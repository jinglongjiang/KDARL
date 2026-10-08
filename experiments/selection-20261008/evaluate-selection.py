"""Frozen checkpoint evaluation; selection is anchored to the real current state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

CELLS = [(5, 'circle_crossing'), (5, 'square_crossing'),
         (10, 'circle_crossing'), (10, 'square_crossing'),
         (20, 'circle_crossing'), (20, 'square_crossing')]
EXPECTED = {'gru': '96f06039a07b75d09e3b067ebb796d6de113b6d80a33214db737ed6d3a245828',
            'kda': 'cb2e20c4cc2f12014fed589e9ff02468d04cfa12573c66fc79617c95356cbf0a'}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, allow_nan=False, indent=2))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(project):
    root = Path(project).resolve()
    os.chdir(str(root))
    sys.path.insert(0, str(root))
    import crowd_nav, crowd_sim, crowd_nav.policy.mamba_rl as policy
    for module in (crowd_nav, crowd_sim, policy):
        Path(module.__file__).resolve().relative_to(root)


def select_indices(robot, humans, rule):
    if rule == 'first5' or len(humans) <= 5:
        return list(range(min(5, len(humans))))
    h = np.array([[x.px, x.py, x.vx, x.vy, x.radius] for x in humans], dtype=np.float64)
    if rule == 'nearest5':
        score = np.hypot(h[:, 0] - robot.px, h[:, 1] - robot.py)
    elif rule == 'ttc5':
        # Match the inherited builder's robot float32 round trip and TTC formula.
        rp = np.array([robot.px, robot.py, robot.vx, robot.vy], dtype=np.float32).astype(np.float64)
        rx, ry = h[:, 0] - rp[0], h[:, 1] - rp[1]
        rvx, rvy = h[:, 2] - rp[2], h[:, 3] - rp[3]
        distances = np.sqrt(rx ** 2 + ry ** 2 + 1e-6)
        closing = -(rx * rvx + ry * rvy) / (distances + 1e-6)
        with np.errstate(divide='ignore', invalid='ignore'):
            score = np.where(closing > .1, distances / (closing + 1e-6), distances * 10)
    else:
        raise ValueError(rule)
    # Preserve source order inside the selected set; existing TTC token sort owns order.
    return sorted(np.argsort(score)[:5].tolist())


def install_selection(net, rule):
    original_build = net._build_joint_state_34
    original_fast = net.score_sarl_candidates
    original_legacy = net._score_sarl_candidates_legacy
    net.selection_rule = rule
    net.selection_indices = None

    def build(robot, humans):
        assert net.selection_indices is not None
        return original_build(robot, [humans[i] for i in net.selection_indices])

    def score(state, legacy=False):
        assert net._phase in ('test', 'val', 'eval'), 'Selection is evaluation-only'
        net.selection_indices = select_indices(state.self_state, state.human_states, net.selection_rule)
        result = (original_legacy if legacy else original_fast)(state)
        assert len(result['commands']) == 80 and result['windows'].shape == (80, 24, 8, 13)
        for key in ('values', 'rewards', 'clearance', 'scores'):
            assert torch.isfinite(result[key]).all().item()
        return result

    net._build_joint_state_34 = build
    net.score_sarl_candidates = lambda state: score(state)
    net.selection_legacy_score = lambda state: score(state, legacy=True)


def load(args):
    pin(args.project)
    from crowd_nav.train import load_config
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_nav import contracts
    cfg = load_config(args.config)
    grid = contracts.init_grid_from_cfg(cfg)
    assert grid['n_speeds'] == 5 and grid['n_headings'] == 16
    assert grid['sampling'] == 'exponential' and grid['v_min'] == .05 and not grid['include_stop']
    cfg.set('mamba', 'temporal_backbone', args.backbone)
    torch.set_num_threads(1)
    torch.manual_seed(419)
    net = MambaRLPolicy(cfg, device='cuda')
    net.build_action_space(1.)
    assert len(net.action_space) == 80
    path = Path(args.arm_dir) / 'rl_model_ep10000.pth'
    digest = sha(path)
    assert digest and digest == EXPECTED[args.backbone], (digest, EXPECTED[args.backbone])
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['episode'] == 10000
    weights = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in payload['policy_state'].items()}
    assert weights
    net.load_state_dict(weights, strict=True)
    net.eval()
    net.multiagent_training = True
    net.use_sarl_predict = True
    net.set_phase('test')
    net.set_training_mode('rl')
    net.epsilon = 0.
    install_selection(net, args.human_select)
    return cfg, net, digest


def environment(cfg, net, cell, case):
    from crowd_nav.train import build_env_and_robot, bind_policy
    n, geometry = CELLS[cell]
    cfg.set('sim', 'human_num', str(n))
    cfg.set('sim', 'test_sim', geometry)
    env, robot = build_env_and_robot(cfg)
    bind_policy(robot, net, env, epsilon=0.)
    env.phase = 'test'
    net.reset_episode_stats()
    env.reset(seed=case, options={'test_case': case})
    assert len(env.humans) == n
    assert float(env.time_limit) == 50. and float(env.time_step) == .25
    return env, robot


def state_of(env, robot):
    from crowd_sim.envs.utils.state import JointState
    return JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])


def step(env, action):
    result = env.step(action)
    if len(result) == 5:
        _, reward, terminal, truncated, info = result
        return float(reward), terminal or truncated, info
    _, reward, done, info = result
    return float(reward), done, info


def event(info):
    name = str(info.get('event', '')) if isinstance(info, dict) else type(info).__name__
    name = name.lower().replace('_', '')
    return 'success' if 'success' in name or 'reachgoal' in name else 'collision' if 'collision' in name else 'timeout'


def preflight(args):
    cfg, net, digest = load(args)
    maximum = {key: 0. for key in ('windows', 'values', 'rewards', 'clearance')}
    nonbitwise = {key: 0 for key in maximum}
    comparisons = 0
    five_equal = 0
    started = time.time()
    with torch.no_grad():
        for cell in range(6):
            env, robot = environment(cfg, net, cell, 89000)
            for _ in range(8):
                state = state_of(env, robot)
                reference = None
                for rule in ('first5', 'nearest5', 'ttc5'):
                    net.selection_rule = rule
                    before = [x.copy() for x in net._history]
                    fast = net.score_sarl_candidates(state)
                    frozen = net.selection_indices[:]
                    legacy = net.selection_legacy_score(state)
                    assert frozen == net.selection_indices
                    assert len(before) == len(net._history)
                    assert all(np.array_equal(x, y) for x, y in zip(before, net._history))
                    for key in maximum:
                        a = fast[key].detach().cpu().numpy()
                        b = legacy[key].detach().cpu().numpy()
                        delta = float(np.max(np.abs(a - b)))
                        maximum[key] = max(maximum[key], delta)
                        nonbitwise[key] += int(not np.array_equal(a, b))
                        if key in ('windows', 'values'):
                            assert np.array_equal(a, b), (rule, key, delta)
                        else:
                            # Inherited scalar/vector reward path already differs at float64 ULP.
                            assert delta <= 2e-15, (rule, key, delta)
                    assert fast['scores'].argmax().item() == legacy['scores'].argmax().item()
                    if CELLS[cell][0] == 5:
                        if reference is None:
                            reference = fast
                        else:
                            assert np.array_equal(reference['current_token'], fast['current_token'])
                            for key in ('windows', 'values', 'rewards', 'clearance', 'scores'):
                                assert torch.equal(reference[key], fast[key]), ('5-person', rule, key)
                            five_equal += 1
                    comparisons += 1
                net.selection_rule = 'first5'
                action = net.predict(state)
                _, done, _ = step(env, action)
                if done:
                    break
    save(args.out, {'status': 'PASS', 'checkpoint_sha256': digest, 'grid_actions': 80,
                    'comparisons': comparisons, 'five_person_equal_comparisons': five_equal,
                    'max_abs_error': maximum, 'nonbitwise_comparisons': nonbitwise,
                    'windows_values_bitwise_required': True,
                    'reward_clearance_absolute_tolerance': 2e-15,
                    'wall_seconds': time.time() - started, 'torch': torch.__version__,
                    'script_sha256': sha(__file__),
                    'note': 'Reward/clearance tolerance is inherited ULP, not bitwise parity claim.'})


def evaluate(args):
    cfg, net, digest = load(args)
    config_sha = sha(args.config)
    rows, latency = [], []
    started = time.time()
    with torch.no_grad():
        env, robot = environment(cfg, net, args.cell, args.case_base)
        n, geometry = CELLS[args.cell]
        for case in range(args.case_base, args.case_base + args.n_cases):
            net.reset_episode_stats()
            env.reset(seed=case, options={'test_case': case})
            assert len(env.humans) == n
            outcome, total, minimum = 'timeout', 0., float('inf')
            selection_digest = hashlib.sha256()
            for k in range(201):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                action = net.predict(state_of(env, robot))
                torch.cuda.synchronize()
                latency.append(1000 * (time.perf_counter() - tick))
                selection_digest.update(bytes(net.selection_indices))
                reward, done, info = step(env, action)
                total += .99 ** k * reward
                minimum = min(minimum, min(float(np.hypot(h.px - robot.px, h.py - robot.py)
                                                  - h.radius - robot.radius) for h in env.humans))
                if done:
                    outcome = event(info)
                    break
            rows.append({'case': case, 'cell': args.cell, 'human_num': n, 'geometry': geometry,
                         'outcome': outcome, 'return': total, 'minimum_clearance': minimum,
                         'time': (k + 1) * .25, 'selection_trace_sha256': selection_digest.hexdigest()})
            if len(rows) % 10 == 0:
                save(str(args.out) + '.progress', {'episodes': rows, 'wall_seconds': time.time() - started})
                print(json.dumps({'rule': args.human_select, 'cell': args.cell, 'episodes': len(rows)}), flush=True)
    assert len(rows) == args.n_cases
    assert sha(args.config) == config_sha and sha(Path(args.arm_dir) / 'rl_model_ep10000.pth') == digest
    save(args.out, {'episodes': rows, 'checkpoint_sha256': digest, 'config_sha256': config_sha,
                    'selection': args.human_select, 'backbone': args.backbone, 'grid_actions': 80,
                    'wall_seconds': time.time() - started, 'torch': torch.__version__,
                    'script_sha256': sha(__file__),
                    'policy_latency_ms_median': float(np.median(latency)),
                    'policy_latency_ms_p95': float(np.percentile(latency, 95)),
                    'protocol': {'case_base': args.case_base, 'n_cases': args.n_cases, 'cell': args.cell}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'evaluate'), default='evaluate')
    parser.add_argument('--project', required=True)
    parser.add_argument('--arm-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--backbone', choices=('gru', 'kda'), required=True)
    parser.add_argument('--human-select', choices=('first5', 'nearest5', 'ttc5'), default='first5')
    parser.add_argument('--cell', type=int, default=0)
    parser.add_argument('--case-base', type=int, default=90000)
    parser.add_argument('--n-cases', type=int, default=500)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    save(str(args.out) + '.pid.json', {'pid': os.getpid(), 'proc_start_ticks': stat[19],
                                      'script': str(Path(__file__).resolve()), 'out': args.out})
    if args.mode == 'preflight':
        preflight(args)
    else:
        evaluate(args)
