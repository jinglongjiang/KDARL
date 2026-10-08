"""Frozen-policy diagnosis. All interventions live outside the project tree."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

CELLS = [(5, 'circle_crossing'), (5, 'square_crossing'),
         (10, 'circle_crossing'), (10, 'square_crossing'),
         (20, 'circle_crossing'), (20, 'square_crossing')]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, allow_nan=False))
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def evaluator(args):
    spec = importlib.util.spec_from_file_location('supplied_eval6', args.eval_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(args.project).resolve()
    module.pin_project(root)
    import crowd_nav, crowd_sim, crowd_nav.policy.mamba_rl as policy
    for item in (crowd_nav, crowd_sim, policy):
        assert Path(item.__file__).resolve().is_relative_to(root)
    return module


def load_model(config, arm_dir, backbone, seed=419):
    from crowd_nav.train import load_config
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    cfg = load_config(config)
    torch.manual_seed(seed)
    cfg.set('mamba', 'temporal_backbone', backbone)
    net = MambaRLPolicy(cfg, device='cuda')
    checkpoint = Path(arm_dir) / 'rl_model_ep10000.pth'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['episode'] == 10000
    weights = {k.removeprefix('_orig_mod.'): v for k, v in payload['policy_state'].items()}
    assert weights
    net.load_state_dict(weights, strict=True)
    net.eval()
    net.multiagent_training = True
    net.use_sarl_predict = True
    net.set_phase('test')
    net.set_training_mode('rl')
    net.epsilon = 0.
    return cfg, net, sha(checkpoint)


def outcome_of(info):
    event = str(info.get('event', '')) if isinstance(info, dict) else type(info).__name__
    event = event.lower().replace('_', '')
    if 'success' in event or 'reachgoal' in event:
        return 'success'
    if 'collision' in event:
        return 'collision'
    return 'timeout'


def env_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        _, reward, terminal, truncated, info = result
        return float(reward), terminal or truncated, info
    _, reward, done, info = result
    return float(reward), done, info


def line_distances(robot, humans):
    origin = np.array([robot.px, robot.py])
    vector = np.array([robot.gx, robot.gy]) - origin
    points = np.array([[h.px, h.py] for h in humans]) - origin
    factor = (points @ vector) / max(float(vector @ vector), 1e-20)
    closest = np.clip(factor, 0., 1.)[:, None] * vector
    return np.linalg.norm(points - closest, axis=1).tolist()


def probe(args):
    from crowd_nav.train import build_env_and_robot, bind_policy
    from crowd_sim.envs.utils.state import JointState
    cfg, net, checkpoint_sha = load_model(args.config, args.arm_dir, args.backbone)
    cross = None
    if args.cross_arm_dir:
        _, cross, _ = load_model(args.cross_config, args.cross_arm_dir, 'mamba')
    original = net.score_sarl_candidates
    captured = {}

    def instrumented(state):
        result = original(state)
        captured['result'] = result
        return result

    net.score_sarl_candidates = instrumented
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.arm_dir) / 'evaluation.json'
    reference = {(r['human_num'], r['geometry'], r['case']): r for r in
                 json.loads(reference_path.read_text())['episodes']}
    summaries = []
    started = time.time()
    with torch.no_grad():
        for cell in [int(i) for i in args.cells.split(',')]:
            n, geometry = CELLS[cell]
            cfg.set('sim', 'human_num', str(n))
            cfg.set('sim', 'test_sim', geometry)
            env, robot = build_env_and_robot(cfg)
            bind_policy(robot, net, env, epsilon=0.)
            env.phase = 'test'
            for case in range(args.case_base, args.case_base + args.n_cases):
                target = out / f'cell{cell}-case{case}.json.gz'
                if target.exists():
                    with gzip.open(target, 'rt') as stream:
                        old = json.load(stream)
                    assert old['checkpoint_sha256'] == checkpoint_sha
                    summaries.append(old['summary'])
                    continue
                net.reset_episode_stats()
                env.reset(seed=case, options={'test_case': case})
                assert len(env.humans) == n
                initial_lines = line_distances(robot, env.humans)
                initial = [[h.px, h.py, h.vx, h.vy, h.radius] for h in env.humans]
                frames = []
                previous = None
                total, minimum, result_name = 0., float('inf'), 'timeout'
                for k in range(201):
                    state = JointState(robot.get_full_state(),
                                       [h.get_observable_state() for h in env.humans])
                    position = np.array([robot.px, robot.py])
                    goal = np.array([robot.gx, robot.gy])
                    distance = float(np.linalg.norm(position - goal))
                    actual = net.predict(state)
                    r = captured['result']
                    commands = np.array([[c.vx, c.vy] for c in r['commands']])
                    assert commands.shape == (80, 2)
                    final = int(net._last_grid_index)
                    assert np.array_equal(commands[final], [actual.vx, actual.vy])
                    raw, post, rewards, values, clearance = [
                        r[name].detach().cpu().numpy() for name in
                        ('raw_scores', 'scores', 'rewards', 'values', 'clearance')]
                    assert all(np.isfinite(x).all() for x in (raw, post, rewards, values, clearance))
                    progress = distance - np.linalg.norm(
                        position[None] + net.time_step * commands - goal[None], axis=1)
                    safe = clearance >= net.test_min_clearance
                    gp = np.flatnonzero(safe & (progress > 0.))
                    raw_top = int(raw.argmax())
                    order = np.argsort(-raw, kind='stable')
                    ranks = np.empty(80, dtype=int)
                    ranks[order] = np.arange(1, 81)
                    best_raw_gp = int(gp[np.argmax(raw[gp])]) if len(gp) else None
                    best_progress_gp = int(gp[np.argmax(progress[gp])]) if len(gp) else None
                    distances = np.linalg.norm(
                        np.array([[h.px, h.py] for h in env.humans]) - position, axis=1)
                    future_humans = np.array([[h.px + net.time_step * h.vx,
                                               h.py + net.time_step * h.vy] for h in env.humans])
                    relative_start = np.array([[h.px, h.py] for h in env.humans]) - position
                    human_velocity = np.array([[h.vx, h.vy] for h in env.humans])
                    displacement = net.time_step * (human_velocity[None] - commands[:, None])
                    denominator = np.sum(displacement ** 2, axis=2)
                    numerator = -np.sum(relative_start[None] * displacement, axis=2)
                    along = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.)
                    along = np.clip(along, 0., 1.)
                    swept_points = relative_start[None] + along[:, :, None] * displacement
                    future_robot = position + net.time_step * commands[final]
                    radii = np.array([h.radius for h in env.humans])
                    swept_matrix = np.linalg.norm(swept_points, axis=2) - radii[None] - robot.radius
                    individual_clearance = (np.linalg.norm(future_humans - future_robot, axis=1)
                                            - radii - robot.radius)
                    critical = int(individual_clearance.argmin())
                    assert abs(float(individual_clearance[critical]) - clearance[final]) < 1e-10
                    reversal = False
                    velocity = commands[final]
                    if previous is not None and np.linalg.norm(previous) * np.linalg.norm(velocity) > 1e-18:
                        reversal = float(previous @ velocity) < 0.
                    collision = clearance < 0.
                    reaching = np.linalg.norm(position[None] + net.time_step * commands - goal[None], axis=1) < net.success_radius
                    nonterminal = ~(collision | reaching)
                    frame = {
                        'step': k, 'time': k * net.time_step,
                        'position': position.tolist(), 'goal': goal.tolist(), 'goal_distance': distance,
                        'commands': commands.tolist(), 'progress': progress.tolist(),
                        'dmin': clearance.tolist(), 'r': rewards.tolist(), 'V': values.tolist(),
                        'dmin_swept_shadow': swept_matrix.min(axis=1).tolist(),
                        'raw_score': raw.tolist(), 'final_score': post.tolist(),
                        'rawtop': raw_top, 'final': final, 'rank_of_final': int(ranks[final]),
                        'rawtop_safe': bool(safe[raw_top]),
                        'rawtop_filtered': bool(not safe[raw_top] and safe.any()),
                        'n_safe': int(safe.sum()), 'all_unsafe': not bool(safe.any()),
                        'best_raw_safe_progress': best_raw_gp,
                        'best_progress_safe': best_progress_gp,
                        'rank_best_raw_safe_progress': int(ranks[best_raw_gp]) if best_raw_gp is not None else None,
                        'rank_best_progress_safe': int(ranks[best_progress_gp]) if best_progress_gp is not None else None,
                        'reversal': reversal,
                        'critical_index': critical, 'nearest_index': int(distances.argmin()),
                        'human_distances': distances.tolist(),
                        'nonterminal': nonterminal.tolist(),
                        'candidate_below_stand_threshold': int((np.linalg.norm(commands, axis=1) < .05).sum()),
                    }
                    if cross is not None:
                        frame['mamba_same_window_V'] = cross.forward_value(r['windows']).cpu().tolist()
                    reward, done, info = env_step(env, actual)
                    total += .99 ** k * reward
                    minimum = min(minimum, min(float(np.hypot(h.px - robot.px, h.py - robot.py)
                                                   - h.radius - robot.radius) for h in env.humans))
                    frame['actual_reward'] = reward
                    frame['actual_clearance_endpoint'] = min(float(np.hypot(h.px - robot.px, h.py - robot.py)
                                                                  - h.radius - robot.radius)
                                                             for h in env.humans)
                    frame['environment_dmin'] = float(info['dmin']) if isinstance(info, dict) and 'dmin' in info else None
                    if frame['environment_dmin'] is not None and outcome_of(info) != 'collision':
                        assert abs(frame['environment_dmin'] - max(float(swept_matrix[final].min()), 0.)) < 1e-9
                    frame['after_goal_distance'] = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
                    frames.append(frame)
                    previous = velocity
                    if done:
                        result_name = outcome_of(info)
                        break
                ref = reference[(n, geometry, case)]
                parity = (ref['outcome'] == result_name and ref['time'] == len(frames) * .25
                          and abs(ref['return'] - total) < 1e-9
                          and abs(ref['minimum_clearance'] - minimum) < 1e-9)
                summary = {
                    'cell': cell, 'human_num': n, 'geometry': geometry, 'case': case,
                    'outcome': result_name, 'steps': len(frames), 'time': len(frames) * .25,
                    'return': total, 'minimum_clearance': minimum,
                    'start_distance': frames[0]['goal_distance'],
                    'end_distance': frames[-1]['after_goal_distance'],
                    'initial_line_distances': initial_lines,
                    'initial_humans': initial, 'parity': parity,
                    'nearest_blind_fraction': float(np.mean([f['nearest_index'] >= 5 for f in frames])),
                    'critical_blind_fraction': float(np.mean([f['critical_index'] >= 5 for f in frames])),
                }
                assert parity, f'Instrumented run differs: {args.backbone}/{cell}/{case}'
                with gzip.open(target, 'wt') as stream:
                    json.dump({'checkpoint_sha256': checkpoint_sha, 'summary': summary,
                               'frames': frames}, stream, allow_nan=False)
                summaries.append(summary)
                save(out / 'summary.json', {'backbone': args.backbone, 'episodes': summaries,
                                           'checkpoint_sha256': checkpoint_sha,
                                           'wall_seconds': time.time() - started})
                print(args.backbone, cell, case, result_name, 'PARITY_OK', flush=True)
    save(out / 'DONE.json', {'episodes': len(summaries), 'wall_seconds': time.time() - started})


def evaluate_variant(args, supplied):
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    original = MambaRLPolicy._build_joint_state_34
    if args.selection == 'distance':
        original_score = MambaRLPolicy.score_sarl_candidates

        def select_at_current_state(self, state):
            robot = state.self_state
            self._diag_selected_indices = sorted(
                range(len(state.human_states)), key=lambda i:
                (state.human_states[i].px - robot.px) ** 2
                + (state.human_states[i].py - robot.py) ** 2)[:5]
            return original_score(self, state)

        def nearest_five(self, robot, humans):
            # One real-state selection per control step; all 80 candidates retain
            # these same actors. Otherwise the vectorized scorer's shared human
            # block would inadvertently select actors using candidate0's position.
            assert hasattr(self, '_diag_selected_indices')
            return original(self, robot, [humans[i] for i in self._diag_selected_indices])
        MambaRLPolicy.score_sarl_candidates = select_at_current_state
        MambaRLPolicy._build_joint_state_34 = nearest_five
    cells = ([(args.population, 'circle_crossing')] if args.population is not None
             else [CELLS[int(i)] for i in args.cells.split(',')])
    supplied.evaluate(args.arm_dir, args.backbone, args.out, 419, cells,
                      list(range(args.case_base, args.case_base + args.n_cases)),
                      10000, args.config)


def continuation(args):
    from crowd_nav.train import build_env_and_robot, bind_policy
    from crowd_sim.envs.utils.state import JointState
    cfg, net, checkpoint_sha = load_model(args.config, args.arm_dir, args.backbone)
    selections = json.loads(Path(args.selections).read_text())
    outputs = []
    with torch.no_grad():
        for choice in selections:
            if choice['arm'] != args.backbone:
                continue
            target = Path(args.out) / f"{choice['cell']}-{choice['case']}-{choice['step']}.json"
            if target.exists():
                outputs.append(json.loads(target.read_text()))
                continue
            n, geometry = CELLS[choice['cell']]
            cfg.set('sim', 'human_num', str(n))
            cfg.set('sim', 'test_sim', geometry)
            env, robot = build_env_and_robot(cfg)
            bind_policy(robot, net, env, epsilon=0.)
            net.reset_episode_stats()
            env.phase = 'test'
            env.reset(seed=choice['case'], options={'test_case': choice['case']})
            source = Path(args.probe_root) / args.backbone / f"cell{choice['cell']}-case{choice['case']}.json.gz"
            with gzip.open(source, 'rt') as stream:
                baseline = json.load(stream)
            captured = {}
            original = net.score_sarl_candidates

            def wrapper(state):
                r = original(state)
                captured['r'] = r
                return r

            net.score_sarl_candidates = wrapper
            total, minimum, result_name = 0., float('inf'), 'timeout'
            root_return = 0.
            minimum_after_root = float('inf')
            intervention = None
            for k in range(201):
                state = JointState(robot.get_full_state(),
                                   [h.get_observable_state() for h in env.humans])
                actual = net.predict(state)
                if k <= choice['step']:
                    before = baseline['frames'][k]
                    assert before['final'] == net._last_grid_index
                    assert np.max(np.abs(np.array(before['commands'][before['final']])
                                         - [actual.vx, actual.vy])) < 1e-12
                    assert np.max(np.abs(captured['r']['raw_scores'].cpu().numpy()
                                         - np.array(before['raw_score']))) < 1e-7
                if k == choice['step']:
                    index = choice['replacement']
                    actual = captured['r']['commands'][index]
                    net._last_grid_index = index
                    net._last_action = actual
                    intervention = {'grid_index': index, 'command': [actual.vx, actual.vy],
                                    'predicted_clearance': float(captured['r']['clearance'][index])}
                reward, done, info = env_step(env, actual)
                total += .99 ** k * reward
                if k >= choice['step']:
                    root_return += .99 ** (k - choice['step']) * reward
                minimum = min(minimum, min(float(np.hypot(h.px - robot.px, h.py - robot.py)
                                               - h.radius - robot.radius) for h in env.humans))
                if k >= choice['step']:
                    minimum_after_root = min(minimum_after_root,
                        min(float(np.hypot(h.px - robot.px, h.py - robot.py)
                                  - h.radius - robot.radius) for h in env.humans))
                if done:
                    result_name = outcome_of(info)
                    break
            net.score_sarl_candidates = original
            assert intervention is not None
            original_root_return = sum(.99 ** i * f['actual_reward'] for i, f in
                                       enumerate(baseline['frames'][choice['step']:]))
            row = dict(choice, checkpoint_sha256=checkpoint_sha,
                       intervention=intervention, outcome=result_name,
                       time=(k + 1) * .25, minimum_clearance=minimum, return_value=total,
                       minimum_clearance_after_root=minimum_after_root,
                       baseline_minimum_after_root=min(f['actual_clearance_endpoint']
                           for f in baseline['frames'][choice['step']:]),
                       baseline_outcome=baseline['summary']['outcome'],
                       baseline_return=baseline['summary']['return'],
                       root_return=root_return, baseline_root_return=original_root_return,
                       prefix_verified=True)
            save(target, row)
            outputs.append(row)
            print('CF', args.backbone, choice['case'], result_name, root_return, flush=True)
    save(Path(args.out) / 'summary.json', outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('probe', 'evaluate', 'continue'))
    parser.add_argument('--project', required=True)
    parser.add_argument('--eval-script', required=True)
    parser.add_argument('--arm-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cells', default='2,3,4,5')
    parser.add_argument('--case-base', type=int, default=89000)
    parser.add_argument('--n-cases', type=int, default=32)
    parser.add_argument('--cross-arm-dir')
    parser.add_argument('--cross-config')
    parser.add_argument('--population', type=int)
    parser.add_argument('--selection', choices=('fixed', 'distance'), default='fixed')
    parser.add_argument('--selections')
    parser.add_argument('--probe-root')
    args = parser.parse_args()
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.ERROR)
    supplied = evaluator(args)
    if args.mode == 'probe':
        probe(args)
    elif args.mode == 'evaluate':
        evaluate_variant(args, supplied)
    else:
        continuation(args)


if __name__ == '__main__':
    main()
