"""Monkey-patch network-input selection only; full crowd remains in safety/reward."""
import sys
sys.dont_write_bytecode = True
import argparse
import gzip
import hashlib
import time
from types import SimpleNamespace
import numpy as np
import torch
from common import BASE, PROJECT, HISTORY, RULES, module, protocol, save, sha

REF = module('ac3_history_harness', HISTORY)
AC3 = module('downloaded_ac3', BASE / 'action-coverage-reference.py')


def canonical_ttc(p, v, humans, k=5):
    assert k == 5
    robot = SimpleNamespace(px=p[0], py=p[1], vx=v[0], vy=v[1])
    rows = [SimpleNamespace(px=h[0], py=h[1], vx=h[2], vy=h[3], radius=h[4]) for h in humans]
    return REF.select_indices(robot, rows, 'ttc5')


# Keep inherited TTC precision and tie handling in all-unsafe and zero-gain fill.
AC3.ttc5_indices = canonical_ttc


def humans_array(humans):
    return np.asarray([[h.px, h.py, h.vx, h.vy, h.radius] for h in humans], dtype=np.float64)


def endpoint_clearances(robot, humans, commands):
    rp = np.array([robot.px, robot.py]) + commands * .25
    hp = humans[:, :2] + humans[:, 2:4] * .25
    return (np.hypot(rp[:, None, 0] - hp[None, :, 0],
                     rp[:, None, 1] - hp[None, :, 1]) - robot.radius - humans[None, :, 4])


def goal_indices(robot, h):
    goal = np.array([robot.gx - robot.px, robot.gy - robot.py])
    norm = np.linalg.norm(goal)
    velocity = goal / norm * robot.v_pref if norm > 0 else np.zeros(2)
    origin = h[:, :2] - np.array([robot.px, robot.py])
    relative = h[:, 2:4] - velocity
    den = (relative ** 2).sum(axis=1)
    t = np.clip(-(origin * relative).sum(axis=1) / np.maximum(den, 1e-12), 0., 1.)
    clear = np.linalg.norm(origin + relative * t[:, None], axis=1) - robot.radius - h[:, 4]
    return sorted(np.argsort(clear, kind='stable')[:5].tolist())


def select(state, commands, rule):
    robot, humans = state.self_state, state.human_states
    h = humans_array(humans)
    u = np.asarray([[a.vx, a.vy] for a in commands])
    assert u.shape == (80, 2)
    ttc = REF.select_indices(robot, humans, 'ttc5')
    clear = endpoint_clearances(robot, h, u)
    safe = clear.min(axis=1) >= .2
    if len(humans) <= 5:
        ids = list(range(len(humans)))
    elif rule == 'ttc5' or (rule == 'ac3' and not safe.any()):
        ids = ttc
    elif rule == 'goal-cv1s':
        ids = goal_indices(robot, h)
    elif rule == 'ac3':
        ids = AC3.select_action_coverage([robot.px, robot.py], [robot.vx, robot.vy],
                                       robot.radius, h, u, k=5)
    else:
        raise ValueError(rule)
    assert ids == sorted(set(ids)) and len(ids) == min(5, len(humans))
    return ids, ttc, clear


def install(net, rule):
    original_build = net._build_joint_state_34
    fast, legacy = net.score_sarl_candidates, net._score_sarl_candidates_legacy
    net.selection_rule = rule
    net.selection_indices = None
    pinned = [None]

    def build(robot, humans):
        assert net.selection_indices == pinned[0]
        return original_build(robot, [humans[i] for i in pinned[0]])

    def score(state, use_legacy=False):
        assert net._phase == 'test' and len(net.action_space) == 80
        assert len(state.human_states) in (5, 20)
        commands = net.candidate_commands()
        ids, ttc, clear = select(state, commands, net.selection_rule)
        pinned[0] = ids[:]
        net.selection_indices = ids[:]
        before = list(net._history)
        result = (legacy if use_legacy else fast)(state)
        assert len(result['commands']) == 80
        assert result['windows'].shape == (80, 24, 8, 13)
        assert ids == net.selection_indices == pinned[0]
        assert len(net._history) == len(before)
        assert all(a is b for a, b in zip(before, net._history))
        assert all(a == b for a, b in zip(commands, result['commands']))
        for key in ('values', 'rewards', 'clearance', 'scores'):
            assert torch.isfinite(result[key]).all().item(), key
        actual = result['clearance'].cpu().numpy()
        assert np.array_equal(clear.min(axis=1), actual), 'Full-crowd safety geometry changed'
        if not (actual >= .2).any() and net.selection_rule == 'ac3':
            assert ids == ttc
        net.ac3_result = result
        net.ac3_meta = dict(indices=ids[:], ttc=ttc[:],
                            n_safe=int((actual >= .2).sum()), cv_matrix=clear)
        return result

    net._build_joint_state_34 = build
    net.score_sarl_candidates = score
    net.ac3_legacy = lambda state: score(state, True)


def load(seed, rule):
    item = next(x for x in protocol()['checkpoints'] if x['seed'] == seed)
    REF.install_selection = lambda net, rule: None
    args = SimpleNamespace(project=str(PROJECT), config=item['config'], backbone='kda',
                           arm_dir=str(__import__('pathlib').Path(item['weight']).parent),
                           expected_checkpoint_sha=item['weight_sha256'], human_select='ttc5')
    cfg, net, digest = REF.load(args)
    assert net.seq_len == 24 and net.test_action_smoothing == .3
    assert net.test_risk_lambda == .8 and net.test_min_clearance == .2
    install(net, rule)
    return item, cfg, net, digest


def correctness(seed):
    item, cfg, net, digest = load(seed, 'ttc5')
    stats = dict(fast_legacy=0, five_person_score_equal=0, five_person_trajectory_equal=0,
                 current_selected_once=True, full_crowd_safety_unchanged=True)
    AC3.test_reference()
    with torch.no_grad():
        for cell in (0, 1, 4, 5):
            env, robot = REF.environment(cfg, net, cell, 90000)
            for step in range(3):
                root = REF.state_of(env, robot)
                results = []
                for rule in RULES:
                    net.selection_rule = rule
                    a = net.score_sarl_candidates(root)
                    b = net.ac3_legacy(root)
                    for key in ('windows', 'values', 'current_token'):
                        assert torch.equal(a[key], b[key]) if torch.is_tensor(a[key]) else np.array_equal(a[key], b[key]), key
                    for key in ('rewards', 'clearance'):
                        assert (a[key] - b[key]).abs().max().item() <= 2e-15, key
                    assert a['scores'].argmax().item() == b['scores'].argmax().item()
                    if results:
                        assert torch.equal(results[0]['rewards'], a['rewards'])
                        assert torch.equal(results[0]['clearance'], a['clearance'])
                        assert results[0]['commands'] == a['commands']
                        if cell in (0, 1):
                            for key in ('windows', 'values', 'scores'):
                                assert torch.equal(results[0][key], a[key]), key
                            stats['five_person_score_equal'] += 1
                    results.append(a)
                    stats['fast_legacy'] += 1
                net.selection_rule = 'ttc5'
                action = net.predict(root)
                _, done, _ = REF.step(env, action)
                if done:
                    break
        for cell in (0, 1):
            traces = []
            for rule in RULES:
                net.selection_rule = rule
                env, robot = REF.environment(cfg, net, cell, 90001)
                trace = []
                for step in range(201):
                    action = net.predict(REF.state_of(env, robot))
                    reward, done, info = REF.step(env, action)
                    trace.append([action.vx, action.vy, reward, robot.px, robot.py,
                                  net.ac3_result['values'].cpu().tolist()])
                    if done:
                        break
                if traces:
                    assert traces[0] == trace, 'Five-person complete trajectory differs'
                    stats['five_person_trajectory_equal'] += 1
                traces.append(trace)
    assert sha(item['weight']) == digest
    stats.update(status='PASS', seed=seed, weight_sha256=digest)
    save(BASE / ('correctness-s%d.json' % seed), stats)
    print(stats, flush=True)


def evaluate(args):
    item, cfg, net, digest = load(args.seed, args.rule)
    folder = BASE / ('smoke' if args.smoke else 'per-case') / ('s%d-%s' % (args.seed, args.rule))
    out = folder / ('cell%d.json' % args.cell)
    assert not out.exists(), out
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    with torch.no_grad(), gzip.open(folder / ('cell%d.steps.jsonl.gz' % args.cell), 'wt') as stream:
        env, robot = REF.environment(cfg, net, args.cell, 90000)
        for case in range(90000, 90000 + args.n_cases):
            net.reset_episode_stats()
            env.reset(seed=case, options={'test_case': case})
            assert len(env.humans) == 20 and not robot.visible
            minimum, total = float('inf'), 0.
            episode = []
            trace = hashlib.sha256()
            for k in range(201):
                state = REF.state_of(env, robot)
                action = net.predict(state)
                result, meta = net.ac3_result, net.ac3_meta
                chosen = int(net._last_grid_index)
                assert result['commands'][chosen] == action
                reward, done, info = REF.step(env, action)
                total += .99 ** k * reward
                actual = np.asarray([np.hypot(h.px - robot.px, h.py - robot.py) - h.radius - robot.radius
                                     for h in env.humans])
                constraint_cv = int(np.argmin(meta['cv_matrix'][chosen]))
                constraint_actual = int(np.argmin(actual))
                minimum = min(minimum, float(actual.min()))
                trace.update(np.asarray([action.vx, action.vy, reward, robot.px, robot.py], dtype=np.float64).tobytes())
                row = dict(case=case, cell=args.cell, step=k, selection_indices=meta['indices'],
                           ttc_indices=meta['ttc'], selected_index=chosen,
                           command=[action.vx, action.vy], n_safe=meta['n_safe'],
                           constraint_cv=constraint_cv, constraint_actual_endpoint=constraint_actual,
                           cv_clearance=float(meta['cv_matrix'][chosen].min()),
                           actual_endpoint_clearance=float(actual.min()),
                           ttc_omits_cv=constraint_cv not in meta['ttc'],
                           selected_includes_cv=constraint_cv in meta['indices'],
                           ttc_omits_actual=constraint_actual not in meta['ttc'],
                           selected_includes_actual=constraint_actual in meta['indices'],
                           selection_differs=meta['indices'] != meta['ttc'])
                episode.append(row)
                if done:
                    break
            outcome = REF.event(info) if done else 'timeout'
            for row in episode:
                row['outcome'] = outcome
                stream.write(__import__('json').dumps(row, allow_nan=False) + '\n')
            rows.append(dict(seed=args.seed, rule=args.rule, cell=args.cell, case=case,
                             outcome=outcome, return_=total, time=(k + 1) * .25,
                             minimum_clearance=minimum, steps=k + 1, trajectory_sha256=trace.hexdigest()))
            if len(rows) % 10 == 0:
                save(str(out) + '.progress', dict(completed=len(rows), wall_seconds=time.time()-started))
                print(dict(seed=args.seed, rule=args.rule, cell=args.cell, episodes=len(rows)), flush=True)
    assert sha(item['weight']) == digest and sha(item['config']) == item['config_sha256']
    assert len(rows) == args.n_cases
    save(out, dict(status='PASS', episodes=rows, checkpoint=item, actions=80, alpha=.3,
                   all_crowd_filter=True, torch=torch.__version__, device=torch.cuda.get_device_name(),
                   wall_seconds=time.time()-started, steps=sum(r['steps'] for r in rows)))
    print(dict(status='PASS', out=str(out), episodes=len(rows), wall_seconds=time.time()-started), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=(419,443), required=True)
    p.add_argument('--rule', choices=RULES, default='ttc5')
    p.add_argument('--cell', type=int, choices=(4,5), default=4)
    p.add_argument('--n-cases', type=int, choices=(3,500), default=500)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--correctness', action='store_true')
    args = p.parse_args()
    if args.correctness:
        correctness(args.seed)
    else:
        assert args.n_cases == (3 if args.smoke else 500)
        evaluate(args)
