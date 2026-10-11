"""Read-only frozen-policy diagnostics; all interventions are instance patches."""
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import kendalltau

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('selection_reference', BASE / 'evaluate-selection.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
ref.EXPECTED['mamba'] = 'f23e411effac2f1959eead081fa49c908a85532642172c50a34427fe40bff2b9'


def same_history(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def install(net, stop):
    from crowd_sim.envs.utils.action import ActionXY
    build = net._build_joint_state_34
    fast = net.score_sarl_candidates
    net.selection_indices = None
    net.selection_rule = 'ttc5'
    net.diag_checks = {'scores': 0, 'selection': 0, 'history': 0, 'stop_contract': 0}
    net.diag_original_fast = fast
    frozen = [None]

    def anchored_build(robot, humans):
        assert net.selection_indices == frozen[0], 'selection changed inside candidate batch'
        return build(robot, [humans[i] for i in frozen[0]])

    def score(state):
        indices = ref.select_indices(state.self_state, state.human_states, 'ttc5')
        net.selection_indices = indices[:]
        frozen[0] = indices[:]
        history = [x.copy() for x in net._history]
        result = fast(state)
        assert len(result['commands']) == 80
        if stop:
            original_commands = net.candidate_commands
            net.candidate_commands = lambda: [ActionXY(0., 0.)]
            try:
                extra = fast(state)
            finally:
                net.candidate_commands = original_commands
            assert torch.equal(result['current_token'], extra['current_token']) if isinstance(result['current_token'], torch.Tensor) else np.array_equal(result['current_token'], extra['current_token'])
            original_windows = result['windows'].clone()
            original_values = result['values'].clone()
            result = dict(result)
            result['commands'] = result['commands'] + [ActionXY(0., 0.)]
            for key in ('windows', 'values', 'rewards', 'clearance', 'raw_scores'):
                result[key] = torch.cat([result[key], extra[key]], dim=0)
            assert torch.equal(result['windows'][:80], original_windows)
            assert torch.equal(result['values'][:80], original_values)
            # Apply the original filter to all 81 candidates together. Index 0 retains its penalty.
            values = result['raw_scores'].clone()
            d = result['clearance']
            if net.test_min_clearance > 0.:
                safe = d >= net.test_min_clearance
                if safe.any():
                    values = values.masked_fill(~safe, -1e9)
            if net.test_risk_lambda > 0.:
                margin = net.test_min_clearance if net.test_min_clearance > 0. else net.discomfort_dist
                values -= net.test_risk_lambda * torch.clamp(margin - d, min=0.)
            values[0] -= 1e-3
            result['scores'] = values
            r = state.self_state
            successors = [net.propagate(h, ActionXY(h.vx, h.vy)) for h in state.human_states]
            robot_next = net.propagate(r, ActionXY(0., 0.))
            assert robot_next.px == r.px and robot_next.py == r.py
            assert robot_next.vx == robot_next.vy == 0.
            expected_reward = net.compute_reward(robot_next, successors, prev_nav=r, action=ActionXY(0., 0.))
            assert abs(float(result['rewards'][80]) - expected_reward) < 2e-7
            dmin = min(np.hypot(r.px - h.px, r.py - h.py) - r.radius - h.radius for h in successors)
            assert abs(float(result['clearance'][80]) - dmin) < 2e-12
            from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
            expected_token = _batch_joint34_to_tokens_vectorized(anchored_build(robot_next, successors).reshape(1, -1))[0]
            assert np.array_equal(result['windows'][80, -1].cpu().numpy(), expected_token)
            net.diag_checks['stop_contract'] += 1
        assert net.selection_indices == indices == frozen[0]
        assert same_history(history, list(net._history))
        assert result['windows'].shape == (80 + int(stop), 24, 8, 13)
        for key in ('values', 'rewards', 'clearance', 'raw_scores', 'scores'):
            assert torch.isfinite(result[key]).all().item(), key
        net.diag_checks['scores'] += 1
        net.diag_checks['selection'] += 1
        net.diag_checks['history'] += 1
        net.diag_result = result
        return result

    net._build_joint_state_34 = anchored_build
    net.score_sarl_candidates = score
    return fast


def load(args):
    # Reuse the supplied harness's weight/grid loader, replacing only its instance installer.
    saved = ref.install_selection
    ref.install_selection = lambda net, rule: None
    args.human_select = 'ttc5'
    try:
        cfg, net, digest = ref.load(args)
    finally:
        ref.install_selection = saved
    alpha = cfg.getfloat('eval_protocol', 'action_smoothing')
    assert alpha == args.alpha
    assert cfg.getfloat('eval_protocol', 'risk_lambda') == .8
    assert net.test_action_smoothing == args.alpha
    assert len(net.action_space) == 80
    assert net.seq_len == 24
    install(net, args.stop)
    return cfg, net, digest


def array_humans(humans):
    return np.array([[h.px, h.py, h.vx, h.vy, h.radius] for h in humans], dtype=np.float64)


def shadow_oracle(net, root, next_humans, baseline, history, previous):
    from crowd_sim.envs.utils.state import ObservableState
    before_hist, before_last = net._history, net._last_action
    propagate = net.propagate
    lookup = {id(h): successor for h, successor in zip(root.human_states, next_humans)}

    def oracle(state, action):
        return lookup[id(state)] if isinstance(state, ObservableState) else propagate(state, action)

    net._history = history
    net._last_action = previous
    net.propagate = oracle
    frozen = net.selection_indices[:]
    try:
        result = net.score_sarl_candidates(root)
        assert net.selection_indices == frozen
        assert baseline['commands'] == result['commands']
        assert np.array_equal(baseline['current_token'], result['current_token'])
        assert torch.equal(baseline['windows'][:, :-1], result['windows'][:, :-1])
        # Robot portion of successor tokens must not change.
        assert torch.equal(baseline['windows'][:, -1, :3], result['windows'][:, -1, :3])
    finally:
        net._history, net._last_action, net.propagate = before_hist, before_last, propagate
    return result


def oracle_record(root, true_humans, baseline, oracle, indices, dt):
    current = array_humans(root.human_states)
    true = array_humans(true_humans)
    assert np.max(np.abs(true[:, :2] - (current[:, :2] + true[:, 2:4] * dt))) < 2e-12
    cv = current.copy()
    cv[:, :2] += dt * cv[:, 2:4]
    errors_p = np.linalg.norm(cv[:, :2] - true[:, :2], axis=1)
    errors_v = np.linalg.norm(cv[:, 2:4] - true[:, 2:4], axis=1)
    va = baseline['values'].cpu().numpy()
    vb = oracle['values'].cpu().numpy()
    i = int(baseline['scores'].argmax()); j = int(oracle['scores'].argmax())
    order_a = np.argsort(-va, kind='stable'); order_b = np.argsort(-vb, kind='stable')
    tau = float(kendalltau(va, vb).statistic)
    ca = baseline['clearance'].cpu().numpy(); cb = oracle['clearance'].cpu().numpy()
    return dict(position_errors_all=errors_p.tolist(), velocity_errors_all=errors_v.tolist(),
        position_errors_selected=errors_p[indices].tolist(), velocity_errors_selected=errors_v[indices].tolist(),
        value_kendall_tau=tau if np.isfinite(tau) else None,
        top5_overlap=len(set(order_a[:5]) & set(order_b[:5])) / 5.,
        top10_overlap=len(set(order_a[:10]) & set(order_b[:10])) / 10.,
        argmax_cv=i, argmax_oracle=j, flipped=i != j,
        value_argmax_flipped=int(np.argmax(va)) != int(np.argmax(vb)),
        cv_value_gap=float(va[i] - va[j]), oracle_value_gap=float(vb[j] - vb[i]),
        cv_clearance_gap=float(ca[j] - ca[i]), oracle_clearance_gap=float(cb[j] - cb[i]),
        cv_values=va.tolist(), oracle_values=vb.tolist(),
        cv_scores=baseline['scores'].cpu().tolist(), oracle_scores=oracle['scores'].cpu().tolist(),
        cv_clearance=ca.tolist(), oracle_clearance=cb.tolist(),
        current_humans=current.tolist(), cv_next_humans=cv.tolist(), true_next_humans=true.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True); p.add_argument('--arm-dir', required=True)
    p.add_argument('--config', required=True); p.add_argument('--backbone', choices=('kda', 'mamba', 'gru'), required=True)
    p.add_argument('--alpha', type=float, choices=(0., .3), required=True)
    p.add_argument('--stop', action='store_true'); p.add_argument('--oracle', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--cell', type=int, required=True); p.add_argument('--n-cases', type=int, required=True)
    p.add_argument('--case-base', type=int, default=90000); p.add_argument('--out', required=True)
    args = p.parse_args()
    assert 90000 <= args.case_base <= 90499 and args.case_base + args.n_cases <= 90500
    assert not (args.oracle and args.stop)
    torch.set_num_threads(1)
    cfg, net, digest = load(args)
    config_sha = ref.sha(args.config); started = time.time(); rows = []
    record_path = Path(args.out).with_suffix('.steps.jsonl.gz')
    record_path.parent.mkdir(parents=True, exist_ok=True)
    records = gzip.open(record_path, 'wt') if args.oracle else None
    try:
        with torch.no_grad():
            env, robot = ref.environment(cfg, net, args.cell, args.case_base)
            for case in range(args.case_base, args.case_base + args.n_cases):
                net.reset_episode_stats(); env.reset(seed=case, options={'test_case': case})
                total, minimum, outcome = 0., float('inf'), 'timeout'
                commands_trace = []; episode_records = []; stop_count = 0
                for k in range(201):
                    root = ref.state_of(env, robot)
                    if args.smoke and k == 0:
                        trial = net.score_sarl_candidates(root)
                        normal = net.diag_original_fast(root)
                        legacy = net._score_sarl_candidates_legacy(root)
                        for key in ('windows', 'values'):
                            assert torch.equal(normal[key], legacy[key]), ('fast/legacy', key)
                            assert torch.equal(trial[key][:80], normal[key]), ('unchanged80', key)
                        for key in ('rewards', 'clearance'):
                            assert torch.max(torch.abs(normal[key] - legacy[key])).item() <= 2e-15
                            assert torch.equal(trial[key][:80], normal[key])
                        assert trial['commands'][:80] == normal['commands'] == legacy['commands']
                    history = net._history.copy(); previous = net._last_action
                    action = net.predict(root)
                    result = net.diag_result
                    selected = int(net._last_grid_index)
                    assert result['commands'][selected] == action
                    indices = net.selection_indices[:]
                    if selected == 80:
                        assert args.stop and action.vx == action.vy == 0.; stop_count += 1
                    reward, done, info = ref.step(env, action)
                    assert robot.vx == action.vx and robot.vy == action.vy
                    total += .99 ** k * reward
                    minimum = min(minimum, min(np.hypot(h.px - robot.px, h.py - robot.py)
                        - h.radius - robot.radius for h in env.humans))
                    commands_trace.append(dict(step=k, action_index=selected, command=[action.vx, action.vy], selection_indices=indices))
                    if args.oracle:
                        true_humans = [h.get_observable_state() for h in env.humans]
                        other = shadow_oracle(net, root, true_humans, result, history, previous)
                        record = oracle_record(root, true_humans, result, other, indices, env.time_step)
                        record.update(case=case, cell=args.cell, step=k, selection_indices=indices,
                            robot_state=[getattr(root.self_state, name) for name in
                                ('px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta')],
                            executed_command=[action.vx, action.vy])
                        episode_records.append(record)
                    if done:
                        outcome = ref.event(info); break
                for record in episode_records:
                    record.update(outcome=outcome, steps_to_terminal=k - record['step'])
                    records.write(json.dumps(record, allow_nan=False) + '\n')
                rows.append(dict(case=case, cell=args.cell, human_num=ref.CELLS[args.cell][0],
                    geometry=ref.CELLS[args.cell][1], outcome=outcome, return_value=total,
                    minimum_clearance=float(minimum), time=(k + 1) * .25,
                    stop_selections=stop_count, commands=commands_trace))
                if len(rows) % 10 == 0:
                    ref.save(str(args.out) + '.progress', dict(episodes=rows, wall_seconds=time.time() - started))
                    print(json.dumps(dict(done=len(rows), cell=args.cell, alpha=args.alpha, stop=args.stop)), flush=True)
        assert len(rows) == args.n_cases
        assert digest == ref.sha(Path(args.arm_dir) / 'rl_model_ep10000.pth')
        assert config_sha == ref.sha(args.config)
        ref.save(args.out, dict(status='已验证通过', episodes=rows, checkpoint_sha256=digest,
            config_sha256=config_sha, script_sha256=ref.sha(__file__), backbone=args.backbone,
            selection='ttc5', smoothing=args.alpha, grid_actions=80, candidate_actions=80 + int(args.stop),
            appended_stop=args.stop, oracle_shadow_only=args.oracle, rule='ttc5',
            checks=net.diag_checks, records=str(record_path) if args.oracle else None,
            wall_seconds=time.time() - started))
        print(json.dumps(dict(status='已验证通过', output=args.out, bytes=Path(args.out).stat().st_size)), flush=True)
    finally:
        if records:
            records.close()


if __name__ == '__main__':
    main()
