"""Frozen, offline H3 replay with joint position/velocity residual rotations."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent
EXPECTED = 'cb2e20c4cc2f12014fed589e9ff02468d04cfa12573c66fc79617c95356cbf0a'
GROUPS = ('all', 'success_all', 'collision_all', 'timeout_all',
          'success_last20', 'collision_last20', 'timeout_last20')
VARIANTS = ('cv', 'oracle', 'random_shared', 'random_independent')
METRICS = ('final_flip', 'value_flip', 'env_clearance_delta', 'actual_clearance_delta',
           'progress_delta', 'env_collision', 'actual_collision', 'env_safe', 'actual_safe')
CONDITIONAL = ('cv_value_gap', 'new_value_gap', 'speed_delta', 'command_delta',
               'direction_degrees', 'env_clearance_delta', 'actual_clearance_delta', 'progress_delta')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def source(cell):
    return PARENT / 'h3/per-case/kda-alpha0.3-a80' / ('cell%d.steps.jsonl.gz' % cell)


def records(cell):
    with gzip.open(source(cell), 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def load_model():
    spec = importlib.util.spec_from_file_location('h3_reference', PARENT / 'evaluate-lookahead.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args = argparse.Namespace(project='/root/cc-diag-20261008/code',
        arm_dir=str(PARENT / 'weights/kda'), config=str(PARENT / 'original-configs/kda.ini'),
        backbone='kda', alpha=.3, stop=False, human_select='ttc5')
    cfg, net, digest = module.load(args)
    assert digest == EXPECTED and net.seq_len == cfg.getint('buffer', 'seq_len') == 24
    assert cfg.getint('temporal', 't') == 24
    assert len(net.action_space) == 80 and net.test_action_smoothing == .3
    assert net.test_min_clearance == .2 and net.test_risk_lambda == .8
    assert net.time_step == .25 and net.gamma == .99 and not cfg.getboolean('robot', 'visible')
    assert net.lookahead_ablation_mode == 'full'
    return module, net


def swept_geometry(record, commands, actual):
    """Same-time relative segments, with either stored or actually updated human velocity."""
    from crowd_sim.envs.utils.utils import point_to_segment_dist
    robot = np.array(record['robot_state'], dtype=np.float64)
    humans = np.array(record['current_humans'], dtype=np.float64)
    if actual:
        true = np.array(record['true_next_humans'], dtype=np.float64)
        velocity = (true[:, :2] - humans[:, :2]) / .25
        assert np.max(np.abs(velocity - true[:, 2:4])) < 1e-11
    else:
        velocity = humans[:, 2:4]
    values, loop_values = [], []
    for command in commands:
        distances = []
        first_negative = None
        for i, h in enumerate(humans):
            x, y = h[:2] - robot[:2]
            vx, vy = velocity[i] - command
            d = float(point_to_segment_dist(x, y, x + .25 * vx, y + .25 * vy, 0., 0.) - h[4] - robot[4])
            distances.append(d)
            if first_negative is None and d < 0:
                first_negative = d
        minimum = min(distances)
        values.append(minimum)
        loop_values.append(first_negative if first_negative is not None else minimum)
    values, loop_values = np.array(values), np.array(loop_values)
    assert np.array_equal(values < 0, loop_values < 0)
    return values, loop_values


def rotate(record, kind, repetition):
    cv = np.array(record['cv_next_humans'], dtype=np.float64)
    true = np.array(record['true_next_humans'], dtype=np.float64)
    residual = true[:, :4] - cv[:, :4]
    assert np.max(np.abs(residual[:, :2] - .25 * residual[:, 2:4])) < 2e-12
    key = '%d/%d/%d/%s/%d/20261009' % (record['cell'], record['case'], record['step'], kind, repetition)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0., 2 * np.pi, 1 if kind == 'random_shared' else len(cv))
    angle = np.broadcast_to(angle, (len(cv),))
    c, s = np.cos(angle), np.sin(angle)
    rotated = residual.copy()
    for offset in (0, 2):
        rotated[:, offset] = c * residual[:, offset] - s * residual[:, offset + 1]
        rotated[:, offset + 1] = s * residual[:, offset] + c * residual[:, offset + 1]
    generated = cv.copy(); generated[:, :4] += rotated
    for offset in (0, 2):
        assert np.allclose(np.linalg.norm(rotated[:, offset:offset + 2], axis=1),
            np.linalg.norm(residual[:, offset:offset + 2], axis=1), rtol=1e-11, atol=2e-12)
    assert np.max(np.abs(rotated[:, :2] - .25 * rotated[:, 2:4])) < 2e-12
    if kind == 'random_shared':
        assert np.allclose(rotated @ rotated.T, residual @ residual.T, atol=2e-11, rtol=1e-10)
    assert np.array_equal(generated[:, 4], cv[:, 4])
    # Validate amplitudes after addition, not only on the intermediate rotation.
    observed = generated[:, :4] - cv[:, :4]
    error = max(float(np.max(np.abs(np.linalg.norm(observed[:, j:j+2], axis=1)
        - np.linalg.norm(residual[:, j:j+2], axis=1)))) for j in (0, 2))
    assert error < 3e-12
    return generated, angle[:1].tolist() if kind == 'random_shared' else angle.tolist(), error


def variant_metrics(result, baseline, commands, env_d, actual_d, env_loop, progress):
    values = result['values'].cpu().numpy()
    cv_values = baseline['values'].cpu().numpy()
    final = int(result['scores'].argmax()); original = int(baseline['scores'].argmax())
    value_id = int(np.argmax(values)); cv_value_id = int(np.argmax(cv_values))
    old, new = commands[original], commands[final]
    old_speed, new_speed = float(np.linalg.norm(old)), float(np.linalg.norm(new))
    angle = None
    if old_speed > 1e-12 and new_speed > 1e-12:
        angle = float(abs(np.degrees(np.arctan2(old[0] * new[1] - old[1] * new[0], np.dot(old, new)))))
    return dict(final_index=final, value_index=value_id, final_flip=float(final != original),
        value_flip=float(value_id != cv_value_id), command=new.tolist(),
        cv_value_gap=float(cv_values[original] - cv_values[final]),
        new_value_gap=float(values[final] - values[original]),
        value_only_cv_gap=float(cv_values[cv_value_id] - cv_values[value_id]),
        value_only_new_gap=float(values[value_id] - values[cv_value_id]),
        speed_delta=new_speed - old_speed, command_delta=float(np.linalg.norm(new - old)),
        direction_degrees=angle, env_clearance=float(env_d[final]), actual_clearance=float(actual_d[final]),
        env_reported_loop_clearance=float(env_loop[final]),
        env_clearance_delta=float(env_d[final] - env_d[original]),
        actual_clearance_delta=float(actual_d[final] - actual_d[original]),
        progress=float(progress[final]), progress_delta=float(progress[final] - progress[original]),
        env_collision=float(env_d[final] < 0), actual_collision=float(actual_d[final] < 0),
        env_safe=float(env_d[final] >= .2), actual_safe=float(actual_d[final] >= .2),
        values=values.tolist(), final_scores=result['scores'].cpu().tolist())


def new_group(repeats):
    return {'n_steps': 0, 'sums': {v: {k: 0. for k in METRICS} for v in VARIANTS},
        'conditional': {v: {k: [0., 0.] for k in CONDITIONAL} for v in VARIANTS},
        'repeat_flips': {v: [0.] * repeats for v in VARIANTS[2:]}}


def accumulate(groups, data, record, repeats):
    names = ['all', record['outcome'] + '_all']
    if record['steps_to_terminal'] < 20:
        names.append(record['outcome'] + '_last20')
    for name in names:
        group = groups.setdefault(name, new_group(repeats)); group['n_steps'] += 1
        for variant in VARIANTS:
            choices = data[variant] if variant.startswith('random') else [data[variant]]
            weight = 1. / len(choices)
            for r, choice in enumerate(choices):
                for metric in METRICS:
                    group['sums'][variant][metric] += weight * choice[metric]
                if variant.startswith('random'):
                    group['repeat_flips'][variant][r] += choice['final_flip']
                if choice['final_flip']:
                    for metric in CONDITIONAL:
                        value = choice[metric]
                        if value is not None:
                            group['conditional'][variant][metric][0] += weight * value
                            group['conditional'][variant][metric][1] += weight


def run(args):
    from types import SimpleNamespace
    torch.set_num_threads(1)
    reference, net = load_model()
    from crowd_sim.envs.utils.action import ActionXY
    from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), 'Never overwrite a completed block'
    start_sha = sha(PARENT / 'weights/kda/rl_model_ep10000.pth')
    assert start_sha == EXPECTED
    start_config = sha(PARENT / 'original-configs/kda.ini')
    input_sha = sha(source(args.cell))
    episodes, active, previous_record = [], None, None
    checked, norm_error, seconds = 0, 0., []
    record_out = out.with_suffix('.steps.jsonl.gz')
    tick = time.time()
    with torch.no_grad(), gzip.open(record_out, 'wt') as stream:
        for record in records(args.cell):
            case = record['case']
            if not (90000 + args.offset <= case < 90000 + args.offset + args.n_cases):
                continue
            if args.mode == 'benchmark' and checked >= 12:
                break
            if active is None or case != active['case']:
                if active is not None:
                    episodes.append(active)
                net.reset_episode_stats(); previous_record = None
                active = dict(cell=args.cell, case=case, outcome=record['outcome'], groups={})
                assert record['step'] == 0
            else:
                assert record['step'] == previous_record['step'] + 1
            humans = [ObservableState(*h) for h in record['current_humans']]
            root = JointState(FullState(*record['robot_state']), humans)
            if previous_record is not None:
                previous_command = previous_record['executed_command']
                assert np.array_equal(np.array(record['current_humans']), np.array(previous_record['true_next_humans']))
                assert np.array_equal(np.array(record['robot_state'][2:4]), np.array(previous_command))
                net._last_action = ActionXY(*previous_command)
            assert net._last_action is None if previous_record is None else True
            before = net._history.copy(); previous_action = net._last_action
            torch.cuda.synchronize(); started = time.perf_counter()
            cv = net.score_sarl_candidates(root)
            commands = np.array([[c.vx, c.vy] for c in cv['commands']], dtype=np.float64)
            assert len(commands) == 80
            assert net.selection_indices == record['selection_indices']
            assert reference.ref.select_indices(root.self_state, humans, 'ttc5') == record['selection_indices']
            for key, old_key in (('values', 'cv_values'), ('scores', 'cv_scores'), ('clearance', 'cv_clearance')):
                actual = cv[key].cpu().numpy()
                assert np.array_equal(actual, np.asarray(record[old_key], dtype=actual.dtype)), ('CV parity', key, args.cell, case, record['step'])
            assert int(cv['scores'].argmax()) == record['argmax_cv']
            assert np.array_equal(commands[record['argmax_cv']], record['executed_command'])
            oracle = reference.shadow_oracle(net, root, [ObservableState(*h) for h in record['true_next_humans']],
                cv, before.copy(), previous_action)
            for key, old_key in (('values', 'oracle_values'), ('scores', 'oracle_scores'), ('clearance', 'oracle_clearance')):
                actual = oracle[key].cpu().numpy()
                assert np.array_equal(actual, np.asarray(record[old_key], dtype=actual.dtype)), ('Oracle parity', key, args.cell, case, record['step'])
            assert int(oracle['scores'].argmax()) == record['argmax_oracle']
            env_d, env_loop = swept_geometry(record, commands, False)
            actual_d, _ = swept_geometry(record, commands, True)
            robot = np.array(record['robot_state'])
            progress = np.linalg.norm(robot[:2] - robot[5:7]) - np.linalg.norm(robot[None, :2] + .25 * commands - robot[None, 5:7], axis=1)
            data = {'cv': variant_metrics(cv, cv, commands, env_d, actual_d, env_loop, progress),
                'oracle': variant_metrics(oracle, cv, commands, env_d, actual_d, env_loop, progress)}
            for variant in VARIANTS[2:]:
                data[variant] = []
                for repetition in range(args.repeats):
                    random_humans, angles, error = rotate(record, variant, repetition)
                    norm_error = max(norm_error, error)
                    result = reference.shadow_oracle(net, root, [ObservableState(*h) for h in random_humans],
                        cv, before.copy(), previous_action)
                    assert net.selection_indices == record['selection_indices']
                    metrics = variant_metrics(result, cv, commands, env_d, actual_d, env_loop, progress)
                    metrics.update(angles_radians=angles, repetition=repetition)
                    data[variant].append(metrics)
            torch.cuda.synchronize(); seconds.append(time.perf_counter() - started)
            accumulate(active['groups'], data, record, args.repeats)
            stream.write(json.dumps(dict(cell=args.cell, case=case, step=record['step'], outcome=record['outcome'],
                steps_to_terminal=record['steps_to_terminal'], selection_indices=record['selection_indices'],
                matched_norm_max_error=norm_error, arms=data), allow_nan=False) + '\n')
            net._history.append(cv['current_token'])
            assert len(net._history) == min(record['step'] + 1, 24)
            previous_record = record; checked += 1
            if record['steps_to_terminal'] == 0:
                print(json.dumps(dict(cell=args.cell, case=case, checked_steps=checked, elapsed_seconds=time.time() - tick)), flush=True)
    if active is not None:
        episodes.append(active)
    assert checked > 0
    if args.mode != 'benchmark':
        assert len(episodes) == args.n_cases
    assert sha(PARENT / 'weights/kda/rl_model_ep10000.pth') == start_sha
    assert sha(PARENT / 'original-configs/kda.ini') == start_config
    assert sha(source(args.cell)) == input_sha
    save(out, dict(status='已验证通过', mode=args.mode, episodes=episodes, checked_steps=checked,
        repeats=args.repeats, checkpoint_sha256=start_sha, config_sha256=start_config, input_sha256=input_sha,
        matched_norm_max_error=norm_error, cv_values_argmax_bitwise=True, oracle_values_argmax_bitwise=True,
        command_and_selection_and_history_frozen=True, per_step=str(record_out), wall_seconds=time.time() - tick,
        steady_seconds_per_step=float(np.mean(seconds[1:])) if len(seconds) > 1 else seconds[0]))
    print(json.dumps(dict(status='已验证通过', output=str(out), bytes=out.stat().st_size,
        records_bytes=record_out.stat().st_size, checked_steps=checked)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cell', type=int, required=True); p.add_argument('--offset', type=int, default=0)
    p.add_argument('--n-cases', type=int, required=True); p.add_argument('--repeats', type=int, required=True)
    p.add_argument('--mode', choices=('benchmark', 'implcheck', 'full'), required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    assert 0 <= args.cell < 6 and 0 <= args.offset < 50 and args.offset + args.n_cases <= 50
    assert 5 <= args.repeats <= 10
    try:
        run(args)
    except BaseException as error:
        save(str(args.out) + '.error.json', dict(status='已确认失败', error=repr(error), fail_closed=True))
        raise
