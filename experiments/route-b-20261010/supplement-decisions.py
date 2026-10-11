"""Supplement saved trajectories by deterministic action replay, never policy search."""
import sys
sys.dont_write_bytecode = True
import argparse
import gzip
import json
import time
from types import SimpleNamespace
import numpy as np
from common import BASE, PROJECT, harness, save, sha, specs, verify
from measure import physical


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def decision_metrics(values, raw, predicted, commands, selected=None):
    values = np.asarray(values)
    raw = np.asarray(raw)
    predicted = np.asarray(predicted)
    commands = np.asarray(commands)
    assert values.shape == raw.shape == predicted.shape == (80,)
    assert commands.shape == (80, 2)
    score = raw - .8 * np.maximum(0., .2 - predicted)
    score[0] -= .001
    safe = predicted >= .2
    n_safe = int(safe.sum())
    eligible = np.flatnonzero(safe) if n_safe else np.arange(80)
    rank = eligible[np.argsort(-score[eligible], kind='stable')]
    winner = int(rank[0])
    if selected is not None:
        assert winner == selected, ('Saved selection differs', winner, selected,
            float(score[winner] - score[selected]))
    gap = float(score[rank[0]]-score[rank[1]]) if len(rank) >= 2 else None
    value_rank = np.argsort(-values, kind='stable')
    top = value_rank[:5]
    speeds = np.hypot(commands[:, 0], commands[:, 1])
    return dict(n_safe=n_safe, eligibility='safe' if n_safe else 'all-unsafe-fallback',
        decision_gap=gap if n_safe >= 2 else None,
        fallback_gap=gap if not n_safe else None,
        predicted_clearance=predicted.tolist(), complete_unmasked_scores=score.tolist(),
        selected_index=winner, selected_command=commands[winner].tolist(),
        selected_speed=float(speeds[winner]), selected_predicted_clearance=float(predicted[winner]),
        selected_value_rank=int(np.flatnonzero(value_rank == winner)[0])+1,
        decision_top2=rank[:2].tolist(),
        decision_top2_speed_difference=float(abs(speeds[rank[0]]-speeds[rank[1]])) if len(rank)>1 else None,
        decision_top2_velocity_difference=float(np.linalg.norm(commands[rank[0]]-commands[rank[1]])) if len(rank)>1 else None,
        value_top5_indices=top.tolist(), value_top5_speeds=speeds[top].tolist(),
        value_top5_speed_spread=float(np.ptp(speeds[top])),
        value_top5_predicted_clearance=predicted[top].tolist(),
        value_top5_predicted_clearance_spread=float(np.ptp(predicted[top])),
        value_top2_velocity_difference=float(np.linalg.norm(commands[top[0]]-commands[top[1]])))


def replay(item, cell, smoke):
    ref = harness()
    ref.pin(str(PROJECT))
    from crowd_nav.train import load_config
    from crowd_nav import contracts
    from crowd_sim.envs.utils.action import ActionXY
    cfg = load_config(item['config'])
    assert contracts.grid_action_dim(contracts.init_grid_from_cfg(cfg)) == 80
    assert sha(item['weight']) == item['weight_sha256']
    assert sha(item['config']) == item['config_sha256']
    stub = SimpleNamespace(multiagent_training=True, reset_episode_stats=lambda: None)
    env, robot = ref.environment(cfg, stub, cell, 90000)
    assert not robot.visible, 'Cannot reuse true humans for alternate commands if robot visible'
    source = BASE/'p1-margin/per-checkpoint'/item['id']/('cell%d.steps.jsonl.gz'%cell)
    target = BASE/'p1-margin'/('decision-smoke' if smoke else 'decision')/item['id']
    target.mkdir(parents=True, exist_ok=True)
    out = target/('cell%d.json'%cell)
    assert not out.exists(), out
    started = time.time()
    count = 0
    episodes = []
    previous = None
    last_done = True
    with gzip.open(target/('cell%d.steps.jsonl.gz'%cell), 'wt') as stream:
        for row in read_rows(source):
            if smoke and row['case'] >= 90003:
                break
            if row['case'] != previous:
                assert last_done, 'Previous replay did not terminate'
                env.reset(seed=row['case'], options={'test_case': row['case']})
                previous = row['case']
                assert row['step'] == 0
            assert abs(env.global_time - row['step']*.25) < 1e-12
            root = ref.state_of(env, robot)
            assert ref.select_indices(root.self_state, root.human_states, 'ttc5') == row['selection_indices']
            cur = np.array([[h.px, h.py, h.vx, h.vy, h.radius] for h in root.human_states])
            cmd = np.asarray(row['candidate_commands'])
            successor = np.array([root.self_state.px, root.self_state.py])[None] + cmd*.25
            next_humans = cur[:, :2] + cur[:, 2:4]*.25
            predicted = (np.hypot(successor[:, None, 0]-next_humans[None, :, 0],
                successor[:, None, 1]-next_humans[None, :, 1])-root.self_state.radius-cur[None, :, 4]).min(axis=1)
            metrics = decision_metrics(row['values'], row['raw_scores'], predicted, cmd, row['executed_index'])
            commands = [ActionXY(*c) for c in cmd]
            reward, done, info = ref.step(env, ActionXY(*row['executed_command']))
            geometry = physical(root, [h.get_observable_state() for h in env.humans], commands)
            for key, expected in row['geometry'].items():
                assert np.array_equal(geometry[key], expected), ('Replay geometry mismatch', item['id'], cell, row['case'], row['step'], key)
            if done:
                assert ref.event(info) == row['outcome'] and row['steps_to_terminal'] == 0
                episodes.append(dict(case=row['case'], outcome=row['outcome'], steps=row['step']+1))
            else:
                assert row['steps_to_terminal'] > 0
            last_done = done
            metrics.update(cell=cell, case=row['case'], step=row['step'], outcome=row['outcome'],
                robot_state=[getattr(root.self_state,k) for k in ('px','py','vx','vy','radius','gx','gy','v_pref','theta')], current_humans=cur.tolist(),
                selected_actual_endpoint_clearance=float(geometry['endpoint'][metrics['selected_index']]),
                selected_actual_swept_clearance=float(geometry['actual_swept'][metrics['selected_index']]))
            stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            count += 1
    assert last_done and len(episodes) == (3 if smoke else 20)
    assert sha(item['weight']) == item['weight_sha256']
    save(out, dict(status='已验证通过', checkpoint=item['id'], cell=cell, episodes=episodes,
        steps=count, wall_seconds=time.time()-started, source_sha256=sha(source),
        geometry_bitwise=True, selected_actions_identical=True, training=0,
        nn_forward_calls=0, new_policy_rollouts=0, robot_visible=False))
    print(json.dumps(dict(id=item['id'], cell=cell, smoke=smoke, steps=count,
        seconds=time.time()-started)), flush=True)


def shared(item, cell):
    source = BASE/'p1-margin/shared'/item['id']/('cell%d.steps.jsonl.gz'%cell)
    reference = BASE/'p1-margin/per-checkpoint/kda-original-t24'/('cell%d.shared.jsonl.gz'%cell)
    physical_source = BASE/'p1-margin/decision/kda-original-t24'/('cell%d.steps.jsonl.gz'%cell)
    target = BASE/'p1-margin/decision-shared'/item['id']
    target.mkdir(parents=True, exist_ok=True)
    out = target/('cell%d.json'%cell)
    assert not out.exists(), out
    count = 0
    rows = iter(read_rows(source)); refs = iter(read_rows(reference)); states = iter(read_rows(physical_source))
    with gzip.open(target/('cell%d.steps.jsonl.gz'%cell), 'wt') as stream:
        while True:
            row, ref, state = next(rows, None), next(refs, None), next(states, None)
            if row is None:
                assert ref is None and state is None
                break
            assert ref is not None and state is not None
            assert all(row[k] == ref[k] == state[k] for k in ('cell','case','step'))
            # Reproduce torch float32 gamma*V before promotion to float64 rewards.
            scaled = np.float32(.99)*np.asarray(row['values'], dtype=np.float32)
            rewards = np.asarray(ref['raw_rewards'])
            # The inherited scorer uses float32 rewards only when all candidates terminate.
            raw = rewards.astype(np.float32)+scaled if np.isin(rewards, [-.5,1.]).all() else rewards+scaled
            if item['id'] == 'kda-original-t24':
                own = np.asarray(state['complete_unmasked_scores'])
            record = decision_metrics(row['values'], raw, state['predicted_clearance'], ref['commands'])
            if item['id'] == 'kda-original-t24':
                assert record['selected_index'] == state['selected_index']
                assert np.array_equal(record['complete_unmasked_scores'], own)
            record.update(cell=cell,case=row['case'],step=row['step'],outcome=state['outcome'],
                selected_actual_endpoint_clearance=ref['geometry']['endpoint'][record['selected_index']],
                selected_actual_swept_clearance=ref['geometry']['actual_swept'][record['selected_index']])
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            count += 1
    save(out, dict(status='已验证通过',steps=count,checkpoint=item['id'],cell=cell,
        reference='kda-original-t24',state_matching=True,nn_forward_calls=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('smoke','own','shared'), required=True)
    p.add_argument('--id'); p.add_argument('--cell', type=int)
    args = p.parse_args()
    verify()
    if args.mode == 'smoke':
        start = time.time()
        for item in specs():
            if item['id'] not in ('kda-original-t24','gru-original-t24','mamba-original-t24'):
                continue
            for cell in range(6):
                replay(item, cell, True)
        summaries = [json.loads(f.read_text()) for f in (BASE/'p1-margin/decision-smoke').glob('*/cell[0-5].json')]
        assert len(summaries) == 18
        elapsed = time.time()-start
        estimate = sum(r['wall_seconds'] for r in summaries)*1320/54
        save(BASE/'decision-smoke-check.json',dict(status='已验证通过',episodes=54,
            wall_seconds=elapsed, projected_worker_seconds_full=estimate, cost_gate_pass=estimate <= 3600,
            decision_argmax_and_geometry_parity=True, original_weight_and_code_verified=verify()))
        assert estimate <= 3600, ('Cost gate not passed', estimate)
    else:
        item = next(s for s in specs() if s['id'] == args.id)
        assert args.cell in range(6)
        (shared if args.mode=='shared' else lambda i,c:replay(i,c,False))(item,args.cell)


if __name__ == '__main__':
    main()
