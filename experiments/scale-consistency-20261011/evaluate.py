"""Extend only the people-count axis of the inherited evaluation harness."""
import sys
sys.dont_write_bytecode = True
import argparse
import gzip
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import BASE, PROJECT, CELLS, checkpoint, helper, save, sha, verify, compare_fields


def run(args):
    verified = verify()
    h = helper()
    item = checkpoint(args.seed, args.backbone, args.length)
    old = None
    if args.stage == 'parity':
        assert args.seed == 443 and args.length == 24 and args.cell in (0, 5)
        old_path = Path(item['config']).parent / 'evaluation' / (args.backbone + '-t24') / ('cell%d.json' % args.cell)
        old = json.loads(old_path.read_text())
        old_rows = {r['case']: r for r in old['episodes']}
        assert len(old_rows) == 500 and old['torch'] == '2.9.1+cu128'
    else:
        assert args.cell in (6, 7, 8, 9)
    loaded = SimpleNamespace(project=str(PROJECT), config=item['config'],
                             backbone=args.backbone, arm_dir=str(Path(item['path']).parent),
                             expected_checkpoint_sha=item['sha256'], human_select='ttc5')
    cfg, net, digest = h.load(loaded)
    assert float(net.test_action_smoothing) == .3
    assert float(net.test_risk_lambda) == .8
    assert float(net.test_min_clearance) == .2
    assert len(net.action_space) == 80
    assert cfg.getfloat('env', 'time_limit') == 50
    assert cfg.getfloat('sim', 'circle_radius') == 4
    assert cfg.getfloat('sim', 'square_width') == 10
    assert cfg.getint('sim', 'human_num') == 5, 'Do not modify the stored training config'
    n, geometry = CELLS[args.cell]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, max_error = [], {'return': 0., 'minimum_clearance': 0.}
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                       old_torch=old['torch'] if old else None,
                       old_device='4090 per seed443 run provenance' if old else None,
                       tolerance=1e-4 if old else None)
    if args.stage == 'parity':
        assert environment['device'] == 'NVIDIA GeForce RTX 3060'
    score_original = net.score_sarl_candidates
    last_scored = {}

    def score(state):
        result = score_original(state)
        selected = list(net.selection_indices)
        expected = h.select_indices(state.self_state, state.human_states, 'ttc5')
        assert selected == expected and len(selected) == 5
        assert len(result['commands']) == 80
        assert result['windows'].shape == (80, args.length, 8, 13)
        assert len(state.human_states) == n
        last_scored['result'] = result
        return result

    net.score_sarl_candidates = score
    started = time.time()
    with torch.no_grad(), gzip.open(str(path) + '.actions.jsonl.gz', 'wt') as stream:
        env, robot = h.environment(cfg, net, args.cell, 91000)
        assert env.time_limit == 50 and env.time_step == .25
        for case in range(91000, 91000 + args.n_cases):
            net.reset_episode_stats()
            env.reset(seed=case, options={'test_case': case})
            assert len(env.humans) == n and env.time_limit == 50
            assert env.circle_radius == 4 and env.square_width == 10
            outcome, total, minimum = 'timeout', 0., float('inf')
            selection_digest = __import__('hashlib').sha256()
            actions = []
            for k in range(201):
                action = net.predict(h.state_of(env, robot))
                selection_digest.update(bytes(net.selection_indices))
                index = net._last_grid_index
                if index is not None:
                    assert int(last_scored['result']['scores'].argmax().item()) == index
                    assert last_scored['result']['commands'][index] == action
                actions.append(dict(step=k, index=index, command=[float(action.vx), float(action.vy)],
                                    selection_indices=list(net.selection_indices)))
                reward, done, info = h.step(env, action)
                total += .99 ** k * reward
                minimum = min(minimum, min(float(np.hypot(human.px - robot.px, human.py - robot.py)
                                                  - human.radius - robot.radius) for human in env.humans))
                if done:
                    outcome = h.event(info)
                    break
            assert np.isfinite(total) and np.isfinite(minimum)
            row = dict(case=case, cell=args.cell, human_num=n, geometry=geometry,
                       outcome=outcome, **{'return': total}, minimum_clearance=minimum,
                       time=(k + 1) * .25, selection_trace_sha256=selection_digest.hexdigest())
            rows.append(row)
            stream.write(json.dumps(dict(case=case, actions=actions)) + '\n')
            stream.flush()
            if old is not None:
                reference = old_rows[case]
                mismatch, deltas = compare_fields(reference, row, 1e-4)
                for key, delta in deltas.items():
                    max_error[key] = max(max_error[key], delta)
                if mismatch:
                    error = dict(status='FAILED', seed=args.seed, backbone=args.backbone, cell=args.cell,
                                 first_mismatching_case=case, mismatches=mismatch,
                                 first_divergence_step=None,
                                 first_divergence_reason='Historical per-step trace was never recorded; cannot infer first divergent step from final hashes',
                                 new_trace_only=str(path) + '.actions.jsonl.gz', environment=environment,
                                 compared_cases=len(rows), old_source=str(old_path))
                    save(str(path) + '.mismatch.json', error)
                    save(str(path) + '.partial.json', dict(episodes=rows, status='FAILED', error=error))
                    raise AssertionError(json.dumps(error))
            if len(rows) % 10 == 0 or case == 91000:
                save(str(path) + '.progress', dict(completed=len(rows), wall_seconds=time.time() - started,
                                                  last_case=case, stage=args.stage))
                print(json.dumps(dict(stage=args.stage, seed=args.seed, backbone=args.backbone,
                                      T=args.length, cell=args.cell, completed=len(rows))), flush=True)
    assert len(rows) == args.n_cases
    assert sha(item['path']) == digest and sha(item['config']) == item['config_sha256']
    result = dict(status='PASS', episodes=rows, checkpoint_sha256=digest,
                  config_sha256=sha(item['config']), backbone=args.backbone, training_seed=args.seed,
                  selection='ttc5', seq_len=args.length, grid_actions=80, time_limit=50,
                  environment=environment, wall_seconds=time.time() - started,
                  max_parity_absolute_error=max_error, frozen_files_verified=verified,
                  protocol_sha256=sha(BASE / 'protocol.json'), script_sha256=sha(__file__),
                  interpretation='Saved-field equivalence only, not historical trajectory reproduction' if old else 'Frozen paired evaluation',
                  protocol=dict(case_base=91000, n_cases=args.n_cases, cell=args.cell))
    save(path, result)
    print(json.dumps(dict(status='PASS', out=str(path), episodes=len(rows),
                          wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(419, 443), required=True)
    parser.add_argument('--backbone', choices=('kda', 'gru'), required=True)
    parser.add_argument('--length', type=int, choices=(12, 24), required=True)
    parser.add_argument('--cell', type=int, required=True)
    parser.add_argument('--stage', choices=('parity', 'smoke', 'full'), required=True)
    parser.add_argument('--n-cases', type=int, required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
