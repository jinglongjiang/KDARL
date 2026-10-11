"""Frozen TTC5 replay and single-command causal interventions. No project edits."""
import sys
sys.dont_write_bytecode = True
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
MAN = json.loads((BASE / 'manifest.json').read_text())
P = MAN['protocol']
spec = importlib.util.spec_from_file_location('causal_original_driver', BASE / 'reference.py')
REF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REF)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2))
    tmp.replace(path)


def digest(a):
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def write(f, row):
    f.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
    f.flush()


def load(model):
    item = next(m for m in P['models'] if m['id'] == model)
    cfg, net, sha = REF.load(SimpleNamespace(project=P['project'], config=item['config'],
        backbone=item['backbone'], arm_dir=str(Path(item['weight']).parent),
        expected_checkpoint_sha=item['weight_sha256'], human_select='ttc5'))
    assert net.seq_len == 24 and net.test_action_smoothing == .3
    assert net.test_min_clearance == .2 and net.test_risk_lambda == .8 and net.gamma == .99
    assert cfg.getfloat('reward', 'progress_reward') == 0.
    original = net.score_sarl_candidates

    def score(state):
        hist = list(net._history)
        out = original(state)
        assert len(out['commands']) == 80
        assert list(net.selection_indices) == REF.select_indices(state.self_state, state.human_states, 'ttc5')
        assert len(hist) == len(net._history) and all(a is b for a, b in zip(hist, net._history))
        net.causal_result = out
        net.causal_history_hash = digest(np.asarray(hist))
        return out

    net.score_sarl_candidates = score
    return item, cfg, net


def state_arrays(state):
    r = state.self_state
    rob = [r.px, r.py, r.vx, r.vy, r.radius, r.gx, r.gy, r.v_pref, r.theta]
    humans = [[h.px, h.py, h.vx, h.vy, h.radius] for h in state.human_states]
    return rob, humans


def swept(state, commands, next_humans=None):
    r, humans = state_arrays(state)
    h = np.asarray(humans)
    u = np.asarray(commands)
    d0 = np.asarray(r[:2])[None, None] - h[None, :, :2]
    hv = h[:, 2:4] if next_humans is None else (np.asarray(next_humans)[:, :2] - h[:, :2]) / .25
    dv = u[:, None, :] - hv[None]
    den = (dv * dv).sum(axis=-1)
    t = np.clip(-(d0 * dv).sum(axis=-1) / np.maximum(den, 1e-30), 0., .25)
    clear = np.linalg.norm(d0 + t[:, :, None] * dv, axis=-1) - r[4] - h[None, :, 4]
    return clear


def inspect_step(net, state, k):
    out = net.causal_result
    rob, humans = state_arrays(state)
    cmd = np.asarray([[a.vx, a.vy] for a in out['commands']])
    d = math.hypot(rob[0] - rob[5], rob[1] - rob[6])
    progress = d - np.hypot(rob[0] + cmd[:, 0] * .25 - rob[5], rob[1] + cmd[:, 1] * .25 - rob[6])
    result = {'step': k, 'robot': rob, 'humans': humans, 'ids': list(net.selection_indices),
              'commands': cmd.tolist(), 'chosen': int(net._last_grid_index),
              'goal_distance': d, 'progress': progress.tolist(),
              'crowd2m': sum(math.hypot(h[0]-rob[0], h[1]-rob[1]) <= 2. for h in humans),
              'history_sha256': net.causal_history_hash,
              'token_sha256': digest(out['current_token']),
              'windows_sha256': digest(out['windows']),
              'current_token': np.asarray(out['current_token']).tolist()}
    for key in ('values', 'rewards', 'clearance', 'raw_scores', 'scores'):
        result[key] = out[key].detach().cpu().tolist()
    result['n_safe'] = sum(x >= .2 for x in result['clearance'])
    result['command'] = result['commands'][result['chosen']]
    return result


def oracle(net, state, real_state, out, selected, reward, terminal):
    from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
    # Preserve root IDs and batch shape; only the predicted last frame is replaced.
    pinned = list(net.selection_indices)
    future = [net._build_joint_state_34(net.propagate(state.self_state, a), real_state.human_states)
              for a in out['commands']]
    tokens = _batch_joint34_to_tokens_vectorized(np.asarray(future))
    window = out['windows'].clone()
    window[:, -1] = torch.from_numpy(tokens).float().to(net.device)
    values = net.forward_value(window)
    assert pinned == net.selection_indices
    discounted = (.99 * values).detach().cpu().tolist()
    value = float(values[selected].item())
    return {'value': value, 'q_oracle1_unmasked': float(reward + discounted[selected]),
            'q_oracle1_terminal_corrected': float(reward if terminal else reward + discounted[selected]),
            'oracle_values80': values.detach().cpu().tolist(),
            'oracle_window_sha256': digest(window), 'terminal': bool(terminal),
            'true_next_robot': state_arrays(real_state)[0], 'true_next_humans': state_arrays(real_state)[1],
            'true_next_ttc_ids': REF.select_indices(real_state.self_state, real_state.human_states, 'ttc5')}


def historical(model, cell):
    item = next(m for m in P['models'] if m['id'] == model)
    if item['backbone'] != 'kda' or cell not in (4, 5):
        return {}, {}
    folder = Path(P['historical_ac3']) / ('s%d-ttc5' % item['seed'])
    data = json.loads((folder / ('cell%d.json' % cell)).read_text())
    assert data['checkpoint']['weight_sha256'] == item['weight_sha256']
    steps = {}
    for row in rows(folder / ('cell%d.steps.jsonl.gz' % cell)):
        steps.setdefault(row['case'], []).append(row)
    return {r['case']: r for r in data['episodes']}, steps


def check_historical(ep, old, old_steps):
    if old is None:
        return
    assert ep['outcome'] == old['outcome'] and ep['time'] == old['time'], (ep['case'], 'old outcome/time')
    key = 'return_' if 'return_' in old else 'return'
    assert ep['return'] == old[key], (ep['case'], 'old return', ep['return'], old[key])
    assert ep['minimum_clearance'] == old['minimum_clearance'], (ep['case'], 'old clearance',ep['minimum_clearance'],old['minimum_clearance'])
    if 'trajectory_sha256' in old:
        assert ep['trajectory_sha256'] == old['trajectory_sha256'], (ep['case'], 'old trajectory')
    if 'selection_trace_sha256' in old:
        assert ep['selection_trace_sha256'] == old['selection_trace_sha256'], (ep['case'], 'old selection')
    if old_steps is not None:
        assert len(ep['steps']) == len(old_steps)
        for a, b in zip(ep['steps'], old_steps):
            assert a['chosen'] == b['selected_index'] and a['command'] == b['command'], (ep['case'], a['step'], 'old action')
            assert a['ids'] == b['selection_indices'], (ep['case'], a['step'], 'old IDs')


def rollout(cfg, net, model, cell, case, reference=None, root=None, selected=None):
    env, robot = REF.environment(cfg, net, cell, case)
    assert not robot.visible
    total, minimum, trace, ids_hash = 0., float('inf'), hashlib.sha256(), hashlib.sha256()
    steps, first_data = [], None
    root_step = root['step'] if root else None
    for k in range(201):
        state = REF.state_of(env, robot)
        action = net.predict(state)
        row = inspect_step(net, state, k)
        out = net.causal_result
        assert [action.vx, action.vy] == row['command']
        # Full scores, physical state, windows and history must replay exactly.
        if reference is not None and (root_step is None or k <= root_step):
            assert row == {key: reference['steps'][k][key] for key in row}, (model, cell, case, k, 'prefix mismatch')
        if k == root_step:
            assert selected is not None
            action = out['commands'][selected]
            net._last_action = action
            net._last_grid_index = selected
            row['executed_override'] = selected
            row['command'] = [action.vx, action.vy]
        chosen = selected if k == root_step else row['chosen']
        before_geo = swept(state, row['commands'])
        reward, done, info = REF.step(env, action)
        real_state = REF.state_of(env, robot)
        rnew, hnew = state_arrays(real_state)
        actual_geo = swept(state, row['commands'], hnew)
        endpoint = min(float(np.hypot(h.px - robot.px, h.py - robot.py) - h.radius - robot.radius) for h in env.humans)
        row.update(reward=reward, terminal=bool(done), event=info.get('event'),
                   env_dmin=info.get('dmin'), env_swept_clearance=float(before_geo[chosen].min()),
                   actual_swept_clearance=float(actual_geo[chosen].min()), endpoint_clearance=float(endpoint),
                   env_constraint=int(before_geo[chosen].argmin()), actual_constraint=int(actual_geo[chosen].argmin()))
        if k == root_step:
            first_data = oracle(net, state, real_state, out, selected, reward, done)
            first_data.update(reward=reward, predicted_value=row['values'][selected],
                q_pred=row['raw_scores'][selected], final_score=row['scores'][selected],
                predicted_clearance=row['clearance'][selected],
                env_swept_clearance=row['env_swept_clearance'], actual_swept_clearance=row['actual_swept_clearance'],
                predicted_hard_masked=row['n_safe'] > 0 and row['clearance'][selected] < .2,
                risk_penalty=.8 * max(0., .2 - row['clearance'][selected]),
                index0_penalty=.001 if selected == 0 else 0.,
                root_ids=row['ids'], candidate_index=selected, command=row['command'])
        total += .99 ** k * reward
        minimum = min(minimum, float(endpoint))
        trace.update(np.asarray([action.vx, action.vy, reward, robot.px, robot.py], dtype=np.float64).tobytes())
        ids_hash.update(bytes(row['ids']))
        steps.append(row)
        if done:
            break
    ep = dict(model=model, cell=cell, case=case, outcome=REF.event(info) if done else 'timeout',
              return_=total, time=(k+1)*.25, minimum_clearance=minimum, steps=steps,
              trajectory_sha256=trace.hexdigest(), selection_trace_sha256=ids_hash.hexdigest())
    ep['return'] = ep.pop('return_')
    if root:
        ep['root_return'] = sum(.99 ** j * s['reward'] for j, s in enumerate(steps[root_step:]))
        ep['first_step'] = first_data
    if reference is not None and root_step is None:
        assert ep == reference, (model, cell, case, 'full replay mismatch')
    return ep


def preflight(model):
    item, cfg, net = load(model)
    evidence = []
    ac3, trace = historical(model, 5)
    with torch.no_grad():
        for cell in (0, 5):
            path = BASE / 'inputs' / model / ('historical-cell%d.json' % cell)
            old = {r['case']: r for r in json.loads(path.read_text())['episodes']}
            for case in (91000,):
                ep = rollout(cfg, net, model, cell, case)
                check_historical(ep, old[case], None)
                repeated = rollout(cfg, net, model, cell, case, reference=ep)
                evidence.append(dict(cell=cell, case=case, historical_summary_exact=True,
                                     rebuilt_action_history_scores_exact=True, steps=len(repeated['steps'])))
        for case in (90000, 90001, 90002):
            ep = rollout(cfg, net, model, 5, case)
            check_historical(ep, ac3.get(case), trace.get(case))
            rollout(cfg, net, model, 5, case, reference=ep)
            # A0 intervention must be a no-op, including smoothing and history.
            k = min(20, len(ep['steps']) - 1)
            repeat = rollout(cfg, net, model, 5, case, reference=ep, root={'step': k}, selected=ep['steps'][k]['chosen'])
            assert repeat['trajectory_sha256'] == ep['trajectory_sha256'] and repeat['return'] == ep['return']
            assert [s['history_sha256'] for s in repeat['steps']] == [s['history_sha256'] for s in ep['steps']]
            evidence.append(dict(cell=5, case=case, historical_step_trace=case in ac3,
                                 A0_exact=True, steps=len(ep['steps'])))
        from crowd_sim.envs.utils.state import JointState
        from crowd_sim.envs.utils.action import ActionXY
        from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
        env, robot = REF.environment(cfg, net, 5, 90003)
        for k in range(3):
            state = REF.state_of(env, robot)
            action = net.predict(state)
            out = net.causal_result
            cv = JointState(net.propagate(state.self_state, action),
                [net.propagate(h, ActionXY(h.vx,h.vy)) for h in state.human_states])
            check = oracle(net,state,cv,out,net._last_grid_index,0.,False)
            assert check['oracle_values80'] == out['values'].cpu().tolist(), 'Oracle reconstruction changes CV'
            assert check['oracle_window_sha256'] == digest(out['windows']), 'Oracle prefix/token mismatch'
            current = _batch_joint34_to_tokens_vectorized(net._build_joint_state_34(state.self_state,state.human_states)[None])[0]
            assert np.array_equal(current,out['current_token'])
            REF.step(env,action)
        evidence.append(dict(oracle_identity_cv_bitwise=True,steps=3))
    save(BASE / 'work' / ('preflight-' + model + '.json'), dict(status='PASS', evidence=evidence,
        torch=torch.__version__, device=torch.cuda.get_device_name(), weight_sha256=REF.sha(item['weight'])))


def baseline(model, cell, n):
    item, cfg, net = load(model)
    old, old_steps = historical(model, cell)
    path = BASE / 'work' / ('baseline-%s-cell%d.jsonl.gz' % (model, cell))
    assert not path.exists()
    started = time.time()
    with torch.no_grad(), gzip.open(path, 'wt', compresslevel=1) as stream:
        for case in range(90000, 90000+n):
            if time.time() > P['started_unix'] + P['deadline_seconds'] - 1200:
                raise RuntimeError('Budget guard before complete baseline; no interventions authorized')
            ep = rollout(cfg, net, model, cell, case)
            check_historical(ep, old.get(case), old_steps.get(case))
            write(stream, ep)
            if (case-90000+1) % 10 == 0:
                print(json.dumps(dict(mode='baseline', model=model, cell=cell, completed=case-90000+1,
                                      seconds=time.time()-started)), flush=True)
    save(str(path)+'.done.json', dict(status='PASS', episodes=n, wall_seconds=time.time()-started,
                                     sha256=REF.sha(path)))


def alternatives(row):
    safe = [i for i in range(80) if row['clearance'][i] >= .2]
    advancing = [i for i in safe if row['progress'][i] >= .02]
    labels = {'A0': row['chosen']}
    if advancing:
        labels['A1'] = max(advancing, key=lambda i: (row['raw_scores'][i], -i))
    if safe:
        labels['A2'] = max(safe, key=lambda i: (row['progress'][i], -i))
    rest = [i for i in safe if i not in labels.values()]
    if rest:
        labels['A3'] = max(rest, key=lambda i: (row['scores'][i], -i))
    return labels


def freeze_roots():
    roots, denominator = [], []
    for cell in (5, 4, 0, 3):
        for model in [x['id'] for x in P['models']]:
            path = BASE / 'work' / ('baseline-%s-cell%d.jsonl.gz' % (model, cell))
            assert Path(str(path)+'.done.json').exists()
            episodes = list(rows(path))
            successes = [e for e in episodes if e['outcome'] == 'success']
            candidates = []
            for ep in episodes:
                if ep['outcome'] == 'success':
                    continue
                chosen, first_unsafe = None, next((s['step'] for s in ep['steps'] if s['n_safe'] == 0), None)
                if ep['outcome'] == 'timeout':
                    for row in ep['steps']:
                        good = any(c >= .2 and p >= .02 for c,p in zip(row['clearance'],row['progress']))
                        if row['step'] >= 20 and row['goal_distance'] > 1. and good and row['progress'][row['chosen']] < .02:
                            chosen = row
                            break
                elif first_unsafe is not None:
                    for row in ep['steps'][max(0,first_unsafe-12):first_unsafe]:
                        if any(c >= .2 and p >= .02 for c,p in zip(row['clearance'],row['progress'])) and len(set(alternatives(row).values())) > 1:
                            chosen = row
                            break
                d = dict(model=model, cell=cell, case=ep['case'], outcome=ep['outcome'], has_root=chosen is not None,
                         first_all_unsafe=first_unsafe, baseline_return=ep['return'])
                denominator.append(d)
                if chosen is None:
                    continue
                k = chosen['step']
                root = dict(model=model, cell=cell, case=ep['case'], step=k, type=ep['outcome'],
                    labels=alternatives(chosen), baseline_return=ep['return'], baseline_outcome=ep['outcome'],
                    first_all_unsafe=first_unsafe,
                    sustained_stagnation=k>=8 and ep['steps'][k-8]['goal_distance']-chosen['goal_distance']<.1,
                    snapshot=chosen, matched_control=None)
                candidates.append(root)
            used = set()
            for root in candidates:
                a = root['snapshot']
                options = []
                for ep in successes:
                    if ep['case'] in used:
                        continue
                    for b in ep['steps']:
                        if root['type'] == 'timeout':
                            eligible = (b['step']>=20 and b['goal_distance']>1. and b['progress'][b['chosen']]<.02
                                and any(c>=.2 and q>=.02 for c,q in zip(b['clearance'],b['progress'])))
                            if not eligible:
                                continue
                        elif not any(c>=.2 and q>=.02 for c,q in zip(b['clearance'],b['progress'])):
                            continue
                        diffs = [abs(b['step']-a['step'])/8.,abs(b['goal_distance']-a['goal_distance']),
                                 abs(b['n_safe']-a['n_safe'])/16.,abs(b['crowd2m']-a['crowd2m'])/2.]
                        if max(diffs) <= 1. and len(set(alternatives(b).values())) > 1:
                            options.append((sum(diffs),ep['case'],b['step'],ep,b,diffs))
                if options:
                    _, case, k, ep, row, diffs = min(options,key=lambda x:x[:3])
                    used.add(case)
                    control = dict(model=model, cell=cell, case=case, step=k, type='success-control',
                        labels=alternatives(row), baseline_return=ep['return'], baseline_outcome='success',
                        snapshot=row, matched_failure_case=root['case'], match_standardized_differences=diffs)
                    root['matched_control'] = [case,k]
                    roots.append(control)
                roots.append(root)
            print(dict(model=model,cell=cell,episodes=len(episodes),failures=len(candidates),matched=len(used)),flush=True)
    roots.sort(key=lambda r: ((0 if r['cell']==5 and r['model'].startswith('kda') else
                              1 if r['cell']==5 else 2 if r['cell']==4 else 3),r['model'],r['case'],r['step']))
    for i,r in enumerate(roots):
        r['root_id'] = i
    frozen = dict(roots=roots, all_failures=denominator, no_root_count=sum(not r['has_root'] for r in denominator),
                  frozen_before_interventions=time.time(), protocol_sha256=REF.sha(BASE/'manifest.json'))
    save(BASE/'work/roots.json',frozen)
    save(BASE/'work/roots.sha256.json',{'sha256':REF.sha(BASE/'work/roots.json')})


def counterfactual(model, cell):
    frozen = BASE/'work/roots.json'
    assert REF.sha(frozen)==json.loads((BASE/'work/roots.sha256.json').read_text())['sha256']
    selected_roots = [r for r in json.loads(frozen.read_text())['roots'] if r['model']==model and r['cell']==cell]
    refs = {e['case']: e for e in rows(BASE/'work'/('baseline-%s-cell%d.jsonl.gz'%(model,cell)))}
    item,cfg,net=load(model)
    outpath=BASE/'work'/('branches-%s-cell%d.jsonl.gz'%(model,cell))
    assert not outpath.exists()
    with torch.no_grad(),gzip.open(outpath,'wt',compresslevel=1) as f:
        for root in selected_roots:
            if time.time()>P['started_unix']+P['deadline_seconds']-900:
                write(f,dict(record_type='budget_stop',next_root_id=root['root_id']))
                break
            a0=root['labels']['A0']
            unique=[a0]+sorted(set(root['labels'].values())-{a0})
            branches=[]
            for index in unique:
                ep=rollout(cfg,net,model,cell,root['case'],reference=refs[root['case']],root=root,selected=index)
                if index==a0:
                    assert ep['trajectory_sha256']==refs[root['case']]['trajectory_sha256']
                    assert ep['return']==refs[root['case']]['return']
                labels=[label for label,value in root['labels'].items() if value==index]
                # Preserve full physical continuation, but 80-score arrays only at the root.
                for row in ep['steps']:
                    for key in ('values','rewards','clearance','raw_scores','scores','commands','progress','current_token'):
                        row.pop(key,None)
                ep.update(labels=labels,candidate_index=index)
                branches.append(ep)
            write(f,dict(record_type='counterfactual',root=root,branches=branches))
            print(json.dumps(dict(mode='counterfactual',model=model,cell=cell,root_id=root['root_id'],
                                  case=root['case'],branches=len(branches))),flush=True)
    assert REF.sha(item['weight'])==item['weight_sha256']
    save(str(outpath)+'.done.json',dict(status='PASS',sha256=REF.sha(outpath)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['preflight','baseline','roots','counterfactual'])
    p.add_argument('--model')
    p.add_argument('--cell',type=int,default=5)
    p.add_argument('--n',type=int,default=500)
    a=p.parse_args()
    (BASE/'work').mkdir(exist_ok=True)
    if a.mode=='preflight':preflight(a.model)
    elif a.mode=='baseline':baseline(a.model,a.cell,a.n)
    elif a.mode=='roots':freeze_roots()
    else:counterfactual(a.model,a.cell)


if __name__=='__main__':
    main()
