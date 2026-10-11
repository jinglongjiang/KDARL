"""P1 only: frozen own-policy episodes and same-window shadow inference."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from common import BASE, PROJECT, CELLS, harness, specs, save, sha


def install(net, ref):
    build, fast, legacy = net._build_joint_state_34, net.score_sarl_candidates, net._score_sarl_candidates_legacy
    net.selection_indices = None
    frozen = [None]

    def selected_build(robot, humans):
        assert net.selection_indices == frozen[0]
        return build(robot, [humans[i] for i in frozen[0]])

    def score(state):
        frozen[0] = ref.select_indices(state.self_state, state.human_states, 'ttc5')
        net.selection_indices = frozen[0][:]
        old = [x.copy() for x in net._history]
        result = fast(state)
        assert result['windows'].shape == (80, net.seq_len, 8, 13)
        assert len(result['commands']) == 80 and frozen[0] == net.selection_indices
        assert len(old) == len(net._history) and all(np.array_equal(x,y) for x,y in zip(old, net._history))
        for key in ('values', 'raw_scores', 'rewards', 'clearance', 'scores'):
            assert torch.isfinite(result[key]).all(), key
        net.route_result = result
        return result

    net._build_joint_state_34 = selected_build
    net.score_sarl_candidates = score
    net.route_legacy = legacy


def load(item):
    ref = harness()
    ref.install_selection = lambda net, rule: None
    args = SimpleNamespace(project=str(PROJECT), config=item['config'],
        backbone=item['backbone'], arm_dir=str(Path(item['weight']).parent),
        human_select='ttc5', expected_checkpoint_sha=item['weight_sha256'])
    cfg, net, digest = ref.load(args)
    assert net.seq_len == item['length'] and net.test_action_smoothing == .3
    assert net.test_risk_lambda == .8 and len(net.action_space) == 80
    install(net, ref)
    return ref, cfg, net, digest


def swept(robot, current, commands, velocities):
    origin = np.array([robot.px, robot.py])[None,None,:] - current[None,:,:2]
    relative = commands[:,None,:] - velocities[None,:,:]
    den = np.sum(relative**2, axis=-1)
    times = np.clip(-np.sum(origin*relative, axis=-1)/np.maximum(den,1e-30), 0., .25)
    distance = np.linalg.norm(origin + relative*times[...,None], axis=-1)
    return (distance - robot.radius - current[None,:,4]).min(axis=1)


def physical(root, humans, commands):
    assert humans and len(humans) == len(root.human_states)
    cur = np.array([[h.px,h.py,h.vx,h.vy,h.radius] for h in root.human_states])
    nxt = np.array([[h.px,h.py,h.vx,h.vy,h.radius] for h in humans])
    cmd = np.array([[a.vx,a.vy] for a in commands])
    robot = root.self_state
    successor = np.array([robot.px,robot.py]) + cmd*.25
    endpoint = (np.linalg.norm(successor[:,None,:]-nxt[None,:,:2],axis=-1)
                - robot.radius - nxt[None,:,4]).min(axis=1)
    actual_velocity = (nxt[:,:2]-cur[:,:2])/.25
    return dict(endpoint=endpoint, env_swept=swept(robot,cur,cmd,cur[:,2:4]),
                actual_swept=swept(robot,cur,cmd,actual_velocity))


def metrics(values, raw, geometry):
    order = np.argsort(-values, kind='stable')
    gap = float(values[order[0]]-values[order[1]])
    scale = abs(float(values[order[0]]))
    rr = np.sort(raw)
    row = dict(v_gap=gap, top1_value=float(values[order[0]]),
        gap_relative=gap/scale if scale > 1e-8 else None,
        raw_gap=float(rr[-1]-rr[-2]), value_top5=order[:5].tolist())
    for key, clearances in geometry.items():
        top = clearances[order[:5]]
        row[key] = dict(spread=float(np.ptp(top)), minimum=float(top.min()),
                        maximum=float(top.max()), individual=top.tolist())
    return row


def own(args, item):
    ref, cfg, net, digest = load(item)
    started = time.time(); rows=[]; episode_rows=[]
    folder=BASE/'p1-margin'/('smoke' if args.smoke else 'per-checkpoint')/item['id']
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/('cell%d.json'%args.cell)
    assert not out.exists(), out
    reference=item['id']=='kda-original-t24'
    prefix=[]; successors=[]; common_info=[]
    with torch.no_grad(), gzip.open(folder/('cell%d.steps.jsonl.gz'%args.cell),'wt') as stream:
        env,robot=ref.environment(cfg,net,args.cell,90000)
        assert not robot.visible, 'True pedestrian next step must be robot-command independent'
        for case in range(90000,90000+args.n_cases):
            net.reset_episode_stats(); env.reset(seed=case,options={'test_case':case})
            episode=[]
            for step in range(201):
                root=ref.state_of(env,robot)
                if args.smoke and step==0:
                    fast=net.score_sarl_candidates(root); legacy=net.route_legacy(root)
                    for key in ('windows','values'):
                        assert torch.equal(fast[key],legacy[key]), ('fast/legacy',key)
                    for key in ('rewards','clearance'):
                        assert (fast[key]-legacy[key]).abs().max() <= 2e-15, key
                action=net.predict(root); result=net.route_result
                commands=result['commands']; selected=int(net._last_grid_index)
                assert commands[selected]==action
                indices=net.selection_indices[:]
                reward,done,info=ref.step(env,action)
                geometry=physical(root,[h.get_observable_state() for h in env.humans],commands)
                values=result['values'].detach().cpu().numpy().reshape(-1)
                raw=result['raw_scores'].detach().cpu().numpy().reshape(-1)
                row=metrics(values,raw,geometry)
                row.update(cell=args.cell,case=case,step=step,selection_indices=indices,
                    values=values.tolist(),raw_scores=raw.tolist(),
                    candidate_commands=[[a.vx,a.vy] for a in commands],
                    executed_index=selected,executed_command=[action.vx,action.vy],
                    geometry={k:v.tolist() for k,v in geometry.items()})
                episode.append(row)
                if reference:
                    win=result['windows'].cpu().numpy()
                    assert np.array_equal(win[:,:-1],np.broadcast_to(win[0:1,:-1],win[:,:-1].shape))
                    prefix.append(win[0,:-1].copy()); successors.append(win[:,-1].copy())
                    common_info.append(dict(cell=args.cell,case=case,step=step,
                        raw_rewards=result['rewards'].cpu().tolist(), geometry=row['geometry'],
                        commands=row['candidate_commands'],reference_values=values.tolist(),
                        selection_indices=indices))
                if done:break
            outcome=ref.event(info) if done else 'timeout'
            for row in episode:
                row['outcome']=outcome;row['steps_to_terminal']=step-row['step']
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            episode_rows.append(dict(cell=args.cell,case=case,outcome=outcome,steps=step+1))
            print(json.dumps(dict(checkpoint=item['id'],cell=args.cell,case=case,steps=step+1)),flush=True)
    if reference:
        np.savez_compressed(folder/('cell%d.shared.npz'%args.cell),
            prefix=np.asarray(prefix),successors=np.asarray(successors))
        with gzip.open(folder/('cell%d.shared.jsonl.gz'%args.cell),'wt') as f:
            for row in common_info:f.write(json.dumps(row,allow_nan=False)+'\n')
    assert digest==sha(item['weight']) and item['config_sha256']==sha(item['config'])
    assert len(episode_rows)==args.n_cases
    save(out,dict(status='已验证通过',checkpoint=item,episodes=episode_rows,
        steps=sum(x['steps'] for x in episode_rows),wall_seconds=time.time()-started,
        torch=torch.__version__,device=torch.cuda.get_device_name(),
        selection='ttc5',alpha=.3,actions=80,smoke=args.smoke,
        fast_legacy_checked=args.smoke,per_step_file=str(folder/('cell%d.steps.jsonl.gz'%args.cell))))


def shared(args,item):
    ref,cfg,net,digest=load(item)
    source=BASE/'p1-margin'/('smoke' if args.smoke else 'per-checkpoint')/'kda-original-t24'
    data=np.load(source/('cell%d.shared.npz'%args.cell),allow_pickle=False)
    info=[json.loads(x) for x in gzip.open(source/('cell%d.shared.jsonl.gz'%args.cell),'rt')]
    assert len(info)==len(data['prefix'])==len(data['successors'])
    folder=BASE/'p1-margin'/('shared-smoke' if args.smoke else 'shared')/item['id']
    folder.mkdir(parents=True,exist_ok=True);out=folder/('cell%d.json'%args.cell)
    assert not out.exists(),out
    started=time.time();exact=0
    with torch.no_grad(),gzip.open(folder/('cell%d.steps.jsonl.gz'%args.cell),'wt') as stream:
        for i,record in enumerate(info):
            pref=data['prefix'][i][-(item['length']-1):]
            succ=data['successors'][i]
            windows=np.concatenate([np.broadcast_to(pref[None],(80,)+pref.shape),succ[:,None]],axis=1)
            x=torch.from_numpy(np.ascontiguousarray(windows)).to(net.device)
            values=net.forward_value(x).detach().cpu().numpy().reshape(-1)
            assert np.isfinite(values).all()
            if item['id']=='kda-original-t24':
                assert np.array_equal(values,np.array(record['reference_values'],dtype=values.dtype))
                exact+=1
            geometry={k:np.asarray(v) for k,v in record['geometry'].items()}
            raw=np.asarray(record['raw_rewards'])+.99*values
            row=metrics(values,raw,geometry)
            row.update(cell=args.cell,case=record['case'],step=record['step'],values=values.tolist())
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    assert digest==sha(item['weight'])
    save(out,dict(status='已验证通过',checkpoint=item,steps=len(info),
        episodes=len({r['case'] for r in info}),wall_seconds=time.time()-started,
        same_reference_values_bitwise_steps=exact,reference='kda-original-t24',
        alpha=.3,selection='ttc5',actions=80,per_step_file=str(folder/('cell%d.steps.jsonl.gz'%args.cell))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--cell',type=int,required=True)
    p.add_argument('--n-cases',type=int,default=20);p.add_argument('--smoke',action='store_true')
    p.add_argument('--shared',action='store_true');args=p.parse_args()
    assert args.cell in range(6) and args.n_cases in (3,20)
    item=next(x for x in specs() if x['id']==args.id)
    (shared if args.shared else own)(args,item)
