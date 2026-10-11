"""Main-scheduled finite root continuations with legal all-N real-history replacement."""
import argparse
import collections
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('a_closedloop_entry',HERE/'run-a.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def swept_clearances(before, after, command, dt=.25):
    robot,humans=before;h=np.asarray(humans,dtype=np.float64)
    d0=np.asarray(robot[:2],dtype=np.float64)-h[:,:2]
    result={}
    for name,velocity in (('env',h[:,2:4]),('actual',(np.asarray(after[1],dtype=np.float64)[:,:2]-h[:,:2])/dt)):
        dv=np.asarray(command,dtype=np.float64)-velocity
        den=np.sum(dv*dv,axis=1)
        t=np.clip(np.divide(-np.sum(d0*dv,axis=1),den,out=np.zeros_like(den),where=den>0.),0.,dt)
        clear=np.linalg.norm(d0+t[:,None]*dv,axis=1)-robot[4]-h[:,4]
        result[name+'_swept_clearance']=float(np.min(clear))
    return result


def root_decision_snapshot(out, windows, current, input_name):
    def array(value):
        return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
    result=dict(input=input_name,windows_sha256=A.digest(array(windows)),current_input_sha256=A.digest(array(current)),
        window_shape=list(windows.shape),current_input_shape=list(current.shape))
    for key in ('values','raw_scores','scores'):
        result[key]=array(out[key]).tolist()
        assert len(result[key])==80
    return result


def freeze(args):
    assert not args.output.exists(), 'Do not replace a previously frozen root plan'
    data=json.loads((args.data/'work/roots.json').read_text())
    roots=[];matching=[]
    for model in args.models:
        pool=[r for r in data['roots'] if r['model']==model and r['cell']==5 and r['case']%5==4]
        failures=sorted([r for r in pool if r['type']!='success-control'],key=lambda r:(r['case'],r['step'],r['root_id']))[:args.per_scope]
        controls=sorted([r for r in pool if r['type']=='success-control'],key=lambda r:(r['case'],r['step'],r['root_id']))
        assert len(failures)==args.per_scope
        used=set();chosen=[]
        for f in failures:
            a=f['snapshot'];options=[]
            for c in controls:
                if c['case'] in used:continue
                b=c['snapshot']
                has_progress=any(clear>=.2 and progress>=.02 for clear,progress in zip(b['clearance'],b['progress']))
                if not has_progress:continue
                if f['type']=='timeout' and not (b['step']>=20 and b['goal_distance']>1. and b['progress'][b['chosen']]<.02):continue
                diffs=[abs(b['step']-a['step'])/8.,abs(b['goal_distance']-a['goal_distance']),abs(b['n_safe']-a['n_safe'])/16.,abs(b['crowd2m']-a['crowd2m'])/2.]
                if max(diffs)<=1.:options.append((sum(diffs),c['case'],c['step'],c['root_id'],c,diffs))
            match=min(options,key=lambda x:x[:4]) if options else None
            control=match[4] if match else None
            matching.append(dict(model=model,failure_root=f['root_id'],control_root=control['root_id'] if control else None,
                status='matched_heldout_caliper' if control else 'unmatched',standardized_differences=match[5] if match else None))
            roots.append(dict(root_id=f['root_id'],model=model,cell=5,case=f['case'],step=f['step'],scope='failure',
                matched_control_root=control['root_id'] if control else None,matching_status='matched_heldout_caliper' if control else 'unmatched'))
            if control:
                used.add(control['case']);chosen.append((control,'matched_heldout_caliper',f['root_id']))
        for c in controls:
            if len(chosen)>=args.per_scope:break
            if c['case'] not in used:used.add(c['case']);chosen.append((c,'independent_not_matched',None))
        for c,status,failure in chosen:
            roots.append(dict(root_id=c['root_id'],model=model,cell=5,case=c['case'],step=c['step'],scope='success-control',
                matching_status=status,matched_failure_root=failure))
    roots.sort(key=lambda r:(r['model'],r['case'],r['step'],r['root_id']))
    A.save(args.output,dict(roots=roots,roots_sha256=A.sha(args.data/'work/roots.json'),
        rule='case%5==4; first10 failures per model, then nearest heldout success roots under original calipers (time2s/dgoal1/nsafe16/crowd2m2), L1 standardized distance, tie case/step/root_id, no replacement of success cases. Unmatched failures retained; remaining up-to10 success protectors explicitly independent_not_matched. No future labels or readout used.',
        matching=matching,
        per_scope=args.per_scope,models=args.models,arms=args.arms,
        question='Does replacing only legal observable crowd input/history change actual finite navigation continuation beyond matched readouts?',
        GO='Same held-out root set improves success without additional success-control destruction or collisions, with comparable head controls; eligibility only, no method contribution claim.',
        KILL='No protected navigation benefit, or matched TTC5 head explains gain: stop full-N-only candidate.',
        budget=f'{len(roots)} roots x{len(args.arms)} arms, <=201 total env steps each; no optimizer or second round.',
        closedloop_semantics='After frozen prefix, TTC5 or allN input with the SAME original frozen V used at EVERY subsequent step. Linear readout gate failed; MC/pairwise arms forbidden.',
        authorization='Plan only; execution requires explicit main scheduling.'))
    print(json.dumps(dict(frozen_roots=len(roots),arms=args.arms,output=str(args.output))),flush=True)


def run(args):
    if not args.main_authorized:raise SystemExit('Main scheduling required: --main-authorized')
    if args.device.startswith('cuda') and not args.gpu_authorized:raise SystemExit('GPU scheduling required')
    plan=json.loads(args.plan.read_text());assert A.sha(args.data/'work/roots.json')==plan['roots_sha256']
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'closedloop-rows.jsonl.gz').exists(), 'Do not overwrite partial/complete runs'
    selected=[r for r in plan['roots'] if not args.models or r['model'] in args.models]
    if getattr(args,'smoke_first_per_scope',0):
        counts=collections.Counter();smoke=[]
        for r in selected:
            key=(r['model'],r['scope'])
            if counts[key]<args.smoke_first_per_scope:smoke.append(r);counts[key]+=1
        selected=smoke
    arms=args.arms or plan['arms'];assert set(arms)<=set(plan['arms'])
    assert set(arms)<=set(('ttc5-V','allN-V')), 'A4 gate FAIL: learned-head closed loop authorization revoked'
    assert selected
    protocol=dict(plan_sha256=A.sha(args.plan),plan=plan,arms=arms,script_sha256=A.sha(__file__),
        public_baseline='Both public history/terminal fixes active; old checkpoint; no archived-score binding.',
        input='N5 contracts bitwise; legal all-human current/past only, CV candidate future. Same13 fields, same256 spatial/temporal widths, attention and maxpool.',
        controls='Same candidate commands/smoothing, all-human endpoint reward/clearance/filter, terminal mask, environment reward. Only crowd encoder input and explicitly named matched head change.',
        intervention='One full-N selected command then original repaired parent continuation' if getattr(args,'once',False) else 'Recursive selected input/head after root',
        sweep_recording='Read-only synchronous min ||d0+t*(executed robot command-human velocity)||-radii, t in[0,.25]; env uses current human velocity, actual uses realized human displacement/.25. Strict clearance<0 counts collision intervals, not distinct people. No decision changes.',
        incomplete_alias='No artificial omitted-position alias claimed; hidden-goal sensitivity is separate main experiment.')
    A.save(args.output/'closedloop-protocol.json',protocol)
    ref=A.module('a_closedloop_reference',args.data/'reference.py');ref.pin(args.project)
    from crowd_nav.train import load_config
    from crowd_nav import contracts
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized as tokens
    from crowd_sim.envs.utils.action import ActionXY
    C=A.module('a_closedloop_contract',HERE/'all-n-contract.py');all_tokens,_=C.extend_converter(tokens)
    torch.set_num_threads(1);manifest=json.loads((args.data/'manifest.json').read_text())
    original_roots={r['root_id']:r for r in json.loads((args.data/'work/roots.json').read_text())['roots']}
    started=time.time();output_rows=[]

    def physical(state):
        r=state.self_state
        return [r.px,r.py,r.vx,r.vy,r.radius,r.gx,r.gy,r.v_pref,r.theta], [[h.px,h.py,h.vx,h.vy,h.radius] for h in state.human_states]

    with torch.no_grad(),gzip.open(args.output/'closedloop-rows.jsonl.gz','wt') as stream:
        for model in sorted({r['model'] for r in selected}):
            item=next(x for x in manifest['protocol']['models'] if x['id']==model)
            weight=args.inputs/model/'rl_model_ep10000.pth';config=args.inputs/model/'config.ini'
            assert A.sha(weight)==item['weight_sha256'] and A.sha(config)==item['config_sha256']
            cfg=load_config(str(config));torch.manual_seed(419)
            grid=contracts.init_grid_from_cfg(cfg)
            assert contracts.grid_action_dim(grid)==80 and grid['n_speeds']==5 and grid['n_headings']==16
            net=MambaRLPolicy(cfg,device=args.device)
            payload=torch.load(weight,map_location='cpu',weights_only=False)
            weights={k[10:] if k.startswith('_orig_mod.') else k:v for k,v in payload['policy_state'].items()}
            net.load_state_dict(weights,strict=True);net.eval();net.build_action_space(1.)
            assert len(net.action_space)==80
            net.multiagent_training=True;net.use_sarl_predict=True;net.set_phase('test');net.set_training_mode('rl');net.epsilon=0.
            ref.install_selection(net,'ttc5');score_original=net.score_sarl_candidates
            refs={e['case']:e for e in A.rows(args.data/'work'/f'baseline-{model}-cell5.jsonl.gz') if any(r['model']==model and r['case']==e['case'] for r in selected)}
            gate_scopes=set()
            for root_spec in [r for r in selected if r['model']==model]:
                rid=root_spec['root_id'];r=original_roots[rid];ep=refs[r['case']];k0=r['step']
                root_control=None;baseline_row=None
                root_arms=list(arms)
                if root_spec['scope'] not in gate_scopes and 'ttc5-V' in root_arms:
                    root_arms.insert(root_arms.index('ttc5-V')+1,'ttc5-V-repeat')
                    gate_scopes.add(root_spec['scope'])
                for arm in root_arms:
                    is_full=arm.startswith('allN');head_mode=arm.split('-')[1]
                    net.value_head.load_state_dict({'weight':weights['value_head.weight'],'bias':weights['value_head.bias']})
                    head_path=None
                    if head_mode!='V':
                        assert args.heads, 'Matched-head arms require --heads directory'
                        tag=('P4-MC' if head_mode=='MC' else 'P4-pairwise') if is_full else ('P2' if head_mode=='MC' else 'P3')
                        head_path=args.heads/f'{model}-cell5-pred-{tag}.pth'
                        head_payload=torch.load(head_path,map_location='cpu',weights_only=False)
                        net.value_head.load_state_dict(head_payload['weight'])
                    env,robot=ref.environment(cfg,net,5,r['case']);net.epsilon=0.;net.set_phase('test')
                    full_history=collections.deque(maxlen=24);prefix=[]
                    for k in range(k0):
                        state=ref.state_of(env,robot);ra,ha=physical(state);s=ep['steps'][k]
                        assert (ra,ha)==(s['robot'],s['humans']), (rid,arm,k,'restore prefix')
                        prefix.append(np.asarray(s['current_token'],dtype=np.float32))
                        full_history.append(all_tokens(C.state_vector(ra,ha)[None],len(ha))[0])
                        reward,done,info=ref.step(env,ActionXY(*s['command']))
                        assert reward==s['reward'] and done==s['terminal'] and not done
                    state=ref.state_of(env,robot)
                    assert physical(state)==(r['snapshot']['robot'],r['snapshot']['humans'])
                    net._history=collections.deque(prefix,maxlen=24)
                    net._last_action=ActionXY(*ep['steps'][k0-1]['command']) if k0 else None
                    pending={}

                    def score(state):
                        nonlocal root_control
                        out=score_original(state)
                        if not rewards:
                            repeated=score_original(state)
                            for key in ('windows','values','raw_scores','scores','clearance','rewards'):
                                assert torch.equal(out[key],repeated[key]), (rid,arm,'A0 score roundtrip',key)
                            assert A.digest(out['windows'].cpu().numpy())==r['snapshot']['windows_sha256'], 'frozen root window restoration'
                            assert A.digest(out['current_token'])==r['snapshot']['token_sha256'], 'frozen root current token'
                            assert A.digest(np.asarray(list(net._history)))==r['snapshot']['history_sha256'], 'frozen root history'
                            if head_mode=='V':
                                v=out['values'].cpu().numpy();archived=np.asarray(r['snapshot']['values'])
                                if args.device.startswith('cuda'):assert np.array_equal(v,archived), 'frozen original V bitwise'
                                else:assert np.max(abs(v-archived))<2e-5, 'CPU/GPU V tolerance'
                                pending['terminal_raw_difference']=dict(predicted_terminal=out['predicted_terminal'].cpu().tolist(),
                                    old_raw=r['snapshot']['raw_scores'],new_raw=out['raw_scores'].cpu().tolist())
                            control=dict(commands=[[c.vx,c.vy] for c in out['commands']],rewards=out['rewards'].cpu().tolist(),
                                clearance=out['clearance'].cpu().tolist(),terminal=out['predicted_terminal'].cpu().tolist())
                            if root_control is None:root_control=control
                            assert root_control==control, 'Root candidate commands/rpred/filter/terminal changed across input intervention'
                            pending['root_decision']=dict(original_parent=root_decision_snapshot(out,out['windows'],out['current_token'],'ttc5'))
                        ra,ha=physical(state);N=len(ha)
                        current=all_tokens(C.state_vector(ra,ha)[None],N)[0]
                        pending['current']=current
                        active_full=is_full and (not getattr(args,'once',False) or not rewards)
                        if not active_full and N!=5:
                            if not rewards:pending['root_decision']['used_input']=dict(pending['root_decision']['original_parent'])
                            pending['scores']=out['scores'].cpu().numpy()
                            pending['commands']=[[c.vx,c.vy] for c in out['commands']]
                            return out
                        cv=[net.propagate(h,ActionXY(h.vx,h.vy)) for h in state.human_states]
                        cvh=[[h.px,h.py,h.vx,h.vy,h.radius] for h in cv]
                        futures=[]
                        for command in out['commands']:
                            rob=net.propagate(state.self_state,command)
                            nr=[rob.px,rob.py,rob.vx,rob.vy,rob.radius,rob.gx,rob.gy,rob.v_pref,rob.theta]
                            futures.append(C.state_vector(nr,cvh))
                        full_windows=C.candidate_windows(list(full_history)+[current],all_tokens(np.asarray(futures),N))
                        if N==5:
                            assert np.array_equal(full_windows,out['windows'].cpu().numpy()), 'N5 windows control'
                        if active_full:
                            values=net.forward_value(torch.from_numpy(full_windows).to(args.device))
                            if N==5:assert torch.equal(values,out['values']), 'N5 values control'
                            raw=out['rewards']+net.gamma*(~out['predicted_terminal']).to(values.dtype)*values
                            scores=raw.clone();clear=out['clearance'];safe=clear>=net.test_min_clearance
                            if safe.any():scores=scores.masked_fill(~safe,-1e9)
                            scores=scores-net.test_risk_lambda*torch.clamp(net.test_min_clearance-clear,min=0.)
                            scores[0]-=1e-3
                            if N==5:assert torch.equal(scores,out['scores']), 'N5 final score control'
                            out.update(values=values,raw_scores=raw,scores=scores)
                        if not rewards:
                            pending['root_decision']['used_input']=root_decision_snapshot(out,
                                full_windows if active_full else out['windows'],current if active_full else out['current_token'],
                                'allN' if active_full else 'ttc5')
                        pending['scores']=out['scores'].cpu().numpy()
                        pending['commands']=[[c.vx,c.vy] for c in out['commands']]
                        return out

                    net.score_sarl_candidates=score
                    rewards=[];commands=[];times=[];path=0.;minimum=float('inf');oldpos=np.asarray([state.self_state.px,state.self_state.py])
                    steps=[];done=False
                    for k in range(k0,201):
                        state=ref.state_of(env,robot)
                        if args.device.startswith('cuda'):torch.cuda.synchronize()
                        tick=time.perf_counter();action=net.predict(state)
                        if args.device.startswith('cuda'):torch.cuda.synchronize()
                        times.append(time.perf_counter()-tick)
                        full_history.append(pending['current'])
                        before_physical=physical(state)
                        reward,done,info=ref.step(env,action);rewards.append(reward);commands.append([action.vx,action.vy])
                        ns=ref.state_of(env,robot);ra,ha=physical(ns);pos=np.asarray(ra[:2]);path+=float(np.linalg.norm(pos-oldpos));oldpos=pos
                        sweep=swept_clearances(before_physical,(ra,ha),commands[-1])
                        minimum=min(minimum,min(float(np.hypot(h[0]-ra[0],h[1]-ra[1])-h[4]-ra[4]) for h in ha))
                        steps.append(dict(step=k,index=net._last_grid_index,command=commands[-1],reward=reward,terminal=done,
                            **sweep,
                            physical_state_sha256=A.digest(np.concatenate([np.asarray(ra,dtype=np.float64),np.asarray(ha,dtype=np.float64).ravel()])),
                            actor_history_sha256=A.digest(np.asarray(list(net._history))),
                            real_history_sha256=A.digest(np.asarray(list(full_history))),
                            scores_sha256=A.digest(pending['scores'])))
                        if done:break
                    net.score_sarl_candidates=score_original
                    row=dict(model=model,root_id=rid,case=r['case'],step=k0,scope=root_spec['scope'],arm=arm,restoration_exact=True,
                        root_controls_exact=True,A0_readonly_roundtrip_exact=True,root_candidate_controls=root_control,
                        root_decision=pending['root_decision'],
                        terminal_raw_difference=pending.get('terminal_raw_difference'),
                        head_sha256=A.sha(head_path) if head_path else None,successor_input='allN' if is_full else 'ttc5',
                        outcome=ref.event(info) if done else 'timeout',time=(k0+len(rewards))*.25,root_return=sum(.99**j*v for j,v in enumerate(rewards)),
                        path_length=path,minimum_endpoint_clearance=minimum,mean_inference_seconds=float(np.mean(times)),steps=steps)
                    for mode in ('env','actual'):
                        row['min_'+mode+'_swept_clearance']=min(s[mode+'_swept_clearance'] for s in steps)
                        row[mode+'_physical_collision_count']=sum(s[mode+'_swept_clearance']<0. for s in steps)
                    row['physical_collision_count']=row['actual_physical_collision_count']
                    row['continuation_sha256']=A.hashlib.sha256(json.dumps(steps,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    if arm=='ttc5-V-repeat':
                        assert baseline_row and row['continuation_sha256']==baseline_row['continuation_sha256'], 'A0 full-rollout bitwise repeat gate'
                        assert (row['outcome'],row['time'],row['root_return'])==(baseline_row['outcome'],baseline_row['time'],baseline_row['root_return'])
                        A.save(args.output/f'A0-repeat-{model}-{root_spec["scope"]}.json',dict(status='PASS',root_id=rid,
                            continuation_sha256=row['continuation_sha256'],steps=len(steps),exact=True))
                        continue
                    if arm=='ttc5-V':baseline_row=row
                    row['baseline_outcome_new']=baseline_row['outcome'] if baseline_row else None
                    row['rescue_eligible_new_baseline']=bool(baseline_row and baseline_row['outcome']!='success')
                    if getattr(args,'once',False) and is_full and baseline_row and steps[0]['index']==baseline_row['steps'][0]['index']:
                        assert [(s['index'],s['command'],s['reward']) for s in steps]==[(s['index'],s['command'],s['reward']) for s in baseline_row['steps']], 'A0 no-op trajectory not exact'
                        row['A0_noop_trajectory_exact']=True
                    output_rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    print(json.dumps(dict(model=model,root_id=rid,arm=arm,outcome=row['outcome'],wall_seconds=time.time()-started)),flush=True)
    A.save(args.output/'closedloop-summary.json',dict(status='PASS',root_plan_sha256=A.sha(args.plan),continuations=len(output_rows),
        wall_seconds=time.time()-started,method_level_progress=False,scientific_verdict='INCONCLUSIVE',
        note='Raw actual continuations, not an overall unselected navigation success-rate benchmark. Must pair by root and protect success controls.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','run']);p.add_argument('--data',type=Path,default=A.DEFAULT_DATA)
    p.add_argument('--inputs',type=Path,default=A.DEFAULT_DATA/'inputs');p.add_argument('--project',type=Path,default=HERE/'project')
    p.add_argument('--models',nargs='+');p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path)
    p.add_argument('--heads',type=Path);p.add_argument('--arms',nargs='+',choices=['ttc5-V','allN-V','ttc5-MC','allN-MC','ttc5-pairwise','allN-pairwise'])
    p.add_argument('--per-scope',type=int,default=10);p.add_argument('--device',default='cpu')
    p.add_argument('--main-authorized',action='store_true');p.add_argument('--gpu-authorized',action='store_true')
    p.add_argument('--once',action='store_true')
    p.add_argument('--smoke-first-per-scope',type=int,default=0)
    args=p.parse_args()
    if args.command=='freeze':
        args.models=args.models or ['kda-s419-t24','kda-s443-t24','gru-s419-t24','gru-s443-t24']
        args.arms=args.arms or ['ttc5-V','allN-V'];freeze(args)
    else:
        assert args.plan;run(args)
