"""Paired 5Circle t=5s continuations: old/fixed history x train/deploy.

Frozen weights, one shared forced A0 command, no IL/RL. Run only when main schedules.
"""
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
spec=importlib.util.spec_from_file_location('a_paired_entry',HERE/'run-a.py')
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


def main(args):
    if not args.main_authorized:
        raise SystemExit('Main scheduling required: --main-authorized; no autonomous continuation collection.')
    if args.device.startswith('cuda') and not args.gpu_authorized:
        raise SystemExit('GPU scheduling required: --gpu-authorized')
    assert 1<=args.fit_steps<=300
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'paired5-rows.jsonl.gz').exists(), 'Never overwrite a completed/partial scientific run'
    cases=list(range(90000,90016)); root_step=20
    protocol=dict(question='At identical 5Circle t5s state and forced action, do history omission and train/deploy continuation change targets or actual outcomes?',
        GO='History fix changes held-out continuation regret/outcome in matched train behavior, or target-mode readout consistently improves held-out G in both seeds; eligible only for bounded controlled validation.',
        KILL='No actual outcome/target benefit in matched arms, or linear readout fails held-out calibration: stop the tested continuation-only candidate, not claim impossibility.',
        cases=cases,root_step=root_step,root_time_seconds=5.,denominator_per_model=16,
        arms=['old-train','fixed-train','old-deploy','fixed-deploy'],
        root='Replay original saved commands to step20. Force identical archived A0 command once, append observed current frame once. Compare tails starting at the identical resulting true state.',
        history_intervention='Only post-root epsilon early-return history omission vs append-every-real-frame. Initial complete deployment prefix identical. Both arms keep repaired terminal bootstrap.',
        buffer_contract='ReplayBufferMC push_episode rebuilds continuous real-state windows, including epsilon frames. This intervention changes online actor continuation/actions/labels, NOT buffer frame availability. No buffer or training update is performed.',
        train_behavior='phase=train; frozen epsilon=.05, no smoothing/filter/risk according to original phase rules. Same frozen checkpoint. Not a recreation of all past training checkpoints.',
        eval_contract='All four arms keep net.eval(), so dropout is disabled. Does NOT reproduce original train-mode actor dropout, mixtures of historical checkpoints, or replay state noise. A controlled frozen continuation contract, not complete original training-behavior reproduction. No new intervention.',
        deploy='phase=test; epsilon=0; smoothing=.3, clearance=.2, risk=.8; repaired terminal baseline.',
        rng='Per-model/case fixed private NumPy exploration RNG, restored around predict so environment RNG is not consumed by exploration; paired tape same initial seed.',
        early_terminal='Cases ending before t5s retained in all16 denominator, marked PRE_ROOT_TERMINAL; no replacement roots.',
        budget=f'{len(args.models)} models x16 fixed cases x4 arms; <=201 total env steps/episode; {args.fit_steps} full-batch CPU optimizer steps/head; no IL/RL.',
        readout='Same true next-window latent256, same train/dev/test cases, old V init, linear257 parameters. MC labels independently for each arm. case%5 split; no dev selection.',
        scope='Finite deterministic realizations with one exploration tape, not expected Q or full-learning causal effect.',
        sweep_recording='Read-only synchronous min ||d0+t*(executed robot command-human velocity)||-radii over[0,.25]; env current velocity versus actual displacement/.25. Negative clearance counts collision intervals. Includes forced root command; no decision changes.',
        models=args.models,git_head=A.subprocess.check_output(['git','-C',str(args.project),'rev-parse','HEAD'],text=True).strip(),
        script_sha256=A.sha(__file__),policy_sha256=A.sha(args.project/'crowd_nav/policy/mamba_rl.py'))
    A.save(args.output/'paired5-protocol.json',protocol)
    ref=A.module('paired5_reference',args.data/'reference.py');ref.pin(args.project)
    from crowd_nav.train import load_config
    from crowd_nav import contracts
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized as tokens
    from crowd_sim.envs.utils.action import ActionXY
    torch.set_num_threads(1)
    manifest=json.loads((args.data/'manifest.json').read_text())
    results=[];feature_rows=[];latents=[];started=time.time()

    def physical(env,robot):
        state=ref.state_of(env,robot);r=state.self_state
        return [r.px,r.py,r.vx,r.vy,r.radius,r.gx,r.gy,r.v_pref,r.theta], [[h.px,h.py,h.vx,h.vy,h.radius] for h in state.human_states]

    with torch.no_grad(),gzip.open(args.output/'paired5-rows.jsonl.gz','wt') as stream:
        for model in args.models:
            item=next(x for x in manifest['protocol']['models'] if x['id']==model)
            weight=args.inputs/model/'rl_model_ep10000.pth';config=args.inputs/model/'config.ini'
            assert A.sha(weight)==item['weight_sha256'] and A.sha(config)==item['config_sha256']
            cfg=load_config(str(config));torch.manual_seed(419)
            grid=contracts.init_grid_from_cfg(cfg)
            assert contracts.grid_action_dim(grid)==80 and grid['n_speeds']==5 and grid['n_headings']==16
            net=MambaRLPolicy(cfg,device=args.device)
            payload=torch.load(weight,map_location='cpu',weights_only=False)
            net.load_state_dict({k[10:] if k.startswith('_orig_mod.') else k:v for k,v in payload['policy_state'].items()},strict=True)
            net.eval();net.build_action_space(1.);net.multiagent_training=True;net.use_sarl_predict=True;net.set_training_mode('rl')
            assert len(net.action_space)==80
            assert net.test_action_smoothing==.3 and net.test_min_clearance==.2 and net.test_risk_lambda==.8 and net.seq_len==24
            fixed_predict=net.predict_sarl_style
            old_predict=A.snapshot_method(args.project,'MambaRLPolicy','predict_sarl_style').__get__(net)
            archive={e['case']:e for e in A.rows(args.data/'work'/f'baseline-{model}-cell0.jsonl.gz') if e['case'] in cases}
            assert set(archive)==set(cases)
            model_rows=[]
            for case in cases:
                ep=archive[case];arms=[];canonical=None;canonical_next=None;canonical_z=None;canonical_reward=None
                if len(ep['steps'])<=root_step:
                    row=dict(model=model,case=case,split=A.split(case),status='PRE_ROOT_TERMINAL',baseline_outcome=ep['outcome'],baseline_time=ep['time'],arms_executed=0,denominator_included=True)
                    model_rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();continue
                for history in ('old','fixed'):
                    for behavior in ('train','deploy'):
                        arm=history+'-'+behavior
                        net.predict_sarl_style=fixed_predict if history=='fixed' else old_predict
                        net.set_phase('test');net.epsilon=0.
                        env,robot=ref.environment(cfg,net,0,case)
                        prefix=[]
                        for k in range(root_step):
                            actual=physical(env,robot);saved=ep['steps'][k]
                            assert actual==(saved['robot'],saved['humans']), (model,case,arm,k,'prefix physical restore')
                            current=tokens(net._build_joint_state_34(ref.state_of(env,robot).self_state,ref.state_of(env,robot).human_states)[None])[0]
                            assert np.array_equal(current,np.asarray(saved['current_token'],dtype=np.float32))
                            prefix.append(current)
                            reward,done,info=ref.step(env,ActionXY(*saved['command']))
                            assert reward==saved['reward'] and done==saved['terminal'], (model,case,arm,k,'prefix reward')
                            assert not done
                        restored=physical(env,robot);saved=ep['steps'][root_step]
                        assert restored==(saved['robot'],saved['humans'])
                        root_state=ref.state_of(env,robot)
                        current=tokens(net._build_joint_state_34(root_state.self_state,root_state.human_states)[None])[0]
                        net._history=collections.deque(prefix,maxlen=24)
                        net._last_action=ActionXY(*ep['steps'][root_step-1]['command'])
                        root_hash=A.digest(np.asarray(list(net._history)+[current]))
                        if canonical is None:canonical=(restored,root_hash)
                        assert canonical==(restored,root_hash), 'paired root physical/history mismatch'
                        forced=ActionXY(*saved['command']);net._last_action=forced;net._last_grid_index=saved['chosen']
                        net._history.append(current)
                        reward,done,info=ref.step(env,forced)
                        assert reward==saved['reward'] and done==saved['terminal'], 'forced A0 successor restore'
                        nextphysical=physical(env,robot)
                        if canonical_next is None:canonical_next=nextphysical;canonical_reward=reward
                        assert nextphysical==canonical_next and reward==canonical_reward
                        nextstate=ref.state_of(env,robot)
                        nexttoken=tokens(net._build_joint_state_34(nextstate.self_state,nextstate.human_states)[None])[0]
                        seq=list(net._history)+[nexttoken];seq=seq[-24:];seq=[seq[0]]*(24-len(seq))+seq
                        t=torch.from_numpy(np.asarray(seq,dtype=np.float32)[None]).to(args.device)
                        z=net.temporal_encoder(net.spatial_encoder(t))[:,-1].cpu().numpy()[0]
                        if canonical_z is None:canonical_z=z.copy()
                        assert np.array_equal(z,canonical_z), 'paired readout input not identical'
                        rewards=[reward];commands=[[forced.vx,forced.vy]];history_trace=[]
                        sweep_steps=[dict(step=root_step,command=commands[0],**swept_clearances(restored,nextphysical,commands[0]))]
                        oldpos=np.asarray(nextphysical[0][:2]);path=float(np.linalg.norm(oldpos-np.asarray(restored[0][:2])))
                        minimum=min(float(np.hypot(h[0]-oldpos[0],h[1]-oldpos[1])-h[4]-nextphysical[0][4]) for h in nextphysical[1])
                        random=np.random.RandomState(args.exploration_seed+case*1000+item['seed'])
                        net.set_phase('train' if behavior=='train' else 'test');net.epsilon=.05 if behavior=='train' else 0.
                        latency=[]
                        for k in range(root_step+1,201):
                            if done:break
                            before_physical=physical(env,robot)
                            before=len(net._history);before_hash=A.digest(np.asarray(list(net._history)))
                            env_rng=np.random.get_state();np.random.set_state(random.get_state())
                            tick=time.perf_counter();action=net.predict(ref.state_of(env,robot))
                            if args.device.startswith('cuda'):torch.cuda.synchronize()
                            latency.append(time.perf_counter()-tick)
                            random.set_state(np.random.get_state());np.random.set_state(env_rng)
                            history_trace.append(dict(step=k,before=before,after=len(net._history),before_sha256=before_hash,
                                after_sha256=A.digest(np.asarray(list(net._history))),index=net._last_grid_index))
                            reward,done,info=ref.step(env,action);rewards.append(reward);commands.append([action.vx,action.vy])
                            p,h=physical(env,robot);pos=np.asarray(p[:2]);path+=float(np.linalg.norm(pos-oldpos));oldpos=pos
                            sweep_steps.append(dict(step=k,command=commands[-1],**swept_clearances(before_physical,(p,h),commands[-1])))
                            minimum=min(minimum,min(float(np.hypot(hh[0]-p[0],hh[1]-p[1])-hh[4]-p[4]) for hh in h))
                        row=dict(model=model,case=case,split=A.split(case),arm=arm,status='PASS',root_step=root_step,
                            root_history_sha256=root_hash,restoration_exact=True,forced_command=saved['command'],forced_reward=canonical_reward,
                            true_next_window_sha256=A.digest(np.asarray(seq,dtype=np.float32)),root_return=sum(.99**j*r for j,r in enumerate(rewards)),
                            outcome=ref.event(info) if done else 'timeout',remaining_steps=len(rewards),time=(root_step+len(rewards))*.25,
                            path_length_after_root=path,minimum_endpoint_clearance=None if minimum==float('inf') else minimum,
                            mean_inference_seconds=float(np.mean(latency)) if latency else None,
                            exploration_seed=args.exploration_seed+case*1000+item['seed'],commands=commands,rewards=rewards,history_trace=history_trace,sweep_steps=sweep_steps)
                        row['network_eval_mode']=not net.training
                        row['network_all_modules_eval']=all(not m.training for m in net.modules())
                        for mode in ('env','actual'):
                            row['min_'+mode+'_swept_clearance']=min(s[mode+'_swept_clearance'] for s in sweep_steps)
                            row[mode+'_physical_collision_count']=sum(s[mode+'_swept_clearance']<0. for s in sweep_steps)
                        row['physical_collision_count']=row['actual_physical_collision_count']
                        arms.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                olddeploy=next(x for x in arms if x['arm']=='old-deploy');fixeddeploy=next(x for x in arms if x['arm']=='fixed-deploy')
                assert olddeploy['commands']==fixeddeploy['commands'] and olddeploy['rewards']==fixeddeploy['rewards'], 'epsilon0 protected history control'
                feature_rows.append(dict(model=model,case=case,split=A.split(case),reward=canonical_reward,terminal=saved['terminal'],labels={x['arm']:x['root_return'] for x in arms}))
                latents.append(canonical_z);model_rows.extend(arms)
                A.save(args.output/'paired5-progress.json',dict(model=model,case=case,wall_seconds=time.time()-started))
                print(json.dumps(dict(model=model,case=case,arms=4,wall_seconds=time.time()-started)),flush=True)
            results.extend(model_rows)
    np.savez_compressed(args.output/'paired5-latents.npz',latent256=np.asarray(latents))
    A.save(args.output/'paired5-labels.json',feature_rows)
    # Each root is kept together across all four label modes and both model seeds.
    summaries=[]
    for model in args.models:
        subset=[r for r in results if r['model']==model]
        valid_cases=sorted({r['case'] for r in subset if r['status']=='PASS'})
        pairs=[]
        for case in valid_cases:
            arms={r['arm']:r for r in subset if r['case']==case and r['status']=='PASS'}
            for factor,left,right in (('history_train','old-train','fixed-train'),('history_deploy','old-deploy','fixed-deploy'),('continuation_old','old-train','old-deploy'),('continuation_fixed','fixed-train','fixed-deploy')):
                a,b=arms[left],arms[right]
                pairs.append(dict(case=case,factor=factor,delta_G=b['root_return']-a['root_return'],outcome_changed=b['outcome']!=a['outcome'],
                    success_gained=a['outcome']!='success' and b['outcome']=='success',success_lost=a['outcome']=='success' and b['outcome']!='success',
                    command_trace_changed=a['commands']!=b['commands']))
        outcome_counts={}
        for arm in ('old-train','fixed-train','old-deploy','fixed-deploy'):
            outcomes=[r['outcome'] for r in subset if r.get('arm')==arm]+[r['baseline_outcome'] for r in subset if r['status']=='PRE_ROOT_TERMINAL']
            assert len(outcomes)==16
            outcome_counts[arm]=dict(collections.Counter(outcomes))
        summaries.append(dict(model=model,denominator=16,valid_roots=len(valid_cases),pre_root_terminal=sum(r['status']=='PRE_ROOT_TERMINAL' for r in subset),outcome_counts_all16=outcome_counts,paired_effects=pairs))
    A.save(args.output/'paired5-summary.json',dict(status='COLLECTION_COMPLETE_READOUT_PENDING',protocol=protocol,models=summaries,
        wall_seconds=time.time()-started,method_level_progress=False,scientific_verdict='INCONCLUSIVE',
        limitation='Frozen-contract intervention with one exploration realization; not the causal effect of retraining.'))
    fit(args)


def fit(args):
    labels=json.loads((args.output/'paired5-labels.json').read_text())
    with np.load(args.output/'paired5-latents.npz') as f:latent=f['latent256'].copy()
    manifest=json.loads((args.data/'manifest.json').read_text())
    result=[]
    for model in args.models:
        ids=[i for i,r in enumerate(labels) if r['model']==model]
        records=[labels[i] for i in ids];X=torch.tensor(latent[ids])
        payload=torch.load(args.inputs/model/'rl_model_ep10000.pth',map_location='cpu',weights_only=False)
        weights={k[10:] if k.startswith('_orig_mod.') else k:v for k,v in payload['policy_state'].items()}
        masks={s:torch.tensor([r['split']==s for r in records]) for s in ('train','dev','test')}
        if any(not m.any() for m in masks.values()):
            result.append(dict(model=model,status='INCOMPLETE',reason='Valid-root partition missing; early cases not replaced'));continue
        R=torch.tensor([r['reward'] for r in records],dtype=torch.float32)
        B=.99*torch.tensor([not r['terminal'] for r in records],dtype=torch.float32)
        for arm in ('old-train','fixed-train','old-deploy','fixed-deploy'):
            Y=torch.tensor([r['labels'][arm] for r in records],dtype=torch.float32)
            head=torch.nn.Linear(256,1)
            head.load_state_dict({'weight':weights['value_head.weight'],'bias':weights['value_head.bias']})
            initial=(R+B*head(X).squeeze(-1)).detach()
            optimizer=torch.optim.Adam(head.parameters(),lr=.001)
            for _ in range(args.fit_steps):
                optimizer.zero_grad();p=R+B*head(X).squeeze(-1);loss=((p[masks['train']]-Y[masks['train']])**2).mean();loss.backward();optimizer.step()
            pred=(R+B*head(X).squeeze(-1)).detach()
            metrics=[]
            for partition,m in masks.items():
                metrics.append(dict(split=partition,cases=int(m.sum()),old_V_mse=float(((initial[m]-Y[m])**2).mean()),
                    fit_mse=float(((pred[m]-Y[m])**2).mean()),target_mean=float(Y[m].mean()),target_std=float(Y[m].std(unbiased=False))))
            result.append(dict(model=model,arm=arm,parameters=257,steps=args.fit_steps,metrics=metrics))
            torch.save(head.state_dict(),args.output/(model+'-'+arm+'-head.pth'))
    A.save(args.output/'paired5-readout.json',dict(status='PASS',results=result,
        limitation='At most16 cases/seed, one root action and exploration tape; no full-learning effect claimed.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=A.DEFAULT_DATA);p.add_argument('--inputs',type=Path,default=A.DEFAULT_DATA/'inputs')
    p.add_argument('--project',type=Path,default=HERE/'project');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=['kda-s419-t24','kda-s443-t24','gru-s419-t24','gru-s443-t24'],default=['gru-s419-t24','gru-s443-t24'])
    p.add_argument('--device',default='cpu');p.add_argument('--main-authorized',action='store_true');p.add_argument('--gpu-authorized',action='store_true')
    p.add_argument('--exploration-seed',type=int,default=20261011);p.add_argument('--fit-steps',type=int,default=300)
    main(p.parse_args())
