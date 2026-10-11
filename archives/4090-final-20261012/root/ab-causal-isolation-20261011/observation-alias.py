"""Legal reset-world alias twins, identical TTC5 history/filter at initial root."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',required=True);p.add_argument('--causal',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();outdir=Path(a.output);outdir.mkdir(parents=True,exist_ok=True)
    assert not (outdir/'plan.json').exists(), 'Do not replace a frozen alias plan'
    sys.path.insert(0,a.causal)
    import audit as A
    A.P['project']=a.project
    roots=json.loads((Path(a.causal)/'work/roots.json').read_text())['roots']
    plan=[];qualification=[];started=time.time()
    def arrays(state):return A.state_arrays(state)
    def modify(env,selected):
        for i,h in enumerate(env.humans):
            if i not in selected:
                x=10000.+20*i
                h.set(x,0.,x,4.,0.,0.,0.)
    with torch.no_grad():
        for item in A.P['models']:
            model=item['id'];item['weight']=str(Path(a.causal)/'inputs'/model/'rl_model_ep10000.pth')
            item['config']=str(Path(a.causal)/'inputs'/model/'config.ini')
            _,cfg,net=A.load(model)
            for scope in ('failure','success-control'):
                accepted=0
                pool=sorted([r for r in roots if r['model']==model and r['cell']==5 and r['step']==0 and (r['type']=='success-control')==(scope=='success-control')],key=lambda r:r['case'])
                for r in pool:
                    if accepted==2:break
                    alternative=next((r['labels'][label] for label in ('A1','A2','A3') if r['labels'][label]!=r['labels']['A0']),None)
                    if alternative is None:continue
                    env,robot=A.REF.environment(cfg,net,5,r['case'])
                    state=A.REF.state_of(env,robot);assert arrays(state)==(r['snapshot']['robot'],r['snapshot']['humans'])
                    original=net.score_sarl_candidates(state);selected=list(net.selection_indices)
                    modify(env,selected)
                    twinstate=A.REF.state_of(env,robot)
                    twin=net.score_sarl_candidates(twinstate)
                    checks={k:bool(torch.equal(original[k],twin[k])) for k in ('windows','values','rewards','clearance','raw_scores','scores')}
                    checks['current_token']=bool(np.array_equal(original['current_token'],twin['current_token']))
                    checks['selection']=list(net.selection_indices)==selected
                    checks['commands']=[[x.vx,x.vy] for x in original['commands']]==[[x.vx,x.vy] for x in twin['commands']]
                    checks['history_empty']=len(net._history)==0
                    qualification.append(dict(root_id=r['root_id'],model=model,case=r['case'],checks=checks))
                    if not all(checks.values()):continue
                    plan.append(dict(root_id=r['root_id'],model=model,case=r['case'],scope=scope,selected=selected,
                        a0=r['labels']['A0'],alternative=alternative,window_sha256=A.digest(original['windows']),
                        root_scores=original['scores'].cpu().tolist(),commands=[[x.vx,x.vy] for x in original['commands']]))
                    accepted+=1
    protocol=dict(question='Can identical legal TTC5 reset history and identical all-human immediate reward/filter require different action preferences because omitted observable crowd differs?',
        design='Original sampled 20Square initial world versus artificial twin moving only omitted humans to independent far lanes. Selected-human states/goals and robot unchanged. Step0 initial padding is a valid full T24 history, not stitched incompatible prior frames. Real ORCA thereafter; no forced human trajectories.',
        decision='If action preference reverses with identical root TTC5/r/filter, information alias is causally demonstrated for these constructed world twins. Without reversal KILL this construction as a preference-ambiguity witness. Natural-distribution major cause requires separate same-world allN navigation benefit.',
        sampling='First2 eligible step0 failure and first2 success controls/model, ordered case; eligibility uses root-only exact checks, no continuation labels.',
        limitation='Artificial twins are not natural test-distribution samples. Moving crowd changes physical task difficulty; preference reversal, not raw SR delta, is the alias evidence. Not a deployable change or an estimated main-cause percentage.',
        original_root_source_sha256=A.REF.sha(Path(a.causal)/'work/roots.json'),
        qualifications=qualification,roots=plan)
    A.save(outdir/'plan.json',protocol)
    results=[]
    with torch.no_grad(),gzip.open(outdir/'records.jsonl.gz','wt') as stream:
        for r in plan:
            for world in ('natural','far-omitted'):
                for label,index in (('A0',r['a0']),('alternative',r['alternative'])):
                    _,cfg,net=A.load(r['model']);env,robot=A.REF.environment(cfg,net,5,r['case'])
                    if world=='far-omitted':modify(env,r['selected'])
                    rewards=[];steps=[]
                    for k in range(201):
                        state=A.REF.state_of(env,robot);action=net.predict(state);row=A.inspect_step(net,state,k)
                        if k==0:
                            assert row['windows_sha256']==r['window_sha256'] and row['ids']==r['selected']
                            assert row['scores']==r['root_scores'] and row['commands']==r['commands']
                            action=net.causal_result['commands'][index]
                            net._last_action=action;net._last_grid_index=index
                        reward,done,info=A.REF.step(env,action);rewards.append(reward)
                        steps.append(dict(step=k,command=[action.vx,action.vy],reward=reward,terminal=done,
                            state=arrays(A.REF.state_of(env,robot))))
                        if done:break
                    result=dict(root_id=r['root_id'],model=r['model'],case=r['case'],scope=r['scope'],world=world,
                        label=label,candidate=index,return_=sum(.99**j*v for j,v in enumerate(rewards)),
                        outcome=A.REF.event(info),time=(k+1)*.25,steps=steps)
                    A.write(stream,result);results.append({key:v for key,v in result.items() if key!='steps'})
                    print(json.dumps(results[-1]),flush=True)
    pairs=[]
    for r in plan:
        delta={w:next(x['return_'] for x in results if x['root_id']==r['root_id'] and x['world']==w and x['label']=='alternative')-next(x['return_'] for x in results if x['root_id']==r['root_id'] and x['world']==w and x['label']=='A0') for w in ('natural','far-omitted')}
        pairs.append(dict(root_id=r['root_id'],model=r['model'],case=r['case'],delta_return=delta,
            preference_reversal=min(delta.values()) < -1e-6 and max(delta.values()) > 1e-6))
    A.save(outdir/'summary.json',dict(status='PASS',roots=len(plan),episodes=len(results),pairs=pairs,results=results,
        reversals=sum(x['preference_reversal'] for x in pairs),wall_seconds=time.time()-started,method_level_progress=False))


if __name__=='__main__':main()
