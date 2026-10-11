"""One forced, preselected filtered action; production safety code remains unchanged."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True)
    p.add_argument('--causal',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'plan.json').exists(), 'Do not overwrite frozen plan'
    sys.path.insert(0,a.causal)
    import audit as A
    A.P['project']=a.project
    roots=json.loads((Path(a.causal)/'work/roots.json').read_text())['roots'];selected=[]
    for item in A.P['models']:
        item['weight']=str(Path(a.causal)/'inputs'/item['id']/'rl_model_ep10000.pth')
        item['config']=str(Path(a.causal)/'inputs'/item['id']/'config.ini')
        for scope in ('failure','success-control'):
            candidates=[]
            for r in roots:
                if r['model']!=item['id'] or r['cell']!=5 or (r['type']=='success-control')!=(scope=='success-control'):continue
                s=r['snapshot'];idx=int(np.argmax(s['raw_scores']))
                if s['n_safe']>0 and s['clearance'][idx]<.2 and idx!=r['labels']['A0']:
                    candidates.append((r,idx))
            selected.extend(sorted(candidates,key=lambda x:(x[0]['case'],x[0]['step']))[:2])
    protocol=dict(roots=[dict(root_id=r['root_id'],model=r['model'],case=r['case'],step=r['step'],scope=r['type'],filtered_candidate=i) for r,i in selected],
        rule='First2 original20Square failure and first2 success-control roots/model whose original rawtop was hard-filtered while some candidate was safe. Alternative fixed to original rawtop, no future-based choice.',
        intervention='Only execute that filtered actual command once at root, synchronize last_action, then restore unchanged TTC5 policy and unchanged filter. This is diagnostic action forcing, not a deployable filter modification.',
        GO='At least one preselected filtered action improves realized G: supports a filtered-out return-improving opportunity at that root. Broad protected mechanism requires both KDA seeds improve and no success or physical safety destruction.',
        KILL='If no filtered alternative improves G, stop tested filter-blocked-loss explanation. If any additional collision/success destruction, KILL this bypass as protected solution.',
        limit='Two roots per scope/model cannot estimate main-cause share. An action violating .2 policy margin can be collision-free yet still violate original safety contract. Return improvement does not prove filter is incorrect.',
        baseline='Terminal-fixed separate continuation, same original checkpoint; compare A0 and forced action in same world.',
        source_sha256=A.REF.sha(Path(a.causal)/'work/roots.json'))
    A.save(out/'plan.json',protocol);records=[];started=time.time()
    with torch.no_grad(),gzip.open(out/'records.jsonl.gz','wt') as stream:
        for r,alternative in selected:
            baseline_root=None
            for label in ('A0','filtered'):
                item,cfg,net=A.load(r['model']);env,robot=A.REF.environment(cfg,net,5,r['case'])
                rewards=[];steps=[];minimum=float('inf');root_details=None
                for k in range(201):
                    state=A.REF.state_of(env,robot);action=net.predict(state);row=A.inspect_step(net,state,k)
                    if k==r['step']:
                        for key in ('robot','humans','history_sha256','windows_sha256','commands'):
                            assert row[key]==r['snapshot'][key], (r['root_id'],key,'changed prefix')
                        if baseline_root is None:baseline_root=row
                        assert row==baseline_root, (r['root_id'],'A0 root score/history mismatch')
                        index=row['chosen'] if label=='A0' else alternative
                        assert row['n_safe']>0 and row['clearance'][alternative]<.2
                        action=net.causal_result['commands'][index];net._last_action=action;net._last_grid_index=index
                        root_details=dict(chosen_original=row['chosen'],forced=index,commands=row['commands'],values=row['values'],
                            rewards=row['rewards'],clearance=row['clearance'],raw_scores=row['raw_scores'],scores=row['scores'])
                    env_sweep=A.swept(state,[[action.vx,action.vy]])[0].min()
                    reward,done,info=A.REF.step(env,action);rewards.append(reward)
                    actual=A.swept(state,[[action.vx,action.vy]],A.state_arrays(A.REF.state_of(env,robot))[1])[0].min()
                    if k>=r['step']:minimum=min(minimum,float(actual))
                    steps.append(dict(step=k,command=[action.vx,action.vy],reward=reward,terminal=done,
                        env_swept_clearance=float(env_sweep),actual_swept_clearance=float(actual)))
                    if done:break
                record=dict(root_id=r['root_id'],model=r['model'],case=r['case'],step=r['step'],scope=r['type'],
                    label=label,outcome=A.REF.event(info),time=(k+1)*.25,
                    return_=sum(.99**j*v for j,v in enumerate(rewards)),root_return=sum(.99**j*v for j,v in enumerate(rewards[r['step']:])),
                    minimum_actual_swept_after_root=minimum,root_details=root_details,steps=steps)
                A.write(stream,record);records.append({key:v for key,v in record.items() if key not in ('root_details','steps')})
                print(json.dumps(records[-1]),flush=True)
    pairs=[]
    for r,_ in selected:
        z=next(x for x in records if x['root_id']==r['root_id'] and x['label']=='A0')
        b=next(x for x in records if x['root_id']==r['root_id'] and x['label']=='filtered')
        pairs.append(dict(root_id=r['root_id'],model=r['model'],case=r['case'],scope=r['type'],
            delta_G=b['root_return']-z['root_return'],rescued=z['outcome']!='success' and b['outcome']=='success',
            destroyed=z['outcome']=='success' and b['outcome']!='success',new_collision=z['outcome']!='collision' and b['outcome']=='collision',
            actual_margin_difference=b['minimum_actual_swept_after_root']-z['minimum_actual_swept_after_root']))
    A.save(out/'summary.json',dict(status='PASS',protocol=protocol,pairs=pairs,records=records,roots=len(selected),episodes=len(records),wall_seconds=time.time()-started))


if __name__=='__main__':main()
