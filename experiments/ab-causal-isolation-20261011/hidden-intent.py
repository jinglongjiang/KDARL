"""Bounded artificial hidden-goal intervention, never a deployment policy."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    p.add_argument('--causal', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.causal)
    import audit as A
    A.P['project'] = a.project
    roots = json.loads((Path(a.causal)/'work/roots.json').read_text())['roots']
    selected = []
    for item in A.P['models']:
        item['weight'] = str(Path(a.causal)/'inputs'/item['id']/'rl_model_ep10000.pth')
        item['config'] = str(Path(a.causal)/'inputs'/item['id']/'config.ini')
        candidates = [r for r in roots if r['model'] == item['id'] and r['cell'] == 5 and r['type'] != 'success-control' and r['labels']['A0'] != r['labels']['A1']]
        selected.extend(sorted(candidates, key=lambda r:(r['case'],r['step']))[:2])
    outdir = Path(a.output); outdir.mkdir(parents=True, exist_ok=True)
    protocol = dict(status='FROZEN_BEFORE_RUN', roots=[r['root_id'] for r in selected],
        worlds=['original', 'omitted_goals_rotate_plus90', 'omitted_goals_rotate_minus90'],
        root_actions=['A0', 'A1'],
        intervention='At the root only, rotate omitted humans goal-displacement vectors. Positions, velocities, selected-human goals, robot, history, commands, CV rewards and filter are identical. All future motion uses actual ORCA.',
        decision='If the same observed root/action pair reverses its realized-return ordering between hidden-intent worlds, KILL interpreting each realized difference as a universal observable-Q error. Otherwise KILL this bounded hidden-intent explanation, not all POMDP ambiguity.',
        limitations='Artificial hidden-goal change, not a sample from original static-goal distribution; demonstrates sensitivity or a counterexample, not actual-distribution probability or primary causal share. Not a five-slot-vs-all-human input intervention.',
        original_baseline='New independent terminal-fixed deployment; original archival Q/G not mixed.',
        sources={str(Path(a.causal)/'work/roots.json'):A.REF.sha(Path(a.causal)/'work/roots.json'),str(Path(a.project)/'crowd_nav/policy/mamba_rl.py'):A.REF.sha(Path(a.project)/'crowd_nav/policy/mamba_rl.py')})
    A.save(outdir/'protocol.json', protocol)
    started = time.time(); results = []
    with torch.no_grad(), gzip.open(outdir/'records.jsonl.gz', 'wt') as stream:
        for r in selected:
            observations = {}; commands_ref = None
            for world in protocol['worlds']:
                for label in ('A0','A1'):
                    item,cfg,net=A.load(r['model'])
                    env,robot=A.REF.environment(cfg,net,5,r['case'])
                    hist_ref=None; trajectory=[]; total=0.; root_total=0.
                    for k in range(201):
                        state=A.REF.state_of(env,robot)
                        command=net.predict(state); row=A.inspect_step(net,state,k)
                        if k == r['step']:
                            for key in ('robot','humans','commands','history_sha256','windows_sha256'):
                                assert row[key] == r['snapshot'][key], (r['root_id'],key,'terminal fix changed prefix; old root ineligible')
                            signature={key:copy.deepcopy(row[key]) for key in ('robot','humans','ids','commands','history_sha256','windows_sha256','values','rewards','clearance','raw_scores','scores')}
                            if commands_ref is None:
                                commands_ref=signature
                            assert signature==commands_ref, (r['root_id'],world,label,'root-visible contract changed')
                            if world != 'original':
                                sign=1 if 'plus' in world else -1
                                for j,h in enumerate(env.humans):
                                    if j not in row['ids']:
                                        dx,dy=h.gx-h.px,h.gy-h.py
                                        h.gx,h.gy=h.px-sign*dy,h.py+sign*dx
                            command=net.causal_result['commands'][r['labels'][label]]
                            net._last_action=command;net._last_grid_index=r['labels'][label]
                        reward,done,info=A.REF.step(env,command)
                        total+=.99**k*reward
                        if k>=r['step']:
                            root_total+=.99**(k-r['step'])*reward
                        trajectory.append(dict(step=k,command=[command.vx,command.vy],reward=reward,
                            robot=[robot.px,robot.py],humans=A.state_arrays(A.REF.state_of(env,robot))[1],
                            terminal=bool(done)))
                        if done:break
                    result=dict(root_id=r['root_id'],model=r['model'],case=r['case'],root_step=r['step'],
                        world=world,label=label,candidate=r['labels'][label],return_=total,root_return=root_total,
                        outcome=A.REF.event(info),time=(k+1)*.25,root_contract_sha256=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),
                        trajectory=trajectory,weight_sha256=item['weight_sha256'])
                    A.write(stream,result);results.append({key:v for key,v in result.items() if key!='trajectory'})
                    print(json.dumps(results[-1]),flush=True)
            deltas={w:next(x['root_return'] for x in results if x['root_id']==r['root_id'] and x['world']==w and x['label']=='A1')-next(x['root_return'] for x in results if x['root_id']==r['root_id'] and x['world']==w and x['label']=='A0') for w in protocol['worlds']}
            observations[r['root_id']]=deltas
    pairs=[]
    for r in selected:
        delta={w:next(x['root_return'] for x in results if x['root_id']==r['root_id'] and x['world']==w and x['label']=='A1')-next(x['root_return'] for x in results if x['root_id']==r['root_id'] and x['world']==w and x['label']=='A0') for w in protocol['worlds']}
        pairs.append(dict(root_id=r['root_id'],model=r['model'],case=r['case'],delta_G=delta,
            ordering_reversed=min(delta.values()) < -1e-6 and max(delta.values()) > 1e-6))
    A.save(outdir/'summary.json',dict(status='PASS',protocol=protocol,results=results,pairs=pairs,
        reversal_roots=sum(x['ordering_reversed'] for x in pairs),roots=len(pairs),episodes=len(results),
        wall_seconds=time.time()-started,method_level_progress=False))


if __name__=='__main__':main()
