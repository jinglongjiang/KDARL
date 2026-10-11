"""Correct offline attribution using the actual next-frame TTC history contract."""
import sys
sys.dont_write_bytecode=True
import collections
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import audit as A

BASE=Path(__file__).resolve().parent


def main():
    assert json.loads((BASE/'work/status.json').read_text())['stage']=='EXPERIMENTS_COMPLETE'
    output=BASE/'work/real-history-oracle.jsonl.gz'
    assert not output.exists()
    protocol=dict(
        reason='Pinned-root IDs with true positions are not necessarily the actually observed next TTC frame. Preserve that control and additionally reconstruct the real next-frame selection.',
        unchanged='All weights, baseline trajectories, roots, executed interventions, rewards and return labels remain unchanged.',
        oracle_control='Original root-pinned true-human oracle must reproduce its saved batch80 values bitwise.',
        actual_oracle='Same actual-history prefix; next real robot/humans encoded by the original current-frame TTC selector at that next state.',
        interpretation='Two-term user decomposition uses actual next history. Three-term split separates physical/reward substitution, selection transition, and remaining future value error.',
        limitation='Offline future information only. No oracle action is executed. This does not establish a deployable improvement or identify training versus input sufficiency.',
        script_sha256=A.REF.sha(__file__))
    A.save(BASE/'work/real-history-oracle.start.json',protocol)
    count=0
    with torch.no_grad(),gzip.open(output,'wt',compresslevel=6) as stream:
        for item in A.P['models']:
            _,_,net=A.load(item['id'])
            from crowd_sim.envs.utils.state import JointState, FullState, ObservableState
            from crowd_sim.envs.utils.action import ActionXY
            from crowd_nav.policy.mamba_rl import MambaRLPolicy
            from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
            for cell in (5,4,0,3):
                refs={e['case']:e for e in A.rows(BASE/'work'/('baseline-%s-cell%d.jsonl.gz'%(item['id'],cell)))}
                path=BASE/'work'/('branches-%s-cell%d.jsonl.gz'%(item['id'],cell))
                for row in A.rows(path):
                    if row['record_type']!='counterfactual':continue
                    root=row['root'];snap=root['snapshot'];k=root['step'];ep=refs[root['case']]
                    history=[np.asarray(s['current_token'],dtype=np.float32) for s in ep['steps'][max(0,k-24):k]]
                    state=JointState(FullState(*snap['robot']),[ObservableState(*h) for h in snap['humans']])
                    net.reset_episode_stats()
                    net._history=collections.deque(history,maxlen=24)
                    net._last_action=ActionXY(*ep['steps'][k-1]['command']) if k else None
                    out=net.score_sarl_candidates(state)
                    assert A.digest(out['windows'])==snap['windows_sha256']
                    assert out['values'].cpu().tolist()==snap['values']
                    a0=next(b for b in row['branches'] if 'A0' in b['labels'])
                    true_humans=a0['first_step']['true_next_humans']
                    assert all(b['first_step']['true_next_humans']==true_humans for b in row['branches'])
                    humans=[ObservableState(*h) for h in true_humans]
                    pinned=[];actual=[];ids=[]
                    for command in out['commands']:
                        robot=net.propagate(state.self_state,command)
                        pinned.append(net._build_joint_state_34(robot,humans))
                        chosen=A.REF.select_indices(robot,humans,'ttc5')
                        ids.append(chosen)
                        actual.append(MambaRLPolicy._build_joint_state_34(net,robot,[humans[i] for i in chosen]))
                    pwindow=out['windows'].clone()
                    pwindow[:,-1]=torch.from_numpy(_batch_joint34_to_tokens_vectorized(np.asarray(pinned))).float().to(net.device)
                    pv=net.forward_value(pwindow)
                    assert pv.cpu().tolist()==a0['first_step']['oracle_values80'],(root['root_id'],'pinned oracle mismatch')
                    window=out['windows'].clone()
                    window[:,-1]=torch.from_numpy(_batch_joint34_to_tokens_vectorized(np.asarray(actual))).float().to(net.device)
                    assert torch.equal(window[:,:-1],out['windows'][:,:-1])
                    values=net.forward_value(window)
                    qdiscount=(.99*values).cpu().tolist()
                    branches=[]
                    for b in row['branches']:
                        i=b['candidate_index'];first=b['first_step']
                        assert ids[i]==first['true_next_ttc_ids'],(root['root_id'],i,'actual next selector mismatch')
                        branches.append(dict(candidate_index=i,labels=b['labels'],
                            value_actual_history=float(values[i].item()),actual_next_ids=ids[i],
                            q_oracle1_actual_history=first['reward']+qdiscount[i],
                            q_oracle1_actual_history_terminal_corrected=first['reward'] if first['terminal'] else first['reward']+qdiscount[i]))
                    A.write(stream,dict(root_id=root['root_id'],model=root['model'],cell=cell,case=root['case'],step=k,
                        root_ids=snap['ids'],actual_next_ids80=ids,values_actual_history80=values.cpu().tolist(),
                        branches=branches,pinned_oracle_bitwise_reproduced=True,
                        actual_window_sha256=A.digest(window),prefix_unchanged=True))
                    count+=1
            del net
            torch.cuda.empty_cache()
    A.save(BASE/'work/real-history-oracle.done.json',dict(status='PASS',roots=count,
        sha256=A.REF.sha(output),protocol=protocol))
    print(json.dumps(dict(status='PASS',roots=count)),flush=True)


if __name__=='__main__':main()
