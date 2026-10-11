"""One-off legal input reencoding; no new action search or environment rollout."""
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
    protocol={'selector':'existing nearest5, no parameter changes',
              'roots':'all completed real-return-improved raw-nonpositive pairs; same frozen roots',
              'history':'reencode every real frame in prior24 with the same nearest5 rule, then current frame and80 successors anchored once',
              'continuation':'restore original TTC5 history after shadow scoring, execute only proposed original smoothed command; use existing exact branch only',
              'unknown':'argmax outside frozen A0-A3 has no outcome label; no additional rollout',
              'decision':'input-set local causal effect supported only for changed argmax with existing verified branch and higher return; otherwise not supported here; never claim five slots insufficient',
              'scope':'mechanism diagnostic, not resurrection or evaluation of a global nearest5 navigation policy'}
    A.save(BASE/'work/input-probe.start.json',dict(protocol=protocol,script_sha256=A.REF.sha(__file__)))
    from crowd_sim.envs.utils.state import JointState,FullState,ObservableState
    from crowd_sim.envs.utils.action import ActionXY
    from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    output=BASE/'work/input-probe.jsonl.gz'
    assert not output.exists()
    count=0
    with torch.no_grad(),gzip.open(output,'wt',compresslevel=6) as stream:
        for item in A.P['models']:
            _,_,net=A.load(item['id'])
            for cell in (5,4,0,3):
                path=BASE/'work'/('branches-%s-cell%d.jsonl.gz'%(item['id'],cell))
                refs={e['case']:e for e in A.rows(BASE/'work'/('baseline-%s-cell%d.jsonl.gz'%(item['id'],cell)))}
                for record in A.rows(path):
                    if record['record_type']!='counterfactual':continue
                    root=record['root'];k=root['step'];original=root['snapshot'];ep=refs[root['case']]
                    a0=next(b for b in record['branches'] if 'A0' in b['labels'])
                    eligible=any(b['candidate_index']!=a0['candidate_index']
                                 and b['root_return']-a0['root_return']>A.P['numerical_return_tolerance']
                                 and b['first_step']['q_pred']<=a0['first_step']['q_pred']
                                 for b in record['branches'])
                    if not eligible:continue
                    result={}
                    for rule in ('ttc5','nearest5'):
                        tokens=[];id_history=[]
                        for row in ep['steps'][max(0,k-24):k+1]:
                            state=JointState(FullState(*row['robot']),[ObservableState(*h) for h in row['humans']])
                            ids=A.REF.select_indices(state.self_state,state.human_states,rule)
                            raw=MambaRLPolicy._build_joint_state_34(net,state.self_state,[state.human_states[i] for i in ids])
                            token=_batch_joint34_to_tokens_vectorized(raw[None])[0]
                            tokens.append(token);id_history.append(ids)
                        net.reset_episode_stats()
                        net._history=collections.deque(tokens[:-1],maxlen=24)
                        net._last_action=ActionXY(*ep['steps'][k-1]['command']) if k else None
                        net.selection_indices=id_history[-1]
                        # Class implementation does no selection; instance build is pinned to IDs above.
                        out=MambaRLPolicy.score_sarl_candidates(net,state)
                        assert list(net.selection_indices)==id_history[-1]
                        assert out['clearance'].cpu().tolist()==original['clearance']
                        assert out['rewards'].cpu().tolist()==original['rewards']
                        assert [[c.vx,c.vy] for c in out['commands']]==original['commands']
                        assert np.array_equal(out['current_token'],tokens[-1])
                        if rule=='ttc5':
                            assert A.digest(out['windows'])==original['windows_sha256']
                            assert out['values'].cpu().tolist()==original['values']
                            assert out['scores'].argmax().item()==a0['candidate_index']
                        result[rule]={'argmax':out['scores'].argmax().item(),
                            'ids':id_history[-1],'history_ids':id_history,'values':out['values'].cpu().tolist(),
                            'raw_scores':out['raw_scores'].cpu().tolist(),'scores':out['scores'].cpu().tolist(),
                            'windows_sha256':A.digest(out['windows'])}
                    selected=result['nearest5']['argmax']
                    branch=next((b for b in record['branches'] if b['candidate_index']==selected),None)
                    evidence=None if branch is None else dict(candidate_index=selected,labels=branch['labels'],
                        delta_root_return=branch['root_return']-a0['root_return'],
                        original_outcome=a0['outcome'],new_outcome=branch['outcome'],
                        original_root_return=a0['root_return'],new_root_return=branch['root_return'],
                        trajectory_sha256=branch['trajectory_sha256'])
                    A.write(stream,dict(root_id=root['root_id'],model=root['model'],cell=cell,case=root['case'],step=k,
                                        changed=selected!=a0['candidate_index'],outcome_verified=branch is not None,
                                        scoring=result,existing_branch_evidence=evidence))
                    count+=1
            del net
            torch.cuda.empty_cache()
    A.save(BASE/'work/input-probe.done.json',dict(status='PASS',roots=count,sha256=A.REF.sha(output),protocol=protocol))
    print(json.dumps(dict(status='PASS',roots=count)),flush=True)


if __name__=='__main__':
    # Pin before importing project classes in main.
    A.REF.pin(A.P['project'])
    main()
