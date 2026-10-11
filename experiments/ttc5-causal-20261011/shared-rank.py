"""Read-only same-window comparison; labels remain the reference policy's Qpi."""
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
    nets={}
    for model in A.P['models']:
        _,_,net=A.load(model['id'])
        nets[model['id']]=net
    from crowd_sim.envs.utils.state import JointState, FullState, ObservableState
    from crowd_sim.envs.utils.action import ActionXY
    path=BASE/'work/shared-rank.jsonl.gz'
    assert not path.exists()
    count=0
    with torch.no_grad(),gzip.open(path,'wt',compresslevel=6) as out:
        for source in sorted((BASE/'work').glob('branches-*.jsonl.gz')):
            # Load only the source model/cell's baseline history, never fabricate actor tracks.
            key=source.name[len('branches-'):-len('.jsonl.gz')]
            refs={e['case']:e for e in A.rows(BASE/'work'/('baseline-'+key+'.jsonl.gz'))}
            for record in A.rows(source):
                if record['record_type']!='counterfactual':continue
                root=record['root'];k=root['step'];snapshot=root['snapshot']
                a0=next(b for b in record['branches'] if 'A0' in b['labels'])
                improved=[b for b in record['branches'] if b['candidate_index']!=a0['candidate_index']
                          and b['root_return']-a0['root_return']>A.P['numerical_return_tolerance']]
                if not improved:continue
                reference=refs[root['case']]
                history=[np.asarray(s['current_token'],dtype=np.float32) for s in reference['steps'][max(0,k-24):k]]
                state=JointState(FullState(*snapshot['robot']),[ObservableState(*h) for h in snapshot['humans']])
                scores={}
                for name,net in nets.items():
                    net.reset_episode_stats()
                    net._history=collections.deque([x.copy() for x in history],maxlen=24)
                    net._last_action=ActionXY(*reference['steps'][k-1]['command']) if k else None
                    data=net.score_sarl_candidates(state)
                    assert A.digest(data['windows'])==snapshot['windows_sha256'],(name,root['root_id'],'window mismatch')
                    if name==root['model']:
                        assert data['values'].cpu().tolist()==snapshot['values'],(name,root['root_id'],'native values mismatch')
                        assert data['raw_scores'].cpu().tolist()==snapshot['raw_scores']
                    scores[name]={'values':data['values'].cpu().tolist(),'raw_scores':data['raw_scores'].cpu().tolist(),
                                  'scores':data['scores'].cpu().tolist()}
                for b in improved:
                    i,j=a0['candidate_index'],b['candidate_index']
                    item=dict(root_id=root['root_id'],reference_model=root['model'],cell=root['cell'],case=root['case'],
                              step=k,original_action=i,alternative_action=j,labels=b['labels'],
                              reference_delta_real=b['root_return']-a0['root_return'],
                              reference_outcomes=[a0['outcome'],b['outcome']],
                              raw_gap_by_model={name:d['raw_scores'][j]-d['raw_scores'][i] for name,d in scores.items()},
                              full_score_gap_by_model={name:d['scores'][j]-d['scores'][i] for name,d in scores.items()},
                              scores=scores,history_sha256=A.digest(np.asarray(history)),
                              windows_sha256=snapshot['windows_sha256'],
                              limitation='Qpi labels use reference checkpoint continuation, not the compared checkpoint continuation; not backbone causal isolation')
                    A.write(out,item);count+=1
    A.save(BASE/'work/shared-rank.done.json',dict(status='PASS',pairs=count,sha256=A.REF.sha(path),
                                               script_sha256=A.REF.sha(Path(__file__))))
    print(json.dumps(dict(status='PASS',pairs=count)),flush=True)


if __name__=='__main__':main()
