"""Independent arithmetic and completeness checks on the deliverable stream."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent


def main():
    counts=collections.Counter()
    baselines={}
    outcomes={}
    roots=None
    tested={}
    a0_histories={}
    comparisons=[]
    with gzip.open(BASE/'root_cause_data.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            kind=row['record_type'];counts[kind]+=1
            if kind=='summary':summary=row
            elif kind=='frozen_roots':
                roots={r['root_id']:r for r in row['roots']}
                assert len(roots)==len(row['roots'])
            elif kind=='counterfactual':
                root=row['root'];rid=root['root_id']
                assert rid not in tested and root==roots[rid]
                keys=(root['model'],root['cell'],root['case'])
                assert all(92000>x['case'] or x['case']>92499 for x in row['branches'])
                a0=next(x for x in row['branches'] if 'A0' in x['labels'])
                snap=root['snapshot']
                safe=[i for i in range(80) if snap['clearance'][i]>=.2]
                advance=[i for i in safe if snap['progress'][i]>=.02]
                expected_labels={'A0':snap['chosen']}
                if advance:expected_labels['A1']=max(advance,key=lambda i:(snap['raw_scores'][i],-i))
                if safe:expected_labels['A2']=max(safe,key=lambda i:(snap['progress'][i],-i))
                other=[i for i in safe if i not in expected_labels.values()]
                if other:expected_labels['A3']=max(other,key=lambda i:(snap['scores'][i],-i))
                assert expected_labels==root['labels'], 'Predefined candidates changed'
                a0_histories[keys]=[(s['history_sha256'],s['windows_sha256'],s['command']) for s in a0['steps']]
                seen=set()
                for branch in row['branches']:
                    assert branch['candidate_index'] not in seen
                    seen.add(branch['candidate_index'])
                    G=sum(.99**i*s['reward'] for i,s in enumerate(branch['steps']))
                    Q=sum(.99**i*s['reward'] for i,s in enumerate(branch['steps'][root['step']:]))
                    assert abs(G-branch['return'])<1e-12
                    assert abs(Q-branch['root_return'])<1e-12
                    assert branch['time']==len(branch['steps'])*.25
                    assert branch['candidate_index']==root['labels'][branch['labels'][0]]
                    for label in branch['labels']:
                        assert root['labels'][label]==branch['candidate_index']
                        if label=='A0':continue
                        comp=row['comparisons'][label]
                        assert abs(comp['delta_real']-(Q-a0['root_return']))<1e-12
                        assert abs(comp['delta_episode_return']-.99**root['step']*comp['delta_real'])<1e-10
                        assert abs(comp['delta_pred']-comp['delta_real']-(comp['successor_reward_error']+comp['future_value_error']))<1e-10
                        assert abs(comp['delta_pred']-comp['delta_real']-(comp['physical_prediction_reward_error']+
                            comp['selection_history_transition_error']+comp['future_value_error']))<1e-10
                        comparisons.append((root,label,comp))
                tested[rid]=(keys,a0['return'],a0['outcome'],a0['trajectory_sha256'])
            elif kind=='baseline_episode':
                key=(row['model'],row['cell'],row['case'])
                assert key not in baselines
                baselines[key]=(row['return'],row['outcome'],row['trajectory_sha256'])
                if key in a0_histories:
                    assert a0_histories[key]==[(s['history_sha256'],s['windows_sha256'],s['command']) for s in row['steps']], (key,'A0 full history/window mismatch')
                assert abs(row['return']-sum(.99**i*s['reward'] for i,s in enumerate(row['steps'])))<1e-12
                outcomes.setdefault((row['model'],row['cell']),collections.Counter())[row['outcome']]+=1
    assert len(tested)==summary['completed_roots']
    for rid,(key,G,event,trace) in tested.items():
        assert baselines[key]==(G,event,trace)
    for key,c in outcomes.items():
        target=summary['baseline_totals']['|'.join(map(str,key))]
        assert sum(c.values())==target['n']
        for event in ('success','collision','timeout'):assert c[event]==target[event]
    for key,s in summary['per_alternative'].items():
        model,cell,kind,label=key.split('|')
        xs=[x for r,l,x in comparisons if (r['model'],r['cell'],r['type'],l)==(model,int(cell),kind,label)]
        assert len(xs)==s['n']
        assert sum(x['delta_real']>1e-6 for x in xs)==s['return_improved']
        assert sum(x['delta_real'] < -1e-6 for x in xs)==s['return_worsened']
        assert sum(x['rescued'] and x['original_outcome']=='timeout' for x in xs)==s['timeout_to_success']
        assert sum(x['rescued'] and x['original_outcome']=='collision' for x in xs)==s['collision_to_success']
        assert sum(x['damaged'] for x in xs)==s['success_to_failure']
    h=hashlib.sha256()
    with open(BASE/'root_cause_data.jsonl.gz','rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    print(json.dumps(dict(status='PASS',records=dict(counts),baseline_episodes=len(baselines),
                         completed_roots=len(tested),A0_complete_history_and_windows_checked=len(a0_histories),
                         predefined_candidates_recomputed=True,three_term_error_identity_checked=True,
                         data_sha256=h.hexdigest()),indent=2))


if __name__=='__main__':main()
