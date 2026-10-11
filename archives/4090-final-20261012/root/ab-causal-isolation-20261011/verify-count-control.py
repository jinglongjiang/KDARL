"""Independent CPU verification of same-root TTC5/fullN/repeated5 continuations."""
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    with gzip.open(p,'rt') as f:
        return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parent)
    a=p.parse_args();base=a.workspace
    formal=base/'a/artifacts/closedloop-complete/closedloop-rows.jsonl.gz'
    control=base/'a/artifacts/count-matched-complete/count-matched-rows.jsonl.gz'
    done=json.loads(control.with_name('count-matched-summary.json').read_text())
    assert done['status']=='PASS' and done['roots']==80 and done['raw_sha256']==sha(control)
    assert done['protocol']['reference_sha256']==sha(formal)
    groups=collections.defaultdict(dict)
    for row in load(formal):
        groups[(row['model'],row['root_id'])][row['arm']]=row
    for row in load(control):
        key=(row['model'],row['root_id'])
        assert key in groups
        if row['arm']=='ttc5-V':
            original=groups[key]['ttc5-V']
            assert all(row[k]==original[k] for k in ('steps','continuation_sha256','root_return','outcome','time','root_decision'))
        else:
            assert row['arm']=='dup5x4-V' and row['root_decision']['used_input']['window_shape']==[80,24,23,13]
            groups[key][row['arm']]=row
    assert len(groups)==80
    records=[];summaries=[]
    spec=importlib.util.spec_from_file_location('a_cpu_summary',base/'a/summarize-a.py')
    S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
    for (model,rid),arms in sorted(groups.items()):
        assert set(arms)=={'ttc5-V','allN-V','dup5x4-V'}
        parent=arms['ttc5-V']
        for arm,row in arms.items():
            assert row['root_candidate_controls']==parent['root_candidate_controls']
            assert abs(sum(.99**j*s['reward'] for j,s in enumerate(row['steps']))-row['root_return'])<1e-12
            assert row['continuation_sha256']==hashlib.sha256(json.dumps(row['steps'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            S.sweep_summary(row,row['steps'])
            decision=row['root_decision']['used_input']
            for k in ('values','raw_scores','scores'):assert len(decision[k])==80
        for arm in ('allN-V','dup5x4-V'):
            row=arms[arm]
            records.append(dict(model=model,root_id=rid,case=row['case'],step=row['step'],scope=row['scope'],arm=arm,
                G0=parent['root_return'],G1=row['root_return'],delta_G=row['root_return']-parent['root_return'],
                original_outcome=parent['outcome'],intervention_outcome=row['outcome'],
                rescue=parent['outcome']!='success' and row['outcome']=='success',
                damage=parent['outcome']=='success' and row['outcome']!='success',
                added_collision=parent['outcome']!='collision' and row['outcome']=='collision',
                first_command_changed=parent['steps'][0]['command']!=row['steps'][0]['command'],
                A0=parent['steps'][0]['command'],A1=row['steps'][0]['command'],
                parent_decision=parent['root_decision']['used_input'],intervention_decision=row['root_decision']['used_input']))
    for model in sorted({r['model'] for r in records}):
        for scope in ('failure','success-control'):
            for arm in ('allN-V','dup5x4-V'):
                rows=[r for r in records if r['model']==model and r['scope']==scope and r['arm']==arm]
                if not rows:continue
                summaries.append(dict(model=model,scope=scope,arm=arm,roots=len(rows),
                    rescues=sum(r['rescue'] for r in rows),damages=sum(r['damage'] for r in rows),
                    added_collision=sum(r['added_collision'] for r in rows),
                    G_improve=sum(r['delta_G']>1e-6 for r in rows),G_worse=sum(r['delta_G']< -1e-6 for r in rows),
                    bootstrap=S.bootstrap(rows,['delta_G'])))
    contrasts=[]
    for (model,rid),arms in sorted(groups.items()):
        contrasts.append(dict(model=model,root_id=rid,case=arms['ttc5-V']['case'],scope=arms['ttc5-V']['scope'],
            delta_G=arms['allN-V']['root_return']-arms['dup5x4-V']['root_return'],
            identical_commands=arms['allN-V']['steps']==arms['dup5x4-V']['steps']))
    comparison=[]
    for model in sorted({r['model'] for r in contrasts}):
        for scope in ('failure','success-control'):
            rows=[r for r in contrasts if r['model']==model and r['scope']==scope]
            if rows:comparison.append(dict(model=model,scope=scope,roots=len(rows),bootstrap=S.bootstrap(rows,['delta_G']),
                identical_trajectories=sum(r['identical_commands'] for r in rows)))
    output=dict(status='PASS',roots=len(groups),sources={str(p.relative_to(base)):sha(p) for p in (formal,control)},
        groups=summaries,allN_minus_duplicate=comparison,root_records=records,
        interpretation='No new deployment gate. Duplication controls repeated-count/self-attention pooling effects, not every possible allN distribution shift. Selected-root effects, not global cause percentages.',
        method_level_progress=False)
    target=base/'a/artifacts/verified-count-control.json'
    target.write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(dict(status='PASS',roots=80,groups=[{k:v for k,v in r.items() if k!='bootstrap'} for r in summaries])))


if __name__=='__main__':main()
