"""Separate V margins from eligible action-score margins; descriptive evidence only."""
import gzip
import json
from collections import Counter
import numpy as np
from common import BASE, save, specs


def describe(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    assert np.isfinite(a).all()
    if not len(a):return dict(n=0,p50=None,p90=None,mean=None)
    return dict(n=len(a),p50=float(np.median(a)),p90=float(np.percentile(a,90)),mean=float(a.mean()))


def clustered_ci(rows,key):
    groups={}
    for r in rows:
        if r[key] is not None:groups.setdefault(r['case'],[]).append(r[key])
    arrays=[np.asarray(v) for _,v in sorted(groups.items())]
    if not arrays:return dict(n_cases=0,ci95=None)
    rng=np.random.default_rng(20261010)
    draws=[]
    for i in range(2000):
        draw=rng.integers(len(arrays),size=len(arrays))
        draws.append(float(np.median(np.concatenate([arrays[j] for j in draw]))))
    return dict(n_cases=len(arrays),ci95=np.percentile(draws,[2.5,97.5]).tolist(),
        estimand='pooled step median, resampling case IDs with all six cells and all within-case steps',
        limit='Only 20 case clusters; descriptive interval, not a formal intervention test.')


def summarize(folder):
    rows=[]
    for cell in range(6):
        meta=json.loads((folder/('cell%d.json'%cell)).read_text())
        assert meta['status']=='已验证通过'
        rr=[json.loads(s) for s in gzip.open(folder/('cell%d.steps.jsonl.gz'%cell),'rt')]
        assert len(rr)==meta['steps']
        rows+=rr
    keys=('decision_gap','fallback_gap','selected_speed','selected_predicted_clearance',
        'selected_actual_endpoint_clearance','selected_actual_swept_clearance','selected_value_rank',
        'value_top5_speed_spread','value_top5_predicted_clearance_spread',
        'value_top2_velocity_difference','decision_top2_speed_difference','decision_top2_velocity_difference')
    out=dict(steps=len(rows),episodes=len({(r['cell'],r['case']) for r in rows}),
        case_clusters=len({r['case'] for r in rows}),
        safe_at_least2_steps=sum(r['n_safe']>=2 for r in rows),
        safe_exactly1_steps=sum(r['n_safe']==1 for r in rows),
        all_unsafe_steps=sum(r['n_safe']==0 for r in rows),
        selected_action_index_counts=dict(sorted(Counter(r['selected_index'] for r in rows).items())),
        metrics={k:describe([r[k] for r in rows]) for k in keys},
        decision_gap_case_cluster_ci=clustered_ci(rows,'decision_gap'),cells={},by_outcome={})
    assert out['safe_at_least2_steps']+out['safe_exactly1_steps']+out['all_unsafe_steps']==len(rows)
    for cell in range(6):
        group=[r for r in rows if r['cell']==cell]
        out['cells'][str(cell)]=dict(steps=len(group),decision_gap=describe([r['decision_gap'] for r in group]),
            all_unsafe_steps=sum(r['n_safe']==0 for r in group))
    for outcome in ('success','collision','timeout'):
        group=[r for r in rows if r['outcome']==outcome]
        out['by_outcome'][outcome]=dict(steps=len(group),episodes=len({(r['cell'],r['case']) for r in group}),
            decision_gap=describe([r['decision_gap'] for r in group]))
    return out


assert json.loads((BASE/'decision-DONE.json').read_text())['status']=='COMPLETED'
result={}
for item in specs():
    result[item['id']]={mode:summarize(BASE/'p1-margin'/folder/item['id'])
        for mode,folder in [('own','decision'),('shared','decision-shared')]}
save(BASE/'p1-margin/decision-summary.json',dict(status='已验证通过',checkpoints=result,
    definitions=dict(decision_gap='Final eligible score gap; at least two safe candidates.',
        single_safe='Gap undefined/null, not infinity or 1e9.',all_unsafe='Separate unmasked fallback gap.'),
    limits=['All results conditional on frozen weights and alpha=0.3.',
        'Small score gaps or clearance spreads do not establish wrong actions or navigation loss.',
        'Cannot attribute cross-configuration commonalities to sparse reward.',
        'Top5 means V ranking; its speed and clearance differences are not labels of action quality.']))
print(json.dumps(dict(status='PASS',checkpoints=len(result))))
