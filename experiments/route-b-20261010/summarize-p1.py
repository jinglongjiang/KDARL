"""Describe checkpoint margins; never infer objective causality or outcome gains."""
import gzip
import json
import numpy as np
from common import BASE, specs, save


def describe(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=None,p50=None,p90=None,p95=None)
    return dict(n=len(x),mean=float(x.mean()),p50=float(np.median(x)),
                p90=float(np.percentile(x,90)),p95=float(np.percentile(x,95)))


def bootstrap_episode_median(rows,key):
    episodes={}
    for r in rows:
        value=r[key]
        if value is not None:episodes.setdefault((r['cell'],r['case']),[]).append(value)
    med=np.array([np.median(v) for v in episodes.values()])
    rng=np.random.default_rng(20261010)
    draws=np.median(med[rng.integers(0,len(med),size=(2000,len(med)))],axis=1)
    return dict(n_episodes=len(med),median_of_episode_medians=float(np.median(med)),
        episode_bootstrap_ci95=np.percentile(draws,[2.5,97.5]).tolist(),
        estimand='median of episode-specific medians, not pooled-step median',
        limit='episodes from repeated case IDs acrosscells may be dependent; descriptive only')


def summarize(folder):
    rows=[];cells={}
    for cell in range(6):
        result=json.loads((folder/('cell%d.json'%cell)).read_text())
        assert result['status']=='已验证通过'
        rr=[json.loads(line) for line in gzip.open(folder/('cell%d.steps.jsonl.gz'%cell),'rt')]
        assert len(rr)==result['steps']
        rows.extend(rr)
        cells[str(cell)]=dict(steps=len(rr),episodes=len({x['case'] for x in rr}),
            v_gap=describe([r['v_gap'] for r in rr]),relative_gap=describe([r['gap_relative'] for r in rr if r['gap_relative'] is not None]))
    out=dict(steps=len(rows),episodes=len({(r['cell'],r['case']) for r in rows}),
        v_gap=describe([r['v_gap'] for r in rows]),
        relative_gap=describe([r['gap_relative'] for r in rows if r['gap_relative'] is not None]),
        nearzero_value_steps=sum(r['gap_relative'] is None for r in rows),
        raw_r_gammaV_gap=describe([r['raw_gap'] for r in rows]),
        episode_gap=bootstrap_episode_median(rows,'v_gap'),cells=cells,geometry={})
    bins=[(-np.inf,0,'<0'),(0,.2,'0-0.2'),(.2,.4,'0.2-0.4'),(.4,.8,'0.4-0.8'),(.8,np.inf,'>=0.8')]
    for metric in ('endpoint','env_swept','actual_swept'):
        strata={}
        for lo,hi,name in bins:
            group=[r for r in rows if lo<=r[metric]['minimum']<hi]
            strata[name]=dict(steps=len(group),episodes=len({(r['cell'],r['case']) for r in group}),
                spread=describe([r[metric]['spread'] for r in group]))
        out['geometry'][metric]=dict(top5_spread=describe([r[metric]['spread'] for r in rows]),strata=strata)
    return out


def main():
    entries={}
    for item in specs():
        entries[item['id']]=dict(checkpoint=item,
            own=summarize(BASE/'p1-margin/per-checkpoint'/item['id']),
            shared=summarize(BASE/'p1-margin/shared'/item['id']))
    save(BASE/'p1-margin/summary.json',dict(status='已验证通过',checkpoints=entries,
        limits=['Raw successor V margin is not the Q action gap in Bellemare2016',
               'Checkpoint replication cannot isolate causality of the shared training objective',
               'One-step clearance spread does not measure long-run return or value-function error',
               'Same windows hold reference selection and action smoothing fixed; T12 trims to last12',
               'Own-trajectory distributions differ, shared-state comparison is only one reference policy',
               'No new selected actions are executed in shared-state shadow scoring']))
    print(json.dumps(dict(status='PASS',checkpoints=len(entries))))


if __name__=='__main__':main()
