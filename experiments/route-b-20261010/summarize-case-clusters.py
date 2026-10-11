"""Supplemental descriptive CIs retaining all cells within each repeated case ID."""
import gzip
import json
import numpy as np
from common import BASE, specs, save


def weighted_medians(x,group,counts):
    order=np.argsort(x);xx=x[order];gg=group[order]
    out=[]
    for start in range(0,len(counts),32):
        w=counts[start:start+32,gg]
        cum=w.cumsum(axis=1);total=cum[:,-1]
        low=np.argmax(cum>=((total-1)//2+1)[:,None],axis=1)
        high=np.argmax(cum>=(total//2+1)[:,None],axis=1)
        out.extend((xx[low]+xx[high])/2)
    return np.asarray(out)


def summarize(folder):
    rows=[]
    for cell in range(6):
        for line in gzip.open(folder/('cell%d.steps.jsonl.gz'%cell),'rt'):
            r=json.loads(line)
            rows.append([r['case'],r['v_gap'],r['gap_relative'],r['endpoint']['spread'],
                         r['env_swept']['spread'],r['actual_swept']['spread']])
    a=np.asarray(rows,float);cases,group=np.unique(a[:,0],return_inverse=True)
    assert len(cases)==20
    rng=np.random.default_rng(20261010)
    draws=rng.integers(0,20,size=(2000,20))
    counts=np.stack([np.bincount(d,minlength=20) for d in draws])
    out=dict(case_clusters=20,episodes=120,steps=len(a),metrics={})
    for i,key in enumerate(('v_gap','gap_relative','endpoint_spread','env_swept_spread','actual_swept_spread'),1):
        valid=np.isfinite(a[:,i]);x=a[valid,i]
        boots=weighted_medians(x,group[valid],counts)
        out['metrics'][key]=dict(pooled_step_median=float(np.median(x)),
            case_cluster_bootstrap_ci95=np.percentile(boots,[2.5,97.5]).tolist())
    return out


def main():
    assert (BASE/'DONE.json').is_file(), 'Primary collection must complete first'
    entries={}
    for s in specs():
        entries[s['id']]={mode:summarize(BASE/'p1-margin'/folder/s['id']) for mode,folder in
                         [('own','per-checkpoint'),('shared','shared')]}
    save(BASE/'p1-margin/case-cluster-summary.json',dict(status='已验证通过',
        scope='Supplemental descriptive CIs, not new confirmatory tests or 11 independent runs',
        bootstrap_unit='case ID, jointly retaining all six cells and all steps',
        bootstrap_repeats=2000,checkpoints=entries))
    print(json.dumps(dict(status='PASS',checkpoints=len(entries))))


if __name__=='__main__':main()
