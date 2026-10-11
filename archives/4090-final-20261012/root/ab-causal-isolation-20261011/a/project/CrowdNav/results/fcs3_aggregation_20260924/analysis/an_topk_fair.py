import json, glob, re, collections
import numpy as np
from scipy import stats
S=(1207,2407,3607,4807); BET=('0.01','0.03','0.1','0.3','1.0')
d=collections.defaultdict(dict)
for f in glob.glob('/home/abc/workspace/topk_fair_20260925/eval/*.json'):
    m=re.match(r'(\d+)_k(\d+)_b(.+?)_(\d+)_(\w+_crossing)_(\d+)\.json', f.split('/')[-1])
    if not m: continue
    j=json.load(open(f))
    if len(j.get('episodes',[]))!=200: continue
    s=j['summary'][0]
    d[(m.group(2),m.group(3),int(m.group(4)),m.group(5),m.group(6))][int(m.group(1))]=(100*s['sr'],100*s['cr'],100*s['tr'])
def ci(v):
    v=np.asarray(v,float); m=v.mean()
    h=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else np.nan
    return m,m-h,m+h
print('=== Top-K 公平比较:每个 k 在开发集独立选 β ===')
sel={}
for k in ('1','2','4','8','0'):
    tot={}
    for b in BET:
        v=[np.mean([x[0] for x in d.get((k,b,n,'circle_crossing','9300000'),{}).values()]) for n in (5,10,20)]
        if all(np.isfinite(v)) and len(v)==3: tot[b]=np.mean(v)
    if tot:
        sel[k]=max(tot,key=tot.get)
        print(f"  k={'全部' if k=='0' else k:<4s} 开发集各β: "+'  '.join(f'{b}:{tot[b]:5.1f}' for b in BET if b in tot)+f"  → 选 β={sel[k]}")
print()
print('=== 测试集(8→4 seed,两场景合并),各 k 用自己的 β ===')
print(f"{'N':>3s} {'k':>6s} {'β':>5s} | {'SR':>6s} {'CR':>6s} {'TO':>6s} | {'全部−k: ΔSR':>22s} {'ΔCR':>22s}")
for n in (20,10,5):
    base=None
    rows={}
    for k in ('0','1','2','4','8'):
        b=sel.get(k)
        if b is None: continue
        v=[]
        for s in S:
            c=d.get((k,b,n,'circle_crossing','9400000'),{}).get(s); q=d.get((k,b,n,'square_crossing','9400000'),{}).get(s)
            if c and q: v.append([(c[i]+q[i])/2 for i in range(3)])
        if len(v)==len(S): rows[k]=np.array(v)
    if '0' not in rows: continue
    for k in ('0','1','2','4','8'):
        if k not in rows: continue
        m=rows[k].mean(axis=0)
        if k=='0':
            print(f'{n:3d} {"全部":>6s} {sel[k]:>5s} | {m[0]:6.1f} {m[1]:6.1f} {m[2]:6.1f} | {"(基准)":>22s} {"":>22s}')
        else:
            ds,_lo,_hi=ci(rows['0'][:,0]-rows[k][:,0]); dc,clo,chi=ci(rows['0'][:,1]-rows[k][:,1])
            print(f'{n:3d} {k:>6s} {sel[k]:>5s} | {m[0]:6.1f} {m[1]:6.1f} {m[2]:6.1f} | '
                  f'{ds:+7.2f} [{_lo:+6.2f},{_hi:+6.2f}] {dc:+7.2f} [{clo:+6.2f},{chi:+6.2f}]')
    print()
