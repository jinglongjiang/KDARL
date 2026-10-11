import json, glob, re, collections
import numpy as np
from scipy import stats
D='/home/abc/workspace/fcs3_core_20260925/eval/'
rec=collections.defaultdict(dict)
for f in glob.glob(D+'*.json'):
    m=re.match(r'(\w+?)_(\d+)_b(.+?)_(\d+)_(\w+_crossing)_(\d+)\.json', f.split('/')[-1])
    if not m: continue
    d=json.load(open(f))
    if len(d.get('episodes',[]))!=200: continue
    ctrl,seed,beta,n,scene,off=m.group(1),int(m.group(2)),float(m.group(3)),int(m.group(4)),m.group(5),int(m.group(6))
    rec[(ctrl,beta,n,scene,off)][seed]=(100*d['summary'][0]['sr'],100*d['summary'][0]['cr'],100*d['summary'][0]['tr'])

print('=== 阶段一:开发集(case 9300000)选 β,circle,4 seed ===')
best={}
for ctrl in ('sarl','lstm'):
    print(f'-- {ctrl} --')
    for n in (5,10,20):
        line=[]; cand=[]
        for b in (0,0.01,0.03,0.1,0.3):
            r=rec.get((ctrl,b,n,'circle_crossing',9300000),{})
            if not r: continue
            sr=np.mean([v[0] for v in r.values()]); line.append(f'β={b:<5g}{sr:5.1f}')
            if b>0: cand.append((sr,b))
        if cand:
            best[(ctrl,n)]=max(cand)[1]
            print(f'  {n:2d}人  '+'  '.join(line)+f'   → 选 β={max(cand)[1]:g}')
# 开发集统一选一个 β(跨密度)
for ctrl in ('sarl','lstm'):
    tot={}
    for b in (0.01,0.03,0.1,0.3):
        v=[np.mean([x[0] for x in rec.get((ctrl,b,n,'circle_crossing',9300000),{}).values()]) for n in (5,10,20)]
        if all(np.isfinite(v)) and len(v)==3: tot[b]=np.mean(v)
    if tot:
        bsel=max(tot,key=tot.get); best[(ctrl,'all')]=bsel
        print(f'  {ctrl}: 跨密度平均最优 β = {bsel:g}  ({tot[bsel]:.2f}%)')

print('\n=== 阶段二:测试集(case 9400000),8 seed,两场景合并,β 由开发集选定 ===')
print(f"{'方法':>6s} {'N':>3s} | {'基线 SR/CR/TO':>20s} | {'+ACT SR/CR/TO':>20s} | {'ΔSR (95% CI)':>26s}")
for ctrl in ('sarl','lstm'):
    b=best.get((ctrl,'all'))
    if b is None: continue
    for n in (5,10,20):
        s0={}; s1={}
        for sc in ('circle_crossing','square_crossing'):
            for seed,v in rec.get((ctrl,0,n,sc,9400000),{}).items(): s0.setdefault(seed,[]).append(v)
            for seed,v in rec.get((ctrl,b,n,sc,9400000),{}).items(): s1.setdefault(seed,[]).append(v)
        ks=sorted(set(s0)&set(s1))
        ks=[k for k in ks if len(s0[k])==2 and len(s1[k])==2]
        if len(ks)<2: continue
        a0=np.array([[np.mean([x[j] for x in s0[k]]) for j in range(3)] for k in ks])
        a1=np.array([[np.mean([x[j] for x in s1[k]]) for j in range(3)] for k in ks])
        d=a1[:,0]-a0[:,0]; m=d.mean(); h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        print(f'{ctrl:>6s} {n:3d} | {a0[:,0].mean():6.2f}/{a0[:,1].mean():5.2f}/{a0[:,2].mean():5.2f}      '
              f'| {a1[:,0].mean():6.2f}/{a1[:,1].mean():5.2f}/{a1[:,2].mean():5.2f}      '
              f'| {m:+7.2f} [{m-h:+7.2f},{m+h:+7.2f}]  n={len(ks)}  (β={b:g})')
