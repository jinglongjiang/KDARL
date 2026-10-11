import json, glob, re, collections
import numpy as np
from scipy import stats
def load(pat, rx):
    a=collections.defaultdict(dict)
    for f in glob.glob(pat):
        m=re.match(rx, f.split('/')[-1])
        if not m: continue
        d=json.load(open(f))
        if len(d.get('episodes',[]))!=200: continue
        a[m.groups()[1:]][int(m.group(1))]=(100*d['summary'][0]['sr'],100*d['summary'][0]['cr'],100*d['summary'][0]['tr'])
    return a
def ci(v):
    v=np.asarray(v,float); m=v.mean()
    return (m, m-stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)),
               m+stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else (m,np.nan,np.nan)

# ---- CADRL 锁定协议 ----
c=load('/home/abc/workspace/cadrl_locked_20260925/eval/cadrl_*.json',
       r'cadrl_(\d+)_b(.+?)_(\d+)_(\w+_crossing)_(\d+)\.json')
print('=== CADRL 锁定协议 ===')
print('开发集(circle,4 seed)按 β:')
for n in (5,10,20):
    row=[]
    for b in ('0','0.01','0.03','0.1','0.3'):
        r=c.get((b,str(n),'circle_crossing','9300000'),{})
        if r: row.append(f'β={b:<5s}{np.mean([x[0] for x in r.values()]):5.1f}')
    if row: print(f'  {n:2d}人  '+'  '.join(row))
tot={}
for b in ('0.01','0.03','0.1','0.3'):
    v=[np.mean([x[0] for x in c.get((b,str(n),'circle_crossing','9300000'),{}).values()]) for n in (5,10,20)]
    if all(np.isfinite(v)): tot[b]=np.mean(v)
bsel=max(tot,key=tot.get) if tot else None
print(f'  → 跨密度最优 β = {bsel}')
if bsel:
    print('测试集(8 seed,两场景):')
    for n in (5,10,20):
        s0,s1={},{}
        for sc in ('circle_crossing','square_crossing'):
            for k,v in c.get(('0',str(n),sc,'9400000'),{}).items(): s0.setdefault(k,[]).append(v)
            for k,v in c.get((bsel,str(n),sc,'9400000'),{}).items(): s1.setdefault(k,[]).append(v)
        ks=[k for k in sorted(set(s0)&set(s1)) if len(s0[k])==2 and len(s1[k])==2]
        if len(ks)<2: continue
        a0=np.array([np.mean([x[0] for x in s0[k]]) for k in ks]); a1=np.array([np.mean([x[0] for x in s1[k]]) for k in ks])
        m,lo,hi=ci(a1-a0)
        print(f'  {n:2d}人  {a0.mean():6.2f} → {a1.mean():6.2f}   Δ {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  n={len(ks)}')

# ---- severe ACT ----
sv=load('/home/abc/workspace/severe_act_20260925/eval/sarl_*.json',
        r'sarl_(\d+)_b(.+?)_(\d+)_(\w+_crossing)_(\d+)\.json')
print('\n=== SARL @ severe 协议 ===')
tot={}
for b in ('0.01','0.03','0.1','0.3'):
    v=[np.mean([x[0] for x in sv.get((b,str(n),'circle_crossing','9300000'),{}).values()]) for n in (5,10,20)]
    if all(np.isfinite(v)): tot[b]=np.mean(v)
bs=max(tot,key=tot.get) if tot else None
print('  开发集各 β 跨密度均值:', {k:round(v,2) for k,v in tot.items()}, '→ 选', bs)
if bs:
    allv=[]
    for n in (5,10,20):
        s0,s1={},{}
        for sc in ('circle_crossing','square_crossing'):
            for k,v in sv.get(('0',str(n),sc,'9400000'),{}).items(): s0.setdefault(k,[]).append(v)
            for k,v in sv.get((bs,str(n),sc,'9400000'),{}).items(): s1.setdefault(k,[]).append(v)
        ks=[k for k in sorted(set(s0)&set(s1)) if len(s0[k])==2 and len(s1[k])==2]
        if len(ks)<2: continue
        a0=np.array([np.mean([x[0] for x in s0[k]]) for k in ks]); a1=np.array([np.mean([x[0] for x in s1[k]]) for k in ks])
        m,lo,hi=ci(a1-a0); allv.append((a0.mean(),a1.mean()))
        print(f'  {n:2d}人  {a0.mean():6.2f} → {a1.mean():6.2f}   Δ {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]')
    if allv: print(f'  三密度均值: {np.mean([x[0] for x in allv]):.2f} → {np.mean([x[1] for x in allv]):.2f}')

# ---- Top-K ----
tk=load('/home/abc/workspace/fcs3_topk_20260925/eval/sarl_*.json', r'sarl_(\d+)_k(\d+)_(\d+)_(\w+_crossing)\.json')
print('\n=== 截断对照:只累计威胁最高的 k 个行人(β=0.1,4 seed,两场景)===')
print(f"{'N':>3s} | "+' | '.join(f'k={k:<2s}' for k in ('1','2','4','8','0(全部)'))+' ')
for n in (20,10):
    row=[]
    for k in ('1','2','4','8','0'):
        v=[]
        for s in (1207,2407,3607,4807):
            a=tk.get((k,str(n),'circle_crossing'),{}).get(s); b=tk.get((k,str(n),'square_crossing'),{}).get(s)
            if a and b: v.append((a[0]+b[0])/2)
        row.append(f'{np.mean(v):5.1f}' if v else '  -  ')
    print(f'{n:3d} | '+' | '.join(row))
# 全部 vs k 的配对
print('\n  全部累计 − top-k(配对,20 人):')
for k in ('1','2','4','8'):
    d=[]
    for s in (1207,2407,3607,4807):
        f=lambda kk,sc: tk.get((kk,'20',sc),{}).get(s)
        a=[f('0','circle_crossing'),f('0','square_crossing')]; b=[f(k,'circle_crossing'),f(k,'square_crossing')]
        if all(a) and all(b): d.append((a[0][0]+a[1][0])/2-(b[0][0]+b[1][0])/2)
    if len(d)>1:
        m,lo,hi=ci(d); print(f'    k={k}: {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]')
