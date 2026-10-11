import json, glob, re, collections
import numpy as np
S=(1207,2407,3607,4807)

# ---------- (1) 安全-保守性 Pareto ----------
d=collections.defaultdict(dict)
for f in glob.glob('/home/abc/workspace/topk_fair_20260925/eval/*_9400000.json'):
    m=re.match(r'(\d+)_k(\d+)_b(.+?)_(\d+)_(\w+_crossing)_9400000\.json', f.split('/')[-1])
    if not m: continue
    j=json.load(open(f))
    if len(j.get('episodes',[]))!=200: continue
    s=j['summary'][0]
    d[(m.group(2),m.group(3),int(m.group(4)),m.group(5))][int(m.group(1))]=(100*s['sr'],100*s['cr'],100*s['tr'])
print('=== (1) 安全-保守性前沿:20 人,两场景合并,4 seed ===')
print(f"{'k':>6s} {'β':>5s} | {'SR':>6s} {'CR':>6s} {'TO':>6s}")
pts=[]
for k in ('1','2','4','8','0'):
    for b in ('0.01','0.03','0.1','0.3','1.0'):
        v=[]
        for s in S:
            c=d.get((k,b,20,'circle_crossing'),{}).get(s); q=d.get((k,b,20,'square_crossing'),{}).get(s)
            if c and q: v.append([(c[i]+q[i])/2 for i in range(3)])
        if len(v)==len(S):
            m=np.mean(v,axis=0); pts.append((k,b,*m))
            print(f'{("全部" if k=="0" else k):>6s} {b:>5s} | {m[0]:6.1f} {m[1]:6.1f} {m[2]:6.1f}')
print('\n  同等碰撞率下谁的超时更低(取 CR 最接近 20% 的配置):')
for k in ('1','2','4','8','0'):
    cand=[p for p in pts if p[0]==k]
    if not cand: continue
    best=min(cand,key=lambda p:abs(p[3]-20.0))
    print(f'    k={("全部" if k=="0" else k):<4s} β={best[1]:<5s} CR={best[3]:5.1f} TO={best[4]:5.1f} SR={best[2]:5.1f}')

# ---------- (3) 运行时开销 ----------
print('\n=== (3) 单步决策耗时(ms,取自评测记录)===')
rt=collections.defaultdict(list)
for f in glob.glob('/home/abc/workspace/fcs3_core_20260925/eval/sarl_*_9400000.json'):
    m=re.match(r'sarl_(\d+)_b(.+?)_(\d+)_(\w+_crossing)_9400000\.json',f.split('/')[-1])
    if not m: continue
    j=json.load(open(f)); s=j['summary'][0]
    rt[(m.group(2),int(m.group(3)))].append((s.get('mean_plan_ms'),s.get('p95_plan_ms')))
print(f"{'N':>3s} | {'原版 mean/p95':>16s} | {'+ACT mean/p95':>16s} | {'增量':>10s}")
for n in (5,10,20):
    a=rt.get(('0',n)); b=rt.get(('0.1',n))
    if not a or not b: continue
    am=np.mean([x[0] for x in a]); ap=np.mean([x[1] for x in a])
    bm=np.mean([x[0] for x in b]); bp=np.mean([x[1] for x in b])
    print(f'{n:3d} | {am:7.2f} / {ap:6.2f} | {bm:7.2f} / {bp:6.2f} | {bm-am:+6.2f} ms')
