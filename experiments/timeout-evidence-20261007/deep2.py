import json, math, statistics as st
from pathlib import Path
from itertools import combinations
W = Path('/home/abc/temp/cc_6env_20261007')
ARMS = ('mamba','gru','kda')
def load(run, arm):
    return {(e['human_num'],e['geometry'],e['case']): e
            for e in json.load(open(W/run/f'{arm}.json'))['episodes']}
def mcn(b,c):
    n=b+c
    if n==0: return 1.0
    k=min(b,c)
    return min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/2**n)

print('='*104)
print('第5步  差异到底显著吗? 同一 case 组内配对检验')
print('='*104)
GROUPS = [('case 88000: 原版3000 + 续训10000 合并 (384 对)',
           [('kda_3000',), ('kda_premerge_88000',)]),
          ('case 89000: 续训10000 + 优化版10000 合并 (384 对)',
           [('kda_premerge_89000',), ('kda',)])]
for gname, runs in GROUPS:
    print(f"\n  [{gname}]")
    D = {a: {} for a in ARMS}
    for (r,) in runs:
        for a in ARMS:
            for k,v in load(r,a).items():
                D[a][(r,)+k] = v
    n = len(D['mamba'])
    print(f"    {'对比':<22}{'ΔSR':>9}{'b':>5}{'c':>5}{'p(成功)':>10}   {'ΔTO':>8}{'超时数':>12}{'p(超时)':>10}")
    for x,y in combinations(ARMS,2):
        keys = sorted(set(D[x])&set(D[y]))
        b = sum(1 for k in keys if D[x][k]['outcome']=='success' and D[y][k]['outcome']!='success')
        c = sum(1 for k in keys if D[x][k]['outcome']!='success' and D[y][k]['outcome']=='success')
        sx = sum(1 for k in keys if D[x][k]['outcome']=='success')/len(keys)*100
        sy = sum(1 for k in keys if D[y][k]['outcome']=='success')/len(keys)*100
        isto = lambda e: e['outcome'] not in ('success','collision')
        tb = sum(1 for k in keys if isto(D[x][k]) and not isto(D[y][k]))
        tc = sum(1 for k in keys if not isto(D[x][k]) and isto(D[y][k]))
        tx = sum(1 for k in keys if isto(D[x][k]))/len(keys)*100
        ty = sum(1 for k in keys if isto(D[y][k]))/len(keys)*100
        print(f"    {y+' − '+x:<22}{sy-sx:>+8.2f}pp{b:>5}{c:>5}{mcn(b,c):>10.4f}   "
              f"{ty-tx:>+7.2f}pp{f'{tb}/{tc}':>12}{mcn(tb,tc):>10.4f}")

print()
print('='*104)
print('第6步  超时率的"跑间波动"有多大 —— 和臂间差异比')
print('='*104)
RUNS = ['kda_3000','kda_premerge_88000','kda_premerge_89000','kda']
to = {}
for a in ARMS:
    vals=[]
    for r in RUNS:
        eps = list(load(r,a).values()); n=len(eps)
        t = sum(1 for e in eps if e['outcome'] not in ('success','collision'))
        vals.append(t/n*100)
    to[a]=vals
print(f"  {'臂':<8}" + ''.join(f"{r.replace('kda_','').replace('premerge_','@'):>14}" for r in RUNS)
      + f"{'均值':>9}{'极差':>9}{'标准差':>9}")
for a in ARMS:
    v=to[a]
    print(f"  {a:<8}" + ''.join(f"{x:>13.2f}%" for x in v)
          + f"{st.mean(v):>8.2f}%{max(v)-min(v):>8.2f}%{st.stdev(v):>8.2f}%")
print()
allv=[x for a in ARMS for x in to[a]]
print(f"  臂间差异(各臂均值的极差): {max(st.mean(to[a]) for a in ARMS)-min(st.mean(to[a]) for a in ARMS):.2f}pp")
print(f"  跑间波动(单臂内极差的均值): {st.mean(max(to[a])-min(to[a]) for a in ARMS):.2f}pp")

print()
print('='*104)
print('第7步  总失败率(CR+TO)的排序是否稳定')
print('='*104)
print(f"  {'跑':<20}" + ''.join(f"{a:>12}" for a in ARMS) + '   排序(低->高)')
for r in RUNS:
    f={}
    for a in ARMS:
        eps=list(load(r,a).values()); n=len(eps)
        s=sum(1 for e in eps if e['outcome']=='success')
        f[a]=(n-s)/n*100
    order=' < '.join(sorted(f,key=f.get))
    print(f"  {r.replace('kda_','').replace('premerge_','@'):<20}" + ''.join(f"{f[a]:>11.2f}%" for a in ARMS) + f"   {order}")
