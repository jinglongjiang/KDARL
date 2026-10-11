import json, statistics as st
from pathlib import Path
W = Path('/home/abc/temp/cc_6env_20261007')
RUNS = [('原版3000',        'kda_3000'),
        ('续训10000',       'kda_premerge_88000'),
        ('续训10000@89000', 'kda_premerge_89000'),
        ('优化版10000',     'kda')]
ARMS = ('mamba', 'gru', 'kda')
CELLS = [(5,'circle_crossing'),(5,'square_crossing'),(10,'circle_crossing'),
         (10,'square_crossing'),(20,'circle_crossing'),(20,'square_crossing')]
TO_TIME = 201 * 0.25   # 评测时限 50s / 0.25s = 200 步, harness 上限 201 步

def load(run, arm):
    f = W / run / f'{arm}.json'
    return json.load(open(f))['episodes'] if f.exists() else None

print('='*104)
print('第1步  你的前提是否成立? —— 四个跑里各臂的超时率')
print('='*104)
print(f"{'跑':<20}" + ''.join(f"{a+' TO':>13}" for a in ARMS) + '   最高超时者')
print('-'*104)
for label, run in RUNS:
    tos = {}
    for a in ARMS:
        eps = load(run, a)
        if eps is None: tos[a] = None; continue
        n = len(eps)
        s = sum(1 for e in eps if e['outcome']=='success')
        c = sum(1 for e in eps if e['outcome']=='collision')
        tos[a] = (n-s-c)/n*100
    if any(v is None for v in tos.values()): continue
    worst = max(tos, key=tos.get)
    print(f"{label:<20}" + ''.join(f"{tos[a]:>12.2f}%" for a in ARMS) + f"   {worst}")
print('-'*104)

print()
print('='*104)
print('第2步  安全-效率权衡: 碰撞率 vs 超时率 (同一个跑内比较)')
print('='*104)
for label, run in RUNS:
    if load(run, 'mamba') is None: continue
    print(f"\n  [{label}]")
    print(f"    {'臂':<8}{'SR':>9}{'CR':>8}{'TO':>8}{'CR+TO':>9}{'失败中超时占比':>14}")
    for a in ARMS:
        eps = load(run, a); n=len(eps)
        s = sum(1 for e in eps if e['outcome']=='success')
        c = sum(1 for e in eps if e['outcome']=='collision')
        t = n-s-c
        fail = c+t
        frac = t/fail*100 if fail else 0
        print(f"    {a:<8}{s/n*100:>8.2f}%{c/n*100:>7.2f}%{t/n*100:>7.2f}%{(c+t)/n*100:>8.2f}%{frac:>13.1f}%")

print()
print('='*104)
print('第3步  成功 episode 的导航时间 (是否"更谨慎所以更慢")')
print('='*104)
print(f"{'跑':<20}{'臂':<8}{'成功数':>7}{'中位用时':>10}{'均值':>9}{'p90':>8}{'最慢':>8}{'中位净距':>10}")
print('-'*104)
for label, run in RUNS:
    if load(run, 'mamba') is None: continue
    for a in ARMS:
        eps = [e for e in load(run, a) if e['outcome']=='success']
        ts = sorted(e['time'] for e in eps)
        cl = st.median(e['minimum_clearance'] for e in eps)
        p90 = ts[int(len(ts)*0.9)]
        print(f"{label:<20}{a:<8}{len(eps):>7}{st.median(ts):>9.2f}s{st.mean(ts):>8.2f}s{p90:>7.2f}s{ts[-1]:>7.2f}s{cl:>10.3f}")
    print('-'*104)

print()
print('='*104)
print('第4步  超时 episode 的性质: 是"卡住不动"还是"绕路绕不到"')
print('='*104)
print('  (超时 episode 的 time 必然=50.25s; 看它的 return 与 clearance 能区分行为)')
print(f"{'跑':<20}{'臂':<8}{'超时数':>7}{'超时中位return':>15}{'成功中位return':>15}{'超时中位净距':>13}{'成功中位净距':>13}")
print('-'*104)
for label, run in RUNS:
    if load(run, 'mamba') is None: continue
    for a in ARMS:
        eps = load(run, a)
        to = [e for e in eps if e['outcome'] not in ('success','collision')]
        su = [e for e in eps if e['outcome']=='success']
        if not to:
            print(f"{label:<20}{a:<8}{0:>7}{'—':>15}{st.median(e['return'] for e in su):>15.4f}{'—':>13}{st.median(e['minimum_clearance'] for e in su):>13.3f}")
            continue
        print(f"{label:<20}{a:<8}{len(to):>7}{st.median(e['return'] for e in to):>15.4f}"
              f"{st.median(e['return'] for e in su):>15.4f}"
              f"{st.median(e['minimum_clearance'] for e in to):>13.3f}"
              f"{st.median(e['minimum_clearance'] for e in su):>13.3f}")
    print('-'*104)
