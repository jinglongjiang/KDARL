import json, sys, statistics as st
from pathlib import Path
CELLS = [(5,'circle_crossing'),(5,'square_crossing'),(10,'circle_crossing'),
         (10,'square_crossing'),(20,'circle_crossing'),(20,'square_crossing')]
def load(p):
    d = json.load(open(p)); return d, d['episodes']
def rates(eps):
    n=len(eps); s=sum(1 for e in eps if e['outcome']=='success')
    c=sum(1 for e in eps if e['outcome']=='collision')
    return n, s/n*100, c/n*100, (n-s-c)/n*100
arms = sys.argv[1:]
data = {}
for a in arms:
    name = Path(a).stem
    data[name] = load(a)
print('=== 总体 (192 ep = 6 cell x 32 case) ===')
print(f"{'臂':<8}{'n':>5}{'SR':>9}{'CR':>8}{'TO':>8}{'中位延迟':>11}{'p95':>9}{'墙钟':>9}")
for name,(d,eps) in data.items():
    n,sr,cr,to = rates(eps)
    print(f"{name:<8}{n:>5}{sr:>8.2f}%{cr:>7.2f}%{to:>7.2f}%"
          f"{d['policy_latency_ms_median']:>10.2f}ms{d['policy_latency_ms_p95']:>8.2f}ms{d['wall_seconds']:>8.0f}s")
print()
print('=== 逐 cell 成功率 ===')
print(f"{'环境':<24}" + ''.join(f"{n:>11}" for n in data))
for hn,geo in CELLS:
    label = f"{hn}人 {geo}"
    row = f"{label:<22}"
    for name,(d,eps) in data.items():
        sub=[e for e in eps if e['human_num']==hn and e['geometry']==geo]
        _,sr,_,_ = rates(sub)
        row += f"{sr:>10.1f}%"
    print(row)
print()
print('=== 逐 cell 碰撞率 ===')
print(f"{'环境':<24}" + ''.join(f"{n:>11}" for n in data))
for hn,geo in CELLS:
    row = f"{f'{hn}人 {geo}':<22}"
    for name,(d,eps) in data.items():
        sub=[e for e in eps if e['human_num']==hn and e['geometry']==geo]
        _,_,cr,_ = rates(sub)
        row += f"{cr:>10.1f}%"
    print(row)
print()
print('=== 中位最小净距 (m) ===')
print(f"{'环境':<24}" + ''.join(f"{n:>11}" for n in data))
for hn,geo in CELLS:
    row = f"{f'{hn}人 {geo}':<22}"
    for name,(d,eps) in data.items():
        sub=[e for e in eps if e['human_num']==hn and e['geometry']==geo]
        row += f"{st.median(e['minimum_clearance'] for e in sub):>11.3f}"
    print(row)
print()
print('=== checkpoint sha256 (前16) ===')
for name,(d,eps) in data.items():
    print(f"  {name:<8}{d['checkpoint_sha256'][:16]}")
