import csv, math
from itertools import combinations
from pathlib import Path
import statistics as st

ARMS = ('formal-gru', 'formal-bayes_mean', 'formal-bayes')
SHORT = {'formal-gru': 'gru', 'formal-bayes_mean': 'bayes_mean', 'formal-bayes': 'bayes'}

def load(a):
    rows = {}
    with open(f'codex3000/{a}.csv') as f:
        for r in csv.DictReader(f):
            key = (r['scenario'], int(r['episode']))
            rows[key] = r
    return rows

D = {a: load(a) for a in ARMS}
for a in ARMS:
    assert len(D[a]) == 3000, (a, len(D[a]))

def mcn(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

print('=' * 100)
print('表 A  Codex 3000-ep 协议 (6 命名环境 x 500 ep), 同 seed 42 / 同 IL / 同权重')
print('=' * 100)
print(f"{'臂':<12}{'n':>6}{'SR':>9}{'CR':>8}{'TO':>8}{'接触ep':>8}{'中位净距':>10}{'中位推理':>10}")
print('-' * 100)
agg = {}
for a in ARMS:
    rs = list(D[a].values()); n = len(rs)
    s = sum(1 for r in rs if r['outcome'] == 'success')
    c = sum(1 for r in rs if r['outcome'] == 'collision')
    t = n - s - c
    ct = sum(1 for r in rs if float(r['geometric_contact_steps']) > 0)
    clr = st.median(float(r['min_clearance_m']) for r in rs)
    inf = st.median(float(r['mean_inference_ms']) for r in rs)
    agg[a] = (s/n*100, c/n*100, t/n*100)
    print(f"{SHORT[a]:<12}{n:>6}{s/n*100:>8.2f}%{c/n*100:>7.2f}%{t/n*100:>7.2f}%{ct:>8}{clr:>10.3f}{inf:>9.2f}ms")

print()
print('=' * 100)
print('表 B  两两配对 McNemar (按 场景+episode 严格配对, 3000 对)')
print('=' * 100)
print(f"{'对比':<28}{'ΔSR':>10}{'b':>6}{'c':>6}{'净差':>7}{'McNemar p':>12}  判读")
print('-' * 100)
for x, y in combinations(ARMS, 2):
    keys = sorted(set(D[x]) & set(D[y])); assert len(keys) == 3000
    b = sum(1 for k in keys if D[x][k]['outcome'] == 'success' and D[y][k]['outcome'] != 'success')
    c = sum(1 for k in keys if D[x][k]['outcome'] != 'success' and D[y][k]['outcome'] == 'success')
    p = mcn(b, c)
    v = '**显著**' if p < 0.05 else '无显著差异'
    if p < 0.0167: v += ' (Bonferroni后仍显著)'
    print(f"{SHORT[y]+' − '+SHORT[x]:<28}{agg[y][0]-agg[x][0]:>+9.2f}pp{b:>6}{c:>6}{c-b:>+7}{p:>12.2e}  {v}")

print()
print('=' * 100)
print('表 C  逐环境成功率 / 碰撞率')
print('=' * 100)
SC = ['baseline_circle', 'baseline_square', 'dense_circle', 'dense_square', 'large_circle', 'large_square']
print(f"{'环境':<18}" + ''.join(f"{SHORT[a]:>13}" for a in ARMS) + '     |' + ''.join(f"{SHORT[a]+' CR':>14}" for a in ARMS))
print('-' * 100)
for sc in SC:
    row = f"{sc:<18}"
    crs = ''
    for a in ARMS:
        sub = [r for k, r in D[a].items() if k[0] == sc]
        s = sum(1 for r in sub if r['outcome'] == 'success')
        c = sum(1 for r in sub if r['outcome'] == 'collision')
        row += f"{s/len(sub)*100:>12.1f}%"
        crs += f"{c/len(sub)*100:>13.1f}%"
    print(row + '     |' + crs)

print()
print('=' * 100)
print('表 D  两套协议的结论是否一致')
print('=' * 100)
MINE = {'gru': 91.15, 'bayes_mean': 88.54, 'bayes': 79.69}   # 我的 192-ep
print(f"{'臂':<12}{'192-ep SR':>12}{'3000-ep SR':>12}{'差':>9}")
print('-' * 100)
for a in ARMS:
    k = SHORT[a]
    print(f"{k:<12}{MINE[k]:>11.2f}%{agg[a][0]:>11.2f}%{agg[a][0]-MINE[k]:>+8.2f}pp")
print('-' * 100)
print(f"  192-ep  排序: {' > '.join(sorted(MINE, key=MINE.get, reverse=True))}")
o = sorted(ARMS, key=lambda a: agg[a][0], reverse=True)
print(f"  3000-ep 排序: {' > '.join(SHORT[a] for a in o)}")
