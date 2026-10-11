import json, glob, collections, re
import numpy as np
from scipy import stats

SEEDS = (1207, 2407, 3607, 4807)
BETAS = ('0', '0.01', '0.03', '0.1', '0.3', '0.6', '1.0')
agg = collections.defaultdict(lambda: np.zeros(6))
for f in glob.glob('/home/abc/workspace/threatmass_20260923/eval/*/*.json'):
    m = re.match(r'b(.+?)_l(.+?)_standard_(\d+)_circle_crossing\.json', f.split('/')[-1])
    d = json.load(open(f))
    if len(d.get('episodes', [])) != 200: continue
    seed = int(f.split('/')[-2])
    for s in d['summary']:
        k = (m.group(1), m.group(2), seed, int(m.group(3)))
        e = s['episodes']
        agg[k] += np.array([s['sr']*e, s['cr']*e, s['tr']*e, e,
                            s['arrival_time_success']*e, s['path_length']*e])

arch = collections.defaultdict(lambda: np.zeros(4))
for f in glob.glob('/home/abc/workspace/classic_baselines_fast_20260922/eval/sarl_*/standard_*_circle_crossing.json'):
    d = json.load(open(f)); seed = int(f.split('/')[-2].split('_')[1])
    for s in d['summary']:
        e = s['episodes']
        arch[(seed, s['human_num'])] += np.array([s['sr']*e, s['cr']*e, s['tr']*e, e])

print('=== 内建对账:beta=0 是否还原归档(standard, circle) ===')
bad = 0
for seed in SEEDS:
    for n in (5, 10, 20):
        a = agg[('0', '1.0', seed, n)][:4]; b = arch[(seed, n)]
        if not np.allclose(a, b): bad += 1; print(f'  不一致 seed {seed} N={n}: {a} vs {b}')
print(f'  {12-bad}/12 格逐数相同' if bad == 0 else f'  {bad}/12 不一致 —— 实验作废')

def ci(v):
    v = np.asarray(v, float); m = v.mean()
    if len(v) < 2: return m, np.nan, np.nan
    h = stats.t.ppf(.975, len(v)-1) * v.std(ddof=1)/np.sqrt(len(v))
    return m, m-h, m+h

for n in (20, 10, 5):
    print(f'\n=== N={n} 人(standard, circle,4 seed × 200 回合)===')
    print(f"{'beta':>5s} {'lam':>4s} {'SR%':>7s} {'CR%':>7s} {'TO%':>7s} {'到达时间':>8s} {'路径长':>7s} {'ΔSR vs beta=0 (95% CI)':>28s}")
    base = {s: 100*agg[('0','1.0',s,n)][0]/agg[('0','1.0',s,n)][3] for s in SEEDS}
    for beta in BETAS:
        for lam in (('1.0',) if beta == '0' else ('1.0','0.0')):
            rows = [agg[(beta,lam,s,n)] for s in SEEDS]
            if any(r[3] == 0 for r in rows): continue
            t = np.sum(rows, axis=0)
            sr = [100*r[0]/r[3] for r in rows]
            m, lo, hi = ci([sr[i]-base[s] for i, s in enumerate(SEEDS)])
            tag = '(基准)' if beta == '0' else f'{m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]'
            succ = t[0] if t[0] > 0 else 1
            print(f'{beta:>5s} {lam:>4s} {100*t[0]/t[3]:7.2f} {100*t[1]/t[3]:7.2f} {100*t[2]/t[3]:7.2f} '
                  f'{t[4]/t[3]:8.2f} {t[5]/t[3]:7.2f} {tag:>28s}')
