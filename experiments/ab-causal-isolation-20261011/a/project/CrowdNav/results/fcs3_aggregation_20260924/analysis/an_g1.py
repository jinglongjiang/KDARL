import json, glob, re, collections
import numpy as np
from scipy import stats
a = collections.defaultdict(dict)
for f in glob.glob('/home/abc/workspace/g1_velocity_20260924/eval/*.json'):
    m = re.match(r'(\w+?)_(\d+)_(oracle|finite_diff|kalman|zero)_(\d+)_(\w+)\.json', f.split('/')[-1])
    if not m: continue
    d = json.load(open(f))
    if len(d.get('episodes', [])) != 200: continue
    a[(m.group(1), m.group(3), int(m.group(4)))][int(m.group(2))] = 100*d['summary'][0]['sr']
print('=== G1:速度来源(360deg 全向,standard,circle,200 回合/格)===')
print(f"{'方法':>6s} {'N':>3s} | {'oracle':>7s} {'finite_diff':>12s} {'kalman':>8s} {'zero':>7s} | {'oracle-finite_diff 95%CI':>26s}")
for ctrl in ('sarl', 'lstm'):
    for n in (5, 10, 20):
        v = {}
        for src in ('oracle', 'finite_diff', 'kalman', 'zero'):
            d = a.get((ctrl, src, n), {})
            if d: v[src] = d
        if 'oracle' not in v or 'finite_diff' not in v: continue
        seeds = sorted(set(v['oracle']) & set(v['finite_diff']))
        dd = [v['oracle'][s] - v['finite_diff'][s] for s in seeds]
        mm = np.mean(dd); h = stats.t.ppf(.975, len(dd)-1)*np.std(dd, ddof=1)/np.sqrt(len(dd)) if len(dd) > 1 else float('nan')
        row = ' '.join(f'{np.mean(list(v[s].values())):7.2f}     ' if s == 'finite_diff'
                       else f'{np.mean(list(v[s].values())):7.2f}' for s in ('oracle','finite_diff','kalman','zero') if s in v)
        print(f'{ctrl:>6s} {n:3d} | {row} | {mm:+8.2f} [{mm-h:+7.2f},{mm+h:+7.2f}]  n={len(seeds)}')
    print()
