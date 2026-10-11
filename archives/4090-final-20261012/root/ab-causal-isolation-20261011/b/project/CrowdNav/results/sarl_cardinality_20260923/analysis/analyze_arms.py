import json, glob, collections
import numpy as np
from scipy import stats
SEEDS = (1207, 2407, 3607, 4807)

def load(pat, keyf):
    a = collections.defaultdict(lambda: np.zeros(6))
    for f in glob.glob(pat):
        d = json.load(open(f))
        if len(d.get('episodes', [])) != 200: continue
        k = keyf(f, d)
        for s in d['summary']:
            e = s['episodes']
            a[k + (d['protocol']['sensor_protocol']['name'], s['human_num'])] += np.array(
                [s['sr']*e, s['cr']*e, s['tr']*e, e, s['arrival_time_success']*e, s['path_length']*e])
    return a

arms = load('/home/abc/workspace/sarl_cardinality_20260923/eval_arms/*/*.json',
            lambda f, d: (f.split('/')[-2].rsplit('_',1)[0], int(f.split('/')[-2].rsplit('_',1)[1])))
base = load('/home/abc/workspace/classic_baselines_fast_20260922/eval/sarl_*/*.json',
            lambda f, d: ('SARL', int(f.split('/')[-2].split('_')[1])))

def ci(v):
    v = np.asarray(v, float); m = v.mean()
    h = stats.t.ppf(.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
    return m, m-h, m+h

for proto in ('standard', 'severe'):
    print(f'\n########## {proto} 协议(两场景合并,4 seed × 400 回合/格)##########')
    print(f"{'臂':>6s} {'N':>3s} {'SR%':>7s} {'CR%':>7s} {'TO%':>7s} {'路径':>6s} {'ΔSR vs SARL (95% CI)':>28s}")
    for n in (5, 10, 20):
        b = {s: 100*base[('SARL', s, proto, n)][0]/base[('SARL', s, proto, n)][3] for s in SEEDS}
        t = np.sum([base[('SARL', s, proto, n)] for s in SEEDS], axis=0)
        print(f"{'SARL':>6s} {n:3d} {100*t[0]/t[3]:7.2f} {100*t[1]/t[3]:7.2f} {100*t[2]/t[3]:7.2f} "
              f"{t[5]/t[3]:6.2f} {'(基准)':>28s}")
        for arm in ('card', 'dual', 'count'):
            rows = [arms[(arm, s, proto, n)] for s in SEEDS]
            if any(r[3] == 0 for r in rows): continue
            t = np.sum(rows, axis=0)
            d = [100*rows[i][0]/rows[i][3] - b[s] for i, s in enumerate(SEEDS)]
            m, lo, hi = ci(d)
            print(f'{arm:>6s} {n:3d} {100*t[0]/t[3]:7.2f} {100*t[1]/t[3]:7.2f} {100*t[2]/t[3]:7.2f} '
                  f'{t[5]/t[3]:6.2f} {f"{m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]":>28s}')
        print()
print('\n=== 预注册判据(standard, 20 人)===')
for arm in ('card', 'dual', 'count'):
    b = {s: 100*base[('SARL', s, 'standard', 20)][0]/base[('SARL', s, 'standard', 20)][3] for s in SEEDS}
    d20 = [100*arms[(arm,s,'standard',20)][0]/arms[(arm,s,'standard',20)][3] - b[s] for s in SEEDS]
    m, lo, hi = ci(d20)
    b5 = {s: 100*base[('SARL', s, 'standard', 5)][0]/base[('SARL', s, 'standard', 5)][3] for s in SEEDS}
    d5 = [100*arms[(arm,s,'standard',5)][0]/arms[(arm,s,'standard',5)][3] - b5[s] for s in SEEDS]
    m5, _, _ = ci(d5)
    t = np.sum([arms[(arm,s,'standard',20)] for s in SEEDS], axis=0)
    cr = 100*t[1]/t[3]
    print(f'{arm:>6s}: 20人 ΔSR {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  '
          f'实用线(≥+15且下界>0) {"过" if m>=15 and lo>0 else "否"}   '
          f'碰撞 {cr:.2f}%(≤60.94 为过) {"过" if cr<=60.94 else "否"}   '
          f'5人 ΔSR {m5:+6.2f}(≥-3 为过) {"过" if m5>=-3 else "否"}')
