"""三臂聚合归因分析:sum(ACT) / mean / softmax,同 phi、同冻结策略、同动作集。"""
import json, glob, re, collections
import numpy as np
from scipy import stats

D = '/home/abc/workspace/fcs3_agg_20260925/eval/'
SEEDS = (1207, 2407, 3607, 4807, 6007, 7207, 8407, 9607)
ARMS = ('sum', 'mean', 'softmax')
sel = json.load(open('/home/abc/workspace/fcs3_agg_20260925/status/selected_beta.json'))


def rates(f):
    """与 an_core.py 同源:成功/碰撞/超时率取自 summary[0] 的 sr/cr/tr。"""
    d = json.load(open(f))
    if len(d.get('episodes', [])) != 200:
        return None
    su = d['summary'][0]
    return (100. * su['sr'], 100. * su['cr'], 100. * su['tr'])


cell = collections.defaultdict(dict)
for f in glob.glob(D + '*_9400000.json'):
    m = re.match(r'(sum|mean|softmax)_(\d+)_b([\d.]+)_(\d+)_(\w+)_9400000\.json', f.split('/')[-1])
    if not m:
        continue
    arm, seed, beta, hum, sc = m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), m.group(5)
    r = rates(f)
    if r is None:
        continue
    key = 'base' if float(beta) == 0. else arm
    cell[(key, hum)][(seed, sc)] = r


def seedmean(key, hum):
    """先在场景内合并(双场景等权),再以 seed 为推断单位。"""
    out = []
    for s in SEEDS:
        v = [cell[(key, hum)].get((s, sc)) for sc in ('circle_crossing', 'square_crossing')]
        if any(x is None for x in v):
            return None
        out.append(np.mean(np.array(v, float), axis=0))
    return np.array(out)                      # (8,3) = SR/CR/TO


print('选出的 beta:', sel)
print(f'\n{"N":>3s} {"臂":>8s} {"beta":>5s} {"SR":>7s} {"CR":>7s} {"TO":>7s}   '
      f'{"dSR vs 基线 (95% CI)":>26s}')
store = {}
for hum in (5, 10, 20):
    base = seedmean('base', hum)
    if base is None:
        print(f'{hum:>3d}  基线不完整'); continue
    print(f'{hum:>3d} {"baseline":>8s} {"0":>5s} {base[:,0].mean():>7.2f} {base[:,1].mean():>7.2f} '
          f'{base[:,2].mean():>7.2f}')
    for arm in ARMS:
        a = seedmean(arm, hum)
        if a is None:
            print(f'    {arm:>8s} 不完整'); continue
        d = a[:, 0] - base[:, 0]
        t = stats.t.interval(.95, len(d) - 1, d.mean(), stats.sem(d)) if d.std() > 0 else (d.mean(), d.mean())
        store[f'{arm}_{hum}'] = dict(sr=a[:,0].mean(), cr=a[:,1].mean(), to=a[:,2].mean(),
                                     dsr=d.mean(), ci=[t[0], t[1]])
        print(f'    {arm:>8s} {sel.get(arm,"?"):>5s} {a[:,0].mean():>7.2f} {a[:,1].mean():>7.2f} '
              f'{a[:,2].mean():>7.2f}   {d.mean():>+8.2f} [{t[0]:+.2f},{t[1]:+.2f}]')

print('\n=== 关键配对:sum 减 mean / sum 减 softmax(同 seed 同场景配对)===')
for hum in (5, 10, 20):
    s = seedmean('sum', hum)
    for other in ('mean', 'softmax'):
        o = seedmean(other, hum)
        if s is None or o is None:
            continue
        for j, name in ((0, 'SR'), (1, 'CR')):
            d = s[:, j] - o[:, j]
            t = stats.t.interval(.95, len(d) - 1, d.mean(), stats.sem(d)) if d.std() > 0 else (d.mean(), d.mean())
            print(f'  {hum:>2d} 人  sum - {other:<8s} {name}: {d.mean():>+7.2f} '
                  f'[{t[0]:+.2f},{t[1]:+.2f}]  {"已分辨" if t[0]*t[1]>0 else "未分辨"}')
json.dump(store, open('/home/abc/workspace/fcs3_agg_20260925/status/agg_summary.json', 'w'))
