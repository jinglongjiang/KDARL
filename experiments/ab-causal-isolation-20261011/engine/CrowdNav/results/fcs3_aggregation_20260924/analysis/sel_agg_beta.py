"""三臂开发集选 beta。规则与 an_core.py 一致:circle、4 seed、跨 5/10/20 人的平均 SR 最大。"""
import json, glob, re, collections
import numpy as np
D = '/home/abc/workspace/fcs3_agg_20260925/eval/'
BETAS = ('0.01', '0.03', '0.1', '0.3', '1.0', '3.0')
acc = collections.defaultdict(dict)
for f in glob.glob(D + '*_9300000.json'):
    m = re.match(r'(sum|mean|softmax)_(\d+)_b([\d.]+)_(\d+)_circle_crossing_9300000\.json',
                 f.split('/')[-1])
    if not m:
        continue
    d = json.load(open(f))
    if len(d.get('episodes', [])) != 200:
        continue
    acc[(m.group(1), m.group(3))][(int(m.group(2)), int(m.group(4)))] = 100. * d['summary'][0]['sr']
sel = {}
for arm in ('sum', 'mean', 'softmax'):
    print('--', arm)
    best = None
    for b in BETAS:
        c = acc.get((arm, b), {})
        if len(c) != 12:
            print('   beta=%-5s 不完整 (%d/12)' % (b, len(c))); continue
        mean_sr = np.mean(list(c.values()))
        per = {h: np.mean([v for (s, hh), v in c.items() if hh == h]) for h in (5, 10, 20)}
        print('   beta=%-5s 跨密度均值 %6.2f   (5人 %5.1f / 10人 %5.1f / 20人 %5.1f)'
              % (b, mean_sr, per[5], per[10], per[20]))
        if best is None or mean_sr > best[1]:
            best = (b, mean_sr)
    if best:
        sel[arm] = best[0]; print('   => 选 beta=%s (%.2f%%)' % best)
b0 = acc.get(('sum', '0'), {})
if len(b0) == 12:
    print('-- beta=0 基线 跨密度均值 %.2f' % np.mean(list(b0.values())))
json.dump(sel, open('/home/abc/workspace/fcs3_agg_20260925/status/selected_beta.json', 'w'))
print('\nSELECTED', sel)
