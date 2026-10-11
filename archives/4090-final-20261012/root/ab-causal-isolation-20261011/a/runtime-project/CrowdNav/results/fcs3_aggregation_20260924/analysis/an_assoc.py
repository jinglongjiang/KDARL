import json, glob, re, collections
import numpy as np
rows, diag = {}, {}
for f in glob.glob('/home/abc/workspace/assoc_gate_20260924/eval/*.json'):
    m = re.match(r'nonet_(\w+?)_g(.+?)_(\d+)_(\d+)_(\w+)\.json', f.split('/')[-1])
    if not m: continue
    d = json.load(open(f))
    if len(d.get('episodes', [])) != 200: continue
    s = d['summary'][0]
    k = (m.group(1), float(m.group(2)), int(m.group(3)))
    rows[k] = (100*s['sr'], 100*s['cr'], 100*s['tr'])
    L = d.get('assoc_log', [])
    if L:
        t = np.array([x['n_tracks'] for x in L]); r = np.array([x['n_ref'] for x in L])
        me = np.array([x['m_rel'] for x in L])
        diag[k] = (float(np.mean(t - r)), float(100*me.mean()))
print('=== Association Gate 开发集(360deg,memory,200 回合/格)===')
print(f"{'关联':>10s} {'门控':>5s} {'N':>3s} | {'SR':>6s} {'CR':>6s} {'TO':>6s} | {'航迹数偏差':>10s} {'M相对误差%':>11s}")
for k in sorted(rows, key=lambda x: (x[2], x[0], x[1])):
    v = rows[k]; dg = diag.get(k)
    ex = f'{dg[0]:+10.2f} {dg[1]:11.1f}' if dg else f"{'-':>10s} {'-':>11s}"
    print(f'{k[0]:>10s} {k[1]:5g} {k[2]:3d} | {v[0]:6.1f} {v[1]:6.1f} {v[2]:6.1f} | {ex}')
print()
for n in (5, 10, 20):
    orc = rows.get(('oracle', 2.5, n))
    if not orc: continue
    hu = [(rows[k][0], k[1]) for k in rows if k[0] == 'hungarian' and k[2] == n]
    gr = [(rows[k][0], k[1]) for k in rows if k[0] == 'greedy' and k[2] == n]
    if hu:
        bh = max(hu); print(f'{n:2d}人  oracle {orc[0]:5.1f} | hungarian 最优 {bh[0]:5.1f} (门控 {bh[1]:g}) '
                            f'| 缺口 {orc[0]-bh[0]:+6.2f}' + (f' | greedy 最优 {max(gr)[0]:5.1f}' if gr else ''))
