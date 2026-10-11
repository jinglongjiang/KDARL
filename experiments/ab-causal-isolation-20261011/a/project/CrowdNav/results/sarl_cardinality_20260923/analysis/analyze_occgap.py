"""遮挡缺口:威胁质量只对可见行人求和,与对全体行人求和的差。
每个信息条件各自扫 beta,比较各自最优点 —— 否则会把信息不足与标定不当混在一起。"""
import json, glob, re, collections
rows = {}
for f in glob.glob('/home/abc/workspace/occgap_20260923/eval/*.json'):
    m = re.match(r'(nonet|sarl)_(observed|oracle)_b(.+?)_(\d+)\.json', f.split('/')[-1])
    d = json.load(open(f))
    if len(d.get('episodes', [])) != 200: continue
    s = d['summary'][0]
    rows[(m.group(1), m.group(2), float(m.group(3)), int(m.group(4)))] = (
        100*s['sr'], 100*s['cr'], 100*s['tr'], s['arrival_time_success'], s['path_length'])

for net, label in (('nonet', '无网络控制器(目标推进 − β·M)'), ('sarl', 'SARL + 外挂')):
    print(f'\n########## {label} ##########')
    print(f"{'N':>3s} {'β':>6s} | {'观测(可见)SR/CR/TO':>24s} | {'上界(全体)SR/CR/TO':>24s} | {'ΔSR':>7s}")
    for n in (5, 10, 20):
        betas = sorted({k[2] for k in rows if k[0] == net and k[3] == n})
        best = {}
        for src in ('observed', 'oracle'):
            cand = [(rows[(net, src, b, n)][0], b) for b in betas if (net, src, b, n) in rows]
            if cand: best[src] = max(cand)
        for b in betas:
            o = rows.get((net, 'observed', b, n)); r = rows.get((net, 'oracle', b, n))
            if not (o and r): continue
            mark = ''
            if best.get('observed', (None,))[1] == b: mark += ' ←观测最优'
            if best.get('oracle', (None,))[1] == b: mark += ' ←上界最优'
            print(f'{n:3d} {b:6.3g} | {o[0]:7.2f}/{o[1]:6.2f}/{o[2]:5.2f}       | '
                  f'{r[0]:7.2f}/{r[1]:6.2f}/{r[2]:5.2f}       | {r[0]-o[0]:+7.2f}{mark}')
        if 'observed' in best and 'oracle' in best:
            print(f'    → N={n} 各自最优点:观测 {best["observed"][0]:.2f}% (β={best["observed"][1]:g})  '
                  f'上界 {best["oracle"][0]:.2f}% (β={best["oracle"][1]:g})  '
                  f'**遮挡缺口 {best["oracle"][0]-best["observed"][0]:+.2f} pp**')
        print()
