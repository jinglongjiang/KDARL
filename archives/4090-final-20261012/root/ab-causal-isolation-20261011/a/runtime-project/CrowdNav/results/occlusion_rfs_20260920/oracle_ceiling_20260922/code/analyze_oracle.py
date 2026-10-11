"""信息上限:同一 Mean 策略、同一测试 case,只把隐藏行人换成真值。
基线 = 归档的正常评测(已验证本地重跑与归档逐回合一致 400/400)。"""
import glob, json, collections, sys
import numpy as np
R = '/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/CrowdNav/results/occlusion_rfs_20260920'
BASE = {'std_rl': R + '/rl_formal_20260921/eval_std', 'std_il': R + '/std_eval_fixed', 'sev_il': R + '/v3_eval_fixed'}
LABEL = {'std_rl': 'Standard RL', 'std_il': 'Standard IL', 'sev_il': 'Severe IL'}
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
W = __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'records')

def rates(rows, humans=None, outcome='reach_goal'):
    v = [r['outcome'] == outcome for r in rows if humans in (None, r['humans'])]
    return 100 * np.mean(v) if v else np.nan

for g in ('std_rl', 'std_il', 'sev_il'):
    print(f'## {LABEL[g]}(Mean 策略)')
    for o in ('hidden', 'range'):
        d, base_sr, orc_sr, rows_n = [], [], [], {}
        per_h = collections.defaultdict(list); cr = []; tr = []
        for s in (1207, 2407, 3607, 4807, 6007, 7207, 8407, 9607):
            f = f'{W}/{g}_{o}_{s}.jsonl'
            try: orc = [json.loads(l) for l in open(f)]
            except FileNotFoundError: continue
            if len(orc) < 1200: continue
            base = [json.loads(l) for l in open(f'{BASE[g]}/mean_{s}.jsonl')]
            d.append(rates(orc) - rates(base)); base_sr.append(rates(base)); orc_sr.append(rates(orc))
            for h in (5, 10, 20): per_h[h].append(rates(orc, h) - rates(base, h))
            cr.append(rates(orc, outcome='collision') - rates(base, outcome='collision'))
            tr.append(rates(orc, outcome='timeout') - rates(base, outcome='timeout'))
        n = len(d)
        if n < 2: print(f'  {o:<6} 完成种子 {n}/8,尚不足以统计'); continue
        m = np.mean(d); half = T[n] * np.std(d, ddof=1) / np.sqrt(n)
        print(f'  {o:<6} n={n}  成功率 {np.mean(base_sr):.2f} → {np.mean(orc_sr):.2f}  '
              f'提升 {m:+.2f} pp  95%CI [{m-half:+.2f}, {m+half:+.2f}]  正向 {sum(x>0 for x in d)}/{n}')
        print(f'         分人数 5/10/20: ' + ' / '.join(f'{np.mean(per_h[h]):+.2f}' for h in (5,10,20))
              + f'   碰撞 {np.mean(cr):+.2f}  超时 {np.mean(tr):+.2f}')
    print()
