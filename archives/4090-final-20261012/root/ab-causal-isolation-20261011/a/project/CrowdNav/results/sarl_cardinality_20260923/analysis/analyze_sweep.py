"""alpha 扫描分析 + 与归档基线的逐 seed 对账。种子为主推断单元。"""
import json, glob, collections
import numpy as np

SEEDS = (1207, 2407, 3607, 4807, 6007, 7207, 8407, 9607)
ALPHAS = ('1.0', '0.75', '0.5', '0.25', '0.0')


def load(pattern, key):
    """返回 {(key, seed, N): (sr, cr, tr, n)},按场景合并。"""
    out = collections.defaultdict(lambda: np.zeros(4))
    for f in glob.glob(pattern):
        d = json.load(open(f))
        if len(d.get('episodes', [])) != 200:
            continue
        k = key(f, d)
        if k is None:
            continue
        for s in d['summary']:
            out[k + (s['human_num'],)] += np.array(
                [s['sr'] * s['episodes'], s['cr'] * s['episodes'], s['tr'] * s['episodes'], s['episodes']])
    return out


sweep = load('/home/abc/workspace/sarl_cardinality_20260923/eval/*/*.json',
             lambda f, d: (f.split('/')[-1].split('_')[0][1:], int(f.split('/')[-2])))
arch = load('/home/abc/workspace/classic_baselines_fast_20260922/eval/sarl_*/standard_*.json',
            lambda f, d: ('archive', int(f.split('/')[-2].split('_')[1])))

print('=== 对账:alpha=1.0 是否复现归档 8 seed 的 standard 结果 ===')
bad = 0
for s in SEEDS:
    for n in (5, 10, 20):
        a = sweep[('1.0', s, n)]; b = arch[('archive', s, n)]
        if not np.allclose(a, b):
            bad += 1
            print(f'  不一致 seed {s} N={n}: 扫描 {a} vs 归档 {b}')
print(f'  {24-bad}/24 格逐数相同' if bad == 0 else f'  {bad} 格不一致')

def seed_ci(vals):
    v = np.asarray(vals, float); m = v.mean()
    if len(v) < 2: return m, np.nan, np.nan
    se = v.std(ddof=1) / np.sqrt(len(v))
    from scipy import stats
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return m, m - h, m + h

print('\n=== 零训练基数扫描(8 seed,每格 400 回合,两场景合并)===')
print(f"{'alpha':>5s} {'N':>3s} {'SR%':>7s} {'CR%':>7s} {'TO%':>7s} {'ΔSR vs alpha=1 (95% CI)':>30s}")
for n in (5, 10, 20):
    base = {s: 100 * sweep[('1.0', s, n)][0] / sweep[('1.0', s, n)][3] for s in SEEDS}
    for a in ALPHAS:
        rows = [sweep[(a, s, n)] for s in SEEDS]
        tot = np.sum(rows, axis=0)
        sr = [100 * r[0] / r[3] for r in rows]
        d = [sr[i] - base[s] for i, s in enumerate(SEEDS)]
        m, lo, hi = seed_ci(d)
        tag = '  (基准)' if a == '1.0' else f'{m:+8.2f} [{lo:+7.2f},{hi:+7.2f}]'
        print(f'{a:>5s} {n:3d} {100*tot[0]/tot[3]:7.2f} {100*tot[1]/tot[3]:7.2f} {100*tot[2]/tot[3]:7.2f} {tag:>30s}')
    print()

print('=== 预注册主机制判据:20 人成功率是否随 alpha 下降而单调上升 ===')
means = [np.mean([100 * sweep[(a, s, 20)][0] / sweep[(a, s, 20)][3] for s in SEEDS]) for a in ALPHAS]
print('  alpha 1.0→0.0 的 20 人成功率:', ' → '.join(f'{m:.2f}' for m in means))
print('  单调上升?', '是' if all(x < y for x, y in zip(means, means[1:])) else '否 —— 判据未通过')
