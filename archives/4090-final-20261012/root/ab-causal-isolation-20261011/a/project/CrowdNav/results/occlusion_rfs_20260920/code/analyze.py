"""遮挡协议冻结评测的分层统计。

配对单位 = (人数, 场景, case, 种子);自助抽样按 case 成簇,因为同一 case 在所有
臂之间共享布局,回合之间的相关性来自布局而不是策略。
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ARMS = ('sensor', 'mean', 'age', 'full', 'gt')
LABEL = {'sensor': '无记忆', 'mean': '后验均值', 'age': '时长查表',
         'full': '全后验', 'gt': '全知上界'}
CORE = (('mean', 'sensor', '记忆的价值'),
        ('age', 'mean', '不确定度的价值'),
        ('full', 'age', '存在概率的价值(全后验 vs 时长)'),
        ('gt', 'full', '距全知上界的缺口'))


def load(directory):
    rows = []
    for path in sorted(Path(directory).glob('*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index(rows):
    table = defaultdict(dict)
    for row in rows:
        table[(row['humans'], row['scene'], row['case'], row['seed'])][row['arm']] = row
    return table


def bootstrap(pairs, draws=10000, seed=0):
    """pairs: {case: [(a, b), ...]}。按 case 成簇重采样。"""
    cases = list(pairs)
    rng = np.random.default_rng(seed)
    sums = np.array([[sum(a for a, _ in pairs[c]), sum(b for _, b in pairs[c]),
                      len(pairs[c])] for c in cases], float)
    point = (sums[:, 0].sum() - sums[:, 1].sum()) / sums[:, 2].sum()
    draw = rng.integers(len(cases), size=(draws, len(cases)))
    taken = sums[draw]
    delta = (taken[:, :, 0].sum(1) - taken[:, :, 1].sum(1)) / taken[:, :, 2].sum(1)
    return point * 100, np.quantile(delta, .025) * 100, np.quantile(delta, .975) * 100


def mcnemar(pairs):
    """精确二项检验(双侧),对大 n 用对数空间避免溢出。"""
    win = sum(1 for c in pairs for a, b in pairs[c] if a and not b)
    loss = sum(1 for c in pairs for a, b in pairs[c] if b and not a)
    n = win + loss
    if n == 0:
        return win, loss, 1.
    from math import lgamma, log, exp
    def log_choose(n, k):
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    k = min(win, loss)
    terms = [log_choose(n, i) - n * log(2.) for i in range(k + 1)]
    peak = max(terms)
    tail = peak + log(sum(exp(t - peak) for t in terms))
    return win, loss, min(1., 2. * exp(tail))


def collect(table, first, second, humans=None, scene=None):
    pairs = defaultdict(list)
    for (n, s, case, _), arms in table.items():
        if humans is not None and n != humans:
            continue
        if scene is not None and s != scene:
            continue
        if first in arms and second in arms:
            pairs[case].append((arms[first]['outcome'] == 'reach_goal',
                                arms[second]['outcome'] == 'reach_goal'))
    return pairs


def rate(rows, arm, humans=None, scene=None, seed=None, outcome='reach_goal'):
    chosen = [r for r in rows if r['arm'] == arm
              and (humans is None or r['humans'] == humans)
              and (scene is None or r['scene'] == scene)
              and (seed is None or r['seed'] == seed)]
    if not chosen:
        return None, 0
    return 100. * sum(r['outcome'] == outcome for r in chosen) / len(chosen), len(chosen)


def main(directory):
    rows = load(directory)
    table = index(rows)
    seeds = sorted({r['seed'] for r in rows})
    densities = sorted({r['humans'] for r in rows})
    scenes = sorted({r['scene'] for r in rows})
    print(f'回合总数 {len(rows)},种子 {seeds},人数 {densities},场景 {scenes}\n')

    print('=== 成功率(%),按人数 × 场景,三种子合并 ===')
    header = '臂'.ljust(12) + ''.join(f'{n}人-{s[:6]}'.rjust(14) for n in densities for s in scenes)
    print(header)
    for arm in ARMS:
        line = f'{LABEL[arm]}({arm})'.ljust(12)
        for n in densities:
            for s in scenes:
                value, count = rate(rows, arm, n, s)
                line += (f'{value:.1f}' if value is not None else '-').rjust(14)
        print(line)

    print('\n=== 每种子成功率(%),人数合并,检查种子间方差 ===')
    print('臂'.ljust(12) + ''.join(f'{n}人'.rjust(8) + ''.join(f's{x}'.rjust(8) for x in seeds)
                                   for n in densities))
    for arm in ARMS:
        line = f'{arm}'.ljust(12)
        for n in densities:
            line += ''.rjust(8)
            for x in seeds:
                value, _ = rate(rows, arm, n, None, x)
                line += (f'{value:.1f}' if value is not None else '-').rjust(8)
        print(line)

    print('\n=== 核心比较:配对自助(按 case 成簇)+ 精确 McNemar ===')
    for first, second, name in CORE:
        print(f'\n-- {name}: {first} − {second}')
        for n in [None] + densities:
            pairs = collect(table, first, second, n)
            if not pairs:
                continue
            point, low, high = bootstrap(pairs)
            win, loss, p = mcnemar(pairs)
            tag = '全部' if n is None else f'{n}人'
            star = '  显著' if (low > 0 or high < 0) else ''
            print(f'  {tag:>5}: Δ={point:+6.2f} pp  95%CI[{low:+6.2f},{high:+6.2f}]  '
                  f'配对 {win}:{loss}  p={p:.3g}{star}')

    print('\n=== 失败构成(%):碰撞 / 超时 ===')
    for n in densities:
        print(f'  {n}人:', end='')
        for arm in ARMS:
            collision, _ = rate(rows, arm, n, outcome='collision')
            timeout, _ = rate(rows, arm, n, outcome='timeout')
            print(f'  {arm}={collision:.0f}/{timeout:.0f}', end='')
        print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
