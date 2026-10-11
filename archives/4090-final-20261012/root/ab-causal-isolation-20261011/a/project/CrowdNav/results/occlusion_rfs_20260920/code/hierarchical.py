"""belief 层级的主统计。判据见 PREREGISTRATION.md 第 5、6 节。

训练 seed 是主要推断单位:每个 seed 给一个配对差值,再在 seed 之间做 t 区间,
这是主结论依据。seed × case 两级自助只用于提高效应量估计精度,不得覆盖 seed 之间
的不确定性;两者结论不一致时按 seed-level 解释。单元格符号检验仅作描述性证据。
"""
from __future__ import annotations
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, '.')
from analyze import load

CORE = (('mean', 'sensor', '记忆'),
        ('age', 'mean', '失观时长不确定度'),
        ('full', 'age', '逐轨迹完整后验'),
        ('gt', 'full', '全状态参考的缺口'))
EQUIVALENCE = 3.0       # 预注册的实际等价区间 ±3 pp
MINIMUM_GAIN = 3.0      # 预注册的最小有实际意义增益 3 pp


def paired(rows, first, second):
    """-> {seed: {case_key: (a_success, b_success)}}"""
    table = defaultdict(dict)
    for row in rows:
        table[(row['seed'], row['humans'], row['scene'], row['case'])][row['arm']] = row
    out = defaultdict(lambda: defaultdict(list))
    for (seed, humans, scene, case), arms in table.items():
        if first in arms and second in arms:
            out[seed][(humans, scene, case)].append(
                (arms[first]['outcome'] == 'reach_goal',
                 arms[second]['outcome'] == 'reach_goal'))
    return out


def hierarchical(data, draws=20000, seed=0, level=.95):
    rng = np.random.default_rng(seed)
    seeds = sorted(data)
    per_seed = {}
    for s in seeds:
        keys = sorted(data[s])
        per_seed[s] = np.array([[sum(a for a, _ in data[s][k]) - sum(b for _, b in data[s][k]),
                                 len(data[s][k])] for k in keys], float)
    point = float(np.mean([m[:, 0].sum() / m[:, 1].sum() for m in per_seed.values()])) * 100
    samples = np.empty(draws)
    for i in range(draws):
        chosen = rng.integers(len(seeds), size=len(seeds))
        values = []
        for j in chosen:
            matrix = per_seed[seeds[j]]
            take = matrix[rng.integers(len(matrix), size=len(matrix))]
            values.append(take[:, 0].sum() / take[:, 1].sum())
        samples[i] = np.mean(values)
    alpha = (1. - level) / 2.
    return point, float(np.quantile(samples, alpha)) * 100, float(np.quantile(samples, 1 - alpha)) * 100


def seed_interval(values, level=.95):
    """把每个种子的差值当一个观测的 t 区间。种子数少时两级自助偏乐观,
    两者不一致时以更保守的为准。"""
    values = np.asarray(values, float)
    n = len(values)
    if n < 2:
        return float('-inf'), float('inf')
    try:
        from scipy import stats
        critical = float(stats.t.ppf(.5 + level / 2., n - 1))
    except ImportError:
        lookup = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
                  10: 2.262, 12: 2.201, 16: 2.131, 20: 2.093, 24: 2.069}
        critical = lookup[min(lookup, key=lambda k: abs(k - n))]
    half = critical * float(np.std(values, ddof=1)) / np.sqrt(n)
    return float(values.mean()) - half, float(values.mean()) + half


def verdict(low95, high90, point, low90):
    """预注册解释规则(§6):先判增益,再判实际等价,两者都不成立则证据不足。"""
    if low95 > 0 and point >= MINIMUM_GAIN:
        return '有实际意义的增益'
    if low90 > -EQUIVALENCE and high90 < EQUIVALENCE:
        return '落入预设实际等价区间 ±3pp'
    return '证据不足(既跨 0 又超出 ±3pp),不得表述为"无用"'


def main(directory):
    rows = load(directory)
    seeds = sorted({r['seed'] for r in rows})
    print(f'训练种子 {len(seeds)} 个: {seeds}')
    print(f'判据: 主结论按 seed-level t 区间;两级自助仅作精度参考'
          f'(PREREGISTRATION.md §5)\n')
    deviations = {}
    for first, second, name in CORE:
        data = paired(rows, first, second)
        per = [100 * (sum(a for k in data[s] for a, _ in data[s][k])
                      - sum(b for k in data[s] for _, b in data[s][k]))
               / sum(len(v) for v in data[s].values()) for s in seeds]
        deviation = float(np.std(per, ddof=1))
        deviations[(first, second)] = deviation
        t_low, t_high = seed_interval(per)
        point, b_low, b_high = hierarchical(data)
        positive = sum(1 for v in per if v > 0)
        print(f'  {first:>6} − {second:<7} ({name})')
        print(f'      逐种子: ' + '  '.join(f'{v:+6.2f}' for v in per))
        print(f'      【主】seed-level  均值 {np.mean(per):+6.2f} pp  '
              f'95%CI [{t_low:+6.2f}, {t_high:+6.2f}]  sd {deviation:5.2f}  '
              f'正向 {positive}/{len(seeds)}')
        print(f'      〔辅〕两级自助    点估计 {point:+6.2f} pp  '
              f'95%CI [{b_low:+6.2f}, {b_high:+6.2f}]'
              + ('   ← 与主结论不一致,以主结论为准'
                 if (t_low <= 0) != (b_low <= 0) else ''))
        if (first, second) in (('full', 'age'), ('age', 'mean'), ('mean', 'sensor')):
            low90, high90 = seed_interval(per, level=.90)
            print(f'      90%CI [{low90:+6.2f}, {high90:+6.2f}]   '
                  f'预注册判定: {verdict(t_low, high90, float(np.mean(per)), low90)}')

    print('\n=== 追加种子的精度规划(按本轮实测 seed-level sd,§7)===')
    try:
        from scipy import stats
        critical = lambda n: float(stats.t.ppf(.975, n - 1))
    except ImportError:
        lookup = {8: 2.365, 12: 2.201, 16: 2.131, 20: 2.093}
        critical = lambda n: lookup[min(lookup, key=lambda k: abs(k - n))]
    for first, second, name in CORE[:3]:
        deviation = deviations[(first, second)]
        widths = '  '.join(f'n={n}:±{critical(n)*deviation/np.sqrt(n):.2f}'
                           for n in (8, 12, 16, 20))
        print(f'  {first:>6} − {second:<7} sd={deviation:5.2f}   {widths}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'eval')
