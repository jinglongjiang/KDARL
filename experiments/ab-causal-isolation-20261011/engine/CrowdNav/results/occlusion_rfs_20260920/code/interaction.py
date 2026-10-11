"""两个追加检验:(1) 逐种子的臂间差值;(2) 差值随密度增长的交互项。"""
import sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.')
from analyze import load, index, CORE, bootstrap, mcnemar, collect

rows = load('eval')
table = index(rows)
seeds = sorted({r['seed'] for r in rows})

print('=== 逐种子臂间差值(pp),按人数 ===')
for first, second, name in CORE:
    print(f'-- {first} − {second}  ({name})')
    for n in (5, 10, 20):
        line = f'   {n:>2}人:'
        for seed in seeds:
            pairs = defaultdict(list)
            for (h, s, case, sd), arms in table.items():
                if h == n and sd == seed and first in arms and second in arms:
                    pairs[case].append((arms[first]['outcome'] == 'reach_goal',
                                        arms[second]['outcome'] == 'reach_goal'))
            point = (sum(a for c in pairs for a, _ in pairs[c])
                     - sum(b for c in pairs for _, b in pairs[c])) / sum(len(v) for v in pairs.values())
            line += f'  s{seed}={point*100:+6.2f}'
        print(line)

print('\n=== 交互项:差值在 20 人与 5 人之间的变化(按 case 成簇自助)===')
rng = np.random.default_rng(0)
for first, second, name in CORE:
    high = collect(table, first, second, 20)
    low = collect(table, first, second, 5)
    cases = sorted(set(high) & set(low))
    stats = np.array([[sum(a for a, _ in high[c]) - sum(b for _, b in high[c]), len(high[c]),
                       sum(a for a, _ in low[c]) - sum(b for _, b in low[c]), len(low[c])]
                      for c in cases], float)
    point = stats[:, 0].sum()/stats[:, 1].sum() - stats[:, 2].sum()/stats[:, 3].sum()
    draw = rng.integers(len(cases), size=(10000, len(cases)))
    taken = stats[draw]
    delta = (taken[:, :, 0].sum(1)/taken[:, :, 1].sum(1)
             - taken[:, :, 2].sum(1)/taken[:, :, 3].sum(1))
    low_q, high_q = np.quantile(delta, [.025, .975])
    star = '  显著' if (low_q > 0 or high_q < 0) else '  不显著'
    print(f'  {first} − {second}: Δ(20人)−Δ(5人) = {point*100:+6.2f} pp  '
          f'95%CI[{low_q*100:+6.2f},{high_q*100:+6.2f}]{star}')
