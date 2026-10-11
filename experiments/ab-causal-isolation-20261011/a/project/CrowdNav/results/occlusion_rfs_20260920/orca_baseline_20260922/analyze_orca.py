"""从 12 个原始结果文件复算 ORCA 基线。用法:python analyze_orca.py [本目录]

每个文件 = 一个(协议, 人数, 场景)单元格,200 回合,case 9400000–9400199。
ORCA 无训练、无种子,因此只报描述统计与逐单元格二项 95% 区间(Wilson)。
"""
import glob, json, math, sys
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main(root):
    cells = {}
    for f in sorted(glob.glob(str(Path(root) / 'results' / '*.json'))):
        d = json.load(open(f))
        proto, humans, *scene = Path(f).stem.split('_')
        g = d['protocol']
        assert g['controller'] == 'orca' and g['observation'] == 'current_visible_pedestrians_only' and g['occlusion']
        ev = [e['event'] for e in d['episodes']]
        assert len(ev) == 200, f
        cells[(proto, int(humans), '_'.join(scene))] = ev
    print(f'单元格 {len(cells)} 个,共 {sum(len(v) for v in cells.values())} 回合')
    print('观测输入:仅当前可见行人(与所有学习方法的合法输入一致),遮挡开\n')
    for proto in ('standard', 'severe'):
        rows = [e for (p, _, _), v in cells.items() if p == proto for e in v]
        n = len(rows); s = rows.count('reach_goal'); c = rows.count('collision'); t = rows.count('timeout')
        lo, hi = wilson(s, n)
        print(f'## {proto}  n={n}  SR {100*s/n:.2f}% [{lo:.2f}, {hi:.2f}]  CR {100*c/n:.2f}%  TO {100*t/n:.2f}%')
        for h in (5, 10, 20):
            for sc in ('circle_crossing', 'square_crossing'):
                v = cells[(proto, h, sc)]
                print(f'   {h:>2}人 {sc:<16} SR {100*v.count("reach_goal")/200:6.2f}  CR {100*v.count("collision")/200:6.2f}  TO {100*v.count("timeout")/200:6.2f}')
        print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
