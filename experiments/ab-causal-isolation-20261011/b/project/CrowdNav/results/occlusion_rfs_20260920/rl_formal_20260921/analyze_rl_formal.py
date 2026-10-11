"""从原始逐回合记录复算 RESULTS_RL_FORMAL.md 的全部数字。

用法(在 results/occlusion_rfs_20260920/ 下):
  mkdir -p /tmp/pilot && tar -xzf rl_pilot_20260921/records_raw.tar.gz -C /tmp/pilot
  python rl_formal_20260921/analyze_rl_formal.py rl_formal_20260921 /tmp/pilot std_eval_fixed
数据:
  rl_formal_20260921/eval_std, eval_severe  —— 正式 24 组(Sensor/Mean 8 种子,Age/Full 新 4 种子),git 中明文存放
  rl_pilot_20260921/records_raw.tar.gz      —— pilot 8 组(Age/Full 种子 1207/2407/3607/4807),LFS
  std_eval_fixed                            —— Standard IL fixed.pt 基线(已在 git)
统计:训练种子为推断单位,seed-level t 区间(与全部预注册一致)。
"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

T975 = {3: 4.303, 4: 3.182, 7: 2.447, 8: 2.365}   # t_{0.975, n-1}
ARMS = ('sensor', 'mean', 'age', 'full')
NEW = (6007, 7207, 8407, 9607)


def load(*dirs):
    table = defaultdict(lambda: defaultdict(list))   # (arm, seed) -> humans -> outcomes
    for d in dirs:
        for f in glob.glob(str(Path(d) / '*.jsonl')):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    table[(r['arm'], r['seed'])][r['humans']].append(r['outcome'])
    return table


def rate(table, arm, seed, outcome='reach_goal', humans=None):
    cells = table[(arm, seed)]
    rows = [o for h, v in cells.items() if humans in (None, h) for o in v]
    return 100. * sum(o == outcome for o in rows) / len(rows)


def seeds_of(table, arm):
    return sorted(s for a, s in table if a == arm)


def interval(values):
    v = np.asarray(values, float); n = len(v)
    half = T975[n] * v.std(ddof=1) / np.sqrt(n)
    return v.mean(), v.mean() - half, v.mean() + half, int((v > 0).sum()), n


def main(formal, pilot, il_dir):
    formal, pilot = Path(formal), Path(pilot)
    std = load(formal / 'eval_std', pilot / 'eval_std')
    sev = load(formal / 'eval_severe', pilot / 'eval_sev')
    il = load(il_dir)
    print('## 完整性')
    for name, t in (('Standard', std), ('Severe', sev)):
        n = sum(len(v) for c in t.values() for v in c.values())
        print(f'  {name}: {len(t)} 个 arm×seed,{n} 回合')

    print('\n## 1. 独立确认(仅新种子 6007/7207/8407/9607,Standard)')
    for arm in ('age', 'full'):
        d = [rate(std, arm, s) - rate(il, arm, s) for s in NEW]
        m, lo, hi, pos, n = interval(d)
        verdict = '确认' if pos == 4 and m >= 5 else '未确认'
        print(f'  {arm:<6} {" ".join(f"{x:+.2f}" for x in d)}  均值 {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  {pos}/{n}  → {verdict}')

    for label, t in (('Standard', std), ('Severe OOD', sev)):
        print(f'\n## {label} 四臂(8 种子)')
        print('  臂      SR     CR     TO    5人   10人   20人')
        for arm in ARMS:
            ss = seeds_of(t, arm)
            f = lambda **k: np.mean([rate(t, arm, s, **k) for s in ss])
            print(f'  {arm:<6} {f():6.2f} {f(outcome="collision"):6.2f} {f(outcome="timeout"):6.2f} '
                  f'{f(humans=5):6.2f} {f(humans=10):6.2f} {f(humans=20):6.2f}')

    print('\n## RL − IL(Standard,同种子配对)')
    for arm in ARMS:
        ss = seeds_of(std, arm)
        m, lo, hi, pos, n = interval([rate(std, arm, s) - rate(il, arm, s) for s in ss])
        print(f'  {arm:<6} IL {np.mean([rate(il, arm, s) for s in ss]):.2f} → RL {np.mean([rate(std, arm, s) for s in ss]):.2f}'
              f'  {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  {pos}/{n}')

    for label, t in (('Standard', std), ('Severe OOD', sev)):
        print(f'\n## {label} 信息阶梯')
        for a, b in (('mean', 'sensor'), ('age', 'mean'), ('full', 'age')):
            ss = sorted(set(seeds_of(t, a)) & set(seeds_of(t, b)))
            m, lo, hi, pos, n = interval([rate(t, a, s) - rate(t, b, s) for s in ss])
            print(f'  {a} − {b}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  {pos}/{n}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
