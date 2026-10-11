#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

OLD = (1207, 2407, 3607, 4807, 6007, 7207, 8407, 9607)
NEW = (10807, 12007, 13207, 14407)


def sr(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 1200:
        raise RuntimeError(f'{path}: expected 1200 rows, got {len(rows)}')
    return float(np.mean([row['outcome'] == 'reach_goal' for row in rows]))


def summarize(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    intervals = {}
    for confidence in (.90, .95):
        half = float(t.ppf((1 + confidence) / 2, len(values) - 1)
                     * sd / math.sqrt(len(values)))
        intervals[str(confidence)] = [mean - half, mean + half]
    ci90, ci95 = intervals['0.9'], intervals['0.95']
    if mean >= .03 and ci95[0] > 0:
        decision = 'meaningful_gain'
    elif ci90[0] >= -.03 and ci90[1] <= .03:
        decision = 'practical_equivalence'
    else:
        decision = 'evidence_insufficient'
    return {'per_seed': values.tolist(), 'mean': mean, 'sd': sd,
            'ci90': ci90, 'ci95': ci95, 'decision': decision}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--new-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    def differences(seeds, root):
        return [sr(root / f'full_{seed}.jsonl') - sr(root / f'age_{seed}.jsonl')
                for seed in seeds]

    old = differences(OLD, args.old_root)
    new = differences(NEW, args.new_root)
    result = {'original_8': summarize(old), 'new_4': summarize(new),
              'pooled_12': summarize(old + new)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    lines = ['# Severe IL Full minus Age: 12-Seed Extension', '',
             '| Cohort | Mean | 95% CI | 90% CI | Decision |',
             '|---|---:|---:|---:|---|']
    for name, value in result.items():
        lines.append(
            f"| {name} | {100*value['mean']:+.2f} pp | "
            f"[{100*value['ci95'][0]:+.2f}, {100*value['ci95'][1]:+.2f}] | "
            f"[{100*value['ci90'][0]:+.2f}, {100*value['ci90'][1]:+.2f}] | "
            f"{value['decision']} |"
        )
    args.out.with_suffix('.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
