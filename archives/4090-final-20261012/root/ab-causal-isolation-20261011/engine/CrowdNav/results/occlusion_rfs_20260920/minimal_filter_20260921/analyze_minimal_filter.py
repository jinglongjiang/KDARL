#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t

ARMS = ('mean', 'age')
SEEDS = (1207, 2407, 3607, 4807, 6007, 7207, 8407, 9607)


def interval(values, confidence=.90):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf((1 + confidence) / 2, len(values) - 1)
                 * values.std(ddof=1) / math.sqrt(len(values)))
    return [mean - half, mean + half]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for arm in ARMS:
        seed_sr, seed_cr, all_rows, deviations = [], [], [], []
        agreements = 0
        action_count = 0
        for seed in SEEDS:
            path = args.root / 'records' / f'{arm}_{seed}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            if len(rows) != 1200:
                raise RuntimeError(f'{path}: expected 1200 rows, got {len(rows)}')
            all_rows.extend(rows)
            reference_success = np.mean(
                [row['reference']['outcome'] == 'reach_goal' for row in rows]
            )
            minimal_success = np.mean(
                [row['minimal']['outcome'] == 'reach_goal' for row in rows]
            )
            reference_collision = np.mean(
                [row['reference']['outcome'] == 'collision' for row in rows]
            )
            minimal_collision = np.mean(
                [row['minimal']['outcome'] == 'collision' for row in rows]
            )
            seed_sr.append(float(minimal_success - reference_success))
            seed_cr.append(float(minimal_collision - reference_collision))
            for row in rows:
                agreements += sum(action['agreement'] for action in row['actions'])
                action_count += len(row['actions'])
                deviations.extend(action['deviation'] for action in row['actions'])

        outcome_agreement = sum(row['outcome_agreement'] for row in all_rows)
        outcome_ci = binomtest(outcome_agreement, len(all_rows)).proportion_ci(.95)
        sr_ci, cr_ci = interval(seed_sr), interval(seed_cr)
        performance_equivalent = (
            sr_ci[0] >= -.03 and sr_ci[1] <= .03
            and cr_ci[0] >= -.02 and cr_ci[1] <= .02
        )
        strong_implementation_equivalent = (
            performance_equivalent and outcome_ci.low >= .95
        )
        values = np.asarray(deviations)
        result[arm] = {
            'sr_difference_minimal_minus_rfs': float(np.mean(seed_sr)),
            'sr_ci90': sr_ci,
            'cr_difference_minimal_minus_rfs': float(np.mean(seed_cr)),
            'cr_ci90': cr_ci,
            'outcome_agreement': outcome_agreement / len(all_rows),
            'outcome_agreement_ci95': [outcome_ci.low, outcome_ci.high],
            'action_agreement': agreements / action_count,
            'action_deviation': {
                'mean': float(values.mean()),
                'median': float(np.median(values)),
                'p90': float(np.percentile(values, 90)),
                'max': float(values.max()),
            },
            'performance_equivalent': performance_equivalent,
            'strong_implementation_equivalent': strong_implementation_equivalent,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    lines = ['# Minimal-Filter Navigation Equivalence', '',
             '| Arm | SR diff (90% CI) | CR diff (90% CI) | Outcome agreement | Action agreement | Decision |',
             '|---|---:|---:|---:|---:|---|']
    for arm, value in result.items():
        decision = ('strong equivalence' if value['strong_implementation_equivalent']
                    else 'performance equivalence' if value['performance_equivalent']
                    else 'not equivalent')
        lines.append(
            f"| {arm} | {100*value['sr_difference_minimal_minus_rfs']:+.2f} pp "
            f"[{100*value['sr_ci90'][0]:+.2f}, {100*value['sr_ci90'][1]:+.2f}] | "
            f"{100*value['cr_difference_minimal_minus_rfs']:+.2f} pp "
            f"[{100*value['cr_ci90'][0]:+.2f}, {100*value['cr_ci90'][1]:+.2f}] | "
            f"{100*value['outcome_agreement']:.2f}% | "
            f"{100*value['action_agreement']:.2f}% | {decision} |"
        )
    args.out.with_suffix('.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
