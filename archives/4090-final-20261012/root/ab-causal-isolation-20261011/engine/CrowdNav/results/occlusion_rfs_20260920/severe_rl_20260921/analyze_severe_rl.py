#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

ARMS = ('mean', 'age', 'full')
SEEDS = (1207, 2407, 3607, 4807, 6007, 7207, 8407, 9607)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rate(rows, field='outcome', value='reach_goal'):
    return float(np.mean([row[field] == value for row in rows]))


def interval(values, confidence):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return [mean, mean]
    half = float(t.ppf((1 + confidence) / 2, len(values) - 1)
                 * values.std(ddof=1) / math.sqrt(len(values)))
    return [mean - half, mean + half]


def contrast(name, values):
    mean = float(np.mean(values))
    ci95, ci90 = interval(values, .95), interval(values, .90)
    if mean >= .03 and ci95[0] > 0:
        decision = 'meaningful_gain'
    elif ci90[0] >= -.03 and ci90[1] <= .03:
        decision = 'practical_equivalence'
    else:
        decision = 'evidence_insufficient'
    return {'name': name, 'per_seed': values, 'mean': mean,
            'ci95': ci95, 'ci90': ci90, 'decision': decision}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--il-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    result = {'protocols': {}, 'training_integrity': {}}
    rows = {}
    for protocol in ('severe', 'standard'):
        rows[protocol] = {}
        result['protocols'][protocol] = {'arms': {}, 'contrasts': {}}
        for arm in ARMS:
            per_seed = {}
            for seed in SEEDS:
                path = args.root / f'eval_{protocol}' / f'{arm}_{seed}.jsonl'
                data = read_jsonl(path)
                if len(data) != 1200:
                    raise RuntimeError(f'{path}: expected 1200 rows, got {len(data)}')
                per_seed[seed] = data
            rows[protocol][arm] = per_seed
            all_rows = [row for seed in SEEDS for row in per_seed[seed]]
            result['protocols'][protocol]['arms'][arm] = {
                'sr': rate(all_rows),
                'cr': rate(all_rows, value='collision'),
                'tr': rate(all_rows, value='timeout'),
                'per_seed_sr': [rate(per_seed[seed]) for seed in SEEDS],
                'by_humans': {
                    str(humans): rate([row for row in all_rows if row['humans'] == humans])
                    for humans in (5, 10, 20)
                },
            }
        for high, low in (('age', 'mean'), ('full', 'age')):
            values = [rate(rows[protocol][high][seed])
                      - rate(rows[protocol][low][seed]) for seed in SEEDS]
            result['protocols'][protocol]['contrasts'][f'{high}-{low}'] = contrast(
                f'{high}-{low}', values
            )

    for arm in ARMS:
        gains = []
        integrity = []
        for seed in SEEDS:
            training = json.loads(
                (args.root / 'runs' / f'{arm}_{seed}' / 'results.json').read_text()
            )
            accepted = sum(bool(row.get('accepted')) for row in training['policy_updates'])
            integrity.append({
                'seed': seed,
                'status': training['status'],
                'environment_steps': training['rl_environment_steps'],
                'optimizer_updates': training['rl_optimizer_updates'],
                'accepted_updates': accepted,
                'selected_rl_step': training['selected_rl_step'],
            })
            il = read_jsonl(args.il_root / f'{arm}_{seed}.jsonl')
            gains.append(rate(rows['severe'][arm][seed]) - rate(il))
        result['training_integrity'][arm] = integrity
        result['protocols']['severe']['arms'][arm]['rl_minus_il'] = contrast(
            f'{arm}:rl-il', gains
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))

    report = ['# Severe-Trained RL Results', '',
              'Primary inference uses Severe testing and eight paired training seeds.', '']
    for protocol in ('severe', 'standard'):
        report += [f'## {protocol.title()} Test', '',
                   '| Arm | SR | CR | TR | 5 humans | 10 humans | 20 humans |',
                   '|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            value = result['protocols'][protocol]['arms'][arm]
            report.append(
                f"| {arm} | {100*value['sr']:.2f}% | {100*value['cr']:.2f}% | "
                f"{100*value['tr']:.2f}% | {100*value['by_humans']['5']:.2f}% | "
                f"{100*value['by_humans']['10']:.2f}% | "
                f"{100*value['by_humans']['20']:.2f}% |"
            )
        report += ['', '| Contrast | Mean | 95% CI | 90% CI | Decision |',
                   '|---|---:|---:|---:|---|']
        for value in result['protocols'][protocol]['contrasts'].values():
            report.append(
                f"| {value['name']} | {100*value['mean']:+.2f} pp | "
                f"[{100*value['ci95'][0]:+.2f}, {100*value['ci95'][1]:+.2f}] | "
                f"[{100*value['ci90'][0]:+.2f}, {100*value['ci90'][1]:+.2f}] | "
                f"{value['decision']} |"
            )
        report.append('')
    report += ['## Severe IL to RL', '', '| Arm | Mean | 95% CI | Decision |',
               '|---|---:|---:|---|']
    for arm in ARMS:
        value = result['protocols']['severe']['arms'][arm]['rl_minus_il']
        report.append(
            f"| {arm} | {100*value['mean']:+.2f} pp | "
            f"[{100*value['ci95'][0]:+.2f}, {100*value['ci95'][1]:+.2f}] | "
            f"{value['decision']} |"
        )
    args.out.with_suffix('.md').write_text('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
