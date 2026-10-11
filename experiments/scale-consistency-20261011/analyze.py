"""Case-paired scale consistency, conditional on eight frozen weights."""
import itertools
import json
from pathlib import Path

import numpy as np

from audit_existing_import import measure
from common import BASE, CELLS, load_protocol, save, verify


def main():
    verified = verify()
    protocol = load_protocol()
    records, sources = {}, []
    for item in protocol['weights']:
        seed, backbone, length = item['seed'], item['backbone'], item['T']
        for cell, (n, geometry) in enumerate(CELLS):
            if cell < 6:
                path = Path(item['config']).parent / 'evaluation' / ('%s-t%d' % (backbone, length)) / ('cell%d.json' % cell)
            else:
                path = BASE / 'evaluation' / ('%d-person' % n) / geometry / ('s%d-%s-t%d.json' % (seed, backbone, length))
            data = json.loads(path.read_text())
            assert data['checkpoint_sha256'] == item['sha256']
            assert data['config_sha256'] == item['config_sha256']
            assert data['grid_actions'] == 80 and data['seq_len'] == length
            assert data['selection'] == 'ttc5' and data['backbone'] == backbone
            rows = sorted(data['episodes'], key=lambda row: row['case'])
            assert [row['case'] for row in rows] == list(range(91000, 91500))
            assert all(row['human_num'] == n and row['geometry'] == geometry for row in rows)
            records[(seed, backbone, length, cell)] = np.array([row['outcome'] == 'success' for row in rows])
            sources.append(str(path))
    assert len(records) == 80
    rng = np.random.default_rng(20261011)
    # The same resampled case IDs are retained across cells, models and seeds for trends.
    indices = rng.integers(0, 500, (10000, 500))
    results, aggregates, ratio_boot = [], [], {}
    for cell, (n, geometry) in enumerate(CELLS):
        entries = {'A': [], 'B': [], 'C': []}
        for backbone, length in itertools.product(('kda', 'gru'), (12, 24)):
            entries['A'].append(((419, backbone, length), (443, backbone, length)))
        for seed, length in itertools.product((419, 443), (12, 24)):
            entries['B'].append(((seed, 'kda', length), (seed, 'gru', length)))
        for seed, backbone in itertools.product((419, 443), ('kda', 'gru')):
            entries['C'].append(((seed, backbone, 24), (seed, backbone, 12)))
        for family, pairs in entries.items():
            numer, denom = np.zeros(10000), np.zeros(10000)
            point_num, point_den = [], []
            for first, second in pairs:
                x, y = records[first + (cell,)], records[second + (cell,)]
                row = measure(x, y, rng)
                row.update(family=family, first=list(first), second=list(second),
                           cell=cell, human_num=n, geometry=geometry)
                results.append(row)
                px, py = x[indices].mean(axis=1), y[indices].mean(axis=1)
                numer += (x ^ y)[indices].mean(axis=1)
                denom += px * (1 - py) + py * (1 - px)
                point_num.append(row['disagreement'])
                point_den.append(row['independent_expectation'])
            ratio = np.divide(numer, denom, out=np.full(10000, np.nan), where=denom > 0)
            ratio_boot[(family, cell)] = ratio
            valid = np.isfinite(ratio)
            aggregates.append(dict(cell=cell, human_num=n, geometry=geometry, family=family,
                disagreement=float(np.mean(point_num)), independent_expectation=float(np.mean(point_den)),
                ratio=float(sum(point_num) / sum(point_den)) if sum(point_den) else None,
                ratio_ci95=np.quantile(ratio[valid], [.025, .975]).tolist() if valid.any() else None,
                aggregation='Ratio of four overlapping pair means; case-cluster resampling preserves dependencies'))
    trends = []
    for geometry in ('circle_crossing', 'square_crossing'):
        group = sorted([row for row in aggregates if row['geometry'] == geometry and row['family'] == 'A'],
                       key=lambda row: row['human_num'])
        counts = np.array([row['human_num'] for row in group], dtype=float)
        ratios = np.array([row['ratio'] for row in group])
        coeff = (counts - counts.mean()) / np.sum((counts - counts.mean()) ** 2)
        samples = np.stack([ratio_boot[('A', row['cell'])] for row in group], axis=1)
        valid = np.isfinite(samples).all(axis=1)
        slopes = samples[valid] @ coeff
        trends.append(dict(geometry=geometry, people=counts.tolist(), ratios=ratios.tolist(),
                           strictly_monotonic_increase=bool(np.all(np.diff(ratios) > 0)),
                           descriptive_linear_slope=float(ratios @ coeff),
                           slope_ci95=np.quantile(slopes, [.025, .975]).tolist(),
                           interpretation='Descriptive trend check, not equivalence proof or new confirmatory test'))
    save(BASE / 'consistency.json', dict(status='COMPLETED_MEASUREMENT_ONLY', comparisons=results,
                                        family_cell_aggregates=aggregates, A_trends=trends,
                                        new_episodes=16000, reused_episodes=24000,
                                        training_seeds=2, sources=sources,
                                        frozen_verified=verified,
                                        no_seed_population_variance_estimate=True))
    lines = ['# Cross-Population Scale Consistency', '',
             'Verified: 16000 new frozen evaluations, 24000 reused evaluations; no training.', '',
             '## Phenomenon', '',
             '| People | Geometry | A change seed | B change backbone | C change T | A independent expectation | A ratio [95% CI] |',
             '|---|---|---:|---:|---:|---:|---|']
    for n in (5, 10, 12, 15, 20):
        for geometry in ('circle_crossing', 'square_crossing'):
            rows = {row['family']: row for row in aggregates if row['human_num'] == n and row['geometry'] == geometry}
            a = rows['A']
            lines.append('| %d | %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.4f [%.4f, %.4f] |' % (
                n, geometry, 100*a['disagreement'], 100*rows['B']['disagreement'],
                100*rows['C']['disagreement'], 100*a['independent_expectation'],
                a['ratio'], *a['ratio_ci95']))
    lines += ['', 'Sources: 80 cell files listed in consistency.json; 500 paired cases per cell/weight.', '',
              '## Question', '',
              'Does normalized case disagreement remain approximately flat on the finer people-count axis?',
              'This is not a search for a variance onset or a performance/backbone ranking.', '',
              '## Evidence', '',
              'Verified: four parity cells passed outcome/time/selection-hash equality for 2000 cases and numeric tolerances.',
              'This is saved-field equivalence, NOT a historical action/state-trajectory reproduction.',
              'Verified: 96 new-cell smoke episodes; actual crowd count, 80 actions and 50-second limit checked.',
              'Verified: all eight weight/config hashes rechecked; original project unchanged.', '',
              '### Descriptive Trend', '', '```json', json.dumps(trends, indent=2), '```', '',
              'No predefined flatness equivalence margin was supplied. A slope interval containing zero does not prove flatness.',
              'Ratios below one show positive paired success association, not independence.', '',
              '### All Pairwise Measurements', '',
              '| People | Geometry | Family | First | Second | b | c | Disagree % | Independent % | Ratio [95% CI] | Net SR pp |',
              '|---|---|---|---|---|---:|---:|---:|---:|---|---:|']
    for row in sorted(results, key=lambda row: (row['human_num'], row['geometry'], row['family'], row['first'])):
        ci = row['ratio_ci95']
        ratio = 'undefined' if row['ratio'] is None else '%.4f [%.4f, %.4f]' % (row['ratio'], *ci)
        lines.append('| %d | %s | %s | %s | %s | %d | %d | %.2f | %.2f | %s | %+.2f |' % (
            row['human_num'], row['geometry'], row['family'], row['first'], row['second'], row['b'], row['c'],
            100*row['disagreement'], 100*row['independent_expectation'], ratio, row['net_SR_pp']))
    lines += ['', 'b = first success/second non-success; c = the reverse. Net SR cancels disagreements.', '',
              '## Options And Recommendation', '',
              'Report the finer-axis measurements and A/B/C magnitudes without backbone stability claims.',
              'Do not start GRU T12 extra-seed training or any new algorithm automatically.',
              'A flat normalized pattern is compatible with a ceiling-effect account but does not causally prove it.', '',
              '## Limits And Engineering Note', '',
              '- Fixed circle radius 4 and square width 10 confound people count with spatial density.',
              '- Training seeds share test worlds. Case bootstrap is conditional on eight weights, not a training-seed population interval.',
              '- These weights come from independent training runs: no new training is done, but training-run differences are present in A/B/C.',
              '- [eval_protocol] time_limit=25 is ignored; [env] time_limit=50 is effective. Enabling the dead key later changes comparability.',
              '- A 50-second timeout has nominal travel-time slack but is not alone proof of stalling.',
              '- Selection IDs are a lossy state summary. The same hash need not imply identical actions or states.',
              '- Net differences do not replace disagreement; no architecture superiority, novel algorithm or explained generalization claim.', '',
              'All restrictions maintained: zero training, no new seed, no original-project edits, no 92000-92499 cases.', '']
    (BASE / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='COMPLETED_MEASUREMENT_ONLY', new_episodes=16000, comparisons=len(results))))


if __name__ == '__main__':
    main()
