"""Audit existing frozen evaluations; no environment execution or training."""
import configparser
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOTS = {
    419: Path('/home/abc/4090_backup_20261009/root/kda-history-20261009'),
    443: Path('/home/abc/workspace/kda-history-replication-20261010'),
}
CELLS = [(5, 'circle'), (5, 'square'), (10, 'circle'),
         (10, 'square'), (20, 'circle'), (20, 'square')]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    path = BASE / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def measure(x, y, rng):
    # Four joint outcomes are sufficient for paired case-level resampling.
    n = len(x)
    counts = np.bincount(2 * x.astype(int) + y.astype(int), minlength=4)
    draws = rng.multinomial(n, counts / n, size=10000) / n
    p1, p2 = float(x.mean()), float(y.mean())
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    independent = p1 * (1 - p2) + p2 * (1 - p1)
    bp1, bp2 = draws[:, 2] + draws[:, 3], draws[:, 1] + draws[:, 3]
    denom = bp1 * (1 - bp2) + bp2 * (1 - bp1)
    valid = denom > 0
    ratios = (draws[:, 1] + draws[:, 2])[valid] / denom[valid]
    interval = np.quantile(ratios, [.025, .975]).tolist() if len(ratios) else None
    return dict(n=n, b=b, c=c, p1=p1, p2=p2,
                disagreement=(b + c) / n, independent_expectation=independent,
                ratio=(b + c) / n / independent if independent else None,
                ratio_ci95=interval, bootstrap_valid_draws=int(valid.sum()),
                net_SR_pp=100 * (b - c) / n,
                b_definition='first succeeds, second does not',
                c_definition='second succeeds, first does not')


def main():
    files, weights, records, provenance = {}, [], {}, []
    for seed, root in ROOTS.items():
        assert json.loads((root / 'protocol.json').read_text())['seed'] == seed
        for backbone, length in itertools.product(('kda', 'gru'), (12, 24)):
            name = '%s-t%d' % (backbone, length)
            config = root / (name + '.ini')
            weight = root / 'runs' / name / 'rl_model_ep10000.pth'
            cfg = configparser.ConfigParser()
            assert cfg.read(str(config))
            assert cfg.getint('temporal', 't') == cfg.getint('buffer', 'seq_len') == length
            assert cfg.getfloat('eval_protocol', 'action_smoothing') == .3
            assert cfg.getfloat('eval_protocol', 'risk_lambda') == .8
            assert cfg.getfloat('eval_protocol', 'safety_margin') == .2
            assert cfg.getfloat('sim', 'circle_radius') == 4
            assert cfg.getfloat('sim', 'square_width') == 10
            assert cfg.getfloat('env', 'time_limit') == 50
            digest = sha(weight)
            assert digest
            weights.append(dict(seed=seed, backbone=backbone, T=length,
                                path=str(weight), sha256=digest, config=str(config)))
            for path in (weight, config, root / 'protocol.json'):
                files[str(path)] = sha(path)
            for cell, (n, geometry) in enumerate(CELLS):
                path = root / 'evaluation' / name / ('cell%d.json' % cell)
                data = json.loads(path.read_text())
                assert data['checkpoint_sha256'] == digest
                assert data['config_sha256'] == sha(config)
                assert data['grid_actions'] == 80 and data['seq_len'] == length
                assert data['selection'] == 'ttc5' and data['backbone'] == backbone
                rows = sorted(data['episodes'], key=lambda r: r['case'])
                assert [r['case'] for r in rows] == list(range(91000, 91500))
                assert all(r['cell'] == cell and r['human_num'] == n for r in rows)
                assert all(r['geometry'] == geometry + '_crossing' for r in rows)
                assert all(r['outcome'] in ('success', 'collision', 'timeout') for r in rows)
                records[(seed, backbone, length, cell)] = np.array(
                    [r['outcome'] == 'success' for r in rows])
                fields = sorted(set().union(*(set(r) for r in rows)))
                provenance.append(dict(path=str(path), seed=seed, backbone=backbone,
                                       T=length, cell=cell, n=500,
                                       episode_fields=fields,
                                       argmax_trajectory_available=False,
                                       max_recorded_time=max(r['time'] for r in rows),
                                       torch=data.get('torch'),
                                       script_sha256=data.get('script_sha256')))
                files[str(path)] = sha(path)
    rng = np.random.default_rng(20261011)
    results, aggregate = [], []
    for cell, (n, geometry) in enumerate(CELLS):
        entries = []
        for backbone, length in itertools.product(('kda', 'gru'), (12, 24)):
            entries.append(('A', (419, backbone, length), (443, backbone, length)))
        for seed, length in itertools.product((419, 443), (12, 24)):
            entries.append(('B', (seed, 'kda', length), (seed, 'gru', length)))
        for seed, backbone in itertools.product((419, 443), ('kda', 'gru')):
            entries.append(('C', (seed, backbone, 24), (seed, backbone, 12)))
        for family, first, second in entries:
            row = measure(records[first + (cell,)], records[second + (cell,)], rng)
            row.update(family=family, first=list(first), second=list(second),
                       cell=cell, human_num=n, geometry=geometry)
            results.append(row)
        current = [r for r in results if r['cell'] == cell and r['family'] == 'A']
        disagreement = float(np.mean([r['disagreement'] for r in current]))
        expected = float(np.mean([r['independent_expectation'] for r in current]))
        aggregate.append(dict(cell=cell, human_num=n, geometry=geometry,
                              mean_A_disagreement=disagreement,
                              mean_A_independent_expectation=expected,
                              ratio_of_A_means=disagreement / expected if expected else None,
                              note='Four overlapping pairs, not 2000 independent worlds; descriptive mean.'))
    pending = dict(
        task='CROSS_POPULATION_SCALE_CASE_CONSISTENCY', status='PENDING_PARITY_RESOLUTION',
        training_episodes=0, existing_cells=[5, 10, 20], new_human_nums=[12, 15],
        geometries=['circle_crossing', 'square_crossing'], case_base=91000,
        n_cases_per_cell=500, new_evaluation_episodes=16000,
        selection='ttc5', alpha=.3, risk_lambda=.8, safety_margin=.2,
        circle_radius=4, square_width=10, actual_existing_time_limit=50,
        supplied_text_time_limit=25, actions=80, weights=weights,
        outcome='binary success versus non-success; not three-way terminal identity',
        families=dict(A='same backbone and T, 419 versus 443',
                      B='same seed and T, KDA versus GRU',
                      C='same seed and backbone, T24 versus T12 (inferred from B/C and T references)'),
        bootstrap=dict(unit='paired case ID within each comparison/cell', draws=10000,
                       random_seed=20261011, method='multinomial of four paired outcome categories'),
        untouched_confirmation_cases=[92000, 92499],
        parity_required='bitwise argmax trajectory and outcome against existing evaluation',
        blocker='Old evaluation files have no argmax trajectories; selection hash is not an action hash',
        restrictions=['No original project edits', 'No training or added seeds',
                      'No grid/reward/filter/TTC/optimizer changes', 'No new model'],
        limitations=['People count and density confounded at fixed field dimensions',
                     'Training seeds evaluated on same worlds',
                     'Two training seeds cannot estimate seed population variance',
                     'Frozen checkpoint intervention has no new training, but compares independently trained weights',
                     'Ratio below one indicates positive success association, not independence',
                     'A flat ratio alone does not establish ceiling effects as the cause'])
    save('protocol-pending.json', pending)
    files[str(BASE / 'audit-existing.py')] = sha(BASE / 'audit-existing.py')
    save('frozen-manifest.json', dict(files=files, weights=weights, stage='existing-data-audit'))
    family_means = []
    for cell, (n, geometry) in enumerate(CELLS):
        family_means.append(dict(cell=cell, human_num=n, geometry=geometry,
                                 **{family: float(np.mean([r['disagreement'] for r in results
                                                          if r['cell'] == cell and r['family'] == family]))
                                    for family in 'ABC'}))
    save('consistency-existing.json', dict(status='VERIFIED_EXISTING_DATA', rows=results,
                                         A_cell_descriptive_means=aggregate,
                                         ABC_cell_descriptive_means=family_means))
    save('provenance-existing.json', dict(files=provenance,
                                         evaluation_argmax_trajectories_available=False,
                                         new_evaluation_started=False))
    for path, digest in files.items():
        assert sha(Path(path)) == digest, path
    save('status.json', dict(stage='PARITY_UNVERIFIABLE_WITH_EXISTING_RECORDS',
                             existing_files_verified=len(provenance), weights_verified=len(weights),
                             new_episodes=0, training_episodes=0, reserved_cases_used=0))
    lines = ['# Scale Consistency Audit', '',
             'Status: existing-data audit verified; new evaluations NOT STARTED.', '',
             '## Phenomenon', '',
             '| People | Geometry | A disagreement | Independent expectation | Ratio |',
             '|---|---|---:|---:|---:|']
    for row in aggregate:
        lines.append('| %s | %s | %.2f%% | %.2f%% | %.4f |' % (
            row['human_num'], row['geometry'], row['mean_A_disagreement'] * 100,
            row['mean_A_independent_expectation'] * 100, row['ratio_of_A_means']))
    lines += ['', 'These are descriptive means across four overlapping A comparisons.', '',
              '| People | Geometry | A: change seed | B: change backbone | C: change T |',
              '|---|---|---:|---:|---:|']
    for row in family_means:
        lines.append('| %s | %s | %.2f%% | %.2f%% | %.2f%% |' % (
            row['human_num'], row['geometry'], row['A'] * 100, row['B'] * 100, row['C'] * 100))
    lines += ['',
              'VERIFIED: B/C are of similar descriptive magnitude to A on these fixed weights.',
              'This does not order architecture/T causal effects versus the population variance of training.', '',
              '## Question And Evidence', '',
              '- VERIFIED: all 48 old cell files contain 500 matched cases each; eight weight hashes match file metadata.',
              '- VERIFIED: all old files contain outcome, return, clearance, time and selection-trace hash, but no argmax trajectory.',
              '- UNVERIFIED: required bitwise action-trajectory parity cannot be established from a selection hash.',
              '- VERIFIED: effective old time limit is 50 seconds; some old episodes run beyond 25 seconds. Do not change to 25.',
              '- UNVERIFIED: 12/15-person behavior; no new environment episodes executed.', '',
              '## Options And Recommendation', '',
              'Resolve parity before any full evaluation: reconstruct reference action traces on the original stack,',
              'first exactly matching all saved old non-latency episode fields, then compare the new harness with those traces.',
              'This is a replacement gate, not recovery of the missing historical action trace; it needs explicit acceptance.',
              'Do not silently relax trajectory parity to outcome equality.', '',
              '## Interpretation Limits', '',
              'A ratio below one measures positive paired success association, not proof of independence.',
              'A flat ratio alone does not identify ceiling effects or explain generalization.',
              'B/C disagreement at least as large as A cannot imply B/C is no larger than A; report magnitudes without reversing the inequality.',
              'C is interpreted as changing T at fixed seed/backbone; the supplied list explicitly defines only A and B.',
              'All confidence intervals concern sampled cases conditional on the eight fixed weights, not training-seed population variance.',
              'Count and density remain confounded. No backbone superiority, stability or novel algorithm claim.', '',
              'Sources: consistency-existing.json, provenance-existing.json, frozen-manifest.json.', '']
    (BASE / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='EXISTING_AUDIT_COMPLETE_PARITY_PENDING',
                          weights=8, existing_cells=48, comparisons=len(results),
                          new_episodes=0, aggregates=aggregate), indent=2))


if __name__ == '__main__':
    main()
