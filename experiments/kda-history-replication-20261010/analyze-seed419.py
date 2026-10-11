"""Describe existing seed419 interaction; this is not a new replication."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats


def analyze(source):
    rows = {}
    for arm in ('kda', 'gru'):
        for length in (12, 24):
            entries = []
            for cell in range(6):
                data = json.loads((source / 'evaluation' / ('%s-t%d' % (arm, length)) / ('cell%d.json' % cell)).read_text())
                assert data['selection'] == 'ttc5' and data['seq_len'] == length and data['grid_actions'] == 80
                assert len(data['episodes']) == 500
                entries.extend(data['episodes'])
            mapped = {(e['cell'], e['case']): e for e in entries}
            assert len(mapped) == len(entries) == 3000
            rows[arm, length] = mapped
    keys = sorted(rows['kda', 12])
    assert all(set(r) == set(keys) for r in rows.values())
    cases = sorted({case for _, case in keys})
    assert cases == list(range(91000, 91500))
    rng = np.random.default_rng(20261010)
    samples = rng.integers(0, len(cases), size=(10000, len(cases)))
    counts = {}
    changes = {}
    tests = {}
    for arm in ('kda', 'gru'):
        tests[arm] = {}
        for metric in ('success', 'collision', 'timeout'):
            old, new = rows[arm, 12], rows[arm, 24]
            b = sum(old[k]['outcome'] == metric and new[k]['outcome'] != metric for k in keys)
            c = sum(old[k]['outcome'] != metric and new[k]['outcome'] == metric for k in keys)
            if b + c:
                if hasattr(stats, 'binomtest'):
                    p = float(stats.binomtest(b, b + c, .5).pvalue)
                else:
                    p = float(stats.binom_test(b, b + c, .5))
            else:
                p = 1.
            tests[arm][metric] = dict(delta_t24_minus_t12_pp=100 * (c-b)/len(keys),
                                     b_t12_only=b, c_t24_only=c, exact_mcnemar_p=p,
                                     prior_family6_corrected_p=min(1., 6*p))
        for length in (12, 24):
            counts[arm, length] = sum(rows[arm, length][k]['outcome'] != 'success' for k in keys)
            assert 0 < counts[arm, length] < len(keys)
        changes[arm] = np.array([np.mean([float(rows[arm, 24][cell, case]['outcome'] == 'success')
                                         - float(rows[arm, 12][cell, case]['outcome'] == 'success')
                                         for cell in range(6)]) for case in cases])
    interaction = changes['kda'] - changes['gru']
    draws = 100 * interaction[samples].mean(axis=1)
    log_effect = {}
    failures = {}
    for arm in ('kda', 'gru'):
        f = {t: np.array([np.mean([rows[arm, t][cell, case]['outcome'] != 'success'
                                   for cell in range(6)]) for case in cases]) for t in (12, 24)}
        f12, f24 = f[12].mean(), f[24].mean()
        effect = np.log(f24/(1-f24)) - np.log(f12/(1-f12))
        log_effect[arm] = effect
        failures[arm] = dict(failure_counts={str(t): counts[arm, t] for t in (12, 24)},
                             failure_percent_t12=100*f12, failure_percent_t24=100*f24,
                             marginal_failure_odds_ratio_t24_vs_t12=float(np.exp(effect)))
        sampled = {t: f[t][samples].mean(axis=1) for t in (12, 24)}
        assert all(np.all((x > 0) & (x < 1)) for x in sampled.values())
        failures[arm]['bootstrap_log_odds_effect'] = np.log(sampled[24]/(1-sampled[24])) - np.log(sampled[12]/(1-sampled[12]))
    log_draws = failures['kda'].pop('bootstrap_log_odds_effect') - failures['gru'].pop('bootstrap_log_odds_effect')
    return dict(status='已验证通过', scope='Existing seed419 development data; NOT second-seed confirmation',
                n_paired_episodes=3000, n_case_clusters=500, bootstrap_replicates=10000,
                sign_convention='D=(SR_KDA24-SR_KDA12)-(SR_GRU24-SR_GRU12)',
                interaction_pp=float(100*interaction.mean()), descriptive_case_cluster_ci95_pp=np.percentile(draws,[2.5,97.5]).tolist(),
                marginal_log_failure_odds_interaction=float(log_effect['kda']-log_effect['gru']),
                descriptive_case_cluster_log_odds_ci95=np.percentile(log_draws,[2.5,97.5]).tolist(),
                failure_odds=failures, group_tests=tests,
                limits=['Intervals are descriptive; new joint confirmatory family has not been specified',
                        'Marginal failure odds ratio is not the matched-pair McNemar odds ratio',
                        'Two training seeds cannot estimate architecture-level seed variance',
                        'History length changes training dynamics, not just deployment memory'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = analyze(args.source)
    assert not args.out.exists(), 'Preserve previous analysis'
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(output, ensure_ascii=False, indent=2))
