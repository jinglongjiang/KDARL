"""Evidence aggregation; classifications are observations, not causal labels."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr, mannwhitneyu

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
args = parser.parse_args()
root = Path(args.root)
result = {}


def fraction_table(counts):
    total = sum(counts.values())
    return {'counts': dict(counts), 'total': total,
            'percent': {k: 100. * counts[k] / total if total else None for k in counts}}


def category(frame, epsilon=0.):
    safe = np.asarray(frame['dmin']) >= .2
    progress = np.asarray(frame['progress']) > epsilon
    if not (safe & progress).any():
        return 'D'
    if safe[frame['final']] and progress[frame['final']]:
        return 'OK'
    if frame['rawtop_filtered']:
        return 'C'
    return 'B'


def moments(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {'n': len(values), 'mean': float(values.mean()),
            'p25': float(np.percentile(values, 25)), 'median': float(np.median(values)),
            'p75': float(np.percentile(values, 75)), 'p95': float(np.percentile(values, 95))}


def case_bootstrap_differences(tables, first, second, measure, draws=5000):
    cases = sorted({k[2] for k in tables[first]})
    deltas = np.array([np.mean([measure(tables[first][key]) - measure(tables[second][key])
                               for key in tables[first] if key[2] == case]) for case in cases])
    random = np.random.default_rng(20261008)
    samples = np.mean(random.choice(deltas, (draws, len(deltas)), replace=True), axis=1)
    return {'delta_pp': float(100 * deltas.mean()),
            'case_cluster_bootstrap_ci95_pp': (100 * np.percentile(samples, [2.5, 97.5])).tolist(),
            'case_cluster_bootstrap_ci98p333_pp': (100 * np.percentile(samples, [.8333333, 99.1666667])).tolist()}


if (root / 'd0' / 'DONE.json').exists():
    tables = {}
    aggregates = {}
    for arm in ('kda', 'mamba', 'gru'):
        rows = []
        cells = []
        hashes = set()
        for cell in range(6):
            data = json.loads((root / 'd0' / arm / f'cell{cell}.json').read_text())
            assert data['checkpoint_sha256']
            hashes.add(data['checkpoint_sha256'])
            assert len(data['episodes']) == 500
            cells.append({'cell': cell, 'counts': dict(Counter(x['outcome'] for x in data['episodes']))})
            rows.extend(data['episodes'])
        assert len(hashes) == 1
        tables[arm] = {(r['human_num'], r['geometry'], r['case']): r for r in rows}
        assert len(tables[arm]) == 3000
        counts = Counter(r['outcome'] for r in rows)
        aggregates[arm] = {'counts': dict(counts), 'rates': {k: v / 3000 for k, v in counts.items()},
                           'cells': cells, 'mean_return': float(np.mean([r['return'] for r in rows]))}
    pairs = []
    for first, second in [('mamba', 'kda'), ('mamba', 'gru'), ('kda', 'gru')]:
        assert set(tables[first]) == set(tables[second])
        for metric in ('success', 'collision', 'timeout'):
            b = sum(tables[first][k]['outcome'] == metric and tables[second][k]['outcome'] != metric
                    for k in tables[first])
            c = sum(tables[first][k]['outcome'] != metric and tables[second][k]['outcome'] == metric
                    for k in tables[first])
            p = float(binomtest(b, b + c, .5).pvalue) if b + c else 1.
            interval = case_bootstrap_differences(
                tables, first, second, lambda row: float(row['outcome'] == metric))
            pairs.append({'first': first, 'second': second, 'metric': metric,
                          'b_first_only': b, 'c_second_only': c, 'discordant': b + c,
                          'mcnemar_exact_p': p, 'bonferroni_3_adjusted_p': min(1., 3 * p),
                          'significant_three_pairs': p < .05 / 3, **interval})
    result['D0'] = {'arms': aggregates, 'pairs': pairs,
                    'scope': 'Three frozen seed419 checkpoints; 500 case clusters, six cells each.'}

datasets = {}
for arm in ('kda', 'mamba', 'gru'):
    folder = root / 'd1' / arm
    paths = sorted(folder.glob('cell*-case*.json.gz'))
    if not paths:
        continue
    episodes = []
    for path in paths:
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        assert data['summary']['parity']
        episodes.append(data)
    datasets[arm] = episodes

if datasets:
    result['D1'] = {arm: {'episodes': len(eps), 'steps': sum(len(e['frames']) for e in eps),
                          'outcomes': dict(Counter(e['summary']['outcome'] for e in eps)),
                          'all_parity': all(e['summary']['parity'] for e in eps)}
                    for arm, eps in datasets.items()}
    result['safety_geometry_shadow'] = {}
    for arm, episodes in datasets.items():
        collisions = [e for e in episodes if e['summary']['outcome'] == 'collision']
        records = []
        for e in collisions:
            f = e['frames'][-1]
            i = f['final']
            records.append({'cell': e['summary']['cell'], 'case': e['summary']['case'],
                            'step': f['step'], 'predicted_endpoint_clearance': f['dmin'][i],
                            'shadow_swept_clearance': f['dmin_swept_shadow'][i],
                            'environment_dmin': f['environment_dmin'],
                            'all_unsafe_endpoint': f['all_unsafe'],
                            'critical_index_endpoint': f['critical_index']})
        mismatch_frames = [f for e in episodes for f in e['frames']
                           if f['dmin'][f['final']] >= .2
                           and f['dmin_swept_shadow'][f['final']] < .2]
        result['safety_geometry_shadow'][arm] = {
            'collision_episodes': len(collisions), 'collision_records': records,
            'collision_endpoint_margin_safe_count': sum(r['predicted_endpoint_clearance'] >= .2 for r in records),
            'collision_endpoint_noncolliding_count': sum(r['predicted_endpoint_clearance'] >= 0. for r in records),
            'executed_endpoint_margin_safe_but_swept_not_margin_safe_steps': len(mismatch_frames)}
    result['D2'], result['D3'], result['D5'], result['D8'] = {}, {}, {}, {}
    for arm, episodes in datasets.items():
        failure = [e for e in episodes if e['summary']['outcome'] != 'success']
        counts, subtypes = Counter({x: 0 for x in ('D', 'C', 'B', 'OK')}), Counter({'A1_proxy': 0, 'A2_proxy': 0})
        per_episode, by_cell = [], {}
        terminal_counts = Counter({x: 0 for x in ('D', 'C', 'B', 'OK')})
        by_outcome = {}
        B_exceptions = []
        blind_distances, blind_clearances = [], []
        for e in failure:
            cell = e['summary']['cell']
            terminal_counts.update(category(f) for f in e['frames'][-40:])
            name = e['summary']['outcome']
            by_outcome.setdefault(name, Counter({x: 0 for x in ('D', 'C', 'B', 'OK')}))
            by_cell.setdefault(cell, Counter({x: 0 for x in ('D', 'C', 'B', 'OK')}))
            ec = Counter({x: 0 for x in ('D', 'C', 'B', 'OK')})
            for f in e['frames']:
                label = category(f)
                counts[label] += 1
                ec[label] += 1
                by_cell[cell][label] += 1
                by_outcome[name][label] += 1
                if label == 'B':
                    proxy = f['critical_index'] >= 5 or f['nearest_index'] >= 5
                    subtypes['A1_proxy' if proxy else 'A2_proxy'] += 1
                    if proxy:
                        blind_distances.append(min(f['human_distances']))
                        blind_clearances.append(f['dmin'][f['final']])
                    if not f['rawtop_safe'] or f['progress'][f['rawtop']] > 0.:
                        B_exceptions.append([cell, e['summary']['case'], f['step']])
            per_episode.append(fraction_table(ec)['percent'])
        assert counts['B'] == sum(subtypes.values())
        equal_episode = {k: float(np.mean([r[k] for r in per_episode])) for k in counts} if per_episode else {}
        sensitivity = {}
        for epsilon in (1e-6, .005, .02):
            sensitivity[str(epsilon)] = fraction_table(Counter(category(f, epsilon) for e in failure for f in e['frames']))
        result['D2'][arm] = {'failure_episodes': len(failure), 'step_weighted': fraction_table(counts),
                              'terminal_last10s': fraction_table(terminal_counts),
                              'by_outcome': {k: fraction_table(v) for k, v in by_outcome.items()},
                              'episode_equal_weighted_percent': equal_episode,
                              'B_conditioned': fraction_table(subtypes),
                              'A1_proxy_nearest_distance': moments(blind_distances),
                              'A1_proxy_selected_clearance': moments(blind_clearances),
                              'A1_proxy_selected_clearance_below_0p5_fraction':
                                  float(np.mean(np.asarray(blind_clearances) < .5)) if blind_clearances else None,
                              'B_not_pure_value_cases': B_exceptions,
                              'by_cell': {k: fraction_table(v) for k, v in by_cell.items()},
                              'progress_epsilon_sensitivity': sensitivity}
        ranges = {name: [] for name in ('r_all', 'r_nonterminal', 'r_safe_nonterminal',
                                        'gamma_V_all', 'risk_all', 'gamma_V_safe_nonterminal')}
        agreement = []
        below_stand = 0
        zero_bias_changes = 0
        zero_bias_failure_changes = 0
        failure_step_total = 0
        for e in episodes:
            for f in e['frames']:
                r, v, d = (np.asarray(f[key]) for key in ('r', 'V', 'dmin'))
                ordinary = np.asarray(f['nonterminal'])
                safe_ordinary = ordinary & (d >= .2)
                risk = -.8 * np.maximum(.2 - d, 0.)
                ranges['r_all'].append(float(np.ptp(r)))
                ranges['gamma_V_all'].append(float(np.ptp(.99 * v)))
                ranges['risk_all'].append(float(np.ptp(risk)))
                if ordinary.any():
                    ranges['r_nonterminal'].append(float(np.ptp(r[ordinary])))
                if safe_ordinary.any():
                    ranges['r_safe_nonterminal'].append(float(np.ptp(r[safe_ordinary])))
                    ranges['gamma_V_safe_nonterminal'].append(float(np.ptp(.99 * v[safe_ordinary])))
                agreement.append(int(np.argmax(v)) == f['rawtop'])
                below_stand += f['candidate_below_stand_threshold']
                unbiased = np.asarray(f['final_score']).copy()
                unbiased[0] += .001
                changed = int(unbiased.argmax()) != f['final']
                zero_bias_changes += int(changed)
                if e['summary']['outcome'] != 'success':
                    failure_step_total += 1
                    zero_bias_failure_changes += int(changed)
        result['D3'][arm] = {'ranges': {k: moments(v) for k, v in ranges.items()},
                              'argmax_V_equals_argmax_r_gamma_V': float(np.mean(agreement)),
                              'candidate_commands_below_stand_threshold_count': below_stand,
                              'index0_bias_argmax_changes_offline': zero_bias_changes,
                              'index0_bias_argmax_changes_on_failure_steps': zero_bias_failure_changes,
                              'failure_step_total': failure_step_total}
        correlation = {}
        for measure in ('nearest_blind_fraction', 'critical_blind_fraction'):
            success = [e['summary'][measure] for e in episodes if e['summary']['outcome'] == 'success']
            failures = [e['summary'][measure] for e in episodes if e['summary']['outcome'] != 'success']
            groups = {}
            for name in ('success', 'collision', 'timeout'):
                groups[name] = moments([e['summary'][measure] for e in episodes if e['summary']['outcome'] == name])
            p = float(mannwhitneyu(success, failures, alternative='two-sided').pvalue) if success and failures else None
            within = {}
            for cell in (2, 3, 4, 5):
                rows = [e['summary'] for e in episodes if e['summary']['cell'] == cell]
                a = [r[measure] for r in rows if r['outcome'] == 'success']
                b = [r[measure] for r in rows if r['outcome'] != 'success']
                within[cell] = {'success': moments(a), 'failure': moments(b),
                                'mann_whitney_p_descriptive': float(mannwhitneyu(a, b).pvalue) if a and b else None}
            correlation[measure] = {'by_outcome': groups, 'pooled_p_descriptive': p, 'within_cell': within}
        line_groups = {}
        for measure in ('first5_mean', 'first5_min', 'first5_fraction_within1m'):
            by_outcome = {}
            vals, labels = [], []
            for name in ('success', 'collision', 'timeout'):
                group = []
                for e in episodes:
                    if e['summary']['outcome'] != name:
                        continue
                    distances = np.asarray(e['summary']['initial_line_distances'][:5])
                    value = {'first5_mean': distances.mean(), 'first5_min': distances.min(),
                             'first5_fraction_within1m': np.mean(distances < 1.)}[measure]
                    group.append(float(value))
                    vals.append(value)
                    labels.append(name != 'success')
                by_outcome[name] = moments(group)
            rho, p = spearmanr(vals, labels)
            line_groups[measure] = {'by_outcome': by_outcome,
                                     'spearman_failure': float(rho), 'p_descriptive': float(p)}
        result['D5'][arm] = {'blind_exposure': correlation, 'initial_line': line_groups,
                              'causal_warning': 'Exposure after actions is endogenous; comparisons are descriptive, not A1 causation.'}
    if 'kda' in datasets:
        paired = []
        all_diff, same_top = [], []
        for e in datasets['kda']:
            if e['summary']['outcome'] == 'success':
                continue
            differences = []
            for f in e['frames']:
                if 'mamba_same_window_V' not in f:
                    continue
                gp = np.flatnonzero((np.asarray(f['dmin']) >= .2) & (np.asarray(f['progress']) > 0.))
                if not len(gp):
                    continue
                kda_raw = np.asarray(f['raw_score'])
                mamba_raw = np.asarray(f['r']) + .99 * np.asarray(f['mamba_same_window_V'])
                mamba_rank = np.empty(80, int)
                mamba_rank[np.argsort(-mamba_raw, kind='stable')] = np.arange(1, 81)
                kda_rank = np.empty(80, int)
                kda_rank[np.argsort(-kda_raw, kind='stable')] = np.arange(1, 81)
                # Same reference candidate: maximal immediate progress among safe candidates.
                reference = int(gp[np.argmax(np.asarray(f['progress'])[gp])])
                delta = int(mamba_rank[reference]) - int(kda_rank[reference])
                differences.append(delta)
                all_diff.append(delta)
                same_top.append(int(kda_raw.argmax()) == int(mamba_raw.argmax()))
            if differences:
                paired.append({'cell': e['summary']['cell'], 'case': e['summary']['case'],
                               'outcome': e['summary']['outcome'],
                               'mean_mamba_minus_kda_rank': float(np.mean(differences)),
                               'steps': len(differences)})
        result['D8'] = {'same_reference_rank_delta': moments(all_diff),
                        'same_raw_top_fraction': float(np.mean(same_top)) if same_top else None,
                        'episode_level': paired,
                        'warning': 'Same windows/commands, but entire trained networks and replay differ; no architecture-only attribution.'}
    # Freeze roots without consulting counterfactual outcomes.
    choices = []
    for arm in ('kda', 'mamba'):
        for e in datasets.get(arm, []):
            if e['summary']['outcome'] == 'success':
                continue
            selected = None
            for f in e['frames']:
                if f['time'] >= 5. and f['goal_distance'] > 1. and category(f) == 'B' and f['rawtop_safe']:
                    if f['progress'][f['rawtop']] <= 0.:
                        selected = f
                        break
            if selected is None:
                continue
            choices.append({'arm': arm, 'cell': e['summary']['cell'], 'case': e['summary']['case'],
                            'step': selected['step'], 'time': selected['time'],
                            'replacement': selected['best_raw_safe_progress'],
                            'original': selected['final'], 'baseline_outcome': e['summary']['outcome'],
                            'blind_proxy': selected['critical_index'] >= 5 or selected['nearest_index'] >= 5})
            if sum(c['arm'] == arm for c in choices) >= 10:
                break
    selection_path = root / 'd7-selections.json'
    if not selection_path.exists():
        selection_path.write_text(json.dumps(choices, indent=2))
    else:
        assert json.loads(selection_path.read_text()) == choices, 'Frozen counterfactual roots changed'

for stage in ('d4', 'd6'):
    data = {}
    for path in sorted((root / stage).glob('*.json')):
        if path.name == 'summary.json':
            continue
        payload = json.loads(path.read_text())
        if 'episodes' in payload:
            counts = Counter(row['outcome'] for row in payload['episodes'])
            data[path.stem] = {'episodes': len(payload['episodes']), 'counts': dict(counts),
                               'mean_return': float(np.mean([r['return'] for r in payload['episodes']]))}
    if data:
        result[stage.upper()] = data
if (root / 'd7').exists():
    rows = []
    for path in sorted((root / 'd7').glob('*/*.json')):
        if path.name != 'summary.json':
            rows.append(json.loads(path.read_text()))
    result['D7'] = {'n': len(rows), 'records': rows}
(root / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps({key: list(value) if isinstance(value, dict) else None for key, value in result.items()}))
