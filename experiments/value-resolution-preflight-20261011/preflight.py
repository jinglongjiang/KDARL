"""CPU-only reanalysis of archived, identical-window frozen-network scores."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parent
P1 = Path('/home/abc/workspace/route-b-20261010/p1-margin')
CF = Path('/home/abc/workspace/ttc5-causal-20261011/work')
MODELS = ['kda-s419-t24', 'kda-s443-t24']
GAMMA = .99


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def save(name, data):
    (BASE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def key(row):
    return row['cell'], row['case'], row['step']


def stats(rows):
    if not rows:
        return {'steps': 0}
    a = np.asarray([[r['m419'], r['m443'], r['centered_disagreement_state_median'],
                     r['pair_disagreement']] for r in rows])
    margins = np.median(np.abs(a[:, :2]), axis=0)
    d = float(np.median(a[:, 2]))
    return dict(steps=len(rows), episodes=len({(r['cell'], r['case']) for r in rows}),
        median_abs_M419=float(margins[0]), median_abs_M443=float(margins[1]),
        median_state_disagreement=d,
        ratios=[d / float(x) if x else None for x in margins],
        median_pair_disagreement=float(np.median(a[:, 3])),
        preference_agreement=float(np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1]))),
        ties419=int(np.sum(a[:, 0] == 0)), ties443=int(np.sum(a[:, 1] == 0)))


def bootstrap(rows, repeats=2000):
    # Resample whole (cell, case) trajectories, never individual control steps.
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r['cell'], r['case']), []).append(i)
    groups = [np.asarray(g) for g in groups.values()]
    data = np.asarray([[abs(r['m419']), abs(r['m443']),
                        r['centered_disagreement_state_median'],
                        float(np.sign(r['m419']) == np.sign(r['m443']))] for r in rows])
    rng = np.random.default_rng(20261011)
    samples = []
    for _ in range(repeats):
        index = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        a = data[index]
        med = np.median(a[:, :3], axis=0)
        samples.append([med[2] / max(med[0], 1e-15),
                        med[2] / max(med[1], 1e-15), np.mean(a[:, 3])])
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return dict(unit='(cell, case), whole episode clusters; not training-seed uncertainty',
                repeats=repeats, ratios95=bounds[:, :2].T.tolist(),
                preference_agreement95=bounds[:, 2].tolist())


def resolution():
    result = []
    sources = []
    for model in MODELS:
        for cell in (4, 5):
            path = CF / ('branches-%s-cell%d.jsonl.gz' % (model, cell))
            sources.append(path)
            for line, record in enumerate(load(path), 1):
                root = record['root']
                snap = root['snapshot']
                branches = record['branches']
                baseline = next(b for b in branches if 'A0' in b['labels'])
                for alt in branches:
                    if alt is baseline:
                        continue
                    i, j = baseline['candidate_index'], alt['candidate_index']
                    if 0 in (i, j):
                        continue
                    dq = GAMMA * (snap['values'][j] - snap['values'][i])
                    dg = alt['root_return'] - baseline['root_return']
                    row = dict(model=model, cell=cell, case=root['case'], root_id=root['root_id'],
                        root_step=root['step'], a0=i, a1=j, delta_gamma_V=dq,
                        delta_real_G=dg, gain_vs_real_return=dq / dg if abs(dg) > 1e-6 else None,
                        outcomes=[baseline['outcome'], alt['outcome']],
                        source=str(path), source_line=line, arrival_gain=None)
                    if baseline['outcome'] == alt['outcome'] == 'success':
                        remaining = [[s for s in b['steps'] if s['step'] >= root['step']]
                                     for b in (baseline, alt)]
                        for branch, steps in zip((baseline, alt), remaining):
                            assert steps and steps[0]['step'] == root['step']
                            actual_g = sum(GAMMA ** t * s['reward'] for t, s in enumerate(steps))
                            assert abs(actual_g - branch['root_return']) <= 1e-12
                            assert steps[-1]['reward'] == 1.0
                        lengths = [len(s) for s in remaining]
                        terminal = [GAMMA ** (n - 1) for n in lengths]
                        dt = terminal[1] - terminal[0]
                        row.update(remaining_steps=lengths,
                            terminal_only_return_difference=dt,
                            nonduration_return_difference=dg - dt,
                            arrival_gain=dq / dt if abs(dt) > 1e-6 else None,
                            root_rewards=[snap['rewards'][i], snap['rewards'][j]])
                    result.append(row)
    with gzip.open(BASE / 'resolution-pairs.jsonl.gz', 'wt') as f:
        for r in result:
            f.write(json.dumps(r, allow_nan=False) + '\n')
    summary = {}
    for model in MODELS:
        rows = [r for r in result if r['model'] == model]
        valid = [r for r in rows if r['arrival_gain'] is not None]
        vals = np.asarray([r['arrival_gain'] for r in valid])
        summary[model] = dict(action_pairs=len(rows), roots=len({r['root_id'] for r in rows}),
            episodes=len({(r['cell'], r['case']) for r in rows}),
            common_success_nonzero_duration_pairs=len(valid),
            common_success_roots=len({r['root_id'] for r in valid}),
            median_signed_arrival_gain=float(np.median(vals)) if len(vals) else None,
            median_abs_arrival_gain=float(np.median(np.abs(vals))) if len(vals) else None,
            opposite_sign_pairs=int(np.sum(vals < 0)),
            absolute_gain_below_point1_pairs=int(np.sum(np.abs(vals) < .1)),
            median_abs_remaining_steps_difference=float(np.median([
                abs(r['remaining_steps'][1] - r['remaining_steps'][0]) for r in valid])) if valid else None,
            fraction_remaining_steps_difference_above1=float(np.mean([
                abs(r['remaining_steps'][1] - r['remaining_steps'][0]) > 1 for r in valid])) if valid else None,
            median_abs_other_rewards_difference=float(np.median([
                abs(r['nonduration_return_difference']) for r in valid])) if valid else None)
    return summary, sources


def main():
    started = time.time()
    assert not (BASE / 'protocol.json').exists(), 'Do not overwrite frozen preflight'
    protocol = dict(status='FROZEN_BEFORE_COMPUTATION', training_authorized=False,
        scope='Shared value-learning decision diagnostics, not KDA-specific attribution.',
        original_T1_T4_spec='NOT LOCATED; this supplement does not amend an unidentified 34-hour plan.',
        weights=MODELS, cells=list(range(6)), cases=[90000, 90019],
        probes='Existing reference-policy shared-window states; reference original KDA top two '
               'unmasked V candidates excluding index0, selected without looking at compared seeds.',
        index0='Reselect top two from indices 1..79; retain original pair and replacement flag.',
        M='gamma*(V(a1)-V(a0)); no hard mask, risk, index0 bias, or immediate reward.',
        centered_disagreement='For each state, median_a |gamma*(V419-mean(V419)) '
                              '- gamma*(V443-mean(V443))|; then median over states.',
        pooled_disagreement='Also report point median over all state-candidate entries.',
        gate='Any primary ratio >=1 cancels sign-only timeline training on this probe population; '
             'both upper95<1 passes this budget gate only; otherwise HOLD. No automatic training.',
        caution='Seed disagreement is not repeated-estimator noise of a verified identical V^pi; '
                'even a large ratio does not prove sign disagreement, causation, or value compression.',
        bootstrap=dict(unit='cell,case episode cluster', repeats=2000, seed=20261011),
        closedloop_acceptance=dict(practical_SR_gain_pp=6.0,
            interpretation='User-imposed resource threshold, NOT a statistical MDE or seed-noise floor.',
            sample_size_and_power='UNVERIFIED until discordant paired outcomes, alpha, power, '
                                  'families, and protection/noninferiority margins are specified.',
            below_threshold='No GO; not engineering FIX merely because efficacy is unverified.'),
        resolution='Use actual branch G as primary comparator; terminal-arrival-only gamma^(L-1) '
                   'difference only for common-success, unequal-duration pairs. Other rewards '
                   'reported separately. Signed and absolute gains both retained.')
    save('protocol.json', protocol)
    sources = [P1.parent / 'measure.py', Path(__file__)]
    meta = {}
    frozen = []
    info_by_key = {}
    for cell in range(6):
        ip = P1 / 'per-checkpoint/kda-original-t24' / ('cell%d.shared.jsonl.gz' % cell)
        rp = P1 / 'per-checkpoint/kda-original-t24' / ('cell%d.steps.jsonl.gz' % cell)
        sources.extend([ip, rp])
        info = load(ip)
        refs = load(rp)
        assert len(info) == len(refs) and info
        for x, ref in zip(info, refs):
            assert key(x) == key(ref)
            values = np.asarray(x['reference_values'])
            original = np.argsort(-values, kind='stable')[:2].tolist()
            pair = (np.argsort(-values[1:], kind='stable')[:2] + 1).tolist()
            frozen.append(dict(cell=cell, case=x['case'], step=x['step'], a0=pair[0], a1=pair[1],
                               original_pair=original, reselected=(pair != original)))
            info_by_key[key(x)] = x
    with gzip.open(BASE / 'frozen-probes.jsonl.gz', 'wt') as f:
        for r in frozen:
            f.write(json.dumps(r) + '\n')
    sources.append(BASE / 'frozen-probes.jsonl.gz')
    scores = {}
    side = {}
    for model in MODELS:
        scores[model] = {}
        side[model] = {}
        for cell in range(6):
            p = P1 / 'shared' / model / ('cell%d.steps.jsonl.gz' % cell)
            d = P1 / 'decision-shared' / model / ('cell%d.steps.jsonl.gz' % cell)
            m = P1 / 'shared' / model / ('cell%d.json' % cell)
            sources.extend([p, d, m])
            item = json.loads(m.read_text())['checkpoint']
            meta[model] = item
            for row in load(p):
                v = np.asarray(row['values'], dtype=float)
                assert v.shape == (80,) and np.isfinite(v).all() and np.max(np.abs(v)) < 1e6
                assert key(row) not in scores[model]
                scores[model][key(row)] = v
            side[model].update({key(r): r for r in load(d)})
        for field in ('weight', 'config'):
            p = Path(item[field])
            assert p.exists() and sha(p) == item[field + '_sha256']
            sources.append(p)
    rows = []
    pooled = []
    assert set(scores[MODELS[0]]) == set(scores[MODELS[1]]) == set(info_by_key)
    for probe in frozen:
        k = key(probe)
        i, j = probe['a0'], probe['a1']
        v, w = [scores[m][k] for m in MODELS]
        diff = np.abs(GAMMA * ((v - v.mean()) - (w - w.mean())))
        pooled.append(diff)
        reward = info_by_key[k]['raw_rewards']
        clear = [side[m][k]['predicted_clearance'] for m in MODELS]
        assert np.array_equal(clear[0], clear[1])
        m1, m2 = GAMMA * (v[j] - v[i]), GAMMA * (w[j] - w[i])
        rows.append(dict(probe, m419=float(m1), m443=float(m2),
            centered_disagreement_state_median=float(np.median(diff)),
            pair_disagreement=float(abs(m1 - m2)),
            rewards=[reward[i], reward[j]], predicted_clearance=[clear[0][i], clear[0][j]],
            safety_eligible=[clear[0][i] >= .2, clear[0][j] >= .2],
            all_unsafe=not any(x >= .2 for x in clear[0]),
            hard_masked=[any(x >= .2 for x in clear[0]) and clear[0][idx] < .2 for idx in (i, j)],
            risk=[.8 * max(0., .2 - clear[0][i]), .8 * max(0., .2 - clear[0][j])],
            index0_penalty=[0., 0.]))
    with gzip.open(BASE / 'probe-results.jsonl.gz', 'wt') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    overall = stats(rows)
    overall['cluster_bootstrap'] = bootstrap(rows)
    overall['median_pooled_candidate_disagreement'] = float(np.median(np.concatenate(pooled)))
    overall['pooled_candidate_ratios'] = [overall['median_pooled_candidate_disagreement'] /
        overall[name] for name in ('median_abs_M419', 'median_abs_M443')]
    ratios = overall['ratios']
    gate = ('CANCEL_SIGN_ONLY_TIMELINE' if max(ratios) >= 1 else
            'PASS_BUDGET_GATE_ONLY' if all(c[1] < 1 for c in
            overall['cluster_bootstrap']['ratios95']) else 'HOLD_GATE_UNRESOLVED')
    gain, paths = resolution()
    sources.extend(paths)
    save('summary.json', dict(status='PASS', execution='CPU archived-data analysis only',
        overall=overall, by_cell={str(c): stats([r for r in rows if r['cell'] == c]) for c in range(6)},
        safe_zero_reward=stats([r for r in rows if not any(r['hard_masked']) and r['rewards'] == [0., 0.]]),
        index0_reselected_steps=sum(r['reselected'] for r in rows), gate=gate,
        resolution=gain, training_started=False, wall_seconds=time.time() - started))
    hashes = {str(p): sha(p) for p in sorted(set(sources))}
    save('frozen-manifest.json', dict(status='PASS', inputs=hashes, model_metadata=meta,
        protocol_sha256=sha(BASE / 'protocol.json')))
    print(json.dumps(dict(overall=overall, resolution=gain, gate=gate,
                         wall_seconds=time.time() - started), indent=2))


def repair_duration():
    started = time.time()
    old_manifest = json.loads((BASE / 'initial-manifest.json').read_text())
    for path, digest in old_manifest['inputs'].items():
        if path != str(Path(__file__)):
            assert sha(Path(path)) == digest, path
    summary = json.loads((BASE / 'invalid-duration-summary.json').read_text())
    summary['resolution'], _ = resolution()
    summary['duration_repair_wall_seconds'] = time.time() - started
    summary['duration_repair'] = 'Full episode prefixes excluded; root-return reward sums verified.'
    save('summary.json', summary)
    old_manifest['inputs'][str(Path(__file__))] = sha(Path(__file__))
    old_manifest['correction'] = 'Initial source hash and invalid duration outputs retained separately.'
    save('frozen-manifest.json', old_manifest)
    print(json.dumps(summary['resolution'], indent=2))


def repair_mask():
    rows = load(BASE / 'initial-probe-results.jsonl.gz')
    side = {}
    for cell in range(6):
        path = P1 / 'decision-shared/kda-s419-t24' / ('cell%d.steps.jsonl.gz' % cell)
        side.update({key(r): r for r in load(path)})
    for row in rows:
        clear = side[key(row)]['predicted_clearance']
        any_safe = any(c >= .2 for c in clear)
        i, j = row['a0'], row['a1']
        row['safety_eligible'] = [clear[idx] >= .2 for idx in (i, j)]
        row['all_unsafe'] = not any_safe
        row['hard_masked'] = [any_safe and clear[idx] < .2 for idx in (i, j)]
    with gzip.open(BASE / 'probe-results.jsonl.gz', 'wt') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    summary = json.loads((BASE / 'summary.json').read_text())
    summary['safe_zero_reward'] = stats([r for r in rows if all(r['safety_eligible']) and
                                       r['rewards'] == [0., 0.]])
    summary['all_unsafe_steps'] = sum(r['all_unsafe'] for r in rows)
    summary['mask_side_column_repair'] = 'all-unsafe fallback is not a hard mask; margins/gate unchanged'
    save('summary.json', summary)
    manifest = json.loads((BASE / 'frozen-manifest.json').read_text())
    for path, digest in manifest['inputs'].items():
        if path != str(Path(__file__)):
            assert sha(Path(path)) == digest
    manifest['inputs'][str(Path(__file__))] = sha(Path(__file__))
    save('frozen-manifest.json', manifest)
    print(json.dumps({'status': 'PASS', 'all_unsafe_steps': summary['all_unsafe_steps']}))


if __name__ == '__main__':
    if '--repair-duration' in sys.argv:
        repair_duration()
    elif '--repair-mask' in sys.argv:
        repair_mask()
    else:
        main()
