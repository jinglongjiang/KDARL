"""Read-only original-data three-term action-gap decomposition; CPU stdlib only."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import time

TOL = 1e-6


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(data, output):
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        question='Do pinned physical/reward and actualTTC history corrections remove original strict action-ranking errors?',
        GO='Residual errors survive: proceed only to fixed-budget matched readout isolation, not method training.',
        KILL='No residual strict error survives: stop value/readout cause candidate for these roots.',
        budget='One streaming pass of all 16 frozen branch files, <=20 CPU minutes; no GPU.',
        controls='Unchanged roots, original weights/commands/safety/rewards/continuation; deduplicate candidate index; separate cell and failure/success.',
        decomposition='D(G-Qpred)=D(Qpinned-Qpred)+D(Qactual-Qpinned)+D(G-Qactual); D=alternative-A0.',
        limitation='Algebraic error decomposition, NOT causal attribution fractions. Pinned includes reward and terminal correction.',
        tolerance=TOL)
    (output / 'a2-protocol.json').write_text(json.dumps(protocol, indent=2))
    started = time.time()
    roots = {r['root_id']: r for r in json.loads((data / 'roots.json').read_text())['roots']}
    oracle = {r['root_id']: r for r in rows(data / 'real-history-oracle.jsonl.gz')}
    groups = {}
    sources = [data / 'roots.json', data / 'real-history-oracle.jsonl.gz']
    maximum = 0.

    def sign(v):
        return 1 if v > TOL else -1 if v < -TOL else 0

    with gzip.open(output / 'a2-pairs.jsonl.gz', 'wt') as f:
        for path in sorted(data.glob('branches-*.jsonl.gz')):
            sources.append(path)
            for line_number, row in enumerate(rows(path), 1):
                if row.get('record_type') != 'counterfactual':
                    continue
                r = row['root']; rid = r['root_id']; o = oracle[rid]
                assert r == roots[rid]
                assert (o['model'], o['cell'], o['case'], o['step']) == (r['model'], r['cell'], r['case'], r['step'])
                assert o['prefix_unchanged'] and o['pinned_oracle_bitwise_reproduced']
                bs = {b['candidate_index']: b for b in row['branches']}
                assert len(bs) == len(row['branches']), 'Unexpected duplicate candidate records'
                qa = {b['candidate_index']: b['q_oracle1_actual_history_terminal_corrected'] for b in o['branches']}
                a = next(b for b in bs.values() if 'A0' in b['labels'])
                assert abs(a['return'] - r['baseline_return']) < 1e-12
                for b in bs.values():
                    g = sum(.99 ** j * s['reward'] for j, s in enumerate(b['steps'][r['step']:]))
                    assert abs(g - b['root_return']) < 1e-12, (rid, b['candidate_index'])
                scope = 'success-control' if r['type'] == 'success-control' else 'failure'
                key = (r['model'], r['cell'], scope)
                z = groups.setdefault(key, dict(count=collections.Counter(), roots=collections.defaultdict(set), cases=collections.defaultdict(set)))

                def count(label):
                    z['count'][label] += 1
                    z['roots'][label].add(rid)
                    z['cases'][label].add(r['case'])

                count('root_records')

                def q(b):
                    s = b['first_step']
                    return [s['q_pred'], s['q_oracle1_terminal_corrected'], qa[b['candidate_index']], b['root_return']]

                for b in bs.values():
                    if b is a:
                        continue
                    count('all_pairs')
                    d = [v - u for u, v in zip(q(a), q(b))]
                    terms = [d[1] - d[0], d[2] - d[1], d[3] - d[2]]
                    error = abs((d[3] - d[0]) - sum(terms))
                    maximum = max(maximum, error)
                    assert error < 1e-12
                    truth = sign(d[3]); stages = [sign(v) for v in d[:3]]
                    flags = []
                    if truth:
                        flags.append('strict_G_pairs')
                        for name, s in zip(('pred', 'pinned', 'actual'), stages):
                            flags.append(name + ('_correct' if s == truth else '_wrong' if s == -truth else '_tie'))
                        for name, left, right in (('physical', stages[0], stages[1]), ('history', stages[1], stages[2]), ('combined', stages[0], stages[2])):
                            if left == -truth and right == truth:
                                flags.append(name + '_correction')
                            if left == truth and right == -truth:
                                flags.append(name + '_reverse_correction')
                        if stages[0] == -truth:
                            flags.append('original_strict_error')
                            if stages[2] == -truth:
                                flags.append('residual_strict_error')
                    if truth == 1:
                        flags.append('G_improved')
                        flags.append('G_improved_actual_' + ('correct' if stages[2] == 1 else 'wrong' if stages[2] == -1 else 'tie'))
                    if any(v['first_step']['terminal'] for v in (a, b)):
                        flags.append('terminal_pairs')
                    if any(v['first_step']['predicted_hard_masked'] for v in (a, b)):
                        flags.append('masked_pairs')
                    if any(v['first_step']['risk_penalty'] for v in (a, b)):
                        flags.append('risk_pairs')
                    for flag in flags:
                        count(flag)
                    evidence = dict(root_id=rid, model=r['model'], cell=r['cell'], case=r['case'], split=('train' if r['case'] % 5 < 3 else 'dev' if r['case'] % 5 == 3 else 'test'),
                        scope=scope, source=str(path), source_line=line_number, a0=a['candidate_index'], alternative=b['candidate_index'],
                        commands=[a['first_step']['command'], b['first_step']['command']],
                        q_a0=q(a), q_alt=q(b), delta_pred_pinned_actual_G=d,
                        physical_reward_terminal=terms[0], actualTTC_history=terms[1], residual_future_value=terms[2], identity_error=error, flags=flags)
                    f.write(json.dumps(evidence, allow_nan=False) + '\n')
    result = dict(protocol=protocol, wall_seconds=time.time()-started, max_identity_error=maximum,
        source_sha256={str(p): sha(p) for p in sources}, groups=[])
    for (model, cell, scope), z in sorted(groups.items()):
        result['groups'].append(dict(model=model, cell=cell, scope=scope,
            metrics={k: dict(action_pairs=v, independent_roots=len(z['roots'][k]), independent_cases=len(z['cases'][k])) for k, v in sorted(z['count'].items())}))
    residual = sum(z['count']['residual_strict_error'] for z in groups.values())
    result.update(scientific_verdict='REFUTED' if residual else 'SUPPORTED',
        proposition='Combined original pinned physical/reward + actualTTC history correction removes all original strict ranking errors in these frozen pairs.',
        research_decision='GO' if residual else 'KILL', method_level_progress=False,
        next_action='Matched frozen-latent readout diagnostics only; no IL/RL or closed-loop authorized.')
    (output / 'a2-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: result[k] for k in ('wall_seconds', 'max_identity_error', 'scientific_verdict', 'research_decision')}, indent=2))
    for g in result['groups']:
        if g['cell'] == 5:
            print(g['model'], g['scope'], json.dumps({k:v for k,v in g['metrics'].items() if k in ('G_improved','G_improved_actual_correct','G_improved_actual_wrong','physical_correction','physical_reverse_correction','history_correction','history_reverse_correction','residual_strict_error')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('/home/abc/workspace/ttc5-causal-20261011/work'))
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'artifacts')
    a = p.parse_args()
    main(a.data, a.output)
