"""Join archived common prefixes to A continuations without new rollouts."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OLD = Path('/home/abc/workspace/ttc5-causal-20261011/work')


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def clearance(current, following, actual):
    robot = np.asarray(current['robot'], dtype=np.float64)
    humans = np.asarray(current['humans'], dtype=np.float64)
    future = np.asarray(following['humans'], dtype=np.float64)
    human_velocity = (future[:, :2] - humans[:, :2]) / .25 if actual else humans[:, 2:4]
    d0 = robot[:2] - humans[:, :2]
    dv = np.asarray(current['command']) - human_velocity
    norm = np.sum(dv * dv, axis=1)
    t = np.clip(-np.sum(d0 * dv, axis=1) / np.maximum(norm, 1e-30), 0., .25)
    t[norm < 1e-20] = 0.
    return float(np.min(np.sqrt(np.sum((d0 + t[:, None] * dv) ** 2, axis=1)) - robot[4] - humans[:, 4]))


def main():
    raw = BASE / 'a/artifacts/closedloop-complete/closedloop-rows.jsonl.gz'
    continuations = list(rows(raw))
    assert len(continuations) == 160, 'Scientific rows exclude eight duplicate acceptance rollouts'
    needed = {(r['model'], r['case']) for r in continuations}
    originals = {}
    sources = {str(raw): sha(raw)}
    for model in sorted({r['model'] for r in continuations}):
        path = OLD / ('baseline-' + model + '-cell5.jsonl.gz')
        sources[str(path)] = sha(path)
        for episode in rows(path):
            key = (model, episode['case'])
            if key in needed:
                originals[key] = episode
    assert set(originals) == needed
    output = []
    max_error = 0.
    unique_prefixes = set()
    for continuation in continuations:
        episode = originals[continuation['model'], continuation['case']]
        k0 = continuation['step']
        env_clearances = []
        actual_clearances = []
        for k in range(k0):
            current, following = episode['steps'][k:k + 2]
            assert current['step'] == k and following['step'] == k + 1
            for actual, target in ((False, env_clearances), (True, actual_clearances)):
                value = clearance(current, following, actual)
                saved = current['actual_swept_clearance' if actual else 'env_swept_clearance']
                error = abs(value - saved)
                max_error = max(max_error, error)
                assert error < 1e-12
                target.append(value)
        assert continuation['steps'][0]['step'] == episode['steps'][k0]['step']
        prefix_actual_min = min(actual_clearances, default=None)
        prefix_env_min = min(env_clearances, default=None)
        combined_actual_min = min(actual_clearances + [s['actual_swept_clearance'] for s in continuation['steps']])
        combined_env_min = min(env_clearances + [s['env_swept_clearance'] for s in continuation['steps']])
        output.append(dict(model=continuation['model'], root_id=continuation['root_id'],
            case=continuation['case'], step=k0, arm=continuation['arm'], outcome=continuation['outcome'],
            prefix_actual_min=prefix_actual_min, prefix_env_min=prefix_env_min,
            complete_episode_actual_min=combined_actual_min, complete_episode_env_min=combined_env_min))
        unique_prefixes.add((continuation['model'], continuation['root_id']))
    result = dict(status='PASS', scope='Archived shared prefix plus already executed continuation; no model or rollout.',
        decision_contract=dict(GO='Complete archived prefixes and numerical reconstruction permit full-episode safety qualification.',
            KILL='Missing or inconsistent records invalidate full-episode safety claims; any negative prefix must be explicitly retained, not dropped.'),
        roots=len(unique_prefixes), continuation_rows=len(output), source_sha256=sources,
        max_prefix_reconstruction_error=max_error,
        roots_with_negative_actual_prefix=len({(r['model'], r['root_id']) for r in output if r['prefix_actual_min'] is not None and r['prefix_actual_min'] < 0.}),
        success_rows_with_actual_overlap=sum(r['outcome'] == 'success' and r['complete_episode_actual_min'] < 0. for r in output),
        records=output)
    (BASE / 'a/artifacts/verified-prefix-safety.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_sha256', 'decision_contract')}))


if __name__ == '__main__':
    main()
