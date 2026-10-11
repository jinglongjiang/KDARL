"""Independent CPU checks of arithmetic, frozen keys and interpretation controls."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
P1 = Path('/home/abc/workspace/route-b-20261010/p1-margin')


def load(path):
    return [json.loads(x) for x in gzip.open(path, 'rt')]


def key(r):
    return r['cell'], r['case'], r['step']


def main():
    manifest = json.loads((BASE / 'frozen-manifest.json').read_text())
    for name, digest in manifest['inputs'].items():
        h = hashlib.sha256()
        with Path(name).open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        assert h.hexdigest() == digest, name
    assert hashlib.sha256((BASE / 'protocol.json').read_bytes()).hexdigest() == manifest['protocol_sha256']
    expected = {}
    for model in ('kda-s419-t24', 'kda-s443-t24'):
        expected[model] = {key(r): r for cell in range(6) for r in load(
            P1 / 'shared' / model / ('cell%d.steps.jsonl.gz' % cell))}
    rows = load(BASE / 'probe-results.jsonl.gz')
    assert len(rows) == 6264 and len({(r['cell'], r['case']) for r in rows}) == 120
    for r in rows:
        i, j = r['a0'], r['a1']
        assert 0 < i < 80 and 0 < j < 80 and i != j
        for model, field in (('kda-s419-t24', 'm419'), ('kda-s443-t24', 'm443')):
            v = expected[model][key(r)]['values']
            assert r[field] == .99 * (v[j] - v[i])
        assert not any(r['hard_masked']) if r['all_unsafe'] else True
        assert all(x >= 0 for x in r['risk'])
    # A large state-centered candidate disagreement need not flip a chosen pair.
    a, b = np.array([1., .9, 0.]), np.array([1.1, 1., -10.])
    disagreement = np.median(np.abs((a - a.mean()) - (b - b.mean())))
    assert disagreement > abs(a[1] - a[0])
    assert np.sign(a[1] - a[0]) == np.sign(b[1] - b[0])
    assert np.allclose(a - a.mean(), (a + 47) - (a + 47).mean())
    gains = load(BASE / 'resolution-pairs.jsonl.gz')
    for r in gains:
        assert 0 not in (r['a0'], r['a1'])
        if r['arrival_gain'] is not None:
            t0, t1 = r['remaining_steps']
            dt = .99 ** (t1 - 1) - .99 ** (t0 - 1)
            assert dt == r['terminal_only_return_difference']
            assert r['arrival_gain'] == r['delta_gamma_V'] / dt
            assert abs(r['delta_real_G'] - dt - r['nonduration_return_difference']) < 1e-15
    result = dict(status='PASS', input_sha_end_verification=True, steps=len(rows),
        episode_clusters=120, resolution_action_pairs=len(gains),
        training_or_GPU_used=False, protocol_unchanged=True,
        checks=['unmasked margins', 'index0 exclusions', 'all-unsafe mask semantics',
                'additive-offset invariance', 'large disagreement without pair flip counterexample',
                'root-relative duration gain arithmetic'])
    (BASE / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
