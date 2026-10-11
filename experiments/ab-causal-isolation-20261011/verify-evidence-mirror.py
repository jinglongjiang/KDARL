"""Compare nonempty immutable result-file inventories across local and remote."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

BASE = Path(__file__).resolve().parent
SCOPES = [
    'a/artifacts/closedloop-complete',
    'a/artifacts/count-matched-complete',
    'a/artifacts/paired5-complete',
    'b/work/main', 'b/work/smoke', 'b/work/branches', 'b/n10/work',
    'reference-mamba-v3',
]


def inventory(base, scopes):
    result = {}
    for scope in scopes:
        paths = sorted(p for p in (base / scope).rglob('*') if p.is_file())
        assert paths, 'FAIL empty scope: ' + scope
        for path in paths:
            if '__pycache__' in path.parts:
                continue
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(block)
            result[str(path.relative_to(base))] = dict(
                size=path.stat().st_size, sha256=digest.hexdigest())
    assert result, 'FAIL empty inventory'
    return result


def main():
    started = time.time()
    local = inventory(BASE, SCOPES)
    source = '__file__ = "/root/ab-causal-isolation-20261011/verify-evidence-mirror.py"\n'
    source += Path(__file__).read_text().split('\ndef main():')[0]
    source += '\nprint(json.dumps(inventory(Path("/root/ab-causal-isolation-20261011"), SCOPES)))\n'
    output = subprocess.check_output([
        'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'root@45.126.120.6', 'python3 -c ' + shlex.quote(source)], text=True)
    remote = json.loads(output)
    assert remote, 'FAIL empty remote inventory'
    missing_local = sorted(set(remote) - set(local))
    missing_remote = sorted(set(local) - set(remote))
    differences = [p for p in sorted(set(local) & set(remote)) if local[p] != remote[p]]
    result = dict(status='PASS' if not (missing_local or missing_remote or differences) else 'FAIL',
        scopes=SCOPES, local=local, remote=remote, missing_local=missing_local,
        missing_remote=missing_remote, differences=differences,
        wall_seconds=time.time() - started,
        scope='Byte-level result mirror, not a certification of causal conclusions. '
              'Original inputs and sources separately checked by seal.py. '
              'Derived A feature tensors are reproducible remote intermediates, not included.')
    (BASE / 'evidence-mirror-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('status', 'missing_local', 'missing_remote', 'differences', 'wall_seconds')}))
    assert result['status'] == 'PASS'


if __name__ == '__main__':
    main()
