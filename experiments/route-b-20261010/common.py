"""Read-only shared utilities for the frozen-checkpoint evidence audit."""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
WORK = Path('/home/abc/workspace')
CELLS = [(5, 'circle_crossing'), (5, 'square_crossing'),
         (10, 'circle_crossing'), (10, 'square_crossing'),
         (20, 'circle_crossing'), (20, 'square_crossing')]


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def specs():
    return json.loads((BASE / 'protocol.json').read_text())['checkpoints']


def verify():
    frozen = json.loads((BASE / 'frozen-manifest.json').read_text())
    assert frozen['files']
    for name, expected in frozen['files'].items():
        assert sha(name) == expected, ('Frozen file changed', name)
    return len(frozen['files'])


def harness():
    return module('route_b_history_reference', WORK / 'kda-history-20261009/evaluate-history.py')
