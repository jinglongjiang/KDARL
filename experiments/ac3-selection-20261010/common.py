"""Isolated frozen-weight AC3 evaluation utilities."""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
HISTORY = Path('/home/abc/workspace/kda-history-20261009/evaluate-history.py')
RULES = ('ttc5', 'goal-cv1s', 'ac3')


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def protocol():
    return json.loads((BASE / 'protocol.json').read_text())


def verify():
    files = json.loads((BASE / 'frozen-manifest.json').read_text())['files']
    assert files, 'Empty manifest'
    for path, expected in files.items():
        assert expected and sha(path) == expected, path
    return len(files)
