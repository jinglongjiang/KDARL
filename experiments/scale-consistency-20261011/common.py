"""Shared utilities for a read-only frozen-checkpoint study."""
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
HISTORY = Path('/home/abc/4090_backup_20261009/root/kda-history-20261009/evaluate-history.py')
CELLS = [(5, 'circle_crossing'), (5, 'square_crossing'),
         (10, 'circle_crossing'), (10, 'square_crossing'),
         (20, 'circle_crossing'), (20, 'square_crossing'),
         (12, 'circle_crossing'), (12, 'square_crossing'),
         (15, 'circle_crossing'), (15, 'square_crossing')]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_protocol():
    return json.loads((BASE / 'protocol.json').read_text())


def verify():
    files = json.loads((BASE / 'frozen-manifest.json').read_text())['files']
    assert files, 'Empty frozen manifest'
    for path, expected in files.items():
        assert expected and sha(path) == expected, path
    return len(files)


def helper():
    spec = importlib.util.spec_from_file_location('scale_history_helper', HISTORY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CELLS = CELLS
    return module


def checkpoint(seed, backbone, length):
    matches = [x for x in load_protocol()['weights']
               if (x['seed'], x['backbone'], x['T']) == (seed, backbone, length)]
    assert len(matches) == 1
    return matches[0]


def compare_fields(reference, row, tolerance):
    mismatch = {key: dict(old=reference[key], new=row[key])
                for key in ('outcome', 'time', 'selection_trace_sha256')
                if reference[key] != row[key]}
    errors = {}
    for key in ('return', 'minimum_clearance'):
        delta = abs(reference[key] - row[key])
        errors[key] = delta
        if delta > tolerance:
            mismatch[key] = dict(old=reference[key], new=row[key], absolute_error=delta)
    return mismatch, errors
