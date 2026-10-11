"""Fixed Mamba TTC extension: exactly 3000 evaluation episodes, no training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parent
PROJECT = Path('/root/cc-diag-20261008/code')
ASSET = Path('/root/cc-diag-20261008/results')
RULES = ('ttc5',)
METRICS = ('success', 'collision', 'timeout')
EXPECTED = 'f23e411effac2f1959eead081fa49c908a85532642172c50a34427fe40bff2b9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return {str(p.relative_to(PROJECT)): sha(p) for p in sorted(PROJECT.rglob('*.py'))}


def command(mode, rule, cell, out):
    args = [sys.executable, str(BASE / 'evaluate-selection.py'), '--project', str(PROJECT),
            '--arm-dir', str(ASSET / 'mamba'), '--config', str(ASSET / 'mamba.ini'),
            '--backbone', 'mamba', '--human-select', rule, '--mode', mode, '--out', str(out)]
    if mode == 'evaluate':
        args += ['--cell', str(cell), '--case-base', '91000', '--n-cases', '500']
    return args


def validate(path, rule, cell):
    data = read(path)
    assert data['checkpoint_sha256'] == EXPECTED and data['backbone'] == 'mamba'
    assert data['grid_actions'] == 80 and data['selection'] == rule
    assert data['script_sha256'] == sha(BASE / 'evaluate-selection.py')
    assert data['config_sha256'] == sha(ASSET / 'mamba.ini')
    assert data['protocol'] == dict(case_base=91000, n_cases=500, cell=cell)
    assert len(data['episodes']) == 500
    assert {x['case'] for x in data['episodes']} == set(range(91000, 91500))
    assert all(x['cell'] == cell and x['outcome'] in METRICS for x in data['episodes'])
    return data['episodes']


def summary(rows):
    counts = {m: sum(x['outcome'] == m for x in rows) for m in METRICS}
    times = [x['time'] for x in rows if x['outcome'] == 'success']
    return dict(n=len(rows), counts=counts,
                rates_percent={m: 100 * v / len(rows) for m, v in counts.items()},
                mean_return=float(np.mean([x['return'] for x in rows])),
                median_minimum_clearance=float(np.median([x['minimum_clearance'] for x in rows])),
                success_time_mean=float(np.mean(times)) if times else None,
                success_time_median=float(np.median(times)) if times else None,
                success_time_p90=float(np.percentile(times, 90)) if times else None)


def aggregate():
    rows = {rule: sum([validate(BASE / 'results' / rule / ('cell%d.json' % c), rule, c)
                      for c in range(6)], []) for rule in RULES}
    assert len(rows['ttc5']) == 3000
    return dict(arm='mamba', rules={r: dict(overall=summary(rows[r]),
                cells={str(c): summary([x for x in rows[r] if x['cell'] == c]) for c in range(6)})
                for r in RULES},
                limits=['One frozen training seed', 'Extension, not a new rule-selection study',
                        'Case IDs reused across cells; cluster intervals accompany McNemar',
                        'No same-case Mamba first5 baseline in this run; no paired rule-effect claim'])


def main():
    lock = (BASE / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.time()
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    save('controller-pid.json', dict(pid=os.getpid(), proc_start_ticks=stat[19], started=start))
    assert sha(ASSET / 'mamba/rl_model_ep10000.pth') == EXPECTED
    before = sources()
    config_sha = sha(ASSET / 'mamba.ini')
    save('frozen-manifest.json', dict(source_sha256=before, config_sha256=config_sha,
         checkpoint_sha256=EXPECTED,
         script_sha256={p.name: sha(p) for p in (BASE / 'run-mamba.py', BASE / 'evaluate-selection.py')}))
    save('status.json', dict(stage='PREFLIGHT', training='NONE'))
    (BASE / 'logs').mkdir(exist_ok=True)
    active = []
    try:
        with (BASE / 'logs/preflight.log').open('w') as stream:
            subprocess.run(command('preflight', 'first5', 0, BASE / 'preflight.json'),
                           cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
        preflight = read(BASE / 'preflight.json')
        assert preflight['status'] == 'PASS' and preflight['grid_actions'] == 80
        assert preflight['checkpoint_sha256'] == EXPECTED
        for rule in RULES:
            (BASE / 'results' / rule).mkdir(parents=True, exist_ok=True)
            for cell in range(6):
                out = BASE / 'results' / rule / ('cell%d.json' % cell)
                stream = (BASE / 'logs' / ('%s-cell%d.log' % (rule, cell))).open('w')
                child = subprocess.Popen(command('evaluate', rule, cell, out), cwd=PROJECT,
                                         stdout=stream, stderr=subprocess.STDOUT)
                active.append((child, stream, cell, out))
            while active:
                for child, stream, cell, out in list(active):
                    code = child.poll()
                    if code is None:
                        continue
                    stream.close()
                    active.remove((child, stream, cell, out))
                    if code != 0:
                        raise RuntimeError('%s cell%d exited %d' % (rule, cell, code))
                    validate(out, rule, cell)
                save('status.json', dict(stage='EVALUATING', rule=rule,
                     active_cells=[c for _, _, c, _ in active], wall_seconds=time.time() - start))
                if time.time() - start > 8 * 3600:
                    raise TimeoutError('Eight-hour evaluation ceiling')
                if active:
                    time.sleep(5)
        assert sources() == before and sha(ASSET / 'mamba.ini') == config_sha
        assert sha(ASSET / 'mamba/rl_model_ep10000.pth') == EXPECTED
        save('summary.json', aggregate())
        save('DONE.json', dict(status='COMPLETED', episodes=3000, training='NONE',
                              source_unchanged=True, wall_seconds=time.time() - start))
        save('status.json', dict(stage='COMPLETED', training='NONE'))
    except BaseException as error:
        for child, stream, _, _ in active:
            if child.poll() is None:
                child.terminate()
        for child, stream, _, _ in active:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            stream.close()
        save('ERROR.json', dict(error=repr(error), wall_seconds=time.time() - start))
        save('status.json', dict(stage='FAILED', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
