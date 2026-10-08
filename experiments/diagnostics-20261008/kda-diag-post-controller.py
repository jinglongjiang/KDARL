"""Run the remaining bounded diagnostics only after D0 is complete."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/root/cc-diag-20261008')
environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONPATH='/root/kda-vl-deps/flash-linear-attention')
started = time.time()
deadline = started + 6 * 3600


def hash_files():
    paths = sorted((root / 'code').rglob('*.py')) + sorted((root / 'results').glob('*/rl_model_ep10000.pth'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_jobs(specs, limit=3):
    pending = list(specs)
    active = []
    while pending or active:
        if time.time() > deadline:
            for child, stream, _ in active:
                child.terminate()
            raise TimeoutError('Diagnostic deadline reached; no further launches')
        while pending and len(active) < limit:
            name, command = pending.pop(0)
            folder = root / 'controller-logs'
            folder.mkdir(exist_ok=True)
            stream = open(folder / (name + '.log'), 'w')
            child = subprocess.Popen(command, cwd=root / 'code', env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
            active.append((child, stream, name))
        for child, stream, name in list(active):
            code = child.poll()
            if code is None:
                continue
            stream.close()
            active.remove((child, stream, name))
            if code:
                for other, _, _ in active:
                    other.terminate()
                raise RuntimeError(f'{name}: exit {code}')
            print('DONE', name, flush=True)
        (root / 'post-status.json').write_text(json.dumps({
            'active': [{'pid': p.pid, 'name': n} for p, _, n in active],
            'remaining': [n for n, _ in pending], 'elapsed_seconds': time.time() - started,
        }))
        if active:
            time.sleep(2)


test = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.1)'])
assert test.poll() is None
assert test.wait() == 0 and test.poll() == 0
(root / 'wait-logic-test.json').write_text(json.dumps({'running_detected': True, 'completion_detected': True}))
before = hash_files()
assert all(before.values())
(root / 'immutable-before.json').write_text(json.dumps(before, indent=2))
(root / 'post-controller.pid').write_text(str(os.getpid()))
while not (root / 'd0' / 'DONE.json').exists():
    if time.time() > deadline:
        raise TimeoutError('D0 did not complete within remaining diagnostic budget')
    status = json.loads((root / 'd0' / 'status.json').read_text())
    pidfile = root / 'd0' / 'controller.pid'
    pid = int(pidfile.read_text())
    if not Path(f'/proc/{pid}').exists():
        raise RuntimeError('D0 controller ended without completion sentinel')
    time.sleep(10)

# Check the shared first32 cases against the original frozen evaluator before probing.
for arm in ('kda', 'mamba', 'gru'):
    old = json.loads((root / 'results' / arm / 'evaluation.json').read_text())
    old_table = {(r['human_num'], r['geometry'], r['case']): r for r in old['episodes']}
    for cell in range(6):
        data = json.loads((root / 'd0' / arm / f'cell{cell}.json').read_text())
        assert data['checkpoint_sha256'] == old['checkpoint_sha256']
        for r in data['episodes'][:32]:
            source = old_table[(r['human_num'], r['geometry'], r['case'])]
            assert source['outcome'] == r['outcome'] and source['time'] == r['time']
            assert abs(source['return'] - r['return']) < 1e-9
(root / 'd0' / 'overlap-parity.json').write_text(json.dumps({'episodes': 576, 'passed': True}))

def command(mode, arm, target):
    return [sys.executable, str(root / 'kda-diag-driver.py'), mode,
            '--project', str(root / 'code'), '--eval-script', str(root / 'eval6.py'),
            '--arm-dir', str(root / 'results' / arm), '--config', str(root / 'results' / (arm + '.ini')),
            '--backbone', arm, '--out', str(target)]


analysis = [sys.executable, str(root / 'kda-diag-offline.py'), '--root', str(root)]
subprocess.run(analysis, check=True, env=environment)
specs = []
for arm in ('kda', 'mamba', 'gru'):
    target = root / 'd1' / arm
    cmd = command('probe', arm, target)
    if arm == 'kda':
        cmd += ['--cross-arm-dir', str(root / 'results' / 'mamba'),
                '--cross-config', str(root / 'results' / 'mamba.ini')]
    specs.append(('D1-' + arm, cmd))
run_jobs(specs)
subprocess.run(analysis, check=True, env=environment)

# Existing D0 rows are identical evaluations of n5/n10/n20; reuse them.
folder = root / 'd4'
folder.mkdir(exist_ok=True)
for arm in ('kda', 'mamba', 'gru'):
    for n, cell in ((5, 0), (10, 2), (20, 4)):
        old = json.loads((root / 'd0' / arm / f'cell{cell}.json').read_text())
        old['episodes'] = old['episodes'][:32]
        old['reuse_source'] = f'D0/{arm}/cell{cell}.json first32'
        (folder / f'{arm}-n{n}.json').write_text(json.dumps(old))
specs = []
for n in (6, 8, 15):
    for arm in ('kda', 'mamba', 'gru'):
        target = root / 'd4' / f'{arm}-n{n}.json'
        specs.append((f'D4-{arm}-n{n}', command('evaluate', arm, target) + ['--population', str(n)]))
run_jobs(specs)
specs = []
for arm in ('kda', 'mamba', 'gru'):
    target = root / 'd6' / f'{arm}-distance.json'
    specs.append(('D6-' + arm, command('evaluate', arm, target)
                  + ['--cells', '4,5', '--selection', 'distance']))
run_jobs(specs)
specs = []
for arm in ('kda', 'mamba'):
    target = root / 'd7' / arm
    specs.append(('D7-' + arm, command('continue', arm, target)
                  + ['--selections', str(root / 'd7-selections.json'), '--probe-root', str(root / 'd1')]))
run_jobs(specs, limit=2)
subprocess.run(analysis, check=True, env=environment)
after = hash_files()
assert before == after, 'Source/checkpoint mutation detected'
(root / 'immutable-after.json').write_text(json.dumps(after, indent=2))
(root / 'POST-DONE.json').write_text(json.dumps({'status': 'COMPLETED',
                                               'wall_seconds': time.time() - started,
                                               'immutable_source_and_weights': True}))
print('ALL_DIAGNOSTICS_DONE', flush=True)
