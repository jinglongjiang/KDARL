"""Bounded, checkpoint-only D0 evaluation using the supplied evaluator."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path('/root/cc-diag-20261008')
out = root / 'd0'
out.mkdir(exist_ok=True)
started = time.time()
environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONPATH='/root/kda-vl-deps/flash-linear-attention')
(out / 'controller.pid').write_text(str(os.getpid()))
completed = []
for arm in ('kda', 'mamba', 'gru'):
    jobs = []
    for cell in range(6):
        folder = out / arm
        folder.mkdir(exist_ok=True)
        target = folder / f'cell{cell}.json'
        if target.exists():
            saved = json.loads(target.read_text())
            assert len(saved['episodes']) == 500
            continue
        if shutil.disk_usage(root).free < 2 * 1024 ** 3:
            raise RuntimeError('Disk below 2 GiB; no evaluation launch')
        stream = open(folder / f'cell{cell}.log', 'w')
        command = [sys.executable, str(root / 'eval6.py'), '--project', str(root / 'code'),
                   '--arm-dir', str(root / 'results' / arm),
                   '--config', str(root / 'results' / (arm + '.ini')),
                   '--backbone', arm, '--case-base', '89000', '--n-cases', '500',
                   '--seed', '419', '--out', str(target), '--cells', str(cell)]
        child = subprocess.Popen(command, cwd=root / 'code', env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
        jobs.append((child, stream, target))
    while jobs:
        if time.time() - started > 7 * 3600:
            for child, _, _ in jobs:
                child.terminate()
            raise TimeoutError('Preserve final hour for remaining audit/report')
        for child, stream, target in list(jobs):
            code = child.poll()
            if code is None:
                continue
            stream.close()
            jobs.remove((child, stream, target))
            if code:
                for other, _, _ in jobs:
                    other.terminate()
                raise RuntimeError(f'{target}: exit {code}')
            data = json.loads(target.read_text())
            assert len(data['episodes']) == 500 and data['checkpoint_sha256']
            completed.append(str(target))
            print('COMPLETED', target, flush=True)
        (out / 'status.json').write_text(json.dumps({
            'arm': arm, 'active_pids': [child.pid for child, _, _ in jobs],
            'completed': completed, 'elapsed_seconds': time.time() - started,
        }))
        if jobs:
            time.sleep(5)
(out / 'DONE.json').write_text(json.dumps({'wall_seconds': time.time() - started,
                                         'status': 'COMPLETED'}))
print('ALL_DONE', flush=True)
