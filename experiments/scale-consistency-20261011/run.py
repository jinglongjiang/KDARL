"""Fail-closed parity, smoke and full-evaluation controller."""
import sys
sys.dont_write_bytecode = True
import fcntl
import json
import os
import subprocess
import time

from common import BASE, CELLS, load_protocol, save, verify


def queue(jobs, stage):
    pending, active = list(jobs), []
    try:
        while pending or active:
            while pending and len(active) < 3:
                name, args = pending.pop(0)
                log = BASE / 'logs' / (name + '.log')
                log.parent.mkdir(exist_ok=True)
                stream = log.open('w')
                proc = subprocess.Popen([sys.executable, '-B', str(BASE / 'evaluate.py')] + args,
                                        stdout=stream, stderr=subprocess.STDOUT)
                active.append((name, proc, stream, time.time(), args))
            save(BASE / 'status.json', dict(stage=stage, active=[dict(job=x[0], pid=x[1].pid) for x in active],
                                           pending=len(pending), training_episodes=0))
            for task in active[:]:
                name, proc, stream, started, args = task
                code = proc.poll()
                if code is None:
                    continue
                stream.close()
                active.remove(task)
                log = BASE / 'logs' / (name + '.log')
                evidence = dict(exit_code=code, wall_seconds=time.time() - started,
                                log=str(log), bytes=log.stat().st_size, args=args,
                                actual_output_tail=log.read_text(errors='replace')[-5000:])
                save(BASE / 'job-evidence' / (name + '.json'), evidence)
                print(json.dumps(dict(job=name, exit_code=code, wall_seconds=evidence['wall_seconds'])), flush=True)
                if code:
                    raise RuntimeError('Failed worker: ' + name + '\n' + evidence['actual_output_tail'])
            time.sleep(1)
    finally:
        for _, proc, _, _, _ in active:
            proc.terminate()
        for name, proc, stream, _, args in active:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stream.close()
            save(BASE / 'job-evidence' / (name + '-terminated.json'),
                 dict(exit_code=proc.returncode, reason='fail_closed', args=args))


def make_jobs(stage):
    p = load_protocol()
    jobs = []
    items = [x for x in p['weights'] if x['seed'] == 443 and x['T'] == 24] if stage == 'parity' else p['weights']
    for item in items:
        for cell in ([0, 5] if stage == 'parity' else [6, 7, 8, 9]):
            seed, model, length = item['seed'], item['backbone'], item['T']
            name = '%s-s%d-%s-t%d-cell%d' % (stage, seed, model, length, cell)
            folder = 'parity' if stage == 'parity' else 'smoke' if stage == 'smoke' else 'evaluation'
            n, geom = CELLS[cell]
            out = BASE / folder / ('%d-person' % n) / geom / ('s%d-%s-t%d.json' % (seed, model, length))
            args = ['--seed', str(seed), '--backbone', model, '--length', str(length),
                    '--cell', str(cell), '--stage', stage, '--n-cases', '3' if stage == 'smoke' else '500',
                    '--out', str(out)]
            jobs.append((name, args))
    return jobs


def main():
    lock = (BASE / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (BASE / 'ERROR.json').exists() and not (BASE / 'DONE.json').exists()
    count = verify()
    started = time.time()
    save(BASE / 'start.json', dict(pid=os.getpid(), unix=started, frozen_files=count))
    for stage in ('parity', 'smoke', 'full'):
        queue(make_jobs(stage), stage)
        verify()
        if stage == 'parity':
            files = sorted((BASE / 'parity').rglob('s*.json'))
            assert len(files) == 4
            data = [json.loads(path.read_text()) for path in files]
            assert all(x['status'] == 'PASS' and len(x['episodes']) == 500 for x in data)
            save(BASE / 'parity-summary.json', dict(status='PASS', episodes=2000,
                 interpretation='Saved-field equivalence, not historical action-trajectory reproduction',
                 sources=[str(path) for path in files],
                 environments=[x['environment'] for x in data],
                 maximum_errors=[x['max_parity_absolute_error'] for x in data]))
        if stage == 'smoke':
            files = sorted((BASE / 'smoke').rglob('s*.json'))
            assert len(files) == 32
            data = [json.loads(path.read_text()) for path in files]
            assert all(x['status'] == 'PASS' and len(x['episodes']) == 3 for x in data)
            worker_seconds = sum(x['wall_seconds'] for x in data)
            save(BASE / 'smoke-summary.json', dict(status='PASS', episodes=96,
                 seconds_per_episode_worker=worker_seconds / 96,
                 projected_full_seconds_at_three_workers=worker_seconds / 96 * 16000 / 3,
                 uncertainty='Three cases/cell estimate, not guaranteed completion time'))
    result = subprocess.run([sys.executable, '-B', str(BASE / 'analyze.py')], capture_output=True, text=True)
    save(BASE / 'job-evidence/analysis.json', dict(exit_code=result.returncode,
                                                 stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0, result.stderr
    save(BASE / 'DONE.json', dict(status='COMPLETED', full_episodes=16000,
                                 parity_episodes=2000, smoke_episodes=96, training_episodes=0,
                                 wall_seconds=time.time() - started, frozen_verified=verify()))
    save(BASE / 'status.json', dict(stage='COMPLETED', active=[], training_episodes=0))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(BASE / 'ERROR.json', dict(status='FAILED_CLOSED', error=repr(exc), automatic_restart=False))
        save(BASE / 'status.json', dict(stage='FAILED_CLOSED', active=[], new_full_started=(BASE / 'evaluation').exists()))
        raise
