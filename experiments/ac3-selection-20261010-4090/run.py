"""Fail-closed, budgeted controller. No background self-matching process queries."""
import sys
sys.dont_write_bytecode = True
import fcntl
import json
import os
import subprocess
import time
from common import BASE, RULES, protocol, save, verify

PY = sys.executable


def queue(jobs, deadline):
    pending = list(jobs)
    active = []
    try:
        while pending or active:
            if time.time() >= deadline:
                raise TimeoutError('Two-hour wall-clock budget reached; partial outputs are not full results')
            while pending and len(active) < 3:
                name, args = pending.pop(0)
                log = BASE / 'logs' / (name + '.log')
                log.parent.mkdir(exist_ok=True)
                stream = log.open('w')
                proc = subprocess.Popen([PY, '-B', str(BASE / 'evaluate.py')] + args,
                                        stdout=stream, stderr=subprocess.STDOUT)
                active.append((name, proc, stream, time.time(), args))
            save(BASE / 'status.json', dict(active=[dict(job=x[0], pid=x[1].pid) for x in active],
                                            pending=len(pending), remaining_seconds=deadline-time.time()))
            for task in active[:]:
                name, proc, stream, started, args = task
                code = proc.poll()
                if code is None:
                    continue
                stream.close()
                active.remove(task)
                log = BASE / 'logs' / (name + '.log')
                evidence = dict(exit_code=code, pid=proc.pid, wall_seconds=time.time()-started,
                                log=str(log), log_bytes=log.stat().st_size,
                                actual_output_tail=log.read_text(errors='replace')[-3000:], args=args)
                save(BASE / 'job-evidence' / (name + '.json'), evidence)
                print(json.dumps(dict(job=name, exit_code=code, wall_seconds=evidence['wall_seconds'])), flush=True)
                if code:
                    raise RuntimeError('Worker failed: '+name+'\n'+evidence['actual_output_tail'])
            time.sleep(1)
    finally:
        for name, proc, stream, started, args in active:
            proc.terminate()
        for name, proc, stream, started, args in active:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stream.close()
            save(BASE / 'job-evidence' / (name + '-terminated.json'),
                 dict(exit_code=proc.returncode, reason='controllerfailclosed_or_budget', args=args))


def main():
    lock = (BASE / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (BASE / 'DONE.json').exists() and not (BASE / 'ERROR.json').exists()
    verify()
    start_file = BASE / 'start.json'
    start = json.loads(start_file.read_text())['unix'] if start_file.exists() else time.time()
    deadline = start + 7200
    save(BASE / 'start.json', dict(unix=start, pid=os.getpid(), deadline=deadline))
    queue([('correctness-s%d' % seed, ['--seed', str(seed), '--correctness']) for seed in (419,443)], deadline)
    jobs = []
    for smoke in (True, False):
        jobs = []
        for seed in (419,443):
            for rule in RULES:
                for cell in (4,5):
                    name = ('smoke' if smoke else 'full')+'-s%d-%s-cell%d' % (seed,rule,cell)
                    args = ['--seed', str(seed), '--rule', rule, '--cell', str(cell),
                            '--n-cases', '3' if smoke else '500'] + (['--smoke'] if smoke else [])
                    jobs.append((name,args))
        queue(jobs, deadline)
        if smoke:
            summaries = [json.loads(p.read_text()) for p in sorted((BASE / 'smoke').glob('*/cell*.json'))]
            assert len(summaries) == 12 and all(len(x['episodes']) == 3 and x['status']=='PASS' for x in summaries)
            worker_s = sum(x['wall_seconds'] for x in summaries)
            # Project measured work under the same three-worker concurrency.
            full_seconds = worker_s / 36 * 6000 / 3
            elapsed = time.time()-start
            gate = dict(status='PASS' if full_seconds+elapsed+120 <= 7200 else 'RESOURCE_LIMIT',
                        smoke_episodes=36, smoke_worker_seconds=worker_s,
                        wall_elapsed_seconds=elapsed, projected_full_wall_seconds=full_seconds,
                        projected_total_seconds=full_seconds+elapsed+120,
                        limit_seconds=7200, workers=3,
                        outcome_metrics_not_used_for_budget_decision=True,
                        uncertainty='3episodes/cell estimate, not throughput guarantee; hard deadline enforced')
            save(BASE / 'cost-gate.json', gate)
            print(json.dumps(gate), flush=True)
            if gate['status'] != 'PASS':
                save(BASE / 'DONE.json', dict(status='RESOURCE_LIMIT', full_episodes=0,
                                             no_sample_reduction=True, train_episodes=0, frozen_files=verify()))
                return
    verify()
    result = subprocess.run([PY, '-B', str(BASE / 'summarize.py')], capture_output=True, text=True)
    save(BASE / 'job-evidence/summary.json', dict(exit_code=result.returncode,
                                               stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0, result.stderr
    save(BASE / 'DONE.json', dict(status='COMPLETED', full_episodes=6000, smoke_episodes=36,
                                 train_episodes=0, wall_seconds=time.time()-start,
                                 frozen_files_verified=verify()))
    save(BASE / 'status.json', dict(stage='COMPLETED', active=[]))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(BASE / 'ERROR.json', dict(status='FAILED_OR_BUDGET_STOP', error=repr(exc), automatic_restart=False))
        raise
