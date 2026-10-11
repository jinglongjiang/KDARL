"""Serial first-round continuations after the validated B-main sentinel."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent
C = Path('/root/ttc5-causal-20261011')


def alive(pid, start):
    p = Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ', 1)[1].split()[19] == start


def state(done, pid, start):
    if done.exists():
        return 'PASS' if json.loads(done.read_text())['status'] == 'PASS' else 'FAIL'
    return 'WAIT' if alive(pid, start) else 'FAIL'


def main():
    os.environ['PYTHONPATH'] = '/root/kda-vl-deps/flash-linear-attention:' + os.environ.get('PYTHONPATH', '')
    folder = BASE/'execution'/'remaining'
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/'journal.jsonl').exists(), 'Never overwrite a partial run'
    with tempfile.TemporaryDirectory() as tmp:
        sentinel = Path(tmp)/'done.json'
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        start = Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()[19]
        assert state(sentinel, child.pid, start) == 'WAIT'
        child.terminate(); child.wait()
        assert state(sentinel, child.pid, start) == 'FAIL'
        sentinel.write_text(json.dumps({'status': 'PASS'}))
        assert state(sentinel, child.pid, start) == 'PASS'
    (folder/'wait-unit.json').write_text(json.dumps({'live': 'WAIT', 'dead': 'FAIL', 'done': 'PASS'}))
    done = BASE/'execution/b-main/done.json'
    if not done.exists():
        targets = []
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                argv = p.read_bytes().split(b'\0')
                if b'b-main' in argv and any(x.endswith(b'/dispatch.py') or x == b'../../dispatch.py' for x in argv):
                    targets.append(int(p.parent.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        assert len(targets) == 1, targets
        pid = targets[0]
        start = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
        while state(done, pid, start) == 'WAIT':
            time.sleep(3)
        assert state(done, pid, start) == 'PASS', 'B-main failed or exited without completion sentinel'
    assert json.loads(done.read_text())['status'] == 'PASS'
    project = BASE/'a/runtime-project'
    shared = ['--data', str(C), '--inputs', str(C/'inputs'), '--project', str(project),
              '--device', 'cuda', '--main-authorized', '--gpu-authorized']
    plan = BASE/'a/artifacts/closedloop-heldout-caliper-plan.json'
    jobs = [
        [sys.executable, str(BASE/'filter-probe.py'), '--project', str(BASE/'engine'), '--causal', str(C), '--output', str(BASE/'filter-probe')],
        [sys.executable, str(BASE/'a/closedloop-a.py'), 'run', *shared, '--plan', str(plan), '--output', str(BASE/'a/artifacts/closedloop-smoke'), '--smoke-first-per-scope', '1'],
        [sys.executable, str(BASE/'a/closedloop-a.py'), 'run', *shared, '--plan', str(plan), '--output', str(BASE/'a/artifacts/closedloop-complete')],
        [sys.executable, str(BASE/'a/paired5-continuation.py'), *shared, '--output', str(BASE/'a/artifacts/paired5-complete'), '--models', 'kda-s419-t24', 'kda-s443-t24', 'gru-s419-t24', 'gru-s443-t24'],
        [sys.executable, str(BASE/'dispatch.py'), 'b-branches'],
    ]
    (folder/'jobs.json').write_text(json.dumps(jobs, indent=2))
    with (folder/'journal.jsonl').open('w') as stream:
        for i, argv in enumerate(jobs):
            assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat('2026-10-11T15:40:00+00:00'), 'INCOMPLETE: deadline'
            tick = time.time()
            with (folder/f'{i:03}.log').open('w') as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                (folder/'active-pid.json').write_text(json.dumps({'pid': child.pid, 'argv': argv}))
                code = child.wait()
            row = {'job': i, 'argv': argv, 'exit_code': code, 'wall_seconds': time.time()-tick}
            stream.write(json.dumps(row)+'\n'); stream.flush()
            print(json.dumps(row), flush=True)
            assert code == 0, f'FAIL_CLOSED: job{i}; inspect preserved log'
    (folder/'done.json').write_text(json.dumps({'status': 'PASS', 'jobs': len(jobs)}))


if __name__ == '__main__':
    main()
