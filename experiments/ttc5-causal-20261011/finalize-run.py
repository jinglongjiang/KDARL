"""Run read-only attribution after all frozen continuations finish."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
manifest = json.loads((BASE / 'manifest.json').read_text())
deadline = manifest['protocol']['started_unix'] + manifest['protocol']['deadline_seconds'] - 300
scripts = ['real-history-oracle.py', 'shared-rank.py', 'input-probe.py']
hashes = {s: hashlib.sha256((BASE / s).read_bytes()).hexdigest() for s in scripts}
start_path = BASE / 'work/finalize-run.start.json'
assert not start_path.exists()
start_path.write_text(json.dumps(dict(pid=os.getpid(), started=time.time(), scripts=hashes), indent=2))
while True:
    status = json.loads((BASE / 'work/status.json').read_text())
    if status['stage'] == 'EXPERIMENTS_COMPLETE':
        break
    assert status['stage'] != 'STOPPED', status
    assert time.time() < deadline, 'Original eight-hour deadline reached'
    time.sleep(30)
for script in scripts:
    assert hashlib.sha256((BASE / script).read_bytes()).hexdigest() == hashes[script]
    started = time.time()
    log = BASE / 'work' / (script.replace('.py', '.log'))
    with log.open('w') as output:
        result = subprocess.run([sys.executable, '-u', str(BASE / script)], cwd=BASE,
                                stdout=output, stderr=subprocess.STDOUT,
                                timeout=max(1, deadline - time.time()))
    record = dict(script=script, exit_code=result.returncode, seconds=time.time()-started,
                  log=str(log), script_sha256=hashes[script])
    (BASE / 'work' / (script.replace('.py', '.exit.json'))).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    assert result.returncode == 0, record
(BASE / 'work/finalize-run.done.json').write_text(json.dumps(dict(status='PASS', ended=time.time())))
