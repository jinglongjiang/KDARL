"""Mirror experiment evidence, excluding checkpoints, without controlling training."""
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
REMOTE = 'root@REMOTE_HOST_REDACTED:/root/kda-history-20261009/'
DEADLINE = time.time() + 9 * 3600

while time.time() < DEADLINE:
    result = subprocess.run([
        'rsync', '-rt', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
        '--include=*/', '--include=*.json', '--include=*.md',
        '--include=*.log', '--exclude=*', REMOTE, str(BASE) + '/',
    ], timeout=240, check=False)
    (BASE / 'sync-status.json').write_text(json.dumps({
        'last_attempt': time.time(), 'returncode': result.returncode,
        'weights_excluded': True,
    }, indent=2))
    if result.returncode == 0 and (BASE / 'DONE.json').exists():
        break
    time.sleep(180)
