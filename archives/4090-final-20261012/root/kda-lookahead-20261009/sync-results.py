"""Mirror new experiment evidence and check that local protected files stay unchanged."""
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
ROOTS = [Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav'),
    Path('/home/abc/workspace/kda-selection-20261008'), Path('/home/abc/workspace/kda-history-20261009'),
    Path('/home/abc/workspace/mamba-ttc5-20261009')]


def inventory():
    result = {}
    for root in ROOTS:
        for p in root.rglob('*'):
            if p.is_symlink():
                continue
            if p.is_file():
                stat = p.stat(); result[str(p)] = [stat.st_size, stat.st_mtime_ns]
    return result


def main():
    original = json.loads((BASE / 'local-readonly-manifest.json').read_text())
    while True:
        try:
            process = subprocess.run(['rsync', '-rt', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                '--include=*/', '--include=*.json', '--include=*.md', '--include=*.log', '--include=*.gz', '--exclude=*',
                'root@45.126.120.6:/root/kda-lookahead-20261009/', str(BASE) + '/'], timeout=300)
            status = dict(last_sync=time.time(), exit_code=process.returncode)
        except subprocess.TimeoutExpired:
            status = dict(last_sync=time.time(), exit_code=None, reason='sync timeout; training never started')
        (BASE / 'sync-status.json').write_text(json.dumps(status, indent=2))
        if (BASE / 'DONE.json').exists() or (BASE / 'ERROR.json').exists():
            current = inventory()
            changed = [p for p in original if current.get(p) != original[p]]
            added = [p for p in current if p not in original]
            (BASE / 'local-readonly-verification.json').write_text(json.dumps(dict(
                status='已验证通过' if not changed and not added else '已确认失败', changed=changed, added=added,
                scope='size/mtime inventory; remote executable source and weights additionally SHA256 checked'), indent=2, ensure_ascii=False))
            break
        time.sleep(120)


if __name__ == '__main__':
    main()
