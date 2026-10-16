"""Mirror only this offline experiment; compare the protected local trees on completion."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parent


def main():
    while True:
        command=['rsync','-rt','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
            '--include=*/','--include=*.json','--include=*.md','--include=*.log','--include=*.gz','--exclude=*',
            'root@REMOTE_HOST_REDACTED:/root/kda-lookahead-20261009/h3-random/',str(BASE)+'/']
        try:
            result=subprocess.run(command,timeout=600)
            (BASE/'sync-status.json').write_text(json.dumps({'exit_code':result.returncode,'time':time.time()}))
        except subprocess.TimeoutExpired:
            (BASE/'sync-status.json').write_text(json.dumps({'exit_code':None,'reason':'sync timeout'}))
        if (BASE/'DONE.json').exists() or (BASE/'ERROR.json').exists():
            protected=json.loads((BASE/'protected-local-inventory.json').read_text())
            changed=[]
            for path,expected in protected['files'].items():
                p=Path(path)
                if not p.exists() or [p.stat().st_size,p.stat().st_mtime_ns]!=expected:changed.append(path)
            current=[]
            for root in protected['roots']:
                current.extend(str(p)for p in Path(root).rglob('*')if p.is_file()and not p.is_symlink())
            added=[p for p in current if p not in protected['files']]
            (BASE/'local-readonly-verification.json').write_text(json.dumps({'status':'已验证通过'if not changed and not added else'已确认失败',
                'changed':changed,'added':added,'scope':'protected local file size/mtime; server scientific inputs and source SHA256 verified'},ensure_ascii=False,indent=2))
            break
        time.sleep(90)


if __name__=='__main__':main()
