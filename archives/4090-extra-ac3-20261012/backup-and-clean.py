"""Back up finished KDA research files; delete only hash-verified large artifacts."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


BASE = Path(__file__).resolve().parent
REPO = Path('/home/abc/workspace/KDARL')
REMOTE = 'root@45.126.120.6'
NAMES = ['ac3-selection-20261010-4090']
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', REMOTE]
COMMON = '''
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def active(names):
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name) in (os.getpid(),os.getppid()): continue
  try:
   a=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
   if any('/root/'+n+'/' in a for n in names) and any(t in a for t in ('train','evaluate','worker','run-','dispatch','collect','python')):
    result.append(dict(pid=int(p.name),command=a[:500]))
  except OSError: pass
 return result
def files(names):
 result=[]
 for name in names:
  root=Path('/root')/name
  if not root.is_dir(): continue
  for parent,dirs,names_ in os.walk(root):
   dirs[:]=sorted(d for d in dirs if d not in ('.git','__pycache__','triton-cache','inductor-cache','.venv','venv'))
   for n in sorted(names_):
    p=Path(parent)/n
    if p.is_file() and not p.is_symlink(): result.append(p)
 return sorted(result)
'''


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    (BASE / name).write_text(json.dumps(data, indent=2) + '\n')


def remote(code, payload=None):
    result = subprocess.run(SSH + ['python3 -c ' + shlex.quote(COMMON + code)],
        input=json.dumps(payload) if payload is not None else None,
        text=True, capture_output=True)
    assert result.returncode == 0, (result.returncode, result.stdout[-2000:], result.stderr[-2000:])
    return json.loads(result.stdout)


def main():
    namespace = {}
    exec(COMMON, namespace)
    probe_name = 'backup-preflight-20261012'
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)',
                              '/root/' + probe_name + '/train.py'])
    try:
        detected = namespace['active']([probe_name])
        assert any(p['pid'] == child.pid for p in detected), 'Active-process guard failed'
    finally:
        child.terminate()
        child.wait()
    assert not namespace['active']([probe_name]), 'Finished-process guard failed'
    save('guard-selftest.json', dict(status='PASS', running_detected=True,
                                    finished_not_detected=True))
    assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip() == b''
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert commit
    save('protocol.json', dict(repository_commit_completed_before_backup=commit,
        names=NAMES, delete='Only explicitly listed, locally and remotely SHA-verified weight/log/large evidence files. No directory or dependency deletion.',
        no_active_research=True, keep_code_configs_reports=True))
    code = '''
names=json.loads(sys.stdin.read())
assert names
busy=active(names); assert not busy,busy
entries=[]
for p in files(names):
 st=p.stat(); digest=sha(p); now=p.stat()
 assert (st.st_size,st.st_mtime_ns)==(now.st_size,now.st_mtime_ns),str(p)
 entries.append(dict(path=str(p),bytes=st.st_size,sha256=digest,mtime_ns=st.st_mtime_ns))
assert entries
print(json.dumps(dict(status='PASS',entries=entries,disk=shutil.disk_usage('/')._asdict())))
'''
    inventory = remote(code, NAMES)
    save('remote-manifest-before.json', inventory)
    entries = inventory['entries']
    print(json.dumps(dict(status='INVENTORIED', files=len(entries),
                          bytes=sum(x['bytes'] for x in entries))), flush=True)
    archived = json.loads((REPO / 'research-archive-manifest.json').read_text())['entries']
    source_index = {x['sha256']: Path(x['source']) for x in archived}
    missing = []
    reused = 0
    for entry in entries:
        target = BASE / entry['path'].lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = entry['sha256']
        obj = REPO / '.git/lfs/objects' / digest[:2] / digest[2:4] / digest
        source = obj if obj.is_file() else source_index.get(digest)
        if target.exists() and target.stat().st_size == entry['bytes'] and sha(target) == digest:
            reused += 1
        elif source is not None and source.is_file() and source.stat().st_size == entry['bytes'] and sha(source) == digest:
            if target.exists():
                raise RuntimeError('Existing nonmatching backup: ' + str(target))
            if obj.is_file():
                os.link(obj, target)
            else:
                shutil.copyfile(source, target)
            reused += 1
        else:
            missing.append(entry['path'].lstrip('/'))
    list_path = BASE / 'rsync-files.txt'
    list_path.write_text('\n'.join(missing) + ('\n' if missing else ''))
    required = sum(x['bytes'] for x in entries if x['path'].lstrip('/') in set(missing))
    assert shutil.disk_usage(BASE).free > required + 1024**3, ('insufficient local disk', required)
    print(json.dumps(dict(status='LOCAL_REUSE', files=reused, transfer_files=len(missing),
                          transfer_bytes=required)), flush=True)
    if missing:
        command = ['rsync', '-rt', '--partial', '--delay-updates', '--stats',
            '--files-from=' + str(list_path), '-e',
            'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4',
            REMOTE + ':/', str(BASE) + '/']
        with (BASE / 'rsync.log').open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        assert result.returncode == 0, ('rsync failed', result.returncode, str(BASE / 'rsync.log'))
    verified = []
    for entry in entries:
        target = BASE / entry['path'].lstrip('/')
        assert target.is_file() and not target.is_symlink()
        assert target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'], str(target)
        verified.append(dict(entry, local=str(target)))
    assert verified and len(verified) == len(entries)
    save('local-backup-verification.json', dict(status='PASS', entries=verified,
        immutable_object_reuse='Existing immutable Git LFS objects may share local storage; all local paths contain real bytes, not pointers.'))
    print(json.dumps(dict(status='LOCAL_SHA_VERIFIED', files=len(verified))), flush=True)
    delete = []
    for entry in entries:
        p = Path(entry['path'])
        n = p.name
        weight_log = n.endswith(('.pth', '.pth.gz', '.pt', '.ckpt', '.safetensors', '.log', '.log.gz'))
        heavy_evidence = entry['bytes'] >= 1024**2 and n.endswith(('.json', '.jsonl', '.jsonl.gz', '.json.gz', '.npz', '.csv'))
        if entry['bytes'] > 0 and (weight_log or heavy_evidence):
            delete.append(entry)
    assert delete and all(x['path'].startswith('/root/') for x in delete)
    save('deletion-plan.json', dict(names=NAMES, entries=delete))
    cleanup = '''
payload=json.loads(sys.stdin.read()); names=payload['names']; expected=payload['all']; delete=payload['delete']
assert names and expected and delete
busy=active(names); assert not busy,busy
observed=files(names)
assert [str(p) for p in observed]==[x['path'] for x in expected],'Remote file list changed; ABORT'
for x in expected:
 p=Path(x['path']); st=p.stat()
 assert st.st_size==x['bytes'] and st.st_mtime_ns==x['mtime_ns'] and sha(p)==x['sha256'],str(p)
before=shutil.disk_usage('/')._asdict()
removed=[]
for x in delete:
 p=Path(x['path'])
 assert any((Path('/root')/n) in p.parents for n in names)
 assert not p.is_symlink() and p.is_file()
 st=p.stat(); assert st.st_size==x['bytes'] and st.st_mtime_ns==x['mtime_ns']
 assert sha(p)==x['sha256'],str(p)
 p.unlink(); removed.append(x)
assert all(not Path(x['path']).exists() for x in removed)
after=shutil.disk_usage('/')._asdict()
print(json.dumps(dict(status='PASS',removed=removed,before=before,after=after,deleted_bytes=sum(x['bytes'] for x in removed))))
'''
    result = remote(cleanup, dict(names=NAMES, all=entries, delete=delete))
    save('cleanup-result.json', result)
    print(json.dumps(dict(status='BACKED_UP_AND_CLEANED', files=len(verified),
        removed_files=len(result['removed']), deleted_bytes=result['deleted_bytes'],
        free_before=result['before']['free'], free_after=result['after']['free'])), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('failure.json', dict(status='FAILED_CLOSED', error=repr(exc)))
        raise
