"""Mirror metadata; on completion verify every file before weight-only cleanup."""
import sys
sys.dont_write_bytecode = True
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

BASE=Path(__file__).resolve().parent
HOST='root@45.126.120.6'
REMOTE='/root/kda-history-replication-20261010'


def save(name,data):
    p=BASE/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def inventory(root):
    assert root.is_dir()
    out={}
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(),('Symlink not allowed in archive inventory',str(p))
        if p.is_file():
            p.resolve().relative_to(root.resolve())
            out[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=digest(p))
    assert out,'FAIL: empty file inventory'
    return out


def weights(items):
    return {p:v for p,v in items.items() if p.endswith(('.pth','.pth.gz'))}


def compare(expected,actual,require_weights=False):
    assert expected and actual,'FAIL: either inventory is empty'
    assert all(v.get('sha256') for v in expected.values()) and all(v.get('sha256') for v in actual.values())
    a=[json.dumps([p,v],sort_keys=True) for p,v in sorted(expected.items())]
    b=[json.dumps([p,v],sort_keys=True) for p,v in sorted(actual.items())]
    delta=list(difflib.unified_diff(a,b,fromfile='verified',tofile='current',lineterm=''))
    assert not delta,'ABORT: manifest diff\n'+'\n'.join(delta[:30])
    if require_weights: assert weights(expected) and weights(actual),'FAIL: either weight inventory is empty'


def alive(pid):
    p=Path('/proc/%d/stat'%pid)
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z'


def remote_state():
    done=BASE/'DONE.json';err=BASE/'ERROR.json';pid=BASE/'controller-pid.json'
    return dict(done=json.loads(done.read_text()) if done.exists() else None,
                error=json.loads(err.read_text()) if err.exists() else None,
                controller_alive=alive(json.loads(pid.read_text())['pid']) if pid.exists() else False)


def remote_delete(expected_path):
    assert str(BASE)==REMOTE and expected_path.resolve()==BASE/'archive-expected-weights.json'
    state=remote_state()
    assert state['done'] and state['done']['status']=='COMPLETED' and not state['error'] and not state['controller_alive']
    expected=json.loads(expected_path.read_text());assert expected
    before=inventory(BASE);current=weights(before)
    compare(expected,current,require_weights=True)
    finals={'runs/%s-t%d/rl_model_ep10000.pth'%(arm,t) for arm in ('kda','gru') for t in (12,24)}
    assert finals.issubset(expected),'FAIL: incomplete final checkpoint list'
    # Re-enumeration, hashing, and a line-by-line diff immediately precede unlink.
    compare(expected,weights(inventory(BASE)),require_weights=True)
    for name in expected:
        p=BASE/name;p.resolve().relative_to(BASE.resolve())
        assert p.is_file() and not p.is_symlink() and name.endswith(('.pth','.pth.gz'))
        p.unlink()
    after=inventory(BASE)
    assert not weights(after),'Weight cleanup incomplete'
    for name,value in before.items():
        if name not in expected:assert after[name]==value,('Non-weight file changed',name)
    receipt=dict(status='已验证通过',deleted_files=sorted(expected),deleted_bytes=sum(v['bytes'] for v in expected.values()),
                 double_sided_sha_verified_before_delete=True,relisted_line_diff_passed=True,non_weights_unchanged=True,
                 directories_deleted=False,time=time.time())
    save('remote-cleanup.json',receipt);print(json.dumps(receipt,ensure_ascii=False))


def ssh(command,timeout=180):
    error=None
    for attempt in range(5):
        try:
            return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,command],text=True,timeout=timeout)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            error=exc
            if attempt<4:time.sleep(10)
    raise error


def remote_cli(flag):
    return 'python3 -B '+shlex.quote(REMOTE+'/sync-and-archive.py')+' '+flag


def rsync(full=False):
    cmd=['rsync','-rt','-e','ssh -o BatchMode=yes -o ConnectTimeout=15']
    if not full:cmd+=['--include=*/','--include=*.json','--include=*.md','--include=*.log','--include=*.ini','--exclude=*']
    cmd += [HOST+':'+REMOTE+'/',str(BASE)+'/']
    for attempt in range(5):
        result=subprocess.run(cmd,timeout=3600 if full else 300)
        save('sync-status.json',dict(exit_code=result.returncode,time=time.time(),full=full,attempt=attempt+1))
        if result.returncode==0:return
        if attempt<4:time.sleep(10)
    raise RuntimeError('rsync failed; never delete remote weights')


def selftest():
    sample={'runs/a.pth':dict(bytes=3,sha256='notempty')}
    tests=[]
    for a,b,w in (({},sample,False),(sample,{},False),({'a.log':sample['runs/a.pth']},{'a.log':sample['runs/a.pth']},True),
                  (sample,{'runs/a.pth':dict(bytes=4,sha256='changed')},True)):
        try:compare(a,b,w)
        except AssertionError:tests.append(True)
        else:tests.append(False)
    assert all(tests);compare(sample,sample,True)
    child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.2)']);assert alive(child.pid);child.wait();assert not alive(child.pid)
    save('archive-selftest.json',dict(status='已验证通过',empty_left_rejected=True,empty_right_rejected=True,
        empty_weights_rejected=True,changed_relisting_aborts=True,equal_nonempty_passes=True,pid_live_dead_test_passed=True,
        remote_deletion_executed=False));print('archive safety selftest PASS')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--remote-state',action='store_true');parser.add_argument('--remote-inventory',action='store_true')
    parser.add_argument('--remote-delete',type=Path);parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if args.remote_state:print(json.dumps(remote_state()));return
    if args.remote_inventory:print(json.dumps(inventory(BASE)));return
    if args.remote_delete:remote_delete(args.remote_delete);return
    if args.selftest:selftest();return
    assert str(BASE).startswith('/home/abc/'),'Archive mirror must run on the local machine'
    while True:
        rsync();state=json.loads(ssh(remote_cli('--remote-state')))
        terminal=state['done'] or state['error']
        if terminal and not state['controller_alive']:break
        time.sleep(120)
    remote_files=json.loads(ssh(remote_cli('--remote-inventory'),timeout=600))
    assert remote_files,'FAIL: empty remote archive'
    assert shutil.disk_usage(BASE).free>=sum(v['bytes'] for v in remote_files.values())+1024**3,'Local disk too small; remote preserved'
    rsync(full=True)
    local_files={name:dict(bytes=(BASE/name).stat().st_size,sha256=digest(BASE/name)) for name in remote_files if (BASE/name).is_file()}
    compare(remote_files,local_files)
    # No training process is alive; the complete directory must remain unchanged through copy.
    compare(remote_files,json.loads(ssh(remote_cli('--remote-inventory'),timeout=600)))
    save('archive-verification.json',dict(status='已验证通过',n_files=len(remote_files),n_weights=len(weights(remote_files)),
        bytes=sum(v['bytes'] for v in remote_files.values()),files=remote_files,all_files_both_side_sha256_equal=True,
        source=REMOTE,time=time.time(),scientific_state=terminal))
    if state['error'] or state['done']['status']!='COMPLETED':
        save('archive-finished.json',dict(status='BACKUP_ONLY',remote_weights_deleted=False,reason='failure/partial; keep recovery assets'));return
    w=weights(remote_files);assert w,'FAIL: empty verified weights'
    save('archive-expected-weights.json',w)
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',str(BASE/'archive-expected-weights.json'),HOST+':'+REMOTE+'/archive-expected-weights.json'],check=True,timeout=120)
    receipt=json.loads(ssh(remote_cli('--remote-delete '+shlex.quote(REMOTE+'/archive-expected-weights.json')),timeout=600))
    save('remote-cleanup.json',receipt)
    assert receipt['status']=='已验证通过' and set(receipt['deleted_files'])==set(w)
    assert all((BASE/name).is_file() and digest(BASE/name)==value['sha256'] for name,value in w.items())
    save('archive-finished.json',dict(status='已验证通过',remote_weights_deleted=True,local_weights_preserved=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if str(BASE).startswith('/home/abc/'):
            save('archive-error.json',dict(error=repr(error),remote_cleanup_not_assumed=True))
        raise
