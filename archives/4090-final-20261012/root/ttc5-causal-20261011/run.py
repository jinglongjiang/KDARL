"""Fail-closed, deadline-bounded controller; child PIDs are owned explicitly."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
M=json.loads((BASE/'manifest.json').read_text())
P=M['protocol']


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def save(p,d):
    p.write_text(json.dumps(d,indent=2))


def verify():
    assert M['files'] and M['project_files']
    for rel,h in M['files'].items():assert h and sha(BASE/rel)==h,rel
    for rel,h in M['project_files'].items():assert h and sha(Path(P['project'])/rel)==h,rel
    assert os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize>=4*1024**3, 'Disk gate <4GiB'


def jobs(commands,concurrency=4):
    running=[]
    pending=list(commands)
    all_runs=[]
    try:
        while pending or running:
            while pending and len(running)<concurrency:
                name,args=pending.pop(0)
                logfile=BASE/'work'/(name+'.log')
                f=logfile.open('w')
                proc=subprocess.Popen([sys.executable,'-u',str(BASE/'audit.py')]+args,
                    stdout=f,stderr=subprocess.STDOUT,cwd=P['project'],start_new_session=True)
                running.append((name,proc,f,time.time()))
                save(BASE/'work'/(name+'.pid.json'),dict(pid=proc.pid,args=args,started=time.time()))
            for entry in running[:]:
                name,proc,f,started=entry
                code=proc.poll()
                if code is not None:
                    f.close();running.remove(entry)
                    result=dict(name=name,exit_code=code,seconds=time.time()-started,
                                output=str(BASE/'work'/(name+'.log')))
                    save(BASE/'work'/(name+'.exit.json'),result)
                    all_runs.append(result)
                    print(json.dumps(result),flush=True)
                    assert code==0, 'Child failed: '+name
            if time.time()>P['started_unix']+P['deadline_seconds']-600:
                raise TimeoutError('8hour budget reserved report time reached')
            if running:time.sleep(5)
        return all_runs
    finally:
        for name,proc,f,_ in running:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            f.close()


def main():
    (BASE/'work').mkdir(exist_ok=True)
    verify()
    names=[m['id'] for m in P['models']]
    started=time.time()
    save(BASE/'work/controller.pid.json',dict(pid=os.getpid(),started=started))
    try:
        jobs([('preflight-'+m,['preflight','--model',m]) for m in names])
        save(BASE/'work/status.json',dict(stage='PREFLIGHT_PASS',time=time.time()))
        # Complete denominators and freeze ALL roots before any branch outcome exists.
        for cell in [5,4,0,3]:
            n=500 if cell in (4,5) else 32
            jobs([('baseline-%s-cell%d'%(m,cell),['baseline','--model',m,'--cell',str(cell),'--n',str(n)]) for m in names])
        jobs([('freeze-roots',['roots'])],1)
        save(BASE/'work/status.json',dict(stage='ROOTS_FROZEN',time=time.time()))
        for names_now,cell in [(names[:2],5),(names[2:],5),(names,4),(names,0),(names,3)]:
            jobs([('counterfactual-%s-cell%d'%(m,cell),['counterfactual','--model',m,'--cell',str(cell)]) for m in names_now])
        save(BASE/'work/status.json',dict(stage='EXPERIMENTS_COMPLETE',time=time.time()))
    except BaseException as e:
        save(BASE/'work/status.json',dict(stage='STOPPED',error=repr(e),time=time.time()))
        raise
    finally:
        save(BASE/'work/end-verification.json',{
            'files':{rel:sha(BASE/rel)==h for rel,h in M['files'].items()},
            'project_files':{rel:sha(Path(P['project'])/rel)==h for rel,h in M['project_files'].items()},
            'wall_seconds':time.time()-started})


if __name__=='__main__':main()
