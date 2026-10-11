"""Fail-closed local inference controller; no training or remote execution."""
import sys
sys.dont_write_bytecode=True
import fcntl
import json
from pathlib import Path
import subprocess
import time
from common import BASE, specs, save, verify

PY='/home/abc/temp/kda-vl-env/bin/python'


def queue(jobs,stage):
    pending=list(jobs);active=[]
    while pending or active:
        while pending and len(active)<3:
            args=pending.pop(0);name=stage+'-'+args[args.index('--id')+1]+'-cell'+args[args.index('--cell')+1]
            log=BASE/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
            stream=log.open('w')
            process=subprocess.Popen([PY,'-B',str(BASE/'measure.py')]+args,stdout=stream,stderr=subprocess.STDOUT)
            active.append((name,process,stream,time.time(),args))
            save(BASE/'status.json',dict(stage=stage,active=[dict(name=x[0],pid=x[1].pid) for x in active],pending=len(pending)))
        for task in active[:]:
            name,process,stream,started,args=task
            code=process.poll()
            if code is None:continue
            stream.close();active.remove(task)
            tail=(BASE/'logs'/(name+'.log')).read_text(errors='replace')[-4000:]
            save(BASE/'job-evidence'/(name+'.json'),dict(name=name,pid=process.pid,exit_code=code,
                wall_seconds=time.time()-started,args=args,actual_output_tail=tail))
            if code:
                for _,child,fh,_,_ in active:
                    child.terminate()
                for _,child,fh,_,_ in active:
                    child.wait();fh.close()
                raise RuntimeError('Fail closed: '+name+' exit '+str(code)+'\n'+tail)
        time.sleep(1)


def main():
    lock=(BASE/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (BASE/'DONE.json').exists() and not (BASE/'ERROR.json').exists()
    started=time.time();verify();items=specs()
    reference=next(x for x in items if x['id']=='kda-original-t24')
    ordered=[reference]+[x for x in items if x is not reference]
    for smoke in (True,False):
        n=3 if smoke else 20
        own=[]
        for item in ordered:
            for cell in range(6):
                own.append(['--id',item['id'],'--cell',str(cell),'--n-cases',str(n)]+(['--smoke'] if smoke else []))
        queue(own,'smoke-own' if smoke else 'full-own')
        shared=[]
        for item in ordered:
            for cell in range(6):
                shared.append(['--id',item['id'],'--cell',str(cell),'--shared']+(['--smoke'] if smoke else []))
        queue(shared,'smoke-shared' if smoke else 'full-shared')
        if smoke:
            summaries=[json.loads(p.read_text()) for p in (BASE/'p1-margin/smoke').glob('*/cell*.json')]
            assert len(summaries)==66 and all(x['status']=='已验证通过' and len(x['episodes'])==3 for x in summaries)
            own_seconds=sum(x['wall_seconds'] for x in summaries)
            own_steps=sum(x['steps'] for x in summaries)
            save(BASE/'smoke-validation.json',dict(status='已验证通过',checkpoints=11,cells=6,
                episodes=198,steps=own_steps,mean_worker_seconds_per_step=own_seconds/own_steps,
                full_cases_per_cell=20,not_expanding_to50=True,
                note='Fixed20/cell; timing is measured under3 worker concurrency, not an isolated GPU benchmark'))
    result=subprocess.run([PY,'-B',str(BASE/'summarize-p1.py')],capture_output=True,text=True)
    save(BASE/'job-evidence/summary.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,result.stderr
    checked=verify()
    save(BASE/'DONE.json',dict(status='COMPLETED',train_episodes=0,own_episodes=1320,
        smoke_episodes=198,checkpoints=11,wall_seconds=time.time()-started,frozen_files_verified=checked))
    save(BASE/'status.json',dict(stage='COMPLETED',active=[],train_episodes=0))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(BASE/'ERROR.json',dict(status='已确认失败',error=repr(exc),automatic_restart=False));raise
