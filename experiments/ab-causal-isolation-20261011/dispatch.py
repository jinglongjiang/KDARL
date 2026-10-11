"""Serial, fail-closed remote inference scheduler. No training commands."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
MODELS=['kda-s419-t24','kda-s443-t24','gru-s419-t24','gru-s443-t24']
C=Path('/root/ttc5-causal-20261011')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['b-smoke','b-main','b-branches','a-smoke','a-extract'])
    a=p.parse_args()
    project=BASE/('b' if a.stage.startswith('b') else 'a')/'runtime-project'
    jobs=[]
    if a.stage in ('b-smoke','b-main'):
        phase=a.stage.split('-')[1]
        cmd=[sys.executable,str(BASE/'b/run-b.py'),'jobs','--phase',phase,'--project',str(project),
             '--causal',str(C),'--acceptance',str(BASE/'acceptance.json')]
        raw=subprocess.check_output(cmd,text=True)
        payload=json.loads(raw)
        items=payload['jobs'] if isinstance(payload,dict) else payload
        jobs=[x['argv'] for x in items]
    elif a.stage=='b-branches':
        for model in MODELS:
            jobs.append([sys.executable,str(BASE/'b/run-b.py'),'roots','--project',str(project),'--model',model])
        for model in MODELS:
            for geometry in ('circle','square'):
                for condition in ('E1','E2','E3','E4','E5'):
                    jobs.append([sys.executable,str(BASE/'b/run-b.py'),'counterfactual','--project',str(project),
                        '--causal',str(C),'--acceptance',str(BASE/'acceptance.json'),'--model',model,
                        '--geometry',geometry,'--condition',condition])
        jobs.append([sys.executable,str(BASE/'b/run-b.py'),'summarize','--project',str(project)])
    else:
        for model in MODELS:
            jobs.append([sys.executable,str(BASE/'a/run-a.py'),'extract','--data',str(C),
                '--inputs',str(C/'inputs'),'--project',str(project),
                '--output',str(BASE/'a/artifacts'/(a.stage+'-'+model)),
                '--model',model,'--cells',*(['5'] if a.stage=='a-smoke' else ['0','3','4','5']),
                '--limit','5' if a.stage=='a-smoke' else '0','--device','cuda','--gpu-authorized'])
    folder=BASE/'execution'/a.stage
    folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'journal.jsonl').exists(), 'Already invoked; inspect artifacts, do not silently retry'
    (folder/'jobs.json').write_text(json.dumps(jobs,indent=2))
    started=time.time()
    with (folder/'journal.jsonl').open('w') as journal:
        for index,cmd in enumerate(jobs):
            # Leave ten minutes for local artifact collection and incomplete report.
            if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-10-11T15:52:00+00:00'):
                raise SystemExit('INCOMPLETE: first-round deadline reached, no new job started')
            tick=time.time()
            with (folder/f'{index:03d}.log').open('w') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                (folder/'active-pid.json').write_text(json.dumps(dict(pid=child.pid,argv=cmd)))
                code=child.wait()
            record=dict(index=index,argv=cmd,pid=child.pid,exit_code=code,
                        wall_seconds=time.time()-tick,stage_elapsed=time.time()-started)
            journal.write(json.dumps(record)+'\n');journal.flush()
            print(json.dumps(record),flush=True)
            if code:
                raise SystemExit(f'FAIL_CLOSED job {index}, exit {code}; see {folder/index.__format__("03d")}.log')
    (folder/'done.json').write_text(json.dumps(dict(status='PASS',jobs=len(jobs),wall_seconds=time.time()-started)))


if __name__=='__main__':main()
