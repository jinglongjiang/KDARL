"""Fail-closed, serial completion of the explicitly required first-round controls."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
C = Path('/root/ttc5-causal-20261011')
OUT = BASE/'execution/supplemental'
PY = sys.executable
MODELS = ['kda-s419-t24','kda-s443-t24','gru-s419-t24','gru-s443-t24']


def main():
    os.environ['PYTHONPATH'] = '/root/kda-vl-deps/flash-linear-attention:' + os.environ.get('PYTHONPATH','')
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'journal.jsonl').exists(), 'Never overwrite evidence'
    sentinel = BASE/'execution/remaining/done.json'
    pid = 3132014
    stat = Path(f'/proc/{pid}/stat')
    start = stat.read_text().split(') ',1)[1].split()[19] if stat.exists() else None
    while not sentinel.exists():
        assert start and stat.exists() and stat.read_text().split(') ',1)[1].split()[19] == start, 'Prior queue exited without PASS'
        time.sleep(3)
    assert json.loads(sentinel.read_text())['status'] == 'PASS'
    common = ['--project',str(BASE/'b/runtime-project'),'--causal',str(C),'--acceptance',str(BASE/'acceptance.json')]
    index = 0
    def run(argv, capture=False):
        nonlocal index
        assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat('2026-10-11T15:40:00+00:00'), 'INCOMPLETE: deadline'
        tick = time.time()
        path = OUT/f'{index:03}.log'
        with path.open('w') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            (OUT/'active-pid.json').write_text(json.dumps({'pid':child.pid,'argv':argv}))
            code = child.wait()
        row = dict(index=index,argv=argv,exit_code=code,wall_seconds=time.time()-tick,log=str(path))
        with (OUT/'journal.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        index += 1
        assert code == 0, f'FAIL_CLOSED: {path}'
        return json.loads(path.read_text()) if capture else None
    for phase in ('smoke','main'):
        jobs = run([PY,str(BASE/'b/b10.py'),'jobs','--phase',phase,*common],True)['jobs']
        assert len(jobs)==12
        for argv in jobs:run(argv)
        if phase=='smoke':run([PY,str(BASE/'b/b10.py'),'smoke-gate','--project',str(BASE/'b/runtime-project'),'--causal',str(C)])
    for model in MODELS:
        run([PY,str(BASE/'b/b10.py'),'roots','--project',str(BASE/'b/runtime-project'),'--causal',str(C),'--model',model])
    for model in MODELS:
        for geometry in ('circle','square'):
            run([PY,str(BASE/'b/b10.py'),'counterfactual',*common,'--model',model,'--geometry',geometry])
    matches = run([PY,str(BASE/'b/match-b10.py')],True)
    (BASE/'b/n10/work/cross-N-matches.json').write_text(json.dumps(matches,indent=2))
    shared = ['--project',str(BASE/'a/runtime-project'),'--data',str(C),'--inputs',str(C/'inputs'),
        '--plan',str(BASE/'a/artifacts/closedloop-heldout-caliper-plan.json'),
        '--reference',str(BASE/'a/artifacts/closedloop-complete'),'--device','cuda','--main-authorized','--gpu-authorized']
    control = BASE/'a/count-matched-a.py'
    smoke = BASE/'a/artifacts/count-matched-smoke'
    formal = BASE/'a/artifacts/count-matched-complete'
    run([PY,str(control),*shared,'--output',str(smoke),'--smoke-first-per-scope','1'])
    gate = json.loads((smoke/'count-matched-summary.json').read_text())
    assert gate['status']=='PASS' and gate['roots']==8
    run([PY,str(control),*shared,'--output',str(formal)])
    gate = json.loads((formal/'count-matched-summary.json').read_text())
    assert gate['status']=='PASS' and gate['roots']==80
    run([PY,str(BASE/'reference-mamba.py'),*common])
    (OUT/'done.json').write_text(json.dumps(dict(status='PASS',jobs=index,full_training_started=False)))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'failure.json').write_text(json.dumps(dict(status='FAIL',error=repr(error))))
        raise
