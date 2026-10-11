"""Fail-closed 42-episode implementation check, then the remaining 258 offline episodes."""
import sys
sys.dont_write_bytecode = True
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent
WEIGHT = PARENT / 'weights/kda/rl_model_ep10000.pth'
EXPECTED = 'cb2e20c4cc2f12014fed589e9ff02468d04cfa12573c66fc79617c95356cbf0a'
ACTIVE = []


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)


def alive(pid):
    p=Path('/proc/%d/stat'%pid)
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z'


def old_freeze():
    spec=importlib.util.spec_from_file_location('previous_controller',PARENT/'run-diagnostics.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.verify_freeze()


def freeze():
    old_freeze(); assert sha(WEIGHT)==EXPECTED
    benchmark=read(BASE/'benchmark/cell0.json')
    assert benchmark['status']=='已验证通过' and benchmark['checked_steps']==12
    seconds=benchmark['steady_seconds_per_step']
    # Select effort by measured cost alone, never by effect estimates.
    estimate10=seconds*(22./12.)*16020
    repeats=10 if estimate10 <= 5400 else 5
    protocol=dict(version='h3-random-matched-v1-20261009',weight_sha256=EXPECTED,
        wrong_history_weight_sha256='24573aefc0cf9d67e32805980acbcf8d8bc648588b5ba0a3371fb466ef29be5e',
        train_episodes=0,new_environment_episodes=0,n_episodes=300,n_steps=16020,
        config=str(PARENT/'original-configs/kda.ini'),seq_len=24,alpha=.3,selection='ttc5',actions=80,
        random_repeats=repeats,seed_rule='SHA256(cell/case/step/variant/repetition/20261009)',
        effort_rule='10 repeats if measured projected serial time <=5400 seconds, otherwise5; no outcome-based selection',
        benchmark=dict(records=12,seconds_per_step_with5=seconds,projected_serial_seconds_with10=estimate10),
        implementation_cases='first7 cases in each6 cell,42 complete episodes; reused in full300',
        main='one shared angle for all humans per step/repetition; same rotation for dp and dv',
        secondary='independent human angles, same angle for each humans dp and dv',
        physics=dict(environment='old human velocity relative segment; first negative short-circuit recorded separately',
            actual='segment from current human position to true next position; same-time relative minimum',
            counterfactual_human_trajectory_assumption='robot.visible=false, current ORCA human response independent of robot command'),
        inference=dict(unit='episode=(cell,case)',bootstrap=10000,paired=True,step_weighted_rates=True,
            random_repeats_averaged_inside_step=True,primary_family=['A_final_flip','B_env_clearance','B_actual_clearance','B_progress'],
            simultaneous_ci_percent=98.75,secondary_ci_percent=95),
        equivalence_margin_flip_fraction=.01,
        B_gate='both clearance definitions and progress Oracle-minus-shared Random simultaneous intervals >0, and observed collision fractions do not increase; one-step only',
        negative_scope='frozen KDA+0.25seconds+alpha0.3 only; not reward/multistep/other KDA; no closed-loop advantage claim')
    assert not (BASE/'protocol.json').exists();save(BASE/'protocol.json',protocol)
    inputs=[PARENT/'h3/per-case/kda-alpha0.3-a80'/('cell%d.steps.jsonl.gz'%c)for c in range(6)]
    manifest=dict(weight={str(WEIGHT):sha(WEIGHT)},config={str(PARENT/'original-configs/kda.ini'):sha(PARENT/'original-configs/kda.ini')},
        inputs={str(p):sha(p)for p in inputs},scripts={str(p):sha(p)for p in BASE.glob('*.py')},
        parent_manifest={str(PARENT/'frozen-manifest.json'):sha(PARENT/'frozen-manifest.json')},
        protocol_sha256=sha(BASE/'protocol.json'),protected_local_inventory_sha256=sha(BASE/'protected-local-inventory.json'))
    save(BASE/'frozen-manifest.json',manifest)
    count=0;ids=set()
    for c,p in enumerate(inputs):
        with gzip.open(p,'rt')as f:
            for line in f:
                x=json.loads(line);assert x['cell']==c and 90000<=x['case']<=90049
                count+=1;ids.add((c,x['case']))
    assert count==16020 and len(ids)==300
    save(BASE/'source-audit.json',dict(status='已验证通过',episodes=len(ids),steps=count,
        physical_models_match=False,source='crowd_sim/envs/crowd_sim.py:348-380,451-460',
        environment_collision_velocity='human.vx/vy BEFORE current-step ORCA action is applied',
        actual_update_velocity='new human_actions from human.act; recorded true_next_humans.vx/vy',
        margin='same-time swept relative segments; NOT unrelated geometric segment intersections'))
    return protocol


def verify():
    old_freeze();m=read(BASE/'frozen-manifest.json')
    assert sha(BASE/'protocol.json')==m['protocol_sha256']
    for group in ('weight','config','inputs','scripts','parent_manifest'):
        for path,digest in m[group].items():assert digest and sha(path)==digest,(group,path)
    assert sha(BASE/'protected-local-inventory.json')==m['protected_local_inventory_sha256']


def jobs(stage,offset,n,repeats):
    queue=[dict(cell=c,output=BASE/stage/('cell%d.json'%c))for c in range(6)]
    finished=[];cores=sorted(os.sched_getaffinity(0));start=time.time()
    while queue or ACTIVE:
        while queue and len(ACTIVE)<min(6,len(cores)):
            job=queue.pop(0);assert not job['output'].exists()
            slot=next(i for i in range(min(6,len(cores)))if i not in [j['slot']for j in ACTIVE])
            path=BASE/'logs'/('%s-cell%d.log'%(stage,job['cell']));path.parent.mkdir(parents=True,exist_ok=True)
            stream=path.open('w');env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            command=[sys.executable,'-B',str(BASE/'random-worker.py'),'--mode',stage,'--cell',str(job['cell']),
                '--offset',str(offset),'--n-cases',str(n),'--repeats',str(repeats),'--out',str(job['output'])]
            def affinity(core=cores[slot]):os.sched_setaffinity(0,{core})
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,preexec_fn=affinity)
            job.update(child=child,stream=stream,slot=slot,log=path,started=time.time());ACTIVE.append(job)
        for job in list(ACTIVE):
            code=job['child'].poll()
            if code is None:
                assert time.time()-job['started']<7200,'worker timed out';continue
            job['stream'].close();ACTIVE.remove(job)
            ev=dict(exit_code=code,cell=job['cell'],output=str(job['output']),
                output_bytes=job['output'].stat().st_size if job['output'].exists() else 0,
                actual_stdout_tail=job['log'].read_text(errors='replace')[-1600:])
            save(BASE/'job-evidence'/('%s-cell%d.json'%(stage,job['cell'])),ev)
            assert code==0 and ev['output_bytes']>0 and 'Traceback (most recent call last)'not in job['log'].read_text(),ev
            data=read(job['output']);assert data['status']=='已验证通过'and len(data['episodes'])==n
            assert data['cv_values_argmax_bitwise']and data['oracle_values_argmax_bitwise']and data['matched_norm_max_error']<3e-12
            finished.append(ev)
        save(BASE/'status.json',dict(status='未验证',stage=stage,active=[{'pid':j['child'].pid,'cell':j['cell']}for j in ACTIVE],
            completed_jobs=len(finished),pending=len(queue),wall_seconds=time.time()-start))
        time.sleep(3)
    verify()
    save(BASE/(stage+'-validation.json'),dict(status='已验证通过',jobs=finished,wall_seconds=time.time()-start))
    return [read(BASE/stage/('cell%d.json'%c))for c in range(6)]


def main():
    lock=(BASE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    probe=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.1)']);assert alive(probe.pid);probe.wait();assert not alive(probe.pid)
    save(BASE/'controller-pid.json',dict(pid=os.getpid(),running_finished_pid_test_passed=True))
    protocol=freeze();verify()
    blocks=jobs('implcheck',0,7,protocol['random_repeats'])
    save(BASE/'implcheck.json',dict(status='已验证通过',n_episodes=42,n_steps=sum(x['checked_steps']for x in blocks),
        cv_bitwise=True,oracle_bitwise=True,matched_joint_position_velocity=True,
        max_norm_error=max(x['matched_norm_max_error']for x in blocks),
        frozen_indices_and_commands_and_history=True,mean_worker_seconds_per_step=sum(x['wall_seconds']for x in blocks)/sum(x['checked_steps']for x in blocks),
        exit_codes=[0]*6))
    jobs('full',7,43,protocol['random_repeats'])
    path=BASE/'logs/summary.log'
    with path.open('w')as stream:
        result=subprocess.run([sys.executable,'-B',str(BASE/'summarize-random.py')],stdout=stream,stderr=subprocess.STDOUT)
    ev=dict(exit_code=result.returncode,actual_stdout_tail=path.read_text()[-2000:],output=str(BASE/'summary.json'),
        output_bytes=(BASE/'summary.json').stat().st_size if(BASE/'summary.json').exists() else 0)
    save(BASE/'job-evidence/summary.json',ev);assert result.returncode==0 and ev['output_bytes']>0,ev
    verify();save(BASE/'DONE.json',dict(status='已验证通过',n_episodes=300,n_steps=16020,train_episodes=0,new_environment_episodes=0,frozen_hashes_verified=True))
    save(BASE/'status.json',dict(status='已验证通过',stage='DONE',active=[],pending=0))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        for j in ACTIVE:
            if j['child'].poll()is None:j['child'].terminate()
        for j in ACTIVE:
            try:j['child'].wait(timeout=20)
            except subprocess.TimeoutExpired:j['child'].kill();j['child'].wait()
            j['stream'].close()
        save(BASE/'ERROR.json',dict(status='已确认失败',error=repr(error),fail_closed=True))
        (BASE/'REPORT.md').write_text('# H3 Random-Matched\n\n已确认失败：'+repr(error)+'\n后续步骤已经停止。不得将部分结果当完整结果。\n')
        raise
