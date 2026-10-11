"""Frozen four-arm training/evaluation, fail closed, without automatic restart."""
import sys
sys.dont_write_bytecode = True
import argparse
import configparser
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE=Path(__file__).resolve().parent
PROJECT=Path('/root/cc-diag-20261008/code')
OLD=Path('/root/kda-history-20261009')
ORDER=(('kda',24),('gru',24),('kda',12),('gru',12))
ACTIVE=[]


def read(path):
    return json.loads(Path(path).read_text())


def save(name,value):
    p=BASE/name; p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)); t.replace(p)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def alive(pid):
    p=Path('/proc/%d/stat'%pid)
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z'


def freeze():
    prep=read(BASE/'preparation.json'); assert prep['status']=='已验证通过' and prep['proposed_seed']==443
    assert shutil.disk_usage(BASE).free>=4*1024**3,'Need >=4GiB; backup/verify old weights before cleanup if insufficient'
    prior=read(OLD/'protocol.json')
    assert prep['dataset_sha256']==prior['dataset_sha256']
    assert len({x['shared_init_sha256'] for x in prep['arms'].values()})==1
    for arm,t in ORDER:
        n='%s-t%d'%(arm,t); c=configparser.RawConfigParser(); c.read(BASE/(n+'.ini'))
        assert c.getint('train','seed')==443 and c.getint('train','save_every')==1000
        assert c.getint('train','il_epochs')==50 and c.getint('train','train_episodes')==10000
        assert c.getint('temporal','t')==c.getint('buffer','seq_len')==t
        assert c.getint('sim','human_num')==5 and c.get('sim','train_val_sim')=='circle_crossing'
        assert sha(BASE/(n+'.ini'))==prep['arms'][n]['config_sha256']
    protocol=dict(task='KDA_GRU_HISTORY_SECOND_SEED_REPLICATION',seed=443,prior_seed=419,
        arms=ORDER,il_epochs=50,rl_episodes=10000,fresh_il=True,
        dataset_sha256=prep['dataset_sha256'],shared_initialization_hashes={n:x['shared_init_sha256'] for n,x in prep['arms'].items()},
        frozen=['reward','safety','ttc5','action_grid80','smoothing','learning_rate','temporal architecture except T'],
        evaluation=dict(case_base=91000,n_cases_per_cell=500,cells=list(range(6)),selection='ttc5',
            scope='same development cases as seed419; comparable, NOT independent case confirmation',reserved_92000_92499_untouched=True),
        primary=dict(family_size=1,alpha=.05,definition='D=(SR_KDA24-SR_KDA12)-(SR_GRU24-SR_GRU12)',
            direction='D>0',confidence_interval='two-sided95% as explicitly requested',unit='case ID clustered across all six cells and all four arms',bootstrap_replicates=10000),
        secondary=dict(family_size=6,tests='KDA/GRU T24-T12 SR/CR/TO',test='paired exact McNemar',bonferroni_alpha=.05/6),
        scale_check='marginal failure OR T24/T12 and log OR_KDA-log OR_GRU; D>0 corresponds to log-failure-odds interaction<0',
        gate=dict(REPLICATED_PRELIMINARY='D>0 and CI excludes0; KDA SR delta>0, GRU<0, both pass secondary correction; scales agree; prior seed419 same signs',
                  PARTIAL='D>0 and CI excludes0, but any full criterion fails, including missing within-arm significance',
                  NOT_CONFIRMED='D<=0 or CI contains0'),
        stage2_authorized=False,stage2='M1 and M2 require new user authorization even if replicated',
        M1_deferred=['full24','replace older12 while retaining recent12','shuffle older frames only','amplitude-matched recent-history positive/control intervention'],
        M1_limits='History replacement creates distribution shift; value/ranking/action sensitivity alone is not memory causal value',
        M2_deferred='Only after authorized M1 localizes an effect; candidate explanations are not established defects',
        n_seeds_scope='n=2 checks sign replication only; do not estimate architecture-level seed variance',
        storage=dict(start_min_free_GiB=4,checkpoint_every=1000,resume_latest_copies=1,
            backup='Immediately after all training and evaluation finish: rsync, both-side every-file SHA256, then weight-only deletion',
            deletion_guard='Neither inventory empty; re-enumerate and hash remote list before deletion; no directory or non-weight deletion',
            no_deletion_on_failure_or_partial=True),
        scheduling=dict(stagger_seconds=60,max_IL_workers=3,max_training_workers=4,max_eval_during_training=2,max_eval_after_training=4),
        wall_budget_hours=8,reserve_report_minutes=30,no_automatic_restart=True,
        other_line='Bayesian CA baseline not coupled to these overnight GPU runs; not launched by this controller')
    assert not (BASE/'protocol.json').exists(),'Do not overwrite frozen protocol'
    save('protocol.json',protocol)
    save('execution-freeze.json',dict(protocol_sha256=sha(BASE/'protocol.json'),preparation_sha256=sha(BASE/'preparation.json'),
        configs={n:sha(BASE/(n+'.ini')) for n in prep['arms']},scripts={p.name:sha(p) for p in BASE.glob('*.py')},
        environment_config_sha256=sha(BASE/'env.config'),
        helpers=prep['helper_sha256'],parent_source_sha256=prep['source_sha256'],dataset=prior['dataset'],dataset_sha256=prep['dataset_sha256'],
        seed419_statistics_sha256=sha(BASE/'seed419-interaction.json')))
    verify(include_dataset=True)


def verify(include_dataset=False):
    f=read(BASE/'execution-freeze.json')
    assert sha(BASE/'protocol.json')==f['protocol_sha256']
    assert sha(BASE/'preparation.json')==f['preparation_sha256']
    assert sha(BASE/'seed419-interaction.json')==f['seed419_statistics_sha256']
    assert sha(BASE/'env.config')==f['environment_config_sha256'],'Required environment/grid configuration changed'
    for n,h in f['configs'].items(): assert sha(BASE/(n+'.ini'))==h
    for n,h in f['scripts'].items(): assert sha(BASE/n)==h
    for n,h in f['helpers'].items(): assert sha(OLD/n)==h
    current={str(p.relative_to(PROJECT)):sha(p) for p in PROJECT.rglob('*.py')}
    assert current and current==f['parent_source_sha256'],'Frozen parent source changed'
    if include_dataset: assert sha(Path(f['dataset']))==f['dataset_sha256'],'Raw demonstration pool changed'


def launch(command,kind,name,core,**meta):
    path=BASE/'logs'/(name+'.log'); path.parent.mkdir(exist_ok=True)
    assert not path.exists(),'Do not silently restart previous jobs'
    stream=path.open('w'); env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                           preexec_fn=lambda:os.sched_setaffinity(0,{core}))
    ACTIVE.append(dict(child=child,stream=stream,kind=kind,name=name,core=core,log=path,**meta))
    save('launches/'+name+'.json',dict(pid=child.pid,command=command,core=core,started=time.time(),kind=kind))


def torch_load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)


def stop_owned():
    for j in ACTIVE:
        if j['child'].poll() is None: j['child'].terminate()
    for j in ACTIVE:
        try:j['child'].wait(timeout=20)
        except subprocess.TimeoutExpired:j['child'].kill();j['child'].wait()
        j['stream'].close()
    ACTIVE.clear()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true'); args=parser.parse_args()
    lock=(BASE/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.freeze_only:
        freeze();print(json.dumps({'status':'已验证通过','protocol_frozen':True,'training_started':False},ensure_ascii=False));return
    assert not (BASE/'DONE.json').exists() and not (BASE/'ERROR.json').exists()
    verify();assert shutil.disk_usage(BASE).free>=4*1024**3
    test=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.2)']);assert alive(test.pid);test.wait();assert not alive(test.pid)
    start=time.time();deadline=start+8*3600;stop_at=deadline-1800;last_launch=0
    save('controller-pid.json',dict(pid=os.getpid(),start=start,deadline=deadline,pid_live_dead_test_passed=True))
    pending=list(ORDER); trained=set(); eval_pending=[]; eval_done=[]; stopping=False
    while time.time()<deadline:
        for j in list(ACTIVE):
            assert 'Traceback (most recent call last)' not in j['log'].read_text(errors='replace'), ('Traceback in active worker',j['name'])
            code=j['child'].poll()
            if code is None: continue
            j['stream'].close();ACTIVE.remove(j)
            evidence=dict(name=j['name'],kind=j['kind'],exit_code=code,log=str(j['log']),
                          actual_output_tail=j['log'].read_text(errors='replace')[-2200:])
            save('job-evidence/'+j['name']+'.json',evidence)
            if code==75 and j['kind']=='train' and (stopping or time.time()>=stop_at):
                stopping=True;assert (j['run']/'PAUSED.json').exists();continue
            assert code==0,evidence
            assert 'Traceback (most recent call last)' not in j['log'].read_text(errors='replace'),evidence
            if j['kind']=='train':
                final=j['run']/'rl_model_ep10000.pth'; il=j['run']/'il_policy.pth'
                a=torch_load(final); b=torch_load(il)
                assert a['episode']==10000 and a['algo']=='sarl' and a['optim_value_state']['state']
                assert b['meta']['epochs']==50 and b['meta']['seq_len']==j['length']
                assert (j['run']/'worker-completed.json').exists() and (j['run']/'il-finished.json').exists()
                digest=sha(final);assert digest
                save('runs/'+j['name']+'/training-completed.json',dict(final_episode=10000,il_epochs=50,fresh_il=True,seed=443,checkpoint_sha256=digest))
                trained.add(j['name']);eval_pending.extend((j['name'],j['arm'],j['length'],c,digest) for c in range(6))
                del a,b
            else:
                d=read(j['output']);assert len(d['episodes'])==500 and d['selection']=='ttc5' and d['grid_actions']==80
                assert d['seq_len']==j['length'] and d['checkpoint_sha256']==j['digest']
                assert {x['case'] for x in d['episodes']}==set(range(91000,91500))
                eval_done.append(j['name'])
        if time.time()>=stop_at or shutil.disk_usage(BASE).free<1536*1024**2:
            if not stopping:
                save('stop-request.json',dict(reason='wall_budget_or_disk',free_disk_bytes=shutil.disk_usage(BASE).free,time=time.time()))
                for j in ACTIVE:
                    if j['kind']=='train': (j['run']/'STOP').touch()
            stopping=True
        trains=[j for j in ACTIVE if j['kind']=='train']
        def in_il(j):
            p=j['run']/'phase.json'
            return not p.exists() or not read(p)['stage'].startswith('RL')
        if pending and not stopping and len(trains)<4 and sum(in_il(j) for j in trains)<3 and time.time()-last_launch>=60:
            verify();arm,t=pending.pop(0);name='%s-t%d'%(arm,t);run=BASE/'runs'/name
            assert not run.exists(),'Formal weights must start fresh'
            command=[sys.executable,'-B',str(OLD/'train-worker.py'),'--project',str(PROJECT),'--deadline',str(stop_at),
                     '--config',str(BASE/(name+'.ini')),'--outdir',str(run),'--device','cuda','--seed','443','--temporal-backbone',arm]
            launch(command,'train',name,sorted(os.sched_getaffinity(0))[ORDER.index((arm,t))],run=run,arm=arm,length=t)
            last_launch=time.time()
        evals=[j for j in ACTIVE if j['kind']=='eval'];cores=sorted(os.sched_getaffinity(0))
        max_eval=2 if trains or pending else 4
        free=[c for c in cores[4:] if c not in [j['core'] for j in evals]]
        if eval_pending and not stopping and len(evals)<max_eval and free:
            n,arm,t,cell,digest=eval_pending.pop(0);out=BASE/'evaluation'/n/('cell%d.json'%cell)
            command=[sys.executable,'-B',str(OLD/'evaluate-history.py'),'--project',str(PROJECT),'--arm-dir',str(BASE/'runs'/n),
                     '--config',str(BASE/(n+'.ini')),'--backbone',arm,'--human-select','ttc5','--expected-checkpoint-sha',digest,
                     '--case-base','91000','--n-cases','500','--cell',str(cell),'--out',str(out)]
            launch(command,'eval',n+'-cell%d'%cell,free[0],output=out,length=t,digest=digest)
        save('status.json',dict(stage='STOP_REQUESTED' if stopping else 'RUNNING',seed=443,
            elapsed_seconds=time.time()-start,remaining_seconds=max(0,deadline-time.time()),free_disk_bytes=shutil.disk_usage(BASE).free,
            active=[dict(pid=j['child'].pid,kind=j['kind'],name=j['name']) for j in ACTIVE],
            pending_training=pending,trained=sorted(trained),pending_evaluation=len(eval_pending),complete_eval_cells=len(eval_done),stage2_started=False))
        if not ACTIVE and (stopping or (not pending and not eval_pending)):break
        time.sleep(10)
    stop_owned();verify(include_dataset=True)
    if len(trained)!=4 or len(eval_done)!=24:
        save('DONE.json',dict(status='PARTIAL_OR_PAUSED',trained=sorted(trained),complete_eval_cells=len(eval_done),stage2_started=False))
        (BASE/'REPORT.md').write_text('# 第二seed复核\n\nPARTIAL_OR_PAUSED：未完成全部四臂10000回合和24格评测，不进行完整比较。\n恢复checkpoint和错误/暂停记录保留，不自动重启，不自动进阶段二。\n')
        save('status.json',dict(stage='PARTIAL_OR_PAUSED',active=[],stage2_started=False));return
    with (BASE/'logs/summary.log').open('w') as f:
        result=subprocess.run([sys.executable,'-B',str(BASE/'analyze-results.py')],stdout=f,stderr=subprocess.STDOUT)
    save('job-evidence/summary.json',dict(exit_code=result.returncode,actual_output_tail=(BASE/'logs/summary.log').read_text()[-2200:]))
    assert result.returncode==0;verify()
    save('DONE.json',dict(status='COMPLETED',seed=443,verdict=read(BASE/'summary.json')['verdict'],
                         wall_seconds=time.time()-start,trained_arms=4,evaluated_episodes=12000,source_unchanged=True,stage2_started=False))
    save('status.json',dict(stage='COMPLETED',active=[],verdict=read(BASE/'summary.json')['verdict'],stage2_started=False))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        stop_owned();save('ERROR.json',dict(status='FAILED',error=repr(error),automatic_restart=False))
        (BASE/'REPORT.md').write_text('# 第二seed复核\n\nFAILED：'+repr(error)+'\n错误日志和权重保留，未静默重启，未进入阶段二。\n')
        raise
