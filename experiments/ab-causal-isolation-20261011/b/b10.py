#!/usr/bin/env python3
"""Frozen native-N10 extension. Main owns GPU; this entrypoint never trains."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
PROTOCOL = ROOT/'b10-protocol.json'
EXT = json.loads(PROTOCOL.read_text())
spec = importlib.util.spec_from_file_location('b_parent_frozen',ROOT/'run-b.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
assert B.sha(ROOT/'run-b.py')==EXT['parent_driver_sha256']
assert B.sha(ROOT/'b-contract.json')==EXT['parent_contract_sha256']
PARENT_P = copy.deepcopy(B.P)
ORIGINAL_SCENE,ORIGINAL_PROVENANCE = B.scene,B.provenance
B.P = dict(copy.deepcopy(PARENT_P),**EXT['overrides'])
B.CONTRACT = PROTOCOL
B.BASE = ROOT/'n10'


def scene(cfg,geometry,condition,case,net=None):
    if condition!='E10':
        current = B.P
        try:
            B.P = PARENT_P
            return ORIGINAL_SCENE(cfg,geometry,condition,case,net)
        finally:
            B.P = current
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot
    assert case in B.P['cases'] and geometry in B.P['geometries']
    cfg = copy.deepcopy(cfg)
    cfg.set('sim','human_num','10')
    cfg.set('sim','test_sim',geometry+'_crossing')
    env = CrowdSim()
    env.configure(cfg)
    robot = Robot(cfg,'robot')
    robot.set_policy(net if net is not None else SimpleNamespace(multiagent_training=True))
    env.set_robot(robot)
    env.phase = 'test'
    if net is not None:
        net.set_env(env)
        if hasattr(net,'set_robot'):
            net.set_robot(robot)
        net.reset_episode_stats()
    env.reset(seed=case,options={'test_case':case})
    assert len(env.humans)==env.human_num==10
    assert (robot.px,robot.py,robot.gx,robot.gy)==(0,-4,0,4)
    assert not robot.visible and env.time_step==.25 and env.time_limit==50
    assert env.circle_radius==4 and env.square_width==10
    assert env.progress_reward==env.time_penalty==env.stand_penalty==0
    return env,robot


def provenance(args):
    out = ORIGINAL_PROVENANCE(args)
    out.update(extension_script_sha256=B.sha(SCRIPT),parent_contract_sha256=EXT['parent_contract_sha256'],
               extension='native-N10-original-B5-completion')
    return out


B.scene,B.provenance = scene,provenance


def cpu(args):
    import numpy as np
    cfg = B.cfg_read(args.config)
    spec = importlib.util.spec_from_file_location('original_n10_reference',Path(args.causal)/'reference.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    started,checks = time.perf_counter(),[]
    for geometry in B.P['geometries']:
        cell = ref.CELLS.index((10,geometry+'_crossing'))
        for case in B.P['cases']:
            env,robot = scene(cfg,geometry,'E10',case)
            oldnet = SimpleNamespace(multiagent_training=True,reset_episode_stats=lambda:None)
            old,oldrobot = ref.environment(copy.deepcopy(cfg),oldnet,cell,case)
            a,b = B.human_array(env.humans),B.human_array(old.humans)
            assert np.array_equal(a,b),('nativeN10 differs original reset',geometry,case)
            rob = [getattr(robot,f) for f in B.FIELDS]
            assert rob==[getattr(oldrobot,f) for f in B.FIELDS]
            assert B.params(env.humans)==B.params(old.humans)
            forwarded = []
            for condition in PARENT_P['conditions']:
                first,_ = scene(cfg,geometry,condition,case)
                current = B.P
                try:
                    B.P = PARENT_P
                    other,_ = ORIGINAL_SCENE(cfg,geometry,condition,case)
                finally:
                    B.P = current
                assert np.array_equal(B.human_array(first.humans),B.human_array(other.humans))
                assert B.params(first.humans)==B.params(other.humans)
                forwarded.append(condition)
            ids = B.ttc_ids(B.state(env))
            cmd = B.grid(cfg)
            rr,cc = B.immediate(env,cmd)
            assert len(ids)==5 and len(cmd)==len(rr)==len(cc)==80
            checks.append({'geometry':geometry,'case':case,'human_num':10,'robot':rob,'humans':a.tolist(),
                'actual_orca_parameters':B.params(env.humans),'native_original_initial_exact':True,
                'old_conditions_forwarded_exact':forwarded,'TTC5_ids':ids,
                'first80_rewards':rr.tolist(),'first80_clearance':cc.tolist()})
    result = {'status':'PASS','checks':checks,'cases':len(checks),'model_inference':False,
        'old_outcomes_consumed':False,'learned_policy_evaluation':False,'provenance':provenance(args),
        'old_reference_source_sha256':B.sha(Path(args.causal)/'reference.py'),
        'wall_seconds':time.perf_counter()-started}
    B.save(B.BASE/'work/cpu-main.json',result)
    print(json.dumps({'status':'PASS','native_N10_initial_pairs':len(checks),
        'old_condition_reset_pairs':len(checks)*5,'wall_seconds':result['wall_seconds']}),flush=True)


def smoke_gate(args):
    cpu_data = json.loads((B.BASE/'work/cpu-main.json').read_text())
    assert cpu_data['status']=='PASS' and cpu_data['cases']==32
    assert cpu_data['provenance']['contract_sha256']==B.sha(PROTOCOL)
    assert cpu_data['provenance']['extension_script_sha256']==B.sha(SCRIPT)
    B.smoke_gate(args)
    gate = json.loads((B.BASE/'work/smoke-accepted.json').read_text())
    assert gate['episodes']==36


def jobs(args):
    result = []
    for controller in ['model']+B.P['references']:
        models = B.P['models'] if controller=='model' else [B.P['models'][0]]
        for model in models:
            for geometry in B.P['geometries']:
                result.append([sys.executable,str(ROOT/'b10.py'),'baseline','--project',args.project,
                    '--causal',args.causal,'--acceptance',args.acceptance or '',
                    '--phase',args.phase,'--controller',controller,'--model',model,'--geometry',geometry])
    print(json.dumps({'jobs':result,'count':len(result),'episodes_per_job':3 if args.phase=='smoke' else 16,
        'starts_jobs':False,'gpu_launch_owner':'main'}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['cpu','baseline','smoke-gate','roots','counterfactual','jobs'])
    p.add_argument('--project',default=str(ROOT/'project'))
    p.add_argument('--causal',default='/home/abc/workspace/ttc5-causal-20261011')
    p.add_argument('--config',default='/home/abc/workspace/ttc5-causal-20261011/inputs/kda-s419-t24/config.ini')
    p.add_argument('--phase',choices=['smoke','main'],default='smoke')
    p.add_argument('--model',choices=B.P['models'],default=B.P['models'][0])
    p.add_argument('--controller',choices=['model']+B.P['references'],default='model')
    p.add_argument('--geometry',choices=B.P['geometries'],default='circle')
    p.add_argument('--condition',choices=['E10'],default='E10')
    p.add_argument('--case',type=int)
    p.add_argument('--acceptance')
    args = p.parse_args()
    if args.mode=='jobs':
        jobs(args)
        return
    B.pin(args.project)
    if args.mode=='cpu':
        cpu(args)
    elif args.mode=='smoke-gate':
        smoke_gate(args)
    else:
        B.budget()
        assert json.loads((B.BASE/'work/cpu-main.json').read_text())['status']=='PASS'
        {'baseline':B.baseline,'roots':B.roots,'counterfactual':B.counterfactual}[args.mode](args)


if __name__=='__main__':
    main()
