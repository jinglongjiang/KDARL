"""Existing Mamba-VL value reference, not a new backbone or training run."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

BASE=Path(__file__).resolve().parent
OUT=BASE/'reference-mamba-v2'
EXPECTED='f23e411effac2f1959eead081fa49c908a85532642172c50a34427fe40bff2b9'


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--causal',required=True)
    p.add_argument('--acceptance',required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('mature_value_reference',BASE/'b/run-b.py')
    B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
    weight=BASE/'reference-mamba/inputs/rl_model_ep10000.pth';config=BASE/'reference-mamba/inputs/mamba.ini'
    assert B.sha(weight)==EXPECTED
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Never overwrite a partial reference run'
    conditions=['E1','E2','E3','E4','E5','E10']
    protocol=dict(question='Do matched known ORCA and existing Mamba-VL controllers also lose return when the physical crowd increases?',
        motivation='The frozen custom CV-MPC reference is not an official mature navigation baseline. Preserve its poor results; add an existing value-lookahead reference to meet the requested evidence level, not to replace any preregistered candidate gate.',
        model='Existing optimized Mamba-VL T24, seed419; different training run from fresh-history KDA/GRU. Descriptive reference, not a matched architectural performance comparison or full submitted-paper reproduction.',
        cases=list(range(90000,90016)),smoke_cases=list(range(90000,90003)),geometries=['circle','square'],conditions=conditions,
        formal_episodes=192,smoke_episodes=36,checkpoint_sha256=EXPECTED,config_sha256=B.sha(config),
        script_sha256=B.sha(__file__),frozen_b_driver_sha256=B.sha(BASE/'b/run-b.py'),
        controls='Same accepted terminal/history maintenance, legal TTC5,80 actions,alpha.3,margin.2,risk.8,original reward,50s,8m robot task. No training, tuning, hidden observations or 92000 cases.',
        decision='E1-to-E2 decline in both existing Mamba-VL and ORCA corroborates physical task difficulty for those references. No optimality or unavoidable-failure claim. Disagreement makes this reference corroboration inconclusive. Never retroactively change the B candidate gate.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    B.BASE=OUT;B.P=copy.deepcopy(B.P);B.P['conditions']=conditions
    original_scene=B.scene
    def scene(cfg,geometry,condition,case,net=None):
        if condition!='E10':return original_scene(cfg,geometry,condition,case,net)
        from crowd_sim.envs.crowd_sim import CrowdSim
        from crowd_sim.envs.utils.robot import Robot
        c=copy.deepcopy(cfg);c.set('sim','human_num','10');c.set('sim','test_sim',geometry+'_crossing')
        env=CrowdSim();env.configure(c);robot=Robot(c,'robot')
        robot.set_policy(net if net is not None else SimpleNamespace(multiagent_training=True));env.set_robot(robot);env.phase='test'
        if net is not None:
            net.set_env(env)
            if hasattr(net,'set_robot'):net.set_robot(robot)
            net.reset_episode_stats()
        env.reset(seed=case,options={'test_case':case})
        assert len(env.humans)==10 and env.time_step==.25 and env.time_limit==50 and not robot.visible
        assert (robot.px,robot.py,robot.gx,robot.gy)==(0,-4,0,4)
        return env,robot
    B.scene=scene
    item=dict(id='mamba-reference',backbone='mamba',seed=419,weight=str(weight),config=str(config),
              weight_sha256=EXPECTED,config_sha256=protocol['config_sha256'])
    B.manifest_item=lambda args,model:copy.deepcopy(item)
    args=SimpleNamespace(project=a.project,causal=a.causal,acceptance=a.acceptance,model='mamba-reference',controller='model',case=None)
    started=time.time()
    for phase in ('smoke','main'):
        args.phase=phase
        for geometry in ('circle','square'):
            args.geometry=geometry
            for condition in conditions:
                args.condition=condition;B.baseline(args)
        expected=3 if phase=='smoke' else 16
        for geometry in ('circle','square'):
            for condition in conditions:
                episodes=B.episodes(phase,'model','mamba-reference',geometry,condition)
                assert len(episodes)==expected
                n=5 if condition=='E1' else 10 if condition=='E10' else 20
                assert all(len(s['humans'])==n for e in episodes for s in e['steps'])
        (OUT/(phase+'-gate.json')).write_text(json.dumps(dict(status='PASS',episodes=expected*12)))
    groups=[];paired=[]
    for geometry in ('circle','square'):
        loaded={c:B.episodes('main','model','mamba-reference',geometry,c) for c in conditions}
        for c,eps in loaded.items():groups.append(dict(geometry=geometry,condition=c,**B.stats(eps)))
        for left,right in [('E1','E2'),('E1','E3'),('E2','E4'),('E2','E5'),('E1','E10'),('E10','E2')]:
            paired.append(dict(geometry=geometry,A=left,B=right,metrics=B.paired(loaded[left],loaded[right])))
        for e1,e3 in zip(loaded['E1'],loaded['E3']):
            for k in ('case','outcome','return','time','path_length','minimum_clearance'):
                assert e1[k]==e3[k], ('Mamba E1/E3 equivalence',geometry,e1['case'],k)
            assert len(e1['steps'])==len(e3['steps'])
            for s1,s3 in zip(e1['steps'],e3['steps']):
                assert s1['humans']==s3['humans'][:5]
                assert all(s1[k]==s3[k] for k in s1 if k!='humans'), ('Mamba E1/E3 step',geometry,e1['case'],s1['step'])
    assert B.sha(weight)==EXPECTED and B.sha(config)==protocol['config_sha256']
    (OUT/'summary.json').write_text(json.dumps(dict(status='PASS',groups=groups,paired=paired,wall_seconds=time.time()-started,
        input_hashes_end_verified=True,reference_level=protocol['model'],method_level_progress=False),indent=2))


if __name__=='__main__':main()
