"""Synthetic engineering checks only; never navigation evidence."""
import gzip
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('summary_test',HERE/'summarize-a.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


def save(path,value):
    path.write_text(json.dumps(value))


class Summary(unittest.TestCase):
    def test_root_decision_snapshot_readonly_and_complete(self):
        source=ast.parse((HERE/'closedloop-a.py').read_text())
        fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='root_decision_snapshot')
        helper=ast.parse((HERE/'run-a.py').read_text())
        digest=next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='digest')
        namespace={'np':np,'torch':torch,'hashlib':hashlib}
        exec(compile(ast.Module(body=[digest],type_ignores=[]),'digest','exec'),namespace)
        from types import SimpleNamespace
        namespace['A']=SimpleNamespace(digest=namespace['digest'])
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'root-log','exec'),namespace)
        values=torch.arange(80,dtype=torch.float32)
        out=dict(values=values,raw_scores=values+1.,scores=values+2.)
        windows=torch.zeros(80,2,8,13);current=np.zeros((8,13),np.float32)
        original={k:v.clone() for k,v in out.items()}
        row=namespace['root_decision_snapshot'](out,windows,current,'ttc5')
        for k,v in out.items():
            self.assertTrue(torch.equal(v,original[k]));self.assertEqual(row[k],v.tolist())
        self.assertEqual(row['windows_sha256'],namespace['digest'](windows.numpy()))
        self.assertEqual(row['current_input_sha256'],namespace['digest'](current))
        out['values'][0]=999.
        self.assertEqual(row['values'][0],0.)

    def test_both_entry_sweeps_geometry_and_no_mutation(self):
        functions=[]
        for name in ('closedloop-a.py','paired5-continuation.py'):
            fn=next(n for n in ast.parse((HERE/name).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='swept_clearances')
            namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),name,'exec'),namespace)
            functions.append(namespace['swept_clearances'])
        robot=[0.,0.,0.,0.,.2,0.,0.,1.,0.]
        before=(robot,[[.5,0.,0.,0.,.2]])
        after=(robot,[[1.5,0.,4.,0.,.2]])
        snapshot=json.dumps([before,after]);rng=np.random.get_state()
        for function in functions:
            result=function(before,after,[4.,0.])
            self.assertAlmostEqual(result['env_swept_clearance'],-.4)
            self.assertAlmostEqual(result['actual_swept_clearance'],.1)
            result=function((robot,[[1.,0.,0.,0.,.2]]),(robot,[[-1.,0.,-8.,0.,.2]]),[0.,0.])
            self.assertAlmostEqual(result['env_swept_clearance'],.6)
            self.assertAlmostEqual(result['actual_swept_clearance'],-.4)
            result=function((robot,[[1.,0.,0.,0.,.2]]),(robot,[[1.,0.,0.,0.,.2]]),[0.,0.])
            self.assertAlmostEqual(result['actual_swept_clearance'],.6)
        self.assertEqual(snapshot,json.dumps([before,after]))
        np.testing.assert_array_equal(rng[1],np.random.get_state()[1])
        self.assertEqual(rng[2:],np.random.get_state()[2:])

    def test_bootstrap_keeps_case_clusters(self):
        rows=[dict(case=1,x=1.),dict(case=1,x=-1.),dict(case=2,x=0.)]
        result=S.bootstrap(rows,['x'],100,17)
        self.assertEqual(result['cases'],2)
        self.assertEqual(result['intervals']['x']['low'],0.)
        self.assertEqual(result['intervals']['x']['high'],0.)

    def test_closedloop_actual_denominators_and_common_success(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            p=Path(tmp);model='synthetic-only';plan=dict(arms=['ttc5-V','allN-V'],roots=[]);rows=[]
            outcomes=[('failure','timeout','success'),('success-control','collision','collision'),('success-control','success','success')]
            for rid,(scope,base,full) in enumerate(outcomes):
                case=90004+rid*5
                plan['roots'].append(dict(model=model,root_id=rid,case=case,scope=scope,matching_status='independent_not_matched'))
                for arm,outcome in (('ttc5-V',base),('allN-V',full)):
                    clearance=-.1 if arm=='allN-V' and rid==0 else .1
                    steps=[dict(step=0,reward=.5,command=[0.,0.],terminal=True,env_swept_clearance=clearance,actual_swept_clearance=clearance)]
                    digest=hashlib.sha256(json.dumps(steps,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    rows.append(dict(model=model,root_id=rid,case=case,scope=scope,arm=arm,step=0,steps=steps,
                        continuation_sha256=digest,restoration_exact=True,root_candidate_controls={},baseline_outcome_new=base,
                        outcome=outcome,root_return=.5,time=.25,path_length=1.,mean_inference_seconds=.01,
                        terminal_raw_difference=dict(old_raw=[.1],new_raw=[.1]),min_env_swept_clearance=clearance,min_actual_swept_clearance=clearance,
                        env_physical_collision_count=int(clearance<0),actual_physical_collision_count=int(clearance<0),physical_collision_count=int(clearance<0)))
            with gzip.open(p/'closedloop-rows.jsonl.gz','wt') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            save(p/'closedloop-summary.json',dict(continuations=6))
            save(p/'closedloop-protocol.json',dict(plan=plan))
            for scope in ('failure','success-control'):
                save(p/('A0-repeat-'+model+'-'+scope+'.json'),dict(status='PASS',exact=True))
            result=S.closedloop(p,draws=100)
            total=next(r for r in result['totals'] if r['old_scope']=='all')
            self.assertEqual(total['denominators'],dict(rescue=2,damage=1,common_success=1))
            self.assertEqual(total['counts']['rescue'],1)
            self.assertEqual(total['counts']['removed_timeout'],1)
            self.assertEqual(total['counts']['allN_success_with_actual_physical_collision'],1)
            self.assertEqual(total['counts']['added_actual_physical_collision'],1)
            self.assertEqual(total['dual_sweep_roots'],3)

    def test_paired5_early_denominator_and_interaction(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            p=Path(tmp);model='synthetic-only';cases=list(range(90000,90016));rows=[]
            rows.append(dict(model=model,case=cases[0],split='train',status='PRE_ROOT_TERMINAL',baseline_outcome='success'))
            for case in cases[1:]:
                for arm in ('old-train','fixed-train','old-deploy','fixed-deploy'):
                    reward=1. if arm=='fixed-train' else 0.
                    clear=-.1 if arm=='fixed-train' else .1
                    rows.append(dict(model=model,case=case,arm=arm,status='PASS',split='test',root_return=.99*reward,
                        rewards=[0.,reward],commands=[[0.,0.],[0.,0.]],forced_command=[0.,0.],forced_reward=0.,restoration_exact=True,root_history_sha256='same',
                        true_next_window_sha256='same',outcome='success',time=1.,path_length_after_root=1.,mean_inference_seconds=None,
                        sweep_steps=[dict(command=[0.,0.],env_swept_clearance=.1,actual_swept_clearance=clear) for _ in range(2)],
                        min_env_swept_clearance=.1,min_actual_swept_clearance=clear,env_physical_collision_count=0,
                        actual_physical_collision_count=2*int(clear<0),physical_collision_count=2*int(clear<0)))
            with gzip.open(p/'paired5-rows.jsonl.gz','wt') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            save(p/'paired5-protocol.json',dict(cases=cases,models=[model]))
            save(p/'paired5-summary.json',dict(models=[]))
            result=S.paired5(p,draws=100)
            effect=next(r for r in result['totals'] if r['factor']=='history_x_deployment_interaction')
            self.assertEqual(effect['denominator_all_cases'],16)
            self.assertEqual(effect['valid_roots'],15)
            self.assertEqual(effect['negative_G'],15)
            self.assertEqual(effect['paired_case_bootstrap']['intervals']['delta_G']['mean'],-.99)
            self.assertEqual(effect['paired_case_bootstrap']['intervals']['delta_actual_physical_collision']['mean'],-15/16)
            self.assertEqual(len(result['arms']),4)


if __name__=='__main__':unittest.main()
