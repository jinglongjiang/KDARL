"""CPU formula and corruption checks; no learned-policy inference or file writes."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


V = module('independent_verifier',ROOT/'verify-b-raw.py')
B = module('frozen_b',ROOT/'run-b.py')
B.pin(ROOT/'project')
CFG = B.cfg_read('/home/abc/workspace/ttc5-causal-20261011/inputs/kda-s419-t24/config.ini')


class Checks(unittest.TestCase):
    def test_independent_formula_parity(self):
        comparisons = 0
        for geo in B.P['geometries']:
            for cond in B.P['conditions']:
                env,_ = B.scene(CFG,geo,cond,90000)
                last = None
                for k in range(5):
                    s = B.state(env)
                    robot = [getattr(env.robot,f) for f in B.FIELDS]
                    humans = B.human_array(env.humans).tolist()
                    ids = V.ttc_indices(robot,humans)
                    self.assertEqual(ids,B.ttc_ids(s))
                    self.assertTrue(np.array_equal(V.visible_token(robot,humans,ids),B.token(s)))
                    commands = B.commands(B.grid(CFG),last)
                    expected = V.action_grid() if last is None else .3*np.asarray(last)[None]+.7*V.action_grid()
                    self.assertTrue(np.array_equal(commands,expected))
                    reward,clear,_,_ = V.immediate(robot,humans,commands)
                    rr,cc = B.immediate(env,commands)
                    self.assertTrue(np.array_equal(reward,rr))
                    self.assertTrue(np.array_equal(clear,cc))
                    self.assertTrue(np.array_equal(V.filter_scores(reward,clear),B.filter_scores(rr,cc)))
                    last = commands[k]
                    B.env_step(env,last)
                    comparisons += 1
        self.assertEqual(comparisons,50)

    def test_raw_pair_catches_hidden_trajectory_corruption(self):
        human = [0.,0.,0.,0.,.3,1.,1.,1.,0.]
        one = {'case':90000,'condition':'E1','steps':[{'step':0,'humans':[human]*5,'history_sha256':'abc'}]}
        far = copy.deepcopy(one)
        far['condition'] = 'E3'
        far['steps'][0]['humans'] += [[10000.+20*i,0.,0.,0.,.3,10000.+20*i,4.,1.,0.] for i in range(15)]
        self.assertTrue(V.compare_e1_e3(one,far)['all_archived_fields_exact'])
        bad = copy.deepcopy(far)
        bad['steps'][0]['humans'][0][2] = 1e-15
        with self.assertRaises(AssertionError):
            V.compare_e1_e3(one,bad)
        bad = copy.deepcopy(far)
        bad['steps'][0]['history_sha256'] = 'changed'
        with self.assertRaises(AssertionError):
            V.compare_e1_e3(one,bad)

    def test_contract_unchanged_and_calipers(self):
        self.assertEqual(V.sha(ROOT/'run-b.py'),V.EXPECTED_DRIVER_SHA)
        self.assertEqual(V.sha(ROOT/'b-contract.json'),V.EXPECTED_CONTRACT_SHA)
        a = {'step':20,'goal_distance':2.,'n_safe':40,'crowd2m':1}
        self.assertEqual(V.distance(a,a),(0.,[0.,0.,0.,0.]))
        self.assertIsNone(V.distance(a,dict(a,step=29)))
        self.assertEqual(V.delta_statistics([0]*16,np.random.RandomState(41995))['case_bootstrap95'],[0.,0.])


if __name__=='__main__':
    unittest.main(verbosity=2)
