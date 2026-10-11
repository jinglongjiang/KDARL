"""CPU implementation checks; these are not learned-policy experimental results."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace, MethodType
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('b_driver',ROOT/'run-b.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
B.pin(ROOT/'project')

import numpy as np
import torch
from collections import deque
from crowd_nav.policy.mamba_rl import MambaRLPolicy

CFG = B.cfg_read('/home/abc/workspace/ttc5-causal-20261011/inputs/kda-s419-t24/config.ini')


def zero_value_scorer(env):
    n = SimpleNamespace(action_space=B.grid(CFG),_phase='test',test_action_smoothing=.3,
        _last_action=None,_history=deque(maxlen=24),time_step=.25,seq_len=24,
        success_radius=.25,progress_reward=0.,time_penalty=0.,stand_penalty=0.,
        discomfort_dist=.2,discomfort_penalty_factor=.5,collision_penalty=-.5,success_reward=1.,
        gamma=.99,device='cpu',lookahead_ablation_mode=None,test_min_clearance=.2,test_risk_lambda=.8,
        _scoring_env=env,forward_value=lambda x:torch.zeros(len(x)))
    n.candidate_commands = MethodType(MambaRLPolicy.candidate_commands,n)
    n.propagate = MethodType(MambaRLPolicy.propagate,n)
    n._lookahead_terminal = MethodType(MambaRLPolicy._lookahead_terminal,n)
    n._build_joint_state_34 = lambda r,hs:MambaRLPolicy._build_joint_state_34(n,r,[hs[i] for i in n.ids])
    return n


class Checks(unittest.TestCase):
    def test_source_reward_clearance_token_commands_parity(self):
        comparisons = 0
        for geometry in ('circle','square'):
            for condition in B.P['conditions']:
                env,_ = B.scene(CFG,geometry,condition,90000)
                n = zero_value_scorer(env)
                for k in range(5):
                    s = B.state(env)
                    n.ids = B.ttc_ids(s)
                    out = MambaRLPolicy.score_sarl_candidates(n,s)
                    cmds = B.commands(n.action_space,n._last_action)
                    rr,cc = B.immediate(env,cmds)
                    self.assertTrue(np.array_equal(rr,out['rewards'].numpy()))
                    self.assertTrue(np.array_equal(cc,out['clearance'].numpy()))
                    self.assertTrue(np.array_equal(B.token(s),out['current_token']))
                    self.assertEqual(cmds,out['commands'])
                    self.assertTrue(np.array_equal(B.filter_scores(rr,cc),out['scores'].numpy()))
                    idx = int(out['scores'].argmax())
                    n._last_action = cmds[idx]
                    n._history.append(out['current_token'])
                    B.env_step(env,cmds[idx])
                    comparisons += 1
        self.assertEqual(comparisons,50)

    def test_reference_replay_and_overrides(self):
        args = SimpleNamespace(model='kda-s419-t24')
        for controller in B.P['references']:
            net = B.ReferenceInterface(CFG)
            one = B.rollout(args,CFG,net,controller,'circle','E1',90000)
            again = B.rollout(args,CFG,net,controller,'circle','E1',90000,reference=one)
            keys = [k for k in one if not k.startswith('latency_ms_')]
            self.assertTrue(all(one[k]==again[k] for k in keys))
            self.assertTrue(one['E3_shadow_exact'])
            row = one['steps'][0]
            alt = B.a1(row)
            self.assertIsNotNone(alt)
            changed = B.rollout(args,CFG,net,controller,'circle','E1',90000,
                override={'step':0,'index':alt},reference=one)
            self.assertEqual(changed['steps'][0]['command'],row['commands'][alt])

    def test_e4_new_starts_and_e5_human_only_map(self):
        for geometry in ('circle','square'):
            e2,_ = B.scene(CFG,geometry,'E2',90001)
            e4,_ = B.scene(CFG,geometry,'E4',90001)
            e5,_ = B.scene(CFG,geometry,'E5',90001)
            mapped = B.human_array(e2.humans)
            mapped[:,[0,1,5,6]] *= 2
            self.assertTrue(np.array_equal(B.human_array(e5.humans),mapped))
            self.assertEqual((e5.robot.py,e5.robot.gy),(-4,4))
            self.assertTrue(np.array_equal(B.human_array(e2.humans[:5]),B.human_array(e4.humans[:5])))
            for i,h in enumerate(e4.humans[5:],start=5):
                for other in [e4.robot]+e4.humans[:i]:
                    self.assertGreaterEqual(np.hypot(h.px-other.px,h.py-other.py),h.radius+other.radius+.2)

    def test_accepts_cherry_pick_by_source_not_ancestry(self):
        accepted = ROOT.parent/'acceptance.json'
        if not accepted.exists():
            self.skipTest('Main acceptance artifact not present locally')
        args = SimpleNamespace(acceptance=str(accepted),project=str(ROOT/'project'))
        self.assertEqual(B.acceptance(args)['status'],'PASS')

    def test_guard_preserves_full_sample_and_prohibits_92000(self):
        self.assertEqual(B.P['cases'],list(range(90000,90016)))
        with self.assertRaises(AssertionError):
            B.scene(CFG,'circle','E1',92000)
        row = {'step':20,'goal_distance':2.,'n_safe':40,'crowd2m':1}
        other = dict(row,step=29)
        self.assertIsNone(B.opportunity_distance(row,other))


if __name__=='__main__':
    started = B.time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    B.save(ROOT/'work/unit-tests.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'wall_seconds':B.time.perf_counter()-started,'test_script_sha256':B.sha(__file__),
        'driver_sha256':B.sha(ROOT/'run-b.py'),'learned_policy_evaluation':False})
    raise SystemExit(0 if result.wasSuccessful() else 1)
