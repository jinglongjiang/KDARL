"""Synthetic corruption tests only. No torch, environments, models or rollouts."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('continuation_verifier',ROOT/'verify-b-continuations.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
V = C.V


def synthetic_episode(condition='E2',override=None):
    n = 10 if condition=='E10' else 20
    humans = [[1000.+20*i,1.,0.,0.,.3,1000.+20*i,1.,1.,0.] for i in range(n)]
    robot,records,history,last = [0.,-4.,0.,0.,.3,0.,4.,1.,np.pi/2],[],[],None
    path_length = 0.
    for k in range(201):
        commands = V.action_grid() if last is None else .3*last[None]+.7*V.action_grid()
        reward,clear,_,_ = V.immediate(robot,humans,commands)
        raw = reward.copy()
        scores = V.filter_scores(raw,clear)
        chosen = int(np.argmax(scores))
        idx = override['index'] if override and k==override['step'] else chosen
        ids = V.ttc_indices(robot,humans)
        token = V.visible_token(robot,humans,ids)
        gd = np.hypot(robot[0]-robot[5],robot[1]-robot[6])
        progress = gd-np.hypot(robot[0]+commands[:,0]*.25-robot[5],robot[1]+commands[:,1]*.25-robot[6])
        row = {'step':k,'robot':list(robot),'humans':copy.deepcopy(humans),'ids':ids,
            'goal_distance':float(gd),'progress':progress.tolist(),'commands':commands.tolist(),
            'chosen':chosen,'crowd2m':0,'token_sha256':V.byte_hash(token),
            'history_sha256':V.byte_hash(np.asarray(history[-24:])),
            'rewards':reward.tolist(),'clearance':clear.tolist(),'values':[0.]*80,
            'raw_scores':raw.tolist(),'scores':scores.tolist(),'n_safe':80,
            'command':commands[idx].tolist(),'reward':-.5 if k==200 else 0.,'terminal':k==200,
            'info':{'event':'timeout' if k==200 else 'nothing','dmin':float(clear[idx])},
            'actual_swept':float(clear[idx]),'CV_swept':float(clear[idx]),'endpoint':float(clear[idx])}
        if override and k==override['step']:
            row['executed_override'] = idx
        records.append(row)
        history.append(token)
        last = commands[idx]
        nxt = [robot[0]+last[0]*.25,robot[1]+last[1]*.25]
        path_length += np.hypot(nxt[0]-robot[0],nxt[1]-robot[1])
        robot[:4] = nxt+last.tolist()
    g = sum(.99**k*r['reward'] for k,r in enumerate(records))
    return {'model':'kda-s419-t24','controller':'model','geometry':'circle','condition':condition,
        'case':90000,'outcome':'timeout','return':g,'discounted_return':g,'undiscounted_return':-.5,
        'time':50.25,'path_length':float(path_length),'minimum_clearance':{
            f:min(r[f] for r in records) for f in ('endpoint','CV_swept','actual_swept')},
        'steps':records,'latency_ms_p50':0.,'latency_ms_p95':0.,'E3_shadow_exact':False,'override':override}


def synthetic_branch(condition='E2'):
    baseline = synthetic_episode(condition)
    row = baseline['steps'][20]
    index = V.a1(row)
    assert index is not None and index!=row['chosen']
    root = {'model':baseline['model'],'geometry':'circle','condition':condition,'case':90000,
        'step':20,'type':'timeout','snapshot':row,'A0':row['chosen'],'A1':index,
        'matched_control':None,'root_id':0}
    one = synthetic_episode(condition,{'step':20,'index':index})
    g0 = sum(.99**(i-20)*r['reward'] for i,r in enumerate(baseline['steps']) if i>=20)
    g1 = sum(.99**(i-20)*r['reward'] for i,r in enumerate(one['steps']) if i>=20)
    return baseline,root,{'root':root,'A0':copy.deepcopy(baseline),'A1':one,'root_G0':g0,'root_G1':g1,
        'delta_root_return':g1-g0,'delta_episode_discounted_return':one['return']-baseline['return'],
        'rescued':False,'destroyed':False,'A0_exact':True,'distinct_action':True}


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline,cls.root,cls.branch = synthetic_branch()

    def test_actual_actions_scores_and_G(self):
        V.check_branch(self.branch,self.root,self.baseline,('model','kda-s419-t24','circle','E2'))
        detail = C.branch_detail(self.branch,Path('synthetic-only-not-an-experiment'))
        self.assertEqual(detail['A0_index'],self.root['A0'])
        self.assertEqual(detail['A1_actual_command'],self.root['snapshot']['commands'][self.root['A1']])
        self.assertEqual(detail['delta_root_return'],0.)
        self.assertEqual(len(detail['all80']['scores']),80)

    def test_native10_raw_shape_without_padding(self):
        baseline,root,branch = synthetic_branch('E10')
        self.assertEqual(len(baseline['steps'][0]['humans']),10)
        V.check_branch(branch,root,baseline,('model','kda-s419-t24','circle','E10'))

    def test_corrupt_A0_history_scores_G_and_double_override(self):
        for change in ('A0-history','score','G','double-override'):
            branch = copy.deepcopy(self.branch)
            if change=='A0-history':
                branch['A0']['steps'][21]['history_sha256'] = 'corrupt'
            elif change=='score':
                branch['A1']['steps'][20]['raw_scores'][0] += .01
            elif change=='G':
                branch['root_G1'] += .01
            else:
                branch['A1']['steps'][21]['executed_override'] = branch['A1']['steps'][21]['chosen']
            with self.assertRaises(AssertionError,msg=change):
                V.check_branch(branch,self.root,self.baseline,('model','kda-s419-t24','circle','E2'))

    def test_no_root_failure_denominator_retained(self):
        ep = {'case':90000,'outcome':'collision','steps':[{'step':0,'n_safe':80}]}
        roots,denom = V.expected_roots([ep],'kda-s419-t24','circle','E2')
        self.assertEqual(roots,[])
        self.assertEqual(len(denom),1)
        self.assertFalse(denom[0]['has_root'])

    def test_strict_all_pair_calipers_and_no_replacement(self):
        def row(i,case,k):
            return {'root_id':i,'case':case,'step':k,'type':'timeout',
                'snapshot':{'step':k,'goal_distance':2.,'n_safe':40,'crowd2m':1}}
        a,b,c = [row(1,90000,20),row(2,90001,20)],[row(3,90000,28)],[row(4,90000,36)]
        self.assertEqual(len(C.strict_pairs(a,b)),1)
        self.assertEqual(C.strict_triples(a,b,c),[])
        c = [row(4,90000,20)]
        self.assertEqual(len(C.strict_triples(a,b,c)),1)
        self.assertEqual(C.strict_triples([],b,c),[])

    def test_remaining_sentinel(self):
        C.require_remaining({'status':'PASS','jobs':5})
        with self.assertRaises(AssertionError):
            C.require_remaining({'status':'RUNNING','jobs':5})


if __name__=='__main__':
    unittest.main(verbosity=2)
