"""N10 extension integration checks, CPU only; no episodes or model loading."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_n10',ROOT/'b10.py')
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)


class Checks(unittest.TestCase):
    def test_frozen_parent_untouched(self):
        self.assertEqual(X.B.sha(ROOT/'run-b.py'),X.EXT['parent_driver_sha256'])
        self.assertEqual(X.B.sha(ROOT/'b-contract.json'),X.EXT['parent_contract_sha256'])
        self.assertEqual(X.B.P['conditions'],['E10'])
        self.assertEqual(X.B.P['cases'],list(range(90000,90016)))
        self.assertEqual(X.B.P['started_unix'],X.PARENT_P['started_unix'])
        self.assertEqual(X.B.P['continuation'],X.PARENT_P['continuation'])
        self.assertEqual(X.B.P['decisions'],X.PARENT_P['decisions'])

    def test_job_counts_no_launch(self):
        for phase,n in [('smoke',36),('main',192)]:
            out = subprocess.check_output([sys.executable,str(ROOT/'b10.py'),'jobs','--phase',phase],text=True)
            data = json.loads(out)
            self.assertEqual(data['count'],12)
            self.assertEqual(data['count']*data['episodes_per_job'],n)
            self.assertFalse(data['starts_jobs'])

    def test_cpu_saved_all_original_reset_pairs(self):
        data = json.loads((ROOT/'n10/work/cpu-main.json').read_text())
        self.assertEqual(data['status'],'PASS')
        self.assertEqual(data['cases'],32)
        self.assertFalse(data['learned_policy_evaluation'])
        self.assertFalse(data['old_outcomes_consumed'])
        self.assertEqual(data['provenance']['extension_script_sha256'],X.B.sha(ROOT/'b10.py'))
        self.assertTrue(all(c['native_original_initial_exact'] for c in data['checks']))
        self.assertTrue(all(len(c['humans'])==10 and c['old_conditions_forwarded_exact']==['E1','E2','E3','E4','E5'] for c in data['checks']))

    def test_matching_strict_without_replacement(self):
        spec = importlib.util.spec_from_file_location('opportunity_matcher',ROOT/'match-b10.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        def row(i,case,step):
            return {'root_id':i,'case':case,'step':step,'type':'timeout',
                'snapshot':{'step':step,'goal_distance':2,'n_safe':40,'crowd2m':1}}
        a,b,c = [row(1,90000,20),row(2,90001,20)],[row(3,90000,20)],[row(4,90000,20)]
        self.assertEqual(len(m.triples(a,b,c)),1)
        self.assertTrue(m.triples(a,b,c)[0]['all_same_case'])
        self.assertEqual(m.triples(a,b,[row(5,90000,29)]),[])


if __name__=='__main__':
    unittest.main(verbosity=2)
