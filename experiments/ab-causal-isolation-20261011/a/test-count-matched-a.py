"""CPU engineering tests for the fixed token-count negative control."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('count_test',HERE/'count-matched-a.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


class Tensor:
    def __init__(self,a):self.a=a
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


class CountControl(unittest.TestCase):
    def test_source_exact_and_only_two_constructions_changed(self):
        path=HERE/'closedloop-a.py'
        self.assertEqual(C.sha(path),C.UPSTREAM_SHA)
        original=ast.parse(path.read_text());changed,names=C.transformed_tree(path.read_text())
        self.assertEqual(sorted(names),['current','full_windows'])
        compile(changed,str(path),'exec')
        class Normalize(ast.NodeTransformer):
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('current','full_windows'):
                    node.value=ast.Constant(value='ONLY_ALLOWED_CONSTRUCTION_CHANGE')
                return self.generic_visit(node)
        self.assertEqual(ast.dump(Normalize().visit(original)),ast.dump(Normalize().visit(copy.deepcopy(changed))))
        self.assertIn('N5 windows control',path.read_text())
        self.assertIn('N5 values control',path.read_text())
        self.assertIn('N5 final score control',path.read_text())

    def test_repeat_each_human_preserve_all_frames_candidates_and_robot(self):
        rng=np.random.RandomState(17);parent=rng.normal(size=(80,24,8,13)).astype(np.float32)
        original=parent.copy();repeated=C.duplicate_tokens(parent)
        self.assertEqual(repeated.shape,(80,24,23,13))
        np.testing.assert_array_equal(repeated[...,:3,:],parent[...,:3,:])
        for h in range(5):
            for repeat in range(4):np.testing.assert_array_equal(repeated[...,3+4*h+repeat,:],parent[...,3+h,:])
        np.testing.assert_array_equal(parent,original)
        out=dict(windows=Tensor(parent),current_token=parent[0,-2])
        def forbidden():raise AssertionError('Must not construct legal-allN window')
        np.testing.assert_array_equal(C.count_windows(out,20,True,forbidden),repeated)
        np.testing.assert_array_equal(C.count_current(out,np.ones((23,13)),20,True),C.duplicate_tokens(out['current_token']))

    def test_n5_and_baseline_identity_no_gate_bypass(self):
        original=np.arange(104,dtype=np.float32).reshape(8,13);out=dict(current_token=original)
        self.assertIs(C.count_current(out,original,5,True),original)
        self.assertIs(C.count_current(out,original,20,False),original)
        calls=[]
        def constructor():calls.append(1);return original
        self.assertIs(C.count_windows(out,5,True,constructor),original)
        self.assertIs(C.count_windows(out,20,False,constructor),original)
        self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
