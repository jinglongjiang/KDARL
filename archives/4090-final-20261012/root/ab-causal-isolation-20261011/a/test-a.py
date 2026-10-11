"""CPU contract/readout engineering tests, not research evidence."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=(HERE/'project/crowd_nav/contracts.py').read_text()
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_batch_joint34_to_tokens_vectorized')
        ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(HERE/'project/crowd_nav/contracts.py'),'exec'),ns)
        cls.old=staticmethod(ns[fn.name]);cls.contract=load('all_n_test',HERE/'all-n-contract.py')
        cls.new=staticmethod(cls.contract.extend_converter(cls.old)[0])

    def test_n5_bitwise(self):
        rng=np.random.RandomState(17)
        for dtype in (np.float32,np.float64):
            x=rng.normal(size=(80,34)).astype(dtype)
            x[0,9:]=0.;x[1,14:]=0.;x[:,7]=np.abs(x[:,7])
            np.testing.assert_array_equal(self.old(x),self.new(x,5))

    def test_full_n_fields_from_original(self):
        rng=np.random.RandomState(19)
        for n in (1,5,10,20):
            x=rng.normal(size=(2,9+5*n));expanded=self.new(x,n)
            self.assertEqual(expanded.shape,(2,3+n,13))
            for b in range(2):
                expected=[]
                for h in x[b,9:].reshape(n,5):
                    old=np.concatenate([x[b,:9],h,np.zeros(20)])[None]
                    token=self.old(old)[0]
                    np.testing.assert_array_equal(expanded[b,:3],token[:3])
                    expected.append(token[3])
                actual=expanded[b,3:]
                for token in expected:
                    self.assertTrue(any(np.array_equal(token,v) for v in actual))

    def test_left_repeat_windows(self):
        for length in (1,2,23,24,30):
            hist=np.arange(length*8*13,dtype=np.float32).reshape(length,8,13)
            nxt=np.zeros((80,8,13),np.float32)
            out=self.contract.candidate_windows(hist,nxt)
            expected=(list(hist)+[nxt[0]])[-24:]
            expected=[expected[0]]*(24-len(expected))+expected
            np.testing.assert_array_equal(out[0],np.asarray(expected))

    def test_readout_equal_budget_and_case_split(self):
        reader=load('reader_test',HERE/'readout-a.py')
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp);features=root/'features';features.mkdir()
            (features/'extract-done.json').write_text('{}')
            for rid,case in enumerate((90000,90001,90003,90004)):
                meta=dict(root_id=rid,model='synthetic-test-only',cell=5,case=case,split='train' if case%5<3 else 'dev' if case%5==3 else 'test',type='success-control',a0=0)
                (features/f'root-{rid}.json').write_text(json.dumps(meta))
                z=np.random.RandomState(rid).normal(size=(80,256)).astype(np.float32)
                arrays={'latent_'+h+n:z for h in ('pred','actual') for n in ('5','N')}
                arrays.update(indices=np.asarray([0,1]),G=np.asarray([.1,.8]),reward=np.zeros(2),terminal=np.zeros(2,bool),qpred=np.zeros(80),values=np.zeros(80),original_head_weight=np.zeros((1,256),np.float32),original_head_bias=np.zeros(1,np.float32))
                np.savez(features/f'root-{rid}.npz',**arrays)
            reader.main(SimpleNamespace(features=[features],output=root/'readout',steps=2,seed=17))
            result=json.loads((root/'readout/readout-summary.json').read_text())
            self.assertEqual(len(result['results']),8)
            for arm in result['results']:
                self.assertEqual(arm['parameters'],257);self.assertEqual(arm['optimizer_steps'],2)
                self.assertEqual(arm['train_candidate_labels'],4)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
