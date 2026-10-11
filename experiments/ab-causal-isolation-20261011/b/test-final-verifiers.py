#!/usr/bin/env python3
"""Synthetic checks for final read-only gate/reference acceptance helpers."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G=load('verify-b-gates')
M=load('verify-mamba-reference')


def group(seed,geo='square',unmatched=0):
    return dict(model='kda-s%d-t24'%seed,geometry=geo,condition='E2',
        failure_roots=2,success_controls=2-unmatched,unmatched_failure_roots=unmatched,
        rescues=1,success_destructions=0,failure_delta_root_returns=[.2,.1],
        control_delta_root_returns=[-.1]*(2-unmatched))


def evidence():
    return dict(execution_status='PASS',original=dict(execution_status='PASS',
        verified=dict(total_episodes=960),E1_E3_pairs=[{}]*12,
        branches=[group(419),group(443)],excess_transfer_baseline=[dict(model='kda-s%d-t24'%seed,
            geometry=geo,A='E1',B='E2',against_references=[],
            both_reference_baseline_threshold_met=False)
            for seed in (419,443) for geo in ('circle','square')]))


class FinalVerifiers(unittest.TestCase):
    def test_A1_pass_does_not_overwrite_excess_failure(self):
        r=G.adjudicate(evidence())
        self.assertEqual(r['A1']['by_geometry']['square']['A1_numeric_gate'],'PASS')
        self.assertEqual(r['excess_transfer']['baseline_necessary_gate'],'FAIL')

    def test_any_unmatched_root_disqualifies_whole_condition(self):
        e=evidence()
        e['original']['branches'][0]=group(419,unmatched=1)
        r=G.adjudicate(e)
        self.assertEqual(len(r['A1']['disqualified_conditions']),1)
        self.assertEqual(r['A1']['by_geometry']['square']['rescues'],1)
        self.assertEqual(r['A1']['by_geometry']['square']['A1_numeric_gate'],'FAIL')

    def test_success_destruction_or_nonpositive_G_fails(self):
        e=evidence()
        e['original']['branches'][0]['success_destructions']=1
        self.assertEqual(G.adjudicate(e)['A1']['by_geometry']['square']['A1_numeric_gate'],'FAIL')
        e=evidence()
        for g in e['original']['branches']:
            g['failure_delta_root_returns']=[0.,0.]
        self.assertEqual(G.adjudicate(e)['A1']['by_geometry']['square']['A1_numeric_gate'],'FAIL')

    def test_duplicate_E3_not_independent_gate_evidence(self):
        e=evidence()
        duplicate=copy.deepcopy(e['original']['branches'][0])
        duplicate['condition']='E3'
        e['original']['branches'].append(duplicate)
        r=G.adjudicate(e)
        self.assertEqual(r['A1']['all_eligible_conditions_descriptive']['rescues'],3)
        self.assertEqual(r['A1']['excluding_duplicate_E3']['rescues'],2)

    def test_reference_requires_accepted_source_and_checkpoint(self):
        start=dict(script_sha256=M.V.EXPECTED_DRIVER_SHA,
            contract_sha256=M.V.EXPECTED_CONTRACT_SHA,phase='main',controller='model',
            acceptance=dict(status='PASS',history_test='PASS',terminal_test='PASS',tested_commit='9f06223'),
            model=dict(id='mamba-reference',seed=419,backbone='mamba',weight_sha256=M.CHECKPOINT,
                config_sha256='config'),source_sha256={'policy':'accepted'})
        M.validate_start(start,dict(config_sha256='config'),'main',{'policy':'accepted'})
        start['source_sha256']['policy']='old'
        with self.assertRaises(AssertionError):
            M.validate_start(start,dict(config_sha256='config'),'main',{'policy':'accepted'})

    def test_reference_summary_derived_data_not_authority(self):
        M.check_summary_group(dict(success=10,return_=.3),dict(geometry='circle',condition='E2',success=10))
        with self.assertRaises(AssertionError):
            M.check_summary_group(dict(success=9),dict(success=10))

    def test_missing_reference_is_incomplete_not_pass(self):
        from types import SimpleNamespace
        args=SimpleNamespace(reference=HERE/'nonexistent-reference',source=HERE/'nonexistent-source')
        r=M.audit(args)
        self.assertEqual(r['execution_status'],'INCOMPLETE')
        self.assertTrue(r['missing'])
        self.assertFalse(r['model_imported'])


if __name__=='__main__':
    unittest.main()
