#!/usr/bin/env python3
"""Independent CPU-only raw acceptance of B continuations and native N10.

Imports only stdlib/NumPy and the independent verifier, never run-b, b10, torch,
models or environment. Prints JSON to stdout; caller may archive that output.
Gate before reading outputs: original remaining/done.json PASS,jobs5; native
N10 cross-N-matches.json plus all8 branch done files (supplemental may still run).
Use --scope original first, then --scope all --original-verified previous.json
to reuse exact SHA-pinned original acceptance without repeating its raw scoring.
No sampling/threshold changes. Zero low-N failure-root matches do NOT refute
generalization loss. Stored matching is independently regenerated and checked.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('independent_b_raw',ROOT/'verify-b-raw.py')
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
N10_SCRIPT_SHA = 'b6ea238ca05ab306a7915918bb50293a633ba61b2276bd89485c3b67e3f54d6c'
N10_CONTRACT_SHA = 'c09320fd15918324f9fba60959a12ea5b147c1075b62c459de1aef82cda36cdf'
PARENT_CONTRACT_SHA = V.EXPECTED_CONTRACT_SHA


def require_remaining(data):
    V.exact(data['status'],'PASS','remaining sentinel status')
    V.exact(data['jobs'],5,'remaining sentinel job count')


def gate(args):
    result = {'status':'PASS','files':[],'missing':[],'errors':[]}
    for folder in ('remaining','supplemental'):
        failed = args.execution/folder/'failure.json'
        if failed.exists() and (folder=='remaining' or args.scope!='original'):
            result['errors'].append({'file':str(failed),'content':V.read_json(failed)})
    try:
        path = args.execution/'remaining/done.json'
        require_remaining(V.read_json(path))
        result['files'].append({'file':str(path),'sha256':V.sha(path)})
        if args.scope=='all':
            expected = [args.n10_work/'cross-N-matches.json']
            parent = V.read_json(args.contract)
            expected += [args.n10_work/'branches'/m/(g+'-E10.jsonl.done.json')
                for m in parent['models'] for g in parent['geometries']]
            for path in expected:
                if not path.is_file():
                    result['missing'].append(str(path))
                else:
                    result['files'].append({'file':str(path),'sha256':V.sha(path)})
                    if path.name.endswith('.done.json'):
                        V.exact(V.read_json(path)['status'],'PASS','N10 branch done status')
            path = args.execution/'supplemental/done.json'
            if path.exists():
                data = V.read_json(path)
                V.exact(data['status'],'PASS','supplemental sentinel')
                V.exact(data['full_training_started'],False,'unexpected full training')
                result['files'].append({'file':str(path),'sha256':V.sha(path)})
    except FileNotFoundError as error:
        result['missing'].append(str(error))
    except (AssertionError,ValueError,KeyError) as error:
        result['errors'].append(str(error))
    result['status'] = 'FAIL' if result['errors'] else 'INCOMPLETE' if result['missing'] else 'PASS'
    return result


def episode_metrics(ep):
    return {k:ep[k] for k in ('case','outcome','return','discounted_return','undiscounted_return',
        'time','path_length','minimum_clearance','latency_ms_p50','latency_ms_p95')}


def branch_detail(record,path):
    root = record['root']
    k,i,j = root['step'],root['A0'],root['A1']
    zero,one = record['A0'],record['A1']
    row = zero['steps'][k]
    resumed = one['steps'][k+1] if k+1<len(one['steps']) else None
    return {'model':root['model'],'geometry':root['geometry'],'condition':root['condition'],
        'root_id':root['root_id'],'case':root['case'],'step':k,'time_seconds':k*.25,'type':root['type'],
        'A0_index':i,'A1_index':j,'A0_actual_command':zero['steps'][k]['command'],
        'A1_actual_command':one['steps'][k]['command'],
        'raw_scores_A0_A1':[row['raw_scores'][i],row['raw_scores'][j]],
        'filtered_scores_A0_A1':[row['scores'][i],row['scores'][j]],
        'values_A0_A1':[row['values'][i],row['values'][j]],
        'immediate_rewards_A0_A1':[row['rewards'][i],row['rewards'][j]],
        'clearance_A0_A1':[row['clearance'][i],row['clearance'][j]],
        'progress_A0_A1':[row['progress'][i],row['progress'][j]],
        'root_goal_distance':row['goal_distance'],'root_safe_count':row['n_safe'],'root_crowd2m':row['crowd2m'],
        'token_sha256':row['token_sha256'],'history_sha256':row['history_sha256'],
        'all80':{f:row[f] for f in ('commands','values','rewards','clearance','progress','raw_scores','scores')},
        'root_G0':record['root_G0'],'root_G1':record['root_G1'],
        'delta_root_return':record['delta_root_return'],
        'delta_episode_discounted_return':record['delta_episode_discounted_return'],
        'rescued':record['rescued'],'destroyed':record['destroyed'],'A0_full_raw_exact':True,
        'single_A1_override_checked':True,'resumed_policy_checked':True,
        'next_A1_branch_chosen_index':resumed['chosen'] if resumed else None,
        'next_A1_branch_commands_sha256':V.byte_hash(np.asarray(resumed['commands'])) if resumed else None,
        'matched_control':root.get('matched_control'),'matched_failure':root.get('matched_failure'),
        'A0':episode_metrics(zero),'A1':episode_metrics(one),'raw_file':str(path),
        'raw_lookup':{'root_id':root['root_id'],'episode_step':k},
        'matched_control_protection':'PENDING_GROUP_LINKAGE' if root['type']!='success-control' else 'SUCCESS_CONTROL'}


class DetailedAudit(V.Audit):
    def __init__(self,work,contract,manifest=None,native=False,extension=None):
        super().__init__(work,contract,manifest)
        self.native = native
        self.result['root_details'] = []
        self.result['root_artifacts'] = []
        if native:
            self.extension = extension
            V.exact(V.sha(extension),N10_CONTRACT_SHA,'N10 protocol changed')
            ext = V.read_json(extension)
            V.exact(ext['parent_driver_sha256'],V.EXPECTED_DRIVER_SHA,'N10 parent driver')
            V.exact(ext['parent_contract_sha256'],V.EXPECTED_CONTRACT_SHA,'N10 parent contract')
            self.p.update(ext['overrides'])
            self.contract = extension
            self.result.update(scope='native_N10_baseline_and_continuations',contract_sha256=V.sha(extension),
                expected={'model_episodes':128,'reference_episodes':64,'total_episodes':192,'main_shards':12},
                unfulfilled_scientific_requirements=['Strict matched opportunities and validated protected effects still required.',
                    'Custom CV-MPC does not fulfill the mature second-reference requirement.'])
        self.cpu_initials = {}

    def check_native_preflight(self):
        path = self.work/'cpu-main.json'
        data = V.read_json(path)
        V.exact(data['status'],'PASS','N10 CPU qualification')
        V.exact(data['cases'],32,'N10 CPU case count')
        V.exact(data['provenance']['contract_sha256'],N10_CONTRACT_SHA,'N10 CPU protocol')
        V.exact(data['provenance']['extension_script_sha256'],N10_SCRIPT_SHA,'N10 CPU script')
        V.exact(data['learned_policy_evaluation'],False,'N10 CPU cannot be model inference')
        V.exact(data['old_outcomes_consumed'],False,'old10S outcomes consumed')
        V.exact(sorted((r['geometry'],r['case']) for r in data['checks']),
            sorted((g,c) for g in self.p['geometries'] for c in self.p['cases']),'native initial case coverage')
        for row in data['checks']:
            V.exact(row['native_original_initial_exact'],True,'N10 original reset equality')
            V.exact(row['old_conditions_forwarded_exact'],['E1','E2','E3','E4','E5'],'old scene forwarding')
            V.require(len(row['humans'])==10,'N10 initial population')
            self.cpu_initials[row['geometry'],row['case']] = row
        gatepath = self.work/'smoke-accepted.json'
        gate = V.read_json(gatepath)
        V.exact(gate['status'],'PASS','N10 smoke gate')
        V.exact(gate['episodes'],36,'N10 smoke count')
        V.exact(gate['contract_sha256'],N10_CONTRACT_SHA,'N10 smoke protocol')
        V.exact(gate['cpu_qualification_sha256'],V.sha(path),'N10 CPU gate hash')
        count,files = 0,[]
        for controller in ['model']+self.p['references']:
            models = self.p['models'] if controller=='model' else [self.p['models'][0]]
            for model in models:
                for geo in self.p['geometries']:
                    raw = self.work/'smoke'/controller/model/(geo+'-E10.jsonl.gz')
                    done,start = V.read_json(raw.with_suffix('.done.json')),V.read_json(raw.with_suffix('.start.json'))
                    V.exact(done['status'],'PASS','N10 smoke done')
                    V.exact(done['output_sha256'],V.sha(raw),'N10 smoke raw hash')
                    V.exact(done['provenance'],start,'N10 smoke start/done')
                    self.native_provenance(start,'smoke',controller,model)
                    cases = []
                    for ep in V.rows(raw):
                        V.check_episode(ep,(controller,model,geo,'E10'))
                        self.check_native_initial(ep)
                        cases.append(ep['case'])
                    V.exact(cases,self.p['smoke_cases'],'N10 smoke fixed3 cases')
                    V.exact(done['episodes'],3,'N10 smoke count')
                    count += len(cases)
                    files.append({'file':str(raw),'sha256':done['output_sha256']})
        self.result['smoke_acceptance'] = {'status':'PASS','episodes':count,'files':files,
            'cpu_qualification':{'file':str(path),'sha256':V.sha(path)},
            'gate':{'file':str(gatepath),'sha256':V.sha(gatepath)}}

    def native_provenance(self,start,phase,controller,model):
        V.exact(start['contract_sha256'],N10_CONTRACT_SHA,'N10 raw contract')
        V.exact(start['parent_contract_sha256'],PARENT_CONTRACT_SHA,'N10 raw parent contract')
        V.exact(start['script_sha256'],V.EXPECTED_DRIVER_SHA,'N10 inherited driver')
        V.exact(start['extension_script_sha256'],N10_SCRIPT_SHA,'N10 wrapper script')
        V.exact(start['extension'],'native-N10-original-B5-completion','N10 provenance label')
        V.exact(start['phase'],phase,'N10 phase')
        V.exact(start['controller'],controller,'N10 controller')
        V.exact(start['model']['id'],model,'N10 model')
        for field in ('status','history_test','terminal_test'):
            V.exact(start['acceptance'][field],'PASS','N10 public acceptance '+field)
        V.exact(start['acceptance']['tested_commit'],'9f06223','N10 accepted revision')
        if self.manifest:
            original = next(m for m in self.manifest['protocol']['models'] if m['id']==model)
            for field in ('weight_sha256','config_sha256','backbone','seed'):
                V.exact(start['model'][field],original[field],'N10 original manifest '+field)

    def check_native_initial(self,ep):
        initial = self.cpu_initials[ep['geometry'],ep['case']]
        V.exact(ep['steps'][0]['humans'],initial['humans'],'N10 raw native initial humans')
        V.exact(ep['steps'][0]['robot'],initial['robot'],'N10 raw native initial robot')

    def load_group(self,key):
        super().load_group(key)
        if self.native:
            for path in self.paths[key]:
                start = V.read_json(path.with_suffix('.start.json'))
                self.native_provenance(start,'main',key[0],key[1])
            for ep in self.raw_group(key):
                self.check_native_initial(ep)

    def check_roots(self,model):
        data = super().check_roots(model)
        path = self.work/('roots-'+model+'.json')
        self.result['root_artifacts'].append({'model':model,'file':str(path),'sha256':V.sha(path),
            'all_failure_denominator':data['all_failure_denominator']})
        return data

    def check_branches(self,model,geometry,condition,data,freeze_sha):
        super().check_branches(model,geometry,condition,data,freeze_sha)
        path = self.work/'branches'/model/(geometry+'-'+condition+'.jsonl.gz')
        done = V.read_json(path.with_suffix('.done.json'))
        if self.manifest:
            original = next(m for m in self.manifest['protocol']['models'] if m['id']==model)
            V.exact(done['weight_sha256'],original['weight_sha256'],'branch original checkpoint hash')
        details = [branch_detail(r,path) for r in V.rows(path)]
        failure_denominator = [d for d in data['all_failure_denominator'] if
            d['geometry']==geometry and d['condition']==condition]
        failure_details = [r for r in details if r['type']!='success-control']
        controls = [r for r in details if r['type']=='success-control']
        pairs = []
        for root in failure_details:
            matches = [r for r in controls if r['matched_failure']==root['case']]
            V.require(len(matches)<=1,'duplicate matched success control')
            control = matches[0] if matches else None
            if control:
                V.exact(root['matched_control'],[control['case'],control['step']],'success protection link')
                root['matched_control_protection'] = 'DESTROYED' if control['destroyed'] else 'VALIDATED_PRESERVED'
            else:
                V.exact(root['matched_control'],None,'missing linked success control')
                root['matched_control_protection'] = 'UNMATCHED_NO_PROTECTION'
            pairs.append({'failure_root':root['root_id'],'failure_case':root['case'],
                'success_control_root':control['root_id'] if control else None,
                'success_control_case':control['case'] if control else None,
                'failure_rescued':root['rescued'],'failure_delta_root_return':root['delta_root_return'],
                'control_destroyed':control['destroyed'] if control else None,
                'control_delta_root_return':control['delta_root_return'] if control else None,
                'status':root['matched_control_protection']})
        group = self.result['branches'][-1]
        group.update(raw_file=str(path),root_freeze_file=str(self.work/('roots-'+model+'.json')),
            root_freeze_sha256=freeze_sha,all_failed_episodes=len(failure_denominator),
            no_eligible_root=sum(not d['has_root'] for d in failure_denominator),
            eligible_but_outside_cap=sum(d['has_root'] and not d['selected_within_cap'] for d in failure_denominator),
            selected_failure_cases=[r['case'] for r in failure_details],
            success_control_cases=[r['case'] for r in controls],failure_control_pairs=pairs,
            protected_rescues=sum(r['rescued'] and r['matched_control_protection']=='VALIDATED_PRESERVED' for r in failure_details),
            full_failure_denominator=failure_denominator)
        self.result['root_details'].extend(details)

    def run(self):
        if not self.native:
            result = super().run()
        else:
            old = V.EXPECTED_CONTRACT_SHA
            try:
                # Inherited independent parser has one contract SHA guard. Keep
                # it scoped to native raw; parent driver/manifest guards stay on.
                self.attempt('N10 CPU/smoke qualification',self.check_native_preflight)
                V.EXPECTED_CONTRACT_SHA = N10_CONTRACT_SHA
                for controller in ['model']+self.p['references']:
                    models = self.p['models'] if controller=='model' else [self.p['models'][0]]
                    for model in models:
                        for geo in self.p['geometries']:
                            key = controller,model,geo,'E10'
                            self.attempt('N10 main '+str(key),lambda k=key:self.load_group(k))
                for model in self.p['models']:
                    self.attempt('N10 roots '+model,lambda m=model:self.check_roots(m))
            finally:
                V.EXPECTED_CONTRACT_SHA = old
            if self.result['errors']:
                self.result['execution_status'] = 'FAIL'
            elif not self.result['missing']:
                for field,value in self.result['expected'].items():
                    V.exact(self.result['verified'][field],value,'N10 formal '+field)
                V.exact(len(self.result['branches']),8,'N10 all8 branch shards')
                V.exact(self.result['smoke_acceptance']['episodes'],36,'N10 all36 smoke')
                self.result['execution_status'] = 'PASS'
            self.result['strict_opportunity_matches'] = []
            result = self.result
        result['raw_branch_shards_verified'] = len(result['branches'])
        result['raw_continuation_roots_verified'] = len(result['root_details'])
        result['continuation_episodes_including_A0_A1'] = 2*len(result['root_details'])
        result['branch_root_evidence'] = 'Actual commands/scores/G/outcomes are in root_details; all80 also retained.'
        return result


def frozen_levels(original,native,model,geo):
    roots,denominators = [],[]
    for source,condition in [(original,'E1'),(native,'E10'),(original,'E2')]:
        artifact = next(r for r in source['root_artifacts'] if r['model']==model)
        data = V.read_json(Path(artifact['file']))
        V.exact(V.sha(Path(artifact['file'])),artifact['sha256'],'matching roots changed after acceptance')
        roots.append([r for r in data['roots'] if r['geometry']==geo and r['condition']==condition and r['type']!='success-control'])
        denominators.append([d for d in artifact['all_failure_denominator'] if d['geometry']==geo and d['condition']==condition])
    return roots,denominators


def strict_distance(a,b):
    return V.distance(a['snapshot'],b['snapshot']) if a['type']==b['type'] else None


def strict_pairs(left,right):
    result,unused = [],{r['root_id'] for r in right}
    for a in left:
        candidates = [(a['case']!=b['case'],strict_distance(a,b),b) for b in right if b['root_id'] in unused]
        candidates = [c for c in candidates if c[1] is not None]
        if candidates:
            _,dist,b = min(candidates,key=lambda x:(x[0],x[1][0],x[2]['case'],x[2]['step']))
            unused.remove(b['root_id'])
            result.append({'A':a['root_id'],'B':b['root_id'],'same_case':a['case']==b['case'],
                'standardized_differences':dist[1]})
    return result


def strict_triples(low,mid,high):
    result,usedmid,usedhigh = [],set(),set()
    for a in low:
        options = []
        for b in mid:
            if b['root_id'] in usedmid:
                continue
            ab = strict_distance(a,b)
            if ab is None:
                continue
            for c in high:
                if c['root_id'] in usedhigh:
                    continue
                ac,bc = strict_distance(a,c),strict_distance(b,c)
                if ac is not None and bc is not None:
                    ds = [ab[1],ac[1],bc[1]]
                    options.append((a['case']==b['case']==c['case'],sum(map(sum,ds)),b,c,ds))
        if options:
            same,_,b,c,ds = min(options,key=lambda x:(not x[0],x[1],x[2]['case'],x[2]['step'],x[3]['case'],x[3]['step']))
            usedmid.add(b['root_id'])
            usedhigh.add(c['root_id'])
            result.append({'N5_root':a['root_id'],'N10_root':b['root_id'],'N20_root':c['root_id'],
                'all_same_case':same,'all_pair_standardized_differences':ds})
    return result


def match_cross_N(original,native,work):
    stored_path = work/'cross-N-matches.json'
    stored = V.read_json(stored_path)
    V.exact(stored['execution_status'],'PASS','stored N matching execution')
    V.exact(stored['errors'],[],'stored matching errors')
    V.exact(stored['missing'],[],'stored matching missing')
    parent = V.read_json(ROOT/'b-contract.json')
    expected,details = [],[]
    for model in parent['models']:
        for geo in parent['geometries']:
            levels,denominators = frozen_levels(original,native,model,geo)
            a,b,c = levels
            pairs = {'N5_N10':strict_pairs(a,b),'N10_N20':strict_pairs(b,c),'N5_N20':strict_pairs(a,c)}
            ts = strict_triples(a,b,c)
            expected.append({'model':model,'geometry':geo,**pairs,'strict_triples':ts,
                'selected_failure_roots_5_10_20':list(map(len,levels)),
                'all_failure_denominators_5_10_20':list(map(len,denominators)),
                'status':'MATCHED_OPPORTUNITIES_NOT_CAUSAL_EFFECTS' if ts else 'INCOMPLETE_NO_STRICT_TRIPLES'})
            branches = []
            for source,condition in [(original,'E1'),(native,'E10'),(original,'E2')]:
                branches.append({r['root_id']:r for r in source['root_details'] if r['model']==model and
                    r['geometry']==geo and r['condition']==condition and r['type']!='success-control'})
            paired_effects = {}
            for label,li,ri in [('N5_N10',0,1),('N10_N20',1,2),('N5_N20',0,2)]:
                items = []
                for match in pairs[label]:
                    l,r = branches[li][match['A']],branches[ri][match['B']]
                    items.append({'match':match,'case_A':l['case'],'case_B':r['case'],
                        'A0_A1_indices_A':[l['A0_index'],l['A1_index']],'A0_A1_indices_B':[r['A0_index'],r['A1_index']],
                        'root_G0_G1_A':[l['root_G0'],l['root_G1']],'root_G0_G1_B':[r['root_G0'],r['root_G1']],
                        'delta_root_return_A':l['delta_root_return'],'delta_root_return_B':r['delta_root_return'],
                        'delta_of_intervention_returns_B_minus_A':r['delta_root_return']-l['delta_root_return'],
                        'rescued_A_B':[l['rescued'],r['rescued']],
                        'success_protection_A_B':[l['matched_control_protection'],r['matched_control_protection']]})
                paired_effects[label] = items
            triple_effects = []
            for triple in ts:
                rs = [branches[i][triple[f]] for i,f in enumerate(('N5_root','N10_root','N20_root'))]
                triple_effects.append({'match':triple,'cases':[r['case'] for r in rs],
                    'delta_root_returns':[r['delta_root_return'] for r in rs],
                    'rescued':[r['rescued'] for r in rs],
                    'success_protection':[r['matched_control_protection'] for r in rs]})
            details.append({'model':model,'geometry':geo,
                'all_failures_5_10_20':list(map(len,denominators)),
                'eligible_failures_5_10_20':[sum(d['has_root'] for d in ds) for ds in denominators],
                'outside_cap_5_10_20':[sum(d['has_root'] and not d['selected_within_cap'] for d in ds) for ds in denominators],
                'selected_failure_roots_5_10_20':list(map(len,levels)),
                'matched_success_controls_5_10_20':[sum(r['matched_control_protection']!='UNMATCHED_NO_PROTECTION' for r in bs.values()) for bs in branches],
                'strict_pair_counts':{k:len(v) for k,v in pairs.items()},'strict_triple_count':len(ts),
                'N10_roots_without_N5_match':len(b)-len(pairs['N5_N10']),
                'N20_roots_without_N5_match':len(c)-len(pairs['N5_N20']),
                'paired_effects':paired_effects,'triple_effects':triple_effects,
                'identifiability_status':'STRICT_OPPORTUNITIES_AVAILABLE_NOT_CAUSAL_PROOF' if ts else 'INCOMPLETE_NO_STRICT_TRIPLES',
                'interpretation':'No low-N failure-root match does not imply no generalization loss; this is failure-conditioned, capped root sampling.'})
    V.exact(stored['groups'],expected,'stored cross-N matching differs from independent regeneration')
    return {'execution_status':'PASS','stored_match_file':str(stored_path),'stored_match_sha256':V.sha(stored_path),
        'groups':details,'strict_triples_total':sum(g['strict_triple_count'] for g in details),
        'strict_pair_totals':{k:sum(g['strict_pair_counts'][k] for g in details) for k in ('N5_N10','N10_N20','N5_N20')},
        'scientific_status':'INCOMPLETE_NO_STRICT_TRIPLES' if not any(g['strict_triple_count'] for g in details) else
            'INCONCLUSIVE_MATCHING_ALONE_NOT_IDENTIFICATION'}


def reuse_original(path):
    cached = V.read_json(path)
    original = cached['original']
    V.exact(cached['execution_status'],'PASS','cached original execution')
    V.exact(original['execution_status'],'PASS','cached original raw execution')
    V.exact(original['verified']['total_episodes'],960,'cached original formal count')
    V.exact(len(original['branches']),40,'cached original all40 branches')
    V.exact(original['contract_sha256'],V.EXPECTED_CONTRACT_SHA,'cached original contract')
    for group in original['shards']:
        for entry in group['files']:
            V.exact(V.sha(Path(entry['file'])),entry['sha256'],'cached baseline bytes changed')
    for group in original['branches']:
        V.exact(V.sha(Path(group['raw_file'])),group['raw_sha256'],'cached branch bytes changed')
        V.exact(V.sha(Path(group['root_freeze_file'])),group['root_freeze_sha256'],'cached roots changed')
    return original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=['original','all'],default='original')
    p.add_argument('--work',type=Path,default=ROOT/'work')
    p.add_argument('--n10-work',type=Path,default=ROOT/'n10/work')
    p.add_argument('--contract',type=Path,default=ROOT/'b-contract.json')
    p.add_argument('--extension',type=Path,default=ROOT/'b10-protocol.json')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--execution',type=Path,default=ROOT.parent/'execution')
    p.add_argument('--original-verified',type=Path)
    args = p.parse_args()
    result = {'execution_status':'INCOMPLETE','scientific_status':'INCOMPLETE','scope':args.scope,
        'read_only':True,'model_imported':False,'rollouts_run':False,'thresholds_modified':False,
        'auditor_sha256':V.sha(SCRIPT),'independent_raw_helper_sha256':V.sha(ROOT/'verify-b-raw.py'),
        'method_level_progress':False,'B4_full_training_started':False}
    try:
        result['gate'] = gate(args)
        if result['gate']['status']!='PASS':
            result['execution_status'] = result['gate']['status']
        else:
            result['original'] = reuse_original(args.original_verified) if args.original_verified else \
                DetailedAudit(args.work,args.contract,args.manifest).run()
            if result['original']['execution_status']!='PASS':
                result['execution_status'] = result['original']['execution_status']
            elif args.scope=='original':
                result['execution_status'] = 'PASS'
                result['scientific_status'] = 'INCOMPLETE_PENDING_NATIVE_N10_AND_REFERENCE'
            else:
                result['native_N10'] = DetailedAudit(args.n10_work,args.contract,args.manifest,
                    native=True,extension=args.extension).run()
                result['execution_status'] = result['native_N10']['execution_status']
                if result['execution_status']=='PASS':
                    result['cross_N'] = match_cross_N(result['original'],result['native_N10'],args.n10_work)
                    result['scientific_status'] = result['cross_N']['scientific_status']
    except (AssertionError,ValueError,KeyError,TypeError,OSError) as error:
        result['execution_status'] = 'FAIL'
        result['error'] = str(error)
    print(json.dumps(result,allow_nan=False,indent=2))
    return 0 if result['execution_status']=='PASS' else 2 if result['execution_status']=='INCOMPLETE' else 1


if __name__=='__main__':
    sys.exit(main())
