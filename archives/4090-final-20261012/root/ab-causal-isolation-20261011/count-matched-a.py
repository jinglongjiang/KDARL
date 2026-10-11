"""Main-scheduled repeated-TTC5 token-count control; no extra crowd information."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
UPSTREAM_SHA='a0db37797b06d704087f45cbc719eb35307624e9eeb42500ba5a39b912abfad3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duplicate_tokens(tokens):
    a=np.asarray(tokens)
    assert a.shape[-2:]==(8,13)
    return np.concatenate((a[...,:3,:],np.repeat(a[...,3:,:],4,axis=-2)),axis=-2)


def count_current(out, original, n, is_full):
    if not is_full or n==5:return original
    assert n==20
    return duplicate_tokens(out['current_token'])


def count_windows(out, n, is_full, original):
    if not is_full or n==5:return original()
    assert n==20
    windows=out['windows'].detach().cpu().numpy()
    assert windows.shape==(80,24,8,13)
    result=duplicate_tokens(windows)
    assert result.shape==(80,24,23,13)
    return result


def transformed_tree(source):
    tree=ast.parse(source);changes=[]
    class Replace(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):return self.generic_visit(node)
            name=node.targets[0].id
            if name=='current' and isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Call) and isinstance(node.value.value.func,ast.Name) and node.value.value.func.id=='all_tokens':
                node.value=ast.Call(func=ast.Name(id='count_current',ctx=ast.Load()),args=[ast.Name(id='out',ctx=ast.Load()),node.value,
                    ast.Name(id='N',ctx=ast.Load()),ast.Name(id='is_full',ctx=ast.Load())],keywords=[])
                changes.append('current')
            elif name=='full_windows' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='candidate_windows':
                original=ast.Lambda(args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=node.value)
                node.value=ast.Call(func=ast.Name(id='count_windows',ctx=ast.Load()),args=[ast.Name(id='out',ctx=ast.Load()),
                    ast.Name(id='N',ctx=ast.Load()),ast.Name(id='is_full',ctx=ast.Load()),original],keywords=[])
                changes.append('full_windows')
            return self.generic_visit(node)
    tree=Replace().visit(tree);assert sorted(changes)==['current','full_windows'],changes
    return ast.fix_missing_locations(tree),changes


def rows(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def main(args):
    assert args.main_authorized, 'Main GPU/continuation scheduling required'
    upstream=HERE/'closedloop-a.py';assert sha(upstream)==UPSTREAM_SHA,'Frozen upstream source changed'
    assert not args.output.exists(), 'Never overwrite a previous control run'
    args.output.mkdir(parents=True)
    plan=json.loads(args.plan.read_text());assert len(plan['roots'])==80 and plan['arms']==['ttc5-V','allN-V']
    reference=args.reference/'closedloop-rows.jsonl.gz' if args.reference.is_dir() else args.reference
    reference_rows={(r['model'],r['root_id']):r for r in rows(reference) if r['arm']=='ttc5-V'}
    expected=[r for r in plan['roots'] if not args.models or r['model'] in args.models]
    if args.smoke_first_per_scope:
        seen={};selected=[]
        for r in expected:
            key=(r['model'],r['scope']);seen[key]=seen.get(key,0)+1
            if seen[key]<=args.smoke_first_per_scope:selected.append(r)
        expected=selected
    assert all((r['model'],r['root_id']) in reference_rows for r in expected), 'Formal TTC baseline missing'
    tree,changes=transformed_tree(upstream.read_text())
    protocol=dict(question='Do original allN gains exceed repeated-TTC5 token-count/pooling-statistic effects on the exact same frozen roots?',
        design='Each frame of parent out.windows already contains the same selected/anchored TTC5 human rows. Repeat each of those five rows four times, preserve the first3 robot/goal/motion rows exactly: [80,24,8,13] -> [80,24,23,13]. No candidate-dependent reselection or additional15 information.',
        interpretation='Statistical negative control, NOT20 independent humans or a new physical world. Original physical humans, commands/reward/filter/smoothing/terminal, frozen encoder/head and seeds unchanged. Recursive input replacement, not single-action branch effect.',
        history='Decision windows come ONLY from parent real TTC history/current/candidate windows. Inherited full_history hash is bookkeeping, not the used decision window; after root it includes repeated current tokens. Root used window/current hashes and80 values/raw/final scores are recorded from the actual repeated input.',
        N5='Unmodified upstream N5 token/window/value/score checks remain executable; helpers return original inputs for N=5. No bypass of existing numerical gate.',
        GO='Only if original allN exceeds duplication AND original protection controls pass; no new numeric threshold.',
        KILL='If duplication matches original allN, stop attributing that gain specifically to added crowd information. This does not alter original scientific gates or prove all input effects impossible.',
        roots_plan_sha256=sha(args.plan),reference_sha256=sha(reference),upstream_source_sha256=sha(upstream),wrapper_sha256=sha(__file__),
        AST_replacements=changes,arms=['ttc5-V','dup5x4-V'],expected_roots=[r['root_id'] for r in expected],
        budget='Smoke: same8roots x2arms +8A0 repeat gates=24 continuations. Formal: exact80roots x2arms +8repeat gates=168. No optimizer, extra model invocation or root reselection.')
    (args.output/'count-matched-protocol.json').write_text(json.dumps(protocol,indent=2))
    namespace=dict(__name__='count_matched_engine',__file__=str(upstream),count_current=count_current,count_windows=count_windows)
    exec(compile(tree,str(upstream),'exec'),namespace)
    args.arms=['ttc5-V','allN-V'];args.once=False;args.heads=None
    namespace['run'](args)
    checks=[];count=0
    with gzip.open(args.output/'count-matched-rows.jsonl.gz','wt') as stream:
        for r in rows(args.output/'closedloop-rows.jsonl.gz'):
            if r['arm']=='ttc5-V':
                ref=reference_rows[(r['model'],r['root_id'])]
                assert r['continuation_sha256']==ref['continuation_sha256'] and r['steps']==ref['steps'],'Formal TTC trajectory not exact'
                assert (r['outcome'],r['time'],r['root_return'])==(ref['outcome'],ref['time'],ref['root_return'])
                r['formal_TTC_trajectory_exact']=True
                checks.append(dict(model=r['model'],root_id=r['root_id'],continuation_sha256=r['continuation_sha256']))
            else:
                assert r['arm']=='allN-V';r['arm']='dup5x4-V';r['successor_input']='dup5x4'
                r['root_decision']['used_input']['input']='dup5x4'
                assert r['root_decision']['used_input']['window_shape']==[80,24,23,13]
            stream.write(json.dumps(r,allow_nan=False)+'\n');count+=1
    assert len(checks)==len(expected) and count==2*len(expected)
    result=dict(status='PASS',roots=len(checks),scientific_continuations=count,formal_TTC_exact=checks,
        protocol=protocol,raw_sha256=sha(args.output/'count-matched-rows.jsonl.gz'),method_level_progress=False,
        limitation='Inherited closedloop-* files are implementation artifacts: their internal allN-V arm means dup5x4 in this wrapper, not legal allN. Use count-matched protocol/rows for scientific comparison.')
    (args.output/'count-matched-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status='PASS',roots=len(checks),formal_TTC_exact=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+');p.add_argument('--device',default='cpu')
    p.add_argument('--smoke-first-per-scope',type=int,default=0)
    p.add_argument('--main-authorized',action='store_true');p.add_argument('--gpu-authorized',action='store_true')
    main(p.parse_args())
