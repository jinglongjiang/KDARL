#!/usr/bin/env python3
"""CPU-only independent acceptance of the existing Mamba-VL reference.

Wait for reference-mamba-v3/summary.json PASS. Read only archived raw and pinned
inputs, never import the wrapper, environment, torch or model. Earlier invalid
launches are neither overwritten nor pooled. No change to B's original gates.
Print JSON to stdout; caller archives it under b/work, not the reference folder.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_raw',HERE/'verify-b-raw.py')
V=importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
CHECKPOINT='f23e411effac2f1959eead081fa49c908a85532642172c50a34427fe40bff2b9'
CONDITIONS=['E1','E2','E3','E4','E5','E10']


def validate_protocol(data,inputs,source):
    V.exact(data['cases'],list(range(90000,90016)),'reference formal cases')
    V.exact(data['smoke_cases'],list(range(90000,90003)),'reference smoke cases')
    V.exact(data['geometries'],['circle','square'],'reference geometries')
    V.exact(data['conditions'],CONDITIONS,'reference conditions')
    V.exact(data['formal_episodes'],192,'reference formal count')
    V.exact(data['smoke_episodes'],36,'reference smoke count')
    V.exact(data['checkpoint_sha256'],CHECKPOINT,'reference checkpoint')
    V.exact(data['frozen_b_driver_sha256'],V.EXPECTED_DRIVER_SHA,'reference B driver')
    V.exact(data['script_sha256'],V.sha(source),'reference wrapper source')
    V.exact(V.sha(inputs/'rl_model_ep10000.pth'),CHECKPOINT,'actual reference checkpoint')
    V.exact(V.sha(inputs/'mamba.ini'),data['config_sha256'],'actual reference config')


def validate_start(start,protocol,phase,source_sha):
    V.exact(start['script_sha256'],V.EXPECTED_DRIVER_SHA,'reference inherited driver')
    V.exact(start['contract_sha256'],V.EXPECTED_CONTRACT_SHA,'reference original contract')
    V.exact(start['phase'],phase,'reference phase')
    V.exact(start['controller'],'model','reference controller')
    for key in ('status','history_test','terminal_test'):
        V.exact(start['acceptance'][key],'PASS','reference acceptance '+key)
    V.exact(start['acceptance']['tested_commit'],'9f06223','reference accepted commit')
    for key,value in [('id','mamba-reference'),('seed',419),('backbone','mamba'),
        ('weight_sha256',CHECKPOINT),('config_sha256',protocol['config_sha256'])]:
        V.exact(start['model'][key],value,'reference model '+key)
    V.exact(start['source_sha256'],source_sha,'reference accepted project source hashes')


def check_summary_group(actual,stored):
    for key,value in stored.items():
        if key not in ('geometry','condition'):
            V.exact(actual[key],value,'reference summary mismatch '+key)


def audit(args):
    result=dict(execution_status='INCOMPLETE',method_level_progress=False,
        read_only=True,model_imported=False,rollouts_run=False,thresholds_modified=False,
        original_candidate_gate_replaced=False,full_training_started=False,
        reference_level='Existing local optimized Mamba-VL T24 seed419 checkpoint; matched decision interface, not fresh-history architecture comparison or full paper reproduction.',
        auditor_sha256=V.sha(Path(__file__)),helper_sha256=V.sha(HERE/'verify-b-raw.py'),
        missing=[],errors=[],groups=[],paired=[],shards=[],E1_E3_pairs=[],
        verified=dict(smoke_episodes=0,formal_episodes=0,raw_steps=0),
        archival_limitations=['Final post-update full human state and ORCA neighbor arrays not archived.',
            'Neural values/windows not recomputed; raw scores, tokens/history hashes, commands and physical transitions independently checked.'])
    try:
        summary=V.read_json(args.reference/'summary.json')
        V.exact(summary['status'],'PASS','reference completion sentinel')
        V.exact(summary['input_hashes_end_verified'],True,'reference end input hashes')
        V.exact(summary['method_level_progress'],False,'reference method claim')
        protocol=V.read_json(args.reference/'protocol.json')
        validate_protocol(protocol,args.inputs,args.source)
        V.exact(V.sha(args.b_work.parent/'run-b.py'),V.EXPECTED_DRIVER_SHA,'frozen B source')
        V.exact(V.sha(args.b_work.parent/'b-contract.json'),V.EXPECTED_CONTRACT_SHA,'frozen B contract')
        parent=V.read_json(args.b_work/'verified-baseline.json')
        V.exact(parent['execution_status'],'PASS','original baseline acceptance')
        source_sha=parent['shards'][0]['files'][0]['provenance']['source_sha256']
        native=V.read_json(args.n10_work/'cpu-main.json')
        V.exact(native['status'],'PASS','N10 reset acceptance')
        initials={(r['geometry'],'E10',r['case']):(r['robot'],r['humans']) for r in native['checks']}
        for geo in protocol['geometries']:
            for condition in CONDITIONS[:-1]:
                path=args.b_work/'main/model/kda-s419-t24'/(geo+'-'+condition+'.jsonl.gz')
                for ep in V.rows(path):
                    row=ep['steps'][0]
                    initials[geo,condition,ep['case']]=(row['robot'],row['humans'])
        smoke_gate=V.read_json(args.reference/'work/smoke-accepted.json')
        V.exact(smoke_gate['status'],'PASS','driver-recognized smoke gate')
        V.exact(smoke_gate['episodes'],36,'driver-recognized smoke count')
        V.exact(smoke_gate['protocol_sha256'],V.sha(args.reference/'protocol.json'),'reference smoke protocol')
        main_gate=V.read_json(args.reference/'main-gate.json')
        V.exact(main_gate['status'],'PASS','reference main gate')
        V.exact(main_gate['episodes'],192,'reference main gate count')
        V.exact(main_gate['protocol_sha256'],V.sha(args.reference/'protocol.json'),'reference main protocol')
        loaded={}
        for phase,cases in [('smoke',protocol['smoke_cases']),('main',protocol['cases'])]:
            for geo in protocol['geometries']:
                for condition in CONDITIONS:
                    path=args.reference/'work'/phase/'model/mamba-reference'/(geo+'-'+condition+'.jsonl.gz')
                    done,start=V.read_json(path.with_suffix('.done.json')),V.read_json(path.with_suffix('.start.json'))
                    V.exact(done['status'],'PASS','reference shard done')
                    V.exact(done['output_sha256'],V.sha(path),'reference raw SHA')
                    V.exact(done['provenance'],start,'reference done/start provenance')
                    validate_start(start,protocol,phase,source_sha)
                    V.exact(start['cases'],cases,'reference frozen case sequence')
                    eps=[]
                    for ep in V.rows(path):
                        result['verified']['raw_steps']+=V.check_episode(ep,('model','mamba-reference',geo,condition))
                        V.exact((ep['steps'][0]['robot'],ep['steps'][0]['humans']),
                            initials[geo,condition,ep['case']],'reference reset differs from matched scene')
                        eps.append(V.compact_episode(ep))
                    V.exact([e['case'] for e in eps],cases,'reference actual case sequence')
                    V.exact(done['episodes'],len(cases),'reference shard count')
                    result['shards'].append(dict(phase=phase,geometry=geo,condition=condition,
                        file=str(path),sha256=done['output_sha256'],episodes=len(eps),wall_seconds=done['wall_seconds']))
                    result['verified']['smoke_episodes' if phase=='smoke' else 'formal_episodes']+=len(eps)
                    if phase=='main':
                        loaded[geo,condition]=eps
                        result['groups'].append(dict(geometry=geo,condition=condition,**V.statistics(eps)))
            for geo in protocol['geometries']:
                directory=args.reference/'work'/phase/'model/mamba-reference'
                checks=[V.compare_e1_e3(a,b) for a,b in zip(V.rows(directory/(geo+'-E1.jsonl.gz')),
                    V.rows(directory/(geo+'-E3.jsonl.gz')))]
                V.exact(len(checks),len(cases),'reference E1/E3 pair count')
                result['E1_E3_pairs'].append(dict(phase=phase,geometry=geo,paired_cases=len(checks),
                    paired_steps=sum(c['steps'] for c in checks),checks=checks))
        for stored in summary['groups']:
            actual=next(g for g in result['groups'] if (g['geometry'],g['condition'])==
                (stored['geometry'],stored['condition']))
            check_summary_group(actual,stored)
        V.exact(len(summary['groups']),12,'reference summary group count')
        for geo in protocol['geometries']:
            for a,b in [('E1','E2'),('E1','E3'),('E2','E4'),('E2','E5'),('E1','E10'),('E10','E2')]:
                metrics=V.paired_statistics(loaded[geo,a],loaded[geo,b])
                stored=next(r for r in summary['paired'] if (r['geometry'],r['A'],r['B'])==(geo,a,b))
                V.exact(stored['metrics'],metrics,'reference paired case statistics')
                result['paired'].append(dict(geometry=geo,A=a,B=b,metrics=metrics))
        V.exact(len(summary['paired']),12,'reference paired summary count')
        V.exact(result['verified']['smoke_episodes'],36,'all reference smoke')
        V.exact(result['verified']['formal_episodes'],192,'all reference main')
        result.update(execution_status='PASS',protocol_sha256=V.sha(args.reference/'protocol.json'),
            wrapper_sha256=protocol['script_sha256'],checkpoint_sha256=CHECKPOINT,
            config_sha256=protocol['config_sha256'],summary_sha256=V.sha(args.reference/'summary.json'),
            wall_seconds=summary['wall_seconds'],reference_directory=str(args.reference))
    except (FileNotFoundError,EOFError) as error:
        result['missing'].append(str(error))
    except (AssertionError,ValueError,KeyError,TypeError,OSError,StopIteration) as error:
        result['execution_status']='FAIL'
        result['errors'].append(str(error))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path('/root/ab-causal-isolation-20261011')
    p.add_argument('--reference',type=Path,default=base/'reference-mamba-v3')
    p.add_argument('--source',type=Path,default=base/'reference-mamba.py')
    p.add_argument('--inputs',type=Path,default=base/'reference-mamba/inputs')
    p.add_argument('--b-work',type=Path,default=base/'b/work')
    p.add_argument('--n10-work',type=Path,default=base/'b/n10/work')
    args=p.parse_args()
    result=audit(args)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['execution_status']=='PASS' else 1 if result['errors'] else 2


if __name__=='__main__':
    sys.exit(main())
