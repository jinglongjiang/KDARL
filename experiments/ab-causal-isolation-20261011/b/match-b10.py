#!/usr/bin/env python3
"""Read-only matching of frozen native5/10/20 opportunity roots; no model imports."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    data = json.loads(path.read_text())
    assert json.loads(path.with_suffix('.sha256.json').read_text())['sha256']==sha(path)
    assert data['branch_results_exist'] is False
    return data


def distance(a,b):
    x,y = a['snapshot'],b['snapshot']
    ds = [abs(x['step']-y['step'])/8,abs(x['goal_distance']-y['goal_distance']),
          abs(x['n_safe']-y['n_safe'])/16,abs(x['crowd2m']-y['crowd2m'])/2]
    return ds if a['type']==b['type'] and max(ds)<=1 else None


def pair(a,b):
    result,used = [],set()
    for root in a:
        options = []
        for other in b:
            ds = distance(root,other)
            if other['root_id'] not in used and ds is not None:
                options.append((root['case']!=other['case'],sum(ds),other['case'],other['step'],other,ds))
        if options:
            chosen = min(options,key=lambda x:x[:4])
            other,ds = chosen[-2:]
            used.add(other['root_id'])
            result.append({'A':root['root_id'],'B':other['root_id'],'same_case':root['case']==other['case'],
                'standardized_differences':ds})
    return result


def triples(a,b,c):
    result,usedb,usedc = [],set(),set()
    for root in a:
        options = []
        for mid in b:
            if mid['root_id'] in usedb or distance(root,mid) is None:
                continue
            for high in c:
                ds = [distance(root,mid),distance(root,high),distance(mid,high)]
                if high['root_id'] not in usedc and all(d is not None for d in ds):
                    options.append((not(root['case']==mid['case']==high['case']),sum(map(sum,ds)),
                        mid['case'],mid['step'],high['case'],high['step'],mid,high,ds))
        if options:
            chosen = min(options,key=lambda x:x[:6])
            mid,high,ds = chosen[-3:]
            usedb.add(mid['root_id'])
            usedc.add(high['root_id'])
            result.append({'N5_root':root['root_id'],'N10_root':mid['root_id'],'N20_root':high['root_id'],
                'all_same_case':root['case']==mid['case']==high['case'],'all_pair_standardized_differences':ds})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=ROOT/'work')
    p.add_argument('--n10-work',type=Path,default=ROOT/'n10/work')
    args = p.parse_args()
    ext = json.loads((ROOT/'b10-protocol.json').read_text())
    result = {'execution_status':'PASS','causal_status':'INCOMPLETE_PENDING_VALIDATED_PAIRED_CONTINUATIONS',
        'read_only':True,'groups':[],'missing':[],'errors':[],'thresholds_modified':False}
    for model in ['kda-s419-t24','kda-s443-t24','gru-s419-t24','gru-s443-t24']:
        try:
            low = read(args.work/('roots-'+model+'.json'))
            mid = read(args.n10_work/('roots-'+model+'.json'))
            assert low['contract_sha256']==ext['parent_contract_sha256']
            assert mid['contract_sha256']==sha(ROOT/'b10-protocol.json')
            for geo in ('circle','square'):
                levels = []
                for data,condition in [(low,'E1'),(mid,'E10'),(low,'E2')]:
                    levels.append([r for r in data['roots'] if r['geometry']==geo and r['condition']==condition
                        and r['type']!='success-control'])
                a,b,c = levels
                ts = triples(a,b,c)
                result['groups'].append({'model':model,'geometry':geo,'N5_N10':pair(a,b),'N10_N20':pair(b,c),
                    'N5_N20':pair(a,c),'strict_triples':ts,
                    'selected_failure_roots_5_10_20':list(map(len,levels)),
                    'all_failure_denominators_5_10_20':[sum(d['geometry']==geo and d['condition']==condition
                        for d in data['all_failure_denominator']) for data,condition in [(low,'E1'),(mid,'E10'),(low,'E2')]],
                    'status':'MATCHED_OPPORTUNITIES_NOT_CAUSAL_EFFECTS' if ts else 'INCOMPLETE_NO_STRICT_TRIPLES'})
        except FileNotFoundError as error:
            result['missing'].append(str(error))
        except (AssertionError,KeyError,ValueError) as error:
            result['errors'].append(str(error))
    if result['errors']:
        result['execution_status'] = 'FAIL'
    elif result['missing']:
        result['execution_status'] = 'INCOMPLETE'
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['execution_status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
