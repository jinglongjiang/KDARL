#!/usr/bin/env python3
"""Read accepted raw verification; adjudicate frozen A1 and excess gates separately.

No new experiments or thresholds. A condition with any unmatched failure root
is excluded in full from the protected A1 claim, regardless of its outcome.
The retained roots are failure-conditioned, capped opportunities, not a deployable
trigger or an independent method. E3 duplicates E1; never independent evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(groups):
    values = [v for g in groups for v in g['failure_delta_root_returns']]
    controls = [v for g in groups for v in g['control_delta_root_returns']]
    result = dict(failure_roots=len(values),success_controls=len(controls),
        rescues=sum(g['rescues'] for g in groups),
        success_destructions=sum(g['success_destructions'] for g in groups),
        paired_mean_delta_root_G=statistics.mean(values) if values else None,
        control_mean_delta_root_G=statistics.mean(controls) if controls else None,
        seeds=sorted({g['model'] for g in groups}))
    result['A1_numeric_gate'] = 'PASS' if (result['rescues']>=2 and values and
        result['paired_mean_delta_root_G']>0 and result['success_destructions']==0
        and len(result['seeds'])==2) else 'FAIL'
    return result


def adjudicate(data):
    assert data['execution_status']=='PASS'
    source=data['original']
    assert source['execution_status']=='PASS'
    assert source['verified']['total_episodes']==960
    assert len(source['E1_E3_pairs'])==12
    all_groups=[g for g in source['branches'] if g['model'].startswith('kda-')]
    valid=[g for g in all_groups if g['failure_roots'] and not g['unmatched_failure_roots']]
    excluded=[dict(model=g['model'],geometry=g['geometry'],condition=g['condition'],
        failure_roots=g['failure_roots'],unmatched_failure_roots=g['unmatched_failure_roots'],
        rescues=g['rescues'],reason='Whole condition disqualified: missing success controls')
        for g in all_groups if g['unmatched_failure_roots']]
    result=dict(A1=dict(eligible_conditions=[{k:g[k] for k in ('model','geometry','condition')}
        for g in valid],disqualified_conditions=excluded,
        by_geometry={geo:summarize([g for g in valid if g['geometry']==geo])
            for geo in ('circle','square')},
        all_eligible_conditions_descriptive=summarize(valid),
        excluding_duplicate_E3=summarize([g for g in valid if g['condition']!='E3']),
        scope='Finite frozen failure-root opportunities only; not an online trigger, whole-distribution benefit or method freeze.'),
        thresholds_modified=False,full_training_started=False,method_level_progress=False)
    rows=[{k:r[k] for k in ('model','geometry','A','B','against_references',
        'both_reference_baseline_threshold_met')} for r in source['excess_transfer_baseline']
        if r['model'].startswith('kda-')]
    assert len(rows)==4
    result['excess_transfer']=dict(baseline_rows=rows,
        baseline_necessary_gate='PASS' if all(r['both_reference_baseline_threshold_met'] for r in rows) else 'FAIL',
        A1_gate_cannot_override_failed_baseline=True,
        decision='KILL extra-value-transfer candidate; not refutation of all value failures')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('verified',type=Path)
    args=p.parse_args()
    data=json.loads(args.verified.read_text())
    result=adjudicate(data)
    result['verified_input_sha256']=hashlib.sha256(args.verified.read_bytes()).hexdigest()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
