#!/usr/bin/env python3
"""Render independently accepted baseline statistics to stdout; no raw edits."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    assert data['execution_status']=='PASS' and data['scope']=='baseline_only'
    assert data['verified']['model_episodes']==640 and data['verified']['reference_episodes']==320
    print('# Independently Verified Formal B Baselines\n')
    print('Source JSON SHA256: '+hashlib.sha256(args.input.read_bytes()).hexdigest())
    print('\nAll cells n=16, cases90000-90015. S/C/T are counts; G is gamma.99 discounted return.')
    print('Time/path are means over all episodes, including collisions and timeouts.')
    print('Physical C counts episodes with negative actual swept clearance; it is not the legacy env CR.')
    print('Latency columns are means of per-episode p50/p95, not pooled step latency percentiles.\n')
    for geo in ('circle','square'):
        print('## '+geo+'\n')
        print('| Controller | Condition | S/C/T | Mean G | Time(s) | Path(m) | Physical C | Latency(ms,p50/p95) |')
        print('|---|---|---|---|---|---|---|---|')
        for g in data['groups']:
            if g['geometry']!=geo:
                continue
            label = g['model'] if g['controller']=='model' else g['controller']
            print('| %s | %s | %d/%d/%d | %.6f | %.4f | %.4f | %d | %.3f/%.3f |'%(
                label,g['condition'],g['success'],g['collision'],g['timeout'],g['return'],
                g['time'],g['path_length'],g['physical_collision_episodes'],g['latency_ms_p50'],g['latency_ms_p95']))
        print()
    print('## Paired Reference Contrasts\n')
    print('Delta=B-A; intervals are deterministic paired case bootstrap2000, seed41995. Not architecture-seed CIs.\n')
    print('| Geometry | Reference | A to B | Delta SR [95%] | Delta G [95%] |')
    print('|---|---|---|---|---|')
    for row in data['paired']:
        if row['controller']=='model' or (row['A'],row['B']) not in [('E1','E2'),('E2','E4'),('E1','E5')]:
            continue
        s,r = row['metrics']['success'],row['metrics']['return']
        print('| %s | %s | %s to %s | %.4f [%.4f,%.4f] | %.6f [%.6f,%.6f] |'%(
            row['geometry'],row['controller'],row['A'],row['B'],s['mean_delta_B_minus_A'],
            *s['case_bootstrap95'],r['mean_delta_B_minus_A'],*r['case_bootstrap95']))
    print('\n## Archival Limits\n')
    for limit in data['archival_limitations']:
        print('- '+limit)
    print('\nCustom CV-MPC is a standard construction, not an official mature second-baseline reproduction.')
    print('B5 continuation results and native N10 are not included in this baseline-only acceptance.')


if __name__=='__main__':
    main()
