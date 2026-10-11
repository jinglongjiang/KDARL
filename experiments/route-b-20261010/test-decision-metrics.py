"""Safety eligibility and full-score gap must not contain mask sentinel values."""
import numpy as np
from common import BASE, module, save

supplement=module('test_decision_supplement',BASE/'supplement-decisions.py')
v=np.zeros(80); raw=np.zeros(80);raw[:3]=[2.,1.,.8]
cmd=np.zeros((80,2));cmd[:,0]=np.linspace(.05,1.,80)
clear=np.zeros(80);clear[1:3]=.3
r=supplement.decision_metrics(v,raw,clear,cmd)
assert r['selected_index']==1 and r['n_safe']==2
assert abs(r['decision_gap']-.2)<1e-12 and r['fallback_gap'] is None
clear[2]=0.
one=supplement.decision_metrics(v,raw,clear,cmd)
assert one['n_safe']==1 and one['decision_gap'] is None
clear[1]=0.
fallback=supplement.decision_metrics(v,raw,clear,cmd)
assert fallback['n_safe']==0 and fallback['decision_gap'] is None
assert fallback['selected_index']==0 and abs(fallback['fallback_gap']-.999)<1e-12
assert all(abs(x)<1e6 for x in fallback['complete_unmasked_scores'])
save(BASE/'decision-metric-unit-checks.json',dict(status='已验证通过',
    two_safe_gap=r['decision_gap'],single_safe_gap=one['decision_gap'],
    all_unsafe_fallback_gap=fallback['fallback_gap'],no_mask_sentinel=True))
print('PASS: eligible2/single-safe/all-unsafe separated; penalty preserved.')
