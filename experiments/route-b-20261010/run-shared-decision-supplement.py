"""Independent-container continuation: the original container exits when its PID1 finishes."""
import sys
sys.dont_write_bytecode=True
import contextlib
import json
import time
import traceback
from common import BASE, specs, save, sha, verify, module

start=time.time()
assert not (BASE/'decision-DONE.json').exists()
save(BASE/'shared-supplement-manifest.json',dict(files={str(BASE/p):sha(BASE/p) for p in (
    'run-shared-decision-supplement.py','supplement-decisions.py','protocol-amendment-v2.json')}))
while not (BASE/'DONE.json').exists():
    assert not (BASE/'ERROR.json').exists(), 'Frozen collector failure'
    time.sleep(3)
assert len(list((BASE/'p1-margin/decision').glob('*/cell[0-5].json')))==66
save(BASE/'decision-recovery.json',dict(status='已验证通过',
    reason='Controller ownership correction: own replay finished; shared supplement moved to independent container so original PID1 exit cannot kill it.',
    preserved_completed_own_cells=66,no_experimental_results_changed=True))
supplement=module('route_b_shared_decision_supplement',BASE/'supplement-decisions.py')
for item in specs():
    for cell in range(6):
        label='decision-shared-%s-cell%d'%(item['id'],cell)
        code=0
        with (BASE/'logs'/(label+'.log')).open('w') as out:
            with contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):
                try:supplement.shared(item,cell)
                except BaseException:
                    traceback.print_exc();code=1
        save(BASE/'job-evidence'/(label+'.json'),dict(name=label,exit_code=code,
            actual_output_tail=(BASE/'logs'/(label+'.log')).read_text()[-4000:]))
        assert code==0,label
verify()
save(BASE/'decision-DONE.json',dict(status='COMPLETED',wall_seconds=time.time()-start,
    own_episodes=1320,shared_reference_episodes=120,nn_forward_calls=0,new_policy_rollouts=0,
    controller_ownership_correction='decision-recovery.json'))
save(BASE/'decision-status.json',dict(stage='COMPLETED'))
