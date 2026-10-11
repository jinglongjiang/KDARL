"""Fail-closed supplement controller. Frozen scripts and prior records stay untouched."""
import sys
sys.dont_write_bytecode = True
import json
import subprocess
import time
import contextlib
import traceback
from common import BASE, specs, save, verify, sha, module

PY = '/home/abc/temp/kda-vl-env/bin/python'


def main():
    assert json.loads((BASE/'decision-smoke-check.json').read_text())['cost_gate_pass']
    start = time.time()
    verify()
    manifest = {str(BASE/name):sha(BASE/name) for name in (
        'protocol-amendment-v2.json','supplement-decisions.py','run-decision-supplement.py',
        'p3-novelty.md','p2-provenance.json')}
    save(BASE/'amendment-frozen-manifest.json', dict(files=manifest, preserved_original_manifest=True))
    script = BASE/'supplement-decisions.py'
    supplement = module('route_b_decision_supplement', script)
    for mode in ('own','shared'):
        if mode == 'shared':
            while not (BASE/'DONE.json').exists():
                assert not (BASE/'ERROR.json').exists(), 'Original frozen collector failed'
                save(BASE/'decision-status.json',dict(stage='waiting-for-original-shared',controller_pid=__import__('os').getpid()))
                time.sleep(3)
        for item in specs():
            for cell in range(6):
                label='decision-%s-%s-cell%d'%(mode,item['id'],cell)
                save(BASE/'decision-status.json',dict(stage=mode,checkpoint=item['id'],cell=cell))
                code = 0
                with (BASE/'logs'/(label+'.log')).open('w') as out:
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                        try:
                            if mode == 'own':
                                supplement.replay(item,cell,False)
                            else:
                                supplement.shared(item,cell)
                        except BaseException:
                            traceback.print_exc()
                            code = 1
                save(BASE/'job-evidence'/(label+'.json'),dict(name=label,exit_code=code,
                    actual_output_tail=(BASE/'logs'/(label+'.log')).read_text()[-4000:]))
                assert code == 0, label
    for name, expected in manifest.items():
        assert sha(name) == expected, name
    verify()
    save(BASE/'decision-DONE.json',dict(status='COMPLETED',wall_seconds=time.time()-start,
        own_episodes=1320,shared_reference_episodes=120,nn_forward_calls=0,new_policy_rollouts=0))
    save(BASE/'decision-status.json',dict(stage='COMPLETED'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save(BASE/'decision-ERROR.json',dict(status='已确认失败',error=repr(error),automatic_restart=False))
        raise
