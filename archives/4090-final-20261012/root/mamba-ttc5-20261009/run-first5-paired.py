"""Add exactly 3000 First-5 evaluations; preserve completed TTC artifacts."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import binomtest

import importlib.util
BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fixed_mamba', BASE / 'run-mamba.py')
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)


def paired(old, new):
    a = {(x['cell'], x['case']): x for x in old}
    b = {(x['cell'], x['case']): x for x in new}
    assert a.keys() == b.keys() and len(a) == len(old) == len(new)
    cases = sorted({k[1] for k in a})
    cells = sorted({k[0] for k in a})
    results = {}
    for metric in fixed.METRICS:
        left = sum(a[k]['outcome'] == metric and b[k]['outcome'] != metric for k in a)
        right = sum(b[k]['outcome'] == metric and a[k]['outcome'] != metric for k in a)
        p = float(binomtest(left, left + right, .5).pvalue) if left + right else 1.
        delta = np.array([np.mean([float(b[cell, case]['outcome'] == metric)
                                  - float(a[cell, case]['outcome'] == metric) for cell in cells])
                          for case in cases])
        draws = 100 * np.random.RandomState(20261008).choice(delta, (5000, len(cases))).mean(axis=1)
        results[metric] = dict(b_first5_only=left, c_ttc5_only=right,
            delta_pp=100 * float(delta.mean()), exact_mcnemar_p=p,
            bonferroni_p=min(1., 3 * p), family_size=3,
            case_cluster_ci95_pp=np.percentile(draws, [2.5, 97.5]).tolist(),
            family_case_cluster_ci_pp=np.percentile(draws, [2.5 / 3, 100 - 2.5 / 3]).tolist())
    return results


def aggregate():
    rows = {r: sum([fixed.validate(BASE / 'results' / r / ('cell%d.json' % c), r, c)
                    for c in range(6)], []) for r in ('first5', 'ttc5')}
    a = {(x['cell'], x['case']): x for x in rows['first5']}
    b = {(x['cell'], x['case']): x for x in rows['ttc5']}
    assert a.keys() == b.keys() and len(a) == 3000
    for k in a:
        if a[k]['human_num'] == 5:
            assert all(a[k][f] == b[k][f] for f in ('outcome', 'return', 'time', 'minimum_clearance'))
    return dict(arm='mamba', checkpoint_sha256=fixed.EXPECTED,
        case_base=91000, cases_per_cell=500, rules={r: dict(overall=fixed.summary(rows[r]),
            cells={str(c): fixed.summary([x for x in rows[r] if x['cell'] == c]) for c in range(6)})
            for r in rows}, paired_overall=paired(rows['first5'], rows['ttc5']),
        paired_cells_descriptive={str(c): paired([x for x in rows['first5'] if x['cell'] == c],
                                                [x for x in rows['ttc5'] if x['cell'] == c])
                                  for c in range(6)},
        limits=['Single frozen training seed, no retraining',
                'Separate predeclared family of 3 overall tests; cell tests descriptive',
                'Case-cluster bootstrap handles case-ID reuse; McNemar independence needs caution',
                'Original TTC outputs untouched; this is fixed-rule retrospective extension'])


def main():
    lock = (BASE / 'first5-controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (BASE / 'first5-DONE.json').exists()
    frozen = fixed.read(BASE / 'frozen-manifest.json')
    assert fixed.sources() == frozen['source_sha256']
    assert fixed.sha(fixed.ASSET / 'mamba.ini') == frozen['config_sha256']
    assert fixed.sha(fixed.ASSET / 'mamba/rl_model_ep10000.pth') == frozen['checkpoint_sha256']
    assert fixed.sha(BASE / 'evaluate-selection.py') == frozen['script_sha256']['evaluate-selection.py']
    for c in range(6):
        fixed.validate(BASE / 'results/ttc5' / ('cell%d.json' % c), 'ttc5', c)
    assert fixed.read(BASE / 'preflight.json')['status'] == 'PASS'
    start = time.time()
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    fixed.save('first5-controller-pid.json', dict(pid=os.getpid(), proc_start_ticks=stat[19], started=start))
    fixed.save('first5-protocol.json', dict(case_base=91000, n_cases_per_cell=500,
        new_episodes=3000, training='NONE', rule='first5', comparator='existing ttc5 same 3000 cases',
        hypothesis_family=['overall SR', 'overall CR', 'overall TO'], bonferroni_family_size=3,
        script_sha256=fixed.sha(__file__)))
    active = []
    (BASE / 'results/first5').mkdir(parents=True, exist_ok=True)
    try:
        for cell in range(6):
            out = BASE / 'results/first5' / ('cell%d.json' % cell)
            assert not out.exists(), 'Do not overwrite completed result'
            stream = (BASE / 'logs' / ('first5-cell%d.log' % cell)).open('w')
            child = subprocess.Popen(fixed.command('evaluate', 'first5', cell, out),
                cwd=fixed.PROJECT, stdout=stream, stderr=subprocess.STDOUT)
            active.append((child, stream, cell, out))
        while active:
            for child, stream, cell, out in list(active):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                active.remove((child, stream, cell, out))
                if code:
                    raise RuntimeError('First5 cell%d exited %d' % (cell, code))
                fixed.validate(out, 'first5', cell)
            fixed.save('first5-status.json', dict(stage='EVALUATING', training='NONE',
                active_cells=[c for _, _, c, _ in active], wall_seconds=time.time() - start))
            if time.time() - start > 8 * 3600:
                raise TimeoutError('Eight-hour ceiling')
            if active:
                time.sleep(5)
        assert fixed.sources() == frozen['source_sha256']
        assert fixed.sha(fixed.ASSET / 'mamba.ini') == frozen['config_sha256']
        fixed.save('paired-summary.json', aggregate())
        fixed.save('first5-DONE.json', dict(status='COMPLETED', new_episodes=3000,
            compared_episode_pairs=3000, source_unchanged=True, training='NONE', wall_seconds=time.time() - start))
        fixed.save('first5-status.json', dict(stage='COMPLETED', training='NONE'))
    except BaseException as error:
        for child, _, _, _ in active:
            if child.poll() is None:
                child.terminate()
        for child, stream, _, _ in active:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            stream.close()
        fixed.save('first5-ERROR.json', dict(error=repr(error), wall_seconds=time.time() - start))
        fixed.save('first5-status.json', dict(stage='FAILED', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
