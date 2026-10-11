"""Two-device deployment study. No training and no interference with foreign jobs."""
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

import numpy as np
from scipy.stats import binom_test

BASE = Path(__file__).resolve().parent
PROJECT = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
CPROJECT = '/workspace/nav_data/mamba/camrl/CrowdNav'
CBASE = '/workspace/kda-selection-20261008'
RBASE = '/root/kda-selection-20261008'
RPROJECT = '/root/cc-diag-20261008/code'
REMOTE = 'root@45.126.120.6'
PYTHON = '/root/miniconda3/envs/mamba/bin/python'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=30', REMOTE]
RULES = ('first5', 'nearest5', 'ttc5')
ARMS = ('gru', 'kda')
METRICS = ('success', 'collision', 'timeout')
EXPECTED = {'gru': '96f06039a07b75d09e3b067ebb796d6de113b6d80a33214db737ed6d3a245828',
            'kda': 'cb2e20c4cc2f12014fed589e9ff02468d04cfa12573c66fc79617c95356cbf0a'}
CANCEL = threading.Event()
START = time.time()
DEADLINE = START + 18 * 3600


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def sources():
    paths = sorted(PROJECT.rglob('*.py'))
    paths += [PROJECT / 'artifacts/vl-optimized-20261007' / (arm + '.ini') for arm in ARMS]
    return {str(p.relative_to(PROJECT)): sha(p) for p in paths}


def guard():
    if CANCEL.is_set():
        raise RuntimeError('Sibling failed; preserve evidence and stop owned workers')
    if time.time() > DEADLINE:
        raise TimeoutError('18-hour ceiling; no further launches')


def remote_json(code):
    command = SSH + [PYTHON + ' -c ' + shlex.quote(code)]
    answer = subprocess.run(command, check=True, timeout=60, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    assert answer.stdout.strip(), 'Empty response cannot pass validation'
    return json.loads(answer.stdout)


def sync():
    subprocess.run(['rsync', '-az', '--timeout=60', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                    '--include=/development/', '--include=/confirmation/', '--include=/preflight/',
                    '--include=*/', '--include=*.json', '--include=*.progress', '--exclude=*',
                    REMOTE + ':' + RBASE + '/', str(BASE) + '/'], check=True, timeout=180)


def wait_foreign():
    code = """import json,pathlib,subprocess,shutil
b=pathlib.Path('/root/bayes-fix-20261008')
p=json.loads((b/'formal-controller-pid.json').read_text())
s=pathlib.Path('/proc/'+str(p['pid'])+'/stat')
alive=False
if s.exists():
 stat=s.read_text().rsplit(')',1)[1].split()
 alive=stat[19]==str(p['proc_start_ticks']) and stat[0]!='Z'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
pids=[int(x.strip()) for x in gpu.splitlines() if x.strip().isdigit()]
complete=json.loads((b/'complete.json').read_text()) if (b/'complete.json').exists() else None
if complete is not None:
 complete={'status':complete.get('status'),'result_count':len(complete.get('results',[]))}
error=json.loads((b/'error.json').read_text()) if (b/'error.json').exists() else None
print(json.dumps(dict(controller_alive=alive,gpu_pids=pids,complete=complete,error=error,
 free_bytes=shutil.disk_usage('/').free)))
"""
    while True:
        guard()
        try:
            status = remote_json(code)
            save(BASE / 'status-kda.json', dict(stage='WAITING_FOREIGN_PIPELINE', remote=status,
                                              wall_seconds=time.time() - START))
            if not status['controller_alive'] and not status['gpu_pids']:
                if status['complete'] and status['complete'].get('status') == 'COMPLETE':
                    assert status['free_bytes'] >= 1024 ** 3, 'Need 1GiB free for evaluation evidence'
                    save(BASE / 'foreign-completion.json', status)
                    return
                if status['error']:
                    raise RuntimeError('Foreign pipeline failed; require resource-state review')
        except (subprocess.SubprocessError, json.JSONDecodeError) as error:
            save(BASE / 'status-kda.json', dict(stage='SSH_RETRY', error=str(error)))
        CANCEL.wait(120)


def specification(arm, mode, rule, stage=None, cell=0):
    rel = 'preflight/' + arm + '.json' if mode == 'preflight' else (
        stage + '/' + arm + '-' + rule + '/cell' + str(cell) + '.json')
    (BASE / rel).parent.mkdir(parents=True, exist_ok=True)
    project, root = (CPROJECT, CBASE) if arm == 'gru' else (RPROJECT, RBASE)
    asset = project + '/artifacts/vl-optimized-20261007' if arm == 'gru' else '/root/cc-diag-20261008/results'
    argv = [PYTHON, root + '/evaluate-selection.py', '--project', project,
            '--arm-dir', asset + '/' + arm, '--config', asset + '/' + arm + '.ini',
            '--backbone', arm, '--mode', mode, '--human-select', rule, '--out', root + '/' + rel]
    if mode == 'evaluate':
        argv += ['--cell', str(cell), '--case-base', '90000' if stage == 'development' else '91000',
                 '--n-cases', '500']
    environment = 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg'
    if arm == 'kda':
        environment += ' PYTHONPATH=/root/kda-vl-deps/flash-linear-attention'
    shell = 'cd ' + shlex.quote(project) + ' && ' + environment + ' ' + ' '.join(map(shlex.quote, argv))
    command = ['docker', 'exec', 'mamba_env', 'bash', '-c', shell] if arm == 'gru' else SSH + [shell]
    return command, BASE / rel, root + '/' + rel


def valid(path, arm, rule, stage, cell):
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    assert data['checkpoint_sha256'] == EXPECTED[arm] and data['grid_actions'] == 80
    assert data['backbone'] == arm and data['selection'] == rule
    assert data['script_sha256'] == sha(BASE / 'evaluate-selection.py')
    assert len(data['episodes']) == 500
    assert data['protocol'] == dict(cell=cell, n_cases=500,
                                  case_base=90000 if stage == 'development' else 91000)
    return True


def stop_owned(arm, out):
    code = """import pathlib,json,os,signal
pfile=pathlib.Path(OUT+'.pid.json')
if pfile.exists():
 p=json.loads(pfile.read_text())
 base=pathlib.Path('/proc/'+str(p['pid']))
 if base.exists():
  stat=(base/'stat').read_text().rsplit(')',1)[1].split()
  args=(base/'cmdline').read_bytes().decode().split(chr(0))
  if stat[19]==p['proc_start_ticks'] and p['script'] in args and OUT in args:
   os.kill(p['pid'],signal.SIGTERM)
""".replace('OUT', repr(out))
    command = ['docker', 'exec', 'mamba_env', PYTHON, '-c', code] if arm == 'gru' else (
        SSH + [PYTHON + ' -c ' + shlex.quote(code)])
    try:
        subprocess.run(command, timeout=30, check=False)
    except subprocess.SubprocessError:
        pass


def preflight(arm):
    command, path, _ = specification(arm, 'preflight', 'first5')
    if not path.exists():
        with open(BASE / 'logs' / (arm + '-preflight.log'), 'w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=True)
        if arm == 'kda':
            sync()
    result = json.loads(path.read_text())
    assert result['script_sha256'] == sha(BASE / 'evaluate-selection.py')
    assert result['status'] == 'PASS' and result['checkpoint_sha256'] == EXPECTED[arm]
    assert result['max_abs_error']['windows'] == result['max_abs_error']['values'] == 0.
    assert result['grid_actions'] == 80


def block(arm, stage, rules):
    for rule in rules:
        guard()
        active = []
        try:
            for cell in range(6):
                command, path, out = specification(arm, 'evaluate', rule, stage, cell)
                if valid(path, arm, rule, stage, cell):
                    continue
                log = BASE / 'logs' / (stage + '-' + arm + '-' + rule + '-cell' + str(cell) + '.log')
                stream = log.open('a')
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                active.append((child, stream, out, cell))
            while active:
                guard()
                for child, stream, out, cell in list(active):
                    code = child.poll()
                    if code is None:
                        continue
                    stream.close()
                    active.remove((child, stream, out, cell))
                    if code:
                        raise RuntimeError('{} {} {} cell{} exit{}'.format(stage, arm, rule, cell, code))
                save(BASE / ('status-' + arm + '.json'), dict(stage=stage, rule=rule,
                     active=[dict(pid=p.pid, cell=c) for p, _, _, c in active],
                     wall_seconds=time.time() - START))
                if active:
                    time.sleep(5)
            if arm == 'kda':
                sync()
            for cell in range(6):
                assert valid(specification(arm, 'evaluate', rule, stage, cell)[1], arm, rule, stage, cell)
        except BaseException:
            for child, stream, out, _ in active:
                stop_owned(arm, out)
                child.terminate()
                stream.close()
            raise
    save(BASE / ('status-' + arm + '.json'), dict(stage=stage + '_COMPLETE', rules=list(rules)))


def rows(stage, arm, rule):
    data = []
    for cell in range(6):
        path = specification(arm, 'evaluate', rule, stage, cell)[1]
        assert valid(path, arm, rule, stage, cell)
        data += json.loads(path.read_text())['episodes']
    assert len(data) == 3000
    return data


def summary(data):
    counts = {m: sum(x['outcome'] == m for x in data) for m in METRICS}
    times = [x['time'] for x in data if x['outcome'] == 'success']
    return dict(n=len(data), counts=counts, rates_percent={m: 100 * n / len(data) for m, n in counts.items()},
                mean_return=float(np.mean([x['return'] for x in data])),
                median_minimum_clearance=float(np.median([x['minimum_clearance'] for x in data])),
                success_time_mean=float(np.mean(times)) if times else None,
                success_time_median=float(np.median(times)) if times else None,
                success_time_p90=float(np.percentile(times, 90)) if times else None)


def paired(old, new, family):
    a = {(x['cell'], x['case']): x for x in old}
    b = {(x['cell'], x['case']): x for x in new}
    assert a.keys() == b.keys()
    output = {}
    for metric in METRICS:
        left = sum(a[k]['outcome'] == metric and b[k]['outcome'] != metric for k in a)
        right = sum(a[k]['outcome'] != metric and b[k]['outcome'] == metric for k in a)
        p = float(binom_test(left, n=left + right, p=.5)) if left + right else 1.
        cases = sorted({k[1] for k in a})
        delta = np.array([np.mean([float(b[k]['outcome'] == metric) - float(a[k]['outcome'] == metric)
                                  for k in a if k[1] == case]) for case in cases])
        rng = np.random.RandomState(20261008)
        draws = np.mean(rng.choice(delta, size=(5000, len(cases)), replace=True), axis=1)
        output[metric] = dict(b_old_only=left, c_new_only=right, delta_pp=100 * float(delta.mean()),
            exact_mcnemar_p=p, bonferroni_p=min(1., family * p), family_size=family,
            ci95_pp=(100 * np.percentile(draws, [2.5, 97.5])).tolist(),
            family_ci_pp=(100 * np.percentile(draws, [2.5 / family, 100 - 2.5 / family])).tolist())
    return output


def aggregate(stage, rules):
    result = dict(stage=stage, arms={}, cross_arm_absolute_comparison='NOT_PERMITTED')
    family = 12 if stage == 'development' else 6
    for arm in ARMS:
        reference = rows(stage, arm, 'first5')
        result['arms'][arm] = {}
        for rule in rules:
            data = rows(stage, arm, rule)
            if rule != 'first5':
                baseline = {(x['cell'], x['case']): x for x in reference if x['human_num'] == 5}
                for x in data:
                    if x['human_num'] == 5:
                        assert all(x[k] == baseline[(x['cell'], x['case'])][k]
                                   for k in ('outcome', 'return', 'minimum_clearance', 'time'))
            entry = dict(overall=summary(data),
                         cells={str(c): summary([x for x in data if x['cell'] == c]) for c in range(6)})
            if rule != 'first5':
                entry['paired'] = paired(reference, data, family)
                entry['paired_20person'] = paired([x for x in reference if x['human_num'] == 20],
                                                 [x for x in data if x['human_num'] == 20], family)
            result['arms'][arm][rule] = entry
    save(BASE / (stage + '-summary.json'), result)
    return result


def report(dev, winner, final, verdict):
    lines = ['# KDA deployment-selection study', '', 'Status: ' + verdict,
             '', 'Zero training. Existing 5-person-trained seed419 ep10000 checkpoints.',
             'Within-arm paired comparisons only; no cross-device absolute SR subtraction.',
             'Development cases90000-90499; confirmation cases91000-91499.',
             'Frozen selected rule: ' + winner,
             'Cell mapping: 0=5-circle; 1=5-square; 2=10-circle; 3=10-square; 4=20-circle; 5=20-square.']
    for label, study in [('Development', dev), ('Confirmation', final)]:
        lines += ['', '## ' + label]
        if study is None:
            lines += ['Not run: first5 won the frozen selection ranking.']
            continue
        for arm, methods in study['arms'].items():
            lines += ['', '### ' + arm,
                '| Rule/cell | N | SR% | CR% | TO% | Return | Median clearance | Success time mean/median/p90 |',
                '|---|---:|---:|---:|---:|---:|---:|---|']
            for rule, entry in methods.items():
                for cell, x in [('overall', entry['overall'])] + list(entry['cells'].items()):
                    rates = x['rates_percent']
                    lines.append('| {} / {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.6f} | {:.6f} | {} / {} / {} |'.format(
                        rule, cell, x['n'], rates['success'], rates['collision'], rates['timeout'],
                        x['mean_return'], x['median_minimum_clearance'], x['success_time_mean'],
                        x['success_time_median'], x['success_time_p90']))
                if rule != 'first5':
                    lines += ['', '| Cohort/metric | b old-only | c new-only | Delta pp | Exact p | Bonferroni p | CI95 pp |',
                              '|---|---:|---:|---:|---:|---:|---|']
                    for cohort in ('paired', 'paired_20person'):
                        for metric, x in entry[cohort].items():
                            lines.append('| {} / {} | {} | {} | {:.3f} | {:.8g} | {:.8g} | {} |'.format(
                                cohort, metric, x['b_old_only'], x['c_new_only'], x['delta_pp'],
                                x['exact_mcnemar_p'], x['bonferroni_p'], x['ci95_pp']))
    lines += ['', '## Boundaries', '- Development results select the rule, not confirm final performance.',
              '- Non-significance is not equivalence; overall and20-person confidence bounds differ.',
              '- 20-person subgroup tests are descriptive; adoption uses only the overall frozen test family.',
              '- TTC is the inherited surrogate, not a guaranteed superior risk measure.',
              '- Shared actor selection improvement is not a KDA memory contribution.',
              '- One trained seed; no final-set tuning or training.',
              '- windows/values parity is bitwise; inherited reward ULP is reported explicitly.',
              '', '## Artifacts', str(BASE), '', '## Next decision', verdict, '']
    (PROJECT / 'KDA-SELECTION-CONFIRMATION-REPORT.md').write_text('\n'.join(lines))


def development(arm):
    if arm == 'kda':
        wait_foreign()
    preflight(arm)
    block(arm, 'development', RULES)


def main():
    lock = (BASE / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (BASE / 'logs').mkdir(exist_ok=True)
    manifest = dict(source_sha256=sources(), protocol_sha256=sha(BASE / 'protocol.json'),
                    script_sha256={x: sha(BASE / x) for x in ('evaluate-selection.py', 'run-study.py')})
    path = BASE / 'frozen-manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Frozen implementation changed; do not overwrite results'
    else:
        save(path, manifest)
    sample = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.1)'])
    assert sample.poll() is None
    assert sample.wait() == 0 and sample.poll() == 0
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    save(BASE / 'controller-pid.json', dict(pid=os.getpid(), proc_start_ticks=stat[19], started=START))
    save(BASE / 'status.json', dict(stage='DEVELOPMENT', training='NONE'))
    try:
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            for job in futures.as_completed([pool.submit(development, arm) for arm in ARMS]):
                try:
                    job.result()
                except BaseException:
                    CANCEL.set()
                    raise
        assert sources() == manifest['source_sha256']
        dev = aggregate('development', RULES)
        primary = dev['arms']['kda']
        def rank(rule):
            x = primary[rule]['overall']['counts']
            return (-x['success'], x['collision'], x['timeout'], RULES.index(rule))
        winner = min(RULES, key=rank)
        save(BASE / 'winner-frozen.json', dict(winner=winner, protocol_sha256=manifest['protocol_sha256'],
                                              selection_data_sha256=sha(BASE / 'development-summary.json')))
        final = None
        verdict = 'NO_ALTERNATIVE_SELECTED'
        if winner != 'first5':
            save(BASE / 'status.json', dict(stage='CONFIRMATION', winner=winner))
            with futures.ThreadPoolExecutor(max_workers=2) as pool:
                for job in futures.as_completed([pool.submit(block, arm, 'confirmation', ('first5', winner))
                                                for arm in ARMS]):
                    try:
                        job.result()
                    except BaseException:
                        CANCEL.set()
                        raise
            final = aggregate('confirmation', ('first5', winner))
            effect = final['arms']['kda'][winner]['paired']
            good = (effect['success']['delta_pp'] > 0 and effect['success']['bonferroni_p'] < .05
                    and effect['collision']['delta_pp'] <= 0 and effect['timeout']['delta_pp'] <= 0)
            verdict = 'CONFIRMED_SELECTION_GAIN' if good else 'FRESH_CONFIRMATION_NOT_PASSED'
        assert sources() == manifest['source_sha256']
        assert sha(BASE / 'protocol.json') == manifest['protocol_sha256']
        report(dev, winner, final, verdict)
        save(BASE / 'DONE.json', dict(status='COMPLETED', verdict=verdict, winner=winner,
             wall_seconds=time.time() - START, source_unchanged=True, training='NONE'))
        save(BASE / 'status.json', dict(stage='COMPLETED', verdict=verdict))
    except BaseException as error:
        CANCEL.set()
        save(BASE / 'ERROR.json', dict(status='ENGINEERING_STOP', error=repr(error),
                                      wall_seconds=time.time() - START))
        save(BASE / 'status.json', dict(stage='ENGINEERING_STOP', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
