"""Bounded fresh-IL history experiment and an isolated safety pilot."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.stats import binomtest

BASE = Path(__file__).resolve().parent
PROJECT = Path('/root/cc-diag-20261008/code')
ASSET = Path('/root/cc-diag-20261008/results')
ARMS = [('kda', 24), ('gru', 24), ('kda', 12), ('gru', 12)]
METRICS = ('success', 'collision', 'timeout')
ACTIVE = []


def save(name, value):
    p = BASE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False)); t.replace(p)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def alive(pid):
    path = Path('/proc/%d/stat' % pid)
    return path.exists() and path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def paired(old, new, family=6):
    a = {(x['cell'], x['case']): x for x in old}
    b = {(x['cell'], x['case']): x for x in new}
    assert a.keys() == b.keys() and len(a) == len(old) == len(new)
    cases = sorted({k[1] for k in a}); cells = sorted({k[0] for k in a})
    result = {}
    for m in METRICS:
        left = sum(a[k]['outcome'] == m and b[k]['outcome'] != m for k in a)
        right = sum(b[k]['outcome'] == m and a[k]['outcome'] != m for k in a)
        p = float(binomtest(left, left + right, .5).pvalue) if left + right else 1.
        delta = np.array([np.mean([float(b[c, case]['outcome'] == m)
                                  - float(a[c, case]['outcome'] == m) for c in cells]) for case in cases])
        draws = 100 * np.random.RandomState(20261009).choice(delta, (5000, len(cases))).mean(axis=1)
        result[m] = dict(b_old_only=left, c_new_only=right, delta_pp=100 * float(delta.mean()),
            exact_mcnemar_p=p, bonferroni_p=min(1., family * p), family_size=family,
            cluster_ci95_pp=np.percentile(draws, [2.5, 97.5]).tolist(),
            family_cluster_ci_pp=np.percentile(draws, [2.5 / family, 100 - 2.5 / family]).tolist())
    return result


def summary(rows):
    counts = {m: sum(x['outcome'] == m for x in rows) for m in METRICS}
    times = [x['time'] for x in rows if x['outcome'] == 'success']
    returns = [x.get('return', x.get('return_value')) for x in rows]
    return dict(n=len(rows), counts=counts, rates_percent={m: 100 * v / len(rows) for m, v in counts.items()},
        mean_return=float(np.mean(returns)), median_minimum_clearance=float(np.median([x['minimum_clearance'] for x in rows])),
        success_time_mean=float(np.mean(times)) if times else None,
        success_time_median=float(np.median(times)) if times else None,
        success_time_p90=float(np.percentile(times, 90)) if times else None)


def aggregate():
    result = dict(main={}, comparisons={}, safety={}, incomplete=[])
    for arm, length in ARMS:
        name = '%s-t%d' % (arm, length)
        paths = [BASE / 'evaluation' / name / ('cell%d.json' % c) for c in range(6)]
        if not all(p.exists() for p in paths):
            result['incomplete'].append(name); continue
        data = [read(p) for p in paths]; rows = sum([d['episodes'] for d in data], [])
        assert all(len(d['episodes']) == 500 and d['seq_len'] == length and d['grid_actions'] == 80 for d in data)
        result['main'][name] = dict(overall=summary(rows),
            cells={str(c): summary(data[c]['episodes']) for c in range(6)}, rows=rows)
    for arm in ('kda', 'gru'):
        a, b = arm + '-t24', arm + '-t12'
        if a in result['main'] and b in result['main']:
            result['comparisons'][arm] = paired(result['main'][a]['rows'], result['main'][b]['rows'])
    for arm in ('kda', 'gru', 'mamba'):
        paths = {r: [BASE / 'safety' / (arm + '-' + r) / ('cell%d.json' % c) for c in range(6)]
                 for r in ('endpoint', 'swept')}
        if not all(p.exists() for group in paths.values() for p in group):
            continue
        result['safety'][arm] = {}
        for rule, group in paths.items():
            rows = sum([read(p)['episodes'] for p in group], [])
            assert len(rows) == 192
            scored = sum(x['scored_steps'] for x in rows)
            unsafe = sum(x['all_unsafe_steps'] for x in rows)
            entry = summary(rows)
            entry.update(scored_steps=scored, all_unsafe_steps=unsafe,
                all_unsafe_fraction=unsafe / scored if scored else None,
                n_safe_histogram=np.sum([x['n_safe_histogram'] for x in rows], axis=0).tolist(),
                cells={str(c): summary([x for x in rows if x['cell'] == c]) for c in range(6)}, rows=rows)
            result['safety'][arm][rule] = entry
        result['safety'][arm]['paired_descriptive'] = paired(result['safety'][arm]['endpoint']['rows'],
                                                           result['safety'][arm]['swept']['rows'], family=9)
    # Full episode evidence remains in individual result files, not duplicated in the report.
    for entry in result['main'].values():
        entry.pop('rows')
    for entry in result['safety'].values():
        for r in ('endpoint', 'swept'):
            entry[r].pop('rows')
    return result


def report(status, results):
    lines = ['# 历史长度与独立扫掠安全实验报告', '', '状态：' + status, '',
        '主线：KDA/GRU × T=12/24，四臂全部从头IL50，同一原始示范池、seed419、共享初始化、5人circle训练；各目标RL10000。',
        '只改变时序长度；T=24旧IL复用权重不作为本轮历史长度因果对照。',
        '评测固定TTC Top-5，91000–91499，六环境各500例。该case块已有使用记录，是开发比较，不冒称新鲜最终确认。',
        '', '## 完成情况', '', '| 臂 | 最新阶段/边界 | 完整训练 | 完整3000评测 | 可恢复证据 |',
        '|---|---|---|---|---|']
    for arm, length in ARMS:
        name = '%s-t%d' % (arm, length); run = BASE / 'runs' / name
        phase = read(run / 'phase.json') if (run / 'phase.json').exists() else {}
        lines.append('| %s | %s / %s | %s | %s | %s |' % (name, phase.get('stage', '未开始'),
            phase.get('number', '-'), (run / 'training-completed.json').exists(),
            name in results['main'], 'RL全状态' if (run / 'resume-latest.pth.gz').exists()
            else 'IL epoch＋RNG' if (run / 'il-resume-state.pth').exists() else '无'))
    lines += ['', '## 完整预算结果', '', '| 臂 | N | SR | CR | TO | 原折扣回报 | 成功平均用时 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, entry in results['main'].items():
        x = entry['overall']; r = x['rates_percent']
        lines.append('| %s | %d | %.2f%% | %.2f%% | %.2f%% | %.6f | %s |' %
            (name, x['n'], r['success'], r['collision'], r['timeout'], x['mean_return'], x['success_time_mean']))
    if not results['main']:
        lines += ['未完成任何全预算＋3000例评测；不能给性能结论。']
    for name, entry in results['main'].items():
        lines += ['', '### ' + name, '| cell | SR | CR | TO |', '|---|---:|---:|---:|']
        for c, x in entry['cells'].items():
            r = x['rates_percent']; lines.append('| %s | %.2f%% | %.2f%% | %.2f%% |' % (c, r['success'], r['collision'], r['timeout']))
    lines += ['', '## T12相对fresh T24的配对比较',
              '| 骨干/指标 | Δpp | 原有→失去 | 新增 | 精确p | 校正p×6 | case聚类95%区间 |',
              '|---|---:|---:|---:|---:|---:|---|']
    for arm, entry in results['comparisons'].items():
        for m, x in entry.items():
            lines.append('| %s/%s | %.3f | %d | %d | %.6g | %.6g | %s |' %
                (arm, m, x['delta_pp'], x['b_old_only'], x['c_new_only'], x['exact_mcnemar_p'],
                 x['bonferroni_p'], x['cluster_ci95_pp']))
    lines += ['', '## 独立扫掠安全辅线',
        '仅冻结旧权重，endpoint/swept两臂，TTC Top-5；每格32例、每臂192例。开发性小样本，不据此宣布方法成功。',
        '只替换安全filter/risk所用clearance；reward、successor与V均保持原值。常速度扫掠并非真实未来ORCA运动。',
        '| 骨干/安全规则 | SR | CR | TO | all-unsafe步骤比例 | 成功平均用时 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, entry in results['safety'].items():
        for rule in ('endpoint', 'swept'):
            x = entry[rule]; r = x['rates_percent']
            lines.append('| %s/%s | %.2f%% | %.2f%% | %.2f%% | %.4f | %s |' %
                (arm, rule, r['success'], r['collision'], r['timeout'], x['all_unsafe_fraction'], x['success_time_mean']))
    if not results['safety']:
        lines += ['没有完整配对辅线结果：资源或时间不足，未完成项保留，不能视为零收益。']
    lines += ['', '## 证据边界与下一步',
        '- 完整训练与完整评测分别核验；部分IL/RL或部分cell不参与最终性能比较。',
        '- 单seed不能证明跨seed稳定性；T变化同时影响学习动态，不能直接证明某种记忆机制。',
        '- 不显著不等于等效；T12接近T24也不证明T48无用。',
        '- 四臂共享GPU；实际耗时和条件成功用时不能单独当作固定加速比或因果效率优势。',
        '- 已完成历史长度比较才决定是否值得后续确认/T48；当前不自动追加任何训练或调参。',
        '- 原TTC结果、旧负结果和母体源码未覆盖。', '', '完整汇总JSON：', str(BASE / 'summary.json'), '']
    (BASE / 'kda-history-overnight-report.md').write_text('\n'.join(lines))


def main():
    lock = (BASE / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    task = read(BASE / 'task-start.json'); started = task['started']
    deadline = started + 8 * 3600; stop_at = deadline - 1800
    cores = sorted(os.sched_getaffinity(0))
    # Validate PID-based liveness for both running and finished states.
    sample = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.1)'])
    assert alive(sample.pid); sample.wait(); assert not alive(sample.pid)
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    save('controller-pid.json', dict(pid=os.getpid(), proc_start_ticks=stat[19], started=time.time()))
    while not (BASE / 'preflight.json').exists():
        if not alive(task['preflight_pid']):
            save('ERROR.json', dict(error='Preflight process exited without PASS; inspect prepare.log'))
            report('PREFLIGHT_FAILED', aggregate()); return
        save('status.json', dict(stage='WAITING_PREFLIGHT', elapsed_seconds=time.time() - started))
        if time.time() >= stop_at:
            save('ERROR.json', dict(error='Preflight consumed budget'))
            report('PREFLIGHT_BUDGET_EXHAUSTED', aggregate()); return
        time.sleep(5)
    preflight = read(BASE / 'preflight.json'); assert preflight['status'] == 'PASS'
    assert read(BASE / 'continuation-preflight.json')['status'] == 'PASS'
    protocol = read(BASE / 'protocol.json')
    save('execution-freeze.json', dict(protocol_sha256=sha(BASE / 'protocol.json'),
        scripts={p.name: sha(p) for p in BASE.glob('*.py')},
        known_resource_limit='At most 3 simultaneous IL phases; four RL arms may overlap',
        actual_deadline=deadline, stop_request_time=stop_at))
    pending = list(ARMS); trained = set(); failed = []
    eval_pending = []; safety_pending = [(a, r, c) for a in ('kda', 'gru', 'mamba')
        for r in ('endpoint', 'swept') for c in range(6)]
    evaluations_added = set(); stop_requested = False
    budget_bytes = 22 * 1024 ** 3

    def train_cost(name):
        phase_path = BASE / 'runs' / name / 'phase.json'
        phase = read(phase_path).get('stage') if phase_path.exists() else 'IL'
        if phase.startswith('RL'):
            return 1024 ** 3
        x = protocol['memory_estimates'][name]
        return x['theoretical_il_tensor_bytes'] + x['minibatch_peak_reserved_bytes'] + 384 * 1024 ** 2

    def launch(command, kind, name, out, affinity_core, **extra):
        log = (BASE / 'logs' / (name + '.log')); log.parent.mkdir(exist_ok=True)
        stream = log.open('w')
        env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        def affinity():
            os.sched_setaffinity(0, {affinity_core})
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 preexec_fn=affinity)
        ACTIVE.append(dict(child=child, stream=stream, kind=kind, name=name, out=out, **extra))

    while time.time() < deadline:
        for job in list(ACTIVE):
            code = job['child'].poll()
            if code is None:
                continue
            job['stream'].close(); ACTIVE.remove(job)
            if job['kind'] == 'train':
                name = job['name']; run = job['out']; final = run / 'rl_model_ep10000.pth'
                if code == 0 and final.exists():
                    payload = torch_load(final)
                    il = torch_load(run / 'il_policy.pth')
                    assert payload['episode'] == 10000 and payload['algo'] == 'sarl'
                    assert il['meta']['epochs'] == 50 and il['meta']['seq_len'] == job['length']
                    assert payload['optim_value_state']['state']
                    save('runs/%s/training-completed.json' % name, dict(final_episode=10000,
                        il_epochs=50, fresh_il=True, checkpoint_sha256=sha(final)))
                    trained.add(name)
                elif code != 75:
                    failed.append(dict(name=name, code=code, log=str(BASE / 'logs' / (name + '.log'))))
            elif code != 0:
                failed.append(dict(name=job['name'], code=code))
        if time.time() >= stop_at and not stop_requested:
            report('FINALIZING_AT_BUDGET_BOUNDARY', aggregate())
            for job in ACTIVE:
                if job['kind'] == 'train':
                    (job['out'] / 'STOP').touch()
            stop_requested = True
        if shutil.disk_usage(BASE).free < 1024 ** 3:
            for job in ACTIVE:
                if job['kind'] == 'train':
                    (job['out'] / 'STOP').touch()
            stop_requested = True
            save('disk-stop.json', dict(free_bytes=shutil.disk_usage(BASE).free))
        active_train = [j for j in ACTIVE if j['kind'] == 'train']
        il_count = sum(train_cost(j['name']) > 1024 ** 3 for j in active_train)
        used_cost = sum(train_cost(j['name']) if j['kind'] == 'train' else 512 * 1024 ** 2 for j in ACTIVE)
        if pending and not stop_requested and il_count < 3:
            arm, length = pending[0]; name = '%s-t%d' % (arm, length)
            cost = train_cost(name)
            if used_cost + cost < budget_bytes:
                pending.pop(0); run = BASE / 'runs' / name
                assert not run.exists(), 'Fresh arm output already exists'
                command = [sys.executable, str(BASE / 'train-worker.py'), '--project', str(PROJECT),
                    '--deadline', str(stop_at), '--config', str(BASE / (name + '.ini')), '--outdir', str(run),
                    '--device', 'cuda', '--seed', '419', '--temporal-backbone', arm]
                launch(command, 'train', name, run, cores[ARMS.index((arm, length))], arm=arm, length=length)
                used_cost += cost
                il_count += 1
        for name in sorted(trained - evaluations_added):
            arm, t = name.split('-t'); checkpoint = BASE / 'runs' / name / 'rl_model_ep10000.pth'
            eval_pending.extend((name, arm, c, sha(checkpoint)) for c in range(6))
            evaluations_added.add(name)
        eval_active = [j for j in ACTIVE if j['kind'] != 'train']
        if not stop_requested and len(eval_active) < 3 and used_cost + 512 * 1024 ** 2 < budget_bytes:
            free_cores = [c for c in cores[4:7] if c not in [j.get('core') for j in eval_active]]
            if free_cores and eval_pending:
                name, arm, cell, digest = eval_pending.pop(0)
                out = BASE / 'evaluation' / name / ('cell%d.json' % cell)
                command = [sys.executable, str(BASE / 'evaluate-history.py'), '--project', str(PROJECT),
                    '--arm-dir', str(BASE / 'runs' / name), '--config', str(BASE / (name + '.ini')),
                    '--backbone', arm, '--human-select', 'ttc5', '--expected-checkpoint-sha', digest,
                    '--case-base', '91000', '--n-cases', '500', '--cell', str(cell), '--out', str(out)]
                launch(command, 'evaluate', name + '-cell%d' % cell, out, free_cores[0], core=free_cores[0])
            elif free_cores and safety_pending and il_count <= 1 and not eval_pending:
                arm, rule, cell = safety_pending.pop(0)
                out = BASE / 'safety' / (arm + '-' + rule) / ('cell%d.json' % cell)
                command = [sys.executable, str(BASE / 'evaluate-safety.py'), '--project', str(PROJECT),
                    '--arm-dir', str(ASSET / arm), '--config', str(ASSET / (arm + '.ini')),
                    '--backbone', arm, '--safety', rule, '--cell', str(cell), '--case-base', '91000',
                    '--n-cases', '32', '--out', str(out)]
                launch(command, 'safety', arm + '-' + rule + '-cell%d' % cell, out, free_cores[0], core=free_cores[0])
        save('status.json', dict(stage='STOP_REQUESTED' if stop_requested else 'RUNNING',
            elapsed_seconds=time.time() - started, remaining_seconds=max(0., deadline - time.time()),
            active=[dict(pid=j['child'].pid, kind=j['kind'], name=j['name']) for j in ACTIVE],
            pending_training=pending, trained=sorted(trained), pending_eval=len(eval_pending),
            pending_safety=len(safety_pending), failures=failed))
        if not ACTIVE and (stop_requested or (not pending and not eval_pending and not safety_pending)):
            break
        time.sleep(10)
    # Bound owned workers; at stop_at they were already asked to save at safe boundaries.
    for job in ACTIVE:
        if job['child'].poll() is None:
            job['child'].terminate()
    for job in ACTIVE:
        try:
            job['child'].wait(timeout=20)
        except subprocess.TimeoutExpired:
            job['child'].kill(); job['child'].wait()
        job['stream'].close()
    result = aggregate()
    result['failures'] = failed
    current = {str(p.relative_to(PROJECT)): sha(p) for p in PROJECT.rglob('*.py')}
    assert current == protocol['source_sha256']
    save('summary.json', result)
    status = ('COMPLETED' if len(result['safety']) == 3 else 'MAIN_COMPLETE_AUXILIARY_PARTIAL') if not result['incomplete'] and not failed else 'PARTIAL_OR_BLOCKED'
    report(status, result)
    save('DONE.json', dict(status=status, wall_seconds=time.time() - started,
        incomplete=result['incomplete'], failures=failed, source_unchanged=True,
        safety_completed_arms=list(result['safety']), no_new_experiments_after_deadline=True))
    save('status.json', dict(stage=status, complete_main_arms=list(result['main']),
                             incomplete=result['incomplete'], failures=failed))


def torch_load(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        for job in ACTIVE:
            if job['kind'] == 'train':
                (job['out'] / 'STOP').touch()
            elif job['child'].poll() is None:
                job['child'].terminate()
        save('ERROR.json', dict(error=repr(error)))
        raise
