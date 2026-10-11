"""Fail-closed H1 -> H3 -> H2 pipeline. Never trains or touches final cases."""
import sys
sys.dont_write_bytecode = True
import configparser
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import binomtest

BASE = Path(__file__).resolve().parent
PROJECT = Path('/root/cc-diag-20261008/code')
ASSETS = BASE / 'weights'
ARMS = ('kda', 'mamba', 'gru')
METRICS = ('success', 'collision', 'timeout')
ACTIVE = []


def save(name, value):
    p = BASE / name; p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); tmp.replace(p)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def alive(pid):
    p = Path('/proc/%d/stat' % pid)
    return p.exists() and p.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def freeze():
    local = read(BASE / 'local-core-source-sha256.json')
    for relative in ('crowd_nav/contracts.py', 'crowd_nav/policy/mamba_rl.py',
                     'crowd_nav/policy/kda_temporal.py', 'crowd_sim/envs/crowd_sim.py'):
        assert sha(PROJECT / relative) == local[relative], ('local/server source mismatch', relative)
    from importlib import util
    spec = util.spec_from_file_location('lookahead_worker', BASE / 'evaluate-lookahead.py')
    module = util.module_from_spec(spec); spec.loader.exec_module(module)
    weights = {}
    for arm in ARMS:
        weight = ASSETS / arm / 'rl_model_ep10000.pth'
        digest = sha(weight); assert digest and digest == module.ref.EXPECTED[arm]
        weights[str(weight)] = digest
        config = configparser.RawConfigParser(); assert config.read(str(BASE / 'original-configs' / (arm + '.ini')))
        for alpha in (0., .3):
            config.set('eval_protocol', 'action_smoothing', str(alpha))
            p = BASE / 'configs' / ('%s-alpha%s.ini' % (arm, alpha))
            p.parent.mkdir(exist_ok=True)
            with p.open('w') as f:
                config.write(f)
    protocol = dict(version='kda-lookahead-diag-v1-20261009', train_episodes=0,
        baseline='ttc5; selection anchored to actual current state, frozen within all candidates',
        case_base=90000, cases_per_cell=500, cells=module.ref.CELLS, forbidden_case_blocks=[[91000, 91499], [92000, 92499]],
        H1=dict(arms=list(ARMS), alpha=[.3, 0.], n=18000),
        H3=dict(arm='kda', alpha=.3, cases_per_cell=50, n=300, tail_K=20,
            oracle='actual next human observations after the same root real step; root/history/previous action restored for shadow scoring',
            deployment='CV unchanged; oracle never executed', no_expansion_below_flip_fraction=.01),
        H2=dict(arms=['kda', 'mamba'], candidate_counts=[80, 81], zero_stop_index=80,
            original_80_mapping_preserved=True, alpha_selection='H1 corrected Pareto winner; otherwise alpha0 primary and alpha0.3 descriptive secondary',
            total_per_alpha=12000, additional_per_alpha=6000, reuse='exact H1 same-config same-case 80-action baseline',
            extra_candidate_scored_separately='preserves bitwise values/windows of original 80; filter recomputed over all81',
            adoption='KDA SR gain corrected p<0.05, observed CR and TO no higher than baseline; this does not prove statistical noninferiority'),
        family=dict(arm='kda', tests=6, comparisons=['H1 off vs on', 'H2 primary81 vs80'], metrics=list(METRICS),
            alpha=.05 / 6., statistic='exact paired McNemar; episode key=(cell,case)',
            case_clustering='same numeric case repeated across cells: additionally report case-cluster bootstrap intervals; pooled McNemar independence remains a limitation'),
        H1_winner='at least one corrected favorable SR/CR/TO change and no corrected unfavorable change; mixed tradeoffs mean no clear winner',
        controls='Mamba/GRU descriptive only, never included in testing family or adoption',
        smoke=dict(cases_per_cell=3, all_cells=True, every_variant_before_formal=True),
        no_novelty_claim=True, no_training_testing_consistency_claim=True,
        H3_negative_scope='only current frozen value consumer at0.25s; not all prediction/multistep planning',
        final_choice='at most one; among validated H1/H2 Pareto gains choose larger KDA SR gain, ties favor H1; otherwise H3 sensitivity is only an unvalidated lead')
    assert not (BASE / 'protocol.json').exists(), 'Protocol already frozen'
    save('protocol.json', protocol)
    manifest = dict(weights=weights, source={str(p): sha(p) for p in PROJECT.rglob('*.py')},
        original_configs={str(BASE / 'original-configs' / (a + '.ini')): sha(BASE / 'original-configs' / (a + '.ini')) for a in ARMS},
        configs={str(p): sha(p) for p in (BASE / 'configs').glob('*.ini')},
        scripts={str(p): sha(p) for p in BASE.glob('*.py')}, protocol_sha256=sha(BASE / 'protocol.json'))
    save('frozen-manifest.json', manifest)
    verify_freeze()


def verify_freeze():
    m = read(BASE / 'frozen-manifest.json')
    assert sha(BASE / 'protocol.json') == m['protocol_sha256']
    for group in ('weights', 'source', 'original_configs', 'configs', 'scripts'):
        for p, expected in m[group].items():
            assert expected and sha(p) == expected, (group, p)


def output_path(stage, arm, alpha, stop, cell):
    name = '%s-alpha%s-a%d' % (arm, alpha, 81 if stop else 80)
    return BASE / stage / 'per-case' / name / ('cell%d.json' % cell)


def tasks(stage, arms, alphas, stop_options, n, smoke=False):
    result = []
    for arm in arms:
        for alpha in alphas:
            for stop in stop_options:
                for cell in range(6):
                    out = output_path(stage, arm, alpha, stop, cell)
                    cmd = [sys.executable, '-B', str(BASE / 'evaluate-lookahead.py'), '--project', str(PROJECT),
                        '--arm-dir', str(ASSETS / arm), '--config', str(BASE / 'configs' / ('%s-alpha%s.ini' % (arm, alpha))),
                        '--backbone', arm, '--alpha', str(alpha), '--cell', str(cell), '--n-cases', str(n),
                        '--case-base', '90000', '--out', str(out)]
                    if stop: cmd.append('--stop')
                    if 'h3' in stage: cmd.append('--oracle')
                    if smoke: cmd.append('--smoke')
                    result.append(dict(command=cmd, output=out, name=stage + '-' + out.parent.name + '-cell%d' % cell,
                        expected_n=n, candidate_count=80 + int(stop)))
    return result


def run_jobs(queue, stage):
    queue = list(queue); completed = []; started = time.time()
    cores = sorted(os.sched_getaffinity(0))
    while queue or ACTIVE:
        while queue and len(ACTIVE) < min(6, len(cores)):
            job = queue.pop(0)
            assert not job['output'].exists(), ('Output exists', job['output'])
            slot = next(i for i in range(min(6, len(cores))) if i not in [x['slot'] for x in ACTIVE])
            log = BASE / 'logs' / (job['name'] + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open('w'); env = os.environ.copy()
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            def affinity(core=cores[slot]): os.sched_setaffinity(0, {core})
            child = subprocess.Popen(job['command'], cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, preexec_fn=affinity)
            job.update(child=child, stream=stream, slot=slot, log=log, started=time.time()); ACTIVE.append(job)
        for job in list(ACTIVE):
            code = job['child'].poll()
            if code is None:
                assert time.time() - job['started'] < 3600, ('job timeout', job['name'])
                continue
            job['stream'].close(); ACTIVE.remove(job)
            evidence = dict(name=job['name'], exit_code=code, output=str(job['output']),
                output_bytes=job['output'].stat().st_size if job['output'].exists() else 0,
                log=str(job['log']), actual_stdout_tail=job['log'].read_text(errors='replace')[-1500:])
            save('job-evidence/%s.json' % job['name'], evidence)
            assert code == 0 and evidence['output_bytes'] > 0, evidence
            assert 'Traceback (most recent call last)' not in job['log'].read_text(errors='replace'), evidence
            data = read(job['output'])
            assert data['status'] == '已验证通过' and len(data['episodes']) == job['expected_n']
            assert data['grid_actions'] == 80 and data['candidate_actions'] == job['candidate_count']
            assert data['selection'] == 'ttc5'
            assert data['checks']['selection'] == data['checks']['scores'] == data['checks']['history'] > 0
            assert all(x['outcome'] in METRICS and np.isfinite(x['return_value']) for x in data['episodes'])
            completed.append(evidence)
        save('status.json', dict(status='未验证', stage=stage, active=[dict(name=x['name'], pid=x['child'].pid) for x in ACTIVE],
            pending=len(queue), completed_jobs=len(completed), wall_seconds=time.time() - started))
        time.sleep(2)
    verify_freeze()
    save(stage + '/validation.json', dict(status='已验证通过', jobs=completed, wall_seconds=time.time() - started))


def rows(stage, arm, alpha, stop):
    data = [read(output_path(stage, arm, alpha, stop, c)) for c in range(6)]
    result = sum([x['episodes'] for x in data], [])
    assert len({(x['cell'], x['case']) for x in result}) == len(result)
    return result


def summarize(data):
    rates = {m: 100 * sum(x['outcome'] == m for x in data) / len(data) for m in METRICS}
    times = [x['time'] for x in data if x['outcome'] == 'success']
    return dict(n=len(data), rates=rates, mean_return=float(np.mean([x['return_value'] for x in data])),
        success_time_mean=float(np.mean(times)) if times else None,
        minimum_clearance_median=float(np.median([x['minimum_clearance'] for x in data])),
        stop_selections=sum(x['stop_selections'] for x in data),
        cells={str(c): {'n': 500, **{m: 100 * sum(x['outcome'] == m for x in data if x['cell'] == c) / 500 for m in METRICS}} for c in range(6)})


def paired(a, b):
    a = {(x['cell'], x['case']): x for x in a}; b = {(x['cell'], x['case']): x for x in b}
    assert a.keys() == b.keys() and len(a) == 3000
    result = {}
    for m in METRICS:
        lost = sum(a[k]['outcome'] == m and b[k]['outcome'] != m for k in a)
        gained = sum(b[k]['outcome'] == m and a[k]['outcome'] != m for k in a)
        p = float(binomtest(lost, lost + gained, .5).pvalue) if lost + gained else 1.
        cases = sorted({k[1] for k in a})
        diffs = np.array([np.mean([float(b[cell, case]['outcome'] == m) - float(a[cell, case]['outcome'] == m)
                                  for cell in range(6)]) for case in cases])
        draws = 100 * np.random.RandomState(20261009).choice(diffs, (10000, len(diffs))).mean(axis=1)
        result[m] = dict(delta_pp=100 * float(diffs.mean()), b=lost, c=gained, exact_p=p,
            bonferroni6_p=min(1., 6 * p), significant=6 * p < .05,
            case_cluster_ci95_pp=np.percentile(draws, [2.5, 97.5]).tolist(),
            case_cluster_family_ci_pp=np.percentile(draws, [.4166666667, 99.5833333333]).tolist())
    return result


def h1_summary():
    output = dict(status='已验证通过', arms={}, kda_paired={})
    for arm in ARMS:
        output['arms'][arm] = {str(alpha): summarize(rows('h1', arm, alpha, False)) for alpha in (.3, 0.)}
    tests = paired(rows('h1', 'kda', .3, False), rows('h1', 'kda', 0., False))
    output['kda_paired'] = tests
    favorable_off = [m for m, x in tests.items() if x['significant'] and ((x['delta_pp'] > 0) == (m == 'success'))]
    favorable_on = [m for m, x in tests.items() if x['significant'] and ((x['delta_pp'] < 0) == (m == 'success'))]
    winner = 0. if favorable_off and not favorable_on else .3 if favorable_on and not favorable_off else None
    output.update(clear_winner_alpha=winner, H2_primary_alpha=winner if winner is not None else 0.,
                  H2_secondary_alpha=.3 if winner is None else None,
                  no_training_testing_consistency_inference=True)
    save('h1/summary.json', output)
    return output


def distribution(values):
    a = np.array(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.median(a)), p90=float(np.percentile(a, 90)), max=float(a.max())) if len(a) else dict(n=0, mean=None, p50=None, p90=None, max=None)


def h3_summary():
    observations = []
    for c in range(6):
        path = output_path('h3', 'kda', .3, False, c).with_suffix('.steps.jsonl.gz')
        with gzip.open(path, 'rt') as stream: observations.extend(json.loads(line) for line in stream)
    groups = {'all_steps': observations}
    for outcome in METRICS:
        groups[outcome + '_all_steps'] = [x for x in observations if x['outcome'] == outcome]
        groups[outcome + '_last20_steps'] = [x for x in observations if x['outcome'] == outcome and x['steps_to_terminal'] < 20]
    output = dict(status='已验证通过', n_episodes=300, tail_K=20, groups={},
        evidence_scope='shadow action sensitivity, no oracle navigation outcome claim',
        limits='Negative result only concerns this frozen0.25s consumer. Studying longer-horizon prediction usefulness requires a consumer with an extended horizon, not this test.')
    for name, data in groups.items():
        flips = [x for x in data if x['flipped']]
        entry = dict(n_steps=len(data), flipped_steps=len(flips), flip_fraction=len(flips) / len(data) if data else None,
            value_argmax_flip_fraction=sum(x['value_argmax_flipped'] for x in data) / len(data) if data else None,
            value_kendall_tau=distribution([x['value_kendall_tau'] for x in data if x['value_kendall_tau'] is not None]),
            top5_overlap=distribution([x['top5_overlap'] for x in data]), top10_overlap=distribution([x['top10_overlap'] for x in data]))
        for key in ('position_errors_all', 'velocity_errors_all', 'position_errors_selected', 'velocity_errors_selected'):
            entry[key] = distribution([y for x in data for y in x[key]])
        entry['flips'] = {key: distribution([x[key] for x in flips]) for key in
            ('cv_value_gap', 'oracle_value_gap', 'cv_clearance_gap', 'oracle_clearance_gap')}
        output['groups'][name] = entry
    output['no_expansion'] = output['groups']['all_steps']['flip_fraction'] < .01
    save('h3/summary.json', output)
    return output


def h2_summary(h1):
    output = dict(status='已验证通过', primary_alpha=h1['H2_primary_alpha'], arms={},
        not_novelty_even_if_5pp=True, negative_is_inconclusive_untrained_zero_condition=True)
    alphas = [h1['H2_primary_alpha']] + ([.3] if h1['H2_secondary_alpha'] is not None else [])
    for alpha in alphas:
        entry = {}
        for arm in ('kda', 'mamba'):
            old = rows('h1', arm, alpha, False); new = rows('h2', arm, alpha, True)
            entry[arm] = dict(baseline=summarize(old), appended_stop=summarize(new))
            if arm == 'kda' and alpha == h1['H2_primary_alpha']:
                entry[arm]['paired'] = paired(old, new)
        output['arms'][str(alpha)] = entry
    primary = output['arms'][str(output['primary_alpha'])]['kda']
    tests = primary['paired']
    output['adoption_supported'] = (tests['success']['delta_pp'] > 0 and tests['success']['significant']
        and tests['collision']['delta_pp'] <= 0 and tests['timeout']['delta_pp'] <= 0)
    save('h2/summary.json', output)
    return output


def report(status, error=None):
    h1 = read(BASE / 'h1/summary.json') if (BASE / 'h1/summary.json').exists() else None
    h3 = read(BASE / 'h3/summary.json') if (BASE / 'h3/summary.json').exists() else None
    h2 = read(BASE / 'h2/summary.json') if (BASE / 'h2/summary.json').exists() else None
    lines = ['# KDA前瞻层机制诊断', '', '状态：' + status, '',
        '## 现象', '零训练；冻结旧ep10000权重，TTC Top-5。只使用90000–90499开发块；91000与92000保留块本轮未使用。',
        '## 问题', 'H1：测试期平滑是否有收益；H3：同状态真实一步行人后继是否改变排序；H2：追加真停止动作是否有收益。',
        '## 证据与原因', '来源：各实验summary.json、per-case目录、h3压缩逐步记录、job-evidence退出码与实际日志；每项配对严格匹配(cell,case)。',
        '确证家族只含KDA的H1/H2各SR/CR/TO共6项；Mamba/GRU仅描述。合并McNemar之外报告case聚类区间，避免忽略跨cell相同case的相关性。']
    if error: lines += ['已确认失败：' + repr(error), '发生非零退出或断言失败后停止后续步骤，失败输出保留。']
    if h1:
        lines += ['', '### H1 测试平滑', '| 骨干 | alpha | N | SR | CR | TO |', '|---|---:|---:|---:|---:|---:|']
        for arm, entries in h1['arms'].items():
            for alpha, x in entries.items():
                r = x['rates']; lines.append('| %s | %s | %d | %.2f | %.2f | %.2f |' % (arm, alpha, x['n'], r['success'], r['collision'], r['timeout']))
        lines += ['已验证通过：完整评测。明确优胜alpha=' + str(h1['clear_winner_alpha']),
            '未验证：训练期×测试期一致性建模价值。本轮没有二维训练对照，不能据H1作判断。']
    if h3:
        lines += ['', '### H3 同状态oracle影子重评分（300例）', '| 分组 | 步数 | argmax翻转率 | 选中行人位置误差均值 | 选中行人速度误差均值 |', '|---|---:|---:|---:|---:|']
        for name, x in h3['groups'].items():
            lines.append('| %s | %d | %s | %s | %s |' % (name, x['n_steps'], x['flip_fraction'], x['position_errors_selected']['mean'], x['velocity_errors_selected']['mean']))
        lines += ['已验证通过：误差与排序敏感性统计。未验证：oracle改变动作后的闭环收益，影子排序不是导航收益。',
            '若翻转率<1%，不扩规模；负结果只支持当前冻结网络在0.25秒前瞻下对该替换不敏感，不淘汰预测或多步前瞻。',
            '若要研究预测精度的控制价值，应同步延长消费者前瞻时域；本轮不做。']
    if h2:
        lines += ['', '### H2 追加index80真停止', '| alpha | 骨干 | 候选 | N | SR | CR | TO |', '|---|---|---:|---:|---:|---:|---:|']
        for alpha, entries in h2['arms'].items():
            for arm, entries2 in entries.items():
                for key, count in (('baseline', 80), ('appended_stop', 81)):
                    x = entries2[key]; r = x['rates']; lines.append('| %s | %s | %d | %d | %.2f | %.2f | %.2f |' % (alpha, arm, count, x['n'], r['success'], r['collision'], r['timeout']))
        lines += ['主实验alpha=' + str(h2['primary_alpha']) + '；80候选基线严格复用H1，原80动作的window与value不变。',
            '正向采纳条件成立：' + str(h2['adoption_supported']),
            '持平或变差只能记未验证，不能否定stop：冻结价值网络未训练零速度动作条件。即使改善5pp也只是性能开发，不等于论文创新。']
    lines += ['', '### KDA配对检验', '| 实验/指标 | Δpp | b原有失去 | c新增 | 精确p | Bonferroni×6 | case聚类95%区间 |', '|---|---:|---:|---:|---:|---:|---|']
    comparisons = {}
    if h1: comparisons['H1 off-on'] = h1['kda_paired']
    if h2: comparisons['H2 stop-parent'] = h2['arms'][str(h2['primary_alpha'])]['kda']['paired']
    for name, entries in comparisons.items():
        for metric, x in entries.items():
            lines.append('| %s/%s | %.3f | %d | %d | %.6g | %.6g | %s |' % (name, metric, x['delta_pp'], x['b'], x['c'], x['exact_p'], x['bonferroni6_p'], x['case_cluster_ci95_pp']))
    candidates = []
    if h1 and h1['clear_winner_alpha'] == 0.:
        change = h1['kda_paired']['success']['delta_pp']
        candidates.append((change, 'H1 测试平滑机制'))
    if h2 and h2['adoption_supported']:
        candidates.append((h2['arms'][str(h2['primary_alpha'])]['kda']['paired']['success']['delta_pp'], 'H2 真停止动作'))
    lines += ['', '## 可选方案与推荐']
    if candidates:
        chosen = sorted(candidates, key=lambda x: -x[0])[0][1]
        lines += ['最多保留一个：' + chosen + '。开发块证据不等于独立最终确认或新颖性，不自动启动任何训练。']
    elif h1 and h2 and h3:
        if h1['clear_winner_alpha'] == .3:
            lines += ['H1支持保留既有alpha0.3，这是部署收益证据，不是新增平滑缺陷或新方法入口。']
        if h3['no_expansion']:
            lines += ['本轮三项没有足以支持前瞻层方法立项的有力线索。结束本轮，不强行将前瞻层写成论文主题；其他候选方向等待用户决定，不自动搜索或训练。']
        else:
            lines += ['H1/H2未建立可采纳收益。唯一保留H3后继敏感性为未验证线索；没有闭环oracle收益证据，不能据此进入算法训练。']
    else: lines += ['未验证：尚未完成三项，不能作机制采纳判定。']
    lines += ['当前KDA-VL不等于第一篇Mamba-VL投稿代码；L0字段表不作为第一篇勘误。',
        '原始逐步数据没有跨线借用；训练位置日志与冻结评测动作记录分开。完整来源审计见L0-C-provenance.json。',
        '实际输出、文件大小及退出码见job-evidence/；所有负结果保留。']
    (BASE / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    lock = (BASE / 'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe = subprocess.Popen([sys.executable, '-B', '-c', 'import time;time.sleep(.1)'])
    assert alive(probe.pid); probe.wait(); assert not alive(probe.pid)
    save('controller-pid.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]))
    if not (BASE / 'protocol.json').exists(): freeze()
    verify_freeze(); report('未验证')
    assert read(BASE / 'L0-C-provenance.json')['status'] == '已验证通过'
    smoke = tasks('smoke/h1', ARMS, (.3, 0.), (False,), 3, True)
    smoke += tasks('smoke/h3', ('kda',), (.3,), (False,), 3, True)
    smoke += tasks('smoke/h2', ('kda', 'mamba'), (0., .3), (True,), 3, True)
    run_jobs(smoke, 'smoke')
    save('smoke/summary.json', dict(status='已验证通过', n_episodes=198, all_variants_tested=True))
    run_jobs(tasks('h1', ARMS, (.3, 0.), (False,), 500), 'h1')
    h1 = h1_summary(); report('未验证')
    run_jobs(tasks('h3', ('kda',), (.3,), (False,), 50), 'h3')
    h3_summary(); report('未验证')
    alphas = [h1['H2_primary_alpha']] + ([.3] if h1['H2_secondary_alpha'] is not None else [])
    run_jobs(tasks('h2', ('kda', 'mamba'), alphas, (True,), 500), 'h2')
    h2_summary(h1); verify_freeze(); report('已验证通过')
    save('DONE.json', dict(status='已验证通过', experiments=['H1', 'H3', 'H2'], final_cases_used=0, train_episodes=0,
        frozen_hashes_verified=True, all_sources_unchanged=True))
    save('status.json', dict(status='已验证通过', stage='DONE', active=[], pending=0))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        for job in ACTIVE:
            if job['child'].poll() is None: job['child'].terminate()
        for job in ACTIVE:
            try: job['child'].wait(timeout=20)
            except subprocess.TimeoutExpired: job['child'].kill(); job['child'].wait()
            job['stream'].close()
        save('ERROR.json', dict(status='已确认失败', error=repr(error), fail_closed=True))
        report('已确认失败', error)
        raise
