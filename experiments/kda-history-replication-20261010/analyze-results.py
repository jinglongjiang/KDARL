"""Pre-registered seed443 interaction analysis; no automatic stage two."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def classify(result):
    d = result['interaction_pp']
    lower, upper = result['descriptive_case_cluster_ci95_pp']
    if d <= 0 or lower <= 0 <= upper or lower <= 0:
        return 'NOT_CONFIRMED', ['D is not positive with its entire 95% interval above zero']
    k = result['group_tests']['kda']['success']
    g = result['group_tests']['gru']['success']
    reasons = []
    if k['delta_t24_minus_t12_pp'] <= 0 or g['delta_t24_minus_t12_pp'] >= 0:
        reasons.append('At least one within-backbone SR sign did not replicate')
    if k['prior_family6_corrected_p'] >= .05 or g['prior_family6_corrected_p'] >= .05:
        reasons.append('At least one within-backbone SR test did not pass the six-test correction')
    if result['marginal_log_failure_odds_interaction'] >= 0:
        reasons.append('SR percentage-point and failure-log-odds scales disagree')
    return ('PARTIAL', reasons) if reasons else ('REPLICATED_PRELIMINARY', [])


def summary(rows):
    counts = {m: sum(x['outcome'] == m for x in rows) for m in ('success', 'collision', 'timeout')}
    assert sum(counts.values()) == len(rows)
    times = [x['time'] for x in rows if x['outcome'] == 'success']
    return dict(n=len(rows), counts=counts, rates_percent={m: 100*n/len(rows) for m,n in counts.items()},
                mean_return=float(np.mean([x['return'] for x in rows])),
                median_minimum_clearance=float(np.median([x['minimum_clearance'] for x in rows])),
                success_time_mean=float(np.mean(times)) if times else None,
                success_time_median=float(np.median(times)) if times else None,
                success_time_p90=float(np.percentile(times,90)) if times else None)


def main():
    spec = importlib.util.spec_from_file_location('interaction', BASE / 'analyze-seed419.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    protocol = read(BASE / 'protocol.json')
    assert protocol['seed'] == 443 and protocol['stage2_authorized'] is False
    result = module.analyze(BASE)
    result.update(seed=443, scope='Second training seed, same development cases; not independent case confirmation',
                  primary_family_size=1, secondary_family_size=6, stage2_started=False,
                  interaction_test='Directional D>0 evaluated using paired case-cluster two-sided 95% bootstrap interval as specified',
                  scale_direction_definition='SR D>0 corresponds to failure-log-odds interaction<0')
    prior = read(BASE / 'seed419-interaction.json')
    verdict, reasons = classify(result)
    prior_verdict, prior_reasons = classify(prior)
    if verdict == 'REPLICATED_PRELIMINARY' and prior_verdict != 'REPLICATED_PRELIMINARY':
        verdict = 'PARTIAL'; reasons.append('Existing seed419 failed the same signs and secondary checks')
    result.update(verdict=verdict, reasons=reasons, previous_seed419=prior,
                  cross_seed='n=2: sign replication only; no architecture-level seed variance or pooled significance claim',
                  arms={})
    result['limits'][0] = 'D is the single primary test; six within-backbone outcome tests are a separate secondary family'
    for arm in ('kda','gru'):
        for t in (12,24):
            name = '%s-t%d' % (arm,t)
            cells = [read(BASE/'evaluation'/name/('cell%d.json'%c)) for c in range(6)]
            final = read(BASE/'runs'/name/'training-completed.json')
            assert all(x['checkpoint_sha256'] == final['checkpoint_sha256'] for x in cells)
            rows = sum([x['episodes'] for x in cells],[])
            result['arms'][name] = dict(overall=summary(rows),
                cells={str(c):summary(cells[c]['episodes']) for c in range(6)},
                twenty_person=summary([x for x in rows if x['human_num']==20]),
                checkpoint_sha256=final['checkpoint_sha256'],
                training=read(BASE/'runs'/name/'worker-completed.json'))
    (BASE/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# KDA/GRU历史长度第二seed复核','', '状态：'+verdict,
        '', '## 范围','seed443，四臂从头IL50+RL10000，仅5人circle训练；TTC5、六格各500例。',
        '91000-91499与seed419相同，可比但非独立case确认；92000-92499未使用。',
        '无T48、无新模型、无reward/filter/TTC/动作/学习率修改，阶段二未授权且未启动。',
        '', '## 完整结果','| 臂 | N | SR | CR | TO | 原折扣回报 | 成功平均用时，秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,e in result['arms'].items():
        x=e['overall']; r=x['rates_percent']
        lines.append('| %s | %d | %.2f%% | %.2f%% | %.2f%% | %.6f | %.3f |' %
            (name,x['n'],r['success'],r['collision'],r['timeout'],x['mean_return'],x['success_time_mean']))
    lines += ['', '## 主交互检验',
        'D=(SR_KDA24-SR_KDA12)-(SR_GRU24-SR_GRU12)，正向为预设方向。四臂、六格按同case共同重采样；500个case聚类、10000次bootstrap，主检验只有一个，不作Bonferroni。',
        '| Training seed | D，pp | case聚类95%区间，pp | 对数失败几率交互 |', '|---|---:|---|---:|']
    for seed,x in ((419,prior),(443,result)):
        lines.append('| %d | %.6f | %s | %.6f |' %
            (seed,x['interaction_pp'],x['descriptive_case_cluster_ci95_pp'],x['marginal_log_failure_odds_interaction']))
    lines += ['', '## 次要检验','T24减T12，两个骨干各SR/CR/TO，六项Bonferroni；per-cell和20人仅描述。',
        '| 骨干/指标 | Δpp | b，T12独有 | c，T24独有 | 精确McNemar p | 校正p×6 |', '|---|---:|---:|---:|---:|---:|']
    for arm,tests in result['group_tests'].items():
        for metric,x in tests.items():
            lines.append('| %s/%s | %.6f | %d | %d | %.8g | %.8g |' %
                (arm,metric,x['delta_t24_minus_t12_pp'],x['b_t12_only'],x['c_t24_only'],x['exact_mcnemar_p'],x['prior_family6_corrected_p']))
    lines += ['', '## 尺度稳健性','成功率pp与失败对数几率的数值符号应相反，才代表同一物理方向。报告的是边际失败几率比，不是McNemar配对几率比。',
        '| 骨干 | T12失败率 | T24失败率 | 失败OR，T24/T12 |','|---|---:|---:|---:|']
    for arm,x in result['failure_odds'].items():
        lines.append('| %s | %.3f%% | %.3f%% | %.6f |' %
            (arm,x['failure_percent_t12'],x['failure_percent_t24'],x['marginal_failure_odds_ratio_t24_vs_t12']))
    lines += ['', '## 分环境，描述性','| 臂 | cell | N | SR | CR | TO |','|---|---:|---:|---:|---:|---:|']
    for name,e in result['arms'].items():
        for cell,x in e['cells'].items():
            r=x['rates_percent']; lines.append('| %s | %s | %d | %.2f%% | %.2f%% | %.2f%% |' %
                (name,cell,x['n'],r['success'],r['collision'],r['timeout']))
    lines += ['', '## 裁决',verdict, '；'.join(reasons) if reasons else '达到预设初步复现门槛，仍需用户再次授权才可开展M1。',
        '只有两个training seed，不能宣称跨seed统计稳定性，不能估计架构层seed方差，也不能把T敏感性直接当作记忆机制因果证据。',
        '即使D显著，如果一臂SR未通过次要校正，也按PARTIAL处理，不据此进入阶段二。',
        '', '## 来源','protocol.json、summary.json、evaluation四臂24份cell JSON、runs/*/training-completed.json、job-evidence/。',
        '完整预算与源码hash由控制器复验；归档和远端权重清理的最终状态见archive-verification.json及remote-cleanup.json。']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='已验证通过',verdict=verdict,D_pp=result['interaction_pp'],stage2_started=False),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
