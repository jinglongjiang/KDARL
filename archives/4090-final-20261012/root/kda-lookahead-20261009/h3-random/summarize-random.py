"""Episode-paired bootstrap, never treating steps or random repeats as independent episodes."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
VARIANTS = ('cv', 'oracle', 'random_shared', 'random_independent')
METRICS = ('final_flip', 'value_flip', 'env_clearance_delta', 'actual_clearance_delta',
           'progress_delta', 'env_collision', 'actual_collision', 'env_safe', 'actual_safe')
CONDITIONAL = ('cv_value_gap', 'new_value_gap', 'speed_delta', 'command_delta',
               'direction_degrees', 'env_clearance_delta', 'actual_clearance_delta', 'progress_delta')
PRIMARY = ('final_flip', 'env_clearance_delta', 'actual_clearance_delta', 'progress_delta')
GROUPS = ('all', 'success_all', 'collision_all', 'timeout_all',
          'success_last20', 'collision_last20', 'timeout_last20')


def read(path):
    return json.loads(Path(path).read_text())


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0)


def intervals(samples):
    samples = samples[np.isfinite(samples)]
    return np.percentile(samples, [2.5, 97.5]).tolist() if len(samples) else None


def main():
    protocol = read(BASE / 'protocol.json')
    episodes = []
    for phase in ('implcheck', 'full'):
        for cell in range(6):
            block = read(BASE / phase / ('cell%d.json' % cell))
            assert block['status'] == '已验证通过' and block['repeats'] == protocol['random_repeats']
            episodes.extend(block['episodes'])
    assert len(episodes) == 300 and len({(x['cell'], x['case']) for x in episodes}) == 300
    output = dict(status='已验证通过', n_episodes=300, n_steps=sum(x['groups']['all']['n_steps'] for x in episodes),
        repeats=protocol['random_repeats'], family='A final flip and B three physical metrics: 4 simultaneous intervals',
        inference_unit='episode=(cell,case), resampled jointly across arms; repeat means fixed within each step',
        bootstrap_replicates=10000, groups={})
    assert output['n_steps'] == 16020
    for group in GROUPS:
        included = [e for e in episodes if group in e['groups']]
        items = [e['groups'][group] for e in included]
        n = len(items)
        if not n:
            output['groups'][group] = dict(n_episodes=0, n_steps=0); continue
        counts = np.array([x['n_steps'] for x in items], float)
        indices = np.random.default_rng(20261009).integers(0, n, (10000, n))
        sampled_counts = counts[indices].sum(axis=1)
        entry = dict(n_episodes=n, n_steps=int(counts.sum()), arms={}, oracle_minus_shared={}, oracle_minus_independent={})
        sums = {}
        for variant in VARIANTS:
            arm = dict(metrics={}, flipped_only={}, random_repeat_final_flip_rates=None)
            sums[variant] = {}
            for metric in METRICS:
                values = np.array([x['sums'][variant][metric] for x in items])
                draws = values[indices].sum(axis=1) / sampled_counts
                sums[variant][metric] = values
                arm['metrics'][metric] = dict(mean=float(values.sum() / counts.sum()), episode_bootstrap_ci95=intervals(draws))
            for metric in CONDITIONAL:
                pairs = np.array([x['conditional'][variant][metric] for x in items])
                draws = divide(pairs[indices, 0].sum(axis=1), pairs[indices, 1].sum(axis=1))
                arm['flipped_only'][metric] = dict(mean=float(pairs[:, 0].sum() / pairs[:, 1].sum()) if pairs[:, 1].sum() else None,
                    weighted_flips=float(pairs[:, 1].sum()), episode_bootstrap_ci95=intervals(draws),
                    limitation='conditional on this arm flipping; different arms can select different state subsets')
            if variant.startswith('random'):
                arm['random_repeat_final_flip_rates'] = (np.array([x['repeat_flips'][variant] for x in items]).sum(axis=0) / counts.sum()).tolist()
            entry['arms'][variant] = arm
        for control in VARIANTS[2:]:
            comparison = entry['oracle_minus_shared' if control == 'random_shared' else 'oracle_minus_independent']
            for metric in METRICS:
                difference = sums['oracle'][metric] - sums[control][metric]
                draws = difference[indices].sum(axis=1) / sampled_counts
                value = dict(difference=float(difference.sum() / counts.sum()), paired_episode_ci95=intervals(draws))
                if metric in PRIMARY and group == 'all' and control == 'random_shared':
                    value['simultaneous_ci98_75'] = np.percentile(draws, [.625, 99.375]).tolist()
                comparison[metric] = value
        output['groups'][group] = entry
    all_steps = output['groups']['all']
    a = all_steps['oracle_minus_shared']['final_flip']
    lo, hi = a['simultaneous_ci98_75']
    if lo > 0:
        verdict_a = 'Oracle翻转显著高于Random：真实残差方向结构有额外影响'
    elif hi < 0:
        verdict_a = 'Oracle翻转显著低于Random：评分对随机方向更敏感'
    elif max(abs(x) for x in a['paired_episode_ci95']) < protocol['equivalence_margin_flip_fraction']:
        verdict_a = '在预注册±1pp诊断等效界限内相近；支持幅度敏感性解释，但不证明预测信息无价值'
    else:
        verdict_a = '未检出显著差异，也未证明在±1pp界限内等效'
    b_metrics = ('env_clearance_delta', 'actual_clearance_delta', 'progress_delta')
    positive_b = all(all_steps['oracle_minus_shared'][m]['simultaneous_ci98_75'][0] > 0 for m in b_metrics)
    collision_guard = all(all_steps['arms']['oracle']['metrics'][m]['mean'] <= all_steps['arms']['random_shared']['metrics'][m]['mean'] + 1e-12
        for m in ('env_collision', 'actual_collision'))
    b = positive_b and collision_guard
    output['A'] = dict(positive_structure_supported=lo > 0, verdict=verdict_a, evidence=a)
    output['B'] = dict(positive_one_step_safety_progress_supported=b,
        definition='All-step paired comparison, both swept-clearance definitions and goal progress simultaneously improve; observed collision frequencies do not increase',
        collision_guard_observed_only=collision_guard,
        evidence={m: all_steps['oracle_minus_shared'][m] for m in b_metrics},
        scope='one-step geometric/control indicators, NOT rollout navigation return or success rate')
    if lo > 0 and b:
        recommendation = 'A、B均成立：值得申请下一阶段预测与价值决策协同研究，先查RGL等先例；不自动训练，未证明终局收益'
    elif lo > 0:
        recommendation = '仅A成立：保留方向结构影响这一线索，未证明安全与前进收益；不启动训练'
    elif b:
        recommendation = '仅B成立：保留一步控制指标优势，但必须解释机制；不等于导航方法成功'
    else:
        recommendation = 'A、B的正向门槛均未通过：暂停当前冻结网络、alpha0.3、0.25秒单步CV改进路线；不外推到多步、reward或其他KDA方向'
    output['recommendation'] = recommendation
    path = BASE / 'summary.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    report(output)
    print(json.dumps(dict(status='已验证通过', output=str(path), bytes=path.stat().st_size, A=output['A']['verdict'], B=b), ensure_ascii=False))


def report(data):
    protocol = read(BASE / 'protocol.json')
    lines = ['# H3 Random-Matched 冻结权重诊断', '', '状态：已验证通过', '',
        '## 现象', '原H3 oracle最终动作翻转率22.05%，但没有检验同幅度无信息方向的扰动，也没有证明翻转改善导航。',
        '## 问题', 'A：真实残差方向结构是否比同幅度随机方向更能影响选择？B：oracle选择是否比随机选择更安全、更有目标进度？A与B分开判断。',
        '## 证据', '零训练、零新环境采样。复用原300例/16020步；每步每种随机对照%d次。42例实现核验通过后，补完其余258例，核验例复用，不扩case。' % data['repeats'],
        '权重sha256='+protocol['weight_sha256'],
        '固定T=24、TTC5、alpha=0.3、80动作；CV及Oracle的value/scores/clearance/argmax逐位复现原H3。误差位置与速度联合同角度旋转，匹配所有行人的各自残差幅度。',
        '来源：summary.json、implcheck.json、implcheck/cell0..5.json、full/cell0..5.json及对应steps.jsonl.gz；退出码、输出大小、实际日志尾部见job-evidence/。',
        '统计单位为(cell,case)的episode，10000次配对聚类bootstrap。主四项使用同时98.75%区间，其余95%区间为描述性；不把步或随机重复当独立episode。',
        '', '### A：最终动作与value-only翻转', '| 分组 | episode数 / 步数 | Oracle最终 | Random共享最终 | Random独立最终 | Oracle value-only | 共享 value-only |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group, entry in data['groups'].items():
        if not entry.get('n_episodes'): continue
        arms = entry['arms']
        rate = lambda v, m: 100 * arms[v]['metrics'][m]['mean']
        lines.append('| %s | %d / %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |' %
            (group, entry['n_episodes'], entry['n_steps'], rate('oracle','final_flip'), rate('random_shared','final_flip'),
             rate('random_independent','final_flip'), rate('oracle','value_flip'), rate('random_shared','value_flip')))
    lines += ['已验证通过：A判定='+data['A']['verdict'],
        'A差值（Oracle减共享Random）='+str(data['A']['evidence']),
        'value-only与完整评分翻转分别报告，但两者不是互斥因果类别；不能仅靠比例唯一归因到value或filter。',
        '', '### B：同一批root上Oracle减共享Random的控制指标', '| 指标 | 均值差 | 同时98.75%配对区间 |', '|---|---:|---|']
    for metric, value in data['B']['evidence'].items():
        lines.append('| %s | %.8f m | %s |' % (metric, value['difference'], value['simultaneous_ci98_75']))
    lines += ['B正向门槛通过='+str(data['B']['positive_one_step_safety_progress_supported']),
        '环境口径与实际更新口径不一致：CrowdSim.step用human.vx/vy计算相对线段判碰，却用本步human_actions更新行人。',
        'env_clearance采用旧速度同时间扫掠线段的全体行人最小余量，符号与环境判碰一致；另保存环境遇到第一个负余量即break时的reported_loop_clearance，负值大小不一定等于全局最小。',
        'actual_clearance采用current_humans至true_next_humans实际更新轨迹，同一时刻求机器人与行人的相对最近接近；不混用CV轨迹。原配置robot.visible=false，人的本步动作不读取机器人，因此在同一root评估不同机器人command的这一段几何可复用实际行人更新。',
        '主要B比较覆盖全部相同root，包括不翻转步，避免按各臂不同翻转集合挑样本。翻转条件下的value差、速度差、方向差、安全余量与进度另在summary各组flipped_only报告，注明条件选择限制。',
        '', '## 可选方案', '若A/B有正信号，仅申请对应下一阶段；若都没有，不靠调整随机种子、阈值或筛选case救结果。',
        '## 推荐', data['recommendation'], '', '## 限制',
        '- Oracle与Random翻转相近不证明预测信息无价值，A与B独立；不显著也不自动等效。',
        '- 共享旋转保留偏差场相对结构；独立旋转同时破坏方向与行人间结构，仅为次要描述。随机扰动不一定是物理上真实可发生的运动。',
        '- 位置与速度残差联合匹配和旋转；本实验不能分别辨识位置误差和速度误差的独立贡献。',
        '- 影子重评分和一步几何指标不是闭环导航收益；没有执行这些新动作并跑至终局。',
        '- 负结果仅适用于本冻结网络+0.25秒前瞻+alpha=0.3，不淘汰多步前瞻、reward或其他KDA方向，也不覆盖alpha=0。',
        '- 碰撞/超时分层按原CV轨迹结局分组，不是随机处理后的结局；超时仅6例。跨cell重复的case ID可能存在相关性，主要bootstrap按用户规定episode聚类，单seed开发块不是独立最终确认。',
        '- 实现、输入冻结和数值复现：已验证通过；最终导航收益与新颖性：未验证。']
    (BASE / 'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
