"""Generate the single integrated report from completed, immutable evidence."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from common import BASE, save, specs, sha, verify


def read(name):return json.loads((BASE/name).read_text())


def main():
    done=read('DONE.json');assert done['status']=='COMPLETED'
    assert read('decision-DONE.json')['status']=='COMPLETED'
    decisions=read('p1-margin/decision-summary.json')['checkpoints']
    summary=read('p1-margin/summary.json');cluster=read('p1-margin/case-cluster-summary.json')
    provenance=read('p2-provenance.json');paired=read('paired-asset-verification.json')
    checkpoints=summary['checkpoints'];assert len(checkpoints)==11
    records=[]
    for s in specs():
        a=checkpoints[s['id']]
        assert a['own']['episodes']==a['shared']['episodes']==120
        records.append((s['id'],a['own'],a['shared']))
    own_steps=sum(a['steps'] for _,a,b in records)
    reference_steps=checkpoints['kda-original-t24']['shared']['steps']
    smoke=read('smoke-validation.json')
    jobs=[json.loads(p.read_text()) for p in (BASE/'job-evidence').glob('*.json')]
    assert jobs and all(j['exit_code']==0 for j in jobs)
    verification=verify()
    n=provenance['observed_cross_seed_variation']['rows']
    status_counts=Counter(e['status'] for e in provenance['entries'])
    screen={}
    for mode in ('own','shared'):
        gaps=[a[mode]['v_gap']['p50'] for a in checkpoints.values()]
        spreads=[a[mode]['geometry']['endpoint']['top5_spread']['p50'] for a in checkpoints.values()]
        screen[mode]=dict(v_gap_median_range=[min(gaps),max(gaps)],
            v_gap_max_min_ratio=max(gaps)/min(gaps),endpoint_spread_median_range_m=[min(spreads),max(spreads)],
            broad_gap_screen=sum(1e-4<=v<=1e-2 for v in gaps),
            broad_spread_screen=sum(.01<=v<=1. for v in spreads))
    save(BASE/'p1-margin/interpretation.json',dict(status='已验证通过',screens=screen,
        limit='Descriptive magnitude ranges, not tests of objective causality, equivalence, calibration or navigation benefit'))
    lines=['# 路线B证据固化报告','',
        '日期：2026-10-10。任务：修订版P1跨检查点测量、P2来源审计、P3先例核查。',
        '完成状态：已验证通过（约定的测量与审计完成）；不等于方法成功。',
        '本轮训练0次、优化器更新0次；4090未使用；只在本地3060采集P1仪表轨迹。原项目及11份权重未修改。','',
        '## 现象与结论','',
        '1. P1的11份检查点全部完成自身轨迹和共同状态重评分。下面给出各自数值，不用一份KDA的旧数字替代其他骨干。',
        '2. 小后继V间距不等于错误Q排序；前五名一步净距差异不等于长程价值错误。本轮没有执行替代动作续跑。',
        '3. action gap、误差敏感性、预测指标不等于导航效果、PBRS不变性均有明确先例。路线B目前是诊断证据，不是已经成立的新算法。',
        '4. 即使跨检查点共有小间距，也不能仅凭本轮把原因归给共同MC训练目标；动作时间尺度、网格密度、共享表示及训练动力学仍是竞争解释。','',
        '5. 修订新增最终可行候选的Decision Gap、速度差异、预测净距及最终动作；三种间距不互相替代。P1仍是现象复核，不是KDA性能优化或Delta机制创新。','',
        '## 问题与实验边界','',
        '- 正式P1：90000–90019，六格各20例，每权重120例；11份共1320条自身策略轨迹。92000–92499没有使用。',
        '- 90000段是既有开发块；这是新增仪表测量，不是独立case确认或新的闭环收益基准。',
        '- Smoke：六格各3例，11份共198条轨迹；另做共同窗口核验，不计入正式统计。',
        '- 共同状态来源：原始KDA-T24参考策略的同一批120例；所有检查点共享候选命令、选择索引和物理后果，T12仅取最后12帧。',
        '- 条件：TTC5在真实当前状态锚定一次，80个候选共用；alpha=0.3，risk=0.8，margin=0.2，dt=0.25秒。H1没有确认优胜alpha。',
        '- gap_V来自未经安全掩码的V；gap_raw来自未过滤r+gamma V；Decision Gap来自奖励、风险及索引0扣分后的最终可行分数。三者都不是实测Q_pi。',
        '- 正常Decision Gap仅在n_safe>=2时定义；n_safe=1记为空；all-unsafe单列fallback间距，不混入正常分布，且绝不把-1e9掩码带进统计。',
        '- “可行/安全候选”仅指原filter的预测clearance>=0.2，不是实际安全保证。完整分数=raw-0.8*max(0,0.2-clearance)-0.001*(index==0)。',
        '- gap_V/abs(V_top1)依赖价值零点；平移V不会改变排序，却会改变该比值。因此只作既定reward口径的描述，不作为跨任务鲁棒性尺度。',
        '- 主表数字是合并决策步中位数。步数不是独立样本数；补充CI按case ID整簇重采样，保留同case六格与全部步。仅20个case簇，CI是描述性的。',
        '- 运行栈：临时只读Docker + 既有隔离环境，torch2.7.1/CUDA12.6、mamba_ssm2.2.5、FLA0.6.0；原4090为另一栈，因此不声称复现旧实验的逐位数值。',
        '- 没有升级原容器或安装包；/home/abc只读挂载，只有本工作目录可写。','',
        '## 执行修订与成本闸门','',
        '原始自身轨迹采集已完成时收到本次修订，因此原protocol.json、manifest及逐步记录全部保留，另写protocol-amendment-v2.json；未把事后补充伪装成原预注册。',
        '收到修订后先暂停旧控制器，再补查P3的Clipped Advantage Learning和任务感知预测评价；P2既有来源审计已完成，随后补采预检，过关后恢复测量。',
        '补采预检：原始KDA/GRU/Mamba×六格×3例，共54个episode；确定性重放保存命令，几何逐位一致、完整评分argmax与原执行动作一致。',
        '随后补采1320条既有自身轨迹：不调用导航网络、不选择新动作。共同状态补充使用同一个参考轨迹的CV净距与各检查点保存的V。',
        '补采控制器的共同状态部分移到独立临时容器，避免原容器PID1完成退出时杀掉它；66个自身补采任务完成后显式终止空闲等待控制器（exit143），非采集失败，原始记录未覆盖。见controller-retirement.json与decision-recovery.json。','',
        '## P1：跨检查点证据','',
        '| 检查点 | 自身步数 | 自身V间距 | 自身间距/abs(V)% | 自身Top5端点净距展度cm | 共同V间距 | 共同相对间距% | 共同展度cm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,a,b in records:
        lines.append('| %s | %d | %.3g | %.4f | %.2f | %.3g | %.4f | %.2f |'%(name,a['steps'],
            a['v_gap']['p50'],a['relative_gap']['p50']*100,a['geometry']['endpoint']['top5_spread']['p50']*100,
            b['v_gap']['p50'],b['relative_gap']['p50']*100,b['geometry']['endpoint']['top5_spread']['p50']*100))
    lines += ['', '来源：p1-margin/summary.json；每行120条自身轨迹和同一批120条参考轨迹重评分。',
        '正式自身轨迹共%d步；共同状态%d步，各权重分别重评分，总计%d次共同状态测量。'%(own_steps,reference_steps,reference_steps*11),
        '共同窗口不是11组新导航结果；参考策略重评分逐位复现自己保存的V，仅证明本轮窗口重建正确。','']
    lines += ['### 最终动作Decision Gap','',
        '| 检查点 | 自身Decision Gap中位 | 共同Decision Gap中位 | 自身安全候选>=2步 | 仅1安全步 | all-unsafe步 | fallback间距中位 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'无样本' if v is None else '%.3g'%v
    for name,_,_ in records:
        a,b=decisions[name]['own'],decisions[name]['shared']
        lines.append('| %s | %s | %s | %d | %d | %d | %s |'%(name,
            fmt(a['metrics']['decision_gap']['p50']),fmt(b['metrics']['decision_gap']['p50']),
            a['safe_at_least2_steps'],a['safe_exactly1_steps'],a['all_unsafe_steps'],fmt(a['metrics']['fallback_gap']['p50'])))
    lines += ['', '来源：p1-margin/decision-summary.json。正常Decision Gap与all-unsafe fallback分别统计，缺少第二个安全动作时不填0或无穷大。',
        'Decision Gap更接近当前动作选择的分数鲁棒性，但小间距仍不证明动作错误、有害振荡或真实导航损失。完整case级聚类描述区间见同文件。','',
        '### Top-V候选与最终动作的几何/速度','',
        '| 检查点 | V前5速度展度m/s | V前2速度向量差m/s | V前5预测净距展度cm | 最终速度m/s | 最终预测净距cm | 最终实际端点净距cm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,_,_ in records:
        a=decisions[name]['own']['metrics']
        lines.append('| %s | %.3f | %.3f | %.2f | %.3f | %.2f | %.2f |'%(name,
            a['value_top5_speed_spread']['p50'],a['value_top2_velocity_difference']['p50'],
            a['value_top5_predicted_clearance_spread']['p50']*100,a['selected_speed']['p50'],
            a['selected_predicted_clearance']['p50']*100,a['selected_actual_endpoint_clearance']['p50']*100))
    lines += ['', '全部为自身轨迹合并步中位数；逐步文件保留最终动作索引、命令、V名次及前五名各速度/预测净距。共同状态同指标保留在JSON。',
        '净距或速度有差别不代表长程动作优劣；本轮没有用这些几何描述给动作贴成功/失败标签。','']
    for mode,label in [('own','自身轨迹'),('shared','共同状态')]:
        q=screen[mode]
        lines.append('%s：V间距中位数范围%.3g–%.3g（最大/最小%.2f倍）；Top5端点展度中位数%.2f–%.2fcm。'%(label,
            *q['v_gap_median_range'],q['v_gap_max_min_ratio'],q['endpoint_spread_median_range_m'][0]*100,
            q['endpoint_spread_median_range_m'][1]*100))
        lines.append('预冻结的宽数量级筛查：V间距[1e-4,1e-2]命中%d/11；展度[1,100]cm命中%d/11。这个宽筛查不是等价性检验。'%(q['broad_gap_screen'],q['broad_spread_screen']))
    all_common=all(v['broad_gap_screen']==11 and v['broad_spread_screen']==11 for v in screen.values())
    lines += ['', ('判定：已验证通过——11份均落入宽数量级区间，支持跨已测11份配置重复的描述性现象；不推广到未测配置，不支持“训练目标造成”的因果结论。' if all_common else
        '判定：已确认失败——“11份均满足所冻结宽数量级条件”的统一表述未通过。必须按上表保留配置差异，不能写成所有骨干的统一性质。'),
        '即使全部通过，也不能把量级筛查称为跨骨干等价，更不能把特定区间外的差异当统计显著。','',
        'Top5净距展度较大只说明近并列V对应不同的一步几何后果；它本身不证明V排序与净距无关，也不证明较低净距的动作有更差长程回报。','',
        '### 三种物理口径与raw前瞻分数','',
        '| 检查点 | 自身raw分数间距 | 共同raw分数间距 | 自身端点展度cm | 自身环境扫掠cm | 自身更新轨迹扫掠cm |',
        '|---|---:|---:|---:|---:|---:|']
    for name,a,b in records:
        lines.append('| %s | %.3g | %.3g | %.2f | %.2f | %.2f |'%(name,a['raw_r_gammaV_gap']['p50'],
            b['raw_r_gammaV_gap']['p50'],*[a['geometry'][k]['top5_spread']['p50']*100 for k in ('endpoint','env_swept','actual_swept')]))
    lines += ['', '端点：真实更新后的人位置与候选机器人后继的净距；环境扫掠：当前人速度口径；更新轨迹扫掠：实际人位移口径。三者不得混称。',
        '人看不见机器人已硬断言，因此同一步实际行人更新可用于全部80候选的几何反事实；扫掠使用同一时刻的相对运动最小距离，不用线段相交代替。',
        '来源：summary.json的geometry与raw_r_gammaV_gap；metric-unit-checks.json含异时交叉、零相对速度及V/raw分数分离测试。','',
        '### 按Top5最小真实端点净距分层','',
        '单元格=展度中位cm（步数/出现的episode数）。这是自身轨迹的条件描述，不把“看起来平”写成统计独立。',
        '| 检查点 | <0m | 0–0.2m | 0.2–0.4m | 0.4–0.8m | >=0.8m |','|---|---:|---:|---:|---:|---:|']
    bins=['<0','0-0.2','0.2-0.4','0.4-0.8','>=0.8']
    for name,a,b in records:
        cells=[]
        for key in bins:
            g=a['geometry']['endpoint']['strata'][key]
            cells.append('缺样本' if not g['steps'] else '%.2f (%d/%d)'%(g['spread']['p50']*100,g['steps'],g['episodes']))
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', '完整共同状态分层及另外两种净距口径见p1-margin/summary.json。置信区间见case-cluster-summary.json。',
        '没有事先冻结平坦性的等价界限，也没有每一层相同覆盖率，故“所有区间都是平的”在本轮仍未验证。','',
        '## P2：两条线来源审计','',
        '已扫描%d个文件路径、%d份唯一内容；备份重复不计作独立实验。11个指定研究目录全部覆盖。'%(len(provenance['entries']),provenance['unique_content_files']),
        '文件提取状态：'+ '，'.join('%s %d份'%(k,v) for k,v in status_counts.items())+'。',
        '这里的“通过”只表示文件解析/来源提取；不是科学结果通过。错误日志及失败尝试全部保留，后续成功不抹去旧失败。',
        '完整逐文件路径、哈希、字段、缺失证据、配置链接与状态见p2-provenance.json和p2-provenance.md。','',
        '| 研究/对比 | 类别 | 证据边界 |', '|---|---|---|',
        '| TTC5、H1、H2、Mamba-TTC5同权重干预 | A | 14组既有对比逐一核验同权重SHA和3000个(cell,case)配对；不同臂绝对性能比较仍属B |',
        '| H3与H3-random | A-影子重评分 | 同根/权重的替换；不是替代动作闭环收益 |',
        '| bayes-decomp | A-影子干预 | 42/43各自冻结导航器；跨seed不是同权重比较 |',
        '| bayes-horizon | A-预测残差测量 | 同轨迹CV基线，不是贝叶斯收益或导航增益 |',
        '| bayes-incr | A-冻结导航 / B-新探针 | 导航器未训，但A1/A2/B/P探针经过独立监督训练，不能笼统归为零训练A类 |',
        '| T419、T443、贝叶斯多臂、原三骨干比较 | B | 独立训练，包含训练路径与seed变异；不能只用case级检验声称骨干普适优势 |',
        '| P1自身/共同状态 | B类检查点描述（共同状态控制输入） | 不重新训练；既有权重不同，同状态只是剥离状态分布差异，不剥离训练差异，不能改标为A类 |','',
        '来源纠正：30列run_label=bayes-optimized-rl10000的CSV属于贝叶斯线，未引用为KDA振荡证据。',
        '另一个时间范围纠正：早期optimized evaluation只有episode级字段；后来H1/H2 JSON已含commands序列。无CSV不等于无逐步命令。','',
        '### 已观察到的同臂跨seed差异','',
        '| 指标 | 四种匹配臂平均绝对差 | 最大绝对差 | 单位 |', '|---|---:|---:|---|']
    for metric,v in n.items():lines.append('| %s | %.4f | %.4f | %s |'%(metric,v['mean_absolute_difference'],v['max_absolute_difference'],v['unit']))
    lines += ['', '来源：419和443各自summary，KDA/GRU×T12/T24共四种同臂配对；各检查点评测六格×500=3000例，case91000–91499；只有2个训练seed。这不是训练方差估计，也不是固定的显著性阈值。','',
        '### “多个测试期干预全零”需收窄','',
        '既有配对数据中的SR净变化如下，每个对比3000例（六格×500），只是来源核对，不是新跑评测，也未重新扩大统计家族。精确case范围和权重绑定见paired-asset-verification.json。',
        '| 对比 | SR净变化pp | 状态/限制 |','|---|---:|---|']
    for r in paired['contrasts']:
        if 'development' not in r['name']:
            lines.append('| %s | %+.3f | 已验证通过：既有文件配对净计数，不等于显著性或机制归因 |'%(r['name'],r['deltas']['success']['delta_pp']))
    lines += ['', 'TTC5确认阶段是明确非零且已有显著证据的公共输入改进，因此不能把所有部署期干预概括为全零。H1/H2未形成采纳证据；H3会翻转动作，但随机匹配后未建立方向结构和安全/进度优势。',
        'T交互seed419为+5.87pp，seed443为-1.33pp且95%区间跨0，原NOT_CONFIRMED裁决保留；本轮P1不推翻这个负结果。','',
        '## P3：先例与新颖性','',
        '| 主张 | 文献判定 |', '|---|---|',
        '| 发现小action gap及其对估计误差敏感 | 已确认失败：Bellemare等AAAI2016及更早advantage learning已有 |',
        '| 导航中需要针对扰动的稳健动作评价 | 已确认失败：Lütjens等CoRL/PMLR已有行人避碰应用 |',
        '| 预测更准不一定导航更好 | 已确认失败：已有直接社交导航经验研究 |',
        '| PBRS一致变换不改变argmax | 已验证通过：经典结论；只改reward不换V则不能照搬 |',
        '| 多检查点、共同状态、近并列V与真实一步净距展度的组合诊断 | 本地实测可报告；是否文献首次未验证，更不是方法增益 |','',
        '完整来源、四条导航研究线的seed报告边界与数学推导见p3-novelty.md。文献阅读不是官方代码执行、完整基线复现或新方法验证。',
        'PBRS CPU核验：一致变换维持动作0；只塑形reward而保留旧V可改选动作1。来源pbrs-check.json。',
        'Ng等的最优策略不变定理不能排除有限IL+MC训练中的实际收益。本轮没有该训练对照，收益未验证；不得写成“解析证明无收益”。方向①Reward研究不因本审计停止，但本轮没有授权新训练。','',
        '## 可选方案与推荐','',
        '| 方案 | 当前裁决 |', '|---|---|',
        '| 直接扩大V间距、加对比损失或换时序骨干 | 不授权：没有证明这些近并列对应错误长程后果，也没有独立技术新颖性 |',
        '| 把P1当作MC目标或KDA失效的因果证明 | 已确认失败：缺目标干预与排除竞争解释 |',
        '| 保存本轮为共有现象/异质性诊断，提出一个后果校准核验 | 推荐，仍不自动启动新实验 |','',
        '针对路线B的唯一最小下一步建议：若要以间距问题立项，先在已保存的自然近并列决策中冻结小批root与选择规则，用确定性重放核验Top-V候选的真实Q_pi差异；同时保留非近并列匹配对照。',
        '该建议需要另行授权，本轮未执行，也不是方向①Reward实验的前置门槛。目的是区分“动作确实近等价”与“价值没有辨别真实代价”，不是先设计扩大间距的方法。',
        '最终裁决：路线B完成证据固化，但仅凭本轮不能立项间距改造或Delta机制新方法；保留诊断，不声称训练目标缺陷、KDA专属缺陷或导航收益。方向①的Reward研究与方向②的Delta机制研究分别记账，本报告不替它们作停止裁决。','',
        '## 验收与文件','',
        '- 全部P1控制器采集任务exit code=0；smoke与完整阶段均无未处理断言失败。环境兼容性阻塞与解决办法单独记录。',
        '- 11份权重及配置、原代码/FLA/冻结脚本共%d个输入文件结束复验一致。'%verification,
        '- fast/legacy的windows/values逐位一致；reward/clearance允许浮点算术差异<=2e-15，未声称这两项逐位一致。',
        '- 采集、来源解析、数学核验与方法成败分开记账；缺失元数据保持缺失。',
        '- 没有训练、模型改动、92000块使用或额外闭环收益评测。',
        '', '主报告：',str(BASE/'REPORT.md'),'',
        '核心数据：',str(BASE/'p1-margin/summary.json'),str(BASE/'p2-provenance.json'),str(BASE/'p3-novelty.md')]
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(BASE/'report-verification.json',dict(status='已验证通过',frozen_files=verification,
        checkpoints=11,own_episodes=1320,own_steps=own_steps,reference_episodes=120,
        reference_steps=reference_steps,jobs=len(jobs),all_exit_codes_zero=True,
        report_bytes=(BASE/'REPORT.md').stat().st_size,report_sha256=sha(BASE/'REPORT.md')))
    print(json.dumps(dict(status='PASS',report=str(BASE/'REPORT.md'),own_steps=own_steps,
                         reference_steps=reference_steps,screens=screen),ensure_ascii=False))


if __name__=='__main__':main()
