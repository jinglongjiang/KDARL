# Q8 rolling deployment and random MPPI control

## yi-12 final research verdict

【研究结论】已验证：冻结Q8滚动决策在base 20S上改善一步策略，但未建立超过ALWAYS的收益；R0上的成功率收益未确定。本轮取得了局部闭环改善证据，没有取得独立算法贡献层面的方法级进展。B的近似等调用率对照支持冻结风险调度整体的成功率优势，不单独证明风险定位机制。

【核心证据】base 20S Q8−1step为+6.8pp [3.0,10.8]，满足原门槛；Q8−ALWAYS为−5.8pp [−8.8,−2.8]，且碰撞增加+2.8pp [1.2,4.6]。R0 20S Q8−1step为+2.4pp [−1.2,5.8]，未通过原门槛。Risk−random成功率+3.0pp [0.2,5.8]。

【科学裁决】SUPPORTED：base 20S中Q8相对一步策略的成功率改善，以及本组近似预算匹配风险调度相对随机调度的成功率改善。INCONCLUSIVE：R0中Q8的成功率改善，以及风险定位本身与时间聚簇贡献的区分。REFUTED：本组base 20S中当前Q8能超过ALWAYS的成功率主张。上述判定仅针对本组冻结条件，不代表否定多步价值前瞻。

【研究决策】KILL：不把当前Q8版本升级为论文主线，不自动修订或调参。理由是强规划对照上的实际劣势，不是把R0的不显著结果当成机制无效。base方形通过注册门槛的事实完整保留，未修改门槛；其他格子各自报告，不设事后合并门槛。

【执行结果或下一动作】本轮A、B已完成；唯一下一动作是将本报告交接主线统一比较，不自动追加实验、训练或fresh confirmation。风险调度结果保留供该决策使用，不声称独立创新成立。

已验证：零训练，冻结实现与参数。未建立独立算法贡献；本测试块非fresh confirmation。

| Mother | Scene | Arm | SR/CR/TO % | Contacts | Mean/p95 ms | Calls |
|---|---|---|---|---|---|---|
| base | square | 1step | 80.8/5.8/13.4 | 28 | 15.26/17.91 | 0 |
| base | square | Q8 | 87.6/4.2/8.2 | 17 | 210.30/219.28 | 0 |
| base | square | Q8-H | 78.2/13.8/8.0 | 63 | 198.78/211.13 | 0 |
| base | square | ALWAYS | 93.4/1.4/5.2 | 6 | 99.45/107.90 | 35260 |
| base | circle | 1step | 96.2/2.0/1.8 | 9 | 15.17/17.68 | 0 |
| base | circle | Q8 | 95.0/3.4/1.6 | 14 | 211.18/219.20 | 0 |
| base | circle | Q8-H | 69.4/29.6/1.0 | 127 | 197.34/211.29 | 0 |
| base | circle | ALWAYS | 97.8/2.2/0.0 | 11 | 103.85/108.20 | 32865 |
| R0 | square | 1step | 87.4/6.2/6.4 | 29 | 15.23/17.88 | 0 |
| R0 | square | Q8 | 89.8/3.0/7.2 | 14 | 210.96/219.55 | 0 |
| R0 | square | Q8-H | 78.2/13.8/8.0 | 63 | 199.62/211.74 | 0 |
| R0 | square | ALWAYS | 90.6/4.0/5.4 | 18 | 103.83/111.03 | 34861 |
| R0 | circle | 1step | 97.8/2.2/0.0 | 10 | 15.08/17.59 | 0 |
| R0 | circle | Q8 | 98.0/1.4/0.6 | 7 | 211.94/219.83 | 0 |
| R0 | circle | Q8-H | 69.4/29.6/1.0 | 127 | 197.67/211.75 | 0 |
| R0 | circle | ALWAYS | 98.2/1.8/0.0 | 10 | 101.45/109.33 | 32773 |

| Mother | Scene | Q8 minus | Delta SR [95%CI] pp | Delta CR [95%CI] pp | Delta TO [95%CI] pp | Rescue/damage |
|---|---|---|---|---|---|---|
| base | square | 1step | +6.8 [+3.0,+10.8] | -1.6 [-4.2,+1.0] | -5.2 [-8.4,-2.2] | 67/33 |
| base | square | Q8-H | +9.4 [+5.2,+13.4] | -9.6 [-12.8,-6.4] | +0.2 [-2.8,+3.4] | 82/35 |
| base | square | ALWAYS | -5.8 [-8.8,-2.8] | +2.8 [+1.2,+4.6] | +3.0 [+0.6,+5.4] | 15/44 |

base square registered SR gate: True.
| base | circle | 1step | -1.2 [-3.6,+1.2] | +1.4 [-0.4,+3.2] | -0.2 [-1.8,+1.4] | 15/21 |
| base | circle | Q8-H | +25.6 [+21.4,+30.0] | -26.2 [-30.4,-22.0] | +0.6 [-0.8,+2.0] | 140/12 |
| base | circle | ALWAYS | -2.8 [-5.0,-0.8] | +1.2 [-0.6,+3.0] | +1.6 [+0.6,+2.8] | 7/21 |

base circle registered SR gate: False.
| R0 | square | 1step | +2.4 [-1.2,+5.8] | -3.2 [-5.6,-1.0] | +0.8 [-2.0,+3.6] | 46/34 |
| R0 | square | Q8-H | +11.6 [+7.4,+15.8] | -10.8 [-14.0,-7.6] | -0.8 [-3.8,+2.2] | 91/33 |
| R0 | square | ALWAYS | -0.8 [-3.8,+2.2] | -1.0 [-3.0,+1.0] | +1.8 [-0.4,+4.2] | 28/32 |

R0 square registered SR gate: False.
| R0 | circle | 1step | +0.2 [-1.4,+1.8] | -0.8 [-2.4,+0.8] | +0.6 [+0.0,+1.4] | 9/8 |
| R0 | circle | Q8-H | +28.6 [+24.4,+33.0] | -28.2 [-32.4,-24.2] | -0.4 [-1.6,+0.6] | 148/5 |
| R0 | circle | ALWAYS | -0.2 [-1.6,+1.2] | -0.4 [-1.6,+0.8] | +0.6 [+0.0,+1.4] | 6/7 |

R0 circle registered SR gate: False.

## B. Random call-rate control

Random SR/CR/TO: 90.0/3.8/6.2; contacts17; mean/p95 74.84/101.02 ms.
Risk-minus-random SR +3.0 [+0.2,+5.8]; CR -1.6 [-3.4,+0.0]; TO -1.4 [-3.6,+0.8]; rescue/damage 32/17.
Actual random calls 27057/37520 (0.7211), risk 25737/35703 (0.7209); total call ratio1.0513; approximate budget matched: True.

| B arm, GRU base 20S | Success/collision/timeout counts | SR/CR/TO % | Contacts | Mean/p95 ms | Actual optimizer calls |
|---|---|---|---|---|---|
| Frozen risk | 465/11/24 | 93.0/2.2/4.8 | 10 | 79.06/110.39 | 25737/35703 |
| Random | 450/19/31 | 90.0/3.8/6.2 | 17 | 74.84/101.02 | 27057/37520 |

已验证：risk−random的SR为+3.0pp [0.2,5.8]，CR为−1.6pp [−3.4,0.0]，TO为−1.4pp [−3.6,0.8]，物理接触为−1.4pp [−3.2,0.2]，救回32例、破坏17例。不能把后面三个跨零或触零的区间称为已经证明降低风险。调用率相差0.0272pp，总调用次数random比risk多5.13%，满足冻结的近似匹配规则，但不是完全等计算量或相同时间聚簇的干预。旧risk时延来自risk-trigger-20261017/RESULT.md，运行负载不同。

## Navigation efficiency

已验证：以下为各臂各自成功回合的平均导航秒数/路径米数，并非共同成功案例配对效应；不能用来单独宣称效率因果改善。完整指标保存在statistics.json。

| Mother | Scene | 1step s/m | Q8 s/m | Q8-H s/m | ALWAYS s/m |
|---|---|---|---|---|---|
| base | square | 18.67/13.95 | 15.73/11.62 | 13.38/10.78 | 16.03/12.68 |
| base | circle | 16.86/12.47 | 15.71/11.24 | 16.28/12.13 | 16.66/12.91 |
| R0 | square | 17.41/13.19 | 17.06/11.81 | 13.38/10.78 | 16.07/12.68 |
| R0 | circle | 16.23/12.16 | 16.79/11.08 | 16.28/12.13 | 16.57/12.85 |

## Execution and engineering history

- 已验证：yi-12传输进程退出0后，才核对manifest、8个冻结文件和726项输入SHA，全部一致；随后运行环境预检退出0。原manifest SHA256为13f951f84f5715d5da128addae40df361deb1911c04574d12eafff2cafc5a62a，协议及判据未变。
- 已验证：40回合旧记录回归、40回合Q8/H冒烟、5940回合A正式部分、2回合随机端点控制、5回合B冒烟、495回合B正式部分，共执行6522回合。其中6500回合进入正式比较，其余22回合为ALWAYS回归和随机端点控制。前5例按冻结协议复用于500例统计，未重复计数。
- 已验证：post-run audit核对4000份Q8逐步文件、254297个决策步、完整案例主键、评分代数、选中/执行动作、逐case配对区间、随机调度位和调用次数；退出0，实验进程均已退出。证据：audit-final.json及audit-final.py。
- 已验证：A-regression、A-smoke、A-formal、B-controls、B-smoke、B-formal、两次统计汇总及主控均正常完成。主控运行10464.98秒，约2小时54分25秒。源项目及旧结果只读，CPU0–5、最多6worker，未训练。
- 已确认失败并保留：yi-11曾在rsync尚未结束时提前核验远端，因summarize.py未传到而退出1；其rsync随后由代理主动停止，退出20。没有产生科学观测。保留failure-preflight.json、yi-11-stopped-result.md，不把本轮成功覆盖成历史从未失败。
- 已确认失败并保留：engineering-probes.json记录冻结前查找不存在旧1step文件夹时的探测错误；未进入评测，不作为科学数据。
- 未验证：fresh confirmation、跨新母体推广、独立算法新颖性、等负载时延优势、风险定位相对时间聚簇的单独贡献。本轮不自动补做。

## Recompute

在远端本目录运行Python summarize.py all可重算冻结统计表；Python audit-final.py仅做结果复算和完整性校验，不运行仿真或训练。代码依赖和权重路径/SHA见manifest.json，原始逐例结果见episodes/，Q8逐步80候选证据见decisions/，配对救回与破坏ID见statistics.json。运行summarize会重新生成基础RESULT表；本节最终研究裁决为复算通过后的人工审阅补充。

## Scope and evidence

Q8 calls rho-tools/rho-tool.py:242-304 verbatim. H changes only terminal callback to frozen Controller.free_value; common1step remains for root control plumbing. Same first-step mask/risk/index0 penalty and executed smoothing. No horizon safety mask added.
Each step executes first command only; no8-step commitment in actual Q8 deployment. Virtual history must leave actual history unchanged. Decision records preserve80 scores, commands, rewards/tails/alive masks, current robot/humans and executed command.
1step full500 records reproduce prior same4090 outcomes/actions/selection/path/return/contact. ALWAYS first5 per cell regressed, full500 cached same4090. ALWAYS/risk old timing workloads differ; no equal-load causal latency claim.
Random matches optimizer-call RATE, not new16-step activation frequency or temporal clustering. Actual call totals may differ with episode duration; unmatched budget bars exclusive risk-targeting attribution. No outcome-based recalibration.
The registered SR criterion alone does not establish safety, value-tail independent benefit, novelty, or fresh confirmation. Q8 versusH andALWAYS plus collisions/contacts remain mandatory.
Inputs/checkpoints pinned in manifest.json. Source dirs read-only. PID files and/proc used, CPU0-5,6workers,1thread,2cores reserved. Full raw outcomes, failures and timing preserved. All stage exits0.
No further algorithm/training/confirmation automatically starts. Final research verdict above was added after independent audit exited0; original frozen gate and scientific observations remain unchanged.
