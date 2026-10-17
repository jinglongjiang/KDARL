# Frozen MPPI plus CV risk trigger

本轮是简单风险触发参考的受控闭环实验，不据通过阈值自动声称原创性或方法级贡献。无训练、无奖励/安全/规划/执行参数改变。

## Development selection

GRU base，310000–310199，20S/20C各200，共400例每格。只比较登记9格；原MPC基线与最强风险格的各10条冒烟记录复用。
| T_r | d_r | Success/collision/timeout | Eligible SR>=mpc | Calls |
|---|---|---|---|---|
| off | off | 373/12/15 | baseline | 11552 |
| 1.0 | 0.1 | 376/9/15 | True | 15970 |
| 1.0 | 0.2 | 380/7/13 | True | 18439 |
| 1.0 | 0.3 | 381/7/12 | True | 21018 |
| 1.5 | 0.1 | 378/10/12 | True | 17571 |
| 1.5 | 0.2 | 376/9/15 | True | 19826 |
| 1.5 | 0.3 | 379/7/14 | True | 22118 |
| 2.0 | 0.1 | 382/7/11 | True | 19043 |
| 2.0 | 0.2 | 381/7/12 | True | 20861 |
| 2.0 | 0.3 | 380/7/13 | True | 22986 |

选定一次并冻结：T_r=2.0s，d_r=0.1m；理由及SHA见selected.json、selection-freeze.json。

## Formal outcomes

| Base | Scene | Arm | SR/CR/TO % | Contact episodes | Mean / pooled step p95 ms | Median episode step-p95 ms | New trigger / intervention % |
|---|---|---|---|---|---|---|---|
| gru | square | mpc | 89.80/4.80/5.40 | 23 | 51.63 / 未验证 | 103.16 | 2.82 / 40.88 |
| gru | square | risk | 93.00/2.20/4.80 | 10 | 79.06 / 110.39 | 109.59 | 5.62 / 71.21 |
| gru | circle | mpc | 98.20/1.40/0.40 | 7 | 48.02 / 未验证 | 102.61 | 2.40 / 37.13 |
| gru | circle | risk | 98.40/1.40/0.20 | 7 | 71.63 / 109.82 | 108.70 | 4.85 / 64.17 |
| kda | square | mpc | 87.60/5.80/6.60 | 28 | 55.44 / 未验证 | 99.16 | 3.21 / 46.69 |
| kda | square | risk | 92.20/2.40/5.40 | 12 | 80.78 / 107.08 | 104.43 | 5.65 / 74.94 |
| kda | circle | mpc | 97.40/2.60/0.00 | 13 | 51.16 / 未验证 | 99.67 | 2.69 / 41.49 |
| kda | circle | risk | 98.60/1.40/0.00 | 7 | 74.13 / 105.25 | 102.62 | 4.96 / 67.97 |
| R0 | square | mpc | 89.00/5.60/5.40 | 27 | 52.52 / 未验证 | 103.36 | 2.87 / 42.04 |
| R0 | square | risk | 90.20/4.20/5.60 | 20 | 73.49 / 104.29 | 99.41 | 5.48 / 69.67 |
| R0 | circle | mpc | 97.20/2.60/0.20 | 14 | 48.84 / 未验证 | 103.22 | 2.45 / 38.00 |
| R0 | circle | risk | 98.00/2.00/0.00 | 10 | 66.94 / 107.98 | 103.31 | 4.40 / 60.53 |

## Paired treatment minus mpc

| Base | Scene | Delta SR [95% CI] pp | Delta CR [95% CI] pp | Delta TO [95% CI] pp | Rescue/damage |
|---|---|---|---|---|---|
| gru | square | 3.20 [0.40, 6.00] | -2.60 [-4.60, -0.60] | -0.60 [-2.60, 1.20] | 32/16 |
| gru | circle | 0.20 [-0.60, 1.00] | 0.00 [-0.80, 0.80] | -0.20 [-0.60, 0.00] | 3/2 |
| kda | square | 4.60 [1.40, 7.80] | -3.40 [-5.60, -1.20] | -1.20 [-3.40, 1.20] | 44/21 |
| kda | circle | 1.20 [-0.20, 2.80] | -1.20 [-2.80, 0.20] | 0.00 [0.00, 0.00] | 11/5 |
| R0 | square | 1.20 [-1.80, 4.20] | -1.40 [-3.40, 0.60] | 0.20 [-1.80, 2.20] | 31/25 |
| R0 | circle | 0.80 [-0.60, 2.20] | -0.60 [-2.00, 0.80] | -0.20 [-0.60, 0.00] | 8/4 |

## Verdict

主判据：风险触发有效。科学裁决：SUPPORTED；研究决策：GO。
指定命题：GRU base20S的CR至少下降1.5pp且95%上界<0，同时SR95%下界>−1.0pp。KDA/R0和圆形只复核，不改主终点。
“无效”仅为CR区间含0时的操作性淘汰标签，不证明零效应；若安全改善但SR门槛不满足，也不判通过。未自动开启下一版或新训练。

## Mechanism and limits

风险仅用当前可观测20人位置/速度/半径及原1step选中的平滑后执行速度。CV按连续时间段计算相对净距，不读取真实未来或隐藏目标。
风险只能在idle且原safe mask非空时新激活，低进展触发优先；active计划不延长。触发后完整复用原MPPI16步、256样本、原代价、自由空间终端启发、首步安全掩码、风险罚、平滑与回退。唯一干预是额外激活机会。
每个触发的条件出现、实际新激活、对应优化器调用、有效执行分别保存，不把风险条件成立当作规划实际执行。原风险关闭分支须对旧记录逐例精确回归。
原strongmother base有效beta=0、intent=false；beta=.03属于原geometry臂，未混入本轮base。机器人不可见，原生50秒、dt=.25、TTC5、80动作。
311000块历史上已用于评测和诊断，本轮只是未用于此次格点选择的固定比较，不是fresh confirmation。单个训练谱系，不估训练seed方差。
原母体单回合SR等已知，但格点只用dev选择并在正式前冻结；没有用最终结果挑参数。简单风险触发本身不构成原创算法。
每步时延边界与原评测器相同，包含网络、风险检查、MPPI及CUDA同步，不包含环境step。新回合保存全部step时间，可算pooled p95。旧记录只保存每回合mean/p95，不能复原pooled历史p95；表中明确未验证，另报各回合p95的中位数。
5例/格关闭风险回归提供当前设备负载下的逐步mean/p95，见statistics.regression_timings；不是500例全量匹配时延。历史并发负载不同，不做完整队列提速/开销因果比较。
全部结果、失败、动作摘要、逐步风险诊断、触发时间、计算代价保存。物理接触与原生collision口径并列，不偷换安全定义。
CPU0-5、最多6worker、线程1，为丙留2核；MPPI CPU，冻结网络CUDA。未修改旧目录。

## Evidence

protocol.json; manifest.json; synthetic-controls.json; smoke-complete.json; development-results.json;
selected.json; selection-freeze.json; regression-complete.json; formal-complete.json; statistics.json;
episodes/; diagnostics/; latencies/; stage-*-exit.json; exit-controller.json.

## Final independent verification

已验证：1669项输入SHA、10项冻结实现/协议SHA、3000条正式回合、209371个决策步、24项独立重算指标均通过；30条正式块关闭风险回归逐例一致，冒烟另有10条关闭风险逐例一致。格点选择在正式评测前冻结，最终结果未用于再次选择。
本轮新增导航回合7030条：开发4000（含复用的20条冒烟）、正式块回归30、正式风险臂3000。正式基线3000条来自已核验旧记录。所有阶段和控制器退出码0，PID文件对应进程均已退出。未启动下一轮实验。

冻结选择selected.json SHA256：661973e48d8026cdd1c6e782dc77bba12977d60d24d0a2069c3f7893e838f6f5。
selection-freeze.json SHA256：4f1a78e356e0a3717c5ded621d4f8fddf4608f539d4e5cc8d18913b54d5d0fbf。
development-results.json SHA256：79f45762f8555d01d66a7b6f735600db7f6541873722c41d76c8815ad2c9558a。

## Risk-specific activation accounting

风险条件成立不等于新触发：正在执行计划时不重启，低进展触发优先。下表百分比以全部决策步为分母；“无有效计划”按原回退逻辑处理，未放松安全条件。实际优化器调用包含计划后续更新，不等于新触发数。

| Base | Scene | Risk condition steps % | Risk new-trigger steps % | Risk activations | Valid plan / no valid plan at activation | First action changed | All optimizer calls mpc -> risk |
|---|---|---|---|---|---|---|---|
| GRU | square | 23.64 | 3.87 | 1381 | 1176 / 205 | 1131 | 15027 -> 25737 |
| GRU | circle | 14.94 | 3.32 | 1111 | 976 / 135 | 957 | 12501 -> 21637 |
| KDA | square | 26.74 | 3.75 | 1388 | 1216 / 172 | 1178 | 18294 -> 27986 |
| KDA | circle | 16.80 | 3.32 | 1147 | 1020 / 127 | 988 | 14510 -> 23657 |
| R0 | square | 25.15 | 3.64 | 1294 | 1079 / 215 | 1037 | 15137 -> 25126 |
| R0 | circle | 12.68 | 2.75 | 909 | 821 / 88 | 788 | 12569 -> 20150 |

## Current-load timing controls

以下仅为关闭风险回归的每格5例，不是500例正式队列时延；均值/p95均从全部逐步时延计算。历史500例基线的pooled p95仍未验证，不用每回合p95的中位数替代。

| Base | Scene | Regression n | Off mean / pooled p95 ms |
|---|---|---|---|
| GRU | square | 5 | 43.33 / 101.14 |
| GRU | circle | 5 | 52.71 / 101.11 |
| KDA | square | 5 | 52.84 / 105.27 |
| KDA | circle | 5 | 45.97 / 101.40 |
| R0 | square | 5 | 45.17 / 101.03 |
| R0 | circle | 5 | 49.41 / 102.36 |

## Navigation efficiency, descriptive only

下表仅取两臂均成功的配对case，是事后描述而非预注册判据；按成功条件筛选，不能用于替代全体任务成败比较。

| Base | Scene | Both-success n | Mean time s mpc -> risk | Mean path m mpc -> risk |
|---|---|---|---|---|
| GRU | square | 433 | 17.065 -> 16.418 | 13.204 -> 12.830 |
| GRU | circle | 489 | 16.816 -> 16.811 | 12.789 -> 12.952 |
| KDA | square | 417 | 18.052 -> 16.767 | 13.765 -> 13.028 |
| KDA | circle | 482 | 17.716 -> 17.438 | 13.424 -> 13.322 |
| R0 | square | 420 | 16.818 -> 16.422 | 13.086 -> 12.891 |
| R0 | circle | 482 | 16.668 -> 16.752 | 12.669 -> 12.829 |

## Research boundary

已验证：GRU方形通过登记主判据，KDA方形复核方向一致且对应区间排除0。未验证：R0方形稳定收益，及圆形的稳定额外收益，相应区间跨0；不得将所有母体概括为均已验证有效。
本轮未建立独立算法贡献，不能标记METHOD_ENTRY_FOUND。额外优化器调用明显增加，尚无相同调用预算的周期/随机触发对照，因而不能把收益唯一归因为CV风险定位优于增加计算机会。零观察事件导致的bootstrap零宽区间也不是总体风险为零的保证。
科学裁决SUPPORTED仅针对登记的GRU方形命题；研究决策GO仅表示保留该参考版本进入后续受控验证，而非自动授权追加仿真、调参或训练。本任务完成并停止，后续唯一优先对照是相同调用预算的简单触发基线。
