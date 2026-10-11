# P3 新颖性与可声称范围

状态：已验证通过（限定范围的原始文献核查与数学核验）；不是系统综述、官方代码执行或基线复现。
检索日期：2026-10-10。没有依据实验结果缩小检索范围。

## 1. Action gap：现象及稳定性动机已有成熟先例

Bellemare、Ostrovski、Guez、Thomas、Munos，AAAI 2016，
*Increasing the Action Gap: New Operators for Reinforcement Learning*。
论文已研究最优与次优Q之间的间距、近似/估计误差对贪心决策的影响，以及保持最优性而扩大间距的算子；特别讨论了细时间尺度控制。
[原文全文](https://arxiv.org/html/1512.04860)，[AAAI记录](https://ojs.aaai.org/index.php/AAAI/article/view/10303)。

本项目测量的两个量必须分开：

- 后继价值间距：gap_V = V(H'_a1) - V(H'_a2)，候选按V排序。
- 未过滤前瞻分数间距：gap_raw = top1(r_a + gamma V(H'_a)) - top2(r_a + gamma V(H'_a))。

修订后同时测量Decision Gap：奖励、风险项及既有索引0扣分后的完整分数，在最终可选集合中取前两名之差。安全候选至少两个时才定义正常Decision Gap；仅一个安全候选时记为空；全部不安全时单列fallback间距。绝不将-1e9掩码带入间距。

三者的前两名可以不是同两个动作。raw和完整分数只是当前系统的单步近似，不是真实Q或最优Q。
不能把gap_V直接命名为论文中的action gap，不能声称首次发现“小间距导致可能不稳定”。
现有MC状态价值拟合也不等同于论文的Q算子迭代；不能未经训练合同设计就移植其算子。

Zhang、Gan、Tan，AAAI 2022，*Robust Action Gap Increasing with Clipped Advantage Learning*，进一步讨论盲目扩大间距在近似最优动作错误时可能带来的误差累积及收敛问题，并提出受控的扩大方式。[AAAI原文记录](https://ojs.aaai.org/index.php/AAAI/article/view/20900)。因此本轮不能从“小间距”直接推导“应扩大间距”。

独立数学说明：若各动作分数误差绝对值不超过epsilon，原最优动作与其他动作的间距大于2epsilon可保证排名不变；反过来不保证一定翻转。
所有候选同时加同一个偏移会改变每个V，却不改变排序。因此“扰动绝对值大于间距的步数占比”不是翻转的充分证据，需要候选之间的差分扰动。
gap_V / abs(V_top1)还依赖价值零点；平移所有V可改变该比值而不改变排序。这一比值只是特定奖励口径的描述量。

## 2. 导航中的扰动与动作价值稳定性也已有先例

Lütjens、Everett、How，CoRL 2019 / PMLR 100（2020），
*Certified Adversarial Robustness for Deep Reinforcement Learning*。
该文对观测扰动下的动作价值做界定，并在人群避碰任务中验证鲁棒决策；不是仅在Atari上的抽象先例。
[会议原文与作者PDF](https://proceedings.mlr.press/v100/lutjens20a.html)。

因此，“导航网络受输入扰动影响”“需要鲁棒动作评价”均不能作为新颖性。
尚可保留的是：对本VL接口的多检查点诊断、共同状态控制，以及把近似并列候选与实际一步净距展度并列测量。
但本轮未发现同样组合，不等于已经证明该组合首次出现。

## 3. 预测更准不等于导航更好：已有直接经验研究

Ivanovic、Pavone，2021，*Rethinking Trajectory Forecasting Evaluation*，指出同等大小的预测误差可具有不同规划后果，并研究任务相关评价。[作者论文](https://arxiv.org/abs/2107.10297)。预测准确率、动作翻转率和导航收益必须分别验证；H3-Random未证实前两者与真实导航收益之间的链条。

Stratton等，HRI 2026，
*How Human Motion Prediction Quality Shapes Social Robot Navigation Performance in Constrained Spaces*。
研究覆盖80名参与者、两个场地与两种平台，报告预测ADE并非导航表现的可靠代理，复杂预测器也未必优于简单障碍物模型。
[作者项目与论文](https://fluentrobotics.com/pred2nav/)，[论文记录](https://arxiv.org/abs/2601.09856)。

它支持“需要任务后果评价”的边界，不等同于本项目0.25秒、冻结critic、alpha=0.3的影子重评分结论。
不能把H3未见收益扩张成所有预测无用，也不能把既有现象重新命名为本项目发现。

## 4. 训练seed方差：一般问题已知，导航报告必须逐文核查

[Henderson等，Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)
和[Agarwal等，Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
已经讨论RL训练变异与少量独立运行下的不确定性。
这支持报告多seed限制，但不能用它们给本导航系统的方差填数字。

本次定向检查的三条导航研究线如下；SoNIC及其扩展不算两条独立证据：

| 研究线 | 文本中可核实的seed口径 | 允许的结论 |
|---|---|---|
| Improving Generalization in RL Training Regimes for Social Robot Navigation | 已查全文未找到明确独立训练seed数量 | 未报告/未核实，不能认定单seed |
| Characterizing the Complexity of Social Robot Navigation Scenarios | 每条件500个场景；固定随机seed用于共同评测场景，非训练seed数量 | 评测配对可核实，独立训练数量未核实 |
| SoNIC及Towards Generalizable Safety扩展 | 主结果明确使用3个训练seed、5个测试seed；部分cost-limit附录固定同一训练seed | 已有明确多训练seed先例；附录消融不能代表全文 |

来源：[Generalization原文](https://arxiv.org/html/2308.14947v2)、
[Complexity原文，IV-C](https://arxiv.org/html/2405.11410v2)、
[SoNIC原文，V-D及附录C/F](https://arxiv.org/html/2407.17460v2)、
[扩展原文，4.4及附录I/J](https://arxiv.org/html/2508.05634v1)。
扩展文的OOD group条件明确报告基线对训练seed高度敏感；这不是本项目方差的估计。

另查[PRISM预印本](https://arxiv.org/html/2609.18125v1)，未检到明确seed说明；仍归为未核实，而非确证单seed。
这是四条研究线的有限文本核查，不是领域总体比例调查。
本次确证“仅一个训练seed支撑全部跨密度结论”的研究线为0条；独立训练数未核实3条，明确多seed1条。
不能据此说领域多数论文单seed，更不能凭未检出词语认定实际训练次数。

本地419/443的四种匹配臂只提供观察到的绝对差，不能宣称已估计训练方差或统计噪声地板。

## 5. PBRS：argmax不变有明确条件，不能用一句话概括

Ng、Harada、Russell，ICML 1999，
*Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*。
[作者原文](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。
[Wiewiora，JAIR 2003](https://arxiv.org/abs/1106.5267)进一步研究势函数塑形与Q初始化的等价关系。
arXiv上传时间2011不是该工作的发表年份。

在势函数有界的折扣无限时域，或终端势正确归零/处理的有限时域，令：

```
r' = r + gamma Phi(s') - Phi(s)
V'_pi(s) = V_pi(s) - Phi(s)
r' + gamma V'_pi(s') = r + gamma V_pi(s') - Phi(s)
```

右侧只减去与候选动作无关的常数，argmax不变。这是已知不变性在一步前瞻表达式中的代数表现，不是新发现。
以上推导是本次独立核验；不是声称当前近似网络满足所有理论前提。

若只给reward加塑形而保持原冻结V，则分数多出gamma Phi(s')，通常会改变argmax。
因此“PBRS在r+gamma V结构下总不改变argmax”不成立。
有限时域终端处理、近似学习误差、历史是否构成充分状态均需另行说明。
最优策略不变不等于有限预算IL和在线MC学习过程不变，也不能排除PBRS改善实际性能。当前没有PBRS训练对照，故其有限训练收益属于未验证，不能标为“解析证明无收益”。本轮零训练审计不否定或停止方向①的Reward实验；但也不据此自动启动任何新训练。
本目录pbrs-check.json保存一致变换不变与reward-only可翻转的两个CPU数值例子；不涉及训练或环境评测。

## 6. 最终可声称与不可声称

| 说法 | 判定 |
|---|---|
| 小action gap、误差敏感性及扩大gap是新概念 | 已确认失败：已被已有RL/导航工作覆盖 |
| 预测更准却未必改善导航是新发现 | 已确认失败：已有直接经验研究 |
| PBRS一致变换的策略不变性是新发现 | 已确认失败：经典理论 |
| 11份本地权重在自己的轨迹和共同状态上的V间距与净距展度 | P1完成后可作为该系统的实测诊断，不能自动声称新颖性 |
| 小间距由MC目标造成、导致碰撞，扩大它可改善导航 | 未验证：缺目标干预、真实后果与修复验证 |
| 小间距解释多个测试期干预全零 | 未验证：先核对各干预并非全零，再检验差分扰动/价值误差/后果链条 |
| 当前结果足以立项一个全新算法 | 未验证：尚无独立技术差分与收益证据 |

推荐：把路线B定位为受控诊断证据，不包装为已验证的新方法；最多保留一个针对真实后果的后续核验建议，本轮不自动执行。

本轮不属于KDA性能优化算法，也不是Delta Rule机制创新。即使11份权重出现相似间距，也只能报告跨已测配置重复；动作网格、前瞻接口、训练流程等共同因素尚未分离，不能归因于稀疏Reward或MC目标。
