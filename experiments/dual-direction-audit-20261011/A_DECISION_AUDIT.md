# A：训练语义、真实动作损失与方法候选审计

【研究结论】存在真实的部署续策动作排序损失，且替换真实下一历史后仍大量保留；本轮没有方法级进展，合格候选为 0。
【核心证据】KDA seed443、case90021、root239：idx69 的实际下一历史 Q=0.675464、根回报=-0.088991、timeout；idx24 的对应值为0.635800、0.547086、success。相同现象也见 GRU。
【科学裁决】SUPPORTED：真实下一历史不能消除全部错误排序。INCONCLUSIVE：训练续策不匹配或 KDA 本身导致这些导航损失。
【研究决策】KILL：停止把“同根真实 MC＋局部排序／续策匹配”作为独立方法路线投资；不否定工程修复价值，不封杀整个导航研究。
【执行结果或下一动作】交付本报告；不启动该通用候选的训练或闭环实验。已知 history 合同问题应列入公平基线维护，不计入方法贡献。

## 1. 范围与执行边界

- 独立读取训练、IL、演示、MC replay、动作评分、空间/时间编码、探索、部署合同及 Git 历史；未读取另一研究 Agent 的推理。
- 已执行：CPU 流式读取原始 gzip、回报重算、根/索引一致性断言、SHA256、原文及官方源码阅读。未执行：神经网络推理、官方代码运行、模块替换、训练、GPU、新闭环评估或源代码修改。
- 共享源目录为 /home/abc/workspace/nav_data/mamba/camrl/CrowdNav，以下相对源码路径均相对此目录；当前 HEAD=86969cd4fc71632bc31f54094dad8c539fc8c8f2。主 Agent 的334文件全量清单核验不冒充 A 自己完成；A 单独读取并哈希下列关键文件。
- 原始数据目录为 /home/abc/workspace/ttc5-causal-20261011。ROOT_CAUSE_RESULT.md 只用于定位；2.4GB root_cause_data.jsonl.gz 只读取前两行。结论依据 work/ 原始分支、roots、oracle、probe、配置和训练日志，而非分析摘要。
- 本报告集中统计 cell5：20人 square、TTC5 观测的既有分支。选择失败根及有限备选动作的结果，不是全部状态/80动作的总体错误率。

## 2. 实际训练的对象不是动作 Q 克隆

|环节|实际计算及精确源码位置|含义与边界|
|---|---|---|
|IL 标签|crowd_nav/train.py:786–888，尤其849–863；1036–1061|逐步完整演示回报 G，24帧窗口，标量 V 的 MSE；当前标量路线不以 action_indices 作动作分类。不能称为动作 BC。|
|在线 RL|train.py:551–584，564、566、576；2696|仅取 states、returns，优化 MSE(V(H),G)。target_net、next_state、done 不进入这项损失；名称 sarl_style_update 不代表当前实际运行 TD。|
|完整 MC|crowd_nav/utils/ppo_buffer.py:340–398，372–376|从 episode 末端 G=0 反向 G=r+0.99G；不是24步截断回报，也不是n-step bootstrap。24 是输入历史长度。|
|续策与采样|train.py:2328、2546–2553、2559–2600；ppo_buffer.py:400–416|不同训练时期的 epsilon-greedy 轨迹混合，epsilon .3→.05/1500ep；均匀 replay，PER关闭。有限样本标签并非精确部署 V。|
|实际动作得分|crowd_nav/policy/mamba_rl.py:793–907，813、845–876、883–900|CV 人体下一状态＋候选机器人动作构造下一窗口，再算 r_hat+0.99V；随后部署 mask/risk。当前 V 网络本身没有动作输入。|
|历史缺口|mamba_rl.py:769–777；explorer.py:466；ppo_buffer.py:390–398|探索提前返回不追加在线历史；replay 却记录每一个真实帧。两者时间采样语义不同，不是“训练完全无历史”。|
|其他训练/部署差异|mamba_rl.py:782–791、892–902；explorer.py:102–128；train.py:2328|部署 smoothing=.3、margin=.2、risk=.8；训练动作不平滑、无相同过滤。replay 对原生34维状态加N(0,.02)，在线决策读干净状态。训练模式下空间 dropout=.1；no_grad 不等于 eval。各项导航损害尚未隔离。|

实际日志：seed419 为 /home/abc/4090_backup_20261009/root/kda-history-20261009/runs/kda-t24/train.log；seed443 为 /home/abc/workspace/kda-history-replication-20261010/runs/kda-t24/train.log。
两者80–110行附近记录：缓存15000条、实际选入 success=15000/collision=0、749318个24帧序列、IL50轮；缓存元数据90.14%不可误写为实际消费的成功比例。train.py:1626–1638 的选择会排除 timeout。
seed419 日志16994、seed443 日志16975：在线均 ok10000/skip0，步数577325/601048；末尾每ep updates=4、epsilon=.05、buffer=200000。不能归因为“在线样本没有进入训练”。
train.py:3694–3743 的 IL warmup=5000 指 episode，不是5000个 transition；它填满200000环形容量。在线步数已超过容量，不能把末期 replay 描述成仍主要由初始 IL 样本占据。低训练MSE（末尾约.0004/.0005）不能证明未见状态的相对动作值正确。

### 空间、时间与覆盖

- mamba_rl.py:35–132：robot token 全13维；human token slots3:8，关系特征、距离排序、attention 后 max pool，得到每帧场景向量，再由时序模块处理。goal/motion 专用 token 未单独进入该编码，但 robot token 已含目标信息，不能称“没有目标输入”。
- mamba_rl.py:734–752：原训练构造器取提供顺序的前5人；训练配置本身为5人 circle。reference.py:46–89 的全20人 TTC5 是本次评测包装，不能倒写为训练时已有的人群选择机制。
- contracts.py:398–513 没有显式绝对时间/剩余时限或人体ID跟踪；每帧池化后学习场景时间关系，不等于追踪同一行人轨迹。机器人速度已有上一执行控制信息，不能仅凭没有 action token 宣称完全不知上一动作。
- 连续 ORCA 演示、有限 epsilon 轨迹，不提供同一根下所有候选动作的标签。每次虽评分80动作，反向更新仍只监督访问到的窗口；“算过80个Q”不等于“训练覆盖80个反事实后继”。现有 replay 不含可重建所有行为概率/策略版本的信息。
- 输入部分可观测、训练5人到评测20人、未显式剩余时间、行为/部署支持差异均可产生残差；它们是已查明的表示/合同事实，不是已隔离的导航因果贡献。不继续逐项开展诊断。

### Git 与已知工程问题

git show 19e6031:./VL-v2-contract.md 第51–65行已明确保留探索 early-return 和 train-unsmoothed/deploy-smoothed。work/training-contract-probe.json 原有单元试验中 epsilon=1 时三次history长度均0→0；epsilon=0为0→1→2→3。A 读取了原始结果和当前代码，没有重跑该单元试验。
这是可复现、此前已知的历史合同问题；不是新发现的方法缺口。其导航损害未隔离。公平基线维护可把“每个已观测当前帧无论探索/贪心均追加一次”作为具体修复，测试随机/贪心/交替路径且避免重复追加；本轮不越权改源。
227210d 的 action-index/online-buffer 修复、717c238 的IL窗口处理、19e6031 的坐标/字段修复，以及86969cd的向量化MC路线均须按当前代码判断；不把旧版本已修错误用来解释现有损失。

## 3. 三种 Q 必须分开

1. Qpred = r_hat(endpoint,CV) + .99 V_hat(H_hat_next)。原始字段 first_step.q_pred；最终决策还可能叠加mask/risk/index0小惩罚。
2. Qactual = r_env + .99(1-terminal)V_hat(H_actual_next)。取 real-history-oracle 的 q_oracle1_actual_history_terminal_corrected，保持根之前的前缀不变，替换真实下一机器人/人体状态并重做下一步TTC选择。不能用 pinned-oracle 字段冒充它。
3. Qreal = sum_j .99^j r_(root+j)：根动作强制一次，之后恢复同一个冻结部署策略，直至该 episode 终止。字段 root_return，不是从episode开头算的 return；更不是Q*。

audit.py:171–230 更新强制动作后的 last_action 并恢复部署续策；226计算 root_return。real-history-oracle.py:42–77 固定原历史、检查所有动作的人体下一状态相同、构造真实下一TTC窗口、使用真实即时奖励和终止修正。
crowd_sim/envs/crowd_sim.py:349–355 与冻结配置 robot.visible=false 表明人体不观察机器人。此世界中不同机器人动作没有不同的人体响应；原分支 true_next_humans 相同亦被断言。动作仍改变机器人位置、观测人选及控制续策。当前损失不支持把“学习动作条件人体反应”立为候选。
这些是固定世界、固定续策下的确定性实现回报；不是对随机人类未来的期望 G。只有在指定条件分布和策略下，有限 MC 样本才是相应价值的样本估计。

用任意同根动作差 Δ，有 Δ(Qreal−Qpred)=Δ(Qactual−Qpred)+Δ(Qreal−Qactual)。这只是可复算的分解，不是两个模块的因果贡献比例。后一残差同时容纳值拟合/泛化、训练行为续策差异、有限历史混叠和部署分布变化；不能直接命名为“KDA长期记忆失效”。
若把这些分支标签拿来训练新策略：标签评估的是 π_old。部署时在许多状态反复切换动作，后续已成为 π_new，不能把单次强制分支收益累加成新策略收益。完整Markov、精确Q、全状态贪心的策略改进结论不直接覆盖当前选根、部分历史、近似值和有限动作标签。
如改用短rollout＋尾部V，则还引入截断bootstrap误差；本次原始标签是完整到终止的回报，不能把它们描述成短视MC或将两种标签混用。

## 4. 原始动作损失：失败与成功对照不混池

每根按 candidate_index 去重，只计 G_alt>G_A0+1e-6；“oracle纠正/仍错”依据 Qactual 的严格排序，均无1e-6内oracle平局。数字是动作对数，不是独立获救episode数。

|模型/seed|失败根G改善对|oracle纠正/仍错|成功对照G改善对|oracle纠正/仍错|合计改善对|
|---|---:|---:|---:|---:|---:|
|KDA419|192|85 / 107|118|41 / 77|310|
|KDA443|216|60 / 156|164|45 / 119|380|
|GRU419|126|40 / 86|96|28 / 68|222|
|GRU443|127|52 / 75|126|40 / 86|253|

全部这些改善对的首步均非terminal，备选动作均未被mask，两动作risk均0。Qpred严格反向除GRU419失败组125/126外均覆盖所有改善对；剩余一个未超过1e-6严格阈值。不能把近似平局报成严格排序错误。
失败组 timeout/collision→success 的改善对分别114、110、62、82，仍非独立episode计数。成功对照中A1改动原成功轨迹后变失败的个数为9/87、9/109、7/75、7/88（分母为与A0不同的A1对）。所以有限动作改进并非无风险，更不能从失败根结果推出整体成功率增益。

### 可定位案例

|模型，case，root，step；gzip行|动作|Qpred|Qactual|Qreal根回报|整集return；终局/时间|
|---|---|---:|---:|---:|---|
|KDA419，90001，1，28；行2|A0 idx67|.6036496162|.5622447729|−.0888181583|−.1198842216；timeout/50.25|
|同根|A3 idx68|.6027882099|.5540406108|.2097715674|.1054672034；success/46|
|KDA443，90021，239，25；行18|A0 idx69|.6770335436|.6754642725|−.0889908902|−.0692190152；timeout/50.25|
|同根|A2 idx24|.6426010728|.6358004212|.5470859837|.4255351635；success/20.25|
|GRU443，90001，682，21；行1|A0 idx60|.5851448774|.5904600024|−.0905505477|−.0763902580；timeout/50.25|
|同根|A1 idx5|.5846170783|.5886909962|.3384784131|.2710064479；success/31.25|

上述文件分别为 work/branches-kda-s419-t24-cell5.jsonl.gz、branches-kda-s443-t24-cell5.jsonl.gz、branches-gru-s443-t24-cell5.jsonl.gz。
实际执行command分别为：root1 A0=(.2702296049,−.2435835572)、A3=(.3331227075,−.3954209384)；root239 A0=(.4135708630,−.4283551064)、A2=(.1456924603,.9183605664)；root682 A0=(.1272520662,−.0553949774)、A1=(.2105821878,.0693173628)。均为平滑后真正执行命令，不把index当成未平滑速度。
root239：首步奖励均0，预测clearance .734442927/.422895924；真实下一TTC IDs由[3,4,8,12,15]变为备选的[3,4,8,9,15]。因此Qactual确实包含观测选择变化。两支全程最小实际clearance .179264007/.102040811，不可仅凭备选成功便宣称更安全。
root1还包含反例：A1 idx5 的 Qpred=.5853680372、Qactual=.5851246715，高于A0的Qactual，G=.6135768082、success/19；同一根的A2 idx24却 G=−.1098834788、timeout。说明真实下一状态纠正和价值残差可共存，最大即时进度也不是普遍解。

## 5. 因果裁决与工程缺陷

|明确命题|裁决|研究含义/处置|
|---|---|---|
|现有备选动作在冻结部署续策下确有更高真实回报，却被低估|SUPPORTED|上表及完整回报复算支持真实损失，不是只看预测误差。|
|所有错误都只是CV一步预测/固定人选引起|REFUTED|真实下一历史仍有107/156/86/75个失败组反向对；不能据此量化剩余原因。|
|这些是KDA特有的时序缺陷|INCONCLUSIVE|GRU有同类损失；共享路线不能识别KDA专属因果，不据此替换骨干。|
|探索历史缺帧、训练/部署续策不一致真实存在|SUPPORTED|代码、原始probe及旧合同共同支持；只是合同事实，不是导航因果效应。|
|它们造成了表中损失，修好即可恢复排序|INCONCLUSIVE|未做隔离干预；普通on-policy MC/FQE是必要竞争解释，不能包装为新算法。|
|当前评分在终止动作后仍bootstrap|SUPPORTED|mamba_rl.py:883–890 未乘(1-terminal)，与完整MC终端语义不一致；属于具体工程错误。|
|terminal bootstrap解释这里全部排序损失|REFUTED|这些改善对首步均非终止，oracle也显式终止修正。该缺陷应修，但不能冒认当前病例归因。|
|filter/risk被全局排除为原因|INCONCLUSIVE|这里只有至多3种备选动作且零mask/risk；只排除此局部对的直接分数惩罚，不能替所有80动作及后续状态免责。|
|同根双真实回报＋局部排序本身建立独立贡献|REFUTED|局部偏好已由ACoB/RankQ实现；真实cost-to-go反事实监督已由rollout PI/LOLS覆盖。标签更可信不等于新增方法。|

环境reward采用swept clearance，策略r_hat主要是endpoint CV（mamba_rl.py:817–843；crowd_sim.py:367–403），到达阈值也有.25与robot radius .3之别。oracle使用真实即时reward，不应把替换后残差仍归给这项误差。环境步进的人体新ORCA动作与即时reward使用的当前速度还需区分；本轮不把实现不一致都提升成主因。

## 6. 最近原文与官方实现边界

均为阅读，未执行官方实现或复现论文结果。公式按计算意义转述；与当前路线的差别不得由论文标题代替源码检查。

|原始来源/公式|核验的官方源码与结论|
|---|---|
|[SARL](https://arxiv.org/pdf/1809.08835)，Bellman与Algorithm1：y=r+γ^(dt·vp)V_target(s')；正文训练用实际后继|[官方 explorer](https://github.com/vita-epfl/CrowdNav/blob/master/crowd_nav/utils/explorer.py) 的IL累计回报与RL一步bootstrap；[multi_human_rl](https://github.com/vita-epfl/CrowdNav/blob/master/crowd_nav/policy/multi_human_rl.py) 的query_env/CV分支。当前MC-only、γ=.99每步和24帧时序不是原SARL训练。|
|[RGL](https://arxiv.org/pdf/1909.13165)，多步规划混合当前V与递归max(r+γV)|[model_predictive_rl.py](https://github.com/ChanganVR/RelationalGraphLearning/blob/master/crowd_nav/policy/model_predictive_rl.py):177–275 的clip/递归规划；[trainer.py](https://github.com/ChanganVR/RelationalGraphLearning/blob/master/crowd_nav/utils/trainer.py):123–129 的target-next-V。论文规划和训练target源码分别看，不能用骨干差异声称解决续策监督。|
|[CAMRL](https://arxiv.org/html/2408.02661v1)，III-A/B：IL后RL、时序编码与值MSE|论文已读。作者官方训练仓库未核验，标UNVERIFIED；本地origin jinglongjiang/camrl不自动视作作者官方实现。不把本地MC改造说成原CAMRL精确复现。|
|[FQE](https://proceedings.mlr.press/v97/le19a/le19a.pdf)，Algorithm3：y=c+γQ_prev(x',π(x'))|[作者实现](https://github.com/clvoloshin/constrained_batch_policy_learning/blob/master/fitted_off_policy_evaluation.py):47–63读取done、指定π后继动作及终止mask。它已经显式绑定被评估续策；普通on-policy MC同样是强基线。|
|[Rollout sampling approximate PI](https://pure.uva.nl/ws/files/4134259/59678_289644.pdf)，逐状态/动作固定续策rollout后做cost-sensitive改进|原文已读；其完整作者代码未核验。真实两支终局回报并不是超出该思想的能力。AggreVaTe作者源码/公式本轮未独立核验，标UNVERIFIED，不将其作为唯一论据。|
|[LOLS](https://proceedings.mlr.press/v37/changb15.pdf)，Algorithm1逐动作的learned/expert continuation cost-to-go；[补充材料](https://proceedings.mlr.press/v37/changb15-supp.pdf)C给出作者代码|[作者VW search.cc](https://github.com/KaiWeiChang/vowpal_wabbit/blob/icmlexp/vowpalwabbit/search.cc) 的POLICY/ORACLE/MIX_PER_ROLL、learn_loss及generate_training_example已读。续策匹配和反事实行动监督不能只与SARL/RGL对比。|
|[RankQ v4](https://arxiv.org/html/2605.11151v4)，Sec3 Eqs4–6：多项相对排序联合TD|[作者项目伪代码](https://horizonrobotics.github.io/gail/rankq/) compute_rankq_loss：softplus(q_neg−q_pos)，成功/扰动/随机启发式标签。读到的是伪代码，完整训练repo未核验；我们的实际branch G区别在标签来源，不自动形成算法差分。|
|[VLA-Precision](https://arxiv.org/html/2609.04355)，III-B1 Eqs9–10：同状态纠正执行动作优于被覆盖提案，平方hinge＋TD|[固定提交官方 agent.py](https://github.com/scy-v/VLA-Precision/blob/41edacde80d8c9d8146d610202299d71b0e79f7c/src/vla_precision/acob/agent.py#L417)：446–464计算good−bad advantage平方hinge；575–588加入critic TD。被覆盖未执行动作不进入TD。两支都执行过提供更强标签，但标准rollout PI已经提供这种监督能力。|
|[REBEL](https://arxiv.org/html/2404.16767)，相对回报与策略log-ratio差回归|原文已读，作者实现未执行/未完整核验；它不是本地V网络训练，但也排除“使用相对奖励本身即创新”的论证。|

终端修复、mask-aware train/deploy一致化是基线维护。成熟shielding工作不因这里需要过滤就变成新能力；本报告不提出shielding候选，也不以未核验代码宣称复现其效果。

## 7. 候选排序与停止条件

**保留候选：0。唯一被具体审查后封存的候选族 C0：部署续策同根反事实MC＋局部偏好。**

- 对应真实失败：root239、root1等“真实下一历史仍偏爱最终timeout动作”；不是用无真实结果的低Q差代替导航失败。
- 精确拟议计算：冻结π_k，克隆同根状态分别执行a/b并续行π_k至终止，得G_k(h,a/b)；以完整回报拟合候选后继值或动作Q，再加 L_pair=[max(0,m−(Q(h,a_good)−Q(h,a_bad)))]²；下一轮更新策略时重新标定续策。仅替换Q头、hinge为softplus、或用G差权重都不增加独立能力。
- 已确认支持：局部真实regret、两种骨干共享残差、MC单窗口监督没有直接同根排序项。未确认支持：该项在未见根/反复部署切换后改善成功率，或比普通部署策略MC/FQE更有效。
- 最强竞争基线：相同预算/根/完整branch标签的普通rollout PI或cost-sensitive动作学习；冻结部署策略on-policy MC/FQE；相同标签下ACoB式hinge/RankQ式softplus；再加不学习的安全最大进度动作。不能只比原始未经合同修复的模型。
- 最近差异判断：ACoB已有同状态局部偏好＋全局回报传播；RankQ已有多项排序；rollout PI/LOLS已有真实续策cost-to-go。当前差异只剩标签获取/任务适配，没有识别出未覆盖的计算能力。因此 KILL 是资源决策，不是“改善在数学上不可能”。
- **METHOD_SPEC_FREEZE=no。** 门槛是给出一个具体新增算子/状态/决策规则及其必要性，能够在相同真实branch标签、相同续策、相同预算下区别于上述强基线；然后才允许冻结最低实现与闭环门槛。不得以“再多收一些动作对”替代此门槛。
- 唯一可能翻转的方法证据项：一份可执行的非等价见证，展示所提新增计算在一个已经出现的导航决策结构上有上述基线做不到的决策能力，而不是标签更多、hinge不同或合同修复。预算上限2 CPU小时、无训练；若只是上述操作的重参数化/组合，维持KILL；若存在明确新增能力且可实现，才转向最小方法实现。当前C0的算子已明确，未提供这种见证，**不是HOLD，也不授权继续诊断**。
- C0不存在能凭更多成功率数字补足独立机制的实验，所以本轮训练/闭环预算为0；不设计“无论什么结果都继续查”的实验，不强造第二候选。动作条件人体响应在当前不可见机器人世界中也没有损失支持。
- 若以后确有新算子通过上述门槛，最小闭环必须按训练seed与环境case分组，与相同数据/预算的最强基线比较；同时报告success、collision、timeout、导航时间/路径效率、推理时延及训练/rollout成本，包含成功对照。单根干预收益不充作闭环增益。建议封顶2seed×200未见cases、单一预注册配置、≤8 GPU小时（成本估计UNVERIFIED）；成功率需提升≥5个百分点且collision不升、效率退化≤5%、推理耗时≤2倍，否则封存。该条件仅是新机制准入后的边界，不是当前候选GO。

## 8. 可独立复算入口

以下自包含只读CPU代码无需临时目录；只流式读cell5分支，重算根回报、核对root及oracle，按原索引去重。与本轮执行脚本相同聚合定义；tol=1e-6，回报一致性tol=1e-12。

```bash
python - <<'PY'
import collections, gzip, json
from pathlib import Path
B=Path('/home/abc/workspace/ttc5-causal-20261011/work')
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f: yield json.loads(line)
O={x['root_id']:x for x in rows(B/'real-history-oracle.jsonl.gz')}
R={x['root_id']:x for x in json.load(open(B/'roots.json'))['roots']}
for model in ('kda-s419-t24','kda-s443-t24','gru-s419-t24','gru-s443-t24'):
    C=collections.defaultdict(collections.Counter)
    for ln,x in enumerate(rows(B/f'branches-{model}-cell5.jsonl.gz'),1):
        r=x['root']; rid=r['root_id']; o=O[rid]; assert r==R[rid]
        assert (o['model'],o['case'],o['step'])==(model,r['case'],r['step'])
        assert o['prefix_unchanged'] and o['pinned_oracle_bitwise_reproduced']
        q={b['candidate_index']:b['q_oracle1_actual_history_terminal_corrected'] for b in o['branches']}
        bs={b['candidate_index']:b for b in x['branches']}
        a=next(b for b in bs.values() if 'A0' in b['labels'])
        assert abs(a['return']-r['baseline_return'])<1e-12
        for b in bs.values():
            g=sum(.99**j*s['reward'] for j,s in enumerate(b['steps'][r['step']:]))
            assert abs(g-b['root_return'])<1e-12
            i=b['candidate_index']; scope='success' if r['type']=='success-control' else 'failure'
            if i!=a['candidate_index'] and g>a['root_return']+1e-6:
                z=C[scope]; z['improved']+=1; d=q[i]-q[a['candidate_index']]
                z['correct' if d>1e-6 else 'wrong' if d< -1e-6 else 'tie']+=1
                z['masked']+=b['first_step']['predicted_hard_masked']
                z['risk']+=any(v['first_step']['risk_penalty']!=0 for v in (a,b))
                z['terminal']+=any(v['first_step']['terminal'] for v in (a,b))
        if rid in (1,239,682):
            print('CASE',model,ln,rid,r['case'],r['step'])
            for b in bs.values():
                print(b['labels'],b['candidate_index'],b['first_step']['command'],
                      b['first_step']['q_pred'],q[b['candidate_index']],
                      b['root_return'],b['return'],b['outcome'],b['time'])
    print(model,dict(C))
PY
```

其他精确定位入口：`git -C /home/abc/workspace/nav_data/mamba/camrl/CrowdNav show 19e6031:./VL-v2-contract.md`；对上述源码用`nl -ba`查看表中行。配置入口 inputs/kda-s419-t24/config.ini；训练合同入口 work/training-contract-probe.json 的 unit_results[*].trials、candidate_commands。日志/配置与manifest已读；未重新哈希整个manifest权重集，不能声称A完成全量provenance复现。

### 本轮实际计算的 SHA256

相对原始数据目录；hash针对压缩文件本体，不是解压文本。
```text
work/roots.json 665da8a3e170b40f8106e0f634c738b576e2dc7d887c508fab3a922f13adacb2
work/real-history-oracle.jsonl.gz c1b348d2336a99f923d2735a60328c7c3c83921642070fb074a34982e4f13029
work/training-contract-probe.json 889dcd0c3daed4adc57ca373d1fb692179f0858f33d67237fb32d942032d5c22
work/branches-kda-s419-t24-cell5.jsonl.gz d3a7dc6d0fc9e2fa7f8aa0bbd050f3f4a217ccfe322fd9a6a092695ad2624415
work/branches-kda-s443-t24-cell5.jsonl.gz 390295f5e6d6486b92cd1a37db6707bf19c5be92dcbdc1f310e904c104b8d233
work/branches-gru-s419-t24-cell5.jsonl.gz f078771b4673eda754dd1242bd785e27a31e751b5cdcb0f94dcdabde0bbc74cd
work/branches-gru-s443-t24-cell5.jsonl.gz b2b14deed0bae40a7464d7e63fd23a7545a119efbcd649d59d5fd88975a2026e
crowd_nav/train.py b777188aedfcb61375f0d28c9748a7655bef8612ae90c5be24f0e40f789315ca
crowd_nav/policy/mamba_rl.py 2fd464618507796fdbc1b072d96563c97180e301f842cac1a6e6f75212ff6616
crowd_nav/utils/ppo_buffer.py 4155eba3c199165320f55aeefa1f4e7693a2b04c523ee3ffadb2892ce96b2744
seed419 train.log 385a0a3a303211ff17d3a7e32f4c1449a4922885b435285a357b76afc1ad559a
seed443 train.log c6dcb0064c06e77a7f951442327bcd290b69b884947fcb7b24b4099fe3581128
```

可复核命令：`sha256sum`加对应完整路径。唯一临时脚本 /home/abc/workspace/dual-direction-audit-20261011/direction-a/audit-original-pairs.py 可以清理；本报告内嵌入口不依赖它。本轮没有额外JSON交付物。
