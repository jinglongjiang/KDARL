# B：人数迁移机制独立审计

【研究结论】本轮没有可独立冻结的方法候选，没有方法级进展。人数上升伴随超时、绕行和动作排序损失，但不能归结为单一 OOD value 机制。
【核心证据】独立复算 4256 条轨迹；严格跨人数局部机会匹配为 0；既有同世界单动作干预可救回部分失败，同时损伤成功对照。官方文献及源码已有人数摘要、风险融合、课程分布扩展和多步规划。
【科学裁决】INCONCLUSIVE：命题“5 人训练到 20 人的退化主要来自 OOD value，而非任务难度、输入混叠、覆盖和执行结构”尚不可识别。
【研究决策】KILL：封存本轮 B 新候选立项，不扩跑泛化曲线，不否定整个研究方向。
【执行结果或下一动作】交付本报告；候选数 0；METHOD_SPEC_FREEZE=NO。主 Agent 可直接综合，不等待 B 新实验。

## 1. 范围与原始证据

- 本审计仅离线读取、聚合已有记录、阅读论文和官方源码；没有训练、GPU、模型推理、新闭环或共享源码修改。文献阅读≠官方代码执行≠受控替换≠完整复现。
- 轨迹根目录：/home/abc/workspace/ttc5-causal-20261011/work/ 。16 个 baseline 文件共 4256 episodes；cell0/3 各模型各 32，cell4/5 各 500。cell0=5Circle、3=10Square、4=20Circle、5=20Square。病例 90000 起。
- 旧六格原始 JSON：419=/home/abc/4090_backup_20261009/root/kda-history-20261009/evaluation/；443=/home/abc/workspace/kda-history-replication-20261010/evaluation/。各含 `{kda,gru}-t24/cell{0..5}.json`，各 500 episodes；病例 91000 起。不能把两批病例串成同一轨迹。
- scale-consistency 的 ERROR.json 为 parity FAILED_CLOSED（443/KDA/20Square，首个不一致 91004）；NEW12/15 未执行，不补格、不推断缺失结果。ROOT_CAUSE_RESULT.md 仅作为索引，以下数字来自原始记录。
- 原始 selector/决策字段明确为每步 `humans`（全部 N）、`ids`（所选 5）、`current_token`、`history_sha256`、`commands`、`chosen`、`progress`、`clearance`、`raw_scores`、`scores`、`n_safe`、`env_constraint`、`actual_swept_clearance`；不是仅凭汇总 SR 猜测机制。
- 实际 selector 见 /home/abc/workspace/ttc5-causal-20261011/reference.py:46–84；生产对应 /home/abc/workspace/kda-selection-20261008/evaluate-selection.py:45–84：closing=−r·Δv/(dist+ε)，closing>0.1 时按 dist/(closing+ε)，否则按 10dist 取最小五人。每真实帧重选，同帧 80 候选固定 ids；不是永远盲于固定的其余 15 人。
- 共享源码根目录 /home/abc/workspace/nav_data/mamba/camrl/CrowdNav/：`crowd_nav/policy/mamba_rl.py:734–753,793–906` 仅五人进入 value token，但全部人进入 CV endpoint reward/clearance；有安全候选才硬屏蔽 clearance<0.2，其后风险惩罚。`crowd_sim/envs/crowd_sim.py:349–389,458–461` 区分环境旧速度扫掠判碰与实际 ORCA 动作后的扫掠。预测 endpoint 安全不是物理安全保证。
- /home/abc/workspace/kda-history-replication-20261010/kda-t24.ini：[env] time_limit=50、dt=0.25；circle R=4、square W=10；5×16 动作，include_stop=false；测试平滑 0.3previous+0.7grid。eval_protocol 的 25 秒不覆盖环境期限。RL 场景配置为 5Circle。B Agent未核验IL人数；主Agent收尾时亲自CPU读取同SHA的原始IL缓存：config.human_num=5，15000轨迹、749318帧全部shape(34,)，机器人起点全部(0,-4)。这补足缓存声明及存储槽位证据；缓存没有逐轨迹场景/实际全人群数量记录，不能由34维反推隐藏人群不存在，也不额外宣称重现了采集过程。

主Agent补核的原始IL缓存为：
/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/data/il_dataset_diverse_v5.0_merged_15000_with_idx.pth

SHA256=940ae49af4b641d23b7c8bdf47f548dddce5db128b7ea0beaabe1e897459a8e2，与seed443 protocol.dataset_sha256一致。缓存config还记录旧教师81动作（even/include_stop=true），而部署80动作；本轮标量IL用原始状态及MC回报，不以这些旧action_indices训练分类器，不能仅凭81就将本轮评测作废。元数据完整原始字段可用torch.load(path,map_location='cpu')['config']复核，未使用GPU。

## 2. 人口统计：不是只有成功率

旧六格 S/C/T（成功/环境碰撞/超时），每格 500；列依次为 5Circle、5Square、10Circle、10Square、20Circle、20Square：

|模型|5C|5S|10C|10S|20C|20S|
|---|---|---|---|---|---|---|
|KDA419|490/2/8|484/6/10|480/14/6|473/19/8|436/35/29|384/50/66|
|GRU419|494/3/3|491/3/6|488/2/10|487/5/8|450/18/32|397/35/68|
|KDA443|494/3/3|491/6/3|485/3/12|480/7/13|451/16/33|355/29/116|
|GRU443|498/1/1|489/1/10|492/1/7|492/4/4|461/14/25|393/37/70|

5Circle 已有 98–99.6% 的绝对提升天花板；10Circle 仍为 96–98.4%，不能泛称十人全部失效。20Square 的损失包括明显超时，非只有碰撞。以下来自另一批逐步轨迹：U=无 endpoint 安全动作的 episode 内帧比例；M=存在 clearance≥0.2 且 progress≥0.02 的动作、但 chosen progress<0.02 的帧比例；L=command speed<0.1 的帧比例。先算每 episode 比例再平均，绝不把相关帧当独立样本。T 为成功条件下平均秒数，D 为所有结局平均指令路径长度（m），不是成功效率的无偏因果量。

|模型/格|S/C/T|U%|M%|L%|T秒|D米|
|---|---|---:|---:|---:|---:|---:|
|GRU419/5C|31/1/0|0.52|2.77|0.20|10.75|8.86|
|GRU419/10S|31/1/0|0.98|7.96|0.49|11.98|9.92|
|GRU419/20C|447/17/36|1.55|16.84|0.58|15.58|14.31|
|GRU419/20S|408/27/65|3.24|20.07|0.74|17.29|15.73|
|GRU443/5C|32/0/0|0.00|6.12|2.01|11.30|9.01|
|GRU443/10S|31/1/0|2.30|9.11|1.19|13.03|10.20|
|GRU443/20C|469/19/12|1.52|19.03|1.86|16.64|12.82|
|GRU443/20S|390/28/82|3.70|25.47|1.72|19.44|15.94|
|KDA419/5C|32/0/0|0.63|5.77|0.13|11.37|9.65|
|KDA419/10S|31/1/0|1.68|9.52|0.79|12.90|10.17|
|KDA419/20C|444/35/21|2.46|21.04|0.78|17.52|13.74|
|KDA419/20S|389/39/72|4.34|21.46|1.81|18.84|15.38|
|KDA443/5C|31/0/1|0.00|8.71|1.94|11.52|9.71|
|KDA443/10S|31/1/0|1.89|12.93|0.53|13.16|10.48|
|KDA443/20C|456/12/32|1.18|23.56|2.35|17.76|13.99|
|KDA443/20S|362/39/99|4.39|25.66|3.55|19.67|15.74|

M 上升支持“已有离散动作中存在未选的局部前进机会”，不证明该动作能安全完成任务，也不等于静止 freeze。KDA20S 每帧平均安全动作约 65.63/66.02；最近环境约束人不在 ids 的比例 29.91%/27.65%，每帧平均换入 1.48/1.43 人。可定位部分输入筛选压力，但全人 safety/reward 路径存在，不能据此断言纯 missing-input 致败。

## 3. 真正相近机会、几何与配对的边界

严格匹配在计算结果前固定：每 episode 取首个 step≥8 且 goal_distance≤4m 的机会；仅同模型、同几何 5C→20C、10S→20S。先要求时间差≤5s、目标距离差≤0.25m、目标坐标系机器人速度差≤0.2m/s、n_safe差≤8、2m人数差≤1、全部人最小表面距离差≤0.2m。再对所选五人做 Hungarian 无序配对，成本 Δposition²/0.5²+Δrelative_velocity²/0.3²，要求 RMS位置≤0.5m、RMS相对速度≤0.3m/s；候选按归一化差异排序、不放回匹配。

|模型|5C机会/20C机会/粗候选边/严格匹配|10S机会/20S机会/粗候选边/严格匹配|
|---|---|---|
|GRU419|31/485/421/0|32/458/364/0|
|GRU443|32/484/361/0|31/436/247/0|
|KDA419|32/462/278/0|31/434/262/0|
|KDA443|32/480/238/0|31/409/280/0|

粗候选是重复使用 episode 的边，不能冒充样本数。严格匹配为零，所以本报告不报告“控制相同机会后的 N20 性能残差”。这也不是总体状态支持不相交的证明：低人数只有 32 条，且所取阶段有生存偏差。即使匹配成功，完整 24 帧历史和未选人的潜在交互也未配平。缺口明确为更多已有低 N 逐步轨迹、10Circle 逐步轨迹、训练 replay/token 分布、行人目标/ORCA 内部状态及全面动作后果；本轮不为填满这些缺口追加实验或 HOLD。

初始帧的嵌套人口核验：八组配对中，原有人群前缀均 32/32 完全相同、机器人起点相同，但 current_token 均 0/32 相同；n_safe 均非零。5C→20C 平均减少 3.5625 个安全动作，10S→20S 减少 1.03125。KDA419/443 初始指令相同分别为 circle 7/32、4/32，square 6/32、2/32。可称“相同初始人群子集加人”，不可称相同世界或相同决策输入；后续人际响应会改变。个别病例加人后由失败变成功，亦不能把人口效应解释为逐病例单调难度。

固定 R4/W10 同时改变人口和密度。名义 N/面积：circle 0.0995/0.1989/0.3979，square 0.05/0.10/0.20 人/m²；它不是实际局部占用。20Circle 比 20Square 成功率高，虽名义密度更高，说明单一密度标量不充分；不证明密度无作用。“低于 1 且平坦的比值”也不等于统计或因果独立。

同 cell、同 case 的 checkpoint 配对可用；跨 N 同 case 不是同一世界。逐步批次 KDA443−419：20C +2.4pp（419独胜36、443独胜48），20S −5.4pp（83/56）；GRU 为 +4.4pp（23/45）、−3.6pp（68/50）。两个 checkpoint 不能估计训练 seed 总体方差，也不能支持稳定 KDA>GRU。四 checkpoint 的事后逐病例成功并集为 20C 498/500、20S 478/500，要求实际扫掠无重叠时并集不变；这是部分世界的可行轨迹见证，不是可部署集成、控制上界或全任务可解证明。

## 4. 同世界已有干预：导航损失可见，但机制不唯一

A1 固定规则：从 clearance≥0.2 且 progress≥0.02 的候选中取最高 raw_score（同分最小索引），只替换一个实际平滑后的动作，然后继续同 checkpoint；包括 A1=A0。不是事后从三个分支挑最佳。以下直接从 branches 的 root.type 与 branch.labels 聚合；救回均要求整段 actual_swept_clearance≥−1e−9。

|模型|20C救回/失败根；损伤/成功对照|20S救回/失败根；损伤/成功对照|
|---|---|---|
|GRU419|15/53；1/51|10/92；7/92|
|GRU443|5/31；0/31|19/110；7/110|
|KDA419|15/56；3/56|31/111；9/111|
|KDA443|15/44；6/44|28/138；9/138|

根选择以已知 baseline 失败为条件，不能将救回率当在线 SR 增益。成功对照见 audit.py:350–374，仅匹配同模型/格、时间≤2s、目标距离≤1m、n_safe≤16、crowd2m≤2；不是完整局部几何或历史匹配。它排除“全部失败都由固有不可避导致”，同时显示强制前进会损害成功。80 动作并未穷举闭环，不能估计最优控制天花板，也未隔离无 stop、平滑、CV、单步 horizon 各自贡献。

真实后继 oracle 的原始连接例：KDA419/20S/case90001/root_id1/step28，A0=67，A3=68；预测 q 为 0.603650 vs 0.602788；实际后继、实际历史及 terminal correction 后仍为 0.562245 vs 0.554041，但 A3 的 root_return 高 0.298590，timeout→46s success。只说明真实后继替换不能消除该世界的长期排序损失，不证明 OOD value；样本实现回报也不等于给定可观测状态的期望 value。20S 失败根的改善分支去重后，KDA419/443 各 192/216 对，其中实际后继 q 仍严格反排 107/156 对；同 root 多分支相关，不能当独立统计检验。

input-probe 是已有筛选根上将 TTC5 历史替成 nearest5 的动作敏感性，不是新闭环：KDA 524 根中动作改变 375，能匹配既有分支的改变动作仅 75；这 75 中 root_return 改善57、恶化17，rescue24、damage4。它支持选择历史影响决策，不支持 nearest5 全分布更优，更不建立缺失输入的不可恢复下界。

|解释|证据支持什么|未被识别的缺口及创新含义|
|---|---|---|
|固有任务更难|人口/几何改变，失败与超时增加|部分同世界救回否定“全都不可解”；不能分离人口、密度、交互变化。难度趋势不是方法贡献。|
|输入缺失/混叠|五人 token、约束人遗漏、替换历史会改动作|全人安全仍在；缺少相同可见历史但不同隐藏世界的决策必要性证据。加人数摘要本身已有工作。|
|OOD value 迁移|真实后继后仍有长期排序损失|缺训练支持分布、可比机会和条件期望校准，不能因果定位 OOD；不复活 value-gap。|
|覆盖/seed 不稳定|同格大量互相救回、几何依赖的 checkpoint 差异|两 seed 非总体方差；成功并集非在线能力。换 backbone 不构成新方法。|
|安全/动作/一步结构|有局部前进机会却未选，CV/实际扫掠不同，无显式 stop|没有完整控制器因子干预或规划最优界；通用多步搜索已成熟，仍缺超越先例的具体能力。|

## 5. 原论文与官方代码：实际读取层级

以下均为原文及官方源码只读核验；没有运行、训练或复现任何文献模型。源码链接固定到实际读取 commit，不能把论文结果视作本项目对照。

**Sigal2023 精确身份。** Adam Sigal、Hsiu-Chin Lin、AJung Moon，Improving Generalization in Reinforcement Learning Training Regimes for Social Robot Navigation，arXiv:2308.14947；2023 初稿与 2024 v2 标题有变化。已读 [v2 §3、§4.3/Table1、§4.5/Table2](https://arxiv.org/html/2308.14947v2)：课程从 circle 到 square、混合行人模型并随机化半径/速度，区分扩大场地与固定面积增密。官方 [train.py:150–164](https://github.com/RAISE-Lab/soc-nav-training/blob/8491d9c02eb7c802d7d0c3db0c4e350434998299/CrowdNav/crowd_nav/train.py#L150-L164) 已读课程阶段/恢复 episode 逻辑及 env.config 的 N5/R4/W10；不是只读摘要，也未确认一条自动命令完整复现全部课程实验。其分布扩展是成熟基线，不提供本模型 OOD 主因证据。

**Don't Freeze, Don't Crash。** Zhang、Xu、Aggarwal，[arXiv:2603.06729v1 §III-B/C、§IV-A](https://arxiv.org/html/2603.06729v1)：固定 K 邻居外加压力/局部占用等摘要；Φ=−w_g d_goal−η(n_local)(w_I C_I+w_P C_P)，r_PSS=γΦ(s′)−Φ(s)。训练 N11–16、测试至21，不是五人训练迁移。官方 [env_social_nav.py:960–1083](https://github.com/jefferyz001/PSS-Social/blob/3c4e75779cb931423ffee4c4479c673d14e9b6d0/env_social_nav.py#L960-L1083) 已读最近邻截断与12维人群摘要，[pss_social.py:251–281](https://github.com/jefferyz001/PSS-Social/blob/3c4e75779cb931423ffee4c4479c673d14e9b6d0/pss_social.py#L251-L281) 已读 potential/η、clip、done 处理。碰撞计数不立即终止，goal-reached 不等于 collision-free；其口径不能直接对比 CrowdNav 碰撞终止。固定维度不等于与人数独立，加入人数/密度特征不能单独申报新意。

**SARL。** Chen 等，[Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning，§III](https://arxiv.org/pdf/1809.08835)：学习 attention 的 softmax 权重汇聚人群特征，reward+discounted next value；可含 occupancy map。官方 [sarl.py:35–64](https://github.com/vita-epfl/CrowdNav/blob/20d678085c06831e658a65b9e20c8bb6f6ecdc10/crowd_nav/policy/sarl.py#L35-L64) 已读 global mean/weighted pooling；[multi_human_rl.py:35–52](https://github.com/vita-epfl/CrowdNav/blob/20d678085c06831e658a65b9e20c8bb6f6ecdc10/crowd_nav/policy/multi_human_rl.py#L35-L52) 已读 query_env 分支。官方默认 query_env=true 可调用环境一步预测，本地 CV+TTC5 不是它的完整复现。可变人数汇聚能力不保证分布迁移成功。

**RGL。** Chen 等，[Relational Graph Learning for Crowd Navigation，§III/IV](https://arxiv.org/pdf/1909.13165)：关系图 A=softmax(XW_aXᵀ)、状态预测及递归规划；实验 circle5/R4。官方 [model_predictive_rl.py:271–299](https://github.com/ChanganVR/RelationalGraphLearning/blob/8e87aa5ed8221efd688f8e6857ba4c38637bf6e1/crowd_nav/policy/model_predictive_rl.py#L271-L299) 已读 value、motion、reward 的深度递归，[mp_separate_dp.py:23–24](https://github.com/ChanganVR/RelationalGraphLearning/blob/8e87aa5ed8221efd688f8e6857ba4c38637bf6e1/crowd_nav/configs/mp_separate_dp.py#L23-L24) 为 depth2/width2。通用多步 lookahead 不是新能力，且未提供当前 5→20 的因果诊断。

**RADR 精确身份。** Sun、Zhang、Wei、Liu，Risk-Aware Deep Reinforcement Learning for Robot Crowd Navigation，Electronics 2023,12(23):4744，DOI10.3390/electronics12234744；不是其他领域同缩写。[原始出版 PDF](https://mdpi-res.com/d_attachment/electronics/electronics-12-04744/article_deploy/electronics-12-04744.pdf) 已读 §3.3.1 Eq9–10：q_i=max(0,(v_R cosα+v_H cosβ)/d²)，跨人 softmax；Eq20–22 学习融合风险与 attention 特征再 GRU；§4 训练20、测试10/20/25，与本案训练分布不同。官方 [selfAttn_srnn_temp_node.py:282–326](https://github.com/QZhSSLab/RADR/blob/bba3f72224b273337329fa1066744c4b96ca360a/rl/networks/selfAttn_srnn_temp_node.py#L282-L326)、[376–398](https://github.com/QZhSSLab/RADR/blob/bba3f72224b273337329fa1066744c4b96ca360a/rl/networks/selfAttn_srnn_temp_node.py#L376-L398)、[579–601](https://github.com/QZhSSLab/RADR/blob/bba3f72224b273337329fa1066744c4b96ca360a/rl/networks/selfAttn_srnn_temp_node.py#L579-L601) 已核验风险计算、融合、实际 forward 调用，不止类定义。config.py:41–51 默认 human_num10、predict_method='inferred'，不同于论文训练20及线性预测描述；精确实验配置复现 UNVERIFIED。仅重命名风险注意力/人数融合没有新颖性依据。

历史查重仅作局部设计先例：仓库 git 根 /home/abc/workspace/nav_data/mamba/camrl；读取 c85ab0e 的 `CrowdNav/CrowdNav/results/fcs3_aggregation_20260924/FINAL_EVIDENCE_20260925.md`、`CrowdNav/results/composition_swap_20260919/README.md`。历史数值未经 B 原始逐案复核，UNVERIFIED，不迁移其负结论。实际 `git show b514e00 -- CrowdNav/crowd_nav/bayesian.py` 中 regret_cost 已实现 floor_i=min_a cost[a,i]，max_i(cost−floor)+max_i floor 与 sum_i cost 的 mixture；故 count-aware max/sum/local-regret 重命名与本地已有设计重合。AC3 的操作门槛失败由主 Agent 原始核验负责；本报告不把 gate fail 写成所有效力或机制不可能。

## 6. 资源决策与冻结条件

合格新候选 **0**。不是因“已有相关论文”自动否决，而是尚无一个新增计算差异同时满足：定位可修复的导航损失、超越上述论文与实际代码及本地已试组合、成本合理且有预注册成功/淘汰门槛。混合 N 训练、更大 K、仅密度输入属于最强简单基线；已有风险融合/通用规划/局部 regret 不包装成两个待续方向。

**METHOD_SPEC_FREEZE=NO**：缺失的是损失定位与具体方法差异，不是一条更完整的泛化曲线。只有明确新增表示/目标/决策计算、与最近源码逐项区别、强简单基线、预先规定 success/collision/timeout/效率/耗时及跨 checkpoint 损伤门槛的闭环检验，才能重新判定冻结。本轮未提出满足条件的假设，所以没有虚构 GO/KILL 数值或新增实验；追加训练/GPU/闭环预算为 0。主决策为 KILL 当前 B 立项，而非 HOLD、整个领域 KILL 或“已证明 OOD 不存在”。

## 7. 内嵌复算入口（清理 scratch 后仍可复核）

以下是实际聚合口径的独立精简入口；只需 Python/numpy/scipy，直接读取原始文件，不依赖本报告派生 JSON。将代码块作为 Python stdin 执行即可；严格匹配函数完整保留阈值，未在零匹配后放宽。输出旧六格、逐步指标、匹配数、A1救回/损伤和 oracle 反排计数；每行原始 episode 等权。

```python
import gzip,json,math,itertools
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
B=Path('/home/abc/workspace/ttc5-causal-20261011/work')
def rows(p):
    with gzip.open(p,'rt') as f: yield from map(json.loads,f)
D={}; O={}
for p in sorted(B.glob('baseline-*.jsonl.gz')):
    E=list(rows(p)); k=(E[0]['model'],E[0]['cell']); D[k]=E; O[k]=[]; V=[]
    for e in E:
        S=e['steps']; speed=[math.hypot(*s['command']) for s in S]
        miss=[any(c>=.2 and g>=.02 for c,g in zip(s['clearance'],s['progress'])) and s['progress'][s['chosen']]<.02 for s in S]
        V.append([np.mean([s['n_safe']==0 for s in S]),np.mean(miss),np.mean(np.array(speed)<.1),sum(speed)*.25])
        i=next((i for i,s in enumerate(S) if i>=8 and s['goal_distance']<=4),None)
        if i is None: continue
        s=S[i]; r=np.array(s['robot']); h=np.array(s['humans']); a=h[s['ids']]
        u=(r[5:7]-r[:2])/s['goal_distance']; R=np.array([u,[-u[1],u[0]]]).T
        O[k].append(([i*.25,s['goal_distance'],s['n_safe'],s['crowd2m'],min(np.linalg.norm(h[:,:2]-r[:2],axis=1)-h[:,4]-r[4])],r[2:4]@R,(a[:,:2]-r[:2])@R,(a[:,2:4]-r[2:4])@R))
    print('STEP',k,Counter(e['outcome'] for e in E),np.mean(V,axis=0),np.mean([e['time'] for e in E if e['outcome']=='success']))
for m in sorted({k[0] for k in D}):
    for lo,hi in [(0,4),(3,5)]:
        edges=[]; coarse=0
        for i,a in enumerate(O[m,lo]):
            for j,b in enumerate(O[m,hi]):
                z=np.abs(np.array(a[0])-b[0]); rv=np.linalg.norm(a[1]-b[1])
                if np.any(z>np.array([5,.25,8,1,.2])) or rv>.2: continue
                coarse+=1; P=((a[2][:,None]-b[2][None])**2).sum(-1); Q=((a[3][:,None]-b[3][None])**2).sum(-1)
                x,y=linear_sum_assignment(P/.5**2+Q/.3**2); pr=np.sqrt(P[x,y].mean()); vr=np.sqrt(Q[x,y].mean())
                if pr<=.5 and vr<=.3: edges.append((sum(z/np.array([5,.25,8,1,.2]))+rv/.2+pr/.5+vr/.3,i,j))
        useda=set(); usedb=set(); matched=0
        for _,i,j in sorted(edges):
            if i not in useda and j not in usedb: useda.add(i); usedb.add(j); matched+=1
        print('MATCH',m,lo,hi,len(O[m,lo]),len(O[m,hi]),coarse,matched)
for seed,root in [(419,'/home/abc/4090_backup_20261009/root/kda-history-20261009/evaluation'),(443,'/home/abc/workspace/kda-history-replication-20261010/evaluation')]:
    for m in ['kda','gru']:
        for c in range(6):
            e=json.loads((Path(root)/f'{m}-t24/cell{c}.json').read_text())['episodes']; print('OLD',seed,m,c,Counter(x['outcome'] for x in e))
oracle={r['root_id']:{b['candidate_index']:b for b in r['branches']} for r in rows(B/'real-history-oracle.jsonl.gz')}
for p in sorted(B.glob('branches-*-cell[45].jsonl.gz')):
    C=Counter(); Q=Counter()
    for r in rows(p):
        if r.get('record_type')!='counterfactual': continue
        t=r['root']; control=t['type']=='success-control'; a0=next(b for b in r['branches'] if 'A0' in b['labels'])
        for b in r['branches']:
            if 'A1' in b['labels']:
                C['control' if control else 'fail']+=1
                if control: C['damage']+=b['outcome']!='success'
                else: C['rescue']+=b['outcome']=='success' and all(s['actual_swept_clearance']>=-1e-9 for s in b['steps'])
            if t['cell']!=5 or control or b['candidate_index']==a0['candidate_index'] or b['root_return']-a0['root_return']<=1e-6: continue
            q=oracle[t['root_id']]; key='q_oracle1_actual_history_terminal_corrected'
            if b['first_step']['q_pred']<=a0['first_step']['q_pred']:
                Q['pairs']+=1; Q['reverse']+=q[b['candidate_index']][key]<q[a0['candidate_index']][key]
    print('BRANCH',p.name,C,Q)
```

其余关键统计可直接复算：同 cell 以 case 内连接，成功指示做配对差；四模型逐 case 对 success 且所有 actual_swept_clearance≥−1e−9 取 OR。嵌套比较每文件前32条，逐 case 比较 steps[0].humans 前缀、current_token、command、n_safe。所有 hash 均为压缩原文件 bytes 的 SHA256，而非解压文本或派生结果。

## 8. 主证据 SHA256

以下 baseline/branches/roots 均相对于第1节 TTC 原始根目录；旧数据标签 old-sSEED-MODEL-cellC 对应第1节旧根目录 /MODEL-t24/cellC.json。保留所有支持人口表与主要干预结论的原始指纹。

```text
baseline-gru-s419-t24-cell0.jsonl.gz b7174e64bf23a648da2cf398865a302e412d37ea7b21f65f74afa0677dcde91c
baseline-gru-s419-t24-cell3.jsonl.gz 47a41175910669a5f9ef03b1cf6a44779df2e41dd3489503284ccf23280ed1b4
baseline-gru-s419-t24-cell4.jsonl.gz fcdba944523f7463cc2b934f75b458e52ddf095be659c1a27f1c9a8db61bdb32
baseline-gru-s419-t24-cell5.jsonl.gz 752dbaaeb94ed284a0312c667958d4a057daf628671befc1b6b35525ee5cffe3
baseline-gru-s443-t24-cell0.jsonl.gz c59d4684852ea62f82b68daee9f6c787070d21afa0ed2b128a9c015c6f671f20
baseline-gru-s443-t24-cell3.jsonl.gz 77477eaaa48cbe0891ab31ed2b483754e1349fb171cb2f56ade97de13c4274e8
baseline-gru-s443-t24-cell4.jsonl.gz 79e358f3ca6138dec22930f9f409ccaee43ba2812479bea249ae4a9cc0911fbd
baseline-gru-s443-t24-cell5.jsonl.gz b0a949b58ca457031962e35a122092817f09646eac998a6b10677e4dfc0e5d98
baseline-kda-s419-t24-cell0.jsonl.gz 81bf46ad58206c2cc8d069b1bd7b7d9a42e2f5c1658b819b0c033d096ecf2caf
baseline-kda-s419-t24-cell3.jsonl.gz fb8df6149de864b884237fd931583f17ae7c7f7e75a423da7a96a133eb8f8fab
baseline-kda-s419-t24-cell4.jsonl.gz 5f7c905b437abdcc36104c90509bf835ba7c629207efdfd7304eb138c8e03082
baseline-kda-s419-t24-cell5.jsonl.gz 9851ac02d19a4f4a1f035cfd6db7737bbc6d9154de96eaf7b9b28042d5fad12a
baseline-kda-s443-t24-cell0.jsonl.gz dbaad4d4b9d41c2096fc7d7ad3b4c31ca63493bfebc0be29cbe1b36727d8ed38
baseline-kda-s443-t24-cell3.jsonl.gz 3beaa8c3b9723766a61e228f2cea156df5457e4d12d99296f1d72e3a6674ba98
baseline-kda-s443-t24-cell4.jsonl.gz abcaa4c9e99b4b52af6ac25af7af8f342206dd71ef773fd348a2b0d9773e63a2
baseline-kda-s443-t24-cell5.jsonl.gz f27433567d49266cbbc643918f59b1ae415984b96b03510474a8727cbd7ba009
branches-gru-s419-t24-cell4.jsonl.gz 1c7a35faaae0ccb45020bdf1d4d77cc7f3d1715350826e7724ed9de4ff11785c
branches-gru-s419-t24-cell5.jsonl.gz f078771b4673eda754dd1242bd785e27a31e751b5cdcb0f94dcdabde0bbc74cd
branches-gru-s443-t24-cell4.jsonl.gz 4488da816a85bc5e12861e226f7c2be603c0b7d6cb6fad2fb66ac08f12fcbdfe
branches-gru-s443-t24-cell5.jsonl.gz b2b14deed0bae40a7464d7e63fd23a7545a119efbcd649d59d5fd88975a2026e
branches-kda-s419-t24-cell4.jsonl.gz 361a76aa03f7a22432205413fbd0ec9b5a0df34f9740a79f529a2787db7617cc
branches-kda-s419-t24-cell5.jsonl.gz d3a7dc6d0fc9e2fa7f8aa0bbd050f3f4a217ccfe322fd9a6a092695ad2624415
branches-kda-s443-t24-cell4.jsonl.gz 240ae779460d6a440e6cb71d0b5b712e68529cc94a560571c823b0709a663f10
branches-kda-s443-t24-cell5.jsonl.gz 390295f5e6d6486b92cd1a37db6707bf19c5be92dcbdc1f310e904c104b8d233
input-probe.jsonl.gz bebed6ebe9b4bffd5d878f7cd95244bc38f0b758ebe0ac83b03f781fd868005c
real-history-oracle.jsonl.gz c1b348d2336a99f923d2735a60328c7c3c83921642070fb074a34982e4f13029
roots.json 665da8a3e170b40f8106e0f634c738b576e2dc7d887c508fab3a922f13adacb2
old-s419-kda-cell0 5b3b130df8a18b84c4fcac8496a1346540444e6bb1b4d6a62aa4c2ad6e93fe3c
old-s419-kda-cell1 daa61b9855118720628e42e8d974dcbf35dc8ffd8949e3b41fa203a21a33ddd2
old-s419-kda-cell2 aeb6175d4215c2e72303b7c5baad7a20ff6406a36769ff9c5726084cad358ec1
old-s419-kda-cell3 085cce569b45c7df2cdae09a6e2b3eac3c4722c676b012f52ef4fc28cd9e9402
old-s419-kda-cell4 487f4d1dbef1df32bd99e8e93ec4091217f5cec07a68dac036919f9847e433bf
old-s419-kda-cell5 4e57cec56c1ad6f59c0b3106f5ca72af8daef8d4e24d2a8cfe5505108f87a632
old-s419-gru-cell0 ec0f78f135e031a3ac661b6cb5c09bf7a45fa8d2aed7ad5a5bdc3b1864f9790f
old-s419-gru-cell1 f891b203f27a0d92bde350b306de796e6432b41b3da80fe23574a3704a87beb5
old-s419-gru-cell2 c52542430ea575b134f4741168411cd1ebd27b8a05a4c725293c5ad83635e2f9
old-s419-gru-cell3 ed5b6d5ef840d33e3a426b6647e7dcfca542108e7f4a7a5694dede827855c3ad
old-s419-gru-cell4 620210d217fb44ed000120872f307b4b5deeb33b34fe40f7281a90c049b149d4
old-s419-gru-cell5 3690db5f421cdd411385aba35be2767570c1cfc24bcd522e06c4633847741097
old-s443-kda-cell0 b05ab9afd2eee08ad85d86c0885be06c0568c90689272974ee38c7023622afe3
old-s443-kda-cell1 ed431fe4292b17806bbc37ac8de2675ff86307ed963a879f3fc032d379864911
old-s443-kda-cell2 0b23f61ffa2c1dc981f8f877b1eefb721e502eaf69683c1a94f8fa8dec7e0f19
old-s443-kda-cell3 7ae323e309b6be3163789079cea71935ec13e05aff362d8a6e502cf62a8e0083
old-s443-kda-cell4 53d50fba3440860ac7629e5958f7fa550ea147054a39183dc6f6546dacb3d10a
old-s443-kda-cell5 10442adec4dd98fc824a977242d51acd056509b7482b520ce879b96f55e8f13b
old-s443-gru-cell0 fb6118c5e3d86952eb755acd18ccc3c35c69a53841ca7a0c367132f4a3e79fa0
old-s443-gru-cell1 c7dcbb1bb3987741412d9c1877b11262529691758b6fc2de36462416cc812b49
old-s443-gru-cell2 976365f9c3981516ff538dd14d89c8e0e1a1f99d369dad661e16d264c05f9dd0
old-s443-gru-cell3 e00681205209f976da3829c6b9506f9a8a1f757837f0ca9d8c0ed89e56ba5494
old-s443-gru-cell4 c596e1e2ede0a20951991604c0768f12d8812e33e47b57ff990e0434bfee7d91
old-s443-gru-cell5 89016dabcb9b41d1e33777c6e8112e9fb7febcc839aefee7365aa91500b28f7b
selector/evaluate-selection.py fa6f0f2e5d5e3b62f874905a8bbe7a2628b29e883da72f339ae3400d5eedd3e8
shared/mamba_rl.py 2fd464618507796fdbc1b072d96563c97180e301f842cac1a6e6f75212ff6616
shared/crowd_sim.py 115aac6e660d0c7fff4425e448f5b9c97683159fe2cb8f9d43ecbfb07a3554dc
RADR-publisher-PDF ee469a065cf7865dbae4fe1fd6a28987c41c30e4c9932c4913d3a94732ee19b8
```

## 9. 临时清理清单

唯一最终 B 交付为本文件。可整体删除的 B scratch：
/home/abc/workspace/dual-direction-audit-20261011/direction-b/

其中包含 analyze-b.py、supplement-b.py、attribution-b.py，statistics.json、supplement.json，SARL/RGL/PSS-Social/RADR/soc-nav-training 五个只读研究 clone，以及 radr-paper.pdf/txt；均非最终证据唯一来源。上述内嵌入口、原始路径、源码 commits、主文件 SHA 已保留，清理不依赖保留这些副本。不得删除原始实验目录或其他 Agent 文件。
