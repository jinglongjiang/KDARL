### 研究问题冻结（Research Brief）
- **RQ1（扰动维度）**：测试协议如何量化泛化性（规模密度、场景拓扑、模型替换、感知噪声）？
- **RQ2（统计规范）**：种子数（seeds）、回合数（episodes）、置信区间（CI）及显著性检验的主流实践为何？
- **RQ3（“够用”基准）**：机器人顶会中兼顾审稿认可度与算力预算的最小充分测试配置是什么？

---

### 最常见的“够用”基准配置归纳（Good-Enough Benchmark）
1. **场景拓扑**：Circle Crossing（$R=4$m）+ Square/Corridor 漫游（$10\times 10$m 至 $14\times 14$m）。
2. **人数与密度**：标称训练 5–10 人（$\rho \approx 0.1$人/m²）；泛化测试外推至 10–20 人（$\rho \approx 0.2$人/m²）。
3. **行人模型**：训练用 ORCA（Invisible 0% 避让）；测试替换为 Social Force (SFM) 或注入 20% 冲刺行人（2.0 m/s）与紧密群组。
4. **感知扰动**：有限圆形视野（$R_{fov}=3\sim 5$m），可选注入位置高斯噪声（$\sigma = 0.05\sim 0.1$m）。
5. **统计实践**：固定 3–5 个测试 seed，每场景 100–500 episodes（总量 $\ge 500$），报告均值 $\pm$ 标准差，涵盖安全、效率与舒适指标。

---

### 1. 逐篇文献精读对照表

| 论文 | 状态表示 | 动作空间 | 网络 / 算法 | 训练方式与环境 (行人模型、人数) | 评测指标 | 核心思想 (一句话) | 原文依据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Diverse-4**<br>Sigal et al. (2023)<br>arXiv:2308.14947<br>NeurIPS GenPlan | 自身与行人相对几何状态（遵循 SARL 全知位姿速度） | 离散 2D 速度（5 档速度 $\times$ 16 偏角 + 停止） | **CD-SARL**（自注意力池化 + 价值迭代） | **固定 10,000 episodes**。训练：Simple Circle（5人, $r=4$m）与 Simple Square（10人, $w=10$m）；行人：ORCA + SFM 混合 + 动静混合 | SR, CR, TR, Time (均值$\pm\sigma$), Discomfort ($<0.2$m 占比), $d_T$ (均值$\pm\sigma$)。测试 200 episodes（Large Circle 12人/Large Square 20人/Dense Circle 10人/Dense Square 20人各 50 轮） | 拓扑场景与动力学模型双重渐进多样化训练，大幅提升未见大场景与密集人群泛化性 | Sec. 4.1–4.5, Table 1, Table 2 |
| **GenSafeNav**<br>Yao et al. (2025)<br>arXiv:2508.05634<br>CoRL 2025 | 自身状态 + 行人位姿速度 + GST 预测 5 步轨迹 + DtACI 自适应不确定性球半径 $r_t$ | 连续全向速度 $(v_x, v_y) \in [-1, 1]^2$，限速 1.0 m/s | **DtACI + GST + CRL (PPO-Lagrangian)**（解耦 HH/RH 图网络） | 标称训练：$12\times 12$m，20 名 ORCA 行人，无噪观测；Invisible setting（100% 不避让） | SR, CR, TR, NT, PL, ITR, SD。评测：5 测试 seed $\times$ 250 episodes = 1250 episodes，报 3 训练 seed 均值$\pm\sigma$。3 类 OOD：20% 冲刺（2.0 m/s）、全员替换为 SF、群组聚集 | 自适应共形推断在线量化轨迹不确定性球，在约束 RL 中惩罚侵入，无需域随机化对冲 OOD 扰动 | Sec. 3.4–3.6, Sec. 4.1–4.4, Table 1, Table 2 |
| **CrowdNav++**<br>Liu et al. (2023)<br>arXiv:2203.01821<br>ICRA 2023 | 自身运动状态 + 行人相对位姿速度 + GST 预测 5 步轨迹（5m 有限半径） | 连续全向速度 $[v_x, v_y]$，最大速率 1.0 m/s | **CrowdNav++**（双重时空图注意力 HH/RH 解耦 + GRU + PPO） | $12\times 12$m 连续人流，最多 20 人。Invisible setting（行人 100% 不避让机器人）。训练 $2\times 10^7$ steps | SR, NT, PL, ITR, SD。评测：500 个未见随机测试回合，仅报均值（无方差/CI）。对比：(1) 随机化动态环境（换目标、速度 $0.5\sim 1.5$m/s、半径 $0.3\sim 0.5$m）；(2) 标称环境 | 解耦人-人与人-机图注意力并引入意图预测，在不避让的 20 人密集人流中实现主动让行 | Sec. III, Sec. IV-A, IV-B, Table I, Table II |
| **SRFM Benchmark**<br>Agrawal et al. (2024)<br>arXiv:2409.14844<br>IROS 2024 | 目标相对距离角度 + 最近 10 名行人相对位姿（不足零填充）+ 上一动作 | 连续差分动作 $(v, w)$，线速 $[-0.5, 0.5]$ m/s，角速 $[-\pi, \pi]$ rad/s | **TD3**（奖励函数引入基于 SRFM 预测的行人偏离惩罚项 $r_{div}$） | $15\times 15$m 自由空间，10 名真实数据拟合参数的 SRFM 行人，动态重采样目标。训练 $2\times 10^6$ steps | Fréchet 偏离距离、最小人机距离、轨迹长度、导航时间。评测：5 大拓扑场景（Footpath, Crosswalk, Crossfootpath, Box, Concert），每场景 100 episodes，双样本 t 检验（$p=0.05$）报均值$\pm\sigma$ | 提出双向考虑人机斥力的 SRFM 模型，并在 5 大拓扑中利用 Fréchet 距离量化机器人引发的行人扰动 | Sec. 3.3, Sec. 3.4, Sec. 4.3, Table 3, Fig. 3 |
| **SocNavBench**<br>Biswas et al. (2022)<br>arXiv:2103.00047<br>ACM THRI 2022 | 多模态输入（全向/单车运动学），提取行人历史当前位姿速度或传感器数据 | 线速度角速度 $(v, \omega)$，最大线速 1.2 m/s | 评测基准框架（横向评估 SA-CADRL, ORCA, Social Forces, DWA 等） | 纯评测基准；利用真实行人轨迹数据集（ETH、UCY 等）回放构建 33 个精选高交互测试场景（Curated Episodes） | 成功率、碰撞不终止总碰撞数、规划耗时、路径长度比、路径不规则度、到达比、能耗、Jerk、加速度。报 33 场景均值$\pm\sigma$ | 提出基于真实轨迹回放且“碰撞不终止”的客观测试基准，消除合成动力学假设中行人过度避让的虚假偏差 | Sec. 3.2–3.4, Sec. 4.2, Table 2, Table 3, Table 4, Fig. 6 |

---

### 2. 共识（Consensus，≤5 条）

1. **人数与密度非对称外推（Scale & Density OOD）是检验泛化性的通用门槛**：普遍在小规模场景（5–10人，$\rho \approx 0.1$人/m²）训练，并在翻倍人数或密度（10–20人，$\rho \approx 0.2$人/m²）下检验；均证实策略在未见高密度下碰撞率显著升高、成功率单调衰减。  
   *出处：[Diverse-4 Sec. 4.3–4.4](https://arxiv.org/abs/2308.14947)；[Flögel 2024 Sec. V-A](https://arxiv.org/abs/2409.10655)*
2. **行人动力学模型替换（Cross-Model Stress Testing）成为防过拟合的必答题**：仅在同构 ORCA 下评测无法反映真实鲁棒性；主流工作公认须将测试行人替换为 Social Force (SFM)、SRFM 或真实轨迹回放（ETH/UCY）以验证抗模型失配能力。  
   *出处：[Diverse-4 Sec. 4.5](https://arxiv.org/abs/2308.14947)；[GenSafeNav Sec. 4.4](https://arxiv.org/abs/2508.05634)；[SocNavBench Sec. 4.2](https://arxiv.org/abs/2103.00047)*
3. **“不可见设置”（Invisible Robot / Non-yielding Humans）成为防流氓策略的标准环境协议**：为防止 RL 策略在测试与训练中利用特权强行逼停行人（Freezing/Bullying），主流仿真协议均将行人设为对机器人不可见（行人 100% 不主动避让机器人，仅人与人间避让）。  
   *出处：[CrowdNav++ Sec. IV-A](https://arxiv.org/abs/2203.01821)；[GenSafeNav Sec. 4.1](https://arxiv.org/abs/2508.05634)*
4. **场景拓扑向结构化多态人流演进**：单一同心圆交叉（Circle Crossing）已不再被视为充分测试；测试协议普遍扩展到正交人流（Crosswalk）、双向平行人流（Footpath）、散布漫游（Square/Random）及极端聚集场景。  
   *出处：[Diverse-4 Table 1](https://arxiv.org/abs/2308.14947)；[SRFM Benchmark Sec. 3.3](https://arxiv.org/abs/2409.14844)；[Flögel 2024 Sec. V-A](https://arxiv.org/abs/2409.10655)*
5. **四维综合评价矩阵取代单一成功率**：统一抛弃单一成功率，形成四维测试矩阵：安全（SR/CR）、效率（Time/Timeout/PL）、舒适（Discomfort Rate）、社交扰动（ITR/Fréchet 偏离）。  
   *出处：[GenSafeNav Sec. 4.1](https://arxiv.org/abs/2508.05634)；[Diverse-4 Sec. 4.2](https://arxiv.org/abs/2308.14947)；[CrowdNav++ Sec. IV-B2](https://arxiv.org/abs/2203.01821)；[SocNavBench Sec. 3.4](https://arxiv.org/abs/2103.00047)*

---

### 3. 分歧与已知弱点（Disagreements & Open Weaknesses，≤5 条）

1. **统计严谨度严重参差：95% 置信区间几乎全员缺席，种子数与 episode 数未统一**：大部分顶会工作（CrowdNav++, HeR-DRL, Mun 2023）仅汇报 500 episodes 单次均值，不报方差与种子；少数（GenSafeNav, Flögel 2024）汇报 3–15 seeds 的均值$\pm\sigma$；唯有 Agrawal 2024 进行了双样本 t 检验（$p=0.05$）；领域内普遍未建立 95% 置信区间报告范式。  
   *出处：[CrowdNav++ Table I](https://arxiv.org/abs/2203.01821)；[GenSafeNav Sec. 4.3](https://arxiv.org/abs/2508.05634)；[SRFM Benchmark Table 3](https://arxiv.org/abs/2409.14844)*
2. **“碰撞是否终止回合”（Early Termination vs Non-termination）的评测协议矛盾**：传统 CrowdNav 体系统一采用“碰撞即刻终止”，导致碰撞率高的策略因早死反而呈现低耗时与短路径的虚假高效；SocNavBench 虽采用“碰撞不终止、累加碰撞”，但导致策略碰撞后陷入非稳态分布，两者至今割裂。  
   *出处：[SocNavBench Sec. 4.2](https://arxiv.org/abs/2103.00047)；[Diverse-4 Sec. 4.5](https://arxiv.org/abs/2308.14947)*
3. **真实轨迹回放（Data Replay）的“幽灵行人/非反应性”两难**：SocNavBench 回放真实轨迹消除了合成模型假设偏差，但被回放行人对机器人动作完全无反应（如幽灵穿行），缺乏反事实闭环交互，难以评估真实的双向避让博弈。  
   *出处：[SocNavBench Sec. 5.3](https://arxiv.org/abs/2103.00047)；[SRFM Benchmark Sec. 1](https://arxiv.org/abs/2409.14844)*
4. **感知扰动评测严重脱节与异质化**：各工作对感知鲁棒性的定义截然割裂：Flögel 2024 采用观测高斯噪声与动作抖动，Mun 2023 采用有限视野与几何遮挡建图，而 Diverse-4 完全基于全知无噪状态，各方法在感知退化下的鲁棒性无法横向对齐。  
   *出处：[Flögel 2024 Sec. V-A](https://arxiv.org/abs/2409.10655)；[Mun 2023 Sec. V-A](https://arxiv.org/abs/2210.00552)；[Diverse-4 Sec. 4.1](https://arxiv.org/abs/2308.14947)*
5. **动力学模型替换下的泛化表现结论相互矛盾**：在 ORCA 训练的策略迁移到 SFM 测试时，Diverse-4 报道 baseline 成功率暴跌（ORCA 从近 100% 降至 36%），而 GenSafeNav 却报道所有策略在 SFM 下优于 ORCA 标称环境（因 SFM 提供了更大的柔性避让空间），暴露出底层仿真器超参对跨模型结论的剧烈扰动。  
   *出处：[Diverse-4 Table 2 & Sec. 4.5](https://arxiv.org/abs/2308.14947)；[GenSafeNav Sec. 4.4](https://arxiv.org/abs/2508.05634)*

---

### 4. 可直接借用的组件（Component Reuse，已全部实测验证）

1. **Diverse-4 跨拓扑跨密度测试基准**
   - **内容**：提供 Simple/Large/Dense $\times$ Circle/Square 六类标准 Gym 环境与 ORCA/SFM 混合采样训练/评测脚本。
   - **链接**：[github.com/RAISE-Lab/soc-nav-training](https://github.com/RAISE-Lab/soc-nav-training)（已核实：HTTP 200 正常访问）
2. **GenSafeNav OOD 分布偏移测试基准**
   - **内容**：集成 20 人密集 CrowdNav 环境、冲刺行人（2.0 m/s）、SFM 替换、紧密群组聚集三类 OOD 测试套件与 DtACI 共形不确定性评测模块。
   - **链接**：[github.com/tasl-lab/GenSafeNav](https://github.com/tasl-lab/GenSafeNav)（已核实：HTTP 200 正常访问）
3. **CrowdNav++ 密集人流与社交指标计算库**
   - **内容**：支持 20 人连续人流与 Invisible setting 仿真环境、GST 意图轨迹预测网络及标准 ITR / SD 社交侵占指标量化脚本。
   - **链接**：[github.com/Shuijing725/CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)（已核实：HTTP 200 正常访问）
4. **SocNavBench 真实轨迹回放评测套件**
   - **内容**：集成 ETH/UCY/Zara 等真实数据集精选的 33 个交互场景，提供“碰撞不终止”元统计、路径质量与平滑度（Jerk/加速度）完整评测管线。
   - **链接**：[github.com/CMU-TBD/SocNavBench](https://github.com/CMU-TBD/SocNavBench)（已核实：HTTP 200 正常访问）
5. **SRFM Benchmark 交互扰动评测框架**
   - **内容**：提供 Footpath/Crosswalk/Crossfootpath/Box/Concert 5 大典型人流拓扑环境、双向 SRFM 动力学仿真器及基于 Fréchet 距离的行人偏离度指标计算工具。
   - **链接**：[github.com/HumanoidsBonn/SRFM-Pedestrian-Deviation-Benchmark](https://github.com/HumanoidsBonn/SRFM-Pedestrian-Deviation-Benchmark)（已核实：HTTP 200 正常访问）
