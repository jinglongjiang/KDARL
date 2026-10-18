### 研究问题冻结（Research Brief）
- **RQ1（混合配比与多样性建模）**：训练集中的非避让/异质行人如何参数化与采样？比例是固定常数、离散随机还是连续谱系？
- **RQ2（等预算控制变量）**：是否在严格固定训练步数/样本预算下，仅改变环境行为分布进行消融对照？
- **RQ3（OOD 分布外评测）**：测试集是否在未见的人群密度、尺度及未知的行人行为动力学分布上检验泛化边界？

---

### 1. 逐篇文献精读对照表

| 论文 | 状态表示 | 动作空间 | 网络 / 算法 | 训练方式与环境（行人模型、人数） | 评测指标 | 核心思想（一句话） | 原文依据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GA3C-CADRL**<br>[Everett et al., 2018](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/everett_1805.01956.txt)<br>arXiv:1805.01956 (IROS) | 自身 $s=[d_g, v_{pref}, \psi, r]$；邻近行人 $\tilde{s}^o=[\tilde{p}_x, \tilde{p}_y, \tilde{v}_x, \tilde{v}_y, \tilde{r}, \tilde{d}_a, \tilde{r}+r]$，按距离由远及近输入 | 离散 11 动作（3 档速度 $\times$ 偏航角增量） | **GA3C**（LSTM 变长邻居编码 + 2层全连接；CADRL 先验预训练初始化） | **两阶段课程**（Phase 1: 2–4人；Phase 2: 2–10人）。环境随机混合：非合作直线、静止、当前学到的策略；仅回传自身策略经验 | 额外用时 $\bar{t}_{eg}$、失败率（碰撞/卡死比例） | 引入 LSTM 消除固定邻居上限假设，并在混合非合作策略的异步多智能体环境中端到端学习策略 | Sec. III-B, III-C, IV-B, Table I |
| **Diverse-4**<br>[Sigal et al., 2023](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/diverse4_2308.14947.txt)<br>arXiv:2308.14947 (NeurIPS GenPlan) | 自身几何状态与各行人相对几何状态（继承 CADRL / LSTM-RL / SARL 规范） | 2D 速度空间（离散速度档位 $\times$ 朝向） | **Deep V-Learning**（横向评估 CADRL、LSTM-RL、SARL；最优为 CD-SARL） | **等预算消融（固定 10,000 episodes）**：Baseline (Circle+ORCA)、Diverse (Circle+ORCA/SF混合)、Curriculum (Circle$\to$Square动静混合)、Curr+Div (Circle ORCA$\to$Square ORCA/SF+动静)，5–10人 | 成功率、碰撞率、超时率、航行时间、不适率（距离<0.2m时间占比）、最近距离 | 证明拓扑场景与动力学模型（ORCA+SFM）双重渐进多样化训练，能大幅提升在分布外密集人群中的泛化性 | Sec. 3, 4.1, 4.3, 4.5, Table 1, Table 2 |
| **CrowdNav++**<br>[Liu et al., 2023](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/crowdnavpp_2203.01821.txt)<br>arXiv:2203.01821 (ICRA) | 自身运动状态 + 行人当前位置 + GST 预训练模型输出的未来 $K=5$ 步意图轨迹 | 连续全向速度 $[v_x, v_y]$，最大 1.0 m/s | **CrowdNav++**（双重注意力时空图神经网络，解耦 HH 与 RH 交互 + GRU）+ **PPO** | 连续空间 $12\times 12$m，最多 20 人。**Invisible Setting（100% 不避让机器人）**，仅人与人间互避；随机化突变目标、半径与最大速度 | 成功率、导航时间、路径长度、侵入时间占比（ITR）、侵入时社交距离（SD） | 通过解耦人人/人机注意力图并显式预测行人未来轨迹个人空间，迫使机器人在完全不避让的人群中主动让行 | Sec. III, IV-A, IV-B, Table I, Table II |
| **PRISM**<br>[Chen et al., 2026](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/prism_2609.18125.txt)<br>arXiv:2609.18125 | 2D LiDAR 扫描特征 + 预训练编码器提取的 8 维隐式交互风格 $\tilde{z}$ 与 1 维时间稳定性置信度 $\alpha$ | 连续非完整约束 $[v, \omega]$（Turtlebot 运动学，$v\le 0.5$ m/s） | **PRISM**（Transformer 编码器 + Rank-N-Contrast 序数对比损失）+ 冻结编码器接入 HEIGHT（**PPO**） | PyBullet 9$\times$9m，8–12 静态家具，5–9 人。**连续社会刚度** $\mu_i \in [0, 1]$ 线性插值 ORCA 参数（$\mu=1$ 谨慎避让，$\mu=0$ 激进不让） | 成功率、总碰撞率、人碰率、障碍物碰撞率、导航时间、路径长度 | 将被动观察到的人人交互历史映射为连续序数交互风格（避让倾向），为下游策略提供显式意图条件 | Sec. 4.1, 4.2, 4.4, Table 1, Table 2 |
| **CP-Mapless**<br>[Anas et al., 2023](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/anas_2304.03593.txt)<br>arXiv:2304.03593 | 2D 激光雷达 + 局部路标目标状态 + 自身位姿与线速 + 危险度 Top-$K$ 障碍物状态 | 连续速度控制（Turtlebot3，线速 $0\sim 0.22$ m/s，角速 $\pm\pi/2$） | **TD3**（结合局部路标密集奖励塑形与碰撞概率 CP 注意力机制） | Gazebo 仿真环境；训练集中所有动态障碍物以随机速度与方向运动，**完全不避让（100% 非合作）**；测试 20 障碍物 | 成功率（SR）、到达时间、自我安全评分（Ego Score）、社交安全评分（Social Score） | 在无地图观测中引入动态障碍物碰撞概率（CP）筛选最危险的 Top-$K$ 移动体，解决非合作人群下的冰冻机器人问题 | Sec. III-A, IV, V-A, Fig. 4, Fig. 6 |

---

### 2. 共识（Consensus）

1. **同构互惠假设必须打破**：仅在纯同构 ORCA 互惠环境下训练会导致策略在非合作现实人群中灾难性退化（成功率从近 100% 暴跌至 30%–40%），必须在训练或评估中引入不避让/行为变异智能体。  
   *出处：[Sigal et al., 2023, Fig. 2 & Table 2](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/diverse4_2308.14947.txt#L380-L420)；[Liu et al., 2023, Sec. IV-A](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/crowdnavpp_2203.01821.txt#L280-L310)*
2. **人人交互（Human-Human, HH）对理解异质人群不可或缺**：只建模人机交互（RH）无法区分行人是被迫转向还是自主避让；显式解耦 HH 拓扑能让机器人在杂乱不避让人群中做出前瞻性避让。  
   *出处：[Liu et al., 2023, Sec. IV-C](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/crowdnavpp_2203.01821.txt#L360-L390)；[Chen et al., 2026, Sec. 1 & 4.3](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/prism_2609.18125.txt#L100-L130)*
3. **“不可见机器人”（Invisible Robot / Non-yielding）是训练保守鲁棒策略的标准环境设计**：将仿真行人设为对机器人完全不可见（即行人 0% 避让机器人，仅人与人避让），能有效抑制 RL 策略学习“强制逼停行人”的过激特权流氓行为。  
   *出处：[Liu et al., 2023, Sec. IV-A](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/crowdnavpp_2203.01821.txt#L295-L315)；[Everett et al., 2018, Sec. III-C](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/everett_1805.01956.txt#L140-L170)*
4. **两阶段/课程学习（Curriculum Learning）是训练收敛的关键支撑**：先在简单稀疏或单一行为环境中建立基础目标导航能力，再过渡至密集、高维或异质非合作人群，能显著稳定训练并提高最终收敛上限。  
   *出处：[Everett et al., 2018, Sec. III-C](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/everett_1805.01956.txt#L165-L185)；[Sigal et al., 2023, Sec. 3](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/diverse4_2308.14947.txt#L215-L255)*
5. **安全性与导航效率的帕累托妥协**：在异质不避让环境中，成功率最高且不适率最低的策略往往伴随轻微上升的导航耗时与路径长度，策略学会了“以距离/时间换取安全冗余”。  
   *出处：[Sigal et al., 2023, Sec. 4.5](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/diverse4_2308.14947.txt#L560-L590)；[Liu et al., 2023, Sec. IV-C](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/crowdnavpp_2203.01821.txt#L400-L420)*

---

### 3. 分歧与已知弱点（Disagreements & Open Weaknesses）

1. **“等训练预算”对照普遍缺失（仅 Diverse-4 严格执行）**：除 Sigal 2023 锁定 10,000 episodes 证明环境异质性优于基线外，Everett 2018、Liu 2023 与 Chen 2026 均未做“固定样本步数、纯环境异质性消融”，性能增益与训练总样本量、网络架构改良混叠。  
   *出处：[Sigal et al., 2023, Sec. 4.1](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/diverse4_2308.14947.txt#L270-L290)；[Everett et al., 2018, Table I](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/everett_1805.01956.txt#L380-L410)*
2. **混合比例设置高度经验化且缺乏敏感性分析**：混合比例存在极大分歧：Everett 仅称“随机抽取（random assortment）”未给分布；Liu 2023 与 Anas 2023 直接设为 100% 不避让；Chen 2026 设为均匀连续谱系；无任何论文探讨“非合作行人比例在 10%–90% 渐变时策略失效的相变临界点”。  
   *出处：[Everett et al., 2018, Sec. III-C](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/everett_1805.01956.txt#L150-L165)；[Chen et al., 2026, Sec. 4.1](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/prism_2609.18125.txt#L210-L235)*
3. **测试集泛化验证范围存在局限（OOD 程度不一）**：Diverse-4 验证了尺度与密度 OOD（大场景+2倍密度）；但 CrowdNav++ 与 PRISM 的测试行人行为依然受限于生成它的底层规则（如 ORCA 随机变种），并未迁移至真实轨迹数据集驱动的黑盒未知动态中检验。  
   *出处：[Sigal et al., 2023, Sec. 4.3](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/diverse4_2308.14947.txt#L330-L360)；[Chen et al., 2026, Sec. 5 Discussion](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/prism_2609.18125.txt#L410-L435)*
4. **模型容量对环境异质性的消化能力存在冲突**：Diverse-4 实验发现 CADRL 即使在混合异质环境下训练，性能也几乎不提升（无容量吸收多样性），而 SARL 显著提升；但不同工作中简单 MLP 与图网络的泛化门槛界限仍未形成统一理论刻画。  
   *出处：[Sigal et al., 2023, Sec. 4.5](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/diverse4_2308.14947.txt#L500-L530)*
5. **合成刚度（Synthetic Stiffness）与真实人类社会规范脱节**：PRISM 作者承认将 ORCA 时域及视距插值作为“社会刚度”属于合成标签，真实人类行为是多维且依赖语境的，sim-to-real 中无法获得干净的刚度监督信号。  
   *出处：[Chen et al., 2026, Sec. 5 & 6](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/prism_2609.18125.txt#L425-L450)*

---

### 4. 可直接借用的组件（Component Reuse）

1. **Diverse-4 训练体系与测试基准代码**
   - **内容**：集成 ORCA 与 Social Force 混合采样的 Gym 环境，含 Simple/Large/Dense $\times$ Circle/Square 六类环境配置与 Curriculum/Diverse 训练脚本。
   - **链接**：[github.com/RAISE-Lab/soc-nav-training](https://github.com/RAISE-Lab/soc-nav-training)（已核实：HTTP 200 正常访问）
2. **CrowdNav++ 时空交互图与意图预测网络**
   - **内容**：解耦人人（HH）与人机（RH）注意力的 PyTorch 网络实现，集成 Gumbel Social Transformer (GST) 轨迹预测与连续动作空间 PPO。
   - **链接**：[github.com/Shuijing725/CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)（已核实：HTTP 200 正常访问）
3. **CrowdNav++ Sim2Real 部署桥接包**
   - **内容**：针对 Turtlebot 平台提供的 2D LiDAR 聚类与真实环境推理节点包。
   - **链接**：[github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot](https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot)（已核实：HTTP 200 正常访问）
4. **GA3C-CADRL ROS 部署包**
   - **内容**：Everett 开源的 ROS 局部规划器节点，支持变长观测输入与速度指令发布。
   - **链接**：[github.com/mfe7/cadrl_ros](https://github.com/mfe7/cadrl_ros)（已核实：HTTP 200 正常访问）
5. **可复用的社交评估指标模块**
   - **侵入时间比（ITR）与侵入社交距离（SD）**：[CrowdNav++ Sec. IV-B](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/crowdnavpp_2203.01821.txt#L330-L360) 提出的基于未来 $K$ 步真实轨迹个人空间侵入量化指标。
   - **不适率（Discomfort Rate）**：[Diverse-4 Sec. 4.2](file:///home/abc/.gemini/antigravity-cli/brain/b6e38420-542f-458e-b331-fe277cb836c3/scratch/diverse4_2308.14947.txt#L295-L315) 统计整个回合中与行人距离侵入 $<0.2$m 的时间步比例。
