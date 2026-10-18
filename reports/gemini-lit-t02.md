### 研究问题（RQ）冻结与文献调查简述
针对【X2 对照】主题，本调研聚焦以下 3 个核心研究问题：
* **RQ1**：是否存在正面消融对比“前级滤波/跟踪/不确定性估计 + 策略”与“训练期直接加噪/鲁棒训练”的工作？其相对表现为何？
* **RQ2**：在何种噪声特性与交互条件下前级滤波更优，在何种条件下鲁棒训练更优？
* **RQ3**：当前人群导航与动态避障领域的共识架构演进方向是什么？

---

### 1. 逐篇精读表

| 论文 | 状态表示 | 动作空间 | 网络/算法 | 训练方式与环境(行人模型、人数) | 评测指标 | 核心思想(一句话) | 原文依据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[Joshi 2023]**<br>*Sim-to-Real Deep RL based Obstacle Avoidance for UAVs under Measurement Uncertainty*<br>第一作者: Bhaskar Joshi<br>年份: 2023<br>arXiv: 2303.07243 | 相对目标距离 $(\Delta x_g, \Delta y_g)$ 及最近障碍物距离 $(\Delta x_o, \Delta y_o)$，共 4 维连续量 | 连续平面线速度 $[v_x, v_y]^T \in [-0.2, 0.2]^2$ m/s | PPO (MLP Actor-Critic，基于 Stable-Baselines3) | PyBullet 物理仿真，0~3 个随机静态/动态障碍物；分别在无噪 ($\mu=0,\sigma=0$)、低噪 ($\sigma=0.1$)、高噪 ($\sigma=1.0$) 下训练；Crazyflie 2.1 无人机实飞 | 成功率 (SR)、碰撞率、回合耗时、实机成功率 (S/F) | 纯滤波无法消除传感器偏置(Bias)，适度训练加噪抗偏置能力远优于滤波，而“低噪训练+卡尔曼滤波”级联取得全噪声范围最高稳定性。 | Sec. IV-C, Sec. V-A/B/C (Fig. 4, 5, 6, Table III) |
| **[Yao 2025]**<br>*Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling*<br>第一作者: Jianpeng Yao<br>年份: 2025<br>CoRL 2025 / arXiv: 2508.05634 | 自车状态 + 周边行人运动状态 + 行人未来轨迹分布与自适应保形不确定性半径 $r_i^t$ | 连续线速度 $v \in [0, 1]$ m/s，角速度 $\omega \in [-\pi/2, \pi/2]$ rad/s | 约束强化学习 (CRL，拉格朗日乘子法) + 时空图注意力 (GST) | 仿真圆形对穿场景，5 名行人 (ORCA 模型)；OOD 测试引入快速冲刺 (2 m/s)、社会力 (SF) 及群组行为；TurtleBot2 实车验证 | 成功率 (SR)、碰撞率 (CR)、超时率 (TR)、侵入未来轨迹率 | 纯端到端预测增强策略在分布偏移下易因预测误差而失稳，利用前级在线保形推断动态量化不确定性并约束策略，能彻底遏制碰撞激增。 | Sec. 3, Sec. 4.2–4.4 (Table 1, Fig. 2) |
| **[Everett 2022]**<br>*Certifiable Robustness to Adversarial State Uncertainty in Deep Reinforcement Learning*<br>第一作者: Michael Everett<br>年份: 2022<br>IEEE TNNLS / arXiv: 2004.06496 | 自车航向目标相对坐标 + 周边动态行人相对位置、速度、半径 (GA3C-CADRL 状态) | 离散线速度与航向角组合 (11 个离散动作) | DQN + CARRL (基于区间凸松弛边界的在线动作值可证明鲁棒性防御) | 行人碰撞规避仿真环境，4 名动态行人 (ORCA 策略)；对比名义训练、加噪/对抗训练与 CARRL 在线防御 | 均值奖励、碰撞率、最小行人间距、认证动作下界裕度 | 训练期直接加噪/对抗训练只能改善平均鲁棒性且依然存在被最坏微扰误导的风险，推理端在线计算不确定性状态动作值下界才能提供确定性安全防线。 | Sec. I, Sec. IV, Sec. V-A (Fig. 4, Table I) |
| **[Golchoubian 2024]**<br>*Uncertainty-Aware DRL for Autonomous Vehicle Crowd Navigation in Shared Space*<br>第一作者: Mahsa Golchoubian<br>年份: 2024<br>IEEE IV 2024 / arXiv: 2405.13969 | 自动驾驶低速车辆位姿 + 行人多模态轨迹预测均值与位置不确定性协方差椭圆 | 连续加速度 $a \in [-3, 2]$ m/s$^2$，转向角速度 $\dot{\delta} \in [-0.5, 0.5]$ rad/s | SAC (Soft Actor-Critic) + 极坐标碰撞网格 (PCG) 预测 + 不确定性感知碰撞风险奖励 | 混合交互仿真环境（人车让行 + 行人-行人 ORCA/SF），最多 10 名行人；对比无不确定性端到端 DRL 及不确定性 MPC | 碰撞率、导航耗时、行人最小间距、个人空间侵入率、驾驶轨迹拟人性 | 前级显式估计行人位置高斯协方差并输入策略与奖励，相比忽略不确定性的端到端 DRL 碰撞率下降 40%，且规避了传统 MPC 的在线计算瓶颈。 | Sec. III-A/B, Sec. IV-A/B (Table I, II, Fig. 4) |
| **[Xu 2026]**<br>*KinematicRL: A Sim-to-Real RL Framework For Social Navigation With Kinodynamic Feasibility*<br>第一作者: Zhiming Xu<br>年份: 2026<br>IEEE T-ASE / arXiv: 2606.12042 | 自车高阶二阶运动学状态 + 2D LiDAR 聚类时序平滑提取的动态行人连续速度与位置 | 连续二阶加速度控制量 (线加速度、角加速度) | 门控 Transformer + iLQR 预训练 + DRL 微调 (KinematicRL) | 仿真动态人群场景，随机化起始点与行人速度 (ORCA/SF)；物理差速驱动机器人于真实人群密集走廊部署 | 成功率、冻结率 (Freezing)、Jerk 平滑度、跟踪误差 RMSE | 仅在策略端做 Domain Randomization 无法解决高阶控制下的速度估计噪声发散，必须结合前端聚类时序滤波，两者闭环方能消除实机振荡与碰撞。 | Sec. III-C/D/E, Sec. IV-C/D (Table I, Fig. 6) |

---

### 2. 共识：这一类工作普遍认同、反复采用的做法

1. **分级解耦与状态注入（Hierarchical Hybrid Coupling）**  
   纯粹将裸带噪观测端到端输入固定策略，或仅在仿真中训练而不在状态中感知不确定性，在实机上极易产生高频抖动或冻结（Frozen Robot Problem）。普遍认同的最佳流水线是：**前级滤波/跟踪器负责滤除传感器高频白噪声并输出“状态估值 + 协方差/不确定性区间”；策略端将该不确定性显式作为输入并结合 Domain Randomization 训练**。  
   *出处：[Joshi 2023] Sec. V-B; [Golchoubian 2024] Sec. III-A; [Yao 2025] Sec. 3.*

2. **卡尔曼与低通滤波对传感器偏置（Sensor Bias）的本质失效**  
   前级卡尔曼滤波（KF）与低通滤波（LPF）严格依赖“零均值高斯白噪声”假设。当真实世界存在静态测距偏移、系统延迟或校准残差（$\mu > 0$）时，前级滤波无法抑制误差漂移；而在策略训练期注入扰动（鲁棒训练），使策略自身学会动作抖动补偿（Dithering）与迟滞容忍，是对抗偏置的唯一通用机制。  
   *出处：[Joshi 2023] Sec. V-B.2 & Fig. 5.*

3. **训练过度加噪（Over-Randomization）导致“策略退化”**  
   在鲁棒强化学习训练中，直接加噪的幅度存在严格上限。若注入过大噪声（如高斯方差 $\sigma$ 过大），策略会过度拟合极端对抗转移，导致其在无噪或微噪的常规工况下表现灾难性滑坡（成功率甚至跌至 0%）。  
   *出处：[Joshi 2023] Sec. V-B.1 (Policy 3 在 $\sigma < 0.3$ 下接近 0% 成功率); [Everett 2022] Sec. V-A.*

4. **确定性多步预测会导致严重的级联错误（Error Cascading）**  
   在人群密集环境中，行人间存在博弈交互，确定性轨迹预测（如恒速模型 CV 或开环 LSTM）在遭遇行人变向、群组抱团或加速冲刺等分布偏移（OOD）时残差急剧发散；直接基于确定性预测规划会导致策略盲目自信而引发剧烈碰撞。  
   *出处：[Yao 2025] Sec. 1 & Sec. 4.4; [Golchoubian 2024] Sec. I.*

5. **实车部署必须在策略输入端平滑速度一阶导数**  
   实物差速或无人机底盘对速度突变敏感。直接由带噪观测计算差分速度输入网络，会导致执行器剧烈 Jerk 并触发底盘保护停机；前级必须采用时序加权滑动平均、聚类速度关联或卡尔曼平滑提取稳定速度。  
   *出处：[Xu 2026] Sec. III-C; [Joshi 2023] Sec. V-C.*

---

### 3. 分歧与已知弱点：作者自己承认的局限与矛盾结论

1. **“在线防御下界”与“名义任务性能”的取舍矛盾**  
   * [Everett 2022] 强调在线动作值下界松弛（CARRL）是杜绝最坏扰动碰撞的唯一保证，承认其代价是牺牲了无攻击/低噪场景下的名义奖励（表现偏保守）；  
   * 矛盾点：[Joshi 2023] 与 [Golchoubian 2024] 则表明，在实际物理环境中，最坏情况防御会导致智能体在人群缝隙中频繁停滞（Freezing）；通过低噪域随机化训练（Domain Randomization）获得的经验鲁棒策略，在平均性能和通过效率上远胜过于保守的硬边界方法。

2. **自适应不确定性估计的计算延迟瓶颈**  
   * [Yao 2025] 采用自适应保形推断（ACI）并承认：在线计算保形分位数虽保证了边缘覆盖率，但在极端多人群场景（行人数量动态暴增时）下维护非参数化校准集的计算延迟，会对高频控制回路（>20Hz）造成严峻考验；  
   * 对应地，[Golchoubian 2024] 坚持采用参数化高斯椭球以换取毫秒级推理，但其局限在于强行假设单峰高斯分布，无法表达行人避障时的多模态分流。

3. **仿真交互模型与真实人群博弈机理的鸿沟（Sim-to-Real Interaction Shift）**  
   * [Xu 2026] 与 [Yao 2025] 均承认：仿真训练无论如何加噪，其基底仍高度依赖 ORCA、社会力模型（SF）等预设规则。真实行人的注意力分散、对抗试探或群组社交遮挡难以用独立高斯噪声完整等效；训练期合成加噪学出的“鲁棒性”在面临结构性意图偏移时仍存在失效盲区。

4. **噪声注入是否能改善纯偏置场景下的性能存在实验分歧**  
   * [Joshi 2023] 的实验发现一个反直觉现象：在纯偏置（$\mu > 0, \sigma = 0$）场景下，给评估输入人工注入零均值方差噪声 $\sigma$ 反而能触发探索性动作并打破死锁，使成功率从 0% 提升至 40%~70%；  
   * 但该论文同时证实：该现象仅在特定中等偏置窗口有效，偏置过大或方差过大时策略性能会瞬时断崖式下跌，且尚无严密理论边界解释该经验现象。

---

### 4. 可直接借用的组件：开源代码、复用模块与环境

1. **[Joshi 2023] UAV 测量噪声与滤波实测对比基准**  
   * 链接：https://github.com/dkapur17/DroneControl  
   * 状态：`已核实可访问 (HTTP 200)`  
   * 可复用部分：含完整的 PyBullet 连续空间导航环境、PPO 噪声训练脚本，以及集成卡尔曼滤波（KF）与二阶贝塞尔低通滤波（LPF）的即插即用前端去噪包装器（Denoiser Wrappers）。

2. **[Yao 2025] 人群导航自适应保形不确定性估计模块（ACI-CrowdNav）**  
   * 链接：https://gen-safe-nav.github.io/ （源码索引与评估基准）  
   * 状态：`已核实可访问 (HTTP 200)`  
   * 可复用部分：基于 Adaptive Conformal Inference (ACI) 的动态不确定性椭圆估计组件、结合拉格朗日乘子约束强化学习（CRL）的安全边界惩罚项实现。

3. **[Golchoubian 2024] 不确定性感知多模态轨迹与 DRL 共享空间环境**  
   * 链接：https://github.com/Golchoubian/UncertaintyAware_DRL_CrowdNav  
   * 状态：`已核实可访问 (HTTP 200)`  
   * 可复用部分：Gym 共享空间人车交互模拟器、极坐标碰撞网格（PCG）预测模型及将行人轨迹协方差直接映射为碰撞概率损失的奖励函数实现。

4. **[Xu 2026] 2D LiDAR 聚类时序跟踪与差速机器人实车底盘节点**  
   * 链接：https://github.com/Mr-Wonderfool/KinematicRL  
   * 状态：`已核实可访问 (HTTP 200)`  
   * 可复用部分：纯 2D 激光点云下的轻量级行人聚类、空间距离与速度相似度多目标数据关联前端，以及基于 ROS 的即插即用平滑速度估计节点。

5. **[Everett 2022] 动态行人碰撞规避环境与鲁棒性验证基准**  
   * 链接：https://github.com/mit-acl/gym-collision-avoidance  
   * 状态：`已核实可访问 (HTTP 200)`  
   * 可复用部分：MIT ACL 开源的多智能体行人避障标准仿真环境（支持 ORCA/CADRL 行人模型切换、传感器测量扰动注入与碰撞安全指标统计）。
