### 研究问题定义 (Research Brief)
1. **RQ1（噪声与漏检建模）**：在人群导航与控制中，传感器观测噪声（位置/速度高斯噪声 $\sigma$）、漏检/遮挡（Missing detection / FoV 几何遮挡）及最坏情况对抗扰动（$L_\infty$ ball $\epsilon$）在数学与工程中如何定量定义？
2. **RQ2（训练-测试分布一致性）**：训练阶段与测试阶段的噪声/扰动分布是否一致（即 Nominal clean / Data augmentation / Domain randomization / Adversarial training 之间的差异）？
3. **RQ3（基线与评测标准）**：现有工作如何权衡成功率（SR）、碰撞率（CR）与社交规范侵犯，报告了哪些典型基线？

---

### 1. 逐篇文献精读与横向对比表

所有论文均已下载并核实原始 PDF 全文。

| 论文 | 状态表示 | 动作空间 | 网络 / 算法 | 训练方式与环境 (行人模型、人数) | 评测指标 | 核心思想 (一句话) | 原文依据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SA-MDP**<br>Zhang et al. (2020)<br>[arXiv:2003.08938](https://arxiv.org/abs/2003.08938)<br>NeurIPS 2020 | 连续控制向量（MuJoCo 归一化连续状态，如 Walker2d 17维、Humanoid 376维） | 连续动作空间（关节力矩，如 Walker2d 6维、Humanoid 17维） | **SA-PPO / SA-DDPG**（在策略损失中加入基于凸松弛 IBP/Backward 或 SGLD 求解的最坏状态扰动正则项） | MuJoCo 连续控制基准。训练期采用扰动半径递增策略（$\epsilon$-schedule，前 3/4 阶段线性升至目标 $\epsilon$），训练集显式注入对抗扰动。 | 50 episodes 累计回报均值 $\pm$ 标准差、不同攻击下的最坏情况回报、收敛箱线图 | 将观测扰动形式化为 SA-MDP，证明策略对状态扰动的平滑性是防御理论下界，通过凸松弛网络输出边界进行鲁棒正则化。 | Section 3, Section 5, Table 1, Table 5, Appendix D/F/G |
| **CARRL**<br>Lütjens & Everett et al. (2019)<br>[arXiv:1910.12908](https://arxiv.org/abs/1910.12908)<br>CoRL 2019 | 相对低维特征向量：自身与行人/障碍物的相对 x-y 位置、相对 x-y 速度、半径及自身目标坐标 | 11 个离散航向偏角 $A \in [-\pi/6, +\pi/6]$，恒定速率 $v = 1.0\text{ m/s}$（单车运动学） | **CARRL-DQN**（离线标准 DQN，在线执行阶段引入 Fast-Lin 凸松弛层认证 $Q$ 值下界） | 运动学仿真环境，2 个智能体（1 个自车 RL，1 个行人由 Static/Non-cooperative/ORCA/CADRL 驱动）。**训练时使用干净无噪观测（clean nominal）**。 | 每轮平均碰撞次数（Average collisions）、回合平均回报（+1 达目标，-0.25 碰撞），统计 $5 \times 100$ episodes | 离线在无噪数据上训练标称 DQN，在线执行时通过快速线性松弛求解扰动球内的最坏情况 $Q$ 值下界 $Q_L$，选取最保守安全的动作。 | Section 3.2, Section 4.1, Section 5.1, Figure 3 |
| **Flögel (2024)**<br>Flögel et al. (2024)<br>[arXiv:2409.10655](https://arxiv.org/abs/2409.10655)<br>IROS 2024 | 连续向量：可观测部分 $s^{i,o}=[p, v, r]$（自身与行人位置、速度、半径）；不可观测隐藏部分包括目标、期望速度、朝向及 Hall 个人空间半径 | 连续 2D 动作空间：线速度 $v_t$ 与航向角增量 $\Delta \theta_t$ | **ODV-PPO-Drop**（引入观测依赖方差 ODV + MC-Dropout / Deep Ensembles 解耦不确定性；超阈值触发保守避碰 SCA） | 微观社会心理学仿真（ORCA + 社会力 + Hall 个人空间规范）。圆周互换、交叉与随机场景。**训练时不注入观测噪声（干净环境）**，训练 $1.4\times 10^7$ 步。 | 碰撞率（Col. %）、超时率（TO %）、个人空间侵犯次数（PV）、回合回报、保守机制避免碰撞比例（Table IV） | 改造 PPO 使动作高斯分布方差直接依赖于当前观测，配合 Dropout 解耦偶然噪声与场景认知不确定性，在不确定性骤升时平滑切换至保守避让策略。 | Section III-A, Section IV-A/B, Section V-A/B, Table I–IV |
| **Yao (2025)**<br>Yao et al. (2025)<br>[arXiv:2508.05634](https://arxiv.org/abs/2508.05634)<br>CoRL 2025 | 复合状态向量 $S_t=[e, h, m]$：自身物理状态 $e$、行人坐标与速度 $h$、模型生成特征 $m$（含预测的 $K$ 步未来轨迹及 DtACI 估计的动态不确定性球半径） | 连续 2D 速度动作 $A_t = (v_x, v_y)$，最大速率 $1.0\text{ m/s}$ | **DtACI + GST + CRL (PPO-Lagrangian)**（结合自注意力编码器，对侵入行人共形不确定性区域施加累积拉格朗日约束） | 经典 CrowdNav 仿真（$12\times 12\text{ m}$，20 名 ORCA 行人）。**训练时无观测噪声（仅加探索动作噪）**。测试时引入速度剧变（$2\text{ m/s}$）、社会力换模与紧密人群群组动力学（OOD）。 | 成功率（SR）、碰撞率（CR）、超时率（TR）、导航时间（NT）、路径长度（PL）、入侵时间比（ITR）、社交距离（SD） | 采用动态自适应共形推断（DtACI）实时量化未来轨迹不确定性球，并在约束强化学习中惩罚对该不确定性区域的侵犯，无需域随机化即可对冲 OOD 扰动。 | Section 3.1–3.4, Section 4.1–4.4, Table 1–2, Appendix K–L |
| **Mun (2023)**<br>Mun et al. (2023)<br>[arXiv:2210.00552](https://arxiv.org/abs/2210.00552)<br>IEEE RA-L 2023 | 局部占用栅格地图（OGM，中心在机器人，$10\times 10\text{ m}$，分辨率 $0.1\text{ m}$，取值 $\{0, 0.5, 1\}$ 表示空闲/遮挡/占用）+ 机器人位姿速度向量 | 连续 2D 全向速度 $(v_x, v_y)$，限速 $2.0\text{ m/s}$，加速度 $1.0\text{ m/s}^2$ | **PaS-VAE + RNN-PPO**（离线全知 GT-VAE 引导，PaS-VAE 编码 4 帧历史观测图恢复潜在障碍，输入 RNN 策略网络） | 仿真环境 6 名 ORCA 行人（半径 $0.3-0.4\text{ m}$，最大速度 $0.5-1.5\text{ m/s}$）。机器人雷达设置圆形视野截断半径 $R_{fov}=3.0\text{ m}$，视野外与障碍后方产生几何遮挡。**训练与测试均在遮挡下端到端执行**。 | 成功率（SR）、碰撞率（CR）、导航时间（Time）、额外距离（ED）、不适频率（DF）、栅格重建误差（BCE/IoU） | 将可见行人的交互避让动态作为间接传感器（"People as Sensors"），利用变分自编码器推断由于有限感知与遮挡导致的盲区目标。 | Section III, Section IV, Section V-A/B, Table I |

---

### 2. 共识：文献普遍认同与反复采用的做法

1. **训练与测试期的噪声/扰动分布普遍采用"非对称"设计（Asymmetric Train/Test Paradigm）**  
   除纯端到端遮挡建图（Mun 2023）外，CARRL、Flögel 2024 与 Yao 2025 均在训练期采用**干净、无噪的标称状态（Nominal clean observations）**，而在测试期注入高斯噪声、遮挡或 OOD 分布偏移。作者们普遍认同：若直接在强噪声/扰动下端到端训练策略网络，会导致策略价值评估极度不稳，且严重损害无扰动自然场景下的标称性能（出处：CARRL Sec. 4.1；Flögel 2024 Sec. V-A；Yao 2025 Sec. 4.1 & App. E）。
2. **朴素对抗训练（Naive Adversarial Training）在连续控制中普遍失效**  
   SA-MDP 与 CARRL 均通过实验证实，将计算机视觉中的 FGSM/PGD 扰动直接施加于观测状态并进行对抗重训练（如 Pattanaik et al. 基线），不仅无法提供鲁棒性泛化保证，反而会导致策略回报急剧崩塌。必须使用凸松弛网络输出边界（IBP / Fast-Lin）或状态平滑正则化来提供有界的防御保证（出处：SA-MDP Sec. 5 & Tab. 1；CARRL Sec. 1 & Sec. 5.1）。
3. **显式不确定性量化与保守降级行为（Uncertainty-to-Fallback / Constraint Guidance）是安全闭环的核心**  
   单纯将不确定性作为额外特征喂给黑盒 MLP 无法阻止碰撞（Yao 2025 中仅加入 ACI 但不加 CRL 的消融模型仍有 23.04% 碰撞率）。主流做法均是将不确定性显式映射为控制级干预：Flögel 2024 在碰撞概率超标时切换为保守避碰（SCA）；Yao 2025 将时变共形置信区间转换为约束侵入代价；CARRL 在不确定性球内计算最坏情况 $Q$ 下界（出处：Flögel 2024 Sec. IV-C & Tab. IV；Yao 2025 Sec. 3.2；CARRL Sec. 4.1）。
4. **两阶段感知滤波解耦（Upstream Filtering + Downstream Planning）是真实部署标准范式**  
   处理实车雷达感知抖动与短时漏检时，学界与工业界一致倾向于先由前置模块（LiDAR + DR-SPAAM 检测 + SORT / EKF 跟踪）吸收高频测量抖动与帧漏检（通过大方差滤波与推断预测补全），再由下层 RL/CRL 策略吸收行为不确定性，而非由策略网络直接硬抗原始带噪点云（出处：Yao 2025 App. K–L；Mun 2023 Sec. IV；CARRL Sec. 2.1）。

---

### 3. 分歧与作者承认的已知弱点

1. **保守度膨胀与"冻结机器人/逃逸"问题（Freezing Robot / Runaway Dilemma）**  
   过度追求观测扰动下的安全性会导致策略过于保守。CARRL 作者明确承认：当鲁棒防御半径 $\epsilon > 0.1\text{ m}$ 时，由于诱发了训练分布之外的状态，破坏了 $Q$ 函数精准性假设，智能体会直接"掉头逃离障碍"而放弃前往目标；Yao 2025 的约束强化学习在引入共形置信椭圆后，路径长度（PL）与超时率（TR）也出现了不可避免的上升（出处：CARRL Sec. 5.1 & Fig. 6f；Yao 2025 Sec. 4.3 & App. I）。
2. **采样推断（Sample-based Inference）与边缘算力实时性的冲突**  
   Flögel 2024 采用 MC-Dropout / Deep Ensembles 评估策略不确定性，作者明确指出该方法在推理期需要多次前向网络传播（Multi-pass），在板载低功耗计算平台上开销过高，必须进一步研发"无采样（sample-free）"替代方案；CARRL 的线性松弛下界（Fast-Lin）同样计算复杂度受限，难以扩展至多智能体大网络（出处：Flögel 2024 Sec. VI-C；CARRL Sec. 6）。
3. **偶然不确定性（噪声）与认知不确定性（OOD）的解耦并不彻底**  
   Flögel 2024 试图通过 ODV 分离传感器噪声（Aleatoric）与陌生场景/人群（Epistemic），但在多行人与加速场景下，两者依然呈现交叉相关；且策略对航向角噪声呈线性敏感，对线速度噪声却表现迟钝（平坦），未达成理想的完全解耦（出处：Flögel 2024 Sec. V-B & Fig. 3）。
4. **"以人为感"（People as Sensors）强先验假设的脆弱性**  
   Mun 2023 假设视野外被遮挡的目标必然会引起视野内行人的避让反应。作者自承：如果外部环境中的行人互不关注（如各自看手机）、遵守不同的社会规约、或产生无规律停顿，该间接推断模型将出现严重虚警或失效（出处：Mun 2023 Sec. III & Sec. VIII）。
5. **理论凸松弛边界过于宽松导致次优性（Loose Relaxation Bound）**  
   SA-MDP 依靠 IBP（区间有界传播）和线性松弛求解，作者指出由于神经网络非线性激活层的层层放大，松弛边界通常比真实对抗边界更宽，导致策略在简单环境（如 Hopper, Walker2d）中为抵御虚假最坏情况而牺牲了名义最佳性能（出处：SA-MDP Sec. 5 & App. G）。

---

### 4. 可直接借用的组件与开源资源

所有仓库链接均已通过网络实际测试，访问状态已明确标注：

1. **[tasl-lab/GenSafeNav](https://github.com/tasl-lab/GenSafeNav)**（**已核实：HTTP 200 可访问**）  
   * **项目主页**：[gen-safe-nav.github.io](https://gen-safe-nav.github.io/)（**已核实：HTTP 200 可访问**）  
   * **ROS2 部署仓库**：[tasl-lab/GenSafeNav-ROS2](https://github.com/tasl-lab/GenSafeNav-ROS2)（**已核实：HTTP 200 可访问**）  
   * **可直接复用模块**：
     - `DtACI`（动态自适应共形推断）时变不确定性球估计算法；
     - 包含 20 人密集动态、速度翻倍突变（$2\text{ m/s}$）、SFM 动力学切换及群组聚集（Group dynamics）的 CrowdNav 拓展仿真环境；
     - 基于 PPO-Lagrangian 的不确定性区域侵入惩罚约束模块；
     - ROS2 节点中针对 6Hz 激光雷达检测抖动与漏检的 SORT + EKF 参数配置与补帧逻辑。
2. **[yejimun/PaS_CrowdNav](https://github.com/yejimun/PaS_CrowdNav)**（**已核实：HTTP 200 可访问**）  
   * **可直接复用模块**：
     - 基于光线投射（Ray tracing）构建有限视野（$R_{fov}=3\text{ m}$）与动态遮挡障碍物阴影的局部栅格地图（OGM，取值 $\{0, 0.5, 1\}$）生成器；
     - 时序 PaS-VAE 网络结构与全知先验 GT-VAE 潜空间对齐损失函数（KL 散度 + 重建损失）；
     - 视野遮挡下的主动避碰奖励函数与评价指标（不适频率 DF、额外绕行距离 ED）。
3. **[chenhongge/StateAdvDRL](https://github.com/chenhongge/StateAdvDRL)**（**已核实：HTTP 200 可访问**）  
   * **可直接复用模块**：
     - SA-PPO 与 SA-DDPG 核心算法实现，包含利用 SGLD 与凸松弛（IBP/CROWN）计算状态观测对偶正则化的 PyTorch 模块；
     - 经典强化学习黑白盒状态对抗攻击套件（MAD 攻击、Robust SARSA 攻击）。
4. **CARRL 与 Flögel 2024 的复用情况说明**  
   - **CARRL**：作者未在论文中提供独立 GitHub 仓库（经核实原文无开源链接，曾致谢 Fast-Lin 算法作者）。其核心思想依赖的神经网络线性松弛库可直接选用开源的 [Certified-Computing/auto_LiRPA](https://github.com/Certified-Computing/auto_LiRPA) 或 [chenhongge/StateAdvDRL](https://github.com/chenhongge/StateAdvDRL)。  
   - **Flögel (2024)**：论文正文中未附带独立开源仓库链接（经核实原文）。其底层依赖 [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) 的 PPO 框架，其 ODV 机制可直接在 Actor 网络输出层将对角协方差改为由观测特征映射的全连接分支实现。
