# wu-01 冻结设计

状态：未验证（在任何训练与选参之前写入）。只读复用原目录，所有新增文件写入本目录。

## 对照和预算

实际 R0 入口为 branch-restart-pilot-20261017/pilot.py 和 formal-worker.py；该目录无 train.py。Pilot 初始化 s443 GRU T24 模型及 AdamW 状态，续训 RNG 为 protocol.json 中 pilot_seed=449。保持 20 人、circle/square 交替、1200000 起递增训练 case、空 MC replay、容量 200000、batch 256、gamma .99、lr 1e-5、原权重衰减/优化器状态、每 201 实际步最多 4 次更新。完成回合才计算 MC，恰好 402000 实际 env.step，末尾未完成回合丢弃。R0 formal-worker 的首次 restoration_test 消耗 correctness 步且重置历史；将复用同一测试并计入两臂预算，以避免隐含预算差异。IL 已包含在母体中，本任务不额外重做 IL。

## X1

在 Pilot.decide 已取得网络候选打分后，用缓存的同一次 score_sarl_candidates 输出构造老师输入；老师 best 为贪心网络选择（不含 epsilon），previous 为决策前实际网络执行动作。直接调用原 risk-trigger controller_class(frozen.Controller)，enabled=True、horizon=2、margin=.1，保留原低进展、16 步激活、无安全候选时停用、每步重规划和硬安全认证。老师计算隔离 torch CPU/CUDA RNG，不调用真实环境。调用后始终返回原 Pilot 网络动作（含原 epsilon），绝不执行老师动作。只有实际调用优化器且通过认证的老师标签进入蒸馏池；认证失败保留日志，不伪造老师动作。

保存共享的 23 帧历史前缀、80 个候选后继 token、原安全/风险/即时奖励打分常数和老师投影离散 index（原 Controller.project）。蒸馏更新重新前向 80 个候选价值：logit=(原常数 + .99 V_theta)/temperature，temperature=.1；保留原候选安全屏蔽与零动作惩罚。每次 R0 更新合并原 MC MSE + lambda * CE，单次 backward/clip/AdamW step，更新次数不变。蒸馏 batch=4，FIFO=2048，独立 random.Random(462) 采样，禁止改变 MC 采样 RNG。lambda=0 且蒸馏关闭直接走原更新函数。

lambda 规则（运行前固定）：分别用 .1、.3、1.0 从同一母体跑 2000 实际步；以独立 dev case 1250000 起、最多 2000 步收集的前 32 个通过认证老师样本作固定验证集，不参与这些 smoke 更新。选 dev 老师动作交叉熵最低的有限候选，完全相等选较小 lambda。任何候选非有限或工程异常，停止选参并修复自写错误后重测，不将异常算效果失败；无足够有效标签则不得正式启动 X1。此规则只检验损失接线，不据此声称导航收益。

## X2

在 Pilot.state 插入 episode-local 观测器；每回合独立 RNG 取 sigma~U(0,.15)、drop~U(0,.2)，每个人位置独立 Gaussian sigma、速度 Gaussian 2sigma，先噪声后漏检，漏检沿用上次观测、不外推；首观测无历史则不漏。沿用 hetero.py 的观测顺序/随机数消耗语义。一个物理时刻仅生成一次观测，token/state 重读返回缓存，环境真实状态和奖励不变。sigma/drop 全关时直接返回父类 state，不消费 RNG。没有信念、滤波或风险不确定性机制。

## 回归、冒烟、资源与评测

regression.py 以原 R0 路径为参照，在相同设备/种子逐步比较动作、完整 simulator snapshot、tokens/windows、经验池内容、每次更新损失、模型和 AdamW 状态；覆盖 warmup 后更新。X1 disabled 与 X2 disabled 必须逐位一致。另检查候选分解重建与原打分一致、蒸馏梯度有限非零、观测生效且真实状态不被修改。正式采集前复用 R0 restoration_test；X1/X2 自有 RNG/缓存须在该 correctness 审计中冻结，审计只检验干净的公共采集路径，不将 audit 老师标签加入训练。

本地最多 3 个本任务计算进程，默认串行；开跑读取 MemAvailable，低于 2 GiB 不加并发、停止新增计算，训练存活另记 PID/start_ticks。容器 mamba_env 的 /workspace 对应主机 workspace；仅用指定 runtime/bin/python。远端源目录位于 /root，与本地 workspace 的相对布局一致；只部署本目录，既有依赖哈希核对。4090 仅先部署与冒烟，正式启动需 loadavg 1min<4，并记录时间；等待期间不触碰其他任务。

评测前在本地和远端扫描旧文件中的 case 字段和区段声明，冻结 410000–410499 的无旧命中证据。代码和报告中本任务的预注册声明不算旧使用；任何旧结果命中即停止该块。全部最终权重以原 ral-strongmother experiment.episode 纯一步前瞻评测；仅用本地 adapter guard 授权新块，不修改原文件。base、R0、X1、X2 两场景各 500；X2/R0 另加 square noise=.1 和 drop=.2 各 500、扰动 RNG=case*17+29，按 case 配对。bootstrap=10000、seed=20261017、percentile 95%，同时报告 SR/CR/TO、路径、成功时间、推理开销。按用户冻结规则：对 R0 ΔSR≥2pp 且 CI 下界>0 为好；上界<0 为坏；否则未知。X2 的 square 干净下界>−2pp 且位置噪声 .1 格下界>0 也判好；漏检 .2 格另列结果（按用户“噪声格”字面执行，不额外要求漏检格也显著）；20S 为主判据，20C 单列。正式 test 不用于调参。插件评测为可选，不挤占必做项目。

新增：DESIGN.md、core.py、regression.py、train.py、evaluate.py、audit.py、summarize.py、pipeline.py、PROGRESS.md、RESULT.md 及运行证据。原目录只读。

## 冒烟接线修正说明（不改选参规则）

X2 的首次 correctness 审计重放第一次实际噪声轨迹的 token/action tape，逐步比较真值世界与历史，并检验当前权重重建和精确 budget 截断；其两次 12 步前缀、1 步对照、3 步小 budget 与原 R0 一样共消耗 28 真实步。audit 不重抽噪声，也不生成新训练标签。原 clean-token 断言只适用于干净观测，不能用于 noisy tape；此替代检查保留同样的物理重放证明和预算。工程失败日志保留，详见 ENGINEERING.md。

所有评测输出先写 JSONL 原始回合及完成哈希，再做严格成对统计。远端使用单例先行评测和余下区段，本地纯一步前瞻评测沿用已核对的官方 episode 函数。结果报告裁决只使用用户导航判据。
