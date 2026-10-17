# R3 Formal Pilot

已验证：402000实际环境步训练、最终权重1000例纯一步前瞻评测和case配对统计完成。
本轮测量R3对base的效果；没有R0同预算比较，不能认定规划器引导采集的独立算法收益或论文贡献。

**研究结论：本轮没有取得方法级进展。** 20S超时率下降4.2pp，但成功率仅+1.6pp、95%区间跨0，同时碰撞率观察值+2.6pp、实际几何接触28→41。20C成功率+0.4pp，区间也跨0。结果有效，但未建立稳定、安全的净导航收益，R3不能据此作为已经通过的论文方案。

## Training

| Steps | Complete trajectories | Updates | MPPI calls | MPPI CPU seconds | MPPI wall seconds | Training hours | Unlabelled tail steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 402000 | 5865 | 7992 | 159428 | 8639.66 | 8650.74 | 5.34 | 64 |

Seed449; original s443 model+AdamW,empty replay,not smoke weights; literal frozen mpc,heuristic terminal unchanged.
All real env.step calls counted. Internal planner model rollout is excluded from environment-step count;its CPU/calls are reported above.
Checkpoints at100000/200000/300000/400000 and402000(final);only final evaluated. Intermediate checkpoints are not lossless resume archives.

## Evaluation

| Scene | Arm | Success / collision / timeout | SR / CR / TO (%) | Physical contacts | Success time(s) | Inference ms |
|---|---|---|---|---:|---:|---:|
| square | base | 404 / 29 / 67 | 80.80 / 5.80 / 13.40 | 28 | 18.67 | 15.26 |
| square | R3 | 412 / 42 / 46 | 82.40 / 8.40 / 9.20 | 41 | 16.17 | 12.61 |
| circle | base | 481 / 10 / 9 | 96.20 / 2.00 / 1.80 | 9 | 16.86 | 15.17 |
| circle | R3 | 483 / 12 / 5 | 96.60 / 2.40 / 1.00 | 13 | 16.03 | 12.36 |

## Paired R3 minus base

| Scene | SR pp [95%CI] | CR pp [95%CI] | TO pp [95%CI] | Rescued / damaged |
|---|---|---|---|---:|
| square | 1.60 [-2.60, 5.60] | 2.60 [-0.20, 5.40] | -4.20 [-7.60, -1.00] | 59 / 51 |
| circle | 0.40 [-1.80, 2.60] | 0.40 [-1.20, 2.00] | -0.80 [-2.20, 0.60] | 16 / 14 |

## Evidence and verdict

科学裁决：INCONCLUSIVE，对应命题“本版R3相对Base具有稳定成功率增益、且不牺牲安全”。两格SR区间均跨0，不能宣布成功，也不能仅据不显著性断言整个机制无效。方形TO下降的配对区间不含0，支持本次固定模型/案例集合中超时减少；不证明这由规划器引导独立造成，因为尚无同预算R0比较。碰撞增加是本集合的观察值，区间跨0，不写成已经确认的总体安全退化。

研究决策：HOLD，暂停后续训练及调参。唯一缺口是乙已安排的同预算R0纯继续RL对照：它能区分普通20人继续训练与规划器引导采集的作用。获取成本是读取既有逐例数据、复算配对统计，分钟级，无需新GPU运行。若R0吸收R3效果，则KILL；只有超出该对照的任务增量和可接受安全结果，才考虑后续GO。这里没有事后新增数值过关阈值。

唯一下一动作：用已安排的冻结R0结果与本次最终R3逐例配对，不重训、不扫参，不仅凭当前R3-Base差值宣称创新。

Training collector mixes one-step epsilon exploration with frozen mpc; MC targets are the actual mixed behavior returns,not automatically pure-deployment values.
Deployment evaluation imports the other pilot evaluate-final.py unchanged;only output root and own hash-verification hook are redirected.
TTC5,80actions,50s,margin.2,risk.8,alpha.3,beta0,intentfalse;no MPPI in deployment.
All initial-scene hashes match saved base per case. Regression before training matches original base on5 cases per scene.
Known checkpoint-compatible input semantics retained;no reward,input/action/dynamics repair in this task.
Single training seed;not fresh confirmation;no post-hoc pass gate,parameter change or checkpoint selection.
Earlier smoke assertion failure and completed smoke remain intact; no source-project edits or other process control.

各臂Success time是各自成功子集的均值，不是共同成功案例的配对效率估计。Base推理毫秒来自历史测量，R3来自本地3060；两列不是同硬件计时对照，不据此声称加速。

第3轮因容器缺少sklearn而在任何训练前停止，完整证据在attempt-formal1/。第4轮用户修复scikit-learn==1.7.2，已核验pip freeze只新增该一行，随后复跑原冻结流水线。算法和预算未改。原母体10例回归逐例一致，远端数值差异备选未启用；训练和最终评测均在本地完成，没有4090占用。授权记录见round4-authorization.json，前次冒烟报告见RESULT-smoke.md，自动统计初稿见RESULT-generated.md。

## Paths

formal/R3/final.pth; formal/R3/checkpoints.jsonl; formal/R3/trajectories.jsonl; formal/R3/planner.jsonl;
evaluation/R3.jsonl; evaluation/regression.jsonl; results.json; formal-protocol.json; formal-manifest.json;
controller-host-pid.json; training-pid.json; progress-R3.json; progress-eval-R3.json; exit-*.json.
All final training/evaluation/controller subprocesses exited0.
