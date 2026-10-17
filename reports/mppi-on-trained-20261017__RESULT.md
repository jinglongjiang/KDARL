# MPPI on trained GRU bases

状态：已验证。R0主判据未通过；按预注册操作性标签为“插件增益被训练吸收”。该标签不适用于全部底座。
R0 square mpc-1step = 1.60pp [-1.00, 4.20].
本轮只检验已有冻结插件在更强底座上的效力，没有建立新算法贡献。本轮没有取得方法级进展。

R1、R2方形上的mpc增益分别为+3.60pp[1.00,6.40]和+3.80pp[1.60,6.20]；它们是次要格子的正向结果，不应隐藏，也不能替换事先指定的R0主判据或用于重启C1。R0+mpc相比base+mpc未建立优势：方形−0.80pp[−3.80,2.40]，圆形−1.00pp[−2.60,0.40]。

## Protocol

20humans, native50s,dt.25,TTC5,80actions,alpha.3,margin.2,risk.8,effective beta0,intentFalse,robot invisible.
All4 bases are frozen GRU T24 models; no training, parameter changes or checkpoint selection.
All4000 1step episodes reused; all8000 planning episodes newly executed on4090 with8 workers and1 CPU thread each.
Both base planning modes are checked against all2000 original frozen planning records.
MPPI remains CPU,horizon16,samples256,lambda.01; hold8 retains/revalidates an8step plan prefix.
MPPI terminal value is a free-space goal-distance heuristic, NOT the value network.

## Results

| Scene | Base | Mode | Success / collision / timeout | SR / CR / TO % | Contact | Success time s | Inference ms | Calls |
|---|---|---|---|---|---:|---:|---:|---:|
| square | base | 1step | 404 / 29 / 67 | 80.80 / 5.80 / 13.40 | 28 | 18.67 | 15.26 | 0 |
| square | base | mpc | 449 / 24 / 27 | 89.80 / 4.80 / 5.40 | 23 | 17.13 | 51.63 | 15027 |
| square | base | hold8 | 442 / 24 / 34 | 88.40 / 4.80 / 6.80 | 23 | 18.15 | 54.36 | 17534 |
| square | R0 | 1step | 437 / 31 / 32 | 87.40 / 6.20 / 6.40 | 29 | 17.41 | 15.26 | 0 |
| square | R0 | mpc | 445 / 28 / 27 | 89.00 / 5.60 / 5.40 | 27 | 16.85 | 52.52 | 15137 |
| square | R0 | hold8 | 445 / 29 / 26 | 89.00 / 5.80 / 5.20 | 28 | 17.26 | 52.14 | 15189 |
| square | R1 | 1step | 434 / 28 / 38 | 86.80 / 5.60 / 7.60 | 26 | 18.79 | 15.31 | 0 |
| square | R1 | mpc | 452 / 26 / 22 | 90.40 / 5.20 / 4.40 | 25 | 17.78 | 52.09 | 15549 |
| square | R1 | hold8 | 442 / 26 / 32 | 88.40 / 5.20 / 6.40 | 25 | 18.52 | 54.01 | 17579 |
| square | R2 | 1step | 433 / 30 / 37 | 86.60 / 6.00 / 7.40 | 30 | 18.39 | 15.27 | 0 |
| square | R2 | mpc | 452 / 24 / 24 | 90.40 / 4.80 / 4.80 | 24 | 17.09 | 51.99 | 14971 |
| square | R2 | hold8 | 444 / 26 / 30 | 88.80 / 5.20 / 6.00 | 25 | 18.18 | 54.21 | 17024 |
| circle | base | 1step | 481 / 10 / 9 | 96.20 / 2.00 / 1.80 | 9 | 16.86 | 15.17 | 0 |
| circle | base | mpc | 491 / 7 / 2 | 98.20 / 1.40 / 0.40 | 7 | 16.82 | 48.02 | 12501 |
| circle | base | hold8 | 492 / 7 / 1 | 98.40 / 1.40 / 0.20 | 7 | 17.41 | 48.77 | 13199 |
| circle | R0 | 1step | 489 / 11 / 0 | 97.80 / 2.20 / 0.00 | 10 | 16.23 | 15.19 | 0 |
| circle | R0 | mpc | 486 / 13 / 1 | 97.20 / 2.60 / 0.20 | 14 | 16.68 | 48.84 | 12569 |
| circle | R0 | hold8 | 488 / 11 / 1 | 97.60 / 2.20 / 0.20 | 11 | 17.17 | 49.07 | 13008 |
| circle | R1 | 1step | 488 / 10 / 2 | 97.60 / 2.00 / 0.40 | 10 | 17.33 | 15.07 | 0 |
| circle | R1 | mpc | 489 / 10 / 1 | 97.80 / 2.00 / 0.20 | 9 | 17.38 | 51.41 | 14302 |
| circle | R1 | hold8 | 491 / 9 / 0 | 98.20 / 1.80 / 0.00 | 10 | 17.85 | 51.43 | 14597 |
| circle | R2 | 1step | 494 / 6 / 0 | 98.80 / 1.20 / 0.00 | 7 | 16.70 | 15.12 | 0 |
| circle | R2 | mpc | 493 / 5 / 2 | 98.60 / 1.00 / 0.40 | 5 | 17.00 | 50.89 | 13852 |
| circle | R2 | hold8 | 492 / 7 / 1 | 98.40 / 1.40 / 0.20 | 7 | 17.21 | 50.46 | 13862 |

## Paired differences

| Scene | Comparison | SR pp [95% CI] | CR pp [95% CI] | TO pp [95% CI] | Contact pp [95% CI] | Rescue / damage |
|---|---|---|---|---|---|---:|
| square | base+mpc - base+1step | 9.00 [6.00, 12.20] | -1.00 [-2.20, 0.00] | -8.00 [-11.00, -5.20] | -1.00 [-2.20, 0.00] | 54 / 9 |
| square | base+hold8 - base+1step | 7.60 [4.60, 10.80] | -1.00 [-2.20, 0.00] | -6.60 [-9.80, -3.80] | -1.00 [-2.20, 0.00] | 52 / 14 |
| square | R0+mpc - R0+1step | 1.60 [-1.00, 4.20] | -0.60 [-1.60, 0.40] | -1.00 [-3.40, 1.40] | -0.40 [-1.40, 0.60] | 26 / 18 |
| square | R0+hold8 - R0+1step | 1.60 [-1.00, 4.20] | -0.40 [-1.60, 0.60] | -1.20 [-3.60, 1.20] | -0.20 [-1.20, 0.80] | 25 / 17 |
| square | R1+mpc - R1+1step | 3.60 [1.00, 6.40] | -0.40 [-1.60, 0.60] | -3.20 [-5.80, -0.80] | -0.20 [-1.20, 0.80] | 34 / 16 |
| square | R1+hold8 - R1+1step | 1.60 [-1.40, 4.60] | -0.40 [-1.60, 0.60] | -1.20 [-4.00, 1.60] | -0.20 [-1.20, 0.80] | 33 / 25 |
| square | R2+mpc - R2+1step | 3.80 [1.60, 6.20] | -1.20 [-2.20, -0.40] | -2.60 [-5.00, -0.40] | -1.20 [-2.20, -0.40] | 28 / 9 |
| square | R2+hold8 - R2+1step | 2.20 [-0.40, 5.00] | -0.80 [-2.00, 0.20] | -1.40 [-4.00, 1.00] | -1.00 [-2.20, 0.00] | 29 / 18 |
| square | R0+mpc - base+mpc | -0.80 [-3.80, 2.40] | 0.80 [-1.60, 3.20] | 0.00 [-1.80, 1.80] | 0.80 [-1.60, 3.20] | 29 / 33 |
| circle | base+mpc - base+1step | 2.00 [0.60, 3.60] | -0.60 [-1.60, 0.40] | -1.40 [-2.60, -0.40] | -0.40 [-1.40, 0.60] | 13 / 3 |
| circle | base+hold8 - base+1step | 2.20 [0.80, 3.60] | -0.60 [-1.60, 0.20] | -1.60 [-2.80, -0.60] | -0.40 [-1.20, 0.40] | 12 / 1 |
| circle | R0+mpc - R0+1step | -0.60 [-1.80, 0.60] | 0.40 [-0.60, 1.60] | 0.20 [0.00, 0.60] | 0.80 [-0.20, 2.00] | 3 / 6 |
| circle | R0+hold8 - R0+1step | -0.20 [-1.20, 0.80] | 0.00 [-1.00, 1.00] | 0.20 [0.00, 0.60] | 0.20 [-0.60, 1.00] | 3 / 4 |
| circle | R1+mpc - R1+1step | 0.20 [-1.00, 1.40] | 0.00 [-1.00, 1.00] | -0.20 [-1.00, 0.40] | -0.20 [-1.20, 0.80] | 5 / 4 |
| circle | R1+hold8 - R1+1step | 0.60 [-0.60, 1.80] | -0.20 [-1.20, 0.80] | -0.40 [-1.00, 0.00] | 0.00 [-1.20, 1.00] | 6 / 3 |
| circle | R2+mpc - R2+1step | -0.20 [-1.40, 1.00] | -0.20 [-1.20, 0.80] | 0.40 [0.00, 1.00] | -0.40 [-1.60, 0.60] | 4 / 5 |
| circle | R2+hold8 - R2+1step | -0.40 [-1.60, 0.80] | 0.20 [-1.00, 1.40] | 0.20 [0.00, 0.60] | 0.00 [-1.20, 1.20] | 4 / 6 |
| circle | R0+mpc - base+mpc | -1.00 [-2.60, 0.40] | 1.20 [-0.20, 2.80] | -0.20 [-0.60, 0.00] | 1.40 [0.00, 3.00] | 5 / 10 |

## Verdict and limits

Scientific verdict: INCONCLUSIVE. Proposition: the frozen R0 square MPPI plugin retains the registered>=3pp signal.
Research decision: KILL. No automatic new training, parameter tuning or experiment is authorized.
该研究决策仅针对本轮“R0满足注册插件保留门槛”的候选主张，不是判定所有MPPI底座无效。指定命题的科学裁决为INCONCLUSIVE：R0区间既包含零，也包含3pp以上的收益；本轮未证明保留，并非证明收益严格为零。
“插件增益被训练吸收”是用户指定的操作性标签；区间跨零不能证明零效应，也不能单独确证训练吸收的因果机制。
For a positive CI below3pp, report SMALL_POSITIVE, not the absorbed label. Negative CIs are reported as harm.
Case-level intervals do not estimate training-seed variance; exploratory cells are not multiplicity-corrected confirmation.
Zero-event bootstrap intervals[0,0] do not prove zero population risk. No safety noninferiority claim is made.
Navigation-time means condition on each arm own successes; they are not paired common-success time differences.
Cached1step timing and new8worker planning timing have different concurrent loads; no matched timing-speedup claim.
The result concerns the entire frozen plugin: low-progress triggering,16stepCV planning, all20 legally observed humans and the free-space terminal heuristic. It does not causally isolate horizon length or a value-network improvement. No parameters, inputs, safety rules or rewards were changed for the trained bases.
本轮沿用yi03的NumPy RandomState(MT19937)按场景生成配对bootstrap；旧MPPI报告使用default_rng(PCG64)。同一数值seed不产生相同重采样序列，因此旧报告的部分区间端点会有0.2pp级蒙特卡罗差异；本轮2000条base规划原始结局和轨迹已逐例完全匹配，这不是导航结果漂移。
C1 remains killed. Loading its frozen models as evaluation substrates does not restart or rescue C1.

## Evidence

protocol.json; manifest.json; smoke-complete.json; formal-complete.json; references.json;
episodes/{smoke,formal}/; diagnostics/; statistics.json; exit-*.json; controller.log.
Smoke covers24 cells*5 episodes=120. No old directory was modified. All parameters/source hashes stayed fixed.
Actual outcome0/1 pairs and rescue/damage case IDs are retained in statistics.json.

Post-run verification is reproducible with verify-results.py:1366 frozen hashes,8000 formal case keys,24 outcome cells,2000 original base planning traces and72 paired SR/CR/TO/contact point estimates and intervals. CPU-only statistics are recomputed from the transferred records, not rerun simulations.

The4090 workers and controller have exited; GPU check after completion reported3MiB and0% utilization. The server is released for the next scheduled task. All code and experiment records were synchronized locally. C1/strongmother source directories were not edited.
