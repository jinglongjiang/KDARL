# Branch-restart pilot round2

Status: VERIFIED. Frozen primary criterion: FAIL.
状态：已验证（实验完整有效；主判据未通过）。
This is a single-seed training pilot, not a publication-level or fresh-confirmation result.

本轮没有取得方法级进展。20S上，结局敏感重启R2没有超过普通续训R0或失败均匀重启R1；相对原母体的提升不能归因于敏感根筛选。按冻结规则封存本版本，不追加调参或训练。20C的R2点估计较好，但相对R0/R1的区间均跨零，不能替代主判据。

## Training

| Arm | Accounted steps | Training / discovery / restore / check | Trajectories | Updates | Restarts | Pool | Hours |
|---|---:|---|---:|---:|---:|---:|---:|
| R0 | 402000 | 401972 / 0 / 0 / 28 | 5529 | 7996 | 0 | 0 | 2.43 |
| R1 | 402000 | 193873 / 0 / 208099 / 28 | 3112 | 7996 | 1528 | 2000 | 1.94 |
| R2 | 402000 | 258698 / 82884 / 60390 / 28 | 3455 | 7996 | 1739 | 61 | 2.22 |

Restart counters include attempts: R1 has1528 attempts and1527 completed restart training suffixes; R2 has1739 attempts and1739 completed suffixes. All unfinished suffixes were discarded without MC labels. Labeled training steps: R0=401932, R1=193780, R2=258661. The remaining charged training steps belong to unfinished final suffixes.

R2 admitted61 sensitive roots:24 rescuable failures and37 fragile successes. Both mechanisms actually activated; the negative primary result is not an unexecuted-module result. Historical branch outcomes were used only to select restart locations, never as value-training targets.

## Evaluation

| Scene | Arm | Success / collision / timeout | SR / CR / TO (%) | Contact | Success time (s) | Inference ms |
|---|---|---|---|---:|---:|---:|
| square | base | 404 / 29 / 67 | 80.80 / 5.80 / 13.40 | 28 | 18.67 | 15.20 |
| square | R0 | 437 / 31 / 32 | 87.40 / 6.20 / 6.40 | 29 | 17.41 | 15.19 |
| square | R1 | 434 / 28 / 38 | 86.80 / 5.60 / 7.60 | 26 | 18.79 | 15.28 |
| square | R2 | 433 / 30 / 37 | 86.60 / 6.00 / 7.40 | 30 | 18.39 | 15.23 |
| circle | base | 481 / 10 / 9 | 96.20 / 2.00 / 1.80 | 9 | 16.86 | 15.15 |
| circle | R0 | 489 / 11 / 0 | 97.80 / 2.20 / 0.00 | 10 | 16.23 | 15.18 |
| circle | R1 | 488 / 10 / 2 | 97.60 / 2.00 / 0.40 | 10 | 17.33 | 15.06 |
| circle | R2 | 494 / 6 / 0 | 98.80 / 1.20 / 0.00 | 7 | 16.70 | 15.11 |

## Paired differences

| Scene | Comparison | SR pp [95% CI] | CR pp [95% CI] | TO pp [95% CI] | Rescue / damage |
|---|---|---|---|---|---:|
| square | R0-base | 6.60 [2.80, 10.60] | 0.40 [-2.20, 3.00] | -7.00 [-10.00, -4.00] | 66 / 33 |
| square | R1-base | 6.00 [2.00, 10.00] | -0.20 [-2.80, 2.40] | -5.80 [-9.00, -2.60] | 67 / 37 |
| square | R2-base | 5.80 [1.80, 9.80] | 0.20 [-2.40, 3.00] | -6.00 [-9.40, -2.80] | 67 / 38 |
| square | R1-R0 | -0.60 [-4.00, 2.80] | -0.60 [-3.20, 2.00] | 1.20 [-1.20, 3.80] | 37 / 40 |
| square | R2-R0 | -0.80 [-4.40, 2.60] | -0.20 [-2.60, 2.20] | 1.00 [-1.60, 3.60] | 36 / 40 |
| square | R2-R1 | -0.20 [-4.00, 3.40] | 0.40 [-2.20, 3.00] | -0.20 [-3.00, 2.80] | 45 / 46 |
| circle | R0-base | 1.60 [-0.20, 3.40] | 0.20 [-1.20, 1.60] | -1.80 [-3.00, -0.80] | 15 / 7 |
| circle | R1-base | 1.40 [-0.60, 3.40] | 0.00 [-1.60, 1.80] | -1.40 [-2.80, -0.20] | 17 / 10 |
| circle | R2-base | 2.60 [0.80, 4.40] | -0.80 [-2.20, 0.60] | -1.80 [-3.00, -0.80] | 17 / 4 |
| circle | R1-R0 | -0.20 [-1.80, 1.40] | -0.20 [-1.60, 1.20] | 0.40 [0.00, 1.00] | 7 / 8 |
| circle | R2-R0 | 1.00 [-0.40, 2.40] | -1.00 [-2.40, 0.40] | 0.00 [0.00, 0.00] | 9 / 4 |
| circle | R2-R1 | 1.20 [-0.40, 2.80] | -0.80 [-2.20, 0.60] | -0.40 [-1.00, 0.00] | 11 / 5 |

## Frozen decision

20S R2-R0 = -0.80pp [-4.40, 2.60]; R2-R1 = -0.20pp.
Scientific verdict: REFUTED. Research decision: KILL.
R0-base is reported separately above; it must not be attributed to sensitive-root sampling.

Scope of the scientific verdict: the frozen, single-seed pilot did not meet its pre-registered20S signal criterion. Its R2-R0 interval upper bound2.60pp is also below the required3.0pp. This does not prove that all possible branch-restart methods lack value; arbitrary small effects and training-seed generalization remain unestablished. The research decision is nevertheless KILL for this version, with no automatic rescue or fresh confirmation.

R0-base measures the total effect of this common20human continuation-training protocol, not an isolated causal effect of crowd size. All arms used the same deployment-aligned safety/risk/smoothing collector with epsilon0.05, resumed the identical model and AdamW state, started with empty replay, and shared the budget-normalized MC update schedule. These conditions differ from historical train-phase behavior and are disclosed in protocol.json.

## MPPI reference and recovered headroom

| Scene | MPPI / fixed8 SR (%) | R2 recovered MPPI / fixed8 headroom (%) |
|---|---|---|
| square | 89.80 / 88.40 | 64.44 / 76.32 |
| circle | 98.20 / 98.40 | 130.00 / 118.18 |

冻结MPPI的终端项是到目标距离的自由空间启发值，不是价值网络。

Read-only source verification: recovery-duration-20261011/research/progress-contract-v2/controller.py:18-28, free_value() and terminal_value_cost(); protocol.json, one_structural_correction and planner.objective. The terminal term is -0.99^16 times V_free(d), where V_free(d)=success_reward*0.99^(max(d-robot_radius,0)/(v_pref*0.25)), zeroed after predicted termination. It never queries the GRU value network. Both ordinary MPPI and fixed8 use this same heuristic.

The recovered-headroom percentages are descriptive differences relative to the old base, including ordinary continuation-training gains. They are not gains attributable specifically to R2, and values above100% do not establish statistical superiority over MPPI. No paired MPPI comparison was computed in this task.

## Evidence and limits

protocol.json; manifest.json; external-gate.json; preflight-R*.json; evaluation/regression.jsonl;
formal/R*/training-trajectories.jsonl; formal/R*/result.json; formal/R*/checkpoints.jsonl;
evaluation/R*.jsonl; results.json; exit-formal-R*.json; exit-eval-R*.json.
All environment calls including discovery and restoration are charged. First2000 preflight steps are included.
The MC suffix uses the current collection policy; old branch outcomes only select restart locations.
No algorithmic novelty or generally superior navigation is established by a pilot pass alone.
The neighbor gate29/32 was user supplied, not independently recomputed here.
Native50s contract and existing checkpoint-compatible input semantics were preserved.
No Bayesian, old recovery or MPPI jobs were restarted or modified.
Exit codes: all training/evaluation arms0.

Zero-event intervals such as circle R2-R0 timeout[0,0] are degenerate empirical bootstrap results, not proof of zero population risk. No safety noninferiority claim is made from those intervals.

## Engineering history and final verification

Round2 was blocked before any remote training by sandbox-denied rsync/scp. The user restored permissions in round3; the missing original checkpoint was restored and all487 frozen hash entries matched before launching. The earlier blocked report and engineering records remain in round2-blocked-result.md and round2-engineering.json. No invalid run was counted as a scientific observation.

The original parent regression matched all10 episodes on outcome, steps, path, return, executed-action/selection hashes, initial world and swept clearance. Each live arm paused exactly at2000 accounted steps and passed restored-state parity, current-weight history rebuild, optimizer-update and budget checks. Those2000 steps are inside402000, not repeated extra work.

After completion, all15 checkpoint files matched their saved SHA256 records; the local487-entry frozen dependency check passed again. All3000 evaluated case keys and complete final-checkpoint outputs were preserved. All36 paired SR/CR/TO point estimates and bootstrap intervals were independently recomputed from the transferred episode records and matched; each success difference also matched(rescues-damages)/500.

Code, final/intermediate weights, training/discovery logs, exit records and evaluation results were synchronized to the local task directory. The local3060 was not used. Another session's MPPI processes and directory were not modified. Formal training, all evaluation arms, regression, statistics and controller exited0.
