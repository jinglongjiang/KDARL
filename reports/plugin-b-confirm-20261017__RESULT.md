未知：冻结风险触发未满足至少4/6格双对照确认门槛；逐格结果如下，不把未通过写成理论无效。

# Plugin B amplification confirmation

【研究结论】已完成冻结确认。未建立独立算法贡献层面的方法级进展；以下裁决只针对冻结插件的任务收益。

【核心证据】30,000个正式回合；仅GRU方形同时通过两项对照门槛（1/6，要求至少4/6）。该格risk−random成功率+3.8pp [2.0,5.7]，risk−default +5.9pp [3.9,8.0]；局部收益成立，不足以确认跨母体、跨场景的预注册命题。

【科学裁决】INCONCLUSIVE：同一格risk−random和risk−default均达到ΔSR≥2pp且95CI下界>0的格数为1/6。
【研究决策】KILL。不把当前冻结插件升级为论文主线，不调参救结果；保留所有逐格效应与不确定区间。
【执行结果或下一动作】本轮结束，交接完整对照结果，不自动追加算法、训练或新队列。

## Outcomes and compute

| Mother | Scene | Arm | SR/CR/TO % | Contact n/% | Calls/steps | Call % | Mean/p50/p95 ms |
|---|---|---|---|---|---|---|---|
| gru | square | 1step | 77.8/7.8/14.4 | 78/7.8 | 0/89690 | 0.00 | 15.68/15.04/19.78 |
| gru | square | default | 86.9/7.3/5.8 | 73/7.3 | 32045/74120 | 43.23 | 50.98/18.12/100.74 |
| gru | square | risk | 92.8/2.5/4.7 | 25/2.5 | 50969/70740 | 72.05 | 74.54/95.26/102.37 |
| gru | square | random | 89.0/4.0/7.0 | 39/3.9 | 53831/74758 | 72.01 | 74.41/95.49/100.90 |
| gru | square | ALWAYS | 91.1/3.4/5.5 | 35/3.5 | 68644/68644 | 100.00 | 97.34/96.51/103.41 |
| gru | circle | 1step | 92.8/2.6/4.6 | 24/2.4 | 0/71831 | 0.00 | 15.56/15.01/19.23 |
| gru | circle | default | 97.5/1.9/0.6 | 18/1.8 | 26026/68180 | 38.17 | 46.80/16.78/99.99 |
| gru | circle | risk | 98.2/1.1/0.7 | 10/1.0 | 45499/68179 | 66.73 | 70.32/95.59/101.72 |
| gru | circle | random | 97.7/1.6/0.7 | 16/1.6 | 45627/68250 | 66.85 | 70.11/95.19/100.42 |
| gru | circle | ALWAYS | 98.5/0.8/0.7 | 8/0.8 | 67498/67498 | 100.00 | 97.88/96.99/103.49 |
| kda | square | 1step | 70.3/8.5/21.2 | 86/8.6 | 0/96946 | 0.00 | 17.34/16.89/20.14 |
| kda | square | default | 86.6/7.7/5.7 | 77/7.7 | 32792/73669 | 44.51 | 54.01/19.13/102.28 |
| kda | square | risk | 90.8/3.3/5.9 | 32/3.2 | 53708/71915 | 74.68 | 79.02/98.31/103.73 |
| kda | square | random | 89.3/4.2/6.5 | 41/4.1 | 56048/75106 | 74.63 | 78.45/97.77/102.28 |
| kda | square | ALWAYS | 90.5/4.4/5.1 | 42/4.2 | 67781/67781 | 100.00 | 99.82/99.12/104.58 |
| kda | circle | 1step | 91.0/3.2/5.8 | 32/3.2 | 0/78904 | 0.00 | 17.32/16.81/20.35 |
| kda | circle | default | 97.5/1.6/0.9 | 16/1.6 | 29786/71527 | 41.64 | 51.49/18.44/101.76 |
| kda | circle | risk | 98.4/1.0/0.6 | 10/1.0 | 48101/69538 | 69.17 | 74.19/97.69/102.98 |
| kda | circle | random | 97.3/1.9/0.8 | 18/1.8 | 48280/69809 | 69.16 | 73.98/97.57/102.05 |
| kda | circle | ALWAYS | 98.3/1.1/0.6 | 11/1.1 | 67107/67107 | 100.00 | 99.59/98.94/104.29 |
| R0 | square | 1step | 88.1/4.9/7.0 | 49/4.9 | 0/76170 | 0.00 | 15.72/15.09/19.82 |
| R0 | square | default | 90.6/4.8/4.6 | 48/4.8 | 29041/71098 | 40.85 | 48.95/17.68/100.34 |
| R0 | square | risk | 92.3/2.7/5.0 | 27/2.7 | 48060/69900 | 68.76 | 71.87/95.17/102.15 |
| R0 | square | random | 90.8/4.8/4.4 | 48/4.8 | 48298/70362 | 68.64 | 71.64/95.27/100.89 |
| R0 | square | ALWAYS | 91.8/2.9/5.3 | 29/2.9 | 68960/68960 | 100.00 | 97.44/96.50/103.66 |
| R0 | circle | 1step | 97.4/1.6/1.0 | 16/1.6 | 0/66988 | 0.00 | 15.49/14.95/19.17 |
| R0 | circle | default | 97.5/1.9/0.6 | 19/1.9 | 26684/67506 | 39.53 | 47.74/17.02/99.73 |
| R0 | circle | risk | 98.4/0.8/0.8 | 8/0.8 | 42364/67543 | 62.72 | 66.85/94.76/101.48 |
| R0 | circle | random | 98.3/1.0/0.7 | 10/1.0 | 42333/67378 | 62.83 | 66.82/94.81/100.31 |
| R0 | circle | ALWAYS | 98.6/0.7/0.7 | 7/0.7 | 67370/67370 | 100.00 | 97.57/96.74/103.41 |

## Paired differences

| Mother | Scene | New−old | ΔSR [95CI] pp | ΔCR [95CI] pp | ΔTO [95CI] pp | Δcontact [95CI] pp | Rescue/damage |
|---|---|---|---|---|---|---|---|
| gru | square | risk-random | +3.8 [+2.0,+5.7] | -1.5 [-2.8,-0.2] | -2.3 [-3.7,-1.0] | -1.4 [-2.7,-0.1] | 65/27 |
| gru | square | risk-default | +5.9 [+3.9,+8.0] | -4.8 [-6.5,-3.2] | -1.1 [-2.5,+0.2] | -4.8 [-6.5,-3.2] | 87/28 |
| gru | square | risk-1step | +15.0 [+12.4,+17.7] | -5.3 [-7.0,-3.7] | -9.7 [-11.8,-7.6] | -5.3 [-7.0,-3.7] | 177/27 |
| gru | square | ALWAYS-risk | -1.7 [-3.4,+0.0] | +0.9 [-0.2,+2.1] | +0.8 [-0.5,+2.1] | +1.0 [-0.1,+2.2] | 30/47 |
| gru | circle | risk-random | +0.5 [-0.3,+1.3] | -0.5 [-1.3,+0.2] | +0.0 [-0.3,+0.3] | -0.6 [-1.4,+0.1] | 11/6 |
| gru | circle | risk-default | +0.7 [-0.1,+1.5] | -0.8 [-1.6,-0.1] | +0.1 [+0.0,+0.3] | -0.8 [-1.6,+0.0] | 12/5 |
| gru | circle | risk-1step | +5.4 [+3.9,+7.0] | -1.5 [-2.5,-0.6] | -3.9 [-5.2,-2.7] | -1.4 [-2.3,-0.5] | 59/5 |
| gru | circle | ALWAYS-risk | +0.3 [-0.4,+1.0] | -0.3 [-1.0,+0.4] | +0.0 [-0.3,+0.3] | -0.2 [-0.8,+0.4] | 8/5 |
| kda | square | risk-random | +1.5 [-0.6,+3.5] | -0.9 [-2.3,+0.5] | -0.6 [-2.2,+1.0] | -0.9 [-2.3,+0.5] | 62/47 |
| kda | square | risk-default | +4.2 [+2.1,+6.3] | -4.4 [-6.1,-2.7] | +0.2 [-1.2,+1.6] | -4.5 [-6.2,-2.8] | 83/41 |
| kda | square | risk-1step | +20.5 [+17.5,+23.5] | -5.2 [-7.0,-3.5] | -15.3 [-18.0,-12.7] | -5.4 [-7.2,-3.6] | 240/35 |
| kda | square | ALWAYS-risk | -0.3 [-2.2,+1.7] | +1.1 [-0.3,+2.5] | -0.8 [-2.2,+0.6] | +1.0 [-0.4,+2.5] | 48/51 |
| kda | circle | risk-random | +1.1 [+0.2,+2.1] | -0.9 [-1.8,+0.0] | -0.2 [-0.5,+0.0] | -0.8 [-1.7,+0.1] | 17/6 |
| kda | circle | risk-default | +0.9 [+0.0,+1.9] | -0.6 [-1.5,+0.3] | -0.3 [-0.7,+0.0] | -0.6 [-1.5,+0.3] | 16/7 |
| kda | circle | risk-1step | +7.4 [+5.7,+9.2] | -2.2 [-3.4,-1.1] | -5.2 [-6.6,-3.9] | -2.2 [-3.4,-1.1] | 81/7 |
| kda | circle | ALWAYS-risk | -0.1 [-0.8,+0.7] | +0.1 [-0.6,+0.8] | +0.0 [-0.3,+0.3] | +0.1 [-0.6,+0.8] | 7/8 |
| R0 | square | risk-random | +1.5 [-0.4,+3.5] | -2.1 [-3.5,-0.7] | +0.6 [-0.8,+2.0] | -2.1 [-3.5,-0.7] | 56/41 |
| R0 | square | risk-default | +1.7 [-0.1,+3.4] | -2.1 [-3.3,-1.0] | +0.4 [-1.0,+1.8] | -2.1 [-3.3,-1.0] | 48/31 |
| R0 | square | risk-1step | +4.2 [+2.1,+6.4] | -2.2 [-3.4,-1.1] | -2.0 [-3.9,-0.2] | -2.2 [-3.4,-1.1] | 82/40 |
| R0 | square | ALWAYS-risk | -0.5 [-2.2,+1.1] | +0.2 [-0.9,+1.3] | +0.3 [-1.0,+1.6] | +0.2 [-0.9,+1.3] | 34/39 |
| R0 | circle | risk-random | +0.1 [-0.6,+0.8] | -0.2 [-0.9,+0.5] | +0.1 [+0.0,+0.3] | -0.2 [-0.9,+0.5] | 7/6 |
| R0 | circle | risk-default | +0.9 [+0.1,+1.7] | -1.1 [-1.9,-0.4] | +0.2 [+0.0,+0.5] | -1.1 [-1.9,-0.4] | 13/4 |
| R0 | circle | risk-1step | +1.0 [+0.2,+1.9] | -0.8 [-1.6,-0.1] | -0.2 [-0.6,+0.2] | -0.8 [-1.6,-0.1] | 15/5 |
| R0 | circle | ALWAYS-risk | +0.2 [-0.5,+0.9] | -0.1 [-0.7,+0.5] | -0.1 [-0.5,+0.2] | -0.1 [-0.7,+0.5] | 7/5 |

## Calls versus ALWAYS gain

| Mother | Scene | Risk/ALWAYS calls % | Risk/ALWAYS gain % | ALWAYS gain pp | Risk gain pp | Random/risk calls | Matched |
|---|---|---|---|---|---|---|---|
| gru | square | 74.3 | 112.8 | +13.3 | +15.0 | 1.0562 | True |
| gru | circle | 67.4 | 94.7 | +5.7 | +5.4 | 1.0028 | True |
| kda | square | 79.2 | 101.5 | +20.2 | +20.5 | 1.0436 | True |
| kda | circle | 71.7 | 101.4 | +7.3 | +7.4 | 1.0037 | True |
| R0 | square | 69.7 | 113.5 | +3.7 | +4.2 | 1.0050 | True |
| R0 | circle | 62.9 | 83.3 | +1.2 | +1.0 | 0.9993 | True |

## Navigation efficiency

Success-only mean seconds/metres, conditional on each arm succeeding; not a common-success causal efficiency effect.
| Mother | Scene | Arm | Success mean seconds | Success mean path m |
|---|---|---|---|---|
| gru | square | 1step | 19.05880462724936 | 14.157302875807062 |
| gru | square | default | 17.588895281933258 | 13.45996174918428 |
| gru | square | risk | 16.41029094827586 | 12.845142379021691 |
| gru | square | random | 16.866011235955057 | 12.773488958048148 |
| gru | square | ALWAYS | 15.663830954994511 | 12.421155982729841 |
| gru | circle | 1step | 16.626616379310345 | 12.287657158315334 |
| gru | circle | default | 17.022051282051283 | 12.841816412150518 |
| gru | circle | risk | 16.938645621181262 | 12.925642461407774 |
| gru | circle | random | 16.994114636642784 | 12.498158785600275 |
| gru | circle | ALWAYS | 16.72005076142132 | 12.91490012415849 |
| kda | square | 1step | 18.806899004267425 | 14.07096562014252 |
| kda | square | default | 17.588337182448036 | 13.429628542877095 |
| kda | square | risk | 16.367015418502202 | 12.821708158653076 |
| kda | square | random | 17.1993281075028 | 13.065198233010225 |
| kda | square | ALWAYS | 15.690883977900553 | 12.445198463800859 |
| kda | circle | 1step | 18.13736263736264 | 13.623830851163099 |
| kda | circle | default | 17.72897435897436 | 13.409454833184613 |
| kda | circle | risk | 17.29268292682927 | 13.194824844582827 |
| kda | circle | random | 17.35688591983556 | 12.922174456381903 |
| kda | circle | ALWAYS | 16.683621566632755 | 12.893585079238369 |
| R0 | square | 1step | 17.393303064699204 | 13.04573303487296 |
| R0 | square | default | 16.875551876379692 | 13.053102555330053 |
| R0 | square | risk | 16.121614301191766 | 12.70110876042108 |
| R0 | square | random | 16.740088105726873 | 12.676818064990627 |
| R0 | square | ALWAYS | 15.76062091503268 | 12.463585293436699 |
| R0 | circle | 1step | 16.509496919917865 | 12.243119391586703 |
| R0 | circle | default | 16.808205128205127 | 12.709527902354614 |
| R0 | circle | risk | 16.692581300813007 | 12.782433199017024 |
| R0 | circle | random | 16.696592065106817 | 12.324746985942125 |
| R0 | circle | ALWAYS | 16.68559837728195 | 12.890920537346483 |

## Evidence, scope, and preserved failures

Cases370000–370999,1000/cell,6cells,5arms; no training. Source/checkpoints and protocol pinned before new outcomes inmanifest-confirm.json. All rules remain frozenT_r=2s,d_r=.1m; original MPPI16steps/256samples/free-space terminal,80actions,TTC5,alpha.3,50s.
Main gate is joint within-cell, not a pooled test. Separate passes:{"risk-random": [["gru", "square"]], "risk-default": [["gru", "square"], ["kda", "square"]]}; joint:[["gru", "square"]]. Bootstrap10000 paired whole cases, nominal95% percentile intervals, no multiplicity correction. Shared worlds across checkpoints are not independent cohorts; R0 andGRU are related checkpoints, not independent architecture replications.
Random p comes from corresponding full1000 risk calls/steps only. Smoke random uses5case rates and is excluded from formal counts. Formal random begins only after rates-formal-freeze.json. No outcome-based recalibration. Budget match is per-step fraction within2pp plus total-call ratiowithin.9–1.1. Unmatched cells do not establish equal-budget mechanism attribution. Temporal clustering and exact compute are not matched.
Latency is pooled original decision-to-CUDA-sync time; environment stepping excluded. Main four-arm order rotates within case, but random runs later under potentially different external load. Do not claim an equal-load causal latency speedup. Zero-event bootstrap bounds are descriptive, not a proof of safety noninferiority. SR gate does not override collision/contact regressions.
Risk/ALWAYS gain uses(SR_risk−SR_1step)/(SR_ALWAYS−SR_1step). Negative or>100% ratios are retained; denominator<=0 isundefined. Point ratios only, no stable ratio interval claimed.
No forbidden block was simulated. Old311000 records were read-only SHA/count/world-alignment checked:old-record-regression.json. On new first5cases, frozen legacy evaluator adapter vs new wrapper was compared for allfour main arms, including full action/selection hashes and planner counters. This is not a claim of rerunning forbidden old cases.
Provenance:audit-remote-370000.json, audit-clearance.json, yi-14-authorization.json. Remote case hits0;1004 interval candidates semantically classified as path dates, lists/counts or non-range function arguments. No unresolved read errors. Local clearance supplied byCC under yi-14, not rerun. Four malformedJSON records used lexical case-field fallback with no relevant hits; retained/deleted/binary-only scope limits are explicit.
yi-13 failures remain:yi-13-stopped-result.md,failure-audit.json,failure-archive.json. Its arbitrary-number audit is withdrawn, not provenance clearance. preparation-review.json records two expected unclassified-filename exits during pre-freeze audit review; no navigation was run then.
Recompute:python statistics-confirm.py; source verification occurs first. Raw results:episodes/, scheduler rate freeze:rates-formal*.json, exact rescue/damage IDs:statistics-confirm.json. No new experiment is started by recomputing statistics. Final post-run audit and delivery receipt are recorded separately.

## yi-15 completion and engineering record

- 已验证：按yi-15补齐100份旧KDA ALWAYS参照，未更改参照内容。归档SHA256为00a3bb28f3b35e8651dff43d32f350806a24cd158158dc4da2a454104500e41f；解包前核对全部成员路径、类型及原SHA，解包后逐文件复验。证据：dependency-restoration-yi15.json、deployment-preflight-yi15.json、yi-15-authorization.json。此后未发生新的部署依赖故障。
- 已验证：运行期预检加载全部三个权重；120个新块主臂回合与120个冻结评测器参照回合逐例一致，含结局、步数、动作/选中动作哈希、路径、回报、接触及规划器计数。另有12个随机端点控制和30个随机冒烟回合全部通过。冒烟/参照不重复计入正式样本。
- 已验证：正式主队列共24,000回合，其中120个为通过回归的同块前5例，另23,880个为余下案例；正式随机臂6,000回合。总正式30,000，总实际记录30,162；2,157,467个正式决策步，随机臂425,663步。非正式162例仅作控制检查。
- 已验证：控制进程于2026-10-17 16:32:50 KST启动，23:15:27 KST结束，墙钟6.710小时。正式主阶段18,250.99秒，正式随机阶段5,631.30秒；含控制/预检的各阶段退出码均为0，最终独立复算退出码0。
- 已验证：post-run-verification.json复核原始源码及权重SHA、全部30格完整案例ID、每步时延/调用计数、随机调度位、初始人群哈希、原配对区间和救回/破坏ID；owned_active_jobs为空。本任务评测已释放资源；不停止或修改其他会话的进程。
- 原始冻结清单SHA256：16345933fbd088b052528a206ce3d5d7abf3911395b3a29e61c5d128d3d1616b。正式随机概率冻结SHA256：770044fcdf7d4215ad371bf8910ac00bc04972a0a3ed12ece046b391277b9388。协议、规则、阈值及主判据未修改。
- 历史工程失败全部保留：yi-13-stopped-result.md、yi-14-stopped-result.md、failure-audit.json、failure-archive.json、failure-confirm-deployment.json，以及preparation-review.json。它们不是科学负例；未混入正式样本。
- 收尾工程错误：复制交付回执delivery-yi15.json时，检查先于rsync退出，远端SHA命令报文件不存在（退出码1）。等待传输明确退出0后，两端SHA已核对一致；本条记录及报告更新随后纳入重新生成的最终交付回执。证据：failure-delivery-receipt-race.json。仅涉及交付元数据，不影响导航数据或裁决；没有因此重跑实验。

## Supplementary compute uncertainty

以下为完成后只读复算，不改主判据。按完整case配对bootstrap 10,000次，统计每步加权平均时延之差和调用步占比之差；p50/p95仍见原表。附加分析说明在正式主阶段期间写入post-analysis-spec.json，不声称它早于全部实验结局预注册。由于随机臂后运行、外部负载可能不同，区间是描述性而非等负载因果速度比较。

| Mother | Scene | risk−random mean latency ms [95CI] | risk−random call fraction pp [95CI] |
|---|---|---|---|
| gru | square | +0.13 [-0.91,+1.09] | +0.04 [-1.20,+1.21] |
| gru | circle | +0.21 [-0.66,+1.06] | -0.12 [-1.17,+0.91] |
| kda | square | +0.57 [-0.50,+1.59] | +0.06 [-1.22,+1.27] |
| kda | circle | +0.21 [-0.59,+1.00] | +0.01 [-0.96,+0.96] |
| R0 | square | +0.22 [-0.94,+1.32] | +0.11 [-1.28,+1.43] |
| R0 | circle | +0.04 [-0.88,+0.95] | -0.11 [-1.22,+1.01] |

原始完整补充统计（包括其余三种配对）在compute-paired-intervals.json。全部六格近似调用预算匹配均通过；不声称逐回合调用数、触发聚集方式或严格总算力完全相同。

## Research disposition

- 已验证：GRU方形存在超出近似等调用随机对照的局部导航收益，并伴随碰撞率下降；不撤回这条正证据。
- 未验证：该收益达到预注册的至少4/6格双对照门槛；实际只达到1/6。未建立独立算法创新，也不把未通过门槛等同于证明插件处处无效。
- 研究决策KILL：停止将本冻结触发器升级为第二篇论文主方法，不调阈值、不扩场景挑赢家；保留为已测实用基线。唯一下一动作是交接本轮冻结结果，由总调度据此决定论文主线；本会话不追加实验。
- 复算入口：statistics-confirm.py复算原统计，verify-confirm.py独立校验并生成补充计算区间；前者会重建机器生成的RESULT.md，手写的本完成附注以交付哈希版本为准。全部文件本地/远端一致性见delivery-yi15.json。
