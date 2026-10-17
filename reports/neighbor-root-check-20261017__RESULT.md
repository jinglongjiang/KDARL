# Neighbor-root check

## 最终结果

状态：已验证，通过事先冻结的门槛。

- 邻近位置仍能挽救：29/32（90.625%），超过20/32门槛。
- GRU seed419：15/16；GRU seed443：14/16，均超过8/16门槛。
- 四个偏移的挽救数：−2步19/32、−1步18/32、+1步22/32、+2步20/32；各列可重叠。
- 成功对照在任一邻近位置被替代动作损伤：1/32（3.125%）。
- 原根已被损伤的对照只有1例，其损伤未在邻近位置保留（0/1）；邻近位置的1例损伤来自另一例对照。
- 768个替代槽位全部完成登记：683个可用、85个缺失；缺失全部留在分母中。
- 去除重复动作及A0等价动作后，实际执行495条不同的替代续行；没有把768槽位冒充768条不同轨迹。
- 256个位置中，231个位置存在且A0回归通过；25个位置超出原回合边界，记为不可用，而非宣称回归通过。
- 冒烟与全量累计运行1034.34秒，约17分14秒，不含准备和最终分析。
- 冒烟、全量、汇总及独立复算退出码均为0；347个冻结输入文件及脚本哈希复验通过。

结论限定：在本轮按原挽救结果筛选的32例失败、两份GRU权重及前后0.5秒范围内，
可挽救分支多数不是仅在原根这个精确时刻存在。该结果不等于在线可识别这些时刻或动作，
也不是所有失败的可挽救率；本轮没有方法级进展。

Status: 已验证. Gate: PASS. No network training; no method-level progress.

## Prespecified Decision

Retained rescue >=60% of 32 failed episodes AND >=50% for each mother. Any of three preset alternatives at any of four offsets is an existence upper bound, not a deployable policy. Missing alternatives remain in all denominators.

Completed alternative slots: 768/768; A0 checks: 256/256.
Neighbor rescue retained: 29/32 (90.62%).
Success controls damaged at any neighbor: 1/32.

| Mother | Original rescued failures | Retained | Fraction |
|---|---:|---:|---:|
| gru-s419-t24 | 16 | 15 | 93.75% |
| gru-s443-t24 | 16 | 14 | 87.50% |

## Geometry and Root Type

| Mother | Cell | Root type | n | Neighbor retained/damaged |
|---|---:|---|---:|---:|
| gru-s419-t24 | 4 | collision | 4 | 4 |
| gru-s419-t24 | 4 | timeout | 4 | 4 |
| gru-s419-t24 | 4 | success-control | 8 | 0 |
| gru-s419-t24 | 5 | collision | 4 | 3 |
| gru-s419-t24 | 5 | timeout | 4 | 4 |
| gru-s419-t24 | 5 | success-control | 8 | 0 |
| gru-s443-t24 | 4 | collision | 4 | 3 |
| gru-s443-t24 | 4 | timeout | 4 | 4 |
| gru-s443-t24 | 4 | success-control | 8 | 0 |
| gru-s443-t24 | 5 | collision | 4 | 4 |
| gru-s443-t24 | 5 | timeout | 4 | 3 |
| gru-s443-t24 | 5 | success-control | 8 | 1 |

## Offset and Action Rule

| Offset steps (seconds) | Label | Failure roots rescued /32 | Controls damaged /32 | Missing /64 |
|---|---|---:|---:|---:|
| -2 (-0.50) | A1 | 3 | 0 | 14 |
| -2 (-0.50) | A2 | 17 | 1 | 13 |
| -2 (-0.50) | A3 | 13 | 0 | 13 |
| -2 (-0.50) | any | 19 | 1 | 13 |
| -1 (-0.25) | A1 | 3 | 0 | 12 |
| -1 (-0.25) | A2 | 12 | 1 | 12 |
| -1 (-0.25) | A3 | 12 | 0 | 12 |
| -1 (-0.25) | any | 18 | 1 | 12 |
| +1 (+0.25) | A1 | 6 | 0 | 2 |
| +1 (+0.25) | A2 | 15 | 0 | 0 |
| +1 (+0.25) | A3 | 14 | 0 | 1 |
| +1 (+0.25) | any | 22 | 0 | 0 |
| +2 (+0.50) | A1 | 7 | 0 | 3 |
| +2 (+0.50) | A2 | 17 | 0 | 1 |
| +2 (+0.50) | A3 | 11 | 1 | 2 |
| +2 (+0.50) | any | 20 | 1 | 1 |

## Successful-Control Protection

Originally harmed controls: 1/32; damage persists in neighbors: 0/1.
Newly damaged controls: 1.
Environment-success branches invalidated by swept collision: 0.

## Per-Episode Evidence

| Root | Mother | Cell | Case | Original step | Type | Neighbor result | Evidence offsets/labels |
|---:|---|---:|---:|---:|---|---|---|
| 937 | gru-s419-t24 | 4 | 90156 | 40 | timeout | True | -2/A2:19, -2/A3:24, -1/A2:19, +2/A2:49, +2/A3:64 |
| 947 | gru-s419-t24 | 4 | 90215 | 17 | collision | True | -2/A2:19, -2/A3:39, -1/A3:39, +1/A2:9, +2/A2:24, +2/A3:18 |
| 951 | gru-s419-t24 | 4 | 90222 | 28 | collision | True | -2/A1:21, -2/A2:29, -2/A3:54, -1/A3:12, +1/A2:24, +2/A2:18, +2/A3:21 |
| 956 | gru-s419-t24 | 4 | 90239 | 34 | timeout | True | -1/A3:25 |
| 970 | gru-s419-t24 | 4 | 90325 | 46 | timeout | True | -2/A2:9, +1/A2:4, +2/A1:14, +2/A2:79 |
| 975 | gru-s419-t24 | 4 | 90343 | 0 | collision | True | +1/A2:24, +1/A3:59, +2/A2:19 |
| 981 | gru-s419-t24 | 4 | 90369 | 17 | collision | True | -2/A2:19, -1/A3:24, +1/A3:7 |
| 984 | gru-s419-t24 | 4 | 90385 | 50 | timeout | True | -2/A1:79, -2/A2:69, -2/A3:19, -1/A1:69, -1/A2:64, -1/A3:9, +1/A1:64, +1/A2:64, +2/A1:64, +2/A2:59 |
| 504 | gru-s419-t24 | 5 | 90009 | 0 | collision | True | +1/A3:8 |
| 527 | gru-s419-t24 | 5 | 90071 | 0 | collision | True | +1/A1:33, +1/A2:24, +1/A3:39 |
| 572 | gru-s419-t24 | 5 | 90181 | 0 | collision | True | +1/A3:34, +2/A1:51, +2/A2:24, +2/A3:56 |
| 619 | gru-s419-t24 | 5 | 90292 | 0 | collision | False | - |
| 620 | gru-s419-t24 | 5 | 90294 | 20 | timeout | True | -2/A2:24, -1/A1:32, -1/A2:24, +1/A1:39, +2/A1:39, +2/A2:24 |
| 661 | gru-s419-t24 | 5 | 90439 | 45 | timeout | True | -1/A2:37, +1/A1:26, +1/A2:26, +2/A2:17 |
| 671 | gru-s419-t24 | 5 | 90471 | 41 | timeout | True | -2/A3:4, -1/A2:24, +1/A3:24 |
| 675 | gru-s419-t24 | 5 | 90480 | 22 | timeout | True | -2/A2:48, -2/A3:53, +2/A1:13, +2/A3:59 |
| 1006 | gru-s443-t24 | 4 | 90002 | 27 | collision | True | -2/A2:24, -2/A3:36, -1/A2:24, -1/A3:28, +1/A2:24, +1/A3:37, +2/A3:34 |
| 1010 | gru-s443-t24 | 4 | 90010 | 28 | collision | False | - |
| 1011 | gru-s443-t24 | 4 | 90032 | 20 | timeout | True | -2/A1:18, -2/A2:19, -1/A1:16, -1/A2:18, -1/A3:53, +1/A2:19, +1/A3:28, +2/A2:34 |
| 1032 | gru-s443-t24 | 4 | 90202 | 0 | collision | True | +1/A3:16, +2/A3:22 |
| 1033 | gru-s443-t24 | 4 | 90206 | 34 | timeout | True | -2/A2:34, -2/A3:26, -1/A2:34, +1/A1:44, +1/A2:34, +2/A1:19, +2/A2:39, +2/A3:13 |
| 1042 | gru-s443-t24 | 4 | 90290 | 22 | timeout | True | -2/A3:16, -1/A2:24, -1/A3:13, +1/A2:24, +1/A3:19, +2/A2:24, +2/A3:3 |
| 1054 | gru-s443-t24 | 4 | 90415 | 16 | collision | True | -2/A2:24, -2/A3:13, -1/A2:24, -1/A3:13 |
| 1062 | gru-s443-t24 | 4 | 90463 | 26 | timeout | True | -2/A2:19, -1/A3:24, +1/A1:33, +1/A2:19, +1/A3:31, +2/A2:19 |
| 687 | gru-s443-t24 | 5 | 90012 | 21 | timeout | False | - |
| 703 | gru-s443-t24 | 5 | 90036 | 26 | timeout | True | -2/A2:24 |
| 799 | gru-s443-t24 | 5 | 90246 | 25 | timeout | True | -1/A2:24, +1/A2:24 |
| 846 | gru-s443-t24 | 5 | 90354 | 8 | collision | True | -2/A2:24, -2/A3:14, -1/A2:24, -1/A3:22, +1/A2:24, +1/A3:26, +2/A2:24, +2/A3:27 |
| 853 | gru-s443-t24 | 5 | 90368 | 23 | collision | True | -2/A2:19, -2/A3:13, -1/A3:33, +1/A2:19, +1/A3:18, +2/A2:19, +2/A3:8 |
| 879 | gru-s443-t24 | 5 | 90428 | 27 | timeout | True | -2/A2:14, -2/A3:8 |
| 884 | gru-s443-t24 | 5 | 90446 | 0 | collision | True | +1/A3:17, +2/A2:24 |
| 893 | gru-s443-t24 | 5 | 90476 | 0 | collision | True | +2/A1:4, +2/A2:24 |
| 902 | gru-s419-t24 | 4 | 90001 | 28 | success-control | False | - |
| 910 | gru-s419-t24 | 4 | 90053 | 29 | success-control | False | - |
| 913 | gru-s419-t24 | 4 | 90064 | 28 | success-control | False | - |
| 943 | gru-s419-t24 | 4 | 90198 | 42 | success-control | False | - |
| 944 | gru-s419-t24 | 4 | 90201 | 24 | success-control | False | - |
| 989 | gru-s419-t24 | 4 | 90412 | 35 | success-control | False | - |
| 992 | gru-s419-t24 | 4 | 90438 | 37 | success-control | False | - |
| 999 | gru-s419-t24 | 4 | 90464 | 60 | success-control | False | - |
| 502 | gru-s419-t24 | 5 | 90006 | 48 | success-control | False | - |
| 503 | gru-s419-t24 | 5 | 90007 | 0 | success-control | False | - |
| 505 | gru-s419-t24 | 5 | 90011 | 0 | success-control | False | - |
| 506 | gru-s419-t24 | 5 | 90013 | 0 | success-control | False | - |
| 542 | gru-s419-t24 | 5 | 90110 | 0 | success-control | False | - |
| 555 | gru-s419-t24 | 5 | 90149 | 20 | success-control | False | - |
| 559 | gru-s419-t24 | 5 | 90154 | 31 | success-control | False | - |
| 578 | gru-s419-t24 | 5 | 90193 | 21 | success-control | False | - |
| 1007 | gru-s443-t24 | 4 | 90004 | 9 | success-control | False | - |
| 1008 | gru-s443-t24 | 4 | 90005 | 1 | success-control | False | - |
| 1024 | gru-s443-t24 | 4 | 90144 | 5 | success-control | False | - |
| 1041 | gru-s443-t24 | 4 | 90249 | 20 | success-control | False | - |
| 1047 | gru-s443-t24 | 4 | 90307 | 38 | success-control | False | - |
| 1049 | gru-s443-t24 | 4 | 90352 | 39 | success-control | False | - |
| 1059 | gru-s443-t24 | 4 | 90440 | 20 | success-control | False | - |
| 1064 | gru-s443-t24 | 4 | 90472 | 35 | success-control | False | - |
| 733 | gru-s443-t24 | 5 | 90100 | 22 | success-control | False | - |
| 739 | gru-s443-t24 | 5 | 90121 | 30 | success-control | False | - |
| 749 | gru-s443-t24 | 5 | 90143 | 21 | success-control | False | - |
| 794 | gru-s443-t24 | 5 | 90236 | 43 | success-control | False | - |
| 798 | gru-s443-t24 | 5 | 90245 | 24 | success-control | True | -2/A2:19, -1/A2:19, +2/A3:44 |
| 812 | gru-s443-t24 | 5 | 90273 | 26 | success-control | False | - |
| 865 | gru-s443-t24 | 5 | 90395 | 20 | success-control | False | - |
| 869 | gru-s443-t24 | 5 | 90403 | 20 | success-control | False | - |

## Reproduction and Limitations

Offsets are control steps: -0.5, -0.25, +0.25, +0.5 seconds. A2 follows the original contract and does not impose a minimum forward progress.
All branches reset at the same case and replay the complete original prefix; the clock is never reset at intervention, and subsequent smoothing uses the forced command.
A0 checks compare each step against archived original causal A0 data, including robot/human states, executed commands, history/token/window hashes, and actual swept clearance. Cross-device raw floating-point value equality is not claimed or required; actions, outcomes, step counts and history hashes must match exactly. Only endpoint_clearance uses the user-authorized absolute tolerance 1e-9 m; actual swept collision acceptance remains strictly clearance >=0, with no tolerance.
Sample is conditioned on original audited rescue and balances failure types; it does not estimate the prevalence of rescuability in all failures. Only two fixed checkpoints and +/-0.5s neighborhoods are covered. Successful controls are sampled from original matched controls, not newly matched to these 32 failures.
Evidence: manifest.json, frozen-references.jsonl.gz, smoke-branches.jsonl.gz, full-branches.jsonl.gz. Reproduction: neighbor-check.py prepare; smoke; full; report (fresh output directory required).
Decision: GO on the specified temporal-persistence gate; stop here, no new method or training authorized.

## Authorized Endpoint-Check Repair

The previous smoke failed on an additional bitwise endpoint-clearance assertion: 0.1971988330465565 versus 0.1971988330465564 m (difference 1.11e-16 m). It was not evidence of an action/outcome/length mismatch. The user authorized endpoint tolerance <=1e-9 m. Only this numerical assertion changed; samples, action rules, acceptance gates, swept-collision audit and all forbidden cases are unchanged.
Original failure: smoke-before-endpoint-fix.jsonl.gz. Previous manifest/script/report: *-before-endpoint-fix.*. manifest.json records the repair SHA256 and preserves the original sample list.
