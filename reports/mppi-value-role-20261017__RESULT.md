# Value Role in Frozen ALWAYS MPPI

Status: 已验证. No method-level contribution established.

# Frozen ALWAYS Value Role Audit

Status: source audit verified; runtime and outcome claims require unit/smoke/full evidence.

## Actual Dependencies

Source paths below are under /home/abc/workspace/strongmother-F-mppi-20261017 unless stated otherwise.

| Component | Value network role | Exact source |
|---|---|---|
| Native one-step choice | r + gamma V, safety/risk, index0 deduction; computed every step | run.py:98-102; /home/abc/workspace/ral-strongmother-20261014/source/kdarl_clean/crowd_nav/policy/mamba_rl.py:793-902 |
| No admitted plan fallback | Returns native learned one-step choice | references/progress-contract/controller.py:104,147-151 |
| Warm start | Previous admitted shifted sequence; otherwise goal-direction grid vector. NO learned value | references/progress-contract/controller.py:129-134 |
| Candidate library | Constant-grid and goal-turn sequences. NO learned value | references/progress-contract/controller.py:13-20 |
| Search running cost | Physical reward, CV motion, rejection geometry. NO learned value | references/progress-contract/controller.py:62-81 |
| Root safety mask | Full-human CV endpoint geometry; independent of weights | run.py:58-68; references/progress-contract/controller.py:141-142 |
| Smoothed commands | Previous executed action and grid; no direct neural dependence. Previous action may have arisen from neural fallback | references/progress-contract/controller.py:49-55; run.py:95-115 |
| Final sequence reranking terminal | Free-space goal-distance heuristic, not learned V | references/progress-contract-v2/controller.py:18-36 |
| MPPI optimizer terminal callback | Cached constructor zero callback, not public attribute subsequently reassigned by v2 | dependencies/pytorch_mppi/mppi.py:157-161,307-325; references/progress-contract-v2/controller.py:16 |
| ALWAYS trigger | Low-progress gate removed; 16-step window/warm starts retained; optimizer runs also on all-unsafe states, native fallback remains | /home/abc/workspace/mppi-always-20261017/run.py:38-57 |

ALWAYS is not value-free: native learned fallback remains. The native value is computed even when not consumed by an admitted plan. Removing it can change later smoothing and planning through the executed fallback action. Search initialization itself is already value independent.

## Intervention Contract

PURE removes neural evaluation and learned one-step ranking. It retains the common geometric safety mask, CV dynamics, actions, smoothing, rewards and MPPI costs. When planning cannot admit a sequence, fallback is the grid command with greatest predicted endpoint clearance, lowest original index on ties. This is a fixed value-independent default, not tuned from outcome data. Geometric one-step safety is not a learned one-step Value Lookahead and is deliberately retained for a fair safety comparison.

ALWAYS-V replaces only the terminal term that actually affects final sequence reranking. It does not repair the cached optimizer callback: doing so would change the frozen planner's search as well as its terminal estimator. Thus the result is about learned terminal reranking in this implementation, not about every possible value-guided MPPI implementation.

Terminal history uses real root history plus all sixteen CV-predicted frames; original per-frame TTC5, field ordering, float32 robot round trip and token conversion are preserved. At the final state the last24 frames comprise eight real/padded frames and sixteen predicted frames. This four-second model-predicted history can be distribution shifted; no claim of exact terminal V or general value uselessness is allowed.

No training, no new reward, no new safety threshold or planner tuning. Both checkpoints are T24 s443, trained on 5-person circle. Parent manifests are verified before and after execution. Parent reports and old outcomes are indexes; comparison outcomes are read from raw local episode JSONL.


## Development Selection

{
  "status": "FROZEN",
  "selection": "Development 20S only, SR/CR/TO/smallest lambda lexicographic",
  "cells": {
    "gru": [
      {
        "weight": 0.25,
        "success": 181,
        "collision": 9,
        "timeout": 10
      },
      {
        "weight": 0.5,
        "success": 182,
        "collision": 12,
        "timeout": 6
      },
      {
        "weight": 1.0,
        "success": 179,
        "collision": 14,
        "timeout": 7
      }
    ],
    "kda": [
      {
        "weight": 0.25,
        "success": 181,
        "collision": 7,
        "timeout": 12
      },
      {
        "weight": 0.5,
        "success": 183,
        "collision": 5,
        "timeout": 12
      },
      {
        "weight": 1.0,
        "success": 179,
        "collision": 5,
        "timeout": 16
      }
    ]
  },
  "gru": 0.5,
  "kda": 0.5
}

## Closed-loop Results

| Mother | Scene | Arm | SR/CR/TO % | ΔSR vs ALWAYS [95% CI], pp |
|---|---|---|---|---|
| gru | square | base | 80.6/5.8/13.6 | - |
| gru | square | always | 93.4/1.4/5.2 | - |
| gru | square | pure | 92.0/2.8/5.2 | -1.4 [-3.0,+0.0] |
| gru | square | value | 94.0/2.4/3.6 | +0.6 [-1.8,+3.0] |
| gru | circle | base | 96.2/2.0/1.8 | - |
| gru | circle | always | 97.8/2.2/0.0 | - |
| gru | circle | pure | 97.8/2.0/0.2 | +0.0 [-1.2,+1.2] |
| gru | circle | value | 97.6/2.4/0.0 | -0.2 [-1.6,+1.0] |
| kda | square | base | 74.0/7.4/18.6 | - |
| kda | square | always | 91.6/3.0/5.4 | - |
| kda | square | pure | 92.0/2.8/5.2 | +0.4 [-1.4,+2.2] |
| kda | square | value | 92.8/2.4/4.8 | +1.2 [-1.4,+3.8] |
| kda | circle | base | 90.8/3.4/5.8 | - |
| kda | circle | always | 98.4/1.6/0.0 | - |
| kda | circle | pure | 97.8/2.0/0.2 | -0.6 [-2.0,+0.8] |
| kda | circle | value | 96.0/2.2/1.8 | -2.4 [-4.0,-1.0] |

## Paired Effects

### gru square
pure-always: success -1.4 pp [-3.0,+0.0]; collision +1.4 pp [+0.2,+2.8]; timeout +0.0 pp [-0.8,+0.8]. Success rescue/damage: 4/11.
value-always: success +0.6 pp [-1.8,+3.0]; collision +1.0 pp [-0.4,+2.4]; timeout -1.6 pp [-3.6,+0.4]. Success rescue/damage: 21/18.
always-base: success +12.8 pp [+9.2,+16.6]; collision -4.4 pp [-6.6,-2.4]; timeout -8.4 pp [-11.6,-5.2]. Success rescue/damage: 81/17.
### gru circle
pure-always: success +0.0 pp [-1.2,+1.2]; collision -0.2 pp [-1.6,+1.0]; timeout +0.2 pp [+0.0,+0.6]. Success rescue/damage: 5/5.
value-always: success -0.2 pp [-1.6,+1.0]; collision +0.2 pp [-1.0,+1.6]; timeout +0.0 pp [+0.0,+0.0]. Success rescue/damage: 5/6.
always-base: success +1.6 pp [-0.2,+3.4]; collision +0.2 pp [-1.4,+1.8]; timeout -1.8 pp [-3.0,-0.8]. Success rescue/damage: 15/7.
### kda square
pure-always: success +0.4 pp [-1.4,+2.2]; collision -0.2 pp [-1.8,+1.2]; timeout -0.2 pp [-1.2,+0.6]. Success rescue/damage: 11/9.
value-always: success +1.2 pp [-1.4,+3.8]; collision -0.6 pp [-2.4,+1.0]; timeout -0.6 pp [-2.8,+1.6]. Success rescue/damage: 26/20.
always-base: success +17.6 pp [+13.4,+21.8]; collision -4.4 pp [-7.0,-1.8]; timeout -13.2 pp [-16.8,-9.6]. Success rescue/damage: 108/20.
### kda circle
pure-always: success -0.6 pp [-2.0,+0.8]; collision +0.4 pp [-1.0,+1.8]; timeout +0.2 pp [+0.0,+0.6]. Success rescue/damage: 5/8.
value-always: success -2.4 pp [-4.0,-1.0]; collision +0.6 pp [-0.4,+1.6]; timeout +1.8 pp [+0.8,+3.0]. Success rescue/damage: 2/14.
always-base: success +7.6 pp [+5.0,+10.4]; collision -1.8 pp [-3.6,-0.2]; timeout -5.8 pp [-8.0,-3.8]. Success rescue/damage: 43/5.

## Matched Configured-load Latency

16 logical cores, single-thread evaluation; one worker for benchmark, four for full runs. 50 preregistered cases/cell, all four arms interleaved per case. P95 pools actual step timings. Bootstrap10000 paired case draws, no multiplicity correction. External machine load recorded separately.

| Mother | Scene | Arm | Mean ms | p95 ms | Neural value calls |
|---|---|---|---|---|---|
| gru | square | base | 11.00 | 15.64 | 4478 |
| gru | square | always | 66.57 | 71.68 | 3523 |
| gru | square | pure | 61.78 | 69.29 | 0 |
| gru | square | value | 270.67 | 307.16 | 18991 |
| gru | circle | base | 10.97 | 16.02 | 3741 |
| gru | circle | always | 66.60 | 71.73 | 3415 |
| gru | circle | pure | 61.53 | 67.81 | 0 |
| gru | circle | value | 273.79 | 310.30 | 17853 |
| kda | square | base | 13.43 | 18.47 | 5102 |
| kda | square | always | 68.88 | 74.35 | 3421 |
| kda | square | pure | 61.74 | 69.47 | 0 |
| kda | square | value | 285.37 | 321.19 | 19397 |
| kda | circle | base | 13.47 | 18.57 | 4359 |
| kda | circle | always | 69.10 | 76.24 | 3325 |
| kda | circle | pure | 61.77 | 69.96 | 0 |
| kda | circle | value | 288.22 | 320.51 | 20282 |

## Verdict

NO_DETECTED_VALUE_TERMINAL_BENEFIT

If intervals both cross zero, the registered wording is "价值终端无用". Scientifically this means no detected SR benefit for this frozen value/reranking implementation and selected weights; it does not establish universal uselessness or equivalence. PURE tests removal of learned fallback as well as saved forward compute, not a claim that every learned planner is dispensable. Mixed terminal weights change final reranking, not cached optimizer search. Four-second CV-predicted histories may be distribution shifted. All original failures and negative outcomes retained. No automatic training or parameter search.

Raw sources: episodes/full, episodes/development, episodes/benchmark; parent local jia05/jia04 episodes. Frozen selection: FROZEN-WEIGHTS.json. Smoke and units: smoke-verified.json, unit-tests.json. Exit code0.

## 最终科学裁决与研究决策

【研究结论】ALWAYS每步仍计算母体Value，但其评分仅在找不到合格规划序列时决定回退动作；热启动、候选库、物理运行成本、几何安全和平滑没有直接读取网络Value。PURE因此不等价于ALWAYS。去除网络后仍保留了较高成功率，但不能宣称完全无损。本轮是机制消融，没有建立独立算法贡献。

【核心证据】20S的PURE为92.0% SR、2.8% CR、5.2% TO（两母体下逐位相同）。相对GRU ALWAYS，少7次成功、碰撞多7次，CR差+1.4pp [0.2,2.8]；相对KDA ALWAYS，SR仅+0.4pp [-1.4,2.2]。网络回退作用不能一概忽略。两母体的PURE是同一个不读取网络权重的控制器，不是两次独立训练复现。

ALWAYS-V均由开发块选为lambda=0.5。20S相对ALWAYS的SR差为GRU +0.6pp [-1.8,3.0]、KDA +1.2pp [-1.4,3.8]，均未达到+2pp且区间下界>0的保留门槛。KDA圆形SR反而-2.4pp [-4.0,-1.0]，新增超时+1.8pp [0.8,3.0]。相关区间均未多重比较校正。

【科学裁决】SUPPORTED：原ALWAYS仍依赖学习型回退，PURE并非同一决策规则。按本轮预注册，两个20S区间均含0，操作性标签为“价值终端无用”；严格科学含义仅为“该冻结终端重排方案未检出稳定SR收益”。关于所有Value终端普遍无用的命题是INCONCLUSIVE，不能把未拒绝零假设写成等价性证明。PURE与ALWAYS在所有情形下无损等价的命题亦不成立。

【研究决策】KILL本轮直接混合冻结Value与启发值的ALWAYS-V候选；不自动训练、调lambda或改优化器回调。当前没有证据支持无条件删除学习型回退，也不否定为规划续策重新学习的Value或其他终端构造。

【执行结果或下一动作】本轮结束：6120次新完整episode（冒烟120、开发1200、正式4000、时延800），无网络训练。正式与重复运行均已完成，原始逐例结果、冻结源码/配置/权重校验及配对救回/破坏case清单保留。

## 重要实现与成本边界

1. 当前固定库缓存了构造时的零终端回调，随后给公开terminal_state_cost属性赋值并不会更新缓存。运行期核验该缓存输出为0。本轮保留jia05实际执行分支，只更换最终候选序列重排的终端项；没有将Value注入MPPI采样更新。不能把结果写成对完整价值引导MPPI的否定。
2. 冻结V对应原母体续策，实际后续控制为ALWAYS MPPI；同时终端输入包含4秒CV预测历史。续策不匹配与预测历史分布偏移均未被本轮隔离，不得把负结果直接归咎Value训练失败。
3. 终端时间/到目标mask保留原实现，没有顺带修复原多步模拟的终止约定。原始任务奖励尺度未重新归一化或校准。
4. 当前ALWAYS-V均值271-288ms、p95 307-321ms，超过250ms控制步长；ALWAYS约67-69ms，PURE约62ms。这是本实现的实测计算成本，不是证明优化后的实现也必然超过实时预算。
5. 同一机器同一单worker配置，四臂按case交错测量；操作系统外部负载并非数学恒定，资源采样保存在resource-monitor.jsonl。没有调用4090，没有向其他任务发信号。

## 最终验收与协议披露

已验证：PURE正式2000例共136302个决策步，全部网络调用0；ALWAYS-V正式2000例的终端Value调用602356次（按实际前向batch计数）。800次时延重放与对应本地正式/父实验记录的动作SHA、结局、步数、Return、路径和净距完全一致。两母体PURE在1000对同case中上述字段均完全一致。证据：final-validation.json。原Reward、净距、窗口和命令的同状态纯几何对照4项均逐位一致，见geometry-parity.json；终端token逐字段96项核验见unit-tests.json。所选lambda文件的SHA256在后续各阶段前后保持不变。

必须披露：冻结protocol.json中“No test reads before frozen selection”措辞过宽，与其明列的311000-311004冒烟阶段矛盾。实际按明列冒烟运行，选型前已读5例/格用于正确性检查；开发选型启动前已在protocol-clarification.md说明，并保留原协议哈希和clarification-manifest.json。选型代码只读取开发块，未读测试结果选lambda；所有方法、阈值、样本量未变。311000块原本已用于jia04/jia05，本轮是复用配对块，不是新独立确认。

预检首次源码解析因缺少去缩进而退出1；在任何仿真及协议冻结之前修复，见preflight-failure.json。之后单元测试、冒烟、开发、选型、正式、时延与分析均退出0。未使用的mamba_ssm可选导入有既有ABI警告，所用GRU/KDA加载和数值回归通过，未把该警告冒充运行通过证据。

本轮bootstrap seed为202610176，jia05使用另一固定seed；相同原始结局的区间末位可因有限重采样而相差0.2pp，不是评测结局发生变化。完整逐例对照与救回/破坏case列表见final-validation.json，统计重现入口为analyze.py final。
