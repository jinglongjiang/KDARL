# ALWAYS MPPI Ablation

Status: 已验证. No training; no method-level contribution established.

## Preregistered Comparison

Two frozen T24 s443 mothers; 20 humans; 500 paired cases 311000-311499 per cell. All efficacy comparisons use round4 local base/mpc only. Parameters, warm starts, window length and safety mask unchanged. ALWAYS removes low-progress gating; all-unsafe states still invoke planning, but root safety mask forces original native fallback.

| Mother | Scene | Arm | Success/Collision/Timeout | ΔSR vs mpc [95% CI], pp |
|---|---|---|---|---|
| gru | square | base | 403/29/68 | - |
| gru | square | mpc | 449/24/27 | - |
| gru | square | always | 467/7/26 | 3.6 [+1.0,+6.2] |
| gru | circle | base | 481/10/9 | - |
| gru | circle | mpc | 491/7/2 | - |
| gru | circle | always | 489/11/0 | -0.4 [-1.6,+0.8] |
| kda | square | base | 370/37/93 | - |
| kda | square | mpc | 438/29/33 | - |
| kda | square | always | 458/15/27 | 4.0 [+0.8,+7.2] |
| kda | circle | base | 454/17/29 | - |
| kda | circle | mpc | 487/13/0 | - |
| kda | circle | always | 492/8/0 | 1.0 [-0.4,+2.6] |

## Paired Effects

### gru square
ALWAYS minus base: success +12.8 pp [+9.0,+16.6]; collision -4.4 pp [-6.6,-2.2]; timeout -8.4 pp [-11.6,-5.2]. Rescue/damage: 81/17.
ALWAYS minus mpc: success +3.6 pp [+1.0,+6.2]; collision -3.4 pp [-5.4,-1.4]; timeout -0.2 pp [-2.0,+1.6]. Rescue/damage: 32/14.
Verdict: TRIGGER_DROPS_GAIN; mean-latency difference CI [28.265099724879445, 35.26119988628833] ms.
### gru circle
ALWAYS minus base: success +1.6 pp [-0.2,+3.4]; collision +0.2 pp [-1.4,+1.8]; timeout -1.8 pp [-3.0,-0.8]. Rescue/damage: 15/7.
ALWAYS minus mpc: success -0.4 pp [-1.6,+0.8]; collision +0.8 pp [-0.4,+2.0]; timeout -0.4 pp [-1.0,+0.0]. Rescue/damage: 4/6.
Verdict: Descriptive; primary decision is square only; mean-latency difference CI [32.972149650510396, 37.80999469888471] ms.
### kda square
ALWAYS minus base: success +17.6 pp [+13.4,+21.8]; collision -4.4 pp [-7.0,-1.8]; timeout -13.2 pp [-16.8,-9.6]. Rescue/damage: 108/20.
ALWAYS minus mpc: success +4.0 pp [+0.8,+7.2]; collision -2.8 pp [-5.2,-0.6]; timeout -1.2 pp [-3.4,+1.2]. Rescue/damage: 43/23.
Verdict: TRIGGER_DROPS_GAIN; mean-latency difference CI [28.416581426368435, 35.07372982287956] ms.
### kda circle
ALWAYS minus base: success +7.6 pp [+5.0,+10.2]; collision -1.8 pp [-3.4,-0.2]; timeout -5.8 pp [-7.8,-3.8]. Rescue/damage: 43/5.
ALWAYS minus mpc: success +1.0 pp [-0.4,+2.6]; collision -1.0 pp [-2.6,+0.4]; timeout +0.0 pp [+0.0,+0.0]. Rescue/damage: 10/5.
Verdict: Descriptive; primary decision is square only; mean-latency difference CI [30.830056136886444, 35.35180296863079] ms.

## Matched-load Latency

16 logical CPUs; one single-thread evaluation worker for this separate benchmark; external R3 untouched. 50 preregistered uniformly sampled cases/cell, each replayed under all three arms in cyclic balanced order. Actual step latencies pooled for p95. Main efficacy runs use two workers and their timings do not determine verdict. External load can vary; this is matched configured load, not identical instantaneous GPU contention. Paired case-cluster bootstrap, 10000 draws, 95% percentile intervals, no multiplicity correction.

| Mother | Scene | Arm | Planning steps % | Active steps % | Mean ms | p95 ms |
|---|---|---|---|---|---|---|
| gru | square | base | 0.0 | 0.0 | 14.17 | 16.89 |
| gru | square | mpc | 41.1 | 40.9 | 38.12 | 72.89 |
| gru | square | always | 100.0 | 98.2 | 69.74 | 73.91 |
| gru | circle | base | 0.0 | 0.0 | 13.55 | 16.69 |
| gru | circle | mpc | 37.2 | 37.1 | 33.98 | 72.26 |
| gru | circle | always | 100.0 | 98.6 | 69.38 | 74.02 |
| kda | square | base | 0.0 | 0.0 | 16.15 | 18.45 |
| kda | square | mpc | 46.9 | 46.7 | 40.39 | 74.52 |
| kda | square | always | 100.0 | 98.2 | 72.01 | 76.30 |
| kda | circle | base | 0.0 | 0.0 | 15.62 | 18.28 |
| kda | circle | mpc | 41.6 | 41.5 | 38.32 | 74.13 |
| kda | circle | always | 100.0 | 98.7 | 71.41 | 75.86 |

## Interpretation and Limits

The upper-bound rule excludes an important ALWAYS advantage; it is NOT an equivalence test. Materially lower ALWAYS SR means triggered is more effective, not merely equivalent. Collision and timeout tradeoffs remain explicit. Mothers are reported separately, never treated as independent episodes pooled across training seeds. This experiment assesses deployment scheduling, not algorithmic novelty.

Raw outcome source: episodes/full; matched-load raw timings: episodes/benchmark; frozen reference sources in frozen-manifest.json. Smoke evidence: smoke-verified.json; safety-fallback test: unit-tests.json. Exit code: 0.

## Final Verdict

已验证：两个母体的20人方形均达到预注册“触发条件丢掉了增益”：GRU +3.6pp [1.0,6.2]，KDA +4.0pp [0.8,7.2]。不能再声称触发式与常驻式同效、更省计算。方形的额外收益主要体现为碰撞率下降（-3.4/-2.8pp），超时差的区间均跨0；不能把额外收益解释为已确认的进一步解卡死作用。

ALWAYS计算更贵：20S均值约69.7/72.0ms，触发式38.1/40.4ms；p95约73.9/76.3ms，触发式72.9/74.5ms。该实测不支持“常驻计算不可承受”的说法；它只显示均值计算代价更高。圆形没有明确额外SR收益，GRU碰撞点估计反而+0.8pp，必须保留。研究决策：KILL“当前低进展触发式已同效最优”这一主张，不外推为否定MPPI路线。无方法级创新进展，不自动开展新实验。

## Load Disclosure and Final Validation

已验证：ALWAYS全量136157个决策步均调用优化器；20次冒烟、200次时延重放与对应全量轨迹的动作SHA、终局、步数、Return和净距完全一致（220次独立重复检查）。base/mpc的400次时延重放及40次冒烟与jia-04本地记录一致。证据：final-validation.json、smoke-verified.json和逐回合文件。

R3 PID 2638568在08:56:05 KST仍存在，08:57:05已不存在；未向其发送信号，未核实其退出原因及训练完成状态。故不能声称外部训练负载全程恒定。每case三臂交错、单worker的配置相同，但绝对时延仍受外部负载影响。事后负载敏感性检查不改变预注册判据：排除退出过渡区间后，R3存在与不存在两段中，四格ALWAYS-minus-mpc均值时延差区间下界均>0。20S：GRU有R3 n=8，差CI [29.3,42.0]ms，无R3 n=40，[27.5,35.3]ms；KDA有R3 n=8，[18.6,38.4]ms，无R3 n=41，[29.6,35.9]ms。分组是事后描述而非新增确证检验，过渡组样本过少不作推断。证据：resource-monitor.jsonl、load-sensitivity.json。

权重：GRU T24 s443 SHA256 212bf66c5d90a3930ef69e027de848da18e32663bbbe78483f51821d7de6713c；KDA T24 s443 SHA256 c6cf228f489793ddc6dacc9bb78aa07ce8fcd9ec0fba1588e3a15c0fb38c23fd。源与配置的完整继承校验见frozen-manifest.json及jia-04的两份manifest。正式评测5590.3s，时延重放2065.8s，分析4.2s；所有阶段退出码0。

工程备注：追加报告时，容器生成文件归root所有，第一次apply_patch因权限失败；仅将本任务RESULT.md所有者改回本机用户后重试成功。未修改评测、原始数据、协议或统计口径。
