# 模块图依据与边界

本图描绘当前 KDA-VL 的 SARL/value-lookahead 主链，不是旧 shixu 探索结构，也不宣称复原第一篇投稿版本。没有改动项目源码、权重或诊断协议。

源码根目录：/home/abc/workspace/nav_data/mamba/camrl/CrowdNav

| 图中模块 | 源码位置 / 核查依据 |
|---|---|
| ORCA 示范与 IL 价值回归 | crowd_nav/train.py：_run_il_phase、_run_policy_il_pretrain、make_il_token_windows；SARL 分支 value_regression |
| 在线采样与 MC 回归 | crowd_nav/train.py：_run_rl_phase、sarl_style_update；后者对 policy.forward_value(states) 和 batch['returns'] 做 MSE，无 TD bootstrap |
| MC 经验池 | crowd_nav/utils/ppo_buffer.py：ReplayBufferMC.store_episode；G = reward + gamma * G，训练历史窗口左侧重复填充 |
| 状态与 token | crowd_nav/policy/mamba_rl.py：_build_joint_state_34；crowd_nav/contracts.py：_batch_joint34_to_tokens_vectorized |
| 空间编码 | crowd_nav/policy/mamba_rl.py：EnhancedSpatialEncoder；机器人 13→64，五人 13+8→64，4-head self-attention，max pooling，128→256 fusion |
| KDA 骨干 | crowd_nav/policy/kda_temporal.py：KDATemporalEncoder；四层 FLA KimiDeltaAttention，head_dim=64，num_heads=4，每层 x + LayerNorm(y) |
| 标量 Value Head | crowd_nav/policy/mamba_rl.py：forward_value；取 temporal_features[:, -1, :]，Linear(256,1) |
| 80 动作与一步后继 | crowd_nav/policy/mamba_rl.py：candidate_commands、score_sarl_candidates、propagate；当前帧锚定的人选机制由评估脚本外部补入 |
| reward / clearance / filter / risk | crowd_nav/policy/mamba_rl.py：score_sarl_candidates；全部 N 个行人参与几何计算 |
| 执行与真实 history 写入 | crowd_nav/policy/mamba_rl.py：predict_sarl_style；选 result['commands'][best_idx]，追加 current_token |
| TTC5 外部评估方案 | kda-selection-20261008/evaluate-selection.py；本次 evaluate-lookahead.py 沿用其 TTC 选择并固定索引 |
| 尺寸与配置 | artifacts/vl-optimized-20261007/kda.ini：d_model=256、n_layers=4、seq_len=24、gamma=0.99、80 动作；平滑 0.3、clearance 0.2、risk 0.8 |

图中必须分清的事实：

1. 仓库默认只取列表前五人；本轮 TTC5 是评估期外部 monkey-patch，不是已经写入仓库的新训练架构。
2. 8 行 token 中目标行和运动行被构造，但当前 EnhancedSpatialEncoder 只读取第 0 行和第 3-7 行。机器人 token 本身含 gx、gy，因此不能据此声称模型没有目标信息。
3. q_head 与 future_pred_head 是已定义但不参与此主链的模块；本图没有把它们误画为当前训练的动作头或预测桥梁。
4. 图中 H1/H2/H3 标记表示本轮诊断作用位置。H2 的停止候选追加到末尾索引 80，不使用 include_stop=true。
5. 时序 backbone 只在窗口内计算，KDA use_cache=False；这不是持久 actor memory。

PNG 用于直接展示，SVG 可缩放。render-module-flow.py 可重新生成两种格式；生成脚本仅使用 Graphviz 和 Pillow，不加载导航模型。
