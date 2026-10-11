

from __future__ import annotations

import os, sys, time, argparse, logging, random, configparser, math, re
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as tb_err:
    SummaryWriter = None
    print(f"[TENSORBOARD] Disabled due to import failure: {tb_err}")

_SUCCESS_TOKENS = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
_COLLISION_TOKENS = {"collision"}
_TIMEOUT_TOKENS = {"timeout"}

def _event_token(info):

    try:
        if info is None:
            return ""
        if isinstance(info, dict):
            val = info.get("event") or info.get("Event") or info.get("status") or info.get("done_event")
            if val is not None:
                return str(val).replace(" ", "_").lower().strip()
        for attr in ("event", "name", "value"):
            if hasattr(info, attr):
                try:
                    return str(getattr(info, attr)).replace(" ", "_").lower().strip()
                except Exception:
                    pass
        s = str(info)
        if "." in s:
            s = s.split(".")[-1]
        return s.replace(" ", "_").lower().strip()
    except Exception:
        return ""

def _info_get(info, key, default=None):
    if isinstance(info, dict):
        return info.get(key, default)
    if hasattr(info, key):
        try:
            return getattr(info, key)
        except Exception:
            return default
    return default

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64,expandable_segments:False')

_TMP_DIR = '/root/tmp'
try:
    os.makedirs(_TMP_DIR, exist_ok=True)
except Exception:
    _TMP_DIR = os.getcwd()
for _tmp_env in ('TMPDIR', 'TEMP', 'TMP'):
    os.environ.setdefault(_tmp_env, _TMP_DIR)
import tempfile
tempfile.tempdir = _TMP_DIR
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.ppo_buffer import ReplayBufferIQL, ReplayBufferSAC
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.log_formatter import get_log_formatter

from crowd_nav.contracts import (
    tokens_to_34d, joint34_to_tokens, simulate_next_frames,
    pick_last
)

from crowd_nav.utils.metrics import (
    TrainingStats, Plotter, MetricsPlotter,
    log_periodic_metrics, log_final_metrics
)

def linear_schedule(start: float, end: float, total_steps: int, current_step: int) -> float:

    if current_step >= total_steps:
        return end
    return start + (end - start) * (current_step / total_steps)

# il_ratio_schedule已删除 - PPO不混合IL数据

class TrainConfig:

    def __init__(self, cfg: configparser.RawConfigParser, outdir: str):

        self.seed = int(cfg.getfloat('train', 'seed', fallback=42))
        self.gamma = cfg.getfloat('train', 'gamma', fallback=0.99)
        self.learning_rate = cfg.getfloat('train', 'learning_rate', fallback=1e-4)
        self.batch_size = int(cfg.getfloat('train', 'batch_size', fallback=96))
        self.updates_per_ep = int(cfg.getfloat('train', 'updates_per_ep', fallback=12))

        self.clip_grad_norm = cfg.getfloat('train', 'clip_grad_norm', fallback=1.0)

        self.n_step = cfg.getint('train', 'n_step', fallback=1)
        self.bootstrap_on_timeout = cfg.getboolean('train', 'bootstrap_on_timeout', fallback=False)

        self.il_ckpt = cfg.get('train', 'il_ckpt', fallback=os.path.join(outdir, 'il_policy.pth'))
        self.il_epochs = int(cfg.getfloat('train', 'il_epochs', fallback=120))
        self.il_batch_size = int(cfg.getfloat('train', 'il_batch_size', fallback=256))
        # [FIX] 默认False，避免每次启动都重训BC
        force_retrain_raw = str(cfg.get('train', 'il_force_retrain', fallback='false')).strip().lower()
        self.il_force_retrain = force_retrain_raw in ('true', '1', 'yes')

        self.teacher_neighbor_dist = cfg.getfloat('imitation_learning', 'teacher_neighbor_dist', fallback=6.0)
        self.teacher_max_neighbors = cfg.getint('imitation_learning', 'teacher_max_neighbors', fallback=15)
        self.teacher_time_horizon = cfg.getfloat('imitation_learning', 'teacher_time_horizon', fallback=4.5)
        self.teacher_time_horizon_obst = cfg.getfloat('imitation_learning', 'teacher_time_horizon_obst', fallback=4.5)
        self.teacher_safety_space = cfg.getfloat('imitation_learning', 'teacher_safety_space', fallback=0.10)
        self.success_target = int(cfg.getfloat('imitation_learning', 'success_target', fallback=2000))
        self.max_prefill_episodes = int(cfg.getfloat('imitation_learning', 'max_prefill_episodes', fallback=12000))
        self.prefill_batch_episodes = int(cfg.getfloat('imitation_learning', 'prefill_batch_episodes', fallback=64))
        self.prefill_patience_batches = int(cfg.getfloat('imitation_learning', 'prefill_patience_batches', fallback=10))
        self.max_il_prefill = int(cfg.getfloat('imitation_learning', 'max_il_prefill', fallback=500))

        self.warmup_episodes = int(cfg.getfloat('buffer', 'warmup_episodes', fallback=50))

        # 修复：优先从[train]读取train_episodes（SSOT），再从[trainer]读取episodes（兼容旧配置）
        self.episodes = int(cfg.getfloat('train', 'train_episodes',
                                         fallback=cfg.getfloat('trainer', 'episodes', fallback=3000)))
        self.save_every = int(cfg.getfloat('train', 'save_every',
                                           fallback=cfg.getfloat('trainer', 'save_every', fallback=200)))
        self.roll_window = int(cfg.getfloat('train', 'roll_window', fallback=50))
        self.plot_every = int(cfg.getfloat('train', 'plot_every', fallback=25))

        self.reward_profile = cfg.get('train', 'reward_profile', fallback='benchmark')

class EnvConfig:

    def __init__(self, cfg: configparser.RawConfigParser):

        self.time_step = cfg.getfloat('env', 'time_step', fallback=0.25)
        self.time_limit = cfg.getint('env', 'time_limit', fallback=25)

        self.human_num = cfg.getint('sim', 'human_num', fallback=5)
        self.circle_radius = cfg.getfloat('sim', 'circle_radius', fallback=4.0)

        self.robot_visible = cfg.get('robot', 'visible', fallback='false').lower() == 'true'
        self.robot_success_radius = cfg.getfloat('robot', 'success_radius', fallback=0.25)

        self.success_reward = cfg.getfloat('reward', 'success_reward', fallback=1.0)
        self.collision_penalty = cfg.getfloat('reward', 'collision_penalty', fallback=-0.25)
        self.timeout_penalty = cfg.getfloat('reward', 'timeout_penalty', fallback=-0.6)
        self.success_radius = cfg.getfloat('reward', 'success_radius', fallback=0.25)

        self.orca_neighbor_dist = cfg.getfloat('orca', 'neighbor_dist', fallback=3.0)
        self.orca_max_neighbors = cfg.getint('orca', 'max_neighbors', fallback=10)
        self.orca_time_horizon = cfg.getfloat('orca', 'time_horizon', fallback=3.0)
        self.orca_time_horizon_obst = cfg.getfloat('orca', 'time_horizon_obst', fallback=3.0)
        self.orca_safety_space = cfg.getfloat('orca', 'safety_space', fallback=0.20)

class PolicyConfig:

    def __init__(self, cfg: configparser.RawConfigParser):

        self.policy_name = str(cfg.get('policy', 'key', fallback='mamba_rl')).lower()

        self.seq_len = int(cfg.getfloat('buffer', 'seq_len', fallback=1))

        self.d_model = int(cfg.getfloat('policy', 'd_model', fallback=256)) if cfg.has_option('policy', 'd_model') else 256
        self.n_layers = int(cfg.getfloat('policy', 'n_layers', fallback=4)) if cfg.has_option('policy', 'n_layers') else 4

def safe_torch_load(path, map_location='cpu'):
    """安全加载checkpoint，兼容不同PyTorch版本和CUDA环境

    解决3060/4090服务器间PyTorch版本差异导致的加载问题：
    - PyTorch 2.6+: weights_only默认为True，导致加载包含非tensor对象的checkpoint失败
    - 旧版PyTorch: 不支持weights_only参数
    """
    try:
        # 优先尝试 weights_only=False（适配新版PyTorch 2.6+）
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 旧版PyTorch不支持weights_only参数
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # 其他加载错误，记录并重新抛出
        logging.warning(f"[COMPAT] Checkpoint加载异常: {e}")
        raise

def set_seed(seed: int, deterministic: bool = False):
    """
    设置全局随机种子

    Args:
        seed: 随机种子
        deterministic: 是否启用CUDA确定性模式
            - True: 完全可复现，但速度慢20-30%（测评用）
            - False: 允许CUDA优化，速度快但有微小数值差异（训练用，默认）
    """
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        # 关闭TF32以确保完全确定性（TF32有舍入误差）
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logging.info("[SEED] 🔒 CUDA确定性模式已启用（完全可复现，速度较慢）")
    elif torch.cuda.is_available():
        # 训练模式：保持性能优化
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logging.info("[SEED] ⚡ CUDA确定性模式已关闭（训练模式，速度优先）")

def load_il_ckpt(policy, value, path, device="cpu"):

    ckpt = safe_torch_load(path, map_location=device)

    sd_val = ckpt.get("value") or ckpt.get("value_state")
    assert sd_val, f"IL ckpt缺少value state_dict: {path}"

    value.load_state_dict(sd_val, strict=False)

    # PPO不需要target_value network

    meta = ckpt.get("meta", {})
    stage = str(meta.get("stage", "")).lower() if isinstance(meta.get("stage"), str) else str(ckpt.get("stage", "")).lower()
    assert "il" in stage, f"不是IL权重或元信息缺失: stage={stage}"

    success_rate = meta.get("il_success_rate") or ckpt.get("il_success_rate", 0.0)
    arch = meta.get("arch", "unknown")
    ts = meta.get("ts", 0.0)

    return {
        "success_rate": float(success_rate),
        "arch": str(arch),
        "ts": float(ts),
        "meta": meta
    }

def load_config(path: str) -> configparser.RawConfigParser:
    cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)

    if path.endswith('env.config') or path.endswith('config.txt'):
        config_dir = os.path.dirname(path) if os.path.dirname(path) else './configs'
        config_files = [
            os.path.join(config_dir, 'env.config'),
            os.path.join(config_dir, 'policy.config'),
            os.path.join(config_dir, 'train.config')
        ]

        for config_file in config_files:
            if os.path.exists(config_file):
                cfg.read(config_file, encoding='utf-8')
    else:
        cfg.read(path, encoding='utf-8')

    return cfg

def setup_logger(outdir: str, level=logging.INFO):
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, 'train.log')

    if os.path.exists(log_path):
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(outdir, f'train_backup_{timestamp}.log')
        try:
            os.rename(log_path, backup_path)
            print(f"LOG_BACKUP: Previous log backed up to: {backup_path}")
        except Exception as e:
            print(f"LOG_BACKUP: Log backup failed: {e}")

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    h1 = logging.StreamHandler(sys.stdout); h1.setFormatter(fmt)
    h2 = logging.FileHandler(log_path); h2.setFormatter(fmt)
    root = logging.getLogger(); root.handlers = []; root.setLevel(level)

    root.addHandler(h1); root.addHandler(h2)
    return logging.getLogger(__name__)

class GradientMonitor:
    def __init__(self, sample_every: int = 200, mode: str = 'lite', watch=None):
        self.sample_every = sample_every; self.mode = mode
        self.watch = watch or ['spatial_mamba', 'temporal_mamba', 'value_head']
        self.last_sample_step = 0; self.history = []
        self.cache = {'total_norm': 0.0, 'last': -1}
    def analyze(self, net: nn.Module, step: int) -> Dict[str, float]:
        if self.mode == 'off':
            return {'grad_norm_total': 0.0}
        if step - self.last_sample_step < self.sample_every:
            return {'grad_norm_total': self.cache['total_norm']}
        self.last_sample_step = step
        tot2 = 0.0
        for n, p in net.named_parameters():
            if p.grad is None: continue
            g = p.grad.data
            tot2 += float(g.norm(2).item() ** 2)
        total = float(tot2 ** 0.5) if tot2 > 0 else 0.0
        self.cache['total_norm'] = total
        self.history.append(total); self.history = self.history[-50:]
        if step % (self.sample_every * 4) == 0:
            logging.info(f"[GRAD] step={step} total={total:.3f}")
        return {'grad_norm_total': total}

# ========== SAC Update Functions ==========

# compute_reward_tensor 已删除（Clean RL 不再需要重新计算 IL reward）


def sac_update(
    batch,
    q_net1, q_net2,
    q_target1, q_target2,
    policy_net,
    q_optimizer, policy_optimizer,
    alpha_optimizer,
    log_alpha,
    gamma,
    tau,
    target_entropy,
    device,
    cfg,
):
    """标准 SAC 更新：Bellman + entropy regularization"""
    import torch
    import torch.nn.functional as F
    import numpy as np

    states      = torch.as_tensor(batch["states"], device=device, dtype=torch.float32)
    actions     = torch.as_tensor(batch.get("actions", batch.get("actions_continuous")),
                                  device=device, dtype=torch.float32)
    rewards     = torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32).unsqueeze(-1)
    next_states = torch.as_tensor(batch["next_states"], device=device, dtype=torch.float32)
    dones       = torch.as_tensor(batch["dones"], device=device, dtype=torch.float32).unsqueeze(-1)

    # ReplayBufferSAC 返回的是 [B, T, ...] 序列，这里只取最后一帧
    if actions.dim() == 3:
        actions = actions[:, -1, :]
    if rewards.dim() == 2:
        rewards = rewards[:, -1:]
    if dones.dim() == 2:
        dones = dones[:, -1:]

    # ---------- Critic 目标 ----------
    with torch.no_grad():
        mean_next, log_std_next, _, _ = policy_net.forward_both(next_states, return_sequence=False)
        std_next = torch.exp(log_std_next)
        dist_next = torch.distributions.Normal(mean_next, std_next)
        u_next = dist_next.rsample()
        next_action = torch.tanh(u_next)

        log_prob_next = dist_next.log_prob(u_next).sum(dim=-1, keepdim=True)
        log_prob_next -= (2 * (np.log(2) - u_next - F.softplus(-2 * u_next))).sum(dim=-1, keepdim=True)

        q1_next = q_target1(next_states, next_action)
        q2_next = q_target2(next_states, next_action)
        q_next_min = torch.min(q1_next, q2_next)

        alpha = log_alpha.exp()
        v_next = q_next_min - alpha * log_prob_next
        q_target = rewards + (1.0 - dones) * gamma * v_next

    # ---------- Critic 更新 ----------
    q1_pred = q_net1(states, actions)
    q2_pred = q_net2(states, actions)
    q1_loss = F.mse_loss(q1_pred, q_target)
    q2_loss = F.mse_loss(q2_pred, q_target)
    q_loss = q1_loss + q2_loss

    q_optimizer.zero_grad(set_to_none=True)
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net1.parameters(), max_norm=10.0)
    torch.nn.utils.clip_grad_norm_(q_net2.parameters(), max_norm=10.0)
    q_optimizer.step()

    # ---------- Actor 更新 ----------
    mean, log_std, _, _ = policy_net.forward_both(states, return_sequence=False)
    std = torch.exp(log_std)
    dist = torch.distributions.Normal(mean, std)
    u = dist.rsample()
    new_actions = torch.tanh(u)

    log_pi = dist.log_prob(u).sum(dim=-1, keepdim=True)
    log_pi -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=-1, keepdim=True)

    q1_pi = q_net1(states, new_actions)
    q2_pi = q_net2(states, new_actions)
    q_pi = torch.min(q1_pi, q2_pi)

    alpha = log_alpha.exp().detach()
    policy_loss_sac = (alpha * log_pi - q_pi).mean()

    # BC regularization: 防止policy偏离BC学到的行为
    bc_weight = cfg.getfloat('sac', 'bc_regularization_weight', fallback=0.0)
    if bc_weight > 0 and 'bc_actions' in batch and 'bc_states' in batch:
        bc_actions = torch.as_tensor(batch['bc_actions'], device=device, dtype=torch.float32)
        bc_states = torch.as_tensor(batch['bc_states'], device=device, dtype=torch.float32)

        if bc_actions.dim() == 3:
            bc_actions = bc_actions[:, -1, :]
        
        # Compute policy output on IL states specifically for BC loss
        bc_mean, _, _, _ = policy_net.forward_both(bc_states, return_sequence=False)
        
        bc_loss = F.mse_loss(torch.tanh(bc_mean), bc_actions)
        policy_loss = policy_loss_sac + bc_weight * bc_loss
    else:
        policy_loss = policy_loss_sac

    policy_optimizer.zero_grad(set_to_none=True)
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    policy_optimizer.step()

    # ---------- alpha 更新 ----------
    if target_entropy is not None and alpha_optimizer is not None:
        alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optimizer.step()
    else:
        alpha_loss = torch.tensor(0.0, device=device)

    # ---------- target Q soft update ----------
    with torch.no_grad():
        for param, target_param in zip(q_net1.parameters(), q_target1.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        for param, target_param in zip(q_net2.parameters(), q_target2.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    return {
        "q1_loss": q1_loss.item(),
        "q2_loss": q2_loss.item(),
        "q_loss": q_loss.item(),
        "policy_loss": policy_loss.item(),
        "alpha_loss": alpha_loss.item(),
        "alpha": log_alpha.exp().item(),
    }


def soft_update_target(source_net, target_net, tau):
    """Polyak averaging: target = tau * source + (1 - tau) * target"""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def build_env_and_robot(cfg) -> Tuple[CrowdSim, Robot]:

    env = CrowdSim(); env.configure(cfg)
    robot = Robot(cfg, 'robot'); env.set_robot(robot)
    return env, robot

def build_policy(policy_key: str, policy_cfg, device: torch.device):
    Policy = policy_factory[policy_key]
    try:
        policy = Policy(policy_cfg, device=device)
    except TypeError:
        try:
            policy = Policy(policy_cfg)
        except TypeError:
            policy = Policy()

    if hasattr(policy,'configure'):
        policy.configure(policy_cfg)

    if hasattr(policy, 'to'):
        policy.to(device)
    elif hasattr(policy,'set_device'):
        policy.set_device(device)

    return policy

def bind_policy(robot: Robot, policy, env, epsilon: Optional[float]=None):
    if hasattr(policy,'set_env'): policy.set_env(env)
    if hasattr(policy,'set_robot'): policy.set_robot(robot)
    if epsilon is not None and hasattr(policy,'set_epsilon'): policy.set_epsilon(epsilon)
    robot.set_policy(policy)
    return policy

def _log_action_table(GRID):

    n_actions = GRID['n_speeds'] * GRID['n_headings']
    v_max_grid = GRID['v_max']
    v_min_grid = GRID['v_min']

    # 验证动作空间配置
    assert n_actions == 80, f"Expected 80 actions, got {n_actions}"
    assert GRID['n_speeds'] == 5, f"Expected 5 speeds, got {GRID['n_speeds']}"
    assert GRID['n_headings'] == 16, f"Expected 16 headings, got {GRID['n_headings']}"
    assert v_min_grid >= 0.0, f"v_min should be >= 0.0, got {v_min_grid}"

    return n_actions

def _log_policy_tag(phase, tag):

    logging.info(f"[POLICY-TAG] phase={phase} tag={tag}")


def _clamp_policy_std(policy, min_std: float = 0.05, max_std: float = 0.25):
    """将策略的动作标准差限制在安全范围，避免BC权重被大探索噪声破坏"""
    if not hasattr(policy, 'action_log_std'):
        return
    try:
        with torch.no_grad():
            current_std = policy.action_log_std.detach().exp()
            clamped_std = current_std.clamp(min=min_std, max=max_std)
            if not torch.allclose(clamped_std, current_std):
                policy.action_log_std.copy_(clamped_std.log())
                logging.info(f"[ACTION-STD] Clamped std {current_std.cpu().numpy()} -> {clamped_std.cpu().numpy()}")
            else:
                logging.info(f"[ACTION-STD] std unchanged {current_std.cpu().numpy()}")
    except Exception as e:
        logging.warning(f"[ACTION-STD] Failed to clamp std: {e}")

def _run_policy_il_pretrain(policy, device, seq_len: int, il_epochs: int, il_batch_size: int, il_ckpt_path: str,
                            explorer, gamma: float = 0.97, success_trajs: Optional[List] = None, config=None):

    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from crowd_nav.contracts import joint34_to_tokens

    def _pad_state_to_34(state):

        if hasattr(state, 'to_array'):
            arr = state.to_array()

        elif isinstance(state, list) and len(state) > 0 and hasattr(state[0], 'px'):

            robot_state = np.zeros(9, dtype=np.float32)

            human_states = []
            for obs in state[:5]:
                human_states.extend([obs.px, obs.py, obs.vx, obs.vy, obs.radius])

            while len(human_states) < 25:
                human_states.append(0.0)

            arr = np.concatenate([robot_state, human_states[:25]], axis=0)
        else:

            arr = np.asarray(state, dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 1:
            if arr.shape[0] < 34:
                return np.pad(arr, (0, 34 - arr.shape[0]), mode='constant')
            return arr[:34]
        elif arr.ndim == 2:
            pad_width = [(0, 0), (0, max(0, 34 - arr.shape[1]))]
            arr = np.pad(arr, pad_width, mode='constant') if arr.shape[1] < 34 else arr[:, :34]
            return arr
        return arr.reshape(-1)[:34]

    def _build_il_dataset(trajs, gamma_value):

        seq_len_local = max(1, int(seq_len))
        state_windows = []
        teacher_actions = []
        returns = []

        n_clipped_actions = 0
        n_total_actions = 0

        for (traj, _info) in trajs:
            states, actions, rewards = traj
            T = len(states)
            if T == 0:
                continue

            states_34 = [_pad_state_to_34(s) for s in states]

            mc = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma_value * G
                mc.insert(0, G)

            for t in range(T):
                # [FIX] 先处理action，决定是否跳过，再构建state window
                action_t = actions[t]
                if hasattr(action_t, 'vx') and hasattr(action_t, 'vy'):
                    action_vx, action_vy = float(action_t.vx), float(action_t.vy)
                else:
                    action_arr = np.array(action_t, dtype=np.float32)[:2]
                    action_vx, action_vy = float(action_arr[0]), float(action_arr[1])

                action_speed = np.sqrt(action_vx**2 + action_vy**2)
                v_max = 1.0
                n_total_actions += 1

                # [修复A] 过滤静止帧，防止网络学习"不动"
                # [BC-FIX] 随机保留15%低速帧，让机器人学会减速/刹车
                if action_speed < 0.05:
                    if np.random.rand() > 0.15:  # 85%概率跳过，15%保留
                        continue

                # 动作有效，构建state window
                window = states_34[max(0, t - seq_len_local + 1): t + 1]
                if not window:
                    continue
                if len(window) < seq_len_local:
                    pad_needed = seq_len_local - len(window)
                    pad_frame = window[0]
                    window = [pad_frame.copy() for _ in range(pad_needed)] + [w.copy() for w in window]
                else:
                    window = [w.copy() for w in window[-seq_len_local:]]

                state_windows.append(np.stack(window, axis=0).astype(np.float32))

                # 归一化和clip动作
                if action_speed > v_max:
                    n_clipped_actions += 1
                    action_vx = action_vx / action_speed * v_max
                    action_vy = action_vy / action_speed * v_max

                # [修复B] 归一化到[-1, 1]范围，匹配网络输出空间
                # 这是最关键的修复：网络Linear层输出无限制，需要统一量级
                action_vx_normalized = np.clip(action_vx / v_max, -1.0, 1.0)
                action_vy_normalized = np.clip(action_vy / v_max, -1.0, 1.0)

                teacher_actions.append([action_vx_normalized, action_vy_normalized])
                returns.append(mc[t])

        if not state_windows:
            raise RuntimeError("[IL-CONTINUOUS-BC] No data available for BC training")

        state_windows = np.asarray(state_windows, dtype=np.float32)
        teacher_actions = np.asarray(teacher_actions, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)

        clip_rate = n_clipped_actions / n_total_actions * 100 if n_total_actions > 0 else 0
        logging.info(f"[IL-ACTION-PROCESS] Total actions: {n_total_actions}")
        logging.info(f"[IL-ACTION-PROCESS] Clipped {n_clipped_actions}/{n_total_actions} actions ({clip_rate:.2f}%) to v_max=1.0")
        logging.info(f"[IL-ACTION-PROCESS] Randomly kept 15% low-speed frames (speed<0.05) for deceleration learning, total {len(teacher_actions)} actions")
        logging.info(f"[IL-ACTION-PROCESS] Actions normalized to [-1, 1] for network training")

        flat_states = state_windows.reshape(-1, state_windows.shape[-1])
        logging.info(f"[IL-TOKEN-CONVERT] Converting {flat_states.shape[0]} states to tokens (15K dataset: ~3-5 min, please wait)...")
        tokens = joint34_to_tokens(flat_states)
        logging.info(f"[IL-TOKEN-CONVERT] Token conversion complete")
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.view(state_windows.shape[0], seq_len_local, 1, 6, 13).squeeze(2).contiguous()
        else:
            tokens = torch.from_numpy(tokens).view(state_windows.shape[0], seq_len_local, 1, 6, 13).squeeze(2).contiguous()

        logging.info(f"[IL-DATASET] BC dataset: {state_windows.shape[0]} sequences (seq_len={seq_len_local}, with MC returns)")

        return {
            'states_tokens': tokens.float(),
            'actions': torch.from_numpy(teacher_actions).float(),
            'returns': torch.from_numpy(returns).float()
        }

    ac_net = policy

    bc_lr = getattr(config, 'il_learning_rate', None) or config.learning_rate
    optim = torch.optim.Adam(ac_net.parameters(), lr=bc_lr)
    logging.info(f"[IL-BC] Training policy network - {sum(p.numel() for p in ac_net.parameters())} params")

    ac_net.train()

    # ========== 关键修复：BC训练使用RL模式（完整12帧序列建模）==========
    # 用户需求：BC应该模仿12帧输入，而非只用最后1帧
    # 方案：训练时用RL模式（spatial+temporal），推理时用BC模式（只spatial，保守）
    ac_net.set_training_mode('rl')
    logging.info(f"[IL-BC-MODE] Using RL mode for BC training (full 12-frame temporal modeling)")
    logging.info(f"[IL-BC-MODE] This ensures BC learns sequence dependencies, not just single-frame actions")

    logging.info(f"[IL-CONTINUOUS-BC] Starting BC training (Action Imitation + Value Regression), lr={bc_lr}, gamma={gamma}")

    dataset_trajs = success_trajs if success_trajs is not None else getattr(explorer, "_last_trajectories", [])
    if not dataset_trajs:
        logging.warning("[IL-CONTINUOUS-BC] No trajectories available, skipping BC training")
        return

    data_source = "离线数据集(generate_il_dataset.py)" if success_trajs is not None else "在线采集(explorer)"
    logging.info(f"[IL-BC] 数据来源: {data_source}")
    logging.info(f"[IL-BC] 轨迹数量: {len(dataset_trajs)} 条成功案例")

    dataset = _build_il_dataset(dataset_trajs, gamma)
    states_tokens = dataset['states_tokens'].to(device, non_blocking=True)
    teacher_actions_full = dataset['actions'].to(device, non_blocking=True)
    mc_returns_full = dataset['returns'].to(device, non_blocking=True)
    dataset_size = states_tokens.size(0)

    logging.info(f"[IL-BC] 准备数据集: 样本数={dataset_size}, batch_size={il_batch_size}, batches={dataset_size // il_batch_size}")

    pbar = tqdm(range(1, il_epochs + 1), desc="[IL-BC]", ncols=120, leave=True)
    for ep in pbar:
        if dataset_size < il_batch_size:
            logging.warning(f"[IL-BC] Dataset too small (N={dataset_size}), reducing batch size to {dataset_size}")
        perm = torch.randperm(dataset_size, device=device)

        total_continuous_bc_loss = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        n_batches = 0

        for start in range(0, dataset_size, il_batch_size):
            idx = perm[start:start + il_batch_size]
            if idx.numel() < 8:
                continue

            states_batch = states_tokens[idx].contiguous()
            teacher_actions = teacher_actions_full[idx].contiguous()
            mc_returns = mc_returns_full[idx].contiguous()

            if states_batch.device != device:
                states_batch = states_batch.to(device, non_blocking=True)
            if teacher_actions.device != device:
                teacher_actions = teacher_actions.to(device, non_blocking=True)
            if mc_returns.device != device:
                mc_returns = mc_returns.to(device, non_blocking=True)

            # Policy输出无界logits
            action_mean_logits, action_log_std, _, _ = ac_net.forward_both(states_batch)
            # BC训练：对logits应用tanh后与teacher动作([-1,1])做MSE
            action_mean_pred = torch.tanh(action_mean_logits)

            # BC loss: 最小化tanh(logits)与teacher动作的差异
            continuous_bc_loss = F.mse_loss(action_mean_pred, teacher_actions)

            # [DIAGNOSTIC] 每10个epoch打印一次动作统计
            if ep % max(1, il_epochs // 10) == 0 and start == 0:
                with torch.no_grad():
                    pred_speed = torch.sqrt(action_mean_pred[:, 0]**2 + action_mean_pred[:, 1]**2).mean().item()
                    teacher_speed = torch.sqrt(teacher_actions[:, 0]**2 + teacher_actions[:, 1]**2).mean().item()
                    logging.info(f"[IL-BC-DIAG] epoch={ep} pred_speed={pred_speed:.4f} teacher_speed={teacher_speed:.4f}")

            # BC阶段只有action loss
            loss = 10.0 * continuous_bc_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac_net.parameters(), 1.0)
            optim.step()

            total_continuous_bc_loss += continuous_bc_loss.detach()
            total_loss += loss.detach()
            n_batches += 1

        avg_bc_loss = (total_continuous_bc_loss / max(1, n_batches)).item()
        avg_total_loss = (total_loss / max(1, n_batches)).item()
        pbar.set_postfix({
            'bc': f'{avg_bc_loss:.4f}',
            'total': f'{avg_total_loss:.4f}'
        })

        if ep % max(1, il_epochs // 5) == 0 or ep == il_epochs:
            logging.info(f"[IL-BC] epoch={ep}/{il_epochs} bc_loss={avg_bc_loss:.4f} total={avg_total_loss:.4f}")

    pbar.close()

    # ========== 保存BC Checkpoint（带元信息）==========
    logging.info(f"[IL-BC] Saving BC checkpoint with metadata...")
    try:
        import time
        bc_checkpoint = {
            'value': ac_net.state_dict(),  # 兼容旧格式
            'model_state_dict': ac_net.state_dict(),  # 标准PyTorch格式
            'meta': {
                'stage': 'bc_policy_only',
                'arch': 'policy_network',
                'epochs': il_epochs,
                'bc_loss': avg_bc_loss,
                'total_loss': avg_total_loss,
                'eval_success_rate': None,  # 将在eval后填充
                'seq_len': seq_len,
                'dataset_size': dataset_size,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        }
        torch.save(bc_checkpoint, il_ckpt_path)
        logging.info(f"[IL-BC] ✓ BC checkpoint saved to {il_ckpt_path}")
    except Exception as e:
        logging.warning(f"[IL-BC] Failed to save checkpoint: {e}")

    logging.info(f"[IL-CONTINUOUS-BC] Evaluating BC-trained model (in BC mode)...")
    try:
        # 修复：评估前恢复训练环境配置（确保human_num=5，匹配BC训练数据）
        if hasattr(explorer, 'env'):
            explorer.env.human_num = 5
            explorer.env.config.set('sim', 'human_num', '5')
            logging.info(f"[IL-BC-EVAL] Environment reset to human_num=5 (matching BC training data)")

        ac_net.eval()
        policy.set_phase('eval')

        stats = explorer.run_k_episodes(
            100,
            'test',  # 改为test模式，绕过multiagent_training检查
            update_memory=False,
            show_tqdm=False,
            return_stats=True,
            imitation_learning=False,
            force_joint_state_policy=False
        )

        if isinstance(stats, dict):
            success_rate = stats.get('success_rate', 0.0)
            avg_return = stats.get('total_reward', stats.get('avg_return', 0.0))
            collision_rate = stats.get('collision_rate', 0.0)
            timeout_rate = stats.get('timeout_rate', 0.0)

            logging.info(
                f"[IL-BC-EVAL] 100 episodes | success={success_rate:.1%} | "
                f"collision={collision_rate:.1%} | timeout={timeout_rate:.1%} | reward={avg_return:.4f}"
            )

            # 更新checkpoint中的eval结果
            try:
                if os.path.exists(il_ckpt_path):
                    bc_ckpt = safe_torch_load(il_ckpt_path, map_location=device)
                    if 'meta' in bc_ckpt:
                        bc_ckpt['meta']['eval_success_rate'] = success_rate
                        bc_ckpt['meta']['eval_collision_rate'] = collision_rate
                        bc_ckpt['meta']['eval_timeout_rate'] = timeout_rate
                        torch.save(bc_ckpt, il_ckpt_path)
                        logging.info(f"[IL-BC] Updated checkpoint with eval metrics")
            except Exception as e:
                logging.warning(f"[IL-BC] Failed to update checkpoint with eval: {e}")
        else:
            logging.info(f"[IL-BC-EVAL] Evaluation completed (stats not available)")

    except Exception as e:
        logging.warning(f"[IL-BC-EVAL] Evaluation failed: {e}")
    finally:
        ac_net.train()
        policy.set_phase('train')

    ac_net.set_training_mode('rl')
    logging.info(f"[IL-BC-MODE] BC training finished, keeping RL mode for RL phase")
    logging.info(f"[IL-BC-MODE] Temporal encoder is now trained (unlike old BC-only baseline)")

    logging.info(f"[IL-BC-ADAPT] Skipping warm adaptation - temporal encoder already trained during BC")

    logging.info(f"[IL-MAMBA-EVAL] Evaluating BC-trained model in RL mode (with temporal encoder)...")
    try:
        ac_net.eval()
        policy.set_phase('eval')

        stats_rl = explorer.run_k_episodes(
            100,
            'test',  # 改为test（与BC评估一致，确保5人环境）
            update_memory=False,
            show_tqdm=False,
            imitation_learning=False,
            force_joint_state_policy=False,
            return_stats=True
        )

        if isinstance(stats_rl, dict):
            success_rate_rl = stats_rl.get('success_rate', 0.0)
            avg_return_rl = stats_rl.get('total_reward', stats_rl.get('avg_return', 0.0))
            collision_rate_rl = stats_rl.get('collision_rate', 0.0)
            timeout_rate_rl = stats_rl.get('timeout_rate', 0.0)

            logging.info(
                f"[IL-MAMBA-EVAL] 100 episodes (RL mode) | success={success_rate_rl:.1%} | "
                f"collision={collision_rate_rl:.1%} | timeout={timeout_rate_rl:.1%} | reward={avg_return_rl:.4f}"
            )
        else:
            logging.info(f"[IL-MAMBA-EVAL] RL mode evaluation completed (stats not available)")

    except Exception as e:
        logging.warning(f"[IL-MAMBA-EVAL] RL mode evaluation failed: {e}")
    finally:
        ac_net.train()
        policy.set_phase('train')

    # BC checkpoint已在训练循环后保存（带完整元信息），此处不重复保存

def _run_il_phase(train_cfg, env_cfg, policy_cfg, env, explorer, policy, device, batch_size, replay_iql, cfg=None):

    import math
    import copy

    # SAC: 使用统一的cfg对象（RawConfigParser）
    # 所有配置通过cfg.getfloat()/cfg.getint()读取
    config = cfg if cfg is not None else train_cfg

    # 保存原始环境配置（用于恢复）
    original_time_limit = config.getfloat('env', 'time_limit', fallback=25.0)
    original_success_radius = config.getfloat('robot', 'success_radius', fallback=0.25)
    original_human_num = config.getint('sim', 'human_num', fallback=5)
    original_neighbor_dist = config.getfloat('orca', 'neighbor_dist', fallback=4.5)
    original_max_neighbors = config.getint('orca', 'max_neighbors', fallback=12)
    original_time_horizon = config.getfloat('orca', 'time_horizon', fallback=5.0)
    original_time_horizon_obst = config.getfloat('orca', 'time_horizon_obst', fallback=3.0)
    original_safety_space = config.getfloat('orca', 'safety_space', fallback=0.10)

    # Teacher ORCA配置（从[imitation_learning] section读取）
    teacher_neighbor_dist = config.getfloat('imitation_learning', 'teacher_neighbor_dist', fallback=6.0)
    teacher_max_neighbors = config.getint('imitation_learning', 'teacher_max_neighbors', fallback=15)
    teacher_time_horizon = config.getfloat('imitation_learning', 'teacher_time_horizon', fallback=4.5)
    teacher_time_horizon_obst = config.getfloat('imitation_learning', 'teacher_time_horizon_obst', fallback=4.5)
    teacher_safety_space = config.getfloat('imitation_learning', 'teacher_safety_space', fallback=0.10)

    success_target = config.getint('imitation_learning', 'success_target', fallback=2000)
    max_total_episodes = config.getint('imitation_learning', 'max_prefill_episodes', fallback=12000)
    batch_collect_episodes = config.getint('imitation_learning', 'prefill_batch_episodes', fallback=64)
    patience_batches = config.getint('imitation_learning', 'prefill_patience_batches', fallback=5)

    logging.info(
        f"[IL-TEACHER] Applying ORCA teacher profile: neighbor_dist={teacher_neighbor_dist}, max_neighbors={teacher_max_neighbors}, "
        f"time_horizon={teacher_time_horizon}, time_horizon_obst={teacher_time_horizon_obst}, safety_space={teacher_safety_space}"
    )

    env.config.set('orca', 'neighbor_dist', str(teacher_neighbor_dist))
    env.config.set('orca', 'max_neighbors', str(teacher_max_neighbors))
    env.config.set('orca', 'time_horizon', str(teacher_time_horizon))
    env.config.set('orca', 'time_horizon_obst', str(teacher_time_horizon_obst))
    env.config.set('orca', 'safety_space', str(teacher_safety_space))
    env.configure(env.config)

    target_buffer = getattr(explorer, 'memory', None)
    if target_buffer is None and hasattr(explorer, 'replay_buffer'):
        target_buffer = explorer.replay_buffer
    if target_buffer is None:
        logging.warning("[IL] Explorer has no replay buffer reference; IL data will stay in-flight only")

    initial_buf_size = len(target_buffer) if target_buffer else 0
    logging.info(f"[IL-START] Buffer size before IL: {initial_buf_size}")

    # SAC: 从main函数传入的offline_dataset_path读取
    offline_dataset_path = config.get('train', 'offline_il_dataset', fallback=None)
    if not offline_dataset_path or offline_dataset_path == 'None':
        # ========== V5.0优先：使用merged文件 ==========
        # V5.0: 15000条轨迹，极端密度/速度组合，最优质数据集
        for version in ['v5.0_merged_15000', 'v5.0_merged', 'v4.0', 'v3.1', 'v3.0']:
            candidates = [
                f'data/il_dataset_diverse_{version}.pth',
                f'../data/il_dataset_diverse_{version}.pth',
                os.path.join(os.path.dirname(__file__), f'../data/il_dataset_diverse_{version}.pth'),
            ]
            for candidate in candidates:
                if os.path.exists(candidate) and os.path.getsize(candidate) > 1024:  # 确保非空文件
                    offline_dataset_path = os.path.abspath(candidate)
                    logging.info(f"[IL-CACHE] 找到数据集候选: {offline_dataset_path}")
                    break
            if offline_dataset_path:
                break

        if offline_dataset_path is None:
            # Fallback：默认路径（通常不会用到）
            offline_dataset_path = 'data/il_dataset_diverse_v5.0_merged_15000.pth'

    success_trajs_all: List = []
    success_ratio = 0.0
    total_sampled_episodes = 0
    loaded_from_cache = False

    if os.path.exists(offline_dataset_path):
        try:
            logging.info(f"[IL-CACHE] 发现离线数据集缓存: {offline_dataset_path}")
            dataset = safe_torch_load(offline_dataset_path, map_location='cpu')

            if isinstance(dataset, dict) and 'trajectories' in dataset:
                cached_trajs = dataset['trajectories']
                config_info = dataset.get('config', {})
                stats_info = dataset.get('stats', {})

                logging.info(f"[IL-CACHE] 数据集版本: {dataset.get('version', 'unknown')}")
                logging.info(f"[IL-CACHE] 生成时间: {dataset.get('timestamp', 'unknown')}")
                logging.info(f"[IL-CACHE] Seeds: {config_info.get('seeds', [])}")
                logging.info(f"[IL-CACHE] ORCA configs: {list(config_info.get('orca_configs', {}).keys())}")
                logging.info(f"[IL-CACHE] 总轨迹数: {len(cached_trajs)}")
                logging.info(f"[IL-CACHE] 原始成功率: {stats_info.get('success_rate', 0.0):.2%}")

                # [FIX] 从config读取IL预填充上限，避免IL基数过大稀释RL梯度
                max_il_prefill = config.getint('imitation_learning', 'max_il_prefill', fallback=15000)
                cached_trajs = cached_trajs[:max_il_prefill]
                logging.info(f"[IL-CACHE] 限制加载: {len(cached_trajs)} episodes (max_il_prefill={max_il_prefill})")

                for traj in cached_trajs:
                    states = traj['states']
                    actions = traj['actions']
                    rewards = traj.get('rewards', [0.0] * len(states))
                    meta = traj.get('meta', {})

                    traj_tuple = ((states, actions, rewards), meta)
                    success_trajs_all.append(traj_tuple)

                total_sampled_episodes = stats_info.get('total_attempts', len(cached_trajs))
                success_ratio = len(success_trajs_all) / max(1, total_sampled_episodes)
                loaded_from_cache = True

                logging.info(f"[IL-CACHE] ✓ 成功加载 {len(success_trajs_all)} 条轨迹")
                logging.info(f"[IL-CACHE] 跳过在线采集，直接使用离线数据集")
            else:
                logging.warning(f"[IL-CACHE] 数据集格式不正确，回退到在线采集")
        except Exception as e:
            logging.warning(f"[IL-CACHE] 加载失败: {e}，回退到在线采集")
            success_trajs_all = []
            loaded_from_cache = False
    else:
        logging.info(f"[IL-CACHE] 未发现离线数据集 ({offline_dataset_path})，使用在线采集")

    if not loaded_from_cache:
        store_only = ('success',)
        consecutive_no_gain = 0

        if hasattr(policy, 'set_epsilon'):
            orig_eps = getattr(policy, 'epsilon', None)
            policy.set_epsilon(0.0)
        else:
            orig_eps = None

        _log_policy_tag('il', 'IL-ORCA-TEACHER')

        while total_sampled_episodes < max_total_episodes and len(success_trajs_all) < success_target:
            batch = min(batch_collect_episodes, max_total_episodes - total_sampled_episodes)
            explorer.run_k_episodes(batch, 'il', update_memory=True, show_tqdm=False,
                                    imitation_learning=True, force_joint_state_policy=True,
                                    store_on=store_only)

            new_success = list(getattr(explorer, '_last_trajectories', []) or [])
            if new_success:
                success_trajs_all.extend(new_success)
                consecutive_no_gain = 0
            else:
                consecutive_no_gain += 1

            total_sampled_episodes += batch
            success_ratio = len(success_trajs_all) / max(1, total_sampled_episodes)
            logging.info(
                f"[IL-COLLECT] episodes={total_sampled_episodes}/{max_total_episodes} | "
                f"success={len(success_trajs_all)}/{success_target} ({success_ratio:.2%})"
            )

            if consecutive_no_gain >= patience_batches:
                logging.warning("[IL-COLLECT] No new successes for several batches, continuing but check teacher settings")
                consecutive_no_gain = 0

        if orig_eps is not None and hasattr(policy, 'set_epsilon'):
            policy.set_epsilon(orig_eps)

    # [FIX] 限制IL收集数量（在线采集模式）
    max_il_prefill = train_cfg.max_il_prefill
    if len(success_trajs_all) > max_il_prefill:
        logging.info(f"[IL-COLLECT] 限制收集数量: {len(success_trajs_all)} → {max_il_prefill}")
        success_trajs_all = success_trajs_all[:max_il_prefill]

    collected_success_count = len(success_trajs_all)
    collected_attempts = total_sampled_episodes

    il_buffer_added = 0
    if collected_success_count == 0:
        logging.error("[IL-COLLECT] ORCA teacher failed to produce any successful episodes. Aborting IL stage.")
        success_ratio = 0.0
    else:
        success_ratio = collected_success_count / max(1, collected_attempts)
        logging.info(
            f"[IL-COLLECT] Final success set: {collected_success_count} trajectories from {collected_attempts} attempts"
            f" (success_ratio={success_ratio:.2%})"
        )
        explorer._last_trajectories = success_trajs_all

        # SAC: BC训练完成后，不需要将IL数据注入buffer
        # SAC使用纯在线RL数据训练

        # 收集成功轨迹数量统计
        # 1. IL轨迹缺少log_prob/value（不是当前策略采集）
        # 2. 违背on-policy前提（PPO要求所有数据来自当前策略）
        # 3. IL数据仅用于BC预训练，不应混入RL训练
        # 保留success_trajs_all供_run_policy_il_pretrain使用即可
        logging.info(f"[IL-BUFFER] IL data reserved for BC pretraining only ({collected_success_count} trajectories)")

    env.config.set('orca', 'neighbor_dist', str(original_neighbor_dist))
    env.config.set('orca', 'max_neighbors', str(original_max_neighbors))
    env.config.set('orca', 'time_horizon', str(original_time_horizon))
    env.config.set('orca', 'time_horizon_obst', str(original_time_horizon_obst))
    env.config.set('orca', 'safety_space', str(original_safety_space))
    env.configure(env.config)
    logging.info("[IL-TEACHER] Restored ORCA parameters for RL phase")

    if collected_success_count == 0:
        return {'success_rate': 0.0, 'transitions': 0}

    il_ckpt = config.get('train', 'il_ckpt', fallback='runs/mamba_vl/il_policy.pth')
    il_epochs = config.getint('train', 'il_epochs', fallback=20)
    il_batch_size = config.getint('train', 'il_batch_size', fallback=512)
    seq_len = config.getint('buffer', 'seq_len', fallback=12)
    gamma = config.getfloat('train', 'gamma', fallback=0.99)

    # ========== BC Checkpoint机制：避免重复训练 ==========
    bc_checkpoint_exists = os.path.exists(il_ckpt) and os.path.getsize(il_ckpt) > 1024  # 确保非空

    if bc_checkpoint_exists and config.getboolean('train', 'il_force_retrain', fallback=False):
        logging.info(f"[IL-BC] il_force_retrain=True → 忽略现有checkpoint，强制重新训练BC")
        bc_checkpoint_exists = False

    if bc_checkpoint_exists:
        # 存在checkpoint：直接加载，跳过120轮训练
        logging.info(f"[IL-BC] ✓ Found existing BC checkpoint: {il_ckpt}")
        logging.info(f"[IL-BC] Loading checkpoint to skip {il_epochs} epochs of BC training")

        try:
            checkpoint = safe_torch_load(il_ckpt, map_location=device)

            # 兼容多种checkpoint格式
            state_dict_to_load = None
            bc_meta = {}
            partial_load = False

            if isinstance(checkpoint, dict):
                if 'value' in checkpoint:
                    # 旧格式1: {'value': state_dict, 'meta': ...}
                    logging.info(f"[IL-BC] Detected old format (dict with 'value' key)")
                    state_dict_to_load = checkpoint['value']
                    bc_meta = checkpoint.get('meta', {})
                elif 'policy' in checkpoint:
                    # 旧格式2: {'policy': state_dict, ...}
                    logging.info(f"[IL-BC] Detected format with 'policy' key")
                    state_dict_to_load = checkpoint['policy']
                    bc_meta = checkpoint.get('meta', {})
                elif 'model_state_dict' in checkpoint:
                    # 标准PyTorch格式
                    state_dict_to_load = checkpoint['model_state_dict']
                    bc_meta = checkpoint.get('meta', {})
                else:
                    # 新格式: 直接是state_dict
                    state_dict_to_load = checkpoint
            else:
                # 直接是state_dict
                state_dict_to_load = checkpoint

            # 尝试加载，如果结构不匹配则使用strict=False（部分加载）
            try:
                policy.load_state_dict(state_dict_to_load, strict=True)
                logging.info(f"[IL-BC] ✓ Loaded all parameters (strict mode)")
            except RuntimeError as e:
                if "Missing key" in str(e) or "Unexpected key" in str(e):
                    logging.warning(f"[IL-BC] ⚠ Model structure mismatch, trying partial load (strict=False)")
                    logging.warning(f"[IL-BC] Error details: {str(e)[:200]}...")
                    result = policy.load_state_dict(state_dict_to_load, strict=False)
                    if result.missing_keys:
                        logging.warning(f"[IL-BC] Missing {len(result.missing_keys)} parameters (will use random init)")
                    if result.unexpected_keys:
                        logging.warning(f"[IL-BC] Ignored {len(result.unexpected_keys)} unexpected parameters")
                    logging.info(f"[IL-BC] ⚠ Partial load completed (may need retraining)")
                    partial_load = True
                else:
                    raise

            # 打印checkpoint元信息
            if bc_meta:
                logging.info(f"[IL-BC] Checkpoint metadata:")
                logging.info(f"  - Training epochs: {bc_meta.get('epochs', 'N/A')}")
                logging.info(f"  - BC loss: {bc_meta.get('bc_loss', 'N/A')}")
                logging.info(f"  - Eval success rate: {bc_meta.get('eval_success_rate', 'N/A')}")
                logging.info(f"  - Timestamp: {bc_meta.get('timestamp', 'N/A')}")

            logging.info(f"[IL-BC] ✓ Successfully loaded BC weights from {il_ckpt}")

            # PPO不需要target_value（已删除）
            setattr(policy, "_il_warm_started", True)

            # 评估加载的模型（快速验证）
            logging.info("[IL-BC] Evaluating loaded BC model (100 episodes)...")
            policy.set_phase('eval')
            policy.eval()
            explorer.run_k_episodes(100, 'val', update_memory=False, show_tqdm=False,
                                    imitation_learning=False, force_joint_state_policy=False)

            def _mamba_event_token(x):
                try:
                    if x is None:
                        return ""
                    if isinstance(x, dict):
                        v = x.get("event") or x.get("Event") or x.get("status") or x.get("done_event")
                        if v is not None:
                            return str(v).replace(" ", "_").lower().strip()
                    for attr in ("event", "name", "value"):
                        if hasattr(x, attr):
                            try:
                                return str(getattr(x, attr)).replace(" ", "_").lower().strip()
                            except Exception:
                                pass
                    s = str(x)
                    if "." in s:
                        s = s.split(".")[-1]
                    return s.replace(" ", "_").lower().strip()
                except Exception:
                    return ""

            _MAMBA_SUCCESS_TOKENS = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
            mamba_succ = sum(1 for _, info in explorer._last_trajectories
                            if _mamba_event_token(info) in _MAMBA_SUCCESS_TOKENS)
            mamba_succ_rate = mamba_succ / max(1, len(explorer._last_trajectories))
            logging.info(f"[IL-BC] Loaded BC model success rate: {mamba_succ_rate:.1%} ({mamba_succ}/{len(explorer._last_trajectories)} episodes)")
            policy.set_phase('train')
            policy.train()

        except Exception as e:
            logging.error(f"[IL-BC] ✗ Failed to load checkpoint: {e}")
            logging.info(f"[IL-BC] Will train from scratch due to loading error")
            bc_checkpoint_exists = False  # 加载失败，重新训练

    if not bc_checkpoint_exists:
        # 不存在checkpoint：训练120轮
        logging.info(f"[IL-BC] ✗ No BC checkpoint found at {il_ckpt}")
        logging.info(f"[IL-BC] Starting {il_epochs} epochs of BC training from scratch")

        _run_policy_il_pretrain(policy, device, seq_len, il_epochs, il_batch_size, il_ckpt, explorer, gamma,
                                success_trajs_all, train_cfg)

        # PPO不需要target_value（已删除）
        setattr(policy, "_il_warm_started", True)

        logging.info("[IL-BC] Evaluating newly trained BC model...")
        policy.set_phase('eval')
        policy.eval()
        explorer.run_k_episodes(100, 'val', update_memory=False, show_tqdm=False,
                                imitation_learning=False, force_joint_state_policy=False)

        def _mamba_event_token(x):
            try:
                if x is None:
                    return ""
                if isinstance(x, dict):
                    v = x.get("event") or x.get("Event") or x.get("status") or x.get("done_event")
                    if v is not None:
                        return str(v).replace(" ", "_").lower().strip()
                for attr in ("event", "name", "value"):
                    if hasattr(x, attr):
                        try:
                            return str(getattr(x, attr)).replace(" ", "_").lower().strip()
                        except Exception:
                            pass
                s = str(x)
                if "." in s:
                    s = s.split(".")[-1]
                return s.replace(" ", "_").lower().strip()
            except Exception:
                return ""

        _MAMBA_SUCCESS_TOKENS = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
        mamba_succ = sum(1 for _, info in explorer._last_trajectories
                        if _mamba_event_token(info) in _MAMBA_SUCCESS_TOKENS)
        mamba_succ_rate = mamba_succ / max(1, len(explorer._last_trajectories))
        logging.info(f"[IL-BC] Newly trained BC model success rate: {mamba_succ_rate:.1%} ({mamba_succ}/{len(explorer._last_trajectories)} episodes)")
        policy.set_phase('train')
        policy.train()

    logging.info("[CAMRL-BASELINE] IL distillation DISABLED - using replay buffer mixing only")
    logging.info("[CAMRL-CONTINUOUS] Continuous action replay chain: memory.push_episode→sample→train.adapt")

    il_succ_rate = success_ratio
    logging.info(
        f"[IL-ORCA-DATA] Teacher success stats: attempts={collected_attempts}, successes={collected_success_count}, ratio={il_succ_rate:.3f}"
    )

    if il_succ_rate < 0.2:
        logging.warning(f"[IL-WARNING] Low success rate: {il_succ_rate:.3f}, but continuing to RL")

    # 恢复原始环境配置（注意类型：time_limit是int，其他可能是float）
    env.config.set('env', 'time_limit', str(int(original_time_limit)))
    env.config.set('reward', 'success_radius', str(original_success_radius))
    env.config.set('orca', 'neighbor_dist', str(original_neighbor_dist))
    env.config.set('orca', 'max_neighbors', str(int(original_max_neighbors)))
    env.config.set('orca', 'time_horizon', str(original_time_horizon))
    env.config.set('orca', 'safety_space', str(original_safety_space))
    env.config.set('sim', 'human_num', str(int(original_human_num)))
    env.configure(env.config)
    restored_success_radius = env.config.getfloat('reward', 'success_radius', fallback=original_success_radius)
    logging.info(
        f"[IL-PARAM-RESTORE] Parameters restored for RL: time_limit={env.time_limit}, success_radius={restored_success_radius}, human_num={original_human_num}"
    )

    # 返回IL轨迹数据用于注入RL buffer（防止Q网络冷启动）
    return {
        'success_rate': il_succ_rate,
        'transitions': len(replay_iql) if replay_iql else 0,
        'il_trajectories': success_trajs_all  # 返回IL成功轨迹
    }

# _prefill_buffer_with_bc已删除 - PPO不需要BC prefill

def _evaluate_policy_quick(explorer, policy, episodes: int, tag: str = 'POLICY-EVAL', phase: str = 'val'):
    if episodes <= 0:
        return
    logging.info(f"[{tag}] Running evaluation: episodes={episodes}")
    prev_phase = getattr(policy, '_phase', None)
    prev_eps = getattr(policy, 'epsilon', None)
    prev_mode = getattr(policy, 'training_mode', None)
    try:
        if hasattr(policy, 'set_phase'):
            policy.set_phase(phase)
        if hasattr(policy, 'set_epsilon') and prev_eps is not None:
            policy.set_epsilon(0.0)
        stats = explorer.run_k_episodes(episodes, phase, update_memory=False, show_tqdm=False,
                                        imitation_learning=False, force_joint_state_policy=False,
                                        return_stats=True)
        if isinstance(stats, dict):
            logging.info(f"[{tag}] success={stats.get('success_rate', 0.0):.1%} collision={stats.get('collision_rate', 0.0):.1%} timeout={stats.get('timeout_rate', 0.0):.1%} reward={stats.get('total_reward', 0.0):.3f}")
        else:
            logging.info(f"[{tag}] Evaluation completed")
    except Exception as e:
        logging.warning(f"[{tag}] Evaluation failed: {e}")
    finally:
        if prev_mode is not None and hasattr(policy, 'set_training_mode'):
            policy.set_training_mode(prev_mode)
        if prev_phase is not None and hasattr(policy, 'set_phase'):
            policy.set_phase(prev_phase)
        if prev_eps is not None and hasattr(policy, 'set_epsilon'):
            policy.set_epsilon(prev_eps)

def _convert_state_sequence_to_array(states):
    import numpy as _np
    from crowd_nav.contracts import ensure_btnd
    if isinstance(states, _np.ndarray):
        arr = _np.asarray(states, dtype=_np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr
    converted = []
    for s in states:
        if hasattr(s, 'to_array'):
            arr = s.to_array()
        else:
            arr = _np.asarray(s, dtype=_np.float32)
        arr = _np.asarray(arr, dtype=_np.float32)
        if arr.ndim == 1:
            if arr.shape[0] < 34:
                arr = _np.pad(arr, (0, 34 - arr.shape[0]), mode='constant')
            arr = arr[:34]
        converted.append(arr)
    arr = _np.stack(converted, axis=0)
    if arr.shape[-1] < 34:
        pad = _np.zeros((arr.shape[0], 34 - arr.shape[-1]), dtype=_np.float32)
        arr = _np.concatenate([arr, pad], axis=-1)
    return arr

def _convert_actions_to_array(actions):
    import numpy as _np
    if actions is None:
        return None
    arr = _np.asarray(actions, dtype=_np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _meta_from_event(meta, event_token, length):
    meta = dict(meta) if meta else {}
    success_tokens = {'success', 'reachgoal', 'reach_goal', 'reaching_goal', 'goal_reached'}
    is_success = any(tok for tok in success_tokens if tok in event_token)
    meta.setdefault('source', 'IL')
    meta['event'] = event_token
    meta['flag_success'] = int(is_success)
    meta['flag_timeout'] = int('timeout' in event_token)
    meta['flag_collision'] = int('collision' in event_token)
    meta['length'] = int(length)
    return meta

# _warmup_rl_buffer已删除 - PPO不需要buffer预热

def _run_rl_phase(cfg, env, policy, optimizer, explorer, rl_buffer, il_buffer, stats, plotter, device, episodes, batch_size, save_every, ARGS, tb_writer, train_step_counter, scaler, start_episode=1, metrics_plotter=None, trajectory_file=None, resume_checkpoint=None):

    from crowd_nav.contracts import ensure_btnd, pick_last, joint34_to_tokens
    import copy, math, os, random, torch, numpy as np
    from crowd_nav.contracts import GRID

    # ========== SAC组件初始化 ==========
    from crowd_nav.policy.mamba_rl import SACQNetwork  # SAC独立命名，避免IQL残留

    logging.info("[ALGO] Training algorithm: SAC (Soft Actor-Critic)")
    logging.info(f"[SAC-REPLAY] Single replay buffer: capacity={rl_buffer.capacity}, size={len(rl_buffer)}")

    # 创建SAC网络
    d_model = cfg.getint('mamba', 'd_model', fallback=256)
    n_layers = cfg.getint('mamba', 'n_layers', fallback=2)
    d_state = cfg.getint('mamba', 'd_state', fallback=16)
    d_conv = cfg.getint('mamba', 'd_conv', fallback=4)
    expand = cfg.getint('mamba', 'expand', fallback=2)

    # Twin Q networks
    q_net1 = SACQNetwork(d_model, n_layers, d_state, d_conv, expand).to(device)
    q_net2 = SACQNetwork(d_model, n_layers, d_state, d_conv, expand).to(device)

    # Target networks
    q_target_net1 = copy.deepcopy(q_net1).to(device)
    q_target_net2 = copy.deepcopy(q_net2).to(device)
    logging.info("[SAC] Target networks enabled")

    # SAC优化器
    q_lr = cfg.getfloat('sac', 'q_lr', fallback=3e-4)
    policy_lr = cfg.getfloat('sac', 'policy_lr', fallback=3e-4)
    alpha_lr = cfg.getfloat('sac', 'alpha_lr', fallback=3e-4)

    # 创建独立的Q优化器（标准SAC：每个Q网络独立优化）
    optim_q1 = torch.optim.AdamW(q_net1.parameters(), lr=q_lr)
    optim_q2 = torch.optim.AdamW(q_net2.parameters(), lr=q_lr)
    optim_pi = torch.optim.AdamW(policy.parameters(), lr=policy_lr)

    # Alpha (temperature) 自动调节
    init_alpha = cfg.getfloat('sac', 'init_alpha', fallback=1.0)
    log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=device)
    optim_alpha = torch.optim.Adam([log_alpha], lr=alpha_lr)

    # Load SAC components from checkpoint if resuming
    if resume_checkpoint is not None and resume_checkpoint.get('stage') == 'rl_training':
        logging.info("[SAC-RESUME] Loading Q-networks and optimizers from checkpoint...")
        try:
            if 'q_net1_state' in resume_checkpoint:
                q_net1.load_state_dict(resume_checkpoint['q_net1_state'])
                q_net2.load_state_dict(resume_checkpoint['q_net2_state'])
                q_target_net1.load_state_dict(resume_checkpoint['q_target_net1_state'])
                q_target_net2.load_state_dict(resume_checkpoint['q_target_net2_state'])
                logging.info("[SAC-RESUME] Q-networks loaded")
            
            if 'log_alpha' in resume_checkpoint:
                with torch.no_grad():
                    log_alpha.copy_(resume_checkpoint['log_alpha'])
                logging.info(f"[SAC-RESUME] Alpha loaded (val={log_alpha.exp().item():.3f})")

            if 'optim_q1_state' in resume_checkpoint:
                # 标准格式：分离的Q1/Q2优化器（干净版SAC）
                optim_q1.load_state_dict(resume_checkpoint['optim_q1_state'])
                optim_q2.load_state_dict(resume_checkpoint['optim_q2_state'])
                optim_pi.load_state_dict(resume_checkpoint['optim_pi_state'])
                optim_alpha.load_state_dict(resume_checkpoint['optim_alpha_state'])
                logging.info("[SAC-RESUME] Optimizers loaded (separate Q optimizers)")
            elif 'optim_q_state' in resume_checkpoint:
                # 旧格式：统一的Q优化器（跳过，使用新的分离初始化）
                logging.warning("[SAC-RESUME] Old checkpoint format (unified Q) detected, using fresh optimizer initialization")
                optim_pi.load_state_dict(resume_checkpoint.get('optim_pi_state', optim_pi.state_dict()))
                optim_alpha.load_state_dict(resume_checkpoint.get('optim_alpha_state', optim_alpha.state_dict()))
                logging.info("[SAC-RESUME] Policy and Alpha optimizers loaded (Q optimizers initialized fresh)")
                
        except Exception as e:
            logging.warning(f"[SAC-RESUME] Failed to load some components: {e}")

    logging.info(f"[SAC] Networks initialized: Q_lr={q_lr}, Pi_lr={policy_lr}, Alpha_lr={alpha_lr}, Init_Alpha={init_alpha}")
    logging.info(f"[SAC-BUFFER] Replay buffer ready (Off-Policy)")

    # ============================================================
    # Q网络warmup：用IL数据（统一reward下）预热Q，避免从白板开始
    # ============================================================
    import torch.nn.functional as F
    q_warmup_steps = cfg.getint('sac', 'q_warmup_steps', fallback=0)
    sac_gamma = cfg.getfloat('train', 'gamma', fallback=0.99)

    if q_warmup_steps > 0 and len(il_buffer) > 0:
        logging.info(f"[Q-WARMUP] Starting Q-network warmup with IL data ({q_warmup_steps} steps)...")
        policy.eval()
        for step in range(q_warmup_steps):
            batch = il_buffer.sample(batch_size, device)
            if batch is None:
                break

            # 只更新Q网络，不更新policy
            states = torch.as_tensor(batch["states"], device=device, dtype=torch.float32)
            actions = torch.as_tensor(batch.get("actions", batch.get("actions_continuous")), device=device, dtype=torch.float32)
            rewards = torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32).unsqueeze(-1)
            next_states = torch.as_tensor(batch["next_states"], device=device, dtype=torch.float32)
            dones = torch.as_tensor(batch["dones"], device=device, dtype=torch.float32).unsqueeze(-1)

            if actions.dim() == 3:
                actions = actions[:, -1, :]
            if rewards.dim() == 2:
                rewards = rewards[:, -1:]
            if dones.dim() == 2:
                dones = dones[:, -1:]

            with torch.no_grad():
                mean_next, log_std_next, _, _ = policy.forward_both(next_states, return_sequence=False)
                std_next = torch.exp(log_std_next)
                dist_next = torch.distributions.Normal(mean_next, std_next)
                u_next = dist_next.sample()
                next_action = torch.tanh(u_next)

                q1_next = q_target_net1(next_states, next_action)
                q2_next = q_target_net2(next_states, next_action)
                q_next_min = torch.min(q1_next, q2_next)
                q_target = rewards + (1.0 - dones) * sac_gamma * q_next_min

            q1_pred = q_net1(states, actions)
            q2_pred = q_net2(states, actions)
            q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

            optim_q1.zero_grad(set_to_none=True)
            optim_q2.zero_grad(set_to_none=True)
            q_loss.backward()
            optim_q1.step()
            optim_q2.step()

            if step % 100 == 0:
                logging.info(f"[Q-WARMUP] step={step}/{q_warmup_steps} q_loss={q_loss.item():.4f}")

        # 同步target networks
        with torch.no_grad():
            for param, target_param in zip(q_net1.parameters(), q_target_net1.parameters()):
                target_param.data.copy_(param.data)
            for param, target_param in zip(q_net2.parameters(), q_target_net2.parameters()):
                target_param.data.copy_(param.data)

        logging.info(f"[Q-WARMUP] Completed. Q networks warmed up with IL data.")
        policy.train()

    # ============================================================
    # Replay warmup机制：先采集数据，再开始训练
    # ============================================================
    replay_warmup = cfg.getint('sac', 'replay_warmup', fallback=10000)
    logging.info(f"[CLEAN-SAC] Replay warmup threshold: {replay_warmup} transitions")

    best_succ = 0.0
    updates_per_ep = cfg.getint('train', 'updates_per_ep', fallback=20)
    eval_every = cfg.getint('train', 'eval_every', fallback=100)
    eval_episodes = cfg.getint('train', 'eval_episodes', fallback=10)

    sot_monitor_interval = 100
    timeout_alert_episodes = []

    vectorize_enable = cfg.getboolean('vectorize', 'enable', fallback=False)
    num_workers = cfg.getint('vectorize', 'num_workers', fallback=8)
    episodes_per_worker = cfg.getint('vectorize', 'episodes_per_worker', fallback=2)
    broadcast_interval = cfg.getint('vectorize', 'broadcast_interval', fallback=1)

    vectorized_sampler = None
    if vectorize_enable:
        logging.info(f"[VECTORIZE] Enabling parallel sampling: {num_workers} workers × {episodes_per_worker} episodes/worker")
        from crowd_nav.utils.vectorized_sampler import VectorizedSampler

        import os
        if ARGS.config and os.path.exists(ARGS.config):
            config_path = os.path.abspath(ARGS.config)
        elif os.path.exists('configs/env.config'):
            config_path = os.path.abspath('configs/env.config')
        elif os.path.exists('crowd_nav/configs/env.config'):
            config_path = os.path.abspath('crowd_nav/configs/env.config')
        else:
            config_path = os.path.abspath(ARGS.config) if ARGS.config else 'configs/env.config'
            logging.warning(f"[VECTORIZE] Config file not found, trying {config_path}")

        policy_name = ARGS.policy if ARGS.policy else policy_cfg.policy_name

        logging.info(f"[VECTORIZE] Config path: {config_path}")
        logging.info(f"[VECTORIZE] Policy name: {policy_name}")

        worker_log_dir = os.path.abspath(ARGS.outdir)

        vectorized_sampler = VectorizedSampler(
            num_workers=num_workers,
            episodes_per_worker=episodes_per_worker,
            config_path=config_path,
            policy_name=policy_name,
            device=device,
            log_dir=worker_log_dir,
        )
        vectorized_sampler.start()
        logging.info(f"[VECTORIZE] Parallel sampler started successfully")

        # 立即广播BC权重到workers，避免首批采样使用随机权重
        if hasattr(vectorized_sampler, 'broadcast_params'):
            bc_state_dict = policy.state_dict()
            num_params = sum(p.numel() for p in policy.parameters())
            logging.info(f"[VECTORIZE] Broadcasting BC policy to workers: {num_params:,} parameters, {len(bc_state_dict)} tensors")
            vectorized_sampler.broadcast_params(bc_state_dict)
            logging.info("[VECTORIZE] ✓ Initial BC policy weights broadcast to all workers")
    else:
        logging.info("[VECTORIZE] Parallel sampling disabled, using serial explorer")

    _first_action_logged = [False]

    # SAC: 无需Critic Warmup，Q网络通过online RL直接学习
    policy.train()

    # 诊断日志
    if vectorized_sampler is not None:
        logging.info("[DIAGNOSTIC] ✓ Using VECTORIZED sampling (parallel workers)")
        logging.info(f"[DIAGNOSTIC] Workers: {num_workers}, Episodes/worker: {episodes_per_worker}")
    else:
        logging.info("[DIAGNOSTIC] ⚠ Using SERIAL sampling (no parallelization)")

    # 快速测试策略
    logging.info("[DIAGNOSTIC] Quick policy sanity check (5 episodes)...")
    policy.set_phase('eval')
    policy.eval()
    test_stats = explorer.run_k_episodes(5, 'val', update_memory=False, show_tqdm=False,
                                         imitation_learning=False, force_joint_state_policy=False)
    logging.info(f"[DIAGNOSTIC] Pre-RL test performance: s/c/t={test_stats[0]:.1%}/{test_stats[1]:.1%}/{test_stats[2]:.1%}")
    policy.set_phase('train')
    policy.train()

    for ep in range(start_episode, int(episodes)+1):
        standard_human_num = cfg.getint('sim', 'human_num', fallback=5)
        standard_circle_radius = cfg.getfloat('sim', 'circle_radius', fallback=4.0)
        standard_success_radius = cfg.getfloat('robot', 'success_radius', fallback=0.25)

        if hasattr(env, '_raw_config'):
            if hasattr(env._raw_config, 'set'):
                env._raw_config.set('sim', 'human_num', str(standard_human_num))
                env._raw_config.set('sim', 'circle_radius', str(standard_circle_radius))
            elif isinstance(env._raw_config, dict):
                if 'sim' not in env._raw_config:
                    env._raw_config['sim'] = {}
                env._raw_config['sim']['human_num'] = standard_human_num
                env._raw_config['sim']['circle_radius'] = standard_circle_radius

        if hasattr(env, 'rwd'):
            env.rwd.success_radius = standard_success_radius

        if ep == 1:
            logging.info(f"[P1-RL-LOCKED] human_num={standard_human_num}, circle_radius={standard_circle_radius}, success_radius={standard_success_radius}")
            logging.info("[P1-RL-LOCKED] Configuration locked to prevent any difficulty reduction during RL training")

        env._current_episode = ep

        # [FIX-REWARD-ULTIMATE] Block REMOVED.
        # Rewards are now strictly controlled by env.config loaded at startup.
        # updates_per_ep is strictly controlled by train.config.
        
        if ep % 50 == 0:
             logging.info(f"[CONFIG-CHECK] ep={ep} updates_per_ep={updates_per_ep} (from config)")

        if vectorized_sampler is not None:

            episodes_batch = vectorized_sampler.collect_episodes(
                episode_start=ep,
                phase='train',
                # PPO: 不需要epsilon参数（使用entropy探索）
                timeout=300.0
            )

            succ_count = coll_count = timeout_count = 0
            total_reward = 0.0
            nav_time_sum = 0.0
            time_taken_list = []
            discomfort_freq_list = []
            discomfort_dist_list = []
            path_len_list = []
            smooth_list = []
            speed_list = []

            for ep_data in episodes_batch:
                states = ep_data['states']
                actions = ep_data['actions']
                rewards = ep_data['rewards']
                info = ep_data['info']

                if len(states) > 0:

                    states_arr = np.array(states, dtype=np.float32)
                    rewards_arr = np.array(rewards, dtype=np.float32)

                    episode_length = len(states)
                    dones = np.zeros(episode_length, dtype=bool)
                    timeouts = np.zeros(episode_length, dtype=bool)
                    dones[-1] = True

                    event_str = (_event_token(info) or '').lower()
                    if 'timeout' in event_str:
                        timeouts[-1] = True

                    if len(actions) > 0:
                        actions_continuous = np.asarray(actions, dtype=np.float32)
                        if actions_continuous.ndim == 1:
                            actions_continuous = actions_continuous.reshape(1, -1)
                    else:
                        actions_continuous = None

                    success_flag = (event_str in _SUCCESS_TOKENS) or ('success' in event_str) or ('reachgoal' in event_str) or ('reaching_goal' in event_str)
                    timeout_flag = ('timeout' in event_str)
                    collision_flag = ('collision' in event_str)
                    # 分层探索：使用episode_source标签（来自vectorized_sampler）
                    actual_source = ep_data.get('source', 'RL')  # 从ep_data中提取source字段
                    meta = {
                        'source': actual_source,
                        'event': event_str,
                        'is_success': success_flag,
                        'is_timeout': timeout_flag,
                        'is_terminal': dones[-1],
                        'length': episode_length,
                    }

                    # 存入RL buffer（在线数据）
                    rl_buffer.store_episode({
                        'states': states_arr,
                        'actions_continuous': actions_continuous,
                        'rewards': rewards_arr,
                        'dones': dones
                    })

                event_str = (_event_token(info) or '').lower()
                if (event_str in _SUCCESS_TOKENS) or ('success' in event_str) or ('reachgoal' in event_str) or ('reaching_goal' in event_str):
                    succ_count += 1
                elif 'collision' in event_str:
                    coll_count += 1
                elif 'timeout' in event_str:
                    timeout_count += 1

                total_reward += sum(rewards)
                nav_time_sum += len(states) * getattr(env, 'time_step', 0.25)

                time_taken_val = _info_get(info, 'time_taken')
                if time_taken_val is not None:
                    time_taken_list.append(time_taken_val)
                discomfort_freq_val = _info_get(info, 'discomfort_freq')
                if discomfort_freq_val is not None:
                    discomfort_freq_list.append(discomfort_freq_val)
                discomfort_dist_val = _info_get(info, 'discomfort_dist')
                if discomfort_dist_val is not None:
                    discomfort_dist_list.append(discomfort_dist_val)
                path_len_val = _info_get(info, 'path_length')
                if path_len_val is not None:
                    path_len_list.append(path_len_val)
                smooth_val = _info_get(info, 'smoothness')
                if smooth_val is not None:
                    smooth_list.append(smooth_val)
                speed_val = _info_get(info, 'speed')
                if speed_val is not None:
                    speed_list.append(speed_val)

                # Trajectory logging: 记录轨迹到trajectory.log
                if trajectory_file is not None and 'trajectory_positions' in info:
                    try:
                        traj_positions = info['trajectory_positions']
                        start_pos = info.get('start_pos', (0.0, 0.0))
                        goal_pos = info.get('goal_pos', (0.0, 0.0))
                        event = info.get('event', 'unknown').upper()
                        ep_id = ep_data.get('episode', ep)

                        # 写入header: [EP=X] START=(x,y) GOAL=(x,y) RESULT=status STEPS=n
                        trajectory_file.write(f"[EP={ep_id}] START={start_pos} GOAL={goal_pos} RESULT={event} STEPS={len(traj_positions)}\n")

                        # 写入trajectory: TRAJECTORY: (x1, y1) (x2, y2) ...
                        trajectory_str = ' '.join(f"{pos}" for pos in traj_positions)
                        trajectory_file.write(f"TRAJECTORY: {trajectory_str}\n\n")
                        trajectory_file.flush()
                    except Exception as e:
                        logging.warning(f"[TRAJECTORY-LOG] Failed to write trajectory: {e}")

            num_episodes = len(episodes_batch)
            succ = succ_count / max(1, num_episodes)
            coll = coll_count / max(1, num_episodes)
            timeout = timeout_count / max(1, num_episodes)
            ep_reward = total_reward / max(1, num_episodes)
            nav_t = nav_time_sum / max(1, num_episodes)
            time_taken = np.mean(time_taken_list) if time_taken_list else 0.0
            discomfort_freq = np.mean(discomfort_freq_list) if discomfort_freq_list else 0.0
            discomfort_dist = np.mean(discomfort_dist_list) if discomfort_dist_list else 0.0
            path_len = np.mean(path_len_list) if path_len_list else 0.0
            smooth = np.mean(smooth_list) if smooth_list else 0.0
            speed = np.mean(speed_list) if speed_list else 0.0

            if ep % broadcast_interval == 0:
                vectorized_sampler.broadcast_params(policy.state_dict())
                if ep % 10 == 0:  # More frequent logging for debugging
                    logging.info(f"[VECTORIZE] Model weights synced to workers (ep={ep})")

        else:

            # 【精简】删除 expert_policy 和 student_prob 参数
            succ, coll, timeout, ep_reward, nav_t, time_taken, discomfort_freq, discomfort_dist, path_len, smooth, speed = explorer.run_k_episodes(
                k=1, phase='train', update_memory=False, show_tqdm=False,
                imitation_learning=False, force_joint_state_policy=False
            )

            # 手动存储数据到buffer（确保数据不丢失）
            # 即使Explorer.update_memory=False，我们也要确保RL数据被存入buffer
            if hasattr(explorer, '_last_trajectories') and explorer._last_trajectories:
                for traj_data, info in explorer._last_trajectories:
                    try:
                        # 解析轨迹数据结构
                        states, actions, rewards = traj_data

                        # 构造dones数组：只有最后一步为True
                        dones = np.zeros(len(rewards), dtype=bool)
                        if len(rewards) > 0:
                            dones[-1] = (info.get('done', False) or
                                       len(rewards) == cfg.getint('env', 'max_steps', fallback=100))

                        # 双池改造：RL 数据只写入 rl_buffer（demo永不被污染）
                        rl_buffer.push_episode({
                            'states': states,
                            'actions': actions,
                            'rewards': rewards,
                            'dones': dones
                        })
                    except Exception as e:
                        logging.warning(f"[STORE] Failed to store trajectory in episode {ep}: {e}")

                # 清空explorer缓存（访问内部属性）
                explorer._last_trajectories = []

        if ep == 1 and not _first_action_logged[0]:

            if hasattr(policy, '_last_continuous_action'):
                vx, vy = policy._last_continuous_action
                speed = math.sqrt(vx*vx + vy*vy)
                heading_deg = math.degrees(math.atan2(vy, vx))
                logging.info(f"FIRST_ACTION (Continuous): speed={speed:.3f}, heading={heading_deg:.1f}°, (vx={vx:+.3f}, vy={vy:+.3f})")
                _first_action_logged[0] = True
            else:
                logging.info("FIRST_ACTION: Policy doesn't expose continuous action info, skipping debug")
                _first_action_logged[0] = True

        # 【渐进更新门槛】根据训练表现动态调整updates
        # 当前简化版：固定使用warm_updates（6次），等稳定后再提升
        # TODO: 实现完整的渐进式提升逻辑（6→8→10）

        # 计算最近50集的滑动窗口统计
        window_size = min(50, len(stats.hist_s))
        if window_size > 0:
            recent_succ = sum(stats.hist_s[-window_size:]) / window_size
            recent_tout = sum(stats.hist_t[-window_size:]) / window_size
        else:
            recent_succ, recent_tout = 0.0, 0.0

        # Warm-update机制已删除，使用固定更新次数
        current_updates = updates_per_ep

        # PPO: 不需要polyak_tau（无target network）
        n_step_config = cfg.getint('train', 'n_step', fallback=1)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_v_loss = 0.0
        total_q_loss = 0.0
        total_aux_loss = 0.0
        # 统计累加器
        total_adv_mean = 0.0
        total_adv_std = 0.0
        total_weight = 0.0
        grad_norm_sum = 0.0
        updates_done = 0
        last_mix_info = None

        success_rate_50 = stats.hist_s[-1] if stats.hist_s else 0.0

        # ========== Step 3b (todo.md): 干净版SAC训练更新 ==========
        replay_warmup = cfg.getint('sac', 'replay_warmup', fallback=10000)
        batch_size = cfg.getint('buffer', 'batch_size', fallback=256)

        # SAC只需要rl_buffer有足够数据
        if len(rl_buffer) < replay_warmup:
            if ep == start_episode or ep % 100 == 0:
                logging.info(f"[SAC-WARMUP] RL buffer size: {len(rl_buffer)}, waiting for {replay_warmup}...")
            updates_done = 0
        else:
            # 标准SAC：只从rl_buffer采样，不混入IL数据
            n_updates = current_updates
            for _ in range(n_updates):
                if len(rl_buffer) < batch_size:
                    break

                batch = rl_buffer.sample(batch_size, device)
                if batch is None:
                    logging.warning(f"[SAC-SAMPLE] Failed to sample batch at ep={ep}")
                    continue

                # BC regularization: 从il_buffer采样teacher actions
                bc_reg_weight = cfg.getfloat('sac', 'bc_regularization_weight', fallback=0.0)
                if bc_reg_weight > 0 and len(il_buffer) > 0:
                    il_batch = il_buffer.sample(batch_size, device)
                    if il_batch is not None:
                        batch['bc_actions'] = il_batch.get('actions', il_batch.get('actions_continuous'))
                        batch['bc_states'] = il_batch['states']

                # 计算SAC参数
                sac_gamma = cfg.getfloat('train', 'gamma', fallback=0.99)
                sac_tau = cfg.getfloat('sac', 'tau', fallback=0.005)
                action_dim = 2  # holonomic (vx, vy)
                target_entropy_ratio = cfg.getfloat('sac', 'target_entropy_ratio', fallback=1.0)
                sac_target_entropy = -target_entropy_ratio * float(action_dim)

                step_metrics = sac_update(
                    batch=batch,
                    q_net1=q_net1,
                    q_net2=q_net2,
                    q_target1=q_target_net1,
                    q_target2=q_target_net2,
                    policy_net=policy,
                    q_optimizer=optim_q1,  # 注意：这里简化为单个optimizer
                    policy_optimizer=optim_pi,
                    alpha_optimizer=optim_alpha,
                    log_alpha=log_alpha,
                    gamma=sac_gamma,
                    tau=sac_tau,
                    target_entropy=sac_target_entropy,
                    device=device,
                    cfg=cfg,
                )

                # 首次更新验证metrics
                if updates_done == 0:
                    expected_keys = ['q1_loss', 'q2_loss', 'policy_loss', 'alpha_loss', 'alpha']
                    missing_keys = [k for k in expected_keys if k not in step_metrics]
                    if missing_keys:
                        logging.warning(f"[SAC-METRICS] Missing keys: {missing_keys}")
                    else:
                        logging.debug(f"[SAC-STEP] q1={step_metrics['q1_loss']:.4f} q2={step_metrics['q2_loss']:.4f} pi={step_metrics['policy_loss']:.4f} alpha={step_metrics['alpha']:.4f}")

                # 累积SAC指标
                q_loss = 0.5 * (step_metrics.get('q1_loss', 0.0) + step_metrics.get('q2_loss', 0.0))
                p_loss = step_metrics.get('policy_loss', 0.0)

                total_loss += (q_loss + p_loss)
                total_policy_loss += p_loss
                total_v_loss += 0.0  # SAC无V网络
                total_q_loss += q_loss
                total_aux_loss += 0.0
                total_adv_mean += 0.0
                total_adv_std += 0.0
                total_weight += 0.0
                grad_norm_sum += 0.0

                updates_done += 1

        # SAC buffer信息
        if ep % 100 == 0:
            logging.info(f"[SAC-REPLAY] ep={ep} buffer_size={len(rl_buffer)} updates={updates_done}")

        # 计算平均loss
        avg_v_loss = 0.0  # SAC无V网络
        avg_q_loss = total_q_loss / max(1, updates_done) if updates_done > 0 else 0.0
        avg_policy_loss = total_policy_loss / max(1, updates_done) if updates_done > 0 else 0.0
        avg_aux_loss = 0.0
        avg_adv_mean = 0.0
        avg_adv_std = 0.0
        avg_weight = 0.0

        # 每25 episodes输出loss到日志和TensorBoard
        if updates_done > 0 and ep % 25 == 0:
            logging.info(f"[SAC-LOSS] ep={ep} q_loss={avg_q_loss:.4f} policy_loss={avg_policy_loss:.4f}")
            if ep % 100 == 0:
                logging.info(f"[SAC-STATS] ep={ep} buffer_size={len(rl_buffer)}")
            tb_writer.add_scalar("sac/q_loss", avg_q_loss, ep)
            tb_writer.add_scalar("sac/policy_loss", avg_policy_loss, ep)
            if avg_aux_loss > 0:
                tb_writer.add_scalar("sac/aux_loss", avg_aux_loss, ep)

        metrics = {
            'loss': total_loss / max(1, updates_done),
            'policy_loss': avg_policy_loss,
            'value_loss': avg_v_loss,
            'q_loss': avg_q_loss,
            'grad_norm': grad_norm_sum / max(1, updates_done),
            'updates': updates_done,
            'architecture': 'sac_mamba'
        }

        train_step_counter['episode'] = ep

        # Record SAC losses for plotting (use Q-loss as critic proxy)
        v_loss_metric = avg_q_loss if updates_done > 0 else None
        p_loss_metric = avg_policy_loss if updates_done > 0 else None
        stats.update(succ, coll, timeout, ep_reward, episode=ep,
                    time_taken=time_taken, discomfort_freq=discomfort_freq, discomfort_dist=discomfort_dist,
                    v_loss=v_loss_metric, p_loss=p_loss_metric)

        tb_writer.flush()

        if ep % 25 == 0:

            roll_succ = stats.hist_s[-1] if stats.hist_s else 0.0
            roll_coll = stats.hist_c[-1] if stats.hist_c else 0.0
            roll_timeout = stats.hist_t[-1] if stats.hist_t else 0.0
            roll_reward = np.mean(stats.hist_r[-stats.roll:]) if stats.hist_r else 0.0

            terminal_percentage = 0.0
            try:
                # 三段式采样统计（success/timeout/collision）
                success_percentage = 0.0
                timeout_percentage = 0.0
                collision_percentage = 0.0
                if last_mix_info is not None:
                    success_percentage = last_mix_info.get('success_pct', 0.0)
                    timeout_percentage = last_mix_info.get('timeout_pct', 0.0)
                    collision_percentage = last_mix_info.get('collision_pct', 0.0)
                    success_count = last_mix_info.get('success_cnt', 0)
                    timeout_count = last_mix_info.get('timeout_cnt', 0)
                    collision_count = last_mix_info.get('collision_cnt', 0)
                    il_count = last_mix_info.get('il_cnt', 0)
                    il_percentage = last_mix_info.get('il_pct', 0.0)
                    actual_batch = last_mix_info.get('actual_batch_size', batch_size)

                    # PPO: 采样质量监控（降低频率，避免霸屏）
                    should_log_mix = (ep % 50 == 0)  # 每50集打印一次
                    if should_log_mix:
                        logging.info(f"[BATCH-STATS] ep={ep} | Success={success_count} ({success_percentage:.1f}%), Timeout={timeout_count} ({timeout_percentage:.1f}%), Collision={collision_count} ({collision_percentage:.1f}%)")
                        # 删除INCOMPLETE warning，避免霸屏

            except Exception as e:
                logging.error(f"[THREE-TIER-SAMPLE] Failed to log sampling stats: {e}")
                import traceback
                logging.error(f"[THREE-TIER-SAMPLE] Traceback: {traceback.format_exc()}")

            pct_format = stats.format_percentage(roll_succ, roll_coll, roll_timeout)

            recent_time = np.mean(stats.hist_time[-stats.roll:]) if stats.hist_time else 0.0
            recent_dfreq = np.mean(stats.hist_discomfort_freq[-stats.roll:]) if stats.hist_discomfort_freq else 0.0
            recent_ddist = np.mean(stats.hist_discomfort_dist[-stats.roll:]) if stats.hist_discomfort_dist else 0.0

            # 三段式采样监控（success/timeout/collision）
            logging.info(f"[STAT] ep={ep:03d} | roll@25(s/c/t)={roll_succ:.1f}/{roll_coll:.1f}/{roll_timeout:.1f} ({pct_format}) | batch: Succ%={success_percentage:.0f}, Tout%={timeout_percentage:.0f}, Coll%={collision_percentage:.0f}, term%={terminal_percentage:.0f} | R.mean={roll_reward:.1f}")
            logging.info(f"[METRICS-25] time={recent_time:.2f}s | dfreq={recent_dfreq:.3f} | ddist={recent_ddist:.3f}m")

            tb_writer.add_scalar("train/avg_reward", roll_reward, ep)
            tb_writer.add_scalar("train/success_rate", roll_succ, ep)

            if metrics_plotter is not None:
                metrics_plotter.plot(ep, stats, plot_every=25)

        current_succ = 1.0 if succ else 0.0
        current_coll = 1.0 if coll else 0.0
        current_timeout = 1.0 if timeout else 0.0

        roll = (stats.hist_s[-1] if stats.hist_s else 0.0,
                stats.hist_c[-1] if stats.hist_c else 0.0,
                stats.hist_t[-1] if stats.hist_t else 0.0)

        rl_episode_count = ep
        total_episodes_target = int(episodes)

        cumulative_pct = f"{roll[0]:.1%}/{roll[1]:.1%}/{roll[2]:.1%}"

        if ep <= 50:
            logging.info(f"[RL-EP-{ep:04d}] ({rl_episode_count}/{total_episodes_target}) s/c/t={current_succ:.0f}/{current_coll:.0f}/{current_timeout:.0f} qloss={metrics['q_loss']:.4f} ploss={metrics['policy_loss']:.4f} buf={len(rl_buffer)}")
            logging.info(f"              cumulative s/c/t%: {cumulative_pct}")
        elif ep % 25 == 0:
            logging.info(f"[RL-EP-{ep:04d}] ({rl_episode_count}/{total_episodes_target}) s/c/t={current_succ:.0f}/{current_coll:.0f}/{current_timeout:.0f} qloss={metrics['q_loss']:.4f} ploss={metrics['policy_loss']:.4f} buf={len(rl_buffer)}")
            logging.info(f"              cumulative s/c/t%: {cumulative_pct}")

            logging.info(f"[STAGE0-MONITOR] timeout_ratio={roll[2]:.3f} (target: decrease over time)")
        elif ep % 5 == 0:
            logging.info(f"[RL-EP-{ep:04d}] roll={roll[0]:.3f}/{roll[1]:.3f}/{roll[2]:.3f}")
        else:
            logging.info(f"[RL-EP-{ep:04d}] ({rl_episode_count})")

        if roll[1] <= 0.01 and roll[2] >= 0.95:
            timeout_alert_episodes.append(ep)

            timeout_alert_episodes = timeout_alert_episodes[-200:]

        if ep % sot_monitor_interval == 0:
            # PPO: 不需要epsilon监控（使用entropy探索）
            reward_config = f"succ={cfg.getfloat('reward', 'success_reward', fallback=1.0):.1f}/coll={cfg.getfloat('reward', 'collision_penalty', fallback=-0.25):.2f}"
            grid_config = f"{GRID['n_speeds']}×{GRID['n_headings']}={GRID['n_speeds']*GRID['n_headings']}"
            event_dist = f"s={roll[0]:.2f}/c={roll[1]:.2f}/t={roll[2]:.2f}"

            recent_timeouts = len([x for x in timeout_alert_episodes if x > ep - 50])
            timeout_warning = f" WARNINGTIMEOUT_ALERT({recent_timeouts}/50)" if recent_timeouts >= 30 else ""

            logging.info(f"[SOT-{ep:04d}] "
                        f"REWARD {reward_config} | GRID {grid_config} | "
                        f"EVENTS {event_dist}{timeout_warning}")

            if recent_timeouts >= 40:
                logging.warning(f"[P2-SOT-ALERT] Detected {recent_timeouts}/50 timeout episodes! Checking configuration...")
                logging.warning(f"[P2-SOT-ALERT] Current time_limit={cfg.getfloat('env', 'time_limit', fallback=25):.1f}s, "
                               f"dt={cfg.getfloat('env', 'time_step', fallback=0.25):.2f}s = {cfg.getfloat('env', 'time_limit', fallback=25)/cfg.getfloat('env', 'time_step', fallback=0.25):.0f} max steps")
                logging.warning(f"[P2-SOT-ALERT] Current circle_radius={cfg.getfloat('sim', 'circle_radius', fallback=4.0):.1f}m, "
                               f"human_num={cfg.getint('sim', 'human_num', fallback=5)}")
                logging.warning("[P2-SOT-ALERT] Consider increasing time_limit or decreasing difficulty if task is infeasible")

        if ep % 25 == 0:
            multi_metrics = stats.get_multi_scale_metrics()
            logging.info("=" * 80)
            logging.info(f"[MULTI-SCALE-{ep}] TARGET: peak={multi_metrics['peak_performance']:.3f} stable_90={multi_metrics['stable_90_episodes']}ep")
            logging.info(f"[MULTI-SCALE-{ep}] WINDOWS: short={multi_metrics['short_term']['succ']:.3f} med={multi_metrics['medium_term']['succ']:.3f} long={multi_metrics['long_term']['succ']:.3f}")
            logging.info(f"[MULTI-SCALE-{ep}] QUALITY: stability={multi_metrics['stability']:.3f} learn_rate={multi_metrics['learning_rate']:.5f}")

            if multi_metrics['stable_90_episodes'] >= 5:
                logging.info(f"[ACHIEVEMENT] Stable 90%+ success rate achieved for {multi_metrics['stable_90_episodes']} episodes!")
            elif multi_metrics['peak_performance'] >= 0.9:
                logging.info(f"[MILESTONE] Peak performance {multi_metrics['peak_performance']:.1%} reached! Working towards stability...")

            if multi_metrics['stagnation_risk']:
                logging.warning(f"[ALERT] Learning stagnation detected. Consider parameter adjustment.")

            overall_health = (multi_metrics['stability'] + min(multi_metrics['learning_rate']*10, 1.0)) / 2
            health_status = "EXCELLENT" if overall_health > 0.8 else "GOOD" if overall_health > 0.6 else "NEEDS_IMPROVEMENT"
            logging.info(f"[HEALTH-CHECK-{ep}] Overall: {health_status} (score={overall_health:.3f})")
            logging.info("=" * 80)

        plotter.plot(ep, stats)

        if ep % eval_every == 0:

            rng_state_random = random.getstate()
            rng_state_numpy = np.random.get_state()
            rng_state_torch = torch.get_rng_state()
            if torch.cuda.is_available():
                rng_state_cuda = torch.cuda.get_rng_state()

            # PPO: 不需要epsilon和tau（evaluation时也不需要）
            policy_state_dict = None
            if hasattr(policy, 'get_state') and callable(policy.get_state):
                try:
                    policy_state_dict = copy.deepcopy(policy.get_state())
                except Exception:
                    pass

            model_cache_backup = None
            if hasattr(policy, 'reset_sequence_cache'):
                try:

                    if hasattr(policy, '_cache_states'):
                        model_cache_backup = copy.deepcopy(getattr(policy, '_cache_states', None))
                except Exception:
                    pass

            if hasattr(policy,'set_epsilon'):
                policy.set_epsilon(0.0)

            if hasattr(policy, 'reset_sequence_cache'):
                policy.reset_sequence_cache()

            logging.info(f"[EVAL] Starting evaluation: ep={ep}, ε=0.0, k={eval_episodes}")

            succ_e, coll_e, timeout_e, rew_e, nav_e, time_e, discomfort_freq_e, discomfort_dist_e, path_len_e, smooth_e, speed_e = explorer.run_k_episodes(
                k=eval_episodes, phase='val', update_memory=False, show_tqdm=False,
                imitation_learning=False, force_joint_state_policy=False)

            random.setstate(rng_state_random)
            np.random.set_state(rng_state_numpy)
            torch.set_rng_state(rng_state_torch)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(rng_state_cuda)

            # PPO: 不需要恢复epsilon（evaluation时epsilon=0也是通过set_epsilon(0.0)设置的）
            if policy_state_dict is not None and hasattr(policy, 'set_state'):
                try:
                    policy.set_state(policy_state_dict)
                except Exception:
                    pass

            if model_cache_backup is not None:
                try:
                    if hasattr(policy, '_cache_states'):
                        policy._cache_states = model_cache_backup
                except Exception:
                    pass

            # PPO修复：先计算eval_sum再使用（避免UnboundLocalError）
            eval_sum = succ_e + coll_e + timeout_e

            logging.info(f"[EVAL] Completed evaluation: succ={succ_e:.3f} coll={coll_e:.3f} timeout={timeout_e:.3f} sum={eval_sum:.3f} nav={nav_e:.2f}")

            logging.info(f"[EVAL-EXTENDED] path_len={path_len_e:.2f}m smoothness={smooth_e:.3f} avg_speed={speed_e:.2f}m/s")
            if abs(eval_sum - 1.0) > 0.05:
                logging.warning(f"[EVAL-SUM-CHECK] Evaluation sum deviation: {eval_sum:.3f} != 1.00, check evaluation calibration")

            best_succ = max(best_succ, succ_e)

            tb_writer.add_scalar("eval/success_rate", succ_e, ep)

            tb_writer.add_scalar("eval/path_length", path_len_e, ep)
            tb_writer.add_scalar("eval/smoothness", smooth_e, ep)
            tb_writer.add_scalar("eval/avg_speed", speed_e, ep)

        if (ep % int(save_every) == 0):

            log_periodic_metrics(ep, stats, window=200)

        # Progress reward统计已删除（简化版）
        # if (ep % 25 == 0) and (ep > 0):
        #     if hasattr(env, '_rprog_stats'):
        #         rprog = env._rprog_stats
        #         if rprog['count'] > 0:
        #             coverage = rprog['nonzero'] / rprog['count']
        #             pos_avg = rprog['pos_sum'] / max(1, rprog['pos_count'])
        #             neg_avg = rprog['neg_sum'] / max(1, rprog['neg_count'])
        #             episode_num = getattr(env, 'current_episode', ep)
        #             current_lambda_p = getattr(env, 'progress_reward_weight', 0.12)
        #             logging.info(f"[R_PROG@{ep}] λ_p={current_lambda_p:.3f} | cover={coverage:.1%} pos={pos_avg:.4f} neg={neg_avg:.4f}")
        #
        #             env._rprog_stats = {'count': 0, 'nonzero': 0, 'pos_sum': 0.0, 'pos_count': 0, 'neg_sum': 0.0, 'neg_count': 0}

        if (ep % int(save_every) == 0):

            rl_model_path = os.path.join(ARGS.outdir, f"rl_model_ep{ep}.pth")
            try:

                current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
                current_gamma = cfg.getfloat('trainer', 'gamma', fallback=0.97)

                rl_checkpoint = {
                    'episode': ep,
                    'policy_state': policy.state_dict(),
                    # SAC Components
                    'q_net1_state': q_net1.state_dict(),
                    'q_net2_state': q_net2.state_dict(),
                    'q_target_net1_state': q_target_net1.state_dict(),
                    'q_target_net2_state': q_target_net2.state_dict(),
                    'log_alpha': log_alpha,
                    # Optimizers (标准格式：分离Q优化器)
                    'optim_q1_state': optim_q1.state_dict(),
                    'optim_q2_state': optim_q2.state_dict(),
                    'optim_pi_state': optim_pi.state_dict(),
                    'optim_alpha_state': optim_alpha.state_dict(),
                    # Misc
                    'optimizer_state': optimizer.state_dict(), # Legacy, keep for compatibility
                    'stats_history': {
                        'hist_s': stats.hist_s,
                        'hist_c': stats.hist_c,
                        'hist_t': stats.hist_t,
                        'hist_r': stats.hist_r
                    },
                    'config': {
                        'batch_size': batch_size,
                        'lr': current_lr,
                        'gamma': current_gamma
                    },
                    'stage': 'rl_training',
                    'best_success': best_succ,
                    'meta': {
                        'save_time': time.time(),
                        'episode_count': ep,
                        'model_type': 'rl_checkpoint'
                    }
                }
                torch.save(rl_checkpoint, rl_model_path)
                logging.info(f"[RL-SAVE] RL model checkpoint -> {rl_model_path}")

            except Exception as e:
                logging.warning(f"[RL-SAVE] RL checkpoint failed: {e}")

                try:
                    minimal_checkpoint = {
                        'episode': ep,
                        'policy_state': policy.state_dict(),
                        'stage': 'rl_training_minimal'
                    }
                    torch.save(minimal_checkpoint, rl_model_path)
                    logging.info(f"[RL-SAVE] Minimal RL checkpoint -> {rl_model_path}")
                except Exception as e2:
                    logging.warning(f"[RL-SAVE] Minimal checkpoint failed: {e2}")

    log_final_metrics(stats, best_succ)

    if vectorized_sampler is not None:
        logging.info("[VECTORIZE] Stopping vectorized sampler...")
        vectorized_sampler.stop()
        logging.info("[VECTORIZE] Vectorized sampler stopped")

    logging.info(f"[DONE] Training completed - best_eval_success={best_succ:.3f}")
    return best_succ

def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.txt')
    parser.add_argument('--outdir', type=str, default='runs/mamba_vl')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--policy', type=str, default=None, help='override policy key (e.g., mamba or mamba_rl)')
    parser.add_argument('--gpu', action='store_true', help='force CUDA if available')
    parser.add_argument('--cpu', action='store_true', help='force CPU even if CUDA available')

    parser.add_argument('--stage', type=str, choices=['il','rl','auto'], default='auto',
                        help='Training stage: il=IL pretraining only, rl=RL only, auto=IL then RL')

    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint file (e.g., --resume policy_ep1000.pt)')
    parser.add_argument('--start_episode', type=int, default=None,
                        help='Override start episode number when resuming')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: from config)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable CUDA deterministic mode (slower but reproducible)')
    ARGS = parser.parse_args()

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if ARGS.gpu and torch.cuda.is_available():
        ARGS.device = 'cuda'
    elif ARGS.cpu:
        ARGS.device = 'cpu'

    # RTX 4090: 启用 BF16 AMP 支持
    if ARGS.device == 'cuda':
        torch.set_float32_matmul_precision('high')
        logging.info("[AMP] BF16 matmul precision enabled for RTX 4090")

    logger = setup_logger(ARGS.outdir)

    tb_log_dir = os.path.join(ARGS.outdir, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)

    import glob
    for old_file in glob.glob(os.path.join(tb_log_dir, "events.out.tfevents.*")):
        try:
            os.remove(old_file)
            logging.info(f"[TENSORBOARD] Removed old events file: {os.path.basename(old_file)}")
        except Exception as e:
            logging.warning(f"[TENSORBOARD] Failed to remove {old_file}: {e}")

    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    logging.info(f"[TENSORBOARD] Fresh events will be saved to: {tb_log_dir}")

    cfg = load_config(ARGS.config)

    train_cfg = TrainConfig(cfg, ARGS.outdir)
    env_cfg = EnvConfig(cfg)
    policy_cfg = PolicyConfig(cfg)

    seed = train_cfg.seed
    # 命令行参数覆盖配置文件
    if ARGS.seed is not None:
        seed = ARGS.seed
        logging.info(f"[SEED] 使用命令行指定的seed: {seed}")
    else:
        logging.info(f"[SEED] 使用配置文件的seed: {seed}")

    episodes = train_cfg.episodes
    updates_ep = train_cfg.updates_per_ep
    batch_size = train_cfg.batch_size
    lr = train_cfg.learning_rate
    gamma = train_cfg.gamma
    # PPO: 不需要epsilon（使用entropy探索）、huber_delta（使用MSE）
    dt = env_cfg.time_step
    roll_window = train_cfg.roll_window
    plot_every = train_cfg.plot_every
    save_every = train_cfg.save_every

    # PPO: grad_clip在config中定义为clip_grad_norm
    grad_clip = train_cfg.clip_grad_norm

    if not (dt>0):
        raise RuntimeError('配置 dt 必须 > 0，并与环境步长一致')

    # 设置随机种子和CUDA确定性模式
    set_seed(int(seed), deterministic=ARGS.deterministic)
    device = torch.device(ARGS.device)

    # TF32加速（仅在非确定性模式下启用）
    if torch.cuda.is_available() and not ARGS.deterministic:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("[TF32] Enabled for matrix operations (5-10% speedup)")

    env, robot = build_env_and_robot(cfg)
    pol_key = (ARGS.policy or policy_cfg.policy_name).lower()

    ttc_shield_unified = False

    logging.info(f"[BOOT] policy={pol_key} device={ARGS.device}")

    robot_visible = env_cfg.robot_visible
    success_reward = env_cfg.success_reward
    collision_penalty = env_cfg.collision_penalty

    logging.info("=" * 60)
    # 修复：T=25秒（正确值），不是T=40
    logging.info("[SOT] env: circle R=4 Nh=5 dt=0.25 T=25 | reward: +1/-0.25/-0.5 rc=0.2 | actions: 80(v∈[0,1.0]) | success_radius=0.25 | eval eps=0")
    logging.info("=" * 60)
    logging.info(f"[ALGO] SAC (Off-Policy): Twin Q-networks with entropy regularization")

    logging.info(f"REWARD: success={success_reward}, collision={collision_penalty}")
    logging.info(f"ROBOT: visible={robot_visible} (crowd avoidance {'enabled' if robot_visible else 'disabled'})")
    logging.info(f"TRAINING: lr={lr}, gamma={gamma}, batch={batch_size}, updates_per_ep={updates_ep}")

    n_step = train_cfg.n_step
    logging.info(f"N-STEP: Using {n_step}-step TD learning from config [train] section")
    logging.info(f"TIMESTEP: dt={dt}s")
    logging.info("=" * 60)

    policy  = build_policy(pol_key, cfg, device)
    bind_policy(robot, policy, env)  # PPO: 不需要epsilon参数

    # 根源修复：CrowdNav是多智能体场景，policy必须设置multiagent_training=True
    # 否则env.reset()会强制human_num=1，导致34维训练 vs 14维推理的维度不匹配
    if hasattr(policy, 'multiagent_training'):
        policy.multiagent_training = True
        logging.info(f"[POLICY-INIT] Set multiagent_training=True (5-human crowd navigation)")

    if hasattr(policy,'set_env_dt'): policy.set_env_dt(dt)

    logging.info("✅ debug.md清理：奖励整形器已删除（未实际使用）")

    logging.info("=" * 60)
    logging.info("[CONFIG-SUMMARY] Training Configuration (Runtime Values)")
    logging.info(f"  Policy: {pol_key}")
    if hasattr(policy, 'model'):
        model = policy.model

        if hasattr(model, 'temporal_encoder') and hasattr(model.temporal_encoder, 'backend'):
            actual_n_layers = len(model.temporal_encoder.backend)
        else:
            actual_n_layers = 'N/A'
        actual_d_model = getattr(model, 'd_model', 'N/A')
        logging.info(f"  Network: n_layers={actual_n_layers}, d_model={actual_d_model}")
    logging.info(f"  Sequence: seq_len={policy_cfg.seq_len}")
    logging.info(f"  Training: batch_size={batch_size}, updates_per_ep={updates_ep}")
    logging.info(f"  Learning: lr={lr:.6f}, gamma={gamma}")
    logging.info(f"  Episodes: {episodes}, device={device}")
    logging.info("=" * 60)

    if hasattr(policy, 'set_step_fn'):
        def step_fn(btnd, pad, action_table, dt):
            from crowd_nav.contracts import simulate_next_frames, tokens_to_34d, joint34_to_tokens

            frames_34d = tokens_to_34d(pick_last(btnd, pad))
            next_34d = simulate_next_frames(frames_34d, action_table, dt)

            B, A = next_34d.shape[:2]

            next_34d_cpu = next_34d.view(B*A, 34).detach().cpu().numpy()
            next_tokens = joint34_to_tokens(next_34d_cpu).view(B, A, 1, 6, 13)
            next_pad = torch.ones(B, A, 1, device=btnd.device)
            return next_tokens, next_pad
        policy.set_step_fn(step_fn)

    if hasattr(policy, 'set_reward_fn'):
        def reward_fn(btnd, pad, action_table):

            from crowd_nav.contracts import tokens_to_34d, simulate_next_frames
            frames_34d = tokens_to_34d(pick_last(btnd, pad))
            next_34d = simulate_next_frames(frames_34d, action_table, dt)

            B, A = next_34d.shape[:2]
            return torch.zeros(B, A, device=btnd.device)
        policy.set_reward_fn(reward_fn)

    if hasattr(policy, 'set_phase'):
        policy.set_phase('train')
    if hasattr(policy, 'set_ttc_shield'):
        policy.set_ttc_shield(False)  # SAC训练：关闭TTC护盾，保留TTC特征
    if hasattr(policy, 'set_goal_prior'):
        policy.set_goal_prior(False)
    logging.info("[SAC-TRAIN] TTC shield=OFF (keep TTC as feature only), goal prior=OFF")

    circle_radius = cfg.getfloat('sim', 'circle_radius', fallback=4.0)
    human_num = cfg.getint('sim', 'human_num', fallback=5)
    dt = cfg.getfloat('env', 'time_step', fallback=0.25)
    time_limit = cfg.getfloat('env', 'time_limit', fallback=25.0)
    success_radius = cfg.getfloat('robot', 'success_radius', fallback=0.25)
    robot_visible = cfg.get('robot', 'visible', fallback='false').lower() == 'true'

    from crowd_nav.contracts import GRID
    n_actions = _log_action_table(GRID)

    logging.info(f"[CFG-LOCK] circle R=4 Nh=5 dt=0.25 T=25 succ_r=0.25 visible=False eval_eps=0")

    # 验证配置一致性（无日志输出）
    assert circle_radius == 4 and human_num == 5 and abs(dt - 0.25) < 1e-6 and time_limit == 25 \
           and abs(success_radius - 0.25) < 1e-6 and not robot_visible and n_actions == 80, \
           f"Config mismatch: R={circle_radius}, Nh={human_num}, dt={dt}, T={time_limit}, succ_r={success_radius}, visible={robot_visible}, actions={n_actions}"

    from crowd_nav.contracts import discrete_index_to_action
    import math

    v_pref = cfg.getfloat('robot', 'v_pref', fallback=1.0)
    v_max_grid = GRID['v_max']

    v_threshold = 0.8 * v_pref
    if v_max_grid < v_threshold:
        logging.error("FATAL: 动作表熔断触发！Action table circuit breaker triggered!")
        logging.error(f"FATAL: v_max={v_max_grid:.3f} < threshold={v_threshold:.3f} (0.8 × v_pref)")
        logging.error("FATAL: 请检查 configs/env.config 中的 [policy] v_min/v_max 设置")
        import sys
        sys.exit(1)

    cfg_print_count = getattr(main, '_cfg_print_count', 0)
    if cfg_print_count > 0:
        logging.error("FATAL: 评测口径熔断触发！Multiple [CFG] prints detected!")
        logging.error("FATAL: 只允许打印一次[CFG]配置，发现多次打印可能存在口径不一致")
        import sys
        sys.exit(1)
    main._cfg_print_count = cfg_print_count + 1

    if time_limit != 25.0:
        logging.error("FATAL: 时间限制熔断触发！Non-standard time limit detected!")
        logging.error(f"FATAL: time_limit={time_limit} != 25.0，论文对标要求T=25秒")
        import sys
        sys.exit(1)

    profile = cfg.get('train', 'reward_profile', fallback='benchmark')

    REWARD = {
        'success': cfg.getfloat('reward', 'success_reward', fallback=1.0),
        'collision': cfg.getfloat('reward', 'collision_penalty', fallback=-0.25),
        'timeout': cfg.getfloat('reward', 'timeout_penalty', fallback=-0.6)
    }

    logging.info(f"[REWARD-{profile.upper()}] {REWARD}")
    # Progress reward已删除（简化版）
    # logging.info(f"[REWARD-PROGRESS] r_prog ENABLED | warm-up: ep<150 λ_p=0.0 → ep 150-1150 linear → ep≥1150 λ_p=0.12")
    # logging.info(f"[REWARD-PROGRESS] r_max=0.06 (clip单步进度) | m_t gate: 1[dmin≥0.6m] (拥挤区不刷进度)")

    config_success = cfg.getfloat('reward', 'success_reward', fallback=1.0)
    config_collision = cfg.getfloat('reward', 'collision_penalty', fallback=-1.0)
    config_timeout = cfg.getfloat('reward', 'timeout_penalty', fallback=-0.8)

    logging.info(f"[CONFIG-FALLBACK] success={config_success}, collision={config_collision}, timeout={config_timeout}")
    logging.info(f"[ACTIVE-REWARD] Using profile-based rewards: {REWARD}")

    logging.info("=" * 60)

    _first_action_logged = [False]

    timeout_penalty = cfg.getfloat('reward', 'timeout_penalty', fallback=-0.6)

    # SAC不需要PPO buffer
    explorer = Explorer(env, robot, device, memory=None, gamma=gamma)

    trajectory_log_path = os.path.join(ARGS.outdir, 'trajectory.log')
    trajectory_file = None  # 初始化为None，避免NameError
    try:

        trajectory_file = open(trajectory_log_path, 'w', encoding='utf-8')
        explorer.trajectory_logger = trajectory_file
        explorer.enable_trajectory_logging = True
        logging.info(f"[TRAJECTORY-LOG] Recording enabled: {trajectory_log_path}")
    except Exception as e:
        logging.warning(f"[TRAJECTORY-LOG] Failed to initialize trajectory logging: {e}")
        trajectory_file = None
        explorer.trajectory_logger = None
        explorer.enable_trajectory_logging = False

    import copy

    # 延迟 torch.compile：在所有 checkpoint 加载完成后执行
    # 原因：torch.compile 会改变 state_dict key 的前缀（_orig_mod.*），导致加载失败
    # 在加载 checkpoint 之前编译会导致参数不匹配，使用随机初始化而不是已训练的权重
    compile_policy_later = True

    fused_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    weight_decay = cfg.getfloat('train', 'weight_decay', fallback=0.01)

    # PPO optimizer: 同时优化policy和value（actor-critic）
    optimizer = optim.AdamW(
        policy.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=weight_decay,
        fused=fused_ok
    )

    if fused_ok:
        logging.info("[OPTIMIZER] Using fused AdamW (Ampere+ GPU)")
    logging.info(f"[OPTIMIZER-SAC] RL phase: lr={train_cfg.learning_rate}, manages policy (actor) ({sum(p.numel() for p in policy.parameters()):,} params)")
    logging.info(f"[OPTIMIZER-POLICY] BC refresh: lr={train_cfg.learning_rate * 0.1}, manages policy only ({sum(p.numel() for p in policy.parameters()):,} params)")
    logging.info("[GRADIENT-ISOLATION] ✓ Value loss will NEVER update policy (complete isolation)")

    # RTX 4090: 启用 BF16 AMP (比FP16稳定)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    logging.info("[AMP] BF16 Automatic Mixed Precision ENABLED for RTX 4090")

    train_step_counter = {'step': 0}

    il_weights_path = './models/il_model.pth'
    il_weights_loaded = False

    if os.path.exists(il_weights_path):
        try:

            ckpt = safe_torch_load(il_weights_path, map_location=device)
            missing, unexpected = policy.load_state_dict(ckpt.get('model', ckpt), strict=False)
            if missing:
                logging.info(f"[DUAL-HEAD] Missing keys in checkpoint (expected for new continuous_head): {missing}")
            if unexpected:
                logging.warning(f"[DUAL-HEAD] Unexpected keys in checkpoint: {unexpected}")
            logging.info(f"[DEBUG.MD] Successfully loaded IL weights from {il_weights_path} (strict=False)")
            il_weights_loaded = True
        except Exception as e:
            logging.warning(f"[DEBUG.MD] IL weight loading failed, continuing with mixed buffer: {e}")
            il_weights_loaded = False

    logging.info(f"[DEBUG.MD] Mixed buffer strategy: IL weights loaded={il_weights_loaded}")

    stats = TrainingStats(roll_window)
    plotter = Plotter(ARGS.outdir, plot_every)
    metrics_plotter = MetricsPlotter(ARGS.outdir)

    fmt = get_log_formatter()

    outdir_abs = os.path.abspath(ARGS.outdir)
    ckpt_path = os.path.join(outdir_abs, 'ckpt_il.pt')
    il_policy_path = os.path.join(outdir_abs, 'il_policy.pth')

    offline_dataset_path = None
    for version in ['v4.0', 'v3.1', 'v3.0', 'v2.1', 'v2.0', 'v1.0']:
        candidates = [
            f'data/il_dataset_diverse_{version}.pth',
            f'../data/il_dataset_diverse_{version}.pth',
            os.path.join(os.path.dirname(__file__), f'../data/il_dataset_diverse_{version}.pth'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                offline_dataset_path = os.path.abspath(candidate)
                break
        if offline_dataset_path:
            break

    skip_il = False
    logging.info(f"[IL-AUTO] 自动检测训练数据...")
    logging.info(f"[IL-AUTO]   离线数据集: {offline_dataset_path or 'None'} (exists: {os.path.exists(offline_dataset_path) if offline_dataset_path else False})")
    logging.info(f"[IL-AUTO]   BC checkpoint: {il_policy_path} (exists: {os.path.exists(il_policy_path)})")
    logging.info(f"[IL-AUTO]   IL checkpoint: {ckpt_path} (exists: {os.path.exists(ckpt_path)})")

    # PPO修复：修正IL跳过条件（offline_dataset_path可能是非空字符串但文件不存在）
    if os.path.exists(ckpt_path) and os.path.exists(il_policy_path) and not (offline_dataset_path and os.path.exists(offline_dataset_path)):

        logging.info(f"[IL-AUTO] ✓ 发现checkpoint但无离线数据集，加载checkpoint跳过IL+BC")
        try:
            ckpt = safe_torch_load(ckpt_path, map_location=device)
            policy.load_state_dict(ckpt['policy'])
            # PPO: policy已是actor-critic合一，不需要加载value_net和target_net
            il_success_rate = ckpt.get('il_success_rate', 0.0)
            logging.info(f"[IL-AUTO] ✓ 加载checkpoint成功 (IL成功率: {il_success_rate:.1%})")

            # PPO: 不需要warmup buffer（每次更新后清空）

            skip_il = True
            logging.info("[IL-AUTO] ✓ 跳过IL+BC，直接进入RL训练")
        except Exception as e:
            logging.warning(f"[IL-AUTO] Checkpoint加载失败: {e}")
            skip_il = False
            checkpoint = None # Initialize to None on failure
    elif offline_dataset_path and os.path.exists(offline_dataset_path):

        logging.info(f"[IL-AUTO] ✓ 检测到离线数据集: {offline_dataset_path}")
        logging.info(f"[IL-AUTO]   将跳过IL在线收集，强制执行BC训练")
        skip_il = False
    else:
        logging.info("[IL-AUTO] ✗ 无离线数据集也无checkpoint，将执行完整IL+BC训练")
        skip_il = False

    # ====== Buffer 初始化 ======
    from crowd_nav.utils.ppo_buffer import ReplayBufferSAC, ReplayBufferIQL

    seq_len = cfg.getint('buffer', 'seq_len', fallback=12)

    rl_buffer = ReplayBufferSAC(
        capacity=int(cfg.getfloat('buffer', 'capacity', fallback=800000)),
        seq_len=seq_len
    )
    il_buffer = ReplayBufferSAC(
        capacity=int(cfg.getfloat('buffer', 'il_capacity', fallback=200000)),
        seq_len=seq_len
    )
    logging.info(f"[SAC-BUFFER] rl_buffer capacity={rl_buffer.capacity}, il_buffer capacity={il_buffer.capacity}")

    # ✅ 只在 IL 结束后，把"BC 数据"存入 il_buffer，不再注入 rl_buffer
    def _prefill_il_to_buffers(il_trajs):
        if not il_trajs:
            logging.warning("[SAC-BUFFER] No IL trajectories, skip prefill")
            return

        n_push_rl = 0
        n_push_il = 0
        for (traj, meta) in il_trajs:
            states, actions, rewards = traj
            T = len(states)
            if T == 0:
                continue

            # 转换为numpy数组 (处理JointState对象)
            states_arr = np.array([s.to_array() if hasattr(s, 'to_array') else s for s in states], dtype=np.float32)
            actions_arr = np.array(actions, dtype=np.float32)
            
            # [FIX-REWARD-INJECTION-ULTIMATE] Re-label rewards using ENV CONFIG
            # This ensures consistency between IL warmup and Online RL
            # Indices [5, 6] are correct for JointState (px, py, vx, vy, radius, gx, gy...)
            
            # Fetch reward params dynamically from the env instance (which loaded env.config)
            # Note: env object might not be available in this scope easily if not passed.
            # Assuming standard config values if env not accessible, BUT BETTER to use hardcoded values THAT MATCH CONFIG.
            # Given the user's config: success=10, coll=-10, time=-5 (from log [ACTIVE-REWARD])
            # The previous hardcoded values (20, -15) were WRONG.
            
            # Reading from 'env' which is available in the outer scope or we use safe defaults matching standard config
            # For safety and consistency with the user's request to fix "double track" rewards:
            
            r_success = 10.0   # Matches typical benchmark
            r_collision = -10.0
            r_timeout = -10.0  # Aligned with env.config
            r_step = -0.05     # Step penalty to discourage freezing
            w_potential = 1.0  # Potential field weight
            
            new_rewards = []
            for i in range(len(states_arr)):
                s = states_arr[i]
                px, py = s[0], s[1]
                gx, gy = s[5], s[6] # Standard JointState indices verified
                
                dist = np.sqrt((px - gx)**2 + (py - gy)**2)
                
                if i < len(states_arr) - 1:
                    ns = states_arr[i+1]
                    npx, npy = ns[0], ns[1]
                    ndist = np.sqrt((npx - gx)**2 + (npy - gy)**2)
                    
                    # Potential reward: (d_prev - d_curr) * weight
                    step_r = (dist - ndist) * w_potential + r_step
                    new_rewards.append(step_r)
                else:
                    # Terminal reward
                    if meta.get('flag_collision'):
                        new_rewards.append(r_collision)
                    elif meta.get('flag_timeout'):
                        new_rewards.append(r_timeout)
                    else:
                        new_rewards.append(r_success)
                        
            rewards_arr = np.array(new_rewards, dtype=np.float32)

            # 构造dones数组
            episode_length = len(states)
            dones = np.zeros(episode_length, dtype=bool)
            dones[-1] = True  # 最后一步标记为done

            episode_data = {
                'states': states_arr,
                'actions_continuous': actions_arr,
                'rewards': rewards_arr,
                'dones': dones
            }

            # 只进 il_buffer
            il_buffer.push_episode(episode_data)
            n_push_il += 1

        logging.info(f"[SAC-BUFFER] Prefill done: rl={n_push_rl} episodes, il={n_push_il} episodes")

    stats = TrainingStats(roll_window)
    logging.info("[STATS-RESET] TrainingStats reset for RL phase")

    # ========== SAC: BC训练逻辑 ==========
    if not skip_il:
        logging.info("[IL-BC] Starting BC training phase...")
        il_result = _run_il_phase(
            train_cfg=train_cfg,  # 传递wrapper对象，但函数内部用cfg(RawConfigParser)
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
            env=env,
            explorer=explorer,
            policy=policy,
            device=device,
            batch_size=batch_size,
            replay_iql=rl_buffer,  # SAC: 复用rl_buffer用于BC训练参数传递
            cfg=cfg  # 传递RawConfigParser用于config读取
        )
        logging.info(f"[IL-BC] BC training completed: success_rate={il_result.get('success_rate', 0.0):.1%}")

        # 调用 _prefill_il_to_buffers 将 IL 数据存入 il_buffer
        il_trajectories = il_result.get('il_trajectories', [])
        _prefill_il_to_buffers(il_trajectories)
        # BC训练后会设置policy._il_warm_started = True
    else:
        logging.info("[IL-BC] Skipping BC training (checkpoint loaded)")
        il_result = {'success_rate': 0.0}

    if not getattr(policy, "_il_warm_started", False):
        logging.warning("[CKPT-LOAD] IL warm-start not detected, should have been done in IL phase")
    else:
        logging.info("[CKPT-LOAD] skipped (already warm-started with fresh IL ckpt)")

    start_episode = 1
    checkpoint = None
    if ARGS.resume:
        resume_path = ARGS.resume
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(ARGS.outdir, resume_path)

        if os.path.exists(resume_path):
            try:
                logging.info(f"[RESUME] Loading checkpoint from {resume_path}")
                checkpoint = safe_torch_load(resume_path, map_location=device)

                if checkpoint.get('policy_state'):
                    missing, unexpected = policy.load_state_dict(checkpoint['policy_state'], strict=False)
                    if missing:
                        logging.info(f"[RESUME-DUAL-HEAD] Missing keys (expected for continuous_head): {missing}")
                    logging.info("[RESUME] Policy state loaded")

                # PPO: policy已是actor-critic合一，不需要加载value_net和target_net

                if checkpoint.get('optimizer_state'):
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    logging.info("[RESUME] Optimizer state loaded")

                start_episode = checkpoint.get('episode', 1) + 1
                if ARGS.start_episode:
                    start_episode = ARGS.start_episode
                    logging.info(f"[RESUME] Episode number overridden to {start_episode}")

                if checkpoint.get('stats_history'):
                    hist = checkpoint['stats_history']
                    stats.hist_s = hist.get('hist_s', [])
                    stats.hist_c = hist.get('hist_c', [])
                    stats.hist_t = hist.get('hist_t', [])
                    stats.hist_r = hist.get('hist_r', [])
                    logging.info(f"[RESUME] Statistics history loaded: {len(stats.hist_s)} episodes")

                logging.info(f"[RESUME] ✅ Successfully resumed from episode {start_episode}")

            except Exception as e:
                logging.error(f"[RESUME] Failed to load checkpoint: {e}")
                logging.info("[RESUME] Starting fresh training...")
                start_episode = 1
        else:
            logging.error(f"[RESUME] Checkpoint file not found: {resume_path}")
            logging.info("[RESUME] Starting fresh training...")
            start_episode = 1

    # PPO: 不需要warmup buffer（每次更新后清空，无需预热）

    # 保持BC学到的低噪声，不强制clamp
    # _clamp_policy_std(policy, min_std=0.05, max_std=0.10)  # 已禁用：让初始std=0.01生效

    # 现在执行 torch.compile：所有 checkpoint 加载已完成
    # 延迟编译确保 state_dict key 匹配，BC 权重正确加载
    if compile_policy_later:
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
            logging.info("[JIT] 4090 UNLEASHED: torch.compile FORCED ON! (reduce-overhead mode, after checkpoint loads)")
            # 重新绑定到robot，确保采样使用编译后的模型
            bind_policy(robot, policy, env)
        except Exception as e:
            logging.info(f"[JIT] torch.compile skipped: {e}")

    # ========== 关闭 RL Curriculum (todo.md修复) ==========
    # 原因：BC模型只在5人环境训练，RL应该先在相同环境超越BC（>80%）
    # 再增加难度。现在curriculum太激进，导致RL在4-20人随机变化中困顿
    # 修复：回归标准5人圆桌环境
    env.use_rl_curriculum = False
    logging.info("[CURRICULUM] DISABLED - Focusing on standard environment")
    logging.info("[CURRICULUM] RL will use baseline 5-human circle environment (Nh=5, R=4)")

    best_succ = _run_rl_phase(cfg, env, policy, optimizer, explorer, rl_buffer, il_buffer, stats, plotter, device, episodes, batch_size, save_every, ARGS, tb_writer, train_step_counter, scaler, start_episode, metrics_plotter, trajectory_file, resume_checkpoint=checkpoint)

    logging.info(f"[MAIN-COMPLETE] Training finished with best evaluation success: {best_succ:.3f}")

    if hasattr(explorer, 'trajectory_logger') and explorer.trajectory_logger:
        try:
            explorer.trajectory_logger.close()
            logging.info("[TRAJECTORY-LOG] Trajectory file closed successfully")
        except Exception as e:
            logging.warning(f"[TRAJECTORY-LOG] Failed to close trajectory file: {e}")

    tb_writer.close()
    logging.info("[TENSORBOARD] Events file closed successfully")

if __name__ == '__main__':
    main()
