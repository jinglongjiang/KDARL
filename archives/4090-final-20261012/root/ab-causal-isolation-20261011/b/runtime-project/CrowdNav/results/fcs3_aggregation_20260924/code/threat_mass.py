"""动作相关的威胁质量 M(a):不归一化的人群风险外挂。

对每个候选动作 a 与每个被观测到的行人 i,用最近接近点(CPA)的几何量构造软威胁
    phi_i(a) = sigmoid((d0 - d_CPA)/tau_d) * sigmoid((T - t_CPA)/tau_t)
然后取 Fréchet 式的两端:
    L(a) = max_i phi_i(a)        最危险单人
    M(a) = sum_i phi_i(a)        累计人群暴露(不归一化,随可见人数增长)
    R(a) = (1-lam) * L(a) + lam * M(a)
最终 Q'(a) = Q(a) - beta * R(a)。

要点:
- phi 是手工几何量,不是标定过的碰撞概率,也不需要任何训练好的风险头;
- 常数取自环境自身的奖励定义(discomfort_dist=0.2)与既有风险视界(2 s),未经调参;
- 只用策略当前实际观测到的行人,不触碰真值;
- beta=0 时逐比特还原原策略。
"""
import numpy as np

HORIZON = 2.0          # 与既有 risk_examples 的视界一致
CLEARANCE_0 = 0.2      # env.config 的 discomfort_dist
TAU_D = 0.2
TAU_T = 0.5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def soft_threat(self_state, human_states, actions):
    """返回 (A, N) 的 phi。actions 为 (A,2) 的 holonomic 速度。"""
    if not human_states:
        return np.zeros((len(actions), 0))
    h = np.asarray([[x.px, x.py, x.vx, x.vy, x.radius] for x in human_states], float)
    a = np.asarray([[x.vx, x.vy] for x in actions], float)
    p = h[None, :, :2] - np.array([[self_state.px, self_state.py]])[None]     # (1,N,2)
    p = np.broadcast_to(p, (len(a), len(h), 2))
    v = h[None, :, 2:4] - a[:, None, :]                                       # (A,N,2)
    denom = np.maximum((v * v).sum(-1), 1e-12)
    t = np.clip(-(p * v).sum(-1) / denom, 0.0, HORIZON)                       # (A,N)
    d = np.linalg.norm(p + t[..., None] * v, axis=-1) - self_state.radius - h[None, :, 4]
    return _sigmoid((CLEARANCE_0 - d) / TAU_D) * _sigmoid((HORIZON - t) / TAU_T)


class ThreatMass:
    """beta=0 即完全不介入。"""

    def __init__(self, beta=0.0, lam=1.0):
        self.beta = float(beta)
        self.lam = float(lam)

    @property
    def active(self):
        return self.beta != 0.0

    def penalty(self, state, actions):
        """返回 (A,) 的 beta*R(a);无可见行人时为 0。"""
        if not self.active:
            return 0.0
        phi = soft_threat(state.self_state, list(state.human_states), actions)
        if phi.shape[1] == 0:
            return np.zeros(len(actions))
        extreme = phi.max(axis=1)
        mass = phi.sum(axis=1)
        return self.beta * ((1.0 - self.lam) * extreme + self.lam * mass)
