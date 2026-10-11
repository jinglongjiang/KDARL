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


def soft_threat(self_state, human_states, actions, return_logit=False):
    """返回 (A, N) 的 phi。actions 为 (A,2) 的 holonomic 速度。

    return_logit=True 时同时返回 (A,N) 的未压缩威胁 logit
        s_i(a) = (d0 - d_CPA)/tau_d + (T - t_CPA)/tau_t
    phi 是它两项各自过 sigmoid 后的乘积;logit 无界且对威胁单调,用作
    risk-aware attention 对照臂的注意力打分(对应 Electronics 2023 的未归一化
    几何风险 a~_i,其 softmax 权重在无界打分上才有意义)。不引入新的可调参数。
    """
    if not human_states:
        z = np.zeros((len(actions), 0))
        return (z, z) if return_logit else z
    h = np.asarray([[x.px, x.py, x.vx, x.vy, x.radius] for x in human_states], float)
    a = np.asarray([[x.vx, x.vy] for x in actions], float)
    p = h[None, :, :2] - np.array([[self_state.px, self_state.py]])[None]     # (1,N,2)
    p = np.broadcast_to(p, (len(a), len(h), 2))
    v = h[None, :, 2:4] - a[:, None, :]                                       # (A,N,2)
    denom = np.maximum((v * v).sum(-1), 1e-12)
    t = np.clip(-(p * v).sum(-1) / denom, 0.0, HORIZON)                       # (A,N)
    d = np.linalg.norm(p + t[..., None] * v, axis=-1) - self_state.radius - h[None, :, 4]
    gd = (CLEARANCE_0 - d) / TAU_D
    gt = (HORIZON - t) / TAU_T
    phi = _sigmoid(gd) * _sigmoid(gt)
    return (phi, gd + gt) if return_logit else phi


AGGREGATIONS = ('sum', 'mean', 'softmax')


class ThreatMass:
    """beta=0 即完全不介入。

    agg 决定累计项如何把每人威胁合成一个动作标量,三臂共用同一个 phi_i(a):
        sum      M(a) = sum_i phi_i(a)              不归一化(本文的 ACT)
        mean     M(a) = (1/N) sum_i phi_i(a)        最纯的"归一化 vs 求和"对照
        softmax  M(a) = sum_i softmax(s_i) phi_i    risk-aware attention 形式的对照
    三者的极值项 L(a)=max_i phi_i(a) 完全相同,因此 lam=1 时差异只来自聚合方式。
    """

    def __init__(self, beta=0.0, lam=1.0, topk=0, agg='sum'):
        self.beta = float(beta)
        self.lam = float(lam)
        if agg not in AGGREGATIONS:
            raise ValueError('Unknown aggregation: %r' % (agg,))
        self.agg = agg
        # topk>0 时只保留每个动作上威胁最高的 k 个行人,复现"截断到最危险者"的设计选择
        # (Anas et al. 2023 的 top-K 观测)。topk=0 表示不截断,即本文的外延统计量。
        self.topk = int(topk)

    @property
    def active(self):
        return self.beta != 0.0

    def penalty(self, state, actions):
        """返回 (A,) 的 beta*R(a);无可见行人时为 0。"""
        if not self.active:
            return 0.0
        need_logit = self.agg == 'softmax'
        out = soft_threat(state.self_state, list(state.human_states), actions,
                          return_logit=need_logit)
        phi, logit = out if need_logit else (out, None)
        if phi.shape[1] == 0:
            return np.zeros(len(actions))
        extreme = phi.max(axis=1)
        if self.topk > 0 and phi.shape[1] > self.topk:
            part = np.partition(phi, -self.topk, axis=1)[:, -self.topk:]
            mass = part.sum(axis=1)
        elif self.agg == 'sum':
            mass = phi.sum(axis=1)
        elif self.agg == 'mean':
            # 归一化聚合:权重恒为 1/N,和不随可见人数增长
            mass = phi.mean(axis=1)
        else:
            w = logit - logit.max(axis=1, keepdims=True)
            np.exp(w, out=w)
            w /= w.sum(axis=1, keepdims=True)
            mass = (w * phi).sum(axis=1)
        return self.beta * ((1.0 - self.lam) * extreme + self.lam * mass)
