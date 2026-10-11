"""部分观测下的累计威胁质量估计器,以及它的完整消融梯队。

决定密集人群安全的量是逐动作的累计威胁质量
    M(a) = sum_i  phi(a, X_i)
问题在于 X_i 对被遮挡/离开视场的行人不可观测,于是观测到的 M 是有偏低估。
本模块把它写成一个显式的估计问题:
    M_b(a) = sum_i  r_i * E_{X~b_i}[ phi(a, X) ]
并提供一整条阶梯,让"贝叶斯是否必要"成为可证伪的问题:

  visible : 只对当前观测到的人求和(r=0 给未观测者)   —— 当前所有方法的做法
  memory  : 记住所有见过的人,匀速外推,r=1            —— 会把早已离开的人算成硬障碍
  ttl     : memory + 固定存活步数后丢弃               —— 最常见的工程做法
  decay   : memory + 指数存在度衰减 exp(-age/tau)      —— 常见的置信度衰减
  rfs     : 伯努利存在概率 + 卡尔曼后验               —— 标准 RFS
  rfs_occ : rfs,但存在度更新区分"没看到"的两种原因    —— 本工作的增量

rfs_occ 的要点:未检测到某条航迹时,先查它的预测位置落在传感器栅格的哪种单元。
落在**已知被遮挡**的单元里,没看到是意料之中,存在度几乎不降;
落在**已确认空旷**的单元里,没看到才是缺席的强证据,存在度大幅下降。
固定 TTL 与指数衰减在结构上无法做这个区分 —— 它们只有"多久没看到",没有"为什么没看到"。

威胁量有两种取法:
  mean   : phi(a, 均值)        —— 忽略后验散布
  expect : E_b[phi(a, X)]      —— sigma point 求期望;phi 非线性,两者不等(Jensen 间隙)
"""
import numpy as np

from threat_mass import soft_threat, HORIZON, CLEARANCE_0, TAU_D, TAU_T

MODES = ('visible', 'memory', 'ttl', 'decay', 'rfs', 'rfs_occ')
ASSOCIATIONS = ('oracle', 'greedy', 'hungarian')
UNKNOWN_SPEEDS = 1.0
UNKNOWN_DIRS = 4
THREATS = ('mean', 'age_proxy', 'trace', 'expect')


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


class Track:
    __slots__ = ('id', 'mean', 'cov', 'existence', 'missed', 'radius', 'hits')

    def __init__(self, identifier, mean, cov, radius):
        self.id = int(identifier)
        self.mean = np.asarray(mean, float)
        self.cov = np.asarray(cov, float)
        self.existence = 1.0
        self.missed = 0
        self.hits = 1
        self.radius = float(radius)


class BeliefTracker:
    """一个航迹集合 + 一条消融梯队。数据关联用仿真提供的 id(暂不研究关联问题)。"""

    def __init__(self, mode='visible', dt=0.25, accel_std=0.55,
                 pos_std=0.0, vel_std=0.0, survival=0.995,
                 ttl_steps=8, decay_tau=6.0, r_min=0.08,
                 occ_survival=0.999, clear_detect=0.95, unknown_scale=0.0,
                 association='oracle', gate=2.5, max_coast=12, confirm_hits=3):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.dt = float(dt)
        self.accel_std = float(accel_std)
        # 观测噪声:standard 协议为 0,给一个下限避免奇异
        self.pos_var = max(float(pos_std), 0.02) ** 2
        self.vel_var = max(float(vel_std), 0.05) ** 2
        self.survival = float(survival)
        self.ttl_steps = int(ttl_steps)
        self.decay_tau = float(decay_tau)
        self.r_min = float(r_min)
        self.occ_survival = float(occ_survival)
        self.clear_detect = float(clear_detect)
        # 未知区域强度的缩放。0 表示不启用;>0 时按当前观测密度自标定。
        self.unknown_scale = float(unknown_scale)
        # 关联方式。oracle 用仿真真值 id;greedy/hungarian 只看无标签的位置量测。
        if association not in ASSOCIATIONS:
            raise ValueError(association)
        self.association = association
        self.gate = float(gate)
        # SORT 式航迹管理,仅在非 oracle 关联下生效。oracle 关联时 id 唯一,不会产生幻影航迹;
        # 非 oracle 下必须有存活上限(max_age)与确认期,否则未分配量测会无界地开新航迹。
        self.max_coast = int(max_coast)
        self.confirm_hits = int(confirm_hits)
        self._next_id = 0
        self.cardinality = 0
        self.last_grid = None
        self.last_mesh = None
        self.last_resolution = 0.25
        self.last_seen = 0
        self.tracks = {}
        self._keymap = {}
        dt = self.dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        g = np.array([dt * dt / 2, dt * dt / 2, dt, dt], float)
        self.Q = self.accel_std ** 2 * np.diag(g ** 2)
        self.R = np.diag([self.pos_var, self.pos_var, self.vel_var, self.vel_var])

    def reset(self):
        self.tracks = {}
        self._keymap = {}
        self._next_id = 0

    def _associate(self, observed):
        """把无标签量测分配给现有航迹。只用位置,不看真值 id。返回 观测下标 -> 航迹键。"""
        keys = list(self.tracks.keys())
        if not observed:
            return {}
        z = np.asarray([[e['px'], e['py']] for e in observed], float)
        out = {}
        if keys:
            # 预测位置(此刻尚未做 predict,故在此就地外推一步)
            pred = np.asarray([(self.F @ self.tracks[k].mean)[:2] for k in keys], float)
            cost = np.linalg.norm(z[:, None, :] - pred[None, :, :], axis=-1)
            big = self.gate * 10.
            gated = np.where(cost <= self.gate, cost, big)
            if self.association == 'hungarian':
                from scipy.optimize import linear_sum_assignment
                ri, ci = linear_sum_assignment(gated)
                for i, j in zip(ri, ci):
                    if gated[i, j] < big:
                        out[int(i)] = keys[int(j)]
            else:  # greedy:按距离升序贪心
                order = np.dstack(np.unravel_index(np.argsort(gated, axis=None), gated.shape))[0]
                used_z, used_t = set(), set()
                for i, j in order:
                    if gated[i, j] >= big:
                        break
                    if int(i) in used_z or int(j) in used_t:
                        continue
                    out[int(i)] = keys[int(j)]
                    used_z.add(int(i)); used_t.add(int(j))
        for i in range(len(observed)):
            if i not in out:
                self._next_id += 1
                out[int(i)] = 10 ** 6 + self._next_id   # 新航迹
        return out

    # ---- 传感器栅格查询:预测位置处是"已知遮挡"还是"已确认空旷" ----
    @staticmethod
    def _cell_is_occluded(grid, mesh, resolution, xy):
        if grid is None or mesh is None:
            return True  # 没有栅格信息时保守处理:当作"不知道"
        mx, my = mesh
        i = int(round((xy[1] - my[0, 0]) / resolution))
        j = int(round((xy[0] - mx[0, 0]) / resolution))
        if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
            return True  # 视场外
        return bool(grid[i, j] == .5)

    def step(self, observed, grid=None, mesh=None, resolution=0.25):
        """observed: 本步被观测到的实体列表,含 id/px/py/vx/vy/radius。"""
        self.last_grid, self.last_mesh = grid, mesh
        self.last_resolution = float(resolution)
        self.last_seen = len(observed)
        self.cardinality = len(observed)
        if self.association == 'oracle':
            seen = {int(e['id']): np.array([e['px'], e['py'], e['vx'], e['vy']], float)
                    for e in observed}
            keymap = {int(e['id']): int(e['id']) for e in observed}
        else:
            keymap = self._associate(observed)
            seen = {keymap[i]: np.array([e['px'], e['py'], e['vx'], e['vy']], float)
                    for i, e in enumerate(observed)}
        self._keymap = keymap

        if self.mode == 'visible':
            self.tracks = {}
            for i, e in enumerate(observed):
                t = Track(i, [e['px'], e['py'], e['vx'], e['vy']],
                          np.zeros((4, 4)), e['radius'])
                self.tracks[t.id] = t
            return

        # 预测
        for t in self.tracks.values():
            t.mean = self.F @ t.mean
            t.cov = self.F @ t.cov @ self.F.T + self.Q
            if self.mode in ('rfs', 'rfs_occ'):
                t.existence *= self.survival

        # 更新 / 新建
        for i, e in enumerate(observed):
            k = self._keymap[i] if self.association != 'oracle' else int(e['id'])
            z = seen[k]
            if k not in self.tracks:
                self.tracks[k] = Track(k, z, self.R.copy(), e['radius'])
                continue
            t = self.tracks[k]
            S = t.cov + self.R
            K = t.cov @ np.linalg.inv(S)
            t.mean = t.mean + K @ (z - t.mean)
            t.cov = (np.eye(4) - K) @ t.cov
            t.missed = 0
            t.hits += 1
            t.existence = 1.0

        # 未被观测到的航迹
        drop = []
        for k, t in self.tracks.items():
            if k in seen:
                continue
            t.missed += 1
            if self.mode == 'memory':
                t.existence = 1.0
            elif self.mode == 'ttl':
                if t.missed > self.ttl_steps:
                    drop.append(k)
                else:
                    t.existence = 1.0
            elif self.mode == 'decay':
                t.existence = float(np.exp(-t.missed * self.dt / self.decay_tau))
            elif self.mode == 'rfs':
                # 标准伯努利漏检更新:不区分漏检原因
                pd = self.clear_detect
                r = t.existence
                t.existence = r * (1 - pd) / max(1 - r * pd, 1e-9)
            else:  # rfs_occ:按"为什么没看到"分情况
                occluded = self._cell_is_occluded(grid, mesh, resolution, t.mean[:2])
                if occluded:
                    # 意料之中的缺席,几乎不是"人已离开"的证据
                    t.existence *= self.occ_survival
                else:
                    pd = self.clear_detect
                    r = t.existence
                    t.existence = r * (1 - pd) / max(1 - r * pd, 1e-9)
            if self.mode in ('decay', 'rfs', 'rfs_occ') and t.existence < self.r_min:
                drop.append(k)
        # 航迹管理对所有臂一视同仁(含 oracle),否则关联方式不是唯一变量。
        for k, t in self.tracks.items():
            if t.missed > self.max_coast:
                drop.append(k)
        for k in set(drop):
            self.tracks.pop(k, None)

    # ---- 威胁质量 ----
    def _active(self):
        """参与威胁质量的航迹。非 oracle 关联下,未达确认次数的暂定航迹不计入。"""
        return [t for t in self.tracks.values()
                if t.hits >= self.confirm_hits or t.missed == 0]

    def _phi_mean(self, self_state, actions):
        out = []
        for t in self._active():
            out.append(_phi_one(self_state, t.mean, t.radius, actions))
        return np.array(out) if out else np.zeros((0, len(actions)))

    def _phi_expect(self, self_state, actions):
        """无迹变换:对 4 维后验取 9 个 sigma point 求 phi 的期望。"""
        out = []
        n, kappa = 4, 1.0
        for t in self._active():
            if not np.any(t.cov):
                out.append(_phi_one(self_state, t.mean, t.radius, actions)); continue
            try:
                L = np.linalg.cholesky((n + kappa) * t.cov)
            except np.linalg.LinAlgError:
                out.append(_phi_one(self_state, t.mean, t.radius, actions)); continue
            pts = [t.mean] + [t.mean + L[:, i] for i in range(n)] + [t.mean - L[:, i] for i in range(n)]
            w = np.array([kappa / (n + kappa)] + [1 / (2 * (n + kappa))] * (2 * n))
            phi = np.stack([_phi_one(self_state, p, t.radius, actions) for p in pts])
            out.append((w[:, None] * phi).sum(0))
        return np.array(out) if out else np.zeros((0, len(actions)))

    def unknown_mass(self, self_state, actions):
        """从未观测区域可能存在的、从未见过的行人所贡献的累计威胁质量。

        传感器栅格值 0.5 的单元表示"不知道那里有没有人"(被遮挡、视场外或超出量程)。
        对这些单元按强度 rho 积分,速度取先验(四个方向、典型速度)下的期望:
            M_unk(a) = rho * area * sum_cells  E_v[ phi(a, [cell, v]) ]
        rho 由当前观测自标定:已观测区域内看到的人数 / 已观测面积。
        这是 RFS 基数后验中"未检测目标"那一部分,现有人群导航方法没有这一项。
        """
        if self.unknown_scale <= 0. or self.last_grid is None:
            return np.zeros(len(actions))
        grid, (mx, my) = self.last_grid, self.last_mesh
        res = self.last_resolution
        stride = 2
        g = grid[::stride, ::stride]
        xs = mx[::stride, ::stride][g == .5]
        ys = my[::stride, ::stride][g == .5]
        if xs.size == 0:
            return np.zeros(len(actions))
        observed_area = float((grid != .5).sum()) * res * res
        rho = self.unknown_scale * (self.last_seen / max(observed_area, 1e-6))
        area = (res * stride) ** 2
        a = np.asarray([[x.vx, x.vy] for x in actions], float)          # (A,2)
        p = np.stack([xs - self_state.px, ys - self_state.py], -1)      # (C,2)
        acc = np.zeros(len(actions))
        for k in range(UNKNOWN_DIRS):
            ang = 2 * np.pi * k / UNKNOWN_DIRS
            hv = UNKNOWN_SPEEDS * np.array([np.cos(ang), np.sin(ang)])
            v = hv[None, None, :] - a[:, None, :]                        # (A,C,2)
            denom = np.maximum((v * v).sum(-1), 1e-12)
            t = np.clip(-(p[None] * v).sum(-1) / denom, 0., HORIZON)
            d = np.linalg.norm(p[None] + t[..., None] * v, axis=-1) - self_state.radius - .3
            acc += _sigmoid((CLEARANCE_0 - d) / TAU_D) * _sigmoid((HORIZON - t) / TAU_T) @ np.ones(xs.size) \
                if False else (_sigmoid((CLEARANCE_0 - d) / TAU_D) * _sigmoid((HORIZON - t) / TAU_T)).sum(-1)
        return rho * area * acc / UNKNOWN_DIRS

    def _phi_age_proxy(self, self_state, actions):
        """最强非贝叶斯代理:用解析的 age 协方差替换真后验协方差,再走同一套无迹期望。

        与 _phi_expect 的唯一差别是协方差从哪来:
          expect    : 卡尔曼递推得到的真后验 Sigma
          age_proxy : Sigma(age) 解析式,只由失观步数决定,不需要滤波器
        两者共用同一无迹变换与同一有界的 phi,量纲可比。
        若两者打平,说明本设置下 Age 是 Full 协方差的近充分代理。

        作废说明:此前实现是把 sigma 加到有效半径上,那不是同量纲对照 ——
        半径膨胀无界(失观 10 s 即 27.5 m),而 E[phi] 恒 <= 1。相关批次已隔离。
        """
        out = []
        n, kappa = 4, 1.0
        for t in self._active():
            sp = self.accel_std * (t.missed * self.dt) ** 2 / 2.
            sv = self.accel_std * (t.missed * self.dt)
            cov = np.diag([max(sp, 1e-3) ** 2, max(sp, 1e-3) ** 2,
                           max(sv, 1e-3) ** 2, max(sv, 1e-3) ** 2])
            try:
                L = np.linalg.cholesky((n + kappa) * cov)
            except np.linalg.LinAlgError:
                out.append(_phi_one(self_state, t.mean, t.radius, actions)); continue
            pts = [t.mean] + [t.mean + L[:, i] for i in range(n)] + [t.mean - L[:, i] for i in range(n)]
            w = np.array([kappa / (n + kappa)] + [1 / (2 * (n + kappa))] * (2 * n))
            phi = np.stack([_phi_one(self_state, q, t.radius, actions) for q in pts])
            out.append((w[:, None] * phi).sum(0))
        return np.array(out) if out else np.zeros((0, len(actions)))

    def _phi_trace(self, self_state, actions, c=1.0):
        """一阶不确定度惩罚:phi(mu) 之后按位置协方差的迹线性加罚,不做完整期望。"""
        out = []
        for t in self._active():
            tr = float(np.trace(t.cov[:2, :2]))
            out.append(np.clip(_phi_one(self_state, t.mean, t.radius, actions) + c * tr, 0., 1.))
        return np.array(out) if out else np.zeros((0, len(actions)))

    def mass(self, self_state, actions, threat='mean'):
        """返回 (A,) 的 sum_i r_i * phi_i(a) 与 (A,) 的 max_i r_i * phi_i(a)。"""
        if threat == 'expect':
            phi = self._phi_expect(self_state, actions)
        elif threat == 'age_proxy':
            phi = self._phi_age_proxy(self_state, actions)
        elif threat == 'trace':
            phi = self._phi_trace(self_state, actions)
        else:
            phi = self._phi_mean(self_state, actions)
        if phi.shape[0] == 0:
            return np.zeros(len(actions)), np.zeros(len(actions))
        r = np.array([t.existence for t in self._active()])[:, None]
        w = r * phi
        return w.sum(0) + self.unknown_mass(self_state, actions), w.max(0)


def _phi_one(self_state, state4, radius, actions):
    """单个行人状态对 A 个候选动作的软威胁,与 threat_mass.soft_threat 同式。"""
    a = np.asarray([[x.vx, x.vy] for x in actions], float)
    p = np.array([state4[0] - self_state.px, state4[1] - self_state.py], float)[None, :]
    v = np.array([state4[2], state4[3]], float)[None, :] - a
    denom = np.maximum((v * v).sum(-1), 1e-12)
    t = np.clip(-(p * v).sum(-1) / denom, 0.0, HORIZON)
    d = np.linalg.norm(p + t[:, None] * v, axis=-1) - self_state.radius - radius
    return _sigmoid((CLEARANCE_0 - d) / TAU_D) * _sigmoid((HORIZON - t) / TAU_T)
