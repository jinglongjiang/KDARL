"""遮挡观测 -> Bernoulli-Gaussian RFS 信念 -> 逐人 token(五个实验臂)。

协议(severe,沿用 bayes_occ_mpc 既有设定,不重新调参):
  位置观测噪声 0.10 m,速度观测噪声 0.20 m/s,检测概率 0.8;
  感知范围 5 m,栅格 0.25 m,行人切线阴影;robot.visible=false;
  过程噪声 acceleration_std=0.55(各向同性,不做速度缩放/各向异性)。
遮挡前后的同一行人用仿真器真实 ID 关联(oracle ID),不实现数据关联。
"""
from pathlib import Path

import numpy as np
from crowd_nav.bayesian_rfs import BayesianRFSBelief, RFSConfig

ARMS = ('sensor', 'mean', 'full', 'age', 'gt')
# severe 协议的默认值(0.10 m / 0.20 m/s / pD=0.8)。Standard 协议在配置文件的
# [occlusion] 段覆盖为 0 / 0 / 1.0。两档只差传感质量,其余一律相同。
NOISE_POSITION, NOISE_VELOCITY, DETECTION_PROBABILITY = .10, .20, .8
POINTS = 5              # 每人的预测速度采样点数,与既有网络接口一致
SIGMA_CLIP = 3.0        # 位置不确定度通道的统一上限(米),五臂同一契约
# age 臂:只允许用"失观了多久"查表,表由五人标定集上 full 臂后验的均值标定,
# 因而两臂的不确定度在尺度上按构造对齐。两表只在 age>=1 上有定义:当前被检测到
# 的行人在四臂上逐位一致,分叉只从漏检/遮挡之后开始。
# 查表不再作为模块常量:severe 与 standard 各有一份,存于
# configs/age_table_<protocol>.json,由 calibrate_age_sigma.py 写入。


def load_age_table(protocol):
    """按协议读取 age 查表。severe 与 standard 的 RFS 后验尺度不同,不可共用。"""
    import json
    path = Path(__file__).resolve().parent / 'configs' / f'age_table_{protocol}.json'
    if not path.exists():
        raise FileNotFoundError(f'{protocol} 协议的 age 查表尚未标定: {path}')
    payload = json.loads(path.read_text())
    return tuple(payload['position_std']), tuple(payload['velocity_std'])


def _lookup(table, age):
    if age <= 0 or not table:
        return 0.
    return float(table[min(age, len(table)) - 1])


def _sensor_frame(env):
    """一次合法观测:本步可见行人的真值(随后由 _corrupt 污染) + 传感器栅格。

    只读取 occlusion 模块按几何算出的 visible_ids 与 sensor_grid,不使用
    policy_entities(那里含各模式自己合成的隐藏实体,会绕过 RFS)。
    """
    if not env.occlusion_enabled():
        raise RuntimeError('本协议要求 [occlusion] enabled=true')
    occ = env.occlusion
    if occ.sensor_grid is None or occ._mesh is None:
        raise RuntimeError('occlusion 尚未更新')
    visible = set(int(i) for i in occ.visible_ids)
    entities = [dict(id=i, px=h.px, py=h.py, vx=h.vx, vy=h.vy, radius=h.radius)
                for i, h in enumerate(env.humans) if i in visible]
    return entities, occ.sensor_grid, occ._mesh


class OcclusionBelief:
    """五个臂共用同一份观测流;只有隐藏行人的表示不同。"""

    def __init__(self, config, arm='full', seed=0):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.dt = config.getfloat('env', 'time_step')
        self.rng = np.random.default_rng(seed)
        g = lambda key, default: config.getfloat('occlusion', key, fallback=default)
        self.noise_position = g('noise_position', NOISE_POSITION)
        self.noise_velocity = g('noise_velocity', NOISE_VELOCITY)
        self.detection_probability = g('detection_probability', DETECTION_PROBABILITY)
        self.protocol = config.get('occlusion', 'protocol', fallback='severe')
        # 诊断用 oracle(只用于测上限,不用于训练):
        #   none   —— 正常
        #   hidden —— 已跟踪但当前未检测到的行人,位置/速度换成真值(完美记忆)
        #   range  —— hidden + 感知范围内从未被跟踪的行人也以真值加入(完全消除遮挡)
        self.oracle = config.get('occlusion', 'oracle', fallback='none')
        #   vis_pos / vis_vel / vis_state —— 只把本步被检测到的行人的位置 / 速度 / 两者换成真值,
        #   遮挡与漏检照旧(漏检的人仍按 RFS 预测处理)。用于拆分可见行人测量噪声的代价。
        if self.oracle not in ('none', 'hidden', 'range', 'vis_pos', 'vis_vel', 'vis_state'):
            raise ValueError(f'未知 oracle: {self.oracle}')
        if arm == 'age':
            self.age_sigma, self.age_velocity = load_age_table(self.protocol)
        else:                                   # 其余臂不查表,标定时表尚不存在
            self.age_sigma, self.age_velocity = (), ()
        self.cfg = RFSConfig(dt=self.dt,
                             detection_probability=self.detection_probability,
                             position_measurement_std=self.noise_position,
                             velocity_measurement_std=self.noise_velocity)
        self.reset()

    def reset(self):
        self.rfs = BayesianRFSBelief(self.cfg)
        self.frame = None

    # ------------------------------------------------------------ 观测污染
    def _corrupt(self, entities):
        """真实注入:位置/速度高斯噪声 + 20% 随机漏检。滤波器参数与此一致。"""
        kept = []
        for e in entities:
            if self.detection_probability < 1. and self.rng.random() > self.detection_probability:
                continue                      # 漏检
            d = dict(e)
            if self.noise_position > 0.:
                d['px'] += float(self.rng.normal(0., self.noise_position))
                d['py'] += float(self.rng.normal(0., self.noise_position))
            if self.noise_velocity > 0.:
                d['vx'] += float(self.rng.normal(0., self.noise_velocity))
                d['vy'] += float(self.rng.normal(0., self.noise_velocity))
            kept.append(d)
        return kept

    # ------------------------------------------------------------ 主接口
    def observe(self, env, raw, frame, remaining):
        """raw 仅为与无遮挡观测器保持同一调用签名,本协议不使用真值向量。"""
        if self.frame is not None and frame not in (self.frame, self.frame + 1):
            raise ValueError('每回合必须 reset,且观测必须逐步连续')
        entities, sensor, mesh = _sensor_frame(env)
        if self.arm == 'gt':
            tracks = [dict(px=h.px, py=h.py, vx=h.vx, vy=h.vy, radius=h.radius,
                           existence=1., age=0, var=np.zeros(2), pvar=0.)
                      for h in env.humans]
        else:
            if frame != self.frame:
                detections = self._corrupt(entities)
                self.rfs.update(detections, sensor, mesh, frame * self.dt)
            tracks = []
            for t in self.rfs.tracks.values():
                if not (t.visible or t.existence >= self.cfg.report_existence):
                    continue
                tracks.append(dict(id=int(t.identifier), px=t.mean[0], py=t.mean[1], vx=t.mean[2], vy=t.mean[3],
                                   radius=t.radius, existence=float(t.existence),
                                   age=int(t.missed_steps),
                                   var=np.diag(t.covariance)[2:4].copy(),
                                   pvar=float(np.linalg.eigvalsh(t.covariance[:2, :2]).max()),
                                   visible=bool(t.visible)))
            if self.arm == 'sensor':
                tracks = [t for t in tracks if t.get('visible', True)]
            elif self.oracle != 'none':
                tracks = self._apply_oracle(env, tracks)
        self.frame = frame
        return self._tokens(env, tracks, remaining)

    def _apply_oracle(self, env, tracks):
        truth = {i: h for i, h in enumerate(env.humans)}
        if self.oracle.startswith('vis_'):
            for t in tracks:
                if t['age'] == 0 and t['id'] in truth:      # 只替换本步被检测到的行人
                    h = truth[t['id']]
                    if self.oracle in ('vis_pos', 'vis_state'):
                        t.update(px=h.px, py=h.py)
                    if self.oracle in ('vis_vel', 'vis_state'):
                        t.update(vx=h.vx, vy=h.vy)
            return tracks
        for t in tracks:
            if t['age'] > 0 and t['id'] in truth:          # 只替换隐藏航迹,可见行人不动
                h = truth[t['id']]
                t.update(px=h.px, py=h.py, vx=h.vx, vy=h.vy, oracle=True)
        if self.oracle == 'range':
            known = {t['id'] for t in tracks}
            radius = float(getattr(env.occlusion, 'fov_radius', 5.0))
            r = env.robot
            for i, h in truth.items():
                if i not in known and (h.px - r.px) ** 2 + (h.py - r.py) ** 2 <= radius ** 2:
                    tracks.append(dict(id=i, px=h.px, py=h.py, vx=h.vx, vy=h.vy, radius=h.radius,
                                       existence=1., age=1, var=np.zeros(2), pvar=0.,
                                       visible=False, oracle=True))
        return tracks

    def _tokens(self, env, tracks, remaining):
        r = env.robot
        direction = np.array([r.gx - r.px, r.gy - r.py], float)
        angle = float(np.arctan2(direction[1], direction[0]))
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, s], [-s, c]])
        n = len(tracks)
        humans = np.zeros((max(n, 0), 7), np.float32)
        modes = np.zeros((max(n, 0), POINTS, 2), np.float32)
        for i, t in enumerate(tracks):
            rel = rotation @ np.array([t['px'] - r.px, t['py'] - r.py])
            vel = rotation @ np.array([t['vx'] - r.vx, t['vy'] - r.vy])
            radius = t['radius']
            existence, sigma = 1., 0.
            # 本步被检测到的行人:四臂逐位一致(existence=1,不确定度=0,不展开)。
            # 测量噪声仍进入 RFS 用于得到公共的滤波均值,但可见行人的后验协方差
            # 不额外暴露给任何一臂。分叉只从漏检/遮挡之后开始。
            if t['age'] > 0:
                if self.arm == 'age':
                    sigma = min(_lookup(self.age_sigma, t['age']), SIGMA_CLIP)
                elif self.arm == 'full':
                    existence = t['existence']
                    sigma = min(float(np.sqrt(max(t['pvar'], 0.))), SIGMA_CLIP)
            humans[i] = [rel[0] / 10., rel[1] / 10., vel[0] / 2., vel[1] / 2.,
                         radius / .3, existence, sigma / SIGMA_CLIP]
            points = np.repeat(np.array([[t['vx'], t['vy']]]), POINTS, axis=0)
            spread = np.zeros(2)
            if t['age'] > 0:
                if self.arm == 'full':
                    spread = np.sqrt(np.maximum(t['var'], 0.))
                elif self.arm == 'age':
                    spread = np.full(2, _lookup(self.age_velocity, t['age']))
            if spread.any():
                offset = np.sqrt(2.5) * spread
                points[1, 0] += offset[0]; points[2, 0] -= offset[0]
                points[3, 1] += offset[1]; points[4, 1] -= offset[1]
            modes[i] = ((points - np.array([r.vx, r.vy])) @ rotation.T / 2.).astype(np.float32)
        velocity = rotation @ np.array([r.vx, r.vy])
        ego = np.array([np.linalg.norm(direction) / 10., velocity[0] / 2., velocity[1] / 2.,
                        r.radius / .3, r.v_pref / 2., remaining, 0., 0.], np.float32)
        return dict(robot=ego, humans=humans, modes=modes,
                    conditional=np.full((n, POINTS), 1. / POINTS, np.float32),
                    weights=np.full(POINTS, 1. / POINTS, np.float32),
                    mask=np.ones(n, np.float32)), rotation
