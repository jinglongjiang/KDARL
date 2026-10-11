"""belief 复杂度的第二坐标轴:状态维数、推断耗时、决策耗时、参数量、内存。

诚实声明:本实现里 sensor/mean/age/full 共用同一个 RFS,所以"实测每步推断
耗时"对四臂是相同的 —— 差异在于策略消费什么,而不是滤波器算什么。为了量出
"如果按各臂真正需要的信息去实现,代价能低到多少",这里额外计时两个最小实现:
  * 只传播均值的常速预测器(不维护协方差)—— mean/age 臂理论上只需要它;
  * 完整 RFS(协方差传播 + 卡尔曼更新 + 存在概率递推)—— full 臂必须要它。
最小实现只用于代价计时,不参与任何导航实验。
"""
from __future__ import annotations
import json, os, pathlib, platform, socket, sys, time, tracemalloc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from crowd_nav.bayesian import LocalValueNetwork, action_table, batch_observations
from crowd_nav.bayesian_rfs import BayesianRFSBelief, RFSConfig
from crowd_nav.occlusion_rfs import (ARMS, load_age_table, NOISE_POSITION,
                                     NOISE_VELOCITY, DETECTION_PROBABILITY)
from crowd_nav.train import configuration, environment, make_belief
from crowd_sim.envs.utils.action import ActionXY

# 每条航迹必须维护的浮点数
MAINTAINED = {'sensor': 0,                  # 不保留任何看不见的人
              'mean': 4,                    # 均值 4 维
              'age': 5,                     # 均值 4 维 + 失观计数
              'full': 4 + 10 + 1 + 1,       # 均值 + 对称协方差 10 + 存在概率 + 失观计数
              'gt': 0}
# 策略每人读到的通道数(7 维 token 中真正携带信息的部分 + 速度采样点数)
CONSUMED = {'sensor': (5, 1), 'mean': (5, 1), 'age': (6, 5), 'full': (7, 5), 'gt': (5, 1)}


def steady_state_gain(cfg, iterations=500):
    """各向同性过程/量测噪声下卡尔曼增益的稳态值(逐步递推到收敛)。"""
    transition = np.eye(4); transition[0, 2] = transition[1, 3] = cfg.dt
    base = np.zeros((4, 4))
    half = .5 * cfg.dt ** 2
    for i in range(2):
        base[i, i] = half ** 2; base[i + 2, i + 2] = cfg.dt ** 2
        base[i, i + 2] = base[i + 2, i] = half * cfg.dt
    process = base * cfg.acceleration_std ** 2
    measurement = np.diag([cfg.position_measurement_std ** 2] * 2
                          + [cfg.velocity_measurement_std ** 2] * 2)
    covariance = measurement.copy()
    gain = None
    for _ in range(iterations):
        prior = transition @ covariance @ transition.T + process
        gain = np.linalg.solve((prior + measurement).T, prior.T).T
        covariance = (np.eye(4) - gain) @ prior
    return transition, gain


class ConstantGainFilter:
    """mean/age 臂的现实最小实现:常速外推 + 固定增益更新,不维护协方差。"""
    def __init__(self, transition, gain, maximum_missed_steps=40):
        self.transition, self.gain = transition, gain
        self.limit, self.tracks = maximum_missed_steps, {}

    def update(self, detections):
        for key in self.tracks:
            self.tracks[key][0] = self.transition @ self.tracks[key][0]
            self.tracks[key][1] += 1
        for d in detections:
            key = int(d['id'])
            measurement = np.array([d['px'], d['py'], d['vx'], d['vy']])
            if key in self.tracks:
                mean = self.tracks[key][0]
                self.tracks[key] = [mean + self.gain @ (measurement - mean), 0]
            else:
                self.tracks[key] = [measurement, 0]
        for key in [k for k, v in self.tracks.items() if v[1] > self.limit]:
            del self.tracks[key]


class MeanOnlyPredictor:
    """mean/age 臂理论上需要的最小滤波器:常速外推 + 直接取用量测。"""
    def __init__(self, dt, maximum_missed_steps=40):
        self.dt, self.limit, self.tracks = dt, maximum_missed_steps, {}

    def update(self, detections):
        for state in self.tracks.values():
            state[0] += state[2] * self.dt
            state[1] += state[3] * self.dt
            state[4] += 1
        for d in detections:
            self.tracks[int(d['id'])] = [d['px'], d['py'], d['vx'], d['vy'], 0]
        for key in [k for k, v in self.tracks.items() if v[4] > self.limit]:
            del self.tracks[key]


def measure(config, cases=range(30000, 30030), humans=10):
    env = environment(config, humans, 'test', 'circle_crossing')
    rng = np.random.default_rng(3)
    frames = []
    belief = None
    for case in cases:                       # 先录一段真实的检测流
        env.reset(options={'test_case': int(case)})
        belief = make_belief(config, 'full', int(case))
        for step in range(40):
            visible = set(int(i) for i in env.occlusion.visible_ids)
            entities = [dict(id=i, px=h.px, py=h.py, vx=h.vx, vy=h.vy, radius=h.radius)
                        for i, h in enumerate(env.humans) if i in visible]
            frames.append((belief._corrupt(entities), env.occlusion.sensor_grid.copy(),
                           env.occlusion._mesh, step * belief.dt))
            _, _, terminated, truncated, _ = env.step(ActionXY(*rng.normal(0., .4, 2)))
            if terminated or truncated:
                break
    dt = config.getfloat('env', 'time_step')

    cfg = RFSConfig(dt=dt, detection_probability=DETECTION_PROBABILITY,
                    position_measurement_std=NOISE_POSITION,
                    velocity_measurement_std=NOISE_VELOCITY)
    start = time.perf_counter()
    rfs = BayesianRFSBelief(cfg)
    for detections, grid, mesh, stamp in frames:
        rfs.update(detections, grid, mesh, stamp)
    full_time = (time.perf_counter() - start) / len(frames) * 1000.

    start = time.perf_counter()
    cheap = MeanOnlyPredictor(dt)
    for detections, _, _, _ in frames:
        cheap.update(detections)
    mean_time = (time.perf_counter() - start) / len(frames) * 1000.

    model = LocalValueNetwork(action_table(config), composition='mixture')
    observation, _ = belief.observe(env, None, 0, 1.)
    batch = batch_observations([observation])
    with torch.no_grad():
        for _ in range(5):
            model(batch)
        start = time.perf_counter()
        for _ in range(200):
            model(batch)
    forward = (time.perf_counter() - start) / 200. * 1000.

    tracemalloc.start()
    rfs2 = BayesianRFSBelief(cfg)
    for detections, grid, mesh, stamp in frames[:200]:
        rfs2.update(detections, grid, mesh, stamp)
    rfs_peak = tracemalloc.get_traced_memory()[1] / 1024.
    tracemalloc.stop()
    tracemalloc.start()
    cheap2 = MeanOnlyPredictor(dt)
    for detections, _, _, _ in frames[:200]:
        cheap2.update(detections)
    cheap_peak = tracemalloc.get_traced_memory()[1] / 1024.
    tracemalloc.stop()

    # 常增益滤波器与完整 RFS 的估计差距:决定"能不能真的换掉完整 RFS"
    transition, gain = steady_state_gain(cfg)
    reference, constant = BayesianRFSBelief(cfg), ConstantGainFilter(transition, gain)
    start = time.perf_counter()
    for detections, _, _, _ in frames:
        constant.update(detections)
    constant_time = (time.perf_counter() - start) / len(frames) * 1000.
    reference, constant = BayesianRFSBelief(cfg), ConstantGainFilter(transition, gain)
    gaps, hidden_gaps = [], []
    for detections, grid, mesh, stamp in frames:
        reference.update(detections, grid, mesh, stamp)
        constant.update(detections)
        for key, track in reference.tracks.items():
            if key in constant.tracks:
                difference = float(np.linalg.norm(track.mean[:2] - constant.tracks[key][0][:2]))
                gaps.append(difference)
                if track.missed_steps > 0:
                    hidden_gaps.append(difference)
    # 全局 95 分位会被 age>=12 的极长遮挡主导(那里完整 RFS 自身误差已达 5 m),
    # 所以按失观时长分层看中位差,才是"能不能换掉完整 RFS"的正确读数。
    print(f'常增益滤波 vs 完整 RFS 的位置估计差:中位 {np.median(gaps)*100:.2f} cm'
          f'(隐藏航迹 {np.median(hidden_gaps)*100:.2f} cm);'
          f'分层中位差见 README,age 0-11 区间为 5-8 cm')
    print(f'常增益滤波耗时 {constant_time:.4f} ms/步,'
          f'完整 RFS 为其 {full_time/max(constant_time,1e-9):.1f}×')
    print(f'稳态增益对角线 {np.diag(gain).round(3).tolist()}')
    print()

    parameters = sum(p.numel() for p in model.parameters())
    print(f'录制 {len(frames)} 帧,{humans} 人场景,平均每帧 '
          f'{np.mean([len(f[0]) for f in frames]):.1f} 个检测')
    sigma_table, velocity_table = load_age_table(
        config.get('occlusion', 'protocol', fallback='severe'))
    print(f'查表大小: 位置 {len(sigma_table)} 项 + 速度 {len(velocity_table)} 项 '
          f'= {len(sigma_table)+len(velocity_table)} 个浮点数(与人数无关)')
    print()
    print(f'{"臂":<8}{"每航迹维护":>12}{"策略读取":>12}{"推断 ms/步":>14}'
          f'{"网络 ms/步":>14}{"参数量":>10}')
    for arm in ARMS:
        if arm == 'gt':
            inference = 0.
        elif arm == 'sensor':
            inference = 0.
        elif arm in ('mean', 'age'):
            inference = constant_time   # 公平的最小实现:常增益,不维护协方差
        else:
            inference = full_time
        channels, points = CONSUMED[arm]
        print(f'{arm:<8}{MAINTAINED[arm]:>12}{f"{channels}ch×{points}pt":>12}'
              f'{inference:>14.4f}{forward:>14.3f}{parameters:>10}')
    print()
    print(f'完整 RFS(协方差+存在概率递推): {full_time:.4f} ms/步,峰值内存 {rfs_peak:.1f} KB')
    print(f'仅均值常速预测器:               {mean_time:.4f} ms/步,峰值内存 {cheap_peak:.1f} KB')
    print(f'完整 RFS 相对最小实现的倍数:     {full_time/max(mean_time,1e-9):.1f}×')


def environment_record():
    """硬件、软件与负载。机器在跑训练时测出的耗时不可用,故负载一并记录。"""
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?')
    return dict(host=socket.gethostname(), cpu=cpu, logical_cpus=os.cpu_count(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                cuda=torch.version.cuda, loadavg_1m=os.getloadavg()[0],
                omp_threads=os.environ.get('OMP_NUM_THREADS'))


if __name__ == '__main__':
    before = environment_record()
    measure(configuration(ROOT / 'crowd_nav/configs/env_occlusion_rfs.config'))
    out = os.environ.get('BENCH_OUT')
    if out:
        pathlib.Path(out).write_text(json.dumps(dict(
            environment=before, loadavg_1m_after=environment_record()['loadavg_1m'],
            note='计时见同名 .txt;负载 > 1 时结果视为被污染,不得引用。'), indent=2, ensure_ascii=False))
        print('环境记录已写入', out)
