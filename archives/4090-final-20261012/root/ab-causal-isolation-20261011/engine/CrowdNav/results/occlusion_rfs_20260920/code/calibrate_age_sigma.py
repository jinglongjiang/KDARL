"""用五人标定集(case 20000-20199)标定 age 臂查表,与 full 臂后验尺度对齐。

同时报告同龄航迹后验的离散程度 —— 它决定 age 臂能否原样复现 full 臂。
选模集(21000-21399)与测试集(9400000+)都不参与标定,三者互不重叠。
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from crowd_nav.train import configuration, environment, make_belief
from crowd_sim.envs.utils.action import ActionXY

MAX_AGE, MINIMUM_SAMPLES = 40, 30


def collect(config, cases, humans=5, steps=60, seed=7):
    env = environment(config, humans, 'val', 'circle_crossing')
    rng = np.random.default_rng(seed)
    position, velocity = {}, {}
    for case in cases:
        env.reset(options={'test_case': int(case)})
        belief = make_belief(config, 'full', int(case))
        for step in range(steps):
            belief.observe(env, None, step, 1.)
            for track in belief.rfs.tracks.values():
                age = int(track.missed_steps)
                if 0 < age <= MAX_AGE:
                    position.setdefault(age, []).append(
                        float(np.sqrt(np.linalg.eigvalsh(track.covariance[:2, :2]).max())))
                    velocity.setdefault(age, []).append(
                        float(np.sqrt(np.diag(track.covariance)[2:4].mean())))
            _, _, terminated, truncated, _ = env.step(ActionXY(*rng.normal(0., .4, 2)))
            if terminated or truncated:
                break
    return position, velocity


def table(samples):
    values, spread = [], []
    for age in range(1, MAX_AGE + 1):
        rows = samples.get(age, [])
        if len(rows) < MINIMUM_SAMPLES:
            break
        values.append(float(np.mean(rows)))
        spread.append(float(np.std(rows) / max(np.mean(rows), 1e-9)))
    return values, spread


def main():
    import sys as _sys
    path = Path(_sys.argv[1]) if len(_sys.argv) > 1 else ROOT / 'crowd_nav/configs/env_occlusion_rfs.config'
    config = configuration(path)
    position, velocity = collect(config, range(20000, 20200))
    sigma, sigma_spread = table(position)
    vel, vel_spread = table(velocity)
    size = min(len(sigma), len(vel))
    sigma, vel = sigma[:size], vel[:size]
    print(f'有效支撑 age∈[1,{size}],样本门槛 {MINIMUM_SAMPLES}')
    print('位置标准差:', ' '.join(f'{v:.3f}' for v in sigma))
    print('速度标准差:', ' '.join(f'{v:.3f}' for v in vel))
    print(f'同龄离散度(变异系数)位置 max {max(sigma_spread[:size]):.4f},'
          f' 速度 max {max(vel_spread[:size]):.4f}')
    import json
    protocol = config.get('occlusion', 'protocol', fallback='severe')
    target = ROOT / 'crowd_nav/configs' / f'age_table_{protocol}.json'
    target.write_text(json.dumps(dict(
        protocol=protocol, support=size, minimum_samples=MINIMUM_SAMPLES,
        calibration_cases=[20000, 20199],
        noise=dict(position=config.getfloat('occlusion', 'noise_position', fallback=.10),
                   velocity=config.getfloat('occlusion', 'noise_velocity', fallback=.20),
                   detection_probability=config.getfloat('occlusion', 'detection_probability',
                                                         fallback=.8)),
        position_std=[round(v, 4) for v in sigma],
        velocity_std=[round(v, 4) for v in vel],
        dispersion=dict(position=round(max(sigma_spread[:size]), 4),
                        velocity=round(max(vel_spread[:size]), 4))), indent=2))
    print(f'已写入 {target}')


if __name__ == '__main__':
    main()
