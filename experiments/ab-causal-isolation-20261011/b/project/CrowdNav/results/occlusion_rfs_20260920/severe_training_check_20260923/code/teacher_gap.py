import sys; sys.path.insert(0, '.')
import numpy as np
from crowd_nav.train import configuration, environment, make_belief, expert
from crowd_nav.bayesian import action_table
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState, JointState

def orca_on(config, raw, humans):
    t = ORCA(); t.configure(config)
    t.safety_space = config.getfloat('policy', 'teacher_safety_space', fallback=t.safety_space)
    r = raw[:9]
    s = FullState(r[0], r[1], r[4], r[5], r[6], r[2], r[3], r[7], r[8])
    a = t.predict(JointState(s, [ObservableState(*h) for h in humans]))
    return np.array([a.vx, a.vy])

for name in ('env_occlusion_standard.config', 'env_occlusion_rfs.config'):
    config = configuration('crowd_nav/configs/' + name); table = action_table(config)
    env = environment(config, 5, 'train', 'circle_crossing')
    same = n = 0; dist = []; hidden_steps = same_h = 0
    for case in range(10000, 10060):                       # 与正式示范同一段 case
        raw, _ = env.reset(options={'test_case': case}); belief = make_belief(config, 'mean', case)
        for step in range(int(np.ceil(env.time_limit / env.time_step))):
            _, rot = belief.observe(env, raw, step, 1.)
            tracks = [t for t in belief.rfs.tracks.values() if t.visible or t.existence >= belief.cfg.report_existence]
            obs_h = [[t.mean[0], t.mean[1], t.mean[2], t.mean[3], t.radius] for t in tracks]
            a_true = expert(config, raw)
            a_obs = orca_on(config, raw, obs_h) if obs_h else a_true * 0 + orca_on(config, raw, [[1e3, 1e3, 0, 0, .3]])
            i_true = int(np.square(table - rot @ a_true).sum(1).argmin()); i_obs = int(np.square(table - rot @ a_obs).sum(1).argmin())
            n += 1; same += i_true == i_obs; dist.append(float(np.linalg.norm(a_true - a_obs)))
            if len(tracks) < len(env.humans) or any(t.missed_steps > 0 for t in tracks):
                hidden_steps += 1; same_h += i_true == i_obs
            raw, _, te, tr, _ = env.step(ActionXY(*a_true))
            if te or tr: break
    d = np.array(dist)
    print(f'{name:<32} 步数 {n}  离散动作一致 {100*same/n:5.1f}%  速度差 中位 {np.median(d):.3f} / 90分位 {np.quantile(d,.9):.3f} m/s  '
          f'| 有遮挡或漏检的步 {100*hidden_steps/n:4.1f}%,其中一致 {100*same_h/max(hidden_steps,1):5.1f}%')
