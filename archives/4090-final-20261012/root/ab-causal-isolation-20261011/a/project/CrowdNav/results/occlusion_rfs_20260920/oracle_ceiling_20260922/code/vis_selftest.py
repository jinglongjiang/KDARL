import sys, copy; sys.path.insert(0, '.')
import numpy as np
from crowd_nav.train import configuration, environment, make_belief
from crowd_sim.envs.utils.action import ActionXY
base = configuration('crowd_nav/configs/env_occlusion_rfs.config')
C = {}
for o in ('none', 'vis_pos', 'vis_vel', 'vis_state'):
    c = copy.deepcopy(base); c.set('occlusion', 'oracle', o); C[o] = c
env = environment(base, 10, 'test', 'circle_crossing'); rng = np.random.default_rng(2)
hid = {o: 0. for o in C}; err = {o: 0. for o in C}; n = 0
for case in range(12):
    env.reset(options={'test_case': 9400000 + case}); B = {o: make_belief(C[o], 'mean', case) for o in C}
    for step in range(50):
        T = {o: B[o].observe(env, None, step, 1.)[0]['humans'] for o in B}
        tr = [t for t in B['none'].rfs.tracks.values() if t.visible or t.existence >= B['none'].cfg.report_existence]
        r = env.robot; a = np.arctan2(r.gy - r.py, r.gx - r.px); R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        for j, t in enumerate(tr):
            h = env.humans[int(t.identifier)]
            pos = R @ np.array([h.px - r.px, h.py - r.py]) / 10.; vel = R @ np.array([h.vx - r.vx, h.vy - r.vy]) / 2.
            for o in ('vis_pos', 'vis_vel', 'vis_state'):
                if t.missed_steps > 0:
                    hid[o] = max(hid[o], float(np.abs(T[o][j] - T['none'][j]).max()))   # 隐藏行人必须不变
                else:
                    n += 1
                    if o in ('vis_pos', 'vis_state'): err[o] = max(err[o], float(np.abs(T[o][j][:2] - pos).max()))
                    if o in ('vis_vel', 'vis_state'): err[o] = max(err[o], float(np.abs(T[o][j][2:4] - vel).max()))
        _, _, te, trn, _ = env.step(ActionXY(*rng.normal(0, .4, 2)))
        if te or trn: break
ok = all(hid[o] == 0 and err[o] < 1e-5 for o in ('vis_pos', 'vis_vel', 'vis_state'))
print({o: (f'隐藏行人变化 {hid[o]:.1e}', f'可见行人与真值误差 {err[o]:.1e}') for o in ('vis_pos', 'vis_vel', 'vis_state')})
print('自检', '通过' if ok and n > 0 else '未通过', f'(可见行人样本 {n//3})')
