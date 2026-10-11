import sys, copy; sys.path.insert(0, '.')
import numpy as np
from crowd_nav.train import configuration, environment, make_belief
from crowd_sim.envs.utils.action import ActionXY
fails = []
for cfgname in ('env_occlusion_standard.config', 'env_occlusion_rfs.config'):
    base = configuration('crowd_nav/configs/' + cfgname)
    cfgs = {}
    for o in ('none', 'hidden', 'range'):
        c = copy.deepcopy(base); c.set('occlusion', 'oracle', o); cfgs[o] = c
    env = environment(base, 10, 'test', 'circle_crossing'); rng = np.random.default_rng(1)
    vis_gap = hid_err = extra_bad = 0.; n_hidden = n_extra = 0
    for case in range(15):
        env.reset(options={'test_case': 9400000 + case})
        B = {o: make_belief(cfgs[o], 'mean', case) for o in cfgs}
        for step in range(50):
            tok = {o: B[o].observe(env, None, step, 1.)[0]['humans'] for o in B}
            tracks = [t for t in B['none'].rfs.tracks.values()
                      if t.visible or t.existence >= B['none'].cfg.report_existence]
            k = len(tokens_none := tok['none'])
            for j, t in enumerate(tracks):
                a, b = tok['none'][j], tok['hidden'][j]
                if t.missed_steps == 0:
                    vis_gap = max(vis_gap, float(np.abs(a - b).max()))       # 可见行人必须逐位相同
                else:
                    n_hidden += 1
            # hidden 下隐藏行人的相对位置必须等于真值
            r = env.robot; ang = np.arctan2(r.gy - r.py, r.gx - r.px); c_, s_ = np.cos(ang), np.sin(ang)
            R = np.array([[c_, s_], [-s_, c_]])
            for j, t in enumerate(tracks):
                if t.missed_steps > 0:
                    h = env.humans[int(t.identifier)]
                    hid_err = max(hid_err, float(np.abs(tok['hidden'][j][:2] - R @ np.array([h.px - r.px, h.py - r.py]) / 10.).max()))
            extra = len(tok['range']) - len(tok['hidden'])
            if extra < 0: extra_bad = 1
            n_extra += extra
            _, _, te, tr, _ = env.step(ActionXY(*rng.normal(0, .4, 2)))
            if te or tr: break
    print(f'{cfgname:<32} 可见行人最大差 {vis_gap:.1e}  隐藏行人位置误差 {hid_err:.1e}  '
          f'替换隐藏航迹 {n_hidden} 次  range 额外加入 {n_extra} 人次')
    if vis_gap > 0 or hid_err > 1e-5 or extra_bad or n_hidden == 0 or n_extra == 0: fails.append(cfgname)
print('自检', '通过' if not fails else f'未通过 {fails}')
