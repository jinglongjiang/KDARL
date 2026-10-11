"""量化各密度下策略实际看到的行人数(遮挡后)。n_ref 必须由此确定,而不是拍 5。"""
import sys, configparser, collections
sys.path.insert(0, '/home/abc/CrowdNav_fast')
import numpy as np, torch, gym
import crowd_sim  # noqa: F401
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_nav.policy.sarl import SARL

RUNS = '/home/abc/workspace/classic_baselines_fast_20260922/runs'
pc = configparser.RawConfigParser(); pc.read('/home/abc/CrowdNav_fast/crowd_nav/configs/policy.config')

for phase, human_num, cases in (('train', 5, range(300)), ('test', 5, range(9900000, 9900050)),
                                ('test', 10, range(9900000, 9900050)), ('test', 20, range(9900000, 9900050))):
    ec = configparser.RawConfigParser(); ec.read(f'{RUNS}/sarl_1207/env.config')
    ec.set('sim', 'human_num', str(human_num))
    policy = SARL(); policy.configure(pc); policy.set_device(torch.device('cpu'))
    policy.get_model().load_state_dict(torch.load(f'{RUNS}/sarl_1207/rl_model.pth', map_location='cpu'))
    env = gym.make('CrowdSim-v0'); env.configure(ec)
    robot = Robot(ec, 'robot'); env.set_robot(robot); robot.set_policy(policy)
    policy.set_phase('test'); policy.set_env(env); robot.print_info = lambda: None
    counts = []
    for case in cases:
        ob = env.reset(phase, case if phase == 'test' else None)
        for _ in range(80):
            counts.append(len(ob))
            st = JointState(robot.get_full_state(), ob)
            if not st.human_states:
                ob, _, done, _ = env.step(policy.predict(st))
            else:
                ob, _, done, _ = env.step(policy.predict(st))
            if done:
                break
    c = np.array(counts)
    hist = collections.Counter(c.tolist())
    print(f'{phase} human_num={human_num:2d}  steps={len(c):5d}  mean={c.mean():5.3f}  '
          f'median={np.median(c):4.1f}  p90={np.percentile(c,90):4.1f}  max={c.max():2d}  '
          f'零可见={100*(c==0).mean():4.1f}%')
    print('   分布:', dict(sorted(hist.items())))
