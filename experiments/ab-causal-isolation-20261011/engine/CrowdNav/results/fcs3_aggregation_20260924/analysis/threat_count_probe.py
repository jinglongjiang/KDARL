"""有效威胁人数:解释 Top-K 在 k≈2-4 处饱和的原因。

对每个决策步、每个候选动作,统计有多少行人的 phi_i(a) 实际在贡献威胁。
既报多个固定阈值下的计数,也报一个免阈值的连续指标 —— 参与率(逆 Simpson):
    n_eff(a) = (sum_i phi_i)^2 / sum_i phi_i^2
对单一支配威胁 n_eff -> 1,对 m 个等量威胁 n_eff -> m。
"""
import sys, configparser, json, collections
sys.path.insert(0, '/home/abc/CrowdNav_fast')
sys.path.insert(0, '/home/abc/workspace/bayes_occ_mpc')
import numpy as np, torch, gym
import crowd_sim  # noqa: F401
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_nav.policy.sarl import SARL
from threat_mass import soft_threat

RUNS='/home/abc/workspace/classic_baselines_fast_20260922/runs'
OUT='/home/abc/workspace/fcs3_threatcount_20260925'

def build(seed,n):
    pc=configparser.RawConfigParser(); pc.read('/home/abc/CrowdNav_fast/crowd_nav/configs/policy.config')
    ec=configparser.RawConfigParser(); ec.read(f'{RUNS}/sarl_{seed}/env.config'); ec.set('sim','human_num',str(n))
    p=SARL(); p.configure(pc); p.set_device(torch.device('cpu'))
    p.get_model().load_state_dict(torch.load(f'{RUNS}/sarl_{seed}/rl_model.pth',map_location='cpu'),strict=False)
    env=gym.make('CrowdSim-v0'); env.configure(ec)
    r=Robot(ec,"robot"); r.set_policy(p); env.set_robot(r); p.set_phase("test"); p.set_env(env); r.print_info=lambda: None
    if p.action_space is None: p.build_action_space(r.v_pref)
    return p,env,r

res=collections.defaultdict(list)
for n in (5,10,20,30):
    for seed in (1207,2407):
        p,env,r=build(seed,n)
        for case in range(9400000,9400020):
            ob=env.reset('test',case)
            for _ in range(60):
                st=JointState(r.get_full_state(),ob)
                hs=list(st.human_states)
                if hs:
                    phi=soft_threat(st.self_state,hs,p.action_space)          # (A,N)
                    a=int(np.argmax(phi.sum(1)))                              # 威胁最大的那个动作
                    v=phi[a]
                    ssum=v.sum(); n_eff=float(ssum**2/max((v**2).sum(),1e-12)) if ssum>0 else 0.
                    res[n].append((len(hs), float((v>0.10).sum()), float((v>0.25).sum()),
                                   float((v>0.50).sum()), n_eff, float(v.max()), float(ssum)))
                ob,_,done,_=env.step(p.predict(st))
                if done: break
print('=== 有效威胁人数(最危险动作上,SARL 冻结策略,2 seed × 20 case)===')
print(f"{'N':>3s} {'步数':>6s} | {'可见人数':>8s} | {'φ>0.10':>7s} {'φ>0.25':>7s} {'φ>0.50':>7s} | {'n_eff':>6s} | {'max φ':>6s} {'Σφ':>6s}")
summary={}
for n in (5,10,20,30):
    a=np.array(res[n])
    if not len(a): continue
    m=a.mean(axis=0); summary[n]=m.tolist()
    print(f'{n:3d} {len(a):6d} | {m[0]:8.2f} | {m[1]:7.2f} {m[2]:7.2f} {m[3]:7.2f} | {m[4]:6.2f} | {m[5]:6.3f} {m[6]:6.2f}')
print('\n  n_eff 的分位数(免阈值的有效威胁人数):')
for n in (5,10,20,30):
    a=np.array(res[n])
    if not len(a): continue
    q=np.percentile(a[:,4],[50,75,90,95,99])
    print(f'    {n:2d}人: 中位 {q[0]:.2f}  p75 {q[1]:.2f}  p90 {q[2]:.2f}  p95 {q[3]:.2f}  p99 {q[4]:.2f}   n_eff>4 的步占 {100*(a[:,4]>4).mean():.1f}%')
import os; os.makedirs(OUT,exist_ok=True)
json.dump({str(k):v for k,v in summary.items()}, open(f'{OUT}/summary.json','w'))
