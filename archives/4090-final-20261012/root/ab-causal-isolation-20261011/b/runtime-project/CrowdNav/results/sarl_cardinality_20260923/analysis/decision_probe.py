"""决策层逐人揭示探针:把揭示实验从 V(s) 推到 80 个候选动作值。

对每个真实状态,按距离由近及远逐个揭示行人。揭示第 i 个人后重算 80 个动作值
    Q_k(a) = reward_k(a) + gamma^(dt*v_pref) * V(rotate([next_robot_a, next_humans_k]))
与评测时策略实际使用的计算完全一致。

核心统计(整体平移不影响 argmax,故一律同时报中心化版本):
  Delta_i(a)      = Q_k(a) - Q_{k-1}(a)
  Delta~_i(a)     = Delta_i(a) - mean_a Delta_i(a)      去掉整体平移
  受威胁动作集     = phi_i(a) 的前 10%(phi 为 CPA 软威胁,与 threat_mass 同式)
  安全动作集       = phi_i(a) 的后 10%
判据:揭示一个行人后,朝他冲过去的动作**相对**价值应当下降。Delta~ > 0 即违反。
"""
import sys, configparser, json, collections
sys.path.insert(0, '/home/abc/CrowdNav_fast')
sys.path.insert(0, '/home/abc/workspace/bayes_occ_mpc')
import numpy as np, torch, gym
import crowd_sim  # noqa: F401
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from threat_mass import soft_threat

RUNS = '/home/abc/workspace/classic_baselines_fast_20260922/runs'
SEEDS = (1207, 2407, 3607, 4807)
OUT = '/home/abc/workspace/sarl_cardinality_20260923'


def build(seed, human_num=20):
    pc = configparser.RawConfigParser(); pc.read('/home/abc/CrowdNav_fast/crowd_nav/configs/policy.config')
    ec = configparser.RawConfigParser(); ec.read(f'{RUNS}/sarl_{seed}/env.config')
    ec.set('sim', 'human_num', str(human_num))
    p = SARL(); p.configure(pc); p.set_device(torch.device('cpu'))
    p.get_model().load_state_dict(torch.load(f'{RUNS}/sarl_{seed}/rl_model.pth', map_location='cpu'), strict=False)
    env = gym.make('CrowdSim-v0'); env.configure(ec)
    robot = Robot(ec, 'robot'); env.set_robot(robot); robot.set_policy(p)
    p.set_phase('test'); p.set_env(env); robot.print_info = lambda: None
    if p.action_space is None:
        p.build_action_space(robot.v_pref)
    return p, env, robot


def action_values(policy, self_state, humans):
    """与 predict_batched_value 同式:reward + gamma^dt * V(next)。"""
    next_humans = [policy.propagate(h, ActionXY(h.vx, h.vy)) for h in humans]
    next_robots = [policy.propagate(self_state, a) for a in policy.action_space]
    x = torch.tensor([[r + h for h in next_humans] for r in next_robots], dtype=torch.float32)
    rot = policy.rotate(x.reshape(-1, x.shape[-1]))
    with torch.no_grad():
        v = policy.model(rot.reshape(len(next_robots), len(humans), -1)).reshape(-1).numpy()
    r = np.array([MultiHumanRL.compute_reward(policy, nr, next_humans) for nr in next_robots])
    return r + policy.gamma ** (policy.time_step * self_state.v_pref) * v


def main():
    stats = collections.defaultdict(list)
    per_outcome = collections.defaultdict(lambda: collections.defaultdict(list))
    for seed in SEEDS:
        p, env, robot = build(seed)
        for case in range(9900000, 9900030):
            ob = env.reset('test', case)
            rows, done, info = [], False, None
            for _ in range(80):
                st = JointState(robot.get_full_state(), ob)
                humans = list(st.human_states)
                if len(humans) >= 4:
                    order = np.argsort([np.hypot(h.px - st.self_state.px, h.py - st.self_state.py)
                                        for h in humans])
                    ordered = [humans[i] for i in order]
                    qs = [action_values(p, st.self_state, ordered[:k]) for k in range(1, len(ordered) + 1)]
                    for k in range(1, len(ordered)):
                        newcomer = ordered[k]
                        phi = soft_threat(st.self_state, [newcomer], p.action_space)[:, 0]
                        d = qs[k] - qs[k - 1]
                        dc = d - d.mean()
                        hi = phi >= np.quantile(phi, 0.9)
                        lo = phi <= np.quantile(phi, 0.1)
                        if hi.sum() == 0 or lo.sum() == 0:
                            continue
                        rows.append(dict(
                            k=k + 1,
                            shift=float(d.mean()),
                            thr_centered=float(dc[hi].mean()),
                            safe_centered=float(dc[lo].mean()),
                            contrast=float(dc[hi].mean() - dc[lo].mean()),
                            top1_changed=int(np.argmax(qs[k]) != np.argmax(qs[k - 1])),
                            top1_phi_up=int(phi[int(np.argmax(qs[k]))] > phi[int(np.argmax(qs[k - 1]))] + 1e-9),
                            top3_changed=int(set(np.argsort(qs[k])[-3:]) != set(np.argsort(qs[k - 1])[-3:])),
                        ))
                ob, _, done, info = env.step(p.predict(st))
                if done:
                    break
            outcome = type(info).__name__ if info is not None else 'none'
            for r in rows:
                for key, val in r.items():
                    stats[key].append(val)
                    per_outcome[outcome][key].append(val)
        print(f'seed {seed} 完成,累计比较 {len(stats["k"])}', flush=True)

    json.dump({k: v for k, v in stats.items()}, open(f'{OUT}/decision_probe.json', 'w'))
    a = {k: np.array(v, float) for k, v in stats.items()}
    n = len(a['k'])
    print(f'\n=== 决策层逐人揭示({len(SEEDS)} seed,20 人,{n} 次"多揭示一人"比较)===')
    print(f'  80 个动作值的整体平移        {a["shift"].mean():+.6f}  (>0 即 V 普遍升高,与状态层一致)')
    print(f'  中心化后 · 受威胁动作(前10%) {a["thr_centered"].mean():+.6f}')
    print(f'  中心化后 · 安全动作(后10%)   {a["safe_centered"].mean():+.6f}')
    print(f'  对比度(受威胁 - 安全)        {a["contrast"].mean():+.6f}   机制正确时应为负')
    viol = (a['contrast'] > 0).mean()
    print(f'  决策级违反率(对比度>0)       {100*viol:.1f}%')
    print(f'\n  top-1 动作改变              {100*a["top1_changed"].mean():.1f}%')
    print(f'  其中改到"更朝新行人冲"的     {100*a["top1_phi_up"].mean():.1f}%  '
          f'(占已改变者 {100*a["top1_phi_up"].sum()/max(a["top1_changed"].sum(),1):.1f}%)')
    print(f'  top-3 集合改变              {100*a["top3_changed"].mean():.1f}%')

    print('\n=== 按该回合最终结局拆分 ===')
    print(f'{"结局":>14s} {"比较数":>7s} {"对比度":>10s} {"违反率":>8s} {"top1改变":>9s} {"改到更危险":>10s}')
    for outcome, d in sorted(per_outcome.items(), key=lambda kv: -len(kv[1]['k'])):
        b = {k: np.array(v, float) for k, v in d.items()}
        print(f'{outcome:>14s} {len(b["k"]):7d} {b["contrast"].mean():+10.6f} '
              f'{100*(b["contrast"]>0).mean():7.1f}% {100*b["top1_changed"].mean():8.1f}% '
              f'{100*b["top1_phi_up"].mean():9.1f}%')


if __name__ == '__main__':
    main()
