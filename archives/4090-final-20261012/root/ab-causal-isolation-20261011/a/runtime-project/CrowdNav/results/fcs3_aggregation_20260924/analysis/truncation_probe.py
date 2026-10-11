"""截断质量的决策层探针:回答"截掉其余行人后,为什么机器人还是选同一个动作"。

n_eff 只说明"有几个人同时在威胁",不足以说明截断是否影响决策。本探针直接测量
被截掉的威胁质量,并把它放到候选动作分差的尺度上比较:

    M(a)   = sum_i phi_i(a)                     完整累计
    M_k(a) = sum_{i in topk(a)} phi_i(a)        截断到前 k
    D_k(a) = M(a) - M_k(a)                      被丢弃的质量
    S(a)   = Q(a) - beta*M(a)                   完整校正后的动作分
    S_k(a) = Q(a) - beta*M_k(a)                 截断校正后的动作分

对每个决策步,取完整校正下的前两名动作 a1、a2:
    margin      = S(a1) - S(a2)                 原本的分差
    beta*|dD_k| = beta*|D_k(a1) - D_k(a2)|      截断对这个分差的扰动
    flip_k      = argmax S_k != a1              截断是否真的改变了 argmax
判据:若 beta*|dD_k| 远小于 margin,则截断不改变决策 —— 这才是决策层解释。
轨迹用完整校正的动作推进,即评测时实际访问的状态分布。
"""
import sys, json, collections
sys.path.insert(0, '/home/abc/workspace/sarl_cardinality_20260923')
import numpy as np
from decision_probe import build, action_values
sys.path.insert(0, '/home/abc/workspace/bayes_occ_mpc')
from threat_mass import soft_threat
from crowd_sim.envs.utils.state import JointState

SEEDS = (1207, 2407, 3607, 4807)
KS = (1, 2, 4, 8)
BETA = 0.1              # 锁定协议在开发集选出的权重
OUT = '/home/abc/workspace/sarl_cardinality_20260923'


def run(human_num, n_cases=20):
    rec = collections.defaultdict(list)
    for seed in SEEDS:
        p, env, robot = build(seed, human_num=human_num)
        for case in range(9900000, 9900000 + n_cases):
            ob = env.reset('test', case)
            for _ in range(80):
                st = JointState(robot.get_full_state(), ob)
                humans = list(st.human_states)
                if len(humans) >= 2:
                    phi = soft_threat(st.self_state, humans, p.action_space)   # (A,N)
                    q = action_values(p, st.self_state, humans)
                    mass = phi.sum(axis=1)
                    s_full = q - BETA * mass
                    order = np.argsort(s_full)
                    a1, a2 = int(order[-1]), int(order[-2])
                    margin = float(s_full[a1] - s_full[a2])
                    for k in KS:
                        if phi.shape[1] > k:
                            mk = np.partition(phi, -k, axis=1)[:, -k:].sum(axis=1)
                        else:
                            mk = mass
                        dk = mass - mk
                        s_k = q - BETA * mk
                        rec['k'].append(k)
                        rec['N_obs'].append(phi.shape[1])
                        rec['margin'].append(margin)
                        rec['retained'].append(float(mk[a1] / mass[a1]) if mass[a1] > 1e-12 else 1.0)
                        rec['discarded'].append(float(dk[a1]))
                        rec['bdD'].append(float(BETA * abs(dk[a1] - dk[a2])))
                        rec['could_flip'].append(int(BETA * abs(dk[a1] - dk[a2]) > margin))
                        rec['flip'].append(int(np.argmax(s_k) != a1))
                    a_exec = p.action_space[a1]
                else:
                    a_exec = p.predict(st)
                ob, _, done, _ = env.step(a_exec)
                if done:
                    break
        print(f'  N={human_num} seed {seed} 完成,累计 {len(rec["k"])//len(KS)} 步', flush=True)
    return {k: np.asarray(v, float) for k, v in rec.items()}


def main():
    allout = {}
    for N in (20, 30):
        print(f'=== {N} 人 ===', flush=True)
        a = run(N)
        allout[str(N)] = {k: v.tolist() for k, v in a.items()}
        steps = len(a['k']) // len(KS)
        print(f'\n  决策步 {steps},平均可见 {a["N_obs"].mean():.2f} 人,beta={BETA}')
        print(f'  {"k":>3s} {"保住质量":>9s} {"丢弃 D_k":>9s} {"beta|dD_k|":>11s} '
              f'{"动作分差":>9s} {"扰动/分差":>10s} {"可能翻转":>9s} {"实际翻转":>9s}')
        for k in KS:
            m = a['k'] == k
            ratio = a['bdD'][m] / np.maximum(a['margin'][m], 1e-12)
            print(f'  {k:>3d} {100*a["retained"][m].mean():>8.1f}% {a["discarded"][m].mean():>9.4f} '
                  f'{a["bdD"][m].mean():>11.5f} {a["margin"][m].mean():>9.5f} '
                  f'{np.median(ratio):>10.3f} {100*a["could_flip"][m].mean():>8.1f}% '
                  f'{100*a["flip"][m].mean():>8.1f}%')
        print('  (扰动/分差 为中位数;"可能翻转"= beta|dD_k| 超过前两名分差)')
    json.dump(allout, open(f'{OUT}/truncation_probe.json', 'w'))
    print('\n写入 truncation_probe.json')


if __name__ == '__main__':
    main()
