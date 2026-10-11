"""同状态配对计时:把 ACT 的开销从"跨 rollout 均值相减"变成隔离测量。

稿件原先的 0.46/0.26/0.10 ms 来自两组各自独立的闭环 rollout,访问状态与可见人数
都不同,只能算端到端观测差。本探针先冻结同一批状态,再在同一批状态上交替计时:

    A: Q(a)                        策略自身的决策成本
    B: Q(a) - beta*R(a)            策略 + ACT

交替顺序做平衡(ABBA),丢弃预热轮,按状态配对给区间。
"""
import sys, json, time
sys.path.insert(0, '/home/abc/workspace/sarl_cardinality_20260923')
import numpy as np
from decision_probe import build, action_values
sys.path.insert(0, '/home/abc/workspace/bayes_occ_mpc')
from threat_mass import ThreatMass
from crowd_sim.envs.utils.state import JointState

SEEDS = (1207, 2407, 3607, 4807)
BETA, REPEATS, WARMUP = 0.1, 7, 2
OUT = '/home/abc/workspace/sarl_cardinality_20260923'


def collect(human_num, n_cases=6, max_steps=60):
    """冻结一批真实访问状态(用 ACT 校正的动作推进)。"""
    tm = ThreatMass(BETA, 1.0, 0, 'sum')
    states = []
    for seed in SEEDS:
        p, env, robot = build(seed, human_num=human_num)
        for case in range(9900000, 9900000 + n_cases):
            ob = env.reset('test', case)
            for _ in range(max_steps):
                st = JointState(robot.get_full_state(), ob)
                if len(list(st.human_states)) >= 1:
                    states.append((p, st))
                    q = action_values(p, st.self_state, list(st.human_states))
                    a = p.action_space[int(np.argmax(q - tm.penalty(st, p.action_space)))]
                else:
                    a = p.predict(st)
                ob, _, done, _ = env.step(a)
                if done:
                    break
    return states


def time_states(states, with_act):
    tm = ThreatMass(BETA, 1.0, 0, 'sum')
    out = np.empty(len(states))
    for j, (p, st) in enumerate(states):
        humans = list(st.human_states)
        t0 = time.perf_counter()
        q = action_values(p, st.self_state, humans)
        if with_act:
            q = q - tm.penalty(st, p.action_space)
        int(np.argmax(q))
        out[j] = (time.perf_counter() - t0) * 1e3
    return out


def main():
    res = {}
    for N in (5, 10, 20):
        states = collect(N)
        a_runs, b_runs = [], []
        # ABBA 平衡:每轮先 A 后 B,次轮先 B 后 A
        for r in range(REPEATS + WARMUP):
            if r % 2 == 0:
                a = time_states(states, False); b = time_states(states, True)
            else:
                b = time_states(states, True); a = time_states(states, False)
            if r >= WARMUP:
                a_runs.append(a); b_runs.append(b)
        A = np.median(np.stack(a_runs), axis=0)      # 每个状态取跨轮中位数
        B = np.median(np.stack(b_runs), axis=0)
        d = B - A
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        res[str(N)] = dict(n_states=n, policy_ms=float(A.mean()), act_ms=float(B.mean()),
                           delta_ms=float(d.mean()), ci=[float(lo), float(hi)],
                           delta_median=float(np.median(d)), pct=float(100 * d.mean() / A.mean()))
        print(f'  N={N:2d}  状态 {n:5d}  策略 {A.mean():7.3f} ms  +ACT {B.mean():7.3f} ms  '
              f'差 {d.mean():+.4f} ms [{lo:+.4f},{hi:+.4f}]  中位差 {np.median(d):+.4f}  '
              f'占比 {100*d.mean()/A.mean():.2f}%', flush=True)
    json.dump(res, open(f'{OUT}/timing_probe.json', 'w'))
    print('写入 timing_probe.json')


if __name__ == '__main__':
    main()
