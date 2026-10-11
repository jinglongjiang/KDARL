"""SARL 基数不敏感性的解析验证 + 真实状态上的匹配基数探针。

命题 1(恒等复制):with_global_state 的 SARL 对 N 个完全相同的行人,
  全局态 mean 等于该行人自身,故所有注意力分数相同、softmax 权重各为 1/N,
  池化结果 = f,与 N 无关。1 个危险行人与 8 个完全相同的危险行人给出同一个值。
命题 2(凸包界):softmax 权重非负且和为 1,故池化向量必落在 {f_i} 的凸包内,
  因而 ||pooled|| <= max_i ||f_i||。无论多少人,人群表示都被"最极端的单人"封顶。
本脚本对 8 个归档 checkpoint 数值验证这两条,并在真实 20 人状态上做逐人揭示探针。
"""
import sys, configparser, json
sys.path.insert(0, '/home/abc/CrowdNav_fast')
import numpy as np, torch, gym
import crowd_sim  # noqa: F401
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState, ObservableState
from crowd_nav.policy.sarl import SARL

RUNS = '/home/abc/workspace/classic_baselines_fast_20260922/runs'
SEEDS = (1207, 2407, 3607, 4807, 6007, 7207, 8407, 9607)
OUT = '/home/abc/workspace/sarl_cardinality_20260923'


def build(seed, human_num=20, **ov):
    pc = configparser.RawConfigParser(); pc.read('/home/abc/CrowdNav_fast/crowd_nav/configs/policy.config')
    for k, v in ov.items():
        pc.set('sarl', k, str(v))
    ec = configparser.RawConfigParser(); ec.read(f'{RUNS}/sarl_{seed}/env.config')
    ec.set('sim', 'human_num', str(human_num))
    p = SARL(); p.configure(pc); p.set_device(torch.device('cpu'))
    p.get_model().load_state_dict(torch.load(f'{RUNS}/sarl_{seed}/rl_model.pth', map_location='cpu'), strict=False)
    env = gym.make('CrowdSim-v0'); env.configure(ec)
    robot = Robot(ec, 'robot'); env.set_robot(robot); robot.set_policy(p)
    p.set_phase('test'); p.set_env(env); robot.print_info = lambda: None
    return p, env, robot


def pooled_and_value(model, x):
    """复现 forward 的中间量:每人特征、softmax 权重、池化向量、状态值。"""
    with torch.no_grad():
        size = x.shape
        self_state = x[:, 0, :model.self_state_dim]
        m1 = model.mlp1(x.view(-1, size[2]))
        m2 = model.mlp2(m1)
        g = torch.mean(m1.view(size[0], size[1], -1), 1, keepdim=True)
        g = g.expand((size[0], size[1], model.global_state_dim)).contiguous().view(-1, model.global_state_dim)
        scores = model.attention(torch.cat([m1, g], dim=1)).view(size[0], size[1])
        se = torch.exp(scores) * (scores != 0).float()
        w = se / se.sum(1, keepdim=True)
        f = m2.view(size[0], size[1], -1)
        pooled = (w.unsqueeze(2) * f).sum(1)
        value = model.mlp3(torch.cat([self_state, pooled], dim=1))
    return f[0], w[0], pooled[0], float(value[0, 0])


def threat_state(n, jitter=0.0, rng=None):
    """机器人在原点朝 +y;n 个行人正面迎头、同距同速。jitter=0 时为完全相同的复制。"""
    from crowd_sim.envs.utils.state import FullState
    self_state = FullState(0., -2., 0., 1., .3, 0., 4., 1., np.pi / 2)
    humans = []
    for i in range(n):
        dx = 0. if jitter == 0 else rng.uniform(-jitter, jitter)
        humans.append(ObservableState(dx, 1.0, 0., -1., .3))
    return JointState(self_state, humans)


def main():
    report = {'identical': {}, 'jittered': {}, 'convex_hull': {}, 'reveal': {}}
    rng = np.random.default_rng(0)
    for seed in SEEDS:
        p, env, robot = build(seed)
        model = p.get_model(); model.eval()

        # --- 命题 1:完全相同的复制 ---
        vals = []
        for n in (1, 2, 4, 8, 16):
            x = p.transform(threat_state(n)).unsqueeze(0)
            _, _, pooled, v = pooled_and_value(model, x)
            vals.append((n, v, float(pooled.norm())))
        report['identical'][seed] = vals

        # --- 同等威胁、轻微抖动的复制(物理可实现) ---
        jv = []
        for n in (1, 2, 4, 8, 16):
            vs = []
            for _ in range(20):
                x = p.transform(threat_state(n, 0.25, rng)).unsqueeze(0)
                _, _, pooled, v = pooled_and_value(model, x)
                vs.append((v, float(pooled.norm())))
            jv.append((n, float(np.mean([a for a, _ in vs])), float(np.mean([b for _, b in vs]))))
        report['jittered'][seed] = jv

        # --- 命题 2:真实轨迹上的凸包界 ---
        viol, worst, total = 0, 0., 0
        reveal = []
        for case in range(9900000, 9900020):
            ob = env.reset('test', case)
            for _ in range(60):
                st = JointState(robot.get_full_state(), ob)
                if not st.human_states:
                    ob, _, done, _ = env.step(p.predict(st))
                    if done: break
                    continue
                x = p.transform(st).unsqueeze(0)
                f, w, pooled, v = pooled_and_value(model, x)
                total += 1
                slack = float(f.norm(dim=1).max() - pooled.norm())
                if slack < -1e-6: viol += 1
                worst = max(worst, -slack)
                # 逐人揭示:按距离从近到远,只给策略看前 k 个人,物理局面完全不变
                if len(st.human_states) >= 6 and len(reveal) < 400:
                    order = np.argsort([np.hypot(h.px - st.self_state.px, h.py - st.self_state.py)
                                        for h in st.human_states])
                    row = []
                    for k in range(1, len(order) + 1):
                        sub = JointState(st.self_state, [st.human_states[i] for i in order[:k]])
                        xk = p.transform(sub).unsqueeze(0)
                        _, _, pk, vk = pooled_and_value(model, xk)
                        row.append((k, vk, float(pk.norm())))
                    reveal.append(row)
                ob, _, done, _ = env.step(p.predict(st))
                if done: break
        report['convex_hull'][seed] = dict(states=total, violations=viol, worst_excess=worst)
        report['reveal'][seed] = reveal
        print(f'seed {seed}: 凸包界 {total-viol}/{total} 成立,最大越界 {worst:.2e}')

    json.dump(report, open(f'{OUT}/multiplicity_probe.json', 'w'))

    print('\n=== 命题 1:N 个完全相同的迎头行人 ===')
    print(f"{'N':>3s} " + ' '.join(f'{s:>9d}' for s in SEEDS) + '   池化范数(seed1207)')
    base = {s: dict((n, (v, q)) for n, v, q in report['identical'][s]) for s in SEEDS}
    for n in (1, 2, 4, 8, 16):
        print(f'{n:3d} ' + ' '.join(f'{base[q][n][0]:9.6f}' for q in SEEDS) + f'   {base[SEEDS[0]][n][1]:.6f}')

    print('\n=== 同等威胁、轻微抖动(20 次平均) ===')
    jb = {s: dict((n, (v, q)) for n, v, q in report['jittered'][s]) for s in SEEDS}
    print(f"{'N':>3s} " + ' '.join(f'{s:>9d}' for s in SEEDS))
    for n in (1, 2, 4, 8, 16):
        print(f'{n:3d} ' + ' '.join(f'{jb[s][n][0]:9.6f}' for s in SEEDS))

    print('\n=== 逐人揭示:真实状态,物理局面不变,只改变让策略看到几个人 ===')
    print(f"{'k':>3s} {'ΔV 相对 k=1':>13s} {'池化范数比 k=1':>16s} {'样本':>6s}")
    agg = {}
    for s in SEEDS:
        for row in report['reveal'][s]:
            v1, q1 = row[0][1], row[0][2]
            for k, vk, qk in row:
                agg.setdefault(k, []).append((vk - v1, qk / max(q1, 1e-9)))
    for k in sorted(agg)[:10]:
        d = np.array(agg[k])
        print(f'{k:3d} {d[:,0].mean():13.6f} {d[:,1].mean():16.4f} {len(d):6d}')


if __name__ == '__main__':
    main()
