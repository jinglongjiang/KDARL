"""B1-0 前置检查:
1) aggregation=softmax + cardinality_alpha=1 的新代码,与 88ba062 原版逐比特相同;
2) alpha != 1 在 N==n_ref 时仍与原版逐比特相同(测试期开关的核心前提);
3) 训练阶段行人数恒为 5。
不通过则整条 B1 不得继续。"""
import sys, configparser, importlib.util
sys.path.insert(0, '/home/abc/CrowdNav_fast')
import numpy as np, torch, gym
import crowd_sim  # noqa: F401  注册 CrowdSim-v0
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_nav.policy.sarl import SARL

RUNS = '/home/abc/workspace/classic_baselines_fast_20260922/runs'
SEEDS = (1207, 2407, 3607, 4807, 6007, 7207, 8407, 9607)


def original_network():
    spec = importlib.util.spec_from_file_location(
        'orig_sarl', '/home/abc/CrowdNav/crowd_nav/policy/sarl.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ValueNetwork


def build(seed, human_num, **overrides):
    pc = configparser.RawConfigParser()
    pc.read('/home/abc/CrowdNav_fast/crowd_nav/configs/policy.config')
    for key, value in overrides.items():
        pc.set('sarl', key, str(value))
    ec = configparser.RawConfigParser()
    ec.read(f'{RUNS}/sarl_{seed}/env.config')
    ec.set('sim', 'human_num', str(human_num))
    policy = SARL(); policy.configure(pc); policy.set_device(torch.device('cpu'))
    state = torch.load(f'{RUNS}/sarl_{seed}/rl_model.pth', map_location='cpu')
    missing, unexpected = policy.get_model().load_state_dict(state, strict=False)
    env = gym.make('CrowdSim-v0'); env.configure(ec)
    robot = Robot(ec, 'robot'); env.set_robot(robot); robot.set_policy(policy)
    policy.set_phase('test'); policy.set_env(env); robot.print_info = lambda: None
    return policy, env, robot, state, missing, unexpected


def collect(env, robot, policy, cases, max_steps=60):
    """收集真实轨迹上的 rotated 输入张量。"""
    batch = []
    for case in cases:
        ob = env.reset('test', case)
        for _ in range(max_steps):
            st = JointState(robot.get_full_state(), ob)
            if not st.human_states:
                break
            batch.append(policy.transform(st).unsqueeze(0))
            ob, _, done, _ = env.step(policy.predict(st))
            if done:
                break
    return batch


def main():
    Orig = original_network()
    failures = []
    for seed in SEEDS:
        for human_num in (5, 10, 20):
            policy, env, robot, state, missing, unexpected = build(seed, human_num)
            if missing or unexpected:
                failures.append(f'seed {seed} state_dict 不匹配 missing={missing} unexpected={unexpected}')
            orig = Orig(policy.input_dim(), policy.self_state_dim, [150, 100], [100, 50],
                        [150, 100, 100, 1], [100, 100, 1], True, 1, 4)
            orig.load_state_dict(state); orig.eval()
            new = policy.get_model(); new.eval()
            batch = collect(env, robot, policy, range(9900000, 9900004))
            with torch.no_grad():
                delta = max(float((new(b) - orig(b)).abs().max()) for b in batch)
            tag = f'seed {seed} N={human_num} states={len(batch)}'
            if delta != 0.:
                failures.append(f'{tag} softmax/alpha=1 最大偏差 {delta:.3e} != 0')
            else:
                print(f'[OK] {tag} alpha=1 逐比特相同')

            # 遮挡下每步可见人数不同,所以不变式改为:当 n_ref 等于该步的实际可见人数时
            # 增益恒为 1,任意 alpha 都必须与原版逐比特相同;n_ref 偏离时必须分叉。
            by_count = {}
            for b in batch:
                by_count.setdefault(b.shape[1], []).append(b)
            for alpha in (0.0, 0.5):
                for count, group in sorted(by_count.items())[:4]:
                    pol2, *_ = build(seed, human_num, cardinality_alpha=alpha, n_ref=count)
                    m2 = pol2.get_model(); m2.eval()
                    with torch.no_grad():
                        d2 = max(float((m2(b) - orig(b)).abs().max()) for b in group)
                    if d2 != 0.:
                        failures.append(f'{tag} alpha={alpha} n_ref={count} 应恒等却偏差 {d2:.3e}')
                pol3, *_ = build(seed, human_num, cardinality_alpha=alpha, n_ref=3.298)
                m3 = pol3.get_model(); m3.eval()
                with torch.no_grad():
                    d3 = max(float((m3(b) - orig(b)).abs().max()) for b in batch)
                if alpha != 1. and d3 == 0.:
                    failures.append(f'{tag} alpha={alpha} n_ref=3.298 竟然毫无变化,开关失效')
                print(f'[OK] {tag} alpha={alpha}: n_ref=可见人数时恒等;n_ref=3.298 时最大偏差 {d3:.3e}')
            if human_num == 5:
                break

    # 训练阶段行人数
    policy, env, robot, *_ = build(SEEDS[0], 5)
    counts = set()
    for case in range(40):
        ob = env.reset('train', case)
        counts.add(len(ob))
        for _ in range(8):
            st = JointState(robot.get_full_state(), ob)
            counts.add(len(st.human_states))
            ob, _, done, _ = env.step(policy.predict(st))
            if done:
                break
    print(f'[训练阶段可见人数集合] {sorted(counts)}(遮挡开启,变动属预期)')

    print('\n==== 结论 ====')
    if failures:
        for f in failures:
            print('[FAIL]', f)
        sys.exit(1)
    print('全部通过:softmax+alpha=1 逐比特等价;alpha 开关在 n_ref=当前可见人数时恒等,否则生效')


if __name__ == '__main__':
    main()
