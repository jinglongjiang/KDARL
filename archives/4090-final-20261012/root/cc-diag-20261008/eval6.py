# -*- coding: utf-8 -*-
"""贝叶斯线 6 环境评测。协议严格照抄 KDA 侧 tools/run_vl_minimal.py::evaluate,
只替换配置/checkpoint 的路径布局, 保证两条线的数字可直接对照。

与 KDA 侧的唯一实现差异: 不调用 initialize_shared (该 helper 在贝叶斯树中不存在)。
因为随后 load_state_dict(strict=True) 会覆盖全部参数与 buffer, 且 strict 会在
缺键/多键时直接报错, 所以不影响被评权重。
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

CELLS = [(5, 'circle_crossing'), (5, 'square_crossing'),
         (10, 'circle_crossing'), (10, 'square_crossing'),
         (20, 'circle_crossing'), (20, 'square_crossing')]
CASES = list(range(89000, 89032))   # 合并后 KDA 跑与贝叶斯跑统一使用这一组


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def pin_project(project):
    """把目标项目根强制置顶, 并硬断言 crowd_nav/crowd_sim 确实解析到它。

    本机同时存在多份 CrowdNav 副本(soc-nav-training 等), 不置顶会静默导入错的那份。
    """
    root = str(Path(project).resolve())
    sys.path.insert(0, root)
    import crowd_nav, crowd_sim, crowd_nav.policy.mamba_rl as _m
    for mod in (crowd_nav, crowd_sim, _m):
        got = str(Path(mod.__file__).resolve())
        assert got.startswith(root), f"{mod.__name__} 解析到 {got}, 不在 {root} 下"
    return root


def evaluate(arm_dir, backbone, out_json, seed, cells, cases, budget=10000, config=None,
             legacy_scorer=False):
    from crowd_nav.train import build_env_and_robot, bind_policy, load_config
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_sim.envs.utils.state import JointState

    if legacy_scorer:
        # 把候选打分换回合并前的参考实现, 用于隔离"合并本身"对评测结果的影响
        if hasattr(MambaRLPolicy, '_score_sarl_candidates_legacy'):
            MambaRLPolicy.score_sarl_candidates = MambaRLPolicy._score_sarl_candidates_legacy
            print('[LEGACY] score_sarl_candidates -> _score_sarl_candidates_legacy')
        elif hasattr(MambaRLPolicy, '_predict_sarl_style_legacy'):
            MambaRLPolicy.predict_sarl_style = MambaRLPolicy._predict_sarl_style_legacy
            print('[LEGACY] predict_sarl_style -> _predict_sarl_style_legacy')
        else:
            raise SystemExit('该树没有 legacy 方法, 无法做隔离对照')

    arm_dir = Path(arm_dir)
    checkpoint = arm_dir / f'rl_model_ep{budget}.pth'
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert data['episode'] == budget, f"episode={data['episode']} != {budget}"

    cfg = load_config(str(config) if config else str(arm_dir / 'config.ini'))
    torch.manual_seed(seed)
    cfg.set('mamba', 'temporal_backbone', backbone)
    net = MambaRLPolicy(cfg, device='cuda')
    # torch.compile 会加 _orig_mod. 前缀; 用 3.8 兼容写法去掉
    weights = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
               for k, v in data['policy_state'].items()}
    net.load_state_dict(weights, strict=True)
    net = net.eval()
    net.multiagent_training = True
    net.use_sarl_predict = True
    net.set_phase('test')
    net.set_training_mode('rl')
    net.epsilon = 0.

    rows, latencies = [], []
    started = time.time()
    with torch.no_grad():
        for n, geometry in cells:
            cfg.set('sim', 'human_num', str(n))
            cfg.set('sim', 'test_sim', geometry)
            env, robot = build_env_and_robot(cfg)
            bind_policy(robot, net, env, epsilon=0.)
            env.phase = 'test'
            for case in cases:
                if hasattr(net, 'reset_episode_stats'):
                    net.reset_episode_stats()
                env.reset(seed=case, options={'test_case': case})
                assert len(env.humans) == n, f"humans={len(env.humans)} != {n}"
                result, total_return, minimum, steps = 'timeout', 0., float('inf'), 0
                while steps < 201:
                    state = JointState(robot.get_full_state(),
                                       [h.get_observable_state() for h in env.humans])
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    action = net.predict(state)
                    torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - tick) * 1000)
                    step = env.step(action)
                    if len(step) == 5:
                        _, reward, terminal, truncated, info = step
                        done = terminal or truncated
                    else:
                        _, reward, done, info = step
                    total_return += .99 ** steps * float(reward)
                    minimum = min(minimum,
                                  min(float(np.hypot(h.px - robot.px, h.py - robot.py)
                                            - h.radius - robot.radius) for h in env.humans))
                    steps += 1
                    if done:
                        event = str(info.get('event', '')) if isinstance(info, dict) else type(info).__name__
                        event = event.lower().replace('_', '')
                        if 'success' in event or 'reachgoal' in event:
                            result = 'success'
                        elif 'collision' in event:
                            result = 'collision'
                        break
                rows.append({'case': case, 'human_num': n, 'geometry': geometry,
                             'outcome': result, 'return': total_return,
                             'minimum_clearance': minimum, 'time': steps * .25})
                save(str(out_json) + '.progress', {'episodes': rows})
    save(out_json, {'checkpoint_sha256': sha(checkpoint), 'episodes': rows,
                    'wall_seconds': time.time() - started,
                    'policy_latency_ms_median': float(np.median(latencies)),
                    'policy_latency_ms_p95': float(np.percentile(latencies, 95)),
                    'latency_includes_full_80_action_scoring': True,
                    'backbone': backbone, 'arm_dir': str(arm_dir), 'budget': budget,
                    'protocol': {'cells': cells, 'cases': [cases[0], cases[-1]], 'n_cases': len(cases)}})
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm-dir', required=True)
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--project', required=True, help='项目根(含 crowd_nav/ 与 crowd_sim/)')
    ap.add_argument('--budget', type=int, default=10000)
    ap.add_argument('--smoke', action='store_true', help='只跑 1 cell x 2 case')
    ap.add_argument('--cells', type=str, default=None,
                    help='只跑这些 cell 下标(逗号分隔), 用于分片并行; 省略=全部 6 个')
    ap.add_argument('--n-cases', type=int, default=None, help='每个 cell 的 case 数; 默认 32, 可设 500 做高分辨率评测')
    ap.add_argument('--config', default=None, help='配置文件路径; 省略=<arm-dir>/config.ini')
    ap.add_argument('--case-base', type=int, default=None, help='case 起点; 省略=89000')
    ap.add_argument('--legacy-scorer', action='store_true',
                    help='改用合并前的参考打分实现(同权重对照)')
    a = ap.parse_args()
    if a.smoke:
        cells, cases = [CELLS[0]], CASES[:2]
    else:
        cells = CELLS if a.cells is None else [CELLS[int(i)] for i in a.cells.split(',')]
        nb = 32 if a.n_cases is None else int(a.n_cases)
        cb = 89000 if a.case_base is None else int(a.case_base)
        cases = list(range(cb, cb + nb))
    root = pin_project(a.project)
    print(f'[PIN] 项目根 = {root}')
    rows = evaluate(a.arm_dir, a.backbone, a.out, a.seed, cells, cases, a.budget, a.config, a.legacy_scorer)
    n = len(rows); s = sum(1 for r in rows if r['outcome'] == 'success')
    c = sum(1 for r in rows if r['outcome'] == 'collision')
    print(f"[DONE] {a.backbone}  n={n}  SR={s/n*100:.2f}%  CR={c/n*100:.2f}%  TO={(n-s-c)/n*100:.2f}%")
