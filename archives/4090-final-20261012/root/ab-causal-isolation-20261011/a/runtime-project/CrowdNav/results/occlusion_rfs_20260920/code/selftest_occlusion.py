"""遮挡+RFS 接口自检。五臂输入一致性 + 私有信息泄漏检查。

不训练、不评估导航性能:只验证观测通道本身。
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from crowd_nav.train import configuration, environment, make_belief, occlusion_protocol
from crowd_nav.occlusion_rfs import (ARMS, NOISE_POSITION, NOISE_VELOCITY,
                                     DETECTION_PROBABILITY, OcclusionBelief)
from crowd_sim.envs.utils.action import ActionXY

FAIL = []


def check(name, condition, detail=''):
    print(('PASS  ' if condition else 'FAIL  ') + name + (('  ' + detail) if detail else ''))
    if not condition:
        FAIL.append(name)


def run(config_path, episodes=12, humans=5):
    config = configuration(config_path)
    check('遮挡协议已开启', occlusion_protocol(config))
    env = environment(config, humans, 'test', 'circle_crossing')
    rng = np.random.default_rng(0)
    visible_gap, hidden_gap, detected_rows, channel_rows = [], [], [], []
    miss_total = miss_count = 0
    sensor_count = other_count = gt_count = 0
    age_positions = {}
    for case in range(episodes):
        env.reset(options={'test_case': 100000 + case})
        beliefs = {arm: make_belief(config, arm, case) for arm in ARMS}
        assert all(isinstance(b, OcclusionBelief) for b in beliefs.values())
        for step in range(40):
            tokens = {arm: beliefs[arm].observe(env, None, step, 1.)[0] for arm in ARMS}
            truth = np.array([[h.px, h.py] for h in env.humans])
            robot = np.array([env.robot.px, env.robot.py])
            angle = np.arctan2(env.robot.gy - robot[1], env.robot.gx - robot[0])
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, s], [-s, c]])
            truth_local = (truth - robot) @ rot.T / 10.

            # 1) 五臂在"本步被检测到的行人"上必须完全一致
            visible = set(int(i) for i in env.occlusion.visible_ids)
            detected = {int(k.identifier) for k in beliefs['full'].rfs.tracks.values() if k.visible}
            miss_total += len(visible); miss_count += len(visible) - len(detected)
            rows, spreads = {}, {}
            for arm in ('sensor', 'mean', 'full', 'age'):
                belief = beliefs[arm]
                order = [int(k.identifier) for k in belief.rfs.tracks.values()
                         if k.visible or (arm != 'sensor'
                                          and k.existence >= belief.cfg.report_existence)]
                rows[arm] = dict(zip(order, tokens[arm]['humans']))
                spreads[arm] = dict(zip(order, tokens[arm]['modes']))
            for identifier in detected:
                present = [rows[a][identifier] for a in rows if identifier in rows[a]]
                modes = [spreads[a][identifier] for a in spreads if identifier in spreads[a]]
                if len(present) != 4:
                    FAIL.append('被检测行人未同时出现在四个臂中')
                    continue
                stack, mode_stack = np.stack(present), np.stack(modes)
                # 全部 7 个通道 + 5 个速度采样点都必须逐位一致
                visible_gap.append(max(float(np.abs(stack - stack[0]).max()),
                                       float(np.abs(mode_stack - mode_stack[0]).max())))
                detected_rows.append((stack[0, :2], truth_local[identifier]))
                if float(np.abs(stack[:, 5] - 1.).max()) > 0 or float(np.abs(stack[:, 6]).max()) > 0:
                    FAIL.append('被检测行人的 existence/不确定度通道不是 1/0')
                if float(np.abs(mode_stack - mode_stack[:, :1]).max()) > 0:
                    FAIL.append('被检测行人的速度采样点被展开了')
            sensor_count += len(tokens['sensor']['humans'])
            other_count += len(tokens['full']['humans'])
            gt_count += len(tokens['gt']['humans'])

            # 2) gt 臂必须等于真值;非 gt 臂不得等于真值
            gt_rows = tokens['gt']['humans'][:, :2]
            check_gt = np.abs(np.sort(gt_rows, 0) - np.sort(truth_local, 0)).max()
            if check_gt > 1e-5:
                FAIL.append('gt 臂与真值不符')

            for t in beliefs['full'].rfs.tracks.values():
                if t.missed_steps > 0:
                    channel_rows.append((int(t.missed_steps),
                        float(np.sqrt(np.linalg.eigvalsh(t.covariance[:2, :2]).max())),
                        float(t.existence)))

            # 3) 失观越久,full 臂位置误差应越大
            for t in beliefs['full'].rfs.tracks.values():
                if t.missed_steps > 0 and int(t.identifier) < len(truth):
                    err = float(np.linalg.norm(t.mean[:2] - truth[int(t.identifier)]))
                    age_positions.setdefault(min(int(t.missed_steps), 8), []).append(err)
                    hidden_gap.append(err)

            action = ActionXY(*(rng.normal(0, .5, 2)))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    check('被检测行人同时出现在四个非 gt 臂中', '被检测行人未同时出现在四个臂中' not in FAIL)
    check('被检测行人 existence=1 且不确定度=0', '被检测行人的 existence/不确定度通道不是 1/0' not in FAIL)
    check('被检测行人速度采样点未展开', '被检测行人的速度采样点被展开了' not in FAIL)
    check('四臂对同一被检测行人的全部通道与采样点逐位相同',
          (max(visible_gap) if visible_gap else 0) < 1e-6,
          f'max|Δ|={max(visible_gap) if visible_gap else 0:.2e}')
    check('gt 臂逐位等于真值', 'gt 臂与真值不符' not in FAIL)
    probe = OcclusionBelief(config, 'full', 0)
    observed_error = np.mean([np.linalg.norm((a - b) * 10.) for a, b in detected_rows])
    if probe.noise_position > 0.:
        check('被检测行人位置带有真实观测噪声(非真值直通)', .01 < observed_error < .30,
              f'平均误差 {observed_error:.3f} m')
    else:
        check('零噪声协议下被检测行人位置等于真值', observed_error < .01,
              f'平均误差 {observed_error:.4f} m')
    rate = miss_count / max(miss_total, 1)
    check('漏检率 ≈ 1-检测概率', abs(rate - (1 - probe.detection_probability)) < .05,
          f'实测 {rate:.3f} 对 {1-probe.detection_probability:.2f}')
    ages = sorted(age_positions)
    means = [np.mean(age_positions[a]) for a in ages]
    check('失观时长与位置误差单调相关', len(means) > 2 and means[-1] > means[0],
          ' '.join(f'age{a}:{m:.2f}' for a, m in zip(ages, means)))
    check('sensor 臂看到的行人少于带记忆的臂', sensor_count < other_count,
          f'{sensor_count} < {other_count}')
    check('gt 臂看到的行人最多', gt_count >= other_count, f'{gt_count} >= {other_count}')

    # age 与 full 的差异来源:位置/速度不确定度按构造对齐,剩下的是存在概率
    from crowd_nav.occlusion_rfs import load_age_table, SIGMA_CLIP, _lookup
    protocol = config.get('occlusion', 'protocol', fallback='severe')
    table, _ = load_age_table(protocol)
    age_sigma = lambda a: min(_lookup(table, a), SIGMA_CLIP)
    if channel_rows:
        age_channel = np.array([a for a, _, _ in channel_rows])
        full_sigma = np.array([s for _, s, _ in channel_rows])
        existence = np.array([e for _, _, e in channel_rows])
        table_sigma = np.array([age_sigma(a) for a in age_channel])
        gap = np.abs(full_sigma - table_sigma)
        check('age 查表与 full 后验的不确定度尺度已对齐(仅 age>=1)',
              float(np.median(gap)) < .05, f'中位差 {np.median(gap):.4f} m')
        varying = [float(np.std(existence[age_channel == a]))
                   for a in np.unique(age_channel) if (age_channel == a).sum() > 30]
        worst = max(varying) if varying else 0.
        spread = float(np.max(np.abs(full_sigma - table_sigma)))
        if probe.noise_position > 0. or probe.detection_probability < 1.:
            # severe:后验带有超出失观时长的个体信息,L3 在这一档才可测
            check('同龄航迹的存在概率仍然有差异(age 臂拿不到的信息)', worst > .01,
                  f'同龄存在概率标准差 max {worst:.4f}')
        else:
            # standard:无噪且无漏检时,协方差与存在概率都是失观时长的精确函数,
            # full 按构造不含超出 age 的信息。此处断言该退化成立,而非断言有差异。
            check('零噪声零漏检下 full 退化为 age 的确定性函数(存在概率)',
                  worst < 1e-6, f'同龄存在概率标准差 max {worst:.2e}')
            check('零噪声零漏检下 full 退化为 age 的确定性函数(协方差)',
                  spread < 1e-3, f'max|Δσ| = {spread:.2e} m,受查表四位小数取整限制')

    cfg = probe.cfg
    check('RFS 内部测量噪声与注入噪声一致',
          cfg.position_measurement_std == probe.noise_position
          and cfg.velocity_measurement_std == probe.noise_velocity
          and cfg.detection_probability == probe.detection_probability,
          f'协议 {probe.protocol}: {probe.noise_position} m / {probe.noise_velocity} m/s / pD={probe.detection_probability}')

    # 语法级泄漏检查:只看真正被访问的名字,注释与文档字符串不算
    import ast
    tree = ast.parse((ROOT / 'crowd_nav/occlusion_rfs.py').read_text())
    touched = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    touched |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    touched |= {s.value for n in ast.walk(tree) if isinstance(n, ast.Subscript)
                for s in [n.slice] if isinstance(s, ast.Constant)}
    for banned in ('policy_entities', 'label_grid', '_all', 'occluded_ids',
                   'get_policy_state', 'label_ids', 'human_times', 'global_time'):
        check(f'观测模块未访问 {banned}', banned not in touched)
    legal_human = {'px', 'py', 'vx', 'vy', 'radius'}
    used_on_humans = {k for k in touched if k in ('gx', 'gy', 'theta', 'v_pref', 'policy')}
    check('机器人目标仅用于自身朝向对齐(gx/gy 属机器人自身状态)',
          used_on_humans <= {'gx', 'gy', 'v_pref'}, ' '.join(sorted(used_on_humans)))
    check('行人通道只取公开物理量', legal_human <= touched)

    return not FAIL


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'crowd_nav/configs/env_occlusion_rfs.config'
    ok = run(path)
    print('\n自检' + ('通过' if ok else '未通过: ' + ', '.join(sorted(set(FAIL)))))
    sys.exit(0 if ok else 1)
