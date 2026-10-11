"""Isolated endpoint versus simultaneous swept-clearance pilot; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import importlib.util
BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('history_evaluator', BASE / 'evaluate-history.py')
fixed = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixed)


def swept_clearance(robot_position, human_positions, robot_velocities, human_velocities,
                    robot_radius, human_radii, dt):
    d0 = np.asarray(robot_position)[None, None, :] - np.asarray(human_positions)[None, :, :]
    dv = np.asarray(robot_velocities)[:, None, :] - np.asarray(human_velocities)[None, :, :]
    speed2 = np.sum(dv * dv, axis=-1)
    t = np.zeros_like(speed2)
    np.divide(-np.sum(d0 * dv, axis=-1), speed2, out=t, where=speed2 > 1e-12)
    t = np.clip(t, 0., dt)
    distance = np.linalg.norm(d0 + t[..., None] * dv, axis=-1)
    return (distance - robot_radius - np.asarray(human_radii)[None, :]).min(axis=1)


def geometry_tests():
    # Same-time crossing, constant separation, and spatial crossing at different times.
    assert abs(swept_clearance([0., 0.], [[1., -1.]], [[1., 0.]], [[0., 1.]], 0., [0.], 2.)[0]) < 1e-12
    assert abs(swept_clearance([0., 0.], [[1., 0.]], [[1., 0.]], [[1., 0.]], .1, [.1], 2.)[0] - .8) < 1e-12
    expected = np.sqrt(.5)
    assert abs(swept_clearance([0., 0.], [[1., -2.]], [[1., 0.]], [[0., 1.]], 0., [0.], 2.)[0] - expected) < 1e-12
    assert abs(swept_clearance([0., 0.], [[2., 0.]], [[-1., 0.]], [[0., 0.]], .1, [.1], .25)[0] - 1.8) < 1e-12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--arm-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--backbone', required=True, choices=('kda', 'gru', 'mamba'))
    parser.add_argument('--safety', required=True, choices=('endpoint', 'swept'))
    parser.add_argument('--cell', type=int, required=True)
    parser.add_argument('--n-cases', type=int, default=32)
    parser.add_argument('--case-base', type=int, default=91000)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    args.human_select = 'ttc5'; args.expected_checkpoint_sha = None
    geometry_tests()
    torch.set_num_threads(1)
    cfg, net, digest = fixed.load(args)
    original = net.score_sarl_candidates
    measurements = []

    def score(state):
        result = original(state)
        robot = state.self_state; humans = state.human_states
        commands = np.asarray([[a.vx, a.vy] for a in result['commands']], dtype=np.float64)
        swept = swept_clearance([robot.px, robot.py], [[h.px, h.py] for h in humans], commands,
            [[h.vx, h.vy] for h in humans], robot.radius, [h.radius for h in humans], net.time_step)
        endpoints = result['clearance'].detach().cpu().numpy()
        assert np.all(swept <= endpoints + 2e-12)
        clearance = endpoints if args.safety == 'endpoint' else swept
        n_safe = int(np.count_nonzero(clearance >= net.test_min_clearance))
        if args.safety == 'swept':
            d = torch.as_tensor(swept, device=net.device, dtype=result['clearance'].dtype)
            values = result['raw_scores'].clone()
            if net.test_min_clearance > 0.:
                mask = d >= net.test_min_clearance
                if mask.any():
                    values = values.masked_fill(~mask, -1e9)
            if net.test_risk_lambda > 0.:
                margin = net.test_min_clearance if net.test_min_clearance > 0. else net.discomfort_dist
                values -= net.test_risk_lambda * torch.clamp(margin - d, min=0.)
            values[0] -= 1e-3
            result['scores'] = values
            result['clearance'] = d
        measurements.append(dict(n_safe=n_safe, all_unsafe=n_safe == 0,
            endpoint_n_safe=int(np.count_nonzero(endpoints >= .2)),
            swept_n_safe=int(np.count_nonzero(swept >= .2)),
            endpoint_safe_swept_collision_candidates=int(np.count_nonzero((endpoints >= .2) & (swept < 0.)))))
        return result

    net.score_sarl_candidates = score
    started = time.time(); rows = []
    with torch.no_grad():
        env, robot = fixed.environment(cfg, net, args.cell, args.case_base)
        for case in range(args.case_base, args.case_base + args.n_cases):
            net.reset_episode_stats(); env.reset(seed=case, options={'test_case': case})
            measurements.clear()
            total, minimum, outcome = 0., float('inf'), 'timeout'
            for k in range(201):
                action = net.predict(fixed.state_of(env, robot))
                reward, done, info = fixed.step(env, action)
                total += .99 ** k * reward
                minimum = min(minimum, min(float(np.hypot(h.px - robot.px, h.py - robot.py)
                    - h.radius - robot.radius) for h in env.humans))
                if done:
                    outcome = fixed.event(info); break
            rows.append(dict(case=case, cell=args.cell, human_num=fixed.CELLS[args.cell][0],
                geometry=fixed.CELLS[args.cell][1], outcome=outcome, return_value=total,
                time=(k + 1) * .25, minimum_clearance=minimum, scored_steps=len(measurements),
                all_unsafe_steps=sum(x['all_unsafe'] for x in measurements),
                n_safe_histogram=np.bincount([x['n_safe'] for x in measurements], minlength=81).tolist(),
                endpoint_safe_swept_collision_candidates=sum(x['endpoint_safe_swept_collision_candidates'] for x in measurements)))
            fixed.save(str(args.out) + '.progress', dict(episodes=rows, wall_seconds=time.time() - started))
    fixed.save(args.out, dict(episodes=rows, backbone=args.backbone, safety=args.safety,
        checkpoint_sha256=digest, config_sha256=fixed.sha(args.config), grid_actions=80,
        script_sha256=fixed.sha(__file__), wall_seconds=time.time() - started,
        rule='ttc5', geometry_tests='PASS', reward_and_value_unchanged=True,
        protocol=dict(cell=args.cell, case_base=args.case_base, n_cases=args.n_cases),
        limitation='Swept clearance assumes constant observed human velocities, not true future ORCA trajectories'))


if __name__ == '__main__':
    main()
