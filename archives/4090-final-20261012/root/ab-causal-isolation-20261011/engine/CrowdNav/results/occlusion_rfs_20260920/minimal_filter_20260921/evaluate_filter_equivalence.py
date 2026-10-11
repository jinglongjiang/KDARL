#!/usr/bin/env python3
"""Paired navigation and shadow-action audit for the minimal tracker."""
from __future__ import annotations

import argparse
import configparser
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def load_runtime(root: Path):
    sys.path.insert(0, str(root.resolve()))
    from crowd_nav.bayesian import SCHEMA, LocalValueNetwork, action_table, batch_observations
    from crowd_nav.occlusion_rfs import OcclusionBelief
    from crowd_nav.train import environment, episode
    from crowd_sim.envs.utils.action import ActionXY
    return SCHEMA, LocalValueNetwork, action_table, batch_observations, OcclusionBelief, environment, episode, ActionXY


def choose(model, observation, batch_observations):
    device = next(model.parameters()).device
    with torch.no_grad():
        scores = model(batch_observations([observation], device))
    return int(scores.argmax(1).item())


def normalized_action_deviation(first, second, maximum_speed):
    speed_first = float(np.linalg.norm(first))
    speed_second = float(np.linalg.norm(second))
    delta_speed = abs(speed_first - speed_second) / max(maximum_speed, 1e-12)
    angle_first = math.atan2(float(first[1]), float(first[0])) if speed_first else 0.0
    angle_second = math.atan2(float(second[1]), float(second[0])) if speed_second else 0.0
    delta_angle = abs((angle_first - angle_second + math.pi) % (2 * math.pi) - math.pi)
    return float(math.hypot(delta_speed, delta_angle / math.pi))


def shadow_episode(
    env,
    model,
    config,
    case,
    arm,
    table,
    batch_observations,
    belief_type,
    action_type,
):
    raw, _ = env.reset(options={'test_case': int(case)})
    reference = belief_type(config, arm, int(case), tracker='rfs')
    minimal = belief_type(config, arm, int(case), tracker='constant')
    maximum_speed = float(np.linalg.norm(table, axis=1).max())
    actions = []
    reward_sum = 0.0
    clearance = float('inf')
    outcome = 'incomplete'
    limit = int(np.ceil(env.time_limit / env.time_step))
    for step in range(limit):
        remaining = max(0.0, 1.0 - env.global_time / env.time_limit)
        reference_obs, rotation = reference.observe(env, raw, step, remaining)
        minimal_obs, minimal_rotation = minimal.observe(env, raw, step, remaining)
        np.testing.assert_allclose(rotation, minimal_rotation, atol=0.0, rtol=0.0)
        reference_action = choose(model, reference_obs, batch_observations)
        minimal_action = choose(model, minimal_obs, batch_observations)
        actions.append({
            'reference': reference_action,
            'minimal': minimal_action,
            'agreement': int(reference_action == minimal_action),
            'deviation': normalized_action_deviation(
                table[reference_action], table[minimal_action], maximum_speed
            ),
        })
        velocity = rotation.T @ table[reference_action]
        raw, reward, terminated, truncated, info = env.step(action_type(*velocity))
        reward_sum += float(reward)
        clearance = min(clearance, float(info.get('dmin', float('inf'))))
        if terminated or truncated:
            outcome = info['event']
            break
    return {
        'outcome': outcome,
        'steps': len(actions),
        'reward': reward_sum,
        'min_clearance': clearance if np.isfinite(clearance) else None,
        'actions': actions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--case-start', type=int, default=9400000)
    parser.add_argument('--humans', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--scenes', nargs='+', default=['circle_crossing', 'square_crossing'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.04)
    args = parser.parse_args()

    (schema, network_type, action_table, batch_observations, belief_type,
     environment, episode, action_type) = load_runtime(args.code_root)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if checkpoint.get('schema') != schema:
        raise ValueError('checkpoint schema mismatch')
    arm = checkpoint['arm']
    if arm not in ('mean', 'age'):
        raise ValueError('minimal-filter audit accepts only mean/age checkpoints')
    config = configparser.ConfigParser()
    config.read_dict(checkpoint['config'])
    model = network_type(
        action_table(config), composition=checkpoint.get('composition', 'mixture')
    ).to(args.device)
    model.load_state_dict(checkpoint['state'], strict=True)
    model.eval()
    table = model.actions.detach().cpu().numpy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as handle:
        for humans in args.humans:
            for scene in args.scenes:
                for index in range(args.episodes):
                    case = args.case_start + index
                    reference_env = environment(config, humans, 'test', scene)
                    shadow = shadow_episode(
                        reference_env, model, config, case, arm, table,
                        batch_observations, belief_type, action_type,
                    )
                    minimal_config = copy.deepcopy(config)
                    minimal_config.set('occlusion', 'tracker', 'constant')
                    minimal_env = environment(minimal_config, humans, 'test', scene)
                    _, minimal = episode(
                        minimal_env, model, minimal_config, case, arm,
                        np.random.default_rng(0),
                    )
                    handle.write(json.dumps({
                        'arm': arm,
                        'seed': int(checkpoint['seed']),
                        'humans': humans,
                        'scene': scene,
                        'case': case,
                        'reference': {k: v for k, v in shadow.items() if k != 'actions'},
                        'minimal': minimal,
                        'outcome_agreement': int(shadow['outcome'] == minimal['outcome']),
                        'actions': shadow['actions'],
                    }) + '\n')
                    handle.flush()
    manifest = {
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'runtime_files': {
            name: hashlib.sha256((args.code_root / 'crowd_nav' / name).read_bytes()).hexdigest()
            for name in ('bayesian_rfs.py', 'occlusion_rfs.py', 'train.py')
        },
        'case_start': args.case_start,
        'episodes_per_cell': args.episodes,
    }
    args.out.with_suffix('.manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8'
    )


if __name__ == '__main__':
    main()
