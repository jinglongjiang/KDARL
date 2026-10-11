"""冻结评测:逐回合落盘,供配对统计使用。

同一 (人数,场景,case) 在所有臂与种子上完全一致,因此可以做配对检验。
测试 case 从 9400000 起,与教师轨迹(10000+)和开发集(20000+)不重叠。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import configparser
import numpy as np
import torch
from crowd_nav.bayesian import SCHEMA, LocalValueNetwork, action_table
from crowd_nav.train import configuration, environment, episode

CASE_START = 9400000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--humans', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--scenes', nargs='+', default=['circle_crossing', 'square_crossing'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.04)
    parser.add_argument('--oracle', choices=('none', 'hidden', 'range', 'vis_pos', 'vis_vel', 'vis_state'), default='none',
                        help='诊断用:测试时把隐藏行人换成真值,测信息上限')
    parser.add_argument('--protocol-config', type=Path,
                        help='用另一档遮挡协议测试同一个检查点(跨协议迁移测试)。'
                             '只允许 [occlusion] 的协议四项不同,其余必须逐项一致。')
    args = parser.parse_args()

    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if checkpoint.get('schema') != SCHEMA:
        raise ValueError('checkpoint schema 不匹配')
    arm, seed = checkpoint['arm'], checkpoint['seed']
    config = configparser.ConfigParser()
    config.read_dict(checkpoint['config'])
    if not config.getboolean('occlusion', 'enabled', fallback=False):
        raise ValueError('该检查点不是遮挡协议下训练的')
    if args.protocol_config:
        override = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        if not override.read(str(args.protocol_config)):
            raise FileNotFoundError(args.protocol_config)
        allowed = {'protocol', 'noise_position', 'detection_probability', 'noise_velocity'}
        for section in set(config.sections()) | set(override.sections()):
            a, b = dict(config[section]), dict(override[section])
            differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
            if differing - (allowed if section == 'occlusion' else set()):
                raise ValueError(f'[{section}] 除协议四项外存在差异: {sorted(differing)}')
        config = override
        print('PROTOCOL_OVERRIDE', override.get('occlusion', 'protocol'), flush=True)
    config.set('occlusion', 'oracle', args.oracle)
    model = LocalValueNetwork(action_table(config),
                             composition=checkpoint.get('composition', 'mixture')).to(args.device)
    model.load_state_dict(checkpoint['state'], strict=True)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as handle:
        for humans in args.humans:
            for scene in args.scenes:
                env = environment(config, humans, 'test', scene)
                rng = np.random.default_rng(0)
                state = np.random.get_state()
                try:
                    for index in range(args.episodes):
                        case = CASE_START + index
                        _, record = episode(env, model, config, case, arm, rng)
                        handle.write(json.dumps(dict(
                            arm=arm, seed=seed, humans=humans, scene=scene, case=case, oracle=args.oracle,
                            outcome=record['outcome'], steps=record['steps'],
                            reward=record['reward'],
                            min_clearance=record['min_clearance'])) + '\n')
                finally:
                    np.random.set_state(state)
                handle.flush()
                print('DONE', arm, seed, humans, scene, flush=True)


if __name__ == '__main__':
    main()
