"""v2 协议一致性与泄漏核验。不看成功率,只看协议是否被原样执行。

对每个已完成的 run 检查参数、开发集规模、case 分区、源码指纹;
再抽一个检查点做 CUDA/CPU 数值一致性对照。
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import configparser
import numpy as np
import torch

EXPECTED = dict(arm=None, composition='mixture', device='cuda', demo_episodes=500,
                il_updates=12000, il_eval_every=2000, eval_episodes=400,
                validation_start=21000, case_start=9400000, batch_size=64,
                rl_episodes=0, rl_steps=0, il_lr=.0003)
CALIBRATION = range(20000, 20200)
DEMOS = range(10000, 10500)
FAIL = []


def check(name, condition, detail=''):
    print(('PASS  ' if condition else 'FAIL  ') + name + (('  ' + detail) if detail else ''))
    if not condition:
        FAIL.append(name)


def verify_runs(directory):
    runs = sorted(Path(directory).glob('*/results.json'))
    print(f'已完成 run: {len(runs)}')
    bad_arguments, bad_curve, sizes, fingerprints = [], [], set(), {}
    for path in runs:
        result = json.loads(path.read_text())
        arguments = result['arguments']
        for key, value in EXPECTED.items():
            if value is None:
                continue
            if arguments.get(key) != value:
                bad_arguments.append(f'{path.parent.name}.{key}={arguments.get(key)}')
        if 'env_occlusion_rfs.config' not in str(arguments.get('config', '')):
            bad_arguments.append(f'{path.parent.name}.config')
        for row in result['il_curve']:
            if row['episodes'] != 400:
                bad_curve.append(f"{path.parent.name}@{row['update']}={row['episodes']}")
            sizes.add(row['humans'])
        updates = [row['update'] for row in result['il_curve']]
        if updates != [2000, 4000, 6000, 8000, 10000, 12000]:
            bad_curve.append(f'{path.parent.name} 评测点 {updates}')
        cases = [record['case'] for row in result['il_curve'] for record in row['records']]
        if cases and (min(cases) != 21000 or max(cases) != 21399):
            bad_curve.append(f'{path.parent.name} 开发 case {min(cases)}-{max(cases)}')
    check('全部 run 的训练参数与预注册一致', not bad_arguments, '; '.join(bad_arguments[:4]))
    check('开发集每次 400 回合,六个评测点', not bad_curve, '; '.join(bad_curve[:4]))
    check('开发集只用五人布局', sizes == {5}, str(sizes))

    for path in runs:
        payload = torch.load(path.parent / 'fixed.pt', map_location='cpu')
        fingerprints.setdefault(json.dumps(payload['source_sha256'], sort_keys=True), []).append(
            path.parent.name)
        if payload['stage'] not in ('il','il_fixed') or payload['train_num_humans'] != 5:
            FAIL.append('检查点阶段或训练人数异常')
    check('全部 run 共用同一个源码 sha256(指纹集合大小应为 1)', len(fingerprints) == 1,
          f'指纹集合大小 {len(fingerprints)},应为 1')
    check('检查点阶段合法且训练人数为 5', '检查点阶段或训练人数异常' not in FAIL)

    # case 分区互不重叠
    development = range(21000, 21400)
    test = range(9400000, 9400200)
    spans = dict(演示=DEMOS, 标定=CALIBRATION, 选模=development, 测试=test)
    overlap = [f'{a}×{b}' for a in spans for b in spans
               if a < b and set(spans[a]) & set(spans[b])]
    check('四段 case 互不重叠', not overlap, '; '.join(overlap))

    source = (ROOT / 'crowd_nav/calibrate_age_sigma.py').read_text()
    check('查表标定用的是 20000-20199', 'range(20000, 20200)' in source)
    return runs


def verify_device(run):
    """同一检查点在 CUDA 与 CPU 上跑同一批 case,比较动作序列与结局。"""
    from crowd_nav.bayesian import LocalValueNetwork, action_table
    from crowd_nav.train import configuration, environment, episode
    payload = torch.load(run / 'final.pt', map_location='cpu')
    config = configparser.ConfigParser()
    config.read_dict(payload['config'])
    outcomes, actions = {}, {}
    for device in ('cpu', 'cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        model = LocalValueNetwork(action_table(config), composition=payload['composition']).to(device)
        model.load_state_dict(payload['state'], strict=True)
        model.eval()
        env = environment(config, 5, 'test', 'circle_crossing')
        rng = np.random.default_rng(0)
        records, sequence = [], []
        state = np.random.get_state()
        try:
            for index in range(20):
                rows, record = episode(env, model, config, 9400000 + index, payload['arm'], rng)
                records.append(record['outcome'])
                sequence.extend(row['action'] for row in rows)
        finally:
            np.random.set_state(state)
        outcomes[device], actions[device] = records, sequence
    if len(outcomes) < 2:
        check('CUDA/CPU 数值一致性', False, '本机无 CUDA,需在 4090 上跑')
        return
    same_outcome = outcomes['cpu'] == outcomes['cuda']
    length = min(len(actions['cpu']), len(actions['cuda']))
    agree = sum(a == b for a, b in zip(actions['cpu'][:length], actions['cuda'][:length]))
    check('CUDA 与 CPU 的 20 回合结局完全一致', same_outcome,
          f"cpu={outcomes['cpu'].count('reach_goal')} cuda={outcomes['cuda'].count('reach_goal')}")
    check('CUDA 与 CPU 的动作选择一致率 > 99%', agree / max(length, 1) > .99,
          f'{agree}/{length} = {100*agree/max(length,1):.2f}%')


if __name__ == '__main__':
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'runs'
    runs = verify_runs(directory)
    if runs:
        verify_device(runs[0].parent)
    print('\n核验' + ('通过' if not FAIL else '未通过: ' + ', '.join(sorted(set(FAIL)))))
    sys.exit(0 if not FAIL else 1)
