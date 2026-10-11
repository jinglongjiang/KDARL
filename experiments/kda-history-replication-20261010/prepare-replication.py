"""Prepare a second-seed replication without starting IL, RL, or evaluation."""
import sys
sys.dont_write_bytecode = True
import argparse
import configparser
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
PROJECT = Path('/root/cc-diag-20261008/code')
PREVIOUS = Path('/root/kda-history-20261009')
DATA = Path('/root/workspace/nav_data/mamba/camrl/CrowdNav/data/il_dataset_diverse_v5.0_merged_15000_with_idx.pth')
ARMS = (('kda', 12), ('kda', 24), ('gru', 12), ('gru', 24))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def save(name, payload):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposed-seed', type=int, default=443)
    args = parser.parse_args()
    assert shutil.disk_usage(BASE).free >= 4 * 1024 ** 3, 'Need at least 4 GiB before launch'
    prior = json.loads((PREVIOUS / 'protocol.json').read_text())
    assert prior['seed'] == 419 and prior['il_epochs'] == 50 and prior['online_episodes'] == 10000
    digest = sha(DATA)
    assert digest and digest == prior['dataset_sha256'], 'Raw IL pool changed'
    source = {str(p.relative_to(PROJECT)): sha(p) for p in sorted(PROJECT.rglob('*.py'))}
    assert source and source == prior['source_sha256'], 'Parent Python source differs from seed419'
    sys.path.insert(0, str(PROJECT))
    import crowd_nav, crowd_sim, crowd_nav.policy.mamba_rl as policy_module
    for module in (crowd_nav, crowd_sim, policy_module):
        Path(module.__file__).resolve().relative_to(PROJECT)
    import torch
    from crowd_nav import contracts
    from crowd_nav.policy.shared_initialization import initialize_shared
    torch.set_num_threads(1)
    entries = {}
    for arm, length in ARMS:
        name = '%s-t%d' % (arm, length)
        original = PREVIOUS / (name + '.ini')
        assert original.exists(), str(original)
        cfg = configparser.RawConfigParser()
        cfg.read(original)
        assert cfg.getint('temporal', 't') == cfg.getint('buffer', 'seq_len') == length
        assert cfg.getint('sim', 'human_num') == 5
        assert cfg.get('sim', 'train_val_sim') == 'circle_crossing'
        run = BASE / 'runs' / name
        replacements = {'seed': str(args.proposed_seed), 'save_every': '1000',
                        'offline_il_dataset': str(DATA), 'il_ckpt': str(run / 'il_policy.pth')}
        for key, value in replacements.items():
            cfg.set('train', key, value)
        assert cfg.getboolean('train', 'il_force_retrain')
        assert cfg.getboolean('train', 'il_fixed_budget')
        assert cfg.getint('train', 'il_epochs') == 50
        assert cfg.getint('train', 'train_episodes') == 10000
        dest = BASE / (name + '.ini')
        assert not dest.exists(), 'Do not overwrite prepared configurations'
        with dest.open('w') as stream:
            cfg.write(stream)
        grid = contracts.init_grid_from_cfg(cfg)
        assert contracts.grid_action_dim(grid) == 80 and not grid['include_stop']
        torch.manual_seed(args.proposed_seed)
        torch.cuda.manual_seed_all(args.proposed_seed)
        net = policy_module.MambaRLPolicy(cfg, device='cuda')
        initialize_shared(net, args.proposed_seed)
        shared, full = hashlib.sha256(), hashlib.sha256()
        for key, value in sorted(net.state_dict().items()):
            data = key.encode() + value.detach().cpu().contiguous().numpy().tobytes()
            full.update(data)
            if not key.startswith('temporal_encoder.'):
                shared.update(data)
        net.eval()
        with torch.no_grad():
            output = net.forward_value(torch.zeros(2, length, 8, 13, device='cuda'))
        assert torch.isfinite(output).all()
        entries[name] = dict(config_sha256=sha(dest), original_config_sha256=sha(original),
                             allowed_configuration_changes=replacements,
                             shared_init_sha256=shared.hexdigest(), full_init_sha256=full.hexdigest(),
                             forward_shape=list(output.shape), grid_actions=80)
        del net, output
        torch.cuda.empty_cache()
    assert len({e['shared_init_sha256'] for e in entries.values()}) == 1
    for arm in ('kda', 'gru'):
        assert entries[arm + '-t12']['full_init_sha256'] == entries[arm + '-t24']['full_init_sha256']
    helpers = {name: sha(PREVIOUS / name) for name in ('train-worker.py', 'evaluate-history.py')}
    save('preparation.json', dict(status='已验证通过', scope='Preparation only; no IL/RL/evaluation launched',
         proposed_seed=args.proposed_seed, prior_seed=419, free_disk_bytes=shutil.disk_usage(BASE).free,
         dataset_sha256=digest, parent_source_unchanged=True, source_sha256=source,
         helper_sha256=helpers, arms=entries, fresh_case_block_92000_used=False,
         pending=['Resolve main task: forecast probe or history replication',
                  'Confirm second training seed and complete stage1/stage2 protocol',
                  'Freeze statistical family and pass criteria before experimental results']))
    print(json.dumps(dict(status='已验证通过', proposed_seed=args.proposed_seed, arms=list(entries),
                         dataset_sha256=digest, shared_initialization_equal=True,
                         training_started=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
