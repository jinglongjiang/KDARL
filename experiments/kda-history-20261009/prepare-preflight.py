"""Freeze four fresh-IL arms and validate the unchanged parent pipeline."""
import configparser
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
PROJECT = Path('/root/cc-diag-20261008/code')
ASSET = Path('/root/cc-diag-20261008/results')
DATA = Path('/root/workspace/nav_data/mamba/camrl/CrowdNav/data/il_dataset_diverse_v5.0_merged_15000_with_idx.pth')
ARMS = [('kda', 12), ('kda', 24), ('gru', 12), ('gru', 24)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def configure(arm, length, destination, epochs=50, episodes=10000, capacity=200000):
    cfg = configparser.RawConfigParser()
    cfg.read(ASSET / (arm + '.ini'))
    cfg.set('temporal', 't', str(length))
    cfg.set('buffer', 'seq_len', str(length))
    cfg.set('buffer', 'capacity', str(capacity))
    cfg.set('train', 'offline_il_dataset', str(DATA))
    cfg.set('train', 'il_ckpt', str(destination / 'il_policy.pth'))
    cfg.set('train', 'il_force_retrain', 'true')
    cfg.set('train', 'il_fixed_budget', 'true')
    cfg.set('train', 'il_epochs', str(epochs))
    cfg.set('train', 'train_episodes', str(episodes))
    assert cfg.getint('sim', 'human_num') == 5
    assert cfg.get('sim', 'train_val_sim') == 'circle_crossing'
    return cfg


def write_cfg(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        cfg.write(stream)


def main():
    BASE.mkdir(exist_ok=True)
    started = time.time()
    save('preflight-status.json', dict(stage='PREPARING', started=started))
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    import crowd_nav, crowd_sim, crowd_nav.policy.mamba_rl as policy_module
    for module in (crowd_nav, crowd_sim, policy_module):
        Path(module.__file__).resolve().relative_to(PROJECT)
    from crowd_nav import contracts, train
    from crowd_nav.policy.shared_initialization import initialize_shared
    from crowd_nav.utils.ppo_buffer import ReplayBufferMC
    dataset = torch.load(DATA, map_location='cpu', weights_only=False)
    assert dataset['config']['human_num'] == 5
    trajectories = dataset['trajectories']
    success = [t for t in trajectories if t.get('meta', {}).get('flag_success')
               or t.get('meta', {}).get('outcome') in ('success', 'ReachGoal')
               or 'success' in str(t.get('meta', {}).get('event', '')).lower()
               or 'reachgoal' in str(t.get('meta', {}).get('event', '')).lower()]
    assert success and all(np.asarray(t['states']).shape[1] == 34 for t in trajectories)
    frames = sum(len(t['states']) for t in success)
    shared, temporal, memory = {}, {}, {}
    baseline_shared = None
    for arm, length in ARMS:
        name = '%s-t%d' % (arm, length)
        cfg = configure(arm, length, BASE / 'runs' / name)
        write_cfg(BASE / (name + '.ini'), cfg)
        contracts.init_grid_from_cfg(cfg)
        assert contracts.grid_action_dim(contracts.GRID) == 80
        torch.manual_seed(419)
        net = policy_module.MambaRLPolicy(cfg, device='cuda')
        initialize_shared(net, 419)
        parts = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        h = hashlib.sha256()
        for key, value in sorted(parts.items()):
            if not key.startswith('temporal_encoder.'):
                h.update(key.encode()); h.update(value.numpy().tobytes())
        shared[name] = h.hexdigest()
        h = hashlib.sha256()
        for key, value in sorted(parts.items()):
            h.update(key.encode()); h.update(value.numpy().tobytes())
        temporal[name] = h.hexdigest()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(256, length, 8, 13, device='cuda')
        output = net.forward_value(x)
        output.square().mean().backward()
        assert torch.isfinite(output).all()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        memory[name] = dict(params=sum(p.numel() for p in net.parameters()),
            minibatch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            theoretical_il_tensor_bytes=frames * length * 8 * 13 * 4)
        net.eval()
        with torch.no_grad():
            sample = x[:2]
            expected = net.forward_value(sample)
            net.load_state_dict(parts, strict=True)
            actual = net.forward_value(sample)
            assert torch.equal(expected, actual)
        del net, x, sample, output, expected, actual, parts
        gc.collect(); torch.cuda.empty_cache()
    assert len(set(shared.values())) == 1
    assert temporal['kda-t12'] == temporal['kda-t24']
    assert temporal['gru-t12'] == temporal['gru-t24']
    import importlib.util
    spec = importlib.util.spec_from_file_location('worker_helpers', BASE / 'train-worker.py')
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    replay = ReplayBufferMC(mc_only=True, capacity=8, seq_len=12)
    for i in range(11):
        replay.store(np.full((12, 8, 13), i, dtype=np.float32), [0., 1.], .1, None, False, mc_return=i / 10.)
    env = type('Counters', (), {})(); env.case_counter = dict(train=11, val=2, test=3)
    rng = worker.rng_state(env)
    packed = worker.pack_buffer(replay)
    tmp = BASE / 'preflight/synthetic-resume.pth.gz'; tmp.parent.mkdir(exist_ok=True)
    import gzip
    with gzip.open(tmp, 'wb', compresslevel=1) as stream:
        torch.save(dict(buffer=packed, rng=rng), stream)
    restored = worker.archive_load(tmp)
    clone = ReplayBufferMC(mc_only=True, capacity=8, seq_len=12)
    worker.restore_buffer(clone, restored['buffer'])
    assert replay.ptr == clone.ptr and replay.size == clone.size
    assert np.array_equal(replay.states, clone.states) and np.array_equal(replay.returns, clone.returns)
    draws = (np.random.rand(8), torch.rand(8), torch.rand(8, device='cuda'))
    worker.restore_rng(restored['rng'], env)
    assert np.array_equal(draws[0], np.random.rand(8))
    assert torch.equal(draws[1], torch.rand(8)) and torch.equal(draws[2], torch.rand(8, device='cuda'))
    token_sample = train.make_il_token_windows(success[0]['states'], 12)
    assert token_sample.shape == (len(success[0]['states']), 12, 8, 13)
    del dataset, success, trajectories, replay, clone, packed, restored, token_sample
    gc.collect(); torch.cuda.empty_cache()
    original_env = configparser.RawConfigParser(); original_env.read(ASSET / 'kda.ini')
    write_cfg(BASE / 'env.config', original_env)
    protocol = dict(task='KDA_GRU_HISTORY_LENGTH_FRESH_IL', seed=419, arms=ARMS,
        il_epochs=50, online_episodes=10000, dataset=str(DATA), dataset_sha256=sha(DATA),
        successful_il_frames=frames, training='five-person circle only',
        shared_initialization_hashes=shared, same_backbone_initialization_hashes=temporal,
        memory_estimates=memory, source_sha256={str(p.relative_to(PROJECT)): sha(p)
            for p in sorted(PROJECT.rglob('*.py'))},
        frozen=['reward', 'grid80', 'filter', 'smoothing', 'KDA/GRU dimensions and LR'],
        eval=dict(rule='ttc5', case_base=91000, n_cases_per_cell=500,
                  scope='single-seed development comparison, not a fresh final confirmation'),
        safety_auxiliary=dict(arms=['kda', 'gru', 'mamba'], rules=['endpoint', 'swept'],
            case_base=91000, n_cases_per_cell=32, scope='bounded development pilot, no threshold tuning'),
        wall_limit_hours=8, reserve_report_minutes=30,
        continuation='epoch/episode-boundary checkpoint includes optimizer, RNG, replay and case counters')
    save('protocol.json', protocol)
    # Full raw demonstration pool, one epoch only; these weights never enter formal runs.
    measurements = {}
    for arm, length in [('kda', 12), ('gru', 24)]:
        name = '%s-t%d' % (arm, length)
        target = BASE / 'preflight' / name
        cfg = configure(arm, length, target, epochs=1, episodes=0)
        config = BASE / ('preflight-' + name + '.ini'); write_cfg(config, cfg)
        save('preflight-status.json', dict(stage='FULL_IL_EPOCH', arm=name))
        command = [sys.executable, str(BASE / 'train-worker.py'), '--project', str(PROJECT),
            '--deadline', str(time.time() + 1800), '--config', str(config), '--outdir', str(target),
            '--device', 'cuda', '--seed', '419', '--temporal-backbone', arm, '--pretrain-only']
        tick = time.time()
        with (BASE / ('preflight-' + name + '.log')).open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
        checkpoint = torch.load(target / 'il_policy.pth', map_location='cpu', weights_only=False)
        assert checkpoint['meta']['seq_len'] == length and checkpoint['meta']['epochs'] == 1
        assert checkpoint['meta']['dataset_size'] == frames
        measurements[name] = json.loads((target / 'phase.json').read_text())
        measurements[name]['complete_probe_wall_seconds'] = time.time() - tick
        measurements[name]['dataset_size'] = frames
        assert (target / 'worker-completed.json').exists()
    save('preflight.json', dict(status='PASS', forward_backward_save_load='PASS',
        replay_ring_and_rng_roundtrip='PASS', shared_initialization='PASS', grid=80,
        full_il_epoch_measurements=measurements, wall_seconds=time.time() - started,
        weights_are_isolated_and_never_reused=True))
    save('preflight-status.json', dict(stage='COMPLETED', status='PASS'))


if __name__ == '__main__':
    main()
