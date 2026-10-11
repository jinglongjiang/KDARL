"""Tiny pipeline continuity test; its random-init weights are never reused."""
import configparser
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prepare_helpers', BASE / 'prepare-preflight.py')
prepare = importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)


def main():
    started = time.time(); root = BASE / 'preflight' / 'rl-continuity'
    root.mkdir(parents=True, exist_ok=True)

    def config(name):
        out = root / name
        cfg = prepare.configure('gru', 12, out, epochs=50, episodes=3, capacity=256)
        cfg.set('train', 'save_every', '1')
        cfg.set('imitation_learning', 'max_il_prefill', '2')
        path = BASE / ('probe-' + name + '.ini'); prepare.write_cfg(path, cfg)
        return out, path

    def run(out, path, extra):
        command = [sys.executable, str(BASE / 'train-worker.py'), '--project', str(prepare.PROJECT),
            '--deadline', str(time.time() + 900), '--config', str(path), '--outdir', str(out),
            '--device', 'cuda', '--seed', '419', '--temporal-backbone', 'gru'] + extra
        with (root / (out.name + '-' + str(len(extra)) + '.log')).open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        return result.returncode

    continuous, continuous_cfg = config('continuous')
    split, split_cfg = config('split')
    assert run(continuous, continuous_cfg, ['--skip-il-pretrain']) == 0
    assert run(split, split_cfg, ['--skip-il-pretrain', '--pause-rl-episode', '2']) == 75
    assert (split / 'resume-latest.pth.gz').exists()
    assert run(split, split_cfg, ['--continue-archive', str(split / 'resume-latest.pth.gz')]) == 0
    a = torch.load(continuous / 'rl_model_ep3.pth', map_location='cpu', weights_only=False)
    b = torch.load(split / 'rl_model_ep3.pth', map_location='cpu', weights_only=False)
    assert a['policy_state'].keys() == b['policy_state'].keys()
    errors = {key: float((a['policy_state'][key] - b['policy_state'][key]).abs().max())
              for key in a['policy_state'] if a['policy_state'][key].numel()}
    maximum = max(errors.values())
    assert maximum <= 1e-6, sorted(errors.items(), key=lambda x: -x[1])[:10]
    assert a['stats_history'] == b['stats_history']
    assert a['optim_value_state']['state'].keys() == b['optim_value_state']['state'].keys()
    optimizer_error = 0.
    for key, state in a['optim_value_state']['state'].items():
        for field, value in state.items():
            other = b['optim_value_state']['state'][key][field]
            if torch.is_tensor(value):
                optimizer_error = max(optimizer_error, float((value - other).abs().max()))
            else:
                assert value == other
    assert optimizer_error <= 1e-6
    assert (split / 'restored-full-state.json').exists()
    prepare.save('continuation-preflight.json', dict(status='PASS',
        test='3-episode tiny GRU: uninterrupted versus pause after ep2 and full-state resume',
        max_policy_error=maximum, max_optimizer_error=optimizer_error,
        stats_history_equal=True, tolerance=1e-6, wall_seconds=time.time() - started,
        formal_weights_never_reused=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        prepare.save('continuation-preflight-error.json', dict(error=repr(error)))
        raise
