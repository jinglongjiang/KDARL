"""Frozen historical RL checkpoint inference only. No environment or optimizer steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--project', required=True, type=Path)
    args = parser.parse_args()
    base, project = args.base, args.project.resolve()
    sys.path.insert(0, str(project))
    import crowd_nav
    import crowd_sim
    import crowd_nav.policy.mamba_rl as policy_module
    for module in (crowd_nav, crowd_sim, policy_module):
        assert Path(module.__file__).resolve().is_relative_to(project)
    from crowd_nav.train import load_config
    from crowd_nav.contracts import init_grid_from_cfg, grid_action_dim
    torch.set_num_threads(1)
    index = json.loads((base / 'checkpoint-index.json').read_text())
    probes = json.loads((base / 'frozen-probes.json').read_text())
    data = np.load(base / 'frozen-windows.npz', allow_pickle=False)
    rows = []
    exact = []
    started = time.time()
    for seed in (419, 443):
        cfg = load_config(str(base / ('config-%d.ini' % seed)))
        init_grid_from_cfg(cfg)
        assert grid_action_dim() == 80
        net = policy_module.MambaRLPolicy(cfg, device='cuda')
        net.eval()
        assert net.seq_len == 24
        assert net.test_action_smoothing == .3 and net.test_min_clearance == .2
        assert net.test_risk_lambda == .8
        own = [r for r in probes if r['seed'] == seed]
        for entry in [r for r in index if r['seed'] == seed]:
            path = base / entry['derived_path']
            assert sha(path) == entry['derived_sha256']
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['episode'] == entry['episode']
            assert state['original_sha256'] == entry['source_sha256']
            net.load_state_dict(state['state_dict'], strict=True)
            for p in own:
                win = torch.from_numpy(data['root%d' % p['root_id']]).float().cuda()
                assert win.shape == (80, 24, 8, 13)
                with torch.no_grad():
                    value = net.forward_value(win).cpu().numpy()
                assert np.isfinite(value).all()
                snap = p['snapshot']
                reward = np.asarray(snap['rewards'])
                raw = reward + .99 * value
                clear = np.asarray(snap['clearance'])
                final = raw.copy()
                safe = clear >= .2
                if safe.any():
                    final[~safe] = -1e9
                risk = .8 * np.maximum(0., .2 - clear)
                final -= risk
                final[0] -= .001
                if entry['episode'] == 10000:
                    stored = np.asarray(snap['values'])
                    assert np.array_equal(value, stored), (seed, p['root_id'], 'final-value parity',
                                                          float(np.max(np.abs(value - stored))))
                    exact.append(p['root_id'])
                a0 = p['labels']['A0']
                margins = []
                for branch in p['branches']:
                    idx = branch['candidate_index']
                    if idx == a0:
                        continue
                    assert a0 != 0 and idx != 0
                    margins.append(dict(a0=a0, alternative=idx, labels=branch['labels'],
                        value_difference=float(value[idx] - value[a0]),
                        raw_Q_difference=float(raw[idx] - raw[a0]),
                        mask=[bool(safe.any() and not safe[x]) for x in (a0, idx)],
                        risk=[float(risk[x]) for x in (a0, idx)],
                        historical_real_return=branch['root_return'], historical_outcome=branch['outcome']))
                rows.append(dict(seed=seed, episode=entry['episode'], root_id=p['root_id'],
                    case=p['case'], root_step=p['step'], root_type=p['root_type'],
                    original_checkpoint_sha256=entry['source_sha256'],
                    windows_sha256=p['windows_sha256'], values=value.tolist(),
                    raw_Q=raw.tolist(), final_scores=final.tolist(),
                    raw_argmax=int(raw.argmax()), final_argmax=int(final.argmax()),
                    margins=margins))
            print(json.dumps(dict(seed=seed, episode=entry['episode'], status='scored')), flush=True)
    assert sorted(exact) == [1, 221, 239, 392]
    result = dict(status='PASS', training=False, original_final_values_bitwise=exact,
        rows=rows, wall_seconds=time.time()-started, torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        source_sha256={str(p): sha(p) for p in (Path(__file__), project/'crowd_nav/policy/mamba_rl.py',
                                             project/'crowd_nav/contracts.py')})
    (base / 'timeline-raw.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
