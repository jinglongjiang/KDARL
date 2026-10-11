"""Inventory real historical states and freeze original root windows; never train."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
PROJECT = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
DATA = Path('/home/abc/workspace/ttc5-causal-20261011/work')
RUNS = {
    419: Path('/home/abc/4090_backup_20261009/root/kda-history-20261009/runs/kda-t24'),
    443: Path('/home/abc/workspace/kda-history-replication-20261010/runs/kda-t24'),
}
IDS = {419: [1, 221], 443: [239, 392]}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def read(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def save(name, value):
    (BASE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    torch.set_num_threads(1)
    assert not (BASE / 'protocol.json').exists()
    save('protocol.json', dict(task='Historical RL preference-formation timeline',
        training_authorized=False, T1='All existing real RL snapshots ep500/1000 through10000',
        T32='Definition requested; not supplied. No substitute experiment authorized.',
        probes=IDS, controls='Original preselected matched success controls, not result-selected',
        frozen='Real original TTC5 history, all80 smoothed commands and original reward/filter',
        margin='Unmasked original r + gamma V. Risk, mask and index0 separate; no probe uses0.',
        attribution='Intervals between saved states only; never reconstruct exact batches from Adam moments.',
        no_IL_rerun=True, no_RL_rerun=True, reserve92000_untouched=True))
    spec = importlib.util.spec_from_file_location('formation_contracts', PROJECT / 'crowd_nav/contracts.py')
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    roots = json.loads((DATA / 'roots.json').read_text())['roots']
    probes = []
    input_paths = [DATA / 'roots.json', PROJECT / 'crowd_nav/contracts.py']
    windows = {}
    for seed in RUNS:
        model = 'kda-s%d-t24' % seed
        wanted = IDS[seed]
        bs = DATA / ('branches-%s-cell5.jsonl.gz' % model)
        ep = DATA / ('baseline-%s-cell5.jsonl.gz' % model)
        input_paths.extend([bs, ep])
        branch = {r['root']['root_id']: r for r in read(bs) if r['root']['root_id'] in wanted}
        cases = {branch[r]['root']['case'] for r in wanted}
        episodes = {r['case']: r for r in read(ep) if r['case'] in cases}
        for rid in wanted:
            record = branch[rid]
            root = record['root']
            assert root == next(r for r in roots if r['root_id'] == rid)
            snap = root['snapshot']
            k = root['step']
            history = [np.asarray(s['current_token'], dtype=np.float32)
                       for s in episodes[root['case']]['steps'][:k+1]]
            prefix = np.asarray(history[-23:])
            if len(prefix) < 23:
                prefix = np.concatenate([np.repeat(prefix[:1], 23-len(prefix), axis=0), prefix])
            commands = np.asarray(snap['commands'], dtype=np.float64)
            assert commands.shape == (80, 2)
            robot = np.asarray(snap['robot'], dtype=np.float32)
            humans = np.asarray(snap['humans'], dtype=np.float64)[snap['ids']].copy()
            humans[:, :2] += .25 * humans[:, 2:4]
            joint = np.concatenate([robot, humans.reshape(-1)])
            joint = np.repeat(joint[None], 80, axis=0)
            rp = np.asarray(snap['robot'][:2], dtype=np.float64)[None] + .25 * commands
            joint[:, :2] = rp.astype(np.float32).astype(joint.dtype)
            joint[:, 2:4] = commands.astype(np.float32).astype(joint.dtype)
            next_tok = contracts._batch_joint34_to_tokens_vectorized(joint)
            win = np.ascontiguousarray(np.concatenate([
                np.broadcast_to(prefix[None], (80, 23, 8, 13)), next_tok[:, None]], axis=1))
            assert digest(win) == snap['windows_sha256'], (rid, 'original window mismatch')
            windows['root%d' % rid] = win
            branches = [{name: b[name] for name in ('candidate_index', 'labels', 'root_return', 'outcome', 'time')}
                        for b in record['branches']]
            probes.append(dict(root_id=rid, seed=seed, model=model, cell=5, case=root['case'],
                step=k, time=.25*k, root_type=root['type'], labels=root['labels'],
                matched_failure_case=root.get('matched_failure_case'),
                snapshot=snap, branches=branches, source=str(bs),
                windows_sha256=digest(win)))
    save('frozen-probes.json', probes)
    np.savez_compressed(BASE / 'frozen-windows.npz', **windows)
    inventory = []
    checkpoint_rows = []
    for seed, folder in RUNS.items():
        for path in sorted(folder.glob('*.pth*')):
            with (gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')) as f:
                payload = torch.load(f, map_location='cpu', weights_only=False)
            entry = dict(seed=seed, path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                keys=list(payload.keys()), episode=payload.get('episode'),
                type='actual historical artifact')
            for name in ('optim_value_state', 'optimizer_state', 'optimizer', 'optim_state'):
                if name in payload:
                    opt = payload[name]
                    states = opt.get('state', {})
                    steps = [float(v['step']) for v in states.values() if 'step' in v]
                    tensors = [v for row in states.values() for v in row.values() if torch.is_tensor(v)]
                    entry[name] = dict(parameter_states=len(states),
                        step_min=min(steps) if steps else None, step_max=max(steps) if steps else None,
                        finite=all(bool(torch.isfinite(v).all()) for v in tensors),
                        sample_fields=list(next(iter(states.values())).keys()) if states else [],
                        param_groups=[{k:v for k,v in g.items() if k != 'params'} for g in opt.get('param_groups', [])])
            if path.name.startswith('rl_model_ep'):
                assert payload['episode'] == int(path.stem.split('ep')[1])
                state = {k[10:] if k.startswith('_orig_mod.') else k: v.detach().clone()
                         for k,v in payload['policy_state'].items()}
                dest = BASE / 'weights' / str(seed) / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state_dict=state, episode=payload['episode'],
                    original_sha256=entry['sha256']), dest)
                checkpoint_rows.append(dict(seed=seed, episode=payload['episode'],
                    source=str(path), source_sha256=entry['sha256'],
                    derived_path=str(dest.relative_to(BASE)), derived_sha256=sha(dest)))
                entry['derived_policy_bytes'] = dest.stat().st_size
            # Structural presence does not prove a replay can reproduce historical batches.
            entry['replay_related_keys'] = [k for k in payload if any(
                s in k.lower() for s in ('replay', 'buffer', 'batch', 'rng', 'random'))]
            inventory.append(entry)
        for name in ('train.log', 'trajectory.log'):
            path = folder / name
            if path.exists():
                input_paths.append(path)
                inventory.append(dict(seed=seed, path=str(path), bytes=path.stat().st_size,
                                      sha256=sha(path), type='historical log'))
    save('asset-inventory.json', inventory)
    save('checkpoint-index.json', sorted(checkpoint_rows, key=lambda r:(r['seed'], r['episode'])))
    save('input-manifest.json', {str(p): sha(p) for p in input_paths})
    print(json.dumps(dict(status='PASS', frozen_probe_ids=[r['root_id'] for r in probes],
        checkpoint_count=len(checkpoint_rows), inventory_count=len(inventory)), indent=2))


if __name__ == '__main__':
    main()
