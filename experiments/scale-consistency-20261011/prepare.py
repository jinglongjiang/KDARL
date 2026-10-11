"""Freeze the revised user gate before running any new environment episodes."""
import json
from common import BASE, PROJECT, HISTORY, CELLS, save, sha


def main():
    assert not (BASE / 'protocol.json').exists(), 'Protocol must not be overwritten'
    old = json.loads((BASE / 'protocol-pending.json').read_text())
    old['status'] = 'FROZEN_REVISED_GATE'
    for item in old['weights']:
        item['config_sha256'] = sha(item['config'])
    old['purpose'] = 'Test whether normalized paired disagreement remains approximately flat on a finer people-count axis, not locate a variance onset'
    old['C_definition'] = 'Same seed and backbone, T24 versus T12'
    old['cells'] = CELLS
    old['parity'] = dict(
        seed=443, models=['kda-t24', 'gru-t24'], cells=[0, 5], cases=[91000, 91499],
        required_cases=2000,
        exact_fields=['outcome', 'time', 'selection_trace_sha256'],
        tolerance_fields=['return', 'minimum_clearance'],
        same_device_absolute_tolerance=1e-6, cross_device_absolute_tolerance=1e-4,
        actual_comparison='3060 torch2.7.1 versus old seed443 torch2.9.1 evaluations; cross-device tolerance for both arms',
        interpretation='Saved-field equivalence only; NOT historical bitwise action/state trajectory reproduction',
        selection_hash_limit='Many state/action paths can produce the same selected-ID sequence, especially N=5 where the set is constant',
        first_divergence_step='Unavailable if old per-step records do not exist; report first mismatching case and all fields without inventing a step',
        stop='Any field mismatch, nonzero exit, traceback or assertion: no new cells',
        no_4090_replay=True, no_reconstructed_historical_trajectory=True)
    old['parity_required'] = 'Revised saved-field equivalence, explicitly not bitwise historical actions'
    old.pop('blocker', None)
    old['smoke'] = dict(cells=[6, 7, 8, 9], cases_per_cell=3,
                        checkpoints=8, episodes=96,
                        required=['actual human count', '80 actions', '50 second limit',
                                  'finite metrics', 'case pairing', 'output files'])
    old['scheduling'] = dict(device='RTX3060', max_workers=3, no_training=True,
                            full_jobs=32, full_episodes=16000,
                            fail_closed=True, automatic_restart=False,
                            no_4090_usage=True)
    old['time_limit'] = 50
    old['engineering_note'] = '[eval_protocol] time_limit=25 is ignored by this environment/harness; enabling it later would change the historical protocol'
    old['metrics'] = ['binary-success paired b/c', 'disagreement (b+c)/n',
                      'same-marginal independent expectation', 'ratio and case-bootstrap CI95',
                      'net SR difference (b-c)/n as cancellation-prone secondary metric']
    old['interpretation'] = dict(
        flat='No obvious additional count-related increase in normalized disagreement on tested weights; not proof of independence or a causal ceiling-effect explanation',
        rising='Monotonic increase is evidence to revise the upstream descriptive claim',
        comparisons='Report A, B and C side by side, not backbone superiority or seed population variance',
        timeout='50s offers substantial nominal travel-time slack but timeout alone does not prove stalling; detours, interactions and speed reductions remain possible')
    old['limitations'].append('No reference per-step actions exist; a mismatching hash cannot identify the first divergent step')
    save(BASE / 'protocol.json', old)
    files = json.loads((BASE / 'frozen-manifest.json').read_text())['files']
    original_freeze = BASE / 'frozen-manifest.json'
    save(BASE / 'existing-audit-manifest.json', json.loads(original_freeze.read_text()))
    source_freeze = json.loads((BASE.parent / 'kda-history-replication-20261010/execution-freeze.json').read_text())
    source_hashes = source_freeze['parent_source_sha256']
    assert source_hashes
    for name, expected in source_hashes.items():
        assert sha(PROJECT / name) == expected, name
    for root in (PROJECT / 'crowd_nav', PROJECT / 'crowd_sim', PROJECT / 'tools',
                 BASE.parent / 'kda-deps/flash-linear-attention/fla'):
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.ini', '.config'):
                files[str(path)] = sha(path)
    for path in [HISTORY, BASE / 'protocol.json', *sorted(BASE.glob('*.py'))]:
        files[str(path)] = sha(path)
    save(BASE / 'frozen-manifest.json', dict(files=files, weights=old['weights'],
                                           stage='revised_gate_frozen',
                                           original_source_files_match=len(source_hashes)))
    save(BASE / 'status.json', dict(stage='READY_FOR_PARITY', new_episodes=0, training_episodes=0))
    print(json.dumps(dict(status='FROZEN', files=len(files), weights=8,
                          parity_episodes=2000, full_episodes=16000)))


if __name__ == '__main__':
    main()
