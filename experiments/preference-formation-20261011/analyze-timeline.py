"""Summarize actual saved RL states; do not infer unrecorded training batches."""
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    raw = json.loads((BASE / 'timeline-raw.json').read_text())
    probes = json.loads((BASE / 'frozen-probes.json').read_text())
    inventory = json.loads((BASE / 'asset-inventory.json').read_text())
    records = json.loads((BASE / 'record-inventory.json').read_text())
    index = json.loads((BASE / 'checkpoint-index.json').read_text())
    assert raw['status'] == records['status'] == 'PASS'
    assert len(index) == 30 and len(raw['rows']) == 60
    pairs = []
    lines = ['seed\troot\tcase\troot_step\tepisode\tadam_step\talternative\tlabels\tM_unmasked\tvalue_difference\tmask_a0\tmask_alt\trisk_a0\trisk_alt\tarchival_return_difference']
    for probe in probes:
        rows = sorted([r for r in raw['rows'] if r['root_id'] == probe['root_id']],
                      key=lambda r: r['episode'])
        a0 = probe['labels']['A0']
        g0 = next(b['root_return'] for b in probe['branches'] if b['candidate_index'] == a0)
        for branch in probe['branches']:
            alternative = branch['candidate_index']
            if alternative == a0:
                continue
            points = []
            for row in rows:
                margin = next(m for m in row['margins'] if m['alternative'] == alternative)
                original = next(r for r in inventory if r['seed'] == probe['seed']
                                and r.get('episode') == row['episode']
                                and Path(r['path']).name.startswith('rl_model_ep'))
                optimizer = original['optim_value_state']
                assert optimizer['finite'] and optimizer['parameter_states'] == 92
                assert optimizer['step_min'] == optimizer['step_max'] == 4 * row['episode']
                assert alternative != 0 and a0 != 0
                delta = row['raw_Q'][alternative] - row['raw_Q'][a0]
                assert delta == margin['raw_Q_difference']
                assert all(not x for x in margin['mask'])
                assert margin['risk'] == [0., 0.]
                point = dict(episode=row['episode'], adam_step=int(optimizer['step_min']),
                             M=delta, values=[row['values'][a0], row['values'][alternative]])
                points.append(point)
                fields = [probe['seed'], probe['root_id'], probe['case'], probe['step'],
                          row['episode'], point['adam_step'], alternative,
                          ','.join(branch['labels']), format(delta, '.12g'),
                          format(margin['value_difference'], '.12g'), *margin['mask'],
                          *margin['risk'], format(branch['root_return'] - g0, '.12g')]
                lines.append('\t'.join(map(str, fields)))
            changes = []
            for before, after in zip(points, points[1:]):
                if (before['M'] > 0) != (after['M'] > 0):
                    changes.append(dict(from_episode=before['episode'],
                        to_episode=after['episode'], from_M=before['M'], to_M=after['M'],
                        update_start_exclusive=before['adam_step'],
                        update_end_inclusive=after['adam_step']))
            pairs.append(dict(seed=probe['seed'], root_id=probe['root_id'], case=probe['case'],
                root_step=probe['step'], root_type=probe['root_type'], a0=a0,
                alternative=alternative, labels=branch['labels'],
                archival_return_difference=branch['root_return'] - g0,
                reference_continuation='Original final-checkpoint policy; not historical-policy Q',
                points=points, sign_changes_between_saved_states=changes,
                earliest_saved_M=points[0]['M'], final_M=points[-1]['M']))
    assert len(pairs) == 12 and len(lines) == 181
    lookup = {(p['root_id'], p['alternative']): p for p in pairs}
    assert [(c['from_episode'], c['to_episode']) for c in
            lookup[(1, 5)]['sign_changes_between_saved_states']] == [(1500, 2000)]
    assert [(c['from_episode'], c['to_episode']) for c in
            lookup[(239, 9)]['sign_changes_between_saved_states']] == [(7000, 8000), (9000, 10000)]
    assert not lookup[(239, 24)]['sign_changes_between_saved_states']
    result = dict(T1_RL='COMPLETED', overall_task='INCOMPLETE',
        T32='NOT EXECUTED: original experiment definition unavailable', training=False,
        saved_checkpoints=30, fixed_roots=4, scored_root_checkpoints=60,
        preselected_action_pairs=12, pair_checkpoint_measurements=180,
        pairs=pairs, exact_batch_attribution='UNIDENTIFIABLE from retained historical assets',
        primary_decision='HOLD',
        decision_scope='No new training; obtain T3.2 definition, then choose IL or logged RL, not both',
        method_level_progress=False)
    (BASE / 'timeline-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (BASE / 'timeline.tsv').write_text('\n'.join(lines) + '\n')

    expected = json.loads((BASE / 'input-manifest.json').read_text())
    for entry in inventory:
        expected[entry['path']] = entry['sha256']
    for entry in records['records']:
        for file in entry['all_files']:
            expected[file['path']] = file['sha256']
    for entry in index:
        expected[str(BASE / entry['derived_path'])] = entry['derived_sha256']
    expected.update(raw['source_sha256'])
    freeze_path = Path('/home/abc/workspace/kda-history-replication-20261010/execution-freeze.json')
    freeze = json.loads(freeze_path.read_text())
    project = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
    historical = Path('/home/abc/4090_backup_20261009/root/cc-diag-20261008/code')
    source_keys = ['crowd_nav/train.py', 'crowd_nav/utils/ppo_buffer.py',
                   'crowd_nav/policy/mamba_rl.py', 'crowd_nav/contracts.py']
    for key in source_keys:
        expected[str(project / key)] = freeze['parent_source_sha256'][key]
        expected[str(historical / key)] = freeze['parent_source_sha256'][key]
    worker = Path('/home/abc/4090_backup_20261009/root/kda-history-20261009/train-worker.py')
    old_freeze = worker.parent / 'execution-freeze.json'
    expected[str(worker)] = json.loads(old_freeze.read_text())['scripts']['train-worker.py']
    for source in (freeze_path, old_freeze):
        expected[str(source)] = sha(source)
    verified = {}
    for path, digest in sorted(expected.items()):
        assert digest and len(digest) == 64 and Path(path).is_file()
        observed = sha(path)
        assert observed == digest, ('source changed', path, digest, observed)
        verified[path] = observed
    local = {str(p): sha(p) for p in sorted(BASE.iterdir())
             if p.is_file() and p.name != 'completion-manifest.json'}
    (BASE / 'completion-manifest.json').write_text(json.dumps(dict(
        status='PASS', historical_and_derived_assets_reverified=verified,
        local_outputs=local, no_training=True, T32_executed=False), indent=2))
    print(json.dumps(dict(status='PASS', T1_RL='COMPLETED', overall_task='INCOMPLETE',
        pair_checkpoint_measurements=180, original_and_derived_assets_verified=len(verified),
        sources_unchanged=True, training=False)))


if __name__ == '__main__':
    main()
