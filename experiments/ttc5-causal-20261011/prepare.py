"""Freeze provenance and the causal audit before observing interventions."""
import hashlib
import json
from pathlib import Path
import shutil
import time

BASE = Path(__file__).resolve().parent
WORK = Path('/home/abc/workspace')
REMOTE = '/root/ttc5-causal-20261011'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    assert not (BASE / 'manifest.json').exists(), 'Protocol already frozen'
    source = json.loads((WORK / 'route-b-20261010/protocol.json').read_text())
    names = ['kda-s419-t24', 'kda-s443-t24', 'gru-s419-t24', 'gru-s443-t24']
    models = []
    originals = {}
    files = {}
    for name in names:
        item = next(x for x in source['checkpoints'] if x['id'] == name)
        dest = BASE / 'inputs' / name
        dest.mkdir(parents=True)
        for key, filename in [('weight', 'rl_model_ep10000.pth'), ('config', 'config.ini')]:
            p = Path(item[key])
            digest = sha(p)
            assert digest and digest == item[key + '_sha256']
            shutil.copy2(p, dest / filename)
            originals[str(p)] = digest
            files[str((dest / filename).relative_to(BASE))] = digest
        old = (Path('/home/abc/4090_backup_20261009/root/kda-history-20261009')
               if item['seed'] == 419 else WORK / 'kda-history-replication-20261010')
        for cell in [0, 5]:
            src = old / 'evaluation' / (item['backbone'] + '-t24') / ('cell%d.json' % cell)
            assert src.exists()
            shutil.copy2(src, dest / ('historical-cell%d.json' % cell))
            files[str((dest / ('historical-cell%d.json' % cell)).relative_to(BASE))] = sha(src)
            originals[str(src)] = sha(src)
        models.append({**item, 'weight': REMOTE + '/inputs/' + name + '/rl_model_ep10000.pth',
                       'config': REMOTE + '/inputs/' + name + '/config.ini'})
    source_harness = WORK / 'kda-history-20261009/evaluate-history.py'
    shutil.copy2(source_harness, BASE / 'reference.py')
    originals[str(source_harness)] = sha(source_harness)
    project = WORK / 'nav_data/mamba/camrl/CrowdNav'
    project_hashes = {}
    for folder in ['crowd_nav', 'crowd_sim']:
        for p in sorted((project / folder).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.ini', '.config'):
                project_hashes[str(p.relative_to(project))] = sha(p)
                originals[str(p)] = sha(p)
    for p in BASE.glob('*.py'):
        files[p.name] = sha(p)
    protocol = {
        'task': 'TTC5-KDA causal attribution, zero training',
        'started_unix': time.time(), 'deadline_seconds': 8 * 3600,
        'models': models, 'project': '/root/ac3-selection-20261010-4090/project',
        'historical_ac3': '/root/ac3-selection-20261010-4090/per-case',
        'case_base': 90000, 'main_cases': 500, 'reference_cases': 32,
        'cells': [5, 4, 0, 3], 'alpha': .3, 'margin': .2, 'risk': .8,
        'gamma': .99, 'dt': .25, 'time_limit': 50., 'actions': 80,
        'root_rule_timeout': 'first t>=5, goal_distance>1, any clearance>=.2 and progress>=.02, selected progress<.02',
        'root_rule_collision': 'first all-unsafe; earliest previous step within12steps with safe progress>=.02 and a distinct safe alternative; otherwise no root',
        'stagnation': 'prior8steps net goal progress<.1; only if full8step history exists',
        'alternatives': {
            'A0': 'original executed candidate',
            'A1': 'highest raw-score safe candidate with progress>=.02',
            'A2': 'largest progress safe candidate (does not require >=.02)',
            'A3': 'highest final-score safe candidate distinct from A0/A1/A2',
            'ties': 'smallest grid index', 'duplicate_execution': 'once; preserve all associated labels'},
        'success_matching': {'same_model_cell': True, 'without_replacement': True,
                             'time_caliper_seconds': 2., 'goal_caliper_m': 1.,
                             'safe_count_caliper': 16, 'crowd_within2m_caliper': 2,
                             'distance': 'sum of normalized absolute differences; ties case then step'},
        'freeze': 'all baseline outcomes and all roots frozen before any intervention',
        'parity': 'historical KDA AC3 exact per-step commands/selection and float64 trajectory hash; original driver vs instrumented replay exact histories/windows/scores; historical GRU episode summaries separately',
        'historical_history_limitation': 'historical histories not stored; reconstructed from unchanged driver and checked against independent repeat, not claimed historical tensor archival parity',
        'continuation': 'replay from start; replace one actual smoothed command; update last_action to executed command; resume TTC5',
        'oracle': 'true successor with root-anchored IDs, identical prefix, batch80; record both inherited unmasked bootstrap and terminal-corrected bootstrap',
        'statistics': 'episode/root unit; seeds separate; case-cluster paired bootstrap across cells; no seed-population variance claim',
        'priority': 'KDA20square, GRU20square, 20circle, low-density references, Q3 read-only',
        'no_root_retained_in_denominator': True,
        'limitations': ['safe alternatives do not test masked-action rescue', '3alternatives are not exhaustive80',
                       'own-policy GRU comparison is not backbone-isolated training causality',
                       'input insufficiency vs learning cannot generally be isolated without retraining'],
        'decision_gate': {'GO': 'at least3distinct KDA cases spanning bothseeds with same module-localized actual regret, including at least2terminal rescues; only bounded mechanism validation, not novelty',
                          'KILL': 'otherwise stop this module-specific method candidate; do not retune roots or actions'},
        'reserved92000_untouched': True,
        'max_workers': 4,
    }
    (BASE / 'manifest.json').write_text(json.dumps({'protocol': protocol, 'files': files,
        'original_local_files': originals, 'project_files': project_hashes}, indent=2))
    print(json.dumps({'status': 'FROZEN', 'models': names, 'project_files': len(project_hashes),
                      'manifest_sha256': sha(BASE / 'manifest.json')}))


if __name__ == '__main__':
    main()
