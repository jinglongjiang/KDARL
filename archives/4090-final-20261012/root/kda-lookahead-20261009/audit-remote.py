"""Supplement local provenance with already-existing server diagnostic assets."""
import sys
sys.dont_write_bytecode = True
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

BASE = Path(__file__).resolve().parent
D = Path('/root/cc-diag-20261008')


def main():
    p = BASE / 'L0-C-provenance.json'; result = json.loads(p.read_text())
    assert 'remote_entries' not in result
    entries = []; valid = []
    scopes = [D / x for x in ('results', 'd0', 'd1', 'd4', 'd6', 'd7')]
    scopes += [Path('/root/bayes-fix-20261008')]
    for scope in scopes:
        if not scope.exists():
            entries.append(dict(path=str(scope), status='未验证', reason='directory missing')); continue
        for path in sorted(scope.rglob('*')):
            if path.is_symlink() or not path.is_file(): continue
            if not (path.suffix in ('.json', '.csv', '.log') or path.name.endswith('.json.gz')): continue
            entry = dict(path=str(path), bytes=path.stat().st_size, run_label=None, backbone=None,
                weight_sha256=None, case_range=None, selection_rule=None, action_definition=None, fields=[])
            if path.suffix == '.csv':
                with path.open(newline='') as stream:
                    reader = csv.DictReader(stream); entry['fields'] = reader.fieldnames or []
                    entry['run_label'] = sorted({x.get('run_label') for x in reader if x.get('run_label')})
                entry['record_level'] = 'episode aggregate'
            elif path.suffix == '.log':
                with path.open(errors='replace') as stream: sample = stream.read(4096)
                entry['fields'] = ['unstructured log']; entry['position_sequence_present'] = 'TRAJECTORY:' in sample
            else:
                try:
                    if path.name.endswith('.gz'):
                        with gzip.open(path, 'rt') as stream: data = json.load(stream)
                    else: data = json.loads(path.read_text())
                except (ValueError, OSError) as error:
                    entry['reason'] = repr(error); entries.append(entry); continue
                if isinstance(data, dict):
                    entry['fields'] = list(data); entry['backbone'] = data.get('backbone')
                    entry['weight_sha256'] = data.get('checkpoint_sha256')
                    entry['selection_rule'] = data.get('selection', data.get('rule'))
                    entry['action_definition'] = {k: data[k] for k in ('grid_actions', 'candidate_actions') if k in data} or None
                    episodes = data.get('episodes', [])
                    if isinstance(episodes, list) and episodes and isinstance(episodes[0], dict):
                        entry['episode_fields'] = list(episodes[0]); cases = [x['case'] for x in episodes if isinstance(x.get('case'), int)]
                        if cases: entry['case_range'] = [min(cases), max(cases)]
                    if scope == D / 'd1' and data.get('frames'):
                        frames = data['frames']; metadata = data['summary']
                        entry.update(backbone=path.parent.name, case_range=[metadata['case'], metadata['case']],
                            selection_rule='first5 (source-path contract, not TTC5)',
                            action_definition='80 original grid indices; executed command=frames[t].commands[frames[t].final]',
                            frame_fields=list(frames[0]), raw_executed_sequence_available=True,
                            n_frames=len(frames), case=metadata['case'], cell=metadata['cell'])
                        assert entry['weight_sha256']
                        for frame in frames:
                            assert len(frame['commands']) == 80 and 0 <= frame['final'] < 80
                        errors = [float(np.max(np.abs(np.asarray(b['position']) - np.asarray(a['position'])
                            - .25 * np.asarray(a['commands'][a['final']])))) for a, b in zip(frames[:-1], frames[1:])]
                        entry['command_position_consistency_max_error'] = max(errors) if errors else 0.
                        assert entry['command_position_consistency_max_error'] < 2e-12
                        valid.append(entry)
            entries.append(entry)
    found = subprocess.run(['find', str(D / 'd1'), '-type', 'f', '-name', '*.json.gz'],
                           text=True, capture_output=True, check=True)
    result['remote_entries'] = entries
    result['remote_scopes'] = [str(x) for x in scopes]
    result['remote_find_actual_output'] = dict(command=['find', str(D / 'd1'), '-type', 'f', '-name', '*.json.gz'],
        exit_code=found.returncode, actual_stdout=found.stdout)
    result['oscillation_metrics_available'] = bool(valid)
    result['raw_frozen_evaluation_action_sequences_available'] = bool(valid)
    result['ttc5_oscillation_metrics_available_before_this_study'] = False
    result['existing_raw_action_episodes_by_backbone'] = {a: sum(x['backbone'] == a for x in valid) for a in ('kda', 'mamba', 'gru')}
    result['conclusion'] = ('已发现旧D1的First-5逐步动作序列，权重SHA/case/commands/final索引可关联，'
        '且相邻位置与执行命令一致。因此“整条KDA/Mamba线无振荡原始数据”不成立。'
        '指定旧TTC5选择/确认评估文件本身没有执行命令序列；不能用First-5或贝叶斯CSV替代TTC5。'
        '新H1/H2独立收集TTC5逐步命令。训练trajectory.log仍单列为位置代理，不冒充冻结评测动作记录。')
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(dict(status='已验证通过', remote_entries=len(entries), raw_episodes=result['existing_raw_action_episodes_by_backbone'],
        oscillation_metrics_available=result['oscillation_metrics_available'], output=str(p), bytes=p.stat().st_size)), flush=True)


if __name__ == '__main__': main()
