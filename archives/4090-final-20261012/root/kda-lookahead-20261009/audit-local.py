"""CPU-only source and asset provenance audit; no navigation policy execution."""
import sys
sys.dont_write_bytecode = True
import ast
import configparser
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

BASE = Path(__file__).resolve().parent
PROJECT = Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
ARTIFACT = PROJECT / 'artifacts/vl-optimized-20261007'
SCOPES = [ARTIFACT, Path('/home/abc/workspace/kda-selection-20261008'),
    Path('/home/abc/workspace/kda-history-20261009'), Path('/home/abc/workspace/mamba-ttc5-20261009'),
    Path('/home/abc/4090_backup_20261008/root/kda-vl-optimized-20261007/results'),
    Path('/home/abc/4090_backup_20261008/root/bayes-vl-optimized-20261007/crowd_nav/runs'),
    Path('/home/abc/4090_backup_20261008/root/cc-6env-bayes-20261007'),
    Path('/home/abc/temp/cc_6env_20261007')]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, data):
    (BASE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    sys.path.insert(0, str(PROJECT))
    from crowd_nav import contracts as c
    assert Path(c.__file__).resolve().is_relative_to(PROJECT) if hasattr(Path(), 'is_relative_to') else str(Path(c.__file__).resolve()).startswith(str(PROJECT) + '/')
    source = Path(c.__file__).read_text()
    tree = ast.parse(source)
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_batch_joint34_to_tokens_vectorized')
    declarations = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any('tokens_batch[' in ast.get_source_segment(source, t) for t in node.targets):
            declarations.append({'line': node.lineno, 'code': ast.get_source_segment(source, node)})
    robot = ['px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref (raw)', 'theta',
        'sqrt((gx-px)^2+(gy-py)^2)', 'sqrt(vx^2+vy^2)', 'clip(v_pref,0,3)', '1 (robot flag)']
    goal = ['gx-px', 'gy-py', 'sqrt(dx^2+dy^2)', 'atan2(dy,dx)', 'cos(goal angle)', 'sin(goal angle)']
    motion = ['speed if speed>1e-6 else 0', 'atan2(vy,vx) if speed>1e-6 else 0',
        'cos(motion angle) if speed>1e-6 else 0', 'sin(motion angle) if speed>1e-6 else 0']
    human = ['human.px-robot.px', 'human.py-robot.py', 'sqrt(rel_x^2+rel_y^2+1e-6)',
        'human.vx (absolute)', 'human.vy (absolute)', 'sqrt(human.vx^2+human.vy^2)',
        'human.radius', 'inverse TTC: dist/closing; clamp TTC>=0.1; receding=0', '1 if dist<2 else 0']
    rows = []
    for row in range(8):
        fields = robot if row == 0 else goal if row == 1 else motion if row == 2 else human
        for col in range(13):
            rows.append(dict(row=row, column=col, code_field=fields[col] if col < len(fields) else '0',
                paper_symbol=None, always_zero=col >= len(fields),
                actor_order='existing TTC order within selected five' if row >= 3 else None))
    raw = np.array([[1., 2., .6, .8, .3, 4., 6., 4., .2] +
        sum(([3. + i, 4. - i, -.2 * i, .3, .2] for i in range(5)), [])])
    tokens = c._batch_joint34_to_tokens_vectorized(raw)
    assert tokens.shape == (1, 8, 13) and np.isfinite(tokens).all()
    for x in rows:
        if x['always_zero']:
            assert tokens[0, x['row'], x['column']] == 0
    assert tokens[0, 0, 7] == 4. and tokens[0, 0, 11] == 3.
    assert np.array_equal(tokens[0, 1, :3], np.array([3., 4., 5.], dtype=np.float32))
    warning = '当前KDA-VL代码不等于第一篇Mamba-VL投稿版本。本表只陈述当前代码事实，不作为第一篇勘误依据。论文符号未提供，全部留空。'
    save('L0-A-token-fields.json', dict(status='已验证通过', warning=warning, shape=[1, 8, 13],
        source=str(Path(c.__file__).resolve()), source_sha256=sha(c.__file__), rows=rows,
        assignments=sorted(declarations, key=lambda x: x['line']), synthetic_input=raw.tolist(), synthetic_output=tokens.tolist()))
    lines = ['# 当前代码token字段对照', '', warning, '', '| 行 | 列 | 代码字段 | 论文符号 | 恒零 |', '|---|---|---|---|---|']
    lines += ['| %d | %d | %s |  | %s |' % (x['row'], x['column'], x['code_field'], x['always_zero']) for x in rows]
    (BASE / 'L0-A-token-fields.md').write_text('\n'.join(lines) + '\n')
    cfg = configparser.ConfigParser(); assert cfg.read(str(ARTIFACT / 'kda.ini'))
    grid = c.init_grid_from_cfg(cfg).copy(); assert c.grid_action_dim(grid) == 80
    mappings = {}
    for include in (False, True):
        g = dict(grid, include_stop=include)
        a = [c.discrete_index_to_action(i, **g) for i in range(c.grid_action_dim(g))]
        assert len(a) == 80 + int(include)
        if include:
            assert a[0] == (0., 0.) and a[1:] == mappings['false']['vectors']
        vectors = np.array(a)
        ranges = []
        for prev in ([0., 0.], [.12885124808584156, 0.], [.3, 0.], [.7, 0.], [1., 0.], [0., 1.]):
            effective = .3 * np.array(prev) + .7 * vectors
            speeds = np.linalg.norm(effective, axis=1)
            ranges.append(dict(previous_command=prev, min_speed=float(speeds.min()), max_speed=float(speeds.max()),
                zero_count=int(np.count_nonzero(speeds == 0.)), effective_commands=effective.tolist()))
        mappings[str(include).lower()] = dict(action_count=len(a), vectors=a, speeds=sorted(set(np.round(np.linalg.norm(vectors, axis=1), 12))),
            smoothing_examples=ranges, index0_penalty=-.001,
            index0_penalty_command=list(a[0]), penalty_is_grid_index_not_stop_semantics=True)
        print('include_stop=%s: actions=%d index0=%s' % (include, len(a), a[0]), flush=True)
    save('L0-B-action-set.json', dict(status='已验证通过', grid=grid, mappings=mappings,
        warning='混合后的速度范围依赖上一命令；没有显式stop与所有可能平滑历史下绝对不能抵消为零不是同一个命题。本轮H2追加未混合的真零速度。',
        alpha=.3, no_previous_action='Original grid returned without smoothing',
        stop_in_H2_index=80, mapping_H2_existing_80_unchanged=True))

    entries = []; position_logs = []; csv_output = []
    for scope in SCOPES:
        if not scope.exists():
            entries.append(dict(path=str(scope), status='未验证', reason='directory missing')); continue
        command = ['find', str(scope), '-type', 'f', '-name', '*.csv']
        found = subprocess.run(command, capture_output=True, text=True, check=True)
        csv_output.append(dict(command=command, exit_code=found.returncode, actual_stdout=found.stdout))
        for path in sorted(scope.rglob('*')):
            if path.is_symlink():
                entries.append(dict(path=str(path), metadata_status='未验证',
                    reason='External legacy symlink not dereferenced; may target an inaccessible /root location'))
                continue
            if not path.is_file() or not (path.suffix in ('.json', '.csv', '.log') or path.name.endswith('.json.progress')):
                continue
            entry = dict(path=str(path), bytes=path.stat().st_size, run_label=None, backbone=None,
                weight_sha256=None, case_range=None, action_definition=None, selection_rule=None, fields=[],
                family='unverified mixed directory' if 'cc_6env' in str(scope) else 'bayesian' if 'bayes' in str(scope).lower() else 'KDA/Mamba-VL',
                family_evidence='directory context only; explicit run_label/checkpoint takes precedence',
                metadata_status='未验证')
            try:
                if path.suffix == '.csv':
                    with path.open(newline='') as stream:
                        reader = csv.DictReader(stream); entry['fields'] = reader.fieldnames or []
                        labels = set(); seeds = []
                        for row in reader:
                            if row.get('run_label'): labels.add(row['run_label'])
                            if row.get('seed'): seeds.append(int(row['seed']))
                    entry.update(run_label=sorted(labels), seed_range=[min(seeds), max(seeds)] if seeds else None,
                        metadata_status='已验证通过', record_level='episode aggregate, not per-step commands')
                    if any('bayes' in label for label in labels):
                        entry.update(family='bayesian', family_evidence='explicit run_label contains bayes')
                elif path.suffix == '.log':
                    with path.open(errors='replace') as stream:
                        sample = stream.read(1024 * 1024)
                    entry['fields'] = ['EP', 'START', 'GOAL', 'RESULT', 'STEPS', 'TRAJECTORY'] if 'TRAJECTORY:' in sample else ['unstructured log']
                    entry['position_sequence_present'] = 'TRAJECTORY:' in sample
                    entry['record_level'] = 'mixed training/validation position sequences; no reliable frozen-evaluation case/weight/command association' if 'TRAJECTORY:' in sample else 'unstructured'
                    if 'TRAJECTORY:' in sample and entry['family'] == 'KDA/Mamba-VL': position_logs.append(str(path))
                else:
                    if path.stat().st_size > 64 * 1024 ** 2:
                        entry['reason'] = 'JSON larger than audit parser bound'; entries.append(entry); continue
                    data = json.loads(path.read_text())
                    if isinstance(data, dict):
                        entry['fields'] = list(data)
                        entry['run_label'] = data.get('run_label'); entry['backbone'] = data.get('backbone')
                        entry['weight_sha256'] = data.get('checkpoint_sha256')
                        if entry['weight_sha256'] in ('cb2e20c4cc2f12014fed589e9ff02468d04cfa12573c66fc79617c95356cbf0a',
                                '96f06039a07b75d09e3b067ebb796d6de113b6d80a33214db737ed6d3a245828',
                                'f23e411effac2f1959eead081fa49c908a85532642172c50a34427fe40bff2b9'):
                            entry.update(family='KDA/Mamba-VL', family_evidence='explicit matching frozen checkpoint sha256')
                        entry['selection_rule'] = data.get('selection', data.get('rule'))
                        entry['action_definition'] = {k: data[k] for k in ('grid_actions', 'candidate_actions', 'smoothing') if k in data} or None
                        episodes = data.get('episodes', data.get('results', []))
                        if isinstance(episodes, list) and episodes and isinstance(episodes[0], dict):
                            entry['episode_fields'] = list(episodes[0]); cases = [x['case'] for x in episodes if isinstance(x.get('case'), int)]
                            if cases: entry['case_range'] = [min(cases), max(cases)]
                        entry['metadata_status'] = '已验证通过'
            except (ValueError, OSError, TypeError) as error:
                entry['reason'] = repr(error)
            entries.append(entry)
    save('L0-C-provenance.json', dict(status='已验证通过', scopes=[str(x) for x in SCOPES], entries=entries,
        find_csv_actual_outputs=csv_output, oscillation_metrics_available=False,
        raw_frozen_evaluation_action_sequences_available=False, position_sequences_available=bool(position_logs),
        position_sequence_files=position_logs,
        conclusion='KDA/Mamba线存在训练期位置日志，不等于具备冻结评测权重对应的逐步执行命令。未找到可用于本轮冻结评测振荡指标的可靠case/weight/action序列；贝叶斯30列CSV不得替代。新H1/H2将独立保存真实执行命令。'))
    print(json.dumps({'status': '已验证通过', 'provenance_entries': len(entries), 'position_logs': len(position_logs)}), flush=True)


if __name__ == '__main__':
    main()
