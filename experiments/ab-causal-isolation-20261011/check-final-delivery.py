"""Fail-closed first-round delivery gate; no training or model imports."""
import datetime
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SECTIONS = ('研究问题', '已经确认的直接机制', '已执行的因果干预',
            '各因素的实际导航效果', '被排除的竞争解释', '无法单独归因的交互作用',
            '最主要的可修复机制', '下一步是否具备算法研发依据')


def read(rel):
    return json.loads((BASE / rel).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = read('frozen-manifest.json')
    end = read('end-verification.json')
    assert start['status'] == end['status'] == 'PASS'
    assert start['original_files'] == end['original_files']
    assert start['models_and_configs'] == end['models_and_configs']
    assert sha(BASE / 'protocol.json') == start['artifact_sources']['protocol.json']
    mirror = read('evidence-mirror-verification.json')
    assert mirror['status'] == 'PASS' and mirror['local'] and mirror['remote']
    assert read('execution/evidence-sync/done.json')['status'] == 'PASS'
    assert read('execution/reference-evidence-sync/done.json')['status'] == 'PASS'
    assert read('final-cpu-acceptance.json')['total_tests'] == 39
    assert read('final-cpu-acceptance.json')['status'] == 'PASS'
    assert read('a/artifacts/a2-factor-split/summary.json')['status'] == 'PASS'
    assert read('a/artifacts/count-matched-complete/count-matched-summary.json')['status'] == 'PASS'
    assert read('a/artifacts/verified-prefix-safety.json')['status'] == 'PASS'
    assert read('b/work/verified-baseline.json')['execution_status'] == 'PASS'
    assert read('b/work/verified-complete.json')['execution_status'] == 'PASS'
    assert read('b/work/verified-mamba-reference.json')['execution_status'] == 'PASS'
    reports = {}
    for name in ('A_ROOT_CAUSE.md', 'B_ROOT_CAUSE.md', 'CAUSAL_DECISION.md'):
        path = BASE / name
        text = path.read_text()
        assert all('【' + section + '】' in text for section in SECTIONS), name
        assert 'INCOMPLETE' in text, name
        assert not re.search(r'PENDING|RUNNING|待执行|正在执行', text), name
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if link.startswith(('http:', 'https:', '#')):
                continue
            assert (BASE / link.split('#')[0]).exists(), (name, link)
        reports[name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    protocol = read('protocol.json')
    assert protocol['round2']['authorized'] is False
    now = datetime.datetime.now(datetime.timezone.utc)
    started = datetime.datetime.fromisoformat(protocol['started_local'])
    elapsed = (now - started).total_seconds()
    assert 0 < elapsed <= protocol['max_wall_seconds'], 'INCOMPLETE: first-round wall-clock limit'
    result = dict(first_round_execution='COMPLETE', overall_root_cause_attribution='INCOMPLETE',
        reason='Matched learning interventions are not authorized; natural cross-population strict opportunity matching yielded zero eligible pairs/triples.',
        method_level_progress=False, method_spec_freeze=False, round2_started=False,
        primary_research_decision='HOLD',
        unique_next_action='Seek separate authorization for the specified two-seed GRU observation-range x training-coverage causal contract; six distinct training jobs conditional on exact N5 equivalence, timing first.',
        completed_at_utc=now.isoformat(), wall_seconds=elapsed, reports=reports,
        original_source_files=len(end['original_files']), frozen_weight_config_files=len(end['models_and_configs']),
        local_raw_mirror_files=len(mirror['local']), original_and_input_hashes_unchanged=True)
    (BASE / 'delivery-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
