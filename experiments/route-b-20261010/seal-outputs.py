"""Seal completed evidence, excluding transient caches and self-referential hashes."""
import json
from common import BASE, save, sha, verify

assert json.loads((BASE/'revised-audit-DONE.json').read_text())['status']=='已验证通过'
records={}
for folder in (BASE/'p1-margin',BASE/'job-evidence',BASE/'logs'):
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            if folder!=BASE/'logs':
                assert path.stat().st_size>0,path
            records[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path),
                empty_console_log=folder==BASE/'logs' and path.stat().st_size==0)
for name in ('REPORT.md','revised-plan.md','p2-provenance.json','p2-provenance.md','p3-novelty.md',
    'protocol.json','protocol-amendment-v2.json','frozen-manifest.json',
    'amendment-frozen-manifest.json','analysis-frozen-manifest.json','shared-supplement-manifest.json',
    'controller-retirement.json','decision-recovery.json','revised-audit-DONE.json',
    'input-verification.json','decision-metric-unit-checks.json','decision-smoke-check.json'):
    path=BASE/name
    assert path.stat().st_size>0,path
    records[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
assert records
save(BASE/'output-manifest.json',dict(status='已验证通过',files=records,
    verified_original_inputs=verify(),exclusions=['runtime caches','output-manifest self hash']))
print(json.dumps(dict(status='PASS',sealed_files=len(records),bytes=sum(v['bytes'] for v in records.values()))))
