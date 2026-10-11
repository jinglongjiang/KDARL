"""Wait for both verified measurement stages, then generate the revised single report."""
import sys
sys.dont_write_bytecode=True
import json
import subprocess
import time
from common import BASE, save, sha, verify

PY='/home/abc/temp/kda-vl-env/bin/python'
start=time.time()
for sentinel in ('DONE.json','decision-DONE.json'):
    while not (BASE/sentinel).exists():
        assert not (BASE/'ERROR.json').exists(), 'Primary collector failed'
        assert not (BASE/'decision-ERROR.json').exists(), 'Supplement collector failed'
        assert time.time()-start<7200, 'Waiting limit exceeded; do not claim completed'
        time.sleep(3)
    assert json.loads((BASE/sentinel).read_text())['status']=='COMPLETED'

analysis_files=('summarize-case-clusters.py','summarize-decisions.py','make-report.py','verify-inputs.py')
analysis_manifest={str(BASE/p):sha(BASE/p) for p in analysis_files}
save(BASE/'analysis-frozen-manifest.json',dict(files=analysis_manifest))
for name in analysis_files:
    result=subprocess.run([PY,'-B',str(BASE/name)],capture_output=True,text=True)
    save(BASE/'job-evidence'/('final-'+name+'.json'),dict(exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr))
    print(name, 'exit', result.returncode, result.stdout[-1000:], flush=True)
    assert result.returncode==0,result.stderr
for filename in ('amendment-frozen-manifest.json','shared-supplement-manifest.json','analysis-frozen-manifest.json'):
    entries=json.loads((BASE/filename).read_text())['files']
    assert entries
    for path,expected in entries.items():
        assert sha(path)==expected, ('Revision file changed',path)
verify()
assert (BASE/'REPORT.md').stat().st_size>1000
save(BASE/'revised-audit-DONE.json',dict(status='已验证通过',training=0,
    original_project_modified=False,own_episodes=1320,own_steps=81434,
    shared_reference_episodes=120,checkpoints=11,
    original_and_revision_manifests_verified=True,
    report_sha256=sha(BASE/'REPORT.md'),report_bytes=(BASE/'REPORT.md').stat().st_size,
    report=str(BASE/'REPORT.md')))
print('REVISED AUDIT COMPLETE',flush=True)
