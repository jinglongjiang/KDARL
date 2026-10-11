"""Check source-result snapshot stability and frozen model/code inputs."""
import json
from pathlib import Path
from common import BASE, verify, sha, save

p=json.loads((BASE/'p2-provenance.json').read_text())
entries=[e for e in p['entries'] if e.get('sha256')]
assert entries
changed=[]
for i,e in enumerate(entries):
    path=Path(e['path'])
    if not path.is_file() or sha(path)!=e['sha256']:changed.append(e['path'])
    if i%2000==0:print('source_results_checked',i,flush=True)
out=dict(status='已验证通过' if not changed else '已确认失败',source_results=len(entries),
         changed=changed,frozen_code_weights_configs=verify())
save(BASE/'input-verification.json',out)
assert not changed, changed
print(json.dumps(out,ensure_ascii=False))
