"""Final source/weight verification after read-only attribution."""
import json
from pathlib import Path
import time
import audit as A

BASE = Path(__file__).resolve().parent
assert json.loads((BASE/'work/finalize-run.done.json').read_text())['status']=='PASS'
files = {rel:A.REF.sha(BASE/rel)==expected for rel,expected in A.MAN['files'].items()}
project = {rel:A.REF.sha(Path(A.P['project'])/rel)==expected
           for rel,expected in A.MAN['project_files'].items()}
assert files and project and all(files.values()) and all(project.values())
data = dict(status='PASS',time=time.time(),files=files,project_files=project)
A.save(BASE/'work/post-attribution-end-verification.json',data)
print(json.dumps(dict(status='PASS',assets=len(files),project_files=len(project))))
