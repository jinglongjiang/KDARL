"""Record implementation repairs explicitly; never edit result-driven thresholds."""
import hashlib
import json
from pathlib import Path
import time
import sys

BASE=Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


p=BASE/'manifest.json'
d=json.loads(p.read_text())
old=sha(p)
changes=[]
for f in BASE.glob('*.py'):
    value=sha(f)
    if d['files'].get(f.name)!=value:
        changes.append({'file':f.name,'before':d['files'].get(f.name),'after':value})
    d['files'][f.name]=value
d['protocol']['success_matching']['eligibility']='timeout controls satisfy the same low-progress opportunity rule; collision controls have safe forward alternatives'
d['protocol']['numerical_return_tolerance']=1e-6
d.setdefault('amendments',[]).append({'time':time.time(),'previous_manifest_sha256':old,
    'changes':changes,'reason':sys.argv[1] if len(sys.argv)>1 else 'Pre-experiment correctness: CV-to-oracle identity assertion, record next TTC IDs, and phenotype-matched success controls',
    'branch_results_exist':False,'invalid_start':'Controller started before initial transfer completion; missing weight gate stopped before evaluation; transfer SIGTERM20 was an intentional optimization, not an experiment result'})
p.write_text(json.dumps(d,indent=2))
print(json.dumps({'manifest_sha256':sha(p),'changes':changes}))
