"""Copy disjoint completed evidence folders with bounded network concurrency."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parent
REMOTE='root@45.126.120.6:/root/ab-causal-isolation-20261011/'
MODELS=['gru-s419-t24','gru-s443-t24','kda-s419-t24','kda-s443-t24']


def main():
    jobs=['b/work/main/model/'+m+'/' for m in MODELS]
    jobs+=['b/work/main/orca-grid/','b/work/main/cv-mpc/','b/work/smoke/','b/work/branches/','b/n10/work/']
    out=BASE/'execution/evidence-sync';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'done.json').exists()
    def copy(rel):
        destination=BASE/rel;destination.mkdir(parents=True,exist_ok=True)
        started=time.time()
        code=subprocess.call(['rsync','-az',REMOTE+rel,str(destination)+'/'])
        return dict(path=rel,exit_code=code,wall_seconds=time.time()-started)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(copy,rel) for rel in jobs]):
            row=future.result();results.append(row)
            with (out/'journal.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            assert row['exit_code']==0
    code=subprocess.call(['rsync','-az','--include=*.json','--exclude=*/','--exclude=*',REMOTE+'b/work/',str(BASE/'b/work')+'/'])
    assert code==0
    (out/'done.json').write_text(json.dumps(dict(status='PASS',jobs=results,
        scope='Complete B main/smoke/branches/native10 raw evidence plus root/verification metadata. Copy only, no deletion or experimental change.')))


if __name__=='__main__':main()
