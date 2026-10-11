import json, sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1]); out = Path(sys.argv[2])
eps, lat, wall, sha, bb = [], [], 0.0, None, None
CELLS=[(5,'circle_crossing'),(5,'square_crossing'),(10,'circle_crossing'),
       (10,'square_crossing'),(20,'circle_crossing'),(20,'square_crossing')]
for i in range(6):
    f = d / f'cell{i}.json'
    assert f.exists(), f'缺分片 {f}'
    j = json.load(open(f))
    rows = j['episodes']
    hn, geo = CELLS[i]
    assert len(rows) == 32, f'cell{i} 只有 {len(rows)} ep'
    assert all(r['human_num']==hn and r['geometry']==geo for r in rows), f'cell{i} 内容不符'
    eps += rows
    wall = max(wall, j['wall_seconds'])
    if sha is None: sha, bb = j['checkpoint_sha256'], j['backbone']
    assert j['checkpoint_sha256'] == sha, '分片 checkpoint 不一致!'
    lat.append((j['policy_latency_ms_median'], len(rows)))
assert len(eps) == 192, len(eps)
med = float(np.average([m for m,_ in lat], weights=[n for _,n in lat]))
json.dump({'checkpoint_sha256': sha, 'backbone': bb, 'episodes': eps,
           'wall_seconds': wall, 'policy_latency_ms_median': med,
           'policy_latency_ms_p95': med, 'sharded': 6,
           'latency_includes_full_80_action_scoring': True}, open(out,'w'), indent=2)
n=len(eps); s=sum(1 for e in eps if e['outcome']=='success'); c=sum(1 for e in eps if e['outcome']=='collision')
print(f'  合并 {out.name}: n={n}  SR={s/n*100:.2f}%  CR={c/n*100:.2f}%  TO={(n-s-c)/n*100:.2f}%  (分片校验通过)')
