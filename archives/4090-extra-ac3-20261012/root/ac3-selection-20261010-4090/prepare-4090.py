"""Freeze a path-only migration, rejecting changed methods, weights or project sources."""
import sys
sys.dont_write_bytecode = True
import json
import os
import platform
import shutil
import time
from pathlib import Path
from common import BASE, PROJECT, HISTORY, save, sha

assert not (BASE/'protocol.json').exists()
assert shutil.disk_usage(BASE).free >= 4*1024**3, 'Need >=4GiB free'
source=json.loads((BASE/'inputs/source-protocol.json').read_text())
old=json.loads((BASE/'inputs/source-frozen-manifest.json').read_text())['files']
frozen={}
for item in source['checkpoints']:
    seed=item['seed']
    for key,file in [('weight','rl_model_ep10000.pth'),('config','kda-t24.ini'),
                     ('training_protocol','training-protocol.json')]:
        original=item[key]
        p=BASE/'inputs'/('s%d'%seed)/file
        assert p.is_file() and p.stat().st_size > 0
        assert sha(p)==old[original], ('Changed input',key,seed)
        item[key]=str(p)
        frozen[str(p)]=sha(p)
old_project=Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
source_files=0
for name,digest in old.items():
    p=Path(name)
    if p.is_relative_to(old_project):
        target=PROJECT/p.relative_to(old_project)
        assert target.is_file() and sha(target)==digest, ('Project differs',str(target))
        frozen[str(target)]=digest
        source_files+=1
assert source_files>100
assert sha(HISTORY)==old['/home/abc/workspace/kda-history-20261009/evaluate-history.py']
frozen[str(HISTORY)]=sha(HISTORY)
for name in ('evaluate.py','action-coverage-reference.py'):
    assert sha(BASE/name)==old['/home/abc/workspace/ac3-selection-20261010/'+name], name
source.update(runtime='remote RTX4090; existing Python3.10/torch2.9.1+cu128/FLA; no environment updates',
              migration=dict(authorization='User explicitly requested migration to4090',
                             previous='3060 RESOURCE_LIMIT; no6000episode results',
                             weights_and_methods_unchanged=True, source_project_files=source_files,
                             old_protocol_sha256=sha(BASE/'inputs/source-protocol.json'),
                             time_budget='new authorized4090attempt,7200s from firstcorrectnessworker'),
              created=time.time())
save(BASE/'protocol.json',source)
for p in sorted(BASE.glob('*.py')):
    frozen[str(p)]=sha(p)
for p in sorted((BASE/'inputs').rglob('*.json')):
    frozen[str(p)]=sha(p)
frozen[str(BASE/'protocol.json')]=sha(BASE/'protocol.json')
import torch
import fla
import mamba_ssm
import rvo2
from fla.layers.kda import KimiDeltaAttention
assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
assert torch.ones(3,device='cuda').sum().item()==3
for p in sorted(Path('/root/kda-vl-deps/flash-linear-attention/fla').rglob('*.py')):
    frozen[str(p)]=sha(p)
save(BASE/'environment-manifest.json',dict(python=sys.version,platform=platform.platform(),
    torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
    fla_file=fla.__file__,mamba_file=mamba_ssm.__file__,rvo2_file=rvo2.__file__,
    no_package_changes=True,remote_free_bytes=shutil.disk_usage(BASE).free))
save(BASE/'frozen-manifest.json',dict(files=frozen,checkpoints=source['checkpoints'],
    method_sha256={p:sha(BASE/p) for p in ('evaluate.py','action-coverage-reference.py')},
    original_project_not_modified=True))
print(json.dumps(dict(status='FROZEN',files=len(frozen),source_files=source_files,
                      disk_free_gib=shutil.disk_usage(BASE).free/1024**3)),flush=True)
