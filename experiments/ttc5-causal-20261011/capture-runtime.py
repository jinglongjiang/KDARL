"""Capture actual dependency provenance without loading a navigation model."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
sys.dont_write_bytecode=True
import numpy
import torch
import fla
import rvo2


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


base=Path(__file__).resolve().parent
root=Path(fla.__file__).resolve().parent
data=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
          torch=torch.__version__,numpy=numpy.__version__,cuda=torch.version.cuda,
          device=torch.cuda.get_device_name(),fla_file=str(fla.__file__),
          rvo2_file=rvo2.__file__,rvo2_sha256=sha(rvo2.__file__),
          pythonpath=os.environ.get('PYTHONPATH'),
          fla_sources={str(p):sha(p) for p in sorted(root.rglob('*.py'))},
          script_sha256=sha(__file__))
path=base/'work/runtime.json'
assert not path.exists()
path.write_text(json.dumps(data,indent=2))
print(json.dumps(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
                     python=sys.version,torch=torch.__version__,device=data['device'])))
