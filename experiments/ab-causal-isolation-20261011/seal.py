"""Hash immutable inputs and preserve auditable first-round execution inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--causal',type=Path,default=Path('/home/abc/workspace/ttc5-causal-20261011'))
    p.add_argument('--original',type=Path,default=Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav'))
    p.add_argument('--phase',choices=['start','end'],required=True)
    a=p.parse_args()
    old=json.loads((a.causal/'manifest.json').read_text())
    mismatches=[]; original={}
    for rel,expected in old['project_files'].items():
        path=a.original/rel
        actual=sha(path) if path.is_file() else None
        original[rel]=actual
        if not actual or actual!=expected:mismatches.append(dict(path=str(path),expected=expected,actual=actual))
    assert original, 'FAIL empty source inventory'
    weights={}
    for model in old['protocol']['models']:
        for field in ('weight','config'):
            path=a.causal/'inputs'/model['id']/('rl_model_ep10000.pth' if field=='weight' else 'config.ini')
            actual=sha(path) if path.is_file() else None
            expected=model[field+'_sha256']
            weights[str(path)]=actual
            if not actual or actual!=expected:mismatches.append(dict(path=str(path),expected=expected,actual=actual))
    files={}
    for root in [a.workspace,a.workspace/'a',a.workspace/'b']:
        for path in sorted(root.glob('*')):
            if path.is_file() and path.suffix in ('.py','.json','.md') and path.name not in ('frozen-manifest.json','end-verification.json'):
                files[str(path.relative_to(a.workspace))]=sha(path)
    source={}
    for name in ('engine','a/project','b/project'):
        path=a.workspace/name/'crowd_nav/policy/mamba_rl.py'
        if path.is_file():source[name]=sha(path)
    result=dict(phase=a.phase,checked_at=time.time(),status='PASS' if not mismatches else 'FAIL',
        original_files=original,models_and_configs=weights,artifact_sources=files,
        isolated_policy_sources=source,mismatches=mismatches,
        commits=dict(parent='cd8b793',history='184f6f0',terminal='9f06223'),
        scope='Hashes verify unchanged originals and known inputs; do not certify unexecuted experiments or causal claims.')
    output=a.workspace/('frozen-manifest.json' if a.phase=='start' else 'end-verification.json')
    if a.phase=='start':assert not output.exists(), 'Do not overwrite start inventory'
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],original_files=len(original),weight_config_files=len(weights),mismatches=mismatches)))
    assert not mismatches
    if a.phase=='end':
        start=json.loads((a.workspace/'frozen-manifest.json').read_text())
        assert start['original_files']==original and start['models_and_configs']==weights


if __name__=='__main__':main()
