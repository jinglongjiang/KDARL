"""Replay one published intervention without rerunning the discovery pipeline."""
import sys
sys.dont_write_bytecode = True
import argparse
import gzip
import json
from pathlib import Path
import torch
import audit as A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--cell', type=int, required=True)
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--label', choices=('A0','A1','A2','A3'), default='A1')
    parser.add_argument('--project', default=A.P['project'])
    parser.add_argument('--inputs', type=Path, default=A.BASE/'inputs')
    parser.add_argument('--package', type=Path, default=A.BASE/'root_cause_data.jsonl.gz')
    args = parser.parse_args()
    assert not 92000 <= args.case <= 92499
    project = Path(args.project).resolve()
    assert A.MAN['project_files']
    assert all(A.REF.sha(project/rel)==expected for rel,expected in A.MAN['project_files'].items())
    A.P['project'] = str(project)
    item = next(m for m in A.P['models'] if m['id']==args.model)
    item['weight'] = str(args.inputs/args.model/'rl_model_ep10000.pth')
    item['config'] = str(args.inputs/args.model/'config.ini')
    assert A.REF.sha(item['weight']) == item['weight_sha256']
    record = reference = None
    sources = [args.package] if args.package.exists() else [
        A.BASE/'work'/('branches-%s-cell%d.jsonl.gz'%(args.model,args.cell)),
        A.BASE/'work'/('baseline-%s-cell%d.jsonl.gz'%(args.model,args.cell))]
    for source in sources:
        for row in A.rows(source):
            if 'record_type' not in row:
                row['record_type']='baseline_episode'
            if row['record_type']=='counterfactual':
                root = row['root']
                if (root['model'],root['cell'],root['case'])==(args.model,args.cell,args.case):
                    record=row
            elif row['record_type']=='baseline_episode':
                if (row['model'],row['cell'],row['case'])==(args.model,args.cell,args.case):
                    reference=row
            if record is not None and reference is not None:
                break
        if record is not None and reference is not None:
            break
    assert record is not None and reference is not None, 'Case has no published root'
    root = record['root']
    expected = next(b for b in record['branches'] if args.label in b['labels'])
    _,cfg,net = A.load(args.model)
    with torch.no_grad():
        reproduced = A.rollout(cfg,net,args.model,args.cell,args.case,reference=reference,
                               root=root,selected=root['labels'][args.label])
    for key in ('return','root_return','outcome','time','trajectory_sha256','selection_trace_sha256'):
        assert reproduced[key]==expected[key], (key,reproduced[key],expected[key])
    assert A.REF.sha(item['weight'])==item['weight_sha256']
    print(json.dumps(dict(status='PASS',model=args.model,cell=args.cell,case=args.case,
                          root_time=root['step']*.25,label=args.label,
                          candidate=root['labels'][args.label],
                          outcome=reproduced['outcome'],root_return=reproduced['root_return'],
                          trajectory_sha256=reproduced['trajectory_sha256']),indent=2))


if __name__=='__main__':main()
