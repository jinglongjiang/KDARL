#!/usr/bin/env python3
"""Compact a completed raw acceptance for transfer; do not change source evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    assert data['execution_status']=='PASS'
    data['full_verification_file'] = str(args.input.resolve())
    data['full_verification_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    data['compact_transfer_only'] = True
    data['omitted_from_compact'] = ['all80 root arrays only; actual A0/A1 commands, scores, G, outcomes and provenance retained']
    for field in ('original','native_N10'):
        for root in data.get(field,{}).get('root_details',[]):
            root.pop('all80',None)
    print(json.dumps(data,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
