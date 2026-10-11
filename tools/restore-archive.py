"""Restore one split immutable evidence file, validating every SHA256."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('descriptor', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    record = json.loads(args.descriptor.read_text())
    output = args.output or repo / record['path']
    assert not output.exists(), 'Output already exists; refusing overwrite'
    overall = hashlib.sha256()
    total = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + '.restore-incomplete')
    with temp.open('xb') as dst:
        for part in record['parts']:
            path = repo / part['path']
            path.resolve().relative_to(repo.resolve())
            count = 0
            digest = hashlib.sha256()
            with path.open('rb') as src:
                for block in iter(lambda: src.read(1024**2), b''):
                    dst.write(block)
                    overall.update(block)
                    digest.update(block)
                    count += len(block)
            assert count == part['bytes'] and digest.hexdigest() == part['sha256'], str(path)
            total += count
    assert total == record['bytes'] and overall.hexdigest() == record['sha256']
    temp.rename(output)
    print(json.dumps(dict(status='PASS', output=str(output), bytes=total,
                          sha256=overall.hexdigest())))


if __name__ == '__main__':
    main()
