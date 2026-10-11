"""Verify Git/LFS archive content against the captured source hashes."""
import hashlib
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((REPO / 'research-archive-manifest.json').read_text())
    entries = manifest['entries']
    assert entries
    staged = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).split(b'\0'))
    checked = set()
    counts = {'git': 0, 'lfs': 0, 'lfs-parts': 0}
    for i, entry in enumerate(entries, 1):
        assert entry['sha256'] and entry['bytes'] >= 0
        counts[entry['storage']] += 1
        if entry['storage'] == 'git':
            assert entry['path'].encode() in staged, ('untracked', entry['path'])
            path = REPO / entry['path']
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        else:
            parts = entry.get('parts', [entry])
            whole = hashlib.sha256() if entry['storage'] == 'lfs-parts' else None
            if whole:
                assert entry['descriptor'].encode() in staged
            for part in parts:
                assert part['path'].encode() in staged, ('untracked', part['path'])
                digest = part['sha256']
                obj = REPO / '.git/lfs/objects' / digest[:2] / digest[2:4] / digest
                assert obj.stat().st_size == part['bytes']
                if digest not in checked:
                    assert sha(obj) == digest
                    checked.add(digest)
                if whole:
                    with obj.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024**2), b''):
                            whole.update(block)
                pointer = (REPO / part['path']).read_text()
                assert 'oid sha256:' + digest in pointer and 'size ' + str(part['bytes']) in pointer
                attrs = subprocess.check_output(['git', 'check-attr', 'filter', '--', part['path']], cwd=REPO)
                assert attrs.rstrip().endswith(b': lfs'), ('missing LFS filter', part['path'])
            if whole:
                assert whole.hexdigest() == entry['sha256']
        if i % 1000 == 0:
            print(json.dumps(dict(verified=i, total=len(entries))), flush=True)
    result = dict(status='PASS', files=len(entries), storage=counts,
        unique_lfs_objects=len(checked), logical_bytes=sum(e['bytes'] for e in entries),
        source_manifest_sha256=sha(REPO / 'research-archive-manifest.json'),
        training_started=False)
    (REPO / 'research-archive-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
