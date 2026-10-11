"""Make retained backups and LFS objects independent of mutable source files."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


BASE = Path(__file__).resolve().parent
CACHE = Path('/home/abc/workspace/KDARL/.git/lfs/objects')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    sources = {}
    for path in CACHE.glob('*/*/*'):
        if path.is_file() and len(path.name) == 64:
            sources[path] = path.name
    for folder in (BASE, BASE / 'extra-ac3'):
        record = json.loads((folder / 'local-backup-verification.json').read_text())
        assert record['status'] == 'PASS' and record['entries']
        for entry in record['entries']:
            sources[Path(entry['local'])] = entry['sha256']
    required = sum(p.stat().st_size for p in sources if p.stat().st_nlink > 1)
    assert shutil.disk_usage(BASE).free > required + 2 * 1024**3
    copied = []
    for i, (path, digest) in enumerate(sources.items(), 1):
        assert path.is_file() and not path.is_symlink()
        if path.stat().st_nlink > 1:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                temporary = Path(tmp.name)
            try:
                shutil.copy2(path, temporary)
                assert sha(temporary) == digest, str(path)
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            copied.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))
        assert path.stat().st_nlink == 1, ('alias remains', str(path))
        if i % 1000 == 0:
            print(json.dumps(dict(checked=i, total=len(sources))), flush=True)
    result = dict(status='PASS', paths_checked=len(sources), independent_copies=len(copied),
        copied_bytes=sum(e['bytes'] for e in copied), copied=copied,
        all_retained_backup_and_lfs_paths_nlink=1,
        content_changed=False, original_source_files_modified=False)
    (BASE / 'storage-isolation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'copied'}), flush=True)


if __name__ == '__main__':
    main()
