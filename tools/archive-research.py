"""Export immutable research evidence into Git/LFS without duplicating local weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = Path('/home/abc/workspace')
PROJECT = WORKSPACE / 'nav_data/mamba/camrl/CrowdNav'
BIG = 8 * 1024**2
CHUNK = 512 * 1024**2
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv',
             'triton-cache', 'inductor-cache', 'torchinductor_abc', 'node_modules'}
SKIP_SUFFIXES = ('.pyc', '.pyo', '.pid', '.pid.json', '.lock', '.json.progress',
                 '.cubin', '.ptx', '.llir', '.ttir', '.ttgir')
WEIGHT_SUFFIXES = ('.pth', '.pth.gz', '.pt', '.ckpt', '.safetensors')
BINARY_SUFFIXES = WEIGHT_SUFFIXES + ('.npz', '.pkl', '.pickle', '.jsonl.gz',
                                   '.json.gz', '.log.gz')
SECRET = re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{30,}'
                    rb'|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}'
                    rb'|sk-[A-Za-z0-9]{32,}')
EXPERIMENTS = [
    'kda-selection-20261008', 'kda-history-20261009',
    'kda-history-replication-20261010', 'kda-lookahead-20261009',
    'mamba-ttc5-20261009', 'route-b-20261010', 'scale-consistency-20261011',
    'ac3-selection-20261010', 'ac3-selection-20261010-4090',
    'ttc5-causal-20261011', 'dual-direction-audit-20261011',
    'ab-causal-isolation-20261011', 'value-resolution-preflight-20261011',
    'preference-formation-20261011',
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def eligible(path):
    return (not path.is_symlink() and path.is_file()
            and not set(path.parts) & SKIP_DIRS
            and not path.name.endswith(SKIP_SUFFIXES)
            and '.bak' not in path.name
            and path.name not in ('.env', 'id_rsa', 'id_ed25519', 'credentials.json')
            and not path.name.startswith('.env.'))


def source_files():
    output = {}
    def tree(src, dst):
        assert src.is_dir(), src
        for folder, dirs, files in os.walk(src):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                p = Path(folder) / name
                if eligible(p):
                    output[str(dst / p.relative_to(src))] = p
    # Preserve the existing curated source boundary, not unrelated legacy runs.
    tracked = subprocess.check_output(['git', 'ls-files', '-z', 'CrowdNav'], cwd=REPO).split(b'\0')
    for item in tracked:
        if item:
            dest = Path(os.fsdecode(item))
            p = PROJECT / dest.relative_to('CrowdNav')
            if eligible(p) and 'artifacts' not in dest.parts:
                output[str(dest)] = p
    for p in PROJECT.glob('*.md'):
        output[str(Path('CrowdNav') / p.name)] = p
    tree(PROJECT / 'artifacts', Path('CrowdNav/artifacts'))
    for name in EXPERIMENTS:
        tree(WORKSPACE / name, Path('experiments') / name)
    tree(Path('/home/abc/4090_backup_20261009/root/kda-history-20261009'),
         Path('archives/history-seed419-20261009'))
    tree(Path('/home/abc/temp/cc_6env_20261007'), Path('experiments/timeout-evidence-20261007'))
    dataset = PROJECT / 'data/il_dataset_diverse_v5.0_merged_15000_with_idx.pth'
    assert sha(dataset) == '940ae49af4b641d23b7c8bdf47f548dddce5db128b7ea0beaabe1e897459a8e2'
    output['CrowdNav/data/' + dataset.name] = dataset
    return output


def lfs_object(src, digest, size):
    dest = REPO / '.git/lfs/objects' / digest[:2] / digest[2:4] / digest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try:
            os.link(src, dest)
        except OSError:
            shutil.copyfile(src, dest)
    assert dest.stat().st_size == size
    return ('version https://git-lfs.github.com/spec/v1\n'
            'oid sha256:%s\nsize %d\n' % (digest, size)).encode('ascii')


def store(src, relative, entries, attributes):
    before = src.stat()
    size = before.st_size
    digest = sha(src)
    assert before.st_size == src.stat().st_size and before.st_mtime_ns == src.stat().st_mtime_ns
    dest = REPO / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if size < BIG and src.suffix in ('.py', '.json', '.md', '.ini', '.config', '.txt', '.sh', '.log'):
        assert not SECRET.search(src.read_bytes()), ('credential-like content', str(src))
    entry = dict(source=str(src), path=relative, bytes=size, sha256=digest)
    if size > CHUNK:
        h = hashlib.sha256()
        parts = []
        with src.open('rb') as stream:
            number = 0
            while True:
                data = stream.read(CHUNK)
                if not data:
                    break
                h.update(data)
                part_digest = hashlib.sha256(data).hexdigest()
                name = relative + '.archive-part-%04d' % number
                with tempfile.NamedTemporaryFile(dir=REPO / '.git', delete=False) as temporary:
                    temp = Path(temporary.name)
                    temporary.write(data)
                try:
                    pointer = lfs_object(temp, part_digest, len(data))
                    (REPO / name).write_bytes(pointer)
                finally:
                    temp.unlink()
                parts.append(dict(path=name, bytes=len(data), sha256=part_digest))
                number += 1
        assert h.hexdigest() == digest
        descriptor = relative + '.reassembly.json'
        entry.update(storage='lfs-parts', parts=parts, descriptor=descriptor)
        (REPO / descriptor).write_text(json.dumps(entry, indent=2) + '\n')
        assert not dest.exists(), ('unexpected unsplit destination', relative)
    elif size >= BIG or src.name.endswith(BINARY_SUFFIXES):
        dest.write_bytes(lfs_object(src, digest, size))
        attributes.add(json.dumps('/' + relative, ensure_ascii=False)
                       + ' filter=lfs diff=lfs merge=lfs -text')
        entry['storage'] = 'lfs'
    else:
        shutil.copyfile(src, dest)
        assert sha(dest) == digest
        entry['storage'] = 'git'
    entries.append(entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--backup', type=Path, help='Append a verified local remote-backup snapshot')
    parser.add_argument('--archive-prefix', default='archives/4090-final-20261012')
    args = parser.parse_args()
    previous = None
    if args.backup:
        prefix = Path(args.archive_prefix)
        assert not prefix.is_absolute() and '..' not in prefix.parts
        backup = args.backup.resolve()
        verification = json.loads((backup / 'local-backup-verification.json').read_text())
        cleanup = json.loads((backup / 'cleanup-result.json').read_text())
        assert verification['status'] == cleanup['status'] == 'PASS'
        previous = json.loads((REPO / 'research-archive-manifest.json').read_text())
        sources = {}
        for entry in verification['entries']:
            p = Path(entry['local'])
            p.resolve().relative_to(backup)
            if eligible(p):
                sources[str(prefix / p.relative_to(backup))] = p
        for name in ('backup-and-clean.py', 'protocol.json', 'guard-selftest.json',
                     'remote-manifest-before.json', 'local-backup-verification.json',
                     'deletion-plan.json', 'cleanup-result.json', 'rsync.log'):
            p = backup / name
            assert p.is_file(), p
            sources[str(prefix / name)] = p
    else:
        sources = source_files()
    total = sum(p.stat().st_size for p in sources.values())
    plan = dict(files=len(sources), logical_bytes=total,
        weights=sum(p.name.endswith(WEIGHT_SUFFIXES) for p in sources.values()),
        experiments=EXPERIMENTS, training=False,
        excluded='Runtime caches, credentials, PID/lock files, unrelated Bayesian/recovery projects. Existing historical exports retained.')
    print(json.dumps(plan, indent=2), flush=True)
    if not args.export:
        return
    assert shutil.disk_usage(REPO).free >= 4 * 1024**3
    entries = previous['entries'].copy() if previous else []
    attributes = set()
    if previous:
        tail = (REPO / '.gitattributes').read_text().split('# Large immutable evidence paths (generated)')
        if len(tail) == 2:
            attributes.update(x for x in tail[1].splitlines() if x.strip())
        assert not {str(x) for x in sources} & {x['path'] for x in entries}
    for i, (relative, source) in enumerate(sorted(sources.items()), 1):
        store(source, relative, entries, attributes)
        if i % 250 == 0:
            print(json.dumps(dict(exported=i, total=len(sources))), flush=True)
    attr = REPO / '.gitattributes'
    current = attr.read_text().split('# Large immutable evidence paths (generated)')[0].rstrip()
    attr.write_text(current + '\n# Large immutable evidence paths (generated)\n' + '\n'.join(sorted(attributes)) + '\n')
    if previous:
        plan = dict(previous['plan'], files=len(entries),
            logical_bytes=sum(x['bytes'] for x in entries),
            weights=sum(x['path'].endswith(WEIGHT_SUFFIXES) for x in entries),
            verified_remote_backup_added=str(args.backup))
    manifest = dict(version='20261012-full-evidence-with-weights', plan=plan, entries=entries,
        historical_manifests='snapshot-manifest.json and export-verification.json describe the previous weight-free export, not this update',
        immutable_LFS_cache='Content-addressed local objects may share storage with existing immutable local originals; do not edit archived objects in place.')
    (REPO / 'research-archive-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='EXPORTED', files=len(entries), bytes=total)), flush=True)


if __name__ == '__main__':
    main()
