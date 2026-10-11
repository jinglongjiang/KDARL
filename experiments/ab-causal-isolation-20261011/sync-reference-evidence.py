"""Copy completed Mamba reference shards with bounded disjoint transfers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
REL = 'reference-mamba-v3/'
REMOTE = 'root@45.126.120.6:/root/ab-causal-isolation-20261011/'


def main():
    folder = BASE / 'execution/reference-evidence-sync'
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'done.json').exists()
    code = subprocess.call(['rsync', '-az', '--exclude=main/', '--exclude=smoke/',
                            REMOTE + REL, str(BASE / REL) + '/'])
    assert code == 0
    jobs = [('main', geo + '-' + condition + '.jsonl.*')
            for geo in ('circle', 'square')
            for condition in ('E1', 'E2', 'E3', 'E4', 'E5', 'E10')]
    jobs += [('smoke', geo + '-*') for geo in ('circle', 'square')]

    def copy(job):
        phase, pattern = job
        rel = REL + 'work/' + phase + '/model/mamba-reference/'
        destination = BASE / rel
        destination.mkdir(parents=True, exist_ok=True)
        started = time.time()
        code = subprocess.call(['rsync', '-az', '--include=' + pattern,
                                '--exclude=*', REMOTE + rel, str(destination) + '/'])
        return dict(phase=phase, pattern=pattern, exit_code=code,
                    wall_seconds=time.time() - started)

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(copy, job) for job in jobs]):
            row = future.result()
            results.append(row)
            with (folder / 'journal.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            assert row['exit_code'] == 0
    (folder / 'done.json').write_text(json.dumps(dict(status='PASS', jobs=results,
        scope='Only immutable completed reference outputs; no experiment, rerun, source edit or deletion.')))


if __name__ == '__main__':
    main()
