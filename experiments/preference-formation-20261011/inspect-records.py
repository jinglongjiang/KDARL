"""Read nested historical recovery state and event tags without training."""
import gzip
import importlib.util
import json
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('formation_prepare', Path(__file__).with_name('prepare.py'))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def main():
    torch.set_num_threads(1)
    result = []
    for seed, run in prep.RUNS.items():
        path = run / 'resume-latest.pth.gz'
        with gzip.open(path, 'rb') as f:
            obj = torch.load(f, map_location='cpu', weights_only=False)
        checkpoint = obj['checkpoint']
        cont = obj['continuation']
        buffer = cont['buffer']
        row = dict(seed=seed, source=str(path), sha256=prep.sha(path),
            episode=checkpoint['episode'], continuation_keys=list(cont),
            checkpoint_keys=list(checkpoint), buffer_keys=list(buffer),
            attributes={k:v for k,v in buffer['attributes'].items() if isinstance(v,(str,int,float,bool,type(None)))},
            arrays={k:dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in buffer['arrays'].items()},
            historical_batch_sequence_present=False)
        del obj, checkpoint, cont, buffer
        il_path = run / 'il-resume-state.pth'
        il = torch.load(il_path, map_location='cpu', weights_only=False)
        row['IL'] = dict(epoch=il['epoch'], best_epoch=il['best_epoch'],
            available='Final saved epoch plus retained best_state, not per-epoch series',
            sha256=prep.sha(il_path))
        del il
        paths = sorted(p for p in run.rglob('*') if p.is_file())
        row['all_files'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=prep.sha(p)) for p in paths]
        events = list((run / 'tensorboard_logs').glob('*'))
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        row['tensorboard'] = []
        for event in events:
            acc = EventAccumulator(str(event), size_guidance={'scalars': 0})
            acc.Reload()
            row['tensorboard'].append(dict(path=str(event), tags=acc.Tags(),
                scalars={tag:dict(count=len(acc.Scalars(tag)),
                    first_step=acc.Scalars(tag)[0].step, last_step=acc.Scalars(tag)[-1].step)
                    for tag in acc.Tags()['scalars']}))
        result.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('all_files','tensorboard')},ensure_ascii=False),flush=True)
    prep.save('record-inventory.json', dict(status='PASS', records=result,
        missing='Per-epoch IL series, intermediate RL replay snapshots and actual sampled-batch sequence. '
                'Intermediate Adam moments ARE present; final replay/RNG ARE present.'))


if __name__ == '__main__':
    main()
