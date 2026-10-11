"""Instrument the frozen parent entry for lossless boundary checkpoints only."""
import argparse
import copy
import gzip
import inspect
import io
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
import torch


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def rng_state(env):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                case_counter=copy.deepcopy(env.case_counter))


def restore_rng(state, env):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    env.case_counter = copy.deepcopy(state['case_counter'])


def pack_buffer(buffer):
    arrays, attributes = {}, {}
    for name, value in vars(buffer).items():
        if isinstance(value, np.ndarray):
            arrays[name] = torch.from_numpy(value[:buffer.size])
        else:
            attributes[name] = copy.deepcopy(value)
    return dict(arrays=arrays, attributes=attributes)


def restore_buffer(buffer, packed):
    assert buffer.capacity == packed['attributes']['capacity']
    assert buffer.seq_len == packed['attributes']['seq_len']
    for name, value in packed['arrays'].items():
        destination = getattr(buffer, name)
        assert destination.shape[1:] == tuple(value.shape[1:])
        destination[:len(value)] = value.cpu().numpy()
    for name, value in packed['attributes'].items():
        setattr(buffer, name, copy.deepcopy(value))


def archive_load(path):
    with gzip.open(path, 'rb') as stream:
        return torch.load(io.BytesIO(stream.read()), map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--continue-archive')
    parser.add_argument('--pause-il-epoch', type=int)
    parser.add_argument('--pause-rl-episode', type=int)
    parser.add_argument('--cpu-core', type=int)
    args, training_args = parser.parse_known_args()
    root = Path(args.project).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    if args.cpu_core is not None:
        os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    import crowd_nav, crowd_sim, crowd_nav.policy.mamba_rl as policy_module
    for module in (crowd_nav, crowd_sim, policy_module):
        Path(module.__file__).resolve().relative_to(root)
    from crowd_nav import train
    out = Path(training_args[training_args.index('--outdir') + 1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    context = {}
    original_save, original_load = torch.save, train.safe_torch_load
    original_rl, original_il = train._run_rl_phase, train._run_policy_il_pretrain
    recovered = archive_load(args.continue_archive) if args.continue_archive else None
    if recovered:
        checkpoint = out / 'resume-policy.pth'
        original_save(recovered['checkpoint'], checkpoint)
        training_args += ['--resume', str(checkpoint), '--resume-rebuild-il-replay']

    def stopping():
        return time.time() >= args.deadline or (out / 'STOP').exists()

    def resource_status(stage, number, seconds):
        save_json(out / 'phase.json', dict(stage=stage, number=number,
            boundary_seconds=seconds, wall_seconds=time.time() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_max_reserved_bytes=torch.cuda.max_memory_reserved(),
            checkpoint_includes_rng=True, checkpoint_includes_replay=stage == 'RL'))

    def checkpoint_save(payload, path, *a, **kw):
        name = str(path)
        if isinstance(payload, dict) and 'objective' in payload and 'epoch' in payload and name.endswith('.progress'):
            frame = inspect.currentframe().f_back
            assert frame.f_code.co_name == '_run_policy_il_pretrain'
            env = frame.f_locals['explorer'].env
            enriched = dict(payload, overnight_rng=rng_state(env),
                            overnight_seq_len=frame.f_locals['seq_len'])
            original_save(enriched, path, *a, **kw)
            # A separate copy survives the parent's final-IL progress-file removal.
            temporary = out / 'il-resume-state.pth.tmp'
            original_save(enriched, temporary)
            temporary.replace(out / 'il-resume-state.pth')
            epoch = int(payload['epoch'])
            now = time.time()
            resource_status('IL', epoch, now - context.get('last_boundary', started))
            context['last_boundary'] = now
            if stopping() or epoch == args.pause_il_epoch:
                save_json(out / 'PAUSED.json', dict(stage='IL', epoch=epoch, resume='original IL progress plus RNG'))
                raise SystemExit(75)
            return
        original_save(payload, path, *a, **kw)
        if (isinstance(payload, dict) and payload.get('stage') == 'rl_training_sarl'
                and Path(name).name.startswith('rl_model_ep')):
            assert context.get('rl_buffer') is not None
            complete = dict(checkpoint=payload,
                continuation=dict(buffer=pack_buffer(context['rl_buffer']),
                    rng=rng_state(context['env']), stats=copy.deepcopy(vars(context['stats'])),
                    train_step_counter=copy.deepcopy(context['train_step_counter'])),
                checkpoint_kind='episode_boundary_full_resume_v1')
            temporary = out / 'resume-latest.pth.gz.tmp'
            with gzip.open(temporary, 'wb', compresslevel=1) as stream:
                original_save(complete, stream)
            temporary.replace(out / 'resume-latest.pth.gz')
            episode = int(payload['episode'])
            now = time.time()
            resource_status('RL', episode, now - context.get('last_boundary', started))
            context['last_boundary'] = now
            if stopping() or episode == args.pause_rl_episode:
                # Parent would rename the .tmp immediately after this call.
                target = Path(name)
                if target.suffix == '.tmp':
                    target.replace(target.with_suffix(''))
                save_json(out / 'PAUSED.json', dict(stage='RL', episode=episode,
                    resume='resume-latest.pth.gz', replay_rng_environment_counters_saved=True))
                raise SystemExit(75)

    def safe_load(path, *a, **kw):
        payload = original_load(path, *a, **kw)
        if str(path).endswith('.progress') and isinstance(payload, dict) and 'overnight_rng' in payload:
            frame = inspect.currentframe().f_back
            assert frame.f_code.co_name == '_run_policy_il_pretrain'
            assert payload['overnight_seq_len'] == frame.f_locals['seq_len']
            restore_rng(payload['overnight_rng'], frame.f_locals['explorer'].env)
        return payload

    def il_phase(*a, **kw):
        result = original_il(*a, **kw)
        torch.cuda.empty_cache()
        save_json(out / 'il-finished.json', dict(status='COMPLETED', fresh_il=True))
        return result

    def rl_phase(*a, **kw):
        bound = inspect.signature(original_rl).bind(*a, **kw).arguments
        context.update({key: bound[key] for key in ('rl_buffer', 'env', 'stats', 'train_step_counter')})
        resource_status('RL_START', int(bound.get('start_episode', 1)), 0.)
        explorer = bound['explorer']
        run = explorer.run_k_episodes
        pending = [recovered is not None]
        if recovered:
            restore_buffer(bound['rl_buffer'], recovered['continuation']['buffer'])
            vars(bound['stats']).update(copy.deepcopy(recovered['continuation']['stats']))
            bound['train_step_counter'].update(recovered['continuation']['train_step_counter'])

            def run_restored(*ra, **rk):
                phase = ra[1] if len(ra) > 1 else rk.get('phase')
                if pending[0] and phase == 'train':
                    restore_rng(recovered['continuation']['rng'], bound['env'])
                    pending[0] = False
                    save_json(out / 'restored-full-state.json', dict(status='RESTORED',
                        replay_size=bound['rl_buffer'].size, replay_ptr=bound['rl_buffer'].ptr))
                return run(*ra, **rk)
            explorer.run_k_episodes = run_restored
        return original_rl(*a, **kw)

    torch.save = checkpoint_save
    train.safe_torch_load = safe_load
    train._run_policy_il_pretrain = il_phase
    train._run_rl_phase = rl_phase
    sys.argv = [str(root / 'crowd_nav/train.py')] + training_args
    try:
        train.main()
    except SystemExit as error:
        if error.code != 75:
            raise
        raise
    else:
        save_json(out / 'worker-completed.json', dict(status='COMPLETED', wall_seconds=time.time() - started))


if __name__ == '__main__':
    main()
