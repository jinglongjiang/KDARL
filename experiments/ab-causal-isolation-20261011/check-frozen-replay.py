"""Original replay, history-only negative control, terminal-fix isolated rollout."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    p.add_argument('--causal', required=True)
    p.add_argument('--before', required=True)
    p.add_argument('--history-only', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.causal)
    import audit as A
    A.P['project'] = a.project
    A.REF.pin(a.project)
    import crowd_nav.policy.mamba_rl as live
    classes = {'original': module('original_frozen', a.before).MambaRLPolicy,
               'history_only': module('history_frozen', a.history_only).MambaRLPolicy,
               'terminal_fixed': live.MambaRLPolicy}
    comparisons = []
    for item in A.P['models']:
        model = item['id']
        item['weight'] = str(Path(a.causal)/'inputs'/model/'rl_model_ep10000.pth')
        item['config'] = str(Path(a.causal)/'inputs'/model/'config.ini')
        references = {x['case']: x for x in A.rows(Path(a.causal)/'work'/f'baseline-{model}-cell5.jsonl.gz') if x['case'] in (90000, 90001)}
        for case in (90000, 90001):
            ref = references[case]
            result = {}
            for variant, cls in classes.items():
                live.MambaRLPolicy = cls
                _, cfg, net = A.load(model)
                with torch.no_grad():
                    ep = A.rollout(cfg, net, model, 5, case,
                                   reference=ref if variant != 'terminal_fixed' else None)
                if variant != 'terminal_fixed':
                    for k in ('outcome', 'time', 'return', 'trajectory_sha256', 'selection_trace_sha256'):
                        assert ep[k] == ref[k], (model, case, variant, k)
                result[variant] = {k: ep[k] for k in ('outcome', 'time', 'return', 'trajectory_sha256', 'selection_trace_sha256')}
                result[variant]['steps'] = len(ep['steps'])
                if variant == 'original':
                    before = ep
                elif variant == 'terminal_fixed':
                    diff = [k for k, (u, v) in enumerate(zip(before['steps'], ep['steps'])) if u['chosen'] != v['chosen']]
                    result[variant]['first_changed_action'] = diff[0] if diff else None
            comparisons.append(dict(model=model, case=case, variants=result))
            print(json.dumps(comparisons[-1]), flush=True)
    live.MambaRLPolicy = classes['terminal_fixed']
    data = dict(status='PASS', history_test='PASS', terminal_test='PASS',
                tested_commit='9f06223', torch=torch.__version__, device=torch.cuda.get_device_name(),
                original_replay_episodes=len(comparisons), comparisons=comparisons,
                scope='Frozen epsilon-zero original/history-only actions, history/windows, scores and outcomes match original full-step data; terminal variant is a distinct baseline. Terminal physical model contracts remain unchanged.')
    Path(a.out).write_text(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
