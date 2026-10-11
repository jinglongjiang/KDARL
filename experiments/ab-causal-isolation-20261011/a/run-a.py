"""A-direction frozen-root diagnostics. No IL/RL; no automatic GPU scheduling."""
import argparse
import ast
import collections
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path('/home/abc/workspace/ttc5-causal-20261011')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def digest(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def split(case):
    return 'train' if case % 5 < 3 else 'dev' if case % 5 == 3 else 'test'


def snapshot_method(project, class_name, method_name):
    source = subprocess.check_output(['git', '-C', str(project), 'show', 'cd8b793:crowd_nav/policy/mamba_rl.py'], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    import torch
    import numpy as np
    namespace = {'torch': torch,'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<snapshot-spatial-forward>', 'exec'), namespace)
    return namespace[method_name]


def extract(args):
    if args.device.startswith('cuda') and not args.gpu_authorized:
        raise SystemExit('GPU blocked: main Agent must schedule and supply --gpu-authorized.')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    ref = module('a_reference', args.data / 'reference.py')
    ref.pin(args.project)
    from crowd_nav.train import load_config
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized as old_tokens
    from crowd_sim.envs.utils.state import JointState, FullState, ObservableState
    from crowd_sim.envs.utils.action import ActionXY
    contract = module('a_all_n', HERE / 'all-n-contract.py')
    all_tokens, expanded_source = contract.extend_converter(old_tokens)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    item = next(x for x in manifest['protocol']['models'] if x['id'] == args.model)
    weight = args.inputs / args.model / 'rl_model_ep10000.pth'
    config = args.inputs / args.model / 'config.ini'
    assert sha(weight) == item['weight_sha256'] and sha(config) == item['config_sha256']
    cfg = load_config(str(config))
    torch.manual_seed(419)
    net = MambaRLPolicy(cfg, device=args.device)
    payload = torch.load(weight, map_location='cpu', weights_only=False)
    assert payload['episode'] == 10000
    weights = {k[10:] if k.startswith('_orig_mod.') else k: v for k,v in payload['policy_state'].items()}
    net.load_state_dict(weights, strict=True)
    net.eval(); net.build_action_space(1.); net.set_phase('test'); net.set_training_mode('rl')
    net.multiagent_training = True; net.use_sarl_predict = True; net.epsilon = 0.
    assert net.seq_len == 24 and net.gamma == .99 and net.test_action_smoothing == .3
    assert net.test_min_clearance == .2 and net.test_risk_lambda == .8
    # Archival features intentionally use the original score, NOT the repaired closed-loop baseline.
    net.score_sarl_candidates=snapshot_method(args.project,'MambaRLPolicy','score_sarl_candidates').__get__(net)
    ref.install_selection(net, 'ttc5')
    frozen_spatial = copy.deepcopy(net.spatial_encoder)
    frozen_spatial.forward = snapshot_method(args.project,'EnhancedSpatialEncoder','forward').__get__(frozen_spatial)
    roots = [r for r in json.loads((args.data / 'work/roots.json').read_text())['roots'] if r['model'] == args.model and r['cell'] in args.cells]
    roots.sort(key=lambda r: (r['cell'], r['case'], r['step'], r['root_id']))
    if args.limit:
        roots = roots[:args.limit]
    wanted = {r['root_id'] for r in roots}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(question='Can a matched linear readout recover frozen-policy branch preferences, and does full-N add recoverable signal?',
        GO='On held-out case groups both seeds improve root-averaged pair accuracy >=0.05 over P1; advance only to main-authorized finite closed-loop.',
        KILL='No consistent held-out improvement or success-control harm: stop this readout candidate; no round-two training.',
        budget=f'Fixed roots sorted cell/case/step; limit={args.limit or "all"}; extraction only; readout <=500 CPU optimizer steps.',
        controls='Same old weights, batch80, commands/rewards/safety, original tokens/windows bitwise, N5 original-vs-expanded bitwise.',
        data_scope='20-person complete G is in-distribution diagnostic labels, NOT OOD evidence.',
        corrections='Original archival score bound from cd8b793. Public repairs may be present in worktree but are intentionally NOT applied to feature scores. This is NOT a new closed-loop baseline.',
        model=args.model, cells=args.cells, root_ids=[r['root_id'] for r in roots],
        config_sha256=sha(config), weight_sha256=sha(weight), roots_sha256=sha(args.data/'work/roots.json'),
        script_sha256=sha(__file__), contract_sha256=sha(HERE/'all-n-contract.py'),
        git_head=subprocess.check_output(['git','-C',str(args.project),'rev-parse','HEAD'],text=True).strip(),
        device=args.device, torch=torch.__version__, expanded_contract=expanded_source)
    protocol['source_sha256']={str(args.project/'crowd_nav/policy/mamba_rl.py'):sha(args.project/'crowd_nav/policy/mamba_rl.py'),
        str(args.project/'crowd_nav/contracts.py'):sha(args.project/'crowd_nav/contracts.py')}
    save(args.output/'extract-protocol.json', protocol)
    start = time.time(); completed = []; max_saved_delta = 0.
    oracle={r['root_id']:r for r in rows(args.data/'work/real-history-oracle.jsonl.gz') if r['root_id'] in wanted}

    def encode(window):
        t = torch.as_tensor(window, device=args.device)
        z = net.temporal_encoder(net.spatial_encoder(t))[:, -1]
        assert z.shape == (80, 256)
        return z.cpu().numpy()

    with torch.no_grad():
        for cell in args.cells:
            cell_roots = [r for r in roots if r['cell'] == cell]
            if not cell_roots:
                continue
            cases = {r['case'] for r in cell_roots}
            baseline_path = args.data/'work'/f'baseline-{args.model}-cell{cell}.jsonl.gz'
            refs = {e['case']:e for e in rows(baseline_path) if e['case'] in cases}
            branch_path = args.data/'work'/f'branches-{args.model}-cell{cell}.jsonl.gz'
            protocol['source_sha256'].update({str(baseline_path):sha(baseline_path),str(branch_path):sha(branch_path)})
            save(args.output/'extract-protocol.json',protocol)
            for row in rows(branch_path):
                if row.get('record_type') != 'counterfactual' or row['root']['root_id'] not in wanted:
                    continue
                r=row['root']; rid=r['root_id']; k=r['step']; ep=refs[r['case']]
                assert r == next(x for x in cell_roots if x['root_id'] == rid)
                hist=[]; full=[]
                for s in ep['steps'][max(0,k-24):k+1]:
                    selected = [s['humans'][i] for i in s['ids']]
                    v=contract.state_vector(s['robot'], selected)
                    token=old_tokens(v[None])[0]
                    assert np.array_equal(token,np.asarray(s['current_token'],dtype=np.float32)), (rid,s['step'],'old frame')
                    assert np.array_equal(token,all_tokens(v[None],5)[0]), 'N5 token gate'
                    hist.append(token)
                    full.append(all_tokens(contract.state_vector(s['robot'],s['humans'])[None],len(s['humans']))[0])
                net.reset_episode_stats()
                net._history=collections.deque(hist[:-1],maxlen=24)
                net._last_action=ActionXY(*ep['steps'][k-1]['command']) if k else None
                s=r['snapshot']
                state=JointState(FullState(*s['robot']),[ObservableState(*h) for h in s['humans']])
                out=net.score_sarl_candidates(state)
                old_window=out['windows'].cpu().numpy()
                assert digest(old_window)==s['windows_sha256'], (rid,'archival windows bitwise')
                assert [[c.vx,c.vy] for c in out['commands']]==s['commands']
                assert np.array_equal(old_window,contract.candidate_windows(hist,old_window[:,-1])), 'window contract'
                original_s=frozen_spatial(out['windows'])
                expanded_s=net.spatial_encoder(out['windows'])
                assert torch.equal(original_s,expanded_s), 'N5 spatial gate'
                original_z=net.temporal_encoder(original_s)[:,-1]
                expanded_z=net.temporal_encoder(expanded_s)[:,-1]
                assert torch.equal(original_z,expanded_z), 'N5 latent gate'
                assert torch.equal(net.value_head(original_z),net.value_head(expanded_z)), 'N5 V gate'
                saved=np.asarray(s['values']); now=out['values'].cpu().numpy()
                delta=float(np.max(abs(saved-now))); max_saved_delta=max(max_saved_delta,delta)
                if args.device.startswith('cuda'):
                    assert np.array_equal(saved,now), (rid,'GPU saved values must be bitwise',delta)
                else:
                    assert delta<2e-5, (rid,'CPU/GPU saved numeric tolerance',delta)
                bs={b['candidate_index']:b for b in row['branches']}
                a0=next(b for b in bs.values() if 'A0' in b['labels'])
                actual_h=a0['first_step']['true_next_humans']
                assert all(b['first_step']['true_next_humans']==actual_h for b in bs.values())
                cv_h=[net.propagate(h,ActionXY(h.vx,h.vy)) for h in state.human_states]
                cv_h=[[h.px,h.py,h.vx,h.vy,h.radius] for h in cv_h]
                robot_next=[]; actual5=[]; predicted_full=[]; actual_full=[]
                N=len(actual_h)
                for c in out['commands']:
                    rob=net.propagate(state.self_state,c)
                    ra=[rob.px,rob.py,rob.vx,rob.vy,rob.radius,rob.gx,rob.gy,rob.v_pref,rob.theta]
                    robot_next.append(ra)
                    ids=ref.select_indices(rob,[ObservableState(*h) for h in actual_h],'ttc5')
                    actual5.append(contract.state_vector(ra,[actual_h[i] for i in ids]))
                    predicted_full.append(contract.state_vector(ra,cv_h))
                    actual_full.append(contract.state_vector(ra,actual_h))
                windows={'pred5':old_window,
                    'actual5':contract.candidate_windows(hist,old_tokens(np.asarray(actual5))),
                    'predN':contract.candidate_windows(full,all_tokens(np.asarray(predicted_full),N)),
                    'actualN':contract.candidate_windows(full,all_tokens(np.asarray(actual_full),N))}
                assert digest(windows['actual5'])==oracle[rid]['actual_window_sha256'], (rid,'actualTTC archival window bitwise')
                if N==5:
                    for left,right in (('pred5','predN'),('actual5','actualN')):
                        assert np.array_equal(windows[left],windows[right]), 'N5 full windows gate'
                arrays={}; ids=sorted(bs)
                for name,w in windows.items():
                    arrays['latent_'+name]=encode(w)
                    arrays['windows_'+name]=w[ids]
                if N==5:
                    assert np.array_equal(arrays['latent_pred5'],arrays['latent_predN'])
                    assert np.array_equal(arrays['latent_actual5'],arrays['latent_actualN'])
                arrays.update(indices=np.asarray(ids), G=np.asarray([bs[i]['root_return'] for i in ids]),
                    reward=np.asarray([bs[i]['first_step']['reward'] for i in ids]),
                    terminal=np.asarray([bs[i]['first_step']['terminal'] for i in ids]),
                    qpred=np.asarray(s['raw_scores']), values=np.asarray(s['values']),
                    commands=np.asarray(s['commands']), clearance=np.asarray(s['clearance']), scores=np.asarray(s['scores']),
                    original_head_weight=net.value_head.weight.cpu().numpy(), original_head_bias=net.value_head.bias.cpu().numpy())
                meta=dict(root_id=rid,case=r['case'],cell=cell,model=args.model,split=split(r['case']),
                    step=k,type=r['type'],a0=a0['candidate_index'],N=N,N5_bitwise=(N==5),
                    original_windows_bitwise=True,spatial_N5_compatibility_bitwise=True,
                    saved_values_bitwise=bool(np.array_equal(saved,now)),saved_values_max_error=delta,
                    branch_source=str(branch_path),baseline_source=str(baseline_path),labels={str(i):bs[i]['labels'] for i in ids})
                np.savez_compressed(args.output/f'root-{rid}.npz',**arrays)
                save(args.output/f'root-{rid}.json',meta)
                completed.append(rid)
                save(args.output/'extract-progress.json',dict(completed=completed,wall_seconds=time.time()-start))
                print(json.dumps(dict(root_id=rid,N=N,split=meta['split'],wall_seconds=time.time()-start)),flush=True)
    assert set(completed)==wanted
    save(args.output/'extract-done.json',dict(status='PASS',roots=len(completed),wall_seconds=time.time()-start,
        max_saved_value_error=max_saved_delta,GPU_archival_bitwise=args.device.startswith('cuda'),
        N5_full_crowd_roots=sum(json.loads(p.read_text())['N5_bitwise'] for p in args.output.glob('root-*.json'))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('a2'); a.add_argument('--data',type=Path,default=DEFAULT_DATA/'work'); a.add_argument('--output',type=Path,default=HERE/'artifacts')
    a=sub.add_parser('extract'); a.add_argument('--data',type=Path,default=DEFAULT_DATA); a.add_argument('--inputs',type=Path,default=DEFAULT_DATA/'inputs')
    a.add_argument('--project',type=Path,default=HERE/'project'); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--model',choices=['kda-s419-t24','kda-s443-t24','gru-s419-t24','gru-s443-t24'],default='kda-s419-t24')
    a.add_argument('--cells',type=int,nargs='+',default=[5]); a.add_argument('--limit',type=int,default=0)
    a.add_argument('--device',default='cpu'); a.add_argument('--gpu-authorized',action='store_true')
    a=sub.add_parser('readout'); a.add_argument('--features',type=Path,nargs='+',required=True); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--steps',type=int,default=300); a.add_argument('--seed',type=int,default=20261011)
    args=p.parse_args()
    if args.command=='a2':
        module('a2',HERE/'a2-decompose.py').main(args.data,args.output)
    elif args.command=='extract':
        extract(args)
    else:
        module('a_readout',HERE/'readout-a.py').main(args)


if __name__=='__main__':
    main()
