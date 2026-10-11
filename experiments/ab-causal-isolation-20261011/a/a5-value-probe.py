"""Five-person frozen deployment-MC calibration. Not a training-causation experiment."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run_a',HERE/'run-a.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def main(args):
    if args.device.startswith('cuda') and not args.gpu_authorized:
        raise SystemExit('GPU requires main scheduling and --gpu-authorized')
    assert args.cell in (0,3) and 1<=args.steps<=500
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(question='Does original five-person V retain deployment-continuation target signal recoverable by a matched linear readout?',
        GO='Both model seeds show >=10% held-out case-MSE reduction without fitted-vs-label variance collapse; supports only target calibration follow-up.',
        KILL='No consistent held-out calibration gain: stop continuation-only candidate for this representation/budget.',
        budget='Cases 90000..90029, first <=64 steps, stride8, <=240 frozen windows/model; linear257 parameters, 300 full-batch CPU steps.',
        controls='Original checkpoint and real TTC5 current-state histories, no synthetic CV next appended; baseline rewards and complete deployment continuation.',
        split='case%5 train0/1/2 dev3 test4 across seeds; no dev tuning.',
        missing='Matched training-behavior continuation returns from these identical roots unavailable; cannot establish train/deploy causal mismatch.',
        training_continuation_status='INCOMPLETE',models=args.models,cell=args.cell,steps=args.steps)
    A.save(args.output/'a5-protocol.json',protocol)
    ref=A.module('a5_reference',args.data/'reference.py');ref.pin(args.project)
    from crowd_nav.train import load_config
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    torch.set_num_threads(1)
    manifest=json.loads((args.data/'manifest.json').read_text())
    results=[]; started=time.time()
    for model in args.models:
        item=next(x for x in manifest['protocol']['models'] if x['id']==model)
        weight=args.inputs/model/'rl_model_ep10000.pth'; config=args.inputs/model/'config.ini'
        assert A.sha(weight)==item['weight_sha256'] and A.sha(config)==item['config_sha256']
        torch.manual_seed(419)
        net=MambaRLPolicy(load_config(str(config)),device=args.device)
        payload=torch.load(weight,map_location='cpu',weights_only=False)
        net.load_state_dict({k[10:] if k.startswith('_orig_mod.') else k:v for k,v in payload['policy_state'].items()},strict=True)
        net.eval()
        examples=[]; zs=[]
        with torch.no_grad():
            for ep in A.rows(args.data/'work'/f'baseline-{model}-cell{args.cell}.jsonl.gz'):
                if not 90000<=ep['case']<90030:continue
                tokens=[np.asarray(s['current_token'],dtype=np.float32) for s in ep['steps']]
                for k in range(0,min(len(tokens),64),8):
                    prefix=tokens[max(0,k-23):k+1]
                    window=np.asarray([prefix[0]]*(24-len(prefix))+prefix,dtype=np.float32)
                    t=torch.from_numpy(window[None]).to(args.device)
                    z=net.temporal_encoder(net.spatial_encoder(t))[:,-1]
                    value=float(net.value_head(z).item())
                    target=sum(.99**j*s['reward'] for j,s in enumerate(ep['steps'][k:]))
                    examples.append(dict(case=ep['case'],step=k,split=A.split(ep['case']),V=value,G=target,window_sha256=A.digest(window)))
                    zs.append(z.cpu().numpy()[0])
        assert len({e['case'] for e in examples})==30
        X=torch.tensor(np.asarray(zs));Y=torch.tensor([e['G'] for e in examples],dtype=torch.float32)
        masks={s:torch.tensor([e['split']==s for e in examples]) for s in ('train','dev','test')}
        head=torch.nn.Linear(256,1)
        head.load_state_dict({k:v.cpu() for k,v in net.value_head.state_dict().items()})
        optimizer=torch.optim.Adam(head.parameters(),lr=.001)
        for _ in range(args.steps):
            optimizer.zero_grad(); pred=head(X[masks['train']]).squeeze(-1)
            loss=torch.mean((pred-Y[masks['train']])**2);loss.backward();optimizer.step()
        with torch.no_grad():fitted=head(X).squeeze(-1).numpy()
        for e,v in zip(examples,fitted):e['MC_readout']=float(v)
        metrics=[]
        train_constant=float(Y[masks['train']].mean())
        for partition in masks:
            subset=[e for e in examples if e['split']==partition]
            for arm in ('V','MC_readout','train_mean'):
                case_errors=collections.defaultdict(list)
                for e in subset:
                    value=train_constant if arm=='train_mean' else e[arm]
                    case_errors[e['case']].append((value-e['G'])**2)
                values=np.asarray([train_constant if arm=='train_mean' else e[arm] for e in subset])
                labels=np.asarray([e['G'] for e in subset])
                metrics.append(dict(split=partition,arm=arm,cases=len(case_errors),windows=len(subset),
                    case_mean_mse=float(np.mean([np.mean(v) for v in case_errors.values()])),
                    prediction_std=float(values.std()),target_std=float(labels.std()),
                    correlation=float(np.corrcoef(values,labels)[0,1]) if values.std()>1e-10 and labels.std()>1e-10 else None))
        np.savez_compressed(args.output/(model+'-a5-latents.npz'),latent256=X.numpy(),G=Y.numpy(),case=np.asarray([e['case'] for e in examples]),step=np.asarray([e['step'] for e in examples]))
        A.save(args.output/(model+'-a5-rows.json'),examples)
        torch.save(head.state_dict(),args.output/(model+'-a5-head.pth'))
        results.append(dict(model=model,weight_sha256=A.sha(weight),config_sha256=A.sha(config),
            baseline_sha256=A.sha(args.data/'work'/f'baseline-{model}-cell{args.cell}.jsonl.gz'),metrics=metrics))
        print(json.dumps(dict(model=model,windows=len(examples),wall_seconds=time.time()-started)),flush=True)
    A.save(args.output/'a5-summary.json',dict(status='INCOMPLETE',reason='Deployment readout completed; matched training-continuation labels absent.',
        protocol=protocol,results=results,wall_seconds=time.time()-started,method_level_progress=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=A.DEFAULT_DATA);p.add_argument('--inputs',type=Path,default=A.DEFAULT_DATA/'inputs')
    p.add_argument('--project',type=Path,default=HERE/'project');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['gru-s419-t24','gru-s443-t24'])
    p.add_argument('--cell',type=int,default=0);p.add_argument('--steps',type=int,default=300)
    p.add_argument('--device',default='cpu');p.add_argument('--gpu-authorized',action='store_true')
    main(p.parse_args())
