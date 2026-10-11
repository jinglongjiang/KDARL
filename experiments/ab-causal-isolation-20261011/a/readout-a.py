"""Fixed-capacity, fixed-label-budget frozen-latent CPU probes, not policy training."""
import collections
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch


def main(args):
    assert 1 <= args.steps <= 500
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    groups=collections.defaultdict(list)
    for folder in args.features:
        assert (folder/'extract-done.json').exists(), f'Incomplete extraction: {folder}'
        for path in sorted(folder.glob('root-*.json')):
            meta=json.loads(path.read_text())
            expected='train' if meta['case']%5<3 else 'dev' if meta['case']%5==3 else 'test'
            assert meta['split']==expected
            with np.load(path.with_suffix('.npz')) as archive:
                arrays={k:archive[k].copy() for k in archive.files}
            groups[(meta['model'],meta['cell'])].append((meta,arrays))
    protocol=dict(question='Do matched linear MC/pairwise readouts expose recoverable preference signal beyond original V, and does all-N improve it?',
        GO='Both seeds show >=0.05 test root-mean ranking-accuracy gain over P1 with no success-control regret increase; only finite authorized closed-loop may follow.',
        KILL='No consistent two-seed held-out gain: seal this candidate; no second-round IL/RL.',
        budget=dict(parameters=257,steps=args.steps,optimizer='Adam',lr=.001,batch='full frozen train partition',seed=args.seed),
        split='case%5: train 0,1,2; dev 3; test 4, across all model seeds. No dev selection or early stopping.',
        label_budget='Identical candidate G set for MC and pairwise; pairwise uses all unordered pairs within root, never additional rollout labels.',
        controls='Same linear256->1 initialization from old V, encoder frozen; P2/P3 TTC5, P4 full-N MC and pairwise. Predicted vs actual windows separate.',
        label_semantics='pred windows: archived rpred+.99*head(z), retaining original non-masked bootstrap for archival isolation. actual windows: r_env+.99*(1-terminal)*head(z), oracle-only diagnostic. Full deployment-policy branch G labels identical. P1-native-qpred is raw original score control.',
        limitation='No inference of navigation gain, training-continuation causation, or OOD from complete 20-person branch G.')
    (args.output/'readout-protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]; started=time.time()
    for (model,cell), records in sorted(groups.items()):
        ids=[m['root_id'] for m,d in records]
        assert len(ids)==len(set(ids)), 'duplicate root across feature folders'
        partitions={s:[x for x in records if x[0]['split']==s] for s in ('train','dev','test')}
        if any(not partitions[s] for s in partitions):
            results.append(dict(model=model,cell=cell,status='INCOMPLETE',reason='Missing train/dev/test root groups',
                roots={s:len(v) for s,v in partitions.items()},cases={s:sorted({m['case'] for m,d in v}) for s,v in partitions.items()}))
            continue
        for history in ('pred','actual'):
            for representation in ('5','N'):
                feature='latent_'+history+representation
                offset=0; slices=[]; x=[]; g=[]; reward=[]; boot=[]; pair_i=[]; pair_j=[]; pair_y=[]
                for m,d in partitions['train']:
                    inds=d['indices']; n=len(inds)
                    if history=='pred':
                        # The archived float32 discount product reconstructs exactly the original rpred.
                        discount=(.99*torch.as_tensor(d['values'][inds],dtype=torch.float32)).numpy()
                        r=d['qpred'][inds]-discount; b=np.full(len(inds),.99)
                    else:
                        r=d['reward']; b=.99*(~d['terminal'])
                    x.append(d[feature][inds]);g.append(d['G']);reward.append(r);boot.append(b)
                    for i,j in itertools.combinations(range(n),2):
                        if abs(d['G'][i]-d['G'][j])>1e-6:
                            pair_i.append(offset+i);pair_j.append(offset+j);pair_y.append(float(d['G'][i]>d['G'][j]))
                    slices.append((offset,offset+n));offset+=n
                X=torch.from_numpy(np.concatenate(x)).float()
                Y=torch.from_numpy(np.concatenate(g)).float()
                R=torch.from_numpy(np.concatenate(reward)).float()
                B=torch.from_numpy(np.concatenate(boot)).float()
                pi=torch.tensor(pair_i,dtype=torch.long); pj=torch.tensor(pair_j,dtype=torch.long); py=torch.tensor(pair_y)
                for objective in ('MC','pairwise'):
                    head=torch.nn.Linear(256,1)
                    init=records[0][1]
                    with torch.no_grad():
                        head.weight.copy_(torch.from_numpy(init['original_head_weight']))
                        head.bias.copy_(torch.from_numpy(init['original_head_bias']))
                    opt=torch.optim.Adam(head.parameters(),lr=.001)
                    if objective=='pairwise' and not len(pi):
                        results.append(dict(model=model,cell=cell,history=history,representation=representation,objective=objective,status='INCOMPLETE',reason='No strict train pair'))
                        continue
                    for _ in range(args.steps):
                        opt.zero_grad(); q=R+B*head(X).squeeze(-1)
                        loss=((q-Y)**2).mean() if objective=='MC' else torch.nn.functional.binary_cross_entropy_with_logits(q[pi]-q[pj],py)
                        assert torch.isfinite(loss)
                        loss.backward();opt.step()
                    arm=('P2' if objective=='MC' else 'P3') if representation=='5' else 'P4-'+objective
                    tag=f'{model}-cell{cell}-{history}-{arm}'
                    torch.save(dict(weight=head.state_dict(),protocol=protocol),args.output/(tag+'.pth'))
                    detail=[]
                    with torch.no_grad():
                        for partition, subset in partitions.items():
                            for m,d in subset:
                                inds=d['indices']; z=torch.from_numpy(d[feature][inds]).float()
                                if history=='pred':
                                    discount=(.99*torch.as_tensor(d['values'][inds],dtype=torch.float32)).numpy()
                                    r=d['qpred'][inds]-discount; b=np.full(len(inds),.99)
                                else:
                                    r=d['reward']; b=.99*(~d['terminal'])
                                score=r+b*head(z).squeeze(-1).numpy()
                                p1=r+b*(z.numpy()@d['original_head_weight'].T+d['original_head_bias']).ravel()
                                native=d['qpred'][inds]
                                for name, values in ((arm,score),('P1-matched-V',p1),('P1-native-Qpred',native)):
                                    # Ranking uses raw Q; candidate-set selection retains original shield/risk/index0.
                                    safe_scores=values.copy()
                                    clearance=d.get('clearance',np.ones(80))[inds]
                                    if np.any(d.get('clearance',np.ones(80))>=.2):
                                        safe_scores[clearance<.2]=-1e9
                                    safe_scores-=.8*np.maximum(0.,.2-clearance)
                                    safe_scores-=.001*(inds==0)
                                    correct=[]; strict=0; correction=0; broken=0
                                    for i,j in itertools.combinations(range(len(inds)),2):
                                        dg=d['G'][i]-d['G'][j]
                                        if abs(dg)<=1e-6: continue
                                        ok=bool((values[i]-values[j])*dg>1e-6*abs(dg)); correct.append(ok); strict+=1
                                        oldok=bool((p1[i]-p1[j])*dg>1e-6*abs(dg))
                                        correction+=int(ok and not oldok); broken+=int(not ok and oldok)
                                    detail.append(dict(root_id=m['root_id'],case=m['case'],scope=m['type'],split=partition,arm=name,
                                        root_pair_accuracy=float(np.mean(correct)) if correct else None,strict_pairs=strict,
                                        corrected_pairs=correction,broken_pairs=broken,regret=float(max(d['G'])-d['G'][int(np.argmax(safe_scores))]),
                                        mse=float(np.mean((values-d['G'])**2)),scores=values.tolist(),indices=inds.tolist()))
                    summary=[]
                    for partition in partitions:
                        for name in (arm,'P1-matched-V','P1-native-Qpred'):
                            for scope in ('all','success-control','failure'):
                                ds=[x for x in detail if x['split']==partition and x['arm']==name and (scope=='all' or (x['scope']=='success-control')==(scope=='success-control'))]
                                valid=[x['root_pair_accuracy'] for x in ds if x['root_pair_accuracy'] is not None]
                                if not ds: continue
                                summary.append(dict(split=partition,arm=name,scope=scope,roots=len(ds),cases=len({x['case'] for x in ds}),
                                    root_mean_pair_accuracy=float(np.mean(valid)) if valid else None,
                                    mean_regret=float(np.mean([x['regret'] for x in ds])),mean_mse=float(np.mean([x['mse'] for x in ds])),
                                    strict_pairs=sum(x['strict_pairs'] for x in ds)))
                    (args.output/(tag+'-roots.json')).write_text(json.dumps(detail,indent=2,allow_nan=False))
                    results.append(dict(model=model,cell=cell,history=history,representation=representation,objective=objective,
                        status='PASS',arm=arm,train_candidate_labels=len(Y),train_strict_pairs=len(pi),parameters=sum(p.numel() for p in head.parameters()),
                        optimizer_steps=args.steps,final_train_loss=float(loss),metrics=summary))
    result=dict(protocol=protocol,results=results,wall_seconds=time.time()-started,
        scientific_verdict='INCONCLUSIVE',method_level_progress=False,
        reason='Readout evidence alone does not establish causal navigation improvement; training/deployment continuation needs matched labels.')
    (args.output/'readout-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status='DONE',groups=len(groups),arms=len(results),wall_seconds=result['wall_seconds'])),flush=True)
