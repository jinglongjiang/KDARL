"""Necessary A3 frozen-weight input intervention, independent of the A4 readout gate."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('input_once_engine',HERE/'closedloop-a.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
A=E.A


def freeze(args):
    assert not args.output.exists(), 'Never replace an existing frozen plan'
    roots=json.loads((args.data/'work/roots.json').read_text())['roots']
    selected=[]
    for model in args.models:
        pool=[r for r in roots if r['model']==model and r['cell']==5]
        failures=sorted([r for r in pool if r['type']!='success-control' and r['case']%5==4],key=lambda r:(r['case'],r['step'],r['root_id']))
        used=set();pairs=[]
        for f in failures:
            controls=sorted([r for r in pool if r['type']=='success-control' and r.get('matched_failure_case')==f['case'] and r['root_id'] not in used],key=lambda r:(r['case'],r['step'],r['root_id']))
            assert controls, (model,f['case'],'missing original matched control')
            c=controls[0];used.add(c['root_id']);pairs.extend([(f,'failure',c['root_id']),(c,'success-control',f['root_id'])])
            if len(used)==8:break
        assert len(used)==8
        for r,scope,other in pairs:
            selected.append(dict(root_id=r['root_id'],model=model,cell=5,case=r['case'],step=r['step'],scope=scope,
                original_matched_root=other,case_split=A.split(r['case'])))
    selected.sort(key=lambda r:(r['model'],r['case'],r['step'],r['root_id']))
    A.save(args.output,dict(roots=selected,roots_sha256=A.sha(args.data/'work/roots.json'),models=args.models,arms=['ttc5-V','allN-V'],
        rule='Each model: first8 failure test cases (case%5==4), original matched success root per failure, no readout/outcome-result root selection. Matched controls retain original case group and are NOT all claimed test cases.',
        question='Can legal all-N frozen-weight input at one root decision improve actual repaired-parent continuation while protecting original successes?',
        GO='Positive paired actual continuation benefit with no added success destruction/collisions; only limited input-mechanism support, not algorithm innovation.',
        KILL='No protected outcome benefit: seal all-N-one-step-only candidate regardless of offline readout movement.',
        budget=f'{len(selected)} frozen roots x2 arms, <=128 episodes over four models, <=201 total environment steps/episode.',
        closedloop_semantics='At root, same80commands/rpred/filter/terminal; only latent input changes. FullN selected command executed once, then repaired original TTC5/V parent resumed.',
        control='A0 repeated read-only scoring must be exact; when allN chooses A0, entire continuation command/reward trajectory must be exact.',
        exclusion='This input experiment is mandatory independent of A4 gate. Matched success controls not reassigned to another data split; no readout-head claim on train-group controls.'))
    print(json.dumps(dict(frozen_roots=len(selected),episodes=len(selected)*2,output=str(args.output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','run']);p.add_argument('--data',type=Path,default=A.DEFAULT_DATA)
    p.add_argument('--inputs',type=Path,default=A.DEFAULT_DATA/'inputs');p.add_argument('--project',type=Path,default=HERE/'project')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path)
    p.add_argument('--models',nargs='+',default=['kda-s419-t24','kda-s443-t24','gru-s419-t24','gru-s443-t24'])
    p.add_argument('--device',default='cpu');p.add_argument('--main-authorized',action='store_true');p.add_argument('--gpu-authorized',action='store_true')
    args=p.parse_args()
    if args.command=='freeze':freeze(args)
    else:
        assert args.plan
        args.once=True;args.heads=None;args.arms=['ttc5-V','allN-V'];E.run(args)
