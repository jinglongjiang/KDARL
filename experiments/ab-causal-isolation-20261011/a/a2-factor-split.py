"""Offline fixed-denominator subdivision of the original pinned F; no rollout/model."""
import argparse
import collections
import configparser
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

TOL=1e-6


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def sign(value):return 1 if value>TOL else -1 if value< -TOL else 0


def main(args):
    assert not args.output.exists(),'Never overwrite original or previous decomposition'
    args.output.mkdir(parents=True);torch.set_num_threads(1);started=time.time()
    wanted=collections.defaultdict(lambda:collections.defaultdict(list));old_summary=json.loads(args.original_summary.read_text())
    with gzip.open(args.original_pairs,'rt') as stream:
        for line in stream:
            pair=json.loads(line)
            if 'original_strict_error' in pair['flags']:wanted[pair['source']][pair['source_line']].append(pair)
    cfg={};configs={}
    for model in {p['model'] for lines in wanted.values() for ps in lines.values() for p in ps}:
        path=args.inputs/model/'config.ini';c=configparser.ConfigParser();c.read(str(path))
        cfg[model]=c.getfloat('robot','success_radius');configs[str(path)]=sha(path)
    old_source=subprocess.check_output(['git','-C',str(args.project),'show','cd8b793:crowd_nav/policy/mamba_rl.py'])
    protocol=dict(question='Within original strict-loss pairs, how much does pinned F change ranking through pinned physical V, immediate reward, and terminal mask separately?',
        denominator='Exactly original a2-pairs original_strict_error: sign(delta G) nonzero and sign(delta saved Qpred) opposite, tolerance1e-6. Fixed across all stages; not all G-improved pairs.',
        stages=['saved Qpred','physical-only: original rpred + float32(.99*true pinned V)',
            'then realReward: real reward + float32(.99*true pinned V)','then terminal: real reward alone if done'],
        floating='Old scoring multiplies gamma*V in torch.float32, then adds reward tensor (float64 if any predicted candidate nonterminal; float32 for all-terminal batch). Oracle converts float32 product to Python float before adding real reward. Preserve both actual contracts and report reconstruction/rounding residuals; no float64 gamma*V substitution.',
        transition='Correction=wrong->correct, reverse=correct->wrong relative to fixed realized G. Physical reverse is structurally0 on an initially-wrong denominator; ties tracked separately. Later reverse can undo earlier correction.',
        interpretation='Physical-only changes the oracle-pinned last-frame V, not pure CV position error: all true next human position/velocity fields with pinned IDs affect encoder output. These are ordered algebraic diagnostic replacements, not additive navigation-cause fractions.',
        GO='Reconstruction matches archived P/F within reported floating precision: publish descriptive subdivision only.',
        KILL='Failed reconstruction: invalidate this subdivision; do not infer mechanism or launch new experiments.',
        budget='One CPU streaming pass over existing branch files; no new model invocation, GPU, rollout, optimizer or selection.',
        old_source_sha256=hashlib.sha256(old_source).hexdigest(),oracle_source_sha256=sha(args.data/'audit.py'),script_sha256=sha(__file__),
        original_pairs_sha256=sha(args.original_pairs),original_summary_sha256=sha(args.original_summary),config_sha256=configs,tolerance=TOL)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    groups={};rounding=collections.defaultdict(float);sources={};processed=0
    with gzip.open(args.output/'pairs.jsonl.gz','wt') as output:
        for filename,lines in sorted(wanted.items()):
            path=Path(filename);sources[filename]=sha(path)
            assert sources[filename]==old_summary['source_sha256'][filename]
            with gzip.open(path,'rt') as stream:
                for number,line in enumerate(stream,1):
                    if number not in lines:continue
                    row=json.loads(line);root=row['root'];snapshot=root['snapshot'];branches={b['candidate_index']:b for b in row['branches']}
                    robot=snapshot['robot'];commands=np.asarray(snapshot['commands'])
                    reaching=np.hypot(robot[0]+.25*commands[:,0]-robot[5],robot[1]+.25*commands[:,1]-robot[6])<cfg[root['model']]
                    predicted_terminal=(np.asarray(snapshot['clearance'])<0.)|reaching
                    reward_dtype=torch.float32 if predicted_terminal.all() else torch.float64
                    q0=(torch.tensor(snapshot['rewards'],dtype=reward_dtype)+.99*torch.tensor(snapshot['values'],dtype=torch.float32)).tolist()
                    for i,value in enumerate(q0):rounding['max_original80_reconstruction_abs']=max(rounding['max_original80_reconstruction_abs'],abs(value-snapshot['raw_scores'][i]))
                    def values(index):
                        s=branches[index]['first_step'];discount=float((.99*torch.tensor(s['value'],dtype=torch.float32)).item())
                        physical=float((torch.tensor(snapshot['rewards'][index],dtype=reward_dtype)+.99*torch.tensor(s['value'],dtype=torch.float32)).item())
                        reward=float(s['reward']+discount);terminal=float(s['reward']) if s['terminal'] else reward
                        residual=dict(P_reconstruction=q0[index]-s['q_pred'],realReward_vs_archived_unmasked=reward-s['q_oracle1_unmasked'],
                            terminal_vs_archived_F=terminal-s['q_oracle1_terminal_corrected'],
                            gamma32_minus_naive64=discount-.99*s['value'],realReward_float32add_minus_contract=float(torch.tensor(s['reward'],dtype=torch.float32)+.99*torch.tensor(s['value'],dtype=torch.float32))-reward)
                        for key,value in residual.items():rounding['max_abs_'+key]=max(rounding['max_abs_'+key],abs(value))
                        assert abs(residual['P_reconstruction'])<1e-6 and abs(residual['realReward_vs_archived_unmasked'])<1e-6 and abs(residual['terminal_vs_archived_F'])<1e-6
                        return dict(q=[s['q_pred'],physical,reward,terminal],predicted_value=s['predicted_value'],pinned_value=s['value'],
                            rpred=snapshot['rewards'][index],real_reward=s['reward'],terminal=s['terminal'],rounding=residual)
                    for old in lines[number]:
                        assert old['root_id']==root['root_id']
                        a,b=values(old['a0']),values(old['alternative']);d=[y-x for x,y in zip(a['q'],b['q'])]
                        dg=old['delta_pred_pinned_actual_G'][3];truth=sign(dg);stages=[sign(v) for v in d];assert stages[0]==-truth and truth
                        key=(old['model'],old['cell'],old['scope']);g=groups.setdefault(key,dict(count=collections.Counter(),roots=collections.defaultdict(set),cases=collections.defaultdict(set)))
                        def count(name):
                            g['count'][name]+=1;g['roots'][name].add(old['root_id']);g['cases'][name].add(old['case'])
                        count('denominator');flags=[]
                        for j,name in enumerate(('physical_only','realReward','terminal'),1):
                            left,right=stages[j-1],stages[j]
                            label=name+('_correct' if right==truth else '_wrong' if right==-truth else '_tie');count(label);flags.append(label)
                            for suffix,condition in (('correction',left==-truth and right==truth),('reverse_correction',left==truth and right==-truth)):
                                if condition:count(name+'_'+suffix);flags.append(name+'_'+suffix)
                        residual=(d[-1]-d[0])-sum(d[j]-d[j-1] for j in range(1,4))
                        rounding['max_telescoping_abs']=max(rounding['max_telescoping_abs'],abs(residual))
                        evidence=dict(root_id=old['root_id'],model=old['model'],cell=old['cell'],case=old['case'],scope=old['scope'],
                            source=filename,source_line=number,a0=old['a0'],alternative=old['alternative'],commands=old['commands'],
                            a0_components=a,alternative_components=b,delta_Qpred_physical_realReward_terminal=d,delta_G=dg,
                            ordered_terms=[d[j]-d[j-1] for j in range(1,4)],archived_delta_F=old['delta_pred_pinned_actual_G'][1],
                            archived_F_gap_residual=d[-1]-old['delta_pred_pinned_actual_G'][1],telescoping_residual=residual,flags=flags)
                        output.write(json.dumps(evidence,allow_nan=False)+'\n');processed+=1
    assert processed==sum(len(ps) for lines in wanted.values() for ps in lines.values())
    names=['denominator']+[n+'_'+s for n in ('physical_only','realReward','terminal') for s in ('correct','wrong','tie','correction','reverse_correction')]
    summary=dict(status='PASS',protocol=protocol,source_sha256=sources,processed_pairs=processed,rounding=dict(rounding),groups=[],wall_seconds=time.time()-started,method_level_progress=False)
    for (model,cell,scope),g in sorted(groups.items()):
        summary['groups'].append(dict(model=model,cell=cell,scope=scope,metrics={n:dict(action_pairs=g['count'][n],independent_roots=len(g['roots'][n]),independent_cases=len(g['cases'][n])) for n in names}))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(dict(status='PASS',pairs=processed,rounding=dict(rounding),wall_seconds=summary['wall_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);here=Path(__file__).resolve().parent
    p.add_argument('--data',type=Path,default=Path('/home/abc/workspace/ttc5-causal-20261011'))
    p.add_argument('--inputs',type=Path,default=Path('/home/abc/workspace/ttc5-causal-20261011/inputs'))
    p.add_argument('--project',type=Path,default=here/'project')
    p.add_argument('--original-pairs',type=Path,default=here/'artifacts/a2-pairs.jsonl.gz')
    p.add_argument('--original-summary',type=Path,default=here/'artifacts/a2-summary.json')
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
