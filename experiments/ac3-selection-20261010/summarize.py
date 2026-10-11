"""Paired case-cluster summaries; no model selection or rule changes."""
import sys
sys.dont_write_bytecode = True
from collections import Counter
import gzip
import json
import numpy as np
from common import BASE, RULES, protocol, save, verify


def statistics(rows):
    counts = Counter(r['outcome'] for r in rows)
    successes = [r['time'] for r in rows if r['outcome']=='success']
    return dict(n=len(rows), counts=dict(counts),
                SR=100*counts['success']/len(rows), CR=100*counts['collision']/len(rows),
                TO=100*counts['timeout']/len(rows), mean_return=float(np.mean([r['return_'] for r in rows])),
                median_minimum_endpoint_clearance=float(np.median([r['minimum_clearance'] for r in rows])),
                success_navtime_mean=float(np.mean(successes)) if successes else None)


def interval(delta, rng):
    # One row per case; preserve cells and, for pooled output, both trainingseeds.
    delta = np.asarray(delta, dtype=float)
    assert delta.shape[0] == 500 and np.isfinite(delta).all()
    draws = rng.integers(0,500,(10000,500))
    means = delta[draws].mean(axis=1)
    return dict(mean=float(delta.mean()), ci95=np.quantile(means,[.025,.975]).tolist(),
                unit='caseID cluster; correlated cells/seeds retained', clusters=500)


def contrast(rows, candidate, baseline, seeds):
    d = {(r['seed'], r['rule'], r['cell'], r['case']): r for r in rows}
    pairs = [[(d[(s,candidate,c,k)],d[(s,baseline,c,k)]) for s in seeds for c in (4,5)]
             for k in range(90000,90500)]
    rng = np.random.default_rng(20261010)
    result = dict(n_pairs=500*len(seeds)*2, training_seeds=seeds, candidate=candidate,baseline=baseline)
    for key, transform in (
        ('return', lambda r:r['return_']),
        ('SR_pp', lambda r:100*int(r['outcome']=='success')),
        ('CR_pp', lambda r:100*int(r['outcome']=='collision')),
        ('TO_pp', lambda r:100*int(r['outcome']=='timeout')),
    ):
        delta = [np.mean([transform(a)-transform(b) for a,b in pair]) for pair in pairs]
        result[key] = interval(delta,rng)
    allpairs = [p for block in pairs for p in block]
    result['rescue'] = sum(a['outcome']=='success' and b['outcome']!='success' for a,b in allpairs)
    result['damage'] = sum(a['outcome']!='success' and b['outcome']=='success' for a,b in allpairs)
    both = [(a,b) for a,b in allpairs if a['outcome']==b['outcome']=='success']
    result['common_success_n'] = len(both)
    # Ratio bootstrap keeps the shared-success subgroup paired and case-clustered.
    numer = np.array([sum(a['time']-b['time'] for a,b in block if a['outcome']==b['outcome']=='success') for block in pairs])
    denom = np.array([sum(a['outcome']==b['outcome']=='success' for a,b in block) for block in pairs])
    idx = rng.integers(0,500,(10000,500))
    totals = denom[idx].sum(axis=1)
    vals = numer[idx].sum(axis=1)/np.maximum(totals,1)
    result['common_success_navtime_delta'] = dict(mean=float(numer.sum()/denom.sum()) if denom.sum() else None,
                                                 ci95=np.quantile(vals[totals>0],[.025,.975]).tolist() if (totals>0).any() else None)
    return result


def main():
    verified = verify()
    rows = []
    sources = sorted((BASE/'per-case').glob('*/cell*.json'))
    assert len(sources)==12
    for p in sources:
        data=json.loads(p.read_text())
        assert data['status']=='PASS' and len(data['episodes'])==500
        rows.extend(data['episodes'])
    assert len(rows)==6000 and len({(r['seed'],r['rule'],r['cell'],r['case']) for r in rows})==6000
    absolute={}
    comparisons={}
    diagnostic={}
    for seed in (419,443):
        for rule in RULES:
            group=[r for r in rows if r['seed']==seed and r['rule']==rule]
            absolute['s%d-%s'%(seed,rule)] = dict(overall=statistics(group),
                cells={str(c):statistics([r for r in group if r['cell']==c]) for c in (4,5)})
            count=Counter()
            for c in (4,5):
                path=BASE/'per-case'/('s%d-%s'%(seed,rule))/('cell%d.steps.jsonl.gz'%c)
                with gzip.open(path,'rt') as f:
                    for line in f:
                        r=json.loads(line)
                        count['steps']+=1
                        for key in ('selection_differs','ttc_omits_cv','selected_includes_cv',
                                    'ttc_omits_actual','selected_includes_actual'):
                            count[key]+=r[key]
                        count['all_unsafe']+=int(r['n_safe']==0)
                        count['補選_cv']+=int(r['ttc_omits_cv'] and r['selected_includes_cv'])
                        count['補選_actual']+=int(r['ttc_omits_actual'] and r['selected_includes_actual'])
                        count['補選_actual_'+r['outcome']]+=int(r['ttc_omits_actual'] and r['selected_includes_actual'])
            diagnostic['s%d-%s'%(seed,rule)]=dict(counts=dict(count),
                rates={k:v/count['steps'] for k,v in count.items() if k!='steps'},
                interpretation='On-policy state distribution; 補選/outcome association, not isolated step causal rescue')
    for seeds in ([419],[443],[419,443]):
        label='+'.join(map(str,seeds))
        comparisons[label]={a+'-'+b:contrast(rows,a,b,seeds)
                            for a,b in (('ac3','ttc5'),('ac3','goal-cv1s'),('goal-cv1s','ttc5'))}
    checks=[]
    for seed in (419,443):
        for baseline in ('ttc5','goal-cv1s'):
            x=comparisons[str(seed)]['ac3-'+baseline]
            checks.append(dict(seed=seed,baseline=baseline,
                               return_positive=x['return']['mean']>0,
                               SR_nonnegative=x['SR_pp']['mean']>=0,
                               CR_nonincrease=x['CR_pp']['mean']<=0))
    accepted=all(all(x[k] for k in ('return_positive','SR_nonnegative','CR_nonincrease')) for x in checks)
    accepted=accepted and all(diagnostic['s%d-ac3'%s]['counts']['selection_differs']>0 for s in (419,443))
    verdict='PROTOTYPE_ENTRY_NOT_CONFIRMED' if accepted else 'REJECT'
    data=dict(status='COMPLETED',verdict=verdict,total_episodes=6000,training_episodes=0,
              absolute=absolute,paired_comparisons=comparisons,diagnostics=diagnostic,
              decision_checks=checks,sources=[str(p) for p in sources],frozen_files_verified=verified,
              protocol_sha256=__import__('common').sha(BASE/'protocol.json'))
    save(BASE/'summary.json',data)
    lines=['# AC3 Frozen-Weight Development Test','',
           '**Status: '+verdict+'; 6000 episodes; zero training.**','',
           '## Question','Does action-coverage selection improve navigation beyond TTC5 and a one-direction Goal-CV1s selector?', '',
           '## Frozen Setup','KDA T24 fresh IL50+RL10000 paired seeds419/443. 20-person circle/square,500cases/cell,90000-90499developmentblock. TTC5/Goal-CV1s/AC3, same80smoothedcommands, alpha0.3, unchanged reward/filter/value network. All20humans remain in reward/filter; only5networkslots change. No92000confirmationcases.','',
           'The two downloaded files are the method sources; latest user protocol supersedes their384-episode sample plan. Canonical TTC robot float32roundtrip/tie behavior is preserved in AC3fallback/fill.','',
           '## Results','Source: per-case/s{seed}-{rule}/cell{4,5}.json. Each overall row is1000episodes; eachcell500.','',
           '| Seed | Selector | Cell | SR % | CR % | TO % | Mean Original Return | Success Time s |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for seed in (419,443):
        for rule in RULES:
            x=absolute['s%d-%s'%(seed,rule)]
            for cell,v in [('overall',x['overall']),*x['cells'].items()]:
                lines.append('| %d | %s | %s | %.2f | %.2f | %.2f | %.6f | %s |'%(seed,rule,cell,v['SR'],v['CR'],v['TO'],v['mean_return'], '%.3f'%v['success_navtime_mean'] if v['success_navtime_mean'] is not None else 'n/a'))
    lines+=['','## Paired Comparisons','Positive Return/SR favors the first selector; positive CR/TO/NavTime is worse. 95%caseIDclusterbootstrap intervals (10000draws), not160k independentsteps or a trainingseedpopulation test. Pooled intervals keep bothseeds andbothcells together for eachcase.','',
            '| Seeds | Contrast | Return Delta [95% CI] | SR Delta pp | CR Delta pp | TO Delta pp | Rescue / Damage | Common Success Time Delta s |',
            '|---|---|---|---:|---:|---:|---|---:|']
    for seeds,cs in comparisons.items():
        for name,x in cs.items():
            ret=x['return'];nt=x['common_success_navtime_delta']['mean']
            lines.append('| %s | %s | %.6f [%.6f, %.6f] | %.2f | %.2f | %.2f | %d / %d | %s |'%(seeds,name,ret['mean'],*ret['ci95'],x['SR_pp']['mean'],x['CR_pp']['mean'],x['TO_pp']['mean'],x['rescue'],x['damage'],'%.3f'%nt if nt is not None else 'n/a'))
    lines+=['','## Selection Diagnostics','Source: per-case/*/cell*.steps.jsonl.gz; actual constraint means minimum actual next-position endpointclearance, not a swept collisioncertificate. Rates are on each method own trajectories and cannot be compared as same-state causal interventions.','',
            '| Seed / Rule | Steps | Selection Different % | TTC Omits Actual Constraint % | Selected Adds Omitted Constraint % | All Unsafe % |',
            '|---|---:|---:|---:|---:|---:|']
    for name,x in diagnostic.items():
        r=x['rates'];lines.append('| %s | %d | %.2f | %.2f | %.2f | %.2f |'%(name,x['counts']['steps'],100*r.get('selection_differs',0),100*r.get('ttc_omits_actual',0),100*r.get('補選_actual',0),100*r.get('all_unsafe',0)))
    lines+=['','## Decision','Frozen gate: positive Returnvsbothcontrols ineachseed, nonnegativeSR and nonincreasingcollisionsvsboth. These strict point-estimate guards are a conservative operational prototype gate, not proof of noninferiority. Full CIs andpaircounts are in summary.json.','', '```json',json.dumps(checks,indent=2),'```','',
            '## Evidence Boundaries','- Verified: correctness/smoke checks, pairedclosed-loop outcomes and allinputSHAverification.','- Notverified: novelty, generalization beyondthese2weights, independentconfirmations, causal benefitofanyindividual補選step.','- Existing23.48%/11737stepomission claim is userprovidedcontext, not independently reproduced here.','- 90000block is developmentdata alreadyused before, notafreshholdout.','- Runtime islocal3060existingtorch2.7.1 isolatedstack; absoluteperformance cannot be assertedbitwiseequal to4090torch2.9.1. All3arms arecomparedinthissamestack.','- SmallerCV1sgeometryerror andbetterselectionhit do not prove betterlong-runvalue.','',
            '## Recommendation',
            'Preserve as an unconfirmed prototype; next only independent confirmation and precise prior-work audit after explicit authorization.' if accepted else 'Stop AC3. No horizon/weight/threshold tuning, no retraining, no additional seeds or92000confirmations to rescue it. The measured failure does not invalidate all action-aware representations.',
            '', 'No new experiment is started automatically.']
    (BASE/'RESULT.md').write_text('\n'.join(lines)+'\n')
    print(dict(verdict=verdict,episodes=6000,report=str(BASE/'RESULT.md')),flush=True)


if __name__=='__main__':
    main()
