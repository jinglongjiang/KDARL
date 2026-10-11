"""CPU postprocessing of pre-registered A artifacts. No model fitting or rollout."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def readout(path):
    original=json.loads(path.read_text());groups=collections.defaultdict(dict)
    for a in original['results']:
        if a.get('status')!='PASS' or a.get('cell')!=5 or a.get('history')!='pred':continue
        metrics={(m['arm'],m['scope']):m for m in a['metrics'] if m['split']=='test'}
        arm=a['arm'];candidate=metrics[(arm,'all')];base=metrics[('P1-native-Qpred','all')]
        gain=candidate['root_mean_pair_accuracy']-base['root_mean_pair_accuracy']
        protected=metrics[(arm,'success-control')]['mean_regret']<=metrics[('P1-native-Qpred','success-control')]['mean_regret']
        groups[(a['model'].split('-s')[0],arm)][a['model']]=dict(ranking_gain=gain,success_control_regret_nonincrease=protected,
            seed_gate=bool(gain>=.05 and protected),candidate_test=candidate,
            success_control=metrics[(arm,'success-control')],P1=base,P1_success_control=metrics[('P1-native-Qpred','success-control')])
    decisions=[]
    for (backbone,arm),seeds in sorted(groups.items()):
        passed=len(seeds)==2 and all(v['seed_gate'] for v in seeds.values())
        decisions.append(dict(backbone=backbone,arm=arm,seeds=seeds,gate='PASS' if passed else 'FAIL',decision='GO' if passed else 'KILL'))
    return dict(source_sha256=sha(path),protocol=original['protocol'],groups=decisions,
        linear_candidate_decision='GO' if any(x['gate']=='PASS' for x in decisions) else 'KILL',method_level_progress=False)


def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def sweep_summary(row, steps):
    if not steps or any('actual_swept_clearance' not in s or 'env_swept_clearance' not in s for s in steps):
        return dict(status='INCOMPLETE',reason='No dual swept records; endpoint/outcome is not substituted for geometry.')
    result=dict(status='PASS',steps=len(steps))
    for mode in ('env','actual'):
        minimum=min(s[mode+'_swept_clearance'] for s in steps)
        count=sum(s[mode+'_swept_clearance']<0. for s in steps)
        assert row['min_'+mode+'_swept_clearance']==minimum
        assert row[mode+'_physical_collision_count']==count
        result[mode+'_minimum_clearance']=minimum
        result[mode+'_collision_intervals']=count
        result[mode+'_any_collision']=count>0
    assert row['physical_collision_count']==result['actual_collision_intervals']
    return result


def bootstrap(records, fields, draws=2000, seed=20261011):
    """Resample paired cases, preserving all rows/seeds belonging to a case."""
    clusters=collections.defaultdict(list)
    for row in records:clusters[row['case']].append(row)
    cases=sorted(clusters);rng=random.Random(seed);samples={k:[] for k in fields}
    if not cases:return dict(cases=0,intervals={})
    for _ in range(draws):
        chosen=[clusters[cases[rng.randrange(len(cases))]] for _ in cases]
        for field in fields:
            value=mean([r.get(field) for group in chosen for r in group])
            if value is not None:samples[field].append(value)
    def quantile(x,p):
        x=sorted(x);index=(len(x)-1)*p;lo=int(index);hi=min(lo+1,len(x)-1)
        return x[lo]+(index-lo)*(x[hi]-x[lo])
    return dict(cases=len(cases),draws=draws,seed=seed,unit='case; both arms and any model seeds remain paired',
        interval='Descriptive percentile95 only; no new scientific gate or significance-based selection.',
        intervals={k:dict(mean=mean([r.get(k) for r in records]),low=quantile(v,.025),high=quantile(v,.975),
                         valid_rows=sum(r.get(k) is not None for r in records)) for k,v in samples.items() if v})


def closedloop(folder, plan_path=None, draws=2000, seed=20261011):
    path=folder/'closedloop-rows.jsonl.gz';done=folder/'closedloop-summary.json'
    if not path.exists() or not done.exists():return dict(status='INCOMPLETE',folder=str(folder))
    completed=json.loads(done.read_text())
    protocol=json.loads((folder/'closedloop-protocol.json').read_text())
    plan=json.loads(plan_path.read_text()) if plan_path else protocol['plan']
    if plan_path:assert sha(plan_path)==protocol['plan_sha256']
    assert plan['arms']==['ttc5-V','allN-V']
    planned={(r['model'],r['root_id']):r for r in plan['roots']}
    grouped=collections.defaultdict(dict)
    with gzip.open(path,'rt') as f:
        for l in f:
            r=json.loads(l);key=(r['model'],r['root_id']);assert r['arm'] not in grouped[key];grouped[key][r['arm']]=r
    assert sum(map(len,grouped.values()))==completed['continuations']
    details=[];totals=collections.defaultdict(collections.Counter)
    for (model,rid),arms in sorted(grouped.items()):
        assert set(arms)=={'ttc5-V','allN-V'}, 'Learned heads forbidden or missing protected baseline'
        a,b=arms['ttc5-V'],arms['allN-V'];assert a['restoration_exact'] and b['restoration_exact']
        specification=planned[(model,rid)]
        assert a['case']==b['case']==specification['case'] and a['scope']==b['scope']==specification['scope']
        for row in (a,b):
            assert abs(sum(.99**j*s['reward'] for j,s in enumerate(row['steps']))-row['root_return'])<1e-12
            assert row['time']==(row['step']+len(row['steps']))*.25
            actual=hashlib.sha256(json.dumps(row['steps'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert actual==row['continuation_sha256']
        assert a['root_candidate_controls']==b['root_candidate_controls']
        assert a['baseline_outcome_new']==a['outcome']==b['baseline_outcome_new']
        flags=dict(new_baseline_failed=a['outcome']!='success',new_baseline_success=a['outcome']=='success',
            rescue=a['outcome']!='success' and b['outcome']=='success',success_destroyed=a['outcome']=='success' and b['outcome']!='success',
            added_collision=a['outcome']!='collision' and b['outcome']=='collision',removed_collision=a['outcome']=='collision' and b['outcome']!='collision',
            delta_G_positive=b['root_return']>a['root_return']+1e-6,delta_G_negative=b['root_return']<a['root_return']-1e-6)
        flags.update(added_timeout=a['outcome']!='timeout' and b['outcome']=='timeout',removed_timeout=a['outcome']=='timeout' and b['outcome']!='timeout',
            first_command_changed=a['steps'][0]['command']!=b['steps'][0]['command'])
        safety={arm:sweep_summary(row,row['steps']) for arm,row in arms.items()}
        decisions={arm:row.get('root_decision') for arm,row in arms.items()}
        decision_comparison=dict(status='INCOMPLETE')
        if all(decisions.values()):
            parent=decisions['ttc5-V']['original_parent'];used=decisions['allN-V']['used_input']
            assert parent==decisions['ttc5-V']['used_input']==decisions['allN-V']['original_parent']
            assert parent['input']=='ttc5' and used['input']=='allN'
            assert all(len(d[k])==80 for d in (parent,used) for k in ('values','raw_scores','scores'))
            decision_comparison=dict(status='PASS',original_parent_exact_across_arms=True,
                parent_raw_top=max(range(80),key=lambda i:parent['raw_scores'][i]),used_raw_top=max(range(80),key=lambda i:used['raw_scores'][i]),
                parent_final_top=max(range(80),key=lambda i:parent['scores'][i]),used_final_top=max(range(80),key=lambda i:used['scores'][i]),
                max_abs_value_delta=max(abs(x-y) for x,y in zip(parent['values'],used['values'])),
                max_abs_raw_delta=max(abs(x-y) for x,y in zip(parent['raw_scores'],used['raw_scores'])),
                max_abs_final_delta=max(abs(x-y) for x,y in zip(parent['scores'],used['scores'])))
        dual_complete=all(s['status']=='PASS' for s in safety.values())
        if dual_complete:
            for mode in ('env','actual'):
                ac=safety['ttc5-V'][mode+'_any_collision'];bc=safety['allN-V'][mode+'_any_collision']
                flags['added_'+mode+'_physical_collision']=not ac and bc
                flags['removed_'+mode+'_physical_collision']=ac and not bc
                flags['baseline_success_with_'+mode+'_physical_collision']=a['outcome']=='success' and ac
                flags['allN_success_with_'+mode+'_physical_collision']=b['outcome']=='success' and bc
        common=a['outcome']==b['outcome']=='success'
        details.append(dict(model=model,root_id=rid,case=a['case'],old_sampling_scope=a['scope'],new_baseline_outcome=a['outcome'],
            matching_status=specification['matching_status'],allN_outcome=b['outcome'],
            baseline_G=a['root_return'],allN_G=b['root_return'],delta_G=b['root_return']-a['root_return'],delta_time=b['time']-a['time'],
            delta_success=int(b['outcome']=='success')-int(a['outcome']=='success'),
            delta_collision=int(b['outcome']=='collision')-int(a['outcome']=='collision'),
            delta_timeout=int(b['outcome']=='timeout')-int(a['outcome']=='timeout'),
            safety=safety,dual_sweep_complete=dual_complete,
            root_decision=decisions,root_decision_comparison=decision_comparison,
            delta_actual_physical_collision=int(safety['allN-V']['actual_any_collision'])-int(safety['ttc5-V']['actual_any_collision']) if dual_complete else None,
            delta_env_physical_collision=int(safety['allN-V']['env_any_collision'])-int(safety['ttc5-V']['env_any_collision']) if dual_complete else None,
            rescue_conditional=int(flags['rescue']) if flags['new_baseline_failed'] else None,
            damage_conditional=int(flags['success_destroyed']) if flags['new_baseline_success'] else None,
            common_success_time_delta=b['time']-a['time'] if common else None,
            common_success_path_delta=b['path_length']-a['path_length'] if common else None,
            baseline_time=a['time'],allN_time=b['time'],baseline_path=a['path_length'],allN_path=b['path_length'],
            baseline_latency=a['mean_inference_seconds'],allN_latency=b['mean_inference_seconds'],
            inference_ratio=b['mean_inference_seconds']/a['mean_inference_seconds'],
            baseline_steps=len(a['steps']),allN_steps=len(b['steps']),
            terminal_root_raw_changed=any(x!=y for x,y in zip(a['terminal_raw_difference']['old_raw'],a['terminal_raw_difference']['new_raw'])),flags=flags))
        for scope in ('all',a['scope'],a['scope']+':'+specification['matching_status']):
            z=totals[(model,scope)];z['roots']+=1
            for name,value in flags.items():z[name]+=int(value)
            z['baseline_'+a['outcome']]+=1;z['allN_'+b['outcome']]+=1
    gates=[]
    for model,scope in sorted({(r['model'],r['old_sampling_scope']) for r in details}):
        p=folder/('A0-repeat-'+model+'-'+scope+'.json')
        gate=json.loads(p.read_text());assert gate['status']=='PASS' and gate['exact'];gates.append(gate)
    reports=[]
    fields=('delta_G','delta_success','delta_collision','delta_timeout','rescue_conditional','damage_conditional',
            'common_success_time_delta','common_success_path_delta','inference_ratio','delta_actual_physical_collision','delta_env_physical_collision')
    for (model,scope),counts in sorted(totals.items()):
        rs=[r for r in details if r['model']==model and (scope=='all' or scope==r['old_sampling_scope'] or scope==r['old_sampling_scope']+':'+r['matching_status'])]
        common=[r for r in rs if r['common_success_time_delta'] is not None]
        reports.append(dict(model=model,old_scope=scope,counts=dict(counts),
            denominators=dict(rescue=counts['new_baseline_failed'],damage=counts['new_baseline_success'],common_success=len(common)),
            dual_sweep_roots=sum(r['dual_sweep_complete'] for r in rs),
            mean_G_delta=mean([r['delta_G'] for r in rs]),mean_common_success_time_delta=mean([r['delta_time'] for r in common]),
            mean_common_success_baseline_time=mean([r['baseline_time'] for r in common]),
            mean_common_success_allN_time=mean([r['allN_time'] for r in common]),
            mean_common_success_path_delta=mean([r['common_success_path_delta'] for r in common]),
            latency=dict(baseline_episode_mean=mean([r['baseline_latency'] for r in rs]),allN_episode_mean=mean([r['allN_latency'] for r in rs]),
                baseline_step_weighted=sum(r['baseline_latency']*r['baseline_steps'] for r in rs)/sum(r['baseline_steps'] for r in rs),
                allN_step_weighted=sum(r['allN_latency']*r['allN_steps'] for r in rs)/sum(r['allN_steps'] for r in rs)),
            paired_case_bootstrap=bootstrap(rs,fields,draws,seed)))
    return dict(status='PASS',source_sha256=sha(path),done_sha256=sha(done),paired_roots=details,A0_repeat_gates=gates,
        planned_roots=len(planned),executed_roots=len(grouped),unexecuted_roots=[dict(model=m,root_id=r) for m,r in sorted(set(planned)-set(grouped))],
        totals=reports,all_models_paired_case_bootstrap=bootstrap(details,fields,draws,seed),
        denominator='Rescue denominator uses actual repaired TTC5-V failure; success destruction uses actual repaired TTC5-V successes. Old scopes only sampling labels.',
        latency_caution='Actual diagnostic implementation includes original TTC scoring plus fullN scoring and root exact-check overhead. This is not optimized allN deployment latency.',
        sweep_convention='Synchronous relative-segment geometry uses actual executed robot command. env=current human velocity; actual=human displacement/.25. Collision intervals are strict negative clearance, not distinct people. New success does not imply collision-free geometry.',
        limitation='Finite selected-root recursive-input effect, not full unselected benchmark success rate or learned-method contribution.')


def paired5(folder, draws=2000, seed=20261011):
    summary=folder/'paired5-summary.json';raw=folder/'paired5-rows.jsonl.gz'
    if not summary.exists() or not raw.exists():return dict(status='INCOMPLETE',folder=str(folder))
    groups=collections.defaultdict(dict);early=[]
    with gzip.open(raw,'rt') as f:
        for l in f:
            r=json.loads(l)
            if r['status']=='PRE_ROOT_TERMINAL':early.append(r);continue
            recomputed=sum(.99**j*v for j,v in enumerate(r['rewards']));assert abs(recomputed-r['root_return'])<1e-12
            assert r['arm'] not in groups[(r['model'],r['case'])]
            groups[(r['model'],r['case'])][r['arm']]=r
    effects=[];totals=collections.defaultdict(collections.Counter)
    for (model,case),arms in sorted(groups.items()):
        assert set(arms)=={'old-train','fixed-train','old-deploy','fixed-deploy'}
        assert len({r['root_history_sha256'] for r in arms.values()})==1
        assert len({r['true_next_window_sha256'] for r in arms.values()})==1
        assert len({tuple(r['forced_command']) for r in arms.values()})==1
        assert len({r['forced_reward'] for r in arms.values()})==1
        assert all(r['restoration_exact'] and r['commands'][0]==r['forced_command'] and r['rewards'][0]==r['forced_reward'] for r in arms.values())
        assert arms['old-deploy']['commands']==arms['fixed-deploy']['commands']
        assert arms['old-deploy']['rewards']==arms['fixed-deploy']['rewards']
        safety={arm:sweep_summary(row,row.get('sweep_steps',[])) for arm,row in arms.items()}
        complete=all(s['status']=='PASS' for s in safety.values())
        if complete:
            for row in arms.values():
                assert len(row['sweep_steps'])==len(row['rewards'])
                assert [s['command'] for s in row['sweep_steps']]==row['commands']
        for factor,left,right in (('history_train','old-train','fixed-train'),('history_deploy','old-deploy','fixed-deploy'),('continuation_old','old-train','old-deploy'),('continuation_fixed','fixed-train','fixed-deploy')):
            a,b=arms[left],arms[right];dg=b['root_return']-a['root_return']
            flags=dict(outcome_changed=a['outcome']!=b['outcome'],success_gained=a['outcome']!='success' and b['outcome']=='success',
                success_lost=a['outcome']=='success' and b['outcome']!='success',command_trace_changed=a['commands']!=b['commands'],
                G_positive=dg>1e-6,G_negative=dg< -1e-6)
            if complete:
                for mode in ('env','actual'):
                    ac=safety[left][mode+'_any_collision'];bc=safety[right][mode+'_any_collision']
                    flags['added_'+mode+'_physical_collision']=not ac and bc
                    flags['removed_'+mode+'_physical_collision']=ac and not bc
            effects.append(dict(model=model,case=case,split=a['split'],factor=factor,status='VALID_ROOT',delta_G=dg,
                delta_success=int(b['outcome']=='success')-int(a['outcome']=='success'),
                delta_collision=int(b['outcome']=='collision')-int(a['outcome']=='collision'),
                delta_timeout=int(b['outcome']=='timeout')-int(a['outcome']=='timeout'),
                delta_actual_physical_collision=int(safety[right]['actual_any_collision'])-int(safety[left]['actual_any_collision']) if complete else None,
                delta_env_physical_collision=int(safety[right]['env_any_collision'])-int(safety[left]['env_any_collision']) if complete else None,
                left_outcome=a['outcome'],right_outcome=b['outcome'],flags=flags))
            z=totals[(model,factor)];z['valid_roots']+=1
            for k,v in flags.items():z[k]+=int(v)
        def contrast(field):
            def value(arm):
                r=arms[arm]
                if field=='G':return r['root_return']
                if field.endswith('_physical_collision'):return int(safety[arm][field.split('_')[0]+'_any_collision'])
                return int(r['outcome']==field)
            return (value('fixed-deploy')-value('old-deploy'))-(value('fixed-train')-value('old-train'))
        effects.append(dict(model=model,case=case,split=arms['old-train']['split'],factor='history_x_deployment_interaction',status='VALID_ROOT',
            delta_G=contrast('G'),delta_success=contrast('success'),delta_collision=contrast('collision'),delta_timeout=contrast('timeout'),
            delta_actual_physical_collision=contrast('actual_physical_collision') if complete else None,
            delta_env_physical_collision=contrast('env_physical_collision') if complete else None))
        totals[(model,'history_x_deployment_interaction')]['valid_roots']+=1
    factors=('history_train','history_deploy','continuation_old','continuation_fixed','history_x_deployment_interaction')
    protocol=json.loads((folder/'paired5-protocol.json').read_text());cases=protocol['cases'];models=protocol['models']
    assert len(cases)==16
    for model in models:
        valid={c for m,c in groups if m==model};pre=[r for r in early if r['model']==model]
        assert len(valid)+len(pre)==16 and valid|{r['case'] for r in pre}==set(cases)
        for r in pre:
            for factor in factors:
                effects.append(dict(model=model,case=r['case'],split=r['split'],factor=factor,status='PRE_ROOT_TERMINAL',
                    delta_G=None,delta_success=0,delta_collision=0,delta_timeout=0,
                    delta_actual_physical_collision=0,delta_env_physical_collision=0))
    contrasts=[]
    for model in models:
        for factor in factors:
            rs=[r for r in effects if r['model']==model and r['factor']==factor]
            assert len(rs)==16
            contrasts.append(dict(model=model,factor=factor,denominator_all_cases=16,
                valid_roots=sum(r['status']=='VALID_ROOT' for r in rs),counts=dict(totals[(model,factor)]),
                positive_G=sum(r['delta_G'] is not None and r['delta_G']>1e-6 for r in rs),
                negative_G=sum(r['delta_G'] is not None and r['delta_G']< -1e-6 for r in rs),
                paired_case_bootstrap=bootstrap(rs,('delta_G','delta_success','delta_collision','delta_timeout','delta_actual_physical_collision','delta_env_physical_collision'),draws,seed)))
    fitpath=folder/'paired5-readout.json'
    fitted=json.loads(fitpath.read_text()) if fitpath.exists() else dict(status='INCOMPLETE')
    for r in fitted.get('results',[]):
        if r.get('status')=='INCOMPLETE':continue
        assert r['parameters']==257 and 1<=r['steps']<=300
    collected=json.loads(summary.read_text());arm_reports=[]
    for model in models:
        pre=[r for r in early if r['model']==model]
        for arm in ('old-train','fixed-train','old-deploy','fixed-deploy'):
            rs=[arms[arm] for (m,c),arms in groups.items() if m==model]
            safety=[sweep_summary(r,r.get('sweep_steps',[])) for r in rs]
            covered=[s for s in safety if s['status']=='PASS']
            outcomes=[r['outcome'] for r in rs]+[r['baseline_outcome'] for r in pre]
            arm_reports.append(dict(model=model,arm=arm,denominator_all_cases=16,valid_roots=len(rs),
                pre_root_terminal=len(pre),actual_outcomes_all16=dict(collections.Counter(outcomes)),
                mean_G_valid_roots=mean([r['root_return'] for r in rs]),
                mean_time_valid_roots=mean([r['time'] for r in rs]),
                mean_path_after_root=mean([r['path_length_after_root'] for r in rs]),
                mean_episode_latency=mean([r['mean_inference_seconds'] for r in rs]),
                sweep=dict(covered_valid_roots=len(covered),missing_valid_roots=len(rs)-len(covered),
                    env_collision_roots=sum(s['env_any_collision'] for s in covered),actual_collision_roots=sum(s['actual_any_collision'] for s in covered),
                    env_collision_intervals=sum(s['env_collision_intervals'] for s in covered),actual_collision_intervals=sum(s['actual_collision_intervals'] for s in covered),
                    env_minimum_clearance=min((s['env_minimum_clearance'] for s in covered),default=None),
                    actual_minimum_clearance=min((s['actual_minimum_clearance'] for s in covered),default=None),
                    success_with_actual_collision=sum(r['outcome']=='success' and s.get('actual_any_collision',False) for r,s in zip(rs,safety)))))
    return dict(status='PASS',source_sha256=sha(raw),summary_sha256=sha(summary),pre_root_terminal=early,effects=effects,
        totals=contrasts,arms=arm_reports,collector_summary=collected['models'],readout=fitted,
        interaction_sign='(fixed-deploy - old-deploy) - (fixed-train - old-train), for G and each actual outcome indicator.',
        early_terminal_convention='All16 cases remain in outcome-effect denominators; pre-root cases have zero arm difference. Root G is undefined for them and excluded only from G means.',
        readout_contract='Only already-collected CPU linear257 readouts <=300steps; no postprocess fitting, tuning, new gate or IL/RL.',
        eval_contract=protocol.get('eval_contract','UNVERIFIED: absent from original protocol'),
        scope='Frozen policy generation/label continuation effects only; continuous ReplayBufferMC window reconstruction not altered; one exploration tape, no retraining.')


def alias(folder, roots_path):
    required=[folder/n for n in ('plan.json','records.jsonl.gz','summary.json')]
    if not all(p.exists() for p in required):return dict(status='INCOMPLETE',folder=str(folder))
    plan=json.loads(required[0].read_text());original=json.loads(roots_path.read_text())
    assert sha(roots_path)==plan['original_root_source_sha256']
    roots={r['root_id']:r for r in original['roots']};selected={r['root_id']:r for r in plan['roots']}
    qualifications={r['root_id']:r for r in plan['qualifications']};grouped=collections.defaultdict(dict)
    with gzip.open(required[1],'rt') as f:
        for line in f:
            r=json.loads(line);rid=r['root_id'];spec=selected[rid];key=(r['world'],r['label'])
            assert key not in grouped[rid];grouped[rid][key]=r
            expected=spec['a0'] if r['label']=='A0' else spec['alternative']
            assert r['candidate']==expected and r['steps'][0]['command']==spec['commands'][expected]
            assert r['time']==len(r['steps'])*.25 and r['steps'][-1]['terminal']
            assert all(s['step']==i and s['terminal']==(i==len(r['steps'])-1) for i,s in enumerate(r['steps']))
            assert abs(sum(.99**j*s['reward'] for j,s in enumerate(r['steps']))-r['return_'])<1e-12
    expected={(w,l) for w in ('natural','far-omitted') for l in ('A0','alternative')};pairs=[]
    for rid,spec in selected.items():
        assert roots[rid]['step']==0 and roots[rid]['cell']==5
        assert len(spec['selected'])==5 and spec['selected']==roots[rid]['snapshot']['ids']
        assert all(qualifications[rid]['checks'].values()) and set(grouped[rid])==expected
        deltas={w:grouped[rid][(w,'alternative')]['return_']-grouped[rid][(w,'A0')]['return_'] for w in ('natural','far-omitted')}
        pairs.append(dict(root_id=rid,model=spec['model'],case=spec['case'],scope=spec['scope'],delta_return=deltas,
            preference_reversal=min(deltas.values())< -1e-6 and max(deltas.values())>1e-6))
    reported=json.loads(required[2].read_text());assert len(pairs)==reported['roots'] and len(pairs)*4==reported['episodes']
    assert sum(r['preference_reversal'] for r in pairs)==reported['reversals']
    assert {r['root_id']:r['delta_return'] for r in pairs}=={r['root_id']:r['delta_return'] for r in reported['pairs']}
    return dict(status='PASS',sources_sha256={p.name:sha(p) for p in required},roots=len(pairs),episodes=len(pairs)*4,
        reversals=sum(r['preference_reversal'] for r in pairs),pairs=pairs,
        limitation='Constructed reset worlds modify omitted positions AND their hidden goals. TTC5 root equivalence is recorded by executed qualification assertions; no GPU re-execution. Reversal frequency is not a natural attribution fraction or a proven allN navigation benefit.')


def filter_probe(folder, roots_path):
    files=[folder/n for n in ('plan.json','records.jsonl.gz','summary.json')]
    if not all(p.exists() for p in files):return dict(status='INCOMPLETE',folder=str(folder))
    plan=json.loads(files[0].read_text());assert sha(roots_path)==plan['source_sha256']
    roots={r['root_id']:r for r in json.loads(roots_path.read_text())['roots']}
    selected={r['root_id']:r for r in plan['roots']};grouped=collections.defaultdict(dict)
    with gzip.open(files[1],'rt') as f:
        for line in f:
            r=json.loads(line);assert r['label'] not in grouped[r['root_id']]
            grouped[r['root_id']][r['label']]=r
            assert abs(sum(.99**j*s['reward'] for j,s in enumerate(r['steps']))-r['return_'])<1e-12
            tail=r['steps'][r['step']:]
            assert abs(sum(.99**j*s['reward'] for j,s in enumerate(tail))-r['root_return'])<1e-12
            assert min(s['actual_swept_clearance'] for s in tail)==r['minimum_actual_swept_after_root']
            assert r['time']==len(r['steps'])*.25 and r['steps'][-1]['terminal']
    reported=json.loads(files[2].read_text());by_root={r['root_id']:r for r in reported['pairs']}
    pairs=[];totals=collections.defaultdict(collections.Counter)
    assert set(grouped)==set(selected)
    for rid,spec in selected.items():
        arms=grouped[rid];assert set(arms)=={'A0','filtered'};a,b=arms['A0'],arms['filtered'];index=spec['filtered_candidate']
        original=roots[rid]['snapshot'];assert index==max(range(80),key=lambda i:original['raw_scores'][i])
        assert original['n_safe']>0 and original['clearance'][index]<.2
        assert a['steps'][:spec['step']]==b['steps'][:spec['step']]
        for key in ('chosen_original','commands','values','rewards','clearance','raw_scores','scores'):
            assert a['root_details'][key]==b['root_details'][key]
        assert b['root_details']['forced']==index and a['root_details']['forced']==a['root_details']['chosen_original']
        for r in (a,b):assert r['steps'][spec['step']]['command']==r['root_details']['commands'][r['root_details']['forced']]
        flags=dict(rescued=a['outcome']!='success' and b['outcome']=='success',destroyed=a['outcome']=='success' and b['outcome']!='success',
            new_collision=a['outcome']!='collision' and b['outcome']=='collision',
            added_actual_physical_collision=a['minimum_actual_swept_after_root']>=0. and b['minimum_actual_swept_after_root']<0.)
        dg=b['root_return']-a['root_return'];expected=by_root[rid]
        assert dg==expected['delta_G'] and all(flags[k]==expected[k] for k in ('rescued','destroyed','new_collision'))
        flags.update(G_improved=dg>1e-6,G_worse=dg< -1e-6)
        pair=dict(root_id=rid,model=a['model'],case=a['case'],scope=a['scope'],delta_G=dg,flags=flags,
            baseline_outcome=a['outcome'],filtered_outcome=b['outcome'],
            filtered_candidate=index,predicted_clearance=b['root_details']['clearance'][index],
            baseline_actual_swept_min=a['minimum_actual_swept_after_root'],filtered_actual_swept_min=b['minimum_actual_swept_after_root'])
        pairs.append(pair);z=totals[a['model']];z['roots']+=1
        for k,v in flags.items():z[k]+=int(v)
    assert len(pairs)==reported['roots'] and len(pairs)*2==reported['episodes']
    fail=any(r['flags']['destroyed'] or r['flags']['new_collision'] or r['flags']['added_actual_physical_collision'] for r in pairs)
    return dict(status='PASS',sources_sha256={p.name:sha(p) for p in files},roots=len(pairs),episodes=len(pairs)*2,pairs=pairs,
        per_model={m:dict(z) for m,z in totals.items()},protection_gate='FAIL' if fail else 'No observed protection violation',
        bypass_decision='KILL' if fail else 'Not decided by this postprocessor',
        limitation='Existing protection gate only. Return gains do not prove a filter error: forced commands violate original .2 margin. Fixed selected roots cannot estimate natural attribution; no production filter change.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--readout',type=Path);p.add_argument('--closedloop',type=Path)
    p.add_argument('--paired5',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--alias',type=Path)
    p.add_argument('--filter-probe',type=Path)
    p.add_argument('--roots',type=Path,default=Path('/home/abc/workspace/ttc5-causal-20261011/work/roots.json'))
    p.add_argument('--bootstrap-draws',type=int,default=2000);p.add_argument('--bootstrap-seed',type=int,default=20261011)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert a.bootstrap_draws>0
    result={}
    if a.readout:result['readout']=readout(a.readout)
    if a.closedloop:result['closedloop']=closedloop(a.closedloop,a.plan,a.bootstrap_draws,a.bootstrap_seed)
    if a.paired5:result['paired5']=paired5(a.paired5,a.bootstrap_draws,a.bootstrap_seed)
    if a.alias:result['observation_alias']=alias(a.alias,a.roots)
    if a.filter_probe:result['filter_probe']=filter_probe(a.filter_probe,a.roots)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v.get('status',v.get('linear_candidate_decision')) for k,v in result.items()}))
