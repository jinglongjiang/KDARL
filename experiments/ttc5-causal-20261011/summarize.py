"""Recompute every claim from completed raw counterfactuals, then package once."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np

BASE=Path(__file__).resolve().parent
MAN=json.loads((BASE/'manifest.json').read_text())
P=MAN['protocol']
EPS=P['numerical_return_tolerance']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def dump(f,row):
    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')


def ci(values):
    if not values:return None
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(271828)
    results=[]
    for _ in range(50):
        results.extend(x[rng.integers(0,len(x),size=(100,len(x)))].mean(axis=1).tolist())
    return np.percentile(results,[2.5,97.5]).tolist()


def distribution(values):
    return None if not values else dict(n=len(values),mean=float(np.mean(values)),
        p50=float(np.median(values)),p90=float(np.percentile(values,90)),
        minimum=float(min(values)),maximum=float(max(values)))


def category_for(real, masked, pred, oracle):
    if real <= EPS:return 'no_return_improvement'
    if masked:return 'hard_filter'
    if pred > 1e-9:return 'postprocessing_index0_or_risk'
    if oracle > 1e-9:return 'oracle1_corrects_pair'
    return 'real_successor_value_residual'


def contrast(base,other,root):
    a,b=base['first_step'],other['first_step']
    pred=b['q_pred']-a['q_pred']
    pinned_oracle=b['q_oracle1_unmasked']-a['q_oracle1_unmasked']
    pinned_terminal=b['q_oracle1_terminal_corrected']-a['q_oracle1_terminal_corrected']
    actual_unmasked=b['q_oracle1_actual_history']-a['q_oracle1_actual_history']
    terminal_oracle=b['q_oracle1_actual_history_terminal_corrected']-a['q_oracle1_actual_history_terminal_corrected']
    oracle=terminal_oracle
    real=other['root_return']-base['root_return']
    final=b['final_score']-a['final_score']
    r0=root['snapshot']['rewards'][base['candidate_index']]
    r1=root['snapshot']['rewards'][other['candidate_index']]
    cv_only0=r0+float(np.float32(np.float32(.99)*np.float32(a['value'])))
    cv_only1=r1+float(np.float32(np.float32(.99)*np.float32(b['value'])))
    reward_only0=a['reward']+float(np.float32(np.float32(.99)*np.float32(a['predicted_value'])))
    reward_only1=b['reward']+float(np.float32(np.float32(.99)*np.float32(b['predicted_value'])))
    human=np.asarray(root['snapshot']['humans'])
    actual=np.asarray(a['true_next_humans'])
    position_errors=np.linalg.norm(actual[:,:2]-(human[:,:2]+.25*human[:,2:4]),axis=1)
    selected_errors=position_errors[root['snapshot']['ids']]
    assert abs((pred-real)-((pred-oracle)+(oracle-real)))<1e-10
    assert abs((pred-real)-((pred-pinned_terminal)+(pinned_terminal-oracle)+(oracle-real)))<1e-10
    category=category_for(real,b['predicted_hard_masked'],pred,oracle)
    return dict(delta_pred=pred,delta_final_score=final,delta_oracle1=oracle,
                delta_oracle1_terminal_corrected=terminal_oracle,delta_real=real,
                delta_oracle1_pinned=pinned_oracle,delta_oracle1_pinned_terminal_corrected=pinned_terminal,
                delta_oracle1_actual_unmasked=actual_unmasked,
                successor_reward_error=pred-oracle,future_value_error=oracle-real,
                physical_prediction_reward_error=pred-pinned_terminal,
                selection_history_transition_error=pinned_terminal-oracle,
                actual_terminal_bootstrap_adjustment=actual_unmasked-oracle,
                first_component_includes_terminal_handling=bool(a['terminal'] or b['terminal']),
                delta_real_immediate_reward=b['reward']-a['reward'],
                delta_predicted_immediate_reward=r1-r0,
                delta_cv_only=cv_only1-cv_only0,
                delta_real_reward_only=reward_only1-reward_only0,
                successor_position_error_all_mean=float(position_errors.mean()),
                successor_position_error_selected_mean=float(selected_errors.mean()),
                successor_position_error_selected_max=float(selected_errors.max()),
                all_current_human_velocities_zero=bool(np.max(np.abs(human[:,2:4]))<1e-12),
                delta_predicted_value=b['predicted_value']-a['predicted_value'],
                delta_true_successor_value=b['value']-a['value'],
                category=category,
                pinned_category=category_for(real,b['predicted_hard_masked'],pred,pinned_terminal),
                strict_raw_misrank=pred < -1e-9 and real>EPS,
                tied_raw=abs(pred)<=1e-9,delta_episode_return=other['return']-base['return'],
                rescued=base['outcome']!='success' and other['outcome']=='success',
                damaged=base['outcome']=='success' and other['outcome']!='success',
                original_outcome=base['outcome'],new_outcome=other['outcome'],
                original_time=base['time'],new_time=other['time'],
                original_root_return=base['root_return'],new_root_return=other['root_return'],
                original_min_env_swept=min(s['env_swept_clearance'] for s in base['steps']),
                alternative_min_env_swept=min(s['env_swept_clearance'] for s in other['steps']),
                original_min_actual_swept=min(s['actual_swept_clearance'] for s in base['steps']),
                alternative_min_actual_swept=min(s['actual_swept_clearance'] for s in other['steps']),
                original_postroot_min_actual_swept=min(s['actual_swept_clearance'] for s in base['steps'][root['step']:]),
                alternative_postroot_min_actual_swept=min(s['actual_swept_clearance'] for s in other['steps'][root['step']:]),
                original_postroot_min_env_swept=min(s['env_swept_clearance'] for s in base['steps'][root['step']:]),
                alternative_postroot_min_env_swept=min(s['env_swept_clearance'] for s in other['steps'][root['step']:]))


def main():
    status=json.loads((BASE/'work/status.json').read_text())
    assert status['stage']=='EXPERIMENTS_COMPLETE',status
    assert json.loads((BASE/'work/finalize-run.done.json').read_text())['status']=='PASS'
    final_check=json.loads((BASE/'work/post-attribution-end-verification.json').read_text())
    assert final_check['status']=='PASS' and all(final_check['files'].values()) and all(final_check['project_files'].values())
    for stem in ('real-history-oracle','shared-rank','input-probe'):
        done=json.loads((BASE/'work'/(stem+'.done.json')).read_text())
        assert done['status']=='PASS' and sha(BASE/'work'/(stem+'.jsonl.gz'))==done['sha256'],stem
    local_inputs={rel:sha(BASE/rel)==expected for rel,expected in MAN['files'].items()}
    assert local_inputs and all(local_inputs.values())
    originals={p:sha(p)==expected for p,expected in MAN['original_local_files'].items()}
    project=Path('/home/abc/workspace/nav_data/mamba/camrl/CrowdNav')
    project_check={p:sha(project/p)==expected for p,expected in MAN['project_files'].items()}
    assert originals and all(originals.values()), 'Original local asset changed'
    assert project_check and all(project_check.values()), 'Original local project changed'
    frozen=json.loads((BASE/'work/roots.json').read_text())
    assert sha(BASE/'work/roots.json')==json.loads((BASE/'work/roots.sha256.json').read_text())['sha256']
    roots={r['root_id']:r for r in frozen['roots']}
    root_by_episode={(r['model'],r['cell'],r['case']):r for r in roots.values()}
    history_descriptions={}
    groups=collections.defaultdict(list)
    totals={}
    baseline_sources=[]
    for path in sorted((BASE/'work').glob('baseline-*.jsonl.gz')):
        assert Path(str(path)+'.done.json').exists()
        assert sha(path)==json.loads(Path(str(path)+'.done.json').read_text())['sha256']
        baseline_sources.append(path)
        for ep in read(path):
            key=(ep['model'],ep['cell'])
            stat=totals.setdefault(key,dict(n=0,success=0,collision=0,timeout=0,steps=0))
            stat['n']+=1;stat[ep['outcome']]+=1;stat['steps']+=len(ep['steps'])
            root=root_by_episode.get((ep['model'],ep['cell'],ep['case']))
            if root:
                k=root['step'];now=ep['steps'][k];end=ep['steps'][-1]
                # Candidate windows contain23 real frames plus one hypothetical successor.
                history=ep['steps'][max(0,k-22):k+1]
                selected=[set(s['ids']) for s in history]
                history_descriptions[root['root_id']]=dict(
                    real_frame_steps=[s['step'] for s in history],left_padding=max(0,23-len(history)),
                    selected_ids=[s['ids'] for s in history],root_ids=now['ids'],
                    selection_changes=sum(a!=b for a,b in zip(selected,selected[1:])),
                    unique_people_in_real_window=len(set().union(*selected)),
                    root_env_constraint=now['env_constraint'],root_actual_constraint=now['actual_constraint'],
                    root_env_constraint_omitted=now['env_constraint'] not in now['ids'],
                    root_actual_constraint_omitted=now['actual_constraint'] not in now['ids'],
                    terminal_env_constraint=end['env_constraint'],terminal_actual_constraint=end['actual_constraint'],
                    terminal_env_constraint_missing_at_root=end['env_constraint'] not in now['ids'],
                    terminal_env_constraint_missing_at_terminal=end['env_constraint'] not in end['ids'],
                    limitation='observational descriptors; omission or switching is not a causal diagnosis')
    records=[]
    actual_oracles={r['root_id']:r for r in read(BASE/'work/real-history-oracle.jsonl.gz')}
    assert len(actual_oracles)==len(roots), 'Actual successor oracle incomplete'
    for path in sorted((BASE/'work').glob('branches-*.jsonl.gz')):
        assert Path(str(path)+'.done.json').exists()
        assert sha(path)==json.loads(Path(str(path)+'.done.json').read_text())['sha256']
        for record in read(path):
            if record['record_type']!='counterfactual':continue
            root=record['root'];rid=root['root_id']
            assert root==roots[rid]
            bylabel={label:b for b in record['branches'] for label in b['labels']}
            actual={b['candidate_index']:b for b in actual_oracles[rid]['branches']}
            for b in record['branches']:
                extra=actual[b['candidate_index']]
                b['first_step'].update({k:v for k,v in extra.items() if k not in ('candidate_index','labels')})
            a0=bylabel['A0']
            assert a0['return']==root['baseline_return'] and a0['outcome']==root['baseline_outcome']
            for b in record['branches']:
                assert b['first_step']['true_next_humans']==a0['first_step']['true_next_humans'], 'Invisible robot human-response mismatch'
                r=root['snapshot']['robot'];u=b['first_step']['command']
                propagated=[r[0]+.25*u[0],r[1]+.25*u[1],u[0],u[1]]+r[4:]
                assert np.array_equal(np.asarray(propagated,dtype=np.float32),
                    np.asarray(b['first_step']['true_next_robot'],dtype=np.float32)), 'Oracle robot successor mismatch'
            record['comparisons']={}
            for label in ('A1','A2','A3'):
                if label not in bylabel:continue
                other=bylabel[label]
                result=contrast(a0,other,root)
                result.update(root_id=rid,model=root['model'],cell=root['cell'],case=root['case'],
                    step=root['step'],root_type=root['type'],label=label,
                    original_action=a0['candidate_index'],alternative_action=other['candidate_index'],
                    actually_different=a0['candidate_index']!=other['candidate_index'],
                    original_prediction=a0['first_step'],alternative_prediction=other['first_step'])
                record['comparisons'][label]=result
                groups[(root['model'],root['cell'],root['type'],label)].append(result)
            records.append({'root':root,'comparisons':record['comparisons']})
    assert len({r['root']['root_id'] for r in records})==len(records)
    complete=len(records)==len(roots)
    summary={}
    for key,items in sorted(groups.items()):
        categories=collections.Counter(x['category'] for x in items)
        summary['|'.join(map(str,key))]=dict(n=len(items),distinct_interventions=sum(x['actually_different'] for x in items),
            return_improved=sum(x['delta_real']>EPS for x in items),return_worsened=sum(x['delta_real'] < -EPS for x in items),
            return_equal=sum(abs(x['delta_real'])<=EPS for x in items),
            timeout_to_success=sum(x['rescued'] and x['original_outcome']=='timeout' for x in items),
            collision_to_success=sum(x['rescued'] and x['original_outcome']=='collision' for x in items),
            success_to_failure=sum(x['damaged'] for x in items),
            outcome_transitions=dict(collections.Counter(x['original_outcome']+'->'+x['new_outcome'] for x in items)),
            improved_without_success=sum(x['delta_real']>EPS and x['new_outcome']!='success' for x in items),
            paired_success_time_changes=[x['new_time']-x['original_time'] for x in items
                                         if x['original_outcome']==x['new_outcome']=='success'],
            rescued_but_actual_swept_negative=sum(x['rescued'] and x['alternative_postroot_min_actual_swept']<0 for x in items),
            mean_delta_root_return=float(np.mean([x['delta_real'] for x in items])),
            ci_delta_root_return=ci([x['delta_real'] for x in items]),
            mean_delta_episode_return=float(np.mean([x['delta_episode_return'] for x in items])),
            ci_delta_episode_return=ci([x['delta_episode_return'] for x in items]),
            categories=dict(categories))
    # Pool geometries only by resampling case clusters; never pool training seeds as IID replicates.
    case_intervals={}
    for model in [m['id'] for m in P['models']]:
        for label in ('A1','A2','A3'):
            selected=[x for key,items in groups.items() if key[0]==model and key[1] in (4,5) and key[2]!='success-control'
                      and key[3]==label for x in items]
            clusters=collections.defaultdict(list)
            for x in selected:clusters[x['case']].append(x['delta_episode_return'])
            if clusters:
                sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
                rng=np.random.default_rng(314159);idx=rng.integers(0,len(sums),(5000,len(sums)))
                boot=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
                case_intervals[model+'|'+label]=dict(cases=len(clusters),episodes=len(selected),
                    mean=float(sums.sum()/counts.sum()),ci=np.percentile(boot,[2.5,97.5]).tolist())
    provenance={'record_type':'provenance','manifest':MAN,'status':status,
        'raw_files':{str(p.relative_to(BASE)):sha(p) for p in sorted((BASE/'work').glob('*')) if p.is_file()},
        'raw_file_bytes':{str(p.relative_to(BASE)):p.stat().st_size for p in sorted((BASE/'work').glob('*')) if p.is_file()},
        'process_exits':{p.name:json.loads(p.read_text()) for p in sorted((BASE/'work').glob('*.exit.json'))},
        'runtime':json.loads((BASE/'work/runtime.json').read_text()),
        'training_contract_unit':json.loads((BASE/'work/training-contract-probe.json').read_text()),
        'training_contract_invalid_attempt':json.loads((BASE/'work/training-contract-probe-attempt1.json').read_text()),
        'local_original_end_verification':originals,
        'local_inputs_and_frozen_scripts_end_verification':local_inputs,
        'local_project_end_verification':project_check,
        'preflight':{p.name:json.loads(p.read_text()) for p in sorted((BASE/'work').glob('preflight-*.json'))
                     if not p.name.endswith(('.pid.json','.exit.json'))},
        'invalid_attempts':{str(p.relative_to(BASE)):p.read_text() for p in sorted((BASE/'invalid-preflight-1').glob('*.log'))},
        'end_verification':json.loads((BASE/'work/end-verification.json').read_text()),
        'analysis_script_sha256':sha(Path(__file__)),
        'reproduction_check':json.loads((BASE/'work/reproduction-check.json').read_text()),
        'artifact_metadata':{p.name:json.loads(p.read_text()) for p in sorted((BASE/'work').glob('*.json'))
                             if p.name!='roots.json'},
        'diagnostic_scripts_sha256':{p.name:sha(p) for p in sorted(BASE.glob('*.py'))}}
    training_logs={}
    for model in P['models']:
        source=next(Path(path) for path,digest in MAN['original_local_files'].items()
                    if path.endswith('.pth') and digest==model['weight_sha256'])
        log=source.parent/'train.log'
        assert log.is_file(),str(log)
        expected=sha(log)
        selected=[]
        for number,line in enumerate(log.read_text().splitlines(),1):
            if number<=30 or '[P1-RL] Base config' in line or '[SEED]' in line:
                selected.append(dict(line=number,text=line))
        assert sha(log)==expected
        training_logs[model['id']]=dict(path=str(log),sha256=expected,lines=selected)
    provenance['original_training_log_evidence']=training_logs
    assert all(provenance['end_verification']['files'].values())
    assert all(provenance['end_verification']['project_files'].values())
    aggregate={'record_type':'summary','baseline_totals':{'|'.join(map(str,k)):v for k,v in totals.items()},
               'root_count':len(roots),'completed_roots':len(records),'complete':complete,
               'all_failure_count':len(frozen['all_failures']),'no_root_count':frozen['no_root_count'],
               'per_alternative':summary,'case_cluster_intervals_separate_seeds':case_intervals}
    allpairs=[x for r in records for x in r['comparisons'].values()]
    unique_pairs=list({(x['root_id'],x['alternative_action']):x for x in allpairs
                       if x['actually_different']}.values())
    failures20=[x for x in unique_pairs if x['cell'] in (4,5) and x['root_type']!='success-control']
    mechanism_summary={}
    for model in [m['id'] for m in P['models']]:
        for cell in (5,4):
            xs=[x for x in failures20 if x['model']==model and x['cell']==cell]
            rs=[r for r in records if r['root']['model']==model and r['root']['cell']==cell
                and r['root']['type']!='success-control']
            mechanism_summary[model+'|'+str(cell)]={
                'completed_failure_roots':len(rs),
                'any_improvement_oracle_roots':len({x['root_id'] for x in xs if x['delta_real']>EPS}),
                'any_rescue_oracle_roots':len({x['root_id'] for x in xs if x['rescued']}),
                'unique_executed_pairs':len(xs),
                'improved_pair_categories':dict(collections.Counter(x['category'] for x in xs if x['delta_real']>EPS)),
                'A1_exclusive_categories':dict(collections.Counter(r['comparisons']['A1']['category']
                    for r in rs if 'A1' in r['comparisons'])),
                'strict_raw_misrank_pairs':sum(x['strict_raw_misrank'] for x in xs),
                'tied_raw_improved_pairs':sum(x['tied_raw'] and x['delta_real']>EPS for x in xs),
                'strict_raw_and_actual_oracle_misrank_pairs':sum(x['strict_raw_misrank'] and x['delta_oracle1'] < -1e-9 for x in xs),
                'actual_oracle_tied_improved_pairs':sum(abs(x['delta_oracle1'])<=1e-9 and x['delta_real']>EPS for x in xs),
                'input_rule':'all alternatives original margin-safe; masked-action effect not identified'}
            positive=[x for x in xs if x['delta_real']>EPS]
            mechanism_summary[model+'|'+str(cell)].update(
                original_pinned_categories=dict(collections.Counter(x['pinned_category'] for x in positive)),
                category_changes=dict(collections.Counter(x['pinned_category']+'->'+x['category'] for x in positive)),
                oracle_corrected_at_initial_step=sum(x['category']=='oracle1_corrects_pair' and x['step']==0 for x in positive),
                physical_only_corrects=sum(x['delta_cv_only']>1e-9 and x['delta_pred']<=1e-9 for x in positive),
                real_reward_only_corrects=sum(x['delta_real_reward_only']>1e-9 and x['delta_pred']<=1e-9 for x in positive),
                numerical_components={name:distribution([x[name] for x in positive]) for name in (
                    'delta_pred','delta_real','physical_prediction_reward_error',
                    'selection_history_transition_error','future_value_error',
                    'successor_position_error_selected_mean')},
                remaining_after_oracle_rescue_roots=len({x['root_id'] for x in positive
                    if x['category']=='real_successor_value_residual' and x['rescued']}))
    aggregate['mechanisms_unique_pairs']=mechanism_summary
    aggregate['matched_success_controls']={m['id']:dict(
        failures=sum(r['model']==m['id'] for r in frozen['all_failures']),
        controls=sum(r['model']==m['id'] and r['type']=='success-control' for r in roots.values()),
        unmatched=[dict(cell=r['cell'],case=r['case'],root_id=r['root_id']) for r in roots.values()
                   if r['model']==m['id'] and r['type']!='success-control' and r['matched_control'] is None])
        for m in P['models']}
    shared_rows=list(read(BASE/'work/shared-rank.jsonl.gz'))
    aggregate['same_state_ranking_descriptive']={m['id']:{other['id']:dict(
        pairs=sum(x['reference_model']==m['id'] for x in shared_rows),
        still_raw_nonpositive=sum(x['reference_model']==m['id'] and x['raw_gap_by_model'][other['id']]<=1e-9 for x in shared_rows),
        prefers_reference_better_action=sum(x['reference_model']==m['id'] and x['raw_gap_by_model'][other['id']]>1e-9 for x in shared_rows))
        for other in P['models']} for m in P['models']}
    input_rows=list(read(BASE/'work/input-probe.jsonl.gz')) if (BASE/'work/input-probe.jsonl.gz').exists() else []
    input_summary={}
    for model in [m['id'] for m in P['models']]:
        xs=[x for x in input_rows if x['model']==model]
        known=[x for x in xs if x['changed'] and x['outcome_verified']]
        input_summary[model]=dict(roots=len(xs),changed=sum(x['changed'] for x in xs),
            changed_with_exact_pretested_command=len(known),
            changed_outcome_unknown=sum(x['changed'] and not x['outcome_verified'] for x in xs),
            return_improved=sum(x['existing_branch_evidence']['delta_root_return']>EPS for x in known),
            return_worsened=sum(x['existing_branch_evidence']['delta_root_return'] < -EPS for x in known),
            rescued=sum(x['existing_branch_evidence']['original_outcome']!='success' and
                        x['existing_branch_evidence']['new_outcome']=='success' for x in known),
            damaged=sum(x['existing_branch_evidence']['original_outcome']=='success' and
                        x['existing_branch_evidence']['new_outcome']!='success' for x in known))
    aggregate['legal_history_input_probe']=input_summary
    gate={}
    for cat in ('real_successor_value_residual','oracle1_corrects_pair','postprocessing_index0_or_risk'):
        xs=[x for x in failures20 if x['model'].startswith('kda') and x['category']==cat]
        gate[cat]=dict(distinct_cases=len({x['case'] for x in xs}),
            models=sorted({x['model'] for x in xs}),
            rescue_cases=len({x['case'] for x in xs if x['rescued']}))
        g=gate[cat]
        g['passes']=g['distinct_cases']>=3 and len(g['models'])==2 and g['rescue_cases']>=2
    selected_module=next((name for name in gate if gate[name]['passes']),None)
    aggregate['decision_gate']=dict(pre_registered_gate=P['decision_gate'],results=gate,
        selected_module=selected_module,decision='GO' if selected_module else 'KILL',
        scope='GO only authorizes recommending a bounded next mechanism validation; no new training or method has been executed')
    def square_a1(model):
        xs=[x for x in allpairs if x['model']==model and x['cell']==5 and x['label']=='A1'
            and x['root_type']!='success-control']
        return '%s: %d/%d挽救、%d个Return改善'%(model,sum(x['rescued'] for x in xs),len(xs),sum(x['delta_real']>EPS for x in xs))
    q1_head='；'.join(square_a1(m['id']) for m in P['models'] if m['backbone']=='kda')
    priority_names={'real_successor_value_residual':'共享未来价值排序/监督与部署合同',
        'oracle1_corrects_pair':'一步后继及下一帧选人/历史转换',
        'postprocessing_index0_or_risk':'评分后处理'}
    priority=priority_names.get(selected_module,'当前候选未通过重复性门槛，停止投入')
    package=BASE/'root_cause_data.jsonl.gz'
    assert not package.exists()
    with gzip.open(package,'wt',compresslevel=6) as out:
        dump(out,provenance);dump(out,aggregate)
        dump(out,{'record_type':'frozen_roots',**frozen})
        for rid,description in history_descriptions.items():
            dump(out,{'record_type':'history_description','root_id':rid,**description})
        comparisons={r['root']['root_id']:r['comparisons'] for r in records}
        for path in sorted((BASE/'work').glob('branches-*.jsonl.gz')):
            for record in read(path):
                if record['record_type']=='counterfactual':
                    record['comparisons']=comparisons[record['root']['root_id']]
                dump(out,record)
        shared_path=BASE/'work/shared-rank.jsonl.gz'
        if shared_path.exists():
            for row in read(shared_path):dump(out,{'record_type':'shared_window_rank',**row})
        for row in input_rows:dump(out,{'record_type':'legal_input_probe',**row})
        for row in actual_oracles.values():dump(out,{'record_type':'real_history_oracle',**row})
        for path in baseline_sources:
            for ep in read(path):
                dump(out,{'record_type':'baseline_episode',**ep})
    successes=[x for x in allpairs if x['rescued']]
    losses=[x for x in allpairs if x['damaged']]
    residual=[x for x in failures20 if x['category']=='real_successor_value_residual']
    corrected=[x for x in failures20 if x['category']=='oracle1_corrects_pair']
    post=[x for x in failures20 if x['category']=='postprocessing_index0_or_risk']
    def unique(items,backbone=None):
        return len({(x['model'],x['cell'],x['case']) for x in items if backbone is None or x['model'].startswith(backbone)})
    lines=[
        '| 问题 | 结论 | 实际证据 | 证据不足之处 |',
        '|---|---|---|---|',
        '| Q1 长期改善 | %s | %d个冻结root全部续跑；A1事先定义，不是事后选最好 | 这些是失败root上的条件结果，不是在线策略收益 |'%(q1_head,len(records)),
        '| Q2 错排来源 | 真实后继认可替代%d对；仍未认可%d对（含并列）；后处理%d对 | 四权重、20人两格、去重改善动作对，每对有三种Q | 不是总体责任百分比；严格错排与并列另列；不执行被mask动作 |'%(len(corrected),len(residual),len(post)),
        '| Q3 KDA独有吗 | KDA真实后继残差root %d，GRU %d | 同样的root规则和单步干预 | 自身轨迹及独立训练权重，不能隔离架构因果效应 |'%(unique(residual,'kda'),unique(residual,'gru')),
        '| Q4 输入还是学习 | 输入重编码的局部因果检验见下表；更深层主要根因未识别 | 最近5人按完整合法历史重编码，命令命中已续跑分支才有后果证据 | 未执行的命令不推断收益；不能由此区分输入容量、学习不足和历史别名 |',
        '| Q5 优先机制 | %s；本轮没有方法级进展 | 预注册重复性门槛见文末，导航损失由真实单步替换检验 | 未开发新算法；不声称论文创新 |'%priority,
        '', '# TTC5冻结权重因果归因结果', '',
        '## 实验合同与验收',
        '- 4090；四份T24、IL50+RL10000权重。零训练、原项目零修改。TTC5、alpha=0.3、80动作、margin=0.2、risk=0.8、gamma=0.99、dt=0.25、[env]时限50秒。',
        '- 20人Circle/Square各500case/model，90000-90499；5人Circle、10人Square各32例/model为次要参考。不使用92000块。',
        '- 完成状态：%s。全部失败%d；无合格root%d，全部保留在失败分母。'%(str(complete),len(frozen['all_failures']),frozen['no_root_count']),
        '- 历史AC3逐步动作、选择及trajectory hash逐位核对；GRU旧91000记录核对终局、return、净距和选择hash。旧历史张量没有存盘，因此不冒充历史张量逐位核验：原驱动重建后再次重放，history/windows/scores逐位相等。',
        '- A0同动作重放必须与原轨迹完全相同。CV作为oracle输入时，80窗口及Value必须逐位不变。首轮np.hypot/math.hypot统计口径错误已保留并修复，不放宽容差。',
        '- crowd_sim/envs/crowd_sim.py:349-355先求ORCA动作，:367-389却用当前行人速度做相对扫掠判碰，:458-461按新动作更新。两套净距都记录；env_swept是该运动口径下全体最小值，原info.dmin在遇到首个碰撞后break，多个重叠时不一定等于全体极小值。环境reward原值不重算、不替换。',
        '- 保留原root固定5人的真坐标oracle作为控制；主归因另外构造真实下一帧TTC5历史，保持真实历史前缀不变。前者逐位复现原oracle；后者才与真正下一步的观测接口一致。终止后不bootstrap，原未mask版本也保留。',
        '- 归因口径修正：原冻结协议的pinned oracle把下一帧选人变化留在了Value残差；因此并列保存原定义与actual-history定义，并以三项分解单列选人/历史转换，不改root、候选、续跑结果或判据。',
        '- 未对冻结原权重做网络优化；每条分支的唯一执行差别是root命令，平滑上一动作同步更新为真正执行命令。',
        '- Q_real是这个冻结仿真世界、这个checkpoint续跑下的实际G；不是对所有可能隐藏世界取期望后的Q。实测错排证明该世界有动作损失，不自动证明网络凭已有合法输入就应当预知它。',
        '- 以下终局按环境规则统计；与实际ORCA更新轨迹一致的几何净距另存。某分支即使环境判成功，若actual_swept<0，也不能据此称其物理上无碰撞；各候选summary单列此数。',
        '', '### 冻结checkpoint（不是原optimized权重）',
        '| 模型 | 训练seed | SHA256 |',
        '|---|---:|---|']
    for m in P['models']:
        lines.append('| %s | %s | %s |'%(m['id'],m['seed'],m['weight_sha256']))
    lines += [
        '', '## 完整失败分母',
        '| 模型 | cell | episodes | success | collision | timeout | 合格失败root | 无root |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for (model,cell),stat in sorted(totals.items()):
        d=[r for r in frozen['all_failures'] if r['model']==model and r['cell']==cell]
        lines.append('| %s | %d | %d | %d | %d | %d | %d | %d |'%(model,cell,stat['n'],stat['success'],stat['collision'],stat['timeout'],sum(x['has_root'] for x in d),sum(not x['has_root'] for x in d)))
    lines += ['', 'cell0=5-circle；cell3=10-square；cell4=20-circle；cell5=20-square。',
        '', '## 各预设替代动作独立效果',
        '以下是诊断root的条件结果，不是一个部署策略的SR。未找到root和未匹配成功对照都没有被丢弃。',
        'Return改善可能仅是碰撞延后或不舒适惩罚减少；终局挽救单列。完整outcome_transitions、改善但仍失败、共同成功的时间差均在summary，不把延后失败叫作到达改善。',
        '| 模型 | cell | root类型 | 候选 | n/不同动作 | Return升/降/等 | TO→成功 | CR→成功 | 成功→失败 | 平均Δepisode Return [95% CI] |',
        '|---|---:|---|---|---:|---|---:|---:|---:|---|']
    for key,s in sorted(summary.items()):
        model,cell,kind,label=key.split('|');lo,hi=s['ci_delta_episode_return']
        lines.append('| %s | %s | %s | %s | %d/%d | %d/%d/%d | %d | %d | %d | %+.6f [%+.6f,%+.6f] |'%(model,cell,kind,label,s['n'],s['distinct_interventions'],s['return_improved'],s['return_worsened'],s['return_equal'],s['timeout_to_success'],s['collision_to_success'],s['success_to_failure'],s['mean_delta_episode_return'],lo,hi))
    lines += ['', '每格内每个episode最多一个root，区间按episode重采样；跨两格的区间按case聚类，保存在summary记录。两个训练seed始终分列，不能从n=2推总体训练方差。Return计数数值容差=1e-6。',
        '', '## 事后headroom上界（不是部署策略）',
        '| 模型/cell | 全部失败 | 任一预设动作提高Return | 任一预设动作挽救 |',
        '|---|---:|---:|---:|']
    for key,s in mechanism_summary.items():
        lines.append('| %s | %d | %d | %d |'%(key,s['completed_failure_roots'],s['any_improvement_oracle_roots'],s['any_rescue_oracle_roots']))
    lines += ['', '三个动作中事后最好结果只给上界；独立A1/A2/A3的效果以上表为准。',
        '', '## 【已确认的因果结果】逐例证据',
        '下面列出所有终局挽救及所有成功破坏。其余正负Return结果全部在压缩原始记录comparisons字段。动作编号均对应实际alpha平滑命令。',
        '| rootID | 模型/cell/case | t | 候选/动作 | V原→替代 | Qpred原→替代 | Qoracle原→替代 | Qreal原→替代 | 终局 | 到达/终止时间原→替代 | 分类 |',
        '|---:|---|---:|---|---|---|---|---|---|---|---|']
    for x in sorted(successes+losses,key=lambda x:(x['root_id'],x['label'])):
        a=x['original_prediction'];b=x['alternative_prediction']
        lines.append('| %d | %s/%d/%d | %.2f | %s:%d→%d | %.9f→%.9f | %.9f→%.9f | %.9f→%.9f | %.6f→%.6f | %s→%s | %.2f→%.2f | %s |'%(x['root_id'],x['model'],x['cell'],x['case'],x['step']*.25,x['label'],x['original_action'],x['alternative_action'],a['predicted_value'],b['predicted_value'],a['q_pred'],b['q_pred'],a['q_oracle1_actual_history_terminal_corrected'],b['q_oracle1_actual_history_terminal_corrected'],x['original_root_return'],x['new_root_return'],x['original_outcome'],x['new_outcome'],x['original_time'],x['new_time'],x['category']))
    lines += ['', '每一行的数据位置：root_cause_data.jsonl.gz中record_type=counterfactual且root.root_id等于表中ID，comparisons[候选]；完整80候选在root.snapshot，后续轨迹在branches[].steps。',
        '', '### 否定反例：有一步推进机会但本组三个动作都未挽救',
        '每个模型按预先case顺序列最早两例，完整否定结果均保留。不由此断言其余动作或多步控制无解。',
        '| rootID | 模型/case | t | 候选/动作 | Qpred原→新 | Qreal原→新 | 终局 |',
        '|---:|---|---:|---|---|---|---|']
    for model in [m['id'] for m in P['models']]:
        negative=sorted([r for r in records if r['root']['model']==model and r['root']['cell']==5
            and r['root']['type']!='success-control' and not any(x['rescued'] for x in r['comparisons'].values())],
            key=lambda r:r['root']['case'])[:2]
        for record in negative:
            for label,x in record['comparisons'].items():
                a=x['original_prediction'];b=x['alternative_prediction']
                lines.append('| %d | %s/%d | %.2f | %s:%d→%d | %.9f→%.9f | %.6f→%.6f | %s→%s |'%(
                    x['root_id'],model,x['case'],x['step']*.25,label,x['original_action'],x['alternative_action'],
                    a['q_pred'],b['q_pred'],x['original_root_return'],x['new_root_return'],x['original_outcome'],x['new_outcome']))
    lines += [
        '', '## 错排分解',
        '| 模型 | 类别 | 改善动作对数 | 涉及独立case数（同cell也按case聚类） |',
        '|---|---|---:|---:|']
    for model in [m['id'] for m in P['models']]:
        for cat in ['hard_filter','postprocessing_index0_or_risk','oracle1_corrects_pair','real_successor_value_residual']:
            xs=[x for x in failures20 if x['model']==model and x['category']==cat]
            lines.append('| %s | %s | %d | %d |'%(model,cat,len(xs),len({x['case'] for x in xs})))
    lines += ['', '预设A1的episode级互斥分类（包括未改善，不用事后最好动作选择类别）：',
              '| 模型/cell | 完成失败root | A1无改善 | A1过滤 | A1后处理 | A1真一步纠正 | A1真一步仍错排 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for key,s in mechanism_summary.items():
        cats=s['A1_exclusive_categories']
        values=[cats.get(k,0) for k in ['no_return_improvement','hard_filter','postprocessing_index0_or_risk',
                                      'oracle1_corrects_pair','real_successor_value_residual']]
        lines.append('| %s | %d | %s |'%(key,s['completed_failure_roots'],' | '.join(map(str,values))))
    lines += ['', '严格错排与数值并列分开（绝对gap<=1e-9仅作数值并列标记，不改变原argmax）：',
        '| 模型/cell | 原raw严格反向 | 原raw并列但G改善 | 原raw和实际oracle均严格反向 | 实际oracle并列但G改善 |',
        '|---|---:|---:|---:|---:|']
    for key,s in mechanism_summary.items():
        lines.append('| %s | %d | %d | %d | %d |'%(key,s['strict_raw_misrank_pairs'],s['tied_raw_improved_pairs'],
            s['strict_raw_and_actual_oracle_misrank_pairs'],s['actual_oracle_tied_improved_pairs']))
    lines += ['', '### 原pinned口径与实际下一帧口径',
        '按改善动作对去重；数字不是因果责任百分比。改变类别只来自离线后继历史修正，真实续跑标签完全不变。',
        '| 模型/cell | 原pinned分类→实际history分类 | 动作对数 |',
        '|---|---|---:|']
    for key,s in mechanism_summary.items():
        for transition,n in sorted(s['category_changes'].items()):
            lines.append('| %s | %s | %d |'%(key,transition,n))
    lines += ['', '三个可加的数值项：Δpred−Δreal = (Δpred−Δpinned) + (Δpinned−Δactual) + (Δactual−Δreal)。依次为物理/即时reward替换、下一帧选人/历史转换、剩余未来价值误差。终止处理另外标记；不能把各项绝对值占比包装成模块因果责任。',
        '| 模型/cell | 真后继纠正且root=0 | 仅真坐标可纠正 | 仅真reward可纠正 | 平均选中行人位置误差中位/m |',
        '|---|---:|---:|---:|---:|']
    for key,s in mechanism_summary.items():
        dist=s['numerical_components']['successor_position_error_selected_mean']
        lines.append('| %s | %d | %d | %d | %s |'%(key,s['oracle_corrected_at_initial_step'],s['physical_only_corrects'],s['real_reward_only_corrects'],
            'NA' if dist is None else '%.6f'%dist['p50']))
    lines += ['', '## 合法输入反事实：完整历史重编码',
              '只在已确认真实回报改善且raw排序未认可的root测试既有nearest5规则；每一历史帧从真实记录重编码，当前帧和80后继共用当前选人集合。原TTC5重构须逐位复现windows、values与argmax。reward、clearance与commands须完全不变。',
              '仅当新的argmax等于预先执行过的A0-A3命令，才引用该分支的导航后果。评分后恢复TTC5原历史；这是root一次性输入干预，不是将nearest5全程部署。命中已测动作的子集不代表全部输入干预，也不代表可部署收益。',
              '| 模型 | root | 动作改变 | 已有执行证据 | 改变但后果未知 | Return升/降 | 挽救/破坏 |',
              '|---|---:|---:|---:|---:|---|---|']
    for model,s in input_summary.items():
        lines.append('| %s | %d | %d | %d | %d | %d/%d | %d/%d |'%(model,s['roots'],s['changed'],
            s['changed_with_exact_pretested_command'],s['changed_outcome_unknown'],s['return_improved'],
            s['return_worsened'],s['rescued'],s['damaged']))
    lines += ['', '以下仅列出动作确实改变且命中已执行分支的输入干预，保留正负结果；未命中分支的评分翻转没有导航收益结论。',
        '| rootID | 模型/cell/case | t | 原动作→输入重编码后动作 | 原raw gap / 重编码raw gap | Qreal原→新 | 终局 |',
        '|---:|---|---:|---|---|---|---|']
    for row in input_rows:
        if not row['changed'] or not row['outcome_verified']:continue
        root=roots[row['root_id']];e=row['existing_branch_evidence'];i=root['labels']['A0'];j=e['candidate_index']
        orig=row['scoring']['ttc5']['raw_scores'];new=row['scoring']['nearest5']['raw_scores']
        lines.append('| %d | %s/%d/%d | %.2f | %d→%d | %+.9f / %+.9f | %.6f→%.6f | %s→%s |'%(
            row['root_id'],row['model'],row['cell'],row['case'],row['step']*.25,i,j,orig[j]-orig[i],new[j]-new[i],
            e['original_root_return'],e['new_root_return'],e['original_outcome'],e['new_outcome']))
    lines += ['', '每例输入证据位于record_type=legal_input_probe，按root_id连接counterfactual；不把输入扰动有响应等同因果收益。']
    lines += ['', '## 同状态、同24帧的跨网络排序',
        '参考网络真实改善的动作对，给四个网络相同windows，原网络分数逐位复现。以下只测排序差异；真回报来自参考网络的后续策略，不是另一个网络的Q，因此不能宣称替换网络即可得到该收益。',
        '| 参考轨迹模型 | 评分模型 | 已有改善动作对 | 仍不偏好替代动作 | 偏好替代动作 |',
        '|---|---|---:|---:|---:|']
    for reference,others in aggregate['same_state_ranking_descriptive'].items():
        for other,s in others.items():
            lines.append('| %s | %s | %d | %d | %d |'%(reference,other,s['pairs'],s['still_raw_nonpositive'],s['prefers_reference_better_action']))
    lines += ['', '- hard_filter为0不能排除过滤器损失：A1/A2/A3按原规则选安全候选，设计未执行被mask动作。',
        '- oracle1_corrects_pair表示原预测排序被真实下一帧（含实际TTC5观测选择）纠正；这可能来自坐标/速度、即时reward或选人转换。physical_prediction_reward_error和selection_history_transition_error分别保存，不能笼统称为CV运动误差。未执行oracle策略，不声称预测模块的闭环收益。',
        '- real_successor_value_residual表示使用真实下一帧观测历史和真实即时reward后，仍未正确排列已验证回报更好的动作；定位到后续价值估计链路，但不能在输入、训练目标/覆盖、历史表示、架构之间分摊责任。',
        '- Q2及本表按实际不同的动作对去重；同一动作同时具有A1/A2标签只计一次。各模块类别在动作对层面互斥，但同一root可以有多个类别，不把动作对比例冒充互斥episode因果百分比。',
        '- 终止根动作同时记录继承的未mask bootstrap及terminal-corrected版本；不能把终止后仍bootstrap造成的误差归为一般预测误差。',
        '', '## 【有证据支持但未完成隔离】',
        '输入限制为当前5人场景token；历史24帧可能曾包含其他行人，不能称另外15人永久失明。集合切换、未选中的约束行人均仅作描述，不能凭相关性认定原因。',
        '成功对照按同model/cell、时间±2秒、目标距离±1m、安全数±16、2m内人数±2匹配，不重复用同一成功episode；超时对照还要求同样低进度机会。匹配不到的root明确保留null。',
        '未匹配对照的失败root：'+json.dumps({k:v['unmatched'] for k,v in aggregate['matched_success_controls'].items()},ensure_ascii=False),
        '', '### 亲自执行的训练合同单元核验',
        '【已验证通过】独立加载KDA419与GRU419，给原训练路径相同的3个真实5人状态。强制epsilon=1时历史长度每次0→0，强制epsilon=0时依次0→1、1→2、2→3。代码在mamba_rl.py:765-768随机返回，:777才append历史；explorer.py:466与train.py:2568-2600仍将完整实际帧写入MC经验池。',
        '【未验证】这处在线采样历史与监督窗口不一致是否造成本轮20人损失，尚未做可隔离重训。训练候选不平滑、评测候选alpha=0.3且另加mask/risk，也只是已核验协议差异，不能直接当成根因。首次单元测试误用了仅允许eval的TTC包装，失败记录保留；随后绑定原训练方法通过。',
        '', '## 【已经排除的假设】仅限实际证据覆盖范围',
        '- 原TTC5未生效、选错权重或96动作污染：真实路径、权重SHA、80动作、历史重建、历史逐步trace以及A0完整重放均通过。',
        '- 被验证的安全替代动作不存在于原控制支持：这些命令本来就在同一80个alpha=0.3候选内，实际执行可改善的个例不需要扩大动作空间。未挽救病例不能作同样推断。',
        '- 真一步后继仍错排的动作对：仅修复一步后继不足以纠正该对的排序。不能将此负结论扩大为预测/多步规划整体无效。',
        '- 若GRU也有同类真后继残差，则“KDA是这些错误的必要条件”被反例否定；KDA相对发生率与内部因果效应仍未隔离。',
        '', '## 【现有实验不可识别】',
        '1. 同一冻结网络的真实后继残差，不能唯一地区分信息缺失、有限历史别名、已见信息未学好和训练分布覆盖不足。',
        '2. 最近5人影子输入替换只对命中已执行分支的命令有因果后果证据；不重训，不能给输入缺失与学习不足编造责任占比。',
        '3. 所测候选无效不证明完整80动作、停止动作、其他平滑、多步控制都无效；也不证明场景无解。',
        '4. 如GRU也出现同类实际错排，只能否定“KDA是该现象必要条件”；不证明两架构的发生率相同或架构无作用。',
        '', '## 最小补充实验合同（不自动启动）',
        '唯一优先补充：等预算2×2冻结表示读出对照，以区分“已有输入可修复”与“增加合法观测才有增量”。复用本轮20人Square的全部预设A0-A3分支及成功对照，不再采环境、不挑挽救例。按case末位0/1留出，其余训练；同case跨seed不得拆分，两个seed结果分列。',
        '因素一：原TTC5历史表示，或全部合法行人的历史表示；后者只将现有关系编码器的5人截断/reshape推广为N，参数全部冻结，不增网络宽度。两者都投到原256维表示。必须独立preflight核对N=5逐位parity。因素二：等样本的MC回报回归，或同root成对回报排序监督；每臂只训练相同257参数线性Value读出，初始化、50epoch、Adam lr=1e-3固定，不扫参数。以(G_root-r_pred)/gamma为回归靶，排序靶来自真实Delta_G；所有未来信息只作label。',
        '只在留出root从预先固定A0-A3中选动作，引用已执行分支的真实后果，故无需新续跑；不是全80动作在线方法。主比较为TTC5排序读出−TTC5回归读出，以及全人群排序读出−TTC5排序读出；case聚类区间，两项Bonferroni alpha=.025。GO要求两个seed均有正Return增量、区间下界>0、挽救保留且成功破坏不增加；任何复杂读出被原线性头或即时进度A2追平则KILL，不追加调参。',
        '同输入能够修复，说明学习/监督有可用修复空间；只有全人群有增量，支持额外观测在该受控表示下有作用；交互则是联合机制。四臂都失败，只能KILL该可修复性候选，不能证明信息无价值。因为使用20人诊断标签，该补充不能被包装成5人训练→20人泛化方法，也不能当论文性能。若用户坚持任何20人监督都禁止，则该输入/学习隔离在当前合同下不可执行。',
        '资源上限：4090，每seed四个读出、共八个；表示重编码及拟合总墙钟90分钟、无IL/RL、无新环境episode，超限就停止并保留部分证据；实际吞吐由小样本preflight核实。全骨干重训不是这个最小对照的必要条件。本轮不启动任何读出训练。',
        '', '## 已有方法边界（亲读原文/源码，非复现）',
        '- SARL官方MultiHumanRL.predict第31-48行提供环境one-step lookahead和CV两条路径，随后r+gammaV；其输入遍历所有human而非固定5槽。它覆盖注意力聚合/可变人数和一步评价思路，不保证解决本轮冻结KDA的特定错排。[官方源码](https://raw.githubusercontent.com/vita-epfl/CrowdNav/master/crowd_nav/policy/multi_human_rl.py)',
        '- RGL官方ModelPredictiveRL第177-275行实现状态预测、动作剪枝、多步Value planning；第277-324行用相对扫掠距离估计reward。因此“增加预测/多步前瞻”或“扫掠reward”不能独立当新贡献。[官方源码](https://raw.githubusercontent.com/ChanganVR/RelationalGraphLearning/master/crowd_nav/policy/model_predictive_rl.py)',
        '- CAMRL原文III-B使用历史状态编码与Mamba价值网络，不能因本轮残差就宣称现有时序价值方法都没处理历史；其公布评价不等同当前TTC5反事实验证。[原文](https://arxiv.org/html/2408.02661v1)',
        '- Social-NCE原文III节已有面向危险未来事件的负采样与对比表示监督，且用于导航RL。故“加负样本/对比损失来区分危险行为”本身不是新贡献；本轮也没有证明缺少失败样本是原因。[原文](https://arxiv.org/html/2012.11717v3)',
        '- 未运行SARL/RGL/CAMRL的同root官方checkpoint；不得宣称它们已解决或一定无法解决本轮案例。没有建立新的算法贡献。',
        '', '## 研究裁决',
        '预注册重复性门槛结果：'+json.dumps(aggregate['decision_gate'],ensure_ascii=False),
        '本轮只取得冻结系统的动作因果与模块定位证据，没有方法级进展；未开发或验证独立新算法。GO只表示值得下一项有限机制验证，不自动启动训练或复活已否决候选。',
        '', '## 复现入口与资产',
        '独立脚本audit.py、run.py、reference.py；manifest.json冻结模型、原源码和合同；root_cause_data.jsonl.gz收录完整baseline、root、反事实、分解和源证据。原项目及权重结束SHA复验通过。',
        '命令：PYTHONPATH=/root/kda-vl-deps/flash-linear-attention /root/miniconda3/envs/mamba/bin/python audit.py preflight --model kda-s419-t24；完整运行由run.py按冻结次序执行。已存在输出不覆盖。',
        'root_cause_data.jsonl.gz SHA256: '+sha(package),
    ]
    (BASE/'ROOT_CAUSE_RESULT.md').write_text('\n'.join(lines)+'\n')
    # The next step reads this same packaged summary; no separate result-report proliferation.
    print(json.dumps(dict(status='PASS',roots=len(roots),complete=complete,
        failure_episodes=len(frozen['all_failures']),decision=aggregate['decision_gate'],
        report=str(BASE/'ROOT_CAUSE_RESULT.md'),report_bytes=(BASE/'ROOT_CAUSE_RESULT.md').stat().st_size,
        data_bytes=package.stat().st_size,data_sha256=sha(package)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
