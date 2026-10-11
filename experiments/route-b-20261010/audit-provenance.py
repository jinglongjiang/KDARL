"""File-level provenance inventory with explicit missing-evidence fields."""
import sys
sys.dont_write_bytecode=True
import configparser
import csv
import gzip
import json
from functools import lru_cache
from pathlib import Path
import re
import time
import zipfile
import numpy as np
from common import BASE, WORK, PROJECT, save, sha, specs

NAMES=['kda-vl-optimized-20261007','kda-selection-20261008','kda-history-20261009',
       'kda-history-replication-20261010','kda-lookahead-20261009','mamba-ttc5-20261009',
       'vl-timeout-analysis-20261008','bayes-fix-20261008','bayes-decomp-20261009',
       'bayes-incr-20261010','bayes-horizon-20261010']
FIELDS={'run_label','backbone','temporal_backbone','seed','training_seed','train_seed',
        'checkpoint_sha256','weight_sha256','config_sha256','selection','selection_rule','rule',
        'smoothing','alpha','action_smoothing','grid_actions','candidate_actions',
        'case_base','n_cases','n_cases_per_cell','seq_len','t','episode','n_episodes',
        'navigation_training','probe_training','n_steps','status','exit_code'}


@lru_cache(maxsize=2048)
def read_json(path):
    if path.stat().st_size > 100*1024**2:return None
    try:return json.loads(path.read_text())
    except (ValueError,UnicodeError,OSError):return None


def walk_meta(obj,loc='',depth=0):
    out=[]
    if not isinstance(obj,dict) or depth>4:return out
    for key,value in obj.items():
        here=loc+'/'+str(key)
        if key in FIELDS and isinstance(value,(str,int,float,bool,type(None))):
            out.append(dict(key=key,value=value,location=here))
        if isinstance(value,dict) and key not in ('episodes','per_step','frames','results','files'):
            out.extend(walk_meta(value,here,depth+1))
    return out


def resolve_roots():
    out={}
    bases=[WORK,Path('/home/abc/temp')]+sorted(Path('/home/abc').glob('4090_backup_*/root'))
    for name in NAMES:
        roots=[b/name for b in bases if (b/name).is_dir()]
        if name=='kda-vl-optimized-20261007':
            roots += [PROJECT/'artifacts/vl-optimized-20261007']
        if name=='bayes-fix-20261008':
            roots += [p for p in [WORK/'CrowdNav(20270731_backup2)/CrowdNav/crowd_nav/runs/bayes-fix-20261008'] if p.is_dir()]
        out[name]=sorted(set(roots))
    return out


def context(path,root):
    evidence=[]
    for parent in [path.parent]+list(path.parent.parents):
        if parent!=root and root not in parent.parents:break
        for name in ('protocol.json','frozen-manifest.json','formal-version.json','run-version.json','training-completed.json'):
            p=parent/name
            if p==path or not p.is_file():continue
            data=read_json(p)
            for row in walk_meta(data):
                row['source']=str(p);evidence.append(row)
        if parent==root:break
    return evidence


def extract(path):
    fields=[];meta=[];cases=[];sample=None;n=None;note=None
    if path.name.endswith('.jsonl.gz'):
        n=0
        with gzip.open(path,'rt') as f:
            for line in f:
                n+=1
                row=json.loads(line)
                if sample is None:sample=row
                if isinstance(row,dict):
                    if isinstance(row.get('case'),int):cases.append(row['case'])
                    for key in row:
                        if key not in fields:fields.append(key)
        if isinstance(sample,dict):
            meta=walk_meta(sample)
        note='JSONL counted completely; case range/field union from all rows; other metadata from first record, not all-record constancy'
    elif path.suffix=='.json' or path.name.endswith('.json.progress'):
        sample=read_json(path)
        if isinstance(sample,dict):
            fields=list(sample);meta=walk_meta(sample)
            if isinstance(sample.get('case'),int):cases.append(sample['case'])
            for key in ('episodes','records','results','frames','rows'):
                rows=sample.get(key)
                if isinstance(rows,list):
                    n=len(rows)
                    if rows and isinstance(rows[0],dict):
                        fields += [key+'.'+str(k) for k in rows[0]]
                        commands=rows[0].get('commands')
                        if isinstance(commands,list) and commands and isinstance(commands[0],dict):
                            fields += [key+'.commands.'+str(k) for k in commands[0]]
                        cases += [r.get('case',r.get('case_index')) for r in rows if isinstance(r,dict)
                                  and isinstance(r.get('case',r.get('case_index')),int)]
                        for meta_key in FIELDS:
                            observed=[]
                            for r in rows:
                                value=r.get(meta_key) if isinstance(r,dict) else None
                                if isinstance(value,(str,int,float,bool)) and value not in observed:observed.append(value)
                            if observed:meta.append(dict(key=meta_key,value=observed[0] if len(observed)==1 else observed,
                                                          location=key+'/*/'+meta_key))
                        labels=sorted({r['run_label'] for r in rows if isinstance(r,dict) and isinstance(r.get('run_label'),str)})
                        if labels:meta.append(dict(key='run_label',value=labels,location=key+'/*'))
                    break
            if isinstance(sample.get('summary'),dict) and isinstance(sample['summary'].get('case'),int):
                cases.append(sample['summary']['case'])
        elif isinstance(sample,list):fields=['list'];n=len(sample)
    elif path.suffix=='.csv':
        labels=set();cases=[];n=0
        with path.open(newline='') as stream:
            reader=csv.DictReader(stream);fields=reader.fieldnames or []
            for row in reader:
                n+=1
                if row.get('run_label'):labels.add(row['run_label'])
                for key in ('case','case_id','test_case'):
                    if row.get(key):
                        try:cases.append(int(row[key]))
                        except ValueError:pass
        if labels:meta.append(dict(key='run_label',value=sorted(labels),location='all CSV rows'))
        note='CSV seed column is not assumed to be training seed; explicit protocol required'
    elif path.suffix=='.npz':
        arrays={}
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                with z.open(name) as stream:
                    version=np.lib.format.read_magic(stream)
                    shape,order,dtype=(np.lib.format.read_array_header_1_0(stream) if version==(1,0)
                        else np.lib.format.read_array_header_2_0(stream))
                arrays[name]=dict(shape=list(shape),dtype=str(dtype))
        fields=list(arrays);note='NPZ headers only; array values are not interpreted as metadata';sample=arrays
    elif path.suffix=='.log':
        with path.open(errors='replace') as stream:
            head=stream.read(131072)
            if path.stat().st_size>131072:stream.seek(max(0,path.stat().st_size-131072));tail=stream.read()
            else:tail=''
        fields=['unstructured_log'];note='Head/tail scan, not proof that every earlier error is absent'
        if 'Traceback (most recent call last)' in head+tail:
            meta.append(dict(key='traceback_observed',value=True,location='log head/tail'))
    if n is None and isinstance(sample,dict) and isinstance(sample.get('steps'),int):
        n=sample['steps'];note=(note or '')+'; sample_count is declared decision steps, not episodes'
    return dict(fields=fields,metadata=meta,case_range=[min(cases),max(cases)] if cases else None,
                sample_count=n,extraction_note=note)


def values(meta,keys):
    result=[]
    for row in meta:
        if row['key'] in keys and row['value'] not in result:result.append(row['value'])
    return result or None


def noise():
    old=json.loads((WORK/'kda-history-20261009/summary.json').read_text())['main']
    new=json.loads((WORK/'kda-history-replication-20261010/summary.json').read_text())['arms']
    rows={};details={}
    for metric in ('SR','SR_5','SR_10','SR_20','CR','TO','Return','NavTime'):
        diffs=[];levels=[];details[metric]={}
        for arm in ('kda-t12','kda-t24','gru-t12','gru-t24'):
            if metric.startswith('SR_'):
                n=int(metric.split('_')[1]);idx={5:(0,1),10:(2,3),20:(4,5)}[n]
                a=np.mean([old[arm]['cells'][str(i)]['rates_percent']['success'] for i in idx])
                b=np.mean([new[arm]['cells'][str(i)]['rates_percent']['success'] for i in idx])
            else:
                x,y=old[arm]['overall'],new[arm]['overall']
                field={'SR':'success','CR':'collision','TO':'timeout'}.get(metric)
                a=x['rates_percent'][field] if field else x['mean_return' if metric=='Return' else 'success_time_mean']
                b=y['rates_percent'][field] if field else y['mean_return' if metric=='Return' else 'success_time_mean']
            diffs.append(abs(b-a));levels.extend([a,b]);details[metric][arm]=dict(seed419=a,seed443=b,absolute_difference=abs(b-a))
        rows[metric]=dict(mean_absolute_difference=float(np.mean(diffs)),max_absolute_difference=float(max(diffs)),
            mean_level=float(np.mean(levels)),unit='seconds' if metric=='NavTime' else 'return' if metric=='Return' else 'percentage points')
    return dict(status='已验证通过',definition='same backbone and T, seed419 vs443; mean/max over four matched arm types',
        training_seeds=2,arm_types=4,rows=rows,details=details,
        limit='Observed absolute differences, NOT an estimated population noise floor or architecture-level variance',
        sources=[str(WORK/'kda-history-20261009/summary.json'),str(WORK/'kda-history-replication-20261010/summary.json')])


def main():
    roots=resolve_roots();entries=[];missing=[];seen=set()
    checkpoints={s['weight_sha256']:s for s in specs()}
    configs={}
    for s in specs():
        cfg=configparser.ConfigParser();assert cfg.read(s['config'])
        configs[s['config_sha256']]=dict(path=s['config'],
            alpha=cfg.getfloat('eval_protocol','action_smoothing'))
    for name,scopes in roots.items():
        if not scopes:missing.append(name)
        for root in scopes:
            for path in sorted(root.rglob('*')):
                if path.is_symlink():continue
                if not path.is_file() or path in seen:continue
                if path.suffix not in ('.json','.csv','.npz','.log') and not path.name.endswith(('.jsonl.gz','.json.progress')):continue
                if any(part in ('code','__pycache__','.git') for part in path.relative_to(root).parts[:-1]):continue
                seen.add(path)
                try:
                    item=extract(path);direct=item['metadata'];ctx=context(path,root)
                    item.update(path=str(path),scope=name,bytes=path.stat().st_size,sha256=sha(path),context_evidence=ctx)
                    meta=direct+ctx
                    item['run_label']=values(direct,{'run_label'})
                    item['backbone']=values(direct,{'backbone','temporal_backbone'})
                    item['checkpoint_sha256']=values(direct,{'checkpoint_sha256','weight_sha256'})
                    item['training_seed']=values(direct,{'training_seed','train_seed'})
                    hashes=item['checkpoint_sha256'] or []
                    matched=[checkpoints[h] for h in hashes if isinstance(h,str) and h in checkpoints]
                    if matched and len(matched)==len(hashes):
                        item['verified_checkpoint_links']=[dict(id=s['id'],weight=s['weight'],
                            sha256=s['weight_sha256'],seed=s['seed'],backbone=s['backbone']) for s in matched]
                        if item['training_seed'] is None:item['training_seed']=sorted({s['seed'] for s in matched})
                        if item['backbone'] is None:item['backbone']=sorted({s['backbone'] for s in matched})
                    if item['training_seed'] is None:
                        item['training_seed_context_candidates']=values(ctx,{'seed','training_seed','train_seed'})
                    item['selection_rule']=values(direct,{'selection','selection_rule','rule'})
                    item['alpha']=values(direct,{'alpha','smoothing','action_smoothing'})
                    config_hashes=values(direct,{'config_sha256'}) or []
                    matched_configs=[configs[h] for h in config_hashes if isinstance(h,str) and h in configs]
                    if item['alpha'] is None and matched_configs and len(matched_configs)==len(config_hashes):
                        item['alpha']=[c['alpha'] for c in matched_configs]
                        item['verified_config_links']=matched_configs
                    item['action_definition']=values(direct,{'grid_actions','candidate_actions'})
                    item['declared_context_values']={k:values(ctx,ks) for k,ks in {
                        'checkpoint_sha256':{'checkpoint_sha256','weight_sha256'},'selection':{'selection','selection_rule','rule'},
                        'alpha':{'alpha','smoothing','action_smoothing'},'actions':{'grid_actions','candidate_actions'}}.items()}
                    if name in ('kda-history-20261009','kda-history-replication-20261010','bayes-fix-20261008','kda-vl-optimized-20261007'):
                        item['evidence_class']='B';item['class_reason']='Comparing independently trained model weights includes training variation'
                    else:
                        item['evidence_class']='A-candidate';item['class_reason']='Frozen-navigation diagnostic family; pairing/hash must be checked per contrast'
                    if name=='bayes-incr-20261010':
                        item['evidence_class']='A-navigation/B-probes'
                        item['class_reason']='Navigation weights frozen but A1/A2/B/P probes trained; probe comparisons include probe training variation'
                    if item['run_label'] and any('bayes' in str(x).lower() for x in item['run_label']):
                        item['line']='Bayesian (explicit run_label)'
                    else:item['line']='Bayesian' if name.startswith('bayes') else 'KDA/GRU/Mamba-VL'
                    has_error=any(r['key']=='traceback_observed' and r['value'] for r in direct)
                    failed=any(r['key']=='exit_code' and isinstance(r['value'],int) and r['value']!=0 for r in direct)
                    failed=failed or any(r['key']=='status' and str(r['value']).lower() in ('error','failed','fail') for r in direct)
                    item['status']='已确认失败' if has_error or failed else '已验证通过' if item['fields'] else '未验证'
                    item['status_scope']='file parse/provenance extraction only; NOT scientific success or full metadata completeness'
                    item['step_command_fields_observed']=[f for f in item['fields'] if
                        'commands.command' in f or f in ('executed_command','candidate_commands')]
                    item['missing_evidence']=[key for key in ('run_label','backbone','checkpoint_sha256','training_seed',
                        'case_range','action_definition','selection_rule','alpha','sample_count') if item.get(key) is None]
                except Exception as exc:
                    item=dict(path=str(path),scope=name,status='未验证',error=repr(exc),fields=[])
                entries.append(item)
    n=noise()
    groups={}
    for e in entries:
        if e.get('sha256'):groups.setdefault(e['sha256'],[]).append(e['path'])
    output=dict(status='已验证通过',scopes={k:[str(p) for p in v] for k,v in roots.items()},
        missing_scopes=missing,entries=entries,observed_cross_seed_variation=n,
        duplicate_content_groups=[dict(sha256=h,paths=p) for h,p in groups.items() if len(p)>1],
        unique_content_files=len(groups),
        classes=dict(A='Same frozen weight, paired inputs/cases: isolates intervention, not finite-sample/runtime uncertainty',
            B='Independent training comparisons: include training variation',
            caveat='Classification belongs to a contrast, not an entire directory; A-candidate is not verified pairing'),
        ambiguity_policy='Missing metadata remains unknown; context candidates are not silently promoted to verified per-file facts',
        known_withdrawal='30-column bayes-optimized-rl10000 CSV belongs to Bayesian line; never substitute for KDA trajectories')
    save(BASE/'p2-provenance.json',output)
    lines=['# 两条线来源审计','','状态：已验证通过（文件解析与来源提取，不代表每条元数据齐全）。',
        'A类按同一冻结权重的配对干预定义；B类包含独立训练差异。bayes-incr另含新训练探针，必须单列。',
        '缺失字段保留为空；上下文候选见JSON，不作为已确认事实。', '',
        '## 同臂跨seed观察差异','','仅两个seed；下面不是已估计的噪声地板。',
        '| 指标 | 四种同臂配对的平均绝对差 | 最大绝对差 | 单位 |','|---|---:|---:|---|']
    for k,v in n['rows'].items():lines.append('| %s | %.5f | %.5f | %s |'%(k,v['mean_absolute_difference'],v['max_absolute_difference'],v['unit']))
    lines += ['', '## 文件级总表', '', '| 路径 | 类别 | run/backbone | 权重SHA | 训练seed | case范围 | 动作 | 选择 | α | 样本量 | 字段 | 状态 |',
              '|---|---|---|---|---|---|---|---|---|---:|---|---|']
    def fmt(x):return '缺失证据' if x is None else json.dumps(x,ensure_ascii=False).replace('|','/')
    for e in entries:
        row=[e['path'],e.get('evidence_class'),[e.get('run_label'),e.get('backbone')],e.get('checkpoint_sha256'),
             e.get('training_seed'),e.get('case_range'),e.get('action_definition'),e.get('selection_rule'),
             e.get('alpha'),e.get('sample_count'),e.get('fields'),e['status']]
        lines.append('| '+' | '.join(fmt(x) for x in row)+' |')
    (BASE/'p2-provenance.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',files=len(entries),missing_scopes=missing,noise=n['rows']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
