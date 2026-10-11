#!/usr/bin/env python3
"""Independent, read-only B raw-shard acceptance. No rollout/network imports.

JSON is printed to stdout only. Missing data stays INCOMPLETE, even if an old
summary says COMPLETE. This validates execution/data, not the scientific claims.
"""
import argparse
import gzip
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
EXPECTED_DRIVER_SHA = '8b4ee9d18c5014e43d30fb7f8944625dae4104b23c0713c4bdddaee19099dece'
EXPECTED_CONTRACT_SHA = 'c89f0ca680e17d77470ee572f8608567aa00e4240cb20be1c08e47b979ed6ac2'
TIMING_KEYS = {'latency_ms_p50', 'latency_ms_p95'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))


def rows(path):
    with gzip.open(path,'rt') as f:
        for line in f:
            yield json.loads(line,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))


def require(ok,message):
    if not ok:
        raise AssertionError(message)


def exact(a,b,message):
    require(a==b,message)


def array_exact(a,b,message):
    aa,bb = np.asarray(a),np.asarray(b)
    require(aa.shape==bb.shape and np.array_equal(aa,bb),message)


def near(a,b,message,tolerance=2e-12):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance,message)


def byte_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def action_grid():
    result = []
    for i in range(80):
        speed = (np.exp((i%5+1)/5)-1)/(np.e-1)
        heading = 2*np.pi*(i//5)/16
        result.append([speed*np.cos(heading),speed*np.sin(heading)])
    return np.asarray(result,dtype='float64')


def ttc_indices(robot,humans):
    if len(humans)<=5:
        return list(range(len(humans)))
    h = np.asarray(humans,dtype='float64')
    r = np.asarray(robot[:4],dtype='float32').astype('float64')
    rx,ry = h[:,0]-r[0],h[:,1]-r[1]
    vx,vy = h[:,2]-r[2],h[:,3]-r[3]
    dist = np.sqrt(rx**2+ry**2+1e-6)
    closing = -(rx*vx+ry*vy)/(dist+1e-6)
    with np.errstate(divide='ignore',invalid='ignore'):
        priority = np.where(closing>.1,dist/(closing+1e-6),dist*10)
    return sorted(np.argsort(priority)[:5].tolist())


def visible_token(robot,humans,ids):
    """Reconstruct archived token bytes from original contract, without imports."""
    r = np.asarray(robot,dtype='float32').astype('float64')
    h = np.asarray(humans,dtype='float64')[ids,:5]
    t = np.zeros((8,13),dtype='float32')
    dx,dy = r[5]-r[0],r[6]-r[1]
    dist = np.sqrt(dx**2+dy**2)
    speed = np.sqrt(r[2]**2+r[3]**2)
    angle = np.arctan2(dy,dx)
    t[0,:9] = r
    t[0,9:] = [dist,speed,np.clip(r[7],0.,3.),1.]
    t[1,:6] = [dx,dy,dist,angle,np.cos(angle),np.sin(angle)]
    if speed>1e-6:
        motion = np.arctan2(r[3],r[2])
        t[2,:4] = [speed,motion,np.cos(motion),np.sin(motion)]
    hx,hy = h[:,0]-r[0],h[:,1]-r[1]
    vx,vy = h[:,2]-r[2],h[:,3]-r[3]
    d = np.sqrt(hx**2+hy**2+1e-6)
    closing = -(hx*vx+hy*vy)/(d+1e-6)
    with np.errstate(divide='ignore',invalid='ignore'):
        priority = np.where(closing>.1,d/(closing+1e-6),d*10)
    for row,index in enumerate(np.argsort(priority),start=3):
        inv = 10.
        if d[index]>1e-6:
            c = -(hx[index]*vx[index]+hy[index]*vy[index])/d[index]
            inv = 1./max(d[index]/c,.1) if c>1e-6 else 0.
        t[row,:9] = [hx[index],hy[index],d[index],h[index,2],h[index,3],
            np.sqrt(h[index,2]**2+h[index,3]**2),h[index,4],inv,float(d[index]<2)]
    return t


def immediate(robot,humans,commands):
    r = np.asarray(robot,dtype='float64')
    h = np.asarray(humans,dtype='float64')
    cmd = np.asarray(commands,dtype='float64')
    nr = r[None,:2]+cmd*.25
    nh = h[:,:2]+h[:,2:4]*.25
    d = np.hypot(nr[:,None,0]-nh[None,:,0],nr[:,None,1]-nh[None,:,1])-r[4]-h[None,:,4]
    clearance = d.min(1)
    collision = (d<0).any(1)
    reaching = np.hypot(nr[:,0]-r[5],nr[:,1]-r[6])<.25
    reward = -np.where(clearance<.2,.5*(.2-clearance)*.25,0.)
    reward = np.where(reaching,1.,reward)
    reward = np.where(collision,-.5,reward)
    return reward,clearance,collision,reaching


def filter_scores(raw,clearance):
    scores = np.asarray(raw,dtype='float64').copy()
    safe = clearance>=.2
    if safe.any():
        scores[~safe] = -1e9
    scores = scores-.8*np.maximum(.2-clearance,0.)
    scores[0] -= .001
    return scores


def check_episode(ep,key,baseline=True):
    controller,model,geometry,condition = key
    for field,value in [('controller',controller),('model',model),('geometry',geometry),('condition',condition)]:
        exact(ep[field],value,'episode metadata '+field)
    require(ep['case'] in range(90000,90016),'case outside frozen block')
    records = ep['steps']
    require(1<=len(records)<=201,'episode length')
    exact([r['step'] for r in records],list(range(len(records))),'step sequence')
    exact([r['terminal'] for r in records],[False]*(len(records)-1)+[True],'terminal sequence')
    expected_event = {'success':'reach_goal','collision':'collision','timeout':'timeout'}[ep['outcome']]
    exact(records[-1]['info']['event'],expected_event,'terminal event')
    exact(ep['time'],len(records)*.25,'navigation time')
    exact(ep['return'],sum(.99**i*r['reward'] for i,r in enumerate(records)),'discounted return')
    exact(ep['discounted_return'],ep['return'],'duplicate discounted return')
    exact(ep['undiscounted_return'],sum(r['reward'] for r in records),'undiscounted return')
    exact(ep['E3_shadow_exact'],condition=='E1','shadow marker')
    if baseline:
        exact(ep['override'],None,'baseline override forbidden')
    grid,last,history = action_grid(),None,[]
    length = 0.
    for i,row in enumerate(records):
        robot,humans = row['robot'],row['humans']
        require(len(robot)==9 and len(humans)==(5 if condition=='E1' else 20),'recorded population/state shape')
        require(all(len(h)==9 for h in humans),'human state dimensions')
        require(np.isfinite(np.asarray(robot+sum(humans,[]),dtype=float)).all(),'nonfinite physical state')
        exact(robot[5:8],[0,4,1.],'robot goal or preferred speed changed')
        if i==0:
            exact(robot[:4],[0,-4,0,0],'robot initial position/velocity changed')
        for field in ('commands','progress','rewards','clearance','values','raw_scores','scores'):
            require(len(row[field])==80 and np.isfinite(np.asarray(row[field],dtype=float)).all(),'invalid80 '+field)
        expected_commands = grid if last is None else .3*last[None]+.7*grid
        array_exact(row['commands'],expected_commands,'80 smoothed commands')
        expected_ids = ttc_indices(robot,humans)
        exact(row['ids'],expected_ids,'TTC5 selected IDs')
        current_token = visible_token(robot,humans,expected_ids)
        exact(row['token_sha256'],byte_hash(current_token),'current token SHA')
        exact(row['history_sha256'],byte_hash(np.asarray(history[-24:])),'real history SHA')
        history.append(current_token)
        reward,clearance,collision,reaching = immediate(robot,humans,row['commands'])
        array_exact(row['rewards'],reward,'all80 predicted rewards')
        array_exact(row['clearance'],clearance,'all80 clearance')
        exact(row['n_safe'],int((clearance>=.2).sum()),'safe candidate count')
        exact(row['crowd2m'],sum(math.hypot(h[0]-robot[0],h[1]-robot[1])<=2 for h in humans),'crowd2m')
        gd = np.hypot(robot[0]-robot[5],robot[1]-robot[6])
        exact(row['goal_distance'],float(gd),'goal distance')
        cmd = np.asarray(row['commands'])
        progress = gd-np.hypot(robot[0]+cmd[:,0]*.25-robot[5],robot[1]+cmd[:,1]*.25-robot[6])
        array_exact(row['progress'],progress,'all80 goal progress')
        require(type(row['chosen']) is int and 0<=row['chosen']<80,'selected grid index')
        if controller=='model':
            values = np.asarray(row['values'],dtype='float32')
            values[collision|reaching|(i>=200)] = 0.
            continuation = np.float32(.99)*values
            rdtype = np.float64 if (~(collision|reaching)).any() else np.float32
            raw = reward.astype(rdtype)+continuation
            array_exact(row['raw_scores'],raw,'predicted terminal-masked Q')
            array_exact(row['scores'],filter_scores(raw,clearance),'model safety/risk scores')
            exact(row['chosen'],int(np.argmax(row['scores'])),'model argmax before override')
        else:
            array_exact(row['values'],np.zeros(80),'reference value-free interface')
            array_exact(row['raw_scores'],reward,'reference adapter raw-r, not planner cost')
            array_exact(row['scores'],filter_scores(reward,clearance),'reference safety adapter')
        executed = row.get('executed_override',row['chosen'])
        if baseline:
            require('executed_override' not in row,'baseline command overridden')
        exact(row['command'],row['commands'][executed],'executed command differs from evaluated command')
        if (clearance>=.2).any():
            require(clearance[executed]>=.2,'executed masked command')
        last = np.asarray(row['command'],dtype='float64')
        nxt = [robot[0]+last[0]*.25,robot[1]+last[1]*.25]
        length += math.hypot(nxt[0]-robot[0],nxt[1]-robot[1])
        if i+1<len(records):
            exact(records[i+1]['robot'][:4],nxt+row['command'],'robot transition')
            h,hnext = np.asarray(humans),np.asarray(records[i+1]['humans'])
            array_exact(h[:,:2]+hnext[:,2:4]*.25,hnext[:,:2],'human transition')
        near(row['reward'],(-.5 if ep['outcome']!='success' else 1.) if row['terminal'] else
            -.5*max(.2-row['info']['dmin'],0.)*.25,'environment reward')
        require(row['info']['event']==('nothing' if not row['terminal'] else expected_event),'nonterminal event')
    exact(ep['path_length'],length,'path length from raw commands')
    for name,field in [('endpoint','endpoint'),('CV_swept','CV_swept'),('actual_swept','actual_swept')]:
        exact(ep['minimum_clearance'][name],min(r[field] for r in records),'minimum '+name)
    return len(records)


def compact_episode(ep):
    result = {k:v for k,v in ep.items() if k!='steps'}
    result['steps_count'] = len(ep['steps'])
    result['safe_count_sum'] = sum(r['n_safe'] for r in ep['steps'])
    result['crowd2m_sum'] = sum(r['crowd2m'] for r in ep['steps'])
    return result


def statistics(eps):
    n = len(eps)
    out = {'episodes':n}
    for outcome,rate in [('success','SR'),('collision','CR'),('timeout','TO')]:
        out[outcome] = sum(e['outcome']==outcome for e in eps)
        out[rate] = out[outcome]/n
    for field in ('return','discounted_return','undiscounted_return','time','path_length','latency_ms_p50','latency_ms_p95'):
        out[field] = float(np.mean([e[field] for e in eps]))
    out['physical_collision_episodes'] = sum(e['minimum_clearance']['actual_swept']<0 for e in eps)
    out['successful_but_physical_collision'] = sum(e['outcome']=='success' and e['minimum_clearance']['actual_swept']<0 for e in eps)
    for name in ('endpoint','CV_swept','actual_swept'):
        out['mean_minimum_'+name] = float(np.mean([e['minimum_clearance'][name] for e in eps]))
    steps = sum(e['steps_count'] for e in eps)
    out['safe_count_step_mean'] = sum(e['safe_count_sum'] for e in eps)/steps
    out['crowd2m_step_mean'] = sum(e['crowd2m_sum'] for e in eps)/steps
    return out


def delta_statistics(delta,rng):
    d = np.asarray(delta)
    resamples = d[rng.randint(0,len(d),(2000,len(d)))].mean(1)
    return {'mean_delta_B_minus_A':float(d.mean()),'case_delta':d.tolist(),
            'case_bootstrap95':np.quantile(resamples,[.025,.975]).tolist()}


def paired_statistics(a,b):
    exact([e['case'] for e in a],[e['case'] for e in b],'paired case block')
    rng,out = np.random.RandomState(41995),{}
    for field in ('success','discounted_return','return','time','path_length'):
        av = np.asarray([int(e['outcome']=='success') if field=='success' else e[field] for e in a])
        bv = np.asarray([int(e['outcome']=='success') if field=='success' else e[field] for e in b])
        out[field] = delta_statistics(bv-av,rng)
    return out


def compare_e1_e3(a,b):
    """Every archived step/field exact, dropping only added humans and timings."""
    exact(a['case'],b['case'],'E1/E3 case')
    exact(len(a['steps']),len(b['steps']),'E1/E3 trajectory length')
    for field in a:
        if field in TIMING_KEYS or field in ('condition','steps','E3_shadow_exact'):
            continue
        exact(a[field],b[field],'E1/E3 episode '+field)
    far_min = float('inf')
    for one,far in zip(a['steps'],b['steps']):
        exact(set(one),set(far),'E1/E3 raw step schema')
        for field in one:
            exact(one[field],far[field][:5] if field=='humans' else far[field],
                  'E1/E3 case%d step%d %s'%(a['case'],one['step'],field))
        remote = np.asarray(far['humans'][5:])
        require((remote[:,0]>9000).all(),'remote entered navigation region')
        d = np.linalg.norm(remote[:,None,:2]-remote[None,:,:2],axis=-1)-remote[:,None,4]-remote[None,:,4]
        np.fill_diagonal(d,np.inf)
        far_min = min(far_min,float(d.min()))
        require(far_min>=0,'remote humans collided')
    return {'case':a['case'],'steps':len(a['steps']),'all_archived_fields_exact':True,
            'minimum_far_far_clearance':far_min}


def a1(row):
    valid = [i for i in range(80) if row['clearance'][i]>=.2 and row['progress'][i]>=.02]
    return max(valid,key=lambda i:(row['raw_scores'][i],-i)) if valid else None


def eligible(row,kind):
    return a1(row) is not None and (kind!='timeout' or
        (row['step']>=20 and row['goal_distance']>1 and row['progress'][row['chosen']]<.02))


def distance(a,b):
    ds = [abs(a['step']-b['step'])/8.,abs(a['goal_distance']-b['goal_distance']),
        abs(a['n_safe']-b['n_safe'])/16.,abs(a['crowd2m']-b['crowd2m'])/2.]
    return (sum(ds),ds) if max(ds)<=1 else None


def expected_roots(eps,model,geometry,condition):
    candidates,denominator = [],[]
    for ep in eps:
        if ep['outcome']=='success':
            continue
        unsafe = next((r['step'] for r in ep['steps'] if r['n_safe']==0),None)
        window = ep['steps'] if ep['outcome']=='timeout' else (
            ep['steps'][max(0,unsafe-12):unsafe] if unsafe is not None else [])
        rootrow = next((r for r in window if eligible(r,ep['outcome']) and
            (ep['outcome']=='timeout' or a1(r)!=r['chosen'])),None)
        denominator.append({'geometry':geometry,'condition':condition,'case':ep['case'],
            'outcome':ep['outcome'],'has_root':rootrow is not None,'selected_within_cap':False})
        if rootrow is not None:
            candidates.append({'model':model,'geometry':geometry,'condition':condition,'case':ep['case'],
                'step':rootrow['step'],'type':ep['outcome'],'snapshot':rootrow,'A0':rootrow['chosen'],
                'A1':a1(rootrow),'matched_control':None})
    selected,used = [],set()
    for root in candidates[:2]:
        next(d for d in denominator if d['case']==root['case'])['selected_within_cap'] = True
        options = []
        for ep in eps:
            if ep['outcome']!='success' or ep['case'] in used:
                continue
            for row in ep['steps']:
                dist = distance(root['snapshot'],row)
                if eligible(row,root['type']) and a1(row)!=row['chosen'] and dist is not None:
                    options.append((dist[0],ep['case'],row['step'],row,dist[1]))
        if options:
            _,case,k,row,ds = min(options,key=lambda x:x[:3])
            used.add(case)
            selected.append({'model':model,'geometry':geometry,'condition':condition,'case':case,'step':k,
                'type':'success-control','snapshot':row,'A0':row['chosen'],'A1':a1(row),
                'matched_failure':root['case'],'match_standardized_differences':ds})
            root['matched_control'] = [case,k]
        selected.append(root)
    return selected,denominator


def compare_a0(ep,baseline):
    exact(set(ep),set(baseline),'A0 episode schema')
    for field in baseline:
        if field not in TIMING_KEYS:
            exact(ep[field],baseline[field],'A0 full raw replay '+field)


def check_branch(record,root,baseline,key):
    exact(record['root'],root,'branch frozen root')
    zero,one = record['A0'],record['A1']
    compare_a0(zero,baseline)
    check_episode(zero,key)
    check_episode(one,key,baseline=False)
    k = root['step']
    distinct = root['A0']!=root['A1']
    exact(record['distinct_action'],distinct,'distinct action flag')
    if not distinct:
        compare_a0(one,zero)
    else:
        exact(one['override'],{'step':k,'index':root['A1']},'single prescribed override')
        exact([r['step'] for r in one['steps'] if 'executed_override' in r],[k],'only one override')
        for i in range(k):
            exact(one['steps'][i],zero['steps'][i],'A1 full shared prefix')
        expected = dict(zero['steps'][k])
        changed = dict(one['steps'][k])
        for field in ('command','reward','terminal','info','actual_swept','CV_swept','endpoint','executed_override'):
            expected.pop(field,None)
            changed.pop(field,None)
        exact(expected,changed,'root score/history/state frozen before execution')
        exact(one['steps'][k]['executed_override'],root['A1'],'A1 override index')
        exact(one['steps'][k]['command'],root['snapshot']['commands'][root['A1']],'A1 prescribed command')
    g0 = sum(.99**(i-k)*r['reward'] for i,r in enumerate(zero['steps']) if i>=k)
    g1 = sum(.99**(i-k)*r['reward'] for i,r in enumerate(one['steps']) if i>=k)
    exact(record['root_G0'],g0,'root G0')
    exact(record['root_G1'],g1,'root G1')
    exact(record['delta_root_return'],g1-g0,'root return difference')
    exact(record['delta_episode_discounted_return'],one['return']-zero['return'],'episode return difference')
    exact(record['rescued'],zero['outcome']!='success' and one['outcome']=='success','rescue flag')
    exact(record['destroyed'],zero['outcome']=='success' and one['outcome']!='success','destruction flag')
    exact(record['A0_exact'],True,'execution A0 marker')


class Audit:
    def __init__(self,work,contract,manifest=None,baseline_only=False):
        self.work,self.contract = work,contract
        self.baseline_only = baseline_only
        self.p = read_json(contract)
        self.manifest = read_json(manifest) if manifest else None
        self.result = {'execution_status':'INCOMPLETE','scientific_scope_status':'INCOMPLETE',
            'expected':{'model_episodes':640,'reference_episodes':320,'total_episodes':960,'main_shards':60},
            'verified':{'model_episodes':0,'reference_episodes':0,'total_episodes':0,'main_shards':0,'raw_steps':0},
            'missing':[],'errors':[],'shards':[],'E1_E3_pairs':[],'roots':[],'branches':[],
            'groups':[],'paired':[],'reference_difficulty':[],'excess_transfer_baseline':[],
            'scope':'baseline_only' if baseline_only else 'baseline_and_continuations',
            'strict_opportunity_matches':[],'reference_evidence_levels':{
                'orca-grid':'Existing local ORCA source execution plus80-command quantization/filter adaptation; not unmodified ORCA official performance reproduction.',
                'cv-mpc':'Standard CV finite-horizon beam/MPC construction implemented here. Not an official checkpoint, published-method baseline reproduction, mathematical optimum or validated task-feasibility upper bound.'},
            'unfulfilled_scientific_requirements':['B5 lacks new paired N10 arm.',
                'B3 mature official second-planner baseline reproduction is not established by custom CV-MPC.',
                'B4 input x training-population factorial has not been executed; unauthorized second round.'],
            'archival_limitations':['Final post-update complete human state not stored per episode.',
                'GPU episode raw lacks per-step ORCA parameter/neighbor-ID arrays; E1 shadow source asserts equality, CPU qualification records comparison results.'],
            'read_only':True,'network_execution':False,'thresholds_modified':False,
            'contract_sha256':sha(contract),'auditor_sha256':sha(Path(__file__))}
        exact(sha(contract),EXPECTED_CONTRACT_SHA,'Frozen contract changed')
        exact(self.p['cases'],list(range(90000,90016)),'Frozen16 cases changed')
        self.loaded = {}
        self.paths = {}

    def attempt(self,label,fn):
        try:
            return fn()
        except (FileNotFoundError,EOFError) as e:
            self.result['missing'].append({'item':label,'reason':str(e)})
        except (AssertionError,ValueError,KeyError,TypeError,OSError) as e:
            self.result['errors'].append({'item':label,'reason':str(e)})
        return None

    def load_group(self,key):
        controller,model,geometry,condition = key
        folder = self.work/'main'/controller/model
        stem = geometry+'-'+condition
        combined = folder/(stem+'.jsonl.gz')
        pieces = sorted(folder.glob(stem+'-*.jsonl.gz'))
        require(not(combined.exists() and pieces),'mixed combined+case shards would double count')
        paths = [combined] if combined.exists() else pieces
        if not paths:
            raise FileNotFoundError(str(combined))
        eps,entries,steps = [],[],0
        for path in paths:
            done,start = read_json(path.with_suffix('.done.json')),read_json(path.with_suffix('.start.json'))
            exact(done['status'],'PASS','shard completion status')
            exact(done['output_sha256'],sha(path),'raw shard SHA')
            exact(done['provenance'],start,'start/done provenance')
            exact(start['contract_sha256'],EXPECTED_CONTRACT_SHA,'shard contract SHA')
            exact(start['script_sha256'],EXPECTED_DRIVER_SHA,'frozen driver SHA')
            exact(start['phase'],'main','smoke contaminated main')
            exact(start['controller'],controller,'controller provenance')
            for field in ('status','history_test','terminal_test'):
                exact(start['acceptance'][field],'PASS','public acceptance '+field)
            exact(start['acceptance']['tested_commit'],'9f06223','accepted public revision')
            exact(start['model']['id'],model,'checkpoint ID')
            if self.manifest:
                original = next(m for m in self.manifest['protocol']['models'] if m['id']==model)
                for field in ('weight_sha256','config_sha256','backbone','seed'):
                    exact(start['model'][field],original[field],'original manifest '+field)
            cases = []
            for ep in rows(path):
                try:
                    steps += check_episode(ep,key)
                except AssertionError as error:
                    raise AssertionError('case%d: %s'%(ep['case'],error)) from error
                cases.append(ep['case'])
                eps.append(compact_episode(ep))
            exact(done['episodes'],len(cases),'done episode count')
            exact(cases,start['cases'],'shard case order')
            entries.append({'file':str(path),'sha256':done['output_sha256'],'episodes':len(cases),
                'wall_seconds':done['wall_seconds'],
                'provenance':{k:start[k] for k in ('project_commit','source_sha256','script_sha256')}})
        eps.sort(key=lambda e:e['case'])
        exact([e['case'] for e in eps],self.p['cases'],'missing/duplicate formal cases')
        self.loaded[key] = eps
        self.paths[key] = paths
        self.result['groups'].append(dict(zip(('controller','model','geometry','condition'),key),
            **statistics(eps),shard_wall_seconds=sum(e['wall_seconds'] for e in entries)))
        self.result['shards'].append({'key':key,'episodes':len(eps),'raw_steps':steps,'files':entries})
        self.result['verified']['model_episodes' if controller=='model' else 'reference_episodes'] += len(eps)
        self.result['verified']['total_episodes'] += len(eps)
        self.result['verified']['main_shards'] += 1
        self.result['verified']['raw_steps'] += steps

    def check_e3(self,controller,model,geometry):
        ka,kb = (controller,model,geometry,'E1'),(controller,model,geometry,'E3')
        if ka not in self.loaded or kb not in self.loaded:
            raise FileNotFoundError('E1 or E3 raw shard incomplete')
        checks = []
        for a,b in zip_longest(self.raw_group(ka),self.raw_group(kb)):
            require(a is not None and b is not None,'E1/E3 missing raw episode')
            checks.append(compare_e1_e3(a,b))
        self.result['E1_E3_pairs'].append({'controller':controller,'model':model,'geometry':geometry,
            'paired_cases':len(checks),'paired_steps':sum(c['steps'] for c in checks),'checks':checks})

    def raw_group(self,key):
        for path in self.paths[key]:
            yield from rows(path)

    def baseline_contrasts(self):
        pairs = [('E1','E2'),('E1','E3'),('E2','E4'),('E1','E5'),('E2','E5')]
        indexed = {}
        for controller in ['model']+self.p['references']:
            models = self.p['models'] if controller=='model' else [self.p['models'][0]]
            for model in models:
                for geo in self.p['geometries']:
                    for a,b in pairs:
                        ka,kb = (controller,model,geo,a),(controller,model,geo,b)
                        if ka not in self.loaded or kb not in self.loaded:
                            continue
                        metrics = paired_statistics(self.loaded[ka],self.loaded[kb])
                        indexed[controller,model,geo,a,b] = metrics
                        self.result['paired'].append({'controller':controller,'model':model,
                            'geometry':geo,'A':a,'B':b,'metrics':metrics})
        for geo in self.p['geometries']:
            for a,b in [('E1','E2'),('E2','E4')]:
                refs = []
                for controller in self.p['references']:
                    key = controller,self.p['models'][0],geo,a,b
                    if key not in indexed:
                        continue
                    m = indexed[key]
                    refs.append({'controller':controller,'success_loss':-m['success']['mean_delta_B_minus_A'],
                        'return_loss':-m['return']['mean_delta_B_minus_A'],
                        'frozen_threshold_met':-m['success']['mean_delta_B_minus_A']>=2/16 or
                            -m['return']['mean_delta_B_minus_A']>.05})
                self.result['reference_difficulty'].append({'geometry':geo,'A':a,'B':b,'references':refs,
                    'frozen_constructed_reference_gate':'MET' if len(refs)==2 and all(r['frozen_threshold_met'] for r in refs) else
                        'NOT_MET' if len(refs)==2 else 'INCOMPLETE',
                    'mature_second_reference_requirement':'INCOMPLETE_CUSTOM_CONSTRUCTION_ONLY'})
            for model in self.p['models']:
                key = 'model',model,geo,'E1','E2'
                if key not in indexed:
                    continue
                m,contrasts = indexed[key],[]
                for controller in self.p['references']:
                    rkey = controller,self.p['models'][0],geo,'E1','E2'
                    if rkey not in indexed:
                        continue
                    r = indexed[rkey]
                    sr = np.asarray(r['success']['case_delta'])-np.asarray(m['success']['case_delta'])
                    ret = np.asarray(r['return']['case_delta'])-np.asarray(m['return']['case_delta'])
                    rng = np.random.RandomState(41995)
                    contrasts.append({'reference':controller,'success_excess_loss':delta_statistics(sr,rng),
                        'return_excess_loss':delta_statistics(ret,rng),
                        'baseline_threshold_met':float(sr.mean())>=2/16 or float(ret.mean())>.05})
                self.result['excess_transfer_baseline'].append({'model':model,'geometry':geo,'A':'E1','B':'E2',
                    'against_references':contrasts,'both_reference_baseline_threshold_met':len(contrasts)==2 and
                        all(c['baseline_threshold_met'] for c in contrasts),'protected_continuation':'PENDING_B5',
                    'scientific_verdict':'INCONCLUSIVE_NOT_A_VALUE_CAUSAL_ATTRIBUTION'})

    def check_roots(self,model):
        path = self.work/('roots-'+model+'.json')
        data = read_json(path)
        exact(read_json(path.with_suffix('.sha256.json'))['sha256'],sha(path),'root freeze SHA')
        exact(data['contract_sha256'],EXPECTED_CONTRACT_SHA,'root contract SHA')
        require(data['branch_results_exist'] is False,'roots frozen after branch outcomes')
        expected,denominator = [],[]
        for geometry in self.p['geometries']:
            for condition in self.p['conditions']:
                key = ('model',model,geometry,condition)
                if key not in self.loaded:
                    raise FileNotFoundError('Baseline group incomplete: '+str(key))
                rs,ds = expected_roots(list(self.raw_group(key)),model,geometry,condition)
                expected.extend(rs)
                denominator.extend(ds)
        expected.sort(key=lambda r:(self.p['geometries'].index(r['geometry']),self.p['conditions'].index(r['condition']),r['case'],r['step']))
        for i,root in enumerate(expected):
            root['root_id'] = i
        exact(data['roots'],expected,'case-ordered root/A1/control selection differs')
        exact(data['all_failure_denominator'],denominator,'all failures/no-root/cap denominator')
        exact(data['unmatched_failure_roots'],sum(r['type']!='success-control' and r['matched_control'] is None for r in expected),'unmatched count')
        self.result['roots'].append({'model':model,'roots':len(expected),'all_failed_episodes':len(denominator),
            'unmatched_failure_roots':data['unmatched_failure_roots'],'freeze_sha256':sha(path)})
        for geometry in self.p['geometries']:
            for condition in self.p['conditions']:
                self.attempt('branches '+str((model,geometry,condition)),lambda g=geometry,c=condition:
                    self.check_branches(model,g,c,data,sha(path)))
            for condition in ('E2','E3','E4','E5'):
                a = [r for r in expected if r['geometry']==geometry and r['condition']=='E1' and r['type']!='success-control']
                b = [r for r in expected if r['geometry']==geometry and r['condition']==condition and r['type']!='success-control']
                matched,used = [],set()
                for root in a:
                    options = [(distance(root['snapshot'],other['snapshot']),other) for other in b
                        if other['root_id'] not in used and other['type']==root['type']]
                    options = [x for x in options if x[0] is not None]
                    if options:
                        dist,other = min(options,key=lambda x:(x[0][0],x[1]['case'],x[1]['step']))
                        used.add(other['root_id'])
                        matched.append({'E1_root':root['root_id'],'other_root':other['root_id'],'standardized_diff':dist[1]})
                self.result['strict_opportunity_matches'].append({'model':model,'geometry':geometry,'B':condition,
                    'matches':matched,'status':'MATCHED_NOT_YET_CAUSAL' if matched else 'INCOMPLETE_NO_STRICT_MATCHES'})
        return data

    def check_branches(self,model,geometry,condition,data,freeze_sha):
        path = self.work/'branches'/model/(geometry+'-'+condition+'.jsonl.gz')
        done = read_json(path.with_suffix('.done.json'))
        exact(done['status'],'PASS','branch done status')
        exact(done['sha256'],sha(path),'branch raw SHA')
        exact(done['root_freeze_sha256'],freeze_sha,'branch root SHA')
        selected = [r for r in data['roots'] if r['geometry']==geometry and r['condition']==condition]
        records = list(rows(path))
        exact([r['root']['root_id'] for r in records],[r['root_id'] for r in selected],'branch root completeness/order')
        exact(done['roots'],len(records),'branch done root count')
        # Filesystem timestamp is an additional ordering check, not trusted alone.
        require(path.stat().st_mtime>=data['frozen_unix'],'branch mtime predates root freeze')
        lookup = {e['case']:e for e in self.raw_group(('model',model,geometry,condition))}
        for record,root in zip(records,selected):
            check_branch(record,root,lookup[root['case']],('model',model,geometry,condition))
        failures = [r for r in records if r['root']['type']!='success-control']
        controls = [r for r in records if r['root']['type']=='success-control']
        self.result['branches'].append({'model':model,'geometry':geometry,'condition':condition,
            'verified_A0_A1_roots':len(records),'failure_roots':len(failures),'success_controls':len(controls),
            'unmatched_failure_roots':sum(r['root']['matched_control'] is None for r in failures),
            'rescues':sum(r['rescued'] for r in failures),'success_destructions':sum(r['destroyed'] for r in controls),
            'failure_delta_root_returns':[r['delta_root_return'] for r in failures],
            'control_delta_root_returns':[r['delta_root_return'] for r in controls],
            'raw_sha256':sha(path),'success_protection_evidence':'MISSING' if len(controls)<len(failures) else
                'NO_FAILURE_ROOTS' if not failures else 'MATCHED_CONTROLS_CHECKED'})

    def run(self):
        for controller in ['model']+self.p['references']:
            models = self.p['models'] if controller=='model' else [self.p['models'][0]]
            for model in models:
                for geometry in self.p['geometries']:
                    for condition in self.p['conditions']:
                        key = (controller,model,geometry,condition)
                        self.attempt('main '+str(key),lambda k=key:self.load_group(k))
                    self.attempt('E1/E3 '+str((controller,model,geometry)),lambda c=controller,m=model,g=geometry:self.check_e3(c,m,g))
        self.baseline_contrasts()
        if not self.baseline_only:
            for model in self.p['models']:
                self.attempt('roots '+model,lambda m=model:self.check_roots(m))
        if self.result['errors']:
            self.result['execution_status'] = 'FAIL'
        elif not self.result['missing']:
            exact(self.result['verified']['total_episodes'],960,'formal episode total')
            exact(self.result['verified']['model_episodes'],640,'learned episode total')
            exact(self.result['verified']['reference_episodes'],320,'reference episode total')
            exact(len(self.result['E1_E3_pairs']),12,'all12 controller/model/geometry E3 groups')
            exact(sum(g['paired_cases'] for g in self.result['E1_E3_pairs']),192,'all192 E1/E3 pairs')
            if not self.baseline_only:
                exact(len(self.result['branches']),40,'all40 branch shards, including empty-root shards')
            self.result['execution_status'] = 'PASS'
        self.result['scientific_scope_status'] = 'INCOMPLETE'
        self.result['scientific_conclusion_from_partial_results'] = 'NOT_PERMITTED'
        return self.result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=BASE/'work')
    p.add_argument('--contract',type=Path,default=BASE/'b-contract.json')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--baseline-only',action='store_true',help='Accept only the frozen960 baselines; do not inspect branches.')
    args = p.parse_args()
    try:
        result = Audit(args.work,args.contract,args.manifest,args.baseline_only).run()
    except (AssertionError,ValueError,KeyError,OSError) as error:
        result = {'execution_status':'FAIL','read_only':True,'error':str(error)}
    print(json.dumps(result,allow_nan=False,indent=2))
    return 0 if result['execution_status']=='PASS' else 2 if result['execution_status']=='INCOMPLETE' else 1


if __name__=='__main__':
    sys.exit(main())
