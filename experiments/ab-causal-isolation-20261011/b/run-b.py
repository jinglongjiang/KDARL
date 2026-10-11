#!/usr/bin/env python3
"""B round one: frozen paired scenes, isolated CPU gates, scheduled evaluation.

Never trains. All outputs are confined to this B directory. GPU entrypoints
require independent public-fix acceptance; CPU scenes need no model weights.
"""
import argparse
import configparser
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import deque
from types import SimpleNamespace

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
CONTRACT = BASE / 'b-contract.json'
P = json.loads(CONTRACT.read_text())
FIELDS = ('px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta')
ORCA_FIELDS = ('neighbor_dist', 'max_neighbors', 'time_horizon', 'time_horizon_obst',
               'radius', 'max_speed', 'time_step', 'safety_space', 'ttc_brake',
               'ttc_threshold', 'ttc_min_scale', 'ttc_smoothing')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path).resolve()
    path.relative_to(BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def write(f, value):
    f.write(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')
    f.flush()


def read_rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def pin(project):
    project = Path(project).resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    import crowd_sim
    Path(crowd_sim.__file__).resolve().relative_to(project)


def cfg_read(path):
    cfg = configparser.RawConfigParser()
    assert cfg.read(str(path)), path
    return cfg


def manifest_item(args, model):
    root = Path(args.causal).resolve()
    man = json.loads((root / 'manifest.json').read_text())
    item = copy.deepcopy(next(x for x in man['protocol']['models'] if x['id'] == model))
    # Relocate only by known model ID; do not trust stale absolute manifest roots.
    for key, filename in [('config', 'config.ini'), ('weight', 'rl_model_ep10000.pth')]:
        candidates = [root / 'inputs4models' / model / filename,
                      root / 'inputs' / model / filename, Path(item[key])]
        found = next((x for x in candidates if x.is_file()), None)
        assert found is not None, (model, key, candidates)
        assert sha(found) == item[key + '_sha256'], (found, 'manifest SHA mismatch')
        item[key] = str(found)
    return item


def human_array(humans):
    import numpy as np
    return np.asarray([[getattr(h, x) for x in FIELDS] for h in humans], dtype='float64')


def params(humans):
    return [{k: getattr(h.policy, k) for k in ORCA_FIELDS} for h in humans]


def scene(cfg, geometry, condition, case, net=None):
    import numpy as np
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.human import Human
    assert case in P['cases'] and condition in P['conditions']
    cfg = copy.deepcopy(cfg)
    n = 5 if condition in ('E1', 'E3') else 20
    cfg.set('sim', 'human_num', str(n))
    cfg.set('sim', 'test_sim', geometry + '_crossing')
    env = CrowdSim()
    env.configure(cfg)
    robot = Robot(cfg, 'robot')
    robot.set_policy(net if net is not None else SimpleNamespace(multiagent_training=True))
    env.set_robot(robot)
    env.phase = 'test'
    if net is not None:
        net.set_env(env)
        if hasattr(net, 'set_robot'):
            net.set_robot(robot)
        net.reset_episode_stats()
    env.reset(seed=case, options={'test_case': case})
    assert (robot.px, robot.py, robot.gx, robot.gy) == (0, -4, 0, 4)
    assert not robot.visible and env.time_step == .25 and env.time_limit == 50
    assert env.progress_reward == 0
    if condition == 'E3':
        for i in range(15):
            h = Human(cfg, 'humans')
            x = 10000. + 20. * i
            h.set(x, 0., x, 4., 0., 0., 0.)
            h.time_step = h.policy.time_step = env.time_step
            env.humans.append(h)
        env.human_num = 20
        env.human_times = [0.] * 20
    elif condition == 'E4':
        points = [(ix * .85, iy * .85) for ix in range(-3, 4) for iy in range(-3, 4)]
        points.sort(key=lambda q: (abs(q[0]), abs(q[1]), q[0], q[1]))
        placed = list(env.humans[:5])
        for h in env.humans[5:]:
            point = next((q for q in points if all(
                np.hypot(q[0]-a.px, q[1]-a.py) >= h.radius+a.radius+.2
                for a in [robot] + placed)), None)
            assert point is not None, ('E4 fixed placement infeasible', case)
            points.remove(point)
            x, y = point
            h.set(x, y, -x if x else .85, -y, 0., 0., 0.)
            placed.append(h)
    elif condition == 'E5':
        for h in env.humans:
            h.px *= 2.
            h.py *= 2.
            h.gx *= 2.
            h.gy *= 2.
        env.circle_radius = 8.
        env.square_width = 20.
    assert len(env.humans) == (5 if condition == 'E1' else 20)
    return env, robot


def grid(cfg):
    from crowd_nav import contracts
    from crowd_sim.envs.utils.action import ActionXY
    g = contracts.init_grid_from_cfg(cfg)
    assert g['n_speeds'] == 5 and g['n_headings'] == 16 and not g['include_stop']
    assert g['sampling'] == 'exponential' and g['v_min'] == .05
    return [ActionXY(*contracts.discrete_index_to_action(i, **g)) for i in range(80)]


def commands(actions, last):
    from crowd_sim.envs.utils.action import ActionXY
    if last is None:
        return list(actions)
    return [ActionXY(.3*last.vx+.7*a.vx, .3*last.vy+.7*a.vy) for a in actions]


def state(env):
    from crowd_sim.envs.utils.state import JointState
    return JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])


def ttc_ids(s):
    import numpy as np
    if len(s.human_states) <= 5:
        return list(range(len(s.human_states)))
    h = np.asarray([[x.px,x.py,x.vx,x.vy,x.radius] for x in s.human_states], dtype='float64')
    r = np.asarray([s.self_state.px,s.self_state.py,s.self_state.vx,s.self_state.vy], dtype='float32').astype('float64')
    dxy, dv = h[:, :2]-r[:2], h[:, 2:4]-r[2:4]
    d = np.sqrt((dxy*dxy).sum(1)+1e-6)
    close = -(dxy*dv).sum(1)/(d+1e-6)
    with np.errstate(divide='ignore', invalid='ignore'):
        score = np.where(close > .1, d/(close+1e-6), d*10)
    return sorted(np.argsort(score)[:5].tolist())


def token(s):
    import numpy as np
    from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
    r = s.self_state
    a = np.asarray([getattr(r,k) for k in FIELDS], dtype='float32')
    hs = np.asarray([[h.px,h.py,h.vx,h.vy,h.radius] for h in
                     [s.human_states[i] for i in ttc_ids(s)]], dtype='float64').reshape(-1)
    return _batch_joint34_to_tokens_vectorized(np.concatenate([a,hs])[None])[0]


def immediate(env, cmds):
    """Exact inherited endpoint reward/clearance, distinct from env swept reward."""
    import numpy as np
    r = env.robot
    xy = np.asarray([r.compute_position(c, env.time_step) for c in cmds])
    hp = np.asarray([[h.px+h.vx*env.time_step,h.py+h.vy*env.time_step] for h in env.humans])
    hr = np.asarray([h.radius for h in env.humans])
    d = np.hypot(xy[:,None,0]-hp[None,:,0],xy[:,None,1]-hp[None,:,1])-r.radius-hr[None]
    clear = d.min(1)
    collision = (d < 0).any(1)
    dist = np.hypot(xy[:,0]-r.gx,xy[:,1]-r.gy)
    speed = np.hypot([c.vx for c in cmds],[c.vy for c in cmds])
    rewards = env.progress_reward*(np.linalg.norm((r.px-r.gx,r.py-r.gy))-dist)+env.time_penalty
    rewards = rewards+np.where(speed<.05,env.stand_penalty,0.)
    rewards = rewards-np.where(clear<env.discomfort_dist,
        env.discomfort_penalty_factor*(env.discomfort_dist-clear)*env.time_step,0.)
    # The network success_radius is .25 while environment uses robot.radius(.3).
    reaching = dist < env.config.getfloat('robot','success_radius')
    rewards = np.where(reaching,env.success_reward,rewards)
    rewards = np.where(collision,env.collision_penalty,rewards)
    return rewards, clear


def filter_scores(raw, clearance):
    import numpy as np
    result = np.asarray(raw,dtype='float64').copy()
    safe = clearance >= .2
    if safe.any():
        result[~safe] = -1e9
    result -= .8*np.maximum(.2-clearance,0.)
    result[0] -= .001
    return result


def neighbors(env, local=5):
    """Actual RVO2 neighbor IDs after doStep, mapped back to environment IDs."""
    result = []
    for i,h in enumerate(env.humans[:local]):
        sim = h.policy.sim
        assert sim is not None and hasattr(sim,'getAgentNumAgentNeighbors'), 'RVO2 neighbor API unavailable'
        order = [i] + [j for j in range(len(env.humans)) if j != i]
        ids = [order[sim.getAgentAgentNeighbor(0,k)] for k in range(sim.getAgentNumAgentNeighbors(0))]
        result.append(sorted(ids))
    return result


def differences(a, b):
    import numpy as np
    a,b = np.asarray(a),np.asarray(b)
    return {'equal': bool(np.array_equal(a,b)), 'max_abs': float(np.max(np.abs(a-b))) if a.size else 0.}


def swept(env, cmds, next_h=None):
    import numpy as np
    r = env.robot
    h = human_array(env.humans)
    u = np.asarray([[a.vx,a.vy] for a in cmds])
    d0 = np.asarray([r.px,r.py])[None,None]-h[None,:,:2]
    hv = h[:,2:4] if next_h is None else (next_h[:,:2]-h[:,:2])/env.time_step
    dv = u[:,None]-hv[None]
    t = np.clip(-(d0*dv).sum(-1)/np.maximum((dv*dv).sum(-1),1e-30),0.,env.time_step)
    return (np.linalg.norm(d0+t[:,:,None]*dv,axis=-1)-r.radius-h[None,:,4]).min(1)


def env_step(env, action):
    _,reward,terminal,truncated,info = env.step(action)
    return float(reward),bool(terminal or truncated),info


def isolation_pair(cfg, geometry, case, actions):
    import numpy as np
    one,_ = scene(cfg,geometry,'E1',case)
    far,_ = scene(cfg,geometry,'E3',case)
    history1,history3,last = [],[],None
    maxima = {k:0. for k in ('humans','token','history','commands','rewards','clearance','robot')}
    first_mismatch = None
    min_far_clear = float('inf')
    checks = []
    for k in range(201):
        cmds1,cmds3 = commands(actions,last),commands(actions,last)
        t1,t3 = token(state(one)),token(state(far))
        history1.append(t1)
        history3.append(t3)
        r1,c1 = immediate(one,cmds1)
        r3,c3 = immediate(far,cmds3)
        values = {'humans':(human_array(one.humans),human_array(far.humans[:5])),
            'token':(t1,t3),'history':(history1[-24:],history3[-24:]),
            'commands':(cmds1,cmds3),'rewards':(r1,r3),'clearance':(c1,c3),
            'robot':([getattr(one.robot,x) for x in FIELDS],[getattr(far.robot,x) for x in FIELDS])}
        check = {key:differences(*ab) for key,ab in values.items()}
        check['parameters_equal'] = params(one.humans) == params(far.humans[:5])
        check['ttc_ids_equal'] = ttc_ids(state(one)) == ttc_ids(state(far)) == list(range(5))
        remote = human_array(far.humans[5:])
        dd = np.linalg.norm(remote[:,None,:2]-remote[None,:,:2],axis=-1)-remote[:,None,4]-remote[None,:,4]
        np.fill_diagonal(dd,np.inf)
        min_far_clear = min(min_far_clear,float(dd.min()))
        check['far_guard'] = bool((remote[:,0]>9000).all() and dd.min()>=0)
        for key in maxima:
            maxima[key] = max(maxima[key],check[key]['max_abs'])
        action = cmds1[[24,4,44,64][(k//20)%4]]
        # Post-terminal updates here are qualification only, not evaluated episodes.
        rr1,done1,info1 = env_step(one,action)
        rr3,done3,info3 = env_step(far,action)
        check['actual_neighbor_ids_equal'] = neighbors(one) == neighbors(far)
        check['local_has_no_remote_neighbors'] = all(all(j<5 for j in ns) for ns in neighbors(far))
        check['env_reward_terminal_equal'] = (rr1,done1,info1)==(rr3,done3,info3)
        check['post_state_equal'] = bool(np.array_equal(human_array(one.humans),human_array(far.humans[:5])))
        check['step'] = k
        okay = all(v['equal'] if isinstance(v,dict) else v for key,v in check.items() if key!='step')
        if not okay and first_mismatch is None:
            first_mismatch = check
        checks.append(check)
        last = action
    return {'geometry':geometry,'case':case,'steps':201,'status':'PASS' if first_mismatch is None else 'INELIGIBLE',
        'max_abs':maxima,'min_far_far_clearance':min_far_clear,'first_mismatch':first_mismatch,
        'final_first5_equal':bool(np.array_equal(human_array(one.humans),human_array(far.humans[:5]))),
        'checks':checks,'note':'Fixed identical robot commands, full horizon including CPU-only post-terminal physics. No trajectory forcing.'}


def provenance(args):
    project = Path(args.project).resolve()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    return {'contract_sha256':sha(CONTRACT),'script_sha256':sha(__file__),'project_commit':commit,
        'source_sha256':{x:sha(project/x) for x in ['crowd_sim/envs/crowd_sim.py',
            'crowd_sim/envs/policy/orca.py','crowd_nav/policy/mamba_rl.py','crowd_nav/contracts.py']},
        'python':sys.version,'started_unix':time.time()}


def cpu(args):
    import numpy as np
    pin(args.project)
    cfg = cfg_read(args.config)
    actions = grid(cfg)
    cases = P['smoke_cases'] if args.phase=='smoke' else P['cases']
    started = time.perf_counter()
    scenes,isolations = [],[]
    for geometry in P['geometries']:
        for case in cases:
            e1,_ = scene(cfg,geometry,'E1',case)
            original5 = human_array(e1.humans)
            e2,_ = scene(cfg,geometry,'E2',case)
            original20 = human_array(e2.humans)
            assert np.array_equal(original5,original20[:5]), 'Original20 is not nested at reset'
            for condition in P['conditions']:
                env,r = scene(cfg,geometry,condition,case)
                assert np.array_equal(human_array(env.humans[:5]),original5) if condition in ('E1','E2','E3','E4') else True
                if condition=='E5':
                    mapped = original20.copy()
                    mapped[:,[0,1,5,6]] *= 2
                    assert np.array_equal(human_array(env.humans),mapped)
                scenes.append({'geometry':geometry,'condition':condition,'case':case,
                    'robot':[getattr(r,x) for x in FIELDS], 'humans':human_array(env.humans).tolist(),
                    'actual_orca_parameters':params(env.humans),
                    'crowd2m':sum(math.hypot(h.px-r.px,h.py-r.py)<=2 for h in env.humans),
                    'nominal_area':(math.pi*env.circle_radius**2 if geometry=='circle' else env.square_width**2) if condition!='E3' else None})
            isolated = isolation_pair(cfg,geometry,case,actions)
            isolations.append(isolated)
            print(json.dumps({k:isolated[k] for k in ('geometry','case','steps','status','max_abs')}),flush=True)
    result = {'provenance':provenance(args),'phase':args.phase,'scenes':scenes,'isolations':isolations,
        'status':'PASS' if all(x['status']=='PASS' for x in isolations) else 'INELIGIBLE',
        'wall_seconds':time.perf_counter()-started,'formal_model_evaluation':False}
    save(BASE/'work'/('cpu-'+args.phase+'.json'),result)


def acceptance(args):
    assert args.acceptance, 'Public independent-fix acceptance JSON required; no GPU work before PASS'
    a = json.loads(Path(args.acceptance).read_text())
    assert a['status']=='PASS' and a['history_test']=='PASS' and a['terminal_test']=='PASS', a
    tested = a['tested_commit']
    for commit in P['required_public_acceptance']:
        subprocess.run(['git','merge-base','--is-ancestor',commit,tested],cwd=args.project,check=True)
    # A cherry-pick has a new SHA. Compare tested policy contents, not branch ancestry.
    source = 'crowd_nav/policy/mamba_rl.py'
    expected = subprocess.check_output(['git','show',tested+':'+source],cwd=args.project)
    assert hashlib.sha256(expected).hexdigest()==sha(Path(args.project)/source), 'Policy differs from accepted code'
    return a


def budget():
    assert time.time()<P['started_unix']+P['budget_wall_seconds']-600, '8h budget reached; preserve outputs, report INCOMPLETE'


def load_net(args):
    import torch
    acceptance(args)
    item = manifest_item(args,args.model)
    spec = importlib.util.spec_from_file_location('b_reference',Path(args.causal)/'reference.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    cfg,net,digest = ref.load(SimpleNamespace(project=args.project,config=item['config'],
        backbone=item['backbone'],arm_dir=str(Path(item['weight']).parent),
        expected_checkpoint_sha=item['weight_sha256'],human_select='ttc5'))
    assert net.seq_len==24 and net.test_action_smoothing==.3
    assert net.test_min_clearance==.2 and net.test_risk_lambda==.8
    assert net.gamma==.99 and cfg.getfloat('reward','progress_reward')==0
    return item,cfg,net


def cv_mpc(env, actions, last):
    """Bounded CV receding-horizon beam search. No hidden goals/futures."""
    import numpy as np
    from crowd_sim.envs.utils.action import ActionXY
    humans = [h.get_observable_state() for h in env.humans]
    # Each node owns only a robot kinematic snapshot, CV observable humans,
    # and actual previously executed command. No env.step or ORCA oracle.
    beams = [(0.,copy.copy(env.robot),humans,last,None)]
    for depth in range(4):
        expanded = []
        for acc,robot,hs,prev,first in beams:
            cmds = commands(actions,prev)
            fake = SimpleNamespace(robot=robot,humans=hs,time_step=env.time_step,config=env.config,
                progress_reward=0.,time_penalty=0.,stand_penalty=0.,discomfort_dist=.2,
                discomfort_penalty_factor=env.discomfort_penalty_factor,
                success_reward=env.success_reward,collision_penalty=env.collision_penalty)
            _,clear = immediate(fake,cmds)
            xy = np.asarray([robot.compute_position(c,.25) for c in cmds])
            prev_dist = np.hypot(robot.px-robot.gx,robot.py-robot.gy)
            d = np.hypot(xy[:,0]-robot.gx,xy[:,1]-robot.gy)
            raw = prev_dist-d-.05*.25-2.*np.maximum(.2-clear,0.)*.25-100.*(clear<0)
            local = filter_scores(raw,clear)
            legal = clear>=.2
            eligible = np.flatnonzero(legal) if legal.any() else np.arange(80)
            order = sorted(eligible.tolist(),key=lambda i:(-local[i],i))[:4]
            nh = [SimpleNamespace(px=h.px+h.vx*.25,py=h.py+h.vy*.25,vx=h.vx,vy=h.vy,radius=h.radius) for h in hs]
            for i in order:
                nr = copy.copy(robot)
                nr.step(cmds[i])
                expanded.append((acc+float(local[i]),nr,nh,cmds[i],i if first is None else first))
        expanded.sort(key=lambda b:(-b[0],b[4]))
        beams = expanded[:4]
    # Aggregate bounded continuations by first action. This is not exhaustive MPC.
    return beams[0][4]


def choose_reference(controller, env, out, actions, last, orca):
    import numpy as np
    if controller=='cv-mpc':
        i = cv_mpc(env,actions,last)
    else:
        proposal = orca.predict(state(env))
        cmd = np.asarray([[a.vx,a.vy] for a in out['commands']])
        raw = -((cmd-np.asarray([proposal.vx,proposal.vy]))**2).sum(1)
        i = int(filter_scores(raw,out['clearance'].detach().cpu().numpy()).argmax())
    return i


class ReferenceInterface:
    """CPU-only shared observation/command/safety interface, no value network."""
    def __init__(self,cfg):
        self.action_space = grid(cfg)
        self.multiagent_training = True
        self._history = deque(maxlen=24)
        self._last_action = None
        self.selection_indices = None
        self.env = None

    def set_env(self,env):
        self.env = env

    def reset_episode_stats(self):
        self._history.clear()
        self._last_action = None

    def score_sarl_candidates(self,s):
        import torch
        self.selection_indices = ttc_ids(s)
        cmds = commands(self.action_space,self._last_action)
        r,c = immediate(self.env,cmds)
        return {'commands':cmds,'current_token':token(s),'rewards':torch.from_numpy(r),
            'clearance':torch.from_numpy(c),'values':torch.zeros(80),
            'raw_scores':torch.from_numpy(r),'scores':torch.from_numpy(filter_scores(r,c))}


def rollout(args, cfg, net, controller, geometry, condition, case, override=None, reference=None):
    import numpy as np
    import torch
    from crowd_sim.envs.policy.orca import ORCA
    env,robot = scene(cfg,geometry,condition,case,net)
    orca = ORCA()
    orca.configure(cfg)
    actions = list(net.action_space)
    shadow = scene(cfg,geometry,'E3',case)[0] if condition=='E1' else None
    last = None
    total,discounted,path_length = 0.,0.,0.
    minima = {'endpoint':1e30,'CV_swept':1e30,'actual_swept':1e30}
    records,latencies = [],[]
    for k in range(201):
        s = state(env)
        if controller=='model':
            torch.cuda.synchronize()
        started = time.perf_counter()
        if controller=='model':
            # Explicit read-only score then one real history append, matching policy.
            out = net.score_sarl_candidates(s)
            idx = int(out['scores'].argmax().item())
        else:
            out = net.score_sarl_candidates(s)
            idx = choose_reference(controller,env,out,actions,last,orca)
        if controller=='model':
            torch.cuda.synchronize()
        latencies.append(1000*(time.perf_counter()-started))
        rob = [getattr(robot,x) for x in FIELDS]
        cmd = np.asarray([[a.vx,a.vy] for a in out['commands']])
        d = np.hypot(robot.px-robot.gx,robot.py-robot.gy)
        progress = d-np.hypot(robot.px+cmd[:,0]*.25-robot.gx,robot.py+cmd[:,1]*.25-robot.gy)
        record = {'step':k,'robot':rob,'humans':human_array(env.humans).tolist(),
            'ids':list(net.selection_indices),'goal_distance':float(d),'progress':progress.tolist(),
            'commands':cmd.tolist(),'chosen':idx,'crowd2m':sum(math.hypot(h.px-robot.px,h.py-robot.py)<=2 for h in env.humans),
            'token_sha256':hashlib.sha256(np.ascontiguousarray(out['current_token']).tobytes()).hexdigest(),
            'history_sha256':hashlib.sha256(np.ascontiguousarray(list(net._history)).tobytes()).hexdigest()}
        for key in ('rewards','clearance','values','raw_scores','scores'):
            record[key] = out[key].detach().cpu().tolist()
        record['n_safe'] = int(sum(x>=.2 for x in record['clearance']))
        if reference is not None and (override is None or k<=override['step']):
            expected = reference['steps'][k]
            assert all(record[key]==expected[key] for key in record), ('prefix replay',case,k)
        if override is not None and k==override['step']:
            idx = override['index']
            record['executed_override'] = idx
        action = out['commands'][idx]
        if shadow is not None:
            # Full legal-world negative control under this exact policy trajectory.
            assert np.array_equal(human_array(env.humans),human_array(shadow.humans[:5])), ('E3 humans',k)
            assert params(env.humans)==params(shadow.humans[:5])
            assert ttc_ids(s)==ttc_ids(state(shadow))
            assert np.array_equal(out['current_token'],token(state(shadow)))
            rr,cc = immediate(shadow,out['commands'])
            assert np.array_equal(rr,np.asarray(record['rewards'])), ('E3 r80',k)
            assert np.array_equal(cc,np.asarray(record['clearance'])), ('E3 clearance80',k)
        net._history.append(out['current_token'])
        net._last_action = action
        net._last_grid_index = idx
        last = action
        pre = copy.copy(env)
        pre.robot = copy.copy(robot)
        pre.humans = [copy.copy(h) for h in env.humans]
        cv = float(swept(pre,[action])[0])
        reward,done,info = env_step(env,action)
        if shadow is not None:
            sr,sd,si = env_step(shadow,action)
            assert (reward,done,info)==(sr,sd,si)
            assert neighbors(env)==neighbors(shadow)
            assert np.array_equal(human_array(env.humans),human_array(shadow.humans[:5]))
        actual = float(swept(pre,[action],human_array(env.humans))[0])
        endpoint = min(float(np.hypot(h.px-robot.px,h.py-robot.py)-h.radius-robot.radius) for h in env.humans)
        for name,value in [('endpoint',endpoint),('CV_swept',cv),('actual_swept',actual)]:
            minima[name] = min(minima[name],value)
        path_length += math.hypot(robot.px-rob[0],robot.py-rob[1])
        total += reward
        discounted += .99**k*reward
        record.update(command=[action.vx,action.vy],reward=reward,terminal=done,
            info=info,actual_swept=actual,CV_swept=cv,endpoint=endpoint)
        records.append(record)
        if k%20==0:
            budget()
        if done:
            event = info['event']
            outcome = 'success' if event=='reach_goal' else 'collision' if event=='collision' else 'timeout'
            break
    else:
        raise AssertionError('No terminal at201steps')
    return {'model':args.model,'controller':controller,'geometry':geometry,'condition':condition,'case':case,
        'outcome':outcome,'return':discounted,'discounted_return':discounted,'undiscounted_return':total,'time':float(env.global_time),
        'path_length':path_length,'minimum_clearance':minima,'steps':records,
        'latency_ms_p50':float(np.quantile(latencies,.5)),'latency_ms_p95':float(np.quantile(latencies,.95)),
        'E3_shadow_exact':condition=='E1','override':override}


def baseline(args):
    import torch
    budget()
    if args.controller=='model':
        item,cfg,net = load_net(args)
    else:
        acceptance(args)
        item = manifest_item(args,args.model)
        cfg = cfg_read(item['config'])
        net = ReferenceInterface(cfg)
    phase_cases = P['smoke_cases'] if args.phase=='smoke' else P['cases']
    cases = phase_cases if args.case is None else [args.case]
    assert all(c in phase_cases for c in cases)
    if args.phase=='main':
        gate = BASE/'work'/'smoke-accepted.json'
        assert gate.is_file() and json.loads(gate.read_text())['status']=='PASS', 'Main-scheduled full smoke gate required'
    output = BASE/'work'/args.phase/args.controller/args.model/(args.geometry+'-'+args.condition+('-'+str(args.case) if args.case is not None else '')+'.jsonl.gz')
    assert not output.exists(), ('Never overwrite outcomes',output)
    output.parent.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    meta = dict(provenance(args),model=item,phase=args.phase,cases=cases,controller=args.controller,
                acceptance=json.loads(Path(args.acceptance).read_text()))
    save(output.with_suffix('.start.json'),meta)
    with torch.no_grad(),gzip.open(output,'wt',compresslevel=1) as f:
        for case in cases:
            episode = rollout(args,cfg,net,args.controller,args.geometry,args.condition,case)
            write(f,episode)
            print(json.dumps({k:episode[k] for k in ('case','condition','outcome','return','time','latency_ms_p50')}),flush=True)
    save(output.with_suffix('.done.json'),{'status':'PASS','episodes':len(cases),'wall_seconds':time.perf_counter()-started,
        'output_sha256':sha(output),'provenance':meta})


def episodes(phase,controller,model,geometry,condition):
    folder = BASE/'work'/phase/controller/model
    combined = folder/(geometry+'-'+condition+'.jsonl.gz')
    files = [combined] if combined.is_file() else sorted(folder.glob(geometry+'-'+condition+'-*.jsonl.gz'))
    result = []
    for file in files:
        done = file.with_suffix('.done.json')
        assert done.is_file(), ('Incomplete shard',file)
        meta = json.loads(done.read_text())
        assert meta['status']=='PASS' and meta['output_sha256']==sha(file), ('Corrupt shard',file)
        result.extend(read_rows(file))
    cases = P['smoke_cases'] if phase=='smoke' else P['cases']
    assert sorted(e['case'] for e in result)==cases, ('Missing/duplicate frozen cases',phase,controller,model,geometry,condition)
    return sorted(result,key=lambda e:e['case'])


def smoke_gate(args):
    missing = []
    counts = 0
    for controller in ['model']+P['references']:
        models = P['models'] if controller=='model' else [P['models'][0]]
        for model in models:
            for geometry in P['geometries']:
                for condition in P['conditions']:
                    try:
                        counts += len(episodes('smoke',controller,model,geometry,condition))
                    except (AssertionError,OSError) as e:
                        missing.append(str(e))
    cpu_gate = json.loads((BASE/'work/cpu-main.json').read_text())
    assert cpu_gate['status']=='PASS', 'CPU E3 qualification failed'
    result = {'status':'PASS' if not missing else 'INCOMPLETE','episodes':counts,'missing':missing,
        'contract_sha256':sha(CONTRACT),'cpu_qualification_sha256':sha(BASE/'work/cpu-main.json')}
    save(BASE/'work/smoke-accepted.json',result)
    print(json.dumps(result),flush=True)
    assert not missing, 'All frozen smoke conditions/references must complete before main'


def a1(row):
    good = [i for i in range(80) if row['clearance'][i]>=.2 and row['progress'][i]>=.02]
    return max(good,key=lambda i:(row['raw_scores'][i],-i)) if good else None


def eligible(row,kind):
    return a1(row) is not None and (kind!='timeout' or
        (row['step']>=20 and row['goal_distance']>1 and row['progress'][row['chosen']]<.02))


def opportunity_distance(a,b):
    diffs = [abs(a['step']-b['step'])/8.,abs(a['goal_distance']-b['goal_distance']),
             abs(a['n_safe']-b['n_safe'])/16.,abs(a['crowd2m']-b['crowd2m'])/2.]
    return (sum(diffs),diffs) if max(diffs)<=1. else None


def roots(args):
    path = BASE/'work'/('roots-'+args.model+'.json')
    assert not path.exists(), 'Roots are immutable after freezing'
    frozen,denominator = [],[]
    for geometry in P['geometries']:
        for condition in P['conditions']:
            eps = episodes('main','model',args.model,geometry,condition)
            successes = [e for e in eps if e['outcome']=='success']
            candidates = []
            for ep in eps:
                if ep['outcome']=='success':
                    continue
                first_unsafe = next((r['step'] for r in ep['steps'] if r['n_safe']==0),None)
                rows = ep['steps'] if ep['outcome']=='timeout' else (
                    ep['steps'][max(0,first_unsafe-12):first_unsafe] if first_unsafe is not None else [])
                row = next((r for r in rows if eligible(r,ep['outcome']) and
                            (ep['outcome']=='timeout' or a1(r)!=r['chosen'])),None)
                denominator.append({'geometry':geometry,'condition':condition,'case':ep['case'],
                    'outcome':ep['outcome'],'has_root':row is not None,'selected_within_cap':False})
                if row is not None:
                    candidates.append({'model':args.model,'geometry':geometry,'condition':condition,
                        'case':ep['case'],'step':row['step'],'type':ep['outcome'],'snapshot':row,
                        'A0':row['chosen'],'A1':a1(row),'matched_control':None})
            used = set()
            # Fixed case order, never based on branch outcome or scoring gap.
            for root in candidates[:2]:
                next(d for d in denominator if d['geometry']==geometry and d['condition']==condition and
                    d['case']==root['case'])['selected_within_cap'] = True
                options = []
                for ep in successes:
                    if ep['case'] in used:
                        continue
                    for row in ep['steps']:
                        if not eligible(row,root['type']) or a1(row)==row['chosen']:
                            continue
                        diff = opportunity_distance(root['snapshot'],row)
                        if diff is not None:
                            options.append((diff[0],ep['case'],row['step'],row,diff[1]))
                if options:
                    _,case,k,row,diffs = min(options,key=lambda x:x[:3])
                    used.add(case)
                    control = {'model':args.model,'geometry':geometry,'condition':condition,
                        'case':case,'step':k,'type':'success-control','snapshot':row,
                        'A0':row['chosen'],'A1':a1(row),'matched_failure':root['case'],
                        'match_standardized_differences':diffs}
                    root['matched_control'] = [case,k]
                    frozen.append(control)
                frozen.append(root)
    frozen.sort(key=lambda r:(P['geometries'].index(r['geometry']),P['conditions'].index(r['condition']),r['case'],r['step']))
    for i,root in enumerate(frozen):
        root['root_id'] = i
    result = {'status':'FROZEN','roots':frozen,'all_failure_denominator':denominator,
        'contract_sha256':sha(CONTRACT),'frozen_unix':time.time(),'branch_results_exist':False,
        'unmatched_failure_roots':sum(r['type']!='success-control' and r['matched_control'] is None for r in frozen)}
    save(path,result)
    save(path.with_suffix('.sha256.json'),{'sha256':sha(path)})
    print(json.dumps({'model':args.model,'roots':len(frozen),'unmatched':result['unmatched_failure_roots']}))


def counterfactual(args):
    import torch
    item,cfg,net = load_net(args)
    frozen = BASE/'work'/('roots-'+args.model+'.json')
    assert sha(frozen)==json.loads(frozen.with_suffix('.sha256.json').read_text())['sha256']
    data = json.loads(frozen.read_text())
    selected = [r for r in data['roots'] if r['geometry']==args.geometry and r['condition']==args.condition]
    refs = {e['case']:e for e in episodes('main','model',args.model,args.geometry,args.condition)}
    output = BASE/'work/branches'/args.model/(args.geometry+'-'+args.condition+'.jsonl.gz')
    assert not output.exists(), output
    output.parent.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    with torch.no_grad(),gzip.open(output,'wt',compresslevel=1) as f:
        for root in selected:
            budget()
            reference = refs[root['case']]
            a0 = rollout(args,cfg,net,'model',args.geometry,args.condition,root['case'],reference=reference)
            deterministic_keys = [k for k in reference if not k.startswith('latency_ms_')]
            assert all(a0[k]==reference[k] for k in deterministic_keys), ('A0 exact full replay',root['root_id'])
            if root['A1']==root['A0']:
                aone = a0
            else:
                aone = rollout(args,cfg,net,'model',args.geometry,args.condition,root['case'],
                    override={'step':root['step'],'index':root['A1']},reference=reference)
            start_step = root['step']
            g0 = sum(.99**(i-start_step)*r['reward'] for i,r in enumerate(a0['steps']) if i>=start_step)
            g1 = sum(.99**(i-start_step)*r['reward'] for i,r in enumerate(aone['steps']) if i>=start_step)
            record = {'root':root,'A0':a0,'A1':aone,'root_G0':g0,'root_G1':g1,
                'delta_root_return':g1-g0,'delta_episode_discounted_return':aone['discounted_return']-a0['discounted_return'],
                'rescued':a0['outcome']!='success' and aone['outcome']=='success',
                'destroyed':a0['outcome']=='success' and aone['outcome']!='success',
                'A0_exact':True,'distinct_action':root['A1']!=root['A0']}
            # Timings are retained but not part of deterministic physical replay.
            write(f,record)
            print(json.dumps({'root':root['root_id'],'delta':g1-g0,'rescued':record['rescued'],'destroyed':record['destroyed']}),flush=True)
    save(output.with_suffix('.done.json'),{'status':'PASS','roots':len(selected),'sha256':sha(output),
        'root_freeze_sha256':sha(frozen),'wall_seconds':time.perf_counter()-start,'weight_sha256':item['weight_sha256']})


def stats(eps):
    import numpy as np
    n = len(eps)
    return {'episodes':n,'success':sum(e['outcome']=='success' for e in eps),
        'collision':sum(e['outcome']=='collision' for e in eps),'timeout':sum(e['outcome']=='timeout' for e in eps),
        'SR':sum(e['outcome']=='success' for e in eps)/n,
        'CR':sum(e['outcome']=='collision' for e in eps)/n,
        'TO':sum(e['outcome']=='timeout' for e in eps)/n,
        **{k:float(np.mean([e[k] for e in eps])) for k in ['return','discounted_return','time','path_length','latency_ms_p50','latency_ms_p95']},
        'physical_collision_episodes':sum(e['minimum_clearance']['actual_swept']<0 for e in eps),
        'successful_but_physical_collision':sum(e['outcome']=='success' and e['minimum_clearance']['actual_swept']<0 for e in eps),
        'safe_count_step_mean':float(np.mean([r['n_safe'] for e in eps for r in e['steps']])),
        'crowd2m_step_mean':float(np.mean([r['crowd2m'] for e in eps for r in e['steps']]))}


def paired(a,b):
    import numpy as np
    assert [e['case'] for e in a]==[e['case'] for e in b]
    rng = np.random.RandomState(41995)
    result = {}
    for key in ('success','discounted_return','return','time','path_length'):
        av = np.asarray([int(e['outcome']=='success') if key=='success' else e[key] for e in a])
        bv = np.asarray([int(e['outcome']=='success') if key=='success' else e[key] for e in b])
        delta = bv-av
        resamples = delta[rng.randint(0,len(delta),(2000,len(delta)))].mean(1)
        result[key] = {'mean_delta_B_minus_A':float(delta.mean()),'case_delta':delta.tolist(),
            'case_bootstrap95':np.quantile(resamples,[.025,.975]).tolist()}
    return result


def summarize(args):
    output = {'status':'COMPLETE','contract_sha256':sha(CONTRACT),'groups':[],'paired':[],
              'branches':[],'cross_condition_opportunity_matches':[],'missing':[]}
    loaded = {}
    for controller in ['model']+P['references']:
        models = P['models'] if controller=='model' else [P['models'][0]]
        for model in models:
            for geometry in P['geometries']:
                for condition in P['conditions']:
                    key = (controller,model,geometry,condition)
                    try:
                        eps = episodes('main',*key)
                        loaded[key] = eps
                        output['groups'].append({'controller':controller,'model':model,'geometry':geometry,'condition':condition,**stats(eps)})
                    except (AssertionError,OSError) as e:
                        output['missing'].append(str(e))
                for a,b in [('E1','E2'),('E1','E3'),('E2','E4'),('E1','E5'),('E2','E5')]:
                    ka,kb = (controller,model,geometry,a),(controller,model,geometry,b)
                    if ka in loaded and kb in loaded:
                        output['paired'].append({'controller':controller,'model':model,'geometry':geometry,'A':a,'B':b,
                            'metrics':paired(loaded[ka],loaded[kb])})
                        if a=='E1' and b=='E3':
                            for e1,e3 in zip(loaded[ka],loaded[kb]):
                                for field in ('outcome','return','discounted_return','undiscounted_return','time','path_length','minimum_clearance'):
                                    assert e1[field]==e3[field], ('E3 model trajectory mismatch',model,geometry,e1['case'],field)
    for model in P['models']:
        rootpath = BASE/'work'/('roots-'+model+'.json')
        if not rootpath.is_file():
            output['missing'].append('roots not frozen: '+model)
            continue
        frozen = json.loads(rootpath.read_text())
        for geometry in P['geometries']:
            for condition in P['conditions']:
                file = BASE/'work/branches'/model/(geometry+'-'+condition+'.jsonl.gz')
                if not file.is_file() or not file.with_suffix('.done.json').is_file():
                    output['missing'].append('branches missing: '+str(file))
                    continue
                branches = list(read_rows(file))
                done = json.loads(file.with_suffix('.done.json').read_text())
                assert done['sha256']==sha(file) and len(branches)==done['roots']
                failures = [b for b in branches if b['root']['type']!='success-control']
                controls = [b for b in branches if b['root']['type']=='success-control']
                output['branches'].append({'model':model,'geometry':geometry,'condition':condition,
                    'all_failed_episodes':sum(d['geometry']==geometry and d['condition']==condition for d in frozen['all_failure_denominator']),
                    'sampled_failure_roots':len(failures),'success_controls':len(controls),
                    'unmatched':sum(b['root']['matched_control'] is None for b in failures),
                    'rescued':sum(b['rescued'] for b in failures),'destroyed_controls':sum(b['destroyed'] for b in controls),
                    'failure_return_deltas':[b['delta_root_return'] for b in failures],
                    'control_return_deltas':[b['delta_root_return'] for b in controls],
                    'all_A0_exact':all(b['A0_exact'] for b in branches)})
            for condition in ['E2','E3','E4','E5']:
                r1 = [r for r in frozen['roots'] if r['geometry']==geometry and r['condition']=='E1' and r['type']!='success-control']
                rb = [r for r in frozen['roots'] if r['geometry']==geometry and r['condition']==condition and r['type']!='success-control']
                matches,used = [],set()
                for root in r1:
                    candidates = [(opportunity_distance(root['snapshot'],q['snapshot']),q) for q in rb if q['root_id'] not in used]
                    candidates = [q for q in candidates if q[0] is not None and q[1]['type']==root['type']]
                    if candidates:
                        dist,q = min(candidates,key=lambda t:(t[0][0],t[1]['case'],t[1]['step']))
                        used.add(q['root_id'])
                        matches.append({'E1_root':root['root_id'],'other_root':q['root_id'],'standardized_diff':dist[1]})
                output['cross_condition_opportunity_matches'].append({'model':model,'geometry':geometry,'B':condition,
                    'matches':matches,'status':'MATCHED' if matches else 'NOT_IDENTIFIED_NO_MATCHES'})
    if output['missing']:
        output['status'] = 'INCOMPLETE'
    output['B4_training_executed'] = False
    output['method_level_progress'] = False
    save(BASE/'work/summary.json',output)
    print(json.dumps({'status':output['status'],'groups':len(output['groups']),'branches':len(output['branches']),
                      'missing':len(output['missing'])}),flush=True)


def jobs(args):
    """Print scheduler argv only. This function never starts subprocess work."""
    common = ['--project',str(Path(args.project).resolve()),'--causal',str(Path(args.causal).resolve())]
    if args.acceptance:
        common += ['--acceptance',str(Path(args.acceptance).resolve())]
    result = []
    for controller in ['model']+P['references']:
        models = P['models'] if controller=='model' else [P['models'][0]]
        for model in models:
            for geometry in P['geometries']:
                for condition in P['conditions']:
                    name = '-'.join([args.phase,controller,model,geometry,condition])
                    result.append({'name':name,'resource':'GPU' if controller=='model' else 'CPU',
                        'episodes':3 if args.phase=='smoke' else 16,
                        'argv':[sys.executable,str(BASE/'run-b.py'),'baseline',*common,'--phase',args.phase,
                                '--model',model,'--controller',controller,'--geometry',geometry,'--condition',condition]})
    print(json.dumps({'phase':args.phase,'jobs':result,'episodes':sum(j['episodes'] for j in result)},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['cpu','baseline','smoke-gate','roots','counterfactual','summarize','jobs'])
    parser.add_argument('--project',default=str(BASE/'project'))
    parser.add_argument('--causal',default='/home/abc/workspace/ttc5-causal-20261011')
    parser.add_argument('--config',default='/home/abc/workspace/ttc5-causal-20261011/inputs/kda-s419-t24/config.ini')
    parser.add_argument('--phase',choices=['smoke','main'],default='smoke')
    parser.add_argument('--model',choices=P['models'],default=P['models'][0])
    parser.add_argument('--controller',choices=['model']+P['references'],default='model')
    parser.add_argument('--geometry',choices=P['geometries'],default='circle')
    parser.add_argument('--condition',choices=P['conditions'],default='E1')
    parser.add_argument('--case',type=int)
    parser.add_argument('--acceptance')
    args = parser.parse_args()
    pin(args.project)
    if args.mode=='cpu':
        cpu(args)
    elif args.mode=='baseline':
        baseline(args)
    elif args.mode=='smoke-gate':
        smoke_gate(args)
    elif args.mode=='roots':
        roots(args)
    elif args.mode=='counterfactual':
        counterfactual(args)
    elif args.mode=='jobs':
        jobs(args)
    else:
        summarize(args)


if __name__=='__main__':
    main()
