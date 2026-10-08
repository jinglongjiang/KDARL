# -*- coding: utf-8 -*-
"""决策级诊断: 在真实评测状态上, 同时记录 value 原始排序、filter 的实际作用、
以及被执行动作的朝目标进度。用来区分三种失败形态的真实成因。"""
import argparse, json, sys, math
from pathlib import Path
import numpy as np, torch

def pin(project):
    root=str(Path(project).resolve()); sys.path.insert(0,root)
    import crowd_nav, crowd_sim, crowd_nav.policy.mamba_rl as m
    for mod in (crowd_nav,crowd_sim,m):
        assert str(Path(mod.__file__).resolve()).startswith(root), f"{mod.__name__} 解析错误"
    return root

def run(project, arm_dir, config, backbone, cells, cases, out, budget=10000):
    from crowd_nav.train import build_env_and_robot, bind_policy, load_config
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_sim.envs.utils.state import JointState
    ck=Path(arm_dir)/f'rl_model_ep{budget}.pth'
    data=torch.load(ck,map_location='cpu',weights_only=False)
    assert data['episode']==budget
    cfg=load_config(config); torch.manual_seed(42)
    cfg.set('mamba','temporal_backbone',backbone)
    net=MambaRLPolicy(cfg,device='cuda')
    w={(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k):v
       for k,v in data['policy_state'].items()}
    net.load_state_dict(w,strict=True); net=net.eval()
    net.multiagent_training=True; net.use_sarl_predict=True
    net.set_phase('test'); net.set_training_mode('rl'); net.epsilon=0.
    MC=net.test_min_clearance; RL=net.test_risk_lambda
    print(f"[CFG] min_clearance={MC} risk_lambda={RL} smoothing={net.test_action_smoothing}")

    eps_out=[]
    with torch.no_grad():
        for n,geo in cells:
            cfg.set('sim','human_num',str(n)); cfg.set('sim','test_sim',geo)
            env,robot=build_env_and_robot(cfg); bind_policy(robot,net,env,epsilon=0.)
            env.phase='test'
            for case in cases:
                if hasattr(net,'reset_episode_stats'): net.reset_episode_stats()
                env.reset(seed=case,options={'test_case':case})
                steps=0; outcome='timeout'; rec=[]
                prev_cmd=None
                while steps<201:
                    st=JointState(robot.get_full_state(),
                                  [h.get_observable_state() for h in env.humans])
                    r=net.score_sarl_candidates(st)
                    raw=r['raw_scores'].detach().cpu().numpy()
                    sc=r['scores'].detach().cpu().numpy()
                    clr=r['clearance'].detach().cpu().numpy()
                    cmds=r['commands']
                    rob=st.self_state
                    d_now=math.hypot(rob.px-rob.gx, rob.py-rob.gy)
                    # 每个候选执行后的朝目标进度
                    prog=np.array([d_now - math.hypot(rob.px+c.vx*net.time_step-rob.gx,
                                                      rob.py+c.vy*net.time_step-rob.gy)
                                   for c in cmds])
                    safe = clr >= MC
                    fin = int(sc.argmax()); rawtop = int(raw.argmax())
                    order = np.argsort(-raw)                     # raw 降序
                    rank_of_final = int(np.where(order==fin)[0][0]) + 1
                    # 最好的"安全且有进度"候选, 它在 raw 排序里第几
                    gp = np.where(safe & (prog>0))[0]
                    if len(gp):
                        best_gp = gp[np.argmax(prog[gp])]
                        rank_gp = int(np.where(order==best_gp)[0][0]) + 1
                        prog_gp = float(prog[best_gp])
                    else:
                        rank_gp=-1; prog_gp=float('nan')
                    cmd=cmds[fin]
                    rev=False
                    if prev_cmd is not None:
                        a=np.array([prev_cmd.vx,prev_cmd.vy]); b=np.array([cmd.vx,cmd.vy])
                        na,nb=np.linalg.norm(a),np.linalg.norm(b)
                        if na>1e-9 and nb>1e-9:
                            rev = float(np.dot(a,b)/(na*nb)) < 0.0   # 夹角>90°
                    rec.append(dict(n_safe=int(safe.sum()),
                                    all_unsafe=bool(safe.sum()==0),
                                    rawtop_safe=bool(safe[rawtop]),
                                    rawtop_filtered=bool((not safe[rawtop]) and safe.sum()>0),
                                    rank_of_final=rank_of_final,
                                    rank_best_safe_progress=rank_gp,
                                    prog_final=float(prog[fin]),
                                    prog_rawtop=float(prog[rawtop]),
                                    prog_best_safe=prog_gp,
                                    clr_final=float(clr[fin]),
                                    speed_final=float(math.hypot(cmd.vx,cmd.vy)),
                                    reversal=rev,
                                    d_goal=d_now))
                    prev_cmd=cmd
                    step=env.step(cmd if False else net.predict(st))
                    if len(step)==5:
                        _,rw,te,tr,info=step; done=te or tr
                    else:
                        _,rw,done,info=step
                    steps+=1
                    if done:
                        ev=(str(info.get('event','')) if isinstance(info,dict) else type(info).__name__).lower().replace('_','')
                        outcome='success' if ('success' in ev or 'reachgoal' in ev) else ('collision' if 'collision' in ev else 'timeout')
                        break
                d_end=math.hypot(robot.px-robot.gx, robot.py-robot.gy)
                path=sum(x['speed_final']*net.time_step for x in rec)
                eps_out.append(dict(case=case,human_num=n,geometry=geo,outcome=outcome,
                                    steps=steps,d_start=rec[0]['d_goal'],d_end=d_end,
                                    net_progress=rec[0]['d_goal']-d_end, path_len=path,
                                    steps_rec=rec))
                print(f"  {geo} n={n} case={case} -> {outcome} steps={steps} "
                      f"净进度={rec[0]['d_goal']-d_end:.2f}m 路程={path:.2f}m", flush=True)
    json.dump({'backbone':backbone,'episodes':eps_out}, open(out,'w'))
    print(f"[DONE] -> {out}")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',required=True); ap.add_argument('--arm-dir',required=True)
    ap.add_argument('--config',required=True); ap.add_argument('--backbone',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--cells',default='5')
    ap.add_argument('--n-cases',type=int,default=32)
    a=ap.parse_args()
    ALL=[(5,'circle_crossing'),(5,'square_crossing'),(10,'circle_crossing'),
         (10,'square_crossing'),(20,'circle_crossing'),(20,'square_crossing')]
    pin(a.project)
    run(a.project,a.arm_dir,a.config,a.backbone,
        [ALL[int(i)] for i in a.cells.split(',')],
        list(range(89000,89000+a.n_cases)), a.out)
