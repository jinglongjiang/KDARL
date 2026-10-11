import glob, json
import numpy as np
R='/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/CrowdNav/results/occlusion_rfs_20260920'
S=(1207,2407,3607,4807); T={4:3.182}
def load(f): return [json.loads(l) for l in open(f)]
def sr(rows,h=None,o='reach_goal'): return 100*np.mean([r['outcome']==o for r in rows if h in (None,r['humans'])])
def ci(d):
    m=np.mean(d); h=T[len(d)]*np.std(d,ddof=1)/np.sqrt(len(d)); return m,m-h,m+h,sum(x>0 for x in d)
print("## ③a 教师只看学生观测(Mean 臂,Severe IL)")
try:
    new={s:load(f'/home/abc/workspace/severe_train_check_20260923/eval/belief_il_mean_{s}.jsonl') for s in S}
    old={s:load(f'{R}/v3_eval_fixed/mean_{s}.jsonl') for s in S}
    d=[sr(new[s])-sr(old[s]) for s in S]; m,lo,hi,p=ci(d)
    print(f"   原协议 {np.mean([sr(old[s]) for s in S]):.2f} → 教师用学生观测 {np.mean([sr(new[s]) for s in S]):.2f}"
          f"   {m:+.2f} pp [{lo:+.2f},{hi:+.2f}] 正向 {p}/4   逐种子 {[round(x,1) for x in d]}")
    for s in S:
        r=json.load(open(f'/home/abc/workspace/severe_train_check_20260923/runs/belief_il_mean_{s}/results.json')); print(f"     seed {s}: 教师成功 {r['teacher_success']}/500  示范转移 {r['demo_transitions']}")
except FileNotFoundError as e: print("   未完成:", e.filename.split('/')[-1])
print("\n## ③b Standard IL 初始化 → Severe RL")
try:
    A={s:load(f'/home/abc/workspace/severe_train_check_20260923/eval/stdil_sevrl_age_{s}.jsonl') for s in S}; M={s:load(f'/home/abc/workspace/severe_train_check_20260923/eval/stdil_sevrl_mean_{s}.jsonl') for s in S}
    print(f"   Mean {np.mean([sr(M[s]) for s in S]):.2f}   Age {np.mean([sr(A[s]) for s in S]):.2f}")
    d=[sr(A[s])-sr(M[s]) for s in S]; m,lo,hi,p=ci(d)
    print(f"   Age − Mean {m:+.2f} pp [{lo:+.2f},{hi:+.2f}] 正向 {p}/4   逐种子 {[round(x,1) for x in d]}")
    print(f"   对照:Severe 训练的 RL 为 +6.03;Severe IL 为 +6.65")
    for lab,X in (('Mean',M),('Age',A)):
        print(f"   {lab} 分人数 " + " / ".join(f"{np.mean([sr(X[s],h) for s in S]):.1f}" for h in (5,10,20))
              + f"   碰撞 {np.mean([sr(X[s],o='collision') for s in S]):.1f} 超时 {np.mean([sr(X[s],o='timeout') for s in S]):.1f}")
except FileNotFoundError as e: print("   未完成:", e.filename.split('/')[-1])
