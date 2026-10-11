"""FCS2 受限教师阶梯:Mean / Age / Full,8 seed,配对分析。

配对单位是训练 seed(不是回合),与 FCS2 既有协议一致。
每个 (arm, seed) 先在场景与密度上合并,再以 seed 为推断单位。
"""
import json, glob, collections, itertools
import numpy as np
from scipy import stats

D = '/home/abc/workspace/fcs2_matched_teacher_20260926/eval/'
ARMS = ('sensor', 'mean', 'age', 'full')
SEEDS = (1207, 2407, 3607, 4807, 6007, 7207, 8407, 9607)

rows = collections.defaultdict(list)
for f in glob.glob(D + '*.jsonl'):
    for line in open(f):
        d = json.loads(line)
        rows[(d['arm'], d['seed'])].append(d)

present = sorted({a for a, _ in rows})
print('臂:', present, '| seed 数/臂:', {a: len({s for x, s in rows if x == a}) for a in present})
n0 = {k: len(v) for k, v in rows.items()}
print('每 (臂,seed) 回合数: 最小 %d 最大 %d' % (min(n0.values()), max(n0.values())))

def rate(recs, outcome):
    return 100. * np.mean([r['outcome'] == outcome for r in recs])

# 按密度拆分,也给总体
dens = sorted({r['humans'] for v in rows.values() for r in v})
print('\n密度:', dens)

def cell(arm, seed, hum=None):
    v = rows[(arm, seed)]
    if hum is not None:
        v = [r for r in v if r['humans'] == hum]
    if not v: return None
    return (rate(v, 'reach_goal'), rate(v, 'collision'), rate(v, 'timeout'), len(v))

print('\n%-6s %-8s %8s %8s %8s %7s' % ('N', 'arm', 'SR', 'CR', 'TO', 'n/seed'))
table = {}
for hum in dens + [None]:
    lab = str(hum) if hum else '全部'
    for arm in present:
        vals = [cell(arm, s, hum) for s in SEEDS]
        vals = [v for v in vals if v]
        if len(vals) != len(SEEDS):
            print('  %-6s %-8s 不完整 (%d/%d seed)' % (lab, arm, len(vals), len(SEEDS))); continue
        a = np.array([v[:3] for v in vals])
        table[(arm, hum)] = a
        print('%-6s %-8s %8.2f %8.2f %8.2f %7d' % (lab, arm, a[:,0].mean(), a[:,1].mean(), a[:,2].mean(), vals[0][3]))

print('\n=== 配对差(同 seed),8 seed t 区间 ===')
print('%-6s %-18s %10s %-22s %s' % ('N', '对比', '差值', '95% CI', '判读'))
for hum in dens + [None]:
    lab = str(hum) if hum else '全部'
    for a, b in (('mean','sensor'), ('age','mean'), ('full','age'), ('full','mean')):
        if (a, hum) not in table or (b, hum) not in table: continue
        d = table[(a,hum)][:,0] - table[(b,hum)][:,0]
        if d.std() == 0:
            lo = hi = d.mean()
        else:
            lo, hi = stats.t.interval(.95, len(d)-1, d.mean(), stats.sem(d))
        verdict = '已分辨' if lo*hi > 0 else ('等价(|CI|<3pp)' if max(abs(lo),abs(hi)) < 3 else '未分辨')
        print('%-6s %-18s %+10.2f  [%+6.2f,%+6.2f]  %s' % (lab, a+' - '+b, d.mean(), lo, hi, verdict))

print('\n=== 碰撞率的配对差(FCS3 的教训:不能只看 SR)===')
print('%-6s %-18s %10s %-22s %s' % ('N', '对比', 'ΔCR', '95% CI', '判读'))
for hum in dens + [None]:
    lab = str(hum) if hum else '全部'
    for a, b in (('age','mean'), ('full','age'), ('full','mean')):
        if (a,hum) not in table or (b,hum) not in table: continue
        d = table[(a,hum)][:,1] - table[(b,hum)][:,1]
        lo, hi = (d.mean(), d.mean()) if d.std()==0 else stats.t.interval(.95, len(d)-1, d.mean(), stats.sem(d))
        print('%-6s %-18s %+10.2f  [%+6.2f,%+6.2f]  %s' % (lab, a+' - '+b, d.mean(), lo, hi,
              '已分辨' if lo*hi>0 else '未分辨'))

print('\n=== 与旧 4-seed 批次(同 belief_il 协议)的一致性核对 ===')
old = collections.defaultdict(list)
for f in glob.glob('/home/abc/workspace/severe_train_check_20260923/eval/belief_il_*.jsonl'):
    for line in open(f):
        d = json.loads(line); old[(d['arm'], d['seed'])].append(d)
oldseeds = sorted({s for _, s in old})
print('旧批 臂/seed:', {a: sorted({s for x,s in old if x==a}) for a in sorted({a for a,_ in old})})
for arm in ('mean','age'):
    o = [(a,s) for a,s in old if a==arm]
    if not o: continue
    osr = np.mean([rate(old[k],'reach_goal') for k in o])
    nsr = np.mean([rate(rows[(arm,s)],'reach_goal') for s in oldseeds if (arm,s) in rows])
    print('  %-5s 旧批(%d seed) SR %.2f  |  新批同 seed SR %.2f  |  差 %+.2f'
          % (arm, len(o), osr, nsr, nsr-osr))
