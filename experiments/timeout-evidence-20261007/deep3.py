import json, statistics as st
from pathlib import Path
W=Path('/home/abc/temp/cc_6env_20261007'); ARMS=('mamba','gru','kda')
RUNS=['kda_3000','kda_premerge_88000','kda_premerge_89000','kda']
CELLS=[(5,'circle_crossing'),(5,'square_crossing'),(10,'circle_crossing'),
       (10,'square_crossing'),(20,'circle_crossing'),(20,'square_crossing')]
def load(r,a): return json.load(open(W/r/f'{a}.json'))['episodes']
isto=lambda e: e['outcome'] not in ('success','collision')

print('='*100)
print('第8步  超时集中在哪里? (4 个跑 x 3 臂 合并 = 每格 128 个 episode)')
print('='*100)
print(f"{'环境':<22}" + ''.join(f"{a:>11}" for a in ARMS) + f"{'三臂合计':>11}{'占总超时':>11}")
print('-'*100)
tot={a:0 for a in ARMS}; cellsum={}
for hn,geo in CELLS:
    row=f"{f'{hn}人 {geo}':<22}"; s=0
    for a in ARMS:
        n=0; t=0
        for r in RUNS:
            sub=[e for e in load(r,a) if e['human_num']==hn and e['geometry']==geo]
            n+=len(sub); t+=sum(1 for e in sub if isto(e))
        row+=f"{t:>6}/{n:<4}"; s+=t; tot[a]+=t
    cellsum[(hn,geo)]=s
    row+=f"{s:>11}"
    print(row, end='')
    print(f"{s/sum(1 for _ in [0]):>0}", end='')
    print()
G=sum(cellsum.values())
print('-'*100)
for hn,geo in CELLS:
    print(f"  {f'{hn}人 {geo}':<22} 占全部超时的 {cellsum[(hn,geo)]/G*100:5.1f}%")
print(f"  {'20 人两格合计':<22} 占全部超时的 {(cellsum[(20,'circle_crossing')]+cellsum[(20,'square_crossing')])/G*100:5.1f}%")
print(f"  {'5 人两格合计':<22} 占全部超时的 {(cellsum[(5,'circle_crossing')]+cellsum[(5,'square_crossing')])/G*100:5.1f}%")
print()
print(f"  各臂总超时数 (768 ep/臂): " + '  '.join(f"{a}={tot[a]}" for a in ARMS))

print()
print('='*100)
print('第9步  超时 vs 碰撞 的 return / 净距 对照 (三臂合并, 判断超时的物理性质)')
print('='*100)
cat={'success':[], 'collision':[], 'timeout':[]}
for a in ARMS:
    for r in RUNS:
        for e in load(r,a):
            k='timeout' if isto(e) else e['outcome']
            cat[k].append(e)
print(f"{'结局':<10}{'n':>6}{'中位return':>13}{'中位净距':>11}{'中位用时':>11}{'净距<0.1占比':>13}")
print('-'*100)
for k in ('success','collision','timeout'):
    v=cat[k]
    low=sum(1 for e in v if e['minimum_clearance']<0.1)/len(v)*100
    print(f"{k:<10}{len(v):>6}{st.median(e['return'] for e in v):>13.4f}"
          f"{st.median(e['minimum_clearance'] for e in v):>11.3f}"
          f"{st.median(e['time'] for e in v):>10.2f}s{low:>12.1f}%")
print()
print('  判读依据: 若超时是"绕远路", 进度奖励会让 return 明显为正; 若是"原地卡住/来回震荡",')
print('  时间惩罚累积而进度≈0 -> return 为负。')
