import json, math
from pathlib import Path

W = Path('/home/abc/temp/cc_6env_20261007')
BAYES_ANALYSIS = Path('/home/abc/workspace/CrowdNav(20270731_backup2)/CrowdNav/crowd_nav/runs/bayes-matched-v1/comprehensive-analysis.json')

def ev(p):
    d = json.load(open(p)); eps = d['episodes']; n = len(eps)
    s = sum(1 for e in eps if e['outcome'] == 'success')
    c = sum(1 for e in eps if e['outcome'] == 'collision')
    return dict(n=n, sr=s/n*100, cr=c/n*100, to=(n-s-c)/n*100,
                lat=d.get('policy_latency_ms_median'), wall=d.get('wall_seconds'),
                sharded=d.get('sharded'))

def line(cols, w):
    return ''.join(str(c).ljust(x) if i == 0 else str(c).rjust(x) for i, (c, x) in enumerate(zip(cols, w)))

print('=' * 104)
print('表 1  KDA 线 — 6环境评测 (6 cell = {5,10,20}人 x {circle,square}, 32 case/cell, 192 ep)')
print('=' * 104)
W1 = [10, 10, 16, 9, 9, 9, 12, 10]
print(line(['臂', '预算', '代码/case组', 'SR', 'CR', 'TO', '中位延迟', '墙钟'], W1))
print('-' * 104)
rows = []
for arm in ('kda', 'mamba', 'gru'):
    for budget, tag, d in ((3000, '优化前 / 88000', 'kda_3000'),
                           (10000, '优化前 / 88000', 'kda_premerge_88000'),
                           (10000, '优化前 / 89000', 'kda_premerge_89000'),
                           (10000, '优化后 / 89000', 'kda')):
        f = W / d / f'{arm}.json'
        if not f.exists(): continue
        r = ev(f)
        lat = f"{r['lat']:.2f}ms" + ('*' if r['sharded'] else '')
        wall = f"{r['wall']:.0f}s" + ('*' if r['sharded'] else '')
        print(line([arm, budget, tag, f"{r['sr']:.2f}%", f"{r['cr']:.2f}%",
                    f"{r['to']:.2f}%", lat, wall], W1))
        rows.append((arm, budget, tag, r))
    print('-' * 104)
print('  * = 本次用 6 进程分片测的, 延迟/墙钟受进程互抢影响, 不可与串行数对比; SR/CR/TO 不受影响(已验证分片等价)')

print()
print('=' * 104)
print('表 2  KDA 线 — 优化前 vs 优化后, 同一组 case (89000-89031) 严格配对')
print('=' * 104)
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
W2 = [10, 11, 11, 11, 9, 9, 11, 16]
print(line(['臂', 'SR 优化前', 'SR 优化后', 'ΔSR', 'b(前胜)', 'c(后胜)', 'McNemar p', '判读'], W2))
print('-' * 104)
for arm in ('kda', 'mamba', 'gru'):
    A = {(e['human_num'], e['geometry'], e['case']): e['outcome']
         for e in json.load(open(W / 'kda_premerge_89000' / f'{arm}.json'))['episodes']}
    B = {(e['human_num'], e['geometry'], e['case']): e['outcome']
         for e in json.load(open(W / 'kda' / f'{arm}.json'))['episodes']}
    keys = sorted(set(A) & set(B))
    b = sum(1 for k in keys if A[k] == 'success' and B[k] != 'success')
    c = sum(1 for k in keys if A[k] != 'success' and B[k] == 'success')
    p = mcnemar(b, c)
    sa = sum(1 for k in keys if A[k] == 'success') / len(keys) * 100
    sb = sum(1 for k in keys if B[k] == 'success') / len(keys) * 100
    print(line([arm, f"{sa:.2f}%", f"{sb:.2f}%", f"{sb-sa:+.2f}pp", b, c, f"{p:.4f}",
                '显著' if p < 0.05 else '无显著差异'], W2))
print('-' * 104)
print('  192 ep 协议的检出下限约 7.29pp (净不一致 >=14 个); 3 次比较 Bonferroni 阈值 p<0.0167')

print()
print('=' * 104)
print('表 3  决定性隔离检验 — 同一份权重, 优化前实现 vs 优化后实现')
print('=' * 104)
W3 = [24, 13, 13, 15, 16, 18]
print(line(['对照', 'SR 旧实现', 'SR 新实现', 'outcome不一致', 'return最大差', 'clearance最大差'], W3))
print('-' * 104)
for name, pl, pn in (('贝叶斯线 formal-gru', 'iso/bayes-formal-gru-legacy.json', 'bayes/formal-gru.json'),
                     ('KDA线 gru', 'iso/kda-gru-legacy.json', 'kda/gru.json')):
    L = {(e['human_num'], e['geometry'], e['case']): e for e in json.load(open(W/pl))['episodes']}
    N = {(e['human_num'], e['geometry'], e['case']): e for e in json.load(open(W/pn))['episodes']}
    keys = sorted(set(L) & set(N))
    nd = sum(1 for k in keys if L[k]['outcome'] != N[k]['outcome'])
    mr = max(abs(L[k]['return'] - N[k]['return']) for k in keys)
    mc = max(abs(L[k]['minimum_clearance'] - N[k]['minimum_clearance']) for k in keys)
    sl = sum(1 for k in keys if L[k]['outcome'] == 'success') / len(keys) * 100
    sn = sum(1 for k in keys if N[k]['outcome'] == 'success') / len(keys) * 100
    print(line([name, f"{sl:.2f}%", f"{sn:.2f}%", f"{nd} / {len(keys)}",
                f"{mr:.3e}", f"{mc:.3e}"], W3))

print()
print('=' * 104)
print('表 4  训练期 FINAL-STATS (10000 ep 全程滚动) + 训练吞吐')
print('=' * 104)
TRAIN = {  # 臂: (优化前 SR/CR/TO/s每ep,  优化后 SR/CR/TO/s每ep)
 'kda':   ((0.979, 0.018, 0.003, 2.240), (0.980, 0.018, 0.002, 1.631)),
 'mamba': ((0.980, 0.018, 0.002, 1.790), (0.979, 0.018, 0.003, 1.272)),
 'gru':   ((0.983, 0.013, 0.004, 1.937), (0.983, 0.013, 0.005, 1.325)),
}
W4 = [10, 10, 10, 10, 10, 10, 10, 12, 10]
print(line(['臂', 'SR 前', 'SR 后', 'CR 前', 'CR 后', 'TO 前', 'TO 后', 's/ep 前→后', '训练加速'], W4))
print('-' * 104)
for arm, (a, b) in TRAIN.items():
    print(line([arm, f"{a[0]*100:.1f}%", f"{b[0]*100:.1f}%", f"{a[1]*100:.1f}%", f"{b[1]*100:.1f}%",
                f"{a[2]*100:.1f}%", f"{b[2]*100:.1f}%", f"{a[3]:.2f}→{b[3]:.2f}",
                f"{a[3]/b[3]:.2f}x"], W4))
print('-' * 104)
print('  优化前那次是从 ep3001 续跑(7000 个 RL ep, 3 臂共享 GPU); 优化后是从 ep1 干净跑(10000 个, 6 臂共享 GPU)')

print()
print('=' * 104)
print('表 5  贝叶斯线 — 注意: 之前用的是另一套 6 环境协议 (6 命名环境 x 500 ep = 3000 ep)')
print('=' * 104)
W5 = [16, 11, 10, 10, 10, 10, 20]
print(line(['臂', '预算', 'n', 'SR', 'CR', 'TO', '协议'], W5))
print('-' * 104)
if BAYES_ANALYSIS.exists():
    d = json.load(open(BAYES_ANALYSIS))
    for budget in ('3000', '8000'):
        for arm in ('bayes', 'bayes_mean', 'gru'):
            a = d['cohorts'][budget][arm]['aggregate']
            n = a['n']
            print(line([arm, f"优化前 {budget}", n, f"{a['success']/n*100:.2f}%",
                        f"{a['collision']/n*100:.2f}%", f"{a['timeout']/n*100:.2f}%",
                        '6环境x500'], W5))
        print('-' * 104)
f = W / 'bayes' / 'formal-gru.json'
if f.exists():
    r = ev(f)
    print(line(['gru (formal-gru)', '优化后 10000', r['n'], f"{r['sr']:.2f}%",
                f"{r['cr']:.2f}%", f"{r['to']:.2f}%", '6cell x32case'], W5))
for arm in ('formal-bayes_mean', 'formal-bayes'):
    f = W / 'bayes' / f'{arm}.json'
    if f.exists():
        r = ev(f)
        print(line([arm, '优化后 10000', r['n'], f"{r['sr']:.2f}%",
                    f"{r['cr']:.2f}%", f"{r['to']:.2f}%", '6cell x32case'], W5))
    else:
        print(line([arm, '优化后 10000', '—', '训练中', '—', '—', '待测'], W5))
print('-' * 104)
print('  贝叶斯线没有"优化前的 10000-ep"结果 — 之前最高预算是 8000, 且协议不同, 跨行不可直接相减')
