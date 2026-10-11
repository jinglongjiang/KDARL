import json, glob, re
rows = {}
for f in glob.glob('eval/*.json'):
    m = re.match(r'(\w+?)_(mean|expect)_b(.+?)_(\d+)\.json', f.split('/')[-1])
    d = json.load(open(f))
    if len(d.get('episodes', [])) != 200: continue
    s = d['summary'][0]
    rows[(m.group(1), m.group(2), float(m.group(3)), int(m.group(4)))] = (100*s['sr'], 100*s['cr'], 100*s['tr'])
print('=== SARL + 信念威胁质量 @ FOV=180(circle,200回合/格)===')
for n in (20, 10):
    bare = 1.8 if n == 20 else 11.9
    print(f'--- {n} 人(SARL 裸机 @180deg = {bare}%)---')
    print(f"{'模式':>9s} {'威胁':>7s} |   b=0.1 SR/CR/TO |   b=0.3 SR/CR/TO | 最优")
    for mode in ('visible', 'memory', 'ttl', 'decay', 'rfs_occ'):
        for th in ('mean', 'expect'):
            c = [rows.get((mode, th, b, n)) for b in (0.1, 0.3)]
            if any(x is None for x in c): continue
            body = ' | '.join(f'{x[0]:6.1f}/{x[1]:5.1f}/{x[2]:5.1f}' for x in c)
            print(f'{mode:>9s} {th:>7s} | {body} | {max(x[0] for x in c):5.1f}')
    print()
