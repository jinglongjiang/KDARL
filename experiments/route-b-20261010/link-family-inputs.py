"""Verify declared Bayesian family checkpoints without inventing per-file links."""
import json
from pathlib import Path
from common import BASE, WORK, save, sha

links=[]
for name in ('bayes-decomp-20261009','bayes-incr-20261010','bayes-horizon-20261010'):
    path=WORK/name/'frozen-manifest.json'
    manifest=json.loads(path.read_text())
    source=manifest.get('inputs',manifest)
    for filename,item in source.items():
        if not isinstance(item,dict) or not filename.endswith(('.pth','.pth.gz')):continue
        p=Path(filename);expected=item.get('sha256')
        assert expected and p.is_file() and sha(p)==expected,(name,filename)
        links.append(dict(scope=name,path=filename,sha256=expected,manifest=str(path),
                          status='已验证通过',link_type='declared family input, not inferred per-file use'))
assert links
p=BASE/'p2-provenance.json'
data=json.loads(p.read_text())
data['verified_family_checkpoint_inputs']=links
save(p,data)
md=BASE/'p2-provenance.md'
text=md.read_text()
marker='## 家族级权重来源（不代替逐文件绑定）'
assert marker not in text
table=['',marker,'','以下从冻结manifest声明并复验实际文件；不能据此把每个中间文件都当作该权重生成。','',
       '| 研究目录 | 权重路径 | SHA256 | 声明来源 | 状态 |','|---|---|---|---|---|']
for row in links:
    table.append('| %s | %s | %s | %s | %s |'%(row['scope'],row['path'],row['sha256'],row['manifest'],row['status']))
md.write_text('\n'.join(table)+'\n\n'+text)
print(json.dumps(dict(status='PASS',verified_family_weights=len(links))))
