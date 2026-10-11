"""Verify existing paired result assets, without running policies or environments."""
import json
from pathlib import Path
from common import BASE, WORK, save, sha


def load_cells(folder):
    episodes={}; hashes=set(); files=[]; command_episodes=0
    for cell in range(6):
        path=folder/('cell%d.json'%cell)
        data=json.loads(path.read_text())
        digest=data.get('checkpoint_sha256'); assert digest and len(digest)==64
        hashes.add(digest)
        assert len(data['episodes'])==500
        for r in data['episodes']:
            key=(cell,r['case']); assert key not in episodes
            episodes[key]=r['outcome']
            if r.get('commands'):command_episodes+=1
        files.append(dict(path=str(path),sha256=sha(path),backbone=data.get('backbone'),
            selection=data.get('selection'),alpha=data.get('smoothing'),
            candidates=data.get('candidate_actions',data.get('grid_actions')),
            fields=list(data)))
    assert len(episodes)==3000 and len(hashes)==1
    return episodes,hashes,files,command_episodes


def contrast(name,left,right):
    a,ha,fa,ca=load_cells(left);b,hb,fb,cb=load_cells(right)
    assert ha==hb and a.keys()==b.keys(), name
    differences={}
    for target in ('success','collision','timeout'):
        lose=sum(a[k]==target and b[k]!=target for k in a)
        gain=sum(a[k]!=target and b[k]==target for k in a)
        differences[target]=dict(left_only=lose,right_only=gain,delta_pp=(gain-lose)/30.)
    return dict(name=name,status='已验证通过',evidence_class='A',
        scope='same checkpoint hash and exact (cell,case) pairing verified; not all runtime/source equivalence',
        checkpoint_sha256=next(iter(ha)),episodes_per_configuration=3000,
        case_range=[min(k[1] for k in a),max(k[1] for k in a)],
        deltas=differences,command_episodes=[ca,cb],files=[fa,fb])


def main():
    rows=[]
    sel=WORK/'kda-selection-20261008'
    for phase in ('development','confirmation'):
        for arm in ('kda','gru'):
            for rule in (('nearest5','ttc5') if phase=='development' else ('ttc5',)):
                rows.append(contrast('TTC-selection/%s/%s/%s'%(phase,arm,rule),
                    sel/phase/(arm+'-first5'),sel/phase/(arm+'-'+rule)))
    m=WORK/'mamba-ttc5-20261009/results'
    rows.append(contrast('Mamba-TTC5',m/'first5',m/'ttc5'))
    h=WORK/'kda-lookahead-20261009'
    for arm in ('kda','gru','mamba'):
        rows.append(contrast('H1/'+arm,
            h/'h1/per-case'/(arm+'-alpha0.3-a80'),h/'h1/per-case'/(arm+'-alpha0.0-a80')))
    for arm in ('kda','mamba'):
        for alpha in ('0.0','0.3'):
            rows.append(contrast('H2/%s/alpha%s'%(arm,alpha),
                h/'h1/per-case'/(arm+'-alpha'+alpha+'-a80'),
                h/'h2/per-case'/(arm+'-alpha'+alpha+'-a81')))
    save(BASE/'paired-asset-verification.json',dict(status='已验证通过',contrasts=rows,
        distinction='These are A contrasts; comparing separately trained backbones remains B',
        trajectory_note='H1/H2 JSON episodes include executed command sequences; old optimized evaluation JSON does not. No Bayesian CSV substitution.'))
    print(json.dumps(dict(status='PASS',paired_contrasts=len(rows),episodes_per_configuration=3000)))


if __name__=='__main__':main()
