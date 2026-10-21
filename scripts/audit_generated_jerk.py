import json
from pathlib import Path
import numpy as np

BASE=Path('logs/formal_qk_validation_v1/formal_g2p339_fullgraph/generations/t2m/val_model_20260928_004735')
NEW=Path('logs/formal_qk499_validation_v1/formal_g2p499_fullgraph/generations/t2m/val_model_20260929_140930')

def load(p):
    x=np.load(p,allow_pickle=True)
    if x.shape==():x=x.item()['joints']
    return x[:,:22].astype(np.float64)

def measure(x,ann):
    j=np.diff(x,n=3,axis=0)*20**3
    per=np.mean(j*j,axis=(0,2))**0.5
    times=np.arange(len(j))+1.5
    cuts=set()
    for k,v in ann.items():
        if k.endswith('_length') and isinstance(v,list):cuts.update(np.cumsum(v)[:-1].tolist())
    near=np.zeros(len(j),dtype=bool)
    for cut in cuts:near|=abs(times-cut)<=3
    return dict(jerk=float(np.mean(j*j)**0.5),per_joint=per.tolist(),
                boundary=float(np.mean(j[near]**2)**0.5) if near.any() else None,
                interior=float(np.mean(j[~near]**2)**0.5) if (~near).any() else None)

def main():
    rows=[]
    for p in sorted((NEW/'gt').glob('*.npy')):
        paths=[BASE/'turn_0/data'/p.name,NEW/'turn_0/data'/p.name]
        if not all(q.exists() for q in paths):continue
        ann=json.loads(p.with_suffix('.json').read_text())
        xs=[load(p),*[load(q) for q in paths]]
        length=min([len(x) for x in xs]+[ann.get('sum_length',len(xs[0]))])
        if length<4:continue
        rows.append(dict(key=p.stem,**{k:measure(x[:length],ann) for k,x in zip(('gt','old','new'),xs)}))
    assert rows
    out={'n':len(rows),'space':'raw saved joints, first 22; not guofeats reconstructed metric'}
    for label in ('gt','old','new'):
        out[label]={k:float(np.mean([r[label][k] for r in rows if r[label][k] is not None])) for k in ('jerk','boundary','interior')}
        out[label]['per_joint']=np.mean([r[label]['per_joint'] for r in rows],axis=0).tolist()
    for label in ('old','new'):
        out[label]['gap']=float(np.mean([abs(r[label]['jerk']-r['gt']['jerk']) for r in rows]))
        out[label]['fraction_above_gt']=float(np.mean([r[label]['jerk']>r['gt']['jerk'] for r in rows]))
    out['fraction_new_jerk_higher']=float(np.mean([r['new']['jerk']>r['old']['jerk'] for r in rows]))
    Path('logs/generated_jerk_audit.json').write_text(json.dumps({'summary':out,'samples':rows}))
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
