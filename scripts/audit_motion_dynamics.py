"""Paired raw-joint motion amplitude and temporal detail diagnostics."""
import json
from pathlib import Path
import numpy as np
from scripts.audit_generated_jerk import BASE,NEW,load

GROUPS={'all':list(range(22)),'wrists':[20,21],'feet':[7,8,10,11],'torso':[3,6,9]}

def measure(x):
    result={}
    for space,y in [('world',x),('root_relative',x-x[:,0:1])]:
        for group,ids in GROUPS.items():
            z=y[:,ids]
            amp=np.mean((z-z.mean(0))**2)**0.5
            v=np.diff(z,axis=0)*20
            a=np.diff(v,axis=0)*20
            j=np.diff(a,axis=0)*20
            # Velocity spectrum removes constant position offsets; window reduces boundary leakage.
            spectrum=np.fft.rfft((v-v.mean(0))*np.hanning(len(v))[:,None,None],axis=0)
            energy=np.abs(spectrum)**2
            freq=np.fft.rfftfreq(len(v),d=1/20)
            high=energy[freq>=3].sum()
            result[space+'/'+group]={'amplitude':float(amp),'speed':float(np.mean(v*v)**0.5),
                'acceleration':float(np.mean(a*a)**0.5),'jerk':float(np.mean(j*j)**0.5),
                'velocity_high_frequency_fraction':float(high/max(energy.sum(),1e-30))}
    return result

def main():
    rows=[]
    for p in sorted((NEW/'gt').glob('*.npy')):
        paths=[BASE/'turn_0/data'/p.name,NEW/'turn_0/data'/p.name]
        if not all(q.exists() for q in paths):continue
        ann=json.loads(p.with_suffix('.json').read_text())
        xs=[load(p),*[load(q) for q in paths]]
        n=min([len(x) for x in xs]+[ann.get('sum_length',len(xs[0]))])
        if n<8:continue
        rows.append({'key':p.stem,**{label:measure(x[:n]) for label,x in zip(('gt','old','new'),xs)}})
    summary={label:{group:{metric:float(np.mean([r[label][group][metric] for r in rows])) for metric in rows[0][label][group]} for group in rows[0][label]} for label in ('gt','old','new')}
    # Fixed reference-defined fast-motion subset avoids selecting on model outputs.
    cutoff=float(np.quantile([r['gt']['root_relative/wrists']['speed'] for r in rows],.75))
    fast=[r for r in rows if r['gt']['root_relative/wrists']['speed']>=cutoff]
    fast_summary={label:{metric:float(np.mean([r[label]['root_relative/wrists'][metric] for r in fast])) for metric in fast[0][label]['root_relative/wrists']} for label in ('gt','old','new')}
    out={'n':len(rows),'summary':summary,'fast_wrist_n':len(fast),'fast_wrist_summary':fast_summary,'samples':rows}
    Path('logs/motion_dynamics_audit.json').write_text(json.dumps(out))
    print(json.dumps({k:v for k,v in out.items() if k!='samples'}),flush=True)

if __name__=='__main__':main()
