"""Fixed noisy validation inputs: graph and temporal-refinement output contributions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_school_spike import seed
from src.tools.inference import move_batch_to_device

ROOT=Path('outputs_amass/formal_qk_g2p_340_499_v1')

def main():
    torch.set_num_threads(4);seed(9029)
    cfg=OmegaConf.load(ROOT/'config.json');cfg.data.preload=False
    data=instantiate(cfg.data,split='val')
    model=instantiate(cfg.diffusion).cuda().eval()
    ck=torch.load(ROOT/'logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state_dict'],strict=True);del ck
    model._trainer=SimpleNamespace(current_epoch=499,max_epochs=500)
    d=model.denoiser
    print(json.dumps(dict(event='settings',refine_gate=float(d.refine_gate.sigmoid()),coord_gate=float(d.coord_gate.sigmoid()),part_decoder=d.use_part_specific_decoder,normalization=d.cross_attention_norm)),flush=True)
    fixed=[]
    class Captured(Exception):pass
    def capture(m,args,kwargs):
        capture.t=kwargs.get('t',args[2] if len(args)>2 else None)
        fixed.append((kwargs.get('x',args[0] if args else None),kwargs.get('y',args[1] if len(args)>1 else None)))
        raise Captured()
    for start in (0,64,128,192):
        batch=move_batch_to_device(data.collate_fn([data[i] for i in range(start,start+4)]),'cuda')
        for timestep in (0,25,75):
            # Three reproducible random-noise probes, retaining their sampled timesteps.
            h=d.register_forward_pre_hook(capture,with_kwargs=True)
            seed(9029+start+timestep)
            try:
                with torch.no_grad():model.diffusion_step(copy.deepcopy(batch),0,training=False)
            except Captured:pass
            h.remove()
            # Captured noisy x is held fixed across interventions. Its original noise level is retained below.
            x,y=fixed.pop()
            for cap,refine,label in ((.5,True,'full'),(.5,False,'no_refine'),(0.,True,'no_graph'),(0.,False,'neither')):
                d.graph_residual_gain_cap=cap;d.use_temporal_refine=refine
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    out=d(x,y,capture.t)[...,:d.motion_nfeats].float()
                mask=y['mask'].bool(); pair=mask[:,1:]&mask[:,:-1]
                vel=out[:,1:]-out[:,:-1]
                stats={}
                for idx,name in enumerate(d.motion_part_names):
                    ids=getattr(d,f'motion_part_index_{idx}')
                    vals=out[...,ids];dv=vel[...,ids]
                    stats[name]={'rms':float(vals[mask].square().mean().sqrt()),'frame_delta_rms':float(dv[pair].square().mean().sqrt())}
                print(json.dumps(dict(event='path',start=start,probe_seed=9029+start+timestep,timesteps=capture.t.tolist(),arm=label,features='normalized model motion features, not world joints',parts=stats)),flush=True)
    print('OUTPUT_PATH_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
