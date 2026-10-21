"""Valid-frame, per-layer encoder input/output audit on matched anomaly data."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_graphlossfix import ROOT
from scripts.diagnose_school_spike import seed
from src.tools.inference import move_batch_to_device

def main():
    torch.set_num_threads(4)
    folder=Path('logs/controlled_reset_adam_299_v1')
    cpu=torch.load(folder/'graph_bypass_anomaly_batch.pt',map_location='cpu',weights_only=False)
    cfg=OmegaConf.load(ROOT/'config.json')
    cfg.diffusion.denoiser.cross_attention_norm='none'
    model=instantiate(cfg.diffusion).cuda().train()
    model._trainer=SimpleNamespace(current_epoch=330,max_epochs=3000)
    model.relation_affinity_loss_weight=model.relation_smoothness_loss_weight=0.0
    model.denoiser.graph_residual_gain_cap=0.0
    masks={}; records={}; handles=[]
    def mask_hook(module,args,kwargs):
        y=kwargs.get('y',args[1] if len(args)>1 else None)
        m=y['mask'].bool(); b,t=m.shape; p=len(model.denoiser.motion_part_names)
        masks['local_temporal_encoder']=m[:,None].expand(b,p,t).reshape(b*p,t)
        masks['part_interaction_encoder']=m.reshape(b*t,1).expand(b*t,p)
        masks['global_temporal_encoder']=m
    handles.append(model.denoiser.register_forward_pre_hook(mask_hook,with_kwargs=True))
    def capture(name,group):
        def hook(module,args,kwargs,out):
            x=args[0] if args else kwargs['src']
            mask=masks[group]
            def stats(z):
                v=z.detach().float()[mask]
                return {'rms':float(v.square().mean().sqrt()),'absmax':float(v.abs().max())}
            row={'input':stats(x),'output':stats(out)}
            if group=='part_interaction_encoder':
                row['part_output_rms']=[float(out.detach().float()[:,p][mask[:,p]].square().mean().sqrt()) for p in range(out.shape[1])]
            records[name]=row
        return hook
    for group in ('local_temporal_encoder','part_interaction_encoder','global_temporal_encoder'):
        encoder=getattr(model.denoiser,group)
        for i,layer in enumerate(encoder.layers):
            handles.append(layer.register_forward_hook(capture(group+'.'+str(i),group),with_kwargs=True))
            def branch_capture(name,group):
                def hook(module,args,out):
                    z=out[0] if isinstance(out,tuple) else out
                    v=z.detach().float()[masks[group]]
                    records[name]={'output_rms':float(v.square().mean().sqrt()),'absmax':float(v.abs().max())}
                return hook
            for branch in ('self_attn','linear2'):
                handles.append(getattr(layer,branch).register_forward_hook(branch_capture(group+'.'+str(i)+'.'+branch,group)))
    cases=[('healthy299',ROOT/'logs/checkpoints/recovery-epoch=299.ckpt','none'),
           ('failed330',folder/'graph_bypass_anomaly.ckpt','none'),
           ('stable339',Path('outputs_amass/formal_qk_g2p_299_339_v1/logs/checkpoints/last.ckpt'),'qk_g2p')]
    for label,path,norm in cases:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['state_dict'],strict=True); del ck
        model.denoiser.cross_attention_norm=norm
        records.clear(); seed(3300000+83)
        batch=move_batch_to_device(copy.deepcopy(cpu),'cuda')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            values=model.diffusion_step(batch,83,training=True)
        print(json.dumps({'checkpoint':label,'graph_bypassed':True,'loss':float(values['loss']),'layers':records}),flush=True)
        del batch,values
    for h in handles: h.remove()

if __name__=='__main__': main()
