"""Read-only fixed-input audit of padding, attention weights and fusion scales."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_graphlossfix import ROOT
from scripts.diagnose_school_spike import seed
from scripts.cross_attention_diagnostics import CrossAttentionDiagnostics
from src.tools.inference import move_batch_to_device


def emit(**row):
    print(json.dumps(row), flush=True)


def main():
    torch.set_num_threads(4)
    cfg = OmegaConf.load(ROOT / 'config.json')
    model = instantiate(cfg.diffusion).cuda().eval()
    model._trainer = SimpleNamespace(current_epoch=330, max_epochs=3000)
    model.relation_affinity_loss_weight = model.relation_smoothness_loss_weight = 0.0
    folder = Path('logs/controlled_reset_adam_299_v1')
    cpu = torch.load(folder/'graph_bypass_anomaly_batch.pt', map_location='cpu', weights_only=False)
    # Capture the actual denoiser inputs once, so all checkpoints and padding
    # variants see identical valid motion, text, noise and diffusion timesteps.
    fixed = []
    def capture(module, args, kwargs):
        x = kwargs.get('x', args[0] if args else None)
        y = kwargs.get('y', args[1] if len(args)>1 else None)
        t = kwargs.get('t', args[2] if len(args)>2 else None)
        lengths = y['mask'].sum(1)
        order = lengths.argsort()
        for index in [int(order[0]), int(order[len(order)//2]), int(order[-1])]:
            length = int(lengths[index])
            yi = {'mask': torch.ones(1,length,device=x.device,dtype=torch.bool), 'length':[length]}
            yi['tx'] = {k:(v[index:index+1].clone() if torch.is_tensor(v) else v) for k,v in y['tx'].items()}
            fixed.append((x[index:index+1,:length].clone(), yi, t[index:index+1].clone()))
        # Stop before the unnecessary full-batch forward; no model state changes.
        raise Captured()
    class Captured(Exception):
        pass
    handle = model.denoiser.register_forward_pre_hook(capture, with_kwargs=True)
    seed(3300000+83)
    try:
        with torch.no_grad():
            model.diffusion_step(move_batch_to_device(copy.deepcopy(cpu),'cuda'),83,training=False)
    except Captured:
        pass
    handle.remove()
    del cpu
    cases = [('healthy299',ROOT/'logs/checkpoints/recovery-epoch=299.ckpt','none',0.0),
             ('failed330',folder/'graph_bypass_anomaly.ckpt','none',0.0),
             ('qk399_bypass',Path('logs/controlled_qknorm_fullgraph_299_399_v1/graph_on_completed.ckpt'),'qk',0.0),
             ('qk399_fullgraph',Path('logs/controlled_qknorm_fullgraph_299_399_v1/graph_on_completed.ckpt'),'qk',0.5)]
    for label,path,scope,cap in cases:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['state_dict'],strict=True)
        del ck
        d=model.denoiser
        d.graph_residual_gain_cap=cap
        norm_handles=[]
        if scope=='qk':
            def normalize(module,args,kwargs):
                for key in ('query','key'):
                    v=kwargs[key]
                    kwargs[key]=F.layer_norm(v.float(),(v.shape[-1],),eps=1e-5).to(v.dtype)
                return args,kwargs
            for name in ('part_to_global_attn','global_to_part_attn'):
                norm_handles.append(getattr(d,name).register_forward_pre_hook(normalize,with_kwargs=True))
        weights={}
        for name in ('part_to_global_attn','global_to_part_attn'):
            m=getattr(d,name)
            weights[name]={k:float(v.detach().float().norm()) for k,v in zip(('Wq_fro','Wk_fro','Wv_fro'),m.in_proj_weight.chunk(3))}
            weights[name]['Wo_fro']=float(m.out_proj.weight.detach().float().norm())
        emit(event='weights',label=label,weights=weights,coord_gate=float(d.coord_gate.sigmoid()),refine_gate=float(d.refine_gate.sigmoid()))
        diagnostic=CrossAttentionDiagnostics(d,scope)
        for sample,(x,y,t) in enumerate(fixed):
            length=x.shape[1]
            for precision in ('fp32','bf16'):
                for intervention in ('original','no_refine','masked_refine'):
                    base=None
                    for padding in (0,8,32):
                        xp=F.pad(x,(0,0,0,padding))
                        yp=dict(y,mask=F.pad(y['mask'],(0,padding),value=False))
                        collected={}
                        handles=[]
                        def output_hook(name):
                            def h(module,args,out):
                                v=out[0] if isinstance(out,tuple) else out
                                v=v.detach().float()
                                if name=='global_to_part_attn':
                                    v=v[:,:length*len(d.motion_part_names)]
                                else:
                                    v=v[:,:length]
                                collected[name]=dict(rms=float(v.square().mean().sqrt()),absmax=float(v.abs().max()))
                            return h
                        for name in ('part_to_global_attn','global_to_part_attn','coord_projection','base_motion_head','refine_motion_head'):
                            handles.append(getattr(d,name).register_forward_hook(output_hook(name)))
                        if intervention=='no_refine':
                            handles.append(d.refine_motion_head.register_forward_hook(lambda m,a,o:torch.zeros_like(o)))
                        elif intervention=='masked_refine':
                            # Diagnostic intervention only: mask between every
                            # temporal convolution so biases cannot leak back.
                            def masked_tcn(module,args,out):
                                value=args[0]
                                mask=yp['mask'][...,None]
                                residual=(value*mask).transpose(1,2)
                                for layer in module.net:
                                    residual=layer(residual)*mask.transpose(1,2)
                                return module.norm(value+residual.transpose(1,2))*mask
                            handles.append(d.refine_tcn.register_forward_hook(masked_tcn))
                        diagnostic.enabled=precision=='fp32' and intervention=='original' and padding==0
                        diagnostic.records.clear()
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                            pred=d(xp,yp,t)[:,:length,:d.motion_nfeats].float()
                        if base is None:base=pred.clone()
                        delta=pred-base
                        emit(event='padding',label=label,sample=sample,length=length,timestep=int(t.item()),precision=precision,intervention=intervention,padding=padding,
                             max_abs_delta=float(delta.abs().max()),rms_delta=float(delta.square().mean().sqrt()),
                             interior_max=float(delta[:,:max(0,length-2)].abs().max()) if length>2 else None,
                             output_rms=float(pred.square().mean().sqrt()),modules=collected,
                             attention=diagnostic.records if diagnostic.enabled else None)
                        for h in handles:h.remove()
        diagnostic.close()
        for h in norm_handles:h.remove()
    emit(event='complete')


if __name__=='__main__':
    main()
