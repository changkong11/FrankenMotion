"""Matched-batch directional intervention and valid-token gradient audit."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_graphlossfix import ROOT
from scripts.diagnose_school_spike import seed
from scripts.cross_attention_diagnostics import CrossAttentionDiagnostics
from src.tools.inference import move_batch_to_device

def main():
    torch.set_num_threads(4)
    folder = Path('logs/controlled_reset_adam_299_v1')
    cpu = torch.load(folder/'graph_bypass_anomaly_batch.pt', map_location='cpu', weights_only=False)
    cfg = OmegaConf.load(ROOT/'config.json')
    cfg.diffusion.denoiser.cross_attention_norm = 'none'
    model = instantiate(cfg.diffusion).cuda().train()
    model._trainer = SimpleNamespace(current_epoch=330, max_epochs=3000)
    model.relation_affinity_loss_weight = model.relation_smoothness_loss_weight = 0.0
    model.denoiser.graph_residual_gain_cap = 0.0
    names = ('part_to_global_attn', 'global_to_part_attn')
    cases = [('original',None,1.0),('p2g_half',names[0],0.5),('g2p_half',names[1],0.5),('p2g_zero',names[0],0.0),('g2p_zero',names[1],0.0)]
    for label,path in [('healthy299',ROOT/'logs/checkpoints/recovery-epoch=299.ckpt'),('failed330',folder/'graph_bypass_anomaly.ckpt')]:
        ck = torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['state_dict'],strict=True)
        del ck
        for case,target,gain in cases:
            records = {}
            diagnostic = CrossAttentionDiagnostics(model.denoiser,'none')
            diagnostic.enabled = True
            handles=[]
            def capture(name):
                def hook(module,args,kwargs,output):
                    q=kwargs['query']; out=output[0]
                    mask=diagnostic.mask
                    parts=q.shape[1]//mask.shape[1] if name==names[1] else 1
                    valid=mask[:,:,None].expand(-1,-1,parts).reshape(q.shape[:2])
                    def rms(x): return float(x.detach().float()[valid].square().mean().sqrt())
                    rec={'input_rms':rms(q),'delta_rms':rms(out),'sum_rms':rms(q+out)}
                    if parts>1:
                        shaped=out.detach().float().reshape(mask.shape[0],mask.shape[1],parts,-1)
                        rec['per_part_delta_rms']=[float(shaped[:,:,p][mask].square().mean().sqrt()) for p in range(parts)]
                    records[name]=rec
                    out.register_hook(lambda grad: rec.update(output_gradient_rms=rms(grad)))
                    return (out*(gain if name==target else 1.0),output[1])
                return hook
            for name in names:
                handles.append(getattr(model.denoiser,name).register_forward_hook(capture(name),with_kwargs=True))
            model.zero_grad(set_to_none=True)
            seed(330*10000+83)
            batch=move_batch_to_device(copy.deepcopy(cpu),'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                values=model.diffusion_step(batch,83,training=True)
                loss=values['loss']
            loss.backward()
            print(json.dumps(dict(checkpoint=label,case=case,loss=float(loss.detach()),directions=records,attention=diagnostic.records)),flush=True)
            diagnostic.close()
            for h in handles: h.remove()
            del batch,values,loss

if __name__=='__main__': main()
