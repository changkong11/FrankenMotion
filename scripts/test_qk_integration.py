"""Check native attention normalization against the experimental hooks."""
import copy
import json
import shutil
import tempfile
from pathlib import Path
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_graphlossfix import ROOT
from scripts.diagnose_school_spike import seed
from src.tools.inference import load_diffusion


def main():
    torch.set_num_threads(4)
    cfg=OmegaConf.load(ROOT/'config.json')
    a=instantiate(cfg.diffusion).cuda()
    ck=torch.load(ROOT/'logs/checkpoints/recovery-epoch=299.ckpt',map_location='cpu',weights_only=False)
    a.load_state_dict(ck['state_dict'],strict=True)
    b=instantiate(cfg.diffusion).cuda()
    b.load_state_dict(ck['state_dict'],strict=True)
    del ck
    seed(190)
    x=torch.randn(2,7,a.denoiser.nfeats,device='cuda')
    mask=torch.tensor([[True]*7,[True]*4+[False]*3],device='cuda')
    y={'mask':mask,'length':[7,4],'tx':{'x':torch.randn(2,3,512,device='cuda'),'mask':torch.ones(2,3,dtype=torch.bool,device='cuda')}}
    t=torch.tensor([10,75],device='cuda')
    for mode in ('none','qk_g2p','qk_p2g','qk_both','qkv_both'):
        handles=[]
        names=[] if mode=='none' else (['global_to_part_attn'] if mode=='qk_g2p' else ['part_to_global_attn'] if mode=='qk_p2g' else ['global_to_part_attn','part_to_global_attn'])
        def hook(module,args,kwargs):
            for key in (('query','key','value') if mode=='qkv_both' else ('query','key')):
                value=kwargs[key]
                kwargs[key]=torch.nn.functional.layer_norm(value.float(),(value.shape[-1],),eps=1e-5).to(value.dtype)
            return args,kwargs
        for name in names:handles.append(getattr(a.denoiser,name).register_forward_pre_hook(hook,with_kwargs=True))
        b.denoiser.cross_attention_norm=mode
        for training in (False,True):
            a.train(training);b.train(training)
            for precision in ('fp32','bf16'):
                outputs=[];grads=[]
                for model in (a,b):
                    model.zero_grad(set_to_none=True);seed(777)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                        out=model.denoiser(x,y,t)
                        loss=out.float().square().mean()
                    loss.backward()
                    outputs.append(out.detach());grads.append(model.denoiser.global_to_part_attn.in_proj_weight.grad.clone())
                torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
                torch.testing.assert_close(grads[0],grads[1],rtol=0,atol=0)
                print(json.dumps(dict(event='hook_equivalence',mode=mode,training=training,precision=precision,exact=True)),flush=True)
        for h in handles:h.remove()
    a.eval();a.denoiser.cross_attention_norm='qk_g2p'
    with tempfile.TemporaryDirectory(prefix='qk_load_') as temp:
        p=Path(temp)
        for name in ('motion_stats','text_stats'):shutil.copytree(ROOT/name,p/name)
        cfg.diffusion.denoiser.cross_attention_norm='qk_g2p'
        (p/'config.json').write_text(json.dumps(OmegaConf.to_container(cfg,resolve=True)))
        torch.save({'state_dict':a.state_dict()},p/'model.ckpt')
        loaded=load_diffusion(OmegaConf.load(p/'config.json'),str(p),'model','cuda')
        assert loaded.denoiser.cross_attention_norm=='qk_g2p'
        with torch.no_grad():torch.testing.assert_close(a.denoiser(x,y,t),loaded.denoiser(x,y,t),rtol=0,atol=0)
        print('CONFIG_CHECKPOINT_INFERENCE_ROUNDTRIP_PASS',flush=True)
    print('QK_INTEGRATION_PASS',flush=True)


if __name__=='__main__':main()
