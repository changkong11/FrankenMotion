"""Paired denoising errors on a deterministic 256-sample validation subset."""
import json
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_school_spike import seed
from src.tools.inference import move_batch_to_device

ROOT=Path('outputs_amass/formal_qk_g2p_340_499_v1')
OUT=Path('logs/part_reconstruction_339_499_v1')

def main():
    torch.set_num_threads(4);seed(9130)
    OUT.mkdir(exist_ok=False)
    cfg=OmegaConf.load(ROOT/'config.json');cfg.data.preload=False
    data=instantiate(cfg.data,split='val')
    ids=np.random.default_rng(9130).choice(len(data),min(256,len(data)),replace=False).tolist()
    batches=[data.collate_fn([data[i] for i in ids[start:start+4]]) for start in range(0,len(ids),4)]
    model=instantiate(cfg.diffusion).cuda().eval()
    assert model.prediction=='x'
    (OUT/'manifest.json').write_text(json.dumps(dict(indices=ids,seed=9130,timesteps=[10,50,90],metric_space='normalized motion features',velocity='adjacent-frame feature difference, not joint world velocity')))
    with (OUT/'samples.jsonl').open('x') as log:
        for epoch,root in [(339,Path('outputs_amass/formal_qk_g2p_299_339_v1')),(499,ROOT)]:
            ck=torch.load(root/'logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
            assert ck['epoch']==epoch
            model.load_state_dict(ck['state_dict'],strict=True);del ck
            for bi,cpu in enumerate(batches):
                batch=move_batch_to_device(cpu,'cuda');mask=batch['mask'].bool()
                with torch.no_grad():
                    x=model.motion_normalizer(batch['x'])*mask[...,None]
                    if 'stats_mask' in batch:x=x*batch['stats_mask']
                    y=dict(length=batch['length'],mask=mask,tx=model.prepare_tx_emb(batch['tx']))
                    for timestep in (10,50,90):
                        seed(913000+bi*100+timestep)
                        t=torch.full((len(x),),timestep,device=x.device,dtype=torch.long)
                        xt=model.q_sample(xstart=x,t=t,noise=torch.randn_like(x)*mask[...,None])*mask[...,None]
                        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model.denoiser(xt,y,t)
                        for j in range(len(x)):
                            n=int(mask[j].sum());parts={}
                            for p,name in enumerate(model.denoiser.motion_part_names):
                                ix=getattr(model.denoiser,f'motion_part_index_{p}')
                                target=x[j,:n,ix].float(); estimate=pred[j,:n,ix].float()
                                dv=target[1:]-target[:-1];pv=estimate[1:]-estimate[:-1]
                                parts[name]=dict(mse=float((estimate-target).square().mean()),delta_mse=float((pv-dv).square().mean()),target_speed=float(dv.square().mean().sqrt()),pred_speed=float(pv.square().mean().sqrt()))
                            log.write(json.dumps(dict(epoch=epoch,index=ids[bi*4+j],t=timestep,parts=parts))+'\n')
                        log.flush()
                if bi%16==0:print(json.dumps(dict(epoch=epoch,batch=bi,total=len(batches))),flush=True)
    print('PART_RECONSTRUCTION_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
