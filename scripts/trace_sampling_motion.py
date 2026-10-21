"""Track predicted clean motion (not noisy latent) through actual sampling."""
import json
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_school_spike import seed
from src.tools.inference import move_batch_to_device

ROOT=Path('outputs_amass/formal_qk_g2p_340_499_v1')
OUT=Path('logs/sampling_motion_trace_v1')

def main():
    torch.set_num_threads(4);seed(9140);OUT.mkdir(exist_ok=False)
    cfg=OmegaConf.load(ROOT/'config.json');cfg.data.preload=False
    data=instantiate(cfg.data,split='val')
    ids=np.random.default_rng(9140).choice(len(data),32,replace=False).tolist()
    model=instantiate(cfg.diffusion).cuda().eval()
    ck=torch.load(ROOT/'logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state_dict'],strict=True);del ck
    assert model.prediction=='x' and not model.uni
    steps={99,90,75,50,25,10,5,0}
    with (OUT/'trace.jsonl').open('x') as log:
        def metrics(z,n):
            result={}
            for p,name in enumerate(model.denoiser.motion_part_names):
                ix=getattr(model.denoiser,f'motion_part_index_{p}')
                v=z[:n,ix].float()
                result[name]={'amplitude':float((v-v.mean(0)).square().mean().sqrt()),'delta_rms':float((v[1:]-v[:-1]).square().mean().sqrt())}
            return result
        def record(output,t,mask,keys,s):
            for i,key in enumerate(keys):
                log.write(json.dumps(dict(seed=s,key=key,t=t,parts=metrics(output[i],int(mask[i].sum()))))+'\n')
            log.flush()
        for start in range(0,32,4):
            batch=move_batch_to_device(data.collate_fn([data[i] for i in ids[start:start+4]]),'cuda')
            keys=[str(k) for k in batch['keyid']];mask=batch['mask']
            with torch.no_grad():gt=model.motion_normalizer(batch['x'])
            record(gt,-1,mask,keys,-1)
            for s in (9029,9030):
                def hook(m,args,out):
                    t=int(args[2][0])
                    if t in steps:record(out.detach(),t,mask,keys,s)
                h=model.denoiser.register_forward_hook(hook)
                seed(s+start*100)
                lengths=batch['length'].tolist()
                infos=dict(all_lengths=lengths,output_lengths=lengths,split='val',featsname=cfg.motion_features,guidance_weight=1.0)
                with torch.no_grad():model.batch_forward(batch,infos,progress_bar=None)
                h.remove()
            print(json.dumps(dict(completed=start+4,total=32)),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(epoch=499,indices=ids,seeds=[9029,9030],space='normalized clean-prediction motion features',note='Independent diagnostic subset; seed stream is not identical to prior full-val generation.')))
    print('SAMPLING_TRACE_COMPLETE',flush=True)

if __name__=='__main__':main()
