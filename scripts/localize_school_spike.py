"""Module-weight substitution diagnostics; no training or checkpoint writes."""
import copy
import json
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.diagnose_school_spike import ROOT, seed
from src.tools.inference import move_batch_to_device

torch.set_num_threads(4)
seed(9025)
cfg = OmegaConf.load(ROOT / 'config.json')
data = instantiate(cfg.data, split='train')
samples = [data[i] for i in range(64)]
batches = [data.collate_fn(samples[i:i+16]) for i in range(0,64,16)]
model = instantiate(cfg.diffusion).cuda().eval()
good_checkpoint = torch.load(ROOT / 'logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
assert good_checkpoint['epoch'] == 65
good = good_checkpoint['state_dict']
bad = torch.load(ROOT / 'logs/checkpoints/diverged-step=13309.ckpt',map_location='cpu',weights_only=False)['state_dict']
groups = {
    'graph': ('dynamic_graph_blocks.',),
    'global_encoder': ('global_temporal_encoder.',),
    'local_encoder': ('local_temporal_encoder.', 'part_interaction_encoder.'),
    'fusion': ('part_to_global_attn.', 'global_to_part_attn.', 'coord_projection.', 'coord_gate'),
    'heads': ('base_motion_head.', 'text_head.', 'refine_tcn.', 'refine_motion_head.', 'refine_gate'),
    'embeddings': ('tx_embedding.', 'motion_global_embedding.', 'action_embedding.', 'part_text_embedding.', 'motion_part_embeddings.', 'part_id_embedding.', 'timestep_encoder.'),
}
for label in ['good','bad',*groups]:
    state = dict(good if label=='good' else bad)
    if label in groups:
        for key in state:
            if any(key.startswith('denoiser.'+prefix) for prefix in groups[label]):
                state[key]=good[key]
    model.load_state_dict(state)
    losses=[]
    for i, cpu in enumerate(batches):
        seed(1200+i)
        batch=move_batch_to_device(copy.deepcopy(cpu),'cuda')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model.diffusion_step(batch,i,training=False)
        losses.append(float(loss.get('diff loss',loss['loss'])))
    print(json.dumps(dict(replaced=label,mean=sum(losses)/len(losses),losses=losses)),flush=True)
