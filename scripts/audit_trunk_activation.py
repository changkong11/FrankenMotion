"""Compare trunk activations and output gradients on one fixed failed batch."""
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

torch.set_num_threads(4)
folder = Path('logs/controlled_reset_adam_299_v1')
cpu = torch.load(folder / 'graph_bypass_anomaly_batch.pt', map_location='cpu', weights_only=False)
cfg = OmegaConf.load(ROOT / 'config.json')
model = instantiate(cfg.diffusion).cuda().train()
model._trainer = SimpleNamespace(current_epoch=330, max_epochs=3000)
model.relation_affinity_loss_weight = model.relation_smoothness_loss_weight = 0.0
model.denoiser.graph_residual_gain_cap = 0.0
records = {}
def hook(name):
    def capture(module, inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        v = value.detach().float()
        records[name] = dict(rms=float(v.square().mean().sqrt()), max=float(v.abs().max()))
        if value.requires_grad:
            def backward(grad):
                g = grad.detach().float()
                records[name]['gradient_norm'] = float(g.norm())
                records[name]['gradient_max'] = float(g.abs().max())
            value.register_hook(backward)
    return capture
names = ['timestep_encoder', 'global_temporal_encoder', 'local_temporal_encoder',
         'part_interaction_encoder', 'part_to_global_attn', 'global_to_part_attn',
         'coord_projection', 'base_motion_head']
handles = [getattr(model.denoiser, name).register_forward_hook(hook(name)) for name in names]
for label, path in [('good', ROOT / 'logs/checkpoints/recovery-epoch=299.ckpt'),
                    ('bad', folder / 'graph_bypass_anomaly.ckpt')]:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    del ck
    model.zero_grad(set_to_none=True)
    records.clear()
    seed(330 * 10000 + 83)
    batch = move_batch_to_device(copy.deepcopy(cpu), 'cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        values = model.diffusion_step(batch, 83, training=True)
    loss = values.get('diff loss', values['loss'])
    loss.backward()
    print(json.dumps(dict(label=label, loss=float(loss.detach()), modules=records)), flush=True)
    del values, loss, batch
