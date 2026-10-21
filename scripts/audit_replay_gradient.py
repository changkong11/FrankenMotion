"""Inspect task gradients on the saved failed batch without any updates."""
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
folder = Path('logs/controlled_replay_299_gpu0_v6')
ck = torch.load(folder / 'graph_off_anomaly.ckpt', map_location='cpu', weights_only=False)
cpu = torch.load(folder / 'graph_off_anomaly_batch.pt', map_location='cpu', weights_only=False)
model = instantiate(OmegaConf.load(ROOT / 'config.json').diffusion).cuda().train()
model.load_state_dict(ck['state_dict'])
epoch, index = ck['epoch'], ck['batch']
del ck
model._trainer = SimpleNamespace(current_epoch=epoch, max_epochs=3000)
model.relation_affinity_loss_weight = model.relation_smoothness_loss_weight = 0.0
for cap in (0.5, 0.0):
    model.denoiser.graph_residual_gain_cap = cap
    model.zero_grad(set_to_none=True)
    seed(epoch * 10000 + index)
    batch = move_batch_to_device(copy.deepcopy(cpu), 'cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses = model.diffusion_step(batch, index, training=True)
    loss = losses.get('diff loss', losses['loss'])
    loss.backward()
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            group = name.split('.')[1] if name.startswith('denoiser.') else name.split('.')[0]
            norms[group] = norms.get(group, 0.0) + float(p.grad.float().square().sum())
    print(json.dumps(dict(cap=cap, loss=float(loss.detach()), gradient_norms={k:v**0.5 for k,v in norms.items()})), flush=True)
    del loss, losses, batch
