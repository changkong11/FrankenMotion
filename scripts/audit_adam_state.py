"""Check saved Adam tensor integrity on CPU without modifying checkpoints."""
import json
from pathlib import Path
import torch
from scripts.diagnose_graphlossfix import ROOT

torch.set_num_threads(4)
paths = [ROOT / 'logs/checkpoints/recovery-epoch=299.ckpt',
         Path('logs/controlled_bypass_299_v1/graph_bypass_anomaly.ckpt')]
for path in paths:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    opt = ck['optimizer_states'][0]
    invalid, negative, steps = [], [], []
    for key, state in opt['state'].items():
        steps.append(float(state['step']))
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                invalid.append([key, name])
        if (state['exp_avg_sq'] < 0).any():
            negative.append(key)
    print(json.dumps(dict(path=str(path), epoch=ck['epoch'], invalid=invalid,
                          negative_variances=negative, step_range=[min(steps), max(steps)],
                          groups=[{k:v for k,v in g.items() if k != 'params'} for g in opt['param_groups']])), flush=True)
