"""Fixed saved anomaly batch, matched noise, valid-token attention audit."""
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
    cpu = torch.load(folder / 'graph_bypass_anomaly_batch.pt', map_location='cpu', weights_only=False)
    cfg = OmegaConf.load(ROOT / 'config.json')
    model = instantiate(cfg.diffusion).cuda().train()
    model._trainer = SimpleNamespace(current_epoch=330, max_epochs=3000)
    model.relation_affinity_loss_weight = model.relation_smoothness_loss_weight = 0.0
    model.denoiser.graph_residual_gain_cap = 0.0
    cases = [
        ('epoch299_original', ROOT / 'logs/checkpoints/recovery-epoch=299.ckpt', 'none'),
        ('failed_original', folder / 'graph_bypass_anomaly.ckpt', 'none'),
        ('failed_qk_intervention', folder / 'graph_bypass_anomaly.ckpt', 'qk'),
        ('qk_epoch339', Path('logs/controlled_qknorm_299_v1/graph_bypass_completed.ckpt'), 'qk'),
        ('qkv_epoch339', Path('logs/controlled_crossnorm_299_v1/graph_bypass_completed.ckpt'), 'qkv'),
    ]
    for label, path, scope in cases:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state_dict'], strict=True)
        del ck
        diagnostic = CrossAttentionDiagnostics(model.denoiser, scope)
        diagnostic.enabled = True
        handles = []
        if scope != 'none':
            def normalize(module, args, kwargs):
                for key in (('query', 'key') if scope == 'qk' else ('query', 'key', 'value')):
                    value = kwargs[key]
                    kwargs[key] = torch.nn.functional.layer_norm(value.float(), (value.shape[-1],), eps=1e-5).to(value.dtype)
                return args, kwargs
            for name in ('part_to_global_attn', 'global_to_part_attn'):
                handles.append(getattr(model.denoiser, name).register_forward_pre_hook(normalize, with_kwargs=True))
        seed(330 * 10000 + 83)
        batch = move_batch_to_device(copy.deepcopy(cpu), 'cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            values = model.diffusion_step(batch, 83, training=True)
        print(json.dumps(dict(label=label, loss=float(values.get('diff loss', values['loss'])), modules=diagnostic.records)), flush=True)
        diagnostic.close()
        for handle in handles:
            handle.remove()
        del values, batch, diagnostic


if __name__ == '__main__':
    main()
