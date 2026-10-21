"""Read-only checkpoint comparisons and isolated, unsaved optimizer probes."""
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.tools.inference import move_batch_to_device


ROOT = Path('outputs_amass/frankenmotion_clip_part_smplrifke_frankenstein-dataset_lgfm_scratch_bf16_b64_3k')


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def emit(**values):
    print(json.dumps(values), flush=True)


def main():
    torch.set_num_threads(4)
    seed(9025)
    cfg = OmegaConf.load(ROOT / 'config.json')
    dataset = instantiate(cfg.data, split='train')
    indices = np.random.default_rng(9025).choice(len(dataset), 256, replace=False)
    samples = [dataset[int(i)] for i in indices]
    small = [dataset.collate_fn(samples[i:i+16]) for i in range(0, 128, 16)]
    large = [dataset.collate_fn(samples[i:i+64]) for i in range(0, 256, 64)]
    model = instantiate(cfg.diffusion).cuda()
    model._trainer = SimpleNamespace(current_epoch=66, max_epochs=3000)
    results = []
    for name in ['recovery-epoch=49.ckpt', 'latest-epoch=65.ckpt', 'diverged-step=13309.ckpt']:
        checkpoint = torch.load(ROOT / 'logs/checkpoints' / name, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        for precision in ['fp32', 'bf16']:
            losses = []
            for i, cpu_batch in enumerate(small):
                batch = move_batch_to_device(copy.deepcopy(cpu_batch), 'cuda')
                seed(1200+i)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                    values = model.diffusion_step(batch, i, training=False)
                losses.append(float(values.get('diff loss', values['loss'])))
                del batch, values
            row = dict(stage='forward', checkpoint=name, precision=precision, losses=losses, mean=float(np.mean(losses)))
            emit(**row)
            results.append(row)
        del checkpoint
    # Same initial weights, optimizer moments, batches, noise and dropout for each arm.
    checkpoint = torch.load(ROOT / 'logs/checkpoints/latest-epoch=65.ckpt', map_location='cpu', weights_only=False)
    for scale in [1.0, 0.25]:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.train()
        optimizer = model.configure_optimizers()['optimizer']
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_states'][0]))
        for group in optimizer.param_groups:
            group['lr'] *= scale
        trajectory = []
        for i in range(60):
            seed(4000+i)
            batch = move_batch_to_device(copy.deepcopy(large[i % len(large)]), 'cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values = model.diffusion_step(batch, i, training=True)
            loss = values['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite diagnostic loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            value = float(values.get('diff loss', loss).detach())
            trajectory.append(value)
            emit(stage='update', lr_scale=scale, step=i, loss=value, grad_norm=float(norm))
            if value > 0.23 or float(norm) > 100:
                emit(stage='probe_stop', reason='loss_or_gradient_limit', lr_scale=scale, step=i)
                break
            optimizer.step()
            del batch, values, loss
        results.append(dict(stage='updates', lr_scale=scale, losses=trajectory))
        del optimizer
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    output = Path('logs/spike_diagnostic_20260925.json')
    output.write_text(json.dumps(results, indent=2))
    emit(stage='done', output=str(output))


if __name__ == '__main__':
    main()
