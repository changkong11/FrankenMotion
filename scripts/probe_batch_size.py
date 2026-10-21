"""Run one real training forward/backward pass to verify GPU batch capacity."""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

from src.tools.inference import move_batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    overrides = [
        "dataset=frankenstein-dataset",
        "device=cpu",
        "data.preload=false",
        f"dataloader.batch_size={args.batch_size}",
        "dataloader.num_workers=0",
        "diffusion.denoiser.use_dynamic_kinematic_graph=true",
        "diffusion.denoiser.use_contact_balance_state=true",
        "diffusion.denoiser.use_part_specific_decoder=true",
        "diffusion.relation_affinity_loss_weight=0.10",
        "diffusion.relation_smoothness_loss_weight=0.01",
        "diffusion.contact_state_loss_weight=0.01",
        "diffusion.balance_state_loss_weight=0.01",
        "+diffusion.motion_normalizer.disable=true",
        "+diffusion.text_normalizer.disable=true",
    ]

    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = hydra.compose(config_name="train_part", overrides=overrides)

    # This lightweight entry point does not create Hydra's runtime metadata.
    # Resolve the three training-time interpolations that normally use it.
    cfg.motion_features = "smplrifke"
    cfg.text_features = "clip_part"
    cfg.run_dir = str(root / "outputs_amass" / "batch_size_probe")
    cfg.device = "cpu"
    cfg.data.text_encoder.device = "cpu"

    torch.manual_seed(1234)
    random.seed(1234)
    dataset = instantiate(cfg.data, split="train")
    loader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        shuffle=True,
    )
    model = instantiate(cfg.diffusion).to(args.device).train()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    iterator = iter(loader)
    total_steps = args.warmup_steps + args.timed_steps
    batches = [
        move_batch_to_device(copy.deepcopy(next(iterator)), args.device)
        for _ in range(total_steps)
    ]

    def run_step(step_idx: int):
        batch = batches[step_idx]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            step_loss = model.diffusion_step(batch, batch_idx=step_idx, training=True)["loss"]
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return step_loss

    for step_idx in range(args.warmup_steps):
        loss = run_step(step_idx)

    torch.cuda.synchronize()
    started = time.perf_counter()
    for step_idx in range(args.timed_steps):
        loss = run_step(args.warmup_steps + step_idx)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"batch_size={args.batch_size}")
    print(f"loss={loss.detach().item():.6f}")
    print(f"timed_steps={args.timed_steps}")
    print(f"seconds_per_step={elapsed / args.timed_steps:.4f}")
    print(f"peak_allocated_gb={allocated:.3f}")
    print(f"peak_reserved_gb={reserved:.3f}")


if __name__ == "__main__":
    main()
