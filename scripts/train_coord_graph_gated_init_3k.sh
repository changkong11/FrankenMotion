#!/usr/bin/env bash
set -euo pipefail

# Initialize the stable LG-FM trunk only; graph parameters and optimizer state
# start fresh. Required: INIT_CKPT=/absolute/or/project-relative/checkpoint.ckpt
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
INIT_CKPT="${INIT_CKPT:?Set INIT_CKPT to the clean LG-FM checkpoint}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-3000}"
RUN_SUFFIX="${RUN_SUFFIX:-_coord_graph_gated_init_3k}"
REUSE_STATS="${REUSE_STATS:-true}"
TRAINER_PRECISION="${TRAINER_PRECISION:-bf16-mixed}"
GRAPH_LR_SCALE="${GRAPH_LR_SCALE:-0.10}"
BASE_LR="${BASE_LR:-0.0001}"
LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-100}"
LR_MIN_SCALE="${LR_MIN_SCALE:-0.10}"
GRAPH_RESIDUAL_GAIN_CAP="${GRAPH_RESIDUAL_GAIN_CAP:-0.50}"
RELATION_AFFINITY_LOSS_WEIGHT="${RELATION_AFFINITY_LOSS_WEIGHT:-0.02}"
RELATION_SMOOTHNESS_LOSS_WEIGHT="${RELATION_SMOOTHNESS_LOSS_WEIGHT:-0.002}"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$PYTHON_BIN" train_part.py \
  dataset=frankenstein-dataset \
  data.preload=false \
  trainer.max_epochs="$MAX_EPOCHS" \
  +trainer.precision="$TRAINER_PRECISION" \
  +trainer.gradient_clip_val=1.0 \
  +trainer.gradient_clip_algorithm=norm \
  dataloader.batch_size="$BATCH_SIZE" \
  dataloader.num_workers="$NUM_WORKERS" \
  run_suffix="$RUN_SUFFIX" \
  reuse_stats="$REUSE_STATS" \
  init_ckpt="$INIT_CKPT" \
  diffusion.lr="$BASE_LR" \
  diffusion.denoiser.use_dynamic_kinematic_graph=true \
  diffusion.denoiser.use_contact_balance_state=false \
  diffusion.denoiser.use_part_specific_decoder=false \
  diffusion.denoiser.graph_residual_gain_cap="$GRAPH_RESIDUAL_GAIN_CAP" \
  diffusion.relation_affinity_loss_weight="$RELATION_AFFINITY_LOSS_WEIGHT" \
  diffusion.relation_smoothness_loss_weight="$RELATION_SMOOTHNESS_LOSS_WEIGHT" \
  diffusion.contact_state_loss_weight=0.0 \
  diffusion.balance_state_loss_weight=0.0 \
  diffusion.coordination_loss_warmup_epochs=1000 \
  diffusion.coordination_loss_max_timestep_ratio=0.25 \
  diffusion.state_loss_warmup_epochs=0 \
  diffusion.graph_lr_scale="$GRAPH_LR_SCALE" \
  diffusion.lr_warmup_epochs="$LR_WARMUP_EPOCHS" \
  diffusion.lr_min_scale="$LR_MIN_SCALE" \
  diffusion.divergence_guard_ratio=8.0 \
  diffusion.divergence_guard_min_observations=100 \
  diffusion.divergence_guard_max_finite_spikes=3 \
  diffusion.divergence_guard_window_batches=200 \
  diffusion.divergence_guard_skip_finite_spikes=true \
  diffusion.divergence_guard_component_ratio=16.0 \
  diffusion.divergence_guard_max_grad_norm=100.0
