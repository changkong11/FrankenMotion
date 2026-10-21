#!/usr/bin/env bash
set -euo pipefail

# Optional overrides: PYTHON_BIN, GPU_ID, BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS,
# RUN_SUFFIX, TRAINER_PRECISION, GRADIENT_CLIP_VAL, REUSE_STATS, RESUME_DIR, CKPT.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
# BF16 preserves the mixed-precision memory benefit but has FP32-like exponent
# range, preventing the fp16 overflows observed in the first DM training run.
BATCH_SIZE="${BATCH_SIZE:-48}"
TRAINER_PRECISION="${TRAINER_PRECISION:-bf16-mixed}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-3000}"
RUN_SUFFIX="${RUN_SUFFIX:-_coord_graph_state_bf16_clip_3k}"
REUSE_STATS="${REUSE_STATS:-false}"
RESUME_DIR="${RESUME_DIR:-}"
CKPT="${CKPT:-last}"
STATE_LOSS_WARMUP_EPOCHS="${STATE_LOSS_WARMUP_EPOCHS:-3000}"
STATE_LOSS_MAX_SCALE="${STATE_LOSS_MAX_SCALE:-0.25}"
STATE_FEEDBACK_GAIN_CAP="${STATE_FEEDBACK_GAIN_CAP:-0.25}"
DYNAMIC_RELATION_GAIN_CAP="${DYNAMIC_RELATION_GAIN_CAP:-1.0}"
DYNAMIC_RELATION_STATIC_START_TIMESTEP_RATIO="${DYNAMIC_RELATION_STATIC_START_TIMESTEP_RATIO:-0.75}"
COORDINATION_MAX_TIMESTEP_RATIO="${COORDINATION_MAX_TIMESTEP_RATIO:-0.25}"
COMPONENT_GUARD_RATIO="${COMPONENT_GUARD_RATIO:-16.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-100.0}"
DYNAMIC_RELATION_TRANSITION_EPOCHS="${DYNAMIC_RELATION_TRANSITION_EPOCHS:-0}"
RELATION_AFFINITY_LOSS_WEIGHT="${RELATION_AFFINITY_LOSS_WEIGHT:-0.10}"
RELATION_SMOOTHNESS_LOSS_WEIGHT="${RELATION_SMOOTHNESS_LOSS_WEIGHT:-0.01}"
CONTACT_STATE_LOSS_WEIGHT="${CONTACT_STATE_LOSS_WEIGHT:-0.01}"
BALANCE_STATE_LOSS_WEIGHT="${BALANCE_STATE_LOSS_WEIGHT:-0.01}"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESUME_ARGS=()
if [[ -n "$RESUME_DIR" ]]; then
  # Keep checkpoint filenames such as `recovery-epoch=799.ckpt` as one Hydra
  # value instead of letting the second equals sign be parsed as syntax.
  RESUME_ARGS=(resume_dir="$RESUME_DIR" "ckpt=\"$CKPT\"")
fi

exec "$PYTHON_BIN" train_part.py \
  dataset=frankenstein-dataset \
  data.preload=false \
  trainer.max_epochs="$MAX_EPOCHS" \
  +trainer.precision="$TRAINER_PRECISION" \
  +trainer.gradient_clip_val="$GRADIENT_CLIP_VAL" \
  +trainer.gradient_clip_algorithm=norm \
  dataloader.batch_size="$BATCH_SIZE" \
  dataloader.num_workers="$NUM_WORKERS" \
  run_suffix="$RUN_SUFFIX" \
  reuse_stats="$REUSE_STATS" \
  "${RESUME_ARGS[@]}" \
  diffusion.denoiser.use_dynamic_kinematic_graph=true \
  diffusion.denoiser.use_contact_balance_state=true \
  diffusion.denoiser.use_part_specific_decoder=true \
  diffusion.relation_affinity_loss_weight="$RELATION_AFFINITY_LOSS_WEIGHT" \
  diffusion.relation_smoothness_loss_weight="$RELATION_SMOOTHNESS_LOSS_WEIGHT" \
  diffusion.contact_state_loss_weight="$CONTACT_STATE_LOSS_WEIGHT" \
  diffusion.balance_state_loss_weight="$BALANCE_STATE_LOSS_WEIGHT" \
  diffusion.coordination_loss_warmup_epochs=1000 \
  diffusion.coordination_loss_max_timestep_ratio="$COORDINATION_MAX_TIMESTEP_RATIO" \
  diffusion.state_loss_warmup_epochs="$STATE_LOSS_WARMUP_EPOCHS" \
  diffusion.state_loss_max_scale="$STATE_LOSS_MAX_SCALE" \
  diffusion.denoiser.state_feedback_gain_cap="$STATE_FEEDBACK_GAIN_CAP" \
  diffusion.denoiser.dynamic_relation_gain_cap="$DYNAMIC_RELATION_GAIN_CAP" \
  diffusion.denoiser.dynamic_relation_static_start_timestep_ratio="$DYNAMIC_RELATION_STATIC_START_TIMESTEP_RATIO" \
  diffusion.dynamic_relation_transition_epochs="$DYNAMIC_RELATION_TRANSITION_EPOCHS" \
  diffusion.divergence_guard_ratio=8.0 \
  diffusion.divergence_guard_min_observations=100 \
  diffusion.divergence_guard_max_finite_spikes=3 \
  diffusion.divergence_guard_window_batches=200 \
  diffusion.divergence_guard_skip_finite_spikes=true \
  diffusion.divergence_guard_component_ratio="$COMPONENT_GUARD_RATIO" \
  diffusion.divergence_guard_max_grad_norm="$MAX_GRAD_NORM"
