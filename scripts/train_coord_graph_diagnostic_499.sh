#!/usr/bin/env bash
set -euo pipefail

# Test the learned relation router from a known-good coordination checkpoint.
# High-noise steps use only fixed kinematic routing; learned edges are blended
# in as the diffusion state becomes reliable. Contact/balance losses and
# state-to-token feedback remain disabled for this isolation run.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESUME_DIR="${RESUME_DIR:?Set RESUME_DIR to the original coordination run directory}"
CKPT="${CKPT:-$RESUME_DIR/logs/checkpoints/resume_epoch_499.ckpt}"

export MAX_EPOCHS="${MAX_EPOCHS:-900}"
export CKPT
export RESUME_DIR
export CONTACT_STATE_LOSS_WEIGHT=0.0
export BALANCE_STATE_LOSS_WEIGHT=0.0
export STATE_FEEDBACK_GAIN_CAP="${STATE_FEEDBACK_GAIN_CAP:-0.0}"
export DYNAMIC_RELATION_GAIN_CAP="${DYNAMIC_RELATION_GAIN_CAP:-1.0}"
export DYNAMIC_RELATION_STATIC_START_TIMESTEP_RATIO="${DYNAMIC_RELATION_STATIC_START_TIMESTEP_RATIO:-0.75}"
export DYNAMIC_RELATION_TRANSITION_EPOCHS="${DYNAMIC_RELATION_TRANSITION_EPOCHS:-20}"
export COORDINATION_MAX_TIMESTEP_RATIO="${COORDINATION_MAX_TIMESTEP_RATIO:-0.25}"
export COMPONENT_GUARD_RATIO="${COMPONENT_GUARD_RATIO:-16.0}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-100.0}"

exec "$ROOT_DIR/scripts/train_coord_graph_state_3k.sh"
