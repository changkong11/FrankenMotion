#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/data_disk/wulw/envs/frankenmotion/bin/python}"
export GPU_ID="${GPU_ID:-0}"
export BATCH_SIZE=64
export TRAINER_PRECISION=32-true
export NUM_WORKERS="${NUM_WORKERS:-8}"
export MAX_EPOCHS="${MAX_EPOCHS:-1}"
export RUN_SUFFIX="${RUN_SUFFIX:-_batch64_fp32_probe}"
export INIT_CKPT="${INIT_CKPT:-pretrained/lgfm_3k.ckpt}"
exec bash "$ROOT_DIR/scripts/train_coord_graph_gated_init_3k.sh"
