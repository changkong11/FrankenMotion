#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN=/data_disk/wulw/envs/frankenmotion/bin/python
export GPU_ID=0
export BATCH_SIZE=64
export NUM_WORKERS=4
export MAX_EPOCHS=3000
export TRAINER_PRECISION=bf16-mixed
export INIT_CKPT=null
export RUN_SUFFIX=_lgfm_scratch_bf16_b64_3k
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$ROOT_DIR"
RUN_DIR="outputs_amass/frankenmotion_clip_part_smplrifke_frankenstein-dataset${RUN_SUFFIX}"
if [[ -e "$RUN_DIR/config.json" ]]; then
  echo "Run already exists: $RUN_DIR. Refusing to overwrite."
  exit 1
fi
exec bash scripts/train_coord_graph_gated_init_3k.sh
