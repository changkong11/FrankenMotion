#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHON_BIN=/data_disk/wulw/envs/frankenmotion/bin/python
export GPU_ID=0 BATCH_SIZE=64 NUM_WORKERS=4 MAX_EPOCHS=3000
export TRAINER_PRECISION=bf16-mixed INIT_CKPT=null
export BASE_LR=0.00001675 GRAPH_LR_SCALE=0.10
export LR_WARMUP_EPOCHS=100 LR_MIN_SCALE=1.0
export RUN_SUFFIX=_lgfm_scratch_bf16_b64_graphlossfix_lr1675_3k
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ -e "outputs_amass/frankenmotion_clip_part_smplrifke_frankenstein-dataset${RUN_SUFFIX}/config.json" ]]; then
  echo 'Run already exists; refusing to overwrite.'
  exit 1
fi
exec bash scripts/train_coord_graph_gated_init_3k.sh
