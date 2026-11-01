#!/usr/bin/env bash
# Sequentially run formal external-baseline stages when an A100 has enough spare capacity.
# A generator is complete only after its .complete marker exists; best.pt remains resumable.
set -Eeuo pipefail
BASE=/data_disk/wulw/baselines
PY=/data_disk/wulw/envs/fm-baselines/bin/python
QUEUE_DIR=$BASE/runs/formal_3000epoch_v3_queue
MIN_FREE_MIB=${MIN_FREE_MIB:-30000}
MAX_UTIL_PCT=${MAX_UTIL_PCT:-10}
POLL_SECONDS=${POLL_SECONDS:-120}
mkdir -p "$QUEUE_DIR"
exec > >(tee -a "$QUEUE_DIR/queue.log") 2>&1
printf '%s queue_online min_free_mib=%s max_util_pct=%s\n' "$(date -Is)" "$MIN_FREE_MIB" "$MAX_UTIL_PCT"

choose_ready_gpu() {
  while true; do
    candidate=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | \
      awk -F, -v min_free="$MIN_FREE_MIB" -v max_util="$MAX_UTIL_PCT" \
      '$2+0 >= min_free && $3+0 <= max_util {gsub(/ /,"",$1); print $1; exit}')
    if [ -n "${candidate:-}" ]; then printf '%s\n' "$candidate"; return; fi
    printf '%s waiting_for_capacity min_free_mib=%s max_util_pct=%s\n' "$(date -Is)" "$MIN_FREE_MIB" "$MAX_UTIL_PCT" >&2
    sleep "$POLL_SECONDS"
  done
}

run_stage() {
  local method="$1" stage="$2" config="$3" completion_marker="$4" codec="${5:-}" resume_ckpt="${6:-}"
  if [ -f "$completion_marker" ]; then
    printf '%s skip_completed method=%s stage=%s marker=%s\n' "$(date -Is)" "$method" "$stage" "$completion_marker"
    return
  fi
  local gpu; gpu=$(choose_ready_gpu)
  printf '%s start method=%s stage=%s physical_gpu=%s\n' "$(date -Is)" "$method" "$stage" "$gpu"
  local args=("$PY" "$BASE/source/$method/frankenstein.py" train --stage "$stage" --device cuda:0 --config "$config")
  if [ -n "$codec" ]; then args+=(--codec "$codec"); fi
  if [ -n "$resume_ckpt" ] && [ -s "$resume_ckpt" ]; then args+=(--resume "$resume_ckpt"); fi
  CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}" 2>&1 | tee -a "$QUEUE_DIR/${method}_${stage}.log"
  touch "$completion_marker"
  printf '%s complete method=%s stage=%s\n' "$(date -Is)" "$method" "$stage"
}

MOSCALE_CFG=$BASE/configs/moscale_3000epoch_server.json
FLOOD_CFG=$BASE/configs/flooddiffusion_3000epoch_server.json
MOSCALE_OUT=$BASE/runs/moscale_frankenstein_v3_total3000_seed1234
FLOOD_OUT=$BASE/runs/flooddiffusion_frankenstein_v3_total3000_seed1234
run_stage MoScale codec "$MOSCALE_CFG" "$MOSCALE_OUT/codec/best.pt"
run_stage MoScale generator "$MOSCALE_CFG" "$MOSCALE_OUT/generator/.complete" "$MOSCALE_OUT/codec/best.pt" "$MOSCALE_OUT/generator/last.pt"
run_stage FloodDiffusion codec "$FLOOD_CFG" "$FLOOD_OUT/codec/best.pt"
run_stage FloodDiffusion generator "$FLOOD_CFG" "$FLOOD_OUT/generator/.complete" "$FLOOD_OUT/codec/best.pt" "$FLOOD_OUT/generator/last.pt"
printf '%s all_formal_stages_complete\n' "$(date -Is)"