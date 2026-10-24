#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

target_steps="${RDBM_TARGET_STEPS:-94900}"
save_every="${RDBM_SAVE_EVERY:-9490}"
workers="${RDBM_WORKERS:-2}"
precision="${RDBM_PRECISION:-bf16}"
output_root="${RDBM_OUTPUT_ROOT:-/data/david_agyei/nyu-dataset/derived}"
accelerations="${RDBM_ACCELERATIONS:-2 6 8}"

for acceleration in $accelerations; do
  case "$acceleration" in
    2|4|6|8) ;;
    *) echo "Unsupported acceleration: $acceleration" >&2; exit 1 ;;
  esac

  output_dir="$output_root/rdbm_${acceleration}x_${precision}"
  mkdir -p "$output_dir"
  latest_checkpoint="$(
    find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-*.pt' -printf '%p\n' \
      | sort \
      | tail -n 1
  )"

  current_step=0
  if [[ -n "$latest_checkpoint" ]]; then
    checkpoint_name="$(basename "$latest_checkpoint" .pt)"
    current_step="${checkpoint_name#checkpoint-}"
    current_step=$((10#$current_step))
  fi
  remaining_steps=$((target_steps - current_step))
  if (( remaining_steps <= 0 )); then
    echo "RDBM ${acceleration}x is complete at step $current_step; skipping."
    continue
  fi

  echo "Starting RDBM ${acceleration}x at step $current_step; target=$target_steps."
  RDBM_ACCELERATION="$acceleration" \
  RDBM_OUTPUT_DIR="$output_dir" \
  RDBM_RESUME="$latest_checkpoint" \
  RDBM_STEPS="$remaining_steps" \
  RDBM_SAVE_EVERY="$save_every" \
  RDBM_WORKERS="$workers" \
  RDBM_PRECISION="$precision" \
  bash scripts/train_rdbm_full.sh
done

echo "RDBM acceleration loop complete: $accelerations"
