#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

output_dir="${MODL_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/modl_4x_fp32}"
slices_per_epoch=1898
warmup_steps=$((5 * slices_per_epoch))
target_steps=$((50 * slices_per_epoch))
mkdir -p "$output_dir"

latest_checkpoint() {
  find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-*.pt' -printf '%p\n' \
    | sort \
    | tail -n 1
}

checkpoint="$(latest_checkpoint)"
completed_steps=0
if [[ -n "$checkpoint" ]]; then
  checkpoint_name="$(basename "$checkpoint" .pt)"
  completed_steps="${checkpoint_name#checkpoint-}"
  completed_steps=$((10#$completed_steps))
fi

if (( completed_steps < warmup_steps )); then
  warmup_remaining=$((warmup_steps - completed_steps))
  warmup_resume=()
  if [[ -n "$checkpoint" ]]; then
    warmup_resume=(--resume "$checkpoint")
  fi
  python -u -m mri_modl.cli.train \
    --output-dir "$output_dir" \
    --steps "$warmup_remaining" \
    --save-every "$slices_per_epoch" \
    --unrolls 1 \
    --cg-iterations 10 \
    "${warmup_resume[@]}" \
    2>&1 | tee -a "$output_dir/train_k1.log"
  checkpoint="$(latest_checkpoint)"
  completed_steps=$warmup_steps
fi

if (( completed_steps < target_steps )); then
  remaining_steps=$((target_steps - completed_steps))
  python -u -m mri_modl.cli.train \
    --output-dir "$output_dir" \
    --steps "$remaining_steps" \
    --save-every "$slices_per_epoch" \
    --unrolls 10 \
    --cg-iterations 10 \
    --resume "$checkpoint" \
    --allow-unroll-change \
    2>&1 | tee -a "$output_dir/train_k10.log"
else
  echo "MoDL training is already complete at step $completed_steps."
fi
