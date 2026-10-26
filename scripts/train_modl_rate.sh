#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {2|4|6|8}" >&2
  exit 2
fi

rate="$1"
case "$rate" in
  2|4|6|8) ;;
  *) echo "Acceleration must be one of: 2, 4, 6, 8" >&2; exit 2 ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${MODL_OUTPUT_ROOT:-$data_root/derived}"
output_dir="$output_root/modl_${rate}x_fp32"
slices_per_pass="${MODL_SLICES_PER_PASS:-1898}"
warmup_passes="${MODL_WARMUP_PASSES:-5}"
total_passes="${MODL_TOTAL_PASSES:-50}"
warmup_steps=$((warmup_passes * slices_per_pass))
target_steps=$((total_passes * slices_per_pass))

if (( target_steps < warmup_steps )); then
  echo "MODL_TOTAL_PASSES must be at least MODL_WARMUP_PASSES" >&2
  exit 2
fi
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

echo "MoDL ${rate}x: ${completed_steps}/${target_steps} updates already complete."

if (( completed_steps < warmup_steps )); then
  warmup_remaining=$((warmup_steps - completed_steps))
  warmup_resume=()
  if [[ -n "$checkpoint" ]]; then
    warmup_resume=(--resume "$checkpoint")
  fi
  python -u -m mri_modl.cli.train \
    --data-root "$data_root" \
    --acceleration "$rate" \
    --output-dir "$output_dir" \
    --steps "$warmup_remaining" \
    --save-every "$slices_per_pass" \
    --unrolls 1 \
    --cg-iterations 10 \
    "${warmup_resume[@]}" \
    2>&1 | tee -a "$output_dir/train_k1.log"
  checkpoint="$(latest_checkpoint)"
  completed_steps=$warmup_steps
fi

if (( completed_steps < target_steps )); then
  if [[ -z "$checkpoint" ]]; then
    echo "No warm-up checkpoint found in $output_dir" >&2
    exit 1
  fi
  remaining_steps=$((target_steps - completed_steps))
  python -u -m mri_modl.cli.train \
    --data-root "$data_root" \
    --acceleration "$rate" \
    --output-dir "$output_dir" \
    --steps "$remaining_steps" \
    --save-every "$slices_per_pass" \
    --unrolls 10 \
    --cg-iterations 10 \
    --resume "$checkpoint" \
    --allow-unroll-change \
    2>&1 | tee -a "$output_dir/train_k10.log"
else
  echo "MoDL ${rate}x training is already complete at step $completed_steps."
fi
