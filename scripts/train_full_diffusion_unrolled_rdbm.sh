#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

acceleration="${FULL_UNROLLED_ACCELERATION:-4}"
case "$acceleration" in
  2|4|6|8) ;;
  *) echo "Unsupported acceleration: $acceleration" >&2; exit 1 ;;
esac

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${FULL_UNROLLED_OUTPUT_ROOT:-$data_root/derived}"
output_dir="${FULL_UNROLLED_OUTPUT_DIR:-$output_root/full_diffusion_unrolled_rdbm_${acceleration}x_scratch}"
epochs="${FULL_UNROLLED_EPOCHS:-50}"
workers="${FULL_UNROLLED_WORKERS:-4}"
mkdir -p "$output_dir"

latest_checkpoint="$(
  find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-epoch-*-step-*.pt' -printf '%p\n' \
    | sort | tail -n 1
)"
resume_args=()
if [[ -n "$latest_checkpoint" ]]; then
  echo "Resuming: $latest_checkpoint"
  resume_args=(--resume "$latest_checkpoint")
fi

python -u -m mri_diffusion_unrolled_rdbm.cli.train \
  --data-root "$data_root" \
  --output-dir "$output_dir" \
  --acceleration "$acceleration" \
  --epochs "$epochs" \
  --save-every-epochs 5 \
  --cg-iterations 6 \
  --learning-rate 2e-5 \
  --precision bf16 \
  --num-workers "$workers" \
  --disable-gradient-checkpointing \
  "${resume_args[@]}" \
  2>&1 | tee -a "$output_dir/train.log"
