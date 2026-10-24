#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

acceleration="${SWINMR_ACCELERATION:-4}"
case "$acceleration" in
  2|4|6|8) ;;
  *) echo "Unsupported acceleration: $acceleration" >&2; exit 1 ;;
esac

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${SWINMR_OUTPUT_ROOT:-$data_root/derived}"
output_dir="${SWINMR_OUTPUT_DIR:-$output_root/swinmr_pi_complex_${acceleration}x}"
export TORCH_HOME="${SWINMR_TORCH_HOME:-$output_root/torch_cache}"
mkdir -p "$output_dir" "$TORCH_HOME"
data_args=(--data-root "$data_root")
resume_args=()
latest_checkpoint="$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-epoch-*-step-*.pt' -printf '%p\n' | sort | tail -n 1)"
if [[ -n "$latest_checkpoint" ]]; then
  resume_args=(--resume "$latest_checkpoint")
fi

python -u -m mri_swinmr.cli.train \
  "${data_args[@]}" \
  --output-dir "$output_dir" \
  --acceleration "$acceleration" \
  --epochs 50 \
  --batch-size 8 \
  --crop-size 96 \
  --learning-rate 0.0002 \
  --ema-decay 0.999 \
  --save-every-epochs 5 \
  "${resume_args[@]}" \
  2>&1 | tee -a "$output_dir/train.log"
