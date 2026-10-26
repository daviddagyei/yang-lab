#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

rate="${1:-${CASCADE_CNN_ACCELERATION:-}}"
case "$rate" in
  2|4|6|8) ;;
  *) echo "Usage: $0 {2|4|6|8}" >&2; exit 2 ;;
esac

output_dir="${CASCADE_CNN_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/multicoil_cascade_cnn_${rate}x}"
mkdir -p "$output_dir"

resume_args=()
latest_checkpoint="$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-epoch-*-step-*.pt' -printf '%p\n' | sort | tail -n 1)"
if [[ -n "$latest_checkpoint" ]]; then
  resume_args=(--resume "$latest_checkpoint")
fi

python -u -m mri_cascade_cnn.cli.train \
  --output-dir "$output_dir" \
  --acceleration "$rate" \
  --epochs 50 \
  --batch-size 10 \
  --learning-rate 0.001 \
  --save-every-epochs 1 \
  "${resume_args[@]}" \
  2>&1 | tee -a "$output_dir/train.log"
