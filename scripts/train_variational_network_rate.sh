#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

rate="${1:-}"
case "$rate" in
  2|4|6|8) ;;
  *) echo "Usage: $0 {2|4|6|8}" >&2; exit 2 ;;
esac

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${VN_OUTPUT_ROOT:-$data_root/derived}"
output_dir="${VN_OUTPUT_DIR:-$output_root/hammernik_variational_network_${rate}x_ipalm}"
iterations="${VN_ITERATIONS:-9700}"
workers="${VN_WORKERS:-4}"
batch_size="${VN_BATCH_SIZE:-10}"
mkdir -p "$output_dir"

latest_checkpoint="$(
  find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-step-*.pt' -printf '%p\n' \
    | sort \
    | tail -n 1
)"
resume_args=()
if [[ -n "$latest_checkpoint" ]]; then
  echo "Resuming ${rate}x: $latest_checkpoint"
  resume_args=(--resume "$latest_checkpoint")
fi

python -u -m mri_variational_network.cli.train \
  --data-root "$data_root" \
  --acceleration "$rate" \
  --output-dir "$output_dir" \
  --optimizer ipalm \
  --iterations "$iterations" \
  --save-every-steps 194 \
  --batch-size "$batch_size" \
  --num-workers "$workers" \
  --disable-gradient-checkpointing \
  "${resume_args[@]}" \
  2>&1 | tee -a "$output_dir/train.log"

