#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

output_dir="${VN_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/hammernik_variational_network_4x_ipalm}"
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
  echo "Resuming: $latest_checkpoint"
  resume_args=(--resume "$latest_checkpoint")
fi

python -u -m mri_variational_network.cli.train \
  --output-dir "$output_dir" \
  --optimizer ipalm \
  --iterations "$iterations" \
  --save-every-steps 194 \
  --batch-size "$batch_size" \
  --num-workers "$workers" \
  --disable-gradient-checkpointing \
  "${resume_args[@]}" \
  2>&1 | tee -a "$output_dir/train.log"
