#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate
export ADADIFF_FIR_BACKEND="${ADADIFF_FIR_BACKEND:-triton}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${ADADIFF_OUTPUT_ROOT:-$data_root/derived}"
output_dir="${ADADIFF_OUTPUT_DIR:-$output_root/adadiff_shared_prior}"
mkdir -p "$output_dir"
resume_args=()
latest_checkpoint="$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-epoch-*-step-*.pt' -printf '%p\n' | sort | tail -n 1)"
if [[ -n "$latest_checkpoint" ]]; then
  resume_args=(--resume "$latest_checkpoint")
fi

torchrun --standalone --nproc-per-node=2 -m mri_adadiff.cli.train \
  --data-root "$data_root" \
  --output-dir "$output_dir" \
  --epochs 500 \
  --batch-size "${ADADIFF_GLOBAL_BATCH_SIZE:-16}" \
  --num-workers "${ADADIFF_NUM_WORKERS_PER_GPU:-2}" \
  --lazy-regularization "${ADADIFF_LAZY_REGULARIZATION:-10}" \
  --save-every-epochs 25 \
  "${resume_args[@]}" \
  2>&1 | tee -a "$output_dir/train.log"
