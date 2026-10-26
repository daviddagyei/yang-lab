#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate
export ADADIFF_FIR_BACKEND="${ADADIFF_FIR_BACKEND:-triton}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${ADADIFF_OUTPUT_ROOT:-$data_root/derived}"
probe_dir="${ADADIFF_PROBE_DIR:-$output_root/adadiff_2gpu_probe}"
mkdir -p "$probe_dir"

torchrun --standalone --nproc-per-node=2 -m mri_adadiff.cli.train \
  --data-root "$data_root" \
  --output-dir "$probe_dir" \
  --epochs 500 \
  --batch-size "${ADADIFF_GLOBAL_BATCH_SIZE:-16}" \
  --num-workers 0 \
  --save-every-epochs 25 \
  --log-every 1 \
  --lazy-regularization "${ADADIFF_LAZY_REGULARIZATION:-10}" \
  --benchmark-warmup-steps 2 \
  --max-steps "${ADADIFF_BENCHMARK_TOTAL_STEPS:-13}" \
  --skip-max-step-checkpoint

