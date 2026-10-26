#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

export ADADIFF_FIR_BACKEND="${ADADIFF_FIR_BACKEND:-triton}"
export CUDA_VISIBLE_DEVICES="${ADADIFF_BENCHMARK_GPU:-0}"

python -u -m mri_adadiff.benchmark_training \
  --batch-size "${ADADIFF_BENCHMARK_BATCH_SIZE:-4}" \
  --steps "${ADADIFF_BENCHMARK_STEPS:-11}" \
  --lazy-regularization "${ADADIFF_LAZY_REGULARIZATION:-10}" \
  --train-slices 1898 \
  --epochs 500
