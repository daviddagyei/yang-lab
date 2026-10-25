#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${FULL_UNROLLED_OUTPUT_ROOT:-$data_root/derived}"
gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( gpu_count < 2 )); then
  echo "Two visible CUDA GPUs are required; found $gpu_count." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 MRI_DATA_ROOT="$data_root" FULL_UNROLLED_OUTPUT_ROOT="$output_root" \
  bash scripts/evaluate_full_unrolled_early_exit.sh 2 &
pid2=$!
CUDA_VISIBLE_DEVICES=1 MRI_DATA_ROOT="$data_root" FULL_UNROLLED_OUTPUT_ROOT="$output_root" \
  bash scripts/evaluate_full_unrolled_early_exit.sh 4 &
pid4=$!
trap 'kill "$pid2" "$pid4" 2>/dev/null || true' INT TERM
status=0
wait "$pid2" || status=$?
wait "$pid4" || status=$?
exit "$status"
