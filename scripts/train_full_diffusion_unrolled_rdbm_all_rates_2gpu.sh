#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${FULL_UNROLLED_OUTPUT_ROOT:-$data_root/derived}"
export MRI_DATA_ROOT="$data_root"
export FULL_UNROLLED_OUTPUT_ROOT="$output_root"
mkdir -p "$output_root"

gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( gpu_count < 2 )); then
  echo "Two visible CUDA GPUs are required; found $gpu_count." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 FULL_UNROLLED_ACCELERATIONS="${FULL_UNROLLED_GPU0_RATES:-2 6}" \
  bash scripts/train_full_diffusion_unrolled_rdbm_loop.sh &
pid_gpu0=$!
CUDA_VISIBLE_DEVICES=1 FULL_UNROLLED_ACCELERATIONS="${FULL_UNROLLED_GPU1_RATES:-4 8}" \
  bash scripts/train_full_diffusion_unrolled_rdbm_loop.sh &
pid_gpu1=$!
trap 'kill "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true' INT TERM
status=0
wait "$pid_gpu0" || status=$?
wait "$pid_gpu1" || status=$?
exit "$status"
