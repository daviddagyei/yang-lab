#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${SWINMR_OUTPUT_ROOT:-$data_root/derived}"
export MRI_DATA_ROOT="$data_root"
export SWINMR_OUTPUT_ROOT="$output_root"

gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( gpu_count < 2 )); then
  echo "Two visible CUDA GPUs are required; found $gpu_count." >&2
  exit 1
fi

evaluate_rates() {
  local gpu="$1"
  shift
  local rate
  for rate in "$@"; do
    echo "[$(date --iso-8601=seconds)] GPU $gpu: validating SwinMR ${rate}x"
    CUDA_VISIBLE_DEVICES="$gpu" SWINMR_ACCELERATION="$rate" \
      bash scripts/validate_swinmr_checkpoints.sh
    echo "[$(date --iso-8601=seconds)] GPU $gpu: testing SwinMR ${rate}x"
    CUDA_VISIBLE_DEVICES="$gpu" SWINMR_ACCELERATION="$rate" \
      bash scripts/evaluate_best_swinmr_test.sh
  done
}

evaluate_rates 0 2 6 &
pid_gpu0=$!
evaluate_rates 1 4 8 &
pid_gpu1=$!
trap 'kill "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true' INT TERM
status=0
wait "$pid_gpu0" || status=$?
wait "$pid_gpu1" || status=$?
exit "$status"
