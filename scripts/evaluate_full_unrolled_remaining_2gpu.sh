#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( gpu_count < 2 )); then
  echo "Two visible CUDA GPUs are required; found $gpu_count." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_full_unrolled_rate.sh 6 &
pid6=$!
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate_full_unrolled_rate.sh 8 &
pid8=$!
trap 'kill "$pid6" "$pid8" 2>/dev/null || true' INT TERM

status=0
wait "$pid6" || status=$?
wait "$pid8" || status=$?
exit "$status"

