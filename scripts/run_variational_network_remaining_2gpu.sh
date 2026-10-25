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

# Each rate is an independent model. GPU 0 runs 2x then 8x; GPU 1 runs 6x.
# With three similarly sized jobs, this is the minimum two-wave schedule.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_variational_network_rate.sh 2 &
pid_gpu0=$!
CUDA_VISIBLE_DEVICES=1 bash scripts/run_variational_network_rate.sh 6 &
pid_gpu1=$!
trap 'kill "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true' INT TERM

status=0
wait "$pid_gpu0" || status=$?
if (( status == 0 )); then
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_variational_network_rate.sh 8 || status=$?
fi
wait "$pid_gpu1" || status=$?
exit "$status"
