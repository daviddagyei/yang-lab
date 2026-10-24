#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${SWINMR_OUTPUT_ROOT:-$data_root/derived}"
export MRI_DATA_ROOT="$data_root"
export SWINMR_OUTPUT_ROOT="$output_root"
export SWINMR_TORCH_HOME="${SWINMR_TORCH_HOME:-$output_root/torch_cache}"
export TORCH_HOME="$SWINMR_TORCH_HOME"
mkdir -p "$output_root" "$TORCH_HOME"

gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( gpu_count < 2 )); then
  echo "Two visible CUDA GPUs are required; found $gpu_count." >&2
  exit 1
fi

# Download/cache the fixed VGG-19 weights once before the two jobs start, which
# avoids simultaneous writes to the same cache file on a fresh workstation.
python -c 'from torchvision.models import VGG19_Weights, vgg19; vgg19(weights=VGG19_Weights.IMAGENET1K_V1); print("VGG-19 weights ready")'

CUDA_VISIBLE_DEVICES=0 SWINMR_ACCELERATIONS="${SWINMR_GPU0_RATES:-2 6}" \
  bash scripts/train_swinmr_acceleration_loop.sh &
pid_gpu0=$!

CUDA_VISIBLE_DEVICES=1 SWINMR_ACCELERATIONS="${SWINMR_GPU1_RATES:-4 8}" \
  bash scripts/train_swinmr_acceleration_loop.sh &
pid_gpu1=$!

trap 'kill "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true' INT TERM
status=0
wait "$pid_gpu0" || status=$?
wait "$pid_gpu1" || status=$?
exit "$status"
