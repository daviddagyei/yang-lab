#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${FULL_UNROLLED_OUTPUT_ROOT:-$data_root/derived}"
accelerations="${FULL_UNROLLED_ACCELERATIONS:-2 4 6 8}"

for acceleration in $accelerations; do
  case "$acceleration" in
    2|4|6|8) ;;
    *) echo "Unsupported acceleration: $acceleration" >&2; exit 1 ;;
  esac
  echo "Starting/resuming full diffusion-unrolled RDBM ${acceleration}x on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}."
  MRI_DATA_ROOT="$data_root" \
  FULL_UNROLLED_OUTPUT_ROOT="$output_root" \
  FULL_UNROLLED_OUTPUT_DIR="$output_root/full_diffusion_unrolled_rdbm_${acceleration}x_scratch" \
  FULL_UNROLLED_ACCELERATION="$acceleration" \
    bash scripts/train_full_diffusion_unrolled_rdbm.sh
done

echo "Full diffusion-unrolled acceleration loop complete: $accelerations"
