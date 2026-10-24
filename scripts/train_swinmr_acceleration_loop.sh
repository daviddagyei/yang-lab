#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${SWINMR_OUTPUT_ROOT:-$data_root/derived}"
accelerations="${SWINMR_ACCELERATIONS:-2 4 6 8}"

for acceleration in $accelerations; do
  case "$acceleration" in
    2|4|6|8) ;;
    *) echo "Unsupported acceleration: $acceleration" >&2; exit 1 ;;
  esac

  echo "Starting/resuming SwinMR ${acceleration}x on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}."
  MRI_DATA_ROOT="$data_root" \
  SWINMR_OUTPUT_ROOT="$output_root" \
  SWINMR_OUTPUT_DIR="$output_root/swinmr_pi_complex_${acceleration}x" \
  SWINMR_ACCELERATION="$acceleration" \
    bash scripts/train_swinmr.sh
done

echo "SwinMR acceleration loop complete: $accelerations"
