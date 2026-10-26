#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

rate="${1:-}"
case "$rate" in
  2|6|8) ;;
  *) echo "Usage: $0 {2|6|8}" >&2; exit 2 ;;
esac

checkpoint_dir="/data/david_agyei/nyu-dataset/derived/multicoil_cascade_cnn_${rate}x"

echo "[$(date --iso-8601=seconds)] Starting Cascade CNN ${rate}x pipeline"
CASCADE_CNN_OUTPUT_DIR="$checkpoint_dir" \
  bash scripts/train_cascade_cnn_rate.sh "$rate"
CASCADE_CNN_CHECKPOINT_DIR="$checkpoint_dir" \
  bash scripts/validate_cascade_cnn_checkpoints.sh
CASCADE_CNN_CHECKPOINT_DIR="$checkpoint_dir" \
  bash scripts/evaluate_best_cascade_cnn_test.sh
echo "[$(date --iso-8601=seconds)] Completed Cascade CNN ${rate}x pipeline"
