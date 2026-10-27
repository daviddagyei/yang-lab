#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${ADADIFF_OUTPUT_ROOT:-$data_root/derived}"
train_dir="${ADADIFF_OUTPUT_DIR:-$output_root/adadiff_shared_prior}"
selection="$train_dir/best_checkpoint.json"
adaptation_iterations="${ADADIFF_TEST_ITERATIONS:-1000}"
gpu0="${ADADIFF_GPU0:-0}"
gpu1="${ADADIFF_GPU1:-1}"

if [[ ! -f "$selection" ]]; then
  echo "Run validation selection first: $selection is missing" >&2
  exit 1
fi
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"

evaluate_rates() {
  local gpu="$1"
  shift
  local acceleration
  for acceleration in "$@"; do
    echo "gpu=$gpu evaluating ${acceleration}x checkpoint=$checkpoint"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m mri_adadiff.cli.evaluate_split \
      --checkpoint "$checkpoint" \
      --split test \
      --acceleration "$acceleration" \
      --adaptation-iterations "$adaptation_iterations" \
      --data-root "$data_root" \
      --output-dir "$train_dir/test_${acceleration}x" \
      2>&1 | tee -a "$train_dir/test-gpu-${gpu}.log"
  done
}

echo "Evaluating the validation-selected AdaDiff prior across two GPUs."
evaluate_rates "$gpu0" 2 6 &
pid0=$!
evaluate_rates "$gpu1" 4 8 &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "AdaDiff held-out evaluation failed with status $status" >&2
  exit "$status"
fi

echo "AdaDiff held-out evaluation complete for 2x, 4x, 6x, and 8x."
