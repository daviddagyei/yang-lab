#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${ADADIFF_OUTPUT_ROOT:-$data_root/derived}"
train_dir="${ADADIFF_OUTPUT_DIR:-$output_root/adadiff_shared_prior}"
validation_dir="$train_dir/validation_4x"
adaptation_iterations="${ADADIFF_VALIDATION_ITERATIONS:-500}"
max_samples="${ADADIFF_VALIDATION_SAMPLES:-20}"
mkdir -p "$validation_dir"

mapfile -t checkpoints < <(find "$train_dir" -maxdepth 1 -type f -name 'checkpoint-epoch-*-step-*.pt' | sort)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No AdaDiff checkpoints found in $train_dir" >&2
  exit 1
fi
for checkpoint in "${checkpoints[@]}"; do
  epoch="$(basename "$checkpoint" | sed -E 's/checkpoint-epoch-([0-9]+)-step-.*/\1/')"
  python -u -m mri_adadiff.cli.evaluate_split \
    --checkpoint "$checkpoint" \
    --split val \
    --acceleration 4 \
    --adaptation-iterations "$adaptation_iterations" \
    --max-samples "$max_samples" \
    --data-root "$data_root" \
    --output-dir "$validation_dir/epoch-$epoch"
done
python -u -m mri_adadiff.cli.select_checkpoint \
  --validation-dir "$validation_dir" \
  --output "$train_dir/best_checkpoint.json"

