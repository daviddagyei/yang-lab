#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_dir="${CASCADE_CNN_CHECKPOINT_DIR:-/data/david_agyei/nyu-dataset/derived/multicoil_cascade_cnn_4x}"
evaluation_root="$checkpoint_dir/evaluation"
epochs="${CASCADE_CNN_VALIDATION_EPOCHS:-10 20 30 40 50}"

for epoch in $epochs; do
  epoch_padded="$(printf '%03d' "$epoch")"
  checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type f -name "checkpoint-epoch-${epoch_padded}-step-*.pt" -print -quit)"
  if [[ -z "$checkpoint" ]]; then
    echo "Missing checkpoint for epoch $epoch" >&2
    exit 1
  fi
  step="${checkpoint##*-step-}"
  step="${step%.pt}"
  output="$evaluation_root/validation/step-$step"
  python -u -m mri_cascade_cnn.cli.evaluate_split \
    --checkpoint "$checkpoint" \
    --split val \
    --output-dir "$output"
done

python -m mri_cascade_cnn.cli.select_checkpoint \
  --checkpoint-dir "$checkpoint_dir" \
  --validation-dir "$evaluation_root/validation" \
  --output "$evaluation_root/best_validation_checkpoint.json"
