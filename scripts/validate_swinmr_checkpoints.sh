#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

acceleration="${SWINMR_ACCELERATION:-4}"
case "$acceleration" in
  2|4|6|8) ;;
  *) echo "Unsupported acceleration: $acceleration" >&2; exit 1 ;;
esac
data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${SWINMR_OUTPUT_ROOT:-$data_root/derived}"
checkpoint_dir="${SWINMR_CHECKPOINT_DIR:-$output_root/swinmr_pi_complex_${acceleration}x}"
evaluation_root="$checkpoint_dir/evaluation"
epochs="${SWINMR_VALIDATION_EPOCHS:-10 20 30 40 50}"

for epoch in $epochs; do
  epoch_padded="$(printf '%03d' "$epoch")"
  checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type f -name "checkpoint-epoch-${epoch_padded}-step-*.pt" -print -quit)"
  if [[ -z "$checkpoint" ]]; then
    echo "Missing checkpoint for epoch $epoch" >&2
    exit 1
  fi
  step="${checkpoint##*-step-}"
  step="${step%.pt}"
  python -u -m mri_swinmr.cli.evaluate_split \
    --checkpoint "$checkpoint" \
    --split val \
    --data-root "$data_root" \
    --output-dir "$evaluation_root/validation/step-$step"
done

python -m mri_swinmr.cli.select_checkpoint \
  --validation-dir "$evaluation_root/validation" \
  --output "$evaluation_root/best_validation_checkpoint.json"
