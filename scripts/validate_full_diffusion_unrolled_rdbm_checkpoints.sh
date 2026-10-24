#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

acceleration="${FULL_UNROLLED_ACCELERATION:-4}"
data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${FULL_UNROLLED_OUTPUT_ROOT:-$data_root/derived}"
checkpoint_dir="${FULL_UNROLLED_CHECKPOINT_DIR:-$output_root/full_diffusion_unrolled_rdbm_${acceleration}x_scratch}"
evaluation_root="$checkpoint_dir/evaluation"
epochs="${FULL_UNROLLED_VALIDATION_EPOCHS:-5 10 15 20 25 30 35 40 45 50}"

for epoch in $epochs; do
  epoch_padded="$(printf '%03d' "$epoch")"
  checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type f -name "checkpoint-epoch-${epoch_padded}-step-*.pt" -print -quit)"
  if [[ -z "$checkpoint" ]]; then
    echo "Missing checkpoint for epoch $epoch" >&2
    exit 1
  fi
  step="${checkpoint##*-step-}"
  step="${step%.pt}"
  python -u -m mri_diffusion_unrolled_rdbm.cli.evaluate_split \
    --checkpoint "$checkpoint" --split val --data-root "$data_root" \
    --output-dir "$evaluation_root/validation/step-$step"
done

python -m mri_diffusion_unrolled_rdbm.cli.select_checkpoint \
  --validation-dir "$evaluation_root/validation" \
  --output "$evaluation_root/best_validation_checkpoint.json" \
  --epochs $epochs
