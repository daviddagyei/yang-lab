#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_root="${UNROLLED_RDBM_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/unrolled_rdbm_4x_scratch}"
evaluation_root="$checkpoint_root/evaluation"
epochs="${UNROLLED_RDBM_VALIDATION_EPOCHS:-10 20 30 40 50}"

for epoch in $epochs; do
  step=$((epoch * 1898))
  checkpoint="$checkpoint_root/checkpoint-epoch-$(printf '%03d' "$epoch")-step-$(printf '%07d' "$step").pt"
  output="$evaluation_root/validation/step-$(printf '%07d' "$step")"
  python -u -m mri_unrolled_rdbm.cli.evaluate_split \
    --checkpoint "$checkpoint" --split val --precision bf16 --output-dir "$output"
done

python -m mri_unrolled_rdbm.cli.select_checkpoint \
  --validation-dir "$evaluation_root/validation" \
  --output "$evaluation_root/best_validation_checkpoint.json" \
  --epochs $epochs
