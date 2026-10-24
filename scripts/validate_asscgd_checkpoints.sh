#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_dir="${ASSCGD_CHECKPOINT_DIR:-/data/david_agyei/nyu-dataset/derived/asscgd_4x_stable}"
evaluation_root="$checkpoint_dir/evaluation"
epochs="${ASSCGD_VALIDATION_EPOCHS:-10 20 30 40 50}"

for epoch in $epochs; do
  step=$((epoch * 1898))
  checkpoint="$checkpoint_dir/checkpoint-epoch-$(printf '%03d' "$epoch")-step-$(printf '%07d' "$step").pt"
  output="$evaluation_root/validation/step-$(printf '%07d' "$step")"
  python -u -m mri_asscgd.cli.evaluate_split --checkpoint "$checkpoint" --split val --output-dir "$output"
done

python -m mri_asscgd.cli.select_checkpoint \
  --validation-dir "$evaluation_root/validation" \
  --output "$evaluation_root/best_validation_checkpoint.json" \
  --epochs $epochs
