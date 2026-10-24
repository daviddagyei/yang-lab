#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_dir="${MODL_CHECKPOINT_DIR:-/data/david_agyei/nyu-dataset/derived/modl_4x_fp32}"
evaluation_root="$checkpoint_dir/evaluation"
steps="${MODL_VALIDATION_STEPS:-18980 37960 56940 75920 94900}"

for step in $steps; do
  checkpoint="$checkpoint_dir/checkpoint-$(printf '%07d' "$step").pt"
  output="$evaluation_root/validation/step-$(printf '%07d' "$step")"
  python -u -m mri_modl.cli.evaluate_split \
    --checkpoint "$checkpoint" \
    --split val \
    --output-dir "$output"
done

python -m mri_modl.cli.select_checkpoint \
  --checkpoint-dir "$checkpoint_dir" \
  --validation-dir "$evaluation_root/validation" \
  --output "$evaluation_root/best_validation_checkpoint.json"
