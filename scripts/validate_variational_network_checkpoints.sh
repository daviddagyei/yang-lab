#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_dir="${VN_CHECKPOINT_DIR:-/data/david_agyei/nyu-dataset/derived/hammernik_variational_network_4x_ipalm}"
evaluation_root="$checkpoint_dir/evaluation"
steps="${VN_VALIDATION_STEPS:-970 1940 2910 3880 4850 5820 6790 7760 8730 9700}"

for step in $steps; do
  checkpoint="$checkpoint_dir/checkpoint-step-$(printf '%07d' "$step").pt"
  output="$evaluation_root/validation/step-$(printf '%07d' "$step")"
  python -u -m mri_variational_network.cli.evaluate_split \
    --checkpoint "$checkpoint" \
    --split val \
    --output-dir "$output"
done

python -m mri_variational_network.cli.select_checkpoint \
  --checkpoint-dir "$checkpoint_dir" \
  --validation-dir "$evaluation_root/validation" \
  --output "$evaluation_root/best_validation_checkpoint.json"
