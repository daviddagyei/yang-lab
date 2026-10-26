#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
rates="${MODL_ABLATION_RATES:-2 4 6 8}"
steps="${MODL_VALIDATION_STEPS:-18980 37960 56940 75920 94900}"

for rate in $rates; do
  checkpoint_dir="$data_root/derived/modl_${rate}x_fp32"
  ablation_dir="$checkpoint_dir/evaluation/unroll_ablation"
  for step in $steps; do
    checkpoint="$checkpoint_dir/checkpoint-$(printf '%07d' "$step").pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2
      exit 1
    fi
    python -u -m mri_modl.cli.evaluate_unroll_ablation \
      --data-root "$data_root" \
      --checkpoint "$checkpoint" \
      --output-dir "$ablation_dir/step-$(printf '%07d' "$step")"
  done
  python -u -m mri_modl.cli.select_unroll_ablation \
    --checkpoint-dir "$checkpoint_dir" \
    --ablation-dir "$ablation_dir" \
    --output "$ablation_dir/selection.json"
done
