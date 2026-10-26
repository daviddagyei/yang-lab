#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {2|4|6|8}" >&2
  exit 2
fi
rate="$1"
case "$rate" in 2|4|6|8) ;; *) echo "Invalid acceleration: $rate" >&2; exit 2 ;; esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
checkpoint_dir="${MODL_CHECKPOINT_DIR:-$data_root/derived/modl_${rate}x_fp32}"
evaluation_root="$checkpoint_dir/evaluation"
steps="${MODL_VALIDATION_STEPS:-18980 37960 56940 75920 94900}"

for step in $steps; do
  checkpoint="$checkpoint_dir/checkpoint-$(printf '%07d' "$step").pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  output="$evaluation_root/validation/step-$(printf '%07d' "$step")"
  python -u -m mri_modl.cli.evaluate_split \
    --data-root "$data_root" \
    --checkpoint "$checkpoint" \
    --split val \
    --output-dir "$output"
done

python -m mri_modl.cli.select_checkpoint \
  --checkpoint-dir "$checkpoint_dir" \
  --validation-dir "$evaluation_root/validation" \
  --output "$evaluation_root/best_validation_checkpoint.json"
