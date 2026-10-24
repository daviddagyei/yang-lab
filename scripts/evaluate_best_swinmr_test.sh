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
selection="$checkpoint_dir/evaluation/best_validation_checkpoint.json"
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"
step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"

python -u -m mri_swinmr.cli.evaluate_split \
  --checkpoint "$checkpoint" \
  --split test \
  --data-root "$data_root" \
  --output-dir "$checkpoint_dir/evaluation/test/step-$(printf '%07d' "$step")"
