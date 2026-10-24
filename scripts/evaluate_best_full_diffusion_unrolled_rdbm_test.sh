#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

acceleration="${FULL_UNROLLED_ACCELERATION:-4}"
data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${FULL_UNROLLED_OUTPUT_ROOT:-$data_root/derived}"
checkpoint_dir="${FULL_UNROLLED_CHECKPOINT_DIR:-$output_root/full_diffusion_unrolled_rdbm_${acceleration}x_scratch}"
selection="$checkpoint_dir/evaluation/best_validation_checkpoint.json"
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"
step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"

python -u -m mri_diffusion_unrolled_rdbm.cli.evaluate_split \
  --checkpoint "$checkpoint" --split test --data-root "$data_root" \
  --output-dir "$checkpoint_dir/evaluation/test/step-$(printf '%07d' "$step")"
