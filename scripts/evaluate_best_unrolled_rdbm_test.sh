#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_root="${UNROLLED_RDBM_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/unrolled_rdbm_4x_scratch}"
selection="$checkpoint_root/evaluation/best_validation_checkpoint.json"
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"
step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"
output="$checkpoint_root/evaluation/test/step-$(printf '%07d' "$step")"

python -u -m mri_unrolled_rdbm.cli.evaluate_split \
  --checkpoint "$checkpoint" --split test --precision bf16 --output-dir "$output"
