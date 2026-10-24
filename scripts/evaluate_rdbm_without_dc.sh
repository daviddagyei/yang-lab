#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_dir="${RDBM_CHECKPOINT_DIR:-/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16}"
evaluation_dir="${RDBM_EVALUATION_DIR:-$checkpoint_dir/evaluation}"
selection="$evaluation_dir/best_validation_checkpoint.json"
precision="${RDBM_EVAL_PRECISION:-bf16}"
workers="${RDBM_EVAL_WORKERS:-2}"

if [[ ! -f "$selection" ]]; then
  echo "Run scripts/validate_rdbm_checkpoints.sh first." >&2
  exit 1
fi
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"
step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"
padded_step="$(printf '%07d' "$step")"
output_dir="$evaluation_dir/test_no_dc/step-$padded_step"
mkdir -p "$(dirname "$output_dir")"

python -u -m mri_model.cli.evaluate_split \
  --checkpoint "$checkpoint" \
  --split test \
  --precision "$precision" \
  --num-workers "$workers" \
  --final-consistency-steps 0 \
  --disable-data-consistency \
  --output-dir "$output_dir" \
  2>&1 | tee -a "$output_dir.log"
