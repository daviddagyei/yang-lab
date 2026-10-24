#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_dir="${RDBM_CHECKPOINT_DIR:-/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16}"
evaluation_dir="${RDBM_EVALUATION_DIR:-$checkpoint_dir/evaluation}"
precision="${RDBM_EVAL_PRECISION:-bf16}"
workers="${RDBM_EVAL_WORKERS:-2}"
steps="${RDBM_VALIDATION_STEPS:-10000 20000 30000 40000 50000}"

summary_paths=()
for step in $steps; do
  padded_step="$(printf '%07d' "$step")"
  checkpoint="$checkpoint_dir/checkpoint-$padded_step.pt"
  output_dir="$evaluation_dir/val/step-$padded_step"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$output_dir")"
  python -u -m mri_model.cli.evaluate_split \
    --checkpoint "$checkpoint" \
    --split val \
    --precision "$precision" \
    --num-workers "$workers" \
    --output-dir "$output_dir" \
    2>&1 | tee -a "$output_dir.log"
  summary_paths+=("$output_dir/summary.json")
done

python -m mri_model.cli.select_checkpoint \
  "${summary_paths[@]}" \
  --output "$evaluation_dir/best_validation_checkpoint.json"
