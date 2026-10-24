#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

step="${RDBM_TEST_STEP:-94900}"
precision="${RDBM_EVAL_PRECISION:-bf16}"
workers="${RDBM_EVAL_WORKERS:-2}"
rates="${RDBM_ACCELERATIONS:-2 4 6 8}"

for rate in $rates; do
  if [[ "$rate" == "4" ]]; then
    checkpoint_dir="/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16"
  else
    checkpoint_dir="/data/david_agyei/nyu-dataset/derived/rdbm_${rate}x_bf16"
  fi
  padded_step="$(printf '%07d' "$step")"
  checkpoint="$checkpoint_dir/checkpoint-$padded_step.pt"
  output_dir="$checkpoint_dir/evaluation/test/step-$padded_step"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  mkdir -p "$output_dir"
  echo "Evaluating RDBM ${rate}x at step $step"
  python -u -m mri_model.cli.evaluate_split \
    --checkpoint "$checkpoint" \
    --split test \
    --precision "$precision" \
    --num-workers "$workers" \
    --output-dir "$output_dir" \
    2>&1 | tee -a "$output_dir.log"
done

echo "RDBM acceleration test evaluations complete: $rates"
