#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_root="${UNROLLED_RDBM_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/unrolled_rdbm_4x_scratch}"
checkpoint="$(find "$checkpoint_root" -maxdepth 1 -type f -name 'checkpoint-epoch-*-step-*.pt' -printf '%p\n' | sort | tail -n 1)"
if [[ -z "$checkpoint" ]]; then
  echo "No completed unrolled-RDBM checkpoint found in $checkpoint_root" >&2
  exit 1
fi
checkpoint_name="$(basename "$checkpoint" .pt)"
step="${checkpoint_name##*-step-}"
output="$checkpoint_root/evaluation/preliminary/step-$step"

python -u -m mri_unrolled_rdbm.cli.evaluate_split \
  --checkpoint "$checkpoint" \
  --split val \
  --precision bf16 \
  --output-dir "$output"
