#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

output_dir="${RDBM_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/rdbm_full}"
checkpoint="${1:-$output_dir/checkpoint-0050000.pt}"
split="${2:-val}"
sample_index="${3:-0}"

python -m mri_model.cli.evaluate_slice \
  --checkpoint "$checkpoint" \
  --split "$split" \
  --sample-index "$sample_index" \
  --output-dir "$output_dir/results"
