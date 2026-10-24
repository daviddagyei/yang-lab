#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

checkpoint_dir="${ASSCGD_CHECKPOINT_DIR:-/data/david_agyei/nyu-dataset/derived/asscgd_4x_stable}"
selection="$checkpoint_dir/evaluation/best_validation_checkpoint.json"
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"
step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"
output="$checkpoint_dir/evaluation/test/step-$(printf '%07d' "$step")"

python -u -m mri_asscgd.cli.evaluate_split --checkpoint "$checkpoint" --split test --output-dir "$output"
