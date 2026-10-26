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
selection="$checkpoint_dir/evaluation/best_validation_checkpoint.json"
if [[ ! -f "$selection" ]]; then
  echo "Missing validation selection: $selection" >&2
  exit 1
fi
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"
step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"
output="$checkpoint_dir/evaluation/test/step-$(printf '%07d' "$step")"

python -u -m mri_modl.cli.evaluate_split \
  --data-root "$data_root" \
  --checkpoint "$checkpoint" \
  --split test \
  --output-dir "$output"
