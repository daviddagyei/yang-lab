#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

rate="${1:-}"
case "$rate" in
  2|4|6|8) ;;
  *) echo "Usage: $0 {2|4|6|8}" >&2; exit 2 ;;
esac

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${VN_OUTPUT_ROOT:-$data_root/derived}"
checkpoint_dir="${VN_CHECKPOINT_DIR:-$output_root/hammernik_variational_network_${rate}x_ipalm}"
selection="$checkpoint_dir/evaluation/best_validation_checkpoint.json"
if [[ ! -f "$selection" ]]; then
  echo "Run validation selection first: $selection is missing" >&2
  exit 1
fi
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"
step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"
output="$checkpoint_dir/evaluation/test/step-$(printf '%07d' "$step")"

python -u -m mri_variational_network.cli.evaluate_split \
  --checkpoint "$checkpoint" \
  --split test \
  --data-root "$data_root" \
  --output-dir "$output"

