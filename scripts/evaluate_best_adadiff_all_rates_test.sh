#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${ADADIFF_OUTPUT_ROOT:-$data_root/derived}"
train_dir="${ADADIFF_OUTPUT_DIR:-$output_root/adadiff_shared_prior}"
selection="$train_dir/best_checkpoint.json"
if [[ ! -f "$selection" ]]; then
  echo "Run validation selection first: $selection is missing" >&2
  exit 1
fi
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"
for acceleration in 2 4 6 8; do
  python -u -m mri_adadiff.cli.evaluate_split \
    --checkpoint "$checkpoint" \
    --split test \
    --acceleration "$acceleration" \
    --adaptation-iterations "${ADADIFF_TEST_ITERATIONS:-1000}" \
    --data-root "$data_root" \
    --output-dir "$train_dir/test_${acceleration}x"
done
