#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

rate="${1:-}"
stage="${2:-}"
case "$rate" in
  2|4|6|8) ;;
  *) echo "Usage: $0 {2|4|6|8} {1..10}" >&2; exit 2 ;;
esac
if [[ ! "$stage" =~ ^([1-9]|10)$ ]]; then
  echo "Usage: $0 {2|4|6|8} {1..10}" >&2
  exit 2
fi

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${FULL_UNROLLED_OUTPUT_ROOT:-$data_root/derived}"
run="$output_root/full_diffusion_unrolled_rdbm_${rate}x_scratch"
selection="$run/evaluation/best_validation_checkpoint.json"
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")"
step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"
output="$run/evaluation/exploratory_test/stage-$(printf '%02d' "$stage")-step-$(printf '%07d' "$step")"

echo "Exploratory test evaluation: acceleration=${rate}x stage=$stage checkpoint_step=$step"
python -u -m mri_diffusion_unrolled_rdbm.cli.evaluate_split \
  --checkpoint "$checkpoint" \
  --split test \
  --data-root "$data_root" \
  --max-stages "$stage" \
  --output-dir "$output"

