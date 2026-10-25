#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

rate="${1:-${FULL_UNROLLED_ACCELERATION:-}}"
case "$rate" in
  2|4|6|8) ;;
  *) echo "Usage: $0 {2|4|6|8}" >&2; exit 2 ;;
esac
data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${FULL_UNROLLED_OUTPUT_ROOT:-$data_root/derived}"
run="$output_root/full_diffusion_unrolled_rdbm_${rate}x_scratch"
baseline_selection="$run/evaluation/best_validation_checkpoint.json"
checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$baseline_selection")"
step="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$baseline_selection")"
early_root="$run/evaluation/early_exit"

python -u -m mri_diffusion_unrolled_rdbm.cli.evaluate_stages \
  --checkpoint "$checkpoint" \
  --data-root "$data_root" \
  --output-dir "$early_root/validation/step-$(printf '%07d' "$step")"

python -m mri_diffusion_unrolled_rdbm.cli.select_early_exit \
  --validation-summary "$early_root/validation/step-$(printf '%07d' "$step")/summary.json" \
  --output "$early_root/selection.json"

selected_stage="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_stage"])' "$early_root/selection.json")"
python -u -m mri_diffusion_unrolled_rdbm.cli.evaluate_split \
  --checkpoint "$checkpoint" \
  --split test \
  --data-root "$data_root" \
  --max-stages "$selected_stage" \
  --output-dir "$early_root/test/stage-$(printf '%02d' "$selected_stage")-step-$(printf '%07d' "$step")"
