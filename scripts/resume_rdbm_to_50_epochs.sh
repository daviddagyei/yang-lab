#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

output_dir="${RDBM_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16}"
checkpoint="${RDBM_RESUME:-$output_dir/checkpoint-0050000.pt}"
target_steps="${RDBM_TARGET_STEPS:-94900}"

if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 1
fi

current_step="$(python - "$checkpoint" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(int(checkpoint["step"]))
PY
)"

remaining_steps=$((target_steps - current_step))
if (( remaining_steps <= 0 )); then
  echo "Checkpoint step $current_step has already reached target step $target_steps."
  exit 0
fi

echo "Resuming step $current_step for $remaining_steps additional steps (target: $target_steps)."

RDBM_OUTPUT_DIR="$output_dir" \
RDBM_RESUME="$checkpoint" \
RDBM_STEPS="$remaining_steps" \
RDBM_SAVE_EVERY="${RDBM_SAVE_EVERY:-5000}" \
RDBM_PRECISION="${RDBM_PRECISION:-bf16}" \
bash scripts/train_rdbm_full.sh
