#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

precision="${RDBM_PRECISION:-bf16}"

python - "$precision" <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; full training was not started.")
print("CUDA device:", torch.cuda.get_device_name(0))
precision = sys.argv[1]
if precision == "bf16" and not torch.cuda.is_bf16_supported():
    raise SystemExit("This GPU does not support BF16. Run with RDBM_PRECISION=fp32.")
print("Training precision:", precision)
PY

output_dir="${RDBM_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/rdbm_full}"
steps="${RDBM_STEPS:-50000}"
save_every="${RDBM_SAVE_EVERY:-1000}"
workers="${RDBM_WORKERS:-4}"
acceleration="${RDBM_ACCELERATION:-4}"
resume_args=()
if [[ -n "${RDBM_RESUME:-}" ]]; then
  resume_args=(--resume "$RDBM_RESUME")
fi

mkdir -p "$output_dir"

python -u -m mri_model.cli.train \
  --steps "$steps" \
  --save-every "$save_every" \
  --learning-rate 8e-5 \
  --dim 64 \
  --timesteps 100 \
  --sampling-timesteps 10 \
  --num-workers "$workers" \
  --acceleration "$acceleration" \
  --precision "$precision" \
  --output-dir "$output_dir" \
  "${resume_args[@]}" \
  2>&1 | tee -a "$output_dir/train.log"
