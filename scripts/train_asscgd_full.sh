#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

output_dir="${ASSCGD_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/asscgd_4x_stable}"
epochs="${ASSCGD_EPOCHS:-25}"
workers="${ASSCGD_WORKERS:-4}"
precision="${ASSCGD_PRECISION:-bf16}"
mkdir -p "$output_dir"

python - "$precision" <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; training was not started.")
precision = sys.argv[1]
if precision == "bf16" and not torch.cuda.is_bf16_supported():
    raise SystemExit("This GPU does not support BF16. Run with ASSCGD_PRECISION=fp32.")
print("CUDA device:", torch.cuda.get_device_name(0))
print("Training precision:", precision)
PY

resume_args=()
latest_checkpoint="${ASSCGD_RESUME:-}"
if [[ -z "$latest_checkpoint" ]]; then
  latest_checkpoint="$(
    find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-epoch-*-step-*.pt' -printf '%p\n' \
      | sort \
      | tail -n 1
  )"
fi
if [[ -n "$latest_checkpoint" ]]; then
  echo "Resuming: $latest_checkpoint"
  resume_args=(--resume "$latest_checkpoint")
fi
if [[ "${ASSCGD_RESET_DISCRIMINATOR:-0}" == "1" ]]; then
  resume_args+=(--reset-discriminator)
  echo "Resetting discriminator while preserving generator progress."
fi

python -u -m mri_asscgd.cli.train \
  --output-dir "$output_dir" \
  --epochs "$epochs" \
  --save-every-epochs 1 \
  --learning-rate 2e-4 \
  --discriminator-learning-rate 2e-5 \
  --adversarial-weight 0.1 \
  --gradient-penalty-weight 1.0 \
  --discriminator-update-interval 4 \
  --real-label-smoothing 0.1 \
  --dim 64 \
  --timesteps 100 \
  --sampling-timesteps 10 \
  --discriminator-channels 32 \
  --input-fraction 0.5 \
  --num-workers "$workers" \
  --precision "$precision" \
  "${resume_args[@]}" \
  2>&1 | tee -a "$output_dir/train.log"
