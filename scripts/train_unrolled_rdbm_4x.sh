#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

output_dir="${UNROLLED_RDBM_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/unrolled_rdbm_4x_scratch}"
epochs="${UNROLLED_RDBM_EPOCHS:-50}"
workers="${UNROLLED_RDBM_WORKERS:-4}"
mkdir -p "$output_dir"

resume_args=()
latest_checkpoint="$(
  find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-epoch-*-step-*.pt' -printf '%p\n' \
    | sort \
    | tail -n 1
)"
if [[ -n "$latest_checkpoint" ]]; then
  echo "Resuming: $latest_checkpoint"
  resume_args=(--resume "$latest_checkpoint")
fi

python -u -m mri_unrolled_rdbm.cli.train \
  --output-dir "$output_dir" \
  --epochs "$epochs" \
  --save-every-epochs 1 \
  --cascades 3 \
  --cg-iterations 6 \
  --learning-rate 2e-5 \
  --precision bf16 \
  --num-workers "$workers" \
  "${resume_args[@]}" \
  2>&1 | tee -a "$output_dir/train.log"
