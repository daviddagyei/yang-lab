#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
output_root="${ADADIFF_OUTPUT_ROOT:-$data_root/derived}"
train_dir="${ADADIFF_OUTPUT_DIR:-$output_root/adadiff_shared_prior}"
validation_dir="$train_dir/validation_4x"
adaptation_iterations="${ADADIFF_VALIDATION_ITERATIONS:-500}"
max_samples="${ADADIFF_VALIDATION_SAMPLES:-}"
gpu0="${ADADIFF_GPU0:-0}"
gpu1="${ADADIFF_GPU1:-1}"
mkdir -p "$validation_dir"

mapfile -t checkpoints < <(
  find "$train_dir" -maxdepth 1 -type f -name 'checkpoint-epoch-*-step-*.pt' | sort
)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No AdaDiff checkpoints found in $train_dir" >&2
  exit 1
fi

evaluate_partition() {
  local gpu="$1"
  local start_index="$2"
  local checkpoint epoch
  local -a sample_args
  sample_args=()
  if [[ -n "$max_samples" ]]; then
    sample_args=(--max-samples "$max_samples")
  fi

  for ((index=start_index; index<${#checkpoints[@]}; index+=2)); do
    checkpoint="${checkpoints[$index]}"
    epoch="$(basename "$checkpoint" | sed -E 's/checkpoint-epoch-([0-9]+)-step-.*/\1/')"
    echo "gpu=$gpu validating epoch=$epoch checkpoint=$checkpoint"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m mri_adadiff.cli.evaluate_split \
      --checkpoint "$checkpoint" \
      --split val \
      --acceleration 4 \
      --adaptation-iterations "$adaptation_iterations" \
      "${sample_args[@]}" \
      --data-root "$data_root" \
      --output-dir "$validation_dir/epoch-$epoch" \
      2>&1 | tee -a "$validation_dir/gpu-${gpu}.log"
  done
}

echo "Validating ${#checkpoints[@]} checkpoints across GPUs $gpu0 and $gpu1."
if [[ -n "$max_samples" ]]; then
  echo "Validation subset: first $max_samples slices per checkpoint."
else
  echo "Validation subset: complete frozen validation split."
fi

evaluate_partition "$gpu0" 0 &
pid0=$!
evaluate_partition "$gpu1" 1 &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "AdaDiff validation failed with status $status" >&2
  exit "$status"
fi

python -u -m mri_adadiff.cli.select_checkpoint \
  --validation-dir "$validation_dir" \
  --output "$train_dir/best_checkpoint.json"

echo "AdaDiff checkpoint selection complete: $train_dir/best_checkpoint.json"
