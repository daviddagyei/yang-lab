#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

remote_host="${SWINMR_REMOTE_HOST:-moj@sn4622128247}"
remote_repo="${SWINMR_REMOTE_REPO:-/home/moj/david/yang-lab}"
remote_data_root="${SWINMR_REMOTE_DATA_ROOT:-/data/ms_safari/david_agyei/nyu-dataset}"
local_data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
remote_output_root="${SWINMR_REMOTE_OUTPUT_ROOT:-$remote_data_root/derived}"
local_output_root="${SWINMR_OUTPUT_ROOT:-$local_data_root/derived}"

control_dir="$(mktemp -d /tmp/swinmr-results-ssh.XXXXXX)"
control_socket="$control_dir/control"
cleanup() {
  ssh -o ControlPath="$control_socket" -O exit "$remote_host" >/dev/null 2>&1 || true
  rmdir "$control_dir" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Opening SSH connection to $remote_host (enter its password when prompted)."
ssh \
  -o ControlMaster=yes \
  -o ControlPath="$control_socket" \
  -o ControlPersist=600 \
  -o StrictHostKeyChecking=accept-new \
  -Nf "$remote_host"

echo "Checking remote summaries and generating the matched qualitative figure."
ssh -o ControlPath="$control_socket" "$remote_host" \
  bash -s -- "$remote_repo" "$remote_data_root" "$remote_output_root" <<'REMOTE'
set -euo pipefail
repo="$1"
data_root="$2"
output_root="$3"
for rate in 2 4 6 8; do
  selection="$output_root/swinmr_pi_complex_${rate}x/evaluation/best_validation_checkpoint.json"
  if [[ ! -f "$selection" ]]; then
    echo "Missing completed validation selection for ${rate}x: $selection" >&2
    exit 1
  fi
  step="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"
  summary="$output_root/swinmr_pi_complex_${rate}x/evaluation/test/step-$(printf '%07d' "$step")/summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "Missing completed test summary for ${rate}x: $summary" >&2
    exit 1
  fi
done
cd "$repo"
source .venv/bin/activate
MRI_DATA_ROOT="$data_root" SWINMR_OUTPUT_ROOT="$output_root" \
  python -u -m mri_swinmr.cli.generate_all_rate_comparison
REMOTE

ssh_transport="ssh -o ControlPath=$control_socket -o StrictHostKeyChecking=accept-new"
for rate in 2 4 6 8; do
  remote_run="$remote_output_root/swinmr_pi_complex_${rate}x"
  local_run="$local_output_root/swinmr_pi_complex_${rate}x"
  mkdir -p "$local_run/evaluation"
  echo "Pulling ${rate}x evaluation artifacts."
  rsync -ah --info=progress2 -e "$ssh_transport" \
    "$remote_host:$remote_run/train.log" "$local_run/train.log"
  rsync -ah --info=progress2 -e "$ssh_transport" \
    "$remote_host:$remote_run/evaluation/" "$local_run/evaluation/"
done

mkdir -p "$project_root/results/swinmr_evaluation/qualitative"
rsync -ah --info=progress2 -e "$ssh_transport" \
  "$remote_host:$remote_repo/results/swinmr_evaluation/qualitative/" \
  "$project_root/results/swinmr_evaluation/qualitative/"

echo "Building and executing the local notebook."
MRI_DATA_ROOT="$local_data_root" \
SWINMR_OUTPUT_ROOT="$local_output_root" \
SWINMR_SKIP_QUALITATIVE=1 \
  bash scripts/build_swinmr_results_notebook.sh

echo "Notebook ready: $project_root/notebooks/swinmr_results.ipynb"
