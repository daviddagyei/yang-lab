#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
remote_host="${FULL_UNROLLED_REMOTE_HOST:-moj@sn4622128247}"
remote_repo="${FULL_UNROLLED_REMOTE_REPO:-/home/moj/david/yang-lab}"
remote_data_root="${FULL_UNROLLED_REMOTE_DATA_ROOT:-/data/ms_safari/david_agyei/nyu-dataset}"
local_data_root="${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
remote_output_root="${FULL_UNROLLED_REMOTE_OUTPUT_ROOT:-$remote_data_root/derived}"
local_output_root="${FULL_UNROLLED_OUTPUT_ROOT:-$local_data_root/derived}"
if (( $# == 0 )); then
  requested_rates=(2 4)
else
  requested_rates=("$@")
fi
for rate in "${requested_rates[@]}"; do
  case "$rate" in
    2|4|6|8) ;;
    *) echo "Unsupported report acceleration: $rate" >&2; exit 2 ;;
  esac
done
rates="${requested_rates[*]}"

control_dir="$(mktemp -d /tmp/full-unrolled-ssh.XXXXXX)"
control_socket="$control_dir/control"
cleanup() {
  ssh -o ControlPath="$control_socket" -O exit "$remote_host" >/dev/null 2>&1 || true
  rmdir "$control_dir" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Opening SSH connection to $remote_host (enter its password when prompted)."
ssh -o ControlMaster=yes -o ControlPath="$control_socket" -o ControlPersist=600 \
  -o StrictHostKeyChecking=accept-new -Nf "$remote_host"

echo "Checking remote test summaries and generating progression figures for: $rates."
ssh -o ControlPath="$control_socket" "$remote_host" \
  bash -s -- "$remote_repo" "$remote_data_root" "$remote_output_root" \
  "${requested_rates[@]}" <<'REMOTE'
set -euo pipefail
repo="$1"; data_root="$2"; output_root="$3"
shift 3
rates="$*"
for rate in $rates; do
  selection="$output_root/full_diffusion_unrolled_rdbm_${rate}x_scratch/evaluation/best_validation_checkpoint.json"
  if [[ ! -f "$selection" ]]; then echo "Missing selection: $selection" >&2; exit 1; fi
  step="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "$selection")"
  summary="$output_root/full_diffusion_unrolled_rdbm_${rate}x_scratch/evaluation/test/step-$(printf '%07d' "$step")/summary.json"
  if [[ ! -f "$summary" ]]; then echo "Missing test summary: $summary" >&2; exit 1; fi
done
cd "$repo"
source .venv/bin/activate
MRI_DATA_ROOT="$data_root" FULL_UNROLLED_OUTPUT_ROOT="$output_root" \
FULL_UNROLLED_REPORT_RATES="$rates" \
  python -u -m mri_diffusion_unrolled_rdbm.cli.generate_progression
REMOTE

ssh_transport="ssh -o ControlPath=$control_socket -o StrictHostKeyChecking=accept-new"
for rate in "${requested_rates[@]}"; do
  remote_run="$remote_output_root/full_diffusion_unrolled_rdbm_${rate}x_scratch"
  local_run="$local_output_root/full_diffusion_unrolled_rdbm_${rate}x_scratch"
  mkdir -p "$local_run/evaluation"
  rsync -ah --info=progress2 -e "$ssh_transport" \
    "$remote_host:$remote_run/train.log" "$local_run/train.log"
  rsync -ah --info=progress2 -e "$ssh_transport" \
    "$remote_host:$remote_run/evaluation/" "$local_run/evaluation/"
done
mkdir -p "$project_root/results/full_diffusion_unrolled_rdbm/qualitative"
rsync -ah --info=progress2 -e "$ssh_transport" \
  "$remote_host:$remote_repo/results/full_diffusion_unrolled_rdbm/qualitative/" \
  "$project_root/results/full_diffusion_unrolled_rdbm/qualitative/"

MRI_DATA_ROOT="$local_data_root" FULL_UNROLLED_OUTPUT_ROOT="$local_output_root" \
FULL_UNROLLED_REPORT_RATES="$rates" FULL_UNROLLED_SKIP_QUALITATIVE=1 \
  bash scripts/build_full_diffusion_unrolled_rdbm_results_notebook.sh
echo "Notebook ready: $project_root/notebooks/full_diffusion_unrolled_rdbm_results.ipynb"
