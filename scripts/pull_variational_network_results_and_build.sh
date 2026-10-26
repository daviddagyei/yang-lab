#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

remote_host="${VN_REMOTE_HOST:-moj@sn4622128247}"
remote_root="${VN_REMOTE_ROOT:-/data/ms_safari/david_agyei/nyu-dataset/derived}"
local_root="${VN_LOCAL_ROOT:-/data/david_agyei/nyu-dataset/derived}"
rates=("${@:-2 6}")

echo "Pulling VarNet results from $remote_host (enter its password when prompted)."
control_socket="$(mktemp -u /tmp/vn-results-ssh.XXXXXX)"
ssh -M -S "$control_socket" -o ControlPersist=300 -fN "$remote_host"
cleanup() {
  ssh -S "$control_socket" -O exit "$remote_host" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for rate in ${rates[*]}; do
  remote_run="$remote_root/hammernik_variational_network_${rate}x_ipalm"
  local_run="$local_root/hammernik_variational_network_${rate}x_ipalm"
  echo "Checking ${rate}x evaluation artifacts."
  ssh -S "$control_socket" "$remote_host" \
    "test -f '$remote_run/evaluation/best_validation_checkpoint.json' && test -n \"\$(find '$remote_run/evaluation/test' -type f -name summary.json -print -quit)\""
  mkdir -p "$local_run"
  rsync -ah --info=progress2 --partial -e "ssh -S $control_socket" \
    --include='train.log' \
    --include='evaluation/' \
    --include='evaluation/***' \
    --exclude='*' \
    "$remote_host:$remote_run/" "$local_run/"
done

source .venv/bin/activate
python notebooks/build_variational_network_results_notebook.py
MPLCONFIGDIR=/tmp/mri-matplotlib jupyter nbconvert \
  --to notebook --execute --inplace notebooks/variational_network_results.ipynb \
  --ExecutePreprocessor.timeout=300

echo "Updated notebooks/variational_network_results.ipynb"
