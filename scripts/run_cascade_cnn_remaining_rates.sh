#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

parallel_jobs="${CASCADE_CNN_PARALLEL_JOBS:-2}"
if [[ ! "$parallel_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "CASCADE_CNN_PARALLEL_JOBS must be a positive integer" >&2
  exit 2
fi

printf '%s\n' 2 6 8 | xargs -r -n 1 -P "$parallel_jobs" \
  bash scripts/run_cascade_cnn_rate.sh
