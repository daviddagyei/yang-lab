#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

rates="${MODL_ACCELERATIONS:-2 6 8}"
for rate in $rates; do
  echo "Starting MoDL ${rate}x training."
  bash scripts/train_modl_rate.sh "$rate"
  echo "Finished MoDL ${rate}x training."
done
