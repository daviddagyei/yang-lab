#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

rate="${1:-}"
case "$rate" in
  2|4|6|8) ;;
  *) echo "Usage: $0 {2|4|6|8}" >&2; exit 2 ;;
esac

export FULL_UNROLLED_ACCELERATION="$rate"
bash scripts/validate_full_diffusion_unrolled_rdbm_checkpoints.sh
bash scripts/evaluate_best_full_diffusion_unrolled_rdbm_test.sh
bash scripts/evaluate_full_unrolled_early_exit.sh "$rate"

