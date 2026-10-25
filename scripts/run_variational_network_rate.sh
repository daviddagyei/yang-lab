#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

rate="${1:-}"
case "$rate" in
  2|4|6|8) ;;
  *) echo "Usage: $0 {2|4|6|8}" >&2; exit 2 ;;
esac

bash scripts/train_variational_network_rate.sh "$rate"
bash scripts/validate_variational_network_rate.sh "$rate"
bash scripts/evaluate_best_variational_network_rate_test.sh "$rate"

