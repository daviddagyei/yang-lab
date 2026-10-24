#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export SWINMR_ACCELERATION=4
exec bash scripts/train_swinmr.sh
