#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RDBM_OUTPUT_DIR="${RDBM_OUTPUT_DIR:-/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16}"
export RDBM_RESUME="${RDBM_RESUME:-/data/david_agyei/nyu-dataset/derived/rdbm_full/checkpoint-0001000.pt}"
export RDBM_STEPS="${RDBM_STEPS:-49000}"
export RDBM_PRECISION="${RDBM_PRECISION:-bf16}"

exec "$project_root/scripts/train_rdbm_full.sh"
