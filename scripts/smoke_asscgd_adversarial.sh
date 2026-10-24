#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

python -m mri_asscgd.adversarial_smoke_test \
  --steps "${ASSCGD_ADVERSARIAL_STEPS:-5}"
