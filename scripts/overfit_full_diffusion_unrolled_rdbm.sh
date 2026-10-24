#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate
python -m mri_diffusion_unrolled_rdbm.overfit_test \
  --data-root "${MRI_DATA_ROOT:-/data/david_agyei/nyu-dataset}"
