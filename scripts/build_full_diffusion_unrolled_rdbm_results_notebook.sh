#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

if [[ "${FULL_UNROLLED_SKIP_QUALITATIVE:-0}" != "1" ]]; then
  python -u -m mri_diffusion_unrolled_rdbm.cli.generate_progression
fi
python notebooks/build_full_diffusion_unrolled_rdbm_results_notebook.py
MPLCONFIGDIR=/tmp/mri-matplotlib jupyter nbconvert \
  --to notebook --execute --inplace notebooks/full_diffusion_unrolled_rdbm_results.ipynb \
  --ExecutePreprocessor.timeout=180
