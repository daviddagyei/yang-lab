#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

if [[ "${SWINMR_SKIP_QUALITATIVE:-0}" != "1" ]]; then
  python -u -m mri_swinmr.cli.generate_all_rate_comparison
fi
python notebooks/build_swinmr_results_notebook.py
MPLCONFIGDIR=/tmp/mri-matplotlib jupyter nbconvert \
  --to notebook --execute --inplace notebooks/swinmr_results.ipynb \
  --ExecutePreprocessor.timeout=180
