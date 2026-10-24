#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

python -u -m mri_modl.cli.generate_progression
python notebooks/build_modl_results_notebook.py
MPLCONFIGDIR=/tmp/mri-matplotlib jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/modl_results.ipynb \
  --ExecutePreprocessor.timeout=180
