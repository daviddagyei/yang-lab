#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

python -u -m mri_model.cli.generate_progression
python notebooks/build_rdbm_results_notebook.py
MPLCONFIGDIR=/tmp/mri-matplotlib jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/rdbm_results.ipynb \
  --ExecutePreprocessor.timeout=180
