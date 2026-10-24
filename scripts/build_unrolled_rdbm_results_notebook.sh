#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

python -u -m mri_unrolled_rdbm.cli.generate_progression
python notebooks/build_unrolled_rdbm_results_notebook.py
MPLCONFIGDIR=/tmp/mri-matplotlib jupyter nbconvert \
  --to notebook --execute --inplace notebooks/unrolled_rdbm_results.ipynb \
  --ExecutePreprocessor.timeout=180
