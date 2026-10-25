#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

report_rates="${FULL_UNROLLED_REPORT_RATES:-2 4}"
if [[ "${FULL_UNROLLED_SKIP_QUALITATIVE:-0}" != "1" ]]; then
  FULL_UNROLLED_REPORT_RATES="$report_rates" \
    python -u -m mri_diffusion_unrolled_rdbm.cli.generate_progression
else
  FULL_UNROLLED_REPORT_RATES="$report_rates" python - <<'PY'
import json
import os
from pathlib import Path

root = Path("results/full_diffusion_unrolled_rdbm/qualitative")
metadata_path = root / "progression.json"
if not metadata_path.exists():
    raise FileNotFoundError(f"missing qualitative metadata: {metadata_path}")
metadata = json.loads(metadata_path.read_text())
rates = os.environ["FULL_UNROLLED_REPORT_RATES"].split()
missing = [rate for rate in rates if rate not in metadata.get("rates", {})]
missing += [rate for rate in rates if not (root / f"progression_{rate}x.png").exists()]
if missing:
    raise FileNotFoundError(
        "qualitative artifacts do not cover requested rates: "
        + ", ".join(sorted(set(missing)))
    )
PY
fi
FULL_UNROLLED_REPORT_RATES="$report_rates" \
  python notebooks/build_full_diffusion_unrolled_rdbm_results_notebook.py
MPLCONFIGDIR=/tmp/mri-matplotlib jupyter nbconvert \
  --to notebook --execute --inplace notebooks/full_diffusion_unrolled_rdbm_results.ipynb \
  --ExecutePreprocessor.timeout=180
