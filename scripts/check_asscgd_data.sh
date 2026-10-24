#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

python -m unittest tests.test_mri_asscgd
python -m mri_asscgd.smoke_test --split train --sample-index 0 --epoch 0
python -m mri_asscgd.smoke_test --split val --sample-index 0
