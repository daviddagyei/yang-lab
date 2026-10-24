#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source .venv/bin/activate

python -m mri_asscgd.adversarial_smoke_test \
  --steps 1 \
  --dim 64 \
  --discriminator-channels 32 \
  --timesteps 100 \
  --sampling-timesteps 10 \
  --precision bf16
