# Full diffusion-unrolled RDBM (Final RDBM)

This package contains the project's main model: an end-to-end unrolling of the
complete sampled Residual Diffusion Bridge Model (RDBM) trajectory with
multicoil MRI data consistency. It is labeled **Final RDBM** in the comparison
notebook and is distinct from the earlier three-stage RDBM-derived prototype.

## Model lineage

| Version | Learned refinement | Data consistency | Purpose |
| --- | --- | --- | --- |
| Physics-guided MRI RDBM | Original 10-step reverse-diffusion bridge | One gradient-SENSE update after each step | Starting MRI model |
| Three-stage RDBM-derived unrolled | Three direct timestep-conditioned U-Net refinements; no complete diffusion trajectory | Six-iteration CG-SENSE after each refinement | Computationally feasible unrolling prototype |
| **Full diffusion-unrolled RDBM** | Exact 10-step reverse-diffusion bridge, jointly trained end to end | Six-iteration CG-SENSE after every step | Final model |

The final model therefore retains the original ten diffusion steps; it does
not replace each step with another ten-step sampler and does not execute 100
diffusion steps. One shared timestep-conditioned U-Net is evaluated once at
each of the ten reverse times.

## One reconstruction from input to output

The input is the normalized zero-filled SENSE image `x_zf`, acquired multicoil
k-space `y`, Cartesian mask `M`, and supplied sensitivity maps `S`. The common
MRI forward model is

```text
A(x) = M F(Sx).
```

The exact reverse schedule is
`99->89->79->69->59->49->39->29->19->9->-1`. At stage `k`:

1. the shared RDBM U-Net predicts a clean complex image `z_k` from the current
   bridge state, `x_zf`, and timestep `t_k`;
2. a differentiable six-iteration CG-SENSE block computes

   ```text
   x_k = argmin_x ||A(x)-y||_2^2 + lambda_k ||x-z_k||_2^2,
   ```

   equivalently solving
   `(A^H A + lambda_k I)x_k = A^H y + lambda_k z_k`;
3. the original deterministic `pred_x_start` bridge equation uses `x_k` to
   advance the state to the next reverse time.

Each stage has its own learned positive `lambda_k`, while all stages share the
same U-Net weights. "Unrolled" means these ten algorithmic stages form one
differentiable computation graph, so the image prior and consistency weights
are optimized jointly from the final and intermediate reconstruction losses.
The final model starts from random weights and recomputes the U-Net prediction
at every stage; it does not consume cached predictions or a pretrained RDBM.

## What was preserved and changed

Preserved from the MRI RDBM:

- the conditional RDBM U-Net and one shared parameter set across reverse time;
- 100 training diffusion times and the exact ten sampled reverse-time pairs;
- the deterministic `pred_x_start` bridge transition;
- two real/imaginary image channels and zero-filled SENSE conditioning; and
- the frozen volume splits, deterministic masks, supplied maps, normalization,
  and complex multicoil forward/adjoint operators.

Changed for the final model:

- each single gradient-SENSE update was replaced by a differentiable
  six-iteration CG-SENSE proximal solve;
- ten separate positive proximal weights were learned;
- the entire ten-stage trajectory was optimized jointly rather than training a
  single randomly sampled diffusion time per update; and
- intermediate reconstructions were explicitly supervised.

## Training and selection

Separate models were trained from scratch at 2x, 4x, 6x, and 8x acceleration.
Each used the 120-volume/1,898-slice training split for 50 passes (94,900
updates), batch size 1, AdamW at `2e-5`, weight decay `1e-5`, and gradient-norm
clipping at 1.0. The U-Net ran in BF16 while the MRI physics remained
complex64. Checkpoints were saved every five epochs.

For the final prediction `x_K`, the training objective was

```text
L = L1_complex
  + 0.1 L1_magnitude
  + 0.1 L_SSIM
  + 0.1 L_measurement
  + 0.1 mean_k<K L1_complex(x_k, target).
```

Selection was completed without using test metrics:

1. choose the checkpoint with the highest volume-averaged validation PSNR;
2. evaluate stages 1--10 of that checkpoint on validation;
3. choose the early-exit stage by validation PSNR, using SSIM only as a tie-break;
4. evaluate that fixed checkpoint and stage once on the 14-volume/218-slice test split.

| Rate | Selected epoch | Checkpoint step | Selected stage |
| ---: | ---: | ---: | ---: |
| 2x | 10 | 18,980 | 6 |
| 4x | 20 | 37,960 | 9 |
| 6x | 45 | 85,410 | 9 |
| 8x | 40 | 75,920 | 9 |

## Ablations and architecture evolution at 4x

All rows below use the same held-out 14-volume test split. Lower NMSE and
measurement residual are better; higher PSNR and SSIM are better.

| Method/output | PSNR (dB) | SSIM | Complex NMSE | Measurement residual | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Zero-filled SENSE | 29.716 | 0.8295 | 0.07136 | 0.16224 | Common input baseline |
| Original 10-step MRI RDBM | 35.601 | 0.9207 | 0.05591 | 0.14497 | Starting architecture |
| Three-stage RDBM-derived unrolled | 36.369 | 0.9088 | 0.04057 | 0.14873 | Short-unroll prototype |
| Full diffusion-unrolled, stage 10 | 36.715 | 0.9088 | **0.03781** | 0.14784 | Full-depth ablation |
| **Full diffusion-unrolled, stage 9** | **36.948** | 0.9202 | 0.04101 | 0.14608 | **Validation-selected primary result** |
| Full diffusion-unrolled, stage 6 | 36.775 | **0.9317** | 0.04271 | **0.14587** | Exploratory post-selection test |

This ablation shows why stage count is treated as a validation-selected
inference parameter. Stage 9 gave the highest validation PSNR and remains the
primary 4x result. The later exploratory stage-6 test improved SSIM, residual,
and runtime but was inspected after primary selection, so it must not replace
stage 9 in confirmatory comparisons.

Disabling activation checkpointing on the 96 GB GPUs was also tested as a
systems ablation. It changes memory use and training speed only; it does not
change the model equations or learned parameterization.

## Final held-out performance

Metrics are computed per slice, averaged within each volume, and then averaged
equally across the 14 test volumes.

| Rate | Stage | PSNR (dB) | SSIM | Complex NMSE | Measurement residual | Seconds/slice* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x | 6 | 41.315 | 0.9641 | 0.01357 | 0.19606 | 0.240 |
| 4x | 9 | 36.948 | 0.9202 | 0.04101 | 0.14608 | 0.357 |
| 6x | 9 | 34.403 | 0.9051 | 0.05449 | 0.12813 | 0.428 |
| 8x | 9 | 33.525 | 0.8941 | 0.07620 | 0.11313 | 0.430 |

`*` Timing is hardware- and environment-dependent and should only be compared
directly when methods were measured on the same workstation under the same
conditions.

Relative to the original MRI RDBM, validation-selected Final RDBM PSNR changed
by +0.49, +1.35, +2.25, and +2.79 dB at 2x, 4x, 6x, and 8x, respectively. The
largest gains therefore occur at the harder acceleration rates.

## Reproducibility and reporting

The training, checkpoint validation, early-exit selection, fixed-stage
evaluation, and executed report are available in:

- `scripts/train_full_diffusion_unrolled_rdbm.sh`;
- `scripts/validate_full_diffusion_unrolled_rdbm_checkpoints.sh`;
- `scripts/evaluate_full_unrolled_early_exit.sh`;
- `scripts/evaluate_full_unrolled_fixed_stage_test.sh`; and
- [`notebooks/full_diffusion_unrolled_rdbm_results.ipynb`](../notebooks/full_diffusion_unrolled_rdbm_results.ipynb).

Report this as the **full diffusion-unrolled RDBM** or **Final RDBM**. It is a
project-derived architecture, not a reproduction of a separately published
unrolled-RDBM method. Also distinguish the validation-selected primary outputs
from explicitly labeled exploratory test ablations.
