# Full diffusion-unrolled RDBM

This package is the from-scratch, end-to-end unrolling of the complete sampled
MRI RDBM reconstruction trajectory. It is separate from
`mri_unrolled_rdbm/`, which is the existing three-stage RDBM-derived direct
refinement model.

The controlled architecture uses:

- one shared, randomly initialized conditional RDBM U-Net;
- the original ten reverse-time pairs: `99→89→...→9→-1`;
- the original deterministic `pred_x_start` bridge transition;
- a six-iteration differentiable CG-SENSE solve after every clean prediction;
- ten learned positive proximal weights;
- the same loss weights, AdamW settings, BF16 policy, 50 training passes, masks,
  splits, and validation-PSNR selection rule as the existing three-stage model.
- activation checkpointing disabled on the 96 GB training GPUs; this changes
  memory/computation only, not the mathematical model or gradients.

No pretrained RDBM weights are loaded. Intermediate supervision is applied to
the clean estimates after CG, never to the intentionally noisy bridge states.

Run the gates:

```bash
bash scripts/smoke_full_diffusion_unrolled_rdbm.sh
bash scripts/overfit_full_diffusion_unrolled_rdbm.sh
```

Train one rate or a sequential list:

```bash
FULL_UNROLLED_ACCELERATION=4 bash scripts/train_full_diffusion_unrolled_rdbm.sh
FULL_UNROLLED_ACCELERATIONS="2 4 6 8" bash scripts/train_full_diffusion_unrolled_rdbm_loop.sh
```
