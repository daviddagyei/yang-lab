# Full diffusion-unrolled RDBM

This package is the project's end-to-end unrolling of the complete sampled MRI
RDBM trajectory. It is labeled **Final RDBM** in the comparison notebook and is
distinct from the earlier three-stage direct-refinement model.

## Preserved from the MRI RDBM

- one shared conditional RDBM U-Net;
- the exact ten reverse-time pairs, `99->89->...->9->-1`;
- the deterministic `pred_x_start` bridge transition between reverse times;
- two-channel complex images and the same zero-filled SENSE condition; and
- the common deterministic masks, supplied maps, splits, and normalization.

## Changed for full unrolling

- the original single gradient consistency update is replaced by a
  differentiable six-iteration CG-SENSE proximal solve after every clean-image
  prediction;
- the ten stages have separate learned positive proximal weights;
- the complete trajectory is trained jointly with final and intermediate
  complex-image, magnitude, SSIM, and measurement losses; and
- the network is initialized and trained from scratch rather than loading the
  separately trained RDBM.

Rate-specific models were trained for 2x, 4x, 6x, and 8x for 50 passes with
AdamW at `2e-5` and BF16 neural-network execution. Checkpoints are selected by
volume-averaged validation PSNR; the executed stage count is then selected on
validation as an early-exit ablation before held-out testing. Disabling
activation checkpointing on the 96 GB GPUs changed memory/computation only, not
the model equations.

This is a project-derived architecture, not a reproduction of a separately
published unrolled-RDBM method.
