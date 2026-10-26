# Physics-guided MRI RDBM

This package adapts the repository's conditional Residual Diffusion Bridge Model
(RDBM), originally written for paired RGB restoration, to supervised complex
multicoil brain MRI reconstruction.

## Preserved from the starting RDBM

- conditional U-Net and residual diffusion-bridge formulation;
- 100 training diffusion times with `pred_x_start` supervision;
- ten deterministic reverse-diffusion steps at inference; and
- one shared, timestep-conditioned network across the reverse trajectory.

## Adapted for this study

- RGB input/output became two real/imaginary channels; RGB range conversion and
  clipping were removed;
- the condition is a normalized zero-filled SENSE reconstruction rather than a
  degraded natural image;
- every predicted clean image receives one gradient-based multicoil SENSE
  consistency update using measured k-space, the mask, and supplied maps;
- training uses complex-image L1, magnitude L1, and acquired-k-space residual
  losses with weights `1.0/0.1/0.1`; and
- raw HDF5 slices are loaded lazily and trained in BF16 with complex64 physics.

The controlled cohort contains 148 volumes with supplied maps, split by volume
into 120/14/14 train/validation/test volumes (1,898/220/218 slices). Separate
models were trained for 2x, 4x, 6x, and 8x deterministic Cartesian masks for 50
data passes. Checkpoints are selected by volume-averaged validation PSNR before
one held-out test evaluation.

Report this method as **physics-guided MRI RDBM**, not as the unchanged RGB
model. Its MRI consistency operation is a gradient update, not a CG-SENSE solve.
