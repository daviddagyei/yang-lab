# Three-stage RDBM-derived unrolled network

This package is a project-specific experiment that turns the RDBM U-Net into a
short, explicitly unrolled reconstruction network. It is not a published
baseline and it is not the full RDBM reverse-diffusion sampler.

## Reused from the MRI RDBM

- the same two-channel conditional U-Net architecture;
- timestep conditioning at three points spanning the RDBM diffusion horizon;
- the common complex representation, masks, supplied sensitivity maps, and
  multicoil forward/adjoint operators; and
- supervised complex, magnitude, structural, and measurement-domain losses.

## Changed for the unrolled variant

- the ten-step bridge trajectory is replaced by three direct U-Net refinement
  stages;
- every refinement is followed by a differentiable six-iteration CG-SENSE
  proximal solve with its own learned positive regularization weight;
- intermediate clean reconstructions receive supervision; and
- all stages are optimized jointly with AdamW at `2e-5` for 50 passes.

The reported 4x experiment was initialized and trained from scratch. Although
the code supports an explicitly labeled warm start from an RDBM checkpoint, it
was not used for the primary result. Selection uses volume-averaged validation
PSNR on the frozen 120/14/14-volume split.

Report this as **three-stage RDBM-derived unrolled**, not as an unrolled
execution of the complete diffusion process. The latter is implemented in
`mri_diffusion_unrolled_rdbm/`.
