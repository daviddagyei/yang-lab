# Multicoil MoDL baseline

This package adapts Aggarwal, Mani, and Jacob's
[MoDL](https://github.com/hkaggarwal/modl) architecture to the controlled NYU
multicoil study. The reference is *MoDL: Model-Based Deep Learning Architecture
for Inverse Problems*, IEEE TMI 38(2), 394-405.

## Preserved from MoDL

- a five-layer, 64-channel residual CNN denoiser;
- sharing the same denoiser weights across ten unrolled iterations;
- a learned positive regularization parameter; and
- alternating learned denoising with a ten-iteration conjugate-gradient data
  consistency solve.

The data-consistency subproblem remains

```text
(A^H A + lambda I)x = A^H y + lambda D_w(x).
```

## Adapted for this study

- the implementation is native PyTorch rather than the authors' released
  TensorFlow pipeline;
- `A` is the project's centered-orthonormal multicoil SENSE operator using raw
  k-space, deterministic Cartesian masks, and supplied sensitivity maps;
- images use real/imaginary channels at 320x320 reconstruction resolution;
- each acceleration rate has separate weights and the frozen volume-level
  train/validation/test split; and
- training uses complex two-channel MSE and Adam at `1e-3` for 50 passes.

For memory-safe initialization, the first five passes use one unroll before the
same checkpoint is expanded to the final ten-unroll model. This warm-up is a
study-specific optimization schedule, not part of the published architecture.
Checkpoint selection is based on volume-averaged validation PSNR.

Report this as a **PyTorch multicoil MoDL adaptation**, not an exact execution of
the authors' released data and training pipeline.
