# AdaDiff multicoil adaptation

This package ports the official
[ICON Lab AdaDiff](https://github.com/icon-lab/AdaDiff) implementation at commit
`15f9e1a04c13678d7c3c60ca4a47cf7b12482b24` to the project's NYU multicoil
reconstruction pipeline.

## Preserved from AdaDiff

- complex two-channel NCSN++ generator with time conditioning and latent
  adaptive group normalization;
- time-conditional large-image discriminator with lazy R1 regularization;
- released DDPM/fan-out variance-scaling initialization;
- eight-step VP adversarial diffusion with the published beta range;
- Adam betas `0.5/0.9`, generator/discriminator learning rates, EMA, and cosine
  decay; and
- per-slice acquired-k-space adaptation after rapid reconstruction.

## Adapted for this study

- lazy HDF5 input and frozen 120/14/14-volume brain MRI splits replace the source
  dataset pipeline;
- reconstruction uses deterministic Cartesian masks, supplied sensitivity maps,
  and the common centered-orthonormal multicoil operator;
- images are 320x320, moving the corresponding attention resolution from 18x18
  to 20x20;
- the released StyleGAN `upfirdn2d` operation is implemented as a fused Triton
  FIR kernel and checked through first and second derivatives against the pinned
  official CUDA operation; and
- volume-balanced PSNR, SSIM, complex NMSE, residual, and timing replace the
  source reporting path.

The image prior is trained once on fully sampled target images and shared across
2x, 4x, 6x, and 8x; masks and maps enter during reconstruction and test-time
adaptation. The default retains the released zero-filled rapid start, while
Gaussian-noise initialization is a labeled ablation. The intended schedule is
500 prior-training epochs and 1,000 adaptation updates per test slice, making
this substantially more expensive than the direct and unrolled baselines.

No AdaDiff benchmark result should be reported until long prior training,
validation selection, and held-out evaluation are complete.
