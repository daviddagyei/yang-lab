# Adapted multicoil D5-C5 Cascade CNN

This package adapts Schlemper et al.'s *Deep Cascade of Convolutional Neural
Networks for Dynamic MR Image Reconstruction* to static 2D multicoil brain MRI.
It was checked against the
[official repository](https://github.com/js3611/Deep-MRI-Reconstruction) at
commit `d8a40efd892e57799c3413630e5cb92d5b035cf8`.

## Preserved from D5-C5

- five independent cascades with no parameter sharing;
- five `3x3` convolutions per cascade with widths
  `2->64->64->64->64->2`;
- ReLU after the first four convolutions, residual connections, He-normal
  initialization, and no batch normalization; and
- complex two-channel squared error, Adam (`1e-3`, betas `0.9/0.999`), and an
  explicit `1e-6` convolution-weight L2 penalty.

## Adapted for this study

- the original precombined single-image input is replaced by raw multicoil
  k-space with supplied sensitivity maps;
- hard single-image Fourier replacement is replaced by a fixed, parameter-free
  ten-iteration CG-SENSE proximal solve after every cascade;
- deterministic 2x, 4x, 6x, and 8x Cartesian masks replace the source sampling
  protocol; and
- rate-specific models use the common 120/14/14-volume split, input-derived
  normalization, batch size 10, FP32, and 50 training passes.

Checkpoint selection uses volume-averaged validation PSNR. Because the
data-consistency operator and dataset differ materially from the source, report
this as **adapted multicoil D5-C5**, not an exact reproduction.
