# Adapted multicoil D5-C5 Cascade CNN

This package implements the static 2D D5-C5 model from Schlemper et al.,
*A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image
Reconstruction* (2017). Each of five independent cascades contains five 3x3
convolutions (2 -> 64 -> 64 -> 64 -> 64 -> 2), ReLU after the first four, a
residual image connection, He-normal initialization, and no batch
normalization.

The official model assumes a precombined single image and replaces measured
Fourier coefficients exactly. Our dataset contains raw multicoil k-space, so
each replacement layer is adapted to a fixed, parameter-free 10-iteration
CG-SENSE proximal solve using the supplied sensitivity maps. The CNN topology
and lack of parameter sharing remain unchanged. Results should therefore be
reported as **adapted multicoil D5-C5**, rather than as an exact reproduction.

Training follows the official source: two-channel complex squared error, Adam
with learning rate 1e-3 and betas (0.9, 0.999), explicit 1e-6 convolution-weight
L2 penalty, batch size 10, and FP32. This project uses its frozen 4x masks,
volume splits, and input-only normalization for a controlled comparison.

The upstream reference is pinned in `third_party/README.md`.
