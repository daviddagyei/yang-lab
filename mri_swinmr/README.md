# Adapted complex multicoil SwinMR-PI

This package ports the official
[SwinMR](https://github.com/ayanglab/SwinMR) parallel-imaging backbone at commit
`30f34271a4a112a1c4aac8c87e7c1664f95fc78f` to the controlled NYU multicoil
study.

## Preserved from SwinMR-PI

- six residual Swin Transformer blocks with six layers per block;
- 180-dimensional features, six attention heads, `8x8` shifted windows, and MLP
  ratio 2;
- direct residual reconstruction without an inserted data-consistency layer;
- image, frequency, and VGG-19 perceptual losses weighted `15/0.1/0.0025`; and
- Adam at `2e-4`, EMA decay `0.999`, `96x96` random crops, and dihedral
  augmentation.

## Adapted for this study

- one magnitude channel becomes two real/imaginary input and output channels;
- supplied sensitivity maps replace estimated normalized maps in coil-image and
  frequency losses;
- loss masking supports 4--20 coils and excludes zero-padded batch entries;
- deterministic 2x, 4x, 6x, and 8x Cartesian masks and frozen volume splits
  replace the source G1D30 protocol;
- learning-rate milestones are scaled to the common 50-pass budget; and
- full 320x320 complex images are used at inference and evaluation.

Separate rate-specific models are selected by volume-averaged validation PSNR.
SwinMR is a single feed-forward reconstruction: it has no iterative trajectory,
CG-SENSE, or acquired-line replacement. Any future `SwinMR + DC` result must be
reported as a separate ablation.
