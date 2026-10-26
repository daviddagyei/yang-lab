# Adapted multicoil SSAD-MRI

This package implements the self-supervised adversarial diffusion strategy from
Safari et al., *Self-supervised adversarial diffusion models for fast MRI
reconstruction* (Medical Physics, 2025; arXiv:2406.15656). It is a paper-guided
multicoil adaptation using the official ReconFormer generator code at commit
`e2e0d5e6e58e04ad1c77a1151e63cf56bec21fb1`.

## Preserved from SSAD-MRI

- the ReconFormer generator with `96/48/24` channels, depths `2/1/1`, and five
  recurrent pyramid passes;
- a four-layer transition discriminator and adversarial diffusion objective;
- a 50% acquired-line partition with four central lines shared by input and loss
  masks;
- Adam at `2e-4` and adversarial weight `0.1`; and
- 100 diffusion times with ten reverse sampling steps.

## Adapted for this study

- supplied sensitivity maps replace ESPIRiT estimation;
- the common deterministic 1D Cartesian 4x mask replaces the paper's sampling
  data, after which acquired lines are partitioned for self-supervision;
- multicoil `MFS` consistency replaces ReconFormer's single-image Fourier
  replacement;
- a padded 320x320 network canvas handles variable native image sizes;
- normalized complex held-out k-space MSE implements the paper's squared-error
  self-supervised objective across varying scan intensities;
- the cohort contains FLAIR, T1-pre, and T1-post brain scans rather than the
  paper's T2 cohort; and
- training uses 50 epochs rather than the paper's 25.

Training is target-free: neither the fully sampled complex target nor stored RSS
reference is loaded. Fully sampled data are used only for validation checkpoint
selection and final evaluation. The current study reports 4x only. Describe this
as **adapted multicoil SSAD-MRI**, not an exact reproduction of the authors'
acquisition and preprocessing pipeline.
