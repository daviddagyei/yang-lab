# Hammernik Variational Network baseline

This package is a PyTorch adaptation of Hammernik et al., *Learning a
Variational Network for Reconstruction of Accelerated MRI Data*. It was checked
against the authors'
[official repository](https://github.com/VLOGroup/mri-variationalnetwork) at
commit `653630b9dcd0687387e6cfa494fa3d049abb83d8`.

## Preserved from the official configuration

- ten unrolled variational-gradient stages;
- 24 stage-specific complex `11x11` filters;
- 31 Gaussian-RBF influence weights per filter over `[-1, 1]`;
- one learned nonnegative SENSE data-term weight per stage;
- zero-mean complex unit-norm filter projections;
- squared magnitude-error supervision; and
- the three-block iPALM optimizer with `0.7` momentum, initial Lipschitz
  estimates of 1,000, and at most 40 backtracking trials.

## Adapted for this study

- TensorFlow 1 and custom `optox` operators are replaced by native differentiable
  PyTorch implementations of the same updates and constraints;
- knee data, source masks, and ESPIRiT maps are replaced by the project's brain
  cohort, deterministic 2x/4x/6x/8x masks, and supplied maps;
- variable coil counts are zero-padded within spatially grouped batches without
  changing the SENSE operator; and
- official maximum-magnitude input normalization and batch size 10 are retained,
  while 9,700 iPALM updates provide 50 passes over 1,898 training slices.

Models are selected by volume-averaged validation PSNR and evaluated on the same
held-out volumes as the other methods. Adam exists only as a labeled ablation;
the primary baseline uses iPALM. Report this as a **PyTorch data-protocol
adaptation of the Hammernik Variational Network**.
