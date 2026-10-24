# MRI Variational Network baseline

This package is a PyTorch adaptation of the architecture in Hammernik et al.,
*Learning a Variational Network for Reconstruction of Accelerated MRI Data*
(MRM 2018). The implementation was checked against the authors' official MIT
licensed repository at commit `653630b9dcd0687387e6cfa494fa3d049abb83d8`:

<https://github.com/VLOGroup/mri-variationalnetwork>

Architecture preserved from the official configuration and update equation:

- 10 unrolled variational-gradient stages;
- 24 stage-specific complex filters of size 11 x 11;
- 31 Gaussian RBF weights per filter over `[-1, 1]`;
- a learned nonnegative SENSE data-term weight per stage;
- zero-mean, complex unit-norm filter projections;
- squared magnitude-error supervision; and
- the authors' three-block iPALM optimizer with 0.7 inertial momentum,
  initial Lipschitz estimates of 1000, and at most 40 backtracking trials.

The TensorFlow 1/custom-op runtime is replaced with native PyTorch. Each iPALM
iteration alternates through the same complex-filter, RBF-weight, and
nonnegative data-term blocks as the official code. The real and imaginary
filter tensors are optimized and projected jointly as one complex block.
The official maximum-magnitude zero-filled input normalization is also used.

The baseline uses the project's brain MRI data, retrospective 4x masks,
sensitivity maps, normalization, and frozen splits so that its results can be
compared with the other models. The published batch size of 10 is retained:
samples are grouped by spatial shape and shorter coil dimensions are
zero-padded, which does not change the SENSE operator. Training uses 9,700
iPALM updates, equal to 50 passes over the current 1,898-slice training set.
The change from the publication's knee dataset to this project's brain dataset
must be reported with the results.
The project also uses its frozen retrospective masks and supplied sensitivity
maps, whereas the original experiment used its own fixed masks and ESPIRiT
maps. These are data-protocol adaptations, not architecture changes.

Run the publication-faithful training path with:

```bash
bash scripts/train_variational_network_4x.sh
```

Adam remains available only as a labeled optimizer ablation through
`python -m mri_variational_network.cli.train --optimizer adam`.

The official knee training set has 200 slices, so its 1,000 batches of 10
represent about 50 data passes. This project has 1,898 training slices and 194
spatially grouped batches per pass. We therefore use 9,700 updates to preserve
the same 50-pass exposure. This intentionally adapts the source repository's
absolute update count to the larger dataset.

```bash
bash scripts/train_variational_network_4x.sh
```
