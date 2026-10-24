# Phase 2: MRI transforms and retrospective undersampling

This module prepares complex multi-coil MRI data for the existing conditional
RDBM. It follows the paper's physics-driven reconstruction path: undersample raw
k-space in the phase-encoding direction, form an aliased input with centered FFTs,
and preserve the acquired measurements for later data-consistency steps.

Conventions:

- Raw k-space is `[slice, coil, readout, phase_encoding]`.
- Complex object images become two real RDBM channels: real and imaginary.
- FFT and IFFT are centered and orthonormal.
- The main project protocol uses 4x masks with an 8% calibration center. The
  RDBM comparison also supports 2x, 6x, and 8x. Their default calibration
  fractions preserve the established 4x/8x rule that calibration occupies 32%
  of the expected retained lines: 16%, 5.33%, and 4%, respectively.
- Remaining lines follow fastMRI's random Cartesian policy: uniform Bernoulli
  sampling outside the fully sampled center, with nominal acceleration achieved
  in expectation over the encoded matrix. A SHA-256 score derived from the volume
  ID and seed makes masks stable and keeps the same mask across a volume's slices.
- ISMRMRD encoding limits define the acquired phase window; encoded padding is
  excluded from retained-line counts and effective acceleration, so the reported
  effective rate can differ from the nominal rate.
- Normalization uses one real scale shared by input, target, and measurements.
  The scale must be estimated from permitted input data, never a held-out target.

The generated mask records its actual sampled-line count and effective acceleration.
No masked copy of the dataset is stored.

`zero_filled_coil_images` and `zero_filled_rss` provide the first reconstruction
check and later serve as the non-learned input/baseline. Physics-guided SENSE
forward, adjoint, gradient-consistency, and hard k-space replacement operators are
implemented in `physics.py`. The operators center-pad reconstruction images to the
encoded grid and use the corresponding center crop in the adjoint.

Verification on the current dataset established the phase-encoding convention for
all 446 volumes. On a real AXFLAIR slice, full-mask RSS reconstruction matched the
stored reference at `6.58e-08` NRMSE and the 4x zero-filled output retained the
expected 320x320 orientation and dimensions.

## Upstream alignment

The paper's [curated repository](https://github.com/mosaf/Awesome-DL-based-CS-MRI)
contains references rather than preprocessing code. Its fastMRI entry points to the
[official fastMRI repository](https://github.com/facebookresearch/fastMRI), whose
centered orthonormal FFT, center crop, complex RSS, seeded per-volume masking, and
`RandomMaskFunc` conventions are followed here. Native PyTorch complex tensors are
used internally; real and imaginary values move to channel-first form only at the
existing RDBM boundary.
