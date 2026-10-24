# MRI Reconstruction Development Plan

## Goal

Adapt the current Residual Diffusion Bridge Model (RDBM) from paired RGB image
restoration to physics-guided reconstruction of complex, multi-coil brain MRI data.
The proposed model will alternate learned diffusion updates with sensitivity-aware
k-space data consistency and will be compared with conventional and learned MRI
reconstruction baselines.

## Dataset audit (2026-10-19)

The extracted dataset currently contains:

- 446 raw multi-coil HDF5 volumes (172 GB), containing 7,060 slices in total.
- 148 sensitivity-map HDF5 volumes (21 GB), all paired with raw volumes and
  containing 2,336 slices.
- Acquisition distribution: 266 AXT2, 103 AXT1POST, 30 AXT1, 26 AXFLAIR, and
  21 AXT1PRE volumes.
- Coil counts vary from 4 to 20; the common counts are 4, 16, and 20.
- Most volumes contain 16 slices, while a small number contain 10, 12, or 14.
- Reconstruction sizes include 320x320 and 384x384, plus a small number of
  rectangular or smaller images.
- Existing sensitivity maps cover 26 AXFLAIR, 101 AXT1POST, and 21 AXT1PRE
  volumes. They do not cover the AXT2 or AXT1 volumes.
- For 130 paired volumes, k-space, reconstruction, and sensitivity-map dimensions
  agree directly. Eighteen paired volumes need an explicit crop/alignment rule.
- The supplied `patient_id` attribute is unique across the 446 volumes.

The current study is restricted to the 148 volumes with supplied sensitivity
maps. This subset contains AXFLAIR, AXT1POST, and AXT1PRE acquisitions, so results
must be reported as paired-subset performance and cannot represent the full dataset.

## Phase 1: Manifest, validation, and frozen splits

**Status: complete.** Packaged under `phase1/`: implementation is in
`phase1/mri_manifest.py` and `phase1/build_mri_manifest.py`, with focused tests in
`phase1/tests/test_mri_manifest.py`. Generated manifests and the successful full
validation log are kept in the same folder. The manifest directory remains ignored
so paths to the workstation dataset are not committed to source control.

Build a lightweight manifest without loading complete arrays into memory. For every
volume, record its path, patient ID, acquisition, slice count, coil count, k-space
shape, reconstruction shape, normalization attributes, header encoding information,
and sensitivity-map availability.

Create deterministic train, validation, and test manifests. Split by `patient_id`
before expanding volumes into slices and stratify by acquisition where possible.
Keep the test manifest frozen for all later experiments.

Verification gate:

- Every source volume occurs in exactly one split.
- No patient ID occurs in more than one split.
- Acquisition and coil-count distributions are reported for each split.
- HDF5 datasets contain finite values and expected complex/real dtypes.

## Phase 2: MRI transforms and retrospective undersampling

**Status: complete.** Implemented in `mri_reconstruction/`, with focused tests in
`tests/test_mri_transforms.py` and `tests/test_mri_sampling.py`. The preprocessing
keeps complex measurements for physics operations and converts complex object
images to real/imaginary channels at the existing conditional RDBM boundary. The
project's experimental protocol is fixed at 4x acceleration.

The manifest audit confirms for all 446 volumes that the final k-space axis agrees
with ISMRMRD `kspace_encoding_step_1`. Masks retain an 8% calibration center at
4x, then follow fastMRI's random Cartesian policy outside the center. SHA-256
scores derived from the volume ID and seed keep evaluation masks
independent of global random state and reproducible. Header-derived acquisition
windows exclude encoded padding when reporting retained lines and effective rates.

Implement centered complex FFT/IFFT, center cropping, normalization, and coil
combination utilities. Read the ISMRMRD header to confirm the encoded and
reconstructed dimensions and treat the final k-space dimension as phase encoding
only after validating that convention across the collection.

Use seeded Cartesian masks at nominal 4x acceleration. Each mask
will retain an auto-calibration center and sample the remaining phase-encoding
lines according to a documented variable-density policy. Record the effective
acceleration, since retaining the center changes it from the nominal value.

Generate masks on demand from a stable seed rather than writing another large copy
of the dataset. Validation and test masks must remain fixed across models.

Verification gate:

- Applying a full mask reproduces the original k-space exactly.
- Zero-filled reconstructions have the expected orientation and dimensions.
- The number of retained lines and effective acceleration are correct.
- The same sample identifier always produces the same evaluation mask.

## Phase 3: Supplied sensitivity-map pipeline

**Status: complete.** The `mri_sensitivity/` package standardizes and validates
the supplied maps and center-crops all 18 spatial mismatches. Sensitivity-map
estimation is outside the current scope.

One middle slice from each of the 148 supplied-map volumes passed layout,
finiteness, RSS-normalization, and SENSE reconstruction checks. Median SENSE-vs-RSS
NRMSE within map support was 0.059.

Standardize supplied maps to `[slice, coil, height, width]`, document their complex
convention, and resolve the 18 spatial mismatches using header-derived cropping.
Validate them by reconstructing fully sampled data with the SENSE forward/adjoint
operators and comparing magnitudes with `reconstruction_rss`.

Verification gate:

- Map slice, coil, and spatial axes match their source measurements.
- Maps are finite and have plausible root-sum-of-squares normalization.
- Fully sampled forward/adjoint reconstruction passes numerical and visual checks.

## Phase 4: HDF5 training dataset

**Status: complete for the 4x paired-map prototype.** Implemented in `mri_data/`.
The loader reads one slice lazily, applies a stable per-volume 4x mask, loads its
supplied map, constructs SENSE input and target images, and derives the shared
normalization scale only from the zero-filled input. Variable coil and spatial
dimensions are handled with batch size one until padded collation is needed.

Replace the rain-removal `RSDataset` with an MRI dataset that returns one slice at
a time:

- undersampled complex multi-coil k-space;
- sampling mask;
- complex sensitivity maps;
- zero-filled/SENSE initial reconstruction;
- a fully sampled target and RSS magnitude reference;
- sample and acquisition metadata.

Represent complex object images as two channels (real and imaginary). Derive a
complex reference with sensitivity-weighted coil combination when maps are
available; retain `reconstruction_rss` for standardized magnitude evaluation.
Support variable coil counts without treating coils as fixed neural-network image
channels (for example, process one volume/slice per batch initially or use a padded
collation mask).

Verification gate:

- A sample can be loaded, transformed, masked, and reconstructed end to end.
- Dataset output shapes and dtypes are asserted.
- The target is never used to create the model input except through the explicitly
  simulated acquisition process.
- Train-only augmentation does not alter validation or test samples.

## Phase 5: Multi-coil physics and data consistency

**Status: complete.** Implemented in `mri_reconstruction/physics.py`. The forward
operator expands the image with the supplied coil maps, center-pads to the encoded
matrix, applies the centered orthonormal FFT and the 4x mask. Its adjoint applies
the matching inverse operations. Differentiable gradient consistency and exact
acquired-sample k-space replacement are available for the later RDBM integration.

Implement the forward operator

`A(x) = M F(S x)`

and adjoint

`A^H(y) = sum_c conj(S_c) F^-1(M y_c)`.

Phase 6 replaces the RDBM's current single-image FFT consistency operation with
these multi-coil operators. Hard replacement supports the noiseless retrospective
experiment, while the gradient update provides soft consistency.

Verification gate:

- The numerical inner-product test confirms that the forward and adjoint operators
  are adjoints.
- Fully sampled reconstruction agrees with the reference within documented
  normalization/cropping tolerance.
- A consistency update reduces measured k-space residual.
- Hard consistency makes residual at acquired samples approximately zero.

## Phase 6: Physics-guided RDBM adaptation

**Status: implementation, smoke gates, and the first full training run complete.** The `mri_model/` package
instantiates the existing conditional U-Net with two real/imaginary channels,
keeps the RDBM bridge schedule, and applies multi-coil SENSE consistency after
every reverse update. The MRI objective combines complex-image, magnitude, and
measurement losses. `python -m mri_model.cli.train` reads the HDF5 pipeline directly and stores
checkpoints on `/data` by default.

The supplied-map 4x run completed 94,900 finite optimization steps, approximately
50 slice-level epochs with batch size one. The final training log contains no
NaN or Inf losses.

A real 320x320 forward/backward step completed successfully. A fixed-noise
one-slice overfit reduced loss from 0.1311 to 0.0445 in ten steps. Checkpoint save
and resume and consistency-enabled sampling are covered by automated tests.

Adapt the existing conditional U-Net from three-channel RGB restoration to a
two-channel complex object image. Condition it on the initial reconstruction and,
if beneficial, acquisition/mask information. Apply sensitivity-aware data
consistency during every reverse diffusion step.

Train with a combination of complex image loss, magnitude-domain loss, and
measurement consistency. Add SSIM only where its scale and windowing are defined
consistently. Keep an ablation configuration that disables data consistency so its
effect can be measured.

Verification gate:

- The model deliberately overfits a tiny set of slices.
- Training, checkpoint resume, and deterministic validation complete successfully.
- Sampling preserves acquired measurements when consistency is enabled.
- No PNG or rain-removal path remains in the MRI training entry point.

## Phase 7: MoDL comparison baseline

**Status: first 4x MoDL run and held-out evaluation complete.** The detailed,
source-linked design is in `MODL_IMPLEMENTATION_PLAN.md`. The differentiable CG
block, shared-weight MoDL model, checkpoint trainer, real-data smoke test, and
one-slice overfit gate pass. Training completed 94,900 finite steps. Validation
selected step 37,960; its held-out test results and unroll progression are saved in
`notebooks/modl_results.ipynb`.

Implement MoDL after the first RDBM training milestone. Retain zero-filled RSS
and zero-filled SENSE as non-learned references. Every comparison must use the
same frozen splits, 4x masks, supplied sensitivity maps, normalization, and
volume-level metrics.

Verification gate:

- MoDL runs on the validation split at 4x acceleration.
- Metrics and runtime are saved per volume, not only as global averages.

## Phase 8: Staged training

**Status: complete for the first RDBM run.** Smoke tests, tiny-set overfitting, and
paired-subset 4x training have all completed.

Train and evaluate only on the verified 148-volume supplied-map subset. Report
results per acquisition because its AXFLAIR, AXT1POST, and AXT1PRE distributions
are uneven.

Recommended progression:

1. One slice and one volume smoke tests.
2. Tiny-set overfit.
3. Paired-subset training at 4x.

## Phase 9: Evaluation and robustness

**Status: RDBM and MoDL checkpoint selection and held-out evaluation complete.**
`python -m mri_model.cli.evaluate_split` records resumable per-slice zero-filled and RDBM metrics,
then aggregates within each volume and gives every volume equal weight. The initial
checkpoint screen uses only the frozen validation split. The test split was kept
sealed until validation selected a checkpoint by volume-averaged PSNR.

Validation selected step 94,900 from checkpoints spanning steps 10,000-94,900.
On the held-out 14-volume test split, the selected RDBM reached PSNR 35.60 dB,
SSIM 0.9207, and complex NMSE 0.05591, compared with zero-filled PSNR 29.72 dB,
SSIM 0.8295, and complex NMSE 0.07136. The executed results notebook includes
per-acquisition tables, checkpoint progression, reverse-diffusion progression,
and the inference-time data-consistency ablation.

For MoDL, validation selected step 37,960 from five checkpoints spanning steps
18,980-94,900. On the same held-out test volumes, MoDL reached PSNR 34.80 dB,
SSIM 0.8024, complex NMSE 0.11039, and a 0.14736 measurement residual. This is a
5.09 dB PSNR gain over zero-filled, but SSIM and complex NMSE are worse. The
acquisition breakdown exposes one AXFLAIR failure, while the two T1 groups improve
complex NMSE. The separate executed MoDL notebook reports these results alongside
the matched RDBM and zero-filled references.

Report NMSE, PSNR, SSIM, measured k-space residual, and reconstruction time. Report
both overall and per-acquisition results. Use volume-level aggregation so scans with
more slices do not dominate. Include qualitative error maps and preserve identical
display scaling when comparing methods.

Required comparisons and ablations:

- zero-filled, MoDL, and RDBM results;
- RDBM with and without multi-coil data consistency;
- consistency only at the final step versus at every diffusion step;
- fixed 4x acceleration with effective acceleration reported per volume;
- performance by acquisition, coil count, and image size;
- fixed-mask and unseen-mask generalization.

The first implementation milestone is complete when Phases 1-5 produce a tested
raw-HDF5 data pipeline and correct multi-coil physics. Model training should begin
only after that foundation passes its verification gates.

## Self-supervised 4x diffusion track

**Status: target-free data foundation implemented.** The `mri_asscgd/` package
implements an ASSCGD/SSAD-MRI-inspired measurement split on top of the existing
4x protocol. Each acquired mask is partitioned into disjoint input and held-out
sets, with the calibration center retained in the input. Training partitions can
change reproducibly by epoch; validation and test partitions remain fixed.

The dataset reads no RSS reference, constructs no fully sampled target, and derives
normalization only from input-set measurements. The held-out k-space loss uses the
same supplied maps and multi-coil forward operator as RDBM and MoDL. This establishes
the leakage-tested boundary needed before adding the diffusion generator,
data-consistency updates, adversarial discriminator, and trainer.

**Generator milestone implemented.** A separate conditional diffusion model now
uses the existing two-channel U-Net, a cosine variance schedule, input-mask-only
multi-coil consistency, and held-out k-space supervision. Synthetic tests cover
finite gradients, loss isolation, consistency reduction, accelerated sampling,
and checkpoint restoration. Its real-slice GPU smoke and one-slice overfit gates
passed before the adversarial discriminator was added.

**Adversarial mapper implemented.** The conditional discriminator has four 3x3
convolution blocks with ReLU and batch normalization, followed by a size-agnostic
logit head. Training constructs true and generated shortened diffusion transitions,
uses a gradient-penalized discriminator objective, and combines the held-out
measurement loss with generator adversarial loss at weight 0.1. A short real-data
GPU stability run is the final gate before the full trainer and validation workflow.

**Full training entry point implemented.** `python -m mri_asscgd.cli.train` runs 25 target-free
epochs (47,450 slice updates) with deterministic per-epoch mask changes, BF16
generator convolutions, finite loss/gradient checks, epoch summaries, and complete
generator/discriminator/optimizer checkpoints. `scripts/train_asscgd_full.sh`
automatically resumes the newest completed epoch. A one-step full-width memory
test must pass before starting the long run.
