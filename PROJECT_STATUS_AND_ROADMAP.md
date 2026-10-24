# Multicoil MRI Reconstruction: Project Status and Roadmap

**Last updated:** October 24, 2026
**Compute:** NVIDIA RTX 6000 Ada Generation, PyTorch/CUDA
**Primary study:** Retrospective multicoil brain MRI reconstruction at 2x, 4x,
6x, and 8x acceleration

## Project goal

Develop and evaluate a physics-guided Residual Diffusion Bridge Model (RDBM) for
accelerated multicoil MRI reconstruction, then compare it fairly against
published supervised, self-supervised, unrolled, transformer, variational, and
diffusion-based reconstruction methods.

Every comparison will use the same eligible NYU brain MRI cohort, frozen
volume-level splits, supplied sensitivity maps, deterministic Cartesian masks,
normalization rules, and volume-balanced evaluation. The final study will cover
2x, 4x, 6x, and 8x acceleration.

## Dataset and experimental cohort

The complete downloaded collection contains:

- 446 raw multicoil HDF5 volumes;
- 7,060 total MRI slices;
- 148 volumes with paired, supplied sensitivity maps;
- 2,336 slices in the paired-map subset;
- 4--20 receiver coils per volume; and
- AXFLAIR, AXT1POST, and AXT1PRE acquisitions in the paired-map subset.

The controlled reconstruction study is restricted to the 148 volumes with
supplied maps:

| Split | Volumes | Slices | Use |
| --- | ---: | ---: | --- |
| Training | 120 | 1,898 | Parameter optimization |
| Validation | 14 | 220 | Checkpoint selection and tuning |
| Test | 14 | 218 | One-time held-out reporting |
| **Total** | **148** | **2,336** | |

Splits are frozen at the patient/volume level to prevent slices from the same
scan appearing in multiple splits. Results describe this paired-map subset and
should not be presented as performance on all 446 volumes.

## Completed data and physics foundation

### 1. Manifest generation and validation

The Phase 1 pipeline in [`phase1/`](phase1/) audits the raw HDF5 files and
records paths, acquisition types, coil counts, shapes, slice counts, patient
identifiers, encoding metadata, and sensitivity-map availability.

Completed checks include:

- finite-value and dtype validation;
- mutually exclusive train, validation, and test splits;
- patient-level leakage prevention;
- acquisition and coil-count distribution reporting; and
- full dataset value validation without loading every volume simultaneously.

### 2. Complex MRI transforms and sampling

[`mri_reconstruction/`](mri_reconstruction/) implements:

- centered orthonormal FFT and inverse FFT;
- complex-to-two-channel conversion and its inverse;
- center cropping and encoded-matrix padding;
- input-derived normalization;
- deterministic Cartesian masks;
- auto-calibration signal (ACS) preservation; and
- configurable 2x, 4x, 6x, and 8x acceleration.

Masks are generated from stable sample identifiers and seeds. Validation and
test masks therefore remain identical across models.

### 3. Sensitivity-map processing

[`mri_sensitivity/`](mri_sensitivity/) loads, standardizes, validates, and aligns
the supplied maps. It handles the 18 volumes whose encoded and reconstructed
spatial dimensions require explicit cropping. Sensitivity-map estimation was
removed from the current study so all experiments use only the 148 supplied-map
volumes.

### 4. Lazy HDF5 dataset

[`mri_data/`](mri_data/) reads one slice at a time and returns:

- undersampled complex multicoil k-space;
- the Cartesian sampling mask;
- supplied complex sensitivity maps;
- a zero-filled SENSE initialization;
- a fully sampled complex target for supervised models;
- an RSS reference and normalization scale; and
- sample, volume, acquisition, coil, and acceleration metadata.

Spatially compatible samples can be batched while shorter coil dimensions are
zero-padded. The forward model is unchanged because the added coil entries and
maps are zero.

### 5. Multicoil MRI physics

The common forward operator is

```text
A(x) = M F(Sx),
```

where `S` applies coil sensitivity maps, `F` is the centered Fourier transform,
and `M` is the sampling mask. The matching adjoint `A^H`, measurement residuals,
gradient consistency, exact acquired-sample replacement, and differentiable
conjugate-gradient solvers are implemented and numerically tested.

## Models completed

### Physics-guided RDBM

[`mri_model/`](mri_model/) adapts the original RGB restoration RDBM to two-channel
complex MRI. It performs a 10-step reverse-diffusion reconstruction and applies a
multicoil gradient-based SENSE consistency update after every reverse step.

Completed work:

- real/imaginary conditional U-Net adaptation;
- complex-image, magnitude, and measurement losses;
- BF16 CNN execution with complex64 MRI physics;
- finite-loss and finite-gradient checks;
- checkpoint save/resume;
- one-slice overfit and real-data smoke tests;
- data-consistency ablation;
- 50-pass training runs at 2x, 4x, 6x, and 8x; and
- validation/test evaluation with qualitative progression figures.

Held-out RDBM results:

| Rate | Zero-filled PSNR | RDBM PSNR | PSNR gain | Zero-filled SSIM | RDBM SSIM | RDBM complex NMSE | Seconds/slice |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x | 35.988 | **40.828** | +4.840 | 0.9304 | **0.9586** | 0.01988 | 0.193 |
| 4x | 29.716 | **35.601** | +5.884 | 0.8295 | **0.9207** | 0.05591 | 0.192 |
| 6x | 26.816 | **32.154** | +5.338 | 0.7736 | **0.8825** | 0.08109 | 0.191 |
| 8x | 25.335 | **30.731** | +5.396 | 0.7441 | **0.8583** | 0.11319 | 0.187 |

The executed report is [`notebooks/rdbm_results.ipynb`](notebooks/rdbm_results.ipynb).

### MoDL baseline

[`mri_modl/`](mri_modl/) implements the published weight-shared MoDL algorithm:

- a five-layer residual CNN;
- ten unrolled denoising/data-consistency iterations;
- a learned positive regularization weight; and
- a differentiable 10-iteration CG-SENSE solve at every unroll.

The 4x model completed 94,900 updates. Validation selected step 37,960. Its
held-out result was 34.805 dB PSNR, 0.8024 SSIM, 0.11039 complex NMSE, and 0.0538
seconds per slice. The executed report is
[`notebooks/modl_results.ipynb`](notebooks/modl_results.ipynb).

### RDBM-derived unrolled network

[`mri_unrolled_rdbm/`](mri_unrolled_rdbm/) tests whether the learned RDBM prior
benefits from explicit end-to-end unrolling. The initial computationally feasible
model uses:

- three timestep-conditioned RDBM-derived U-Net refinements;
- a six-iteration CG-SENSE solve after each refinement; and
- joint end-to-end training from scratch.

The 4x run completed 50 passes; validation selected epoch 20/step 37,960. It
reached 36.369 dB PSNR, 0.9088 SSIM, 0.04057 complex NMSE, and 0.0696 seconds per
slice. It improved PSNR and NMSE over the original 4x RDBM, while SSIM decreased.
The executed report is
[`notebooks/unrolled_rdbm_results.ipynb`](notebooks/unrolled_rdbm_results.ipynb).

### Full diffusion-unrolled RDBM

[`mri_diffusion_unrolled_rdbm/`](mri_diffusion_unrolled_rdbm/) implements the
from-scratch, end-to-end unrolling of the complete sampled RDBM trajectory. It
keeps the original ten reverse-time pairs (`99→89→...→9→-1`), the original
deterministic bridge transitions, one shared timestep-conditioned U-Net, and a
six-iteration CG-SENSE solve after every clean prediction. It uses the same
loss weights, optimizer, 50-pass exposure, masks, splits, and validation-PSNR
selection rule as the existing three-stage RDBM-derived model.

The model never loads pretrained RDBM parameters. Its full real-data CUDA smoke
test passed with finite loss and gradients, and the tiny ten-stage overfit gate
reduced its objective. A non-checkpointed real-data update used 26.70 GiB peak
allocated memory, fitting beside SwinMR on each 96 GB GPU. Training entry points
are ready for 2x, 4x, 6x, and 8x; full training has not started.

### ASSCGD-inspired self-supervised diffusion model

[`mri_asscgd/`](mri_asscgd/) implements a target-free 4x reconstruction track.
For every training slice, acquired measurements are partitioned into disjoint
network-input and held-out loss sets. The ACS region remains in the input, the
partition changes deterministically by epoch, and the fully sampled target is
never read during training.

The model combines:

- the existing conditional diffusion U-Net;
- input-only multicoil consistency;
- held-out k-space supervision;
- an adversarial transition discriminator; and
- stability controls for discriminator updates and adversarial weighting.

The stable run completed 50 epochs/94,900 updates. Validation selected epoch 50.
On the held-out test split it reached 33.837 dB PSNR, 0.7569 SSIM, 0.03730 complex
NMSE, and 0.2995 seconds per slice. Its NMSE is strong, while its magnitude SSIM
remains below the supervised RDBM. This is an ASSCGD/SSAD-MRI-inspired adaptation,
because the published method uses ReconFormer as its generator. The executed
report is [`notebooks/asscgd_results.ipynb`](notebooks/asscgd_results.ipynb).

### Adapted multicoil D5-C5 Cascade CNN

[`mri_cascade_cnn/`](mri_cascade_cnn/) preserves the published five-cascade,
five-convolution architecture with independent cascade weights, 64 hidden
features, ReLU activations, residual connections, and He initialization. Its
single-image hard Fourier replacement is adapted to a fixed 10-iteration
CG-SENSE proximal solve for raw multicoil measurements.

The 4x model completed 50 epochs/9,700 updates. Validation selected epoch 50.
On the held-out test split it reached 35.276 dB PSNR, 0.9080 SSIM, 0.04223
complex NMSE, and 0.1501 seconds per slice. It improves every reported image
metric over zero-filled SENSE. Compared with RDBM, it has slightly lower PSNR
and SSIM but lower complex NMSE. The executed report is
[`notebooks/cascade_cnn_results.ipynb`](notebooks/cascade_cnn_results.ipynb).

## Current 4x comparison

All entries below use the same 14-volume/218-slice held-out test split.

| Method | Supervision | PSNR (dB) | SSIM | Complex NMSE | Seconds/slice |
| --- | --- | ---: | ---: | ---: | ---: |
| Zero-filled SENSE | None | 29.716 | 0.8295 | 0.07136 | -- |
| MoDL | Fully supervised | 34.805 | 0.8024 | 0.11039 | **0.0538** |
| RDBM | Fully supervised | 35.601 | **0.9207** | 0.05591 | 0.1916 |
| Unrolled RDBM-derived | Fully supervised | **36.369** | 0.9088 | 0.04057 | 0.0696 |
| ASSCGD-inspired | Self-supervised | 33.837 | 0.7569 | **0.03730** | 0.2995 |
| Cascade CNN | Fully supervised | 35.276 | 0.9080 | 0.04223 | 0.1501 |

PSNR and SSIM are magnitude-domain metrics; complex NMSE measures complex-valued
error. A method can therefore rank differently across these metrics.

## Model currently training

The following status is a snapshot from October 23, 2026 and will become stale as
the jobs continue.

### Hammernik Variational Network

[`mri_variational_network/`](mri_variational_network/) is a PyTorch port checked
against the authors' official repository. It preserves:

- ten unrolled variational-gradient stages;
- 24 stage-specific complex 11x11 filters;
- 31 Gaussian RBF weights per filter;
- learned nonnegative SENSE data weights;
- constrained filter projections; and
- the official three-block iPALM optimizer.

The 4x run targets 9,700 iPALM updates, equivalent to 50 passes over 1,898
training slices. At this snapshot it had reached approximately **3,360/9,700
updates (35%)** with finite losses and gradients.

## Model ready to train

### Adapted complex SwinMR-PI

[`mri_swinmr/`](mri_swinmr/) ports the official SwinMR backbone pinned at commit
`30f34271a4a112a1c4aac8c87e7c1664f95fc78f`. It preserves the published six
residual Swin Transformer blocks, six attention layers per block, 180-dimensional
embeddings, six attention heads, 8x8 windows, direct residual reconstruction,
Adam optimizer, EMA weights, and the image-, frequency-, and VGG-based loss.

The project adaptation replaces the official fixed 12-coil magnitude interface
with two-channel complex reconstruction, supports 4--20 valid coils, uses the
supplied sensitivity maps and frozen masks/splits, and evaluates full 320x320
images. It deliberately has no explicit data-consistency layer, matching the
published SwinMR forward model. The planned 4x run uses 96x96 training crops and
50 matched dataset passes. The real-data CUDA smoke and overfit gates passed;
full training has not started.

## Validation and reporting protocol

Every completed model follows the same evaluation rules:

1. Train only on the 120-volume training split.
2. Screen checkpoints only on the 14-volume validation split.
3. Select the checkpoint with the highest volume-averaged validation PSNR.
4. Evaluate the selected checkpoint once on the sealed 14-volume test split.
5. Save per-slice metrics, then aggregate slices within each volume.
6. Give each volume equal weight in the final mean.
7. Report overall and acquisition-specific results.
8. Report PSNR, SSIM, complex NMSE, measurement residual, and reconstruction time.
9. Use matched display and error scales for qualitative comparisons.

The project contains automated tests for complex transforms, masks, sensitivity
maps, forward/adjoint physics, data consistency, datasets, models, gradients,
checkpoint restoration, and metric aggregation. The latest full CPU suite passed
all **85 tests**.

## Full model and acceleration roadmap

The intended end state is a controlled evaluation at all four acceleration rates.

| Model | 2x | 4x | 6x | 8x | Current state |
| --- | :---: | :---: | :---: | :---: | --- |
| Zero-filled SENSE | Done | Done | Done | Done | Reference baseline |
| RDBM | Done | Done | Done | Done | Training and test evaluation complete |
| RDBM-derived unrolled | Planned | Done | Planned | Planned | 4x complete |
| Full diffusion-unrolled RDBM | Ready | Ready | Ready | Ready | Multi-rate implementation and gates complete |
| MoDL | Planned | Done | Planned | Planned | 4x complete |
| ASSCGD-inspired | Planned | Done | Planned | Planned | 4x complete |
| Variational Network | Planned | Training | Planned | Planned | 4x active |
| Cascade CNN | Planned | Done | Planned | Planned | 4x evaluation complete |
| ReconFormer | Planned | Planned | Planned | Planned | Not implemented yet |
| SwinMR | Ready | Ready | Ready | Ready | Multi-rate implementation and 4x gates complete |
| SSDU | Planned | Planned | Planned | Planned | Official baseline not implemented yet |
| Noise2Recon/meddlr | Planned | Planned | Planned | Planned | Not implemented yet |
| RARE | Planned | Planned | Planned | Planned | Not implemented yet |
| AdaDiff | Planned | Planned | Planned | Planned | Not implemented yet |
| Score-MRI | Planned | Planned | Planned | Planned | Not implemented yet |

### Published baseline references

- Variational Network: <https://github.com/VLOGroup/mri-variationalnetwork>
- Cascade CNN: <https://github.com/js3611/Deep-MRI-Reconstruction>
- MoDL: <https://github.com/hkaggarwal/modl>
- ReconFormer: <https://github.com/guopengf/ReconFormer>
- SwinMR: <https://github.com/ayanglab/SwinMR>
- SSDU: <https://github.com/byaman14/SSDU>
- Noise2Recon/meddlr: <https://github.com/ad12/meddlr>
- RARE: <https://github.com/wustl-cig/RARE>
- AdaDiff: <https://github.com/icon-lab/AdaDiff>
- Score-MRI: <https://github.com/HJ-harry/score-MRI>

## Next steps

### Immediate

1. Allow the 4x Variational Network run to finish.
2. Validate its checkpoints at matched exposure intervals.
3. Select its best checkpoint using volume-averaged validation PSNR.
4. Evaluate the selected checkpoint once on the test split.
5. Produce a concise, executed VarNet notebook with metrics, timing,
   reconstructions, and error maps.
6. Add the VarNet result to the matched 4x comparison table.
7. Start the verified 4x SwinMR training run after GPU capacity is available.

The existing commands are:

```bash
bash scripts/validate_variational_network_checkpoints.sh
bash scripts/evaluate_best_variational_network_test.sh

```

### Acceleration expansion

1. Generalize each model's training and evaluation entry points to accept an
   explicit acceleration argument and rate-specific output directory.
2. Reuse the frozen 2x, 4x, 6x, and 8x masks already supported by the shared data
   layer.
3. Train separate weights at each acceleration rate unless a model is explicitly
   designed and labeled as a single multi-rate model.
4. Select checkpoints independently on validation for each model-rate pair.
5. Keep the test split sealed until selection is complete.

### Remaining baseline implementation

For ReconFormer, SSDU, Noise2Recon, RARE, AdaDiff, and Score-MRI:

1. Pin and document the official repository commit.
2. Reproduce the official architecture and optimizer before adding project-specific
   changes.
3. Document every necessary adaptation for raw multicoil data, supplied maps, and
   the common masks.
4. Pass synthetic numerical and gradient tests.
5. Pass one real-slice smoke and overfit gates.
6. Train and validate the 4x model first.
7. Expand the verified implementation to 2x, 6x, and 8x.
8. Generate a model-specific notebook and update the combined comparison.

### Final analysis

- compare supervised and self-supervised methods separately and together;
- report performance as acceleration increases;
- analyze results by acquisition, coil count, and image size;
- compare reconstruction quality against inference time and training cost;
- include data-consistency and architecture ablations;
- quantify per-volume variability and confidence intervals;
- select representative and failure-case reconstructions without cherry-picking;
- document adaptations that prevent any experiment from being called an exact
  reproduction; and
- prepare final paper figures, tables, presentation material, and resume-ready
  project outcomes.

## Reproducibility and engineering work

The project now includes:

- lazy HDF5 loading for the 172 GB raw dataset;
- deterministic masks and volume-level splits;
- native complex PyTorch MRI operations;
- CUDA execution through PyTorch, cuDNN, cuBLAS, and cuFFT;
- FP32 and BF16 training paths where numerically appropriate;
- resumable checkpoints and `tmux`-safe shell scripts;
- finite loss/gradient gates;
- variable-coil padded batching;
- per-slice JSONL metrics and volume-balanced summaries;
- isolated packages for each reconstruction model; and
- executed Jupyter notebooks for completed experiments.

No custom CUDA C++ kernels have been written; GPU computation is expressed through
PyTorch's CUDA operations.

## Important limitations

- Only 148 of 446 raw volumes have supplied sensitivity maps and are currently
  eligible for the controlled study.
- The paired subset excludes AXT2 and AXT1 acquisitions.
- Some baselines require principled multicoil adaptations because their official
  implementations assume coil-combined or differently preprocessed inputs.
- The Cascade CNN uses adapted CG-SENSE consistency instead of the publication's
  single-image hard Fourier replacement.
- ASSCGD is inspired by the paper but currently uses the project's diffusion U-Net
  instead of ReconFormer.
- Different metrics evaluate different properties; a PSNR improvement does not
  guarantee an SSIM or complex-NMSE improvement.
- Simultaneous jobs fit in GPU memory but compete for the same compute resources.
- Planned model-rate cells must remain labeled as planned until training,
  validation selection, and held-out evaluation are complete.
