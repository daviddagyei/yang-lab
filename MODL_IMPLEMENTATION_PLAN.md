# MoDL Unrolled Reconstruction Plan

## Goal and scope

Implement a PyTorch adaptation of **MoDL: Model-Based Deep Learning Architecture
for Inverse Problems** as the unrolled comparison model for the completed 4x RDBM
experiment. This milestone uses only the verified 148 volumes with supplied
sensitivity maps and the frozen 120/14/14 train/validation/test split.

The later RDBM experiments at 2x, 6x, and 8x acceleration are outside this
milestone. Keeping the first MoDL run at 4x makes its comparison with the existing
RDBM controlled and interpretable.

**Current status: complete for the first 4x run.** The `mri_modl/` package contains
the shared residual CNN, positive learned regularization weight, differentiable
complex CG solver, and weight-shared unrolled forward pass. The numerical,
gradient, real-data, overfit, checkpoint, and resume gates pass. Training completed
94,900 finite steps (50 slice-level epochs). Validation selected step 37,960 by
equal-volume mean PSNR; that checkpoint was evaluated once on the held-out test
split. The executed report is `notebooks/modl_results.ipynb`.

On 14 test volumes, MoDL reached PSNR 34.80 dB, SSIM 0.8024, and complex NMSE
0.11039, compared with zero-filled PSNR 29.72 dB, SSIM 0.8295, and complex NMSE
0.07136. Its 5.09 dB PSNR gain is encouraging, but the worse SSIM and complex NMSE
must be reported; the acquisition breakdown identifies one AXFLAIR failure as the
largest NMSE contributor. The matched RDBM remains stronger overall at PSNR 35.60
dB, SSIM 0.9207, and complex NMSE 0.05591, while MoDL is about 3.56 times faster
per slice in the present evaluation.

## Primary references

- H. K. Aggarwal, M. P. Mani, and M. Jacob, *MoDL: Model-Based Deep Learning
  Architecture for Inverse Problems*, IEEE Transactions on Medical Imaging,
  38(2):394-405. DOI: https://doi.org/10.1109/TMI.2018.2865356
- Authors' official implementation: https://github.com/hkaggarwal/modl
- The project review paper, *Advancing MRI Reconstruction*, Section 3.2 and
  Table 5, which identifies MoDL as a supervised unrolled optimization method
  for parallel MRI.

The official repository is TensorFlow 1 code tied to its own preprocessed data.
We will reproduce its algorithm and default architectural choices in PyTorch while
using this project's existing raw-HDF5 pipeline and controlled evaluation protocol.

## Algorithm to reproduce

MoDL learns a reconstruction under

```text
min_x ||A x - y||_2^2 + lambda ||x - D_w(x)||_2^2,
```

where `A(x) = M F(Sx)` is the multicoil MRI forward operator and `D_w` is a
residual CNN. One unrolled iteration computes

```text
z_k       = D_w(x_k)
x_(k+1)   = (A^H A + lambda I)^(-1) (A^H y + lambda z_k).
```

The inverse is not formed explicitly. A differentiable complex conjugate-gradient
(CG) solver applies it. The same CNN weights and the same learned `lambda` are
reused at every unrolled iteration, matching the paper's recursive architecture.

## What we can reuse

| Existing component | MoDL use |
| --- | --- |
| `mri_data.MRISliceDataset` | Reuse unchanged for the first 4x run. It already returns normalized measured k-space, masks, supplied maps, zero-filled input, target, and metadata. |
| `mri_reconstruction.forward_operator` | Reuse as `A`. |
| `mri_reconstruction.adjoint_operator` | Reuse as `A^H` and to compute `A^H y`. |
| Complex/channel transforms | Reuse for the two-channel CNN representation. |
| Frozen manifests | Reuse the exact train, validation, and test volume split. |
| Metrics and aggregation | Reuse NMSE, PSNR, SSIM, measurement residual, per-volume averaging, and per-acquisition reporting. |
| Checkpoint workflow | Adapt the existing save/resume, validation screening, and held-out-test scripts. |

The RDBM model and checkpoints remain unchanged. MoDL will live in a separate
`mri_modl/` package and use separate `/data/.../modl_4x/` outputs.

## Faithful starting configuration

These choices come from the authors' official code unless marked as a project
protocol choice:

- complex image represented as two real/imaginary channels;
- 5 convolutional layers, 3x3 kernels, and 64 hidden feature maps;
- batch normalization and ReLU after each hidden convolution;
- linear final convolution followed by the residual skip `D_w(x) = x + CNN(x)`;
- one positive learned `lambda`, initialized to 0.05 and shared across unrolls;
- 10 CG iterations per data-consistency solve;
- shared CNN weights across all unrolls;
- Adam optimizer and complex-image mean-squared error;
- batch size 1, required by the current variable-coil/variable-size loader;
- 4x masks and the existing input-derived normalization (project protocol);
- 50 training epochs as the full-run target, matching both the official training
  example and the completed RDBM exposure: 1,898 slices x 50 = 94,900 steps.

We will start with one unroll for numerical and overfit gates, then initialize the
10-unroll model from the same shared weights, as recommended by the official code
for higher unroll counts. Ten unrolls is the single full baseline configuration;
the GPU memory/time smoke test has passed, so this count is frozen before training
and validation screening.

CG and MRI physics remain complex64/float32 even if CNN convolutions later use
BF16. This avoids weakening the numerical solve. The first smoke and overfit runs
use FP32 throughout.

## Implementation sequence

### 1. Shared numerical block

Add a differentiable batched complex CG solver and an operator for
`A^H A + lambda I`.

Verification:

- CG solves a known Hermitian positive-definite system to the requested tolerance.
- The MoDL normal-equation residual decreases over CG iterations.
- Gradients reach both the CNN and `lambda` through the CG block.

### 2. MoDL model package

Create:

```text
mri_modl/
  __init__.py
  cg.py
  model.py
  training.py
  README.md
```

The model accepts the same batch dictionary as `PhysicsGuidedRDBM` and returns a
`[B, 2, H, W]` complex-channel reconstruction. It can optionally return every
unroll output for qualitative progression figures.

Verification:

- output shape and dtype are correct;
- all forward values, loss values, and gradients are finite;
- increasing the requested unroll count does not create new CNN parameters;
- learned `lambda` remains positive;
- a data-consistency solve reduces the normal-equation residual.

### 3. Training and checkpointing

Add `python -m mri_modl.cli.train` and small shell scripts for smoke, overfit, and full training.
Checkpoints record the model configuration, optimizer, step, epoch-equivalent, and
CG/unroll settings. Resume must reproduce the saved step and optimizer state.

Gates before a full run:

1. one real 320x320 forward/backward step;
2. one-slice overfit with a clearly falling MSE;
3. resume from a saved checkpoint;
4. 10-unroll GPU memory/time measurement;
5. no NaN/Inf values in loss, gradients, reconstruction, or learned `lambda`.

### 4. Evaluation

Add a MoDL split evaluator using the existing frozen protocol. Screen checkpoints
only on validation using equal-volume mean PSNR. Evaluate the selected checkpoint
once on test and report:

- complex NMSE, magnitude PSNR, magnitude SSIM;
- measured k-space residual and seconds per slice;
- overall equal-volume means;
- AXFLAIR, AXT1POST, and AXT1PRE results;
- zero-filled, RDBM step 94,900, and MoDL on the same 4x masks;
- representative reconstructions, error maps, and unroll progression.

### 5. Completion criteria

The 4x MoDL milestone is complete when:

- numerical, model, checkpoint, and resume tests pass;
- smoke and one-slice overfit gates pass;
- the full training log is finite;
- validation selects a checkpoint without using test results;
- held-out test metrics and runtime are saved per slice and per volume;
- the results notebook contains a matched zero-filled/RDBM/MoDL comparison.

## Deferred acceleration work

After the 4x MoDL comparison is complete, make acceleration an explicit dataset
and CLI parameter, generate frozen 2x/6x/8x masks, and train or evaluate the RDBM
under a documented multi-rate protocol. That change should be made once in the
shared data pipeline so both models can consume identical masks where needed.
