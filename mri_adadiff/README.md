# AdaDiff multicoil adaptation

This package ports the official [ICON Lab AdaDiff](https://github.com/icon-lab/AdaDiff)
implementation at commit `15f9e1a04c13678d7c3c60ca4a47cf7b12482b24` to the
project's controlled NYU multicoil experiment.

Preserved components:

- complex two-channel NCSN++ generator;
- time conditioning and latent adaptive group normalization;
- time-conditional large-image discriminator and lazy R1 regularization;
- the released DDPM/fan-out variance-scaling initialization rules;
- eight-step VP adversarial diffusion with the published beta range;
- Adam `(0.5, 0.9)`, generator/discriminator learning rates, EMA, and cosine decay;
- hard multicoil data consistency between rapid reverse steps; and
- per-slice prior reset followed by acquired-k-space L1 adaptation.

Necessary adaptations:

- lazy HDF5 input and the frozen 120/14/14 volume splits;
- supplied sensitivity maps and the tested centered-orthonormal multicoil operator;
- 320x320 images with attention at 20x20 rather than 288-wide images at 18;
- a fused Triton FIR implementation that avoids StyleGAN's explicit
  zero-insertion tensor while preserving its `upfirdn2d` mathematics and
  first/second derivatives; and
- volume-balanced PSNR, SSIM, complex NMSE, residual, and timing reports.

The prior sees only fully sampled target images during training. One checkpoint
is shared across 2x, 4x, 6x, and 8x; masks and sensitivity maps enter only during
reconstruction. The released multicoil code initializes rapid diffusion from the
zero-filled image even though the paper describes Gaussian noise. The released
behavior is the default, while `--rapid-start noise` is an explicit ablation.

Run the gates before long training:

```bash
bash scripts/smoke_adadiff.sh
bash scripts/overfit_adadiff.sh
bash scripts/smoke_adadiff_full_architecture.sh
```

Train the single shared prior:

```bash
bash scripts/train_adadiff_prior.sh
```

On a two-GPU workstation, first benchmark the synchronized training path and then run
the data-parallel trainer:

```bash
bash scripts/benchmark_adadiff_2gpu.sh
bash scripts/train_adadiff_prior_2gpu.sh
```

The throughput-oriented two-GPU default uses a global batch of 16 (eight images
per GPU), synchronizes gradients after each update, and retains the official
per-process minibatch-standard-deviation calculation. The published command used
a global batch of four; set `ADADIFF_GLOBAL_BATCH_SIZE=4` for that exact batch
setting. Batch 16 is an explicitly documented compute adaptation that preserves
500 dataset passes but performs fewer optimizer updates. Both paths write the
same checkpoint format.

Screen checkpoints only on 4x validation before running any held-out test:

```bash
bash scripts/validate_adadiff_checkpoints.sh
bash scripts/evaluate_best_adadiff_all_rates_test.sh
```

The official setting is 500 training epochs and 1,000 adaptation updates per
slice. This is intentionally much more expensive at inference than RDBM.

## FIR implementation and performance gate

Production scripts use `ADADIFF_FIR_BACKEND=triton` by default. This is a fused
CUDA implementation of the released StyleGAN operator and supports the double
backward required by R1. Tests compare its outputs, gradients, and second
gradients against the literal reference. An `official` backend dynamically
loads the pinned repository's CUDA source for cross-checking; it also passed the
same numerical test. `optimized` is an exact grouped-convolution fallback.
`reference` explicitly materializes zero-filled feature maps and is prohibited
for a full run.

On the local RTX 6000 Ada, the old reference behavior took about 229 seconds per
update. The fused backend reduced ordinary batch-four updates to about 0.40
seconds; the lazy-R1 update remains about 15 seconds and is now the dominant
cost. Batch 8 completed the full 11-update benchmark at 39.7 GiB peak allocation
and projected about 70 hours on one local GPU. The two-GPU Blackwell projection
must be measured by the DDP gate rather than inferred from VRAM capacity.

Before starting the 500-epoch job on new hardware, run:

```bash
bash scripts/benchmark_adadiff_training.sh       # isolated one-GPU compute gate
bash scripts/benchmark_adadiff_2gpu.sh           # actual two-GPU data/DDP gate
```

Do not launch the full run unless the benchmark completes with finite metrics
and a practical projected duration. The two-GPU gate warms up two updates, then
measures eleven updates including one lazy-R1 update and prints a projected
500-epoch duration. Channels-last execution was benchmarked locally and rejected
because it was slower for this architecture.

