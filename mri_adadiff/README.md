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
- an exact pure-PyTorch transcription of the native StyleGAN `upfirdn2d`
  mathematics instead of its legacy compiled CUDA kernel; and
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

Screen checkpoints only on 4x validation before running any held-out test:

```bash
bash scripts/validate_adadiff_checkpoints.sh
bash scripts/evaluate_best_adadiff_all_rates_test.sh
```

The official setting is 500 training epochs and 1,000 adaptation updates per
slice. This is intentionally much more expensive at inference than RDBM.

