# Adapted complex multicoil SwinMR-PI

This package ports the authors' official SwinMR backbone from commit
`30f34271a4a112a1c4aac8c87e7c1664f95fc78f` and adapts its published parallel
imaging loss to the controlled NYU multicoil study.

Preserved architecture and training choices:

- six residual Swin Transformer blocks with six layers per block;
- 180-dimensional features, six heads, 8x8 shifted windows, and MLP ratio 2;
- direct residual prediction without an inserted data-consistency layer;
- image, frequency, and VGG-19 perceptual losses weighted 15, 0.1, and 0.0025;
- Adam at `2e-4`, EMA decay `0.999`, 96x96 random crops, and dihedral augmentation.

Required adaptations:

- two real/imaginary input and output channels replace one magnitude channel;
- supplied complex sensitivity maps replace the paper's estimated normalized maps;
- the coil losses support 4--20 coils and exclude zero-padded batch entries;
- deterministic 2x, 4x, 6x, or 8x Cartesian masks and the frozen project splits
  replace G1D30;
- the official learning-rate milestones are scaled to the matched 50-pass budget;
- full 320x320 images are used for inference and common complex-valued evaluation.

The primary model intentionally has no CG-SENSE or acquired-line replacement.
Any future `SwinMR + DC` experiment must be reported as a separate ablation.

Run the gates and training with:

```bash
bash scripts/smoke_swinmr.sh
bash scripts/overfit_swinmr.sh
bash scripts/train_swinmr_4x.sh
```

For one resumable sequential loop over selected rates:

```bash
SWINMR_ACCELERATIONS="2 4 6 8" bash scripts/train_swinmr_acceleration_loop.sh
```

For the intended two-GPU schedule, GPU 0 runs 2x then 6x while GPU 1 runs 4x
then 8x:

```bash
MRI_DATA_ROOT=/data/ms_safari/david_agyei/nyu-dataset \
  bash scripts/train_swinmr_all_rates_2gpu.sh
```

Every rate writes to its own directory and resumes from its latest checkpoint.
Set `SWINMR_GPU0_RATES` or `SWINMR_GPU1_RATES` to change the assignment.

The full training command downloads fixed ImageNet VGG-19 weights on first use
into `/data/david_agyei/nyu-dataset/derived/torch_cache`. After training:

```bash
bash scripts/validate_swinmr_checkpoints.sh
bash scripts/evaluate_best_swinmr_test.sh
```
