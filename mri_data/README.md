# MRI training data

This package implements the slice-level HDF5 input pipeline for the 2x, 4x, 6x,
and 8x retrospective protocols on the 148 paired volumes. It lazily reads raw multi-coil k-space,
applies a deterministic per-volume mask, loads supplied sensitivity maps, constructs a SENSE
zero-filled input and fully sampled complex target, and returns the stored RSS
reference for magnitude evaluation.

Normalization uses one scale computed only from the zero-filled input. K-space,
input, target, and RSS reference share that scale. Complex object images are
returned as two real channels. Because coil counts and spatial sizes vary, the
initial loader deliberately uses one slice per batch.

Volumes without supplied sensitivity maps are outside the current study scope.

Run the real-data smoke test from the project root:

```bash
python -m mri_data.smoke_test --split train
```

Use `--acceleration 2`, `4`, `6`, or `8` to inspect another protocol.
