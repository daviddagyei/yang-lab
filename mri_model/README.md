# Physics-guided MRI RDBM

This package preserves the project's conditional Residual Diffusion Bridge Model
and adapts its U-Net from three RGB channels to two real/imaginary MRI channels.
The model consumes the normalized zero-filled SENSE image from `mri_data` and
applies the multi-coil consistency operator after each reverse-diffusion update.

Training combines real/imaginary L1 loss, magnitude L1 loss, and normalized
measured-k-space residual. Inputs retain their input-derived MRI normalization;
the RGB `[0, 1]` conversion and clipping are not used. The dedicated
`python -m mri_model.cli.train` entry point reads HDF5 data directly and has no PNG or rain-removal
path.

The default checkpoint location is
`/data/david_agyei/nyu-dataset/derived/rdbm`. Full training should start only
after the one-batch and tiny-overfit gates pass.

Run the real one-slice overfit gate with:

```bash
python -m mri_model.overfit_test --steps 25
```

`python -m mri_model.cli.evaluate_slice` loads the architecture configuration from a checkpoint, runs
one selected train/validation/test slice, reports NMSE, PSNR, SSIM and measurement
residual, and saves a four-panel comparison PNG.
