# Phase 1: Manifest, validation, and frozen splits

This folder contains the complete Phase 1 pipeline:

- `mri_manifest.py`: HDF5 inspection, validation, deterministic patient splitting,
  and manifest writing.
- `build_mri_manifest.py`: command-line entry point.
- `tests/`: focused unit tests.
- `manifests/`: generated full, train, validation, and test manifests plus summary.
- `phase1_validation.log`: successful full-value validation output for all 446
  volumes and 7,060 slices.

Run from the project root:

```bash
python -m phase1.build_mri_manifest --value-check full
python -m unittest discover -s phase1/tests -v
```

The generated manifests contain workstation-specific absolute dataset paths and
remain excluded from source control.
