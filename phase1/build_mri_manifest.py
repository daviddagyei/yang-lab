"""Build validated, deterministic train/validation/test manifests for MRI data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mri_manifest import (
    MRIManifestBuilder,
    ManifestConfig,
    ManifestWriter,
    PatientStratifiedSplitter,
)


DEFAULT_DATA_ROOT = Path("/data/david_agyei/nyu-dataset/extracted/FastMRI_Share")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests",
    )
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--value-check",
        choices=("none", "sampled", "full"),
        default="sampled",
        help="Finite-value checking level. 'full' reads all 193 GB and can take a long time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    config = ManifestConfig(data_root=args.data_root, value_check=args.value_check)
    records, warnings = MRIManifestBuilder(config).build()
    splitter = PatientStratifiedSplitter(seed=args.seed, ratios=ratios)
    assignments = splitter.split(records)
    summary = ManifestWriter().write(
        args.output_dir,
        records,
        assignments,
        warnings,
        source_root=args.data_root,
        seed=args.seed,
        ratios=ratios,
        value_check=args.value_check,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
