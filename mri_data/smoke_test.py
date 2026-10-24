"""Load one real MRI slice through the complete Phase 4 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import MRISliceDataset, build_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--acceleration", type=int, choices=(2, 4, 6, 8), default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    manifest = args.manifest or PROJECT_ROOT / "phase1" / "manifests" / f"{args.split}.jsonl"
    dataset = MRISliceDataset(
        manifest, split=args.split, acceleration=args.acceleration
    )
    batch = next(iter(build_dataloader(dataset)))
    tensor_summary = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "eligible_volumes": dataset.volume_count,
                "eligible_slices": len(dataset),
                "acceleration": args.acceleration,
                "sample_id": batch["metadata"]["sample_id"][0],
                "sensitivity_source": batch["metadata"]["sensitivity_source"][0],
                "tensors": tensor_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
