"""Validate the target-free 4x self-supervised data path on one real slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import SelfSupervisedMRISliceDataset
from .losses import heldout_kspace_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--input-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    manifest = args.manifest or PROJECT_ROOT / "phase1" / "manifests" / f"{args.split}.jsonl"
    dataset = SelfSupervisedMRISliceDataset(
        manifest,
        split=args.split,
        input_fraction=args.input_fraction,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    dataset.set_epoch(args.epoch)
    sample = dataset[args.sample_index]

    forbidden = {"target", "rss_reference", "undersampled_kspace"} & sample.keys()
    if forbidden:
        raise RuntimeError(f"supervised fields leaked into sample: {sorted(forbidden)}")
    overlap = int((sample["input_mask"] & sample["loss_mask"]).sum().item())
    union_matches = torch.equal(
        sample["input_mask"] | sample["loss_mask"], sample["acquired_mask"]
    )
    acs_held_out = int((sample["acs_mask"] & sample["loss_mask"]).sum().item())
    if overlap or not union_matches or acs_held_out:
        raise RuntimeError("invalid self-supervised mask partition")

    initial_loss = heldout_kspace_loss(
        sample["initial_reconstruction"],
        sample["loss_kspace"],
        sample["sensitivity_maps"],
        sample["loss_mask"],
    )
    if not torch.isfinite(initial_loss):
        raise RuntimeError("initial held-out loss is not finite")

    tensor_summary = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sample.items()
        if isinstance(value, torch.Tensor)
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "protocol": "ASSCGD-inspired target-free 4x",
                "split": args.split,
                "eligible_volumes": dataset.volume_count,
                "eligible_slices": len(dataset),
                "sample_id": sample["metadata"]["sample_id"],
                "mask_partition": {
                    "acquired_lines": sample["metadata"]["acquired_lines"],
                    "input_lines": sample["metadata"]["input_lines"],
                    "heldout_lines": sample["metadata"]["loss_lines"],
                    "acs_lines": sample["metadata"]["acs_lines"],
                    "overlap_lines": overlap,
                    "union_matches_acquired": union_matches,
                    "acs_heldout_lines": acs_held_out,
                    "dynamic": sample["metadata"]["partition_is_dynamic"],
                    "epoch": sample["metadata"]["partition_epoch"],
                },
                "initial_heldout_relative_l1": float(initial_loss),
                "supervised_fields_present": sorted(forbidden),
                "tensors": tensor_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
