"""Verify that AdaDiff test-time adaptation fits one slice's acquired k-space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mri_adadiff import (
    DiffusionSchedule,
    build_adadiff_generator,
    reconstruct_with_adaptation,
)
from mri_data import MRISliceDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AdaDiff overfit gate requires CUDA")
    if args.iterations <= 1:
        raise ValueError("overfit gate needs at least two iterations")
    torch.manual_seed(1024)
    device = torch.device("cuda")
    dataset = MRISliceDataset(
        PROJECT_ROOT / "phase1/manifests/train.jsonl",
        split="train",
        acceleration=4,
        data_root=args.data_root,
    )
    sample = dataset[0]
    batch = {
        key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }
    model = build_adadiff_generator(
        base_channels=16,
        channel_multipliers=(1, 1, 2, 2, 2),
        style_dimension=64,
        latent_dimension=32,
        mapping_layers=1,
    ).to(device)
    result = reconstruct_with_adaptation(
        model,
        batch,
        schedule=DiffusionSchedule(),
        adaptation_iterations=args.iterations,
    )
    initial = float(result.initial_adaptation_loss or float("nan"))
    final = float(result.final_adaptation_loss or float("nan"))
    if not torch.isfinite(torch.tensor([initial, final])).all():
        raise FloatingPointError("nonfinite AdaDiff overfit loss")
    if final >= initial:
        raise RuntimeError(
            f"AdaDiff acquired-k-space loss did not decrease: {initial} -> {final}"
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_id": sample["metadata"]["sample_id"],
                "iterations": args.iterations,
                "initial_adaptation_loss": initial,
                "final_adaptation_loss": final,
                "relative_reduction": (initial - final) / initial,
                "supervised_target_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

