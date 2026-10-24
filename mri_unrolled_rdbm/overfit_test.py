"""Verify that the unrolled RDBM can reduce its objective on one real slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mri_data import MRISliceDataset, build_dataloader

from .model import build_unrolled_rdbm


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(20261021)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MRISliceDataset(
        ROOT / "phase1" / "manifests" / "train.jsonl",
        split="train",
        acceleration=4,
    )
    batch = next(iter(build_dataloader(dataset)))
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model = build_unrolled_rdbm(
        rdbm_config={
            "dim": 8,
            "dim_mults": (1, 2, 4, 8),
            "timesteps": 12,
            "sampling_timesteps": 3,
            "image_size": 320,
            "dc_step_size": 1.0,
        },
        cascades=3,
        cg_iterations=3,
        gradient_checkpointing=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    values = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_losses(batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(losses["loss"].detach())
        values.append(value)
        if step == 1 or step % 5 == 0:
            print(f"step={step} loss={value:.7f}", flush=True)
    initial = sum(values[:5]) / min(5, len(values))
    final = sum(values[-5:]) / min(5, len(values))
    if final >= initial:
        raise RuntimeError(f"tiny overfit failed: initial={initial:.7f}, final={final:.7f}")
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_id": batch["metadata"]["sample_id"][0],
                "steps": args.steps,
                "initial_window_loss": initial,
                "final_window_loss": final,
                "relative_reduction": 1.0 - final / initial,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
