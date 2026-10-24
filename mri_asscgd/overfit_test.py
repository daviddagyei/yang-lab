"""Check that the target-free diffusion generator can fit one held-out mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from mri_data import build_dataloader

from .dataset import SelfSupervisedMRISliceDataset
from .model import build_asscgd_diffusion
from .training import move_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "phase1" / "manifests" / "train.jsonl",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.steps < 10:
        raise ValueError("steps must be at least 10")
    if not torch.cuda.is_available():
        raise RuntimeError("the real-data diffusion overfit test requires the CUDA GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dataset = SelfSupervisedMRISliceDataset(args.manifest, split="train")
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    batch = next(iter(build_dataloader(Subset(dataset, [args.sample_index]))))
    batch = move_batch(batch, device)
    model = build_asscgd_diffusion(
        dim=args.dim,
        timesteps=10,
        sampling_timesteps=5,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    timestep = torch.full((1,), 5, dtype=torch.long, device=device)
    noise = torch.randn_like(batch["initial_reconstruction"])

    losses: list[float] = []
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model.training_losses(batch, t=timestep, noise=noise)
        output["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(output["loss"]) or not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite value at overfit step {step}")
        optimizer.step()
        value = float(output["loss"].detach())
        losses.append(value)
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(f"step={step} heldout_loss={value:.7f}", flush=True)

    window = min(5, args.steps // 2)
    initial_mean = float(np.mean(losses[:window]))
    final_mean = float(np.mean(losses[-window:]))
    if final_mean >= initial_mean:
        raise RuntimeError(
            f"overfit gate failed: initial mean {initial_mean:.7f}, final mean {final_mean:.7f}"
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_id": str(batch["metadata"]["sample_id"][0]),
                "steps": args.steps,
                "initial_window_loss": initial_mean,
                "final_window_loss": final_mean,
                "relative_reduction": (initial_mean - final_mean) / initial_mean,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
