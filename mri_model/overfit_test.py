"""Verify that the MRI RDBM can overfit one real slice with fixed diffusion noise."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mri_data import MRISliceDataset, build_dataloader
from mri_model import build_mri_rdbm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MRISliceDataset(
        PROJECT_ROOT / "phase1" / "manifests" / "train.jsonl",
        split="train",
        mask_seed=20261019,
    )
    batch = next(iter(build_dataloader(dataset)))
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model = build_mri_rdbm(
        dim=args.dim,
        timesteps=4,
        sampling_timesteps=2,
        image_size=batch["target"].shape[-1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    timestep = torch.full((1,), 1, dtype=torch.long, device=device)
    noise = torch.randn_like(batch["target"])

    model.train()
    initial_loss = None
    final_loss = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_losses(batch, t=timestep, noise=noise)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(losses["loss"].detach())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        print(f"step={step} loss={value:.6f}", flush=True)

    assert initial_loss is not None and final_loss is not None
    if final_loss >= initial_loss:
        raise RuntimeError(
            f"tiny overfit failed: loss changed from {initial_loss:.6f} to {final_loss:.6f}"
        )
    print(
        {
            "status": "ok",
            "sample_id": batch["metadata"]["sample_id"][0],
            "steps": args.steps,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        }
    )


if __name__ == "__main__":
    main()
