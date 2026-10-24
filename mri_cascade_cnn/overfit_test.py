"""Verify that full D5-C5 can overfit one real MRI slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from mri_data import MRISliceDataset, build_dataloader

from .model import build_cascade_cnn
from .training import CascadeCNNTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "phase1/manifests/train.jsonl")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.steps < 10:
        raise ValueError("steps must be at least 10")
    if not torch.cuda.is_available():
        raise RuntimeError("the real-data Cascade CNN overfit test requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = MRISliceDataset(args.manifest, split="train", acceleration=4)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    batch = next(iter(build_dataloader(Subset(dataset, [args.sample_index]))))
    trainer = CascadeCNNTrainer(build_cascade_cnn(), device="cuda")
    losses: list[float] = []
    for step in range(1, args.steps + 1):
        metrics = trainer.train_step(batch)
        losses.append(metrics["complex_mse"])
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(f"step={step} complex_mse={metrics['complex_mse']:.8f}", flush=True)
    window = min(5, args.steps // 2)
    initial_mean = float(np.mean(losses[:window]))
    final_mean = float(np.mean(losses[-window:]))
    if not final_mean < initial_mean:
        raise RuntimeError(f"overfit gate failed: {initial_mean:.8f} -> {final_mean:.8f}")
    print(json.dumps({
        "status": "ok",
        "sample_id": str(batch["metadata"]["sample_id"][0]),
        "steps": args.steps,
        "initial_window_complex_mse": initial_mean,
        "final_window_complex_mse": final_mean,
        "relative_reduction": (initial_mean - final_mean) / initial_mean,
    }, indent=2))


if __name__ == "__main__":
    main()
