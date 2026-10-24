"""Verify that the official-size SwinMR backbone can overfit one MRI crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from mri_data import MRISliceDataset, build_dataloader

from .model import build_swinmr
from .training import SwinMRTrainer, random_crop_and_augment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "phase1/manifests/train.jsonl")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.steps < 10:
        raise ValueError("steps must be at least 10")
    if not torch.cuda.is_available():
        raise RuntimeError("the real-data SwinMR overfit test requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = MRISliceDataset(args.manifest, split="train", acceleration=4)
    raw_batch = next(iter(build_dataloader(Subset(dataset, [args.sample_index]))))
    batch = random_crop_and_augment(
        raw_batch,
        crop_size=args.crop_size,
        generator=torch.Generator().manual_seed(args.seed),
        augment=False,
    )
    trainer = SwinMRTrainer(
        build_swinmr(image_size=args.crop_size, perceptual_weight=0.0),
        device="cuda",
        load_perceptual=False,
    )
    losses = []
    for step in range(1, args.steps + 1):
        metrics = trainer.train_step(batch)
        losses.append(metrics["loss"])
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(f"step={step} loss={metrics['loss']:.8f}", flush=True)
    window = min(5, args.steps // 2)
    initial_mean = float(np.mean(losses[:window]))
    final_mean = float(np.mean(losses[-window:]))
    if not final_mean < initial_mean:
        raise RuntimeError(f"overfit gate failed: {initial_mean:.8f} -> {final_mean:.8f}")
    print(json.dumps({
        "status": "ok",
        "sample_id": str(raw_batch["metadata"]["sample_id"][0]),
        "steps": args.steps,
        "initial_window_loss": initial_mean,
        "final_window_loss": final_mean,
        "relative_reduction": (initial_mean - final_mean) / initial_mean,
    }, indent=2))


if __name__ == "__main__":
    main()
