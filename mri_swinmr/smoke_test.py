"""Run one real-data update through the official-size SwinMR backbone."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from mri_data import MRISliceDataset, build_dataloader

from .model import OFFICIAL_COMMIT, build_swinmr
from .training import SwinMRTrainer, random_crop_and_augment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "phase1/manifests/train.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=96)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the real-data SwinMR smoke test requires CUDA")
    dataset = MRISliceDataset(args.manifest, split="train", acceleration=4)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    raw_batch = next(iter(build_dataloader(Subset(dataset, [args.sample_index]))))
    batch = random_crop_and_augment(
        raw_batch,
        crop_size=args.crop_size,
        generator=torch.Generator().manual_seed(20261025),
        augment=False,
    )
    model = build_swinmr(image_size=args.crop_size, perceptual_weight=0.0)
    trainer = SwinMRTrainer(model, device="cuda", load_perceptual=False)
    torch.cuda.reset_peak_memory_stats(trainer.device)
    torch.cuda.synchronize(trainer.device)
    started = time.perf_counter()
    metrics = trainer.train_step(batch)
    torch.cuda.synchronize(trainer.device)
    print(json.dumps({
        "status": "ok",
        "architecture": "official SwinMR backbone, complex two-channel adaptation",
        "official_commit": OFFICIAL_COMMIT,
        "sample_id": str(raw_batch["metadata"]["sample_id"][0]),
        "crop_size": args.crop_size,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": metrics,
        "seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(trainer.device) / 1024**3,
        "perceptual_in_smoke_gate": False,
    }, indent=2))


if __name__ == "__main__":
    main()
