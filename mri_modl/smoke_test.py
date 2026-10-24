"""Run one real MoDL forward/backward update without saving a checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from mri_data import MRISliceDataset, build_dataloader

from .model import build_modl
from .training import MoDLTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "phase1" / "manifests" / "train.jsonl",
    )
    parser.add_argument("--unrolls", type=int, default=1)
    parser.add_argument("--cg-iterations", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the real-data MoDL smoke test requires the CUDA GPU")
    dataset = MRISliceDataset(args.manifest, split="train")
    batch = next(iter(build_dataloader(Subset(dataset, [0]))))
    model = build_modl(unrolls=args.unrolls, cg_iterations=args.cg_iterations)
    trainer = MoDLTrainer(model, device="cuda")
    torch.cuda.reset_peak_memory_stats(trainer.device)
    torch.cuda.synchronize(trainer.device)
    started = time.perf_counter()
    metrics = trainer.train_step(batch)
    torch.cuda.synchronize(trainer.device)
    metadata = batch["metadata"]
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_id": str(metadata["sample_id"][0]),
                "unrolls": args.unrolls,
                "cg_iterations": args.cg_iterations,
                "loss": metrics["loss"],
                "lambda": metrics["lambda"],
                "seconds": time.perf_counter() - started,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(trainer.device) / 1024**3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
