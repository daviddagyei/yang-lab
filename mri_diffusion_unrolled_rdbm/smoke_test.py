"""Run one real-data update through the full ten-stage model from scratch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mri_data import MRISliceDataset, build_dataloader
from .model import build_diffusion_unrolled_rdbm
from .training import DiffusionUnrolledRDBMTrainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--acceleration", type=int, choices=(2, 4, 6, 8), default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-size smoke test requires CUDA")
    torch.manual_seed(20261021)
    model = build_diffusion_unrolled_rdbm(
        cg_iterations=6,
        gradient_checkpointing=False,
    )
    trainer = DiffusionUnrolledRDBMTrainer(
        model, learning_rate=2e-5, device="cuda", precision="bf16"
    )
    dataset = MRISliceDataset(
        ROOT / "phase1/manifests/train.jsonl",
        split="train",
        acceleration=args.acceleration,
        data_root=args.data_root,
    )
    batch = next(iter(build_dataloader(dataset)))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = trainer.train_step(batch)
    torch.cuda.synchronize()
    print(json.dumps({
        "status": "ok",
        "architecture": "full diffusion-faithful unrolled RDBM",
        "initialization": "scratch",
        "sample_id": batch["metadata"]["sample_id"][0],
        "acceleration": args.acceleration,
        "unroll_steps": model.unroll_steps,
        "schedule": model.time_pairs.tolist(),
        "cg_iterations_per_stage": model.cg_iterations,
        "gradient_checkpointing": model.gradient_checkpointing,
        "metrics": metrics,
        "seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }, indent=2))


if __name__ == "__main__":
    main()
