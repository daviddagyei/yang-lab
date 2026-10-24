"""Run one real target-free diffusion forward/backward update on the GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from mri_data import build_dataloader

from .dataset import SelfSupervisedMRISliceDataset
from .model import build_asscgd_diffusion
from .training import ASSCGDTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "phase1" / "manifests" / "train.jsonl",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the real-data diffusion smoke test requires the CUDA GPU")

    dataset = SelfSupervisedMRISliceDataset(args.manifest, split="train")
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    batch = next(iter(build_dataloader(Subset(dataset, [args.sample_index]))))
    model = build_asscgd_diffusion(
        dim=args.dim,
        timesteps=args.timesteps,
        sampling_timesteps=min(5, args.timesteps),
    )
    trainer = ASSCGDTrainer(model, device="cuda", precision="fp32")
    torch.cuda.reset_peak_memory_stats(trainer.device)
    torch.cuda.synchronize(trainer.device)
    started = time.perf_counter()
    metrics = trainer.train_step(batch)
    torch.cuda.synchronize(trainer.device)
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_id": str(batch["metadata"]["sample_id"][0]),
                "loss": metrics["loss"],
                "input_residual": metrics["input_residual"],
                "seconds": time.perf_counter() - started,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(trainer.device) / 1024**3,
                "supervised_target_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
