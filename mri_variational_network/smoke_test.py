"""Run one full official-configuration Variational Network training step."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from mri_data import MRISliceDataset, build_dataloader

from .model import build_variational_network
from .training import VariationalNetworkTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "phase1" / "manifests" / "train.jsonl",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the real-data Variational Network smoke test requires CUDA")

    dataset = MRISliceDataset(
        args.manifest,
        split="train",
        normalization_quantile=1.0,
        acceleration=4,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    batch = next(iter(build_dataloader(Subset(dataset, [args.sample_index]))))
    model = build_variational_network()
    trainer = VariationalNetworkTrainer(
        model, optimizer="ipalm", device="cuda"
    )
    torch.cuda.reset_peak_memory_stats(trainer.device)
    torch.cuda.synchronize(trainer.device)
    started = time.perf_counter()
    metrics = trainer.train_step(batch)
    torch.cuda.synchronize(trainer.device)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "status": "ok",
                "architecture": "Hammernik et al. Variational Network",
                "optimizer": trainer.optimizer_name,
                "sample_id": str(batch["metadata"]["sample_id"][0]),
                "stages": model.num_stages,
                "filters": model.num_filters,
                "kernel_size": model.kernel_size,
                "rbf_weights": model.num_rbf_weights,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "metrics": metrics,
                "seconds": elapsed,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(trainer.device) / 1024**3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
