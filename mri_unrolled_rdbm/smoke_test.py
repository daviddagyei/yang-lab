"""Run one full-size forward/backward step for the warm-started unrolled RDBM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mri_data import MRISliceDataset, build_dataloader

from .model import build_unrolled_rdbm
from .training import UnrolledRDBMTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RDBM = Path(
    "/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16/checkpoint-0094900.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdbm-checkpoint", type=Path, default=DEFAULT_RDBM)
    parser.add_argument("--cascades", type=int, default=3)
    parser.add_argument("--cg-iterations", type=int, default=6)
    parser.add_argument("--warm-start", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-size smoke test requires CUDA")
    checkpoint = torch.load(args.rdbm_checkpoint, map_location="cpu", weights_only=True)
    model = build_unrolled_rdbm(
        rdbm_config=checkpoint["model_config"],
        cascades=args.cascades,
        cg_iterations=args.cg_iterations,
        gradient_checkpointing=True,
    )
    if args.warm_start:
        model.rdbm.load_state_dict(checkpoint["model"])
    pretrained_step = int(checkpoint["step"]) if args.warm_start else 0
    trainer = UnrolledRDBMTrainer(
        model,
        learning_rate=2e-5,
        device="cuda",
        precision="bf16",
        pretrained_rdbm_step=pretrained_step,
    )
    dataset = MRISliceDataset(
        ROOT / "phase1" / "manifests" / "train.jsonl",
        split="train",
        acceleration=4,
    )
    batch = next(iter(build_dataloader(dataset)))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = trainer.train_step(batch)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_id": batch["metadata"]["sample_id"][0],
                "cascades": args.cascades,
                "cg_iterations": args.cg_iterations,
                "initialization": "warm-start" if args.warm_start else "scratch",
                "warm_start_step": pretrained_step,
                "metrics": metrics,
                "seconds": time.perf_counter() - started,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
