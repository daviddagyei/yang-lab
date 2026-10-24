"""Run short real-data adversarial diffusion updates and report GPU use."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from mri_data import build_dataloader

from .adversarial import ConditionalDiffusionDiscriminator
from .dataset import SelfSupervisedMRISliceDataset
from .model import build_asscgd_diffusion
from .training import AdversarialASSCGDTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "phase1" / "manifests" / "train.jsonl",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--discriminator-channels", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--sampling-timesteps", type=int, default=5)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the adversarial smoke test requires the CUDA GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = SelfSupervisedMRISliceDataset(args.manifest, split="train")
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    batch = next(iter(build_dataloader(Subset(dataset, [args.sample_index]))))
    model = build_asscgd_diffusion(
        dim=args.dim,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
    )
    discriminator = ConditionalDiffusionDiscriminator(
        base_channels=args.discriminator_channels
    )
    trainer = AdversarialASSCGDTrainer(
        model,
        discriminator,
        device="cuda",
        precision=args.precision,
    )

    torch.cuda.reset_peak_memory_stats(trainer.device)
    torch.cuda.synchronize(trainer.device)
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        metrics = trainer.train_step(batch)
        history.append(metrics)
        print(
            f"step={step} loss={metrics['loss']:.6f} "
            f"heldout={metrics['heldout_kspace']:.6f} "
            f"generator_adv={metrics['generator_adversarial']:.6f} "
            f"discriminator={metrics['discriminator']:.6f}",
            flush=True,
        )
    torch.cuda.synchronize(trainer.device)
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_id": str(batch["metadata"]["sample_id"][0]),
                "steps": args.steps,
                "dim": args.dim,
                "discriminator_channels": args.discriminator_channels,
                "precision": args.precision,
                "final_metrics": history[-1],
                "seconds": time.perf_counter() - started,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(trainer.device) / 1024**3,
                "supervised_target_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
