"""Benchmark the production AdaDiff training step before a long run."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from .diffusion import DiffusionSchedule
from .model import build_adadiff_discriminator, build_adadiff_generator, fir_backend
from .training import AdaDiffTrainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--train-slices", type=int, default=1898)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--lazy-regularization", type=int, default=10)
    parser.add_argument("--channels-last", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("AdaDiff training benchmark requires CUDA")
    if min(
        args.batch_size,
        args.steps,
        args.train_slices,
        args.epochs,
        args.lazy_regularization,
    ) <= 0:
        raise ValueError("all benchmark arguments must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    generator = build_adadiff_generator()
    discriminator = build_adadiff_discriminator()
    if args.channels_last:
        generator = generator.to(memory_format=torch.channels_last)
        discriminator = discriminator.to(memory_format=torch.channels_last)
    trainer = AdaDiffTrainer(
        generator,
        discriminator,
        schedule=DiffusionSchedule(timesteps=8, beta_min=0.1, beta_max=20.0),
        total_epochs=args.epochs,
        lazy_regularization=args.lazy_regularization,
        data_config={"benchmark": True},
        device=device,
    )
    # Start at one so an 11-step benchmark contains one representative lazy-R1
    # update rather than putting the expensive update in the CUDA warm-up step.
    trainer.step = 1
    image = torch.randn(
        args.batch_size, 2, 320, 320, device=device, dtype=torch.float32
    ).clamp_(-1.0, 1.0)
    if args.channels_last:
        image = image.contiguous(memory_format=torch.channels_last)
    batch = {"image": image}

    torch.cuda.reset_peak_memory_stats(device)
    trainer.train_step(batch)
    torch.cuda.synchronize(device)
    durations: list[float] = []
    last_metrics: dict[str, float] = {}
    for _ in range(args.steps):
        started = time.perf_counter()
        last_metrics = trainer.train_step(batch)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)

    mean_seconds = sum(durations) / len(durations)
    steps_per_epoch = math.ceil(args.train_slices / args.batch_size)
    projected_hours = mean_seconds * steps_per_epoch * args.epochs / 3600.0
    result = {
        "status": "ok",
        "device": torch.cuda.get_device_name(device),
        "fir_backend": fir_backend(),
        "channels_last": args.channels_last,
        "lazy_regularization": args.lazy_regularization,
        "batch_size": args.batch_size,
        "timed_steps": args.steps,
        "step_seconds": durations,
        "mean_step_seconds": mean_seconds,
        "steps_per_epoch": steps_per_epoch,
        "projected_500_epoch_hours_single_gpu": projected_hours,
        "projected_500_epoch_days_single_gpu": projected_hours / 24.0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "last_metrics": last_metrics,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
