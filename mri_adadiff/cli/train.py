"""Train one operator-independent AdaDiff prior for every acceleration rate."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mri_adadiff import (
    AdaDiffPriorDataset,
    AdaDiffTrainer,
    DiffusionSchedule,
    build_adadiff_discriminator,
    build_adadiff_generator,
    prior_collate,
)
from mri_data import MRISliceDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "phase1/manifests/train.jsonl")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every-epochs", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("full AdaDiff training requires CUDA")
    if args.epochs <= 0 or args.batch_size <= 0 or args.save_every_epochs <= 0:
        raise ValueError("invalid training duration or batch settings")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    # Acceleration is used only inside the shared loader before target-only
    # renormalization; no mask, measurement, or initialization reaches AdaDiff.
    source = MRISliceDataset(
        args.manifest,
        split="train",
        acceleration=4,
        data_root=args.data_root,
    )
    dataset = AdaDiffPriorDataset(source)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=prior_collate,
        generator=loader_generator,
    )
    generator = build_adadiff_generator()
    discriminator = build_adadiff_discriminator()
    schedule = DiffusionSchedule(timesteps=8, beta_min=0.1, beta_max=20.0)
    trainer = AdaDiffTrainer(
        generator,
        discriminator,
        schedule=schedule,
        total_epochs=args.epochs,
        data_config={
            "train_volumes": source.volume_count,
            "train_slices": len(source),
            "prior_normalization_quantile": 0.99,
            "operator_independent": True,
        },
        device=device,
    )
    if args.resume:
        trainer.load(args.resume)
    if trainer.completed_epochs > args.epochs:
        raise ValueError("checkpoint already exceeds requested epochs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={torch.cuda.get_device_name(device)} slices={len(dataset)} "
        f"batch_size={args.batch_size} epochs={args.epochs} steps_per_epoch={len(loader)} "
        f"start_epoch={trainer.completed_epochs} start_step={trainer.step} "
        "architecture=official_AdaDiff_NCSNpp timesteps=8 precision=fp32",
        flush=True,
    )
    stop = False
    for epoch in range(trainer.completed_epochs, args.epochs):
        for epoch_step, batch in enumerate(loader, start=1):
            metrics = trainer.train_step(batch)
            if trainer.step == 1 or trainer.step % args.log_every == 0:
                sample_id = batch["metadata"]["sample_id"][0]
                print(
                    f"epoch={epoch + 1}/{args.epochs} epoch_step={epoch_step}/{len(loader)} "
                    f"step={trainer.step} sample={sample_id} "
                    f"g={metrics['generator']:.6f} d={metrics['discriminator']:.6f} "
                    f"real={metrics['discriminator_real']:.6f} "
                    f"fake={metrics['discriminator_fake']:.6f} r1={metrics['r1']:.6f}",
                    flush=True,
                )
            if args.max_steps is not None and trainer.step >= args.max_steps:
                stop = True
                break
        if stop:
            checkpoint = args.output_dir / f"checkpoint-epoch-{epoch + 1:04d}-step-{trainer.step:07d}.pt"
            trainer.save(checkpoint)
            print(f"stopped_at_max_steps checkpoint={checkpoint}", flush=True)
            break
        trainer.complete_epoch()
        if trainer.completed_epochs % args.save_every_epochs == 0 or trainer.completed_epochs == args.epochs:
            checkpoint = args.output_dir / (
                f"checkpoint-epoch-{trainer.completed_epochs:04d}-step-{trainer.step:07d}.pt"
            )
            trainer.save(checkpoint)
            print(f"saved={checkpoint}", flush=True)
    expected_updates = math.floor(len(dataset) / args.batch_size) * args.epochs
    print(
        f"status={'stopped' if stop else 'complete'} step={trainer.step} "
        f"completed_epochs={trainer.completed_epochs} expected_full_updates={expected_updates}",
        flush=True,
    )


if __name__ == "__main__":
    main()

