"""Train adapted complex SwinMR-PI for matched dataset passes."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from mri_data import MRISliceDataset, build_grouped_dataloader, grouped_batch_schedule
from mri_swinmr import OFFICIAL_COMMIT, SwinMRTrainer, build_swinmr
from mri_swinmr.training import random_crop_and_augment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/data/david_agyei/nyu-dataset/derived")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "phase1/manifests/train.jsonl")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--acceleration", type=int, choices=(2, 4, 6, 8), default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--perceptual-weight", type=float, default=0.0025)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--mask-seed", type=int, default=20261019)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if min(args.epochs, args.batch_size, args.crop_size, args.save_every_epochs, args.log_every) <= 0:
        raise ValueError("training values must be positive")
    if args.num_workers < 0 or args.perceptual_weight < 0:
        raise ValueError("invalid worker count or perceptual weight")
    if not torch.cuda.is_available():
        raise RuntimeError("full SwinMR training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_dataset = MRISliceDataset(
        args.manifest,
        split="train",
        normalization_quantile=0.99,
        mask_seed=args.mask_seed,
        acceleration=args.acceleration,
        data_root=args.data_root,
    )
    dataset: torch.utils.data.Dataset = base_dataset
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(base_dataset):
            raise ValueError("max-samples must be within the dataset")
        dataset = Subset(base_dataset, range(args.max_samples))
    data_config = {
        "acceleration": args.acceleration,
        "center_fraction": base_dataset.mask_config.resolved_center_fraction,
        "mask_seed": args.mask_seed,
        "normalization_quantile": 0.99,
        "training_crop_size": args.crop_size,
        "augmentation": not args.disable_augmentation,
    }
    batches_per_epoch = len(
        grouped_batch_schedule(dataset, batch_size=args.batch_size, seed=args.seed)
    )
    total_steps = batches_per_epoch * args.epochs
    milestones = tuple(
        sorted({max(1, round(total_steps * fraction)) for fraction in (0.5, 0.7, 0.9)})
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("model_name") != "adapted_complex_swinmr_pi":
            raise ValueError("resume checkpoint is not SwinMR-PI")
        model = build_swinmr(**checkpoint["model_config"])
        milestones = tuple(checkpoint["scheduler_milestones"])
    else:
        model = build_swinmr(
            image_size=args.crop_size,
            perceptual_weight=args.perceptual_weight,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    trainer = SwinMRTrainer(
        model,
        learning_rate=args.learning_rate,
        ema_decay=args.ema_decay,
        scheduler_milestones=milestones,
        scheduler_gamma=0.5,
        device="cuda",
        data_config=data_config,
    )
    if args.resume:
        trainer.load(args.resume)

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"swinmr_pi_complex_{args.acceleration}x"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={torch.cuda.get_device_name(0)} precision=fp32 slices={len(dataset)} "
        f"acceleration={args.acceleration}x batch_size={args.batch_size} "
        f"batches_per_epoch={batches_per_epoch} epochs={args.epochs} "
        f"crop={model.image_size} embed_dim={model.embed_dim} depths={model.depths} "
        f"heads={model.num_heads} window={model.window_size} parameters="
        f"{sum(parameter.numel() for parameter in model.parameters())} "
        f"official_commit={OFFICIAL_COMMIT} lr_milestones={milestones} "
        f"start_epoch={trainer.completed_epochs} start_step={trainer.step}",
        flush=True,
    )
    if trainer.completed_epochs >= args.epochs:
        print("training is already complete", flush=True)
        return
    for epoch_index in range(trainer.completed_epochs, args.epochs):
        schedule = grouped_batch_schedule(
            dataset, batch_size=args.batch_size, seed=args.seed + epoch_index
        )
        loader = build_grouped_dataloader(
            dataset, schedule, num_workers=args.num_workers, pin_memory=True
        )
        crop_generator = torch.Generator().manual_seed(args.seed * 10_000 + epoch_index)
        totals = {
            "loss": 0.0,
            "coil_image_charbonnier": 0.0,
            "coil_kspace_charbonnier": 0.0,
            "perceptual": 0.0,
        }
        for epoch_step, raw_batch in enumerate(loader, start=1):
            batch = random_crop_and_augment(
                raw_batch,
                crop_size=args.crop_size,
                generator=crop_generator,
                augment=not args.disable_augmentation,
            )
            metrics = trainer.train_step(batch)
            for name in totals:
                totals[name] += metrics[name]
            if epoch_step == 1 or epoch_step % args.log_every == 0:
                print(
                    f"epoch={epoch_index + 1}/{args.epochs} epoch_step={epoch_step}/{len(loader)} "
                    f"step={trainer.step} sample={raw_batch['metadata']['sample_id'][0]} "
                    f"loss={metrics['loss']:.7f} image={metrics['coil_image_charbonnier']:.7f} "
                    f"frequency={metrics['coil_kspace_charbonnier']:.7f} "
                    f"perceptual={metrics['perceptual']:.7f} "
                    f"gradient_norm={metrics['gradient_norm']:.5f} lr={metrics['learning_rate']:.8f}",
                    flush=True,
                )
        trainer.completed_epochs = epoch_index + 1
        means = {name: value / len(loader) for name, value in totals.items()}
        print(
            f"epoch_complete={trainer.completed_epochs}/{args.epochs} step={trainer.step} "
            + " ".join(f"mean_{key}={value:.8f}" for key, value in means.items()),
            flush=True,
        )
        if trainer.completed_epochs % args.save_every_epochs == 0 or trainer.completed_epochs == args.epochs:
            path = output_dir / (
                f"checkpoint-epoch-{trainer.completed_epochs:03d}-step-{trainer.step:07d}.pt"
            )
            trainer.save(path)
            print(f"saved={path}", flush=True)


if __name__ == "__main__":
    main()
