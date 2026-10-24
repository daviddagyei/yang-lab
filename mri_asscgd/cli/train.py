"""Train the 4x target-free adversarial diffusion reconstruction model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mri_asscgd import (
    AdversarialASSCGDTrainer,
    ConditionalDiffusionDiscriminator,
    SelfSupervisedMRISliceDataset,
    build_asscgd_diffusion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/data/david_agyei/nyu-dataset/derived/asscgd_4x_stable")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "phase1" / "manifests" / "train.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--discriminator-learning-rate", type=float, default=2e-5)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--gradient-penalty-weight", type=float, default=1.0)
    parser.add_argument("--discriminator-update-interval", type=int, default=4)
    parser.add_argument("--real-label-smoothing", type=float, default=0.1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sampling-timesteps", type=int, default=10)
    parser.add_argument("--discriminator-channels", type=int, default=32)
    parser.add_argument("--input-fraction", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--mask-seed", type=int, default=20261019)
    parser.add_argument("--partition-seed", type=int, default=20261020)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-discriminator",
        action="store_true",
        help="Resume generator/optimizer progress but initialize a new discriminator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.epochs <= 0 or args.save_every_epochs <= 0 or args.log_every <= 0:
        raise ValueError("epochs, save-every-epochs, and log-every must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("full ASSCGD-inspired training requires the CUDA GPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = SelfSupervisedMRISliceDataset(
        args.manifest,
        split="train",
        mask_seed=args.mask_seed,
        partition_seed=args.partition_seed,
        input_fraction=args.input_fraction,
        dynamic_partitions=True,
    )
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
        learning_rate=args.learning_rate,
        discriminator_learning_rate=args.discriminator_learning_rate,
        adversarial_weight=args.adversarial_weight,
        gradient_penalty_weight=args.gradient_penalty_weight,
        discriminator_update_interval=args.discriminator_update_interval,
        real_label_smoothing=args.real_label_smoothing,
        adaptive_adversarial_weight=True,
        device="cuda",
        precision=args.precision,
    )
    if args.resume:
        trainer.load(args.resume, reset_discriminator=args.reset_discriminator)
    if trainer.completed_epochs >= args.epochs:
        print(
            f"training already reached {trainer.completed_epochs} epochs "
            f"({trainer.step} steps)",
            flush=True,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={torch.cuda.get_device_name(0)} precision={args.precision} "
        f"volumes={dataset.volume_count} slices={len(dataset)} "
        f"start_epoch={trainer.completed_epochs + 1} target_epochs={args.epochs}",
        flush=True,
    )
    for epoch_index in range(trainer.completed_epochs, args.epochs):
        dataset.set_epoch(epoch_index)
        shuffle_generator = torch.Generator().manual_seed(args.seed + epoch_index)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
            generator=shuffle_generator,
        )
        totals = {
            "loss": 0.0,
            "heldout_kspace": 0.0,
            "generator_adversarial": 0.0,
            "discriminator": 0.0,
        }
        for epoch_step, batch in enumerate(loader, start=1):
            metrics = trainer.train_step(batch)
            for key in totals:
                totals[key] += metrics[key]
            if epoch_step == 1 or epoch_step % args.log_every == 0:
                sample_id = str(batch["metadata"]["sample_id"][0])
                print(
                    f"epoch={epoch_index + 1}/{args.epochs} "
                    f"epoch_step={epoch_step}/{len(dataset)} step={trainer.step} "
                    f"sample={sample_id} loss={metrics['loss']:.6f} "
                    f"heldout={metrics['heldout_kspace']:.6f} "
                    f"g_adv={metrics['generator_adversarial']:.6f} "
                    f"g_weight={metrics['effective_adversarial_weight']:.6f} "
                    f"d={metrics['discriminator']:.6f} "
                    f"d_update={int(metrics['discriminator_updated'])}",
                    flush=True,
                )
        trainer.completed_epochs = epoch_index + 1
        means = {key: value / len(dataset) for key, value in totals.items()}
        print(
            f"epoch_complete={trainer.completed_epochs}/{args.epochs} step={trainer.step} "
            f"mean_loss={means['loss']:.6f} mean_heldout={means['heldout_kspace']:.6f} "
            f"mean_g_adv={means['generator_adversarial']:.6f} "
            f"mean_d={means['discriminator']:.6f}",
            flush=True,
        )
        if means["discriminator"] < 0.2:
            raise RuntimeError(
                "discriminator collapse detected (mean loss < 0.2); "
                "the unstable epoch was not checkpointed"
            )
        if (
            trainer.completed_epochs % args.save_every_epochs == 0
            or trainer.completed_epochs == args.epochs
        ):
            checkpoint = args.output_dir / (
                f"checkpoint-epoch-{trainer.completed_epochs:03d}-step-{trainer.step:07d}.pt"
            )
            trainer.save(checkpoint)
            print(f"saved={checkpoint}", flush=True)
        del loader


if __name__ == "__main__":
    main()
