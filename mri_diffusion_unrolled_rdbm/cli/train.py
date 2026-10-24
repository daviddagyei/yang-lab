"""Train the full ten-stage diffusion-unrolled MRI RDBM from scratch."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mri_data import MRISliceDataset
from mri_diffusion_unrolled_rdbm import (
    DiffusionUnrolledRDBMTrainer,
    build_diffusion_unrolled_rdbm,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/data/david_agyei/nyu-dataset")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "phase1/manifests/train.jsonl"
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--acceleration", type=int, choices=(2, 4, 6, 8), default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--cg-iterations", type=int, default=6)
    parser.add_argument("--initial-lambda", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--mask-seed", type=int, default=20261019)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if min(
        args.epochs,
        args.save_every_epochs,
        args.log_every,
        args.cg_iterations,
    ) <= 0:
        raise ValueError("training, logging, and CG values must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("full diffusion-unrolled training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_root = args.data_root or Path(
        os.environ.get("MRI_DATA_ROOT", str(DEFAULT_DATA_ROOT))
    )
    output_dir = args.output_dir or (
        data_root / "derived" / f"full_diffusion_unrolled_rdbm_{args.acceleration}x_scratch"
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("model_name") != "full_diffusion_unrolled_rdbm":
            raise ValueError("resume checkpoint is not a full diffusion-unrolled RDBM")
        model_config = dict(checkpoint["model_config"])
        if args.disable_gradient_checkpointing:
            model_config["gradient_checkpointing"] = False
        model = build_diffusion_unrolled_rdbm(**model_config)
    else:
        model = build_diffusion_unrolled_rdbm(
            cg_iterations=args.cg_iterations,
            initial_lambda=args.initial_lambda,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
        )

    dataset = MRISliceDataset(
        args.manifest,
        split="train",
        mask_seed=args.mask_seed,
        acceleration=args.acceleration,
        data_root=data_root,
    )
    data_config = {
        "acceleration": args.acceleration,
        "center_fraction": dataset.mask_config.resolved_center_fraction,
        "mask_seed": args.mask_seed,
        "normalization_quantile": dataset.normalization_quantile,
    }
    trainer = DiffusionUnrolledRDBMTrainer(
        model,
        learning_rate=args.learning_rate,
        device="cuda",
        precision=args.precision,
        data_config=data_config,
    )
    if args.resume:
        trainer.load(args.resume)
    if trainer.completed_epochs >= args.epochs:
        print(
            f"training already reached {trainer.completed_epochs} epochs ({trainer.step} steps)",
            flush=True,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={torch.cuda.get_device_name(0)} precision={args.precision} "
        f"initialization=scratch acceleration={args.acceleration}x slices={len(dataset)} "
        f"unroll_steps={model.unroll_steps} schedule={model.time_pairs.tolist()} "
        f"cg_iterations={model.cg_iterations} shared_parameters="
        f"{sum(parameter.numel() for parameter in model.rdbm.parameters())} "
        f"start_epoch={trainer.completed_epochs + 1} target_epochs={args.epochs}",
        flush=True,
    )
    for epoch_index in range(trainer.completed_epochs, args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch_index)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
            generator=generator,
        )
        total_keys = (
            "loss",
            "complex_l1",
            "magnitude_l1",
            "ssim_loss",
            "measurement",
            "intermediate_l1",
        )
        totals = {key: 0.0 for key in total_keys}
        for epoch_step, batch in enumerate(loader, start=1):
            metrics = trainer.train_step(batch)
            for key in total_keys:
                totals[key] += metrics[key]
            if epoch_step == 1 or epoch_step % args.log_every == 0:
                print(
                    f"epoch={epoch_index + 1}/{args.epochs} "
                    f"epoch_step={epoch_step}/{len(dataset)} step={trainer.step} "
                    f"sample={batch['metadata']['sample_id'][0]} "
                    f"loss={metrics['loss']:.6f} complex={metrics['complex_l1']:.6f} "
                    f"ssim_loss={metrics['ssim_loss']:.6f} "
                    f"measurement={metrics['measurement']:.6f} "
                    f"lambda={metrics['lambda_mean']:.6f} "
                    f"gradient_norm={metrics['gradient_norm']:.6f}",
                    flush=True,
                )
        trainer.completed_epochs = epoch_index + 1
        means = {key: value / len(dataset) for key, value in totals.items()}
        print(
            f"epoch_complete={trainer.completed_epochs}/{args.epochs} step={trainer.step} "
            + " ".join(f"mean_{key}={value:.6f}" for key, value in means.items()),
            flush=True,
        )
        if (
            trainer.completed_epochs % args.save_every_epochs == 0
            or trainer.completed_epochs == args.epochs
        ):
            checkpoint_path = output_dir / (
                f"checkpoint-epoch-{trainer.completed_epochs:03d}-step-{trainer.step:07d}.pt"
            )
            trainer.save(checkpoint_path)
            print(f"saved={checkpoint_path}", flush=True)
        del loader


if __name__ == "__main__":
    main()
