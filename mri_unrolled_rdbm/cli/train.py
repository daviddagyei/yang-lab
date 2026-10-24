"""Train a 4x three-cascade end-to-end unrolled RDBM."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mri_data import MRISliceDataset
from mri_unrolled_rdbm import UnrolledRDBMTrainer, build_unrolled_rdbm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RDBM = Path(
    "/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16/checkpoint-0094900.pt"
)
DEFAULT_OUTPUT = Path("/data/david_agyei/nyu-dataset/derived/unrolled_rdbm_4x_scratch")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "phase1" / "manifests" / "train.jsonl"
    )
    parser.add_argument("--rdbm-checkpoint", type=Path, default=DEFAULT_RDBM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--cascades", type=int, default=3)
    parser.add_argument("--cg-iterations", type=int, default=6)
    parser.add_argument("--initial-lambda", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--mask-seed", type=int, default=20261019)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Initialize from the standard 94,900-step RDBM instead of random weights.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Keep cascade activations instead of recomputing them during backward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if min(args.epochs, args.save_every_epochs, args.log_every) <= 0:
        raise ValueError("epochs and logging/checkpoint intervals must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("full unrolled RDBM training requires the CUDA GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume_checkpoint.get("model_name") != "unrolled_rdbm":
            raise ValueError("resume checkpoint is not an unrolled RDBM")
        model = build_unrolled_rdbm(**resume_checkpoint["model_config"])
        pretrained_step = int(resume_checkpoint.get("pretrained_rdbm_step", 0))
    else:
        base_checkpoint = torch.load(args.rdbm_checkpoint, map_location="cpu", weights_only=True)
        if int(base_checkpoint.get("data_config", {}).get("acceleration", 4)) != 4:
            raise ValueError("the warm-start RDBM checkpoint must use 4x acceleration")
        model = build_unrolled_rdbm(
            rdbm_config=base_checkpoint["model_config"],
            cascades=args.cascades,
            cg_iterations=args.cg_iterations,
            initial_lambda=args.initial_lambda,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
        )
        if args.warm_start:
            model.rdbm.load_state_dict(base_checkpoint["model"])
            pretrained_step = int(base_checkpoint.get("step", 0))
        else:
            pretrained_step = 0
        del base_checkpoint

    dataset = MRISliceDataset(
        args.manifest,
        split="train",
        mask_seed=args.mask_seed,
        acceleration=4,
    )
    data_config = {
        "acceleration": 4,
        "center_fraction": dataset.mask_config.resolved_center_fraction,
        "mask_seed": args.mask_seed,
    }
    trainer = UnrolledRDBMTrainer(
        model,
        learning_rate=args.learning_rate,
        device="cuda",
        precision=args.precision,
        data_config=data_config,
        pretrained_rdbm_step=pretrained_step,
    )
    if args.resume:
        trainer.load(args.resume)
    if trainer.completed_epochs >= args.epochs:
        print(
            f"training already reached {trainer.completed_epochs} epochs ({trainer.step} steps)",
            flush=True,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={torch.cuda.get_device_name(0)} precision={args.precision} "
        f"slices={len(dataset)} "
        f"initialization={'warm-start' if trainer.pretrained_rdbm_step else 'scratch'} "
        f"warm_start_step={trainer.pretrained_rdbm_step} "
        f"cascades={model.cascades} cg_iterations={model.cg_iterations} "
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
                    f"lambda={metrics['lambda_mean']:.6f}",
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
            checkpoint = args.output_dir / (
                f"checkpoint-epoch-{trainer.completed_epochs:03d}-step-{trainer.step:07d}.pt"
            )
            trainer.save(checkpoint)
            print(f"saved={checkpoint}", flush=True)
        del loader


if __name__ == "__main__":
    main()
