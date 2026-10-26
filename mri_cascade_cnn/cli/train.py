"""Train the adapted multicoil D5-C5 model for 50 passes."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from mri_cascade_cnn import CascadeCNNTrainer, build_cascade_cnn
from mri_data import MRISliceDataset, build_grouped_dataloader, grouped_batch_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/data/david_agyei/nyu-dataset/derived/multicoil_cascade_cnn_4x")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "phase1/manifests/train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--acceleration", type=int, choices=(2, 4, 6, 8), default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--mask-seed", type=int, default=20261019)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if min(args.epochs, args.batch_size, args.save_every_epochs, args.log_every) <= 0:
        raise ValueError("training values must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("full Cascade CNN training requires CUDA")
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
    }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("model_name") != "multicoil_cascade_cnn":
            raise ValueError("resume checkpoint is not a Cascade CNN")
        model = build_cascade_cnn(**checkpoint["model_config"])
    else:
        model = build_cascade_cnn(
            gradient_checkpointing=not args.disable_gradient_checkpointing
        )
    trainer = CascadeCNNTrainer(
        model,
        learning_rate=args.learning_rate,
        device="cuda",
        data_config=data_config,
    )
    if args.resume:
        trainer.load(args.resume)
        if args.disable_gradient_checkpointing:
            trainer.model.gradient_checkpointing = False
            trainer.model.model_config["gradient_checkpointing"] = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batches_per_epoch = len(
        grouped_batch_schedule(dataset, batch_size=args.batch_size, seed=args.seed)
    )
    print(
        f"device={torch.cuda.get_device_name(0)} precision=fp32 "
        f"acceleration={args.acceleration}x slices={len(dataset)} "
        f"batch_size={args.batch_size} batches_per_epoch={batches_per_epoch} "
        f"epochs={args.epochs} cascades={model.cascades} depth={model.depth} "
        f"features={model.features} cg_iterations={model.cg_iterations} "
        f"dc_weight={model.dc_weight} gradient_checkpointing={model.gradient_checkpointing} "
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
        totals = {"loss": 0.0, "complex_mse": 0.0, "l2_regularization": 0.0}
        for epoch_step, batch in enumerate(loader, start=1):
            metrics = trainer.train_step(batch)
            for name in totals:
                totals[name] += metrics[name]
            if epoch_step == 1 or epoch_step % args.log_every == 0:
                print(
                    f"epoch={epoch_index + 1}/{args.epochs} "
                    f"epoch_step={epoch_step}/{len(loader)} step={trainer.step} "
                    f"sample={batch['metadata']['sample_id'][0]} "
                    f"loss={metrics['loss']:.7f} complex_mse={metrics['complex_mse']:.7f} "
                    f"l2={metrics['l2_regularization']:.7f} "
                    f"gradient_norm={metrics['gradient_norm']:.5f}",
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
            path = args.output_dir / (
                f"checkpoint-epoch-{trainer.completed_epochs:03d}-step-{trainer.step:07d}.pt"
            )
            trainer.save(path)
            print(f"saved={path}", flush=True)


if __name__ == "__main__":
    main()
