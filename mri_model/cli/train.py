"""Train the physics-guided RDBM on the supplied-map MRI subset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Subset

from mri_data import MRISliceDataset, build_dataloader
from mri_model import MRITrainer, build_mri_rdbm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/data/david_agyei/nyu-dataset/derived/rdbm")


def repeat_loader(loader: torch.utils.data.DataLoader) -> Iterator[dict]:
    """Repeat epochs without retaining every large MRI batch in memory."""
    while True:
        yield from loader


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "phase1" / "manifests" / "train.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sampling-timesteps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--mask-seed", type=int, default=20261019)
    parser.add_argument("--acceleration", type=int, choices=(2, 4, 6, 8), default=4)
    parser.add_argument("--center-fraction", type=float)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
        help="Training precision. BF16 is preferred on supported CUDA GPUs.",
    )
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.steps <= 0 or args.save_every <= 0:
        raise ValueError("steps and save-every must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = MRISliceDataset(
        args.manifest,
        split="train",
        mask_seed=args.mask_seed,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
    )
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(dataset):
            raise ValueError("max-samples must be between 1 and the dataset size")
        dataset = Subset(dataset, range(args.max_samples))
    loader = build_dataloader(
        dataset,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_mri_rdbm(
        dim=args.dim,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
    )
    trainer = MRITrainer(
        model,
        learning_rate=args.learning_rate,
        precision=args.precision,
        data_config={
            "acceleration": args.acceleration,
            "center_fraction": dataset.mask_config.resolved_center_fraction,
            "mask_seed": args.mask_seed,
        },
    )
    if args.resume:
        trainer.load(args.resume)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"acceleration={args.acceleration}x "
        f"center_fraction={dataset.mask_config.resolved_center_fraction:.8f} "
        f"start_step={trainer.step} additional_steps={args.steps}",
        flush=True,
    )
    batches = repeat_loader(loader)
    final_step = trainer.step + args.steps
    while trainer.step < final_step:
        batch = next(batches)
        metadata = batch.get("metadata", {})
        sample_value = metadata.get("sample_id", "unknown")
        sample_id = sample_value[0] if isinstance(sample_value, (list, tuple)) else sample_value
        losses = trainer.train_step(batch)
        print(
            f"step={trainer.step} sample={sample_id} loss={losses['loss']:.6f} "
            f"image={losses['complex_l1']:.6f} magnitude={losses['magnitude_l1']:.6f} "
            f"measurement={losses['measurement']:.6f}",
            flush=True,
        )
        if trainer.step % args.save_every == 0 or trainer.step == final_step:
            trainer.save(args.output_dir / f"checkpoint-{trainer.step:07d}.pt")


if __name__ == "__main__":
    main()
