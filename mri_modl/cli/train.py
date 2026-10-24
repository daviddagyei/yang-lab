"""Train the 4x multicoil MoDL baseline on the supplied-map subset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Subset

from mri_data import MRISliceDataset, build_dataloader
from mri_modl import MoDLTrainer, build_modl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/data/david_agyei/nyu-dataset/derived/modl_4x_fp32")


def repeat_loader(loader: torch.utils.data.DataLoader) -> Iterator[dict]:
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
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=1898)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unrolls", type=int, default=10)
    parser.add_argument("--cg-iterations", type=int, default=10)
    parser.add_argument("--denoiser-layers", type=int, default=5)
    parser.add_argument("--denoiser-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--mask-seed", type=int, default=20261019)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--allow-unroll-change",
        action="store_true",
        help="Allow a one-unroll checkpoint to initialize the shared ten-unroll model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.steps <= 0 or args.save_every <= 0:
        raise ValueError("steps and save-every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("full MoDL training requires the CUDA GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = MRISliceDataset(args.manifest, split="train", mask_seed=args.mask_seed)
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(dataset):
            raise ValueError("max-samples must be between 1 and the dataset size")
        dataset = Subset(dataset, range(args.max_samples))
    loader = build_dataloader(
        dataset,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = build_modl(
        denoiser_channels=args.denoiser_channels,
        denoiser_layers=args.denoiser_layers,
        unrolls=args.unrolls,
        cg_iterations=args.cg_iterations,
    )
    trainer = MoDLTrainer(model, learning_rate=args.learning_rate, device="cuda")
    if args.resume:
        trainer.load(args.resume, allow_unroll_change=args.allow_unroll_change)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batches = repeat_loader(loader)
    final_step = trainer.step + args.steps
    print(
        f"device={trainer.device} start_step={trainer.step} final_step={final_step} "
        f"unrolls={args.unrolls} cg_iterations={args.cg_iterations}",
        flush=True,
    )
    while trainer.step < final_step:
        batch = next(batches)
        metadata = batch.get("metadata", {})
        sample_value = metadata.get("sample_id", "unknown")
        current_sample = sample_value[0] if isinstance(sample_value, (list, tuple)) else sample_value
        metrics = trainer.train_step(batch)
        print(
            f"step={trainer.step} sample={current_sample} loss={metrics['loss']:.7f} "
            f"lambda={metrics['lambda']:.7f}",
            flush=True,
        )
        if trainer.step % args.save_every == 0 or trainer.step == final_step:
            trainer.save(args.output_dir / f"checkpoint-{trainer.step:07d}.pt")


if __name__ == "__main__":
    main()
