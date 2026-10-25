"""Train the official-architecture Variational Network at one acceleration."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mri_data import MRISliceDataset, collate_padded_coils
from mri_variational_network import (
    VariationalNetworkTrainer,
    build_variational_network,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(
    "/data/david_agyei/nyu-dataset/derived/"
    "hammernik_variational_network_4x_ipalm"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "phase1" / "manifests" / "train.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--acceleration", type=int, choices=(2, 4, 6, 8), default=4
    )
    parser.add_argument(
        "--optimizer",
        choices=("ipalm", "adam"),
        default="ipalm",
        help="iPALM reproduces the official optimizer; Adam is an ablation.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=9700,
        help=(
            "Total iPALM updates; 9700 gives 50 passes over the current "
            "1898-slice training set."
        ),
    )
    parser.add_argument("--save-every-steps", type=int, default=194)
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Used only by the optional Adam ablation.",
    )
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Used only by Adam.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Official configuration: 10; coils are zero-padded per batch.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--mask-seed", type=int, default=20261019)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--ipalm-initial-lipschitz", type=float, default=1000.0)
    parser.add_argument("--ipalm-momentum", type=float, default=0.7)
    parser.add_argument("--ipalm-max-backtracking", type=int, default=40)
    return parser.parse_args()


def batch_group_key(
    dataset: torch.utils.data.Dataset, index: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    if not isinstance(dataset, MRISliceDataset):
        raise TypeError("grouped MRI batching requires MRISliceDataset")
    return dataset.batch_group_key(index)


def grouped_batch_schedule(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    groups: dict[
        tuple[tuple[int, int], tuple[int, int]], list[int]
    ] = {}
    for index in range(len(dataset)):
        groups.setdefault(batch_group_key(dataset, index), []).append(index)
    generator = torch.Generator().manual_seed(seed)
    batches: list[list[int]] = []
    for indices in groups.values():
        order = torch.randperm(
            len(indices), generator=generator
        ).tolist()
        shuffled = [indices[position] for position in order]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            if len(batch) < batch_size:
                missing = batch_size - len(batch)
                repeats = (
                    shuffled
                    * ((missing + len(shuffled) - 1) // len(shuffled))
                )[:missing]
                batch.extend(repeats)
            batches.append(batch)
    batch_order = torch.randperm(
        len(batches), generator=generator
    ).tolist()
    return [batches[position] for position in batch_order]


def build_grouped_loader(
    dataset: torch.utils.data.Dataset,
    batches: list[list[int]],
    *,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=collate_padded_coils,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )


def log_metrics(
    *,
    prefix: str,
    trainer: VariationalNetworkTrainer,
    batch: dict,
    metrics: dict[str, float],
) -> None:
    line = (
        f"{prefix} step={trainer.step} "
        f"sample={batch['metadata']['sample_id'][0]} "
        f"loss={metrics['loss']:.6f} "
        f"magnitude_mse={metrics['magnitude_mse']:.8f} "
        f"relative_rmse={metrics['relative_rmse']:.6f} "
        f"data_weight={metrics['data_weight_mean']:.6f} "
        f"gradient_norm={metrics['gradient_norm']:.6f}"
    )
    if trainer.optimizer_name == "ipalm":
        line += (
            f" L=({metrics['kernels_lipschitz']:.3g},"
            f"{metrics['activation_weights_lipschitz']:.3g},"
            f"{metrics['data_term_weights_lipschitz']:.3g})"
            f" bt=({int(metrics['kernels_backtracking'])},"
            f"{int(metrics['activation_weights_backtracking'])},"
            f"{int(metrics['data_term_weights_backtracking'])})"
        )
    print(line, flush=True)


def train_ipalm(
    args: argparse.Namespace,
    dataset: torch.utils.data.Dataset,
    trainer: VariationalNetworkTrainer,
) -> None:
    if trainer.step >= args.iterations:
        print(
            f"training already reached iteration {trainer.step}/"
            f"{args.iterations}",
            flush=True,
        )
        return
    schedule_size = len(
        grouped_batch_schedule(
            dataset, batch_size=args.batch_size, seed=args.seed
        )
    )
    while trainer.step < args.iterations:
        pass_index = trainer.step // schedule_size
        offset = trainer.step % schedule_size
        schedule = grouped_batch_schedule(
            dataset,
            batch_size=args.batch_size,
            seed=args.seed + pass_index,
        )
        remaining = args.iterations - trainer.step
        batches = schedule[offset : offset + remaining]
        loader = build_grouped_loader(
            dataset, batches, num_workers=args.num_workers
        )
        for batch in loader:
            metrics = trainer.train_step(batch)
            trainer.completed_epochs = trainer.step // schedule_size
            if trainer.step == 1 or trainer.step % args.log_every == 0:
                log_metrics(
                    prefix=f"iteration={trainer.step}/{args.iterations}",
                    trainer=trainer,
                    batch=batch,
                    metrics=metrics,
                )
            if (
                trainer.step % args.save_every_steps == 0
                or trainer.step == args.iterations
            ):
                path = (
                    args.output_dir
                    / f"checkpoint-step-{trainer.step:07d}.pt"
                )
                trainer.save(path)
                print(f"saved={path}", flush=True)
            if trainer.step >= args.iterations:
                break


def train_adam(
    args: argparse.Namespace,
    dataset: torch.utils.data.Dataset,
    trainer: VariationalNetworkTrainer,
) -> None:
    if trainer.completed_epochs >= args.epochs:
        print(
            f"training already reached epoch {trainer.completed_epochs} "
            f"at step {trainer.step}",
            flush=True,
        )
        return
    metric_names = (
        "loss",
        "magnitude_mse",
        "relative_rmse",
        "data_weight_mean",
    )
    for epoch_index in range(trainer.completed_epochs, args.epochs):
        schedule = grouped_batch_schedule(
            dataset,
            batch_size=args.batch_size,
            seed=args.seed + epoch_index,
        )
        loader = build_grouped_loader(
            dataset, schedule, num_workers=args.num_workers
        )
        totals = {name: 0.0 for name in metric_names}
        for epoch_step, batch in enumerate(loader, start=1):
            metrics = trainer.train_step(batch)
            for name in metric_names:
                totals[name] += metrics[name]
            if epoch_step == 1 or epoch_step % args.log_every == 0:
                log_metrics(
                    prefix=(
                        f"epoch={epoch_index + 1}/{args.epochs} "
                        f"epoch_step={epoch_step}/{len(loader)}"
                    ),
                    trainer=trainer,
                    batch=batch,
                    metrics=metrics,
                )
        trainer.completed_epochs = epoch_index + 1
        means = {
            name: value / len(loader)
            for name, value in totals.items()
        }
        print(
            f"epoch_complete={trainer.completed_epochs}/{args.epochs} "
            f"step={trainer.step} "
            + " ".join(
                f"mean_{name}={value:.8f}"
                for name, value in means.items()
            ),
            flush=True,
        )
        if (
            trainer.completed_epochs % args.save_every_epochs == 0
            or trainer.completed_epochs == args.epochs
        ):
            path = args.output_dir / (
                f"checkpoint-epoch-{trainer.completed_epochs:03d}-"
                f"step-{trainer.step:07d}.pt"
            )
            trainer.save(path)
            print(f"saved={path}", flush=True)


def main() -> None:
    args = parse_arguments()
    positive = (
        args.iterations,
        args.save_every_steps,
        args.epochs,
        args.save_every_epochs,
        args.log_every,
    )
    if min(positive) <= 0:
        raise ValueError("training and checkpoint intervals must be positive")
    if args.num_workers < 0 or args.batch_size <= 0:
        raise ValueError(
            "num-workers cannot be negative and batch-size must be positive"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("full Variational Network training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_dataset = MRISliceDataset(
        args.manifest,
        split="train",
        normalization_quantile=1.0,
        mask_seed=args.mask_seed,
        acceleration=args.acceleration,
        data_root=args.data_root,
    )
    data_config = {
        "acceleration": args.acceleration,
        "center_fraction": (
            base_dataset.mask_config.resolved_center_fraction
        ),
        "mask_seed": args.mask_seed,
        "normalization_quantile": 1.0,
    }
    dataset: torch.utils.data.Dataset = base_dataset
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(base_dataset):
            raise ValueError(
                "max-samples must be between 1 and the dataset size"
            )
        dataset = Subset(base_dataset, range(args.max_samples))

    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("model_name")
            != "hammernik_variational_network"
        ):
            raise ValueError(
                "resume checkpoint is not a Variational Network"
            )
        saved_optimizer = checkpoint.get("optimizer_name", "adam")
        if saved_optimizer != args.optimizer:
            raise ValueError(
                f"resume checkpoint uses {saved_optimizer}, but "
                f"--optimizer is {args.optimizer}"
            )
        model = build_variational_network(**checkpoint["model_config"])
    else:
        model = build_variational_network(
            gradient_checkpointing=(
                not args.disable_gradient_checkpointing
            )
        )
    trainer = VariationalNetworkTrainer(
        model,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        device="cuda",
        data_config=data_config,
        ipalm_initial_lipschitz=args.ipalm_initial_lipschitz,
        ipalm_momentum=args.ipalm_momentum,
        ipalm_max_backtracking=args.ipalm_max_backtracking,
    )
    if args.resume:
        trainer.load(args.resume)
        if args.disable_gradient_checkpointing:
            trainer.model.gradient_checkpointing = False
            trainer.model.model_config["gradient_checkpointing"] = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = (
        f"iterations={args.iterations}"
        if args.optimizer == "ipalm"
        else f"epochs={args.epochs}"
    )
    print(
        f"device={torch.cuda.get_device_name(0)} precision=fp32 "
        f"acceleration={args.acceleration}x slices={len(dataset)} "
        f"batch_size={args.batch_size} "
        f"optimizer={args.optimizer} {target} "
        f"stages={model.num_stages} filters={model.num_filters} "
        f"kernel={model.kernel_size} "
        f"rbf_weights={model.num_rbf_weights} "
        f"gradient_checkpointing={model.gradient_checkpointing} "
        f"start_step={trainer.step}",
        flush=True,
    )
    if args.optimizer == "ipalm":
        train_ipalm(args, dataset, trainer)
    else:
        train_adam(args, dataset, trainer)


if __name__ == "__main__":
    main()
