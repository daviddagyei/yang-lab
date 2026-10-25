"""Evaluate one AdaDiff prior with per-slice adaptation on a frozen split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from mri_adadiff import (
    OFFICIAL_COMMIT,
    DiffusionSchedule,
    build_adadiff_generator,
    rapid_reconstruct,
    reconstruct_with_adaptation,
    summarize_evaluation,
)
from mri_data import MRISliceDataset, build_dataloader
from mri_model import reconstruction_metrics
from mri_reconstruction import channels_to_complex, measurement_residual


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--acceleration", type=int, choices=(2, 4, 6, 8), required=True)
    parser.add_argument("--adaptation-iterations", type=int, default=1_000)
    parser.add_argument("--adaptation-learning-rate", type=float, default=1e-3)
    parser.add_argument("--rapid-start", choices=("zero_filled", "noise"), default="zero_filled")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online-weights", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path, configuration: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        for key, value in configuration.items():
            if row.get(key) != value:
                raise ValueError(f"existing AdaDiff metrics differ at {key}: {path}")
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("full AdaDiff evaluation requires CUDA")
    if args.adaptation_iterations < 0 or args.adaptation_learning_rate <= 0:
        raise ValueError("invalid adaptation settings")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("model_name") != "adadiff_multicoil_adaptation"
        or checkpoint.get("official_commit") != OFFICIAL_COMMIT
    ):
        raise ValueError("checkpoint is not this AdaDiff implementation")
    generator = build_adadiff_generator(**checkpoint["generator_config"])
    weight_key = "generator" if args.online_weights else "ema_generator"
    generator.load_state_dict(checkpoint[weight_key])
    generator = generator.to(device).eval()
    schedule = DiffusionSchedule(**checkpoint["schedule_config"])
    step = int(checkpoint["step"])
    epoch = int(checkpoint["completed_epochs"])
    del checkpoint

    dataset = MRISliceDataset(
        PROJECT_ROOT / "phase1/manifests" / f"{args.split}.jsonl",
        split=args.split,
        acceleration=args.acceleration,
        data_root=args.data_root,
    )
    expected = len(dataset)
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(dataset):
            raise ValueError("max-samples must fit the requested split")
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
        expected = args.max_samples
    loader = build_dataloader(dataset, num_workers=args.num_workers, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "slice_metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    configuration = {
        "checkpoint_step": step,
        "split": args.split,
        "acceleration": args.acceleration,
        "adaptation_iterations": args.adaptation_iterations,
        "rapid_start": args.rapid_start,
    }
    rows = _load_rows(rows_path, configuration)
    complete = {str(row["sample_id"]) for row in rows}
    print(
        f"checkpoint={step} epoch={epoch} split={args.split} rate={args.acceleration}x "
        f"adaptation_iterations={args.adaptation_iterations} completed={len(complete)}/{expected}",
        flush=True,
    )
    with rows_path.open("a") as destination:
        for sample_index, batch in enumerate(loader):
            sample_id = str(batch["metadata"]["sample_id"][0])
            if sample_id in complete:
                continue
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            torch.cuda.synchronize(device)
            rapid_started = time.perf_counter()
            rapid = rapid_reconstruct(
                generator,
                batch,
                schedule=schedule,
                seed=args.seed + sample_index,
                start=args.rapid_start,
            )
            torch.cuda.synchronize(device)
            rapid_seconds = time.perf_counter() - rapid_started

            adaptation_started = time.perf_counter()
            reconstruction = reconstruct_with_adaptation(
                generator,
                batch,
                schedule=schedule,
                adaptation_iterations=args.adaptation_iterations,
                adaptation_learning_rate=args.adaptation_learning_rate,
                seed=args.seed + sample_index,
                rapid_start=args.rapid_start,
                rapid_initial=rapid,
            )
            torch.cuda.synchronize(device)
            adaptation_seconds = time.perf_counter() - adaptation_started
            total_seconds = rapid_seconds + adaptation_seconds

            initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
            target = channels_to_complex(batch["target"], channel_dim=1)[0]
            rapid_image = channels_to_complex(reconstruction.rapid, channel_dim=1)[0]
            adapted_image = channels_to_complex(reconstruction.adapted, channel_dim=1)[0]
            metrics = {
                "zero_filled": reconstruction_metrics(target, initial),
                "rapid_adadiff": reconstruction_metrics(target, rapid_image),
                "adadiff": reconstruction_metrics(target, adapted_image),
            }
            row: dict[str, Any] = {
                **configuration,
                "completed_epochs": epoch,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "volume_id": str(batch["metadata"]["volume_id"][0]),
                "slice_index": int(batch["metadata"]["slice_index"][0]),
                "acquisition": str(batch["metadata"]["acquisition"][0]),
                "coils": int(batch["metadata"]["coils"][0]),
                "target_energy": metrics["zero_filled"]["target_energy"],
                "initial_adaptation_loss": reconstruction.initial_adaptation_loss,
                "final_adaptation_loss": reconstruction.final_adaptation_loss,
            }
            for method, result in metrics.items():
                row.update(
                    {
                        f"{method}_error_energy": result["error_energy"],
                        f"{method}_nmse": result["nmse"],
                        f"{method}_psnr": result["psnr"],
                        f"{method}_ssim": result["ssim"],
                    }
                )
            row["rapid_adadiff_reconstruction_seconds"] = rapid_seconds
            row["adadiff_reconstruction_seconds"] = total_seconds
            for method, image in (
                ("zero_filled", initial),
                ("rapid_adadiff", rapid_image),
                ("adadiff", adapted_image),
            ):
                row[f"{method}_measurement_residual"] = float(
                    measurement_residual(
                        image.unsqueeze(0),
                        batch["undersampled_kspace"],
                        batch["sensitivity_maps"],
                        batch["mask"],
                    )
                )
            numeric = [
                value
                for value in row.values()
                if isinstance(value, (float, int)) and value is not None
            ]
            if not all(bool(torch.isfinite(torch.tensor(value))) for value in numeric):
                raise FloatingPointError(f"nonfinite AdaDiff metric for {sample_id}")
            destination.write(json.dumps(row) + "\n")
            destination.flush()
            rows.append(row)
            complete.add(sample_id)
            print(
                f"[{len(complete)}/{expected}] {sample_id} "
                f"rapid={row['rapid_adadiff_psnr']:.2f}dB "
                f"adapted={row['adadiff_psnr']:.2f}dB "
                f"seconds={total_seconds:.1f}",
                flush=True,
            )
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "completed_epochs": epoch,
        "weights": weight_key,
        "split": args.split,
        "acceleration": args.acceleration,
        "adaptation_iterations": args.adaptation_iterations,
        "adaptation_learning_rate": args.adaptation_learning_rate,
        "rapid_start": args.rapid_start,
        "official_commit": OFFICIAL_COMMIT,
        **summarize_evaluation(rows),
    }
    _atomic_json(summary_path, summary)
    printable = {**summary, "volumes": f"{len(summary['volumes'])} records"}
    printable["summary_path"] = str(summary_path)
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()

