"""Evaluate every unrolled stage of one checkpoint on validation data."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mri_data import MRISliceDataset, build_dataloader
from mri_diffusion_unrolled_rdbm import (
    build_diffusion_unrolled_rdbm,
    summarize_evaluation,
)
from mri_model import reconstruction_metrics
from mri_reconstruction import channels_to_complex, measurement_residual


ROOT = Path(__file__).resolve().parents[2]
METHOD = "diffusion_unrolled_rdbm"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def stage_summary(rows: list[dict[str, Any]], stage: int) -> dict[str, Any]:
    mapped = []
    for row in rows:
        base = {
            key: value
            for key, value in row.items()
            if key.startswith("zero_filled_")
            or key in ("volume_id", "acquisition", "target_energy")
        }
        for suffix in (
            "error_energy",
            "nmse",
            "psnr",
            "ssim",
            "measurement_residual",
        ):
            base[f"{METHOD}_{suffix}"] = row[f"stage_{stage}_{suffix}"]
        base[f"{METHOD}_reconstruction_seconds"] = row["full_trajectory_seconds"]
        mapped.append(base)
    return summarize_evaluation(mapped)[METHOD]


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("stage evaluation requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU does not support BF16")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "full_diffusion_unrolled_rdbm":
        raise ValueError("checkpoint is not a full diffusion-unrolled RDBM")
    step = int(checkpoint["step"])
    completed_epochs = int(checkpoint.get("completed_epochs", 0))
    data_config = dict(checkpoint["data_config"])
    model = build_diffusion_unrolled_rdbm(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    device = torch.device("cuda")
    model = model.to(device).eval()

    dataset = MRISliceDataset(
        ROOT / "phase1/manifests/val.jsonl",
        split="val",
        acceleration=int(data_config["acceleration"]),
        center_fraction=float(data_config["center_fraction"]),
        mask_seed=int(data_config["mask_seed"]),
        normalization_quantile=float(data_config.get("normalization_quantile", 0.99)),
        data_root=args.data_root,
    )
    loader = build_dataloader(dataset, num_workers=args.num_workers, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "slice_stage_metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    rows: list[dict[str, Any]] = []
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
        if any(
            row.get("checkpoint_step") != step
            or row.get("precision") != args.precision
            for row in rows
        ):
            raise ValueError(f"existing metrics do not match this run: {rows_path}")
    completed = {str(row["sample_id"]) for row in rows}
    print(
        f"checkpoint={step} epoch={completed_epochs} acceleration="
        f"{data_config['acceleration']}x split=val precision={args.precision} "
        f"stages=1-{model.unroll_steps} completed={len(completed)}/{len(dataset)}",
        flush=True,
    )

    with rows_path.open("a") as destination:
        for sample_index, batch in enumerate(loader):
            sample_id = str(batch["metadata"]["sample_id"][0])
            if sample_id in completed:
                continue
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.precision == "bf16"
                else nullcontext()
            )
            torch.cuda.synchronize(device)
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            with torch.inference_mode(), autocast:
                trajectory = model(
                    batch["initial_reconstruction"],
                    batch["undersampled_kspace"],
                    batch["sensitivity_maps"],
                    batch["mask"],
                    return_trajectory=True,
                )
            finished.record()
            torch.cuda.synchronize(device)
            assert isinstance(trajectory, list)
            initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
            target = channels_to_complex(batch["target"], channel_dim=1)[0]
            zero = reconstruction_metrics(target, initial)
            row: dict[str, Any] = {
                "checkpoint_step": step,
                "completed_epochs": completed_epochs,
                "precision": args.precision,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "volume_id": str(batch["metadata"]["volume_id"][0]),
                "slice_index": int(batch["metadata"]["slice_index"][0]),
                "acquisition": str(batch["metadata"]["acquisition"][0]),
                "target_energy": zero["target_energy"],
                "zero_filled_error_energy": zero["error_energy"],
                "zero_filled_nmse": zero["nmse"],
                "zero_filled_psnr": zero["psnr"],
                "zero_filled_ssim": zero["ssim"],
                "zero_filled_measurement_residual": float(
                    measurement_residual(
                        initial.unsqueeze(0),
                        batch["undersampled_kspace"],
                        batch["sensitivity_maps"],
                        batch["mask"],
                    )
                ),
                "full_trajectory_seconds": float(started.elapsed_time(finished) / 1000.0),
            }
            for stage, stage_channels in enumerate(trajectory[1:], start=1):
                image = channels_to_complex(stage_channels.float(), channel_dim=1)[0]
                metrics = reconstruction_metrics(target, image)
                row.update(
                    {
                        f"stage_{stage}_error_energy": metrics["error_energy"],
                        f"stage_{stage}_nmse": metrics["nmse"],
                        f"stage_{stage}_psnr": metrics["psnr"],
                        f"stage_{stage}_ssim": metrics["ssim"],
                        f"stage_{stage}_measurement_residual": float(
                            measurement_residual(
                                image.unsqueeze(0),
                                batch["undersampled_kspace"],
                                batch["sensitivity_maps"],
                                batch["mask"],
                            )
                        ),
                    }
                )
            numeric = [value for value in row.values() if isinstance(value, float)]
            if not all(np.isfinite(value) for value in numeric):
                raise FloatingPointError(f"nonfinite stage metric for {sample_id}")
            destination.write(json.dumps(row) + "\n")
            destination.flush()
            rows.append(row)
            completed.add(sample_id)
            if len(completed) == 1 or len(completed) % 10 == 0:
                print(f"[{len(completed)}/{len(dataset)}] {sample_id}", flush=True)

    if len(rows) != len(dataset):
        raise RuntimeError(f"expected {len(dataset)} rows, found {len(rows)}")
    stages = {
        str(stage): stage_summary(rows, stage)
        for stage in range(1, model.unroll_steps + 1)
    }
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "completed_epochs": completed_epochs,
        "acceleration": int(data_config["acceleration"]),
        "split": "val",
        "precision": args.precision,
        "selection_scope": "all ten post-CG clean reconstruction stages",
        "volume_count": len({row["volume_id"] for row in rows}),
        "slice_count": len(rows),
        "stages": stages,
    }
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(summary_path)
    print(json.dumps({**summary, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
