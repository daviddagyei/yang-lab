"""Evaluate one unrolled-RDBM checkpoint on a frozen MRI split."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mri_data import MRISliceDataset, build_dataloader
from mri_model import reconstruction_metrics
from mri_reconstruction import channels_to_complex, measurement_residual
from mri_unrolled_rdbm import build_unrolled_rdbm


ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["volume_id"])].append(row)
    volumes = []
    for volume_id, slices in sorted(grouped.items()):
        target_energy = sum(float(row["target_energy"]) for row in slices)
        volumes.append(
            {
                "volume_id": volume_id,
                "acquisition": slices[0]["acquisition"],
                "slice_count": len(slices),
                "zero_filled": {
                    "nmse": sum(float(row["zero_filled_error_energy"]) for row in slices)
                    / target_energy,
                    "psnr": _mean(slices, "zero_filled_psnr"),
                    "ssim": _mean(slices, "zero_filled_ssim"),
                    "measurement_residual": _mean(
                        slices, "zero_filled_measurement_residual"
                    ),
                },
                "unrolled_rdbm": {
                    "nmse": sum(float(row["unrolled_rdbm_error_energy"]) for row in slices)
                    / target_energy,
                    "psnr": _mean(slices, "unrolled_rdbm_psnr"),
                    "ssim": _mean(slices, "unrolled_rdbm_ssim"),
                    "measurement_residual": _mean(
                        slices, "unrolled_rdbm_measurement_residual"
                    ),
                    "seconds_per_slice": _mean(
                        slices, "unrolled_rdbm_reconstruction_seconds"
                    ),
                },
            }
        )

    def macro(method: str) -> dict[str, float]:
        metrics = ["nmse", "psnr", "ssim", "measurement_residual"]
        if method == "unrolled_rdbm":
            metrics.append("seconds_per_slice")
        return {
            metric: float(np.mean([volume[method][metric] for volume in volumes]))
            for metric in metrics
        }

    return {
        "volume_count": len(volumes),
        "slice_count": len(rows),
        "aggregation": "slice metrics within each volume, then equal-weight mean across volumes",
        "zero_filled": macro("zero_filled"),
        "unrolled_rdbm": macro("unrolled_rdbm"),
        "volumes": volumes,
    }


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("unrolled-RDBM evaluation requires the CUDA GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU does not support BF16")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "unrolled_rdbm":
        raise ValueError("checkpoint is not an unrolled RDBM")
    step = int(checkpoint["step"])
    completed_epochs = int(checkpoint.get("completed_epochs", 0))
    acceleration = int(checkpoint.get("data_config", {}).get("acceleration", 4))
    model = build_unrolled_rdbm(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    device = torch.device("cuda")
    model = model.to(device).eval()

    dataset = MRISliceDataset(
        ROOT / "phase1" / "manifests" / f"{args.split}.jsonl",
        split=args.split,
        acceleration=acceleration,
    )
    loader = build_dataloader(dataset, num_workers=args.num_workers, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "slice_metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    rows = []
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
        if any(
            row.get("checkpoint_step") != step
            or row.get("split") != args.split
            or row.get("precision") != args.precision
            for row in rows
        ):
            raise ValueError(f"existing metrics do not match this run: {rows_path}")
    completed = {str(row["sample_id"]) for row in rows}
    print(
        f"checkpoint={step} epoch={completed_epochs} split={args.split} "
        f"precision={args.precision} completed={len(completed)}/{len(dataset)}",
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
            started = time.perf_counter()
            with torch.inference_mode(), autocast:
                prediction = model(
                    batch["initial_reconstruction"],
                    batch["undersampled_kspace"],
                    batch["sensitivity_maps"],
                    batch["mask"],
                )
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            assert isinstance(prediction, torch.Tensor)
            initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
            target = channels_to_complex(batch["target"], channel_dim=1)[0]
            reconstructed = channels_to_complex(prediction, channel_dim=1)[0]
            zero_filled = reconstruction_metrics(target, initial)
            result = reconstruction_metrics(target, reconstructed)
            row = {
                "checkpoint_step": step,
                "completed_epochs": completed_epochs,
                "split": args.split,
                "precision": args.precision,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "volume_id": str(batch["metadata"]["volume_id"][0]),
                "slice_index": int(batch["metadata"]["slice_index"][0]),
                "acquisition": str(batch["metadata"]["acquisition"][0]),
                "target_energy": zero_filled["target_energy"],
                "zero_filled_error_energy": zero_filled["error_energy"],
                "zero_filled_nmse": zero_filled["nmse"],
                "zero_filled_psnr": zero_filled["psnr"],
                "zero_filled_ssim": zero_filled["ssim"],
                "zero_filled_measurement_residual": float(
                    measurement_residual(
                        initial.unsqueeze(0), batch["undersampled_kspace"],
                        batch["sensitivity_maps"], batch["mask"]
                    )
                ),
                "unrolled_rdbm_error_energy": result["error_energy"],
                "unrolled_rdbm_nmse": result["nmse"],
                "unrolled_rdbm_psnr": result["psnr"],
                "unrolled_rdbm_ssim": result["ssim"],
                "unrolled_rdbm_reconstruction_seconds": seconds,
                "unrolled_rdbm_measurement_residual": float(
                    measurement_residual(
                        reconstructed.unsqueeze(0), batch["undersampled_kspace"],
                        batch["sensitivity_maps"], batch["mask"]
                    )
                ),
            }
            numeric = [
                value for key, value in row.items()
                if key.endswith(("energy", "nmse", "psnr", "ssim", "residual", "seconds"))
            ]
            if not all(np.isfinite(float(value)) for value in numeric):
                raise FloatingPointError(f"nonfinite metric for {sample_id}")
            destination.write(json.dumps(row) + "\n")
            destination.flush()
            rows.append(row)
            completed.add(sample_id)
            if len(completed) == 1 or len(completed) % 10 == 0:
                print(
                    f"[{len(completed)}/{len(dataset)}] {sample_id} "
                    f"ZF_PSNR={row['zero_filled_psnr']:.2f} "
                    f"UNROLLED_PSNR={row['unrolled_rdbm_psnr']:.2f}",
                    flush=True,
                )

    if len(rows) != len(dataset):
        raise RuntimeError(f"expected {len(dataset)} rows, found {len(rows)}")
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "completed_epochs": completed_epochs,
        "acceleration": acceleration,
        "split": args.split,
        "precision": args.precision,
        "timing_context": "shared GPU if another process was active",
        **summarize(rows),
    }
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(summary_path)
    print(json.dumps({**summary, "volumes": f"{len(summary['volumes'])} records"}, indent=2))


if __name__ == "__main__":
    main()
