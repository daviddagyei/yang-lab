"""Evaluate one full diffusion-unrolled RDBM checkpoint."""

from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-stages",
        type=int,
        help="stop after this many unrolled stages; defaults to the full trajectory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
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
    output_stage = model.unroll_steps if args.max_stages is None else args.max_stages
    if not 1 <= output_stage <= model.unroll_steps:
        raise ValueError(f"max-stages must be between 1 and {model.unroll_steps}")

    dataset = MRISliceDataset(
        ROOT / "phase1/manifests" / f"{args.split}.jsonl",
        split=args.split,
        acceleration=int(data_config["acceleration"]),
        center_fraction=float(data_config["center_fraction"]),
        mask_seed=int(data_config["mask_seed"]),
        normalization_quantile=float(data_config.get("normalization_quantile", 0.99)),
        data_root=args.data_root,
    )
    loader = build_dataloader(dataset, num_workers=args.num_workers, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "slice_metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    rows: list[dict[str, Any]] = []
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
        if any(
            row.get("checkpoint_step") != step
            or row.get("split") != args.split
            or row.get("precision") != args.precision
            or int(row.get("output_stage", model.unroll_steps)) != output_stage
            for row in rows
        ):
            raise ValueError(f"existing metrics do not match this run: {rows_path}")
    completed = {str(row["sample_id"]) for row in rows}
    print(
        f"checkpoint={step} epoch={completed_epochs} acceleration="
        f"{data_config['acceleration']}x split={args.split} precision={args.precision} "
        f"output_stage={output_stage} "
        f"completed={len(completed)}/{len(dataset)}",
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
                    max_stages=output_stage,
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
                "output_stage": output_stage,
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
                "zero_filled_measurement_residual": float(measurement_residual(
                    initial.unsqueeze(0), batch["undersampled_kspace"],
                    batch["sensitivity_maps"], batch["mask"]
                )),
                f"{METHOD}_error_energy": result["error_energy"],
                f"{METHOD}_nmse": result["nmse"],
                f"{METHOD}_psnr": result["psnr"],
                f"{METHOD}_ssim": result["ssim"],
                f"{METHOD}_reconstruction_seconds": seconds,
                f"{METHOD}_measurement_residual": float(measurement_residual(
                    reconstructed.unsqueeze(0), batch["undersampled_kspace"],
                    batch["sensitivity_maps"], batch["mask"]
                )),
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
                    f"FULL_UNROLLED_PSNR={row[f'{METHOD}_psnr']:.2f}",
                    flush=True,
                )

    if len(rows) != len(dataset):
        raise RuntimeError(f"expected {len(dataset)} rows, found {len(rows)}")
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "completed_epochs": completed_epochs,
        "acceleration": int(data_config["acceleration"]),
        "split": args.split,
        "precision": args.precision,
        "output_stage": output_stage,
        "initialization": "scratch",
        "architecture": "ten-stage diffusion-faithful RDBM with six-iteration CG-SENSE",
        **summarize_evaluation(rows),
    }
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(summary_path)
    print(json.dumps({**summary, "volumes": f"{len(summary['volumes'])} records"}, indent=2))


if __name__ == "__main__":
    main()
