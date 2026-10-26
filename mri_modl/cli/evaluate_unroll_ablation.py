"""Evaluate every intermediate MoDL unroll for one validation checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from mri_data import MRISliceDataset, build_dataloader
from mri_model import reconstruction_metrics
from mri_modl import build_modl, summarize_evaluation
from mri_reconstruction import channels_to_complex, measurement_residual


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def read_rows(path: Path, *, step: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(row.get("checkpoint_step") != step or row.get("split") != "val" for row in rows):
        raise ValueError(f"existing metrics do not match this run: {path}")
    return rows


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stage_rows(rows: list[dict[str, Any]], stage: int) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        metrics = row["stages"][str(stage)]
        output.append(
            {
                "volume_id": row["volume_id"],
                "acquisition": row["acquisition"],
                "target_energy": row["target_energy"],
                "zero_filled_error_energy": row["zero_filled_error_energy"],
                "zero_filled_psnr": row["zero_filled_psnr"],
                "zero_filled_ssim": row["zero_filled_ssim"],
                "zero_filled_measurement_residual": row["zero_filled_measurement_residual"],
                "modl_error_energy": metrics["error_energy"],
                "modl_psnr": metrics["psnr"],
                "modl_ssim": metrics["ssim"],
                "modl_measurement_residual": metrics["measurement_residual"],
                "modl_reconstruction_seconds": row["full_trajectory_seconds"],
            }
        )
    return output


def remove_non_stage_timing(summary: dict[str, Any]) -> None:
    summary["modl"].pop("seconds_per_slice", None)
    for group in summary["by_acquisition"].values():
        group["modl"].pop("seconds_per_slice", None)
    for volume in summary["volumes"]:
        volume["modl"].pop("seconds_per_slice", None)


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("MoDL unroll ablation requires a CUDA GPU")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "modl":
        raise ValueError("checkpoint is not a MoDL checkpoint")
    step = int(checkpoint["step"])
    data_config = dict(
        checkpoint.get(
            "data_config",
            {
                "acceleration": 4,
                "center_fraction": 0.08,
                "mask_seed": 20261019,
                "normalization_quantile": 0.99,
            },
        )
    )
    model = build_modl(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    unrolls = int(model.unrolls)
    if unrolls <= 1:
        raise ValueError("unroll ablation requires a multi-unroll checkpoint")

    dataset = MRISliceDataset(
        PROJECT_ROOT / "phase1/manifests/val.jsonl",
        split="val",
        data_root=args.data_root,
        acceleration=int(data_config["acceleration"]),
        center_fraction=float(data_config["center_fraction"]),
        mask_seed=int(data_config["mask_seed"]),
        normalization_quantile=float(data_config["normalization_quantile"]),
    )
    expected_samples = len(dataset)
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(dataset):
            raise ValueError("max-samples must be between 1 and the validation size")
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
        expected_samples = args.max_samples
    loader = build_dataloader(dataset, num_workers=args.num_workers, pin_memory=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "slice_metrics.jsonl"
    rows = read_rows(rows_path, step=step)
    completed = {str(row["sample_id"]) for row in rows}
    print(
        f"checkpoint={step} acceleration={data_config['acceleration']}x "
        f"unrolls={unrolls} completed={len(completed)}/{expected_samples}",
        flush=True,
    )

    with rows_path.open("a") as destination:
        for sample_index, batch in enumerate(loader):
            current_id = str(batch["metadata"]["sample_id"][0])
            if current_id in completed:
                continue
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                trajectory = model(
                    batch["undersampled_kspace"],
                    batch["sensitivity_maps"],
                    batch["mask"],
                    return_trajectory=True,
                )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            assert isinstance(trajectory, list) and len(trajectory) == unrolls + 1
            target = channels_to_complex(batch["target"], channel_dim=1)[0]
            initial = channels_to_complex(trajectory[0], channel_dim=1)[0]
            zero_filled = reconstruction_metrics(target, initial)
            row: dict[str, Any] = {
                "checkpoint_step": step,
                "split": "val",
                "sample_index": sample_index,
                "sample_id": current_id,
                "volume_id": str(batch["metadata"]["volume_id"][0]),
                "acquisition": str(batch["metadata"]["acquisition"][0]),
                "target_energy": zero_filled["target_energy"],
                "zero_filled_error_energy": zero_filled["error_energy"],
                "zero_filled_psnr": zero_filled["psnr"],
                "zero_filled_ssim": zero_filled["ssim"],
                "zero_filled_measurement_residual": float(
                    measurement_residual(
                        initial.unsqueeze(0),
                        batch["undersampled_kspace"],
                        batch["sensitivity_maps"],
                        batch["mask"],
                    )
                ),
                "full_trajectory_seconds": elapsed,
                "stages": {},
            }
            for stage, channels in enumerate(trajectory[1:], start=1):
                image = channels_to_complex(channels, channel_dim=1)[0]
                metrics = reconstruction_metrics(target, image)
                row["stages"][str(stage)] = {
                    **metrics,
                    "measurement_residual": float(
                        measurement_residual(
                            image.unsqueeze(0),
                            batch["undersampled_kspace"],
                            batch["sensitivity_maps"],
                            batch["mask"],
                        )
                    ),
                }
            destination.write(json.dumps(row) + "\n")
            destination.flush()
            rows.append(row)
            completed.add(current_id)
            print(f"[{len(completed)}/{expected_samples}] {current_id}", flush=True)

    if len(rows) != expected_samples:
        raise RuntimeError(f"expected {expected_samples} rows, found {len(rows)}")
    stages: dict[str, Any] = {}
    for stage in range(1, unrolls + 1):
        stage_summary = summarize_evaluation(stage_rows(rows, stage))
        remove_non_stage_timing(stage_summary)
        stages[str(stage)] = stage_summary
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "split": "val",
        "acceleration": int(data_config["acceleration"]),
        "unrolls": unrolls,
        "full_trajectory_seconds_per_slice": sum(
            float(row["full_trajectory_seconds"]) for row in rows
        )
        / len(rows),
        "stages": stages,
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps({"status": "ok", "summary": str(args.output_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
