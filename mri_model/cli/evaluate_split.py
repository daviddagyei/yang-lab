"""Evaluate one MRI checkpoint on every eligible slice in a frozen split."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from mri_data import MRISliceDataset, build_dataloader
from mri_model import build_mri_rdbm, reconstruction_metrics, summarize_evaluation
from mri_reconstruction import channels_to_complex, measurement_residual


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-consistency-steps", type=int, default=4)
    parser.add_argument(
        "--disable-data-consistency",
        action="store_true",
        help="Disable consistency during reverse sampling and final correction.",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--acceleration", type=int, choices=(2, 4, 6, 8))
    return parser.parse_args()


def _read_rows(
    path: Path,
    *,
    step: int,
    split: str,
    precision: str,
    final_consistency_steps: int,
    apply_data_consistency: bool,
    acceleration: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(
        row.get("checkpoint_step") != step
        or row.get("split") != split
        or row.get("precision") != precision
        or row.get("final_consistency_steps") != final_consistency_steps
        or row.get("apply_data_consistency", True) != apply_data_consistency
        or int(row.get("acceleration", 4)) != acceleration
        for row in rows
    ):
        raise ValueError(f"existing metrics do not match this run: {path}")
    return rows


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_arguments()
    if args.final_consistency_steps < 0:
        raise ValueError("final-consistency-steps cannot be negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 evaluation requires a supported CUDA GPU; use --precision fp32")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    step = int(checkpoint["step"])
    checkpoint_acceleration = int(checkpoint.get("data_config", {}).get("acceleration", 4))
    acceleration = args.acceleration or checkpoint_acceleration
    if args.acceleration is not None and args.acceleration != checkpoint_acceleration:
        raise ValueError("requested acceleration differs from checkpoint data configuration")
    model = build_mri_rdbm(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model = model.to(device)
    model.eval()

    manifest = PROJECT_ROOT / "phase1" / "manifests" / f"{args.split}.jsonl"
    dataset = MRISliceDataset(manifest, split=args.split, acceleration=acceleration)
    expected_samples = len(dataset)
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(dataset):
            raise ValueError("max-samples must be between 1 and the split size")
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
        expected_samples = args.max_samples
    loader = build_dataloader(
        dataset,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "slice_metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    rows = _read_rows(
        rows_path,
        step=step,
        split=args.split,
        precision=args.precision,
        final_consistency_steps=args.final_consistency_steps,
        apply_data_consistency=not args.disable_data_consistency,
        acceleration=acceleration,
    )
    completed = {str(row["sample_id"]) for row in rows}
    print(
        f"checkpoint={step} split={args.split} acceleration={acceleration}x "
        f"device={device} precision={args.precision} "
        f"data_consistency={not args.disable_data_consistency} "
        f"completed={len(completed)}/{expected_samples}",
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            reconstruction_start = time.perf_counter()
            with torch.inference_mode(), autocast:
                reconstruction = model.reconstruct(
                    batch["initial_reconstruction"],
                    batch["undersampled_kspace"],
                    batch["sensitivity_maps"],
                    batch["mask"],
                    apply_data_consistency=not args.disable_data_consistency,
                    final_consistency_steps=args.final_consistency_steps,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            reconstruction_seconds = time.perf_counter() - reconstruction_start

            initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
            target = channels_to_complex(batch["target"], channel_dim=1)[0]
            reconstructed = channels_to_complex(reconstruction, channel_dim=1)[0]
            zero_filled = reconstruction_metrics(target, initial)
            rdbm = reconstruction_metrics(target, reconstructed)
            row = {
                "checkpoint_step": step,
                "acceleration": acceleration,
                "split": args.split,
                "precision": args.precision,
                "final_consistency_steps": args.final_consistency_steps,
                "apply_data_consistency": not args.disable_data_consistency,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "volume_id": str(batch["metadata"]["volume_id"][0]),
                "slice_index": int(batch["metadata"]["slice_index"][0]),
                "acquisition": str(batch["metadata"]["acquisition"][0]),
                "coils": int(batch["metadata"]["coils"][0]),
                "target_energy": zero_filled["target_energy"],
                "zero_filled_error_energy": zero_filled["error_energy"],
                "zero_filled_nmse": zero_filled["nmse"],
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
                "rdbm_error_energy": rdbm["error_energy"],
                "rdbm_nmse": rdbm["nmse"],
                "rdbm_psnr": rdbm["psnr"],
                "rdbm_ssim": rdbm["ssim"],
                "rdbm_reconstruction_seconds": reconstruction_seconds,
                "rdbm_measurement_residual": float(
                    measurement_residual(
                        reconstructed.unsqueeze(0),
                        batch["undersampled_kspace"],
                        batch["sensitivity_maps"],
                        batch["mask"],
                    )
                ),
            }
            numeric_metrics = [
                value
                for key, value in row.items()
                if key.endswith(("energy", "nmse", "psnr", "ssim", "residual", "seconds"))
            ]
            if not all(torch.isfinite(torch.tensor(value)) for value in numeric_metrics):
                raise FloatingPointError(f"nonfinite evaluation metric for {sample_id}")
            destination.write(json.dumps(row) + "\n")
            destination.flush()
            rows.append(row)
            completed.add(sample_id)
            print(
                f"[{len(completed)}/{expected_samples}] {sample_id} "
                f"ZF_PSNR={row['zero_filled_psnr']:.2f} RDBM_PSNR={row['rdbm_psnr']:.2f} "
                f"RDBM_SSIM={row['rdbm_ssim']:.4f}",
                flush=True,
            )

    if len(rows) != expected_samples:
        raise RuntimeError(f"expected {expected_samples} unique rows, found {len(rows)}")
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "acceleration": acceleration,
        "split": args.split,
        "precision": args.precision,
        "final_consistency_steps": args.final_consistency_steps,
        "apply_data_consistency": not args.disable_data_consistency,
        **summarize_evaluation(rows),
    }
    _write_summary(summary_path, summary)
    print(json.dumps({**summary, "volumes": f"{len(summary['volumes'])} records", "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
