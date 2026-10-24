"""Evaluate one adapted SwinMR checkpoint on a frozen MRI split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from mri_data import MRISliceDataset, build_dataloader
from mri_model import reconstruction_metrics
from mri_reconstruction import channels_to_complex, measurement_residual
from mri_swinmr import build_swinmr, summarize_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD = "swinmr"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--online-weights", action="store_true", help="evaluate online instead of EMA weights")
    return parser.parse_args()


def read_rows(path: Path, *, step: int, split: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(row.get("checkpoint_step") != step or row.get("split") != split for row in rows):
        raise ValueError(f"existing metrics do not match this run: {path}")
    return rows


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("full SwinMR evaluation requires CUDA")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "adapted_complex_swinmr_pi":
        raise ValueError("checkpoint is not an adapted SwinMR-PI checkpoint")
    step = int(checkpoint["step"])
    completed_epochs = int(checkpoint.get("completed_epochs", 0))
    data_config = dict(checkpoint["data_config"])
    model = build_swinmr(**checkpoint["model_config"])
    weight_key = "model" if args.online_weights else "ema_model"
    model.load_state_dict(checkpoint[weight_key])
    official_commit = str(checkpoint["official_commit"])
    del checkpoint
    model = model.to(device).eval()

    manifest = PROJECT_ROOT / "phase1" / "manifests" / f"{args.split}.jsonl"
    dataset = MRISliceDataset(
        manifest,
        split=args.split,
        normalization_quantile=float(data_config.get("normalization_quantile", 0.99)),
        acceleration=int(data_config["acceleration"]),
        center_fraction=float(data_config["center_fraction"]),
        mask_seed=int(data_config["mask_seed"]),
        data_root=args.data_root,
    )
    expected_samples = len(dataset)
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(dataset):
            raise ValueError("max-samples must be between 1 and the split size")
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
        expected_samples = args.max_samples
    loader = build_dataloader(dataset, num_workers=args.num_workers, pin_memory=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "slice_metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    rows = read_rows(rows_path, step=step, split=args.split)
    completed = {str(row["sample_id"]) for row in rows}
    print(
        f"checkpoint={step} epoch={completed_epochs} split={args.split} device={device} "
        f"weights={weight_key} completed={len(completed)}/{expected_samples}",
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
                prediction = model(batch["initial_reconstruction"])
            torch.cuda.synchronize(device)
            reconstruction_seconds = time.perf_counter() - started
            initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
            target = channels_to_complex(batch["target"], channel_dim=1)[0]
            reconstructed = channels_to_complex(prediction, channel_dim=1)[0]
            zero_filled = reconstruction_metrics(target, initial)
            result = reconstruction_metrics(target, reconstructed)
            row = {
                "checkpoint_step": step,
                "completed_epochs": completed_epochs,
                "split": args.split,
                "sample_index": sample_index,
                "sample_id": current_id,
                "volume_id": str(batch["metadata"]["volume_id"][0]),
                "slice_index": int(batch["metadata"]["slice_index"][0]),
                "acquisition": str(batch["metadata"]["acquisition"][0]),
                "coils": int(batch["metadata"]["coils"][0]),
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
                f"{METHOD}_reconstruction_seconds": reconstruction_seconds,
                f"{METHOD}_measurement_residual": float(measurement_residual(
                    reconstructed.unsqueeze(0), batch["undersampled_kspace"],
                    batch["sensitivity_maps"], batch["mask"]
                )),
            }
            numeric = [
                value for key, value in row.items()
                if key.endswith(("energy", "nmse", "psnr", "ssim", "residual", "seconds"))
            ]
            if not all(torch.isfinite(torch.tensor(value)) for value in numeric):
                raise FloatingPointError(f"nonfinite metric for {current_id}")
            destination.write(json.dumps(row) + "\n")
            destination.flush()
            rows.append(row)
            completed.add(current_id)
            print(
                f"[{len(completed)}/{expected_samples}] {current_id} "
                f"ZF_PSNR={row['zero_filled_psnr']:.2f} "
                f"SwinMR_PSNR={row[f'{METHOD}_psnr']:.2f} "
                f"SwinMR_SSIM={row[f'{METHOD}_ssim']:.4f}",
                flush=True,
            )
    if len(rows) != expected_samples:
        raise RuntimeError(f"expected {expected_samples} rows, found {len(rows)}")
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "completed_epochs": completed_epochs,
        "weights": weight_key,
        "split": args.split,
        "precision": "fp32",
        "architecture": "official SwinMR-PI backbone, adapted complex multicoil input and loss",
        "official_commit": official_commit,
        "model_config": model.model_config,
        **summarize_evaluation(rows),
    }
    write_json_atomic(summary_path, summary)
    printable = {**summary, "volumes": f"{len(summary['volumes'])} records"}
    printable["summary_path"] = str(summary_path)
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
