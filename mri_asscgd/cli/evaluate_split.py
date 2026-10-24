"""Evaluate one ASSCGD checkpoint on every eligible slice in a frozen split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from mri_asscgd import build_asscgd_diffusion, summarize_evaluation
from mri_data import MRISliceDataset, build_dataloader
from mri_model import reconstruction_metrics
from mri_reconstruction import channels_to_complex, measurement_residual


ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--noise-seed", type=int, default=20261021)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def sample_noise(initial: torch.Tensor, sample_id: str, base_seed: int) -> tuple[torch.Tensor, int]:
    """Create resume-safe inference noise determined only by sample identity."""

    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(initial.shape, dtype=torch.float32, generator=generator)
    return noise.to(device=initial.device, dtype=initial.dtype), seed


def read_rows(
    path: Path, *, step: int, split: str, precision: str, noise_seed: int
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(
        row.get("checkpoint_step") != step
        or row.get("split") != split
        or row.get("precision") != precision
        or row.get("noise_seed") != noise_seed
        for row in rows
    ):
        raise ValueError(f"existing metrics do not match this run: {path}")
    return rows


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("full ASSCGD evaluation requires the CUDA GPU")
    device = torch.device("cuda")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU does not support BF16; use --precision fp32")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "asscgd_inspired_adversarial_diffusion":
        raise ValueError("checkpoint is not an ASSCGD adversarial diffusion checkpoint")
    step = int(checkpoint["step"])
    completed_epochs = int(checkpoint.get("completed_epochs", 0))
    model_config = dict(checkpoint["model_config"])
    model = build_asscgd_diffusion(**model_config)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model = model.to(device).eval()

    dataset = MRISliceDataset(
        ROOT / "phase1" / "manifests" / f"{args.split}.jsonl",
        split=args.split,
        acceleration=4,
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
    rows = read_rows(
        rows_path,
        step=step,
        split=args.split,
        precision=args.precision,
        noise_seed=args.noise_seed,
    )
    completed = {str(row["sample_id"]) for row in rows}
    print(
        f"checkpoint={step} epoch={completed_epochs} split={args.split} device={device} "
        f"precision={args.precision} completed={len(completed)}/{expected_samples}",
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
            noise, per_sample_seed = sample_noise(
                batch["initial_reconstruction"], sample_id, args.noise_seed
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.precision == "bf16"
                else nullcontext()
            )
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode(), autocast:
                prediction = model.reconstruct(
                    batch["initial_reconstruction"],
                    batch["undersampled_kspace"],
                    batch["sensitivity_maps"],
                    batch["mask"],
                    noise=noise,
                )
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            assert isinstance(prediction, torch.Tensor)
            initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
            target = channels_to_complex(batch["target"], channel_dim=1)[0]
            reconstructed = channels_to_complex(prediction, channel_dim=1)[0]
            zero_filled = reconstruction_metrics(target, initial)
            asscgd = reconstruction_metrics(target, reconstructed)
            row = {
                "checkpoint_step": step,
                "completed_epochs": completed_epochs,
                "split": args.split,
                "precision": args.precision,
                "noise_seed": args.noise_seed,
                "sample_noise_seed": per_sample_seed,
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
                        initial.unsqueeze(0), batch["undersampled_kspace"],
                        batch["sensitivity_maps"], batch["mask"]
                    )
                ),
                "asscgd_error_energy": asscgd["error_energy"],
                "asscgd_nmse": asscgd["nmse"],
                "asscgd_psnr": asscgd["psnr"],
                "asscgd_ssim": asscgd["ssim"],
                "asscgd_reconstruction_seconds": seconds,
                "asscgd_measurement_residual": float(
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
            if not all(torch.isfinite(torch.tensor(value)) for value in numeric):
                raise FloatingPointError(f"nonfinite evaluation metric for {sample_id}")
            destination.write(json.dumps(row) + "\n")
            destination.flush()
            rows.append(row)
            completed.add(sample_id)
            print(
                f"[{len(completed)}/{expected_samples}] {sample_id} "
                f"ZF_PSNR={row['zero_filled_psnr']:.2f} "
                f"ASSCGD_PSNR={row['asscgd_psnr']:.2f} ASSCGD_SSIM={row['asscgd_ssim']:.4f}",
                flush=True,
            )

    if len(rows) != expected_samples:
        raise RuntimeError(f"expected {expected_samples} unique rows, found {len(rows)}")
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "completed_epochs": completed_epochs,
        "acceleration": 4,
        "split": args.split,
        "precision": args.precision,
        "noise_seed": args.noise_seed,
        "model_config": model_config,
        **summarize_evaluation(rows),
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps({**summary, "volumes": f"{len(summary['volumes'])} records", "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
