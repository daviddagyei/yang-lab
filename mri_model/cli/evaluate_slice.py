"""Reconstruct one MRI slice from a checkpoint and save metrics and a comparison image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mri-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from mri_data import MRISliceDataset, build_dataloader
from mri_model import build_mri_rdbm
from mri_reconstruction import channels_to_complex, measurement_residual


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "mri_rdbm")
    parser.add_argument("--final-consistency-steps", type=int, default=4)
    parser.add_argument("--acceleration", type=int, choices=(2, 4, 6, 8))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_config" not in checkpoint:
        raise ValueError("checkpoint lacks model configuration; create it with the current trainer")
    model = build_mri_rdbm(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    checkpoint_acceleration = int(checkpoint.get("data_config", {}).get("acceleration", 4))
    acceleration = args.acceleration or checkpoint_acceleration
    if args.acceleration is not None and args.acceleration != checkpoint_acceleration:
        raise ValueError("requested acceleration differs from checkpoint data configuration")

    manifest = PROJECT_ROOT / "phase1" / "manifests" / f"{args.split}.jsonl"
    dataset = MRISliceDataset(manifest, split=args.split, acceleration=acceleration)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    batch = next(iter(build_dataloader(torch.utils.data.Subset(dataset, [args.sample_index]))))
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        reconstruction = model.reconstruct(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            final_consistency_steps=args.final_consistency_steps,
        )

    initial_complex = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)
    reconstructed_complex = channels_to_complex(reconstruction, channel_dim=1)
    target_complex = channels_to_complex(batch["target"], channel_dim=1)
    initial_magnitude = torch.abs(initial_complex)[0]
    reconstructed_magnitude = torch.abs(reconstructed_complex)[0]
    target_magnitude = torch.abs(target_complex)[0]

    target_np = target_magnitude.detach().cpu().numpy()
    reconstructed_np = reconstructed_magnitude.detach().cpu().numpy()
    initial_np = initial_magnitude.detach().cpu().numpy()
    data_range = max(float(target_np.max() - target_np.min()), 1e-8)
    metrics = {
        "checkpoint_step": int(checkpoint["step"]),
        "acceleration": acceleration,
        "split": args.split,
        "sample_id": batch["metadata"]["sample_id"][0],
        "nmse": float(
            torch.sum(torch.abs(reconstructed_complex - target_complex) ** 2)
            / torch.sum(torch.abs(target_complex) ** 2).clamp_min(1e-12)
        ),
        "psnr": float(peak_signal_noise_ratio(target_np, reconstructed_np, data_range=data_range)),
        "ssim": float(structural_similarity(target_np, reconstructed_np, data_range=data_range)),
        "initial_measurement_residual": float(
            measurement_residual(
                initial_complex,
                batch["undersampled_kspace"],
                batch["sensitivity_maps"],
                batch["mask"],
            )
        ),
        "reconstruction_measurement_residual": float(
            measurement_residual(
                reconstructed_complex,
                batch["undersampled_kspace"],
                batch["sensitivity_maps"],
                batch["mask"],
            )
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = metrics["sample_id"].replace(":", "_step")
    metrics_path = args.output_dir / f"{stem}_metrics.json"
    image_path = args.output_dir / f"{stem}_comparison.png"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    vmax = max(float(np.quantile(target_np, 0.995)), 1e-8)
    error = np.abs(reconstructed_np - target_np)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = (
        (initial_np, "Zero-filled SENSE", "gray", 0.0, vmax),
        (reconstructed_np, "RDBM", "gray", 0.0, vmax),
        (target_np, "Target", "gray", 0.0, vmax),
        (error, "Absolute error", "magma", 0.0, max(float(np.quantile(error, 0.995)), 1e-8)),
    )
    for axis, (image, title, cmap, vmin, panel_max) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=panel_max)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"{metrics['sample_id']} | PSNR {metrics['psnr']:.2f} dB | SSIM {metrics['ssim']:.4f}")
    figure.tight_layout()
    figure.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps({**metrics, "metrics_path": str(metrics_path), "image_path": str(image_path)}, indent=2))


if __name__ == "__main__":
    main()
