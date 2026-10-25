"""Generate matched full diffusion-unrolled RDBM stage progressions."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mri-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mri_data import MRISliceDataset, build_dataloader
from mri_diffusion_unrolled_rdbm import build_diffusion_unrolled_rdbm
from mri_model import reconstruction_metrics
from mri_reconstruction import channels_to_complex


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MRI_DATA_ROOT", "/data/david_agyei/nyu-dataset"))
OUTPUT_ROOT = Path(os.environ.get("FULL_UNROLLED_OUTPUT_ROOT", DATA_ROOT / "derived"))
OUTPUT = ROOT / "results/full_diffusion_unrolled_rdbm/qualitative"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def report_rates() -> tuple[int, ...]:
    rates = tuple(int(value) for value in os.environ.get("FULL_UNROLLED_REPORT_RATES", "2 4").split())
    if not rates or any(rate not in (2, 4, 6, 8) for rate in rates):
        raise ValueError("FULL_UNROLLED_REPORT_RATES must contain 2, 4, 6, or 8")
    return rates


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("qualitative generation requires CUDA")
    rates = report_rates()
    selection_rate = 4 if 4 in rates else rates[0]
    reference_run = OUTPUT_ROOT / f"full_diffusion_unrolled_rdbm_{selection_rate}x_scratch"
    reference_selection = read_json(reference_run / "evaluation/best_validation_checkpoint.json")
    reference_test = reference_run / "evaluation/test" / f"step-{reference_selection['checkpoint_step']:07d}"
    reference_rows = [
        json.loads(line)
        for line in (reference_test / "slice_metrics.jsonl").read_text().splitlines()
        if line
    ]
    gains = np.array([
        row["diffusion_unrolled_rdbm_psnr"] - row["zero_filled_psnr"]
        for row in reference_rows
    ])
    median_gain = float(np.median(gains))
    representative = min(
        reference_rows,
        key=lambda row: abs(
            row["diffusion_unrolled_rdbm_psnr"]
            - row["zero_filled_psnr"]
            - median_gain
        ),
    )
    sample_index = int(representative["sample_index"])
    device = torch.device("cuda")
    selected_indices = (0, 2, 4, 6, 8, 10)
    labels = ("Zero-filled", "Stage 2", "Stage 4", "Stage 6", "Stage 8", "Stage 10")
    metadata: dict[str, object] = {
        "sample_index": sample_index,
        "sample_id": representative["sample_id"],
        "selection_reason": f"test slice closest to median {selection_rate}x PSNR gain",
        "rates": {},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for rate in rates:
        run = OUTPUT_ROOT / f"full_diffusion_unrolled_rdbm_{rate}x_scratch"
        selection = read_json(run / "evaluation/best_validation_checkpoint.json")
        checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)
        data_config = checkpoint["data_config"]
        dataset = MRISliceDataset(
            ROOT / "phase1/manifests/test.jsonl",
            split="test",
            acceleration=int(data_config["acceleration"]),
            center_fraction=float(data_config["center_fraction"]),
            mask_seed=int(data_config["mask_seed"]),
            normalization_quantile=float(data_config.get("normalization_quantile", 0.99)),
            data_root=DATA_ROOT,
        )
        batch = next(iter(build_dataloader(torch.utils.data.Subset(dataset, [sample_index]))))
        sample_id = str(batch["metadata"]["sample_id"][0])
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        model = build_diffusion_unrolled_rdbm(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model = model.to(device).eval()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_bf16_supported()
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            trajectory_channels = model(
                batch["initial_reconstruction"],
                batch["undersampled_kspace"],
                batch["sensitivity_maps"],
                batch["mask"],
                return_trajectory=True,
            )
        assert isinstance(trajectory_channels, list)
        trajectory = [
            channels_to_complex(frame.float(), channel_dim=1)[0]
            for frame in trajectory_channels
        ]
        target = channels_to_complex(batch["target"], channel_dim=1)[0]
        selected = [trajectory[index] for index in selected_indices]
        metrics = {
            label: reconstruction_metrics(target, image)
            for label, image in zip(labels, selected)
        }
        images = [torch.abs(image).float().cpu().numpy() for image in selected]
        target_magnitude = torch.abs(target).float().cpu().numpy()
        images.append(target_magnitude)
        plot_labels = (*labels, "Target")
        errors = [np.abs(image - target_magnitude) for image in images]
        vmax = max(float(np.quantile(target_magnitude, 0.995)), 1e-8)
        error_vmax = max(float(np.quantile(np.stack(errors[:-1]), 0.995)), 1e-8)
        figure, axes = plt.subplots(2, len(images), figsize=(21, 6))
        for index, (image, error, label) in enumerate(zip(images, errors, plot_labels)):
            metric = metrics.get(label)
            title = label if metric is None else f"{label}\n{metric['psnr']:.2f} dB"
            axes[0, index].imshow(image, cmap="gray", vmin=0.0, vmax=vmax)
            axes[0, index].set_title(title, fontsize=10)
            axes[1, index].imshow(error, cmap="magma", vmin=0.0, vmax=error_vmax)
            axes[1, index].set_title("Absolute error", fontsize=9)
            axes[0, index].axis("off")
            axes[1, index].axis("off")
        figure.suptitle(f"Full diffusion-unrolled RDBM at {rate}× | {sample_id}", fontsize=14)
        figure.tight_layout()
        figure.savefig(OUTPUT / f"progression_{rate}x.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        metadata["rates"][str(rate)] = {
            "selected_epoch": int(selection["completed_epochs"]),
            "selected_step": int(selection["checkpoint_step"]),
            "metrics": metrics,
        }
        del model, checkpoint, batch, trajectory_channels, trajectory
        torch.cuda.empty_cache()

    (OUTPUT / "progression.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
