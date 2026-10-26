"""Generate a representative Variational Network stage progression."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mri-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mri_data import MRISliceDataset, build_dataloader
from mri_model import reconstruction_metrics
from mri_reconstruction import channels_to_complex
from mri_variational_network import build_variational_network


ROOT = Path(__file__).resolve().parents[2]
RUN = Path(
    os.environ.get(
        "VN_CHECKPOINT_DIR",
        "/data/david_agyei/nyu-dataset/derived/hammernik_variational_network_4x_ipalm",
    )
)
OUTPUT = ROOT / "results/variational_network_evaluation/qualitative"


def magnitude(image: torch.Tensor) -> np.ndarray:
    return torch.abs(image).float().cpu().numpy()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("figure generation requires CUDA")
    selection = json.loads(
        (RUN / "evaluation/best_validation_checkpoint.json").read_text()
    )
    step = int(selection["checkpoint_step"])
    test_dir = RUN / "evaluation/test" / f"step-{step:07d}"
    rows = [
        json.loads(line)
        for line in (test_dir / "slice_metrics.jsonl").read_text().splitlines()
        if line
    ]
    gains = np.array(
        [row["variational_network_psnr"] - row["zero_filled_psnr"] for row in rows]
    )
    median_gain = float(np.median(gains))
    representative = min(
        rows,
        key=lambda row: abs(
            row["variational_network_psnr"]
            - row["zero_filled_psnr"]
            - median_gain
        ),
    )
    sample_index = int(representative["sample_index"])

    checkpoint = torch.load(
        selection["checkpoint"], map_location="cpu", weights_only=True
    )
    data_config = checkpoint["data_config"]
    dataset = MRISliceDataset(
        ROOT / "phase1/manifests/test.jsonl",
        split="test",
        normalization_quantile=float(data_config["normalization_quantile"]),
        acceleration=int(data_config["acceleration"]),
        center_fraction=float(data_config["center_fraction"]),
        mask_seed=int(data_config["mask_seed"]),
    )
    batch = next(
        iter(build_dataloader(torch.utils.data.Subset(dataset, [sample_index])))
    )
    sample_id = str(batch["metadata"]["sample_id"][0])
    device = torch.device("cuda")
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    model = build_variational_network(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    with torch.inference_mode():
        trajectory_channels = model(
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            initial_reconstruction=batch["initial_reconstruction"],
            return_trajectory=True,
        )
    assert isinstance(trajectory_channels, list)
    trajectory = [
        channels_to_complex(frame, channel_dim=1)[0]
        for frame in trajectory_channels
    ]
    target = channels_to_complex(batch["target"], channel_dim=1)[0]

    selected_indices = (0, 2, 4, 6, 8, 10)
    labels = ["Zero-filled", "Stage 2", "Stage 4", "Stage 6", "Stage 8", "Stage 10"]
    selected_images = [trajectory[index] for index in selected_indices]
    metrics = {
        label: reconstruction_metrics(target, image)
        for label, image in zip(labels, selected_images)
    }
    images = [magnitude(image) for image in selected_images] + [magnitude(target)]
    plot_labels = labels + ["Target"]
    target_image = images[-1]
    errors = [np.abs(image - target_image) for image in images]
    vmax = max(float(np.quantile(target_image, 0.995)), 1e-8)
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
    figure.suptitle(
        f"Hammernik Variational Network at 4× | {sample_id}", fontsize=14
    )
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "stage_progression.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "selection_reason": "test slice closest to median VarNet PSNR gain",
        "selected_epoch": int(selection["completed_epochs"]),
        "selected_step": step,
        "metrics": metrics,
    }
    (OUTPUT / "stage_progression.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {**metadata, "output": str(OUTPUT / "stage_progression.png")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
