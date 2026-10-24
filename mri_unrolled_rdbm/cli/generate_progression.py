"""Generate a representative cascade comparison for the selected unrolled RDBM."""

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
from mri_model import build_mri_rdbm, reconstruction_metrics
from mri_reconstruction import channels_to_complex
from mri_unrolled_rdbm import build_unrolled_rdbm


ROOT = Path(__file__).resolve().parents[2]
UNROLLED_ROOT = Path("/data/david_agyei/nyu-dataset/derived/unrolled_rdbm_4x_scratch")
RDBM_CHECKPOINT = Path(
    "/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16/checkpoint-0094900.pt"
)
OUTPUT = ROOT / "results" / "unrolled_rdbm_evaluation" / "qualitative"


def magnitude(image: torch.Tensor) -> np.ndarray:
    return torch.abs(image).float().cpu().numpy()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("figure generation requires CUDA")
    selection = json.loads(
        (UNROLLED_ROOT / "evaluation" / "best_validation_checkpoint.json").read_text()
    )
    step = int(selection["checkpoint_step"])
    test_dir = UNROLLED_ROOT / "evaluation" / "test" / f"step-{step:07d}"
    rows = [
        json.loads(line)
        for line in (test_dir / "slice_metrics.jsonl").read_text().splitlines()
        if line
    ]
    median_gain = float(
        np.median([row["unrolled_rdbm_psnr"] - row["zero_filled_psnr"] for row in rows])
    )
    representative = min(
        rows,
        key=lambda row: abs(
            row["unrolled_rdbm_psnr"] - row["zero_filled_psnr"] - median_gain
        ),
    )
    sample_index = int(representative["sample_index"])
    dataset = MRISliceDataset(
        ROOT / "phase1" / "manifests" / "test.jsonl", split="test", acceleration=4
    )
    batch = next(iter(build_dataloader(torch.utils.data.Subset(dataset, [sample_index]))))
    sample_id = str(batch["metadata"]["sample_id"][0])
    device = torch.device("cuda")
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)
    unrolled = build_unrolled_rdbm(**checkpoint["model_config"])
    unrolled.load_state_dict(checkpoint["model"])
    unrolled = unrolled.to(device).eval()
    rdbm_checkpoint = torch.load(RDBM_CHECKPOINT, map_location="cpu", weights_only=True)
    rdbm = build_mri_rdbm(**rdbm_checkpoint["model_config"])
    rdbm.load_state_dict(rdbm_checkpoint["model"])
    rdbm = rdbm.to(device).eval()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        trajectory = unrolled(
            batch["initial_reconstruction"], batch["undersampled_kspace"],
            batch["sensitivity_maps"], batch["mask"], return_trajectory=True,
        )
        rdbm_output = rdbm.reconstruct(
            batch["initial_reconstruction"], batch["undersampled_kspace"],
            batch["sensitivity_maps"], batch["mask"],
        )
    assert isinstance(trajectory, list) and isinstance(rdbm_output, torch.Tensor)
    target = channels_to_complex(batch["target"], channel_dim=1)[0]
    complex_images = [
        channels_to_complex(frame, channel_dim=1)[0] for frame in trajectory
    ] + [channels_to_complex(rdbm_output, channel_dim=1)[0], target]
    labels = ["Zero-filled", "Cascade 1", "Cascade 2", "Cascade 3", "RDBM", "Target"]
    metrics = {
        label: reconstruction_metrics(target, image)
        for label, image in zip(labels[:-1], complex_images[:-1])
    }
    images = [magnitude(image) for image in complex_images]
    target_image = images[-1]
    errors = [np.abs(image - target_image) for image in images]
    vmax = max(float(np.quantile(target_image, 0.995)), 1e-8)
    error_vmax = max(float(np.quantile(np.stack(errors[:-1]), 0.995)), 1e-8)
    figure, axes = plt.subplots(2, len(images), figsize=(18, 6))
    for index, (image, error, label) in enumerate(zip(images, errors, labels)):
        metric = metrics.get(label)
        title = label if metric is None else f"{label}\n{metric['psnr']:.2f} dB"
        axes[0, index].imshow(image, cmap="gray", vmin=0, vmax=vmax)
        axes[0, index].set_title(title)
        axes[1, index].imshow(error, cmap="magma", vmin=0, vmax=error_vmax)
        axes[1, index].set_title("Absolute error")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.suptitle(f"Unrolled RDBM cascade progression | {sample_id}", fontsize=14)
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "cascade_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "selection_reason": "test slice closest to median unrolled PSNR gain",
        "selected_epoch": int(selection["completed_epochs"]),
        "selected_step": step,
        "metrics": metrics,
    }
    (OUTPUT / "cascade_comparison.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**metadata, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
