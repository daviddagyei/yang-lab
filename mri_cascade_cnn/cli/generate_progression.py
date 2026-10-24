"""Generate a representative D5-C5 cascade progression and RDBM comparison."""

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

from mri_cascade_cnn import build_cascade_cnn
from mri_data import MRISliceDataset, build_dataloader
from mri_model import build_mri_rdbm, reconstruction_metrics
from mri_reconstruction import channels_to_complex


ROOT = Path(__file__).resolve().parents[2]
CASCADE_ROOT = Path(
    "/data/david_agyei/nyu-dataset/derived/multicoil_cascade_cnn_4x"
)
RDBM_CHECKPOINT = Path(
    "/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16/checkpoint-0094900.pt"
)
OUTPUT = ROOT / "results" / "cascade_cnn_evaluation" / "qualitative"


def magnitude(image: torch.Tensor) -> np.ndarray:
    return torch.abs(image).float().cpu().numpy()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("figure generation requires the CUDA GPU")
    selection = json.loads(
        (CASCADE_ROOT / "evaluation/best_validation_checkpoint.json").read_text()
    )
    step = int(selection["checkpoint_step"])
    test_dir = CASCADE_ROOT / "evaluation/test" / f"step-{step:07d}"
    rows = [
        json.loads(line)
        for line in (test_dir / "slice_metrics.jsonl").read_text().splitlines()
        if line
    ]
    gains = np.array(
        [row["cascade_cnn_psnr"] - row["zero_filled_psnr"] for row in rows]
    )
    median_gain = float(np.median(gains))
    representative = min(
        rows,
        key=lambda row: abs(
            row["cascade_cnn_psnr"] - row["zero_filled_psnr"] - median_gain
        ),
    )
    sample_index = int(representative["sample_index"])
    dataset = MRISliceDataset(
        ROOT / "phase1/manifests/test.jsonl", split="test", acceleration=4
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

    checkpoint = torch.load(
        selection["checkpoint"], map_location="cpu", weights_only=True
    )
    cascade = build_cascade_cnn(**checkpoint["model_config"])
    cascade.load_state_dict(checkpoint["model"])
    cascade = cascade.to(device).eval()
    with torch.inference_mode():
        trajectory_channels = cascade(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            return_trajectory=True,
        )
    assert isinstance(trajectory_channels, list)
    trajectory = [
        channels_to_complex(frame, channel_dim=1)[0]
        for frame in trajectory_channels
    ]
    del cascade
    torch.cuda.empty_cache()

    rdbm_checkpoint = torch.load(
        RDBM_CHECKPOINT, map_location="cpu", weights_only=True
    )
    rdbm = build_mri_rdbm(**rdbm_checkpoint["model_config"])
    rdbm.load_state_dict(rdbm_checkpoint["model"])
    rdbm = rdbm.to(device).eval()
    torch.manual_seed(20261025)
    torch.cuda.manual_seed_all(20261025)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        rdbm_channels = rdbm.reconstruct(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            final_consistency_steps=4,
        )
    assert isinstance(rdbm_channels, torch.Tensor)
    rdbm_image = channels_to_complex(rdbm_channels, channel_dim=1)[0]
    target = channels_to_complex(batch["target"], channel_dim=1)[0]

    labels = ["Zero-filled", *[f"Cascade {index}" for index in range(1, 6)]]
    learned_images = trajectory + [rdbm_image]
    learned_labels = labels + ["RDBM"]
    metrics = {
        label: reconstruction_metrics(target, image)
        for label, image in zip(learned_labels, learned_images)
    }
    images = [magnitude(image) for image in learned_images] + [magnitude(target)]
    plot_labels = learned_labels + ["Target"]
    target_image = images[-1]
    errors = [np.abs(image - target_image) for image in images]
    vmax = max(float(np.quantile(target_image, 0.995)), 1e-8)
    error_vmax = max(float(np.quantile(np.stack(errors[:-1]), 0.995)), 1e-8)
    figure, axes = plt.subplots(2, len(images), figsize=(24, 6))
    for index, (image, error, label) in enumerate(
        zip(images, errors, plot_labels)
    ):
        metric = metrics.get(label)
        title = label if metric is None else f"{label}\n{metric['psnr']:.2f} dB"
        axes[0, index].imshow(image, cmap="gray", vmin=0.0, vmax=vmax)
        axes[0, index].set_title(title, fontsize=10)
        axes[1, index].imshow(
            error, cmap="magma", vmin=0.0, vmax=error_vmax
        )
        axes[1, index].set_title("Absolute error", fontsize=9)
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.suptitle(
        f"Adapted D5-C5 reconstruction across cascades | {sample_id}",
        fontsize=14,
    )
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT / "cascade_progression.png", dpi=160, bbox_inches="tight"
    )
    plt.close(figure)
    metadata = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "selection_reason": "test slice closest to median Cascade CNN PSNR gain",
        "selected_epoch": int(selection["completed_epochs"]),
        "selected_step": step,
        "rdbm_checkpoint_step": int(rdbm_checkpoint["step"]),
        "metrics": metrics,
    }
    (OUTPUT / "cascade_progression.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {**metadata, "output": str(OUTPUT / "cascade_progression.png")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
