"""Generate representative MoDL unroll and RDBM comparison figures."""

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
from mri_modl import build_modl
from mri_reconstruction import channels_to_complex


ROOT = Path(__file__).resolve().parents[2]
MODL_ROOT = Path("/data/david_agyei/nyu-dataset/derived/modl_4x_fp32")
RDBM_CHECKPOINT = Path(
    "/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16/checkpoint-0094900.pt"
)
OUTPUT = ROOT / "results" / "modl_evaluation" / "qualitative"


def choose_representative(metrics_path: Path) -> tuple[int, str]:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    gains = np.array([row["modl_psnr"] - row["zero_filled_psnr"] for row in rows])
    median_gain = float(np.median(gains))
    selected = min(
        rows,
        key=lambda row: abs(row["modl_psnr"] - row["zero_filled_psnr"] - median_gain),
    )
    return int(selected["sample_index"]), "test slice closest to the median MoDL PSNR gain"


def plot_grid(
    images: list[np.ndarray],
    labels: list[str],
    target: np.ndarray,
    destination: Path,
    title: str,
) -> None:
    vmax = max(float(np.quantile(target, 0.995)), 1e-8)
    errors = [np.abs(image - target) for image in images]
    error_max = max(float(np.quantile(np.stack(errors[:-1]), 0.995)), 1e-8)
    figure, axes = plt.subplots(2, len(images), figsize=(3.0 * len(images), 6.0))
    for column, (image, error, label) in enumerate(zip(images, errors, labels)):
        axes[0, column].imshow(image, cmap="gray", vmin=0.0, vmax=vmax)
        axes[0, column].set_title(label, fontsize=10)
        axes[1, column].imshow(error, cmap="magma", vmin=0.0, vmax=error_max)
        axes[1, column].set_title("Absolute error", fontsize=9)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("progression generation requires the CUDA GPU")
    device = torch.device("cuda")
    selection = json.loads((MODL_ROOT / "evaluation" / "best_validation_checkpoint.json").read_text())
    step = int(selection["checkpoint_step"])
    metrics_path = MODL_ROOT / "evaluation" / "test" / f"step-{step:07d}" / "slice_metrics.jsonl"
    sample_index, reason = choose_representative(metrics_path)
    dataset = MRISliceDataset(ROOT / "phase1" / "manifests" / "test.jsonl", split="test")
    batch = next(iter(build_dataloader(torch.utils.data.Subset(dataset, [sample_index]))))
    sample_id = str(batch["metadata"]["sample_id"][0])
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
    target = channels_to_complex(batch["target"], channel_dim=1)[0]

    modl_checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)
    modl = build_modl(**modl_checkpoint["model_config"])
    modl.load_state_dict(modl_checkpoint["model"])
    modl = modl.to(device).eval()
    with torch.inference_mode():
        trajectory_channels = modl(
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            return_trajectory=True,
        )
    assert isinstance(trajectory_channels, list)
    trajectory = [channels_to_complex(item, channel_dim=1)[0] for item in trajectory_channels]

    rdbm_checkpoint = torch.load(RDBM_CHECKPOINT, map_location="cpu", weights_only=True)
    rdbm = build_mri_rdbm(**rdbm_checkpoint["model_config"])
    rdbm.load_state_dict(rdbm_checkpoint["model"])
    rdbm = rdbm.to(device).eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        rdbm_channels = rdbm.reconstruct(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            final_consistency_steps=4,
        )
    assert isinstance(rdbm_channels, torch.Tensor)
    rdbm_complex = channels_to_complex(rdbm_channels, channel_dim=1)[0]

    metrics = {
        "zero_filled": reconstruction_metrics(target, initial),
        "modl": reconstruction_metrics(target, trajectory[-1]),
        "rdbm": reconstruction_metrics(target, rdbm_complex),
    }

    def magnitude(image: torch.Tensor) -> np.ndarray:
        return torch.abs(image).float().cpu().numpy()

    target_image = magnitude(target)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    indices = [0, 1, 2, 4, 6, 8, 10]
    plot_grid(
        [magnitude(trajectory[index]) for index in indices] + [target_image],
        ["Initial / Aᴴy" if index == 0 else f"Unroll {index}" for index in indices] + ["Target"],
        target_image,
        OUTPUT / "unroll_progression.png",
        f"MoDL reconstruction across unrolled iterations | {sample_id}",
    )
    plot_grid(
        [magnitude(initial), magnitude(rdbm_complex), magnitude(trajectory[-1]), target_image],
        [
            f"Zero-filled\nPSNR {metrics['zero_filled']['psnr']:.2f}",
            f"RDBM\nPSNR {metrics['rdbm']['psnr']:.2f}",
            f"MoDL\nPSNR {metrics['modl']['psnr']:.2f}",
            "Target",
        ],
        target_image,
        OUTPUT / "model_comparison.png",
        f"Matched 4× reconstruction comparison | {sample_id}",
    )
    metadata = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "selection_reason": reason,
        "modl_checkpoint_step": step,
        "rdbm_checkpoint_step": int(rdbm_checkpoint["step"]),
        "unroll_indices": indices,
        "metrics": metrics,
    }
    (OUTPUT / "progression_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**metadata, "output_dir": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
