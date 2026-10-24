"""Generate representative ASSCGD checkpoint and reverse-diffusion figures."""

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

from mri_asscgd.cli.evaluate_split import sample_noise
from mri_asscgd import build_asscgd_diffusion
from mri_data import MRISliceDataset, build_dataloader
from mri_model import reconstruction_metrics
from mri_reconstruction import channels_to_complex


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = Path("/data/david_agyei/nyu-dataset/derived/asscgd_4x_stable")
OUTPUT = ROOT / "results" / "asscgd_evaluation" / "qualitative"
NOISE_SEED = 20261021


def choose_representative(metrics_path: Path) -> tuple[int, str]:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    gains = np.array([row["asscgd_psnr"] - row["zero_filled_psnr"] for row in rows])
    median_gain = float(np.median(gains))
    selected = min(
        rows,
        key=lambda row: abs(row["asscgd_psnr"] - row["zero_filled_psnr"] - median_gain),
    )
    return int(selected["sample_index"]), "test slice closest to median ASSCGD PSNR gain"


def magnitude(image: torch.Tensor) -> np.ndarray:
    return torch.abs(image).float().cpu().numpy()


def plot_grid(
    images: list[np.ndarray], labels: list[str], target: np.ndarray, destination: Path, title: str
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
    selection = json.loads(
        (CHECKPOINT_ROOT / "evaluation" / "best_validation_checkpoint.json").read_text()
    )
    selected_step = int(selection["checkpoint_step"])
    metrics_path = (
        CHECKPOINT_ROOT / "evaluation" / "test" / f"step-{selected_step:07d}" /
        "slice_metrics.jsonl"
    )
    sample_index, reason = choose_representative(metrics_path)
    dataset = MRISliceDataset(
        ROOT / "phase1" / "manifests" / "test.jsonl", split="test", acceleration=4
    )
    batch = next(iter(build_dataloader(torch.utils.data.Subset(dataset, [sample_index]))))
    sample_id = str(batch["metadata"]["sample_id"][0])
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
    target = channels_to_complex(batch["target"], channel_dim=1)[0]
    noise, sample_seed = sample_noise(batch["initial_reconstruction"], sample_id, NOISE_SEED)
    target_image = magnitude(target)

    candidates = sorted(selection["candidates"], key=lambda row: row["checkpoint_step"])
    checkpoint_outputs: list[np.ndarray] = []
    metrics: dict[str, dict[str, float]] = {
        "zero_filled": reconstruction_metrics(target, initial)
    }
    selected_trajectory: list[np.ndarray] | None = None
    for candidate in candidates:
        step = int(candidate["checkpoint_step"])
        checkpoint = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=True)
        model = build_asscgd_diffusion(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model = model.to(device).eval()
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), autocast:
            result = model.reconstruct(
                batch["initial_reconstruction"], batch["undersampled_kspace"],
                batch["sensitivity_maps"], batch["mask"], noise=noise,
                return_trajectory=step == selected_step,
            )
        if step == selected_step:
            assert isinstance(result, list)
            final = result[-1]
            selected_trajectory = [
                magnitude(channels_to_complex(frame, channel_dim=1)[0]) for frame in result
            ]
        else:
            assert isinstance(result, torch.Tensor)
            final = result
        reconstructed = channels_to_complex(final, channel_dim=1)[0]
        checkpoint_outputs.append(magnitude(reconstructed))
        metrics[f"step_{step}"] = reconstruction_metrics(target, reconstructed)
        del model, result, final, reconstructed
        torch.cuda.empty_cache()

    if selected_trajectory is None:
        raise RuntimeError("selected checkpoint is absent from validation candidates")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint_images = [magnitude(initial), *checkpoint_outputs, target_image]
    checkpoint_labels = [
        f"Zero-filled\nPSNR {metrics['zero_filled']['psnr']:.2f}",
        *[
            "Epoch {}\nPSNR {:.2f}".format(
                item["completed_epochs"],
                metrics[f"step_{item['checkpoint_step']}"]["psnr"],
            )
            for item in candidates
        ],
        "Target",
    ]
    plot_grid(
        checkpoint_images, checkpoint_labels, target_image,
        OUTPUT / "checkpoint_progression.png",
        f"ASSCGD training-checkpoint progression | {sample_id}",
    )

    indices = sorted(set(np.linspace(0, len(selected_trajectory) - 1, 7, dtype=int).tolist()))
    trajectory_images = [selected_trajectory[index] for index in indices] + [target_image]
    trajectory_labels = [
        "Initial / zero-filled" if index == 0 else (
            "Final reconstruction" if index == len(selected_trajectory) - 1
            else f"Reverse stage {index}"
        )
        for index in indices
    ] + ["Target"]
    plot_grid(
        trajectory_images, trajectory_labels, target_image,
        OUTPUT / "reverse_diffusion_progression.png",
        f"ASSCGD reverse-diffusion progression | {sample_id}",
    )
    metadata = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "selection_reason": reason,
        "noise_seed": NOISE_SEED,
        "sample_noise_seed": sample_seed,
        "selected_checkpoint_step": selected_step,
        "checkpoint_steps": [item["checkpoint_step"] for item in candidates],
        "trajectory_indices": indices,
        "metrics": metrics,
    }
    (OUTPUT / "progression_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**metadata, "metrics": f"{len(metrics)} records", "output_dir": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
