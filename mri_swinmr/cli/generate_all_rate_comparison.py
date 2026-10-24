"""Generate a matched qualitative SwinMR comparison across accelerations."""

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
from mri_swinmr import build_swinmr


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MRI_DATA_ROOT", "/data/david_agyei/nyu-dataset"))
OUTPUT_ROOT = Path(os.environ.get("SWINMR_OUTPUT_ROOT", DATA_ROOT / "derived"))
OUTPUT = ROOT / "results/swinmr_evaluation/qualitative"
RATES = (2, 4, 6, 8)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def magnitude(image: torch.Tensor) -> np.ndarray:
    return torch.abs(image).float().cpu().numpy()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SwinMR figure generation requires CUDA")
    run4 = OUTPUT_ROOT / "swinmr_pi_complex_4x"
    selection4 = read_json(run4 / "evaluation/best_validation_checkpoint.json")
    test4 = run4 / "evaluation/test" / f"step-{selection4['checkpoint_step']:07d}"
    rows4 = [
        json.loads(line)
        for line in (test4 / "slice_metrics.jsonl").read_text().splitlines()
        if line
    ]
    gains = np.array([row["swinmr_psnr"] - row["zero_filled_psnr"] for row in rows4])
    median_gain = float(np.median(gains))
    representative = min(
        rows4,
        key=lambda row: abs(
            row["swinmr_psnr"] - row["zero_filled_psnr"] - median_gain
        ),
    )
    sample_index = int(representative["sample_index"])
    device = torch.device("cuda")
    images: list[np.ndarray] = []
    labels: list[str] = []
    metrics: dict[str, dict[str, float]] = {}
    target_image: torch.Tensor | None = None
    sample_id = ""

    for rate in RATES:
        run = OUTPUT_ROOT / f"swinmr_pi_complex_{rate}x"
        selection = read_json(run / "evaluation/best_validation_checkpoint.json")
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
            data_root=DATA_ROOT,
        )
        batch = next(
            iter(build_dataloader(torch.utils.data.Subset(dataset, [sample_index])))
        )
        sample_id = str(batch["metadata"]["sample_id"][0])
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        model = build_swinmr(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["ema_model"])
        model = model.to(device).eval()
        with torch.inference_mode():
            prediction = model(batch["initial_reconstruction"])
        initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
        reconstructed = channels_to_complex(prediction, channel_dim=1)[0]
        target_image = channels_to_complex(batch["target"], channel_dim=1)[0]
        for label, image in ((f"ZF {rate}×", initial), (f"SwinMR {rate}×", reconstructed)):
            images.append(magnitude(image))
            labels.append(label)
            metrics[label] = reconstruction_metrics(target_image, image)
        del model, checkpoint, batch
        torch.cuda.empty_cache()

    assert target_image is not None
    target = magnitude(target_image)
    images.append(target)
    labels.append("Target")
    errors = [np.abs(image - target) for image in images]
    vmax = max(float(np.quantile(target, 0.995)), 1e-8)
    error_vmax = max(float(np.quantile(np.stack(errors[:-1]), 0.995)), 1e-8)
    figure, axes = plt.subplots(2, len(images), figsize=(27, 6))
    for index, (image, error, label) in enumerate(zip(images, errors, labels)):
        metric = metrics.get(label)
        title = label if metric is None else f"{label}\n{metric['psnr']:.2f} dB"
        axes[0, index].imshow(image, cmap="gray", vmin=0.0, vmax=vmax)
        axes[0, index].set_title(title, fontsize=10)
        axes[1, index].imshow(error, cmap="magma", vmin=0.0, vmax=error_vmax)
        axes[1, index].set_title("Absolute error", fontsize=9)
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.suptitle(f"SwinMR across acceleration rates | {sample_id}", fontsize=14)
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "all_rate_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "selection_reason": "test slice closest to median 4x SwinMR PSNR gain",
        "metrics": metrics,
    }
    (OUTPUT / "all_rate_comparison.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
