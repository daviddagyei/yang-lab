"""Generate a matched qualitative comparison of final RDBM acceleration models."""

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


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/data/david_agyei/nyu-dataset/derived")
OUTPUT_DIR = ROOT / "results" / "rdbm_evaluation" / "acceleration_comparison"
CHECKPOINT_STEP = 94900
RATES = (2, 4, 6, 8)


def checkpoint_root(rate: int) -> Path:
    return DATA_ROOT / ("rdbm_full_bf16" if rate == 4 else f"rdbm_{rate}x_bf16")


def representative_index() -> tuple[int, str]:
    existing = ROOT / "results" / "rdbm_evaluation" / "qualitative" / "progression_metadata.json"
    if existing.exists():
        metadata = json.loads(existing.read_text())
        return int(metadata["sample_index"]), "same representative slice as the 4x notebook"
    rows_path = (
        checkpoint_root(4)
        / "evaluation"
        / "test"
        / f"step-{CHECKPOINT_STEP:07d}"
        / "slice_metrics.jsonl"
    )
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    gains = np.asarray([row["rdbm_psnr"] - row["zero_filled_psnr"] for row in rows])
    median = float(np.median(gains))
    selected = min(
        rows,
        key=lambda row: abs(row["rdbm_psnr"] - row["zero_filled_psnr"] - median),
    )
    return int(selected["sample_index"]), "test slice closest to median 4x PSNR gain"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("comparison generation requires a BF16-capable CUDA GPU")

    sample_index, selection_reason = representative_index()
    manifest = ROOT / "phase1" / "manifests" / "test.jsonl"
    images: dict[int, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    sample_id: str | None = None
    target_image: np.ndarray | None = None

    for rate in RATES:
        dataset = MRISliceDataset(manifest, split="test", acceleration=rate)
        batch = next(
            iter(build_dataloader(torch.utils.data.Subset(dataset, [sample_index])))
        )
        current_sample_id = str(batch["metadata"]["sample_id"][0])
        if sample_id is None:
            sample_id = current_sample_id
        elif current_sample_id != sample_id:
            raise RuntimeError("sample order differs across acceleration datasets")
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        checkpoint_path = checkpoint_root(rate) / f"checkpoint-{CHECKPOINT_STEP:07d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if int(checkpoint.get("data_config", {}).get("acceleration", 4)) != rate:
            raise ValueError(f"checkpoint acceleration mismatch: {checkpoint_path}")
        model = build_mri_rdbm(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model = model.to(device).eval()
        del checkpoint

        torch.manual_seed(20261021)
        torch.cuda.manual_seed_all(20261021)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), autocast:
            reconstruction = model.reconstruct(
                batch["initial_reconstruction"],
                batch["undersampled_kspace"],
                batch["sensitivity_maps"],
                batch["mask"],
                final_consistency_steps=4,
            )
        initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
        target = channels_to_complex(batch["target"], channel_dim=1)[0]
        reconstructed = channels_to_complex(reconstruction, channel_dim=1)[0]
        scale = batch["scale"].reshape(-1)[0]
        initial_physical = initial * scale
        target_physical = target * scale
        reconstructed_physical = reconstructed * scale
        images[rate] = {
            "zero_filled": torch.abs(initial_physical).float().cpu().numpy(),
            "rdbm": torch.abs(reconstructed_physical).float().cpu().numpy(),
        }
        target_image = torch.abs(target_physical).float().cpu().numpy()
        metrics[str(rate)] = {
            "zero_filled": reconstruction_metrics(target_physical, initial_physical),
            "rdbm": reconstruction_metrics(target_physical, reconstructed_physical),
        }
        del model, reconstruction, batch
        torch.cuda.empty_cache()
        print(f"generated {rate}x", flush=True)

    assert sample_id is not None and target_image is not None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vmax = max(float(np.quantile(target_image, 0.995)), 1e-8)
    errors = [np.abs(images[rate]["rdbm"] - target_image) for rate in RATES]
    error_max = max(float(np.quantile(np.stack(errors), 0.995)), 1e-8)

    figure = plt.figure(figsize=(16, 9))
    grid = figure.add_gridspec(3, 5, width_ratios=(1, 1, 1, 1, 1.08))
    for column, rate in enumerate(RATES):
        zero_axis = figure.add_subplot(grid[0, column])
        model_axis = figure.add_subplot(grid[1, column])
        error_axis = figure.add_subplot(grid[2, column])
        zero_axis.imshow(images[rate]["zero_filled"], cmap="gray", vmin=0, vmax=vmax)
        model_axis.imshow(images[rate]["rdbm"], cmap="gray", vmin=0, vmax=vmax)
        error_axis.imshow(errors[column], cmap="magma", vmin=0, vmax=error_max)
        zero_axis.set_title(
            f"{rate}x | ZF {metrics[str(rate)]['zero_filled']['psnr']:.2f} dB",
            fontsize=10,
        )
        model_axis.set_title(
            f"RDBM {metrics[str(rate)]['rdbm']['psnr']:.2f} dB",
            fontsize=10,
        )
        error_axis.set_title("RDBM absolute error", fontsize=10)
        for axis in (zero_axis, model_axis, error_axis):
            axis.axis("off")
    target_axis = figure.add_subplot(grid[:2, 4])
    target_axis.imshow(target_image, cmap="gray", vmin=0, vmax=vmax)
    target_axis.set_title("Fully sampled target", fontsize=11)
    target_axis.axis("off")
    text_axis = figure.add_subplot(grid[2, 4])
    text_axis.axis("off")
    text_axis.text(
        0.0,
        1.0,
        "Matched slice\nSame 94,900-step budget\nSame diffusion seed\nShared image/error scales",
        va="top",
        fontsize=11,
        linespacing=1.5,
    )
    figure.suptitle(f"RDBM reconstruction across acceleration rates | {sample_id}", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "rdbm_acceleration_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    np.savez_compressed(
        OUTPUT_DIR / "rdbm_acceleration_images.npz",
        target=target_image,
        **{
            f"{name}_{rate}x": images[rate][name]
            for rate in RATES
            for name in ("zero_filled", "rdbm")
        },
    )
    metadata = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "selection_reason": selection_reason,
        "checkpoint_step": CHECKPOINT_STEP,
        "accelerations": list(RATES),
        "metrics": metrics,
    }
    (OUTPUT_DIR / "rdbm_acceleration_comparison.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps({**metadata, "metrics": "stored", "output_dir": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
