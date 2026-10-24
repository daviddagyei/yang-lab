"""Generate checkpoint and reverse-diffusion reconstruction progression figures."""

from __future__ import annotations

import argparse
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
from mri_model import build_mri_rdbm, reconstruction_metrics
from mri_reconstruction import channels_to_complex


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = Path("/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16")


def format_step(step: int) -> str:
    return f"{step / 1000:g}k"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "rdbm_evaluation" / "qualitative",
    )
    parser.add_argument("--sample-index", type=int)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="Checkpoints to display. Defaults to a sparse trajectory ending at the selected step.",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--final-consistency-steps", type=int, default=4)
    return parser.parse_args()


def choose_representative_index(metrics_path: Path) -> tuple[int, str]:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    gains = np.array([row["rdbm_psnr"] - row["zero_filled_psnr"] for row in rows])
    median_gain = float(np.median(gains))
    selected = min(rows, key=lambda row: abs((row["rdbm_psnr"] - row["zero_filled_psnr"]) - median_gain))
    return int(selected["sample_index"]), "median test-slice PSNR gain"


def plot_progression(
    images: list[np.ndarray],
    labels: list[str],
    target: np.ndarray,
    destination: Path,
    title: str,
) -> None:
    vmax = max(float(np.quantile(target, 0.995)), 1e-8)
    errors = [np.abs(image - target) for image in images]
    nonzero_errors = errors[:-1] if labels[-1] == "Target" else errors
    error_max = max(float(np.quantile(np.stack(nonzero_errors), 0.995)), 1e-8)
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
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("progression generation requires the CUDA GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU does not support BF16; use --precision fp32")

    evaluation_root = args.checkpoint_dir / "evaluation"
    selection = json.loads((evaluation_root / "best_validation_checkpoint.json").read_text())
    selected_step = int(selection["checkpoint_step"])
    checkpoint_steps = (
        list(args.steps)
        if args.steps is not None
        else sorted({10000, 30000, 50000, 70000, selected_step})
    )
    if selected_step not in checkpoint_steps:
        raise ValueError("--steps must include the selected validation checkpoint")
    metrics_path = evaluation_root / "test" / f"step-{selected_step:07d}" / "slice_metrics.jsonl"
    if args.sample_index is None:
        sample_index, selection_reason = choose_representative_index(metrics_path)
    else:
        sample_index, selection_reason = args.sample_index, "explicit sample index"

    dataset = MRISliceDataset(PROJECT_ROOT / "phase1" / "manifests" / "test.jsonl", split="test")
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset)})")
    batch = next(iter(build_dataloader(torch.utils.data.Subset(dataset, [sample_index]))))
    sample_id = str(batch["metadata"]["sample_id"][0])
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)[0]
    target = channels_to_complex(batch["target"], channel_dim=1)[0]
    target_magnitude = torch.abs(target).float().cpu().numpy()
    initial_magnitude = torch.abs(initial).float().cpu().numpy()

    outputs: dict[int, np.ndarray] = {}
    no_dc_magnitude: np.ndarray | None = None
    metrics: dict[str, dict[str, float]] = {
        "zero_filled": reconstruction_metrics(target, initial)
    }
    final_trajectory: list[np.ndarray] | None = None
    for step in checkpoint_steps:
        checkpoint_path = args.checkpoint_dir / f"checkpoint-{step:07d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_mri_rdbm(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        model = model.to(device).eval()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.precision == "bf16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            result = model.reconstruct(
                batch["initial_reconstruction"],
                batch["undersampled_kspace"],
                batch["sensitivity_maps"],
                batch["mask"],
                final_consistency_steps=args.final_consistency_steps,
                return_trajectory=step == selected_step,
            )
        if step == selected_step:
            assert isinstance(result, list)
            final_tensor = result[-1]
            final_trajectory = [
                torch.abs(channels_to_complex(frame, channel_dim=1)[0]).float().cpu().numpy()
                for frame in result
            ]
            no_dc_autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.precision == "bf16"
                else nullcontext()
            )
            with torch.inference_mode(), no_dc_autocast:
                no_dc_tensor = model.reconstruct(
                    batch["initial_reconstruction"],
                    batch["undersampled_kspace"],
                    batch["sensitivity_maps"],
                    batch["mask"],
                    apply_data_consistency=False,
                    final_consistency_steps=0,
                )
            assert isinstance(no_dc_tensor, torch.Tensor)
            no_dc_complex = channels_to_complex(no_dc_tensor, channel_dim=1)[0]
            no_dc_magnitude = torch.abs(no_dc_complex).float().cpu().numpy()
            metrics["selected_step_no_dc"] = reconstruction_metrics(target, no_dc_complex)
            del no_dc_tensor, no_dc_complex
        else:
            assert isinstance(result, torch.Tensor)
            final_tensor = result
        reconstructed = channels_to_complex(final_tensor, channel_dim=1)[0]
        outputs[step] = torch.abs(reconstructed).float().cpu().numpy()
        metrics[f"step_{step}"] = reconstruction_metrics(target, reconstructed)
        del model, result, final_tensor, reconstructed
        torch.cuda.empty_cache()
        print(f"generated step {step:,}", flush=True)

    if final_trajectory is None or no_dc_magnitude is None:
        raise RuntimeError("the selected checkpoint must be included in --steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_images = [initial_magnitude] + [outputs[step] for step in checkpoint_steps] + [target_magnitude]
    checkpoint_labels = [
        f"Zero-filled\nPSNR {metrics['zero_filled']['psnr']:.2f}",
        *[
            f"Step {format_step(step)}\nPSNR {metrics[f'step_{step}']['psnr']:.2f}"
            for step in checkpoint_steps
        ],
        "Target",
    ]
    plot_progression(
        checkpoint_images,
        checkpoint_labels,
        target_magnitude,
        args.output_dir / "checkpoint_progression.png",
        f"Training-checkpoint reconstruction progression | 4× acceleration | {sample_id}",
    )

    trajectory_indices = sorted(set(np.linspace(0, len(final_trajectory) - 1, 7, dtype=int).tolist()))
    trajectory_images = [final_trajectory[index] for index in trajectory_indices] + [target_magnitude]
    trajectory_labels = [
        "Initial / zero-filled" if index == 0 else (
            "Final + data consistency" if index == len(final_trajectory) - 1 else f"Reverse stage {index}"
        )
        for index in trajectory_indices
    ] + ["Target"]
    plot_progression(
        trajectory_images,
        trajectory_labels,
        target_magnitude,
        args.output_dir / "reverse_diffusion_progression.png",
        f"Final-model reverse-diffusion progression | 4× acceleration | {sample_id}",
    )

    plot_progression(
        [initial_magnitude, no_dc_magnitude, outputs[selected_step], target_magnitude],
        [
            f"Zero-filled\nPSNR {metrics['zero_filled']['psnr']:.2f}",
            f"RDBM without DC\nPSNR {metrics['selected_step_no_dc']['psnr']:.2f}",
            f"RDBM with DC\nPSNR {metrics[f'step_{selected_step}']['psnr']:.2f}",
            "Target",
        ],
        target_magnitude,
        args.output_dir / "data_consistency_ablation.png",
        f"Inference-time data-consistency ablation | 4× acceleration | {sample_id}",
    )

    np.savez_compressed(
        args.output_dir / "progression_images.npz",
        target=target_magnitude,
        zero_filled=initial_magnitude,
        selected_step_no_dc=no_dc_magnitude,
        **{f"step_{step}": image for step, image in outputs.items()},
        **{f"trajectory_{index}": final_trajectory[index] for index in trajectory_indices},
    )
    metadata = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "selection_reason": selection_reason,
        "acceleration": 4,
        "selected_checkpoint_step": selected_step,
        "checkpoint_steps": checkpoint_steps,
        "trajectory_indices": trajectory_indices,
        "metrics": metrics,
    }
    (args.output_dir / "progression_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**metadata, "metrics": f"{len(metrics)} records", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
