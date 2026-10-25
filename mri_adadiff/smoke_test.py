"""One real-slice AdaDiff architecture, adversarial, and adaptation gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mri_adadiff import (
    AdaDiffPriorDataset,
    AdaDiffTrainer,
    DiffusionSchedule,
    build_adadiff_discriminator,
    build_adadiff_generator,
    reconstruct_with_adaptation,
)
from mri_data import MRISliceDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--full-architecture", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AdaDiff real-data smoke test requires CUDA")
    torch.manual_seed(1024)
    device = torch.device("cuda")
    dataset = MRISliceDataset(
        PROJECT_ROOT / "phase1/manifests/train.jsonl",
        split="train",
        acceleration=4,
        data_root=args.data_root,
    )
    sample = dataset[0]
    batch = {
        key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }
    prior = AdaDiffPriorDataset(dataset)[0]
    prior_batch = {"image": prior["image"].unsqueeze(0), "metadata": prior["metadata"]}
    if args.full_architecture:
        generator = build_adadiff_generator()
        discriminator = build_adadiff_discriminator()
    else:
        generator = build_adadiff_generator(
            base_channels=16,
            channel_multipliers=(1, 1, 2, 2, 2),
            style_dimension=64,
            latent_dimension=32,
            mapping_layers=1,
        )
        discriminator = build_adadiff_discriminator(
            base_channels=16, time_dimension=64, depth=5
        )
    schedule = DiffusionSchedule()
    trainer = AdaDiffTrainer(
        generator,
        discriminator,
        schedule=schedule,
        total_epochs=1,
        data_config={},
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train_metrics = trainer.train_step(prior_batch)
    reconstruction = reconstruct_with_adaptation(
        trainer.ema_generator,
        batch,
        schedule=schedule,
        adaptation_iterations=2,
    )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    report = {
        "status": "ok",
        "full_architecture": args.full_architecture,
        "sample_id": sample["metadata"]["sample_id"],
        "train_metrics": train_metrics,
        "rapid_shape": list(reconstruction.rapid.shape),
        "adapted_shape": list(reconstruction.adapted.shape),
        "initial_adaptation_loss": reconstruction.initial_adaptation_loss,
        "final_adaptation_loss": reconstruction.final_adaptation_loss,
        "seconds": seconds,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
