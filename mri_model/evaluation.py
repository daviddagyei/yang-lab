"""Metrics and volume-level aggregation for MRI reconstruction evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def reconstruction_metrics(
    target: torch.Tensor,
    estimate: torch.Tensor,
) -> dict[str, float]:
    """Return complex NMSE and magnitude PSNR/SSIM for one 2D slice."""

    if target.shape != estimate.shape or target.ndim != 2:
        raise ValueError("target and estimate must be matching 2D complex images")
    target_energy = torch.sum(torch.abs(target) ** 2)
    error_energy = torch.sum(torch.abs(estimate - target) ** 2)
    target_magnitude = torch.abs(target).detach().float().cpu().numpy()
    estimate_magnitude = torch.abs(estimate).detach().float().cpu().numpy()
    data_range = max(float(np.ptp(target_magnitude)), 1e-8)
    psnr = (
        float("inf")
        if float(error_energy) == 0.0
        else float(
            peak_signal_noise_ratio(
                target_magnitude,
                estimate_magnitude,
                data_range=data_range,
            )
        )
    )
    return {
        "error_energy": float(error_energy),
        "target_energy": float(target_energy),
        "nmse": float(error_energy / target_energy.clamp_min(1e-12)),
        "psnr": psnr,
        "ssim": float(
            structural_similarity(
                target_magnitude,
                estimate_magnitude,
                data_range=data_range,
            )
        ),
    }


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_volume_metrics(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate slice records within each volume."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["volume_id"])].append(row)
    volumes: list[dict[str, Any]] = []
    for volume_id, slices in sorted(grouped.items()):
        target_energy = sum(float(row["target_energy"]) for row in slices)
        if target_energy <= 0:
            raise ValueError(f"volume {volume_id} has no target energy")
        volumes.append(
            {
                "volume_id": volume_id,
                "acquisition": str(slices[0]["acquisition"]),
                "slice_count": len(slices),
                "zero_filled": {
                    "nmse": sum(float(row["zero_filled_error_energy"]) for row in slices)
                    / target_energy,
                    "psnr": _mean(slices, "zero_filled_psnr"),
                    "ssim": _mean(slices, "zero_filled_ssim"),
                    "measurement_residual": _mean(
                        slices, "zero_filled_measurement_residual"
                    ),
                },
                "rdbm": {
                    "nmse": sum(float(row["rdbm_error_energy"]) for row in slices)
                    / target_energy,
                    "psnr": _mean(slices, "rdbm_psnr"),
                    "ssim": _mean(slices, "rdbm_ssim"),
                    "measurement_residual": _mean(
                        slices, "rdbm_measurement_residual"
                    ),
                    "seconds_per_slice": _mean(slices, "rdbm_reconstruction_seconds"),
                },
            }
        )
    return volumes


def _macro_average(volumes: list[Mapping[str, Any]], method: str) -> dict[str, float]:
    if not volumes:
        raise ValueError("cannot aggregate an empty evaluation")
    metrics = ("nmse", "psnr", "ssim", "measurement_residual")
    if method == "rdbm":
        metrics += ("seconds_per_slice",)
    return {
        metric: float(np.mean([float(volume[method][metric]) for volume in volumes]))
        for metric in metrics
    }


def summarize_evaluation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return equal-volume overall and acquisition-stratified metrics."""

    volumes = aggregate_volume_metrics(rows)
    by_acquisition: dict[str, Any] = {}
    acquisitions = sorted({str(volume["acquisition"]) for volume in volumes})
    for acquisition in acquisitions:
        subset = [volume for volume in volumes if volume["acquisition"] == acquisition]
        by_acquisition[acquisition] = {
            "volume_count": len(subset),
            "slice_count": sum(int(volume["slice_count"]) for volume in subset),
            "zero_filled": _macro_average(subset, "zero_filled"),
            "rdbm": _macro_average(subset, "rdbm"),
        }
    return {
        "volume_count": len(volumes),
        "slice_count": len(rows),
        "aggregation": "slice metrics within each volume, then equal-weight mean across volumes",
        "zero_filled": _macro_average(volumes, "zero_filled"),
        "rdbm": _macro_average(volumes, "rdbm"),
        "by_acquisition": by_acquisition,
        "volumes": volumes,
    }
