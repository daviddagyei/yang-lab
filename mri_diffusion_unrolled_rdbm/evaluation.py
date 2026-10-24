"""Volume-balanced metrics for the full diffusion-unrolled RDBM."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

METHOD = "diffusion_unrolled_rdbm"


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summarize_evaluation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["volume_id"])].append(row)
    volumes = []
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
                    "measurement_residual": _mean(slices, "zero_filled_measurement_residual"),
                },
                METHOD: {
                    "nmse": sum(float(row[f"{METHOD}_error_energy"]) for row in slices)
                    / target_energy,
                    "psnr": _mean(slices, f"{METHOD}_psnr"),
                    "ssim": _mean(slices, f"{METHOD}_ssim"),
                    "measurement_residual": _mean(slices, f"{METHOD}_measurement_residual"),
                    "seconds_per_slice": _mean(slices, f"{METHOD}_reconstruction_seconds"),
                },
            }
        )

    def macro(method: str) -> dict[str, float]:
        metrics = ["nmse", "psnr", "ssim", "measurement_residual"]
        if method == METHOD:
            metrics.append("seconds_per_slice")
        return {
            metric: float(np.mean([volume[method][metric] for volume in volumes]))
            for metric in metrics
        }

    return {
        "volume_count": len(volumes),
        "slice_count": len(rows),
        "aggregation": "slice metrics within each volume, then equal-weight mean across volumes",
        "zero_filled": macro("zero_filled"),
        METHOD: macro(METHOD),
        "volumes": volumes,
    }
