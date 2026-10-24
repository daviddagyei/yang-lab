"""Volume-balanced evaluation aggregation for SwinMR."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np


METHOD = "swinmr"


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _aggregate(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["volume_id"])].append(row)
    volumes = []
    for volume_id, slices in sorted(grouped.items()):
        energy = sum(float(row["target_energy"]) for row in slices)
        if energy <= 0:
            raise ValueError(f"volume {volume_id} has no target energy")
        volume: dict[str, Any] = {
            "volume_id": volume_id,
            "acquisition": str(slices[0]["acquisition"]),
            "slice_count": len(slices),
        }
        for method in ("zero_filled", METHOD):
            volume[method] = {
                "nmse": sum(float(row[f"{method}_error_energy"]) for row in slices) / energy,
                "psnr": _mean(slices, f"{method}_psnr"),
                "ssim": _mean(slices, f"{method}_ssim"),
                "measurement_residual": _mean(slices, f"{method}_measurement_residual"),
            }
            if method == METHOD:
                volume[method]["seconds_per_slice"] = _mean(
                    slices, f"{method}_reconstruction_seconds"
                )
        volumes.append(volume)
    return volumes


def _macro(volumes: list[Mapping[str, Any]], method: str) -> dict[str, float]:
    if not volumes:
        raise ValueError("cannot aggregate an empty evaluation")
    metrics = ["nmse", "psnr", "ssim", "measurement_residual"]
    if method == METHOD:
        metrics.append("seconds_per_slice")
    return {
        metric: float(np.mean([float(volume[method][metric]) for volume in volumes]))
        for metric in metrics
    }


def summarize_evaluation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    volumes = _aggregate(rows)
    by_acquisition = {}
    for acquisition in sorted({str(volume["acquisition"]) for volume in volumes}):
        subset = [volume for volume in volumes if volume["acquisition"] == acquisition]
        by_acquisition[acquisition] = {
            "volume_count": len(subset),
            "slice_count": sum(int(volume["slice_count"]) for volume in subset),
            "zero_filled": _macro(subset, "zero_filled"),
            METHOD: _macro(subset, METHOD),
        }
    return {
        "volume_count": len(volumes),
        "slice_count": len(rows),
        "aggregation": "slice metrics within each volume, then equal-weight mean across volumes",
        "zero_filled": _macro(volumes, "zero_filled"),
        METHOD: _macro(volumes, METHOD),
        "by_acquisition": by_acquisition,
        "volumes": volumes,
    }
