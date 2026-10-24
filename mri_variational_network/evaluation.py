"""Volume-balanced evaluation aggregation for the Variational Network."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


METHOD = "variational_network"


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_volume_metrics(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
                METHOD: {
                    "nmse": sum(float(row[f"{METHOD}_error_energy"]) for row in slices)
                    / target_energy,
                    "psnr": _mean(slices, f"{METHOD}_psnr"),
                    "ssim": _mean(slices, f"{METHOD}_ssim"),
                    "measurement_residual": _mean(
                        slices, f"{METHOD}_measurement_residual"
                    ),
                    "seconds_per_slice": _mean(
                        slices, f"{METHOD}_reconstruction_seconds"
                    ),
                },
            }
        )
    return volumes


def _macro_average(volumes: list[Mapping[str, Any]], method: str) -> dict[str, float]:
    if not volumes:
        raise ValueError("cannot aggregate an empty evaluation")
    metrics = ("nmse", "psnr", "ssim", "measurement_residual")
    if method == METHOD:
        metrics += ("seconds_per_slice",)
    return {
        metric: float(np.mean([float(volume[method][metric]) for volume in volumes]))
        for metric in metrics
    }


def summarize_evaluation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    volumes = aggregate_volume_metrics(rows)
    by_acquisition: dict[str, Any] = {}
    for acquisition in sorted({str(volume["acquisition"]) for volume in volumes}):
        subset = [volume for volume in volumes if volume["acquisition"] == acquisition]
        by_acquisition[acquisition] = {
            "volume_count": len(subset),
            "slice_count": sum(int(volume["slice_count"]) for volume in subset),
            "zero_filled": _macro_average(subset, "zero_filled"),
            METHOD: _macro_average(subset, METHOD),
        }
    return {
        "volume_count": len(volumes),
        "slice_count": len(rows),
        "aggregation": "slice metrics within each volume, then equal-weight mean across volumes",
        "zero_filled": _macro_average(volumes, "zero_filled"),
        METHOD: _macro_average(volumes, METHOD),
        "by_acquisition": by_acquisition,
        "volumes": volumes,
    }
