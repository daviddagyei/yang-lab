"""Load and validate supplied complex coil-sensitivity maps.

The convention is ``coil_image[c] = sensitivity[c] * object_image``. Therefore,
coil combination uses ``conj(sensitivity)`` as in the SENSE adjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from mri_reconstruction import center_crop, ifft2c, sense_combine


@dataclass(frozen=True)
class SensitivityMetrics:
    coils: int
    height: int
    width: int
    support_fraction: float
    rss_median: float
    rss_max_error: float


def standardize_sensitivity_slice(
    data: np.ndarray | torch.Tensor,
    layout: str,
    output_shape: Sequence[int],
) -> torch.Tensor:
    """Convert one supplied map slice to complex ``[coil, height, width]``."""

    maps = torch.as_tensor(data)
    if maps.ndim != 3 or not torch.is_complex(maps):
        raise ValueError(f"sensitivity slice must be rank-3 complex data, got {maps.shape} {maps.dtype}")
    if layout == "slice_height_width_coil":
        maps = maps.permute(2, 0, 1)
    elif layout != "slice_coil_height_width":
        raise ValueError(f"unsupported sensitivity layout {layout!r}")
    maps = maps.contiguous()
    return center_crop(maps, output_shape)


def load_sensitivity_slice(record: Mapping[str, Any], slice_index: int) -> torch.Tensor:
    """Load, standardize, and align one supplied sensitivity-map slice."""

    path_value = record.get("sensitivity_path")
    if not path_value:
        raise ValueError("manifest record has no supplied sensitivity map")
    slices = int(record["kspace_shape"][0])
    if not 0 <= slice_index < slices:
        raise IndexError(f"slice index {slice_index} is outside [0, {slices})")
    with h5py.File(Path(path_value), "r") as source:
        data = source["sensmap"][slice_index]
    maps = standardize_sensitivity_slice(
        data,
        str(record["sensitivity_layout"]),
        tuple(int(value) for value in record["reconstruction_shape"][-2:]),
    )
    validate_sensitivities(
        maps,
        expected_coils=int(record["kspace_shape"][1]),
        expected_shape=tuple(int(value) for value in record["reconstruction_shape"][-2:]),
    )
    return maps


def normalize_sensitivities(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize maps so their coil-wise RSS is one inside map support."""

    if maps.ndim != 3 or not torch.is_complex(maps):
        raise ValueError("maps must be complex [coil, height, width]")
    rss = torch.sqrt(torch.sum(torch.abs(maps) ** 2, dim=0))
    support = rss > eps
    result = torch.zeros_like(maps)
    result[:, support] = maps[:, support] / rss[support]
    return result


def validate_sensitivities(
    maps: torch.Tensor,
    *,
    expected_coils: int | None = None,
    expected_shape: Sequence[int] | None = None,
    support_threshold: float = 1e-3,
    rss_tolerance: float = 5e-3,
) -> SensitivityMetrics:
    """Validate shape, finiteness, and RSS normalization of one map slice."""

    if maps.ndim != 3 or not torch.is_complex(maps):
        raise ValueError(f"maps must be complex [coil, height, width], got {maps.shape} {maps.dtype}")
    if expected_coils is not None and maps.shape[0] != expected_coils:
        raise ValueError(f"expected {expected_coils} coils, got {maps.shape[0]}")
    if expected_shape is not None and tuple(maps.shape[-2:]) != tuple(expected_shape):
        raise ValueError(f"expected spatial shape {tuple(expected_shape)}, got {tuple(maps.shape[-2:])}")
    if not torch.isfinite(maps).all():
        raise ValueError("sensitivity maps contain non-finite values")

    rss = torch.sqrt(torch.sum(torch.abs(maps) ** 2, dim=0))
    support = rss > support_threshold
    if not support.any():
        raise ValueError("sensitivity maps have empty support")
    supported_rss = rss[support]
    rss_max_error = float(torch.max(torch.abs(supported_rss - 1)).item())
    if rss_max_error > rss_tolerance:
        raise ValueError(f"sensitivity RSS differs from one by up to {rss_max_error:.4g}")
    return SensitivityMetrics(
        coils=int(maps.shape[0]),
        height=int(maps.shape[1]),
        width=int(maps.shape[2]),
        support_fraction=float(support.float().mean().item()),
        rss_median=float(torch.median(supported_rss).item()),
        rss_max_error=rss_max_error,
    )


def reconstruct_sense(
    kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    output_shape: Sequence[int],
) -> torch.Tensor:
    """Reconstruct a fully sampled complex object with the SENSE adjoint."""

    if kspace.ndim != 3 or not torch.is_complex(kspace):
        raise ValueError("k-space must be complex [coil, readout, phase]")
    coil_images = center_crop(ifft2c(kspace), output_shape)
    if coil_images.shape != sensitivity_maps.shape:
        raise ValueError(
            f"coil images and maps do not align: {coil_images.shape} versus {sensitivity_maps.shape}"
        )
    return sense_combine(coil_images, sensitivity_maps)
