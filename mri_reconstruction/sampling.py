"""Deterministic fastMRI-style random Cartesian undersampling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class CartesianMaskConfig:
    """Configuration for a 1-D phase-encoding mask.

    The 4x and 8x defaults follow common fastMRI retrospective settings: 8%
    and 4%, respectively. The 2x and 6x experiment defaults extend the same
    policy by keeping the calibration center at 32% of the expected retained
    lines. Following fastMRI's ``RandomMaskFunc``, the center is fully sampled
    and every outer line is sampled uniformly with a probability chosen to
    achieve the nominal acceleration in expectation.
    """

    acceleration: int
    center_fraction: float | None = None
    seed: int = 20261019

    def __post_init__(self) -> None:
        if self.acceleration not in {2, 4, 6, 8}:
            raise ValueError("supported nominal accelerations are 2, 4, 6, and 8")
        fraction = self.resolved_center_fraction
        if not 0.0 < fraction <= 1.0:
            raise ValueError("center_fraction must be in (0, 1]")

    @property
    def resolved_center_fraction(self) -> float:
        if self.center_fraction is not None:
            return self.center_fraction
        return 0.32 / self.acceleration


@dataclass(frozen=True)
class CartesianMask:
    mask: torch.Tensor
    acceleration: int
    center_lines: int
    sampled_lines: int
    total_lines: int
    acquired_lines: int
    outer_sampling_probability: float
    effective_acceleration: float


def _uniform_from_hash(payload: str) -> float:
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return (value + 0.5) / 2**64


def make_cartesian_mask(
    phase_lines: int,
    sample_id: str,
    config: CartesianMaskConfig,
    *,
    acquired_range: tuple[int, int] | None = None,
    device: torch.device | str | None = None,
) -> CartesianMask:
    """Create a stable mask for one sample identifier.

    The mask follows fastMRI's random Cartesian policy while replacing mutable RNG
    state with per-line SHA-256 scores. Passing the ISMRMRD-derived acquisition
    range clears encoded padding exactly as fastMRI's data transform does. Use the
    volume ID as ``sample_id`` to keep one fixed mask across all slices of a volume.
    """

    if phase_lines <= 0:
        raise ValueError("phase_lines must be positive")
    if not sample_id:
        raise ValueError("sample_id must be non-empty")

    acquired_start, acquired_stop = acquired_range or (0, phase_lines)
    if not 0 <= acquired_start < acquired_stop <= phase_lines:
        raise ValueError(
            f"acquired_range must fit [0, {phase_lines}), got {(acquired_start, acquired_stop)}"
        )

    center_lines = max(1, round(phase_lines * config.resolved_center_fraction))
    center_start = (phase_lines - center_lines + 1) // 2
    center_stop = center_start + center_lines
    expected_total = phase_lines / config.acceleration
    outer_probability = (expected_total - center_lines) / (phase_lines - center_lines)
    outer_probability = min(max(outer_probability, 0.0), 1.0)

    mask = torch.zeros(phase_lines, dtype=torch.bool, device=device)
    mask[center_start:center_stop] = True
    for line in range(phase_lines):
        if center_start <= line < center_stop:
            continue
        payload = (
            f"{config.seed}|{sample_id}|R={config.acceleration}|N={phase_lines}|"
            f"center={center_lines}|line={line}"
        )
        if _uniform_from_hash(payload) < outer_probability:
            mask[line] = True

    mask[:acquired_start] = False
    mask[acquired_stop:] = False

    sampled_lines = int(mask.sum().item())
    acquired_lines = acquired_stop - acquired_start
    if sampled_lines == 0:
        raise ValueError("mask did not retain any acquired phase-encoding lines")
    return CartesianMask(
        mask=mask,
        acceleration=config.acceleration,
        center_lines=center_lines,
        sampled_lines=sampled_lines,
        total_lines=phase_lines,
        acquired_lines=acquired_lines,
        outer_sampling_probability=outer_probability,
        effective_acceleration=acquired_lines / sampled_lines,
    )


def make_cartesian_mask_for_record(
    record: Mapping[str, Any],
    config: CartesianMaskConfig,
    *,
    device: torch.device | str | None = None,
) -> CartesianMask:
    """Create the fixed per-volume mask prescribed by one manifest record."""

    from .transforms import phase_encoding_window

    try:
        phase_lines = int(record["kspace_shape"][-1])
        volume_id = str(record["volume_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("record lacks a volume ID or k-space shape") from error
    return make_cartesian_mask(
        phase_lines,
        volume_id,
        config,
        acquired_range=phase_encoding_window(record),
        device=device,
    )


def apply_cartesian_mask(
    kspace: torch.Tensor,
    mask: torch.Tensor,
    phase_dim: int = -1,
) -> torch.Tensor:
    """Zero unmeasured phase-encoding lines while preserving measured values."""

    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise TypeError("mask must be a one-dimensional boolean tensor")
    phase_dim = phase_dim % kspace.ndim
    if kspace.shape[phase_dim] != mask.numel():
        raise ValueError(
            f"mask has {mask.numel()} lines but k-space phase axis has {kspace.shape[phase_dim]}"
        )
    shape = [1] * kspace.ndim
    shape[phase_dim] = mask.numel()
    broadcast_mask = mask.to(device=kspace.device).reshape(shape)
    return torch.where(broadcast_mask, kspace, torch.zeros((), dtype=kspace.dtype, device=kspace.device))
