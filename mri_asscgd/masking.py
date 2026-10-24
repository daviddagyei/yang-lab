"""Deterministic SSDU-style partitions of an acquired Cartesian mask."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelfSupervisedMaskConfig:
    """Configuration for splitting acquired lines into input and loss sets.

    ``input_fraction`` applies to acquired lines outside the auto-calibration
    region. Calibration lines always remain in the model-input set.
    """

    input_fraction: float = 0.5
    seed: int = 20261020

    def __post_init__(self) -> None:
        if not 0.0 < self.input_fraction < 1.0:
            raise ValueError("input_fraction must be in (0, 1)")


@dataclass(frozen=True)
class SelfSupervisedMaskSplit:
    """Disjoint input/loss masks whose union is the acquired mask."""

    acquired_mask: torch.Tensor
    input_mask: torch.Tensor
    loss_mask: torch.Tensor
    acs_mask: torch.Tensor
    acquired_lines: int
    input_lines: int
    loss_lines: int
    acs_lines: int


def _score(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def split_acquired_mask(
    acquired_mask: torch.Tensor,
    sample_id: str,
    config: SelfSupervisedMaskConfig,
    *,
    center_lines: int,
    epoch: int = 0,
    device: torch.device | str | None = None,
) -> SelfSupervisedMaskSplit:
    """Split one 1-D acquired mask without exposing held-out lines to the input.

    Hash ranking gives an exact-sized, reproducible partition without mutable
    random-number-generator state. Changing ``epoch`` changes the training
    partition; validation and test callers should keep it at zero.
    """

    if acquired_mask.ndim != 1 or acquired_mask.dtype != torch.bool:
        raise TypeError("acquired_mask must be a one-dimensional boolean tensor")
    if not sample_id:
        raise ValueError("sample_id must be non-empty")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    phase_lines = acquired_mask.numel()
    if not 0 < center_lines <= phase_lines:
        raise ValueError("center_lines must be between 1 and the mask length")

    target_device = torch.device(device) if device is not None else acquired_mask.device
    acquired = acquired_mask.to(device=target_device).clone()
    center_start = (phase_lines - center_lines + 1) // 2
    center_stop = center_start + center_lines
    acs = torch.zeros(phase_lines, dtype=torch.bool, device=target_device)
    acs[center_start:center_stop] = True
    acs &= acquired

    non_acs_indices = torch.nonzero(acquired & ~acs, as_tuple=False).flatten().tolist()
    if len(non_acs_indices) < 2:
        raise ValueError("at least two acquired non-ACS lines are required for self-supervision")

    input_count = round(len(non_acs_indices) * config.input_fraction)
    input_count = min(max(input_count, 1), len(non_acs_indices) - 1)
    ranked = sorted(
        non_acs_indices,
        key=lambda line: _score(
            f"{config.seed}|{sample_id}|epoch={epoch}|N={phase_lines}|line={line}"
        ),
    )

    input_mask = acs.clone()
    input_mask[ranked[:input_count]] = True
    loss_mask = acquired & ~input_mask

    if torch.any(input_mask & loss_mask):
        raise RuntimeError("self-supervised masks overlap")
    if not torch.equal(input_mask | loss_mask, acquired):
        raise RuntimeError("self-supervised masks do not cover the acquired mask")
    if torch.any(loss_mask & acs):
        raise RuntimeError("ACS lines leaked into the held-out loss mask")

    return SelfSupervisedMaskSplit(
        acquired_mask=acquired,
        input_mask=input_mask,
        loss_mask=loss_mask,
        acs_mask=acs,
        acquired_lines=int(acquired.sum().item()),
        input_lines=int(input_mask.sum().item()),
        loss_lines=int(loss_mask.sum().item()),
        acs_lines=int(acs.sum().item()),
    )
