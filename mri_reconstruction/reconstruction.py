"""Non-learned reconstruction helpers for Phase 2 verification and baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .sampling import apply_cartesian_mask
from .transforms import center_crop, ifft2c, root_sum_of_squares


def zero_filled_coil_images(
    kspace: torch.Tensor,
    mask: torch.Tensor,
    output_shape: Sequence[int] | None = None,
    *,
    phase_dim: int = -1,
) -> torch.Tensor:
    """Apply a Cartesian mask and reconstruct each coil with a centered IFFT."""

    coil_images = ifft2c(apply_cartesian_mask(kspace, mask, phase_dim=phase_dim))
    return center_crop(coil_images, output_shape) if output_shape is not None else coil_images


def zero_filled_rss(
    kspace: torch.Tensor,
    mask: torch.Tensor,
    output_shape: Sequence[int] | None = None,
    *,
    coil_dim: int = -3,
    phase_dim: int = -1,
) -> torch.Tensor:
    """Return a zero-filled root-sum-of-squares magnitude reconstruction."""

    coil_images = zero_filled_coil_images(kspace, mask, output_shape, phase_dim=phase_dim)
    return root_sum_of_squares(coil_images, coil_dim=coil_dim)
