"""Spatial adapters for AdaDiff's fixed-size released architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F


Padding = tuple[int, int, int, int]


def center_pad_to_size(
    value: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, Padding]:
    """Center-pad the final two dimensions to a square model input."""

    height, width = (int(dimension) for dimension in value.shape[-2:])
    if height > size or width > size:
        raise ValueError(
            f"AdaDiff image {(height, width)} exceeds model size {(size, size)}"
        )
    vertical = size - height
    horizontal = size - width
    top = vertical // 2
    bottom = vertical - top
    left = horizontal // 2
    right = horizontal - left
    padding = (left, right, top, bottom)
    return F.pad(value, padding), padding


def remove_center_padding(value: torch.Tensor, padding: Padding) -> torch.Tensor:
    """Crop a model output back to its pre-padding spatial shape."""

    left, right, top, bottom = padding
    height_stop = value.shape[-2] - bottom if bottom else value.shape[-2]
    width_stop = value.shape[-1] - right if right else value.shape[-1]
    return value[..., top:height_stop, left:width_stop]
