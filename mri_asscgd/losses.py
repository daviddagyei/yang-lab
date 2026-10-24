"""Measurement-domain losses for self-supervised MRI reconstruction."""

from __future__ import annotations

import torch

from mri_reconstruction import (
    apply_sampling_mask,
    channels_to_complex,
    forward_operator,
)


def heldout_kspace_loss(
    prediction: torch.Tensor,
    loss_kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    kind: str = "relative_l1",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare a reconstruction with measurements only on held-out lines."""

    image = prediction if torch.is_complex(prediction) else channels_to_complex(prediction)
    predicted = forward_operator(image, sensitivity_maps, loss_mask, loss_kspace.shape[-2:])
    measured = apply_sampling_mask(loss_kspace, loss_mask)
    residual = predicted - measured

    if kind == "l1":
        return torch.abs(residual).mean()
    if kind == "mse":
        return torch.abs(residual).square().mean()
    if kind == "relative_l1":
        return torch.abs(residual).sum() / torch.abs(measured).sum().clamp_min(eps)
    if kind == "nmse":
        return torch.abs(residual).square().sum() / torch.abs(measured).square().sum().clamp_min(eps)
    raise ValueError("kind must be one of: l1, mse, relative_l1, nmse")
