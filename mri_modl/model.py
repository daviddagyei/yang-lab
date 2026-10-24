"""PyTorch adaptation of the authors' recursive multicoil MoDL architecture."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from mri_reconstruction import (
    adjoint_operator,
    channels_to_complex,
    complex_to_channels,
    forward_operator,
)

from .cg import conjugate_gradient


class ResidualDenoiser(nn.Module):
    """Five-layer residual CNN used as ``D_w`` in the official MoDL code."""

    def __init__(self, *, channels: int = 64, layers: int = 5) -> None:
        super().__init__()
        if channels <= 0 or layers < 2:
            raise ValueError("channels must be positive and layers must be at least 2")
        widths = [2, *([channels] * (layers - 1)), 2]
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(widths[index], widths[index + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(widths[index + 1]),
            )
            for index in range(layers)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = image
        for index, block in enumerate(self.blocks):
            residual = block(residual)
            if index + 1 < len(self.blocks):
                residual = F.relu(residual)
        return image + residual


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class MoDL(nn.Module):
    """Weight-shared MoDL unrolling with CG multicoil data consistency."""

    def __init__(
        self,
        *,
        denoiser_channels: int = 64,
        denoiser_layers: int = 5,
        unrolls: int = 10,
        cg_iterations: int = 10,
        cg_tolerance: float = 1e-10,
        initial_lambda: float = 0.05,
    ) -> None:
        super().__init__()
        if unrolls <= 0 or cg_iterations <= 0:
            raise ValueError("unrolls and cg_iterations must be positive")
        if cg_tolerance < 0 or initial_lambda <= 0:
            raise ValueError("cg_tolerance must be nonnegative and initial_lambda positive")
        self.denoiser = ResidualDenoiser(
            channels=denoiser_channels,
            layers=denoiser_layers,
        )
        self.raw_lambda = nn.Parameter(torch.tensor(_inverse_softplus(initial_lambda)))
        self.unrolls = unrolls
        self.cg_iterations = cg_iterations
        self.cg_tolerance = cg_tolerance
        self.model_config = {
            "denoiser_channels": denoiser_channels,
            "denoiser_layers": denoiser_layers,
            "unrolls": unrolls,
            "cg_iterations": cg_iterations,
            "cg_tolerance": cg_tolerance,
            "initial_lambda": initial_lambda,
        }

    @property
    def regularization_weight(self) -> torch.Tensor:
        """Positive shared MoDL regularization weight."""

        return F.softplus(self.raw_lambda)

    @staticmethod
    def _validate_inputs(
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if not torch.is_complex(measured_kspace) or not torch.is_complex(sensitivity_maps):
            raise TypeError("measured k-space and sensitivity maps must be complex")
        if measured_kspace.ndim != 4 or sensitivity_maps.ndim != 4:
            raise ValueError("MoDL expects batched [B, C, H, W] k-space and maps")
        if mask.ndim != 2 or mask.shape[0] != measured_kspace.shape[0]:
            raise ValueError("MoDL expects a batched [B, W] mask")

    def forward(
        self,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Reconstruct by alternating the shared CNN and CG consistency blocks."""

        self._validate_inputs(measured_kspace, sensitivity_maps, mask)
        encoded_shape = measured_kspace.shape[-2:]
        atb = adjoint_operator(measured_kspace, sensitivity_maps, mask)
        reconstruction = atb
        trajectory = [complex_to_channels(atb, channel_dim=1)]

        for _ in range(self.unrolls):
            denoised_channels = self.denoiser(
                complex_to_channels(reconstruction, channel_dim=1)
            )
            denoised = channels_to_complex(denoised_channels, channel_dim=1)
            weight = self.regularization_weight.to(denoised.real.dtype)
            right_hand_side = atb + weight * denoised

            def normal_operator(image: torch.Tensor) -> torch.Tensor:
                encoded = forward_operator(image, sensitivity_maps, mask, encoded_shape)
                return adjoint_operator(encoded, sensitivity_maps, mask) + weight * image

            reconstruction = conjugate_gradient(
                normal_operator,
                right_hand_side,
                max_iterations=self.cg_iterations,
                tolerance=self.cg_tolerance,
            )
            trajectory.append(complex_to_channels(reconstruction, channel_dim=1))

        return trajectory if return_trajectory else trajectory[-1]

    def training_losses(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Use the official complex two-channel MSE training objective."""

        required = {"undersampled_kspace", "sensitivity_maps", "mask", "target"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"MRI batch is missing: {', '.join(sorted(missing))}")
        prediction = self(
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        assert isinstance(prediction, torch.Tensor)
        target = batch["target"]
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        loss = F.mse_loss(prediction, target)
        return {
            "loss": loss,
            "complex_mse": loss,
            "prediction": prediction,
            "lambda": self.regularization_weight,
        }


def build_modl(**kwargs: Any) -> MoDL:
    """Build a MoDL model while preserving checkpoint-friendly configuration."""

    return MoDL(**kwargs)
