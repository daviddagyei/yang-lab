"""PyTorch adaptation of Hammernik et al.'s MRI Variational Network.

The architecture follows the official VLOGroup implementation: every stage has
its own complex analysis filters, Gaussian-RBF influence functions, and
nonnegative SENSE data-term weight. Only the framework and data interface are
adapted from TensorFlow 1/custom optox operators to native PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from mri_reconstruction import (
    adjoint_operator,
    channels_to_complex,
    complex_to_channels,
    forward_operator,
)


@dataclass(frozen=True)
class MRIDataConfig:
    """Dataset settings stored with a checkpoint for reproducible evaluation."""

    acceleration: int = 4
    center_fraction: float = 0.08
    mask_seed: int = 20261019


def _symmetric_pad(image: torch.Tensor, pad: int) -> torch.Tensor:
    """Match TensorFlow ``SYMMETRIC`` padding, which repeats edge pixels."""

    if pad == 0:
        return image
    if image.ndim != 4:
        raise ValueError("symmetric padding expects [B, C, H, W]")
    height, width = image.shape[-2:]
    if pad > min(height, width):
        raise ValueError("padding cannot exceed the image dimensions")
    row_indices = torch.cat(
        (
            torch.arange(pad - 1, -1, -1, device=image.device),
            torch.arange(height, device=image.device),
            torch.arange(height - 1, height - pad - 1, -1, device=image.device),
        )
    )
    column_indices = torch.cat(
        (
            torch.arange(pad - 1, -1, -1, device=image.device),
            torch.arange(width, device=image.device),
            torch.arange(width - 1, width - pad - 1, -1, device=image.device),
        )
    )
    return image.index_select(-2, row_indices).index_select(-1, column_indices)


class _GaussianRBFFunction(torch.autograd.Function):
    """Memory-bounded form of the Gaussian RBF operator used by optox."""

    @staticmethod
    def forward(
        ctx: Any,
        response: torch.Tensor,
        weights: torch.Tensor,
        vmin: float,
        vmax: float,
        basis_scale: float,
        chunk_size: int,
    ) -> torch.Tensor:
        if response.ndim != 4 or weights.ndim != 2:
            raise ValueError("RBF response and weights must be [B,F,H,W] and [F,N]")
        if response.shape[1] != weights.shape[0] or weights.shape[1] < 2:
            raise ValueError("RBF filter or knot dimensions differ")
        sigma = (vmax - vmin) / (weights.shape[1] - 1)
        centers = torch.linspace(
            vmin,
            vmax,
            weights.shape[1],
            device=response.device,
            dtype=response.dtype,
        )
        output = torch.zeros_like(response)
        expanded = response.unsqueeze(-1)
        for start in range(0, weights.shape[1], chunk_size):
            stop = min(start + chunk_size, weights.shape[1])
            difference = expanded - centers[start:stop]
            basis = torch.exp(-difference.square() / (2.0 * sigma * sigma)) * basis_scale
            output = output + torch.sum(
                basis * weights[:, start:stop].view(1, weights.shape[0], 1, 1, -1),
                dim=-1,
            )
        ctx.save_for_backward(response, weights)
        ctx.vmin = vmin
        ctx.vmax = vmax
        ctx.basis_scale = basis_scale
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor):
        response, weights = ctx.saved_tensors
        sigma = (ctx.vmax - ctx.vmin) / (weights.shape[1] - 1)
        sigma_squared = sigma * sigma
        centers = torch.linspace(
            ctx.vmin,
            ctx.vmax,
            weights.shape[1],
            device=response.device,
            dtype=response.dtype,
        )
        gradient_response = torch.zeros_like(response)
        gradient_weights = torch.zeros_like(weights)
        expanded = response.unsqueeze(-1)
        expanded_gradient = gradient.unsqueeze(-1)
        for start in range(0, weights.shape[1], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, weights.shape[1])
            difference = expanded - centers[start:stop]
            basis = (
                torch.exp(-difference.square() / (2.0 * sigma_squared))
                * ctx.basis_scale
            )
            weighted_gradient = expanded_gradient * basis
            gradient_weights[:, start:stop] = weighted_gradient.sum(dim=(0, 2, 3))
            basis_derivative = basis * (-difference / sigma_squared)
            gradient_response = gradient_response + torch.sum(
                expanded_gradient
                * basis_derivative
                * weights[:, start:stop].view(1, weights.shape[0], 1, 1, -1),
                dim=-1,
            )
        return gradient_response, gradient_weights, None, None, None, None


def gaussian_rbf(
    response: torch.Tensor,
    weights: torch.Tensor,
    *,
    vmin: float,
    vmax: float,
    basis_scale: float = 0.4,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Apply the official optox Gaussian basis expansion per feature map."""

    return _GaussianRBFFunction.apply(
        response,
        weights,
        vmin,
        vmax,
        basis_scale,
        chunk_size,
    )


class VariationalNetwork(nn.Module):
    """Ten-stage variational gradient network from Hammernik et al."""

    def __init__(
        self,
        *,
        num_stages: int = 10,
        num_filters: int = 24,
        kernel_size: int = 11,
        boundary_pad: int = 11,
        num_rbf_weights: int = 31,
        rbf_vmin: float = -1.0,
        rbf_vmax: float = 1.0,
        rbf_basis_scale: float = 0.4,
        activation_init_scale: float = 0.04,
        data_weight_init: float = 1.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if min(num_stages, num_filters, kernel_size, num_rbf_weights) <= 0:
            raise ValueError("stage, filter, kernel, and RBF counts must be positive")
        if kernel_size % 2 != 1 or num_rbf_weights < 2:
            raise ValueError("kernel size must be odd and at least two RBF weights are required")
        if boundary_pad < 0 or rbf_vmax <= rbf_vmin:
            raise ValueError("invalid boundary padding or RBF interval")
        if min(rbf_basis_scale, activation_init_scale, data_weight_init) <= 0:
            raise ValueError("initial scales and data weight must be positive")

        shape = (num_stages, num_filters, 1, kernel_size, kernel_size)
        standard_deviation = 1.0 / kernel_size
        self.kernel_real = nn.Parameter(torch.randn(shape) * standard_deviation)
        self.kernel_imag = nn.Parameter(torch.randn(shape) * standard_deviation)
        knots = torch.linspace(rbf_vmin, rbf_vmax, num_rbf_weights)
        self.activation_weights = nn.Parameter(
            activation_init_scale
            * knots.view(1, 1, -1).repeat(num_stages, num_filters, 1)
        )
        self.data_term_weights = nn.Parameter(
            torch.full((num_stages,), data_weight_init, dtype=torch.float32)
        )

        self.num_stages = num_stages
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.boundary_pad = boundary_pad
        self.num_rbf_weights = num_rbf_weights
        self.rbf_vmin = rbf_vmin
        self.rbf_vmax = rbf_vmax
        self.rbf_basis_scale = rbf_basis_scale
        self.activation_init_scale = activation_init_scale
        self.data_weight_init = data_weight_init
        self.gradient_checkpointing = gradient_checkpointing
        self.model_config = {
            "num_stages": num_stages,
            "num_filters": num_filters,
            "kernel_size": kernel_size,
            "boundary_pad": boundary_pad,
            "num_rbf_weights": num_rbf_weights,
            "rbf_vmin": rbf_vmin,
            "rbf_vmax": rbf_vmax,
            "rbf_basis_scale": rbf_basis_scale,
            "activation_init_scale": activation_init_scale,
            "data_weight_init": data_weight_init,
            "gradient_checkpointing": gradient_checkpointing,
        }
        self.project_parameters()

    @torch.no_grad()
    def project_parameters(self) -> None:
        """Apply the official zero-mean, unit-norm, and positivity constraints."""

        dimensions = (2, 3, 4)
        self.kernel_real.sub_(self.kernel_real.mean(dim=dimensions, keepdim=True))
        self.kernel_imag.sub_(self.kernel_imag.mean(dim=dimensions, keepdim=True))
        norm = torch.sqrt(
            self.kernel_real.square().sum(dim=dimensions, keepdim=True)
            + self.kernel_imag.square().sum(dim=dimensions, keepdim=True)
        )
        scale = torch.clamp(norm, min=1.0)
        self.kernel_real.div_(scale)
        self.kernel_imag.div_(scale)
        self.data_term_weights.clamp_(min=0.0)

    @staticmethod
    def _validate_inputs(
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if not torch.is_complex(measured_kspace) or not torch.is_complex(sensitivity_maps):
            raise TypeError("k-space and sensitivity maps must be complex tensors")
        if measured_kspace.ndim != 4 or sensitivity_maps.ndim != 4:
            raise ValueError("expected batched [B,C,H,W] k-space and sensitivity maps")
        if mask.ndim != 2 or mask.shape[0] != measured_kspace.shape[0]:
            raise ValueError("expected a batched [B,W] sampling mask")

    def _regularizer(self, stage: int, image: torch.Tensor) -> torch.Tensor:
        real = image.real.float().unsqueeze(1)
        imaginary = image.imag.float().unsqueeze(1)
        real = _symmetric_pad(real, self.boundary_pad)
        imaginary = _symmetric_pad(imaginary, self.boundary_pad)
        convolution_padding = self.kernel_size // 2
        response = F.conv2d(
            real,
            self.kernel_real[stage],
            padding=convolution_padding,
        ) + F.conv2d(
            imaginary,
            self.kernel_imag[stage],
            padding=convolution_padding,
        )
        influence = gaussian_rbf(
            response,
            self.activation_weights[stage],
            vmin=self.rbf_vmin,
            vmax=self.rbf_vmax,
            basis_scale=self.rbf_basis_scale,
        )
        regularizer_real = F.conv_transpose2d(
            influence,
            self.kernel_real[stage],
            padding=convolution_padding,
        )
        regularizer_imag = F.conv_transpose2d(
            influence,
            self.kernel_imag[stage],
            padding=convolution_padding,
        )
        if self.boundary_pad:
            crop = self.boundary_pad
            regularizer_real = regularizer_real[..., crop:-crop, crop:-crop]
            regularizer_imag = regularizer_imag[..., crop:-crop, crop:-crop]
        return torch.complex(
            regularizer_real[:, 0] / self.num_filters,
            regularizer_imag[:, 0] / self.num_filters,
        )

    def _stage(
        self,
        stage: int,
        image: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        regularizer = self._regularizer(stage, image)
        prediction = forward_operator(
            image,
            sensitivity_maps,
            mask,
            measured_kspace.shape[-2:],
        )
        data_gradient = adjoint_operator(
            prediction - measured_kspace,
            sensitivity_maps,
            mask,
        )
        weight = self.data_term_weights[stage].to(data_gradient.real.dtype)
        return image - regularizer - weight * data_gradient

    def forward(
        self,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        *,
        initial_reconstruction: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Apply all variational stages beginning from the zero-filled SENSE image."""

        self._validate_inputs(measured_kspace, sensitivity_maps, mask)
        if initial_reconstruction is None:
            reconstruction = adjoint_operator(measured_kspace, sensitivity_maps, mask)
        else:
            reconstruction = channels_to_complex(initial_reconstruction, channel_dim=1)
        trajectory = [complex_to_channels(reconstruction, channel_dim=1)]
        for stage in range(self.num_stages):
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                reconstruction = activation_checkpoint(
                    lambda current, stage_index=stage: self._stage(
                        stage_index,
                        current,
                        measured_kspace,
                        sensitivity_maps,
                        mask,
                    ),
                    reconstruction,
                    use_reentrant=False,
                )
            else:
                reconstruction = self._stage(
                    stage,
                    reconstruction,
                    measured_kspace,
                    sensitivity_maps,
                    mask,
                )
            if return_trajectory:
                trajectory.append(complex_to_channels(reconstruction, channel_dim=1))
        result = complex_to_channels(reconstruction, channel_dim=1)
        return trajectory if return_trajectory else result

    def training_losses(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Use the official squared magnitude-error objective."""

        required = {
            "undersampled_kspace",
            "sensitivity_maps",
            "mask",
            "initial_reconstruction",
            "target",
        }
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"MRI batch is missing: {', '.join(sorted(missing))}")
        prediction = self(
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            initial_reconstruction=batch["initial_reconstruction"],
        )
        assert isinstance(prediction, torch.Tensor)
        target = batch["target"]
        prediction_complex = channels_to_complex(prediction, channel_dim=1)
        target_complex = channels_to_complex(target, channel_dim=1)
        # Match the official TensorFlow loss, including its small smoothing
        # term at zero magnitude.
        prediction_magnitude = torch.sqrt(
            prediction_complex.real.square()
            + prediction_complex.imag.square()
            + 1e-12
        )
        target_magnitude = torch.sqrt(
            target_complex.real.square()
            + target_complex.imag.square()
            + 1e-12
        )
        magnitude_squared_error = (
            prediction_magnitude - target_magnitude
        ).square()
        energy = magnitude_squared_error.flatten(1).sum(dim=1).mean()
        complex_error = torch.abs(prediction_complex - target_complex).square().flatten(1).sum(dim=1)
        target_energy = torch.abs(target_complex).square().flatten(1).sum(dim=1)
        return {
            "loss": energy,
            "magnitude_mse": magnitude_squared_error.mean(),
            "relative_rmse": torch.sqrt(
                complex_error / target_energy.clamp_min(1e-12)
            ).mean(),
            "prediction": prediction,
            "data_weight_mean": self.data_term_weights.mean(),
        }


def build_variational_network(**kwargs: Any) -> VariationalNetwork:
    """Build the official architecture with checkpoint-friendly configuration."""

    return VariationalNetwork(**kwargs)
