"""D5-C5 CNN cascades with multicoil CG-SENSE data consistency."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from mri_modl import conjugate_gradient
from mri_reconstruction import (
    adjoint_operator,
    channels_to_complex,
    complex_to_channels,
    forward_operator,
)


class CascadeCNNBlock(nn.Module):
    """One independent D5 residual CNN from the published D5-C5 model."""

    def __init__(self, *, depth: int = 5, features: int = 64) -> None:
        super().__init__()
        if depth < 2 or features <= 0:
            raise ValueError("depth must be at least 2 and features positive")
        widths = [2, *([features] * (depth - 1)), 2]
        self.layers = nn.ModuleList(
            nn.Conv2d(
                widths[index],
                widths[index + 1],
                kernel_size=3,
                padding=1,
                bias=True,
            )
            for index in range(depth)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match the official Lasagne He-normal weights and zero biases."""

        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = image
        for index, layer in enumerate(self.layers):
            residual = layer(residual)
            if index + 1 < len(self.layers):
                residual = F.relu(residual)
        return image + residual


class CascadeCNN(nn.Module):
    """Five independent D5 CNNs alternated with multicoil consistency solves.

    The published model performs exact Fourier replacement on a precombined
    single image. This adaptation keeps each CNN unchanged and replaces that
    operation with the parameter-free proximal CG-SENSE solve needed for raw
    multicoil measurements.
    """

    def __init__(
        self,
        *,
        cascades: int = 5,
        depth: int = 5,
        features: int = 64,
        cg_iterations: int = 10,
        cg_tolerance: float = 1e-10,
        dc_weight: float = 0.05,
        l2_penalty: float = 1e-6,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if cascades <= 0 or cg_iterations <= 0:
            raise ValueError("cascades and CG iterations must be positive")
        if cg_tolerance < 0 or dc_weight <= 0 or l2_penalty < 0:
            raise ValueError("invalid data-consistency or L2 configuration")
        self.blocks = nn.ModuleList(
            CascadeCNNBlock(depth=depth, features=features)
            for _ in range(cascades)
        )
        self.cascades = cascades
        self.depth = depth
        self.features = features
        self.cg_iterations = cg_iterations
        self.cg_tolerance = cg_tolerance
        self.dc_weight = float(dc_weight)
        self.l2_penalty = float(l2_penalty)
        self.gradient_checkpointing = gradient_checkpointing
        self.model_config = {
            "cascades": cascades,
            "depth": depth,
            "features": features,
            "cg_iterations": cg_iterations,
            "cg_tolerance": cg_tolerance,
            "dc_weight": dc_weight,
            "l2_penalty": l2_penalty,
            "gradient_checkpointing": gradient_checkpointing,
        }

    @staticmethod
    def _validate_inputs(
        initial_reconstruction: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if initial_reconstruction.ndim != 4 or initial_reconstruction.shape[1] != 2:
            raise ValueError("initial reconstruction must be [B, 2, H, W]")
        if not torch.is_complex(measured_kspace) or not torch.is_complex(sensitivity_maps):
            raise TypeError("measured k-space and sensitivity maps must be complex")
        if measured_kspace.ndim != 4 or sensitivity_maps.ndim != 4:
            raise ValueError("k-space and sensitivity maps must be [B, C, H, W]")
        if mask.ndim != 2 or mask.shape[0] != measured_kspace.shape[0]:
            raise ValueError("mask must be [B, W]")
        if initial_reconstruction.shape[0] != measured_kspace.shape[0]:
            raise ValueError("input batch dimensions differ")

    def _cascade(
        self,
        reconstruction: torch.Tensor,
        block: CascadeCNNBlock,
        atb: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        channels = complex_to_channels(reconstruction, channel_dim=1)
        refined = channels_to_complex(block(channels), channel_dim=1)
        weight = torch.as_tensor(
            self.dc_weight, dtype=refined.real.dtype, device=refined.device
        )
        right_hand_side = atb + weight * refined
        encoded_shape = measured_kspace.shape[-2:]

        def normal_operator(image: torch.Tensor) -> torch.Tensor:
            encoded = forward_operator(image, sensitivity_maps, mask, encoded_shape)
            return adjoint_operator(encoded, sensitivity_maps, mask) + weight * image

        return conjugate_gradient(
            normal_operator,
            right_hand_side,
            initial=reconstruction,
            max_iterations=self.cg_iterations,
            tolerance=self.cg_tolerance,
        )

    def forward(
        self,
        initial_reconstruction: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        self._validate_inputs(
            initial_reconstruction, measured_kspace, sensitivity_maps, mask
        )
        reconstruction = channels_to_complex(initial_reconstruction, channel_dim=1)
        atb = adjoint_operator(measured_kspace, sensitivity_maps, mask)
        trajectory = [initial_reconstruction]
        for block in self.blocks:
            def apply_cascade(value: torch.Tensor, current: CascadeCNNBlock = block) -> torch.Tensor:
                return self._cascade(
                    value,
                    current,
                    atb,
                    measured_kspace,
                    sensitivity_maps,
                    mask,
                )

            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                reconstruction = activation_checkpoint(
                    apply_cascade, reconstruction, use_reentrant=False
                )
            else:
                reconstruction = apply_cascade(reconstruction)
            trajectory.append(complex_to_channels(reconstruction, channel_dim=1))
        return trajectory if return_trajectory else trajectory[-1]

    def training_losses(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        required = {
            "initial_reconstruction",
            "undersampled_kspace",
            "sensitivity_maps",
            "mask",
            "target",
        }
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"MRI batch is missing: {', '.join(sorted(missing))}")
        prediction = self(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        assert isinstance(prediction, torch.Tensor)
        if prediction.shape != batch["target"].shape:
            raise ValueError("prediction and target shapes differ")
        # The official Theano objective multiplies the two-channel mean by two.
        complex_mse = 2.0 * F.mse_loss(prediction, batch["target"])
        weight_l2 = torch.stack(
            [layer.weight.square().sum() for block in self.blocks for layer in block.layers]
        ).sum()
        regularization = self.l2_penalty * weight_l2
        return {
            "loss": complex_mse + regularization,
            "complex_mse": complex_mse,
            "l2_regularization": regularization,
            "prediction": prediction,
        }


def build_cascade_cnn(**kwargs: Any) -> CascadeCNN:
    return CascadeCNN(**kwargs)
