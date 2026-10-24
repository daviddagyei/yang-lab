"""End-to-end unrolling of the physics-guided MRI RDBM."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from mri_model import build_mri_rdbm
from mri_modl import conjugate_gradient
from mri_reconstruction import (
    adjoint_operator,
    channels_to_complex,
    complex_to_channels,
    forward_operator,
    measurement_residual,
)


@dataclass(frozen=True)
class UnrolledLossWeights:
    """Weights for final, structural, measurement, and intermediate objectives."""

    complex_image: float = 1.0
    magnitude: float = 0.1
    ssim: float = 0.1
    measurement: float = 0.1
    intermediate: float = 0.1

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if min(values) < 0:
            raise ValueError("loss weights cannot be negative")
        if sum(values) == 0:
            raise ValueError("at least one loss weight must be positive")


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def magnitude_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 7,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Differentiable local SSIM loss for batched complex images."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("SSIM inputs must be matching complex [B, H, W] images")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    prediction_magnitude = torch.abs(prediction).unsqueeze(1)
    target_magnitude = torch.abs(target).unsqueeze(1)
    flattened = target_magnitude.flatten(1)
    data_range = (flattened.max(dim=1).values - flattened.min(dim=1).values).clamp_min(
        epsilon
    )
    scale = data_range.reshape(-1, 1, 1, 1)
    prediction_magnitude = prediction_magnitude / scale
    target_magnitude = target_magnitude / scale

    maximum_window = min(prediction.shape[-2:])
    window = min(window_size, maximum_window)
    if window % 2 == 0:
        window -= 1
    window = max(window, 1)
    padding = window // 2

    def average(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(value, kernel_size=window, stride=1, padding=padding)

    mu_prediction = average(prediction_magnitude)
    mu_target = average(target_magnitude)
    prediction_variance = average(prediction_magnitude.square()) - mu_prediction.square()
    target_variance = average(target_magnitude.square()) - mu_target.square()
    covariance = average(prediction_magnitude * target_magnitude) - mu_prediction * mu_target
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_prediction * mu_target + c1) * (2 * covariance + c2)
    denominator = (
        (mu_prediction.square() + mu_target.square() + c1)
        * (prediction_variance + target_variance + c2)
    ).clamp_min(epsilon)
    return (1.0 - (numerator / denominator).mean()).clamp(0.0, 2.0)


class UnrolledRDBM(nn.Module):
    """Shared RDBM prior alternated with differentiable CG consistency solves."""

    def __init__(
        self,
        *,
        rdbm_config: Mapping[str, Any] | None = None,
        cascades: int = 3,
        cg_iterations: int = 6,
        cg_tolerance: float = 1e-10,
        initial_lambda: float = 0.05,
        loss_weights: UnrolledLossWeights | Mapping[str, float] | None = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if cascades <= 0 or cg_iterations <= 0:
            raise ValueError("cascades and cg_iterations must be positive")
        if cg_tolerance < 0 or initial_lambda <= 0:
            raise ValueError("cg_tolerance must be nonnegative and initial_lambda positive")
        if loss_weights is None:
            resolved_loss_weights = UnrolledLossWeights()
        elif isinstance(loss_weights, UnrolledLossWeights):
            resolved_loss_weights = loss_weights
        else:
            resolved_loss_weights = UnrolledLossWeights(**dict(loss_weights))

        self.rdbm = build_mri_rdbm(**dict(rdbm_config or {}))
        self.cascades = cascades
        self.cg_iterations = cg_iterations
        self.cg_tolerance = cg_tolerance
        self.loss_weights = resolved_loss_weights
        self.gradient_checkpointing = gradient_checkpointing
        self.raw_lambdas = nn.Parameter(
            torch.full((cascades,), _inverse_softplus(initial_lambda), dtype=torch.float32)
        )
        cascade_times = torch.linspace(
            self.rdbm.num_timesteps - 1,
            0,
            cascades,
        ).round().long()
        self.register_buffer("cascade_times", cascade_times, persistent=False)
        self.model_config = {
            "rdbm_config": dict(self.rdbm.model_config),
            "cascades": cascades,
            "cg_iterations": cg_iterations,
            "cg_tolerance": cg_tolerance,
            "initial_lambda": initial_lambda,
            "loss_weights": asdict(resolved_loss_weights),
            "gradient_checkpointing": gradient_checkpointing,
        }

    @property
    def regularization_weights(self) -> torch.Tensor:
        """Positive data-consistency regularization weight for every cascade."""

        return F.softplus(self.raw_lambdas)

    def load_pretrained_rdbm(self, checkpoint_path: str | Any) -> int:
        """Initialize the shared prior from a standard MRI RDBM checkpoint."""

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_config" not in checkpoint or "model" not in checkpoint:
            raise ValueError("invalid MRI RDBM checkpoint")
        saved_config = dict(checkpoint["model_config"])
        if saved_config != dict(self.rdbm.model_config):
            raise ValueError(
                f"RDBM checkpoint configuration differs: saved={saved_config}, "
                f"current={self.rdbm.model_config}"
            )
        self.rdbm.load_state_dict(checkpoint["model"])
        return int(checkpoint.get("step", 0))

    @staticmethod
    def _validate_inputs(
        initial: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if initial.ndim != 4 or initial.shape[1] != 2:
            raise ValueError("initial reconstruction must be [B, 2, H, W]")
        if not torch.is_complex(measured_kspace) or not torch.is_complex(sensitivity_maps):
            raise TypeError("measured k-space and sensitivity maps must be complex")
        if measured_kspace.ndim != 4 or sensitivity_maps.ndim != 4:
            raise ValueError("k-space and sensitivity maps must be batched [B, C, H, W]")
        if mask.ndim != 2 or mask.shape[0] != initial.shape[0]:
            raise ValueError("mask must be batched [B, W]")

    def _rdbm_prediction(
        self,
        image: torch.Tensor,
        initial: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return activation_checkpoint(
                self.rdbm.model,
                image,
                initial,
                timestep,
                use_reentrant=False,
            )
        return self.rdbm.model(image, initial, timestep)

    def forward(
        self,
        initial: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Apply scheduled RDBM refinements and CG consistency end to end."""

        self._validate_inputs(initial, measured_kspace, sensitivity_maps, mask)
        encoded_shape = measured_kspace.shape[-2:]
        atb = adjoint_operator(measured_kspace, sensitivity_maps, mask)
        reconstruction = channels_to_complex(initial, channel_dim=1)
        trajectory = [initial]
        weights = self.regularization_weights

        for cascade_index, cascade_time in enumerate(self.cascade_times):
            timestep = torch.full(
                (initial.shape[0],),
                int(cascade_time),
                dtype=torch.long,
                device=initial.device,
            )
            denoised_channels = self._rdbm_prediction(
                complex_to_channels(reconstruction, channel_dim=1),
                initial,
                timestep,
            )
            denoised = channels_to_complex(denoised_channels, channel_dim=1)
            weight = weights[cascade_index].to(denoised.real.dtype)
            right_hand_side = atb + weight * denoised

            def normal_operator(image: torch.Tensor) -> torch.Tensor:
                encoded = forward_operator(image, sensitivity_maps, mask, encoded_shape)
                return adjoint_operator(encoded, sensitivity_maps, mask) + weight * image

            reconstruction = conjugate_gradient(
                normal_operator,
                right_hand_side,
                initial=reconstruction,
                max_iterations=self.cg_iterations,
                tolerance=self.cg_tolerance,
            )
            trajectory.append(complex_to_channels(reconstruction, channel_dim=1))

        return trajectory if return_trajectory else trajectory[-1]

    def training_losses(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Train all cascades jointly with image, SSIM, and physics objectives."""

        required = {
            "initial_reconstruction",
            "target",
            "undersampled_kspace",
            "sensitivity_maps",
            "mask",
        }
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"MRI batch is missing: {', '.join(sorted(missing))}")
        trajectory = self(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            return_trajectory=True,
        )
        assert isinstance(trajectory, list)
        prediction = trajectory[-1]
        target = batch["target"]
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        prediction_complex = channels_to_complex(prediction, channel_dim=1)
        target_complex = channels_to_complex(target, channel_dim=1)
        complex_loss = F.l1_loss(prediction, target)
        magnitude_loss = F.l1_loss(
            torch.abs(prediction_complex), torch.abs(target_complex)
        )
        ssim_loss = magnitude_ssim_loss(prediction_complex, target_complex)
        measurement_loss = measurement_residual(
            prediction_complex,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        ).square()
        intermediate_outputs = trajectory[1:-1]
        intermediate_loss = (
            torch.stack([F.l1_loss(output, target) for output in intermediate_outputs]).mean()
            if intermediate_outputs
            else prediction.new_zeros(())
        )
        weights = self.loss_weights
        total = (
            weights.complex_image * complex_loss
            + weights.magnitude * magnitude_loss
            + weights.ssim * ssim_loss
            + weights.measurement * measurement_loss
            + weights.intermediate * intermediate_loss
        )
        return {
            "loss": total,
            "complex_l1": complex_loss,
            "magnitude_l1": magnitude_loss,
            "ssim_loss": ssim_loss,
            "measurement": measurement_loss,
            "intermediate_l1": intermediate_loss,
            "lambda_mean": self.regularization_weights.mean(),
            "prediction": prediction,
        }


def build_unrolled_rdbm(**kwargs: Any) -> UnrolledRDBM:
    """Build a checkpoint-friendly unrolled RDBM."""

    return UnrolledRDBM(**kwargs)
