"""MRI adaptation of the project's conditional Residual Diffusion Bridge Model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from mri_reconstruction import (
    channels_to_complex,
    complex_to_channels,
    gradient_data_consistency,
    measurement_residual,
)
from networks import Unet
from rdbm import RDBM


@dataclass(frozen=True)
class LossWeights:
    """Weights for image, magnitude, and measurement-domain supervision."""

    complex_image: float = 1.0
    magnitude: float = 0.1
    measurement: float = 0.1

    def __post_init__(self) -> None:
        if min(self.complex_image, self.magnitude, self.measurement) < 0:
            raise ValueError("loss weights cannot be negative")
        if self.complex_image + self.magnitude + self.measurement == 0:
            raise ValueError("at least one loss weight must be positive")


class PhysicsGuidedRDBM(RDBM):
    """Conditional RDBM operating on real/imaginary MRI channels.

    The diffusion bridge remains the original project architecture. MRI tensors
    keep their input-derived normalization and are never mapped through the RGB
    ``[0, 1]`` to ``[-1, 1]`` conversion. A differentiable SENSE consistency step
    is applied to every predicted clean image.
    """

    def __init__(
        self,
        model: Unet,
        *,
        image_size: int = 320,
        objective: str = "pred_x_start",
        sampling_type: str = "pred_x_start",
        timesteps: int = 100,
        sampling_timesteps: int = 10,
        dc_step_size: float = 1.0,
        loss_weights: LossWeights | None = None,
    ) -> None:
        if model.channels != 2:
            raise ValueError("MRI RDBM requires two real/imaginary model channels")
        if not 0.0 < dc_step_size <= 1.0:
            raise ValueError("dc_step_size must be in (0, 1]")
        super().__init__(
            model,
            image_size=image_size,
            objective=objective,
            sampling_type=sampling_type,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            condition=True,
        )
        self.dc_step_size = dc_step_size
        self.loss_weights = loss_weights or LossWeights()

    def model_predictions(self, x_t: torch.Tensor, mu: torch.Tensor, t: torch.Tensor, clip_denoised: bool = False):
        """Predict without RGB-range clipping."""

        return super().model_predictions(x_t, mu, t, clip_denoised=clip_denoised)

    def apply_mri_consistency(
        self,
        image_channels: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one differentiable multi-coil consistency step."""

        image = channels_to_complex(image_channels, channel_dim=1)
        corrected = gradient_data_consistency(
            image,
            measured_kspace,
            sensitivity_maps,
            mask,
            step_size=self.dc_step_size,
        )
        return complex_to_channels(corrected, channel_dim=1)

    @staticmethod
    def _validate_batch(batch: Mapping[str, Any]) -> None:
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
        initial = batch["initial_reconstruction"]
        target = batch["target"]
        if initial.ndim != 4 or initial.shape[1] != 2 or initial.shape != target.shape:
            raise ValueError("initial reconstruction and target must be matching [B, 2, H, W] tensors")

    def training_losses(
        self,
        batch: Mapping[str, Any],
        *,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        apply_data_consistency: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute the physics-guided RDBM training objective for one batch."""

        self._validate_batch(batch)
        target = batch["target"]
        initial = batch["initial_reconstruction"]
        measured = batch["undersampled_kspace"]
        maps = batch["sensitivity_maps"]
        mask = batch["mask"]
        batch_size = target.shape[0]
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=target.device)
        noise = torch.randn_like(target) if noise is None else noise

        bridge_sample = self.q_sample(target, initial, t, noise=noise)
        prediction = self.model(bridge_sample, initial, t)
        if apply_data_consistency:
            prediction = self.apply_mri_consistency(prediction, measured, maps, mask)

        prediction_complex = channels_to_complex(prediction, channel_dim=1)
        target_complex = channels_to_complex(target, channel_dim=1)
        complex_loss = F.l1_loss(prediction, target)
        magnitude_loss = F.l1_loss(torch.abs(prediction_complex), torch.abs(target_complex))
        consistency_loss = measurement_residual(
            prediction_complex,
            measured,
            maps,
            mask,
        ).square()
        total = (
            self.loss_weights.complex_image * complex_loss
            + self.loss_weights.magnitude * magnitude_loss
            + self.loss_weights.measurement * consistency_loss
        )
        return {
            "loss": total,
            "complex_l1": complex_loss,
            "magnitude_l1": magnitude_loss,
            "measurement": consistency_loss,
            "prediction": prediction,
        }

    def forward(self, batch: Mapping[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.training_losses(batch, *args, **kwargs)["loss"]

    @torch.no_grad()
    def reconstruct(
        self,
        initial: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        *,
        apply_data_consistency: bool = True,
        final_consistency_steps: int = 1,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run DDIM sampling with SENSE consistency after every reverse step."""

        if initial.ndim != 4 or initial.shape[1] != 2:
            raise ValueError("initial must be [B, 2, H, W]")
        if final_consistency_steps < 0:
            raise ValueError("final_consistency_steps cannot be negative")
        batch_size = initial.shape[0]
        times = torch.linspace(
            -1,
            self.num_timesteps - 1,
            steps=self.sampling_timesteps + 1,
            device=initial.device,
        )
        # Construct pairs explicitly to preserve the original RDBM DDIM ordering.
        ordered = list(reversed(times.long().tolist()))
        time_pairs = list(zip(ordered[:-1], ordered[1:]))
        image = initial
        trajectory = [initial] if return_trajectory else []

        for time, time_next in time_pairs:
            time_condition = torch.full((batch_size,), time, device=initial.device, dtype=torch.long)
            predictions = self.model_predictions(image, initial, time_condition)
            noise = predictions.pred_noise
            x_start = predictions.pred_x_start
            if apply_data_consistency:
                x_start = self.apply_mri_consistency(x_start, measured_kspace, sensitivity_maps, mask)
                noise = self.predict_noise_from_x_start(image, time_condition, initial, x_start)

            if time_next < 0:
                image = x_start
            elif self.sampling_type == "pred_noise":
                if time == self.num_timesteps - 1:
                    image = initial
                else:
                    image = initial + (self.Theta[time_next] / self.Theta[time]) * (image - initial) - (
                        (self.Theta[time_next] / self.Theta[time]) * self.Sigma[time] - self.Sigma[time_next]
                    ) * noise
            else:
                if time == self.num_timesteps - 1:
                    image = initial + self.Theta[time_next] * (x_start - initial)
                else:
                    image = initial + (self.Sigma[time_next] / self.Sigma[time]) * (image - initial) + (
                        self.Theta[time_next] - self.Theta[time] * self.Sigma[time_next] / self.Sigma[time]
                    ) * (x_start - initial)
            if return_trajectory:
                trajectory.append(image)

        if apply_data_consistency:
            for _ in range(final_consistency_steps):
                image = self.apply_mri_consistency(image, measured_kspace, sensitivity_maps, mask)
            if return_trajectory and final_consistency_steps:
                trajectory.append(image)
        return trajectory if return_trajectory else image


def build_mri_rdbm(
    *,
    dim: int = 64,
    dim_mults: Sequence[int] = (1, 2, 4, 8),
    timesteps: int = 100,
    sampling_timesteps: int = 10,
    image_size: int = 320,
    dc_step_size: float = 1.0,
    loss_weights: LossWeights | None = None,
) -> PhysicsGuidedRDBM:
    """Build the existing conditional U-Net as a two-channel MRI RDBM."""

    groups = next(group for group in (8, 4, 2, 1) if dim % group == 0)
    network = Unet(
        dim=dim,
        dim_mults=tuple(dim_mults),
        channels=2,
        condition=True,
        resnet_block_groups=groups,
    )
    result = PhysicsGuidedRDBM(
        network,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        dc_step_size=dc_step_size,
        loss_weights=loss_weights,
    )
    result.model_config = {
        "dim": dim,
        "dim_mults": tuple(dim_mults),
        "timesteps": timesteps,
        "sampling_timesteps": sampling_timesteps,
        "image_size": image_size,
        "dc_step_size": dc_step_size,
    }
    return result
