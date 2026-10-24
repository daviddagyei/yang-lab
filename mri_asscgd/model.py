"""Target-free conditional diffusion reconstruction for multi-coil MRI."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from mri_reconstruction import (
    channels_to_complex,
    complex_to_channels,
    gradient_data_consistency,
    measurement_residual,
)
from networks import Unet

from .losses import heldout_kspace_loss


def _cosine_betas(timesteps: int, maximum: float = 0.999) -> torch.Tensor:
    def alpha_bar(time: float) -> float:
        return math.cos((time + 0.008) / 1.008 * math.pi / 2) ** 2

    return torch.tensor(
        [
            min(1.0 - alpha_bar((index + 1) / timesteps) / alpha_bar(index / timesteps), maximum)
            for index in range(timesteps)
        ],
        dtype=torch.float32,
    )


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(selected.shape[0], *([1] * (len(shape) - 1)))


class SelfSupervisedDiffusionMRI(nn.Module):
    """Conditional diffusion trained only from held-out acquired measurements.

    The denoiser sees the noisy image, the reconstruction from ``input_mask``,
    and the diffusion timestep. Multi-coil consistency is then applied using
    only ``input_kspace``. The objective compares the result with
    ``loss_kspace`` only at ``loss_mask`` locations.
    """

    def __init__(
        self,
        denoiser: Unet,
        *,
        timesteps: int = 100,
        sampling_timesteps: int = 10,
        dc_step_size: float = 1.0,
        heldout_loss_kind: str = "relative_l1",
        inference_noise_variance: float = 0.1,
    ) -> None:
        super().__init__()
        if denoiser.channels != 2:
            raise ValueError("self-supervised MRI diffusion requires two channels")
        if timesteps <= 1 or not 1 <= sampling_timesteps <= timesteps:
            raise ValueError("require 1 <= sampling_timesteps <= timesteps and timesteps > 1")
        if not 0.0 < dc_step_size <= 1.0:
            raise ValueError("dc_step_size must be in (0, 1]")
        if heldout_loss_kind not in {"l1", "mse", "relative_l1", "nmse"}:
            raise ValueError("unsupported held-out loss kind")
        if inference_noise_variance < 0:
            raise ValueError("inference_noise_variance cannot be negative")

        self.denoiser = denoiser
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.dc_step_size = dc_step_size
        self.heldout_loss_kind = heldout_loss_kind
        self.inference_noise_variance = inference_noise_variance

        betas = _cosine_betas(timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    @staticmethod
    def _validate_batch(batch: Mapping[str, Any]) -> None:
        required = {
            "initial_reconstruction",
            "input_kspace",
            "input_mask",
            "loss_kspace",
            "loss_mask",
            "sensitivity_maps",
        }
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"self-supervised MRI batch is missing: {', '.join(sorted(missing))}")
        initial = batch["initial_reconstruction"]
        if initial.ndim != 4 or initial.shape[1] != 2:
            raise ValueError("initial_reconstruction must be [B, 2, H, W]")
        if batch["input_mask"].ndim != 2 or batch["loss_mask"].ndim != 2:
            raise ValueError("input and loss masks must be batched [B, W]")

    def q_sample(
        self,
        image: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the standard variance-preserving forward diffusion process."""

        alpha_bar = _extract(self.alpha_bars, timesteps, image.shape)
        return alpha_bar.sqrt() * image + (1.0 - alpha_bar).sqrt() * noise

    def apply_input_consistency(
        self,
        image_channels: torch.Tensor,
        input_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        image = channels_to_complex(image_channels, channel_dim=1)
        corrected = gradient_data_consistency(
            image,
            input_kspace,
            sensitivity_maps,
            input_mask,
            step_size=self.dc_step_size,
        )
        return complex_to_channels(corrected, channel_dim=1)

    def predict_clean(
        self,
        noisy: torch.Tensor,
        initial: torch.Tensor,
        timesteps: torch.Tensor,
        input_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.denoiser(noisy, initial, timesteps)
        return self.apply_input_consistency(
            prediction,
            input_kspace,
            sensitivity_maps,
            input_mask,
        )

    def training_losses(
        self,
        batch: Mapping[str, Any],
        *,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute a target-free loss from measurements withheld from the model."""

        self._validate_batch(batch)
        initial = batch["initial_reconstruction"]
        batch_size = initial.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=initial.device)
        if t.shape != (batch_size,) or t.dtype != torch.long:
            raise ValueError("t must be a long tensor with one timestep per batch item")
        noise = torch.randn_like(initial) if noise is None else noise
        if noise.shape != initial.shape:
            raise ValueError("noise and initial reconstruction shapes differ")

        noisy = self.q_sample(initial, t, noise)
        prediction = self.predict_clean(
            noisy,
            initial,
            t,
            batch["input_kspace"],
            batch["sensitivity_maps"],
            batch["input_mask"],
        )
        heldout = heldout_kspace_loss(
            prediction,
            batch["loss_kspace"],
            batch["sensitivity_maps"],
            batch["loss_mask"],
            kind=self.heldout_loss_kind,
        )
        input_residual = measurement_residual(
            channels_to_complex(prediction, channel_dim=1),
            batch["input_kspace"],
            batch["sensitivity_maps"],
            batch["input_mask"],
        )
        return {
            "loss": heldout,
            "heldout_kspace": heldout,
            "input_residual": input_residual,
            "prediction": prediction,
        }

    def adversarial_states(
        self,
        batch: Mapping[str, Any],
        *,
        t: torch.Tensor | None = None,
        step_gap: int = 1,
        base_noise: torch.Tensor | None = None,
        transition_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Construct real and generated states for a shortened reverse transition."""

        self._validate_batch(batch)
        if not 1 <= step_gap < self.timesteps:
            raise ValueError("step_gap must be between 1 and timesteps - 1")
        initial = batch["initial_reconstruction"]
        batch_size = initial.shape[0]
        if t is None:
            t = torch.randint(
                0,
                self.timesteps - step_gap,
                (batch_size,),
                device=initial.device,
            )
        if t.shape != (batch_size,) or t.dtype != torch.long:
            raise ValueError("t must be a long tensor with one timestep per batch item")
        if bool(torch.any(t < 0)) or bool(torch.any(t + step_gap >= self.timesteps)):
            raise ValueError("t and step_gap exceed the diffusion schedule")
        later_t = t + step_gap
        base_noise = torch.randn_like(initial) if base_noise is None else base_noise
        transition_noise = (
            torch.randn_like(initial) if transition_noise is None else transition_noise
        )
        if base_noise.shape != initial.shape or transition_noise.shape != initial.shape:
            raise ValueError("adversarial noises and initial reconstruction shapes differ")

        real_state = self.q_sample(initial, t, base_noise)
        alpha_now = _extract(self.alpha_bars, t, initial.shape)
        alpha_later = _extract(self.alpha_bars, later_t, initial.shape)
        transition_alpha = (alpha_later / alpha_now).clamp(0.0, 1.0)
        later_state = (
            transition_alpha.sqrt() * real_state
            + (1.0 - transition_alpha).sqrt() * transition_noise
        )
        clean = self.predict_clean(
            later_state,
            initial,
            later_t,
            batch["input_kspace"],
            batch["sensitivity_maps"],
            batch["input_mask"],
        )
        predicted_noise = (
            later_state - alpha_later.sqrt() * clean
        ) / (1.0 - alpha_later).sqrt().clamp_min(1e-8)
        fake_state = alpha_now.sqrt() * clean + (1.0 - alpha_now).sqrt() * predicted_noise
        heldout = heldout_kspace_loss(
            clean,
            batch["loss_kspace"],
            batch["sensitivity_maps"],
            batch["loss_mask"],
            kind=self.heldout_loss_kind,
        )
        return {
            "real_state": real_state,
            "fake_state": fake_state,
            "condition": later_state,
            "clean_prediction": clean,
            "heldout_kspace": heldout,
            "t": t,
            "later_t": later_t,
        }

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.training_losses(batch)["loss"]

    @torch.no_grad()
    def reconstruct(
        self,
        initial: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run accelerated DDIM-style sampling using all acquired test lines."""

        if initial.ndim != 4 or initial.shape[1] != 2:
            raise ValueError("initial must be [B, 2, H, W]")
        noise = torch.randn_like(initial) if noise is None else noise
        if noise.shape != initial.shape:
            raise ValueError("noise and initial reconstruction shapes differ")
        image = initial + math.sqrt(self.inference_noise_variance) * noise
        trajectory = [initial] if return_trajectory else []
        times = torch.linspace(
            self.timesteps - 1,
            0,
            self.sampling_timesteps,
            device=initial.device,
        ).round().long().unique_consecutive()
        next_times = torch.cat((times[1:], torch.tensor([-1], device=times.device)))

        for time, next_time in zip(times.tolist(), next_times.tolist()):
            t = torch.full((initial.shape[0],), time, dtype=torch.long, device=initial.device)
            clean = self.predict_clean(
                image,
                initial,
                t,
                measured_kspace,
                sensitivity_maps,
                mask,
            )
            if next_time < 0:
                image = clean
            else:
                alpha_now = self.alpha_bars[time]
                alpha_next = self.alpha_bars[next_time]
                predicted_noise = (image - alpha_now.sqrt() * clean) / (1.0 - alpha_now).sqrt().clamp_min(1e-8)
                image = alpha_next.sqrt() * clean + (1.0 - alpha_next).sqrt() * predicted_noise
            if return_trajectory:
                trajectory.append(image)
        return trajectory if return_trajectory else image


def build_asscgd_diffusion(
    *,
    dim: int = 64,
    dim_mults: Sequence[int] = (1, 2, 4, 8),
    timesteps: int = 100,
    sampling_timesteps: int = 10,
    dc_step_size: float = 1.0,
    heldout_loss_kind: str = "relative_l1",
    inference_noise_variance: float = 0.1,
) -> SelfSupervisedDiffusionMRI:
    """Build the ASSCGD-inspired generator from the project's conditional U-Net."""

    groups = next(group for group in (8, 4, 2, 1) if dim % group == 0)
    denoiser = Unet(
        dim=dim,
        dim_mults=tuple(dim_mults),
        channels=2,
        condition=True,
        resnet_block_groups=groups,
    )
    model = SelfSupervisedDiffusionMRI(
        denoiser,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        dc_step_size=dc_step_size,
        heldout_loss_kind=heldout_loss_kind,
        inference_noise_variance=inference_noise_variance,
    )
    model.model_config = {
        "dim": dim,
        "dim_mults": tuple(dim_mults),
        "timesteps": timesteps,
        "sampling_timesteps": sampling_timesteps,
        "dc_step_size": dc_step_size,
        "heldout_loss_kind": heldout_loss_kind,
        "inference_noise_variance": inference_noise_variance,
    }
    return model
