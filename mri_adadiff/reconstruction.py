"""Rapid diffusion and subject-specific AdaDiff prior adaptation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from mri_reconstruction import (
    channels_to_complex,
    complex_to_channels,
    forward_operator,
    image_data_consistency,
)

from .diffusion import DiffusionSchedule
from .model import AdaDiffGenerator


def hard_data_consistency(
    channels: torch.Tensor,
    measured_kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    image = channels_to_complex(channels, channel_dim=1)
    consistent = image_data_consistency(
        image, measured_kspace, sensitivity_maps, mask, weight=1.0
    )
    return complex_to_channels(consistent, channel_dim=1)


def acquired_kspace_l1(
    channels: torch.Tensor,
    measured_kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    image = channels_to_complex(channels, channel_dim=1)
    predicted = forward_operator(
        image, sensitivity_maps, mask, measured_kspace.shape[-2:]
    )
    difference = torch.view_as_real(predicted - measured_kspace)
    support = mask.reshape(mask.shape[0], 1, 1, mask.shape[-1], 1)
    selected = difference.masked_select(support)
    if selected.numel() == 0:
        raise ValueError("AdaDiff adaptation mask has no acquired samples")
    return selected.abs().mean()


@torch.no_grad()
def rapid_reconstruct(
    generator_model: AdaDiffGenerator,
    batch: Mapping[str, Any],
    *,
    schedule: DiffusionSchedule,
    seed: int = 42,
    start: str = "zero_filled",
    deterministic_posterior: bool = False,
) -> torch.Tensor:
    """Run the official eight-step diffusion/DC initialization.

    ``zero_filled`` reproduces the released multicoil code. ``noise`` follows
    the paper text and is retained as an explicit ablation.
    """

    initial = batch["initial_reconstruction"]
    if start not in {"zero_filled", "noise"}:
        raise ValueError("rapid diffusion start must be zero_filled or noise")
    random = torch.Generator(device=initial.device).manual_seed(seed)
    state = (
        initial.clone()
        if start == "zero_filled"
        else torch.randn(initial.shape, device=initial.device, dtype=initial.dtype, generator=random)
    )
    last_time = torch.zeros(initial.shape[0], dtype=torch.long, device=initial.device)
    last_latent = torch.zeros(
        initial.shape[0], generator_model.latent_dimension, device=initial.device
    )
    for index in reversed(range(schedule.timesteps)):
        timesteps = torch.full(
            (initial.shape[0],), index, dtype=torch.long, device=initial.device
        )
        latent = torch.randn(
            (initial.shape[0], generator_model.latent_dimension),
            device=initial.device,
            generator=random,
        )
        predicted_clean = generator_model(state, timesteps, latent)
        predicted_clean = hard_data_consistency(
            predicted_clean,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        state = schedule.sample_posterior(
            predicted_clean,
            state,
            timesteps,
            generator=random,
            deterministic=deterministic_posterior,
        )
        last_time, last_latent = timesteps, latent
    # The released code performs this final t=0 generator projection after the
    # eight posterior transitions; retain it for exact behavioral alignment.
    return generator_model(state, last_time, last_latent)


@dataclass
class AdaDiffReconstruction:
    rapid: torch.Tensor
    adapted: torch.Tensor
    initial_adaptation_loss: float | None
    final_adaptation_loss: float | None


def reconstruct_with_adaptation(
    trained_generator: AdaDiffGenerator,
    batch: Mapping[str, Any],
    *,
    schedule: DiffusionSchedule,
    adaptation_iterations: int = 1_000,
    adaptation_learning_rate: float = 1e-3,
    beta1: float = 0.5,
    beta2: float = 0.9,
    seed: int = 42,
    rapid_start: str = "zero_filled",
    rapid_initial: torch.Tensor | None = None,
) -> AdaDiffReconstruction:
    if adaptation_iterations < 0 or adaptation_learning_rate <= 0:
        raise ValueError("invalid AdaDiff adaptation settings")
    model = copy.deepcopy(trained_generator).to(batch["initial_reconstruction"].device)
    # EMA models are intentionally frozen during prior training. Test-time
    # adaptation operates on a private copy, so re-enable gradients on that
    # copy without mutating the stored EMA prior.
    model.requires_grad_(True)
    model.eval()
    rapid = (
        rapid_reconstruct(model, batch, schedule=schedule, seed=seed, start=rapid_start)
        if rapid_initial is None
        else rapid_initial
    ).detach()
    if rapid.shape != batch["initial_reconstruction"].shape:
        raise ValueError("provided rapid AdaDiff reconstruction has the wrong shape")
    if adaptation_iterations == 0:
        adapted = hard_data_consistency(
            rapid,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        return AdaDiffReconstruction(rapid, adapted, None, None)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=adaptation_learning_rate, betas=(beta1, beta2)
    )
    random = torch.Generator(device=rapid.device).manual_seed(seed + 1)
    latent = torch.randn(
        (rapid.shape[0], model.latent_dimension), device=rapid.device, generator=random
    )
    time = torch.zeros(rapid.shape[0], dtype=torch.long, device=rapid.device)
    initial_loss: float | None = None
    prediction = rapid
    model.train()
    for iteration in range(adaptation_iterations):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(rapid, time, latent)
        loss = acquired_kspace_l1(
            prediction,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite AdaDiff adaptation loss at iteration {iteration + 1}")
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError(
                f"nonfinite AdaDiff adaptation gradient at iteration {iteration + 1}"
            )
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(rapid, time, latent)
        final_loss = acquired_kspace_l1(
            prediction,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        adapted = hard_data_consistency(
            prediction,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
    return AdaDiffReconstruction(
        rapid=rapid,
        adapted=adapted,
        initial_adaptation_loss=initial_loss,
        final_adaptation_loss=float(final_loss),
    )
