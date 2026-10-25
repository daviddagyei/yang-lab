"""AdaDiff forward noising and large-step posterior sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(shape[0], *([1] * (len(shape) - 1)))


@dataclass
class DiffusionSchedule:
    """Official VP schedule with eight large reverse transitions by default."""

    timesteps: int = 8
    beta_min: float = 0.1
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        if self.timesteps <= 0 or self.beta_min <= 0 or self.beta_max <= self.beta_min:
            raise ValueError("invalid AdaDiff schedule")

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        time_values = np.arange(0, self.timesteps + 1, dtype=np.float64)
        time_values = time_values / self.timesteps
        time = torch.from_numpy(time_values * (1.0 - 1e-3) + 1e-3)
        log_mean = -0.25 * time.square() * (self.beta_max - self.beta_min)
        log_mean = log_mean - 0.5 * time * self.beta_min
        variance = 1.0 - torch.exp(2.0 * log_mean)
        alpha_bars = 1.0 - variance
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.cat((torch.tensor([1e-8]), betas)).float().to(device)
        step_alpha = torch.sqrt(1.0 - betas)
        cumulative_alpha = torch.from_numpy(np.cumprod(step_alpha.cpu().numpy())).to(device)
        cumulative_sigma = torch.from_numpy(
            np.sqrt(1.0 - cumulative_alpha.cpu().numpy() ** 2)
        ).to(device)

        posterior_betas = betas[1:]
        alphas = 1.0 - posterior_betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = torch.cat((torch.ones(1, device=device), cumulative[:-1]))
        posterior_variance = posterior_betas * (1.0 - previous) / (1.0 - cumulative)
        return {
            "step_alpha": step_alpha,
            "step_sigma": torch.sqrt(betas),
            "cumulative_alpha": cumulative_alpha,
            "cumulative_sigma": cumulative_sigma,
            "posterior_mean_clean": posterior_betas * torch.sqrt(previous) / (1.0 - cumulative),
            "posterior_mean_noisy": (1.0 - previous) * torch.sqrt(alphas) / (1.0 - cumulative),
            "posterior_log_variance": torch.log(posterior_variance.clamp_min(1e-20)),
        }

    def sample_pair(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.tensors(clean.device)
        # This intentionally follows the official release: q(x_t|x_0) and
        # q(x_{t+1}|x_t) draw independent Gaussian tensors.
        first_noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        current = (
            _extract(coefficients["cumulative_alpha"], timesteps, clean.shape) * clean
            + _extract(coefficients["cumulative_sigma"], timesteps, clean.shape) * first_noise
        )
        transition_noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        next_step = (
            _extract(coefficients["step_alpha"], timesteps + 1, clean.shape) * current
            + _extract(coefficients["step_sigma"], timesteps + 1, clean.shape)
            * transition_noise
        )
        return current, next_step

    def sample_posterior(
        self,
        predicted_clean: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        coefficients = self.tensors(noisy.device)
        mean = (
            _extract(coefficients["posterior_mean_clean"], timesteps, noisy.shape)
            * predicted_clean
            + _extract(coefficients["posterior_mean_noisy"], timesteps, noisy.shape) * noisy
        )
        if deterministic:
            return mean
        noise = torch.randn(
            noisy.shape, device=noisy.device, dtype=noisy.dtype, generator=generator
        )
        active = (timesteps != 0).to(noisy.dtype).reshape(
            noisy.shape[0], *([1] * (noisy.ndim - 1))
        )
        return mean + active * torch.exp(
            0.5 * _extract(coefficients["posterior_log_variance"], timesteps, noisy.shape)
        ) * noise


