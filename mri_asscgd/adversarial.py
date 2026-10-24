"""Conditional discriminator for the ASSCGD/SSAD-MRI adversarial mapper."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class ConditionalDiffusionDiscriminator(nn.Module):
    """Score a candidate diffusion state conditioned on a later noisy state.

    The four 3x3 convolution blocks follow the paper's discriminator description.
    A pooled linear head produces one logit per slice without imposing a fixed
    spatial input size.
    """

    def __init__(self, *, base_channels: int = 32, use_spectral_norm: bool = True) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        widths = (4, base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        blocks: list[nn.Module] = []
        for index in range(4):
            stride = 2 if index < 3 else 1
            convolution: nn.Module = nn.Conv2d(
                widths[index],
                widths[index + 1],
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            if use_spectral_norm:
                convolution = spectral_norm(convolution)
            blocks.extend(
                (
                    convolution,
                    nn.BatchNorm2d(widths[index + 1]),
                    nn.ReLU(inplace=False),
                )
            )
        self.features = nn.Sequential(*blocks)
        head: nn.Module = nn.Linear(widths[-1], 1)
        self.head = spectral_norm(head) if use_spectral_norm else head
        self.base_channels = base_channels
        self.use_spectral_norm = use_spectral_norm

    def forward(self, candidate: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if candidate.shape != condition.shape:
            raise ValueError("candidate and condition shapes differ")
        if candidate.ndim != 4 or candidate.shape[1] != 2:
            raise ValueError("discriminator inputs must be [B, 2, H, W]")
        features = self.features(torch.cat((candidate, condition), dim=1))
        pooled = features.mean(dim=(-2, -1))
        return self.head(pooled).squeeze(1)


def discriminator_objective(
    discriminator: ConditionalDiffusionDiscriminator,
    real_state: torch.Tensor,
    fake_state: torch.Tensor,
    condition: torch.Tensor,
    *,
    gradient_penalty_weight: float = 1.0,
    real_label_smoothing: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Non-saturating discriminator loss with a resolution-normalized R1 penalty."""

    if gradient_penalty_weight < 0:
        raise ValueError("gradient_penalty_weight cannot be negative")
    if not 0.0 <= real_label_smoothing < 0.5:
        raise ValueError("real_label_smoothing must be in [0, 0.5)")
    real = real_state.detach().requires_grad_(gradient_penalty_weight > 0)
    fixed_condition = condition.detach()
    real_logits = discriminator(real, fixed_condition)
    fake_logits = discriminator(fake_state.detach(), fixed_condition)
    real_targets = torch.full_like(real_logits, 1.0 - real_label_smoothing)
    fake_targets = torch.zeros_like(fake_logits)
    adversarial = torch.nn.functional.binary_cross_entropy_with_logits(
        real_logits, real_targets
    )
    adversarial = adversarial + torch.nn.functional.binary_cross_entropy_with_logits(
        fake_logits, fake_targets
    )

    penalty = torch.zeros((), device=real.device, dtype=real.dtype)
    if gradient_penalty_weight > 0:
        gradient = torch.autograd.grad(
            real_logits.sum(),
            real,
            create_graph=True,
            retain_graph=True,
        )[0]
        penalty = 0.5 * gradient.square().flatten(1).sum(dim=1).mean()
    total = adversarial + gradient_penalty_weight * penalty
    return {
        "loss": total,
        "adversarial": adversarial,
        "gradient_penalty": penalty,
        "real_logit": real_logits.mean(),
        "fake_logit": fake_logits.mean(),
    }


def generator_adversarial_loss(
    discriminator: ConditionalDiffusionDiscriminator,
    fake_state: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    """Make generated reverse states score as real under the discriminator."""

    return torch.nn.functional.softplus(-discriminator(fake_state, condition.detach())).mean()
