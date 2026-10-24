"""Complex multicoil adaptation of the official SwinMR-PI model."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from mri_reconstruction import channels_to_complex

from .backbone import SwinIR


OFFICIAL_COMMIT = "30f34271a4a112a1c4aac8c87e7c1664f95fc78f"


def charbonnier_mean(
    difference: torch.Tensor,
    *,
    epsilon: float,
    coil_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the official robust L1-style loss with optional valid-coil masking."""

    if epsilon <= 0:
        raise ValueError("Charbonnier epsilon must be positive")
    values = torch.sqrt(difference.square() + epsilon)
    if coil_mask is None:
        return values.mean()
    if values.ndim != 4 or coil_mask.shape != values.shape[:2]:
        raise ValueError("coil mask must match the first two loss dimensions")
    weights = coil_mask.to(dtype=values.dtype, device=values.device)[..., None, None]
    denominator = weights.sum() * values.shape[-2] * values.shape[-1]
    return (values * weights).sum() / denominator.clamp_min(1)


def valid_coil_mask(
    sensitivity_maps: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Identify real coils while excluding zero-padded batch entries."""

    batch_size, coil_count = sensitivity_maps.shape[:2]
    if metadata is not None and "coils" in metadata:
        counts = torch.as_tensor(
            metadata["coils"], device=sensitivity_maps.device, dtype=torch.long
        ).reshape(-1)
        if counts.numel() != batch_size:
            raise ValueError("metadata coil counts do not match the batch")
        return torch.arange(coil_count, device=sensitivity_maps.device)[None] < counts[:, None]
    energy = sensitivity_maps.abs().square().sum(dim=(-2, -1))
    return energy > 0


class ComplexSwinMR(nn.Module):
    """Official SwinMR backbone with complex two-channel MRI input and output.

    The published SwinMR-PI model predicts a residual correction to a one-channel
    zero-filled magnitude image. This adaptation changes only the first and last
    image channels to real/imaginary pairs. Sensitivity maps remain outside the
    forward network and enter through the published coil-image and k-space losses.
    """

    def __init__(
        self,
        *,
        image_size: int = 96,
        window_size: int = 8,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        embed_dim: int = 180,
        num_heads: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.1,
        gradient_checkpointing: bool = False,
        charbonnier_epsilon: float = 1e-9,
        image_weight: float = 15.0,
        frequency_weight: float = 0.1,
        perceptual_weight: float = 0.0025,
    ) -> None:
        super().__init__()
        if image_size <= 0 or window_size <= 0 or embed_dim <= 0:
            raise ValueError("image, window, and embedding sizes must be positive")
        if image_size % window_size:
            raise ValueError("training image size must be divisible by the window size")
        if len(depths) != len(num_heads) or not depths:
            raise ValueError("depths and attention heads must have equal nonzero length")
        if any(depth <= 0 for depth in depths) or any(head <= 0 for head in num_heads):
            raise ValueError("transformer depths and heads must be positive")
        if embed_dim % max(num_heads):
            raise ValueError("embedding dimension must be divisible by every head count")
        if min(image_weight, frequency_weight, perceptual_weight) < 0:
            raise ValueError("loss weights cannot be negative")
        self.network = SwinIR(
            img_size=image_size,
            patch_size=1,
            in_chans=2,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            ape=False,
            patch_norm=True,
            use_checkpoint=gradient_checkpointing,
            upscale=1,
            img_range=1.0,
            upsampler=None,
            resi_connection="1conv",
        )
        self.image_size = int(image_size)
        self.window_size = int(window_size)
        self.depths = tuple(int(value) for value in depths)
        self.embed_dim = int(embed_dim)
        self.num_heads = tuple(int(value) for value in num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.drop_path_rate = float(drop_path_rate)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.image_weight = float(image_weight)
        self.frequency_weight = float(frequency_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.model_config = {
            "image_size": self.image_size,
            "window_size": self.window_size,
            "depths": self.depths,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "drop_path_rate": self.drop_path_rate,
            "gradient_checkpointing": self.gradient_checkpointing,
            "charbonnier_epsilon": self.charbonnier_epsilon,
            "image_weight": self.image_weight,
            "frequency_weight": self.frequency_weight,
            "perceptual_weight": self.perceptual_weight,
        }

    def forward(self, initial_reconstruction: torch.Tensor) -> torch.Tensor:
        if initial_reconstruction.ndim != 4 or initial_reconstruction.shape[1] != 2:
            raise ValueError("initial reconstruction must be [B, 2, H, W]")
        if not torch.is_floating_point(initial_reconstruction):
            raise TypeError("initial reconstruction must be real floating point channels")
        return self.network(initial_reconstruction)

    def training_losses(
        self,
        batch: Mapping[str, Any],
        *,
        perceptual_loss: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        required = {"initial_reconstruction", "sensitivity_maps", "target"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"MRI batch is missing: {', '.join(sorted(missing))}")
        maps = batch["sensitivity_maps"]
        if not torch.is_complex(maps) or maps.ndim != 4:
            raise TypeError("sensitivity maps must be complex [B, C, H, W]")
        prediction = self(batch["initial_reconstruction"])
        target = batch["target"]
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        prediction_complex = channels_to_complex(prediction, channel_dim=1)
        target_complex = channels_to_complex(target, channel_dim=1)
        if maps.shape[0] != prediction.shape[0] or maps.shape[-2:] != prediction.shape[-2:]:
            raise ValueError("sensitivity maps and images have incompatible shapes")
        coil_mask = valid_coil_mask(maps, batch.get("metadata"))
        predicted_coils = maps * prediction_complex[:, None]
        target_coils = maps * target_complex[:, None]
        coil_difference = predicted_coils - target_coils
        image_loss = charbonnier_mean(
            coil_difference.abs(),
            epsilon=self.charbonnier_epsilon,
            coil_mask=coil_mask,
        )
        # The authors' SwinMR-PI loss uses torch.fft.fftn with its default
        # (unnormalized) convention. Keep that scale so beta=0.1 retains the
        # meaning it has in the official training configuration. Centering does
        # not affect this spatially averaged loss.
        predicted_kspace = torch.fft.fftn(predicted_coils, dim=(-2, -1))
        target_kspace = torch.fft.fftn(target_coils, dim=(-2, -1))
        frequency_loss = 0.5 * (
            charbonnier_mean(
                predicted_kspace.real - target_kspace.real,
                epsilon=self.charbonnier_epsilon,
                coil_mask=coil_mask,
            )
            + charbonnier_mean(
                predicted_kspace.imag - target_kspace.imag,
                epsilon=self.charbonnier_epsilon,
                coil_mask=coil_mask,
            )
        )
        if self.perceptual_weight:
            if perceptual_loss is None:
                raise ValueError("the configured SwinMR loss requires a perceptual model")
            prediction_magnitude = prediction_complex.abs().clamp(0, 1)[:, None].repeat(1, 3, 1, 1)
            target_magnitude = target_complex.abs().clamp(0, 1)[:, None].repeat(1, 3, 1, 1)
            perceptual = perceptual_loss(prediction_magnitude, target_magnitude)
        else:
            perceptual = prediction.new_zeros(())
        total = (
            self.image_weight * image_loss
            + self.frequency_weight * frequency_loss
            + self.perceptual_weight * perceptual
        )
        return {
            "loss": total,
            "coil_image_charbonnier": image_loss,
            "coil_kspace_charbonnier": frequency_loss,
            "perceptual": perceptual,
            "prediction": prediction,
        }


def build_swinmr(**kwargs: Any) -> ComplexSwinMR:
    return ComplexSwinMR(**kwargs)
