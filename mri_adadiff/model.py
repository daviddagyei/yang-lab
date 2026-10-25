"""Pure-PyTorch port of the official AdaDiff adversarial diffusion networks.

The topology and conditioning follow icon-lab/AdaDiff. The repository's custom
StyleGAN CUDA kernel is replaced by an exact, differentiable transcription of
its native ``upfirdn2d`` reference math for current PyTorch/CUDA compatibility.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


OFFICIAL_REPOSITORY = "https://github.com/icon-lab/AdaDiff"
OFFICIAL_COMMIT = "15f9e1a04c13678d7c3c60ca4a47cf7b12482b24"


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a vector")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return F.pad(embedding, (0, dimension - embedding.shape[1]))


def _groups(channels: int) -> int:
    return min(max(channels // 4, 1), 32)


def _fan_average(shape: torch.Size | tuple[int, ...]) -> float:
    if len(shape) < 2:
        raise ValueError("variance scaling requires at least two dimensions")
    receptive = math.prod(shape[2:]) if len(shape) > 2 else 1
    fan_in = shape[1] * receptive
    fan_out = shape[0] * receptive
    return (fan_in + fan_out) / 2.0


def variance_scaling_uniform_(
    tensor: torch.Tensor, *, scale: float = 1.0, mode: str = "fan_avg"
) -> torch.Tensor:
    """Exact uniform variance-scaling initializer used by released AdaDiff."""

    effective_scale = 1e-10 if scale == 0 else scale
    if mode == "fan_avg":
        denominator = _fan_average(tensor.shape)
    elif mode == "fan_out":
        receptive = math.prod(tensor.shape[2:]) if tensor.ndim > 2 else 1
        denominator = tensor.shape[0] * receptive
    else:
        raise ValueError("initialization mode must be fan_avg or fan_out")
    bound = math.sqrt(3.0 * effective_scale / denominator)
    return nn.init.uniform_(tensor, -bound, bound)


def ddpm_linear(input_features: int, output_features: int, *, scale: float = 1.0) -> nn.Linear:
    layer = nn.Linear(input_features, output_features)
    variance_scaling_uniform_(layer.weight, scale=scale, mode="fan_avg")
    nn.init.zeros_(layer.bias)
    return layer


def dense_linear(input_features: int, output_features: int, *, scale: float = 1.0) -> nn.Linear:
    """Match AdaDiff dense_layer.py, including its fan-out behavior."""

    layer = nn.Linear(input_features, output_features)
    variance_scaling_uniform_(layer.weight, scale=scale, mode="fan_out")
    nn.init.zeros_(layer.bias)
    return layer


def ddpm_conv(
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    *,
    padding: int = 0,
    bias: bool = True,
    scale: float = 1.0,
) -> nn.Conv2d:
    layer = nn.Conv2d(
        input_channels, output_channels, kernel_size, padding=padding, bias=bias
    )
    variance_scaling_uniform_(layer.weight, scale=scale, mode="fan_avg")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def dense_conv(
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    *,
    padding: int = 0,
    bias: bool = True,
    scale: float = 1.0,
) -> nn.Conv2d:
    layer = nn.Conv2d(
        input_channels, output_channels, kernel_size, padding=padding, bias=bias
    )
    variance_scaling_uniform_(layer.weight, scale=scale, mode="fan_out")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class PixelNorm(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + 1e-8)


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, channels: int, style_dimension: int) -> None:
        super().__init__()
        self.channels = channels
        self.normalization = nn.GroupNorm(
            _groups(channels), channels, affine=False, eps=1e-6
        )
        self.style = dense_linear(style_dimension, 2 * channels)
        with torch.no_grad():
            self.style.bias[:channels].fill_(1.0)

    def forward(self, value: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        scale, shift = self.style(style).view(-1, 2 * self.channels, 1, 1).chunk(2, dim=1)
        return scale * self.normalization(value) + shift


def _fir_kernel(values: Sequence[int | float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vector = torch.as_tensor(values, device=device, dtype=dtype)
    kernel = torch.outer(vector, vector)
    return kernel / kernel.sum()


def upfirdn2d_native(
    value: torch.Tensor,
    kernel: torch.Tensor,
    *,
    up: int = 1,
    down: int = 1,
    pad: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Autograd-compatible transcription of AdaDiff's native StyleGAN op."""

    if value.ndim != 4 or kernel.ndim != 2 or up < 1 or down < 1:
        raise ValueError("invalid upfirdn2d input")
    batch, channels, input_height, input_width = value.shape
    pad_before, pad_after = (int(item) for item in pad)
    minor = 1
    output = value.reshape(-1, input_height, input_width, minor)
    output = output.view(-1, input_height, 1, input_width, 1, minor)
    output = F.pad(output, (0, 0, 0, up - 1, 0, 0, 0, up - 1))
    output = output.view(-1, input_height * up, input_width * up, minor)
    output = F.pad(
        output,
        (
            0,
            0,
            max(pad_before, 0),
            max(pad_after, 0),
            max(pad_before, 0),
            max(pad_after, 0),
        ),
    )
    output = output[
        :,
        max(-pad_before, 0) : output.shape[1] - max(-pad_after, 0),
        max(-pad_before, 0) : output.shape[2] - max(-pad_after, 0),
        :,
    ]
    padded_height = input_height * up + 2 * pad_before if pad_before == pad_after else input_height * up + pad_before + pad_after
    padded_width = input_width * up + 2 * pad_before if pad_before == pad_after else input_width * up + pad_before + pad_after
    output = output.permute(0, 3, 1, 2).reshape(-1, 1, padded_height, padded_width)
    flipped = torch.flip(kernel.to(device=value.device, dtype=value.dtype), (0, 1))
    output = F.conv2d(output, flipped.view(1, 1, *flipped.shape))
    filtered_height = padded_height - kernel.shape[0] + 1
    filtered_width = padded_width - kernel.shape[1] + 1
    output = output.reshape(-1, minor, filtered_height, filtered_width)
    output = output.permute(0, 2, 3, 1)[:, ::down, ::down, :]
    output_height = (input_height * up + pad_before + pad_after - kernel.shape[0]) // down + 1
    output_width = (input_width * up + pad_before + pad_after - kernel.shape[1]) // down + 1
    return output.reshape(batch, channels, output_height, output_width)


def upsample(value: torch.Tensor, kernel: Sequence[int | float]) -> torch.Tensor:
    spatial = _fir_kernel(kernel, value.device, value.dtype) * 4.0
    padding = spatial.shape[0] - 2
    return upfirdn2d_native(
        value,
        spatial,
        up=2,
        pad=((padding + 1) // 2 + 1, padding // 2),
    )


def downsample(value: torch.Tensor, kernel: Sequence[int | float]) -> torch.Tensor:
    spatial = _fir_kernel(kernel, value.device, value.dtype)
    padding = spatial.shape[0] - 2
    return upfirdn2d_native(
        value,
        spatial,
        down=2,
        pad=((padding + 1) // 2, padding // 2),
    )


class DownsampleConv2d(nn.Module):
    """Exact official FIR-then-strided-convolution progressive-input layer."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int = 3,
        fir_kernel: Sequence[int | float] = (1, 3, 3, 1),
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.fir_kernel = tuple(fir_kernel)
        self.weight = nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(output_channels))
        variance_scaling_uniform_(self.weight, mode="fan_avg")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        spatial = _fir_kernel(self.fir_kernel, value.device, value.dtype)
        padding = (spatial.shape[0] - 2) + (self.kernel_size - 1)
        filtered = upfirdn2d_native(
            value,
            spatial,
            pad=((padding + 1) // 2, padding // 2),
        )
        return F.conv2d(filtered, self.weight, self.bias, stride=2)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        time_dimension: int,
        style_dimension: int,
        dropout: float,
        up: bool = False,
        down: bool = False,
        fir_kernel: Sequence[int | float] = (1, 3, 3, 1),
    ) -> None:
        super().__init__()
        if up and down:
            raise ValueError("a residual block cannot upsample and downsample")
        self.up = up
        self.down = down
        self.fir_kernel = tuple(fir_kernel)
        self.norm1 = AdaptiveGroupNorm(input_channels, style_dimension)
        self.conv1 = ddpm_conv(input_channels, output_channels, 3, padding=1)
        self.time_projection = ddpm_linear(time_dimension, output_channels)
        self.norm2 = AdaptiveGroupNorm(output_channels, style_dimension)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = ddpm_conv(
            output_channels, output_channels, 3, padding=1, scale=0.0
        )
        self.shortcut = (
            ddpm_conv(input_channels, output_channels, 1)
            if input_channels != output_channels or up or down
            else nn.Identity()
        )

    def _resample(self, value: torch.Tensor) -> torch.Tensor:
        if self.up:
            return upsample(value, self.fir_kernel)
        if self.down:
            return downsample(value, self.fir_kernel)
        return value

    def forward(
        self, value: torch.Tensor, time_embedding: torch.Tensor, style: torch.Tensor
    ) -> torch.Tensor:
        residual = self._resample(value)
        hidden = self._resample(F.silu(self.norm1(value, style)))
        hidden = self.conv1(hidden)
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden, style))))
        return (self.shortcut(residual) + hidden) / math.sqrt(2.0)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels, eps=1e-6)
        self.query = ddpm_conv(channels, channels, 1, scale=0.1)
        self.key = ddpm_conv(channels, channels, 1, scale=0.1)
        self.value = ddpm_conv(channels, channels, 1, scale=0.1)
        self.output = ddpm_conv(channels, channels, 1, scale=0.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        normalized = self.norm(value)
        query = self.query(normalized)
        key = self.key(normalized)
        content = self.value(normalized)
        query = query.flatten(2).transpose(1, 2)
        key = key.flatten(2).transpose(1, 2)
        content = content.flatten(2).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, content)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return (value + self.output(attended)) / math.sqrt(2.0)


class AdaDiffGenerator(nn.Module):
    """NCSN++ generator with time and latent adaptive-normalization inputs."""

    def __init__(
        self,
        *,
        image_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 1, 1, 2, 2),
        residual_blocks: int = 2,
        attention_resolutions: Sequence[int] = (20,),
        image_size: int = 320,
        latent_dimension: int = 100,
        style_dimension: int = 256,
        mapping_layers: int = 3,
        dropout: float = 0.0,
        fir_kernel: Sequence[int | float] = (1, 3, 3, 1),
        output_tanh: bool = True,
    ) -> None:
        super().__init__()
        if image_size <= 0 or base_channels <= 0 or residual_blocks <= 0:
            raise ValueError("invalid AdaDiff generator dimensions")
        if len(channel_multipliers) < 2 or mapping_layers < 1:
            raise ValueError("AdaDiff requires multiple resolutions and a latent MLP")
        divisor = 2 ** (len(channel_multipliers) - 1)
        if image_size % divisor:
            raise ValueError(f"image size must be divisible by {divisor}")
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.channel_multipliers = tuple(int(value) for value in channel_multipliers)
        self.residual_blocks = residual_blocks
        self.attention_resolutions = tuple(int(value) for value in attention_resolutions)
        self.image_size = image_size
        self.latent_dimension = latent_dimension
        self.style_dimension = style_dimension
        self.mapping_layers = mapping_layers
        self.dropout_probability = dropout
        self.fir_kernel = tuple(fir_kernel)
        self.output_tanh = output_tanh

        time_dimension = 4 * base_channels
        self.time_mlp = nn.Sequential(
            ddpm_linear(base_channels, time_dimension),
            nn.SiLU(),
            ddpm_linear(time_dimension, time_dimension),
        )
        mapping: list[nn.Module] = [PixelNorm()]
        input_dimension = latent_dimension
        for _ in range(mapping_layers + 1):
            mapping.extend((dense_linear(input_dimension, style_dimension), nn.SiLU()))
            input_dimension = style_dimension
        self.style_mlp = nn.Sequential(*mapping)
        self.input = ddpm_conv(image_channels, base_channels, 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_attention = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.pyramid_projections = nn.ModuleList()
        skip_channels = [base_channels]
        channels = base_channels
        pyramid_channels = image_channels
        resolution = image_size
        for level, multiplier in enumerate(self.channel_multipliers):
            output_channels = base_channels * multiplier
            level_blocks = nn.ModuleList()
            level_attention = nn.ModuleList()
            for _ in range(residual_blocks):
                level_blocks.append(
                    ResidualBlock(
                        channels,
                        output_channels,
                        time_dimension=time_dimension,
                        style_dimension=style_dimension,
                        dropout=dropout,
                        fir_kernel=fir_kernel,
                    )
                )
                channels = output_channels
                level_attention.append(
                    AttentionBlock(channels)
                    if resolution in self.attention_resolutions
                    else nn.Identity()
                )
                skip_channels.append(channels)
            self.encoder_blocks.append(level_blocks)
            self.encoder_attention.append(level_attention)
            if level != len(self.channel_multipliers) - 1:
                self.down_blocks.append(
                    ResidualBlock(
                        channels,
                        channels,
                        time_dimension=time_dimension,
                        style_dimension=style_dimension,
                        dropout=dropout,
                        down=True,
                        fir_kernel=fir_kernel,
                    )
                )
                self.pyramid_projections.append(
                    DownsampleConv2d(
                        pyramid_channels,
                        channels,
                        kernel_size=3,
                        fir_kernel=fir_kernel,
                    )
                )
                pyramid_channels = channels
                skip_channels.append(channels)
                resolution //= 2

        self.middle1 = ResidualBlock(
            channels,
            channels,
            time_dimension=time_dimension,
            style_dimension=style_dimension,
            dropout=dropout,
            fir_kernel=fir_kernel,
        )
        self.middle_attention = AttentionBlock(channels)
        self.middle2 = ResidualBlock(
            channels,
            channels,
            time_dimension=time_dimension,
            style_dimension=style_dimension,
            dropout=dropout,
            fir_kernel=fir_kernel,
        )

        self.decoder_blocks = nn.ModuleList()
        self.decoder_attention = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for reverse_level, multiplier in enumerate(reversed(self.channel_multipliers)):
            output_channels = base_channels * multiplier
            level_blocks = nn.ModuleList()
            level_attention = nn.ModuleList()
            for _ in range(residual_blocks + 1):
                skip = skip_channels.pop()
                level_blocks.append(
                    ResidualBlock(
                        channels + skip,
                        output_channels,
                        time_dimension=time_dimension,
                        style_dimension=style_dimension,
                        dropout=dropout,
                        fir_kernel=fir_kernel,
                    )
                )
                channels = output_channels
                level_resolution = image_size // 2 ** (len(self.channel_multipliers) - 1 - reverse_level)
                level_attention.append(
                    AttentionBlock(channels)
                    if level_resolution in self.attention_resolutions
                    else nn.Identity()
                )
            self.decoder_blocks.append(level_blocks)
            self.decoder_attention.append(level_attention)
            if reverse_level != len(self.channel_multipliers) - 1:
                self.up_blocks.append(
                    ResidualBlock(
                        channels,
                        channels,
                        time_dimension=time_dimension,
                        style_dimension=style_dimension,
                        dropout=dropout,
                        up=True,
                        fir_kernel=fir_kernel,
                    )
                )
        if skip_channels:
            raise RuntimeError("AdaDiff generator skip topology did not balance")
        self.output_norm = nn.GroupNorm(_groups(channels), channels, eps=1e-6)
        self.output = ddpm_conv(channels, image_channels, 3, padding=1, scale=0.0)

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "image_channels": self.image_channels,
            "base_channels": self.base_channels,
            "channel_multipliers": self.channel_multipliers,
            "residual_blocks": self.residual_blocks,
            "attention_resolutions": self.attention_resolutions,
            "image_size": self.image_size,
            "latent_dimension": self.latent_dimension,
            "style_dimension": self.style_dimension,
            "mapping_layers": self.mapping_layers,
            "dropout": self.dropout_probability,
            "fir_kernel": self.fir_kernel,
            "output_tanh": self.output_tanh,
        }

    def forward(
        self, value: torch.Tensor, timesteps: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.image_channels:
            raise ValueError("AdaDiff generator expects [batch, 2, height, width]")
        if value.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"generator expects {(self.image_size, self.image_size)}, got {value.shape[-2:]}"
            )
        if timesteps.shape != (value.shape[0],) or latent.shape != (
            value.shape[0],
            self.latent_dimension,
        ):
            raise ValueError("AdaDiff time or latent conditioning shape differs")
        time = self.time_mlp(timestep_embedding(timesteps, self.base_channels))
        style = self.style_mlp(latent)
        input_pyramid = value
        hidden = self.input(value)
        skips = [hidden]
        for level, blocks in enumerate(self.encoder_blocks):
            for block, attention in zip(blocks, self.encoder_attention[level]):
                hidden = attention(block(hidden, time, style))
                skips.append(hidden)
            if level < len(self.down_blocks):
                hidden = self.down_blocks[level](hidden, time, style)
                input_pyramid = self.pyramid_projections[level](input_pyramid)
                hidden = (hidden + input_pyramid) / math.sqrt(2.0)
                skips.append(hidden)
        hidden = self.middle2(
            self.middle_attention(self.middle1(hidden, time, style)), time, style
        )
        for level, blocks in enumerate(self.decoder_blocks):
            for block, attention in zip(blocks, self.decoder_attention[level]):
                hidden = attention(block(torch.cat((hidden, skips.pop()), dim=1), time, style))
            if level < len(self.up_blocks):
                hidden = self.up_blocks[level](hidden, time, style)
        if skips:
            raise RuntimeError("AdaDiff generator left unused skip tensors")
        output = self.output(F.silu(self.output_norm(hidden)))
        return torch.tanh(output) if self.output_tanh else output


class DiscriminatorResidualBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, time_dimension: int, *, down: bool
    ) -> None:
        super().__init__()
        self.down = down
        self.conv1 = dense_conv(input_channels, output_channels, 3, padding=1)
        self.conv2 = dense_conv(
            output_channels, output_channels, 3, padding=1, scale=0.0
        )
        self.time = dense_linear(time_dimension, output_channels)
        self.shortcut = dense_conv(
            input_channels, output_channels, 1, bias=False
        )

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        residual = value
        hidden = self.conv1(F.leaky_relu(value, 0.2))
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = F.leaky_relu(hidden, 0.2)
        if self.down:
            hidden = downsample(hidden, (1, 3, 3, 1))
            residual = downsample(residual, (1, 3, 3, 1))
        return (self.conv2(hidden) + self.shortcut(residual)) / math.sqrt(2.0)


class AdaDiffDiscriminator(nn.Module):
    """Time-conditional large-image discriminator from the released model."""

    def __init__(
        self,
        *,
        image_channels: int = 2,
        base_channels: int = 64,
        time_dimension: int = 256,
        depth: int = 6,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("discriminator depth must be positive")
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.time_dimension = time_dimension
        self.depth = depth
        self.time_mlp = nn.Sequential(
            dense_linear(time_dimension, time_dimension),
            nn.LeakyReLU(0.2),
            dense_linear(time_dimension, time_dimension),
        )
        channels = 2 * base_channels
        self.input = dense_conv(2 * image_channels, channels, 1)
        blocks = []
        for index in range(depth):
            output_channels = min(4 * base_channels if index == 0 else 8 * base_channels, 8 * base_channels)
            blocks.append(
                DiscriminatorResidualBlock(channels, output_channels, time_dimension, down=True)
            )
            channels = output_channels
        self.blocks = nn.ModuleList(blocks)
        self.final = dense_conv(channels + 1, channels, 3, padding=1)
        self.classifier = dense_linear(channels, 1)

    @property
    def model_config(self) -> dict[str, int]:
        return {
            "image_channels": self.image_channels,
            "base_channels": self.base_channels,
            "time_dimension": self.time_dimension,
            "depth": self.depth,
        }

    def forward(
        self, candidate: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if candidate.shape != condition.shape:
            raise ValueError("discriminator candidate and condition must match")
        time = F.leaky_relu(
            self.time_mlp(timestep_embedding(timesteps, self.time_dimension)), 0.2
        )
        hidden = self.input(torch.cat((candidate, condition), dim=1))
        for block in self.blocks:
            hidden = block(hidden, time)
        deviation = torch.sqrt(hidden.var(dim=0, unbiased=False, keepdim=True) + 1e-8)
        deviation = deviation.mean(dim=(1, 2, 3), keepdim=True).expand(
            hidden.shape[0], 1, hidden.shape[2], hidden.shape[3]
        )
        hidden = F.leaky_relu(self.final(torch.cat((hidden, deviation), dim=1)), 0.2)
        return self.classifier(hidden.sum(dim=(2, 3))).flatten()


def build_adadiff_generator(**kwargs: object) -> AdaDiffGenerator:
    return AdaDiffGenerator(**kwargs)


def build_adadiff_discriminator(**kwargs: object) -> AdaDiffDiscriminator:
    return AdaDiffDiscriminator(**kwargs)
