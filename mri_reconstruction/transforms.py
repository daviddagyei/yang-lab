"""Complex MRI transforms with the last two axes interpreted as spatial axes.

The raw files use ``[..., readout, phase_encoding]``. FFTs are orthonormal so the
forward and inverse transforms can later be used directly in adjoint tests and
multi-coil data-consistency operators.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


def _require_complex(data: torch.Tensor, name: str) -> None:
    if not torch.is_complex(data):
        raise TypeError(f"{name} must be a complex tensor, got {data.dtype}")


def fft2c(image: torch.Tensor) -> torch.Tensor:
    """Apply a centered, orthonormal 2-D FFT over the final two axes."""

    _require_complex(image, "image")
    shifted = torch.fft.ifftshift(image, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fft2(shifted, dim=(-2, -1), norm="ortho"), dim=(-2, -1))


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """Apply the inverse of :func:`fft2c` over the final two axes."""

    _require_complex(kspace, "kspace")
    shifted = torch.fft.ifftshift(kspace, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho"), dim=(-2, -1))


def center_crop(data: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Crop the final two axes around their center without padding."""

    if len(shape) != 2:
        raise ValueError("crop shape must contain exactly height and width")
    target_height, target_width = (int(value) for value in shape)
    height, width = data.shape[-2:]
    if target_height <= 0 or target_width <= 0:
        raise ValueError("crop dimensions must be positive")
    if target_height > height or target_width > width:
        raise ValueError(f"cannot crop shape {(height, width)} to {(target_height, target_width)}")
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return data[..., top : top + target_height, left : left + target_width]


def center_pad(data: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Zero-pad the final two axes around their center.

    This is the adjoint of :func:`center_crop` for the same source and target
    shapes, including odd padding differences.
    """

    if len(shape) != 2:
        raise ValueError("pad shape must contain exactly height and width")
    target_height, target_width = (int(value) for value in shape)
    height, width = data.shape[-2:]
    if target_height < height or target_width < width:
        raise ValueError(f"cannot pad shape {(height, width)} to {(target_height, target_width)}")
    top = (target_height - height) // 2
    bottom = target_height - height - top
    left = (target_width - width) // 2
    right = target_width - width - left
    return F.pad(data, (left, right, top, bottom))


def root_sum_of_squares(coil_images: torch.Tensor, coil_dim: int = -3) -> torch.Tensor:
    """Combine coil magnitudes using root-sum-of-squares (RSS)."""

    return torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=coil_dim))


def sense_combine(
    coil_images: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    coil_dim: int = -3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sensitivity-weighted complex coil combination.

    Computes ``sum(conj(S) * x_c) / sum(abs(S)^2)``. The denominator makes the
    operation robust when supplied maps are not exactly RSS-normalized.
    """

    _require_complex(coil_images, "coil_images")
    _require_complex(sensitivity_maps, "sensitivity_maps")
    if coil_images.shape != sensitivity_maps.shape:
        raise ValueError(
            "coil images and sensitivity maps must have identical shapes, "
            f"got {coil_images.shape} and {sensitivity_maps.shape}"
        )
    numerator = torch.sum(torch.conj(sensitivity_maps) * coil_images, dim=coil_dim)
    denominator = torch.sum(torch.abs(sensitivity_maps) ** 2, dim=coil_dim).clamp_min(eps)
    return numerator / denominator


def magnitude_scale(data: torch.Tensor, quantile: float = 0.99, eps: float = 1e-8) -> torch.Tensor:
    """Return a robust scalar magnitude scale for one MRI sample.

    Callers must estimate this from an allowed model input (for example, a
    zero-filled reconstruction), rather than from a held-out target.
    """

    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    magnitudes = torch.abs(data).reshape(-1).float()
    if magnitudes.numel() == 0 or not torch.isfinite(magnitudes).all():
        raise ValueError("normalization input must be non-empty and finite")
    return torch.quantile(magnitudes, quantile).clamp_min(eps)


def normalize_complex(data: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    """Normalize complex data by one positive, finite real scale."""

    _require_complex(data, "data")
    scale_tensor = torch.as_tensor(scale, device=data.device, dtype=data.real.dtype)
    if scale_tensor.numel() != 1 or not torch.isfinite(scale_tensor).item() or scale_tensor.item() <= 0:
        raise ValueError("scale must be one positive finite value")
    return data / scale_tensor


def complex_to_channels(data: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    """Convert a complex image to the RDBM's two real channels."""

    _require_complex(data, "data")
    return torch.stack((data.real, data.imag), dim=channel_dim)


def channels_to_complex(data: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    """Convert two real channels back to a complex image."""

    if torch.is_complex(data):
        raise TypeError("two-channel data must be real-valued")
    channel_dim = channel_dim % data.ndim
    if data.shape[channel_dim] != 2:
        raise ValueError(f"complex channel axis must have size 2, got {data.shape[channel_dim]}")
    real, imaginary = torch.unbind(data, dim=channel_dim)
    return torch.complex(real, imaginary)


def validate_phase_encoding_record(record: Mapping[str, Any]) -> int:
    """Validate the dataset convention and return the phase-encoding axis.

    The ISMRMRD encoded ``x`` and ``y`` sizes must match the final two k-space
    axes. Encoding-step-1 limits must fit the final axis, establishing that
    ``kspace[..., readout, phase_encoding]`` uses ``phase_encoding == -1``.
    """

    try:
        shape = tuple(int(value) for value in record["kspace_shape"])
        header = record["header"]
        encoded = tuple(int(value) for value in header["encoded_matrix"])
        trajectory = str(header["trajectory"]).lower()
        minimum = int(header["phase_encoding_min"])
        center = int(header["phase_encoding_center"])
        maximum = int(header["phase_encoding_max"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("record lacks complete k-space encoding metadata") from error

    if len(shape) != 4:
        raise ValueError(f"k-space must be [slice, coil, readout, phase], got {shape}")
    if trajectory != "cartesian":
        raise ValueError(f"expected Cartesian trajectory, got {trajectory!r}")
    if len(encoded) != 3 or shape[-2:] != encoded[:2]:
        raise ValueError(f"k-space spatial shape {shape[-2:]} does not match encoded x/y {encoded[:2]}")
    phase_lines = shape[-1]
    if not minimum <= center <= maximum:
        raise ValueError(
            f"phase-encoding limits must be ordered, got {(minimum, center, maximum)}"
        )
    acquired_start = phase_lines // 2 - center + minimum
    acquired_stop = phase_lines // 2 - center + maximum + 1
    if not 0 <= acquired_start < acquired_stop <= phase_lines:
        raise ValueError(
            f"mapped acquisition window {(acquired_start, acquired_stop)} does not fit final axis "
            f"of size {phase_lines}"
        )
    return -1


def phase_encoding_window(record: Mapping[str, Any]) -> tuple[int, int]:
    """Return the acquired half-open phase-encoding range from ISMRMRD limits."""

    validate_phase_encoding_record(record)
    phase_lines = int(record["kspace_shape"][-1])
    header = record["header"]
    start = phase_lines // 2 - int(header["phase_encoding_center"]) + int(header["phase_encoding_min"])
    stop = phase_lines // 2 - int(header["phase_encoding_center"]) + int(header["phase_encoding_max"]) + 1
    return start, stop
