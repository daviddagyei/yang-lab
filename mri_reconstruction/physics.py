"""Sensitivity-aware multi-coil MRI encoding and data consistency."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .transforms import center_crop, center_pad, fft2c, ifft2c, sense_combine


def _require_complex(data: torch.Tensor, name: str) -> None:
    if not torch.is_complex(data):
        raise TypeError(f"{name} must be complex-valued, got {data.dtype}")


def _validate_image_and_maps(image: torch.Tensor, sensitivity_maps: torch.Tensor) -> None:
    _require_complex(image, "image")
    _require_complex(sensitivity_maps, "sensitivity_maps")
    if sensitivity_maps.ndim != image.ndim + 1:
        raise ValueError("sensitivity maps must add one coil axis before the spatial axes")
    if sensitivity_maps.shape[:-3] != image.shape[:-2]:
        raise ValueError("image and sensitivity-map batch dimensions differ")
    if sensitivity_maps.shape[-2:] != image.shape[-2:]:
        raise ValueError("image and sensitivity maps must have the same spatial shape")


def _broadcast_mask(mask: torch.Tensor, kspace: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool or mask.ndim < 1:
        raise TypeError("mask must be a boolean tensor with phase encoding on its final axis")
    if mask.shape[-1] != kspace.shape[-1]:
        raise ValueError("mask and k-space phase dimensions differ")
    leading = mask.shape[:-1]
    if tuple(kspace.shape[: len(leading)]) != tuple(leading):
        raise ValueError("mask and k-space batch dimensions differ")
    singleton_axes = kspace.ndim - mask.ndim
    if singleton_axes < 2:
        raise ValueError("mask must not contain coil or readout axes")
    return mask.to(kspace.device).reshape(*leading, *([1] * singleton_axes), mask.shape[-1])


def apply_sampling_mask(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a shared or batched Cartesian mask to multi-coil k-space."""

    _require_complex(kspace, "kspace")
    broadcast_mask = _broadcast_mask(mask, kspace)
    return torch.where(
        broadcast_mask,
        kspace,
        torch.zeros((), dtype=kspace.dtype, device=kspace.device),
    )


def sense_encode(
    image: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    encoded_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    """Encode an object image into unmasked multi-coil k-space."""

    _validate_image_and_maps(image, sensitivity_maps)
    encoded_shape = tuple(encoded_shape or image.shape[-2:])
    coil_images = sensitivity_maps * image.unsqueeze(-3)
    return fft2c(center_pad(coil_images, encoded_shape))


def forward_operator(
    image: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    mask: torch.Tensor,
    encoded_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    """Apply ``A(x) = M F(Sx)``."""

    return apply_sampling_mask(sense_encode(image, sensitivity_maps, encoded_shape), mask)


def adjoint_operator(
    kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply ``A^H(y) = sum_c conj(S_c) crop(F^-1(My_c))``."""

    _require_complex(kspace, "kspace")
    _require_complex(sensitivity_maps, "sensitivity_maps")
    if kspace.shape[:-3] != sensitivity_maps.shape[:-3]:
        raise ValueError("k-space and sensitivity-map batch dimensions differ")
    if kspace.shape[-3] != sensitivity_maps.shape[-3]:
        raise ValueError("k-space and sensitivity maps have different coil counts")
    coil_images = center_crop(
        ifft2c(apply_sampling_mask(kspace, mask)),
        sensitivity_maps.shape[-2:],
    )
    return torch.sum(torch.conj(sensitivity_maps) * coil_images, dim=-3)


def measurement_residual(
    image: torch.Tensor,
    measured_kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    mask: torch.Tensor,
    *,
    relative: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return the norm of ``A(x) - y`` at acquired samples."""

    predicted = forward_operator(image, sensitivity_maps, mask, measured_kspace.shape[-2:])
    measured = apply_sampling_mask(measured_kspace, mask)
    residual = torch.linalg.vector_norm(predicted - measured)
    if not relative:
        return residual
    return residual / torch.linalg.vector_norm(measured).clamp_min(eps)


def gradient_data_consistency(
    image: torch.Tensor,
    measured_kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    mask: torch.Tensor,
    *,
    step_size: float = 1.0,
) -> torch.Tensor:
    """Take one differentiable gradient step on measurement consistency."""

    if not 0.0 < step_size <= 1.0:
        raise ValueError("step_size must be in (0, 1]")
    predicted = forward_operator(image, sensitivity_maps, mask, measured_kspace.shape[-2:])
    residual = predicted - apply_sampling_mask(measured_kspace, mask)
    return image - step_size * adjoint_operator(residual, sensitivity_maps, mask)


def replace_acquired_kspace(
    image: torch.Tensor,
    measured_kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    mask: torch.Tensor,
    *,
    weight: float = 1.0,
) -> torch.Tensor:
    """Replace or blend acquired samples in the predicted coil k-space.

    ``weight=1`` performs exact hard replacement. The returned k-space therefore
    agrees with the measurements exactly at every acquired location.
    """

    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    predicted = sense_encode(image, sensitivity_maps, measured_kspace.shape[-2:])
    measured = apply_sampling_mask(measured_kspace, mask)
    broadcast_mask = _broadcast_mask(mask, predicted)
    acquired = weight * measured + (1.0 - weight) * predicted
    return torch.where(broadcast_mask, acquired, predicted)


def image_data_consistency(
    image: torch.Tensor,
    measured_kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
    mask: torch.Tensor,
    *,
    weight: float = 1.0,
) -> torch.Tensor:
    """Replace measured coil samples and combine them back into image space."""

    corrected_kspace = replace_acquired_kspace(
        image,
        measured_kspace,
        sensitivity_maps,
        mask,
        weight=weight,
    )
    corrected_coils = center_crop(ifft2c(corrected_kspace), sensitivity_maps.shape[-2:])
    return sense_combine(corrected_coils, sensitivity_maps, coil_dim=-3)
