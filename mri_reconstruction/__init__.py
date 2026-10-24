"""MRI preprocessing primitives used by the physics-guided RDBM pipeline."""

from .sampling import (
    CartesianMask,
    CartesianMaskConfig,
    apply_cartesian_mask,
    make_cartesian_mask,
    make_cartesian_mask_for_record,
)
from .reconstruction import zero_filled_coil_images, zero_filled_rss
from .physics import (
    adjoint_operator,
    apply_sampling_mask,
    forward_operator,
    gradient_data_consistency,
    image_data_consistency,
    measurement_residual,
    replace_acquired_kspace,
    sense_encode,
)
from .transforms import (
    channels_to_complex,
    center_crop,
    center_pad,
    complex_to_channels,
    fft2c,
    ifft2c,
    magnitude_scale,
    normalize_complex,
    phase_encoding_window,
    root_sum_of_squares,
    sense_combine,
    validate_phase_encoding_record,
)

__all__ = [
    "CartesianMask",
    "CartesianMaskConfig",
    "apply_cartesian_mask",
    "apply_sampling_mask",
    "adjoint_operator",
    "center_crop",
    "center_pad",
    "channels_to_complex",
    "complex_to_channels",
    "fft2c",
    "forward_operator",
    "gradient_data_consistency",
    "image_data_consistency",
    "ifft2c",
    "magnitude_scale",
    "make_cartesian_mask",
    "make_cartesian_mask_for_record",
    "measurement_residual",
    "normalize_complex",
    "phase_encoding_window",
    "root_sum_of_squares",
    "replace_acquired_kspace",
    "sense_encode",
    "sense_combine",
    "validate_phase_encoding_record",
    "zero_filled_coil_images",
    "zero_filled_rss",
]
