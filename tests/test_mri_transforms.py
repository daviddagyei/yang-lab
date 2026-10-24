from __future__ import annotations

import unittest

import torch

from mri_reconstruction.reconstruction import zero_filled_rss
from mri_reconstruction.transforms import (
    channels_to_complex,
    center_crop,
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


class MRITransformTests(unittest.TestCase):
    def test_centered_fft_round_trip_and_energy(self) -> None:
        generator = torch.Generator().manual_seed(11)
        image = torch.complex(
            torch.randn(3, 17, 20, generator=generator),
            torch.randn(3, 17, 20, generator=generator),
        )
        kspace = fft2c(image)
        torch.testing.assert_close(ifft2c(kspace), image, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(torch.linalg.vector_norm(kspace), torch.linalg.vector_norm(image))

    def test_center_crop_uses_final_spatial_axes(self) -> None:
        data = torch.arange(2 * 8 * 10).reshape(2, 8, 10)
        cropped = center_crop(data, (4, 6))
        self.assertEqual(cropped.shape, (2, 4, 6))
        torch.testing.assert_close(cropped, data[:, 2:6, 2:8])

    def test_full_mask_zero_filled_rss_has_expected_orientation_and_shape(self) -> None:
        generator = torch.Generator().manual_seed(21)
        coil_images = torch.complex(
            torch.randn(4, 12, 10, generator=generator),
            torch.randn(4, 12, 10, generator=generator),
        )
        kspace = fft2c(coil_images)
        reconstruction = zero_filled_rss(kspace, torch.ones(10, dtype=torch.bool), (8, 6))
        expected = root_sum_of_squares(center_crop(coil_images, (8, 6)))
        self.assertEqual(reconstruction.shape, (8, 6))
        torch.testing.assert_close(reconstruction, expected, rtol=1e-5, atol=1e-5)

    def test_coil_combination(self) -> None:
        sensitivity = torch.ones(4, 6, 5, dtype=torch.complex64) / 2
        image = torch.complex(torch.arange(30, dtype=torch.float32).reshape(6, 5), torch.ones(6, 5))
        coils = sensitivity * image
        torch.testing.assert_close(sense_combine(coils, sensitivity), image)
        torch.testing.assert_close(root_sum_of_squares(coils), torch.abs(image))

    def test_normalization_and_two_channel_round_trip(self) -> None:
        image = torch.complex(torch.arange(1, 17, dtype=torch.float32).reshape(4, 4), torch.ones(4, 4))
        scale = magnitude_scale(image, quantile=1.0)
        normalized = normalize_complex(image, scale)
        self.assertAlmostEqual(float(normalized.abs().max()), 1.0, places=6)
        torch.testing.assert_close(channels_to_complex(complex_to_channels(normalized)), normalized)

    def test_validates_phase_encoding_as_final_axis(self) -> None:
        record = {
            "kspace_shape": [16, 4, 640, 320],
            "header": {
                "trajectory": "cartesian",
                "encoded_matrix": [640, 320, 1],
                "phase_encoding_min": 0,
                "phase_encoding_center": 112,
                "phase_encoding_max": 223,
            },
        }
        self.assertEqual(validate_phase_encoding_record(record), -1)
        self.assertEqual(phase_encoding_window(record), (48, 272))
        record["header"]["encoded_matrix"] = [320, 640, 1]
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_phase_encoding_record(record)


if __name__ == "__main__":
    unittest.main()
