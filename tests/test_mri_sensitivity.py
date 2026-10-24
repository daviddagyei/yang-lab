from __future__ import annotations

import unittest
import numpy as np
import torch

from mri_reconstruction import fft2c
from mri_sensitivity import (
    normalize_sensitivities,
    reconstruct_sense,
    standardize_sensitivity_slice,
    validate_sensitivities,
)


class SensitivityMapTests(unittest.TestCase):
    def test_standardizes_hwc_layout_and_center_crops_mismatches(self) -> None:
        data = np.ones((320, 320, 4), dtype=np.complex64) / 2
        maps = standardize_sensitivity_slice(data, "slice_height_width_coil", (276, 276))
        self.assertEqual(maps.shape, (4, 276, 276))
        validate_sensitivities(maps, expected_coils=4, expected_shape=(276, 276))

    def test_sense_convention_reconstructs_complex_object(self) -> None:
        maps = torch.ones(4, 24, 20, dtype=torch.complex64) / 2
        image = torch.complex(torch.randn(24, 20), torch.randn(24, 20))
        kspace = fft2c(maps * image)
        reconstruction = reconstruct_sense(kspace, maps, image.shape)
        torch.testing.assert_close(reconstruction, image, rtol=1e-5, atol=1e-5)

    def test_normalization_and_validation_reject_nonfinite_maps(self) -> None:
        maps = torch.complex(torch.rand(3, 12, 10), torch.rand(3, 12, 10))
        normalized = normalize_sensitivities(maps)
        metrics = validate_sensitivities(normalized)
        self.assertAlmostEqual(metrics.rss_median, 1.0, places=5)
        normalized[0, 0, 0] = torch.complex(torch.tensor(float("nan")), torch.tensor(0.0))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_sensitivities(normalized)

if __name__ == "__main__":
    unittest.main()
