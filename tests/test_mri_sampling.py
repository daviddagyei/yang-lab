from __future__ import annotations

import unittest

import torch

from mri_reconstruction.sampling import (
    CartesianMaskConfig,
    apply_cartesian_mask,
    make_cartesian_mask,
    make_cartesian_mask_for_record,
)


class CartesianMaskTests(unittest.TestCase):
    def test_full_mask_reproduces_kspace_exactly(self) -> None:
        kspace = torch.complex(torch.randn(3, 4, 12, 20), torch.randn(3, 4, 12, 20))
        result = apply_cartesian_mask(kspace, torch.ones(20, dtype=torch.bool))
        self.assertTrue(torch.equal(result, kspace))

    def test_fastmri_center_fractions_and_outer_probability(self) -> None:
        for acceleration, center_lines in ((2, 51), (4, 26), (6, 17), (8, 13)):
            result = make_cartesian_mask(
                320,
                "volume-a",
                CartesianMaskConfig(acceleration=acceleration),
            )
            self.assertEqual(result.center_lines, center_lines)
            expected_probability = (320 / acceleration - center_lines) / (320 - center_lines)
            self.assertAlmostEqual(result.outer_sampling_probability, expected_probability)
            self.assertEqual(result.acquired_lines, 320)
            self.assertAlmostEqual(result.effective_acceleration, 320 / result.sampled_lines)
            start = (320 - center_lines + 1) // 2
            self.assertTrue(result.mask[start : start + center_lines].all())

    def test_random_policy_reaches_nominal_acceleration_in_expectation(self) -> None:
        for acceleration in (2, 4, 6, 8):
            counts = [
                make_cartesian_mask(320, f"volume-{index}", CartesianMaskConfig(acceleration)).sampled_lines
                for index in range(200)
            ]
            self.assertAlmostEqual(sum(counts) / len(counts), 320 / acceleration, delta=1.5)

    def test_sample_identifier_is_deterministic(self) -> None:
        config = CartesianMaskConfig(acceleration=4, seed=17)
        first = make_cartesian_mask(322, "fixed-evaluation-sample", config)
        second = make_cartesian_mask(322, "fixed-evaluation-sample", config)
        other = make_cartesian_mask(322, "different-sample", config)
        self.assertTrue(torch.equal(first.mask, second.mask))
        self.assertFalse(torch.equal(first.mask, other.mask))
        self.assertAlmostEqual(first.effective_acceleration, 322 / first.sampled_lines)

    def test_acquisition_padding_is_excluded_from_mask_and_effective_rate(self) -> None:
        result = make_cartesian_mask(
            320,
            "padded-volume",
            CartesianMaskConfig(acceleration=4),
            acquired_range=(48, 272),
        )
        self.assertFalse(result.mask[:48].any())
        self.assertFalse(result.mask[272:].any())
        self.assertEqual(result.acquired_lines, 224)
        self.assertAlmostEqual(result.effective_acceleration, 224 / result.sampled_lines)

    def test_manifest_helper_uses_volume_id_and_header_window(self) -> None:
        record = {
            "volume_id": "volume-a",
            "kspace_shape": [16, 4, 640, 320],
            "header": {
                "trajectory": "cartesian",
                "encoded_matrix": [640, 320, 1],
                "phase_encoding_min": 0,
                "phase_encoding_center": 112,
                "phase_encoding_max": 223,
            },
        }
        result = make_cartesian_mask_for_record(record, CartesianMaskConfig(4))
        direct = make_cartesian_mask(320, "volume-a", CartesianMaskConfig(4), acquired_range=(48, 272))
        self.assertTrue(torch.equal(result.mask, direct.mask))

    def test_center_alignment_matches_fastmri_for_odd_padding(self) -> None:
        result = make_cartesian_mask(264, "odd-padding", CartesianMaskConfig(acceleration=4))
        self.assertEqual(result.center_lines, 21)
        self.assertTrue(result.mask[122:143].all())

    def test_mask_broadcasts_only_across_phase_axis(self) -> None:
        kspace = torch.ones(2, 3, 5, 7, dtype=torch.complex64)
        mask = torch.tensor([True, False, True, False, False, True, False])
        masked = apply_cartesian_mask(kspace, mask)
        self.assertTrue(torch.equal(masked[..., mask], kspace[..., mask]))
        self.assertEqual(torch.count_nonzero(masked[..., ~mask]).item(), 0)


if __name__ == "__main__":
    unittest.main()
