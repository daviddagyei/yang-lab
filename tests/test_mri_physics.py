from __future__ import annotations

import unittest

import torch

from mri_reconstruction import (
    adjoint_operator,
    center_crop,
    center_pad,
    forward_operator,
    gradient_data_consistency,
    image_data_consistency,
    measurement_residual,
    replace_acquired_kspace,
    sense_encode,
)


def normalized_maps(coils: int, height: int, width: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    maps = torch.complex(
        torch.randn(coils, height, width, generator=generator),
        torch.randn(coils, height, width, generator=generator),
    )
    rss = torch.sqrt(torch.sum(torch.abs(maps) ** 2, dim=0))
    return maps / rss


class MRIPhysicsTests(unittest.TestCase):
    def test_center_pad_and_crop_are_adjoint_for_odd_differences(self) -> None:
        generator = torch.Generator().manual_seed(3)
        small = torch.randn(3, 7, 8, generator=generator)
        large = torch.randn(3, 12, 11, generator=generator)
        left = torch.sum(center_pad(small, large.shape[-2:]) * large)
        right = torch.sum(small * center_crop(large, small.shape[-2:]))
        torch.testing.assert_close(left, right)

    def test_forward_and_adjoint_pass_complex_inner_product_test(self) -> None:
        generator = torch.Generator().manual_seed(13)
        image = torch.complex(
            torch.randn(13, 10, generator=generator),
            torch.randn(13, 10, generator=generator),
        )
        maps = normalized_maps(4, 13, 10)
        mask = torch.rand(14, generator=generator) > 0.45
        encoded_shape = (19, 14)
        probe = torch.complex(
            torch.randn(4, *encoded_shape, generator=generator),
            torch.randn(4, *encoded_shape, generator=generator),
        )
        forward = forward_operator(image, maps, mask, encoded_shape)
        adjoint = adjoint_operator(probe, maps, mask)
        torch.testing.assert_close(
            torch.vdot(forward.flatten(), probe.flatten()),
            torch.vdot(image.flatten(), adjoint.flatten()),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_fully_sampled_adjoint_reconstructs_normalized_sense_image(self) -> None:
        generator = torch.Generator().manual_seed(17)
        image = torch.complex(
            torch.randn(16, 12, generator=generator),
            torch.randn(16, 12, generator=generator),
        )
        maps = normalized_maps(5, 16, 12)
        mask = torch.ones(12, dtype=torch.bool)
        reconstructed = adjoint_operator(
            forward_operator(image, maps, mask, (24, 12)),
            maps,
            mask,
        )
        torch.testing.assert_close(reconstructed, image, rtol=2e-5, atol=2e-5)

    def test_gradient_consistency_reduces_measurement_residual(self) -> None:
        generator = torch.Generator().manual_seed(29)
        maps = normalized_maps(4, 18, 16)
        target = torch.complex(
            torch.randn(18, 16, generator=generator),
            torch.randn(18, 16, generator=generator),
        )
        prediction = torch.complex(
            torch.randn(18, 16, generator=generator),
            torch.randn(18, 16, generator=generator),
        )
        mask = torch.rand(16, generator=generator) > 0.7
        measured = forward_operator(target, maps, mask, (26, 16))
        before = measurement_residual(prediction, measured, maps, mask)
        corrected = gradient_data_consistency(prediction, measured, maps, mask)
        after = measurement_residual(corrected, measured, maps, mask)
        self.assertLess(float(after), float(before))

    def test_hard_replacement_is_exact_at_acquired_samples(self) -> None:
        generator = torch.Generator().manual_seed(41)
        maps = normalized_maps(3, 12, 10)
        target = torch.complex(
            torch.randn(12, 10, generator=generator),
            torch.randn(12, 10, generator=generator),
        )
        prediction = torch.zeros_like(target)
        mask = torch.rand(10, generator=generator) > 0.5
        measured = sense_encode(target, maps, (18, 10))
        corrected_kspace = replace_acquired_kspace(prediction, measured, maps, mask)
        torch.testing.assert_close(corrected_kspace[..., mask], measured[..., mask])
        corrected_image = image_data_consistency(prediction, measured, maps, mask)
        self.assertEqual(corrected_image.shape, target.shape)
        self.assertTrue(torch.isfinite(corrected_image).all())

    def test_batched_masks_broadcast_over_coils_and_readout(self) -> None:
        maps = normalized_maps(4, 8, 6).repeat(2, 1, 1, 1)
        images = torch.ones(2, 8, 6, dtype=torch.complex64)
        masks = torch.tensor(
            [[True, False, True, False, True, False], [False, True, False, True, False, True]]
        )
        encoded = forward_operator(images, maps, masks, (10, 6))
        self.assertEqual(encoded.shape, (2, 4, 10, 6))
        self.assertTrue(torch.all(encoded[0, ..., ~masks[0]] == 0))
        self.assertTrue(torch.all(encoded[1, ..., ~masks[1]] == 0))


if __name__ == "__main__":
    unittest.main()
