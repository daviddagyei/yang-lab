from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mri_adadiff import (
    AdaDiffPriorDataset,
    AdaDiffTrainer,
    DiffusionSchedule,
    acquired_kspace_l1,
    build_adadiff_discriminator,
    build_adadiff_generator,
    grouped_prior_batch_schedule,
    hard_data_consistency,
    rapid_reconstruct,
    reconstruct_with_adaptation,
    summarize_evaluation,
)
from mri_reconstruction import (
    channels_to_complex,
    complex_to_channels,
    forward_operator,
    measurement_residual,
)
from mri_adadiff.model import (
    DownsampleConv2d,
    ddpm_conv,
    downsample,
    upfirdn2d_native,
    upfirdn2d_optimized,
    minibatch_standard_deviation,
    upsample,
)
from mri_adadiff.spatial import center_pad_to_size, remove_center_padding


def tiny_generator():
    return build_adadiff_generator(
        image_channels=2,
        base_channels=8,
        channel_multipliers=(1, 2),
        residual_blocks=1,
        attention_resolutions=(4,),
        image_size=8,
        latent_dimension=8,
        style_dimension=16,
        mapping_layers=1,
    )


def tiny_discriminator():
    return build_adadiff_discriminator(
        image_channels=2, base_channels=8, time_dimension=16, depth=2
    )


def synthetic_batch() -> dict[str, torch.Tensor]:
    random = torch.Generator().manual_seed(4)
    target = 0.2 * torch.complex(
        torch.randn(1, 8, 8, generator=random),
        torch.randn(1, 8, 8, generator=random),
    )
    maps = torch.ones(1, 2, 8, 8, dtype=torch.complex64) / (2**0.5)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, [1, 3, 4, 6]] = True
    kspace = forward_operator(target, maps, mask)
    initial = torch.zeros_like(target)
    return {
        "initial_reconstruction": complex_to_channels(initial, channel_dim=1),
        "undersampled_kspace": kspace,
        "sensitivity_maps": maps,
        "mask": mask,
        "target": complex_to_channels(target, channel_dim=1),
    }


class AdaDiffTests(unittest.TestCase):
    def test_fixed_size_padding_roundtrip(self) -> None:
        for shape in ((320, 320), (320, 270), (320, 260), (276, 276), (213, 213)):
            image = torch.randn(2, *shape)
            padded, padding = center_pad_to_size(image, 320)
            self.assertEqual(tuple(padded.shape), (2, 320, 320))
            torch.testing.assert_close(remove_center_padding(padded, padding), image)

    def test_grouped_prior_batches_do_not_mix_spatial_shapes(self) -> None:
        class Source:
            def __len__(self) -> int:
                return 6

            def batch_group_key(self, index: int):
                reconstruction = (8, 8) if index < 4 else (10, 10)
                return ((12, 12), reconstruction)

        dataset = AdaDiffPriorDataset.__new__(AdaDiffPriorDataset)
        dataset.source = Source()
        dataset.quantile = 0.99
        dataset.image_size = 320
        batches = grouped_prior_batch_schedule(dataset, batch_size=2, seed=7)
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(len(batch) == 2 for batch in batches))
        for batch in batches:
            shapes = {dataset.batch_group_key(index) for index in batch}
            self.assertEqual(len(shapes), 1)

    def test_fir_resampling_matches_official_native_math(self) -> None:
        value = torch.randn(1, 2, 6, 6, requires_grad=True)
        kernel_1d = torch.tensor([1.0, 3.0, 3.0, 1.0])
        kernel = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel / kernel.sum()

        inserted = torch.zeros(1, 2, 12, 12)
        inserted[..., ::2, ::2] = value.detach()
        up_weight = torch.flip(4.0 * kernel, (0, 1)).view(1, 1, 4, 4).repeat(2, 1, 1, 1)
        expected_up = F.conv2d(F.pad(inserted, (2, 1, 2, 1)), up_weight, groups=2)
        actual_up = upsample(value, (1, 3, 3, 1))
        torch.testing.assert_close(actual_up, expected_up)

        down_weight = torch.flip(kernel, (0, 1)).view(1, 1, 4, 4).repeat(2, 1, 1, 1)
        expected_down = F.conv2d(
            F.pad(value.detach(), (1, 1, 1, 1)), down_weight, stride=2, groups=2
        )
        actual_down = downsample(value, (1, 3, 3, 1))
        torch.testing.assert_close(actual_down, expected_down)
        (actual_up.square().mean() + actual_down.square().mean()).backward()
        self.assertTrue(torch.isfinite(value.grad).all())

        fused = DownsampleConv2d(2, 3)
        expected_filtered = F.conv2d(
            F.pad(value.detach(), (2, 2, 2, 2)), down_weight, groups=2
        )
        expected_fused = F.conv2d(
            expected_filtered, fused.weight.detach(), fused.bias.detach(), stride=2
        )
        torch.testing.assert_close(fused(value.detach()), expected_fused)

    def test_optimized_fir_matches_reference_values_and_gradients(self) -> None:
        for up, down, pad in ((2, 1, (2, 1)), (1, 2, (1, 1)), (1, 1, (2, 2))):
            torch.manual_seed(17 + up + down)
            reference_input = torch.randn(2, 3, 7, 7, dtype=torch.float64, requires_grad=True)
            optimized_input = reference_input.detach().clone().requires_grad_(True)
            kernel = torch.arange(1.0, 17.0, dtype=torch.float64).reshape(4, 4)
            reference = upfirdn2d_native(
                reference_input, kernel, up=up, down=down, pad=pad
            )
            optimized = upfirdn2d_optimized(
                optimized_input, kernel, up=up, down=down, pad=pad
            )
            torch.testing.assert_close(optimized, reference, rtol=1e-12, atol=1e-12)
            probe = torch.randn_like(reference)
            reference_gradient = torch.autograd.grad((reference * probe).sum(), reference_input)[0]
            optimized_gradient = torch.autograd.grad((optimized * probe).sum(), optimized_input)[0]
            torch.testing.assert_close(
                optimized_gradient, reference_gradient, rtol=1e-12, atol=1e-12
            )

    def test_minibatch_standard_deviation_matches_official_grouping(self) -> None:
        torch.manual_seed(29)
        value = torch.randn(8, 6, 3, 3)
        expected = value.view(4, -1, 1, 6, 3, 3)
        expected = torch.sqrt(expected.var(dim=0, unbiased=False) + 1e-8)
        expected = expected.mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
        expected = expected.repeat(4, 1, 3, 3)
        torch.testing.assert_close(minibatch_standard_deviation(value), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton FIR")
    def test_triton_fir_matches_reference_through_double_backward(self) -> None:
        from mri_adadiff.triton_upfirdn import triton_upfirdn2d

        for up, down, pad in ((2, 1, (2, 1)), (1, 2, (1, 1)), (1, 1, (2, 2))):
            torch.manual_seed(41 + up + down)
            reference_input = torch.randn(
                2, 3, 8, 8, device="cuda", dtype=torch.float32, requires_grad=True
            )
            triton_input = reference_input.detach().clone().requires_grad_(True)
            vector = torch.tensor([1.0, 3.0, 3.0, 1.0], device="cuda")
            kernel = torch.outer(vector, vector) / vector.sum().square()
            reference = upfirdn2d_native(
                reference_input, kernel, up=up, down=down, pad=pad
            )
            actual = triton_upfirdn2d(triton_input, kernel, up=up, down=down, pad=pad)
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
            reference_gradient = torch.autograd.grad(
                reference.square().sum(), reference_input, create_graph=True
            )[0]
            actual_gradient = torch.autograd.grad(
                actual.square().sum(), triton_input, create_graph=True
            )[0]
            torch.testing.assert_close(
                actual_gradient, reference_gradient, rtol=3e-5, atol=3e-6
            )
            reference_second = torch.autograd.grad(
                reference_gradient.square().sum(), reference_input
            )[0]
            actual_second = torch.autograd.grad(
                actual_gradient.square().sum(), triton_input
            )[0]
            torch.testing.assert_close(
                actual_second, reference_second, rtol=5e-5, atol=5e-6
            )

    def test_ddpm_initialization_matches_official_variance_scaling(self) -> None:
        torch.manual_seed(91)
        expected = nn.Conv2d(3, 5, 3, padding=1)
        fan_average = ((3 * 3 * 3) + (5 * 3 * 3)) / 2
        bound = (3.0 * 1e-10 / fan_average) ** 0.5
        nn.init.uniform_(expected.weight, -bound, bound)
        nn.init.zeros_(expected.bias)
        torch.manual_seed(91)
        actual = ddpm_conv(3, 5, 3, padding=1, scale=0.0)
        torch.testing.assert_close(actual.weight, expected.weight)
        torch.testing.assert_close(actual.bias, expected.bias)

    def test_default_configuration_matches_released_multicoil_command(self) -> None:
        generator = build_adadiff_generator()
        discriminator = build_adadiff_discriminator()
        self.assertEqual(generator.base_channels, 64)
        self.assertEqual(generator.channel_multipliers, (1, 1, 1, 2, 2))
        self.assertEqual(generator.residual_blocks, 2)
        self.assertEqual(generator.attention_resolutions, (20,))
        self.assertEqual(generator.latent_dimension, 100)
        self.assertEqual(generator.style_dimension, 256)
        self.assertEqual(discriminator.base_channels, 64)
        self.assertEqual(discriminator.depth, 6)

    def test_schedule_shapes_and_t0_posterior_is_deterministic(self) -> None:
        schedule = DiffusionSchedule(timesteps=8)
        clean = torch.randn(2, 2, 8, 8)
        current, next_step = schedule.sample_pair(clean, torch.tensor([0, 7]))
        self.assertEqual(current.shape, clean.shape)
        self.assertEqual(next_step.shape, clean.shape)
        first = schedule.sample_posterior(clean, next_step, torch.zeros(2, dtype=torch.long))
        second = schedule.sample_posterior(clean, next_step, torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(first, second)

    def test_generator_and_discriminator_backward(self) -> None:
        generator = tiny_generator()
        discriminator = tiny_discriminator()
        image = torch.randn(2, 2, 8, 8)
        time = torch.tensor([1, 3])
        latent = torch.randn(2, 8)
        output = generator(image, time, latent)
        logits = discriminator(output, time, image)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(logits.shape, (2,))
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in generator.parameters()))

    def test_paired_discriminator_forward_preserves_separate_batch_statistics(self) -> None:
        discriminator = tiny_discriminator().eval()
        condition = torch.randn(2, 2, 8, 8)
        first = torch.randn_like(condition)
        second = torch.randn_like(condition)
        time = torch.tensor([1, 3])
        expected_first = discriminator(first, time, condition)
        expected_second = discriminator(second, time, condition)
        actual = discriminator(first, time, condition, second)
        self.assertIsInstance(actual, tuple)
        torch.testing.assert_close(actual[0], expected_first)
        torch.testing.assert_close(actual[1], expected_second)

    def test_hard_data_consistency_reduces_residual(self) -> None:
        batch = synthetic_batch()
        before = measurement_residual(
            torch.zeros(1, 8, 8, dtype=torch.complex64),
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        consistent = hard_data_consistency(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        after = measurement_residual(
            torch.complex(consistent[:, 0], consistent[:, 1]),
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        self.assertLess(float(after), float(before))

    def test_rapid_diffusion_is_reproducible(self) -> None:
        model = tiny_generator().eval()
        batch = synthetic_batch()
        schedule = DiffusionSchedule(timesteps=2)
        first = rapid_reconstruct(model, batch, schedule=schedule, seed=9)
        second = rapid_reconstruct(model, batch, schedule=schedule, seed=9)
        torch.testing.assert_close(first, second)

    def test_rapid_diffusion_restores_non_square_shape(self) -> None:
        batch = synthetic_batch()
        target = channels_to_complex(batch["target"], channel_dim=1)[..., :6]
        maps = batch["sensitivity_maps"][..., :6]
        mask = batch["mask"][..., :6]
        kspace = forward_operator(target, maps, mask)
        rectangular = {
            **batch,
            "initial_reconstruction": batch["initial_reconstruction"][..., :6],
            "undersampled_kspace": kspace,
            "sensitivity_maps": maps,
            "mask": mask,
            "target": batch["target"][..., :6],
        }
        reconstruction = rapid_reconstruct(
            tiny_generator(),
            rectangular,
            schedule=DiffusionSchedule(timesteps=2),
            seed=9,
        )
        self.assertEqual(reconstruction.shape, rectangular["target"].shape)

    def test_adaptation_loss_has_generator_gradients(self) -> None:
        model = tiny_generator()
        batch = synthetic_batch()
        prediction = model(
            batch["initial_reconstruction"], torch.zeros(1, dtype=torch.long), torch.randn(1, 8)
        )
        loss = acquired_kspace_l1(
            prediction,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_adaptation_reenables_gradients_on_frozen_ema_copy(self) -> None:
        model = tiny_generator().eval().requires_grad_(False)
        batch = synthetic_batch()
        result = reconstruct_with_adaptation(
            model,
            batch,
            schedule=DiffusionSchedule(timesteps=2),
            adaptation_iterations=1,
            rapid_initial=batch["initial_reconstruction"],
        )
        self.assertIsNotNone(result.initial_adaptation_loss)
        self.assertIsNotNone(result.final_adaptation_loss)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_trainer_checkpoint_roundtrip(self) -> None:
        trainer = AdaDiffTrainer(
            tiny_generator(),
            tiny_discriminator(),
            schedule=DiffusionSchedule(timesteps=2),
            total_epochs=2,
            data_config={"operator_independent": True},
            device="cpu",
        )
        batch = {"image": torch.randn(2, 2, 8, 8)}
        metrics = trainer.train_step(batch)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        trainer.complete_epoch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save(path)
            restored = AdaDiffTrainer(
                tiny_generator(),
                tiny_discriminator(),
                schedule=DiffusionSchedule(timesteps=2),
                total_epochs=2,
                data_config={"operator_independent": True},
                device="cpu",
            )
            restored.load(path)
            self.assertEqual(restored.step, 1)
            self.assertEqual(restored.completed_epochs, 1)

    def test_volume_balanced_summary(self) -> None:
        rows = []
        for volume_id, psnr in (("a", 10.0), ("a", 14.0), ("b", 20.0)):
            row = {
                "volume_id": volume_id,
                "acquisition": "AXFLAIR",
                "target_energy": 2.0,
            }
            for method in ("zero_filled", "rapid_adadiff", "adadiff"):
                row.update(
                    {
                        f"{method}_error_energy": 1.0,
                        f"{method}_psnr": psnr,
                        f"{method}_ssim": 0.5,
                        f"{method}_measurement_residual": 0.1,
                    }
                )
                if method != "zero_filled":
                    row[f"{method}_reconstruction_seconds"] = 1.0
            rows.append(row)
        summary = summarize_evaluation(rows)
        self.assertAlmostEqual(summary["adadiff"]["psnr"], 16.0)


if __name__ == "__main__":
    unittest.main()
