from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from mri_modl import MoDLTrainer, build_modl, conjugate_gradient, summarize_evaluation
from mri_reconstruction import complex_to_channels, forward_operator


def synthetic_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(211)
    batch, coils, height, width = 1, 3, 12, 10
    maps = torch.complex(
        torch.randn(batch, coils, height, width, generator=generator),
        torch.randn(batch, coils, height, width, generator=generator),
    )
    maps = maps / torch.sqrt(torch.sum(torch.abs(maps) ** 2, dim=1, keepdim=True))
    target = 0.2 * torch.complex(
        torch.randn(batch, height, width, generator=generator),
        torch.randn(batch, height, width, generator=generator),
    )
    mask = torch.zeros(batch, width, dtype=torch.bool)
    mask[:, width // 2 - 1 : width // 2 + 2] = True
    mask[:, ::3] = True
    measured = forward_operator(target, maps, mask, (16, width))
    return {
        "target": complex_to_channels(target, channel_dim=1),
        "undersampled_kspace": measured,
        "sensitivity_maps": maps,
        "mask": mask,
    }


class ConjugateGradientTests(unittest.TestCase):
    def test_solves_batched_complex_diagonal_system(self) -> None:
        diagonal = torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(1, 2, 2)
        expected = torch.complex(
            torch.tensor([[[1.0, -2.0], [0.5, 3.0]]]),
            torch.tensor([[[0.5, 1.0], [-1.0, 2.0]]]),
        )
        right_hand_side = diagonal * expected
        actual = conjugate_gradient(
            lambda value: diagonal * value,
            right_hand_side,
            max_iterations=8,
            tolerance=1e-14,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class MoDLTests(unittest.TestCase):
    def test_forward_backward_is_finite_and_reaches_shared_parameters(self) -> None:
        model = build_modl(
            denoiser_channels=8,
            denoiser_layers=3,
            unrolls=2,
            cg_iterations=4,
        )
        batch = synthetic_batch()
        losses = model.training_losses(batch)
        self.assertEqual(losses["prediction"].shape, batch["target"].shape)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(losses["lambda"].detach()), 0.0)
        losses["loss"].backward()
        self.assertIsNotNone(model.raw_lambda.grad)
        self.assertTrue(torch.isfinite(model.raw_lambda.grad))
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.denoiser.parameters())
        )

    def test_trajectory_has_initial_plus_one_output_per_unroll(self) -> None:
        model = build_modl(
            denoiser_channels=8,
            denoiser_layers=3,
            unrolls=3,
            cg_iterations=2,
        ).eval()
        batch = synthetic_batch()
        trajectory = model(
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            return_trajectory=True,
        )
        self.assertIsInstance(trajectory, list)
        self.assertEqual(len(trajectory), 4)
        self.assertTrue(all(item.shape == batch["target"].shape for item in trajectory))

    def test_checkpoint_can_expand_shared_unrolls(self) -> None:
        batch = synthetic_batch()
        warmup = build_modl(
            denoiser_channels=8,
            denoiser_layers=3,
            unrolls=1,
            cg_iterations=2,
        )
        trainer = MoDLTrainer(warmup, learning_rate=1e-3, device="cpu")
        trainer.train_step(batch)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "checkpoint.pt")
            expanded = build_modl(
                denoiser_channels=8,
                denoiser_layers=3,
                unrolls=3,
                cg_iterations=2,
            )
            resumed = MoDLTrainer(expanded, learning_rate=1e-3, device="cpu")
            resumed.load(checkpoint, allow_unroll_change=True)
            self.assertEqual(resumed.step, 1)
            for expected, actual in zip(trainer.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_summary_weights_volumes_equally(self) -> None:
        def row(volume: str, psnr: float, error: float) -> dict[str, object]:
            return {
                "volume_id": volume,
                "acquisition": "AXT1POST",
                "target_energy": 10.0,
                "zero_filled_error_energy": 5.0,
                "modl_error_energy": error,
                "zero_filled_psnr": 5.0,
                "modl_psnr": psnr,
                "zero_filled_ssim": 0.5,
                "modl_ssim": 0.8,
                "modl_reconstruction_seconds": 0.2,
                "zero_filled_measurement_residual": 0.2,
                "modl_measurement_residual": 0.1,
            }

        summary = summarize_evaluation(
            [row("volume-a", 10.0, 2.0), row("volume-a", 20.0, 2.0), row("volume-b", 40.0, 1.0)]
        )
        self.assertAlmostEqual(summary["modl"]["psnr"], 27.5)
        self.assertAlmostEqual(summary["modl"]["nmse"], 0.15)


if __name__ == "__main__":
    unittest.main()
