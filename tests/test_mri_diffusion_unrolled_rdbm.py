from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mri_diffusion_unrolled_rdbm import (
    DiffusionUnrolledRDBMTrainer,
    build_diffusion_unrolled_rdbm,
    reverse_time_pairs,
    summarize_evaluation,
)
from mri_reconstruction import adjoint_operator, complex_to_channels, forward_operator


def synthetic_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(624)
    batch, coils, height, width = 1, 3, 16, 16
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
    mask[:, width // 2 - 2 : width // 2 + 2] = True
    mask[:, ::4] = True
    measured = forward_operator(target, maps, mask, (20, width))
    initial = adjoint_operator(measured, maps, mask)
    return {
        "initial_reconstruction": complex_to_channels(initial, channel_dim=1),
        "target": complex_to_channels(target, channel_dim=1),
        "undersampled_kspace": measured,
        "sensitivity_maps": maps,
        "mask": mask,
    }


def tiny_model(*, gradient_checkpointing: bool = False):
    return build_diffusion_unrolled_rdbm(
        rdbm_config={
            "dim": 8,
            "dim_mults": (1, 2),
            "timesteps": 8,
            "sampling_timesteps": 3,
            "image_size": 16,
            "dc_step_size": 1.0,
        },
        cg_iterations=2,
        gradient_checkpointing=gradient_checkpointing,
    )


class DiffusionUnrolledRDBMTests(unittest.TestCase):
    def test_schedule_exactly_matches_original_ten_step_sampler(self) -> None:
        self.assertEqual(
            reverse_time_pairs(100, 10),
            (
                (99, 89), (89, 79), (79, 69), (69, 59), (59, 49),
                (49, 39), (39, 29), (29, 19), (19, 9), (9, -1),
            ),
        )

    def test_forward_backward_crosses_every_stage(self) -> None:
        model = tiny_model()
        losses = model.training_losses(synthetic_batch())
        self.assertEqual(model.unroll_steps, 3)
        self.assertEqual(losses["prediction"].shape, (1, 2, 16, 16))
        for key in ("loss", "complex_l1", "ssim_loss", "measurement"):
            self.assertTrue(torch.isfinite(losses[key]))
        losses["loss"].backward()
        self.assertIsNotNone(model.raw_lambdas.grad)
        self.assertTrue(torch.isfinite(model.raw_lambdas.grad).all())
        self.assertTrue(any(parameter.grad is not None for parameter in model.rdbm.parameters()))

    def test_trajectory_contains_initial_and_all_clean_estimates(self) -> None:
        model = tiny_model().eval()
        batch = synthetic_batch()
        trajectory = model(
            batch["initial_reconstruction"], batch["undersampled_kspace"],
            batch["sensitivity_maps"], batch["mask"], return_trajectory=True,
        )
        self.assertIsInstance(trajectory, list)
        self.assertEqual(len(trajectory), model.unroll_steps + 1)

    def test_early_exit_matches_requested_trajectory_stage(self) -> None:
        model = tiny_model().eval()
        batch = synthetic_batch()
        trajectory = model(
            batch["initial_reconstruction"], batch["undersampled_kspace"],
            batch["sensitivity_maps"], batch["mask"], return_trajectory=True,
        )
        early = model(
            batch["initial_reconstruction"], batch["undersampled_kspace"],
            batch["sensitivity_maps"], batch["mask"], max_stages=2,
        )
        assert isinstance(trajectory, list)
        assert isinstance(early, torch.Tensor)
        torch.testing.assert_close(early, trajectory[2])
        with self.assertRaisesRegex(ValueError, "max_stages"):
            model(
                batch["initial_reconstruction"], batch["undersampled_kspace"],
                batch["sensitivity_maps"], batch["mask"], max_stages=0,
            )

    def test_bridge_transition_matches_original_pred_x_start_equation(self) -> None:
        model = tiny_model()
        generator = torch.Generator().manual_seed(7)
        state = torch.randn(1, 2, 16, 16, generator=generator)
        initial = torch.randn(1, 2, 16, 16, generator=generator)
        clean = torch.randn(1, 2, 16, 16, generator=generator)
        time, time_next = 4, 1
        ratio = model.rdbm.Sigma[time_next] / model.rdbm.Sigma[time]
        expected = (
            initial + ratio * (state - initial)
            + (model.rdbm.Theta[time_next] - model.rdbm.Theta[time] * ratio)
            * (clean - initial)
        )
        actual = model._bridge_transition(state, initial, clean, time, time_next)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            model._bridge_transition(state, initial, clean, 1, -1), clean
        )

    def test_checkpoint_round_trip_is_explicitly_scratch(self) -> None:
        trainer = DiffusionUnrolledRDBMTrainer(
            tiny_model(gradient_checkpointing=True),
            learning_rate=1e-4, device="cpu", precision="fp32",
            data_config={"acceleration": 4},
        )
        trainer.train_step(synthetic_batch())
        trainer.completed_epochs = 1
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save(Path(directory) / "checkpoint.pt")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["initialization"], "scratch")
            restored = DiffusionUnrolledRDBMTrainer(
                tiny_model(), learning_rate=1e-4, device="cpu", precision="fp32",
                data_config={"acceleration": 4},
            )
            restored.load(path)
            self.assertEqual((restored.step, restored.completed_epochs), (1, 1))
            self.assertFalse(restored.model.gradient_checkpointing)
            for expected, actual in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_summary_weights_volumes_equally(self) -> None:
        method = "diffusion_unrolled_rdbm"
        def row(volume: str, psnr: float, error: float) -> dict[str, object]:
            return {
                "volume_id": volume, "acquisition": "AXFLAIR", "target_energy": 10.0,
                "zero_filled_error_energy": 5.0, f"{method}_error_energy": error,
                "zero_filled_psnr": 5.0, f"{method}_psnr": psnr,
                "zero_filled_ssim": 0.5, f"{method}_ssim": 0.8,
                f"{method}_reconstruction_seconds": 0.2,
                "zero_filled_measurement_residual": 0.2,
                f"{method}_measurement_residual": 0.1,
            }
        summary = summarize_evaluation([
            row("a", 10.0, 2.0), row("a", 20.0, 2.0), row("b", 40.0, 1.0)
        ])
        self.assertAlmostEqual(summary[method]["psnr"], 27.5)
        self.assertAlmostEqual(summary[method]["nmse"], 0.15)


if __name__ == "__main__":
    unittest.main()
