from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mri_model import MRITrainer, build_mri_rdbm, reconstruction_metrics, summarize_evaluation
from mri_reconstruction import (
    adjoint_operator,
    complex_to_channels,
    forward_operator,
    measurement_residual,
)


def synthetic_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(73)
    batch, coils, height, width = 1, 4, 16, 16
    maps = torch.complex(
        torch.randn(batch, coils, height, width, generator=generator),
        torch.randn(batch, coils, height, width, generator=generator),
    )
    maps = maps / torch.sqrt(torch.sum(torch.abs(maps) ** 2, dim=1, keepdim=True))
    target = torch.complex(
        torch.randn(batch, height, width, generator=generator),
        torch.randn(batch, height, width, generator=generator),
    ) * 0.2
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


class MRIModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)
        self.model = build_mri_rdbm(
            dim=8,
            dim_mults=(1, 2),
            timesteps=4,
            sampling_timesteps=2,
            image_size=16,
        )
        self.batch = synthetic_batch()

    def test_two_channel_forward_backward_is_finite(self) -> None:
        losses = self.model.training_losses(
            self.batch,
            t=torch.tensor([1]),
            noise=torch.zeros_like(self.batch["target"]),
        )
        self.assertEqual(losses["prediction"].shape, (1, 2, 16, 16))
        for name in ("loss", "complex_l1", "magnitude_l1", "measurement"):
            self.assertTrue(torch.isfinite(losses[name]))
        losses["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_consistency_during_sampling_reduces_residual(self) -> None:
        initial = self.batch["initial_reconstruction"]
        common = (
            self.batch["undersampled_kspace"],
            self.batch["sensitivity_maps"],
            self.batch["mask"],
        )
        torch.manual_seed(5)
        without_dc = self.model.reconstruct(initial, *common, apply_data_consistency=False)
        torch.manual_seed(5)
        with_dc = self.model.reconstruct(
            initial,
            *common,
            apply_data_consistency=True,
            final_consistency_steps=4,
        )
        residual_without = measurement_residual(
            torch.complex(without_dc[:, 0], without_dc[:, 1]), *common
        )
        residual_with = measurement_residual(
            torch.complex(with_dc[:, 0], with_dc[:, 1]), *common
        )
        self.assertLess(float(residual_with), float(residual_without))

    def test_checkpoint_round_trip_restores_step_and_parameters(self) -> None:
        data_config = {"acceleration": 6, "center_fraction": 0.32 / 6, "mask_seed": 7}
        trainer = MRITrainer(
            self.model,
            learning_rate=1e-3,
            device="cpu",
            data_config=data_config,
        )
        metrics = trainer.train_step(self.batch)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save(Path(directory) / "checkpoint.pt")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["model_config"]["dim"], 8)
            restored_model = build_mri_rdbm(
                dim=8,
                dim_mults=(1, 2),
                timesteps=4,
                sampling_timesteps=2,
                image_size=16,
            )
            restored = MRITrainer(
                restored_model,
                learning_rate=1e-3,
                device="cpu",
                data_config=data_config,
            )
            restored.load(path)
            self.assertEqual(restored.step, 1)
            for expected, actual in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(expected, actual)

            mismatched = MRITrainer(
                restored_model,
                learning_rate=1e-3,
                device="cpu",
                data_config={"acceleration": 8, "center_fraction": 0.04, "mask_seed": 7},
            )
            with self.assertRaisesRegex(ValueError, "data configuration differs"):
                mismatched.load(path)

    def test_nonfinite_loss_stops_before_parameter_update(self) -> None:
        trainer = MRITrainer(self.model, learning_rate=1e-3, device="cpu")
        batch = dict(self.batch)
        batch["target"] = torch.full_like(batch["target"], float("nan"))
        before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
        with self.assertRaisesRegex(FloatingPointError, "nonfinite loss"):
            trainer.train_step(batch)
        self.assertEqual(trainer.step, 0)
        for expected, actual in zip(before, trainer.model.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_reconstruction_metrics_identical_images(self) -> None:
        target = torch.complex(
            torch.arange(64, dtype=torch.float32).reshape(8, 8),
            torch.zeros(8, 8),
        )
        metrics = reconstruction_metrics(target, target)
        self.assertEqual(metrics["nmse"], 0.0)
        self.assertEqual(metrics["ssim"], 1.0)
        self.assertTrue(metrics["psnr"] == float("inf"))

    def test_summary_weights_volumes_equally(self) -> None:
        def row(volume: str, psnr: float, error: float) -> dict[str, object]:
            return {
                "volume_id": volume,
                "acquisition": "AXFLAIR",
                "target_energy": 10.0,
                "zero_filled_error_energy": 5.0,
                "rdbm_error_energy": error,
                "zero_filled_psnr": 5.0,
                "rdbm_psnr": psnr,
                "zero_filled_ssim": 0.5,
                "rdbm_ssim": 0.8,
                "rdbm_reconstruction_seconds": 0.25,
                "zero_filled_measurement_residual": 0.2,
                "rdbm_measurement_residual": 0.1,
            }

        summary = summarize_evaluation(
            [row("volume-a", 10.0, 2.0), row("volume-a", 20.0, 2.0), row("volume-b", 40.0, 1.0)]
        )
        self.assertEqual(summary["volume_count"], 2)
        self.assertEqual(summary["slice_count"], 3)
        self.assertAlmostEqual(summary["rdbm"]["psnr"], 27.5)
        self.assertAlmostEqual(summary["rdbm"]["nmse"], 0.15)
        self.assertAlmostEqual(summary["rdbm"]["seconds_per_slice"], 0.25)


if __name__ == "__main__":
    unittest.main()
