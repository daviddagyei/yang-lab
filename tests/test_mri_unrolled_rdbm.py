from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mri_reconstruction import adjoint_operator, complex_to_channels, forward_operator
from mri_unrolled_rdbm import (
    UnrolledRDBMTrainer,
    build_unrolled_rdbm,
    magnitude_ssim_loss,
)


def synthetic_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(311)
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


def tiny_model(cascades: int = 3):
    return build_unrolled_rdbm(
        rdbm_config={
            "dim": 8,
            "dim_mults": (1, 2),
            "timesteps": 8,
            "sampling_timesteps": 3,
            "image_size": 16,
            "dc_step_size": 1.0,
        },
        cascades=cascades,
        cg_iterations=3,
        gradient_checkpointing=False,
    )


class UnrolledRDBMTests(unittest.TestCase):
    def test_identical_complex_images_have_zero_ssim_loss(self) -> None:
        image = torch.complex(torch.rand(2, 16, 16), torch.rand(2, 16, 16))
        self.assertAlmostEqual(float(magnitude_ssim_loss(image, image)), 0.0, places=5)

    def test_forward_backward_reaches_shared_prior_and_lambdas(self) -> None:
        model = tiny_model()
        losses = model.training_losses(synthetic_batch())
        self.assertEqual(losses["prediction"].shape, (1, 2, 16, 16))
        for key in ("loss", "complex_l1", "ssim_loss", "measurement"):
            self.assertTrue(torch.isfinite(losses[key]))
        losses["loss"].backward()
        self.assertIsNotNone(model.raw_lambdas.grad)
        self.assertTrue(torch.isfinite(model.raw_lambdas.grad).all())
        self.assertTrue(any(parameter.grad is not None for parameter in model.rdbm.parameters()))

    def test_trajectory_contains_initial_and_each_cascade(self) -> None:
        model = tiny_model(cascades=3).eval()
        batch = synthetic_batch()
        trajectory = model(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            return_trajectory=True,
        )
        self.assertIsInstance(trajectory, list)
        self.assertEqual(len(trajectory), 4)

    def test_checkpoint_round_trip(self) -> None:
        model = tiny_model(cascades=2)
        trainer = UnrolledRDBMTrainer(
            model,
            learning_rate=1e-4,
            device="cpu",
            precision="fp32",
            data_config={"acceleration": 4},
            pretrained_rdbm_step=94900,
        )
        trainer.train_step(synthetic_batch())
        trainer.completed_epochs = 1
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save(Path(directory) / "checkpoint.pt")
            restored = UnrolledRDBMTrainer(
                tiny_model(cascades=2),
                learning_rate=1e-4,
                device="cpu",
                precision="fp32",
                data_config={"acceleration": 4},
            )
            restored.load(path)
            self.assertEqual(restored.step, 1)
            self.assertEqual(restored.completed_epochs, 1)
            self.assertEqual(restored.pretrained_rdbm_step, 94900)
            for expected, actual in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
