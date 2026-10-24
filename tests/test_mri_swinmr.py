from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from mri_reconstruction import complex_to_channels
from mri_swinmr import SwinMRTrainer, build_swinmr, summarize_evaluation
from mri_swinmr.training import random_crop_and_augment


def synthetic_batch(*, padded: bool = False) -> dict[str, object]:
    generator = torch.Generator().manual_seed(571)
    batch, coils, height, width = 2, 3, 8, 8
    maps = torch.complex(
        torch.randn(batch, coils, height, width, generator=generator),
        torch.randn(batch, coils, height, width, generator=generator),
    )
    maps = maps / torch.sqrt(torch.sum(torch.abs(maps) ** 2, dim=1, keepdim=True))
    target = 0.1 * torch.complex(
        torch.randn(batch, height, width, generator=generator),
        torch.randn(batch, height, width, generator=generator),
    )
    initial = target + 0.05 * torch.complex(
        torch.randn(batch, height, width, generator=generator),
        torch.randn(batch, height, width, generator=generator),
    )
    counts = torch.tensor([coils, coils])
    if padded:
        maps = torch.cat((maps, torch.zeros(batch, 2, height, width, dtype=maps.dtype)), dim=1)
    return {
        "target": complex_to_channels(target, channel_dim=1),
        "initial_reconstruction": complex_to_channels(initial, channel_dim=1),
        "sensitivity_maps": maps,
        "metadata": {"coils": counts},
    }


def small_model(**kwargs: object):
    return build_swinmr(
        image_size=8,
        window_size=4,
        depths=(2, 2),
        embed_dim=12,
        num_heads=(3, 3),
        mlp_ratio=2,
        drop_path_rate=0,
        perceptual_weight=0,
        **kwargs,
    )


class DummyPerceptual(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.l1_loss(prediction, target)


class SwinMRTests(unittest.TestCase):
    def test_default_configuration_matches_official_backbone(self) -> None:
        model = build_swinmr(perceptual_weight=0)
        self.assertEqual(model.depths, (6, 6, 6, 6, 6, 6))
        self.assertEqual(model.embed_dim, 180)
        self.assertEqual(model.num_heads, (6, 6, 6, 6, 6, 6))
        self.assertEqual(model.window_size, 8)
        self.assertEqual(model.mlp_ratio, 2)
        self.assertEqual(len(model.network.layers), 6)
        self.assertEqual(model.network.conv_first.in_channels, 2)
        self.assertEqual(model.network.conv_last.out_channels, 2)

    def test_forward_backward_and_losses_are_finite(self) -> None:
        model = small_model()
        batch = synthetic_batch()
        losses = model.training_losses(batch)
        self.assertEqual(losses["prediction"].shape, batch["target"].shape)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters())
        )

    def test_zero_padded_coils_do_not_change_loss(self) -> None:
        model = small_model().eval()
        base = model.training_losses(synthetic_batch())
        padded = model.training_losses(synthetic_batch(padded=True))
        torch.testing.assert_close(
            base["coil_image_charbonnier"], padded["coil_image_charbonnier"]
        )
        torch.testing.assert_close(
            base["coil_kspace_charbonnier"], padded["coil_kspace_charbonnier"]
        )

    def test_perceptual_term_is_weighted(self) -> None:
        model = build_swinmr(
            image_size=8, window_size=4, depths=(2,), embed_dim=12,
            num_heads=(3,), drop_path_rate=0, perceptual_weight=0.25,
        )
        losses = model.training_losses(synthetic_batch(), perceptual_loss=DummyPerceptual())
        expected = (
            model.image_weight * losses["coil_image_charbonnier"]
            + model.frequency_weight * losses["coil_kspace_charbonnier"]
            + model.perceptual_weight * losses["perceptual"]
        )
        torch.testing.assert_close(losses["loss"], expected)

    def test_crop_and_augmentation_are_deterministic(self) -> None:
        batch = synthetic_batch()
        first = random_crop_and_augment(
            batch, crop_size=4, generator=torch.Generator().manual_seed(5)
        )
        second = random_crop_and_augment(
            batch, crop_size=4, generator=torch.Generator().manual_seed(5)
        )
        for key in ("initial_reconstruction", "target", "sensitivity_maps"):
            torch.testing.assert_close(first[key], second[key])
        self.assertEqual(first["target"].shape, (2, 2, 4, 4))

    def test_checkpoint_round_trip_restores_ema_and_schedule(self) -> None:
        model = small_model()
        trainer = SwinMRTrainer(
            model,
            device="cpu",
            load_perceptual=False,
            scheduler_milestones=(2, 3),
            data_config={"acceleration": 4},
        )
        trainer.train_step(synthetic_batch())
        trainer.completed_epochs = 1
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save(Path(directory) / "checkpoint.pt")
            restored = SwinMRTrainer(
                small_model(),
                device="cpu",
                load_perceptual=False,
                scheduler_milestones=(2, 3),
                data_config={"acceleration": 4},
            )
            restored.load(path)
            self.assertEqual((restored.step, restored.completed_epochs), (1, 1))
            for expected, actual in zip(trainer.ema_model.parameters(), restored.ema_model.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_summary_weights_volumes_equally(self) -> None:
        def row(volume: str, psnr: float, error: float) -> dict[str, object]:
            return {
                "volume_id": volume, "acquisition": "AXFLAIR", "target_energy": 10.0,
                "zero_filled_error_energy": 5.0, "swinmr_error_energy": error,
                "zero_filled_psnr": 5.0, "swinmr_psnr": psnr,
                "zero_filled_ssim": 0.5, "swinmr_ssim": 0.8,
                "swinmr_reconstruction_seconds": 0.2,
                "zero_filled_measurement_residual": 0.2,
                "swinmr_measurement_residual": 0.1,
            }
        summary = summarize_evaluation([
            row("a", 10.0, 2.0), row("a", 20.0, 2.0), row("b", 40.0, 1.0)
        ])
        self.assertAlmostEqual(summary["swinmr"]["psnr"], 27.5)
        self.assertAlmostEqual(summary["swinmr"]["nmse"], 0.15)


if __name__ == "__main__":
    unittest.main()
