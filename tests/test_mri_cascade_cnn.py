from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mri_cascade_cnn import (
    CascadeCNNTrainer,
    build_cascade_cnn,
    summarize_evaluation,
)
from mri_reconstruction import (
    adjoint_operator,
    complex_to_channels,
    forward_operator,
    measurement_residual,
)


def synthetic_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(419)
    batch, coils, height, width = 1, 3, 12, 10
    maps = torch.complex(
        torch.randn(batch, coils, height, width, generator=generator),
        torch.randn(batch, coils, height, width, generator=generator),
    )
    maps = maps / torch.sqrt(torch.sum(torch.abs(maps) ** 2, dim=1, keepdim=True))
    target = 0.1 * torch.complex(
        torch.randn(batch, height, width, generator=generator),
        torch.randn(batch, height, width, generator=generator),
    )
    mask = torch.zeros(batch, width, dtype=torch.bool)
    mask[:, width // 2 - 1 : width // 2 + 2] = True
    mask[:, ::3] = True
    measured = forward_operator(target, maps, mask, (16, width))
    initial = adjoint_operator(measured, maps, mask)
    return {
        "target": complex_to_channels(target, channel_dim=1),
        "initial_reconstruction": complex_to_channels(initial, channel_dim=1),
        "undersampled_kspace": measured,
        "sensitivity_maps": maps,
        "mask": mask,
    }


class CascadeCNNTests(unittest.TestCase):
    def test_default_configuration_matches_published_d5_c5(self) -> None:
        model = build_cascade_cnn()
        self.assertEqual(model.cascades, 5)
        self.assertEqual(model.depth, 5)
        self.assertEqual(model.features, 64)
        self.assertEqual(len(model.blocks), 5)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 565770)
        self.assertIsNot(model.blocks[0].layers[0].weight, model.blocks[1].layers[0].weight)
        for block in model.blocks:
            self.assertEqual([layer.kernel_size for layer in block.layers], [(3, 3)] * 5)
            self.assertEqual(block.layers[0].in_channels, 2)
            self.assertEqual(block.layers[-1].out_channels, 2)

    def test_forward_backward_and_trajectory_are_finite(self) -> None:
        model = build_cascade_cnn(
            cascades=2,
            depth=3,
            features=4,
            cg_iterations=3,
            gradient_checkpointing=False,
        )
        batch = synthetic_batch()
        losses = model.training_losses(batch)
        self.assertEqual(losses["prediction"].shape, batch["target"].shape)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        trajectory = model(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            return_trajectory=True,
        )
        self.assertEqual(len(trajectory), 3)

    def test_consistency_reduces_measurement_residual_for_identity_cnn(self) -> None:
        model = build_cascade_cnn(
            cascades=1,
            depth=2,
            features=2,
            cg_iterations=10,
            dc_weight=0.05,
            gradient_checkpointing=False,
        ).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        batch = synthetic_batch()
        initial_complex = torch.view_as_complex(
            batch["initial_reconstruction"].movedim(1, -1).contiguous()
        )
        initial_residual = measurement_residual(
            initial_complex,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        output = model(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        assert isinstance(output, torch.Tensor)
        output_complex = torch.view_as_complex(output.movedim(1, -1).contiguous())
        output_residual = measurement_residual(
            output_complex,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        )
        self.assertLess(
            float(output_residual.detach()), float(initial_residual.detach())
        )

    def test_checkpointed_cascade_backpropagates(self) -> None:
        model = build_cascade_cnn(
            cascades=2,
            depth=2,
            features=2,
            cg_iterations=2,
            gradient_checkpointing=True,
        )
        loss = model.training_losses(synthetic_batch())["loss"]
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_checkpoint_round_trip(self) -> None:
        config = dict(cascades=1, depth=2, features=2, cg_iterations=2, gradient_checkpointing=False)
        trainer = CascadeCNNTrainer(
            build_cascade_cnn(**config),
            device="cpu",
            data_config={"acceleration": 4},
        )
        trainer.train_step(synthetic_batch())
        trainer.completed_epochs = 1
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save(Path(directory) / "checkpoint.pt")
            restored = CascadeCNNTrainer(
                build_cascade_cnn(**config),
                device="cpu",
                data_config={"acceleration": 4},
            )
            restored.load(path)
            self.assertEqual((restored.step, restored.completed_epochs), (1, 1))
            for expected, actual in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_summary_weights_volumes_equally(self) -> None:
        def row(volume: str, psnr: float, error: float) -> dict[str, object]:
            return {
                "volume_id": volume,
                "acquisition": "AXFLAIR",
                "target_energy": 10.0,
                "zero_filled_error_energy": 5.0,
                "cascade_cnn_error_energy": error,
                "zero_filled_psnr": 5.0,
                "cascade_cnn_psnr": psnr,
                "zero_filled_ssim": 0.5,
                "cascade_cnn_ssim": 0.8,
                "cascade_cnn_reconstruction_seconds": 0.2,
                "zero_filled_measurement_residual": 0.2,
                "cascade_cnn_measurement_residual": 0.1,
            }
        summary = summarize_evaluation([
            row("a", 10.0, 2.0), row("a", 20.0, 2.0), row("b", 40.0, 1.0)
        ])
        self.assertAlmostEqual(summary["cascade_cnn"]["psnr"], 27.5)
        self.assertAlmostEqual(summary["cascade_cnn"]["nmse"], 0.15)


if __name__ == "__main__":
    unittest.main()
