from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mri_reconstruction import (
    adjoint_operator,
    channels_to_complex,
    complex_to_channels,
    forward_operator,
)
from mri_variational_network import (
    IPALMOptimizer,
    VariationalNetworkTrainer,
    build_variational_network,
    summarize_evaluation,
)
from mri_variational_network.model import _symmetric_pad, gaussian_rbf


def synthetic_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(317)
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
    initial = torch.sum(
        torch.conj(maps)
        * torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(measured, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ),
            dim=(-2, -1),
        )[..., 2:-2, :],
        dim=1,
    )
    return {
        "target": complex_to_channels(target, channel_dim=1),
        "initial_reconstruction": complex_to_channels(initial, channel_dim=1),
        "undersampled_kspace": measured,
        "sensitivity_maps": maps,
        "mask": mask,
    }


class VariationalNetworkTests(unittest.TestCase):
    def test_default_configuration_matches_official_architecture(self) -> None:
        model = build_variational_network()
        self.assertEqual(model.num_stages, 10)
        self.assertEqual(model.num_filters, 24)
        self.assertEqual(model.kernel_size, 11)
        self.assertEqual(model.num_rbf_weights, 31)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 65530)

    def test_rbf_matches_official_gaussian_definition_and_has_gradients(self) -> None:
        response = torch.tensor([[[[-0.25, 0.5]]]], requires_grad=True)
        weights = torch.tensor([[[-0.04, 0.0, 0.04]]]).squeeze(0).requires_grad_()
        actual = gaussian_rbf(response, weights, vmin=-1.0, vmax=1.0)
        centers = torch.linspace(-1.0, 1.0, 3)
        sigma = 1.0
        expected = torch.sum(
            0.4
            * torch.exp(-((response.unsqueeze(-1) - centers) ** 2) / (2 * sigma**2))
            * weights.view(1, 1, 1, 1, 3),
            dim=-1,
        )
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(response.grad).all())
        self.assertTrue(torch.isfinite(weights.grad).all())

    def test_symmetric_padding_repeats_edges(self) -> None:
        image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        expected = torch.tensor(
            [[[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
               [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]]]]
        )
        torch.testing.assert_close(_symmetric_pad(image, 1), expected)

    def test_forward_backward_trajectory_and_constraints(self) -> None:
        model = build_variational_network(
            num_stages=2,
            num_filters=3,
            kernel_size=3,
            boundary_pad=2,
            num_rbf_weights=5,
            gradient_checkpointing=False,
        )
        batch = synthetic_batch()
        losses = model.training_losses(batch)
        self.assertEqual(losses["prediction"].shape, batch["target"].shape)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        trajectory = model(
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            initial_reconstruction=batch["initial_reconstruction"],
            return_trajectory=True,
        )
        self.assertEqual(len(trajectory), 3)
        with torch.no_grad():
            model.kernel_real.add_(2.0)
            model.kernel_imag.mul_(20.0)
            model.data_term_weights.fill_(-1.0)
        model.project_parameters()
        self.assertLess(
            float(model.kernel_real.mean(dim=(2, 3, 4)).abs().max().detach()),
            1e-6,
        )
        norms = torch.sqrt(
            model.kernel_real.square().sum(dim=(2, 3, 4))
            + model.kernel_imag.square().sum(dim=(2, 3, 4))
        )
        self.assertLessEqual(float(norms.max().detach()), 1.000001)
        self.assertGreaterEqual(float(model.data_term_weights.min().detach()), 0.0)

    def test_stage_matches_variational_gradient_equation_without_regularizer(self) -> None:
        model = build_variational_network(
            num_stages=1,
            num_filters=2,
            kernel_size=3,
            boundary_pad=1,
            num_rbf_weights=5,
            gradient_checkpointing=False,
        ).eval()
        with torch.no_grad():
            model.kernel_real.zero_()
            model.kernel_imag.zero_()
            model.activation_weights.zero_()
            model.data_term_weights.fill_(0.25)
        batch = synthetic_batch()
        initial = channels_to_complex(batch["initial_reconstruction"], channel_dim=1)
        residual = forward_operator(
            initial,
            batch["sensitivity_maps"],
            batch["mask"],
            batch["undersampled_kspace"].shape[-2:],
        ) - batch["undersampled_kspace"]
        expected = initial - 0.25 * adjoint_operator(
            residual,
            batch["sensitivity_maps"],
            batch["mask"],
        )
        actual_channels = model(
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            initial_reconstruction=batch["initial_reconstruction"],
        )
        assert isinstance(actual_channels, torch.Tensor)
        actual = channels_to_complex(actual_channels, channel_dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_checkpoint_round_trip(self) -> None:
        config = dict(
            num_stages=1,
            num_filters=2,
            kernel_size=3,
            boundary_pad=1,
            num_rbf_weights=5,
            gradient_checkpointing=False,
        )
        trainer = VariationalNetworkTrainer(
            build_variational_network(**config),
            device="cpu",
            data_config={"acceleration": 4},
        )
        trainer.train_step(synthetic_batch())
        self.assertEqual(trainer.optimizer_name, "ipalm")
        self.assertIsInstance(trainer.optimizer, IPALMOptimizer)
        self.assertEqual(trainer.optimizer.iterations, 1)
        trainer.completed_epochs = 1
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "checkpoint.pt")
            restored = VariationalNetworkTrainer(
                build_variational_network(**config),
                device="cpu",
                data_config={"acceleration": 4},
            )
            restored.load(checkpoint)
            self.assertEqual(restored.step, 1)
            self.assertEqual(restored.completed_epochs, 1)
            self.assertEqual(
                restored.optimizer.lipschitz,
                trainer.optimizer.lipschitz,
            )
            self.assertEqual(restored.optimizer.iterations, 1)
            for expected, actual in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_ipalm_defaults_match_official_optimizer(self) -> None:
        model = build_variational_network(
            num_stages=1,
            num_filters=2,
            kernel_size=3,
            boundary_pad=1,
            num_rbf_weights=5,
            gradient_checkpointing=False,
        )
        optimizer = IPALMOptimizer(model)
        self.assertEqual(
            optimizer.BLOCK_NAMES,
            ("kernels", "activation_weights", "data_term_weights"),
        )
        self.assertEqual(optimizer.initial_lipschitz, 1000.0)
        self.assertEqual(optimizer.momentum, 0.7)
        self.assertEqual(optimizer.max_backtracking, 40)

    def test_summary_weights_volumes_equally(self) -> None:
        def row(volume: str, psnr: float, error: float) -> dict[str, object]:
            return {
                "volume_id": volume,
                "acquisition": "AXFLAIR",
                "target_energy": 10.0,
                "zero_filled_error_energy": 5.0,
                "variational_network_error_energy": error,
                "zero_filled_psnr": 5.0,
                "variational_network_psnr": psnr,
                "zero_filled_ssim": 0.5,
                "variational_network_ssim": 0.8,
                "variational_network_reconstruction_seconds": 0.2,
                "zero_filled_measurement_residual": 0.2,
                "variational_network_measurement_residual": 0.1,
            }

        summary = summarize_evaluation(
            [row("a", 10.0, 2.0), row("a", 20.0, 2.0), row("b", 40.0, 1.0)]
        )
        self.assertAlmostEqual(summary["variational_network"]["psnr"], 27.5)
        self.assertAlmostEqual(summary["variational_network"]["nmse"], 0.15)


if __name__ == "__main__":
    unittest.main()
