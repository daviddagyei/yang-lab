from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from mri_asscgd import (
    ASSCGDTrainer,
    AdversarialASSCGDTrainer,
    ConditionalDiffusionDiscriminator,
    SelfSupervisedMRISliceDataset,
    SelfSupervisedMaskConfig,
    build_asscgd_diffusion,
    heldout_kspace_loss,
    split_acquired_mask,
)
from mri_reconstruction import (
    adjoint_operator,
    channels_to_complex,
    complex_to_channels,
    fft2c,
    forward_operator,
    measurement_residual,
)


class SelfSupervisedMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.acquired = torch.zeros(64, dtype=torch.bool)
        self.acquired[28:37] = True
        self.acquired[[2, 7, 13, 19, 43, 51, 59]] = True
        self.config = SelfSupervisedMaskConfig(input_fraction=0.5, seed=17)

    def test_partition_is_disjoint_complete_and_keeps_acs_in_input(self) -> None:
        partition = split_acquired_mask(
            self.acquired, "volume:0", self.config, center_lines=5, epoch=0
        )
        self.assertFalse(torch.any(partition.input_mask & partition.loss_mask))
        self.assertTrue(torch.equal(partition.input_mask | partition.loss_mask, self.acquired))
        self.assertTrue(torch.all(partition.input_mask[30:35]))
        self.assertFalse(torch.any(partition.loss_mask[30:35]))
        self.assertGreater(partition.input_lines, partition.acs_lines)
        self.assertGreater(partition.loss_lines, 0)

    def test_partition_is_reproducible_and_can_change_by_epoch(self) -> None:
        first = split_acquired_mask(
            self.acquired, "volume:0", self.config, center_lines=5, epoch=0
        )
        repeated = split_acquired_mask(
            self.acquired, "volume:0", self.config, center_lines=5, epoch=0
        )
        changed = split_acquired_mask(
            self.acquired, "volume:0", self.config, center_lines=5, epoch=1
        )
        torch.testing.assert_close(first.input_mask, repeated.input_mask)
        self.assertFalse(torch.equal(first.input_mask, changed.input_mask))


class SelfSupervisedDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source_path = root / "volume.h5"
        self.sensitivity_path = root / "maps.h5"

        height = width = 64
        coils = 4
        generator = torch.Generator().manual_seed(91)
        self.image = torch.complex(
            torch.randn(height, width, generator=generator),
            torch.randn(height, width, generator=generator),
        ).to(torch.complex64)
        self.maps = torch.ones(coils, height, width, dtype=torch.complex64) / np.sqrt(coils)
        kspace = fft2c(self.maps * self.image)
        with h5py.File(self.source_path, "w") as source:
            # Omitting reconstruction_rss proves the target-free loader never reads it.
            source.create_dataset("kspace", data=kspace.unsqueeze(0).numpy())
        with h5py.File(self.sensitivity_path, "w") as source:
            source.create_dataset(
                "sensmap", data=self.maps.permute(1, 2, 0).unsqueeze(0).numpy()
            )

        self.record = {
            "volume_id": "synthetic",
            "path": str(self.source_path),
            "split": "train",
            "acquisition": "AXT2",
            "kspace_shape": [1, coils, height, width],
            "reconstruction_shape": [1, height, width],
            "sensitivity_path": str(self.sensitivity_path),
            "sensitivity_layout": "slice_height_width_coil",
            "header": {
                "trajectory": "cartesian",
                "encoded_matrix": [height, width, 1],
                "phase_encoding_min": 0,
                "phase_encoding_center": width // 2,
                "phase_encoding_max": width - 1,
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_sample_is_target_free_and_has_expected_support(self) -> None:
        sample = SelfSupervisedMRISliceDataset([self.record], split="train")[0]
        self.assertNotIn("target", sample)
        self.assertNotIn("rss_reference", sample)
        self.assertEqual(sample["initial_reconstruction"].shape, (2, 64, 64))
        self.assertEqual(sample["input_kspace"].shape, (4, 64, 64))
        self.assertTrue(torch.all(sample["input_kspace"][..., ~sample["input_mask"]] == 0))
        self.assertTrue(torch.all(sample["loss_kspace"][..., ~sample["loss_mask"]] == 0))
        self.assertFalse(torch.any(sample["input_mask"] & sample["loss_mask"]))
        self.assertTrue(
            torch.equal(sample["input_mask"] | sample["loss_mask"], sample["acquired_mask"])
        )

    def test_heldout_values_do_not_change_model_input_or_scale(self) -> None:
        dataset = SelfSupervisedMRISliceDataset([self.record], split="train")
        before = dataset[0]
        with h5py.File(self.source_path, "r+") as source:
            data = source["kspace"][0]
            data[..., before["loss_mask"].numpy()] *= 1000
            source["kspace"][0] = data
        after = dataset[0]
        torch.testing.assert_close(before["scale"], after["scale"])
        torch.testing.assert_close(before["input_kspace"], after["input_kspace"])
        torch.testing.assert_close(
            before["initial_reconstruction"], after["initial_reconstruction"]
        )
        self.assertFalse(torch.equal(before["loss_kspace"], after["loss_kspace"]))

    def test_validation_partition_stays_fixed_across_epoch_changes(self) -> None:
        record = dict(self.record, split="val")
        dataset = SelfSupervisedMRISliceDataset([record], split="val")
        first = dataset[0]
        dataset.set_epoch(9)
        repeated = dataset[0]
        torch.testing.assert_close(first["input_mask"], repeated["input_mask"])
        self.assertEqual(repeated["metadata"]["partition_epoch"], 0)

    def test_heldout_loss_is_zero_for_matching_image(self) -> None:
        mask = torch.zeros(64, dtype=torch.bool)
        mask[::4] = True
        measured = forward_operator(self.image, self.maps, mask, (64, 64))
        loss = heldout_kspace_loss(self.image, measured, self.maps, mask, kind="nmse")
        self.assertLess(float(loss), 1e-12)


def synthetic_self_supervised_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(173)
    batch_size, coils, height, width = 1, 4, 16, 16
    maps = torch.complex(
        torch.randn(batch_size, coils, height, width, generator=generator),
        torch.randn(batch_size, coils, height, width, generator=generator),
    )
    maps = maps / torch.sqrt(torch.sum(torch.abs(maps) ** 2, dim=1, keepdim=True))
    target = 0.2 * torch.complex(
        torch.randn(batch_size, height, width, generator=generator),
        torch.randn(batch_size, height, width, generator=generator),
    )
    input_mask = torch.zeros(batch_size, width, dtype=torch.bool)
    input_mask[:, [2, 6, 7, 8, 9, 13]] = True
    loss_mask = torch.zeros_like(input_mask)
    loss_mask[:, [0, 4, 11, 15]] = True
    input_kspace = forward_operator(target, maps, input_mask, (20, width))
    loss_kspace = forward_operator(target, maps, loss_mask, (20, width))
    initial = adjoint_operator(input_kspace, maps, input_mask)
    return {
        "initial_reconstruction": complex_to_channels(initial, channel_dim=1),
        "input_kspace": input_kspace,
        "input_mask": input_mask,
        "loss_kspace": loss_kspace,
        "loss_mask": loss_mask,
        "sensitivity_maps": maps,
        "metadata": {"sample_id": ["synthetic:0"]},
    }


class SelfSupervisedDiffusionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(201)
        self.model = build_asscgd_diffusion(
            dim=8,
            dim_mults=(1, 2),
            timesteps=4,
            sampling_timesteps=2,
        )
        self.batch = synthetic_self_supervised_batch()
        self.timestep = torch.tensor([2], dtype=torch.long)
        self.noise = torch.randn_like(self.batch["initial_reconstruction"])

    def test_target_free_forward_backward_is_finite(self) -> None:
        losses = self.model.training_losses(
            self.batch,
            t=self.timestep,
            noise=self.noise,
        )
        self.assertEqual(losses["prediction"].shape, (1, 2, 16, 16))
        for key in ("loss", "heldout_kspace", "input_residual"):
            self.assertTrue(torch.isfinite(losses[key]))
        losses["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_heldout_measurements_affect_loss_but_not_prediction(self) -> None:
        first = self.model.training_losses(
            self.batch,
            t=self.timestep,
            noise=self.noise,
        )
        changed_batch = dict(self.batch)
        changed_batch["loss_kspace"] = self.batch["loss_kspace"] * 2
        changed = self.model.training_losses(
            changed_batch,
            t=self.timestep,
            noise=self.noise,
        )
        torch.testing.assert_close(first["prediction"], changed["prediction"])
        self.assertNotEqual(float(first["loss"].detach()), float(changed["loss"].detach()))

    def test_input_consistency_reduces_input_residual(self) -> None:
        initial = self.batch["initial_reconstruction"]
        noisy = self.model.q_sample(initial, self.timestep, self.noise)
        raw = self.model.denoiser(noisy, initial, self.timestep)
        corrected = self.model.apply_input_consistency(
            raw,
            self.batch["input_kspace"],
            self.batch["sensitivity_maps"],
            self.batch["input_mask"],
        )
        common = (
            self.batch["input_kspace"],
            self.batch["sensitivity_maps"],
            self.batch["input_mask"],
        )
        before = measurement_residual(channels_to_complex(raw, channel_dim=1), *common)
        after = measurement_residual(channels_to_complex(corrected, channel_dim=1), *common)
        self.assertLess(float(after.detach()), float(before.detach()))

    def test_reconstruction_and_checkpoint_round_trip(self) -> None:
        full_mask = self.batch["input_mask"] | self.batch["loss_mask"]
        full_kspace = self.batch["input_kspace"] + self.batch["loss_kspace"]
        full_initial = adjoint_operator(
            full_kspace,
            self.batch["sensitivity_maps"],
            full_mask,
        )
        output = self.model.reconstruct(
            complex_to_channels(full_initial, channel_dim=1),
            full_kspace,
            self.batch["sensitivity_maps"],
            full_mask,
            noise=torch.zeros_like(self.batch["initial_reconstruction"]),
        )
        self.assertEqual(output.shape, (1, 2, 16, 16))
        self.assertTrue(torch.isfinite(output).all())

        trainer = ASSCGDTrainer(self.model, learning_rate=1e-3, device="cpu")
        trainer.train_step(self.batch)
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save(Path(directory) / "checkpoint.pt")
            restored_model = build_asscgd_diffusion(
                dim=8,
                dim_mults=(1, 2),
                timesteps=4,
                sampling_timesteps=2,
            )
            restored = ASSCGDTrainer(restored_model, learning_rate=1e-3, device="cpu")
            restored.load(path)
            self.assertEqual(restored.step, 1)
            for expected, actual in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(expected, actual)


class AdversarialDiffusionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(301)
        self.model = build_asscgd_diffusion(
            dim=8,
            dim_mults=(1, 2),
            timesteps=4,
            sampling_timesteps=2,
        )
        self.discriminator = ConditionalDiffusionDiscriminator(base_channels=4)
        self.batch = synthetic_self_supervised_batch()

    def test_discriminator_has_four_convolutions_and_finite_logits(self) -> None:
        convolutions = [
            module
            for module in self.discriminator.modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        self.assertEqual(len(convolutions), 4)
        candidate = self.batch["initial_reconstruction"]
        logits = self.discriminator(candidate, candidate)
        changed_logits = self.discriminator(candidate * 2, candidate)
        self.assertEqual(logits.shape, (1,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(torch.equal(logits, changed_logits))

    def test_adversarial_step_updates_both_networks_and_is_finite(self) -> None:
        trainer = AdversarialASSCGDTrainer(
            self.model,
            self.discriminator,
            learning_rate=1e-3,
            device="cpu",
        )
        generator_before = next(self.model.parameters()).detach().clone()
        discriminator_before = next(self.discriminator.parameters()).detach().clone()
        metrics = trainer.train_step(self.batch)
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertLessEqual(metrics["effective_adversarial_weight"], 0.100001)
        self.assertEqual(metrics["discriminator_updated"], 1.0)
        second_metrics = trainer.train_step(self.batch)
        self.assertEqual(second_metrics["discriminator_updated"], 0.0)
        self.assertFalse(torch.equal(generator_before, next(self.model.parameters())))
        self.assertFalse(
            torch.equal(discriminator_before, next(self.discriminator.parameters()))
        )

    def test_adversarial_checkpoint_round_trip(self) -> None:
        trainer = AdversarialASSCGDTrainer(
            self.model,
            self.discriminator,
            learning_rate=1e-3,
            device="cpu",
        )
        trainer.train_step(self.batch)
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save(Path(directory) / "adversarial.pt")
            restored = AdversarialASSCGDTrainer(
                build_asscgd_diffusion(
                    dim=8,
                    dim_mults=(1, 2),
                    timesteps=4,
                    sampling_timesteps=2,
                ),
                ConditionalDiffusionDiscriminator(base_channels=4),
                learning_rate=1e-3,
                device="cpu",
            )
            restored.load(path)
            self.assertEqual(restored.step, 1)
            for expected, actual in zip(
                trainer.discriminator.parameters(), restored.discriminator.parameters()
            ):
                torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
