from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from mri_data import (
    MRISliceDataset,
    build_dataloader,
    collate_padded_coils,
    load_manifest,
)
from mri_reconstruction import channels_to_complex, fft2c


class MRIDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source_path = root / "volume.h5"
        self.sensitivity_path = root / "maps.h5"
        self.manifest_path = root / "train.jsonl"

        height = width = 32
        coils = 4
        generator = torch.Generator().manual_seed(91)
        image = torch.complex(
            torch.randn(height, width, generator=generator),
            torch.randn(height, width, generator=generator),
        ).to(torch.complex64)
        maps = torch.ones(coils, height, width, dtype=torch.complex64) / np.sqrt(coils)
        kspace = fft2c(maps * image)
        with h5py.File(self.source_path, "w") as source:
            source.create_dataset("kspace", data=kspace.unsqueeze(0).numpy())
            source.create_dataset("reconstruction_rss", data=torch.abs(image).unsqueeze(0).numpy())
        with h5py.File(self.sensitivity_path, "w") as source:
            source.create_dataset("sensmap", data=maps.permute(1, 2, 0).unsqueeze(0).numpy())

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
        self.manifest_path.write_text(json.dumps(self.record) + "\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loads_masked_normalized_slice_with_expected_shapes(self) -> None:
        dataset = MRISliceDataset(self.manifest_path, split="train")
        sample = dataset[0]
        self.assertEqual(dataset.volume_count, 1)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample["undersampled_kspace"].shape, (4, 32, 32))
        self.assertEqual(sample["sensitivity_maps"].shape, (4, 32, 32))
        self.assertEqual(sample["initial_reconstruction"].shape, (2, 32, 32))
        self.assertEqual(sample["target"].shape, (2, 32, 32))
        self.assertEqual(sample["rss_reference"].shape, (32, 32))
        self.assertEqual(sample["mask"].dtype, torch.bool)
        self.assertEqual(sample["undersampled_kspace"].dtype, torch.complex64)
        self.assertEqual(sample["target"].dtype, torch.float32)
        self.assertEqual(sample["metadata"]["sensitivity_source"], "supplied")
        self.assertTrue(torch.all(sample["undersampled_kspace"][..., ~sample["mask"]] == 0))
        self.assertTrue(torch.isfinite(channels_to_complex(sample["target"], channel_dim=0)).all())

    def test_normalization_scale_does_not_depend_on_rss_target(self) -> None:
        dataset = MRISliceDataset([self.record], split="train")
        before = dataset[0]
        with h5py.File(self.source_path, "r+") as source:
            source["reconstruction_rss"][...] *= 1000
        after = dataset[0]
        torch.testing.assert_close(before["scale"], after["scale"])
        torch.testing.assert_close(before["initial_reconstruction"], after["initial_reconstruction"])
        torch.testing.assert_close(before["target"], after["target"])
        self.assertFalse(torch.equal(before["rss_reference"], after["rss_reference"]))

    def test_loader_batches_one_variable_coil_sample(self) -> None:
        dataset = MRISliceDataset([self.record], split="train")
        batch = next(iter(build_dataloader(dataset)))
        self.assertEqual(batch["undersampled_kspace"].shape, (1, 4, 32, 32))
        self.assertEqual(batch["initial_reconstruction"].shape, (1, 2, 32, 32))
        with self.assertRaisesRegex(ValueError, "batch_size must be 1"):
            build_dataloader(dataset, batch_size=2)

    def test_padded_collation_supports_variable_coil_counts(self) -> None:
        dataset = MRISliceDataset([self.record], split="train")
        four_coils = dataset[0]
        two_coils = dict(four_coils)
        two_coils["undersampled_kspace"] = four_coils[
            "undersampled_kspace"
        ][:2]
        two_coils["sensitivity_maps"] = four_coils[
            "sensitivity_maps"
        ][:2]
        two_coils["metadata"] = {
            **four_coils["metadata"],
            "sample_id": "synthetic-two-coils",
            "coils": 2,
        }
        batch = collate_padded_coils([four_coils, two_coils])
        self.assertEqual(
            batch["undersampled_kspace"].shape, (2, 4, 32, 32)
        )
        self.assertEqual(
            batch["sensitivity_maps"].shape, (2, 4, 32, 32)
        )
        self.assertTrue(
            torch.equal(
                batch["undersampled_kspace"][1, 2:],
                torch.zeros_like(
                    batch["undersampled_kspace"][1, 2:]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["sensitivity_maps"][1, 2:],
                torch.zeros_like(batch["sensitivity_maps"][1, 2:]),
            )
        )

    def test_acceleration_is_configurable_and_reported(self) -> None:
        for acceleration in (2, 4, 6, 8):
            dataset = MRISliceDataset(
                [self.record], split="train", acceleration=acceleration
            )
            sample = dataset[0]
            self.assertEqual(sample["metadata"]["nominal_acceleration"], acceleration)
            self.assertAlmostEqual(
                sample["metadata"]["center_fraction"], 0.32 / acceleration
            )

    def test_manifest_enforces_frozen_split(self) -> None:
        self.assertEqual(load_manifest(self.manifest_path, expected_split="train")[0]["split"], "train")
        with self.assertRaisesRegex(ValueError, "expected 'val'"):
            load_manifest(self.manifest_path, expected_split="val")

    def test_manifest_paths_can_be_relocated_to_another_workstation(self) -> None:
        relocated_record = {
            **self.record,
            "path": "/data/old/nyu-dataset/extracted/raw/volume.h5",
            "sensitivity_path": "/data/old/nyu-dataset/extracted/maps/maps.h5",
        }
        self.manifest_path.write_text(json.dumps(relocated_record) + "\n")
        destination = Path(self.temporary.name) / "new-nyu-dataset"
        record = load_manifest(
            self.manifest_path,
            expected_split="train",
            data_root=destination,
        )[0]
        self.assertEqual(record["path"], str(destination / "extracted/raw/volume.h5"))
        self.assertEqual(
            record["sensitivity_path"],
            str(destination / "extracted/maps/maps.h5"),
        )


if __name__ == "__main__":
    unittest.main()
