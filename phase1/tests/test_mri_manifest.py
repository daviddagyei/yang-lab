from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from phase1.mri_manifest import (
    MRIManifestBuilder,
    ManifestConfig,
    ManifestError,
    ManifestWriter,
    PatientStratifiedSplitter,
)


def ismrmrd_header(coils: int = 4) -> bytes:
    return f"""<?xml version="1.0"?>
    <ismrmrdHeader>
      <acquisitionSystemInformation><receiverChannels>{coils}</receiverChannels></acquisitionSystemInformation>
      <encoding>
        <encodedSpace><matrixSize><x>8</x><y>6</y><z>1</z></matrixSize></encodedSpace>
        <reconSpace><matrixSize><x>6</x><y>6</y><z>1</z></matrixSize></reconSpace>
        <trajectory>cartesian</trajectory>
        <encodingLimits><kspace_encoding_step_1><minimum>0</minimum><maximum>5</maximum><center>3</center></kspace_encoding_step_1></encodingLimits>
      </encoding>
    </ismrmrdHeader>""".encode()


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "multicoil_train 2"
        self.sens = self.root / "sensmap_output"
        self.raw.mkdir()
        self.sens.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create_volume(
        self,
        name: str,
        patient: str,
        acquisition: str,
        *,
        with_sensitivity: bool = False,
        kspace_dtype: np.dtype = np.dtype("complex64"),
    ) -> None:
        with h5py.File(self.raw / f"{name}.h5", "w") as target:
            target.create_dataset("kspace", data=np.ones((2, 4, 8, 6), dtype=kspace_dtype))
            target.create_dataset("reconstruction_rss", data=np.ones((2, 6, 6), dtype="float32"))
            target.create_dataset("ismrmrd_header", data=ismrmrd_header())
            target.attrs["patient_id"] = patient
            target.attrs["acquisition"] = acquisition
            target.attrs["norm"] = 1.5
            target.attrs["max"] = 2.5
        if with_sensitivity:
            with h5py.File(self.sens / f"{name}_sensmap.h5", "w") as target:
                target.create_dataset("sensmap", data=np.ones((2, 6, 6, 4), dtype="complex64"))

    def test_build_split_and_write_are_deterministic_and_patient_exclusive(self) -> None:
        acquisitions = ["AXT2"] * 8 + ["AXFLAIR"] * 5
        for index, acquisition in enumerate(acquisitions):
            self.create_volume(
                f"volume_{index:02d}",
                patient=f"patient_{index:02d}",
                acquisition=acquisition,
                with_sensitivity=index % 2 == 0,
            )

        config = ManifestConfig(self.root, value_check="sampled")
        records, warnings = MRIManifestBuilder(config).build()
        splitter = PatientStratifiedSplitter(seed=17)
        first = splitter.split(records)
        second = splitter.split(list(reversed(records)))

        self.assertEqual(first, second)
        self.assertEqual(len(records), 13)
        self.assertFalse(warnings)
        self.assertEqual({first[record.volume_id] for record in records}, {"train", "val", "test"})

        output = self.root / "output"
        summary = ManifestWriter().write(
            output,
            records,
            first,
            warnings,
            source_root=self.root,
            seed=17,
            ratios=splitter.ratios,
            value_check=config.value_check,
        )
        self.assertEqual(summary["total_volumes"], 13)
        rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
        patient_splits: dict[str, set[str]] = {}
        for row in rows:
            patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in patient_splits.values()))

    def test_rejects_non_complex_kspace(self) -> None:
        self.create_volume("bad", "patient", "AXT2", kspace_dtype=np.dtype("float32"))
        with self.assertRaisesRegex(ManifestError, "kspace must be complex-valued"):
            MRIManifestBuilder(ManifestConfig(self.root)).build()

    def test_rejects_non_finite_normalization_attribute(self) -> None:
        self.create_volume("bad_norm", "patient", "AXT2")
        with h5py.File(self.raw / "bad_norm.h5", "a") as target:
            target.attrs["norm"] = np.inf
        with self.assertRaisesRegex(ManifestError, "attribute 'norm' must be finite"):
            MRIManifestBuilder(ManifestConfig(self.root)).build()

    def test_keeps_multiple_volumes_from_one_patient_together(self) -> None:
        self.create_volume("first", "shared", "AXT1")
        self.create_volume("second", "shared", "AXT1POST")
        for index in range(5):
            self.create_volume(f"other_{index}", f"patient_{index}", "AXT1")
        records, _ = MRIManifestBuilder(ManifestConfig(self.root)).build()
        assignments = PatientStratifiedSplitter(seed=3).split(records)
        self.assertEqual(assignments["first"], assignments["second"])


if __name__ == "__main__":
    unittest.main()
