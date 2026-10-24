"""Dataset discovery, validation, and patient-level splitting for brain MRI HDF5 data."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


SPLITS = ("train", "val", "test")


class ManifestError(RuntimeError):
    """Raised when source data cannot produce a trustworthy manifest."""


@dataclass(frozen=True)
class HeaderMetadata:
    trajectory: str | None
    encoded_matrix: tuple[int, int, int] | None
    reconstruction_matrix: tuple[int, int, int] | None
    phase_encoding_min: int | None
    phase_encoding_max: int | None
    phase_encoding_center: int | None
    receiver_channels: int | None


@dataclass(frozen=True)
class VolumeRecord:
    volume_id: str
    path: str
    patient_id: str
    acquisition: str
    kspace_shape: tuple[int, int, int, int]
    kspace_dtype: str
    reconstruction_shape: tuple[int, int, int]
    reconstruction_dtype: str
    norm: float | None
    maximum: float | None
    header: HeaderMetadata
    sensitivity_path: str | None = None
    sensitivity_shape: tuple[int, int, int, int] | None = None
    sensitivity_dtype: str | None = None
    sensitivity_layout: str | None = None
    sensitivity_spatial_match: bool | None = None

    @property
    def slices(self) -> int:
        return self.kspace_shape[0]

    @property
    def coils(self) -> int:
        return self.kspace_shape[1]

    def to_dict(self, split: str | None = None) -> dict[str, Any]:
        result = asdict(self)
        if split is not None:
            result["split"] = split
        return result


@dataclass(frozen=True)
class ManifestConfig:
    data_root: Path
    raw_subdirectory: str = "multicoil_train 2"
    sensitivity_subdirectory: str = "sensmap_output"
    value_check: str = "sampled"

    def __post_init__(self) -> None:
        if self.value_check not in {"none", "sampled", "full"}:
            raise ValueError("value_check must be one of: none, sampled, full")

    @property
    def raw_directory(self) -> Path:
        return self.data_root / self.raw_subdirectory

    @property
    def sensitivity_directory(self) -> Path:
        return self.data_root / self.sensitivity_subdirectory


class H5VolumeInspector:
    """Validate one raw volume and extract only the metadata needed downstream."""

    REQUIRED_DATASETS = ("kspace", "reconstruction_rss", "ismrmrd_header")

    def __init__(self, value_check: str = "sampled") -> None:
        if value_check not in {"none", "sampled", "full"}:
            raise ValueError("value_check must be one of: none, sampled, full")
        self.value_check = value_check

    def inspect(self, raw_path: Path, sensitivity_path: Path | None) -> tuple[VolumeRecord, list[str]]:
        warnings: list[str] = []
        with h5py.File(raw_path, "r") as source:
            missing = [name for name in self.REQUIRED_DATASETS if name not in source]
            if missing:
                raise ManifestError(f"missing datasets: {', '.join(missing)}")

            kspace = source["kspace"]
            reference = source["reconstruction_rss"]
            self._validate_raw_shapes(kspace, reference)
            self._check_values(kspace, "kspace", coil_axis=1)
            self._check_values(reference, "reconstruction_rss")

            patient_id = self._required_attribute(source, "patient_id")
            acquisition = self._required_attribute(source, "acquisition")
            header = self._parse_header(source["ismrmrd_header"][()])

            if header.trajectory not in {None, "cartesian"}:
                warnings.append(f"{raw_path.name}: trajectory is {header.trajectory!r}, not Cartesian")
            if header.receiver_channels not in {None, kspace.shape[1]}:
                warnings.append(
                    f"{raw_path.name}: header reports {header.receiver_channels} receiver channels, "
                    f"but k-space contains {kspace.shape[1]}"
                )
            if header.encoded_matrix is not None and tuple(kspace.shape[-2:]) != header.encoded_matrix[:2]:
                warnings.append(
                    f"{raw_path.name}: k-space spatial shape {tuple(kspace.shape[-2:])} differs "
                    f"from encoded matrix {header.encoded_matrix[:2]}"
                )

            sensitivity = self._inspect_sensitivity(
                sensitivity_path, tuple(kspace.shape), tuple(reference.shape)
            )

            record = VolumeRecord(
                volume_id=raw_path.stem,
                path=str(raw_path.resolve()),
                patient_id=patient_id,
                acquisition=acquisition,
                kspace_shape=tuple(int(value) for value in kspace.shape),
                kspace_dtype=str(kspace.dtype),
                reconstruction_shape=tuple(int(value) for value in reference.shape),
                reconstruction_dtype=str(reference.dtype),
                norm=self._optional_float_attribute(source, "norm"),
                maximum=self._optional_float_attribute(source, "max"),
                header=header,
                **sensitivity,
            )
        return record, warnings

    @staticmethod
    def _validate_raw_shapes(kspace: h5py.Dataset, reference: h5py.Dataset) -> None:
        if kspace.ndim != 4:
            raise ManifestError(f"kspace must be [slice, coil, readout, phase], got {kspace.shape}")
        if not np.issubdtype(kspace.dtype, np.complexfloating):
            raise ManifestError(f"kspace must be complex-valued, got {kspace.dtype}")
        if reference.ndim != 3:
            raise ManifestError(f"reconstruction_rss must be [slice, height, width], got {reference.shape}")
        if not np.issubdtype(reference.dtype, np.floating):
            raise ManifestError(f"reconstruction_rss must be real-valued, got {reference.dtype}")
        if kspace.shape[0] != reference.shape[0]:
            raise ManifestError(
                f"slice mismatch between kspace {kspace.shape[0]} and reconstruction {reference.shape[0]}"
            )
        if any(size <= 0 for size in (*kspace.shape, *reference.shape)):
            raise ManifestError("datasets cannot contain empty dimensions")

    def _inspect_sensitivity(
        self,
        path: Path | None,
        kspace_shape: tuple[int, ...],
        reference_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        if path is None:
            return {}
        with h5py.File(path, "r") as source:
            if "sensmap" not in source:
                raise ManifestError(f"{path.name}: missing sensmap dataset")
            dataset = source["sensmap"]
            if dataset.ndim != 4 or not np.issubdtype(dataset.dtype, np.complexfloating):
                raise ManifestError(f"{path.name}: sensmap must be rank-4 complex data, got {dataset.shape} {dataset.dtype}")
            if dataset.shape[0] != kspace_shape[0]:
                raise ManifestError(f"{path.name}: sensitivity-map slice count does not match k-space")

            if dataset.shape[-1] == kspace_shape[1]:
                layout = "slice_height_width_coil"
                spatial_shape = tuple(dataset.shape[1:3])
                coil_axis = 3
            elif dataset.shape[1] == kspace_shape[1]:
                layout = "slice_coil_height_width"
                spatial_shape = tuple(dataset.shape[2:4])
                coil_axis = 1
            else:
                raise ManifestError(
                    f"{path.name}: no sensmap axis matches k-space coil count {kspace_shape[1]}"
                )

            self._check_values(dataset, "sensmap", coil_axis=coil_axis)
            return {
                "sensitivity_path": str(path.resolve()),
                "sensitivity_shape": tuple(int(value) for value in dataset.shape),
                "sensitivity_dtype": str(dataset.dtype),
                "sensitivity_layout": layout,
                "sensitivity_spatial_match": spatial_shape == tuple(reference_shape[-2:]),
            }

    def _check_values(self, dataset: h5py.Dataset, name: str, coil_axis: int | None = None) -> None:
        if self.value_check == "none":
            return
        if self.value_check == "full":
            chunks: Iterable[np.ndarray] = (dataset[index] for index in range(dataset.shape[0]))
        else:
            indices = sorted({0, dataset.shape[0] // 2, dataset.shape[0] - 1})
            rows = sorted({0, dataset.shape[-2] // 2, dataset.shape[-2] - 1})
            chunks = (self._read_row(dataset, index, row, coil_axis) for index in indices for row in rows)
        if any(not np.isfinite(chunk).all() for chunk in chunks):
            raise ManifestError(f"{name} contains non-finite values")

    @staticmethod
    def _read_row(dataset: h5py.Dataset, index: int, row: int, coil_axis: int | None) -> np.ndarray:
        if dataset.ndim == 3:
            return dataset[index, row, :]
        if coil_axis == 1:
            return dataset[index, :, row, :]
        if coil_axis == 3:
            return dataset[index, row, :, :]
        raise ManifestError(f"cannot sample-check dataset with shape {dataset.shape}")

    @staticmethod
    def _required_attribute(source: h5py.File, name: str) -> str:
        if name not in source.attrs:
            raise ManifestError(f"missing required attribute {name!r}")
        value = source.attrs[name]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result = str(value).strip()
        if not result:
            raise ManifestError(f"attribute {name!r} is empty")
        return result

    @staticmethod
    def _optional_float_attribute(source: h5py.File, name: str) -> float | None:
        if name not in source.attrs:
            return None
        value = float(source.attrs[name])
        if not math.isfinite(value):
            raise ManifestError(f"attribute {name!r} must be finite")
        return value

    @staticmethod
    def _parse_header(raw_header: Any) -> HeaderMetadata:
        if isinstance(raw_header, np.ndarray):
            raw_header = raw_header.item()
        if isinstance(raw_header, bytes):
            raw_header = raw_header.decode("utf-8")
        try:
            root = ET.fromstring(str(raw_header))
        except ET.ParseError as error:
            raise ManifestError(f"invalid ISMRMRD XML header: {error}") from error
        for element in root.iter():
            element.tag = element.tag.rsplit("}", 1)[-1]

        def integer(path: str) -> int | None:
            element = root.find(path)
            return int(element.text) if element is not None and element.text is not None else None

        def matrix(path: str) -> tuple[int, int, int] | None:
            values = tuple(integer(f"{path}/{axis}") for axis in ("x", "y", "z"))
            return values if all(value is not None for value in values) else None  # type: ignore[return-value]

        trajectory = root.findtext("./encoding/trajectory")
        return HeaderMetadata(
            trajectory=trajectory.strip().lower() if trajectory else None,
            encoded_matrix=matrix("./encoding/encodedSpace/matrixSize"),
            reconstruction_matrix=matrix("./encoding/reconSpace/matrixSize"),
            phase_encoding_min=integer("./encoding/encodingLimits/kspace_encoding_step_1/minimum"),
            phase_encoding_max=integer("./encoding/encodingLimits/kspace_encoding_step_1/maximum"),
            phase_encoding_center=integer("./encoding/encodingLimits/kspace_encoding_step_1/center"),
            receiver_channels=integer("./acquisitionSystemInformation/receiverChannels"),
        )


class MRIManifestBuilder:
    """Discover raw volumes, pair sensitivity maps, and aggregate validation errors."""

    def __init__(self, config: ManifestConfig) -> None:
        self.config = config
        self.inspector = H5VolumeInspector(config.value_check)

    def build(self) -> tuple[list[VolumeRecord], list[str]]:
        raw_paths = sorted(self.config.raw_directory.glob("*.h5"))
        if not raw_paths:
            raise ManifestError(f"no HDF5 files found in {self.config.raw_directory}")
        sensitivity_index = self._sensitivity_index()
        records: list[VolumeRecord] = []
        warnings: list[str] = []
        failures: list[str] = []

        raw_ids = {path.stem for path in raw_paths}
        orphaned = sorted(set(sensitivity_index) - raw_ids)
        if orphaned:
            warnings.append(f"{len(orphaned)} sensitivity maps have no matching raw volume")

        for raw_path in raw_paths:
            try:
                record, file_warnings = self.inspector.inspect(raw_path, sensitivity_index.get(raw_path.stem))
                records.append(record)
                warnings.extend(file_warnings)
            except (OSError, ValueError, ManifestError) as error:
                failures.append(f"{raw_path.name}: {error}")

        if failures:
            preview = "\n".join(f"- {failure}" for failure in failures[:20])
            suffix = f"\n... and {len(failures) - 20} more" if len(failures) > 20 else ""
            raise ManifestError(f"validation failed for {len(failures)} volume(s):\n{preview}{suffix}")
        return records, warnings

    def _sensitivity_index(self) -> dict[str, Path]:
        if not self.config.sensitivity_directory.exists():
            return {}
        result: dict[str, Path] = {}
        for path in sorted(self.config.sensitivity_directory.glob("*_sensmap.h5")):
            volume_id = path.stem.removesuffix("_sensmap")
            if volume_id in result:
                raise ManifestError(f"duplicate sensitivity maps for {volume_id}")
            result[volume_id] = path
        return result


class PatientStratifiedSplitter:
    """Deterministically split patient groups while approximately preserving acquisition mix."""

    def __init__(
        self,
        seed: int = 20261019,
        ratios: Mapping[str, float] | None = None,
    ) -> None:
        self.seed = seed
        self.ratios = dict(ratios or {"train": 0.8, "val": 0.1, "test": 0.1})
        if set(self.ratios) != set(SPLITS):
            raise ValueError(f"ratios must define exactly {SPLITS}")
        if any(value < 0 for value in self.ratios.values()) or not math.isclose(sum(self.ratios.values()), 1.0):
            raise ValueError("split ratios must be non-negative and sum to 1")

    def split(self, records: Sequence[VolumeRecord]) -> dict[str, str]:
        by_patient: dict[str, list[VolumeRecord]] = defaultdict(list)
        for record in records:
            by_patient[record.patient_id].append(record)

        strata: dict[str, list[str]] = defaultdict(list)
        for patient_id, patient_records in by_patient.items():
            label = "+".join(sorted({record.acquisition for record in patient_records}))
            strata[label].append(patient_id)

        ordered_strata = {
            label: sorted(patient_ids, key=lambda patient: self._stable_key(label, patient))
            for label, patient_ids in sorted(strata.items())
        }
        allocations = {
            label: self._allocate_counts(len(patient_ids))
            for label, patient_ids in ordered_strata.items()
        }
        self._rebalance_to_global_target(allocations, ordered_strata)

        patient_split: dict[str, str] = {}
        for label, ordered in ordered_strata.items():
            counts = allocations[label]
            offset = 0
            for split_name in SPLITS:
                for patient_id in ordered[offset : offset + counts[split_name]]:
                    patient_split[patient_id] = split_name
                offset += counts[split_name]

        assignments = {record.volume_id: patient_split[record.patient_id] for record in records}
        if len(assignments) != len(records):
            raise ManifestError("volume IDs must be unique")
        return assignments

    def _rebalance_to_global_target(
        self,
        allocations: dict[str, dict[str, int]],
        strata: Mapping[str, Sequence[str]],
    ) -> None:
        target = self._allocate_counts(sum(len(patients) for patients in strata.values()))
        totals = {
            split_name: sum(counts[split_name] for counts in allocations.values())
            for split_name in SPLITS
        }
        while totals != target:
            donors = [name for name in SPLITS if totals[name] > target[name]]
            receivers = [name for name in SPLITS if totals[name] < target[name]]
            candidates: list[tuple[float, str, str, str]] = []
            for label, counts in allocations.items():
                size = len(strata[label])
                enough_for_every_split = size >= sum(self.ratios[name] > 0 for name in SPLITS)
                desired = {name: size * self.ratios[name] for name in SPLITS}
                for donor in donors:
                    donor_minimum = 1 if enough_for_every_split and self.ratios[donor] > 0 else 0
                    if counts[donor] <= donor_minimum:
                        continue
                    for receiver in receivers:
                        before = (counts[donor] - desired[donor]) ** 2 + (counts[receiver] - desired[receiver]) ** 2
                        after = (counts[donor] - 1 - desired[donor]) ** 2 + (
                            counts[receiver] + 1 - desired[receiver]
                        ) ** 2
                        candidates.append((after - before, label, donor, receiver))
            if not candidates:
                # Very small strata can make exact global counts incompatible with
                # keeping every represented acquisition in every split.
                return
            _, label, donor, receiver = min(candidates)
            allocations[label][donor] -= 1
            allocations[label][receiver] += 1
            totals[donor] -= 1
            totals[receiver] += 1

    def _stable_key(self, stratum: str, patient_id: str) -> bytes:
        return hashlib.sha256(f"{self.seed}:{stratum}:{patient_id}".encode("utf-8")).digest()

    def _allocate_counts(self, total: int) -> dict[str, int]:
        desired = {name: total * self.ratios[name] for name in SPLITS}
        counts = {name: math.floor(desired[name]) for name in SPLITS}
        remaining = total - sum(counts.values())
        order = sorted(SPLITS, key=lambda name: (-(desired[name] - counts[name]), SPLITS.index(name)))
        for name in order[:remaining]:
            counts[name] += 1

        positive = [name for name in SPLITS if self.ratios[name] > 0]
        if total >= len(positive):
            for empty in (name for name in positive if counts[name] == 0):
                donor = max(positive, key=lambda name: counts[name])
                counts[donor] -= 1
                counts[empty] += 1
        return counts


class ManifestWriter:
    """Write complete and per-split JSONL manifests plus an auditable summary."""

    def write(
        self,
        output_directory: Path,
        records: Sequence[VolumeRecord],
        assignments: Mapping[str, str],
        warnings: Sequence[str],
        *,
        source_root: Path,
        seed: int,
        ratios: Mapping[str, float],
        value_check: str,
    ) -> dict[str, Any]:
        output_directory.mkdir(parents=True, exist_ok=True)
        ordered = sorted(records, key=lambda record: record.volume_id)
        self._write_jsonl(
            output_directory / "manifest.jsonl",
            (record.to_dict(assignments[record.volume_id]) for record in ordered),
        )
        for split_name in SPLITS:
            self._write_jsonl(
                output_directory / f"{split_name}.jsonl",
                (
                    record.to_dict(split_name)
                    for record in ordered
                    if assignments[record.volume_id] == split_name
                ),
            )

        summary = self._summary(ordered, assignments)
        summary.update(
            {
                "source_root": str(source_root.resolve()),
                "seed": seed,
                "ratios": dict(ratios),
                "value_check": value_check,
                "warnings": list(warnings),
            }
        )
        self._atomic_text(output_directory / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    @staticmethod
    def _summary(records: Sequence[VolumeRecord], assignments: Mapping[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_volumes": len(records),
            "total_slices": sum(record.slices for record in records),
            "unique_patients": len({record.patient_id for record in records}),
            "volumes_with_sensitivity_maps": sum(record.sensitivity_path is not None for record in records),
            "splits": {},
        }
        for split_name in SPLITS:
            subset = [record for record in records if assignments[record.volume_id] == split_name]
            result["splits"][split_name] = {
                "volumes": len(subset),
                "slices": sum(record.slices for record in subset),
                "patients": len({record.patient_id for record in subset}),
                "with_sensitivity_maps": sum(record.sensitivity_path is not None for record in subset),
                "acquisitions": dict(sorted(Counter(record.acquisition for record in subset).items())),
                "coil_counts": {
                    str(key): value for key, value in sorted(Counter(record.coils for record in subset).items())
                },
            }
        return result

    @classmethod
    def _write_jsonl(cls, path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        cls._atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
