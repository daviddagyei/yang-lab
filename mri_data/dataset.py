"""Lazy slice-level dataset for multi-coil MRI acceleration experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from mri_reconstruction import (
    CartesianMaskConfig,
    apply_cartesian_mask,
    complex_to_channels,
    magnitude_scale,
    make_cartesian_mask_for_record,
    normalize_complex,
    sense_combine,
    zero_filled_coil_images,
)
from mri_sensitivity import (
    load_sensitivity_slice,
    reconstruct_sense,
    validate_sensitivities,
)


def _relocate_data_paths(
    record: Mapping[str, Any], data_root: str | Path | None
) -> dict[str, Any]:
    """Relocate manifest data paths while preserving their extracted subtree."""

    relocated = dict(record)
    if data_root is None:
        return relocated
    root = Path(data_root).expanduser().resolve()
    for key in ("path", "sensitivity_path"):
        value = relocated.get(key)
        if not value:
            continue
        source = Path(str(value))
        try:
            extracted_index = source.parts.index("extracted")
        except ValueError as error:
            raise ValueError(
                f"cannot relocate manifest {key} without an extracted/ component: {source}"
            ) from error
        relocated[key] = str(root.joinpath(*source.parts[extracted_index:]))
    return relocated


def load_manifest(
    path: str | Path,
    *,
    expected_split: str | None = None,
    data_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Read a JSONL manifest and optionally enforce its frozen split."""

    manifest_path = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on {manifest_path}:{line_number}") from error
        if expected_split is not None and record.get("split") != expected_split:
            raise ValueError(
                f"{record.get('volume_id', '<unknown>')} belongs to split "
                f"{record.get('split')!r}, expected {expected_split!r}"
            )
        records.append(_relocate_data_paths(record, data_root))
    if not records:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return records


class MRISliceDataset(Dataset[dict[str, Any]]):
    """Load and transform one multi-coil MRI slice at a time.

    Only the 148 volumes with supplied sensitivity maps are indexed. Every
    normalization scale is computed from the zero-filled model input.
    """

    def __init__(
        self,
        manifest: str | Path | Sequence[Mapping[str, Any]],
        *,
        split: str | None = None,
        normalization_quantile: float = 0.99,
        mask_seed: int = 20261019,
        acceleration: int = 4,
        center_fraction: float | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        if not 0.0 < normalization_quantile <= 1.0:
            raise ValueError("normalization_quantile must be in (0, 1]")
        if isinstance(manifest, (str, Path)):
            records = load_manifest(
                manifest,
                expected_split=split,
                data_root=data_root or os.environ.get("MRI_DATA_ROOT"),
            )
        else:
            root_override = data_root or os.environ.get("MRI_DATA_ROOT")
            records = [_relocate_data_paths(record, root_override) for record in manifest]
            if split is not None:
                wrong = [record.get("volume_id", "<unknown>") for record in records if record.get("split") != split]
                if wrong:
                    raise ValueError(f"records outside requested split {split!r}: {wrong[:3]}")
        self.records = records
        self.normalization_quantile = normalization_quantile
        self.mask_config = CartesianMaskConfig(
            acceleration=acceleration,
            center_fraction=center_fraction,
            seed=mask_seed,
        )

        self._eligible_records: set[int] = set()
        self._examples: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            supplied = record.get("sensitivity_path")
            if not supplied or not Path(str(supplied)).is_file():
                continue
            self._eligible_records.add(record_index)
            slices = int(record["kspace_shape"][0])
            self._examples.extend((record_index, slice_index) for slice_index in range(slices))
        if not self._examples:
            raise ValueError("no volumes have supplied sensitivity maps")

    @property
    def volume_count(self) -> int:
        return len(self._eligible_records)

    def __len__(self) -> int:
        return len(self._examples)

    def batch_group_key(
        self, index: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return spatial shapes that must match within one padded batch."""

        record_index, _ = self._examples[index]
        record = self.records[record_index]
        return (
            tuple(int(value) for value in record["kspace_shape"][-2:]),
            tuple(
                int(value)
                for value in record["reconstruction_shape"][-2:]
            ),
        )

    def _load_maps(self, record: Mapping[str, Any], slice_index: int) -> torch.Tensor:
        maps = load_sensitivity_slice(record, slice_index).to(torch.complex64)
        validate_sensitivities(
            maps,
            expected_coils=int(record["kspace_shape"][1]),
            expected_shape=tuple(int(value) for value in record["reconstruction_shape"][-2:]),
        )
        return maps

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, slice_index = self._examples[index]
        record = self.records[record_index]
        output_shape = tuple(int(value) for value in record["reconstruction_shape"][-2:])

        with h5py.File(Path(str(record["path"])), "r") as source:
            kspace = torch.from_numpy(source["kspace"][slice_index]).to(torch.complex64)
            rss_reference = torch.from_numpy(source["reconstruction_rss"][slice_index]).float()

        maps = self._load_maps(record, slice_index)
        sampling = make_cartesian_mask_for_record(record, self.mask_config)
        masked_kspace = apply_cartesian_mask(kspace, sampling.mask)
        initial_coils = zero_filled_coil_images(kspace, sampling.mask, output_shape)
        initial = sense_combine(initial_coils, maps)
        target = reconstruct_sense(kspace, maps, output_shape)

        # This scale uses only the simulated acquired measurements through the
        # zero-filled input. Neither the complex target nor RSS reference affects it.
        scale = magnitude_scale(initial, self.normalization_quantile)
        masked_kspace = normalize_complex(masked_kspace, scale)
        initial = normalize_complex(initial, scale)
        target = normalize_complex(target, scale)
        rss_reference = rss_reference / scale

        return {
            "undersampled_kspace": masked_kspace,
            "mask": sampling.mask,
            "sensitivity_maps": maps,
            "initial_reconstruction": complex_to_channels(initial, channel_dim=0),
            "target": complex_to_channels(target, channel_dim=0),
            "rss_reference": rss_reference,
            "scale": scale,
            "metadata": {
                "sample_id": f"{record['volume_id']}:{slice_index}",
                "volume_id": str(record["volume_id"]),
                "slice_index": slice_index,
                "split": str(record.get("split", "")),
                "acquisition": str(record.get("acquisition", "")),
                "coils": int(kspace.shape[0]),
                "nominal_acceleration": self.mask_config.acceleration,
                "center_fraction": self.mask_config.resolved_center_fraction,
                "effective_acceleration": sampling.effective_acceleration,
                "sensitivity_source": "supplied",
            },
        }


def build_dataloader(
    dataset: MRISliceDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Any]]:
    """Create a loader while variable coil and spatial sizes require batch one."""

    if batch_size != 1:
        raise ValueError("batch_size must be 1 until variable-size padded collation is added")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def collate_padded_coils(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Batch equal-size MRI slices while zero-padding their coil dimension."""

    if not samples:
        raise ValueError("cannot collate an empty MRI batch")
    kspace_shapes = {
        tuple(sample["undersampled_kspace"].shape[-2:])
        for sample in samples
    }
    map_shapes = {
        tuple(sample["sensitivity_maps"].shape[-2:])
        for sample in samples
    }
    if len(kspace_shapes) != 1 or len(map_shapes) != 1:
        raise ValueError(
            "spatially different MRI samples must be placed in separate batches"
        )
    max_coils = max(
        int(sample["undersampled_kspace"].shape[0])
        for sample in samples
    )
    padded: list[dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        for key in ("undersampled_kspace", "sensitivity_maps"):
            value = sample[key]
            missing = max_coils - int(value.shape[0])
            if missing:
                zeros = torch.zeros(
                    (missing, *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                value = torch.cat((value, zeros), dim=0)
            item[key] = value
        padded.append(item)
    return default_collate(padded)
