"""Target-free slice dataset for 4x self-supervised MRI reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import torch

from mri_data import MRISliceDataset
from mri_reconstruction import (
    apply_cartesian_mask,
    complex_to_channels,
    magnitude_scale,
    make_cartesian_mask_for_record,
    normalize_complex,
    sense_combine,
    zero_filled_coil_images,
)

from .masking import SelfSupervisedMaskConfig, split_acquired_mask


class SelfSupervisedMRISliceDataset(MRISliceDataset):
    """Return only acquired measurements needed for self-supervised training.

    The fixed 4x mask ``Omega`` is partitioned into model-input lines ``Theta``
    and held-out loss lines ``Lambda``. No fully sampled image or RSS reference
    is loaded or returned. The normalization scale is computed from ``Theta``.
    """

    def __init__(
        self,
        manifest: str | Path | Sequence[Mapping[str, Any]],
        *,
        split: str | None = None,
        normalization_quantile: float = 0.99,
        mask_seed: int = 20261019,
        partition_seed: int = 20261020,
        input_fraction: float = 0.5,
        dynamic_partitions: bool | None = None,
    ) -> None:
        super().__init__(
            manifest,
            split=split,
            normalization_quantile=normalization_quantile,
            mask_seed=mask_seed,
        )
        record_splits = {str(record.get("split", "")) for record in self.records}
        inferred_split = split or (next(iter(record_splits)) if len(record_splits) == 1 else "")
        self.split = inferred_split
        self.dynamic_partitions = inferred_split == "train" if dynamic_partitions is None else dynamic_partitions
        if self.dynamic_partitions and inferred_split not in {"", "train"}:
            raise ValueError("dynamic partitions are allowed only for the training split")
        self.partition_config = SelfSupervisedMaskConfig(
            input_fraction=input_fraction,
            seed=partition_seed,
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the train partition for an epoch; validation/test stay fixed."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, slice_index = self._examples[index]
        record = self.records[record_index]
        output_shape = tuple(int(value) for value in record["reconstruction_shape"][-2:])
        sample_id = f"{record['volume_id']}:{slice_index}"

        # Retrospective experiments must read the source k-space to simulate the
        # acquisition, but no fully sampled image-domain target is constructed.
        with h5py.File(Path(str(record["path"])), "r") as source:
            kspace = torch.from_numpy(source["kspace"][slice_index]).to(torch.complex64)

        maps = self._load_maps(record, slice_index)
        sampling = make_cartesian_mask_for_record(record, self.mask_config)
        partition_epoch = self.epoch if self.dynamic_partitions else 0
        partition = split_acquired_mask(
            sampling.mask,
            sample_id,
            self.partition_config,
            center_lines=sampling.center_lines,
            epoch=partition_epoch,
        )

        input_kspace = apply_cartesian_mask(kspace, partition.input_mask)
        loss_kspace = apply_cartesian_mask(kspace, partition.loss_mask)
        initial_coils = zero_filled_coil_images(kspace, partition.input_mask, output_shape)
        initial = sense_combine(initial_coils, maps)

        # The scale is input-derived: held-out values cannot affect model input.
        scale = magnitude_scale(initial, self.normalization_quantile)
        input_kspace = normalize_complex(input_kspace, scale)
        loss_kspace = normalize_complex(loss_kspace, scale)
        initial = normalize_complex(initial, scale)

        return {
            "input_kspace": input_kspace,
            "input_mask": partition.input_mask,
            "loss_kspace": loss_kspace,
            "loss_mask": partition.loss_mask,
            "acquired_mask": partition.acquired_mask,
            "acs_mask": partition.acs_mask,
            "sensitivity_maps": maps,
            "initial_reconstruction": complex_to_channels(initial, channel_dim=0),
            "scale": scale,
            "metadata": {
                "sample_id": sample_id,
                "volume_id": str(record["volume_id"]),
                "slice_index": slice_index,
                "split": str(record.get("split", "")),
                "acquisition": str(record.get("acquisition", "")),
                "coils": int(kspace.shape[0]),
                "nominal_acceleration": 4,
                "acquired_lines": partition.acquired_lines,
                "input_lines": partition.input_lines,
                "loss_lines": partition.loss_lines,
                "acs_lines": partition.acs_lines,
                "acquired_effective_acceleration": sampling.effective_acceleration,
                "input_effective_acceleration": sampling.acquired_lines / partition.input_lines,
                "partition_epoch": partition_epoch,
                "partition_is_dynamic": self.dynamic_partitions,
                "sensitivity_source": "supplied",
            },
        }
