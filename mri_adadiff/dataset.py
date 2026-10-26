"""Target-only training view for AdaDiff's unconditional image prior."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset

from mri_data import MRISliceDataset
from mri_reconstruction import channels_to_complex, complex_to_channels, magnitude_scale


class AdaDiffPriorDataset(Dataset[dict[str, Any]]):
    """Expose fully sampled images without masks or measurements to the prior.

    The shared loader initially applies its leakage-safe input normalization.
    Re-normalizing by the target's own magnitude quantile algebraically removes
    that input scale, leaving a target-only unconditional training image.
    """

    def __init__(self, source: MRISliceDataset, *, quantile: float = 0.99) -> None:
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        self.source = source
        self.quantile = quantile

    def __len__(self) -> int:
        return len(self.source)

    def batch_group_key(self, index: int) -> tuple[int, int]:
        """Return the target-image shape that must match within a batch."""

        _, reconstruction_shape = self.source.batch_group_key(index)
        return reconstruction_shape

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        target = channels_to_complex(sample["target"], channel_dim=0)
        target = target / magnitude_scale(target, self.quantile)
        return {
            "image": complex_to_channels(target, channel_dim=0),
            "metadata": sample["metadata"],
        }


def prior_collate(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty AdaDiff batch")
    shapes = {tuple(sample["image"].shape) for sample in samples}
    if len(shapes) != 1:
        raise ValueError("AdaDiff prior images in one batch must share shape")
    metadata = {
        key: [sample["metadata"][key] for sample in samples]
        for key in samples[0]["metadata"]
    }
    return {"image": torch.stack([sample["image"] for sample in samples]), "metadata": metadata}


def grouped_prior_batch_schedule(
    dataset: AdaDiffPriorDataset,
    *,
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    """Build deterministic full batches without mixing target spatial shapes."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(len(dataset)):
        groups.setdefault(dataset.batch_group_key(index), []).append(index)
    generator = torch.Generator().manual_seed(seed)
    batches: list[list[int]] = []
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[position] for position in order]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            if len(batch) < batch_size:
                missing = batch_size - len(batch)
                repeats = (
                    shuffled
                    * ((missing + len(shuffled) - 1) // len(shuffled))
                )[:missing]
                batch.extend(repeats)
            batches.append(batch)
    batch_order = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[position] for position in batch_order]


