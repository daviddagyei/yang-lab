"""Deterministic batching for variable-coil MRI slices."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from .dataset import MRISliceDataset, collate_padded_coils


def batch_group_key(
    dataset: torch.utils.data.Dataset, index: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    if not isinstance(dataset, MRISliceDataset):
        raise TypeError("grouped MRI batching requires MRISliceDataset")
    return dataset.batch_group_key(index)


def grouped_batch_schedule(
    dataset: torch.utils.data.Dataset, *, batch_size: int, seed: int
) -> list[list[int]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    groups: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = {}
    for index in range(len(dataset)):
        groups.setdefault(batch_group_key(dataset, index), []).append(index)
    generator = torch.Generator().manual_seed(seed)
    batches: list[list[int]] = []
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[position] for position in order]
        for start in range(0, len(shuffled), batch_size):
            batches.append(shuffled[start : start + batch_size])
    order = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[position] for position in order]


def build_grouped_dataloader(
    dataset: torch.utils.data.Dataset,
    batches: list[list[int]],
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=collate_padded_coils,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
