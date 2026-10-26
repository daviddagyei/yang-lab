"""Loader for the pinned official AdaDiff/StyleGAN fused CUDA FIR operator."""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch


OFFICIAL_COMMIT = "15f9e1a04c13678d7c3c60ca4a47cf7b12482b24"


@lru_cache(maxsize=1)
def _official_module() -> ModuleType:
    root = Path(
        os.environ.get(
            "ADADIFF_OFFICIAL_ROOT",
            Path(__file__).resolve().parents[1] / "third_party" / "AdaDiff",
        )
    )
    source = root / "utils" / "op" / "upfirdn2d.py"
    if not source.is_file():
        raise RuntimeError(
            f"official AdaDiff FIR source is missing at {source}; "
            "run scripts/setup_adadiff_official_backend.sh"
        )
    specification = importlib.util.spec_from_file_location(
        "adadiff_official_upfirdn2d", source
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load official AdaDiff FIR module from {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def official_upfirdn2d(
    value: torch.Tensor,
    kernel: torch.Tensor,
    *,
    up: int = 1,
    down: int = 1,
    pad: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Call the CUDA operator distributed by the official AdaDiff repository."""

    if not value.is_cuda:
        raise ValueError("official AdaDiff FIR backend is CUDA-only")
    return _official_module().upfirdn2d(
        value, kernel, up=int(up), down=int(down), pad=tuple(pad)
    )
