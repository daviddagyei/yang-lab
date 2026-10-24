"""Minimal trainer and checkpoint support for the MRI RDBM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .model import PhysicsGuidedRDBM


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _sample_id(batch: Mapping[str, Any]) -> str:
    metadata = batch.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return "unknown"
    value = metadata.get("sample_id", "unknown")
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else "unknown"
    return str(value)


class MRITrainer:
    """Single-device trainer used for smoke tests and staged training."""

    def __init__(
        self,
        model: PhysicsGuidedRDBM,
        *,
        learning_rate: float = 8e-5,
        device: str | torch.device | None = None,
        max_grad_norm: float = 1.0,
        precision: str = "fp32",
        data_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be one of: fp32, bf16, fp16")
        if precision != "fp32" and self.device.type != "cuda":
            raise RuntimeError("mixed-precision training requires an available CUDA GPU")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA GPU does not support BF16; use --precision fp32")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        self.precision = precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        self.data_config = dict(data_config or {})
        self.step = 0

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        batch = _move_batch(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        with torch.autocast(
            device_type=self.device.type,
            dtype=autocast_dtype,
            enabled=self.precision != "fp32",
        ):
            losses = self.model.training_losses(batch)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(
                f"nonfinite loss before optimizer step {self.step + 1} "
                f"for sample {_sample_id(batch)}"
            )
        self.scaler.scale(losses["loss"]).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm, error_if_nonfinite=False
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"nonfinite gradient before optimizer step {self.step + 1} "
                f"for sample {_sample_id(batch)}"
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        return {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if key != "prediction"
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "step": self.step,
                "model_config": self.model.model_config,
                "data_config": self.data_config,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            destination,
        )
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported MRI checkpoint format")
        checkpoint_data_config = dict(checkpoint.get("data_config", {}))
        if self.data_config:
            if not checkpoint_data_config:
                if int(self.data_config.get("acceleration", 4)) != 4:
                    raise ValueError(
                        "legacy checkpoints are assumed to use 4x acceleration"
                    )
            elif checkpoint_data_config != self.data_config:
                raise ValueError(
                    "checkpoint data configuration differs from this training run"
                )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler.is_enabled() and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = int(checkpoint["step"])
