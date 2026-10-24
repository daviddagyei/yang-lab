"""Training and checkpoint support for the full diffusion-unrolled RDBM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .model import DiffusionUnrolledRDBM


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
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


class DiffusionUnrolledRDBMTrainer:
    """Single-GPU end-to-end trainer with no pretrained model state."""

    def __init__(
        self,
        model: DiffusionUnrolledRDBM,
        *,
        learning_rate: float = 2e-5,
        device: str | torch.device | None = None,
        max_grad_norm: float = 1.0,
        precision: str = "bf16",
        data_config: Mapping[str, Any] | None = None,
    ) -> None:
        if learning_rate <= 0 or max_grad_norm <= 0:
            raise ValueError("learning rate and gradient norm must be positive")
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if precision != "fp32" and self.device.type != "cuda":
            raise RuntimeError("mixed precision requires CUDA")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support BF16")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.precision = precision
        self.data_config = dict(data_config or {})
        self.step = 0
        self.completed_epochs = 0

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        moved = _move_batch(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        with torch.autocast(
            device_type=self.device.type,
            dtype=autocast_dtype,
            enabled=self.precision != "fp32",
        ):
            losses = self.model.training_losses(moved)
        if not bool(torch.isfinite(losses["loss"])):
            raise FloatingPointError(
                f"nonfinite loss before step {self.step + 1} for {_sample_id(moved)}"
            )
        self.scaler.scale(losses["loss"]).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm, error_if_nonfinite=False
        )
        if not bool(torch.isfinite(gradient_norm)):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"nonfinite gradient before step {self.step + 1} for {_sample_id(moved)}"
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        return {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if key != "prediction"
        } | {"gradient_norm": float(gradient_norm.detach().cpu())}

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_name": "full_diffusion_unrolled_rdbm",
                "initialization": "scratch",
                "step": self.step,
                "completed_epochs": self.completed_epochs,
                "learning_rate": self.learning_rate,
                "max_grad_norm": self.max_grad_norm,
                "precision": self.precision,
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
        if (
            checkpoint.get("format_version") != 1
            or checkpoint.get("model_name") != "full_diffusion_unrolled_rdbm"
            or checkpoint.get("initialization") != "scratch"
        ):
            raise ValueError("unsupported full diffusion-unrolled checkpoint")
        saved_model_config = dict(checkpoint["model_config"])
        current_model_config = dict(self.model.model_config)
        # Activation checkpointing only trades memory for recomputation and has
        # no parameters, so it may be changed safely when resuming.
        saved_model_config.pop("gradient_checkpointing", None)
        current_model_config.pop("gradient_checkpointing", None)
        if saved_model_config != current_model_config:
            raise ValueError("checkpoint model configuration differs")
        if self.data_config and dict(checkpoint.get("data_config", {})) != self.data_config:
            raise ValueError("checkpoint data configuration differs")
        if float(checkpoint["learning_rate"]) != self.learning_rate:
            raise ValueError("checkpoint learning rate differs")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler.is_enabled() and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = int(checkpoint["step"])
        self.completed_epochs = int(checkpoint.get("completed_epochs", 0))
