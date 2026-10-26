"""Training and checkpoint support for the MoDL baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .model import MoDL


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def sample_id(batch: Mapping[str, Any]) -> str:
    metadata = batch.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return "unknown"
    value = metadata.get("sample_id", "unknown")
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else "unknown"
    return str(value)


class MoDLTrainer:
    """Single-GPU FP32 trainer matching the official MoDL optimization setup."""

    def __init__(
        self,
        model: MoDL,
        *,
        learning_rate: float = 1e-3,
        device: str | torch.device | None = None,
        gradient_clip_value: float = 1.0,
        data_config: Mapping[str, Any] | None = None,
    ) -> None:
        if learning_rate <= 0 or gradient_clip_value <= 0:
            raise ValueError("learning_rate and gradient_clip_value must be positive")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gradient_clip_value = gradient_clip_value
        self.data_config = dict(data_config or {})
        self.step = 0

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        batch = move_batch(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.model.training_losses(batch)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(
                f"nonfinite loss before optimizer step {self.step + 1} "
                f"for sample {sample_id(batch)}"
            )
        losses["loss"].backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"nonfinite gradient before optimizer step {self.step + 1} "
                f"for sample {sample_id(batch)}"
            )
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.gradient_clip_value)
        self.optimizer.step()
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
                "model_name": "modl",
                "step": self.step,
                "model_config": self.model.model_config,
                "data_config": self.data_config,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            destination,
        )
        return destination

    def load(self, path: str | Path, *, allow_unroll_change: bool = False) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("model_name") != "modl":
            raise ValueError("unsupported MoDL checkpoint format")
        saved_config = dict(checkpoint["model_config"])
        current_config = dict(self.model.model_config)
        if allow_unroll_change:
            saved_config.pop("unrolls", None)
            current_config.pop("unrolls", None)
        if saved_config != current_config:
            raise ValueError(
                f"checkpoint model configuration differs: saved={saved_config}, current={current_config}"
            )
        saved_data_config = checkpoint.get("data_config")
        if saved_data_config is not None:
            saved_data_config = dict(saved_data_config)
            if self.data_config and saved_data_config != self.data_config:
                raise ValueError(
                    "checkpoint data configuration differs: "
                    f"saved={saved_data_config}, current={self.data_config}"
                )
            self.data_config = saved_data_config
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = int(checkpoint["step"])
