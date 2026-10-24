"""Training and checkpoint support for the adapted D5-C5 baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .model import CascadeCNN


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class CascadeCNNTrainer:
    def __init__(
        self,
        model: CascadeCNN,
        *,
        learning_rate: float = 1e-3,
        device: str | torch.device | None = None,
        data_config: Mapping[str, Any] | None = None,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999)
        )
        self.learning_rate = float(learning_rate)
        self.data_config = dict(data_config or {})
        self.step = 0
        self.completed_epochs = 0

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        batch = move_batch(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.model.training_losses(batch)
        if not bool(torch.isfinite(losses["loss"])):
            raise FloatingPointError(f"nonfinite loss before step {self.step + 1}")
        losses["loss"].backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"nonfinite gradient before step {self.step + 1}")
        gradient_norm = torch.sqrt(
            sum(torch.sum(gradient.detach().square()) for gradient in gradients)
        )
        self.optimizer.step()
        self.step += 1
        return {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if key != "prediction"
        } | {"gradient_norm": float(gradient_norm.cpu())}

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_name": "multicoil_cascade_cnn",
                "step": self.step,
                "completed_epochs": self.completed_epochs,
                "learning_rate": self.learning_rate,
                "model_config": self.model.model_config,
                "data_config": self.data_config,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            destination,
        )
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("model_name") != "multicoil_cascade_cnn":
            raise ValueError("unsupported Cascade CNN checkpoint")
        if dict(checkpoint["model_config"]) != dict(self.model.model_config):
            raise ValueError("checkpoint model configuration differs")
        if dict(checkpoint.get("data_config", {})) != self.data_config:
            raise ValueError("checkpoint data configuration differs")
        if float(checkpoint.get("learning_rate", self.learning_rate)) != self.learning_rate:
            raise ValueError("checkpoint learning rate differs")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = int(checkpoint["step"])
        self.completed_epochs = int(checkpoint.get("completed_epochs", 0))
