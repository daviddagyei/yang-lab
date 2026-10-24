"""Training, augmentation, EMA, and checkpoint support for SwinMR-PI."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch

from .model import OFFICIAL_COMMIT, ComplexSwinMR
from .perceptual import VGG19PerceptualLoss


def _transform_spatial(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    if mode >= 4:
        tensor = torch.flip(tensor, dims=(-1,))
    return torch.rot90(tensor, k=mode % 4, dims=(-2, -1))


def random_crop_and_augment(
    batch: Mapping[str, Any],
    *,
    crop_size: int,
    generator: torch.Generator,
    augment: bool = True,
) -> dict[str, Any]:
    """Apply the official paired random crop and dihedral augmentation."""

    initial = batch["initial_reconstruction"]
    target = batch["target"]
    maps = batch["sensitivity_maps"]
    if initial.ndim != 4 or target.shape != initial.shape or maps.ndim != 4:
        raise ValueError("training images and maps must be batched 2D tensors")
    height, width = initial.shape[-2:]
    if maps.shape[0] != initial.shape[0] or maps.shape[-2:] != (height, width):
        raise ValueError("training sensitivity maps and images do not align")
    if crop_size <= 0 or crop_size > min(height, width):
        raise ValueError("crop size must fit the training image")
    initial_crops: list[torch.Tensor] = []
    target_crops: list[torch.Tensor] = []
    map_crops: list[torch.Tensor] = []
    for index in range(initial.shape[0]):
        top = int(torch.randint(height - crop_size + 1, (), generator=generator))
        left = int(torch.randint(width - crop_size + 1, (), generator=generator))
        mode = int(torch.randint(8, (), generator=generator)) if augment else 0
        spatial = (..., slice(top, top + crop_size), slice(left, left + crop_size))
        initial_crops.append(_transform_spatial(initial[index][spatial], mode))
        target_crops.append(_transform_spatial(target[index][spatial], mode))
        map_crops.append(_transform_spatial(maps[index][spatial], mode))
    return {
        "initial_reconstruction": torch.stack(initial_crops),
        "target": torch.stack(target_crops),
        "sensitivity_maps": torch.stack(map_crops),
        "metadata": batch.get("metadata", {}),
    }


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


class SwinMRTrainer:
    def __init__(
        self,
        model: ComplexSwinMR,
        *,
        learning_rate: float = 2e-4,
        ema_decay: float = 0.999,
        scheduler_milestones: tuple[int, ...] = (),
        scheduler_gamma: float = 0.5,
        device: str | torch.device | None = None,
        data_config: Mapping[str, Any] | None = None,
        load_perceptual: bool = True,
    ) -> None:
        if learning_rate <= 0 or not 0 <= ema_decay < 1:
            raise ValueError("invalid optimizer or EMA configuration")
        if scheduler_gamma <= 0 or any(step <= 0 for step in scheduler_milestones):
            raise ValueError("invalid learning-rate schedule")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.ema_model = copy.deepcopy(model).to(self.device).eval()
        self.ema_model.requires_grad_(False)
        self.perceptual = (
            VGG19PerceptualLoss().to(self.device).eval()
            if load_perceptual and model.perceptual_weight > 0
            else None
        )
        if model.perceptual_weight > 0 and load_perceptual is False:
            raise ValueError("perceptual loading can only be disabled when its loss weight is zero")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=list(scheduler_milestones),
            gamma=scheduler_gamma,
        )
        self.learning_rate = float(learning_rate)
        self.ema_decay = float(ema_decay)
        self.scheduler_milestones = tuple(int(step) for step in scheduler_milestones)
        self.scheduler_gamma = float(scheduler_gamma)
        self.data_config = dict(data_config or {})
        self.step = 0
        self.completed_epochs = 0

    @torch.no_grad()
    def _update_ema(self) -> None:
        online = dict(self.model.named_parameters())
        for name, parameter in self.ema_model.named_parameters():
            parameter.lerp_(online[name].detach(), 1.0 - self.ema_decay)
        online_buffers = dict(self.model.named_buffers())
        for name, buffer in self.ema_model.named_buffers():
            buffer.copy_(online_buffers[name])

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        moved = _move(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.model.training_losses(moved, perceptual_loss=self.perceptual)
        if not bool(torch.isfinite(losses["loss"])):
            raise FloatingPointError(f"nonfinite loss before step {self.step + 1}")
        losses["loss"].backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"nonfinite gradient before step {self.step + 1}")
        gradient_norm = torch.sqrt(
            sum(torch.sum(gradient.detach().float().square()) for gradient in gradients)
        )
        self.optimizer.step()
        self.step += 1
        self.scheduler.step()
        self._update_ema()
        return {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if key != "prediction"
        } | {
            "gradient_norm": float(gradient_norm.cpu()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_name": "adapted_complex_swinmr_pi",
                "official_commit": OFFICIAL_COMMIT,
                "step": self.step,
                "completed_epochs": self.completed_epochs,
                "learning_rate": self.learning_rate,
                "ema_decay": self.ema_decay,
                "scheduler_milestones": self.scheduler_milestones,
                "scheduler_gamma": self.scheduler_gamma,
                "model_config": self.model.model_config,
                "data_config": self.data_config,
                "model": self.model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            destination,
        )
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("model_name") != "adapted_complex_swinmr_pi":
            raise ValueError("unsupported SwinMR checkpoint")
        if checkpoint.get("official_commit") != OFFICIAL_COMMIT:
            raise ValueError("checkpoint references a different official SwinMR revision")
        checks = (
            (dict(checkpoint["model_config"]), dict(self.model.model_config), "model"),
            (dict(checkpoint.get("data_config", {})), self.data_config, "data"),
            (tuple(checkpoint.get("scheduler_milestones", ())), self.scheduler_milestones, "schedule"),
        )
        for actual, expected, name in checks:
            if actual != expected:
                raise ValueError(f"checkpoint {name} configuration differs")
        if float(checkpoint["learning_rate"]) != self.learning_rate:
            raise ValueError("checkpoint learning rate differs")
        if float(checkpoint["ema_decay"]) != self.ema_decay:
            raise ValueError("checkpoint EMA decay differs")
        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = int(checkpoint["step"])
        self.completed_epochs = int(checkpoint.get("completed_epochs", 0))
