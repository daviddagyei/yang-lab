"""Trainer and checkpoints for target-free diffusion reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .model import SelfSupervisedDiffusionMRI
from .adversarial import (
    ConditionalDiffusionDiscriminator,
    discriminator_objective,
    generator_adversarial_loss,
)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _sample_id(batch: Mapping[str, Any]) -> str:
    metadata = batch.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return "unknown"
    value = metadata.get("sample_id", "unknown")
    return str(value[0] if isinstance(value, (list, tuple)) and value else value)


class ASSCGDTrainer:
    """Single-device trainer with finite-value and gradient checks."""

    def __init__(
        self,
        model: SelfSupervisedDiffusionMRI,
        *,
        learning_rate: float = 2e-4,
        device: str | torch.device | None = None,
        max_grad_norm: float = 1.0,
        precision: str = "fp32",
    ) -> None:
        if learning_rate <= 0 or max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be one of: fp32, bf16, fp16")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if precision != "fp32" and self.device.type != "cuda":
            raise RuntimeError("mixed precision requires CUDA")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA GPU does not support BF16")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        self.precision = precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        self.step = 0
        self.completed_epochs = 0

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        batch = move_batch(batch, self.device)
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
                f"nonfinite loss before step {self.step + 1} for {_sample_id(batch)}"
            )
        self.scaler.scale(losses["loss"]).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm, error_if_nonfinite=False
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"nonfinite gradient before step {self.step + 1} for {_sample_id(batch)}"
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
                "model_name": "asscgd_inspired_diffusion",
                "step": self.step,
                "completed_epochs": self.completed_epochs,
                "model_config": self.model.model_config,
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
            or checkpoint.get("model_name") != "asscgd_inspired_diffusion"
        ):
            raise ValueError("unsupported ASSCGD-inspired checkpoint format")
        if dict(checkpoint["model_config"]) != dict(self.model.model_config):
            raise ValueError("checkpoint model configuration differs")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler.is_enabled() and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = int(checkpoint["step"])
        self.completed_epochs = int(checkpoint.get("completed_epochs", 0))


class AdversarialASSCGDTrainer:
    """Alternate discriminator and target-free diffusion generator updates."""

    def __init__(
        self,
        model: SelfSupervisedDiffusionMRI,
        discriminator: ConditionalDiffusionDiscriminator,
        *,
        learning_rate: float = 2e-4,
        discriminator_learning_rate: float | None = None,
        adversarial_weight: float = 0.1,
        gradient_penalty_weight: float = 1.0,
        discriminator_update_interval: int = 4,
        real_label_smoothing: float = 0.1,
        adaptive_adversarial_weight: bool = True,
        device: str | torch.device | None = None,
        max_grad_norm: float = 1.0,
        precision: str = "fp32",
    ) -> None:
        discriminator_learning_rate = (
            learning_rate * 0.1
            if discriminator_learning_rate is None
            else discriminator_learning_rate
        )
        if (
            learning_rate <= 0
            or discriminator_learning_rate <= 0
            or adversarial_weight < 0
            or gradient_penalty_weight < 0
        ):
            raise ValueError("invalid adversarial trainer hyperparameters")
        if discriminator_update_interval <= 0:
            raise ValueError("discriminator_update_interval must be positive")
        if not 0.0 <= real_label_smoothing < 0.5:
            raise ValueError("real_label_smoothing must be in [0, 0.5)")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if precision not in {"fp32", "bf16"}:
            raise ValueError("adversarial precision must be fp32 or bf16")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if precision == "bf16" and self.device.type != "cuda":
            raise RuntimeError("BF16 adversarial training requires CUDA")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA GPU does not support BF16")
        self.model = model.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.generator_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_learning_rate
        )
        self.generator_learning_rate = learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.adversarial_weight = adversarial_weight
        self.gradient_penalty_weight = gradient_penalty_weight
        self.discriminator_update_interval = discriminator_update_interval
        self.real_label_smoothing = real_label_smoothing
        self.adaptive_adversarial_weight = adaptive_adversarial_weight
        self.max_grad_norm = max_grad_norm
        self.precision = precision
        self.step = 0
        self.completed_epochs = 0

    @staticmethod
    def _finite_gradients(module: torch.nn.Module) -> bool:
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        return bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        self.discriminator.train()
        batch = move_batch(batch, self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.precision == "bf16",
        ):
            states = self.model.adversarial_states(batch)

        update_discriminator = self.step % self.discriminator_update_interval == 0
        if update_discriminator:
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_losses = discriminator_objective(
                self.discriminator,
                states["real_state"],
                states["fake_state"],
                states["condition"],
                gradient_penalty_weight=self.gradient_penalty_weight,
                real_label_smoothing=self.real_label_smoothing,
            )
            if not torch.isfinite(discriminator_losses["loss"]):
                raise FloatingPointError(f"nonfinite discriminator loss at step {self.step + 1}")
            discriminator_losses["loss"].backward()
            if not self._finite_gradients(self.discriminator):
                raise FloatingPointError(
                    f"nonfinite discriminator gradient at step {self.step + 1}"
                )
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()
        else:
            with torch.no_grad():
                discriminator_losses = discriminator_objective(
                    self.discriminator,
                    states["real_state"],
                    states["fake_state"],
                    states["condition"],
                    gradient_penalty_weight=0.0,
                    real_label_smoothing=self.real_label_smoothing,
                )

        self.generator_optimizer.zero_grad(set_to_none=True)
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)
        self.discriminator.eval()
        adversarial = generator_adversarial_loss(
            self.discriminator,
            states["fake_state"],
            states["condition"],
        )
        effective_adversarial_weight = torch.as_tensor(
            self.adversarial_weight,
            device=adversarial.device,
            dtype=adversarial.dtype,
        )
        if self.adaptive_adversarial_weight:
            balance = states["heldout_kspace"].detach() / adversarial.detach().clamp_min(1e-8)
            effective_adversarial_weight = effective_adversarial_weight * balance.clamp(max=1.0)
        generator_loss = (
            states["heldout_kspace"] + effective_adversarial_weight * adversarial
        )
        if not torch.isfinite(generator_loss):
            raise FloatingPointError(
                f"nonfinite generator loss at step {self.step + 1} for {_sample_id(batch)}"
            )
        generator_loss.backward()
        self.discriminator.train()
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(True)
        if not self._finite_gradients(self.model):
            raise FloatingPointError(f"nonfinite generator gradient at step {self.step + 1}")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.generator_optimizer.step()
        self.step += 1

        return {
            "loss": float(generator_loss.detach().cpu()),
            "heldout_kspace": float(states["heldout_kspace"].detach().cpu()),
            "generator_adversarial": float(adversarial.detach().cpu()),
            "effective_adversarial_weight": float(
                effective_adversarial_weight.detach().cpu()
            ),
            "discriminator": float(discriminator_losses["loss"].detach().cpu()),
            "discriminator_updated": float(update_discriminator),
            "gradient_penalty": float(
                discriminator_losses["gradient_penalty"].detach().cpu()
            ),
            "real_logit": float(discriminator_losses["real_logit"].detach().cpu()),
            "fake_logit": float(discriminator_losses["fake_logit"].detach().cpu()),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_name": "asscgd_inspired_adversarial_diffusion",
                "step": self.step,
                "completed_epochs": self.completed_epochs,
                "model_config": self.model.model_config,
                "discriminator_config": {
                    "base_channels": self.discriminator.base_channels,
                    "use_spectral_norm": self.discriminator.use_spectral_norm,
                },
                "generator_learning_rate": self.generator_learning_rate,
                "discriminator_learning_rate": self.discriminator_learning_rate,
                "adversarial_weight": self.adversarial_weight,
                "gradient_penalty_weight": self.gradient_penalty_weight,
                "discriminator_update_interval": self.discriminator_update_interval,
                "real_label_smoothing": self.real_label_smoothing,
                "adaptive_adversarial_weight": self.adaptive_adversarial_weight,
                "precision": self.precision,
                "model": self.model.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "generator_optimizer": self.generator_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            },
            destination,
        )
        return destination

    def load(self, path: str | Path, *, reset_discriminator: bool = False) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if (
            checkpoint.get("format_version") != 1
            or checkpoint.get("model_name") != "asscgd_inspired_adversarial_diffusion"
        ):
            raise ValueError("unsupported adversarial diffusion checkpoint format")
        if dict(checkpoint["model_config"]) != dict(self.model.model_config):
            raise ValueError("checkpoint generator configuration differs")
        self.model.load_state_dict(checkpoint["model"])
        self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        if not reset_discriminator:
            expected_discriminator = {
                "base_channels": self.discriminator.base_channels,
                "use_spectral_norm": self.discriminator.use_spectral_norm,
            }
            if dict(checkpoint["discriminator_config"]) != expected_discriminator:
                raise ValueError("checkpoint discriminator configuration differs")
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.discriminator_optimizer.load_state_dict(
                checkpoint["discriminator_optimizer"]
            )
        self.step = int(checkpoint["step"])
        self.completed_epochs = int(checkpoint.get("completed_epochs", 0))
