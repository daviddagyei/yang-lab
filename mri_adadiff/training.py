"""Official-style adversarial prior training with EMA and resumable checkpoints."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .diffusion import DiffusionSchedule
from .model import (
    OFFICIAL_COMMIT,
    AdaDiffDiscriminator,
    AdaDiffGenerator,
)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


class AdaDiffTrainer:
    def __init__(
        self,
        generator: AdaDiffGenerator,
        discriminator: AdaDiffDiscriminator,
        *,
        schedule: DiffusionSchedule | None = None,
        generator_learning_rate: float = 1.6e-4,
        discriminator_learning_rate: float = 1e-4,
        beta1: float = 0.5,
        beta2: float = 0.9,
        ema_decay: float = 0.999,
        r1_gamma: float = 1.0,
        lazy_regularization: int = 10,
        total_epochs: int = 500,
        device: str | torch.device | None = None,
        data_config: Mapping[str, Any] | None = None,
    ) -> None:
        if generator_learning_rate <= 0 or discriminator_learning_rate <= 0:
            raise ValueError("AdaDiff learning rates must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or not beta1 < beta2:
            raise ValueError("invalid Adam beta values")
        if not 0 <= ema_decay < 1 or r1_gamma < 0 or lazy_regularization <= 0:
            raise ValueError("invalid AdaDiff EMA or R1 settings")
        if total_epochs <= 0:
            raise ValueError("total epochs must be positive")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.ema_generator = copy.deepcopy(generator).to(self.device).eval()
        self.ema_generator.requires_grad_(False)
        self.schedule = schedule or DiffusionSchedule()
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=generator_learning_rate,
            betas=(beta1, beta2),
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_learning_rate,
            betas=(beta1, beta2),
        )
        self.generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.generator_optimizer, T_max=total_epochs, eta_min=1e-5
        )
        self.discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.discriminator_optimizer, T_max=total_epochs, eta_min=1e-5
        )
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.ema_decay = ema_decay
        self.r1_gamma = r1_gamma
        self.lazy_regularization = lazy_regularization
        self.total_epochs = total_epochs
        self.data_config = dict(data_config or {})
        self.step = 0
        self.completed_epochs = 0
        self.ema_initialized = False

    @torch.no_grad()
    def _update_ema(self) -> None:
        online = dict(self.generator.named_parameters())
        for name, parameter in self.ema_generator.named_parameters():
            if self.ema_initialized:
                parameter.lerp_(online[name].detach(), 1.0 - self.ema_decay)
            else:
                parameter.copy_(online[name].detach())
        online_buffers = dict(self.generator.named_buffers())
        for name, buffer in self.ema_generator.named_buffers():
            buffer.copy_(online_buffers[name])
        self.ema_initialized = True

    @staticmethod
    def _finite_gradients(module: torch.nn.Module) -> bool:
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        return bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        image = _move(batch["image"], self.device).float()
        batch_size = image.shape[0]

        self.discriminator.requires_grad_(True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(
            self.schedule.timesteps, (batch_size,), device=self.device
        )
        real_state, condition = self.schedule.sample_pair(image, timesteps)
        real_state.requires_grad_(self.step % self.lazy_regularization == 0)
        real_logits = self.discriminator(real_state, timesteps, condition.detach())
        discriminator_real = F.softplus(-real_logits).mean()
        latent = torch.randn(
            batch_size, self.generator.latent_dimension, device=self.device
        )
        with torch.no_grad():
            predicted_clean = self.generator(condition.detach(), timesteps, latent)
            fake_state = self.schedule.sample_posterior(
                predicted_clean, condition, timesteps
            )
        fake_logits = self.discriminator(fake_state.detach(), timesteps, condition.detach())
        discriminator_fake = F.softplus(fake_logits).mean()
        r1 = torch.zeros((), device=self.device)
        if self.step % self.lazy_regularization == 0:
            gradient = torch.autograd.grad(
                real_logits.sum(), real_state, create_graph=True
            )[0]
            r1 = 0.5 * self.r1_gamma * gradient.flatten(1).square().sum(1).mean()
        discriminator_loss = discriminator_real + discriminator_fake + r1
        if not bool(torch.isfinite(discriminator_loss)):
            raise FloatingPointError(f"nonfinite AdaDiff discriminator loss at step {self.step + 1}")
        discriminator_loss.backward()
        if not self._finite_gradients(self.discriminator):
            raise FloatingPointError(
                f"nonfinite AdaDiff discriminator gradient at step {self.step + 1}"
            )
        self.discriminator_optimizer.step()

        self.discriminator.requires_grad_(False)
        self.generator_optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(
            self.schedule.timesteps, (batch_size,), device=self.device
        )
        _, condition = self.schedule.sample_pair(image, timesteps)
        latent = torch.randn(
            batch_size, self.generator.latent_dimension, device=self.device
        )
        predicted_clean = self.generator(condition.detach(), timesteps, latent)
        fake_state = self.schedule.sample_posterior(
            predicted_clean, condition, timesteps
        )
        generator_logits = self.discriminator(fake_state, timesteps, condition.detach())
        generator_loss = F.softplus(-generator_logits).mean()
        if not bool(torch.isfinite(generator_loss)):
            raise FloatingPointError(f"nonfinite AdaDiff generator loss at step {self.step + 1}")
        generator_loss.backward()
        if not self._finite_gradients(self.generator):
            raise FloatingPointError(
                f"nonfinite AdaDiff generator gradient at step {self.step + 1}"
            )
        self.generator_optimizer.step()
        self.step += 1
        self._update_ema()
        return {
            "generator": float(generator_loss.detach()),
            "discriminator": float(discriminator_loss.detach()),
            "discriminator_real": float(discriminator_real.detach()),
            "discriminator_fake": float(discriminator_fake.detach()),
            "r1": float(r1.detach()),
            "generator_lr": float(self.generator_optimizer.param_groups[0]["lr"]),
            "discriminator_lr": float(self.discriminator_optimizer.param_groups[0]["lr"]),
        }

    def complete_epoch(self) -> None:
        self.completed_epochs += 1
        self.generator_scheduler.step()
        self.discriminator_scheduler.step()

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_name": "adadiff_multicoil_adaptation",
                "official_commit": OFFICIAL_COMMIT,
                "step": self.step,
                "completed_epochs": self.completed_epochs,
                "ema_initialized": self.ema_initialized,
                "generator_config": self.generator.model_config,
                "discriminator_config": self.discriminator.model_config,
                "schedule_config": {
                    "timesteps": self.schedule.timesteps,
                    "beta_min": self.schedule.beta_min,
                    "beta_max": self.schedule.beta_max,
                },
                "training_config": {
                    "generator_learning_rate": self.generator_learning_rate,
                    "discriminator_learning_rate": self.discriminator_learning_rate,
                    "beta1": self.beta1,
                    "beta2": self.beta2,
                    "ema_decay": self.ema_decay,
                    "r1_gamma": self.r1_gamma,
                    "lazy_regularization": self.lazy_regularization,
                    "total_epochs": self.total_epochs,
                },
                "data_config": self.data_config,
                "generator": self.generator.state_dict(),
                "ema_generator": self.ema_generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "generator_optimizer": self.generator_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                "generator_scheduler": self.generator_scheduler.state_dict(),
                "discriminator_scheduler": self.discriminator_scheduler.state_dict(),
            },
            destination,
        )
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if (
            checkpoint.get("format_version") != 1
            or checkpoint.get("model_name") != "adadiff_multicoil_adaptation"
            or checkpoint.get("official_commit") != OFFICIAL_COMMIT
        ):
            raise ValueError("unsupported AdaDiff checkpoint")
        expected = {
            "generator_config": self.generator.model_config,
            "discriminator_config": self.discriminator.model_config,
            "schedule_config": {
                "timesteps": self.schedule.timesteps,
                "beta_min": self.schedule.beta_min,
                "beta_max": self.schedule.beta_max,
            },
            "training_config": {
                "generator_learning_rate": self.generator_learning_rate,
                "discriminator_learning_rate": self.discriminator_learning_rate,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "ema_decay": self.ema_decay,
                "r1_gamma": self.r1_gamma,
                "lazy_regularization": self.lazy_regularization,
                "total_epochs": self.total_epochs,
            },
            "data_config": self.data_config,
        }
        for key, value in expected.items():
            if dict(checkpoint.get(key, {})) != dict(value):
                raise ValueError(f"checkpoint {key} differs")
        self.generator.load_state_dict(checkpoint["generator"])
        self.ema_generator.load_state_dict(checkpoint["ema_generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        self.generator_scheduler.load_state_dict(checkpoint["generator_scheduler"])
        self.discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler"])
        self.step = int(checkpoint["step"])
        self.completed_epochs = int(checkpoint["completed_epochs"])
        self.ema_initialized = bool(checkpoint.get("ema_initialized", True))


