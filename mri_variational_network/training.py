"""Training and checkpoint support for the MRI Variational Network baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .model import VariationalNetwork


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


class IPALMOptimizer:
    """PyTorch port of the three-block iPALM optimizer in the official repo.

    The official TensorFlow implementation alternates between the complex
    convolution kernels, RBF weights, and data-term weights. The real and
    imaginary kernel tensors here form one joint complex-valued block.
    """

    BLOCK_NAMES = ("kernels", "activation_weights", "data_term_weights")

    def __init__(
        self,
        model: VariationalNetwork,
        *,
        initial_lipschitz: float = 1000.0,
        momentum: float = 0.7,
        max_backtracking: int = 40,
        minimum_lipschitz: float = 1e-3,
        maximum_lipschitz: float = 1e12,
        sufficient_decrease_tolerance: float = 1e-3,
    ) -> None:
        if initial_lipschitz <= 0:
            raise ValueError("initial Lipschitz estimate must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("iPALM momentum must be in [0, 1)")
        if max_backtracking <= 0:
            raise ValueError("max backtracking trials must be positive")
        if not 0 < minimum_lipschitz <= initial_lipschitz <= maximum_lipschitz:
            raise ValueError("invalid Lipschitz bounds")
        if sufficient_decrease_tolerance < 0:
            raise ValueError("sufficient-decrease tolerance cannot be negative")

        self.model = model
        self.initial_lipschitz = float(initial_lipschitz)
        self.momentum = float(momentum)
        self.max_backtracking = int(max_backtracking)
        self.minimum_lipschitz = float(minimum_lipschitz)
        self.maximum_lipschitz = float(maximum_lipschitz)
        self.sufficient_decrease_tolerance = float(sufficient_decrease_tolerance)
        self.lipschitz = [self.initial_lipschitz] * len(self.BLOCK_NAMES)
        self.previous: list[list[torch.Tensor]] = []
        self.iterations = 0
        self._reset_memory()

    @property
    def config(self) -> dict[str, float | int]:
        return {
            "initial_lipschitz": self.initial_lipschitz,
            "momentum": self.momentum,
            "max_backtracking": self.max_backtracking,
            "minimum_lipschitz": self.minimum_lipschitz,
            "maximum_lipschitz": self.maximum_lipschitz,
            "sufficient_decrease_tolerance": self.sufficient_decrease_tolerance,
        }

    def _blocks(self) -> tuple[tuple[torch.nn.Parameter, ...], ...]:
        return (
            (self.model.kernel_real, self.model.kernel_imag),
            (self.model.activation_weights,),
            (self.model.data_term_weights,),
        )

    @torch.no_grad()
    def _reset_memory(self) -> None:
        self.model.project_parameters()
        self.previous = [
            [parameter.detach().clone() for parameter in block]
            for block in self._blocks()
        ]

    @staticmethod
    @torch.no_grad()
    def _assign(
        parameters: Sequence[torch.nn.Parameter],
        values: Sequence[torch.Tensor],
    ) -> None:
        for parameter, value in zip(parameters, values, strict=True):
            parameter.copy_(value)

    @torch.no_grad()
    def _project(self, block_index: int) -> None:
        if block_index == 0:
            real = self.model.kernel_real
            imaginary = self.model.kernel_imag
            dimensions = (2, 3, 4)
            real.sub_(real.mean(dim=dimensions, keepdim=True))
            imaginary.sub_(imaginary.mean(dim=dimensions, keepdim=True))
            norm = torch.sqrt(
                real.square().sum(dim=dimensions, keepdim=True)
                + imaginary.square().sum(dim=dimensions, keepdim=True)
            )
            scale = torch.clamp(norm, min=1.0)
            real.div_(scale)
            imaginary.div_(scale)
        elif block_index == 2:
            self.model.data_term_weights.clamp_(min=0.0)

    def step(
        self,
        closure: Callable[[], Mapping[str, torch.Tensor]],
    ) -> dict[str, float]:
        """Perform one full alternating update across all three blocks."""

        block_metrics: dict[str, float] = {}
        total_gradient_norm_squared = 0.0
        for block_index, (block_name, parameters) in enumerate(
            zip(self.BLOCK_NAMES, self._blocks(), strict=True)
        ):
            current = [parameter.detach().clone() for parameter in parameters]
            extrapolated = [
                value + self.momentum * (value - previous)
                for value, previous in zip(
                    current, self.previous[block_index], strict=True
                )
            ]
            self._assign(parameters, extrapolated)

            losses = closure()
            energy_old = losses["loss"]
            if not bool(torch.isfinite(energy_old)):
                self._assign(parameters, current)
                raise FloatingPointError(
                    f"nonfinite iPALM energy in block {block_name}"
                )
            gradients = torch.autograd.grad(energy_old, parameters)
            if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                self._assign(parameters, current)
                raise FloatingPointError(
                    f"nonfinite iPALM gradient in block {block_name}"
                )
            gradient_norm_squared = sum(
                float(gradient.detach().square().sum().cpu())
                for gradient in gradients
            )
            total_gradient_norm_squared += gradient_norm_squared
            energy_value = float(energy_old.detach().cpu())

            accepted = False
            last_lhs = math.inf
            last_rhs = math.nan
            trials = 0
            for trials in range(1, self.max_backtracking + 1):
                current_lipschitz = self.lipschitz[block_index]
                candidates = [
                    value - gradient.detach() / current_lipschitz
                    for value, gradient in zip(
                        extrapolated, gradients, strict=True
                    )
                ]
                self._assign(parameters, candidates)
                self._project(block_index)
                projected = [
                    parameter.detach().clone() for parameter in parameters
                ]
                with torch.no_grad():
                    candidate_energy = closure()["loss"]
                last_lhs = float(candidate_energy.detach().cpu())
                differences = [
                    value - origin
                    for value, origin in zip(
                        projected, extrapolated, strict=True
                    )
                ]
                linear_term = sum(
                    float((difference * gradient.detach()).sum().cpu())
                    for difference, gradient in zip(
                        differences, gradients, strict=True
                    )
                )
                squared_norm = sum(
                    float(difference.square().sum().cpu())
                    for difference in differences
                )
                last_rhs = (
                    energy_value
                    + linear_term
                    + 0.5 * current_lipschitz * squared_norm
                )
                if last_rhs > 0:
                    tolerance_factor = 1.0 + self.sufficient_decrease_tolerance
                elif last_rhs < 0:
                    tolerance_factor = 1.0 - self.sufficient_decrease_tolerance
                else:
                    tolerance_factor = 1.0
                if (
                    math.isfinite(last_lhs)
                    and last_lhs <= last_rhs * tolerance_factor
                ):
                    self.lipschitz[block_index] = max(
                        self.minimum_lipschitz,
                        current_lipschitz * 0.75,
                    )
                    accepted = True
                    break
                self.lipschitz[block_index] = min(
                    self.maximum_lipschitz,
                    current_lipschitz * 2.0,
                )

            if not accepted:
                self._assign(parameters, current)
                raise RuntimeError(
                    f"iPALM backtracking failed for {block_name} after "
                    f"{trials} trials (Q_lhs={last_lhs:.8g}, "
                    f"Q_rhs={last_rhs:.8g})"
                )
            self.previous[block_index] = current
            block_metrics[
                f"{block_name}_lipschitz"
            ] = self.lipschitz[block_index]
            block_metrics[
                f"{block_name}_backtracking"
            ] = float(trials)

        self.iterations += 1
        with torch.no_grad():
            final_losses = closure()
        return {
            key: float(value.detach().cpu())
            for key, value in final_losses.items()
            if key != "prediction"
        } | {
            "gradient_norm": math.sqrt(total_gradient_norm_squared),
            **block_metrics,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "iterations": self.iterations,
            "lipschitz": list(self.lipschitz),
            "previous": [
                [value.detach().clone() for value in block]
                for block in self.previous
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_config = dict(state["config"])
        if saved_config != self.config:
            raise ValueError(
                "checkpoint iPALM configuration differs from the current run"
            )
        lipschitz = [float(value) for value in state["lipschitz"]]
        previous = state["previous"]
        blocks = self._blocks()
        if len(lipschitz) != len(blocks) or len(previous) != len(blocks):
            raise ValueError("invalid iPALM block state")
        restored: list[list[torch.Tensor]] = []
        for saved_block, parameters in zip(previous, blocks, strict=True):
            if len(saved_block) != len(parameters):
                raise ValueError("invalid iPALM parameter-block state")
            restored_block = []
            for saved, parameter in zip(
                saved_block, parameters, strict=True
            ):
                value = saved.to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if value.shape != parameter.shape:
                    raise ValueError(
                        "iPALM memory shape differs from model parameter"
                    )
                restored_block.append(value)
            restored.append(restored_block)
        self.lipschitz = lipschitz
        self.previous = restored
        self.iterations = int(state["iterations"])


class VariationalNetworkTrainer:
    """Train with publication-faithful iPALM or an explicit Adam ablation."""

    def __init__(
        self,
        model: VariationalNetwork,
        *,
        optimizer: str = "ipalm",
        learning_rate: float = 1e-3,
        device: str | torch.device | None = None,
        gradient_clip_norm: float = 1.0,
        data_config: Mapping[str, Any] | None = None,
        ipalm_initial_lipschitz: float = 1000.0,
        ipalm_momentum: float = 0.7,
        ipalm_max_backtracking: int = 40,
    ) -> None:
        if optimizer not in {"ipalm", "adam"}:
            raise ValueError("optimizer must be 'ipalm' or 'adam'")
        if learning_rate <= 0 or gradient_clip_norm <= 0:
            raise ValueError(
                "learning rate and gradient clipping must be positive"
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer_name = optimizer
        if optimizer == "ipalm":
            self.optimizer: IPALMOptimizer | torch.optim.Adam = (
                IPALMOptimizer(
                    self.model,
                    initial_lipschitz=ipalm_initial_lipschitz,
                    momentum=ipalm_momentum,
                    max_backtracking=ipalm_max_backtracking,
                )
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate
            )
        self.gradient_clip_norm = gradient_clip_norm
        self.data_config = dict(data_config or {})
        self.step = 0
        self.completed_epochs = 0

    def _train_step_adam(
        self, batch: Mapping[str, Any]
    ) -> dict[str, float]:
        assert isinstance(self.optimizer, torch.optim.Adam)
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.model.training_losses(batch)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(
                f"nonfinite loss before step {self.step + 1} "
                f"for {sample_id(batch)}"
            )
        losses["loss"].backward()
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(
            bool(torch.isfinite(value).all()) for value in gradients
        ):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"nonfinite gradient before step {self.step + 1} "
                f"for {sample_id(batch)}"
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        self.optimizer.step()
        self.model.project_parameters()
        return {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if key != "prediction"
        } | {"gradient_norm": float(gradient_norm.detach().cpu())}

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self.model.train()
        batch = move_batch(batch, self.device)
        if self.optimizer_name == "ipalm":
            assert isinstance(self.optimizer, IPALMOptimizer)
            metrics = self.optimizer.step(
                lambda: self.model.training_losses(batch)
            )
        else:
            metrics = self._train_step_adam(batch)
        self.step += 1
        return metrics

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "model_name": "hammernik_variational_network",
                "optimizer_name": self.optimizer_name,
                "step": self.step,
                "completed_epochs": self.completed_epochs,
                "model_config": self.model.model_config,
                "data_config": self.data_config,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            destination,
        )
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=True
        )
        if (
            checkpoint.get("format_version") not in {1, 2}
            or checkpoint.get("model_name")
            != "hammernik_variational_network"
        ):
            raise ValueError("unsupported Variational Network checkpoint")
        saved_optimizer = checkpoint.get("optimizer_name", "adam")
        if saved_optimizer != self.optimizer_name:
            raise ValueError(
                f"checkpoint uses {saved_optimizer}, current trainer uses "
                f"{self.optimizer_name}"
            )
        if dict(checkpoint["model_config"]) != dict(
            self.model.model_config
        ):
            raise ValueError(
                "checkpoint architecture differs from the current model"
            )
        saved_data = dict(checkpoint.get("data_config", {}))
        if self.data_config and saved_data != self.data_config:
            raise ValueError(
                "checkpoint data configuration differs from the current run"
            )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = int(checkpoint["step"])
        self.completed_epochs = int(
            checkpoint.get("completed_epochs", 0)
        )
        self.data_config = saved_data
