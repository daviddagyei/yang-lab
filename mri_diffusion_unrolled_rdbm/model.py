"""Diffusion-faithful end-to-end unrolling of the MRI RDBM sampler."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from mri_model import build_mri_rdbm
from mri_modl import conjugate_gradient
from mri_reconstruction import (
    adjoint_operator,
    channels_to_complex,
    complex_to_channels,
    forward_operator,
    measurement_residual,
)
from mri_unrolled_rdbm import UnrolledLossWeights, magnitude_ssim_loss


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def reverse_time_pairs(num_timesteps: int, sampling_timesteps: int) -> tuple[tuple[int, int], ...]:
    """Return the exact integer schedule used by ``PhysicsGuidedRDBM.reconstruct``."""

    if num_timesteps <= 0 or not 0 < sampling_timesteps <= num_timesteps:
        raise ValueError("sampling timesteps must be within the diffusion horizon")
    times = torch.linspace(-1, num_timesteps - 1, steps=sampling_timesteps + 1)
    ordered = list(reversed(times.long().tolist()))
    return tuple(zip(ordered[:-1], ordered[1:]))


class DiffusionUnrolledRDBM(nn.Module):
    """Train the complete sampled RDBM trajectory and CG-SENSE blocks end to end.

    One shared, randomly initialized RDBM U-Net is reused at every original
    reverse-diffusion timestep. Each predicted clean image is refined by a
    differentiable CG-SENSE proximal solve before the original bridge equation
    advances the latent state.
    """

    def __init__(
        self,
        *,
        rdbm_config: Mapping[str, Any] | None = None,
        cg_iterations: int = 6,
        cg_tolerance: float = 1e-10,
        initial_lambda: float = 0.05,
        loss_weights: UnrolledLossWeights | Mapping[str, float] | None = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if cg_iterations <= 0 or cg_tolerance < 0 or initial_lambda <= 0:
            raise ValueError("invalid CG or regularization configuration")
        if loss_weights is None:
            resolved_weights = UnrolledLossWeights()
        elif isinstance(loss_weights, UnrolledLossWeights):
            resolved_weights = loss_weights
        else:
            resolved_weights = UnrolledLossWeights(**dict(loss_weights))

        self.rdbm = build_mri_rdbm(**dict(rdbm_config or {}))
        if self.rdbm.objective != "pred_x_start" or self.rdbm.sampling_type != "pred_x_start":
            raise ValueError("diffusion-faithful unrolling currently requires pred_x_start")
        pairs = reverse_time_pairs(self.rdbm.num_timesteps, self.rdbm.sampling_timesteps)
        self.register_buffer(
            "time_pairs",
            torch.tensor(pairs, dtype=torch.long),
            persistent=False,
        )
        self.cg_iterations = int(cg_iterations)
        self.cg_tolerance = float(cg_tolerance)
        self.loss_weights = resolved_weights
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.raw_lambdas = nn.Parameter(
            torch.full(
                (len(pairs),),
                _inverse_softplus(initial_lambda),
                dtype=torch.float32,
            )
        )
        self.model_config = {
            "rdbm_config": dict(self.rdbm.model_config),
            "cg_iterations": self.cg_iterations,
            "cg_tolerance": self.cg_tolerance,
            "initial_lambda": float(initial_lambda),
            "loss_weights": asdict(resolved_weights),
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    @property
    def unroll_steps(self) -> int:
        return int(self.time_pairs.shape[0])

    @property
    def regularization_weights(self) -> torch.Tensor:
        return F.softplus(self.raw_lambdas)

    @staticmethod
    def _validate_inputs(
        initial: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if initial.ndim != 4 or initial.shape[1] != 2:
            raise ValueError("initial reconstruction must be [B, 2, H, W]")
        if not torch.is_complex(measured_kspace) or not torch.is_complex(sensitivity_maps):
            raise TypeError("measured k-space and sensitivity maps must be complex")
        if measured_kspace.ndim != 4 or sensitivity_maps.ndim != 4:
            raise ValueError("k-space and sensitivity maps must be batched [B, C, H, W]")
        if mask.ndim != 2 or mask.shape[0] != initial.shape[0]:
            raise ValueError("mask must be batched [B, W]")

    def _predict_clean(
        self,
        state: torch.Tensor,
        initial: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return activation_checkpoint(
                self.rdbm.model,
                state,
                initial,
                timestep,
                use_reentrant=False,
            )
        return self.rdbm.model(state, initial, timestep)

    def _cg_consistency(
        self,
        predicted_clean: torch.Tensor,
        atb: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        encoded_shape = measured_kspace.shape[-2:]
        predicted_complex = channels_to_complex(predicted_clean, channel_dim=1)
        resolved_weight = weight.to(predicted_complex.real.dtype)
        right_hand_side = atb + resolved_weight * predicted_complex

        def normal_operator(image: torch.Tensor) -> torch.Tensor:
            encoded = forward_operator(image, sensitivity_maps, mask, encoded_shape)
            return (
                adjoint_operator(encoded, sensitivity_maps, mask)
                + resolved_weight * image
            )

        return conjugate_gradient(
            normal_operator,
            right_hand_side,
            initial=predicted_complex,
            max_iterations=self.cg_iterations,
            tolerance=self.cg_tolerance,
        )

    def _bridge_transition(
        self,
        state: torch.Tensor,
        initial: torch.Tensor,
        clean: torch.Tensor,
        time: int,
        time_next: int,
    ) -> torch.Tensor:
        """Apply the original deterministic pred-x-start RDBM bridge equation."""

        if time_next < 0:
            return clean
        # Keep the float32 bridge coefficients used by the original sampler;
        # autocast may run the U-Net in BF16, while the trajectory remains stable.
        theta_next = self.rdbm.Theta[time_next]
        if time == self.rdbm.num_timesteps - 1:
            return initial + theta_next * (clean - initial)
        theta_now = self.rdbm.Theta[time]
        sigma_now = self.rdbm.Sigma[time]
        sigma_next = self.rdbm.Sigma[time_next]
        ratio = sigma_next / sigma_now
        return (
            initial
            + ratio * (state - initial)
            + (theta_next - theta_now * ratio) * (clean - initial)
        )

    def forward(
        self,
        initial: torch.Tensor,
        measured_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run the complete sampled diffusion trajectory with CG after every prediction."""

        self._validate_inputs(initial, measured_kspace, sensitivity_maps, mask)
        atb = adjoint_operator(measured_kspace, sensitivity_maps, mask)
        state = initial
        clean_trajectory = [initial]
        for stage, pair in enumerate(self.time_pairs.tolist()):
            time, time_next = int(pair[0]), int(pair[1])
            timestep = torch.full(
                (initial.shape[0],), time, dtype=torch.long, device=initial.device
            )
            predicted_clean = self._predict_clean(state, initial, timestep)
            consistent_complex = self._cg_consistency(
                predicted_clean,
                atb,
                measured_kspace,
                sensitivity_maps,
                mask,
                self.regularization_weights[stage],
            )
            consistent_clean = complex_to_channels(consistent_complex, channel_dim=1)
            clean_trajectory.append(consistent_clean)
            state = self._bridge_transition(
                state, initial, consistent_clean, time, time_next
            )
        return clean_trajectory if return_trajectory else clean_trajectory[-1]

    def training_losses(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Use the same supervised objective as the existing three-stage model."""

        required = {
            "initial_reconstruction",
            "target",
            "undersampled_kspace",
            "sensitivity_maps",
            "mask",
        }
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"MRI batch is missing: {', '.join(sorted(missing))}")
        trajectory = self(
            batch["initial_reconstruction"],
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
            return_trajectory=True,
        )
        assert isinstance(trajectory, list)
        prediction = trajectory[-1]
        target = batch["target"]
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        prediction_complex = channels_to_complex(prediction, channel_dim=1)
        target_complex = channels_to_complex(target, channel_dim=1)
        complex_loss = F.l1_loss(prediction, target)
        magnitude_loss = F.l1_loss(
            torch.abs(prediction_complex), torch.abs(target_complex)
        )
        ssim_loss = magnitude_ssim_loss(prediction_complex, target_complex)
        measurement_loss = measurement_residual(
            prediction_complex,
            batch["undersampled_kspace"],
            batch["sensitivity_maps"],
            batch["mask"],
        ).square()
        intermediate_loss = torch.stack(
            [F.l1_loss(clean, target) for clean in trajectory[1:-1]]
        ).mean()
        weights = self.loss_weights
        total = (
            weights.complex_image * complex_loss
            + weights.magnitude * magnitude_loss
            + weights.ssim * ssim_loss
            + weights.measurement * measurement_loss
            + weights.intermediate * intermediate_loss
        )
        return {
            "loss": total,
            "complex_l1": complex_loss,
            "magnitude_l1": magnitude_loss,
            "ssim_loss": ssim_loss,
            "measurement": measurement_loss,
            "intermediate_l1": intermediate_loss,
            "lambda_mean": self.regularization_weights.mean(),
            "prediction": prediction,
        }


def build_diffusion_unrolled_rdbm(**kwargs: Any) -> DiffusionUnrolledRDBM:
    return DiffusionUnrolledRDBM(**kwargs)
