"""Differentiable complex conjugate gradients for MoDL data consistency."""

from __future__ import annotations

from collections.abc import Callable

import torch


ComplexOperator = Callable[[torch.Tensor], torch.Tensor]


def _batch_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return a real inner product for each batch item."""

    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("CG tensors must have matching [B, ...] shapes")
    dimensions = tuple(range(1, left.ndim))
    return torch.sum(torch.conj(left) * right, dim=dimensions).real


def _expand_batch(values: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.reshape(values.shape[0], *([1] * (ndim - 1)))


def conjugate_gradient(
    operator: ComplexOperator,
    right_hand_side: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    max_iterations: int = 10,
    tolerance: float = 1e-10,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Solve a batched Hermitian positive-definite system without detaching autograd.

    ``tolerance`` is applied to the squared residual norm, following the authors'
    official MoDL implementation. The loop always has a finite iteration cap.
    """

    if not torch.is_complex(right_hand_side):
        raise TypeError("right_hand_side must be complex")
    if right_hand_side.ndim < 2:
        raise ValueError("right_hand_side must have a batch dimension")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0 or epsilon <= 0:
        raise ValueError("tolerance must be nonnegative and epsilon must be positive")

    solution = torch.zeros_like(right_hand_side) if initial is None else initial
    if solution.shape != right_hand_side.shape:
        raise ValueError("initial and right_hand_side shapes differ")
    residual = right_hand_side - operator(solution)
    direction = residual
    residual_energy = _batch_inner(residual, residual)

    for _ in range(max_iterations):
        if bool(torch.all(residual_energy.detach() <= tolerance)):
            break
        applied = operator(direction)
        denominator = _batch_inner(direction, applied).clamp_min(epsilon)
        alpha = _expand_batch(residual_energy / denominator, right_hand_side.ndim)
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        next_energy = _batch_inner(residual, residual)
        beta = _expand_batch(
            next_energy / residual_energy.clamp_min(epsilon), right_hand_side.ndim
        )
        direction = residual + beta * direction
        residual_energy = next_energy

    return solution
