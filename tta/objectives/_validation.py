"""Shared fail-closed validation for Stage-B label-free objectives."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


class StageBObjectiveError(ValueError):
    """An objective input or computed value violates the Stage-B contract."""


def finite_real(
    value: Any,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Return a validated finite Python float while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise StageBObjectiveError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise StageBObjectiveError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise StageBObjectiveError(f"{name} must be non-negative")
    return result


def probability_eps(value: Any, *, name: str = "eps") -> float:
    result = finite_real(value, name=name, positive=True)
    if result >= 0.5:
        raise StageBObjectiveError(f"{name} must be strictly less than 0.5")
    return result


def validate_spatial_tensor(value: Tensor, *, name: str) -> None:
    """Validate a non-empty real floating-point BCHW tensor."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 1:
        raise StageBObjectiveError(f"{name} must have shape [B,1,H,W]")
    if any(int(size) <= 0 for size in value.shape):
        raise StageBObjectiveError(f"{name} dimensions must all be positive")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError(f"{name} must be a real floating-point tensor")
    if not bool(torch.isfinite(value).all().detach().item()):
        raise StageBObjectiveError(f"{name} must contain only finite values")


def require_detached(value: Tensor, *, name: str) -> None:
    if value.requires_grad or value.grad_fn is not None:
        raise StageBObjectiveError(
            f"{name} must be detached from the adaptation autograd graph"
        )


def require_same_layout(value: Tensor, reference: Tensor, *, name: str) -> None:
    if value.shape != reference.shape:
        raise StageBObjectiveError(
            f"{name} shape must match the student tensor exactly"
        )
    if value.device != reference.device:
        raise StageBObjectiveError(
            f"{name} device must match the student tensor exactly"
        )
    if value.dtype != reference.dtype:
        raise StageBObjectiveError(
            f"{name} dtype must match the student tensor exactly"
        )


def validate_probability(
    value: Tensor,
    *,
    name: str,
    reference: Tensor | None = None,
    detached: bool = False,
) -> None:
    validate_spatial_tensor(value, name=name)
    if reference is not None:
        require_same_layout(value, reference, name=name)
    if detached:
        require_detached(value, name=name)
    outside = torch.logical_or(value < 0.0, value > 1.0)
    if bool(outside.any().detach().item()):
        raise StageBObjectiveError(f"{name} values must lie in [0,1]")


def validate_weight(
    value: Tensor,
    *,
    name: str,
    reference: Tensor,
) -> None:
    validate_probability(
        value,
        name=name,
        reference=reference,
        detached=True,
    )


def accumulation_dtype(value: Tensor) -> torch.dtype:
    if value.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return value.dtype


def detached_weight_sum(weight: Tensor) -> Tensor:
    return weight.detach().sum(dtype=accumulation_dtype(weight))


def has_positive_weight(weight_sum: Tensor, *, name: str) -> bool:
    if not bool(torch.isfinite(weight_sum).all().item()):
        raise StageBObjectiveError(f"{name} sum must be finite")
    return bool((weight_sum > 0.0).item())


def differentiable_zero(reference: Tensor) -> Tensor:
    """Return exact scalar zero connected to ``reference``'s graph."""

    return reference.reshape(-1)[0] * 0.0


def weighted_mean(
    value: Tensor,
    weight: Tensor,
    *,
    eps: float,
    weight_sum: Tensor | None = None,
) -> Tensor:
    """Compute a weighted mean with a low-precision-safe accumulator."""

    if value.shape != weight.shape:
        raise StageBObjectiveError("value and weight shapes must match exactly")
    dtype = accumulation_dtype(value)
    numerator = (value * weight).sum(dtype=dtype)
    denominator = (
        detached_weight_sum(weight) if weight_sum is None else weight_sum
    )
    result = numerator / (denominator + eps)
    ensure_finite_scalar(result, name="weighted mean")
    return result


def ensure_finite_scalar(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise StageBObjectiveError(f"{name} must be a scalar tensor")
    if not bool(torch.isfinite(value).all().detach().item()):
        raise StageBObjectiveError(f"{name} must be finite")
