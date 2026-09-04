"""Foreground/background-balanced Bernoulli entropy (Stage-B O1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._validation import (
    StageBObjectiveError,
    accumulation_dtype,
    detached_weight_sum,
    differentiable_zero,
    ensure_finite_scalar,
    has_positive_weight,
    probability_eps,
    validate_spatial_tensor,
    validate_weight,
    weighted_mean,
)


@dataclass(frozen=True)
class BalancedEntropyOutput:
    """Auditable O1 loss terms with detached regional weight sums."""

    total: Tensor
    foreground: Tensor
    background: Tensor
    foreground_weight_sum: Tensor
    background_weight_sum: Tensor
    has_foreground: bool


def bernoulli_entropy_map(logits: Tensor, *, eps: float = 1.0e-6) -> Tensor:
    """Return finite elementwise binary entropy for BCHW logits."""

    validate_spatial_tensor(logits, name="logits")
    checked_eps = probability_eps(eps)
    working = logits
    if logits.dtype in (torch.float16, torch.bfloat16):
        working = logits.to(dtype=torch.float32)
    probability = torch.sigmoid(working).clamp(checked_eps, 1.0 - checked_eps)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log1p(-probability)
    )
    if entropy.dtype != accumulation_dtype(logits):
        entropy = entropy.to(dtype=accumulation_dtype(logits))
    if not bool(torch.isfinite(entropy).all().detach().item()):
        raise StageBObjectiveError("Bernoulli entropy produced NaN/Inf")
    return entropy


def balanced_binary_entropy(
    logits: Tensor,
    foreground_weight: Tensor,
    background_weight: Tensor,
    *,
    eps: float = 1.0e-6,
) -> BalancedEntropyOutput:
    """Compute O1 with independently normalized, equally weighted regions.

    The foreground region may be empty.  Its term is then an exact scalar
    zero that remains connected to the student graph.  A missing reliable
    background region is a protocol error and is never silently ignored.
    """

    validate_spatial_tensor(logits, name="logits")
    checked_eps = probability_eps(eps)
    validate_weight(
        foreground_weight,
        name="foreground_weight",
        reference=logits,
    )
    validate_weight(
        background_weight,
        name="background_weight",
        reference=logits,
    )

    foreground_sum = detached_weight_sum(foreground_weight)
    background_sum = detached_weight_sum(background_weight)
    has_foreground = has_positive_weight(
        foreground_sum, name="foreground_weight"
    )
    if not has_positive_weight(background_sum, name="background_weight"):
        raise StageBObjectiveError("reliable-background weight is empty")

    entropy = bernoulli_entropy_map(logits, eps=checked_eps)
    foreground = (
        weighted_mean(
            entropy,
            foreground_weight,
            eps=checked_eps,
            weight_sum=foreground_sum,
        )
        if has_foreground
        else differentiable_zero(logits)
    )
    background = weighted_mean(
        entropy,
        background_weight,
        eps=checked_eps,
        weight_sum=background_sum,
    )
    total = 0.5 * foreground + 0.5 * background
    ensure_finite_scalar(total, name="balanced entropy")
    return BalancedEntropyOutput(
        total=total,
        foreground=foreground,
        background=background,
        foreground_weight_sum=foreground_sum,
        background_weight_sum=background_sum,
        has_foreground=has_foreground,
    )


foreground_background_balanced_entropy = balanced_binary_entropy
balanced_entropy = balanced_binary_entropy


__all__ = [
    "BalancedEntropyOutput",
    "balanced_binary_entropy",
    "balanced_entropy",
    "bernoulli_entropy_map",
    "foreground_background_balanced_entropy",
]
