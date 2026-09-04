"""One-sided reliable-background foreground-mass guard (Stage-B O4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._validation import (
    StageBObjectiveError,
    detached_weight_sum,
    ensure_finite_scalar,
    finite_real,
    has_positive_weight,
    probability_eps,
    validate_probability,
    validate_spatial_tensor,
    validate_weight,
    weighted_mean,
)


@dataclass(frozen=True)
class ForegroundMassGuardOutput:
    """Loss and auditable mass values for the one-sided guard."""

    loss: Tensor
    current_background_mass: Tensor
    source_background_mass: Tensor
    background_weight_sum: Tensor
    margin: float


def foreground_mass_guard_components(
    probability: Tensor,
    teacher_probability: Tensor,
    background_weight: Tensor,
    *,
    margin: float,
    eps: float = 1.0e-6,
) -> ForegroundMassGuardOutput:
    """Penalize only foreground-probability growth on reliable background."""

    validate_probability(probability, name="probability")
    validate_probability(
        teacher_probability,
        name="teacher_probability",
        reference=probability,
        detached=True,
    )
    validate_weight(
        background_weight,
        name="background_weight",
        reference=probability,
    )
    checked_margin = finite_real(
        margin, name="margin", nonnegative=True
    )
    checked_eps = probability_eps(eps)
    background_sum = detached_weight_sum(background_weight)
    if not has_positive_weight(background_sum, name="background_weight"):
        raise StageBObjectiveError("reliable-background weight is empty")

    current_mass = weighted_mean(
        probability,
        background_weight,
        eps=checked_eps,
        weight_sum=background_sum,
    )
    source_mass = weighted_mean(
        teacher_probability,
        background_weight,
        eps=checked_eps,
        weight_sum=background_sum,
    )
    loss = torch.relu(current_mass - source_mass - checked_margin)
    ensure_finite_scalar(loss, name="foreground-mass guard")
    return ForegroundMassGuardOutput(
        loss=loss,
        current_background_mass=current_mass,
        source_background_mass=source_mass,
        background_weight_sum=background_sum,
        margin=checked_margin,
    )


def foreground_mass_guard(
    probability: Tensor,
    teacher_probability: Tensor,
    background_weight: Tensor,
    *,
    margin: float,
    eps: float = 1.0e-6,
) -> Tensor:
    """Return the scalar one-sided mass penalty."""

    return foreground_mass_guard_components(
        probability,
        teacher_probability,
        background_weight,
        margin=margin,
        eps=eps,
    ).loss


def foreground_mass_guard_from_logits(
    logits: Tensor,
    teacher_probability: Tensor,
    background_weight: Tensor,
    *,
    margin: float,
    eps: float = 1.0e-6,
) -> Tensor:
    """Convenience wrapper for student logits."""

    validate_spatial_tensor(logits, name="logits")
    return foreground_mass_guard(
        torch.sigmoid(logits),
        teacher_probability,
        background_weight,
        margin=margin,
        eps=eps,
    )


__all__ = [
    "ForegroundMassGuardOutput",
    "foreground_mass_guard",
    "foreground_mass_guard_components",
    "foreground_mass_guard_from_logits",
]
