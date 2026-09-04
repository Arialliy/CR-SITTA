"""Source-teacher foreground soft-overlap anchors for Stage-B."""

from __future__ import annotations

from typing import Literal

from torch import Tensor

from ._validation import (
    StageBObjectiveError,
    accumulation_dtype,
    detached_weight_sum,
    differentiable_zero,
    ensure_finite_scalar,
    has_positive_weight,
    probability_eps,
    validate_probability,
    validate_weight,
)


ForegroundOverlap = Literal["dice", "iou"]


def _foreground_overlap_anchor(
    probability: Tensor,
    teacher_probability: Tensor,
    foreground_weight: Tensor,
    *,
    kind: ForegroundOverlap,
    eps: float,
) -> Tensor:
    validate_probability(probability, name="probability")
    validate_probability(
        teacher_probability,
        name="teacher_probability",
        reference=probability,
        detached=True,
    )
    validate_weight(
        foreground_weight,
        name="foreground_weight",
        reference=probability,
    )
    checked_eps = probability_eps(eps)
    if kind not in ("dice", "iou"):
        raise StageBObjectiveError("kind must be exactly 'dice' or 'iou'")

    foreground_sum = detached_weight_sum(foreground_weight)
    if not has_positive_weight(foreground_sum, name="foreground_weight"):
        return differentiable_zero(probability)

    dtype = accumulation_dtype(probability)
    intersection = (
        foreground_weight * probability * teacher_probability
    ).sum(dtype=dtype)
    student_mass = (foreground_weight * probability).sum(dtype=dtype)
    teacher_mass = (foreground_weight * teacher_probability).sum(dtype=dtype)
    if kind == "dice":
        score = (2.0 * intersection + checked_eps) / (
            student_mass + teacher_mass + checked_eps
        )
    else:
        union = student_mass + teacher_mass - intersection
        score = (intersection + checked_eps) / (union + checked_eps)
    loss = 1.0 - score
    ensure_finite_scalar(loss, name=f"foreground soft-{kind} anchor")
    return loss


def foreground_soft_dice_anchor(
    probability: Tensor,
    teacher_probability: Tensor,
    foreground_weight: Tensor,
    eps: float = 1.0e-6,
) -> Tensor:
    """Return the weighted soft-Dice loss against a detached teacher."""

    return _foreground_overlap_anchor(
        probability,
        teacher_probability,
        foreground_weight,
        kind="dice",
        eps=eps,
    )


def foreground_soft_iou_anchor(
    probability: Tensor,
    teacher_probability: Tensor,
    foreground_weight: Tensor,
    eps: float = 1.0e-6,
) -> Tensor:
    """Return the weighted soft-IoU loss against a detached teacher."""

    return _foreground_overlap_anchor(
        probability,
        teacher_probability,
        foreground_weight,
        kind="iou",
        eps=eps,
    )


soft_dice_anchor = foreground_soft_dice_anchor
soft_iou_anchor = foreground_soft_iou_anchor


__all__ = [
    "ForegroundOverlap",
    "foreground_soft_dice_anchor",
    "foreground_soft_iou_anchor",
    "soft_dice_anchor",
    "soft_iou_anchor",
]
