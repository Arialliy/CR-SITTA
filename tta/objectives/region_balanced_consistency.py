"""Source-anchored, region-balanced consistency primitives (O2--O4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ._validation import (
    StageBObjectiveError,
    detached_weight_sum,
    differentiable_zero,
    ensure_finite_scalar,
    finite_real,
    has_positive_weight,
    probability_eps,
    require_detached,
    require_same_layout,
    validate_probability,
    validate_spatial_tensor,
    validate_weight,
    weighted_mean,
)
from .foreground_mass_guard import foreground_mass_guard_components
from .soft_iou_anchor import (
    ForegroundOverlap,
    foreground_soft_dice_anchor,
    foreground_soft_iou_anchor,
)


ConsistencyDivergence = Literal["bce", "js"]


@dataclass(frozen=True)
class SourceAnchoredRegionWeights:
    """Detached high-recall foreground and reliable-background weights."""

    foreground: Tensor
    background: Tensor
    foreground_weight_sum: Tensor
    background_weight_sum: Tensor
    has_foreground: bool


@dataclass(frozen=True)
class RegionConsistencyOutput:
    """One aligned-view consistency loss with separate normalization."""

    total: Tensor
    foreground: Tensor
    background: Tensor
    foreground_weight_sum: Tensor
    background_weight_sum: Tensor
    has_foreground: bool
    divergence: ConsistencyDivergence


@dataclass(frozen=True)
class MultiViewConsistencyOutput:
    """Mean consistency and the auditable per-view scalar components."""

    total: Tensor
    foreground: Tensor
    background: Tensor
    per_view_total: Tensor
    per_view_foreground: Tensor
    per_view_background: Tensor
    foreground_weight_sum: Tensor
    background_weight_sum: Tensor
    has_foreground: bool
    view_count: int
    divergence: ConsistencyDivergence


@dataclass(frozen=True)
class RegionLossOutput:
    """The v5 O3/O4 region loss terms for one aligned student view."""

    total: Tensor
    foreground: Tensor
    background: Tensor
    mass_guard: Tensor
    foreground_weight_sum: Tensor
    background_weight_sum: Tensor
    has_foreground: bool
    foreground_overlap: ForegroundOverlap


def _bounded_unit_interval(value: float, *, name: str) -> float:
    result = finite_real(value, name=name, nonnegative=True)
    if result > 1.0:
        raise StageBObjectiveError(f"{name} must lie in [0,1]")
    return result


def build_source_anchored_region_weights(
    teacher_probability: Tensor,
    teacher_uncertainty: Tensor,
    *,
    foreground_threshold: float,
    background_threshold: float,
    foreground_gamma: float = 1.0,
    background_gamma: float = 1.0,
    foreground_temperature: float = 1.0,
    background_temperature: float = 1.0,
    foreground_guard: Tensor | None = None,
) -> SourceAnchoredRegionWeights:
    """Build the detached v5 weights using only source-teacher quantities.

    ``foreground_guard`` is a detached soft guard band in ``[0,1]``.  It is
    applied only to reliable background; it is not a label-bearing input.
    """

    validate_probability(
        teacher_probability,
        name="teacher_probability",
        detached=True,
    )
    validate_spatial_tensor(teacher_uncertainty, name="teacher_uncertainty")
    require_same_layout(
        teacher_uncertainty,
        teacher_probability,
        name="teacher_uncertainty",
    )
    require_detached(teacher_uncertainty, name="teacher_uncertainty")
    if bool((teacher_uncertainty < 0.0).any().detach().item()):
        raise StageBObjectiveError("teacher_uncertainty must be non-negative")

    foreground_threshold = _bounded_unit_interval(
        foreground_threshold, name="foreground_threshold"
    )
    background_threshold = _bounded_unit_interval(
        background_threshold, name="background_threshold"
    )
    if not background_threshold < foreground_threshold:
        raise StageBObjectiveError(
            "background_threshold must be strictly below foreground_threshold"
        )
    foreground_gamma = finite_real(
        foreground_gamma, name="foreground_gamma", nonnegative=True
    )
    background_gamma = finite_real(
        background_gamma, name="background_gamma", nonnegative=True
    )
    foreground_temperature = finite_real(
        foreground_temperature, name="foreground_temperature", positive=True
    )
    background_temperature = finite_real(
        background_temperature, name="background_temperature", positive=True
    )

    if foreground_guard is None:
        guard = torch.zeros_like(teacher_probability)
    else:
        validate_probability(
            foreground_guard,
            name="foreground_guard",
            reference=teacher_probability,
            detached=True,
        )
        guard = foreground_guard

    foreground_selector = (
        teacher_probability >= foreground_threshold
    ).to(dtype=teacher_probability.dtype)
    background_selector = (
        teacher_probability <= background_threshold
    ).to(dtype=teacher_probability.dtype)
    foreground = (
        foreground_selector
        * teacher_probability.pow(foreground_gamma)
        * torch.exp(-teacher_uncertainty / foreground_temperature)
    ).detach()
    background = (
        background_selector
        * (1.0 - teacher_probability).pow(background_gamma)
        * torch.exp(-teacher_uncertainty / background_temperature)
        * (1.0 - guard)
    ).detach()
    validate_weight(
        foreground,
        name="foreground_weight",
        reference=teacher_probability,
    )
    validate_weight(
        background,
        name="background_weight",
        reference=teacher_probability,
    )
    foreground_sum = detached_weight_sum(foreground)
    background_sum = detached_weight_sum(background)
    has_foreground = has_positive_weight(
        foreground_sum, name="foreground_weight"
    )
    if not has_positive_weight(background_sum, name="background_weight"):
        raise StageBObjectiveError("reliable-background weight is empty")
    return SourceAnchoredRegionWeights(
        foreground=foreground,
        background=background,
        foreground_weight_sum=foreground_sum,
        background_weight_sum=background_sum,
        has_foreground=has_foreground,
    )


def bernoulli_jensen_shannon_map(
    probability: Tensor,
    teacher_probability: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    """Elementwise Jensen--Shannon divergence between Bernoulli variables."""

    validate_probability(probability, name="probability")
    validate_probability(
        teacher_probability,
        name="teacher_probability",
        reference=probability,
        detached=True,
    )
    checked_eps = probability_eps(eps)
    working_probability = probability
    working_teacher = teacher_probability
    if probability.dtype in (torch.float16, torch.bfloat16):
        working_probability = probability.to(dtype=torch.float32)
        working_teacher = teacher_probability.to(dtype=torch.float32)
    student = working_probability.clamp(checked_eps, 1.0 - checked_eps)
    teacher = working_teacher.clamp(checked_eps, 1.0 - checked_eps)
    midpoint = 0.5 * (student + teacher)
    student_kl = (
        student * (torch.log(student) - torch.log(midpoint))
        + (1.0 - student)
        * (torch.log1p(-student) - torch.log1p(-midpoint))
    )
    teacher_kl = (
        teacher * (torch.log(teacher) - torch.log(midpoint))
        + (1.0 - teacher)
        * (torch.log1p(-teacher) - torch.log1p(-midpoint))
    )
    divergence = 0.5 * (student_kl + teacher_kl)
    if not bool(torch.isfinite(divergence).all().detach().item()):
        raise StageBObjectiveError("Bernoulli Jensen-Shannon produced NaN/Inf")
    return divergence


def _consistency_map(
    logits: Tensor,
    teacher_probability: Tensor,
    *,
    divergence: ConsistencyDivergence,
    eps: float,
) -> Tensor:
    if divergence == "bce":
        return F.binary_cross_entropy_with_logits(
            logits,
            teacher_probability,
            reduction="none",
        )
    if divergence == "js":
        return bernoulli_jensen_shannon_map(
            torch.sigmoid(logits), teacher_probability, eps=eps
        )
    raise StageBObjectiveError("divergence must be exactly 'bce' or 'js'")


def foreground_soft_bce_consistency(
    logits: Tensor,
    teacher_probability: Tensor,
    foreground_weight: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    """Foreground teacher BCE; an absent foreground returns graph-zero."""

    validate_spatial_tensor(logits, name="logits")
    validate_probability(
        teacher_probability,
        name="teacher_probability",
        reference=logits,
        detached=True,
    )
    validate_weight(
        foreground_weight,
        name="foreground_weight",
        reference=logits,
    )
    checked_eps = probability_eps(eps)
    foreground_sum = detached_weight_sum(foreground_weight)
    if not has_positive_weight(foreground_sum, name="foreground_weight"):
        return differentiable_zero(logits)
    value = F.binary_cross_entropy_with_logits(
        logits, teacher_probability, reduction="none"
    )
    return weighted_mean(
        value,
        foreground_weight,
        eps=checked_eps,
        weight_sum=foreground_sum,
    )


def reliable_background_soft_bce(
    logits: Tensor,
    teacher_probability: Tensor,
    background_weight: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    """Reliable-background soft BCE normalized only by background weight."""

    validate_spatial_tensor(logits, name="logits")
    validate_probability(
        teacher_probability,
        name="teacher_probability",
        reference=logits,
        detached=True,
    )
    validate_weight(
        background_weight,
        name="background_weight",
        reference=logits,
    )
    checked_eps = probability_eps(eps)
    background_sum = detached_weight_sum(background_weight)
    if not has_positive_weight(background_sum, name="background_weight"):
        raise StageBObjectiveError("reliable-background weight is empty")
    value = F.binary_cross_entropy_with_logits(
        logits, teacher_probability, reduction="none"
    )
    return weighted_mean(
        value,
        background_weight,
        eps=checked_eps,
        weight_sum=background_sum,
    )


def region_balanced_consistency(
    logits: Tensor,
    teacher_probability: Tensor,
    foreground_weight: Tensor,
    background_weight: Tensor,
    *,
    divergence: ConsistencyDivergence = "bce",
    eps: float = 1.0e-6,
) -> RegionConsistencyOutput:
    """Compute one inverse-aligned view's separately normalized O2 terms."""

    validate_spatial_tensor(logits, name="logits")
    validate_probability(
        teacher_probability,
        name="teacher_probability",
        reference=logits,
        detached=True,
    )
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
    checked_eps = probability_eps(eps)
    if divergence not in ("bce", "js"):
        raise StageBObjectiveError("divergence must be exactly 'bce' or 'js'")
    foreground_sum = detached_weight_sum(foreground_weight)
    background_sum = detached_weight_sum(background_weight)
    has_foreground = has_positive_weight(
        foreground_sum, name="foreground_weight"
    )
    if not has_positive_weight(background_sum, name="background_weight"):
        raise StageBObjectiveError("reliable-background weight is empty")
    value = _consistency_map(
        logits,
        teacher_probability,
        divergence=divergence,
        eps=checked_eps,
    )
    foreground = (
        weighted_mean(
            value,
            foreground_weight,
            eps=checked_eps,
            weight_sum=foreground_sum,
        )
        if has_foreground
        else differentiable_zero(logits)
    )
    background = weighted_mean(
        value,
        background_weight,
        eps=checked_eps,
        weight_sum=background_sum,
    )
    total = 0.5 * foreground + 0.5 * background
    ensure_finite_scalar(total, name="region-balanced consistency")
    return RegionConsistencyOutput(
        total=total,
        foreground=foreground,
        background=background,
        foreground_weight_sum=foreground_sum,
        background_weight_sum=background_sum,
        has_foreground=has_foreground,
        divergence=divergence,
    )


def _normalise_aligned_logits(
    aligned_student_logits: Tensor | Sequence[Tensor],
) -> tuple[Tensor, ...]:
    if isinstance(aligned_student_logits, Tensor):
        if aligned_student_logits.ndim == 5:
            values = tuple(aligned_student_logits.unbind(dim=0))
        elif aligned_student_logits.ndim == 4:
            values = (aligned_student_logits,)
        else:
            raise StageBObjectiveError(
                "aligned_student_logits tensor must have shape "
                "[V,B,1,H,W] or [B,1,H,W]"
            )
    elif isinstance(aligned_student_logits, Sequence):
        values = tuple(aligned_student_logits)
    else:
        raise TypeError(
            "aligned_student_logits must be a tensor or a sequence of tensors"
        )
    if not values:
        raise StageBObjectiveError(
            "aligned_student_logits must contain at least one view"
        )
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError("every aligned student view must be a torch.Tensor")
    return values


def source_anchored_multiview_consistency(
    aligned_student_logits: Tensor | Sequence[Tensor],
    teacher_probability: Tensor,
    foreground_weight: Tensor,
    background_weight: Tensor,
    *,
    divergence: ConsistencyDivergence = "bce",
    eps: float = 1.0e-6,
) -> MultiViewConsistencyOutput:
    """Average O2 across student views already inverse-aligned to source."""

    views = _normalise_aligned_logits(aligned_student_logits)
    outputs = tuple(
        region_balanced_consistency(
            logits,
            teacher_probability,
            foreground_weight,
            background_weight,
            divergence=divergence,
            eps=eps,
        )
        for logits in views
    )
    per_view_total = torch.stack([output.total for output in outputs])
    per_view_foreground = torch.stack(
        [output.foreground for output in outputs]
    )
    per_view_background = torch.stack(
        [output.background for output in outputs]
    )
    total = per_view_total.mean()
    foreground = per_view_foreground.mean()
    background = per_view_background.mean()
    ensure_finite_scalar(total, name="source-anchored multiview consistency")
    first = outputs[0]
    return MultiViewConsistencyOutput(
        total=total,
        foreground=foreground,
        background=background,
        per_view_total=per_view_total,
        per_view_foreground=per_view_foreground,
        per_view_background=per_view_background,
        foreground_weight_sum=first.foreground_weight_sum,
        background_weight_sum=first.background_weight_sum,
        has_foreground=first.has_foreground,
        view_count=len(outputs),
        divergence=divergence,
    )


region_balanced_multiview_consistency = source_anchored_multiview_consistency


def region_balanced_loss(
    logits: Tensor,
    teacher_probability: Tensor,
    foreground_weight: Tensor,
    background_weight: Tensor,
    *,
    mass_margin: float,
    eps: float = 1.0e-6,
    foreground_overlap: ForegroundOverlap = "dice",
) -> RegionLossOutput:
    """Compute the v5 soft-overlap + background BCE + mass-guard loss."""

    validate_spatial_tensor(logits, name="logits")
    validate_probability(
        teacher_probability,
        name="teacher_probability",
        reference=logits,
        detached=True,
    )
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
    if foreground_overlap not in ("dice", "iou"):
        raise StageBObjectiveError(
            "foreground_overlap must be exactly 'dice' or 'iou'"
        )
    probability = torch.sigmoid(logits)
    foreground = (
        foreground_soft_dice_anchor(
            probability,
            teacher_probability,
            foreground_weight,
            eps,
        )
        if foreground_overlap == "dice"
        else foreground_soft_iou_anchor(
            probability,
            teacher_probability,
            foreground_weight,
            eps,
        )
    )
    background = reliable_background_soft_bce(
        logits,
        teacher_probability,
        background_weight,
        eps=eps,
    )
    mass = foreground_mass_guard_components(
        probability,
        teacher_probability,
        background_weight,
        margin=mass_margin,
        eps=eps,
    )
    total = foreground + background + mass.loss
    ensure_finite_scalar(total, name="region-balanced loss")
    foreground_sum = detached_weight_sum(foreground_weight)
    return RegionLossOutput(
        total=total,
        foreground=foreground,
        background=background,
        mass_guard=mass.loss,
        foreground_weight_sum=foreground_sum,
        background_weight_sum=mass.background_weight_sum,
        has_foreground=has_positive_weight(
            foreground_sum, name="foreground_weight"
        ),
        foreground_overlap=foreground_overlap,
    )


__all__ = [
    "ConsistencyDivergence",
    "MultiViewConsistencyOutput",
    "RegionConsistencyOutput",
    "RegionLossOutput",
    "SourceAnchoredRegionWeights",
    "bernoulli_jensen_shannon_map",
    "build_source_anchored_region_weights",
    "foreground_soft_bce_consistency",
    "region_balanced_consistency",
    "region_balanced_loss",
    "region_balanced_multiview_consistency",
    "reliable_background_soft_bce",
    "source_anchored_multiview_consistency",
]
