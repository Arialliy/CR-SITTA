"""Composable ASB-SFR proposal objective for CR-SITTA Stage-C.

The objective consumes only detached Source-teacher evidence and detached
candidate metadata.  It intentionally has no dataset, corruption, severity,
ground-truth, or outer-evaluation input.  Its five weights are mandatory
keyword arguments so every experimental configuration records the exact
objective rather than inheriting hidden defaults.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ._validation import (
    StageBObjectiveError,
    accumulation_dtype,
    ensure_finite_scalar,
    finite_real,
    require_detached,
)
from .active_bootstrap import active_bootstrap_binary
from .local_contrast_consistency import candidate_local_contrast_consistency
from .router_regularization import router_regularization


@dataclass(frozen=True)
class ASBSFRObjectiveOutput:
    """Unweighted/weighted components and detached activity evidence."""

    total: Tensor
    foreground: Tensor
    background: Tensor
    candidate_contrast: Tensor
    adapter_l2: Tensor
    spatial_tv: Tensor
    weighted_foreground: Tensor
    weighted_background: Tensor
    weighted_candidate_contrast: Tensor
    weighted_adapter_l2: Tensor
    weighted_spatial_tv: Tensor
    active_fraction: Tensor
    active_foreground_weight: Tensor
    active_background_weight: Tensor
    candidate_weight_sum: Tensor
    active_pixel_count: int
    candidate_count: int
    has_active_support: bool
    active_mask: Tensor


def _validate_candidate_weights(
    candidate_weights: Tensor,
    *,
    candidate_count: int,
    reference: Tensor,
) -> Tensor:
    if not isinstance(candidate_weights, Tensor):
        raise TypeError("candidate_weights must be a torch.Tensor")
    if candidate_weights.ndim != 1 or int(candidate_weights.shape[0]) != candidate_count:
        raise StageBObjectiveError(
            "candidate_weights must have shape [candidate_count]"
        )
    if candidate_weights.device != reference.device:
        raise StageBObjectiveError(
            "candidate_weights device must match student_logits"
        )
    if candidate_weights.dtype != reference.dtype:
        raise StageBObjectiveError(
            "candidate_weights dtype must match student_logits"
        )
    if not candidate_weights.is_floating_point() or candidate_weights.is_complex():
        raise TypeError("candidate_weights must be real floating-point")
    require_detached(candidate_weights, name="candidate_weights")
    if not bool(torch.isfinite(candidate_weights).all().item()):
        raise StageBObjectiveError("candidate_weights must contain only finite values")
    if bool((candidate_weights < 0.0).any().item()):
        raise StageBObjectiveError("candidate_weights must be non-negative")
    return candidate_weights


def _weighted_candidate_contrast(
    student_logits: Tensor,
    teacher_logits: Tensor,
    candidate_cores: Sequence[Tensor],
    candidate_rings: Sequence[Tensor],
    candidate_weights: Tensor,
    *,
    temperature: float,
    ring_weight: float,
    huber_delta: float,
) -> tuple[Tensor, int, Tensor]:
    if isinstance(candidate_cores, Tensor) or isinstance(
        candidate_cores, (str, bytes)
    ):
        raise TypeError("candidate_cores must be a sequence of boolean tensors")
    if isinstance(candidate_rings, Tensor) or isinstance(
        candidate_rings, (str, bytes)
    ):
        raise TypeError("candidate_rings must be a sequence of boolean tensors")
    cores = tuple(candidate_cores)
    rings = tuple(candidate_rings)
    if len(cores) != len(rings):
        raise StageBObjectiveError(
            "candidate_cores and candidate_rings must have identical lengths"
        )
    checked_weights = _validate_candidate_weights(
        candidate_weights,
        candidate_count=len(cores),
        reference=student_logits,
    )
    weight_sum = checked_weights.sum(
        dtype=accumulation_dtype(checked_weights)
    ).detach()
    ensure_finite_scalar(weight_sum, name="candidate weight sum")
    if not cores:
        # Call the primitive once so teacher layout/detachment is validated
        # even when there are no candidate masks.
        empty_output = candidate_local_contrast_consistency(
            student_logits,
            teacher_logits,
            (),
            (),
            temperature=temperature,
            ring_weight=ring_weight,
            huber_delta=huber_delta,
        )
        return empty_output.total, 0, weight_sum

    # Calling the primitive per candidate preserves its live Huber loss while
    # permitting the explicit a_k weighting from v6 Eq. 4.6.3.
    losses: list[Tensor] = []
    for core, ring in zip(cores, rings, strict=True):
        output = candidate_local_contrast_consistency(
            student_logits,
            teacher_logits,
            (core,),
            (ring,),
            temperature=temperature,
            ring_weight=ring_weight,
            huber_delta=huber_delta,
        )
        losses.append(output.total)
    stacked = torch.stack(losses)
    result = (stacked * checked_weights).sum()
    ensure_finite_scalar(result, name="weighted candidate contrast")
    return result, len(cores), weight_sum


def asb_sfr_proposal_objective(
    student_logits: Tensor,
    teacher_probability: Tensor,
    teacher_logits: Tensor,
    target_weight: Tensor,
    background_weight: Tensor,
    stable_mask: Tensor,
    candidate_cores: Sequence[Tensor],
    candidate_rings: Sequence[Tensor],
    candidate_weights: Tensor,
    router_parameters: Mapping[str, Tensor],
    *,
    entropy_margin: float,
    foreground_loss_weight: float,
    background_loss_weight: float,
    candidate_contrast_loss_weight: float,
    adapter_l2_weight: float,
    spatial_tv_weight: float,
    min_active_pixels: int = 1,
    contrast_temperature: float = 0.25,
    contrast_ring_weight: float = 1.0,
    contrast_huber_delta: float = 1.0,
    eps: float = 1.0e-6,
) -> ASBSFRObjectiveOutput:
    """Build one label-free deterioration-probe proposal loss.

    Foreground and background BCE terms are normalized independently by
    :func:`active_bootstrap_binary`.  Candidate Huber losses follow
    ``sum_k a_k loss_k``.  L2 and spatial TV regularize only explicitly named
    router coefficient maps.
    """

    checked_foreground_weight = finite_real(
        foreground_loss_weight,
        name="foreground_loss_weight",
        nonnegative=True,
    )
    checked_background_weight = finite_real(
        background_loss_weight,
        name="background_loss_weight",
        nonnegative=True,
    )
    checked_contrast_weight = finite_real(
        candidate_contrast_loss_weight,
        name="candidate_contrast_loss_weight",
        nonnegative=True,
    )
    checked_l2_weight = finite_real(
        adapter_l2_weight,
        name="adapter_l2_weight",
        nonnegative=True,
    )
    checked_tv_weight = finite_real(
        spatial_tv_weight,
        name="spatial_tv_weight",
        nonnegative=True,
    )

    bootstrap = active_bootstrap_binary(
        student_logits,
        teacher_probability,
        target_weight,
        background_weight,
        stable_mask,
        entropy_margin=entropy_margin,
        min_active_pixels=min_active_pixels,
        eps=eps,
    )
    candidate_contrast, candidate_count, candidate_weight_sum = (
        _weighted_candidate_contrast(
            student_logits,
            teacher_logits,
            candidate_cores,
            candidate_rings,
            candidate_weights,
            temperature=contrast_temperature,
            ring_weight=contrast_ring_weight,
            huber_delta=contrast_huber_delta,
        )
    )
    regularization = router_regularization(router_parameters)

    # Stage-C is fail-closed at the episode level: candidate contrast or an
    # adapter regularizer must never manufacture a proposal when the teacher
    # reliability test produced no usable active support.
    support_gate = student_logits.new_tensor(
        1.0 if bootstrap.has_active_support else 0.0
    ).detach()
    weighted_foreground = (
        bootstrap.foreground * checked_foreground_weight * support_gate
    )
    weighted_background = (
        bootstrap.background * checked_background_weight * support_gate
    )
    weighted_candidate_contrast = (
        candidate_contrast * checked_contrast_weight * support_gate
    )
    weighted_adapter_l2 = (
        regularization.adapter_l2 * checked_l2_weight * support_gate
    )
    weighted_spatial_tv = (
        regularization.spatial_tv * checked_tv_weight * support_gate
    )
    total = (
        weighted_foreground
        + weighted_background
        + weighted_candidate_contrast
        + weighted_adapter_l2
        + weighted_spatial_tv
    )
    ensure_finite_scalar(total, name="ASB-SFR proposal objective")

    return ASBSFRObjectiveOutput(
        total=total,
        foreground=bootstrap.foreground,
        background=bootstrap.background,
        candidate_contrast=candidate_contrast,
        adapter_l2=regularization.adapter_l2,
        spatial_tv=regularization.spatial_tv,
        weighted_foreground=weighted_foreground,
        weighted_background=weighted_background,
        weighted_candidate_contrast=weighted_candidate_contrast,
        weighted_adapter_l2=weighted_adapter_l2,
        weighted_spatial_tv=weighted_spatial_tv,
        active_fraction=bootstrap.active_fraction,
        active_foreground_weight=bootstrap.active_foreground_weight,
        active_background_weight=bootstrap.active_background_weight,
        candidate_weight_sum=candidate_weight_sum,
        active_pixel_count=bootstrap.active_pixel_count,
        candidate_count=candidate_count,
        has_active_support=bootstrap.has_active_support,
        active_mask=bootstrap.active_mask,
    )


__all__ = ["ASBSFRObjectiveOutput", "asb_sfr_proposal_objective"]
