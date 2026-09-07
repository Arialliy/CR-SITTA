"""Candidate-local contrast restoration for Stage-C proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor

from ._validation import (
    StageBObjectiveError,
    differentiable_zero,
    ensure_finite_scalar,
    finite_real,
    require_detached,
    validate_spatial_tensor,
)


@dataclass(frozen=True)
class LocalContrastConsistencyOutput:
    """Scalar loss and detached per-candidate response audit."""

    total: Tensor
    per_candidate_loss: Tensor
    student_absolute_response: Tensor
    teacher_absolute_response: Tensor
    student_local_contrast: Tensor
    teacher_local_contrast: Tensor
    candidate_count: int


def _validate_masks(
    masks: Sequence[Tensor],
    *,
    name: str,
    reference: Tensor,
) -> tuple[Tensor, ...]:
    if isinstance(masks, Tensor) or isinstance(masks, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of boolean tensors")
    result = tuple(masks)
    for index, mask in enumerate(result):
        if not isinstance(mask, Tensor):
            raise TypeError(f"{name}[{index}] must be a torch.Tensor")
        if mask.shape != reference.shape or mask.device != reference.device:
            raise StageBObjectiveError(
                f"{name}[{index}] layout must match logits exactly"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"{name}[{index}] must have boolean dtype")
        require_detached(mask, name=f"{name}[{index}]")
    return result


def _smooth_max(values: Tensor, temperature: float) -> Tensor:
    # The area-normalized LSE avoids a candidate-size-dependent offset.
    count = values.numel()
    if count <= 0:
        raise StageBObjectiveError("candidate core must be non-empty")
    return temperature * (
        torch.logsumexp(values / temperature, dim=0)
        - torch.log(values.new_tensor(float(count)))
    )


def candidate_local_contrast_consistency(
    student_logits: Tensor,
    teacher_logits: Tensor,
    candidate_cores: Sequence[Tensor],
    candidate_rings: Sequence[Tensor],
    *,
    temperature: float = 0.25,
    ring_weight: float = 1.0,
    huber_delta: float = 1.0,
) -> LocalContrastConsistencyOutput:
    """Restore detached Source candidate contrast on a deteriorated view.

    Empty candidate collections return graph-connected zero.  A candidate
    with an empty core or ring fails closed because its contrast is undefined.
    """

    validate_spatial_tensor(student_logits, name="student_logits")
    validate_spatial_tensor(teacher_logits, name="teacher_logits")
    if teacher_logits.shape != student_logits.shape:
        raise StageBObjectiveError("teacher_logits shape must match student_logits")
    if teacher_logits.device != student_logits.device:
        raise StageBObjectiveError("teacher_logits device must match student_logits")
    if teacher_logits.dtype != student_logits.dtype:
        raise StageBObjectiveError("teacher_logits dtype must match student_logits")
    require_detached(teacher_logits, name="teacher_logits")
    cores = _validate_masks(
        candidate_cores, name="candidate_cores", reference=student_logits
    )
    rings = _validate_masks(
        candidate_rings, name="candidate_rings", reference=student_logits
    )
    if len(cores) != len(rings):
        raise StageBObjectiveError(
            "candidate_cores and candidate_rings must have identical lengths"
        )
    checked_temperature = finite_real(
        temperature, name="temperature", positive=True
    )
    checked_ring_weight = finite_real(
        ring_weight, name="ring_weight", nonnegative=True
    )
    checked_delta = finite_real(huber_delta, name="huber_delta", positive=True)

    if not cores:
        empty = student_logits.new_empty((0,)).detach()
        return LocalContrastConsistencyOutput(
            total=differentiable_zero(student_logits),
            per_candidate_loss=empty,
            student_absolute_response=empty,
            teacher_absolute_response=empty,
            student_local_contrast=empty,
            teacher_local_contrast=empty,
            candidate_count=0,
        )

    student_absolute: list[Tensor] = []
    teacher_absolute: list[Tensor] = []
    student_contrast: list[Tensor] = []
    teacher_contrast: list[Tensor] = []
    losses: list[Tensor] = []
    for index, (core, ring) in enumerate(zip(cores, rings, strict=True)):
        if not bool(core.any().item()):
            raise StageBObjectiveError(f"candidate core {index} is empty")
        if not bool(ring.any().item()):
            raise StageBObjectiveError(f"candidate ring {index} is empty")
        student_abs = _smooth_max(student_logits[core], checked_temperature)
        teacher_abs = _smooth_max(teacher_logits[core], checked_temperature)
        student_value = student_abs - checked_ring_weight * student_logits[ring].mean()
        teacher_value = teacher_abs - checked_ring_weight * teacher_logits[ring].mean()
        loss = F.huber_loss(
            student_value,
            teacher_value,
            reduction="sum",
            delta=checked_delta,
        )
        student_absolute.append(student_abs)
        teacher_absolute.append(teacher_abs)
        student_contrast.append(student_value)
        teacher_contrast.append(teacher_value)
        losses.append(loss)

    per_candidate = torch.stack(losses)
    total = per_candidate.mean()
    ensure_finite_scalar(total, name="local contrast consistency")
    return LocalContrastConsistencyOutput(
        total=total,
        per_candidate_loss=per_candidate.detach(),
        student_absolute_response=torch.stack(student_absolute).detach(),
        teacher_absolute_response=torch.stack(teacher_absolute).detach(),
        student_local_contrast=torch.stack(student_contrast).detach(),
        teacher_local_contrast=torch.stack(teacher_contrast).detach(),
        candidate_count=len(cores),
    )


__all__ = [
    "LocalContrastConsistencyOutput",
    "candidate_local_contrast_consistency",
]
