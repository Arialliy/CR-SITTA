"""Pure label-free teacher/student signal measurements for Stage-C0.

This module deliberately accepts only a detached teacher probability and a
student logit tensor.  It has no dataset, split, target, corruption-family, or
severity argument, so the signal definition cannot accidentally become an
outer-oracle or condition-routed objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor


class StageC0GapError(ValueError):
    """The teacher/student tensors violate the Stage-C0 audit contract."""


@dataclass(frozen=True, slots=True)
class TeacherStudentGap:
    """JSON-safe sufficient statistics for one teacher/student pair."""

    teacher_student_gap_l1: float
    teacher_student_logit_gap_mean: float
    teacher_student_logit_gap_max: float
    teacher_entropy_mean: float
    student_entropy_mean: float
    student_less_certain_pixel_fraction: float
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probability_eps(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("eps must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 0.5:
        raise StageC0GapError("eps must be finite and lie in (0, 0.5)")
    return result


def _validate(
    teacher_probability: Tensor,
    student_logits: Tensor,
) -> None:
    for value, name in (
        (teacher_probability, "teacher_probability"),
        (student_logits, "student_logits"),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 4 or value.shape[1] != 1 or value.numel() == 0:
            raise StageC0GapError(f"{name} must have non-empty BCHW layout")
        if not value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must be real floating-point")
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise StageC0GapError(f"{name} contains NaN or Inf")
    if (
        teacher_probability.shape != student_logits.shape
        or teacher_probability.dtype != student_logits.dtype
        or teacher_probability.device != student_logits.device
    ):
        raise StageC0GapError(
            "teacher_probability and student_logits must have identical layout"
        )
    if teacher_probability.requires_grad or teacher_probability.grad_fn is not None:
        raise StageC0GapError("teacher_probability must be detached")
    if bool(
        (
            (teacher_probability < 0.0)
            | (teacher_probability > 1.0)
        ).any().item()
    ):
        raise StageC0GapError("teacher_probability must lie in [0, 1]")


def _entropy(probability: Tensor, eps: float) -> Tensor:
    bounded = probability.clamp(eps, 1.0 - eps)
    return -(
        bounded * torch.log(bounded)
        + (1.0 - bounded) * torch.log1p(-bounded)
    )


def measure_teacher_student_gap(
    teacher_probability: Tensor,
    student_logits: Tensor,
    *,
    eps: float = 1.0e-6,
) -> TeacherStudentGap:
    """Measure a detached teacher against one differentiable student output."""

    _validate(teacher_probability, student_logits)
    checked_eps = _probability_eps(eps)
    with torch.no_grad():
        teacher = teacher_probability.detach()
        student = torch.sigmoid(student_logits.detach())
        teacher_logits = torch.logit(
            teacher.clamp(checked_eps, 1.0 - checked_eps)
        )
        absolute_logit_gap = (student_logits.detach() - teacher_logits).abs()
        teacher_entropy = _entropy(teacher, checked_eps)
        student_entropy = _entropy(student, checked_eps)
        values = {
            "teacher_student_gap_l1": float((student - teacher).abs().mean().item()),
            "teacher_student_logit_gap_mean": float(
                absolute_logit_gap.mean().item()
            ),
            "teacher_student_logit_gap_max": float(
                absolute_logit_gap.amax().item()
            ),
            "teacher_entropy_mean": float(teacher_entropy.mean().item()),
            "student_entropy_mean": float(student_entropy.mean().item()),
            "student_less_certain_pixel_fraction": float(
                (student_entropy > teacher_entropy).to(torch.float64).mean().item()
            ),
        }
    finite = all(math.isfinite(value) for value in values.values())
    if not finite:
        raise StageC0GapError("gap measurement produced NaN or Inf")
    return TeacherStudentGap(**values, finite=True)


__all__ = [
    "StageC0GapError",
    "TeacherStudentGap",
    "measure_teacher_student_gap",
]
