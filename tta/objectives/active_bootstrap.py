"""Active, region-balanced binary self-bootstrapping for Stage-C.

The active set is computed only from detached teacher/student uncertainty and
an explicitly supplied detached stability mask.  Ground-truth masks and
benchmark condition labels are intentionally absent from the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

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


@dataclass(frozen=True)
class ActiveBootstrapOutput:
    """Loss components and detached activity evidence for one probe."""

    total: Tensor
    foreground: Tensor
    background: Tensor
    active_fraction: Tensor
    active_foreground_weight: Tensor
    active_background_weight: Tensor
    active_pixel_count: int
    has_active_support: bool
    active_mask: Tensor


def bernoulli_probability_entropy(
    probability: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    """Return the elementwise Bernoulli entropy of a probability tensor."""

    validate_probability(probability, name="probability")
    checked_eps = probability_eps(eps)
    working = probability
    if probability.dtype in (torch.float16, torch.bfloat16):
        working = probability.to(dtype=torch.float32)
    bounded = working.clamp(checked_eps, 1.0 - checked_eps)
    entropy = -(
        bounded * torch.log(bounded)
        + (1.0 - bounded) * torch.log1p(-bounded)
    )
    if not bool(torch.isfinite(entropy).all().detach().item()):
        raise StageBObjectiveError("Bernoulli entropy produced NaN or Inf")
    return entropy.to(dtype=probability.dtype)


def _validate_stable_mask(stable_mask: Tensor, reference: Tensor) -> None:
    if not isinstance(stable_mask, Tensor):
        raise TypeError("stable_mask must be a torch.Tensor")
    if stable_mask.shape != reference.shape:
        raise StageBObjectiveError(
            "stable_mask shape must match the student tensor exactly"
        )
    if stable_mask.device != reference.device:
        raise StageBObjectiveError(
            "stable_mask device must match the student tensor exactly"
        )
    require_detached(stable_mask, name="stable_mask")
    if stable_mask.dtype == torch.bool:
        return
    if not stable_mask.is_floating_point() or stable_mask.is_complex():
        raise TypeError("stable_mask must be boolean or real floating-point")
    if stable_mask.dtype != reference.dtype:
        raise StageBObjectiveError(
            "stable_mask dtype must match the student tensor exactly"
        )
    if not bool(torch.isfinite(stable_mask).all().item()):
        raise StageBObjectiveError("stable_mask must contain only finite values")
    if bool(((stable_mask < 0.0) | (stable_mask > 1.0)).any().item()):
        raise StageBObjectiveError("stable_mask values must lie in [0,1]")


def active_bootstrap_binary(
    student_logits: Tensor,
    teacher_probability: Tensor,
    target_weight: Tensor,
    background_weight: Tensor,
    stable_mask: Tensor,
    *,
    entropy_margin: float,
    min_active_pixels: int = 1,
    eps: float = 1.0e-6,
) -> ActiveBootstrapOutput:
    """Bootstrap only where the detached teacher is demonstrably more certain.

    Foreground and reliable-background losses are normalized independently.
    If the active set is smaller than ``min_active_pixels`` or carries no
    target/background weight, an exact graph-connected zero is returned so
    the caller can fail closed without amplifying a tiny denominator.
    """

    validate_spatial_tensor(student_logits, name="student_logits")
    validate_probability(
        teacher_probability,
        name="teacher_probability",
        reference=student_logits,
        detached=True,
    )
    validate_weight(
        target_weight,
        name="target_weight",
        reference=student_logits,
    )
    validate_weight(
        background_weight,
        name="background_weight",
        reference=student_logits,
    )
    _validate_stable_mask(stable_mask, student_logits)
    checked_margin = finite_real(
        entropy_margin,
        name="entropy_margin",
        nonnegative=True,
    )
    checked_eps = probability_eps(eps)
    if isinstance(min_active_pixels, bool) or not isinstance(
        min_active_pixels, Integral
    ):
        raise TypeError("min_active_pixels must be a positive integer")
    checked_min_pixels = int(min_active_pixels)
    if checked_min_pixels < 1:
        raise StageBObjectiveError("min_active_pixels must be positive")

    teacher_entropy = bernoulli_probability_entropy(teacher_probability)
    student_probability = torch.sigmoid(student_logits).detach()
    student_entropy = bernoulli_probability_entropy(student_probability)
    stable = stable_mask.bool()
    active_bool = (
        (teacher_entropy + checked_margin < student_entropy) & stable
    ).detach()
    active_pixel_count = int(active_bool.sum().item())
    if active_pixel_count < checked_min_pixels:
        active_bool = torch.zeros_like(active_bool)
        active_pixel_count = 0
    active = active_bool.to(dtype=student_logits.dtype).detach()
    active_foreground = (active * target_weight.detach()).detach()
    active_background = (active * background_weight.detach()).detach()
    foreground_sum = detached_weight_sum(active_foreground)
    background_sum = detached_weight_sum(active_background)
    has_foreground = has_positive_weight(
        foreground_sum, name="active_foreground_weight"
    )
    has_background = has_positive_weight(
        background_sum, name="active_background_weight"
    )
    has_active_support = (
        active_pixel_count >= checked_min_pixels
        and (has_foreground or has_background)
    )

    pixel_loss = F.binary_cross_entropy_with_logits(
        student_logits,
        teacher_probability,
        reduction="none",
    )
    foreground = (
        weighted_mean(
            pixel_loss,
            active_foreground,
            eps=checked_eps,
            weight_sum=foreground_sum,
        )
        if has_foreground
        else differentiable_zero(student_logits)
    )
    background = (
        weighted_mean(
            pixel_loss,
            active_background,
            eps=checked_eps,
            weight_sum=background_sum,
        )
        if has_background
        else differentiable_zero(student_logits)
    )
    total = foreground + background
    ensure_finite_scalar(total, name="active bootstrap")
    return ActiveBootstrapOutput(
        total=total,
        foreground=foreground,
        background=background,
        active_fraction=active.mean().detach(),
        active_foreground_weight=foreground_sum.detach(),
        active_background_weight=background_sum.detach(),
        active_pixel_count=active_pixel_count,
        has_active_support=has_active_support,
        active_mask=active,
    )


__all__ = [
    "ActiveBootstrapOutput",
    "active_bootstrap_binary",
    "bernoulli_probability_entropy",
]
