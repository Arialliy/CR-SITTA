"""Pure label-free active-support accounting for Stage-C0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral
from typing import Any

import torch
from torch import Tensor


class StageC0ActiveSupportError(ValueError):
    """Active-support inputs violate the frozen label-free contract."""


@dataclass(frozen=True, slots=True)
class ActiveSupportAudit:
    active_episode: bool
    active_pixel_count: int
    active_pixel_fraction: float
    active_target_weight: float
    active_background_weight: float
    active_near_candidate_pixel_count: int
    active_near_candidate_fraction: float
    active_far_background_pixel_count: int
    active_only_far_background: bool
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_mask(value: Tensor, reference: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != reference.shape or value.device != reference.device:
        raise StageC0ActiveSupportError(f"{name} layout differs")
    if value.requires_grad or value.grad_fn is not None:
        raise StageC0ActiveSupportError(f"{name} must be detached")
    if value.dtype == torch.bool:
        return
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must be boolean or real floating-point")
    if not bool(torch.isfinite(value).all().item()):
        raise StageC0ActiveSupportError(f"{name} contains NaN or Inf")
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise StageC0ActiveSupportError(f"{name} must lie in [0, 1]")


def audit_active_support(
    active_mask: Tensor,
    target_weight: Tensor,
    background_weight: Tensor,
    near_candidate_mask: Tensor,
    *,
    min_active_pixels: int = 1,
) -> ActiveSupportAudit:
    """Account for activity without consulting a ground-truth mask.

    ``near_candidate_mask`` is derived from the strong teacher's candidate
    core/ring/guard metadata.  Consequently ``active_only_far_background`` is
    a label-free diagnostic and must not be interpreted as a GT statement.
    """

    if not isinstance(active_mask, Tensor):
        raise TypeError("active_mask must be a torch.Tensor")
    if active_mask.ndim != 4 or active_mask.shape[1] != 1 or active_mask.numel() == 0:
        raise StageC0ActiveSupportError("active_mask must have non-empty BCHW layout")
    if active_mask.requires_grad or active_mask.grad_fn is not None:
        raise StageC0ActiveSupportError("active_mask must be detached")
    _validate_mask(target_weight, active_mask, name="target_weight")
    _validate_mask(background_weight, active_mask, name="background_weight")
    _validate_mask(near_candidate_mask, active_mask, name="near_candidate_mask")
    if isinstance(min_active_pixels, bool) or not isinstance(
        min_active_pixels, Integral
    ):
        raise TypeError("min_active_pixels must be an integer")
    minimum = int(min_active_pixels)
    if minimum <= 0:
        raise StageC0ActiveSupportError("min_active_pixels must be positive")

    if active_mask.dtype == torch.bool:
        active_bool = active_mask
    elif active_mask.is_floating_point() and not active_mask.is_complex():
        if not bool(torch.isfinite(active_mask).all().item()):
            raise StageC0ActiveSupportError("active_mask contains NaN or Inf")
        if bool(((active_mask < 0.0) | (active_mask > 1.0)).any().item()):
            raise StageC0ActiveSupportError("active_mask must lie in [0, 1]")
        active_bool = active_mask > 0.0
    else:
        raise TypeError("active_mask must be boolean or real floating-point")

    active_count = int(active_bool.sum().item())
    near_bool = near_candidate_mask.bool()
    near_count = int((active_bool & near_bool).sum().item())
    far_count = active_count - near_count
    active_episode = active_count >= minimum
    denominator = max(active_count, 1)
    active_float = active_bool.to(dtype=target_weight.dtype)
    values = {
        "active_pixel_fraction": float(
            active_bool.to(torch.float64).mean().item()
        ),
        "active_target_weight": float(
            (active_float * target_weight).sum().item()
        ),
        "active_background_weight": float(
            (active_float * background_weight).sum().item()
        ),
        "active_near_candidate_fraction": float(near_count / denominator),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise StageC0ActiveSupportError("active-support audit produced NaN or Inf")
    return ActiveSupportAudit(
        active_episode=active_episode,
        active_pixel_count=active_count,
        active_pixel_fraction=values["active_pixel_fraction"],
        active_target_weight=values["active_target_weight"],
        active_background_weight=values["active_background_weight"],
        active_near_candidate_pixel_count=near_count,
        active_near_candidate_fraction=values["active_near_candidate_fraction"],
        active_far_background_pixel_count=far_count,
        active_only_far_background=bool(active_episode and near_count == 0),
        finite=True,
    )


__all__ = [
    "ActiveSupportAudit",
    "StageC0ActiveSupportError",
    "audit_active_support",
]
