"""Explicit ImageNet-normalized/physical image-space conversions."""

from __future__ import annotations

import torch
from torch import Tensor

from .fourier_low_mask import DeteriorationContractError, validate_physical_image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _statistics(reference: Tensor) -> tuple[Tensor, Tensor]:
    if reference.ndim != 4 or reference.shape[1] != 3:
        raise DeteriorationContractError(
            "Stage-C NS-FPN image must have shape [B,3,H,W]"
        )
    mean = reference.new_tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = reference.new_tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    return mean, std


def imagenet_normalize(physical_image: Tensor) -> Tensor:
    """Normalize a finite physical RGB image for the frozen NS-FPN."""

    validate_physical_image(physical_image)
    mean, std = _statistics(physical_image)
    result = (physical_image - mean) / std
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("ImageNet normalization produced NaN or Inf")
    return result


def imagenet_denormalize(
    normalized_image: Tensor,
    *,
    range_tolerance: float = 2.0e-6,
) -> Tensor:
    """Recover physical RGB values and reject preprocessing drift."""

    if not isinstance(normalized_image, Tensor):
        raise TypeError("normalized_image must be a torch.Tensor")
    if not normalized_image.is_floating_point() or normalized_image.is_complex():
        raise TypeError("normalized_image must be real floating-point")
    mean, std = _statistics(normalized_image)
    if not bool(torch.isfinite(normalized_image).all().item()):
        raise DeteriorationContractError("normalized_image contains NaN or Inf")
    if not isinstance(range_tolerance, (int, float)) or isinstance(
        range_tolerance, bool
    ):
        raise TypeError("range_tolerance must be a real number")
    tolerance = float(range_tolerance)
    if not 0.0 <= tolerance < 0.01:
        raise DeteriorationContractError(
            "range_tolerance must lie in [0,0.01)"
        )
    physical = normalized_image * std + mean
    if bool(
        ((physical < -tolerance) | (physical > 1.0 + tolerance)).any().item()
    ):
        raise DeteriorationContractError(
            "denormalized image leaves physical [0,1] range"
        )
    # The cache was built from clipped physical values.  Clamping only removes
    # inverse-normalization round-off at the two endpoints.
    return physical.clamp(0.0, 1.0)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "imagenet_denormalize",
    "imagenet_normalize",
]
