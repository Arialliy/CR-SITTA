"""Label-free view and teacher-region primitives for Stage-B3.

This module deliberately has no target/GT interface.  Geometric views are
selected from an immutable whitelist and have exact spatial inverses.  The
photometric student transform is deterministic and is performed in physical
``[0, 1]`` intensity space after undoing ImageNet normalization.

Region weights are derived exclusively from detached teacher mean and
population-variance tensors.  They therefore cannot create a gradient path
back into the frozen teacher.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
NORMALIZED_RGB_INPUT_CONTRACT = (
    "BCHW float RGB obtained by channel-wise ImageNet normalization of "
    "finite physical intensities in [0, 1]"
)
MILD_CONTRAST_FACTOR = 0.95


def _validate_bchw(tensor: Tensor, *, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W]")
    if any(int(size) <= 0 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must all be positive")


def _identity(tensor: Tensor) -> Tensor:
    _validate_bchw(tensor, name="tensor")
    return tensor


def _hflip(tensor: Tensor) -> Tensor:
    _validate_bchw(tensor, name="tensor")
    return torch.flip(tensor, dims=(-1,))


def _vflip(tensor: Tensor) -> Tensor:
    _validate_bchw(tensor, name="tensor")
    return torch.flip(tensor, dims=(-2,))


def _hvflip(tensor: Tensor) -> Tensor:
    _validate_bchw(tensor, name="tensor")
    return torch.flip(tensor, dims=(-2, -1))


@dataclass(frozen=True)
class InvertibleView:
    """A fixed spatial view and the corresponding prediction-space inverse."""

    name: str
    forward: Callable[[Tensor], Tensor]
    inverse_prediction: Callable[[Tensor], Tensor]


_INVERTIBLE_VIEW_REGISTRY = {
    "identity": InvertibleView("identity", _identity, _identity),
    "hflip": InvertibleView("hflip", _hflip, _hflip),
    "vflip": InvertibleView("vflip", _vflip, _vflip),
    "hvflip": InvertibleView("hvflip", _hvflip, _hvflip),
}

# MappingProxyType makes accidental runtime registry edits fail immediately.
INVERTIBLE_VIEW_REGISTRY: Mapping[str, InvertibleView] = MappingProxyType(
    _INVERTIBLE_VIEW_REGISTRY
)
GEOMETRIC_VIEW_NAMES: tuple[str, ...] = tuple(INVERTIBLE_VIEW_REGISTRY)


def _validate_names(names: Sequence[str], *, kind: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise TypeError(f"{kind} names must be a sequence of strings, not a string")
    checked = tuple(names)
    if not checked:
        raise ValueError(f"at least one {kind} must be selected")
    if any(not isinstance(name, str) for name in checked):
        raise TypeError(f"every {kind} name must be a string")
    return checked


def validated_weak_views(names: Sequence[str]) -> tuple[InvertibleView, ...]:
    """Resolve geometric views from the immutable Stage-B whitelist.

    Requested order is preserved.  Unknown names fail closed; there is no
    dynamic transform construction or random parameter sampling here.
    """

    checked = _validate_names(names, kind="weak view")
    unknown = tuple(name for name in checked if name not in INVERTIBLE_VIEW_REGISTRY)
    if unknown:
        allowed = ", ".join(GEOMETRIC_VIEW_NAMES)
        raise ValueError(
            f"unknown weak view name(s) {unknown!r}; allowed views are {allowed}"
        )
    return tuple(INVERTIBLE_VIEW_REGISTRY[name] for name in checked)


def _imagenet_statistics(image: Tensor) -> tuple[Tensor, Tensor]:
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return mean, std


def _validate_normalized_rgb(image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    _validate_bchw(image, name="image")
    if not image.is_floating_point():
        raise TypeError("image must have a floating-point dtype")
    if image.shape[1] != 3:
        raise ValueError(
            "student photometric perturbations require three RGB channels"
        )
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError("image must contain only finite values")

    mean, std = _imagenet_statistics(image)
    physical = image * std + mean
    tolerance = max(1e-5, 8.0 * torch.finfo(image.dtype).eps)
    if bool((physical < -tolerance).any().item()) or bool(
        (physical > 1.0 + tolerance).any().item()
    ):
        raise ValueError(
            "image violates the normalized-input contract: expected ImageNet "
            "normalization of physical RGB intensities in [0, 1]"
        )
    return physical.clamp(0.0, 1.0), mean, std


def _normalized_identity(image: Tensor) -> Tensor:
    # Validate the same explicit domain contract as every other student view.
    _validate_normalized_rgb(image)
    return image


def apply_fixed_mild_contrast(image: Tensor) -> Tensor:
    """Apply a fixed 0.95 contrast transform in physical intensity space.

    The input must satisfy :data:`NORMALIZED_RGB_INPUT_CONTRACT`.  Contrast is
    measured around physical mid-gray (0.5), clipped to ``[0, 1]``, and then
    re-normalized.  No random-number generator is read or mutated.  The same
    physical transform is applied to all channels, so replicated grayscale
    RGB inputs remain channel-identical after de-normalization.
    """

    physical, mean, std = _validate_normalized_rgb(image)
    transformed = (physical - 0.5) * MILD_CONTRAST_FACTOR + 0.5
    transformed = transformed.clamp(0.0, 1.0)
    return (transformed - mean) / std


@dataclass(frozen=True)
class StudentPerturbation:
    """A deterministic, corruption-agnostic student-input perturbation."""

    name: str
    forward: Callable[[Tensor], Tensor]
    input_contract: str = NORMALIZED_RGB_INPUT_CONTRACT


_STUDENT_PERTURBATION_REGISTRY = {
    "identity": StudentPerturbation("identity", _normalized_identity),
    "mild_contrast_0p95": StudentPerturbation(
        "mild_contrast_0p95", apply_fixed_mild_contrast
    ),
}
STUDENT_PERTURBATION_REGISTRY: Mapping[str, StudentPerturbation] = (
    MappingProxyType(_STUDENT_PERTURBATION_REGISTRY)
)


def validated_student_perturbations(
    names: Sequence[str],
) -> tuple[StudentPerturbation, ...]:
    """Resolve deterministic student perturbations from the frozen whitelist."""

    checked = _validate_names(names, kind="student perturbation")
    unknown = tuple(
        name for name in checked if name not in STUDENT_PERTURBATION_REGISTRY
    )
    if unknown:
        allowed = ", ".join(STUDENT_PERTURBATION_REGISTRY)
        raise ValueError(
            f"unknown student perturbation name(s) {unknown!r}; "
            f"allowed perturbations are {allowed}"
        )
    return tuple(STUDENT_PERTURBATION_REGISTRY[name] for name in checked)


@dataclass(frozen=True)
class DetachedRegionWeights:
    """Teacher-derived, gradient-free Stage-B foreground/background regions."""

    foreground_weight: Tensor
    background_weight: Tensor
    protection_band: Tensor
    foreground_indicator: Tensor
    background_indicator: Tensor

    @property
    def foreground(self) -> Tensor:
        """Alias used by objective code that expects a concise field name."""

        return self.foreground_weight

    @property
    def background(self) -> Tensor:
        """Alias used by objective code that expects a concise field name."""

        return self.background_weight


def _validate_teacher_statistics(
    teacher_mean: Tensor, teacher_variance: Tensor
) -> None:
    _validate_bchw(teacher_mean, name="teacher_mean")
    _validate_bchw(teacher_variance, name="teacher_variance")
    if teacher_mean.shape != teacher_variance.shape:
        raise ValueError("teacher_mean and teacher_variance must have identical shapes")
    if teacher_mean.shape[1] != 1:
        raise ValueError("teacher statistics must have exactly one probability channel")
    if not teacher_mean.is_floating_point() or not teacher_variance.is_floating_point():
        raise TypeError("teacher statistics must have floating-point dtypes")
    if teacher_mean.dtype != teacher_variance.dtype:
        raise ValueError("teacher statistics must have the same dtype")
    if teacher_mean.device != teacher_variance.device:
        raise ValueError("teacher statistics must be on the same device")
    if not bool(torch.isfinite(teacher_mean).all().item()) or not bool(
        torch.isfinite(teacher_variance).all().item()
    ):
        raise ValueError("teacher statistics must contain only finite values")
    if bool((teacher_mean < 0).any().item()) or bool(
        (teacher_mean > 1).any().item()
    ):
        raise ValueError("teacher_mean must lie in [0, 1]")
    if bool((teacher_variance < 0).any().item()):
        raise ValueError("teacher_variance must be non-negative")


def _finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _dilate_indicator(indicator: Tensor, *, radius: int) -> Tensor:
    if radius == 0:
        return indicator
    kernel_size = 2 * radius + 1
    return F.max_pool2d(
        indicator,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )


def build_detached_region_weights(
    teacher_mean: Tensor,
    teacher_variance: Tensor,
    *,
    tau_background: float,
    tau_foreground: float,
    gamma_foreground: float = 1.0,
    gamma_background: float = 1.0,
    temperature_foreground: float = 0.05,
    temperature_background: float = 0.05,
    protection_radius: int = 1,
) -> DetachedRegionWeights:
    """Build detached foreground/background weights from teacher statistics.

    The foreground candidate set is ``teacher_mean >= tau_foreground`` and the
    reliable-background set starts as ``teacher_mean <= tau_background``.
    A square Chebyshev-radius dilation of the foreground set forms the
    protection band, which is removed from reliable background.  Consequently
    ``tau_background < tau_foreground`` leaves an ignored uncertainty interval
    and nearby low-valued pixels cannot be reinforced as background.

    No ground truth, target, prediction mask, corruption name, or test label is
    accepted by this API.
    """

    _validate_teacher_statistics(teacher_mean, teacher_variance)
    tau_b = _finite_real(tau_background, name="tau_background")
    tau_f = _finite_real(tau_foreground, name="tau_foreground")
    gamma_f = _finite_real(gamma_foreground, name="gamma_foreground")
    gamma_b = _finite_real(gamma_background, name="gamma_background")
    temperature_f = _finite_real(
        temperature_foreground, name="temperature_foreground"
    )
    temperature_b = _finite_real(
        temperature_background, name="temperature_background"
    )

    if not 0.0 <= tau_b <= 1.0 or not 0.0 <= tau_f <= 1.0:
        raise ValueError("tau_background and tau_foreground must lie in [0, 1]")
    if not tau_b < tau_f:
        raise ValueError("tau_background must be strictly less than tau_foreground")
    if gamma_f < 0.0 or gamma_b < 0.0:
        raise ValueError("region-weight exponents must be non-negative")
    if temperature_f <= 0.0 or temperature_b <= 0.0:
        raise ValueError("uncertainty temperatures must be greater than zero")
    if isinstance(protection_radius, bool) or not isinstance(protection_radius, int):
        raise TypeError("protection_radius must be an integer")
    if protection_radius < 0:
        raise ValueError("protection_radius must be non-negative")

    # Detach before every operation and stay under no_grad so even callers that
    # pass differentiable teacher tensors receive weights with no grad_fn.
    with torch.no_grad():
        mean = teacher_mean.detach()
        variance = teacher_variance.detach()
        foreground_indicator = (mean >= tau_f).to(dtype=mean.dtype)
        initial_background = (mean <= tau_b).to(dtype=mean.dtype)
        protection_band = _dilate_indicator(
            foreground_indicator, radius=protection_radius
        ).clamp(0.0, 1.0)
        background_indicator = initial_background * (1.0 - protection_band)

        foreground_weight = (
            foreground_indicator
            * mean.pow(gamma_f)
            * torch.exp(-variance / temperature_f)
        )
        background_weight = (
            background_indicator
            * (1.0 - mean).pow(gamma_b)
            * torch.exp(-variance / temperature_b)
        )

        return DetachedRegionWeights(
            foreground_weight=foreground_weight.detach(),
            background_weight=background_weight.detach(),
            protection_band=protection_band.detach(),
            foreground_indicator=foreground_indicator.detach(),
            background_indicator=background_indicator.detach(),
        )


__all__ = [
    "DetachedRegionWeights",
    "GEOMETRIC_VIEW_NAMES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INVERTIBLE_VIEW_REGISTRY",
    "InvertibleView",
    "MILD_CONTRAST_FACTOR",
    "NORMALIZED_RGB_INPUT_CONTRACT",
    "STUDENT_PERTURBATION_REGISTRY",
    "StudentPerturbation",
    "apply_fixed_mild_contrast",
    "build_detached_region_weights",
    "validated_student_perturbations",
    "validated_weak_views",
]
