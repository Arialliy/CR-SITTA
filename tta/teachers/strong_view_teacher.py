"""Strong original/geometric teacher for Stage-C ASB-SFR.

Only the immutable geometric-view registry is accepted.  Every teacher,
uncertainty, region-weight, and candidate tensor is detached before return;
the public interface deliberately has no ground-truth, split, or benchmark
condition input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from metrics.connected_components import label_connected_components
from tta.proposals.source_multiview_teacher import frozen_probability_forward
from tta.views.irstd_views import (
    build_detached_region_weights,
    validated_weak_views,
)


StrongAggregation = Literal["median", "trimmed_mean"]


@dataclass(frozen=True)
class StrongTeacherOutput:
    """Detached label-free evidence emitted by the strong teacher."""

    probability: Tensor
    logits: Tensor
    view_variance: Tensor
    stability_mask: Tensor
    local_contrast_map: Tensor
    target_weight: Tensor
    background_weight: Tensor
    candidate_core: tuple[Tensor, ...]
    candidate_ring: tuple[Tensor, ...]
    guard_union: Tensor
    view_names: tuple[str, ...]
    aggregation: StrongAggregation


def _finite_real(
    value: Real,
    *,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    lower_ok = result >= lower if lower_inclusive else result > lower
    upper_ok = result <= upper if upper_inclusive else result < upper
    if not lower_ok or not upper_ok:
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _positive_integer(value: int, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_image(image: Tensor) -> None:
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] <= 0:
        raise ValueError("Stage-C strong teacher requires shape [1,C,H,W]")
    if min(int(image.shape[-2]), int(image.shape[-1])) <= 0:
        raise ValueError("image spatial dimensions must be positive")
    if not image.is_floating_point() or image.is_complex():
        raise TypeError("image must be a real floating-point tensor")
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError("image must contain only finite values")


def _aggregate(
    probabilities: Tensor,
    *,
    method: StrongAggregation,
    trim_each_side: int,
) -> Tensor:
    if method == "median":
        # ``torch.median(dim=...)`` dispatches to a CUDA value+index kernel
        # that PyTorch marks nondeterministic even when only values are used.
        # Sorting and selecting the lower middle order statistic is the exact
        # same even-view convention and remains usable under the formal strict
        # determinism policy.
        ordered = probabilities.sort(dim=0, stable=True).values
        return ordered[(int(probabilities.shape[0]) - 1) // 2]
    if method != "trimmed_mean":
        raise ValueError("aggregation must be exactly 'median' or 'trimmed_mean'")
    trim = _positive_integer(
        trim_each_side, name="trim_each_side", allow_zero=True
    )
    if probabilities.shape[0] - 2 * trim <= 0:
        raise ValueError("trim_each_side removes every teacher view")
    ordered = probabilities.sort(dim=0).values
    selected = ordered[trim : probabilities.shape[0] - trim] if trim else ordered
    return selected.mean(dim=0)


def _dilate(mask: Tensor, radius: int) -> Tensor:
    if radius == 0:
        return mask
    return F.max_pool2d(
        mask.to(dtype=torch.float32),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ).bool()


def _candidate_masks(
    probability: Tensor,
    *,
    threshold: float,
    min_area: int,
    ring_inner_radius: int,
    ring_outer_radius: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    # Match the repository's frozen detector rule: probability strictly
    # greater than 0.5 is positive (equality remains background).
    binary = (probability[0, 0] > threshold).detach().cpu().numpy()
    extraction = label_connected_components(
        binary, connectivity=2, min_area=min_area
    )
    cores: list[Tensor] = []
    for component in extraction.components:
        core_2d = torch.as_tensor(
            extraction.labels == component.label,
            device=probability.device,
            dtype=torch.bool,
        )
        core = core_2d.reshape(1, 1, *core_2d.shape)
        cores.append(core.detach())
    if not cores:
        return (), ()
    all_candidate_union = torch.stack(cores, dim=0).any(dim=0)
    protected_union = _dilate(all_candidate_union, ring_inner_radius)
    rings: list[Tensor] = []
    for core in cores:
        outer = _dilate(core, ring_outer_radius)
        # A candidate's background ring may not use another Source-predicted
        # candidate (or its guard band) as background evidence.
        ring = outer & ~protected_union
        rings.append(ring.detach())
    return tuple(cores), tuple(rings)


def build_strong_teacher_from_aligned_probabilities(
    aligned_probabilities: Tensor,
    view_names: Sequence[str],
    *,
    expected_view_names: Sequence[str] | None = None,
    aggregation: StrongAggregation = "median",
    trim_each_side: int = 1,
    probability_eps: float = 1.0e-6,
    variance_threshold: float = 0.01,
    target_threshold: float = 0.10,
    background_threshold: float = 0.01,
    uncertainty_temperature: float = 0.01,
    protection_radius: int = 3,
    candidate_threshold: float = 0.50,
    candidate_min_area: int = 1,
    ring_inner_radius: int = 3,
    ring_outer_radius: int = 7,
    local_contrast_radius: int = 3,
) -> StrongTeacherOutput:
    """Build a strong teacher from a sealed, source-aligned view stack.

    ``aligned_probabilities`` has layout ``[V,1,1,H,W]``.  This path lets a
    formal runner consume an already verified label-free teacher cache without
    repeating model inference.  When ``expected_view_names`` is supplied, it is
    an exact ordered contract and must include ``identity``.  This validates the
    caller's declared view layout only; it does not establish cache provenance,
    which remains the runner's manifest/hash responsibility.
    """

    if not isinstance(aligned_probabilities, Tensor):
        raise TypeError("aligned_probabilities must be a torch.Tensor")
    if (
        aligned_probabilities.ndim != 5
        or aligned_probabilities.shape[0] <= 0
        or tuple(aligned_probabilities.shape[1:3]) != (1, 1)
        or min(int(value) for value in aligned_probabilities.shape[-2:]) <= 0
    ):
        raise ValueError(
            "aligned_probabilities must have shape [V,1,1,H,W]"
        )
    if not aligned_probabilities.is_floating_point() or aligned_probabilities.is_complex():
        raise TypeError("aligned_probabilities must be real floating-point")
    if not bool(torch.isfinite(aligned_probabilities).all().item()):
        raise ValueError("aligned_probabilities must contain only finite values")
    if bool(
        (
            (aligned_probabilities < 0.0)
            | (aligned_probabilities > 1.0)
        ).any().item()
    ):
        raise ValueError("aligned_probabilities must lie in [0,1]")
    if isinstance(view_names, (str, bytes)):
        raise TypeError("view_names must be a sequence of strings")
    checked_view_names = tuple(view_names)
    if (
        len(checked_view_names) != int(aligned_probabilities.shape[0])
        or len(set(checked_view_names)) != len(checked_view_names)
        or any(not isinstance(name, str) or not name for name in checked_view_names)
    ):
        raise ValueError("view_names must uniquely identify every aligned view")
    if expected_view_names is not None:
        if isinstance(expected_view_names, (str, bytes)):
            raise TypeError("expected_view_names must be a sequence of strings")
        checked_expected_names = tuple(expected_view_names)
        if (
            not checked_expected_names
            or len(set(checked_expected_names)) != len(checked_expected_names)
            or any(
                not isinstance(name, str) or not name
                for name in checked_expected_names
            )
        ):
            raise ValueError(
                "expected_view_names must be a non-empty sequence of unique names"
            )
        if "identity" not in checked_expected_names:
            raise ValueError("expected_view_names must include identity")
        if checked_view_names != checked_expected_names:
            raise ValueError(
                "view_names must exactly match expected_view_names in order"
            )
    if aggregation not in ("median", "trimmed_mean"):
        raise ValueError("aggregation must be exactly 'median' or 'trimmed_mean'")
    checked_eps = _finite_real(
        probability_eps,
        name="probability_eps",
        lower=0.0,
        upper=0.5,
        lower_inclusive=False,
        upper_inclusive=False,
    )
    checked_variance = _finite_real(
        variance_threshold,
        name="variance_threshold",
        lower=0.0,
        upper=1.0,
    )
    checked_target = _finite_real(
        target_threshold, name="target_threshold", lower=0.0, upper=1.0
    )
    checked_background = _finite_real(
        background_threshold,
        name="background_threshold",
        lower=0.0,
        upper=1.0,
    )
    if checked_background >= checked_target:
        raise ValueError("background_threshold must be below target_threshold")
    checked_temperature = _finite_real(
        uncertainty_temperature,
        name="uncertainty_temperature",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    checked_candidate = _finite_real(
        candidate_threshold,
        name="candidate_threshold",
        lower=0.0,
        upper=1.0,
    )
    checked_protection = _positive_integer(
        protection_radius, name="protection_radius", allow_zero=True
    )
    checked_min_area = _positive_integer(
        candidate_min_area, name="candidate_min_area"
    )
    checked_inner = _positive_integer(
        ring_inner_radius, name="ring_inner_radius", allow_zero=True
    )
    checked_outer = _positive_integer(
        ring_outer_radius, name="ring_outer_radius"
    )
    if checked_outer <= checked_inner:
        raise ValueError("ring_outer_radius must exceed ring_inner_radius")
    checked_contrast = _positive_integer(
        local_contrast_radius, name="local_contrast_radius"
    )

    with torch.no_grad():
        stacked = aligned_probabilities.detach()
        teacher = _aggregate(
            stacked,
            method=aggregation,
            trim_each_side=trim_each_side,
        ).clamp(0.0, 1.0)
        variance = stacked.var(dim=0, unbiased=False)
        logits = torch.logit(teacher.clamp(checked_eps, 1.0 - checked_eps))
        regions = build_detached_region_weights(
            teacher,
            variance,
            tau_background=checked_background,
            tau_foreground=checked_target,
            temperature_foreground=checked_temperature,
            temperature_background=checked_temperature,
            protection_radius=checked_protection,
        )
        stability = (variance < checked_variance).detach()
        kernel = 2 * checked_contrast + 1
        local_mean = F.avg_pool2d(
            teacher,
            kernel_size=kernel,
            stride=1,
            padding=checked_contrast,
            count_include_pad=False,
        )
        contrast = teacher - local_mean
        cores, rings = _candidate_masks(
            teacher,
            threshold=checked_candidate,
            min_area=checked_min_area,
            ring_inner_radius=checked_inner,
            ring_outer_radius=checked_outer,
        )

    output = StrongTeacherOutput(
        probability=teacher.detach(),
        logits=logits.detach(),
        view_variance=variance.detach(),
        stability_mask=stability.detach(),
        local_contrast_map=contrast.detach(),
        target_weight=regions.foreground_weight.detach(),
        background_weight=regions.background_weight.detach(),
        candidate_core=cores,
        candidate_ring=rings,
        guard_union=regions.protection_band.bool().detach(),
        view_names=checked_view_names,
        aggregation=aggregation,
    )
    tensor_fields = (
        output.probability,
        output.logits,
        output.view_variance,
        output.stability_mask,
        output.local_contrast_map,
        output.target_weight,
        output.background_weight,
        output.guard_union,
        *output.candidate_core,
        *output.candidate_ring,
    )
    if any(value.requires_grad or value.grad_fn is not None for value in tensor_fields):
        raise RuntimeError("strong teacher returned an attached tensor")
    if not all(bool(torch.isfinite(value).all().item()) for value in tensor_fields):
        raise RuntimeError("strong teacher returned a non-finite tensor")
    return output


def build_strong_teacher(
    predictor: object,
    image: Tensor,
    view_set: Sequence[str],
    *,
    aggregation: StrongAggregation = "median",
    trim_each_side: int = 1,
    probability_eps: float = 1.0e-6,
    variance_threshold: float = 0.01,
    target_threshold: float = 0.10,
    background_threshold: float = 0.01,
    uncertainty_temperature: float = 0.01,
    protection_radius: int = 3,
    candidate_threshold: float = 0.50,
    candidate_min_area: int = 1,
    ring_inner_radius: int = 3,
    ring_outer_radius: int = 7,
    local_contrast_radius: int = 3,
) -> StrongTeacherOutput:
    """Build a strong teacher using only immutable geometric views."""

    _validate_image(image)
    views = validated_weak_views(view_set)
    probabilities: list[Tensor] = []
    with torch.no_grad():
        for view in views:
            transformed = view.forward(image)
            probability = frozen_probability_forward(predictor, transformed)
            probabilities.append(view.inverse_prediction(probability).detach())
    return build_strong_teacher_from_aligned_probabilities(
        torch.stack(probabilities, dim=0),
        tuple(view.name for view in views),
        aggregation=aggregation,
        trim_each_side=trim_each_side,
        probability_eps=probability_eps,
        variance_threshold=variance_threshold,
        target_threshold=target_threshold,
        background_threshold=background_threshold,
        uncertainty_temperature=uncertainty_temperature,
        protection_radius=protection_radius,
        candidate_threshold=candidate_threshold,
        candidate_min_area=candidate_min_area,
        ring_inner_radius=ring_inner_radius,
        ring_outer_radius=ring_outer_radius,
        local_contrast_radius=local_contrast_radius,
    )


__all__ = [
    "StrongAggregation",
    "StrongTeacherOutput",
    "build_strong_teacher",
    "build_strong_teacher_from_aligned_probabilities",
]
