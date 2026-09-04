"""Optional source feature-statistics alignment primitive (Stage-B O5)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ._validation import (
    StageBObjectiveError,
    ensure_finite_scalar,
    finite_real,
    probability_eps,
    require_detached,
)


@dataclass(frozen=True)
class SourceFeatureStatistics:
    """A detached per-channel Source mean and standard deviation."""

    mean: Tensor
    std: Tensor


@dataclass(frozen=True)
class FeatureStatisticsAlignmentOutput:
    """Auditable current moments and the two L1 alignment components."""

    total: Tensor
    mean_alignment: Tensor
    log_std_alignment: Tensor
    current_mean: Tensor
    current_std: Tensor


def _validate_feature(feature: Tensor) -> None:
    if not isinstance(feature, Tensor):
        raise TypeError("feature must be a torch.Tensor")
    if feature.ndim != 4:
        raise StageBObjectiveError("feature must have shape [B,C,H,W]")
    if any(int(size) <= 0 for size in feature.shape):
        raise StageBObjectiveError("feature dimensions must all be positive")
    if not torch.is_floating_point(feature) or feature.is_complex():
        raise TypeError("feature must be a real floating-point tensor")
    if not bool(torch.isfinite(feature).all().detach().item()):
        raise StageBObjectiveError("feature must contain only finite values")


def _validate_source_statistic(
    value: Tensor,
    *,
    name: str,
    feature: Tensor,
    strictly_positive: bool,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or value.shape[0] != feature.shape[1]:
        raise StageBObjectiveError(
            f"{name} must have shape [C] matching feature channels"
        )
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError(f"{name} must be real floating-point")
    if value.device != feature.device or value.dtype != feature.dtype:
        raise StageBObjectiveError(
            f"{name} dtype/device must match feature exactly"
        )
    require_detached(value, name=name)
    if not bool(torch.isfinite(value).all().detach().item()):
        raise StageBObjectiveError(f"{name} must contain only finite values")
    if strictly_positive and bool((value <= 0.0).any().detach().item()):
        raise StageBObjectiveError(f"{name} must be strictly positive")


def feature_statistics_alignment_components(
    feature: Tensor,
    source_mean: Tensor,
    source_std: Tensor,
    *,
    eps: float = 1.0e-6,
) -> FeatureStatisticsAlignmentOutput:
    """Align per-channel mean and log-standard-deviation to Source stats."""

    _validate_feature(feature)
    _validate_source_statistic(
        source_mean,
        name="source_mean",
        feature=feature,
        strictly_positive=False,
    )
    _validate_source_statistic(
        source_std,
        name="source_std",
        feature=feature,
        strictly_positive=True,
    )
    checked_eps = probability_eps(eps)
    working_feature = feature
    working_mean = source_mean
    working_std = source_std
    if feature.dtype in (torch.float16, torch.bfloat16):
        working_feature = feature.to(dtype=torch.float32)
        working_mean = source_mean.to(dtype=torch.float32)
        working_std = source_std.to(dtype=torch.float32)

    variance, current_mean = torch.var_mean(
        working_feature,
        dim=(0, 2, 3),
        correction=0,
    )
    # Adding eps^2 keeps the zero-variance gradient finite while remaining a
    # faithful stabilized log-sigma implementation.
    current_std = torch.sqrt(variance + checked_eps * checked_eps)
    mean_alignment = torch.abs(current_mean - working_mean).sum()
    log_std_alignment = torch.abs(
        torch.log(current_std) - torch.log(working_std)
    ).sum()
    total = mean_alignment + log_std_alignment
    for value, name in (
        (current_mean, "current feature mean"),
        (current_std, "current feature std"),
    ):
        if not bool(torch.isfinite(value).all().detach().item()):
            raise StageBObjectiveError(f"{name} must be finite")
    ensure_finite_scalar(mean_alignment, name="feature mean alignment")
    ensure_finite_scalar(log_std_alignment, name="feature log-std alignment")
    ensure_finite_scalar(total, name="feature-statistics alignment")
    return FeatureStatisticsAlignmentOutput(
        total=total,
        mean_alignment=mean_alignment,
        log_std_alignment=log_std_alignment,
        current_mean=current_mean,
        current_std=current_std,
    )


def feature_statistics_alignment(
    feature: Tensor,
    source_mean: Tensor,
    source_std: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    """Return the scalar Source feature-statistics alignment loss."""

    return feature_statistics_alignment_components(
        feature,
        source_mean,
        source_std,
        eps=eps,
    ).total


feature_statistics_alignment_loss = feature_statistics_alignment


def multi_layer_feature_statistics_alignment(
    features: Mapping[str, Tensor],
    source_statistics: Mapping[str, SourceFeatureStatistics],
    *,
    layer_weights: Mapping[str, float] | None = None,
    eps: float = 1.0e-6,
) -> Tensor:
    """Sum optional weighted O5 terms over an exact ordered layer topology."""

    if not isinstance(features, Mapping) or not isinstance(
        source_statistics, Mapping
    ):
        raise TypeError("features and source_statistics must be mappings")
    names = tuple(features)
    if not names:
        raise StageBObjectiveError("feature-statistics alignment needs a layer")
    if names != tuple(source_statistics):
        raise StageBObjectiveError(
            "feature and Source-statistic layer names/order must match exactly"
        )
    if layer_weights is not None:
        if not isinstance(layer_weights, Mapping):
            raise TypeError("layer_weights must be a mapping when provided")
        if names != tuple(layer_weights):
            raise StageBObjectiveError(
                "layer_weights names/order must match features exactly"
            )

    terms: list[Tensor] = []
    any_positive_weight = False
    for name in names:
        source = source_statistics[name]
        if not isinstance(source, SourceFeatureStatistics):
            raise TypeError(
                f"source_statistics[{name!r}] must be SourceFeatureStatistics"
            )
        weight: Any = 1.0 if layer_weights is None else layer_weights[name]
        checked_weight = finite_real(
            weight,
            name=f"layer_weights[{name!r}]",
            nonnegative=True,
        )
        any_positive_weight = any_positive_weight or checked_weight > 0.0
        terms.append(
            checked_weight
            * feature_statistics_alignment(
                features[name], source.mean, source.std, eps=eps
            )
        )
    if not any_positive_weight:
        raise StageBObjectiveError("at least one layer weight must be positive")
    devices = {term.device for term in terms}
    if len(devices) != 1:
        raise StageBObjectiveError(
            "all feature-statistics terms must share one device"
        )
    total = torch.stack(terms).sum()
    ensure_finite_scalar(total, name="multi-layer feature-statistics alignment")
    return total


__all__ = [
    "FeatureStatisticsAlignmentOutput",
    "SourceFeatureStatistics",
    "feature_statistics_alignment",
    "feature_statistics_alignment_components",
    "feature_statistics_alignment_loss",
    "multi_layer_feature_statistics_alignment",
]
