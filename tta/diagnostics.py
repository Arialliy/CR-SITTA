"""Pure diagnostics for single-image episodic adaptation.

The functions in this module do not run an adaptation method and never choose
hyperparameters.  They compare already-produced pre/post states and classify
an episode using the frozen priority

``numeric -> functional -> threshold -> metric -> task_effective``.

Ground truth is used only for outer, source-train-side diagnostic strata and
official sufficient statistics.  Callers are responsible for enforcing that
label boundary; :mod:`analysis.analyze_tent_ss_noop` provides the fail-closed
artifact runner for that purpose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from metrics.connected_components import extract_connected_components
from metrics.irstd_metrics import (
    IRSTDEvaluationProtocol,
    UnifiedResearchEvaluator,
    probabilities_from_logits,
)


NUMERIC_NOOP = "numeric_noop"
FUNCTIONAL_NOOP = "functional_noop"
THRESHOLD_NOOP = "threshold_noop"
METRIC_NOOP = "metric_noop"
TASK_EFFECTIVE_CHANGE = "task_effective_change"
NOOP_CLASSIFICATION_ORDER = (
    NUMERIC_NOOP,
    FUNCTIONAL_NOOP,
    THRESHOLD_NOOP,
    METRIC_NOOP,
    TASK_EFFECTIVE_CHANGE,
)

MARGIN_STRATA = (
    "all",
    "near_threshold",
    "gt_target",
    "gt_background",
    "source_false_positive",
    "source_missed_target",
)


class DiagnosticError(ValueError):
    """Raised when diagnostic inputs are incomplete, inconsistent, or unsafe."""


@dataclass(frozen=True)
class NoOpThresholds:
    """Pre-registered numerical and evaluator thresholds.

    Null floors are inclusive: a maximum absolute change less than or equal to
    its floor is treated as indistinguishable from the corresponding null.
    Prediction thresholding remains strict (``probability > threshold``).
    """

    parameter_null_floor: float = 0.0
    logit_null_floor: float = 0.0
    probability_null_floor: float = 0.0
    probability_logit_consistency_atol: float = 1e-6
    prediction_threshold: float = 0.5
    near_threshold_lower: float = 0.45
    near_threshold_upper: float = 0.55
    entropy_eps: float = 1e-6
    connectivity: int = 2
    min_component_area: int = 1
    max_centroid_distance: float = 3.0

    def __post_init__(self) -> None:
        finite_nonnegative = (
            "parameter_null_floor",
            "logit_null_floor",
            "probability_null_floor",
            "probability_logit_consistency_atol",
        )
        for name in finite_nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DiagnosticError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise DiagnosticError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)

        for name in (
            "prediction_threshold",
            "near_threshold_lower",
            "near_threshold_upper",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DiagnosticError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise DiagnosticError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        if not (
            self.near_threshold_lower
            <= self.prediction_threshold
            <= self.near_threshold_upper
        ):
            raise DiagnosticError(
                "near-threshold interval must contain prediction_threshold"
            )
        entropy_eps = float(self.entropy_eps)
        if not math.isfinite(entropy_eps) or not 0.0 < entropy_eps < 0.5:
            raise DiagnosticError("entropy_eps must lie strictly between 0 and 0.5")
        object.__setattr__(self, "entropy_eps", entropy_eps)
        if isinstance(self.connectivity, bool) or self.connectivity not in (1, 2):
            raise DiagnosticError("connectivity must be 1 or 2")
        if (
            isinstance(self.min_component_area, bool)
            or not isinstance(self.min_component_area, (int, np.integer))
            or self.min_component_area < 1
        ):
            raise DiagnosticError("min_component_area must be a positive integer")
        distance = float(self.max_centroid_distance)
        if not math.isfinite(distance) or distance <= 0.0:
            raise DiagnosticError(
                "max_centroid_distance must be finite and positive"
            )
        object.__setattr__(self, "connectivity", int(self.connectivity))
        object.__setattr__(self, "min_component_area", int(self.min_component_area))
        object.__setattr__(self, "max_centroid_distance", distance)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NoOpThresholds":
        if not isinstance(value, Mapping):
            raise DiagnosticError("thresholds must be a mapping")
        expected = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if unknown or missing:
            raise DiagnosticError(
                f"threshold fields must be exact; missing={missing}, unknown={unknown}"
            )
        return cls(**{name: value[name] for name in expected})


def _to_numpy(value: Any, *, name: str) -> NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "buifc":
        raise DiagnosticError(f"{name} must have a numeric dtype")
    if array.size == 0:
        raise DiagnosticError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise DiagnosticError(f"{name} contains NaN or Inf")
    return array


def _as_single_image_2d(value: Any, *, name: str) -> NDArray[np.float64]:
    array = _to_numpy(value, name=name)
    if array.ndim == 2:
        result = array
    elif array.ndim == 3 and array.shape[0] == 1:
        result = array[0]
    elif array.ndim == 4 and array.shape[:2] == (1, 1):
        result = array[0, 0]
    else:
        raise DiagnosticError(
            f"{name} must have shape [H,W], [1,H,W], or [1,1,H,W]; "
            f"got {array.shape}"
        )
    if 0 in result.shape:
        raise DiagnosticError(f"{name} has an empty spatial dimension")
    return result.astype(np.float64, copy=False)


def _probability_2d(value: Any, *, name: str) -> NDArray[np.float64]:
    array = _as_single_image_2d(value, name=name)
    if (array < 0.0).any() or (array > 1.0).any():
        raise DiagnosticError(f"{name} must lie in [0, 1]")
    return array


def _quantile_summary(values: NDArray[Any]) -> dict[str, int | float | None]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "q50": None,
            "q90": None,
            "q99": None,
            "max": None,
        }
    quantiles = np.quantile(flattened, (0.5, 0.9, 0.99), method="linear")
    return {
        "count": int(flattened.size),
        "mean": float(flattened.mean(dtype=np.float64)),
        "min": float(flattened.min()),
        "q50": float(quantiles[0]),
        "q90": float(quantiles[1]),
        "q99": float(quantiles[2]),
        "max": float(flattened.max()),
    }


def _delta_distribution(pre: NDArray[Any], post: NDArray[Any]) -> dict[str, Any]:
    if pre.shape != post.shape:
        raise DiagnosticError(
            f"pre/post shapes differ: {pre.shape} versus {post.shape}"
        )
    delta = post.astype(np.float64, copy=False) - pre.astype(np.float64, copy=False)
    absolute = np.abs(delta)
    return {
        "absolute": _quantile_summary(absolute),
        "l2_norm": float(np.linalg.norm(delta.reshape(-1), ord=2)),
        "signed": _quantile_summary(delta),
    }


def parameter_change_report(
    parameter_pre: Mapping[str, Any],
    parameter_post: Mapping[str, Any],
    *,
    null_floor: float = 0.0,
    relative_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Return exact topology checks plus global/per-tensor delta statistics."""

    if not isinstance(parameter_pre, Mapping) or not isinstance(
        parameter_post, Mapping
    ):
        raise DiagnosticError("parameter snapshots must be mappings")
    if not parameter_pre:
        raise DiagnosticError("parameter snapshots cannot be empty")
    if tuple(parameter_pre) != tuple(parameter_post):
        raise DiagnosticError("parameter snapshot names/order differ")
    if not math.isfinite(float(null_floor)) or float(null_floor) < 0.0:
        raise DiagnosticError("parameter null_floor must be finite and nonnegative")
    if not math.isfinite(float(relative_epsilon)) or float(relative_epsilon) <= 0.0:
        raise DiagnosticError("relative_epsilon must be finite and positive")

    source_flat: list[NDArray[np.float64]] = []
    delta_flat: list[NDArray[np.float64]] = []
    per_tensor: list[dict[str, Any]] = []
    for name in parameter_pre:
        if not isinstance(name, str) or not name:
            raise DiagnosticError("parameter names must be non-empty strings")
        pre = _to_numpy(parameter_pre[name], name=f"parameter_pre[{name}]")
        post = _to_numpy(parameter_post[name], name=f"parameter_post[{name}]")
        if pre.shape != post.shape or pre.dtype != post.dtype:
            raise DiagnosticError(f"parameter topology changed: {name}")
        pre64 = pre.astype(np.float64, copy=False).reshape(-1)
        delta64 = post.astype(np.float64, copy=False).reshape(-1) - pre64
        source_l2 = float(np.linalg.norm(pre64, ord=2))
        delta_l2 = float(np.linalg.norm(delta64, ord=2))
        max_abs = float(np.abs(delta64).max())
        per_tensor.append(
            {
                "changed_above_null_floor": bool(max_abs > null_floor),
                "l2_delta": delta_l2,
                "max_abs_delta": max_abs,
                "name": name,
                "relative_l2_delta": float(
                    delta_l2 / (source_l2 + float(relative_epsilon))
                ),
                "scalar_count": int(delta64.size),
                "source_l2_norm": source_l2,
            }
        )
        source_flat.append(pre64)
        delta_flat.append(delta64)

    source_vector = np.concatenate(source_flat)
    delta_vector = np.concatenate(delta_flat)
    source_norm = float(np.linalg.norm(source_vector, ord=2))
    delta_norm = float(np.linalg.norm(delta_vector, ord=2))
    max_abs_delta = float(np.abs(delta_vector).max())
    return {
        "absolute_delta_quantiles": _quantile_summary(np.abs(delta_vector)),
        "changed_tensor_count": sum(
            bool(record["changed_above_null_floor"]) for record in per_tensor
        ),
        "l2_delta": delta_norm,
        "max_abs_delta": max_abs_delta,
        "null_floor": float(null_floor),
        "numerically_zero_at_floor": bool(max_abs_delta <= null_floor),
        "per_tensor": per_tensor,
        "relative_l2_delta": float(
            delta_norm / (source_norm + float(relative_epsilon))
        ),
        "scalar_count": int(delta_vector.size),
        "signed_delta_quantiles": _quantile_summary(delta_vector),
        "source_l2_norm": source_norm,
        "tensor_count": len(per_tensor),
    }


def _binary_entropy(probability: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    clipped = np.clip(probability, eps, 1.0 - eps)
    return -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )


def _state_summary(
    probability: NDArray[np.float64],
    foreground: NDArray[np.bool_],
    *,
    thresholds: NoOpThresholds,
) -> dict[str, Any]:
    entropy = _binary_entropy(probability, thresholds.entropy_eps)
    components = extract_connected_components(
        foreground,
        connectivity=thresholds.connectivity,
        min_area=thresholds.min_component_area,
    )
    return {
        "component_count": len(components),
        "entropy_mean": float(entropy.mean(dtype=np.float64)),
        "entropy_sum": float(entropy.sum(dtype=np.float64)),
        "foreground_fraction": float(foreground.mean(dtype=np.float64)),
        "foreground_pixel_count": int(foreground.sum()),
        "foreground_probability_mass_mean": float(
            probability.mean(dtype=np.float64)
        ),
        "foreground_probability_mass_sum": float(
            probability.sum(dtype=np.float64)
        ),
        "pixel_count": int(probability.size),
    }


def _official_metric_counts(
    probability: NDArray[np.float64],
    target: NDArray[np.bool_],
    *,
    thresholds: NoOpThresholds,
) -> dict[str, int]:
    protocol = IRSTDEvaluationProtocol(
        fixed_probability_threshold=thresholds.prediction_threshold,
        froc_probability_thresholds=(thresholds.prediction_threshold,),
        connectivity=thresholds.connectivity,
        max_centroid_distance=thresholds.max_centroid_distance,
        min_component_area=thresholds.min_component_area,
    )
    evaluator = UnifiedResearchEvaluator(protocol)
    evaluator.update_probabilities(probability, target)
    fixed = evaluator.compute().fixed
    return {
        "detected_targets": fixed.target.detected_targets,
        "false_alarm_pixels": fixed.target.false_alarm_pixels,
        "false_negative_pixels": fixed.pixel.false_negative_pixels,
        "false_positive_components": fixed.target.false_positive_components,
        "false_positive_pixels": fixed.pixel.false_positive_pixels,
        "intersection_pixels": fixed.pixel.true_positive_pixels,
        "predicted_positive_pixels": fixed.pixel.predicted_positive_pixels,
        "target_positive_pixels": fixed.pixel.target_positive_pixels,
        "total_image_pixels": fixed.target.total_image_pixels,
        "total_targets": fixed.target.total_targets,
        "true_negative_pixels": fixed.pixel.true_negative_pixels,
        "union_pixels": (
            fixed.pixel.true_positive_pixels
            + fixed.pixel.false_positive_pixels
            + fixed.pixel.false_negative_pixels
        ),
    }


def _margin_stratum(
    selector: NDArray[np.bool_],
    *,
    margin: NDArray[np.float64],
    probability_delta: NDArray[np.float64],
    binary_xor: NDArray[np.bool_],
    bg_to_fg: NDArray[np.bool_],
    fg_to_bg: NDArray[np.bool_],
) -> dict[str, Any]:
    selected_margin = margin[selector]
    selected_delta = probability_delta[selector]
    selected_abs_delta = np.abs(selected_delta)
    count = int(selector.sum())
    greater_than_margin = int((selected_abs_delta > selected_margin).sum())
    greater_than_tenth_margin = int(
        (selected_abs_delta > 0.1 * selected_margin).sum()
    )
    return {
        "absolute_probability_delta": _quantile_summary(selected_abs_delta),
        "abs_delta_gt_0_1_margin_count": greater_than_tenth_margin,
        "abs_delta_gt_margin_count": greater_than_margin,
        "bg_to_fg_count": int(bg_to_fg[selector].sum()),
        "binary_xor_count": int(binary_xor[selector].sum()),
        "fg_to_bg_count": int(fg_to_bg[selector].sum()),
        "fraction_abs_delta_gt_0_1_margin": (
            float(greater_than_tenth_margin / count) if count else None
        ),
        "fraction_abs_delta_gt_margin": (
            float(greater_than_margin / count) if count else None
        ),
        "pixel_count": count,
        "signed_probability_delta": _quantile_summary(selected_delta),
        "threshold_margin_pre": _quantile_summary(selected_margin),
    }


def threshold_margin_report(
    probability_pre: Any,
    probability_post: Any,
    target: Any,
    *,
    thresholds: NoOpThresholds | None = None,
) -> dict[str, Any]:
    """Compare ``|delta p|`` with the pre-update distance to the threshold."""

    thresholds = thresholds or NoOpThresholds()
    pre = _probability_2d(probability_pre, name="probability_pre")
    post = _probability_2d(probability_post, name="probability_post")
    target_array = _as_single_image_2d(target, name="target") > 0
    if pre.shape != post.shape or pre.shape != target_array.shape:
        raise DiagnosticError("probability/target spatial shapes differ")

    pre_foreground = pre > thresholds.prediction_threshold
    post_foreground = post > thresholds.prediction_threshold
    probability_delta = post - pre
    margin = np.abs(pre - thresholds.prediction_threshold)
    binary_xor = np.logical_xor(pre_foreground, post_foreground)
    bg_to_fg = np.logical_and(~pre_foreground, post_foreground)
    fg_to_bg = np.logical_and(pre_foreground, ~post_foreground)
    masks = {
        "all": np.ones(pre.shape, dtype=np.bool_),
        "near_threshold": np.logical_and(
            pre >= thresholds.near_threshold_lower,
            pre <= thresholds.near_threshold_upper,
        ),
        "gt_target": target_array,
        "gt_background": ~target_array,
        "source_false_positive": np.logical_and(pre_foreground, ~target_array),
        "source_missed_target": np.logical_and(~pre_foreground, target_array),
    }
    return {
        "near_threshold_interval": [
            thresholds.near_threshold_lower,
            thresholds.near_threshold_upper,
        ],
        "prediction_threshold": thresholds.prediction_threshold,
        "strata": {
            name: _margin_stratum(
                masks[name],
                margin=margin,
                probability_delta=probability_delta,
                binary_xor=binary_xor,
                bg_to_fg=bg_to_fg,
                fg_to_bg=fg_to_bg,
            )
            for name in MARGIN_STRATA
        },
        "threshold_rule": "strict_probability_greater_than_threshold",
    }


def analyze_noop_episode(
    *,
    parameter_pre: Mapping[str, Any],
    parameter_post: Mapping[str, Any],
    logits_pre: Any,
    logits_post: Any,
    target: Any,
    probability_pre: Any | None = None,
    probability_post: Any | None = None,
    thresholds: NoOpThresholds | None = None,
) -> dict[str, Any]:
    """Compute the complete four-level no-op and threshold-margin report."""

    thresholds = thresholds or NoOpThresholds()
    pre_logits = _as_single_image_2d(logits_pre, name="logits_pre")
    post_logits = _as_single_image_2d(logits_post, name="logits_post")
    target_array = _as_single_image_2d(target, name="target") > 0
    if pre_logits.shape != post_logits.shape or pre_logits.shape != target_array.shape:
        raise DiagnosticError("logit/target spatial shapes differ")

    derived_pre_probability = _as_single_image_2d(
        probabilities_from_logits(pre_logits), name="sigmoid(logits_pre)"
    )
    derived_post_probability = _as_single_image_2d(
        probabilities_from_logits(post_logits), name="sigmoid(logits_post)"
    )
    if (probability_pre is None) != (probability_post is None):
        raise DiagnosticError(
            "probability_pre and probability_post must be supplied together"
        )
    if probability_pre is None:
        pre_probability = derived_pre_probability
        post_probability = derived_post_probability
        probability_source = "stable_sigmoid_from_logits"
    else:
        pre_probability = _probability_2d(probability_pre, name="probability_pre")
        post_probability = _probability_2d(probability_post, name="probability_post")
        if pre_probability.shape != pre_logits.shape or post_probability.shape != pre_logits.shape:
            raise DiagnosticError("provided probability/logit spatial shapes differ")
        maximum_consistency_error = max(
            float(np.max(np.abs(pre_probability - derived_pre_probability))),
            float(np.max(np.abs(post_probability - derived_post_probability))),
        )
        if maximum_consistency_error > thresholds.probability_logit_consistency_atol:
            raise DiagnosticError(
                "provided probabilities do not match sigmoid(logits) within the "
                "pre-registered tolerance"
            )
        probability_source = "provided_and_verified_against_logits"

    parameter = parameter_change_report(
        parameter_pre,
        parameter_post,
        null_floor=thresholds.parameter_null_floor,
    )
    logit_change = _delta_distribution(pre_logits, post_logits)
    probability_change = _delta_distribution(pre_probability, post_probability)
    pre_foreground = pre_probability > thresholds.prediction_threshold
    post_foreground = post_probability > thresholds.prediction_threshold
    binary_xor = np.logical_xor(pre_foreground, post_foreground)
    bg_to_fg = np.logical_and(~pre_foreground, post_foreground)
    fg_to_bg = np.logical_and(pre_foreground, ~post_foreground)
    metric_pre = _official_metric_counts(
        pre_probability, target_array, thresholds=thresholds
    )
    metric_post = _official_metric_counts(
        post_probability, target_array, thresholds=thresholds
    )
    metrics_identical = metric_pre == metric_post

    logit_max = float(logit_change["absolute"]["max"])
    probability_max = float(probability_change["absolute"]["max"])
    xor_count = int(binary_xor.sum())
    if parameter["numerically_zero_at_floor"]:
        classification = NUMERIC_NOOP
    elif (
        logit_max <= thresholds.logit_null_floor
        and probability_max <= thresholds.probability_null_floor
    ):
        classification = FUNCTIONAL_NOOP
    elif xor_count == 0:
        classification = THRESHOLD_NOOP
    elif metrics_identical:
        classification = METRIC_NOOP
    else:
        classification = TASK_EFFECTIVE_CHANGE

    pre_state = _state_summary(
        pre_probability, pre_foreground, thresholds=thresholds
    )
    post_state = _state_summary(
        post_probability, post_foreground, thresholds=thresholds
    )
    logit_signed = logit_change["signed"]
    logit_absolute = logit_change["absolute"]
    probability_signed = probability_change["signed"]
    probability_absolute = probability_change["absolute"]

    return {
        # Flat canonical fields follow the names in the Stage-1 failure
        # diagnosis document.  The richer nested structures below retain the
        # signed/absolute distinction and per-tensor evidence.
        "BG_to_FG_pixel_count": int(bg_to_fg.sum()),
        "FG_to_BG_pixel_count": int(fg_to_bg.sum()),
        "binary_transitions": {
            "BG_to_FG_pixel_count": int(bg_to_fg.sum()),
            "FG_to_BG_pixel_count": int(fg_to_bg.sum()),
            "binary_pixel_xor_count": xor_count,
        },
        "binary_pixel_xor_count": xor_count,
        "classification": classification,
        "classification_priority": list(NOOP_CLASSIFICATION_ORDER),
        "component_count_post": post_state["component_count"],
        "component_count_pre": pre_state["component_count"],
        "entropy_post": post_state["entropy_mean"],
        "entropy_pre": pre_state["entropy_mean"],
        "foreground_pixel_fraction_post": post_state["foreground_fraction"],
        "foreground_pixel_fraction_pre": pre_state["foreground_fraction"],
        "foreground_probability_mass_post": post_state[
            "foreground_probability_mass_mean"
        ],
        "foreground_probability_mass_pre": pre_state[
            "foreground_probability_mass_mean"
        ],
        "functional_change": {
            "functionally_zero_at_floors": bool(
                logit_max <= thresholds.logit_null_floor
                and probability_max <= thresholds.probability_null_floor
            ),
            "logit": logit_change,
            "probability": probability_change,
            "probability_source": probability_source,
        },
        "logit_delta_abs_max": logit_absolute["max"],
        "logit_delta_abs_mean": logit_absolute["mean"],
        "logit_delta_mean": logit_signed["mean"],
        "logit_delta_q50": logit_signed["q50"],
        "logit_delta_q90": logit_signed["q90"],
        "logit_delta_q99": logit_signed["q99"],
        "metric_counts": {
            "identical": metrics_identical,
            "post": metric_post,
            "pre": metric_pre,
        },
        "metric_counts_identical": metrics_identical,
        "parameter_change": parameter,
        "parameter_l2_delta": parameter["l2_delta"],
        "parameter_max_abs_delta": parameter["max_abs_delta"],
        "prob_delta_abs_max": probability_absolute["max"],
        "prob_delta_abs_mean": probability_absolute["mean"],
        "prob_delta_mean": probability_signed["mean"],
        "prob_delta_q50": probability_signed["q50"],
        "prob_delta_q90": probability_signed["q90"],
        "prob_delta_q99": probability_signed["q99"],
        "relative_step_norm": parameter["relative_l2_delta"],
        "schema_version": 1,
        "state": {
            "post": post_state,
            "pre": pre_state,
        },
        "threshold_margin": threshold_margin_report(
            pre_probability,
            post_probability,
            target_array,
            thresholds=thresholds,
        ),
        "thresholds": thresholds.to_dict(),
    }


__all__ = [
    "DiagnosticError",
    "FUNCTIONAL_NOOP",
    "MARGIN_STRATA",
    "METRIC_NOOP",
    "NOOP_CLASSIFICATION_ORDER",
    "NUMERIC_NOOP",
    "NoOpThresholds",
    "TASK_EFFECTIVE_CHANGE",
    "THRESHOLD_NOOP",
    "analyze_noop_episode",
    "parameter_change_report",
    "threshold_margin_report",
]
