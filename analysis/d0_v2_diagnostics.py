"""Strict, CPU-only D0-v2 no-op diagnostics.

The frozen v1 implementation in :mod:`tta.diagnostics` remains the source of
truth for the four-level no-op classification and official metric counts.
This module adds only the pre-registered D0-v2 measurements that v1 did not
publish, while failing closed if the v1 result shape changes.

The prediction rule is fixed to ``probability > 0.5``.  Connectivity and the
minimum retained component area are taken from the caller-supplied, frozen
:class:`tta.diagnostics.NoOpThresholds` instance.  All functions are pure:
they do not mutate inputs, touch the filesystem, inspect a model, or select a
candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from metrics.connected_components import extract_connected_components
from metrics.irstd_metrics import probabilities_from_logits
from tta.diagnostics import (
    MARGIN_STRATA,
    NOOP_CLASSIFICATION_ORDER,
    DiagnosticError,
    NoOpThresholds,
    analyze_noop_episode,
)


D0_V2_NOOP_ARTIFACT_TYPE: Final = "cr_sitta_d0_v2_noop_episode_diagnostics"
D0_V2_NOOP_SCHEMA_VERSION: Final = 2
FROZEN_PREDICTION_THRESHOLD: Final = 0.5
THRESHOLD_RULE: Final = "strict_probability_greater_than_0_5"

MARGIN_BIN_SPECS: Final = (
    ("ge_0_lt_1e_minus_4", 0.0, 1.0e-4),
    ("ge_1e_minus_4_lt_1e_minus_3", 1.0e-4, 1.0e-3),
    ("ge_1e_minus_3_lt_1e_minus_2", 1.0e-3, 1.0e-2),
    ("ge_1e_minus_2_lt_5e_minus_2", 1.0e-2, 5.0e-2),
)
MARGIN_BIN_KEYS: Final = tuple(item[0] for item in MARGIN_BIN_SPECS)
MARGIN_OUTSIDE_LOWER_BOUND: Final = 5.0e-2

_V1_REPORT_FIELDS: Final = frozenset(
    {
        "BG_to_FG_pixel_count",
        "FG_to_BG_pixel_count",
        "binary_transitions",
        "binary_pixel_xor_count",
        "classification",
        "classification_priority",
        "component_count_post",
        "component_count_pre",
        "entropy_post",
        "entropy_pre",
        "foreground_pixel_fraction_post",
        "foreground_pixel_fraction_pre",
        "foreground_probability_mass_post",
        "foreground_probability_mass_pre",
        "functional_change",
        "logit_delta_abs_max",
        "logit_delta_abs_mean",
        "logit_delta_mean",
        "logit_delta_q50",
        "logit_delta_q90",
        "logit_delta_q99",
        "metric_counts",
        "metric_counts_identical",
        "parameter_change",
        "parameter_l2_delta",
        "parameter_max_abs_delta",
        "prob_delta_abs_max",
        "prob_delta_abs_mean",
        "prob_delta_mean",
        "prob_delta_q50",
        "prob_delta_q90",
        "prob_delta_q99",
        "relative_step_norm",
        "schema_version",
        "state",
        "threshold_margin",
        "thresholds",
    }
)
_V2_ADDITIONAL_FIELDS: Final = frozenset(
    {
        "artifact_type",
        "p95_abs_delta_logit",
        "p95_abs_delta_probability",
        "foreground_probability_mass_delta",
        "largest_component_area_pre",
        "largest_component_area_post",
        "largest_component_area_delta",
        "near_threshold_pixel_count_by_margin_bin",
        "threshold_margin_outside_ge_5e_minus_2_pixel_count",
        "threshold_margin_partition_pixel_count",
        "threshold_rule_v2",
    }
)
D0_V2_NOOP_REPORT_FIELDS: Final = _V1_REPORT_FIELDS | _V2_ADDITIONAL_FIELDS

_DELTA_DISTRIBUTION_FIELDS: Final = frozenset(
    {"absolute", "l2_norm", "signed"}
)
_QUANTILE_FIELDS: Final = frozenset(
    {"count", "mean", "min", "q50", "q90", "q99", "max"}
)
_STATE_FIELDS: Final = frozenset(
    {
        "component_count",
        "entropy_mean",
        "entropy_sum",
        "foreground_fraction",
        "foreground_pixel_count",
        "foreground_probability_mass_mean",
        "foreground_probability_mass_sum",
        "pixel_count",
    }
)
_METRIC_COUNT_FIELDS: Final = frozenset(
    {
        "detected_targets",
        "false_alarm_pixels",
        "false_negative_pixels",
        "false_positive_components",
        "false_positive_pixels",
        "intersection_pixels",
        "predicted_positive_pixels",
        "target_positive_pixels",
        "total_image_pixels",
        "total_targets",
        "true_negative_pixels",
        "union_pixels",
    }
)
_MARGIN_STRATUM_FIELDS: Final = frozenset(
    {
        "absolute_probability_delta",
        "abs_delta_gt_0_1_margin_count",
        "abs_delta_gt_margin_count",
        "bg_to_fg_count",
        "binary_xor_count",
        "fg_to_bg_count",
        "fraction_abs_delta_gt_0_1_margin",
        "fraction_abs_delta_gt_margin",
        "pixel_count",
        "signed_probability_delta",
        "threshold_margin_pre",
    }
)
_PARAMETER_CHANGE_FIELDS: Final = frozenset(
    {
        "absolute_delta_quantiles",
        "changed_tensor_count",
        "l2_delta",
        "max_abs_delta",
        "null_floor",
        "numerically_zero_at_floor",
        "per_tensor",
        "relative_l2_delta",
        "scalar_count",
        "signed_delta_quantiles",
        "source_l2_norm",
        "tensor_count",
    }
)
_PER_TENSOR_FIELDS: Final = frozenset(
    {
        "changed_above_null_floor",
        "l2_delta",
        "max_abs_delta",
        "name",
        "relative_l2_delta",
        "scalar_count",
        "source_l2_norm",
    }
)
_INPUT_FIELDS_WITH_DERIVED_PROBABILITIES: Final = frozenset(
    {
        "parameter_pre",
        "parameter_post",
        "logits_pre",
        "logits_post",
        "target",
        "thresholds",
    }
)
_INPUT_FIELDS_WITH_PROVIDED_PROBABILITIES: Final = (
    _INPUT_FIELDS_WITH_DERIVED_PROBABILITIES
    | {"probability_pre", "probability_post"}
)


class D0V2DiagnosticError(DiagnosticError):
    """A D0-v2 input, result schema, or numerical invariant failed closed."""


def _require_exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0V2DiagnosticError(f"{label} must be a mapping")
    keys = set(value)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        raise D0V2DiagnosticError(
            f"{label} fields must be exact; missing={missing}, unknown={unknown}"
        )
    return value


def _finite_float(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D0V2DiagnosticError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise D0V2DiagnosticError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise D0V2DiagnosticError(f"{label} must be nonnegative")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise D0V2DiagnosticError(f"{label} must be a nonnegative integer")
    return value


def _finite_numeric_tree(value: Any, label: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise D0V2DiagnosticError(f"{label} contains a non-string key")
            _finite_numeric_tree(child, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_numeric_tree(child, f"{label}[{index}]")
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise D0V2DiagnosticError(f"{label} must be finite")
        return
    raise D0V2DiagnosticError(
        f"{label} has a non-JSON scalar type {type(value).__name__}"
    )


def _single_image_2d(value: Any, label: str) -> NDArray[np.float64]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "buif":
        raise D0V2DiagnosticError(f"{label} must have a real numeric dtype")
    if array.ndim == 2:
        result = array
    elif array.ndim == 3 and array.shape[0] == 1:
        result = array[0]
    elif array.ndim == 4 and array.shape[:2] == (1, 1):
        result = array[0, 0]
    else:
        raise D0V2DiagnosticError(
            f"{label} must have shape [H,W], [1,H,W], or [1,1,H,W]; "
            f"got {array.shape}"
        )
    if result.size == 0 or 0 in result.shape:
        raise D0V2DiagnosticError(f"{label} cannot be empty")
    result64 = result.astype(np.float64, copy=False)
    if not np.isfinite(result64).all():
        raise D0V2DiagnosticError(f"{label} contains NaN or Inf")
    return result64


def _validate_parameter_snapshots(
    parameter_pre: Any, parameter_post: Any
) -> None:
    if not isinstance(parameter_pre, Mapping) or not isinstance(
        parameter_post, Mapping
    ):
        raise D0V2DiagnosticError("parameter snapshots must be mappings")
    if not parameter_pre:
        raise D0V2DiagnosticError("parameter snapshots cannot be empty")
    if tuple(parameter_pre) != tuple(parameter_post):
        raise D0V2DiagnosticError("parameter snapshot names/order differ")
    for name in parameter_pre:
        if not isinstance(name, str) or not name:
            raise D0V2DiagnosticError(
                "parameter names must be non-empty strings"
            )
        pre = parameter_pre[name]
        post = parameter_post[name]
        if hasattr(pre, "detach"):
            pre = pre.detach().cpu().numpy()
        if hasattr(post, "detach"):
            post = post.detach().cpu().numpy()
        pre_array = np.asarray(pre)
        post_array = np.asarray(post)
        if pre_array.dtype.kind not in "buif" or post_array.dtype.kind not in "buif":
            raise D0V2DiagnosticError(
                f"parameter snapshot {name!r} must have a real numeric dtype"
            )
        if pre_array.size == 0 or post_array.size == 0:
            raise D0V2DiagnosticError(
                f"parameter snapshot {name!r} cannot be empty"
            )
        if pre_array.shape != post_array.shape or pre_array.dtype != post_array.dtype:
            raise D0V2DiagnosticError(f"parameter topology changed: {name}")
        if not np.isfinite(pre_array).all() or not np.isfinite(post_array).all():
            raise D0V2DiagnosticError(
                f"parameter snapshot {name!r} contains NaN or Inf"
            )


def _probability_2d(value: Any, label: str) -> NDArray[np.float64]:
    result = _single_image_2d(value, label)
    if (result < 0.0).any() or (result > 1.0).any():
        raise D0V2DiagnosticError(f"{label} must lie in [0, 1]")
    return result


def _frozen_thresholds(value: Any) -> NoOpThresholds:
    if type(value) is not NoOpThresholds:
        raise D0V2DiagnosticError(
            "thresholds must be a frozen NoOpThresholds instance"
        )
    if value.prediction_threshold != FROZEN_PREDICTION_THRESHOLD:
        raise D0V2DiagnosticError(
            "D0-v2 prediction_threshold must be exactly 0.5"
        )
    return value


def _absolute_p95(pre: NDArray[np.float64], post: NDArray[np.float64]) -> float:
    if pre.shape != post.shape:
        raise D0V2DiagnosticError(
            f"pre/post spatial shapes differ: {pre.shape} versus {post.shape}"
        )
    values = np.abs(post - pre).reshape(-1)
    result = float(np.quantile(values, 0.95, method="linear"))
    if not math.isfinite(result):
        raise D0V2DiagnosticError("absolute delta p95 is not finite")
    return result


def _margin_bin_counts(
    probability_pre: NDArray[np.float64],
) -> tuple[dict[str, int], int]:
    margin = np.abs(probability_pre - FROZEN_PREDICTION_THRESHOLD)
    counts = {
        name: int(np.logical_and(margin >= lower, margin < upper).sum())
        for name, lower, upper in MARGIN_BIN_SPECS
    }
    outside = int((margin >= MARGIN_OUTSIDE_LOWER_BOUND).sum())
    if sum(counts.values()) + outside != int(margin.size):
        raise D0V2DiagnosticError(
            "threshold margin bins do not form an exhaustive disjoint partition"
        )
    return counts, outside


def _largest_component_area(
    probability: NDArray[np.float64], thresholds: NoOpThresholds
) -> int:
    # Strictly greater is intentional: p == 0.5 remains background.
    foreground = probability > FROZEN_PREDICTION_THRESHOLD
    components = extract_connected_components(
        foreground,
        connectivity=thresholds.connectivity,
        min_area=thresholds.min_component_area,
    )
    return max((component.area for component in components), default=0)


def _validate_quantile_summary(value: Any, label: str) -> None:
    summary = _require_exact_keys(value, _QUANTILE_FIELDS, label)
    count = _nonnegative_int(summary["count"], f"{label}.count")
    numeric_keys = ("mean", "min", "q50", "q90", "q99", "max")
    if count == 0:
        if any(summary[key] is not None for key in numeric_keys):
            raise D0V2DiagnosticError(
                f"{label} empty summary values must all be null"
            )
        return
    values = [_finite_float(summary[key], f"{label}.{key}") for key in numeric_keys]
    if not (values[1] <= values[2] <= values[3] <= values[4] <= values[5]):
        raise D0V2DiagnosticError(f"{label} quantiles are not ordered")


def _validate_delta_distribution(value: Any, label: str) -> None:
    delta = _require_exact_keys(value, _DELTA_DISTRIBUTION_FIELDS, label)
    _validate_quantile_summary(delta["absolute"], f"{label}.absolute")
    _validate_quantile_summary(delta["signed"], f"{label}.signed")
    _finite_float(delta["l2_norm"], f"{label}.l2_norm", nonnegative=True)


def _validate_v1_nested_schema(report: Mapping[str, Any]) -> None:
    _require_exact_keys(
        report["binary_transitions"],
        frozenset(
            {
                "BG_to_FG_pixel_count",
                "FG_to_BG_pixel_count",
                "binary_pixel_xor_count",
            }
        ),
        "report.binary_transitions",
    )
    functional = _require_exact_keys(
        report["functional_change"],
        frozenset(
            {
                "functionally_zero_at_floors",
                "logit",
                "probability",
                "probability_source",
            }
        ),
        "report.functional_change",
    )
    _validate_delta_distribution(functional["logit"], "report.functional_change.logit")
    _validate_delta_distribution(
        functional["probability"], "report.functional_change.probability"
    )

    metric_counts = _require_exact_keys(
        report["metric_counts"],
        frozenset({"identical", "post", "pre"}),
        "report.metric_counts",
    )
    for state_name in ("pre", "post"):
        counts = _require_exact_keys(
            metric_counts[state_name],
            _METRIC_COUNT_FIELDS,
            f"report.metric_counts.{state_name}",
        )
        for key, value in counts.items():
            _nonnegative_int(value, f"report.metric_counts.{state_name}.{key}")

    parameter = _require_exact_keys(
        report["parameter_change"],
        _PARAMETER_CHANGE_FIELDS,
        "report.parameter_change",
    )
    _validate_quantile_summary(
        parameter["absolute_delta_quantiles"],
        "report.parameter_change.absolute_delta_quantiles",
    )
    _validate_quantile_summary(
        parameter["signed_delta_quantiles"],
        "report.parameter_change.signed_delta_quantiles",
    )
    per_tensor = parameter["per_tensor"]
    if not isinstance(per_tensor, list) or not per_tensor:
        raise D0V2DiagnosticError(
            "report.parameter_change.per_tensor must be a non-empty list"
        )
    for index, tensor in enumerate(per_tensor):
        _require_exact_keys(
            tensor,
            _PER_TENSOR_FIELDS,
            f"report.parameter_change.per_tensor[{index}]",
        )

    states = _require_exact_keys(
        report["state"], frozenset({"post", "pre"}), "report.state"
    )
    for state_name in ("pre", "post"):
        state = _require_exact_keys(
            states[state_name], _STATE_FIELDS, f"report.state.{state_name}"
        )
        _nonnegative_int(
            state["component_count"], f"report.state.{state_name}.component_count"
        )
        _nonnegative_int(
            state["foreground_pixel_count"],
            f"report.state.{state_name}.foreground_pixel_count",
        )
        _nonnegative_int(
            state["pixel_count"], f"report.state.{state_name}.pixel_count"
        )

    threshold_margin = _require_exact_keys(
        report["threshold_margin"],
        frozenset(
            {
                "near_threshold_interval",
                "prediction_threshold",
                "strata",
                "threshold_rule",
            }
        ),
        "report.threshold_margin",
    )
    strata = _require_exact_keys(
        threshold_margin["strata"],
        frozenset(MARGIN_STRATA),
        "report.threshold_margin.strata",
    )
    for stratum_name in MARGIN_STRATA:
        stratum = _require_exact_keys(
            strata[stratum_name],
            _MARGIN_STRATUM_FIELDS,
            f"report.threshold_margin.strata.{stratum_name}",
        )
        for key in (
            "absolute_probability_delta",
            "signed_probability_delta",
            "threshold_margin_pre",
        ):
            _validate_quantile_summary(
                stratum[key],
                f"report.threshold_margin.strata.{stratum_name}.{key}",
            )


def validate_noop_episode_v2_report(value: Any) -> dict[str, Any]:
    """Validate and return a shallow copy of one exact D0-v2 report.

    This verifier is intentionally strict: unknown/missing fields, non-finite
    numeric leaves, changed nested v1 schemas, or inconsistent derived values
    are rejected instead of being ignored.
    """

    report = _require_exact_keys(value, D0_V2_NOOP_REPORT_FIELDS, "report")
    _finite_numeric_tree(report)
    if report["artifact_type"] != D0_V2_NOOP_ARTIFACT_TYPE:
        raise D0V2DiagnosticError("report.artifact_type is not D0-v2 no-op")
    if report["schema_version"] != D0_V2_NOOP_SCHEMA_VERSION:
        raise D0V2DiagnosticError("report.schema_version must be exactly 2")
    if report["threshold_rule_v2"] != THRESHOLD_RULE:
        raise D0V2DiagnosticError("report.threshold_rule_v2 is not frozen")
    if report["classification_priority"] != list(NOOP_CLASSIFICATION_ORDER):
        raise D0V2DiagnosticError("report.classification_priority is not frozen")

    _validate_v1_nested_schema(report)
    try:
        thresholds = NoOpThresholds.from_mapping(report["thresholds"])
    except DiagnosticError as error:
        raise D0V2DiagnosticError(str(error)) from error
    _frozen_thresholds(thresholds)
    threshold_margin = report["threshold_margin"]
    if threshold_margin["prediction_threshold"] != FROZEN_PREDICTION_THRESHOLD:
        raise D0V2DiagnosticError(
            "report.threshold_margin.prediction_threshold must equal 0.5"
        )
    if (
        threshold_margin["threshold_rule"]
        != "strict_probability_greater_than_threshold"
    ):
        raise D0V2DiagnosticError("report.threshold_margin.threshold_rule changed")

    bins = _require_exact_keys(
        report["near_threshold_pixel_count_by_margin_bin"],
        frozenset(MARGIN_BIN_KEYS),
        "report.near_threshold_pixel_count_by_margin_bin",
    )
    bin_total = sum(
        _nonnegative_int(
            value,
            f"report.near_threshold_pixel_count_by_margin_bin.{key}",
        )
        for key, value in bins.items()
    )
    outside = _nonnegative_int(
        report["threshold_margin_outside_ge_5e_minus_2_pixel_count"],
        "report.threshold_margin_outside_ge_5e_minus_2_pixel_count",
    )
    partition_count = _nonnegative_int(
        report["threshold_margin_partition_pixel_count"],
        "report.threshold_margin_partition_pixel_count",
    )
    pre_pixel_count = _nonnegative_int(
        report["state"]["pre"]["pixel_count"], "report.state.pre.pixel_count"
    )
    if bin_total + outside != partition_count or partition_count != pre_pixel_count:
        raise D0V2DiagnosticError("threshold margin partition count is inconsistent")

    for key in ("p95_abs_delta_logit", "p95_abs_delta_probability"):
        _finite_float(report[key], f"report.{key}", nonnegative=True)
    if report["p95_abs_delta_logit"] > report["logit_delta_abs_max"]:
        raise D0V2DiagnosticError("logit absolute p95 exceeds maximum")
    if report["p95_abs_delta_probability"] > report["prob_delta_abs_max"]:
        raise D0V2DiagnosticError("probability absolute p95 exceeds maximum")

    mass_pre = _finite_float(
        report["foreground_probability_mass_pre"],
        "report.foreground_probability_mass_pre",
        nonnegative=True,
    )
    mass_post = _finite_float(
        report["foreground_probability_mass_post"],
        "report.foreground_probability_mass_post",
        nonnegative=True,
    )
    mass_delta = _finite_float(
        report["foreground_probability_mass_delta"],
        "report.foreground_probability_mass_delta",
    )
    if mass_pre != report["state"]["pre"]["foreground_probability_mass_sum"]:
        raise D0V2DiagnosticError("pre foreground probability mass is not SUM")
    if mass_post != report["state"]["post"]["foreground_probability_mass_sum"]:
        raise D0V2DiagnosticError("post foreground probability mass is not SUM")
    if mass_delta != mass_post - mass_pre:
        raise D0V2DiagnosticError("foreground probability mass delta is inconsistent")

    area_pre = _nonnegative_int(
        report["largest_component_area_pre"],
        "report.largest_component_area_pre",
    )
    area_post = _nonnegative_int(
        report["largest_component_area_post"],
        "report.largest_component_area_post",
    )
    area_delta = report["largest_component_area_delta"]
    if isinstance(area_delta, bool) or not isinstance(area_delta, int):
        raise D0V2DiagnosticError(
            "report.largest_component_area_delta must be an integer"
        )
    if area_delta != area_post - area_pre:
        raise D0V2DiagnosticError("largest component area delta is inconsistent")
    for state_name, area in (("pre", area_pre), ("post", area_post)):
        state = report["state"][state_name]
        if area > state["foreground_pixel_count"]:
            raise D0V2DiagnosticError(
                f"largest {state_name} component exceeds foreground pixel count"
            )
        if (state["component_count"] == 0) != (area == 0):
            raise D0V2DiagnosticError(
                f"largest {state_name} component/count zero status differs"
            )

    return dict(report)


def analyze_noop_episode_v2(
    *,
    parameter_pre: Mapping[str, Any],
    parameter_post: Mapping[str, Any],
    logits_pre: Any,
    logits_post: Any,
    target: Any,
    thresholds: NoOpThresholds,
    probability_pre: Any | None = None,
    probability_post: Any | None = None,
) -> dict[str, Any]:
    """Return the exact, finite D0-v2 report for one already-run episode."""

    frozen_thresholds = _frozen_thresholds(thresholds)
    _validate_parameter_snapshots(parameter_pre, parameter_post)
    pre_logits = _single_image_2d(logits_pre, "logits_pre")
    post_logits = _single_image_2d(logits_post, "logits_post")
    target_array = _single_image_2d(target, "target")
    if (
        pre_logits.shape != post_logits.shape
        or pre_logits.shape != target_array.shape
    ):
        raise D0V2DiagnosticError("logit/target spatial shapes differ")

    if (probability_pre is None) != (probability_post is None):
        raise D0V2DiagnosticError(
            "probability_pre and probability_post must be supplied together"
        )
    if probability_pre is None:
        pre_probability = _probability_2d(
            probabilities_from_logits(pre_logits), "sigmoid(logits_pre)"
        )
        post_probability = _probability_2d(
            probabilities_from_logits(post_logits), "sigmoid(logits_post)"
        )
    else:
        pre_probability = _probability_2d(probability_pre, "probability_pre")
        post_probability = _probability_2d(probability_post, "probability_post")
    if pre_probability.shape != post_probability.shape:
        raise D0V2DiagnosticError("pre/post probability spatial shapes differ")

    try:
        report = analyze_noop_episode(
            parameter_pre=parameter_pre,
            parameter_post=parameter_post,
            logits_pre=logits_pre,
            logits_post=logits_post,
            target=target,
            probability_pre=probability_pre,
            probability_post=probability_post,
            thresholds=frozen_thresholds,
        )
    except DiagnosticError as error:
        raise D0V2DiagnosticError(str(error)) from error
    if set(report) != _V1_REPORT_FIELDS:
        missing = sorted(_V1_REPORT_FIELDS - set(report))
        unknown = sorted(set(report) - _V1_REPORT_FIELDS)
        raise D0V2DiagnosticError(
            "frozen v1 no-op report schema changed; "
            f"missing={missing}, unknown={unknown}"
        )

    margin_bins, outside_count = _margin_bin_counts(pre_probability)
    mass_pre = float(pre_probability.sum(dtype=np.float64))
    mass_post = float(post_probability.sum(dtype=np.float64))
    largest_pre = _largest_component_area(pre_probability, frozen_thresholds)
    largest_post = _largest_component_area(post_probability, frozen_thresholds)

    # v1's flat mass aliases referred to means.  D0-v2 follows the v4 protocol
    # literally and publishes SUM; both mean and sum remain explicit in state.
    report.update(
        {
            "artifact_type": D0_V2_NOOP_ARTIFACT_TYPE,
            "foreground_probability_mass_delta": mass_post - mass_pre,
            "foreground_probability_mass_post": mass_post,
            "foreground_probability_mass_pre": mass_pre,
            "largest_component_area_delta": largest_post - largest_pre,
            "largest_component_area_post": largest_post,
            "largest_component_area_pre": largest_pre,
            "near_threshold_pixel_count_by_margin_bin": margin_bins,
            "p95_abs_delta_logit": _absolute_p95(pre_logits, post_logits),
            "p95_abs_delta_probability": _absolute_p95(
                pre_probability, post_probability
            ),
            "schema_version": D0_V2_NOOP_SCHEMA_VERSION,
            "threshold_margin_outside_ge_5e_minus_2_pixel_count": outside_count,
            "threshold_margin_partition_pixel_count": int(pre_probability.size),
            "threshold_rule_v2": THRESHOLD_RULE,
        }
    )
    return validate_noop_episode_v2_report(report)


def analyze_noop_episode_v2_from_mapping(value: Any) -> dict[str, Any]:
    """Parse one of the two exact input schemas and run D0-v2 diagnostics."""

    if not isinstance(value, Mapping):
        raise D0V2DiagnosticError("input must be a mapping")
    keys = set(value)
    allowed = (
        _INPUT_FIELDS_WITH_DERIVED_PROBABILITIES,
        _INPUT_FIELDS_WITH_PROVIDED_PROBABILITIES,
    )
    if keys not in allowed:
        candidates = []
        for expected in allowed:
            candidates.append(
                {
                    "missing": sorted(expected - keys),
                    "unknown": sorted(keys - expected),
                }
            )
        raise D0V2DiagnosticError(
            f"input fields must match one exact schema; candidates={candidates}"
        )
    thresholds_value = value["thresholds"]
    if isinstance(thresholds_value, Mapping):
        try:
            thresholds = NoOpThresholds.from_mapping(thresholds_value)
        except DiagnosticError as error:
            raise D0V2DiagnosticError(str(error)) from error
    else:
        thresholds = _frozen_thresholds(thresholds_value)
    kwargs = {key: value[key] for key in keys if key != "thresholds"}
    return analyze_noop_episode_v2(thresholds=thresholds, **kwargs)


__all__ = [
    "D0V2DiagnosticError",
    "D0_V2_NOOP_ARTIFACT_TYPE",
    "D0_V2_NOOP_REPORT_FIELDS",
    "D0_V2_NOOP_SCHEMA_VERSION",
    "FROZEN_PREDICTION_THRESHOLD",
    "MARGIN_BIN_KEYS",
    "MARGIN_BIN_SPECS",
    "MARGIN_OUTSIDE_LOWER_BOUND",
    "THRESHOLD_RULE",
    "analyze_noop_episode_v2",
    "analyze_noop_episode_v2_from_mapping",
    "validate_noop_episode_v2_report",
]
