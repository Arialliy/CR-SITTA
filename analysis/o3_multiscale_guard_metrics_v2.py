"""Fixed train8 fit readout for the background-growth-penalized v2 branch.

These are prespecified advancement goals for one supervised fitting run, not
statistical tests or evidence of performance on unseen images. This module
only summarizes supplied endpoint records; it never reads data or runs models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from scripts import run_p3_stage_b4_full_pilot64_v1 as b4

DATASET = "NUDT-SIRST"
METHODS = ("source", "o3", "previous", "guarded")
METRICS = ("iou", "normalized_iou", "pd", "fa_per_million")
FAMILIES = ("gaussian_noise", "gaussian_blur", "low_contrast", "stripe_noise")
CONDITIONS = tuple(b4.CONDITIONS)
LOSS_FIELDS = (
    "segmentation_full_fit_loss",
    "background_guard_full_fit_loss",
    "augmented_full_fit_loss",
)
TOLERANCE = 1e-12
TARGET_FIELDS = ("target_positive_pixels", "total_targets", "total_image_pixels", "image_count")


class GuardMetricsError(ValueError):
    """Incomplete, incomparable or invalid endpoint evidence is rejected."""


def _losses(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(LOSS_FIELDS):
        raise GuardMetricsError(f"{label} must contain exactly the three full-fit loss fields")
    result = {}
    for key in LOSS_FIELDS:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GuardMetricsError(f"{label}.{key} must be numeric, not bool")
        if not math.isfinite(value) or value < 0.0:
            raise GuardMetricsError(f"{label}.{key} must be finite and nonnegative")
        result[key] = float(value)
    if result["augmented_full_fit_loss"] + TOLERANCE < result["segmentation_full_fit_loss"]:
        raise GuardMetricsError(f"{label} augmented loss cannot be below its segmentation term")
    return result


def _validated_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence) or len(cells) != 13:
        raise GuardMetricsError("exactly 13 condition endpoint records required")
    expected = {b4._condition_key(*condition): condition for condition in CONDITIONS}
    validated: dict[str, dict[str, Any]] = {}
    expected_targets = None
    for row in cells:
        if not isinstance(row, Mapping):
            raise GuardMetricsError("each condition endpoint must be a mapping")
        condition = row.get("condition")
        if not isinstance(condition, str) or condition not in expected or condition in validated:
            raise GuardMetricsError("condition key is missing, duplicate, or not in the frozen 13 conditions")
        corruption, severity = expected[condition]
        if (row.get("corruption") != corruption or type(row.get("severity")) is not int
                or row["severity"] != severity or row.get("dataset", DATASET) != DATASET):
            raise GuardMetricsError(f"condition/dataset identity differs: {condition}")
        value: dict[str, Any] = {"dataset": DATASET, "condition": condition,
                                  "corruption": corruption, "severity": severity}
        for method in METHODS:
            try:
                endpoint = b4._validated_endpoint_summary(row.get(method),
                    label=f"{condition}/{method}", expected_image_count=8)
            except (b4.StageB4ProtocolError, TypeError, ValueError) as exc:
                raise GuardMetricsError(f"invalid train8 endpoint {condition}/{method}: {exc}") from exc
            target_counts = tuple(endpoint[field] for field in TARGET_FIELDS)
            if expected_targets is None:
                expected_targets = target_counts
            elif target_counts != expected_targets:
                raise GuardMetricsError(f"paired unchanged train8 GT accounting differs: {condition}/{method}")
            value[method] = endpoint
        validated[condition] = value
    if set(validated) != set(expected):
        raise GuardMetricsError("condition Cartesian keyset is incomplete")
    return [validated[b4._condition_key(*condition)] for condition in CONDITIONS]


def _macro(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    return {method: {metric: math.fsum(row[method][metric] for row in rows) / len(rows)
                     for metric in METRICS} for method in METHODS}


def _deltas(means: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for reference in ("o3", "previous"):
        result[f"vs_{reference}"] = {
            "iou_pp": 100 * (means["guarded"]["iou"] - means[reference]["iou"]),
            "normalized_iou_pp": 100 * (means["guarded"]["normalized_iou"] - means[reference]["normalized_iou"]),
            "pd_pp": 100 * (means["guarded"]["pd"] - means[reference]["pd"]),
            "fa_per_million": means["guarded"]["fa_per_million"] - means[reference]["fa_per_million"],
        }
    return result


def summarize(
    cells: Sequence[Mapping[str, Any]],
    initial_losses: Mapping[str, Any],
    final_losses: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize all endpoints and apply the fixed v2 train8 performance goals.

    Endpoints use the existing lowercase B4 keys including ``normalized_iou``
    and all 11 raw integer counts. Supplied row order is immaterial; severity,
    condition identity, counts, paired GT totals and the exact 13 cells are not.
    Loss dictionaries contain exactly the keys declared in ``LOSS_FIELDS``.
    """
    rows = _validated_cells(cells)
    initial = _losses(initial_losses, "initial_losses")
    final = _losses(final_losses, "final_losses")
    nonclean = [row for row in rows if row["corruption"] != "clean"]
    clean = next(row for row in rows if row["corruption"] == "clean")
    overall = _macro(nonclean)
    families = {family: _macro([row for row in nonclean if row["corruption"] == family])
                for family in FAMILIES}
    clean_means = _macro([clean])
    goals = {
        "augmented_full_fit_loss_decreased": final["augmented_full_fit_loss"] < initial["augmented_full_fit_loss"] - TOLERANCE,
        "nonclean_iou_above_previous": overall["guarded"]["iou"] > overall["previous"]["iou"] + TOLERANCE,
        "nonclean_pd_at_least_previous": overall["guarded"]["pd"] >= overall["previous"]["pd"] - TOLERANCE,
        "nonclean_fa_at_most_previous": overall["guarded"]["fa_per_million"] <= overall["previous"]["fa_per_million"] + TOLERANCE,
    }
    for family in ("low_contrast", "stripe_noise"):
        means = families[family]
        goals[f"{family}_iou_at_least_previous"] = means["guarded"]["iou"] >= means["previous"]["iou"] - TOLERANCE
    for family in ("gaussian_blur", "gaussian_noise"):
        means = families[family]
        goals[f"{family}_iou_at_least_o3"] = means["guarded"]["iou"] >= means["o3"]["iou"] - TOLERANCE
        goals[f"{family}_fa_at_most_o3"] = means["guarded"]["fa_per_million"] <= means["o3"]["fa_per_million"] + TOLERANCE
    for metric in ("iou", "pd"):
        goals[f"clean_{metric}_at_least_o3"] = clean["guarded"][metric] >= clean["o3"][metric] - TOLERANCE
    goals["clean_fa_at_most_o3"] = clean["guarded"]["fa_per_million"] <= clean["o3"]["fa_per_million"] + TOLERANCE
    learning_key = "augmented_full_fit_loss_decreased"
    return {
        "dataset": DATASET, "split_name": "train", "image_count_per_condition": 8,
        "condition_count": 13, "nonclean_condition_count": 12,
        "family_condition_counts": {family: 3 for family in FAMILIES},
        "ordered_conditions": [row["condition"] for row in rows],
        "nonclean_macro": overall, "family_macro": families, "clean": clean,
        "all_conditions_macro_descriptive_only": _macro(rows),
        "deltas": {"nonclean": _deltas(overall),
                   "families": {family: _deltas(means) for family, means in families.items()},
                   "clean": _deltas(clean_means)},
        "initial_losses": initial, "final_losses": final,
        "goals": goals, "all_goals_met": all(goals.values()),
        "passed_goal_count": sum(goals.values()), "goal_count": len(goals),
        "failed_goals": [name for name, passed in goals.items() if not passed],
        "learning_check_passed": goals[learning_key],
        "fit_performance_signal_passed": all(passed for name, passed in goals.items() if name != learning_key),
        "numerical_tolerance": TOLERANCE,
        "averaging": "unweighted condition-level arithmetic mean; 12 nonclean cells, 3 per family; clean separate",
        "delta_units": "IoU, nIoU and Pd percentage points; Fa false-alarm pixels per million image pixels",
        "fit_and_measure_on_same_images": True, "no_validation_split": True,
        "generalization_claim": False, "statistical_significance_claim": False,
        "paper_result": False, "formal_test": False,
        "automatic_full_training_allowed": False, "automatic_formal_test_allowed": False,
        "automatic_retry_or_hyperparameter_search_allowed": False,
        "interpretation": "Prespecified supervised train8 fitting goals, not a significance test or unseen-image evaluation; all failures must remain visible.",
    }


__all__ = ["summarize", "GuardMetricsError", "LOSS_FIELDS", "CONDITIONS", "TOLERANCE"]
