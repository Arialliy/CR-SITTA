"""Prespecified two-arm anchored-residual train8 fitting readout.

The control measures additional training of v1; candidate-minus-control is the
parameterization comparison at equal added training budget. These supplied
same-image fitting results cannot establish generalization or significance.
Target transitions must come from the unchanged v3 object helper; this module
checks their accounting and aggregates them, without implementing a matcher.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from analysis.o3_reachability_objects_v3 import STATES
from scripts import run_p3_stage_b4_full_pilot64_v1 as b4

DATASET = "NUDT-SIRST"
ORDERED_IMAGE_IDS = ("000891", "001005", "000641", "001179", "000328", "001048", "000460", "001228")
CONDITIONS = tuple(b4.CONDITIONS)
METHODS = ("source", "o3", "v1", "control", "candidate")
ARMS = ("control", "candidate")
FAMILIES = ("gaussian_noise", "gaussian_blur", "low_contrast", "stripe_noise")
METRICS = ("iou", "normalized_iou", "pd", "fa_per_million")
TOLERANCE = 1e-12
TARGET_FIELDS = ("target_positive_pixels", "total_targets", "total_image_pixels", "image_count")


class AnchoredMetricsError(ValueError):
    """Evidence is incomplete, inconsistent or outside the fixed train8 scope."""


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AnchoredMetricsError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise AnchoredMetricsError(f"{label} must be finite numeric >= {minimum}")
    return float(value)


def _losses(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(ARMS):
        raise AnchoredMetricsError(f"{label} requires exactly control and candidate full-fit losses")
    return {arm: _number(values[arm], f"{label}.{arm}") for arm in ARMS}


def _cells(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)) or len(values) != 13:
        raise AnchoredMetricsError("exactly 13 ordered condition cells required")
    validated = []
    gt_counts = None
    for row, (family, severity) in zip(values, CONDITIONS):
        condition = b4._condition_key(family, severity)
        if (not isinstance(row, Mapping) or row.get("dataset") != DATASET
                or row.get("condition") != condition or row.get("corruption") != family
                or type(row.get("severity")) is not int or row["severity"] != severity):
            raise AnchoredMetricsError(f"frozen dataset/condition order differs at {condition}")
        cell = {"dataset": DATASET, "condition": condition, "corruption": family, "severity": severity}
        for method in METHODS:
            try:
                endpoint = b4._validated_endpoint_summary(row.get(method),
                    label=f"{condition}/{method}", expected_image_count=8)
            except (b4.StageB4ProtocolError, TypeError, ValueError) as exc:
                raise AnchoredMetricsError(f"invalid {condition}/{method} endpoint: {exc}") from exc
            observed_gt = tuple(endpoint[field] for field in TARGET_FIELDS)
            if gt_counts is None:
                gt_counts = observed_gt
            elif observed_gt != gt_counts:
                raise AnchoredMetricsError("unchanged paired train8 GT count accounting differs")
            cell[method] = endpoint
        validated.append(cell)
    return validated


def _geometry(record: Mapping[str, Any]) -> tuple[Any, ...]:
    if not isinstance(record, Mapping):
        raise AnchoredMetricsError("each target record must be a mapping")
    identifier = _int(record.get("id"), "target ID", 1)
    area = _int(record.get("area"), "target area", 1)
    centroid, bbox = record.get("centroid"), record.get("bbox")
    if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
        raise AnchoredMetricsError("target centroid must be [row,column]")
    center = tuple(_number(value, "target centroid") for value in centroid)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise AnchoredMetricsError("target bbox must contain four integer coordinates")
    bounds = tuple(_int(value, "target bbox") for value in bbox)
    r0, c0, r1, c1 = bounds
    if not (r0 < r1 <= 256 and c0 < c1 <= 256 and r0 <= center[0] < r1
            and c0 <= center[1] < c1 and area <= (r1-r0)*(c1-c0)):
        raise AnchoredMetricsError("target geometry is outside its 256-square image/bbox")
    return identifier, area, center, bounds


def _object_summary(value: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if not isinstance(value, Mapping):
        raise AnchoredMetricsError("v3 object_transitions result required")
    protocol = value.get("protocol", {})
    expected_protocol = {"connectivity": 8, "scipy_connectivity_argument": 2,
        "minimum_component_area": 1, "matching": "shared_research_evaluator_one_to_one_hungarian",
        "centroid_coordinates": "row_column", "max_centroid_distance": 3.0,
        "distance_boundary": "strict_less_than", "target_order": "ascending_gt_component_label",
        "input_rule": "already_binary_bool_or_exact_0_1_no_implicit_threshold",
        "false_alarm_pixels": "sum_of_areas_of_unmatched_prediction_components"}
    if protocol != expected_protocol or value.get("image_shape") != [256, 256] or value.get("schema_version") != 1:
        raise AnchoredMetricsError("object transition geometry/matching protocol differs")
    total = _int(value.get("total_targets"), "total targets")
    records = value.get("targets")
    if not isinstance(records, (list, tuple)) or len(records) != total:
        raise AnchoredMetricsError("target detail count differs")
    geometry = tuple(_geometry(record) for record in records)
    if tuple(item[0] for item in geometry) != tuple(range(1, total + 1)):
        raise AnchoredMetricsError("target IDs must follow original ascending contiguous labels")
    counts = {state: 0 for state in STATES}
    match_labels = {"previous": [], "diagnostic": []}
    for record in records:
        if type(record.get("prev_hit")) is not bool or type(record.get("new_hit")) is not bool:
            raise AnchoredMetricsError("target hit flags must be bool")
        state = ("TP" if record["prev_hit"] else "FN") + "→" + ("TP" if record["new_hit"] else "FN")
        if record.get("state") != state:
            raise AnchoredMetricsError("target state differs from its hit flags")
        counts[state] += 1
        for label, hit_key, match_key in (("previous", "prev_hit", "prev_match"), ("diagnostic", "new_hit", "new_match")):
            match = record.get(match_key)
            if not record[hit_key]:
                if match is not None:
                    raise AnchoredMetricsError("unhit target cannot carry a match")
                continue
            if not isinstance(match, Mapping):
                raise AnchoredMetricsError("hit target requires the v3 match record")
            prediction_label = _int(match.get("prediction_label"), "prediction label", 1)
            if _int(match.get("prediction_index"), "prediction index") != prediction_label - 1:
                raise AnchoredMetricsError("prediction index/label accounting differs")
            if _number(match.get("centroid_distance"), "matched centroid distance") >= 3.0:
                raise AnchoredMetricsError("matched centroid distance is not strictly less than three")
            match_labels[label].append(prediction_label)
    raw_counts = value.get("transition_counts")
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(STATES):
        raise AnchoredMetricsError("four transition count fields required")
    for state in STATES:
        if _int(raw_counts[state], f"transition count {state}") != counts[state]:
            raise AnchoredMetricsError("transition counts disagree with per-target states")
    summaries = {}
    for label in ("previous", "diagnostic"):
        endpoint = value.get(label)
        if not isinstance(endpoint, Mapping):
            raise AnchoredMetricsError("previous/diagnostic object summary required")
        fields = ("detected_targets", "false_negative_targets", "prediction_component_count",
                  "false_positive_components", "false_alarm_pixels")
        numbers = {field: _int(endpoint.get(field), f"{label}.{field}") for field in fields}
        hits = len(match_labels[label])
        if (numbers["detected_targets"] != hits or numbers["false_negative_targets"] + hits != total
                or numbers["prediction_component_count"] != hits + numbers["false_positive_components"]
                or numbers["false_alarm_pixels"] < numbers["false_positive_components"]
                or len(set(match_labels[label])) != hits):
            raise AnchoredMetricsError("one-to-one object hit/component accounting differs")
        unmatched = endpoint.get("unmatched_prediction_labels")
        if not isinstance(unmatched, (list, tuple)) or len(unmatched) != numbers["false_positive_components"]:
            raise AnchoredMetricsError("unmatched prediction labels differ")
        unmatched = [_int(k, "unmatched prediction label", 1) for k in unmatched]
        all_labels = match_labels[label] + unmatched
        if sorted(all_labels) != list(range(1, numbers["prediction_component_count"] + 1)):
            raise AnchoredMetricsError("matched/unmatched prediction label partition differs")
        pd = _number(endpoint.get("pd"), f"{label}.pd")
        expected_pd = hits / total if total else 0.0
        if not math.isclose(pd, expected_pd, rel_tol=0.0, abs_tol=TOLERANCE):
            raise AnchoredMetricsError("object Pd does not match integer target counts")
        summaries[label] = numbers
    if (type(value.get("net_detected_target_change")) is not int
            or value["net_detected_target_change"] != summaries["diagnostic"]["detected_targets"] - summaries["previous"]["detected_targets"]):
        raise AnchoredMetricsError("net detected-target change differs")
    return {"total_targets": total, "gt_positive_pixels": sum(item[1] for item in geometry),
            "transition_counts": counts, **summaries}, geometry


def _transitions(values: Sequence[Mapping[str, Any]], cells) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(values, (list, tuple)) or len(values) != 104:
        raise AnchoredMetricsError("exactly 104 ordered image-condition transitions required")
    result = {arm: [] for arm in ARMS}
    gt_geometry = {}
    for index, row in enumerate(values):
        ci, ii = divmod(index, 8)
        if (not isinstance(row, Mapping) or row.get("condition") != cells[ci]["condition"]
                or row.get("image_id") != ORDERED_IMAGE_IDS[ii]):
            raise AnchoredMetricsError("transition order/condition/frozen train8 ID differs")
        for arm in ARMS:
            summary, geometry = _object_summary(row.get(arm))
            if ii not in gt_geometry:
                gt_geometry[ii] = geometry
            elif geometry != gt_geometry[ii]:
                raise AnchoredMetricsError("same train image has changed GT target geometry")
            result[arm].append(summary)
        if (result["control"][-1]["previous"] != result["candidate"][-1]["previous"]
                or [r["prev_hit"] for r in row["control"]["targets"]]
                   != [r["prev_hit"] for r in row["candidate"]["targets"]]):
            raise AnchoredMetricsError("the two arms do not share the same v1 target states")
    for ci, cell in enumerate(cells):
        for arm in ARMS:
            selected = result[arm][ci*8:(ci+1)*8]
            if (sum(row["total_targets"] for row in selected) != cell["v1"]["total_targets"]
                    or sum(row["gt_positive_pixels"] for row in selected) != cell["v1"]["target_positive_pixels"]):
                raise AnchoredMetricsError("object GT totals disagree with the condition endpoint")
            for label, method in (("previous", "v1"), ("diagnostic", arm)):
                for field in ("detected_targets", "false_alarm_pixels"):
                    if sum(row[label][field] for row in selected) != cell[method][field]:
                        raise AnchoredMetricsError(f"object/cell {method}.{field} accounting differs")
    return result


def _transition_totals(rows):
    counts = {state: sum(row["transition_counts"][state] for row in rows) for state in STATES}
    return {"image_condition_observations": len(rows), "total_targets": sum(counts.values()),
            "transition_counts": counts,
            "previous_detected_targets": counts["TP→TP"] + counts["TP→FN"],
            "new_detected_targets": counts["TP→TP"] + counts["FN→TP"],
            "net_detected_target_change": counts["FN→TP"] - counts["TP→FN"]}


def _macro(cells):
    return {method: {metric: math.fsum(row[method][metric] for row in cells) / len(cells)
                     for metric in METRICS} for method in METHODS}


def _deltas(means):
    return {f"{method}_minus_{reference}": {
                "iou_pp": 100 * (means[method]["iou"] - means[reference]["iou"]),
                "normalized_iou_pp": 100 * (means[method]["normalized_iou"] - means[reference]["normalized_iou"]),
                "pd_pp": 100 * (means[method]["pd"] - means[reference]["pd"]),
                "fa_per_million": means[method]["fa_per_million"] - means[reference]["fa_per_million"]}
            for method, reference in (("candidate", "v1"), ("candidate", "control"), ("control", "v1"))}


def summarize(cells, transitions, initial_losses, final_losses) -> dict[str, Any]:
    """Check fixed train8 evidence and apply 18 performance + 2 learning goals."""
    rows = _cells(cells)
    initial, final = _losses(initial_losses, "initial_losses"), _losses(final_losses, "final_losses")
    states = _transitions(transitions, rows)
    nonclean, clean = rows[1:], rows[0]
    means = _macro(nonclean)
    families = {family: _macro([row for row in nonclean if row["corruption"] == family]) for family in FAMILIES}
    totals = {arm: {"all": _transition_totals(states[arm]), "nonclean": _transition_totals(states[arm][8:])} for arm in ARMS}
    learning = {f"{arm}_full_fit_loss_decreased": final[arm] < initial[arm] - TOLERANCE for arm in ARMS}
    goals = dict(learning)
    for reference in ("v1", "control"):
        goals[f"candidate_nonclean_iou_above_{reference}"] = means["candidate"]["iou"] > means[reference]["iou"] + TOLERANCE
        goals[f"candidate_nonclean_pd_at_least_{reference}"] = means["candidate"]["pd"] >= means[reference]["pd"] - TOLERANCE
        goals[f"candidate_nonclean_fa_at_most_{reference}"] = means["candidate"]["fa_per_million"] <= means[reference]["fa_per_million"] + TOLERANCE
    for metric in ("iou", "pd"):
        goals[f"candidate_clean_{metric}_at_least_v1"] = clean["candidate"][metric] >= clean["v1"][metric] - TOLERANCE
    goals["candidate_clean_fa_at_most_v1"] = clean["candidate"]["fa_per_million"] <= clean["v1"]["fa_per_million"] + TOLERANCE
    goals["candidate_no_v1_tp_to_fn_all_conditions"] = totals["candidate"]["all"]["transition_counts"]["TP→FN"] == 0
    for family in FAMILIES:
        for metric in ("iou", "pd"):
            goals[f"candidate_{family}_{metric}_at_least_v1"] = families[family]["candidate"][metric] >= families[family]["v1"][metric] - TOLERANCE
    return {
        "dataset": DATASET, "image_ids": list(ORDERED_IMAGE_IDS), "condition_count": 13,
        "nonclean_condition_count": 12, "samples": 104, "image_count_per_condition": 8,
        "nonclean_macro": means, "families": families, "clean": clean,
        "all_conditions_macro_descriptive_only": _macro(rows),
        "deltas": {"nonclean": _deltas(means), "families": {key: _deltas(value) for key, value in families.items()},
                   "clean": _deltas(_macro([clean]))},
        "target_transitions": totals, "initial_losses": initial, "final_losses": final,
        "goals": goals, "goal_count": len(goals), "passed_goal_count": sum(goals.values()),
        "failed_goals": [name for name, passed in goals.items() if not passed], "all_goals_met": all(goals.values()),
        "learning_checks": learning, "learning_checks_passed": all(learning.values()),
        "fit_performance_signal_passed": all(passed for name, passed in goals.items() if name not in learning),
        "numerical_tolerance": TOLERANCE,
        "averaging": "equal means of condition metrics: 12 nonclean, 3 per family; clean separate",
        "comparison_interpretation": {"control_minus_v1": "effect of 128 additional fixed-budget training steps",
            "candidate_minus_control": "parameterization comparison at equal added training budget",
            "candidate_minus_v1": "combined parameterization and additional-training difference",
            "family_fa_is_reported_not_an_extra_gate": True},
        "target_transition_source": "unchanged v3 object_transitions with shared 8-connected strict-centroid<3 Hungarian matcher",
        "transition_accounting_checked_here_matching_not_recomputed": True,
        "fit_and_measure_on_same_images": True, "no_validation_split": True,
        "generalization_claim": False, "statistical_significance_claim": False,
        "paper_result": False, "formal_test": False,
        "automatic_full_training_allowed": False, "automatic_formal_test_allowed": False,
        "automatic_retry_or_hyperparameter_search_allowed": False,
        "engineering_tests_are_performance_evidence": False,
    }


__all__ = ["summarize", "AnchoredMetricsError", "ORDERED_IMAGE_IDS", "CONDITIONS", "TOLERANCE"]
