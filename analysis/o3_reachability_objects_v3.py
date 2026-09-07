"""GT-target transitions for a materialized offline pixel-IoU diagnostic mask.

Component extraction and one-to-one matching are the exact shared research
evaluator implementations. No probabilities, threshold tuning, adaptation, or
oracle optimization are performed by this readout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metrics import connected_components as components_api
from metrics import target_matching as matching_api

STATES = ("TP→TP", "TP→FN", "FN→TP", "FN→FN")


def _binary_mask(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or isinstance(value, np.ma.MaskedArray):
        raise TypeError(f"{name} must be an unmasked numpy ndarray")
    if value.ndim != 2 or any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional mask")
    if value.dtype.kind not in "buif":
        raise TypeError(f"{name} must have a real boolean/integer/float dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if not np.logical_or(value == 0, value == 1).all():
        raise ValueError(f"{name} must contain exactly 0/1 values; implicit thresholding is forbidden")
    return np.array(value, dtype=np.bool_, copy=True, order="C")


def _prediction_summary(predictions, matching, total_targets: int) -> dict[str, Any]:
    false_alarm_pixels = sum(predictions[index].area for index in matching.unmatched_prediction_indices)
    return {
        "detected_targets": int(matching.true_positives),
        "false_negative_targets": int(matching.false_negatives),
        "prediction_component_count": len(predictions),
        "false_positive_components": int(matching.false_positives),
        "false_alarm_pixels": int(false_alarm_pixels),
        "pd": float(matching.true_positives / total_targets) if total_targets else 0.0,
        "unmatched_prediction_labels": [int(predictions[index].label)
                                         for index in matching.unmatched_prediction_indices],
    }


def _match_record(match, predictions) -> dict[str, Any] | None:
    if match is None:
        return None
    component = predictions[match.prediction_index]
    return {"prediction_index": int(match.prediction_index),
            "prediction_label": int(component.label), "area": int(component.area),
            "centroid": [float(v) for v in component.centroid],
            "bbox": [int(v) for v in component.bbox],
            "centroid_distance": float(match.centroid_distance)}


def object_transitions(
    target: np.ndarray,
    previous_mask: np.ndarray,
    diagnostic_mask: np.ndarray,
) -> dict[str, Any]:
    """Compare v1 and diagnostic binary masks against fixed ordered GT objects.

    Input masks must already be binary. ``id`` is the GT connected-component
    label (one based, ascending label order); centroids are ``[row, column]``.
    ``prev_hit`` and ``new_hit`` reflect the existing Hungarian matcher, not
    local overlap, nearest-neighbor heuristics, or continuous target scores.
    """
    gt = _binary_mask(target, "target")
    previous = _binary_mask(previous_mask, "previous_mask")
    diagnostic = _binary_mask(diagnostic_mask, "diagnostic_mask")
    if gt.shape != previous.shape or gt.shape != diagnostic.shape:
        raise ValueError("target, previous_mask and diagnostic_mask must have identical shapes")
    ground_truth = components_api.extract_connected_components(gt, connectivity=2, min_area=1)
    old_components = components_api.extract_connected_components(previous, connectivity=2, min_area=1)
    new_components = components_api.extract_connected_components(diagnostic, connectivity=2, min_area=1)
    old_matches = matching_api.match_components(old_components, ground_truth, max_centroid_distance=3.0)
    new_matches = matching_api.match_components(new_components, ground_truth, max_centroid_distance=3.0)
    old_by_target = {match.target_index: match for match in old_matches.matches}
    new_by_target = {match.target_index: match for match in new_matches.matches}
    counts = {state: 0 for state in STATES}
    records = []
    for index, component in enumerate(ground_truth):
        old_match, new_match = old_by_target.get(index), new_by_target.get(index)
        old_hit, new_hit = old_match is not None, new_match is not None
        state = ("TP" if old_hit else "FN") + "→" + ("TP" if new_hit else "FN")
        counts[state] += 1
        records.append({"id": int(component.label), "area": int(component.area),
            "centroid": [float(v) for v in component.centroid],
            "bbox": [int(v) for v in component.bbox],
            "prev_hit": bool(old_hit), "new_hit": bool(new_hit), "state": state,
            "prev_match": _match_record(old_match, old_components),
            "new_match": _match_record(new_match, new_components)})
    total = len(ground_truth)
    if (sum(counts.values()) != total
            or counts["TP→TP"] + counts["TP→FN"] != old_matches.true_positives
            or counts["TP→TP"] + counts["FN→TP"] != new_matches.true_positives):
        raise RuntimeError("target transition accounting does not match official assignments")
    return {
        "schema_version": 1, "comparison": "v1→offline_pixel_iou_diagnostic",
        "image_shape": [int(v) for v in gt.shape], "total_targets": total,
        "transition_counts": counts, "targets": records,
        "previous": _prediction_summary(old_components, old_matches, total),
        "diagnostic": _prediction_summary(new_components, new_matches, total),
        "net_detected_target_change": int(new_matches.true_positives - old_matches.true_positives),
        "protocol": {
            "connectivity": 8, "scipy_connectivity_argument": 2, "minimum_component_area": 1,
            "matching": "shared_research_evaluator_one_to_one_hungarian",
            "centroid_coordinates": "row_column", "max_centroid_distance": 3.0,
            "distance_boundary": "strict_less_than", "target_order": "ascending_gt_component_label",
            "input_rule": "already_binary_bool_or_exact_0_1_no_implicit_threshold",
            "false_alarm_pixels": "sum_of_areas_of_unmatched_prediction_components",
        },
        "interpretation": {
            "offline_gt_diagnostic_only": True,
            "oracle_optimality_verified_by_this_function": False,
            "pixel_iou_oracle_is_pd_upper_bound": False,
            "pixel_iou_oracle_is_fa_upper_bound": False,
            "tp_to_fn_is_proof_of_unprotectable_target": False,
            "caveat": "An offline pixel-IoU-maximizing mask need not optimize Pd or Fa. A TP→FN transition describes this chosen mask, not proof that the available expression space cannot preserve that target.",
        },
    }


__all__ = ["object_transitions", "STATES"]
