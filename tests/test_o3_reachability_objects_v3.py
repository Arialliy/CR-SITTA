"""Synthetic masks only; parity with the shared component/matching evaluator."""

import json

import numpy as np
import pytest

from analysis import o3_reachability_objects_v3 as objects
from metrics.irstd_metrics import IRSTDEvaluationProtocol
from metrics.irstd_metrics_v2 import FORMAL_FROC_THRESHOLDS_V2, UnifiedResearchEvaluatorV2
from metrics.target_transitions import evaluate_target_transitions


def _protocol():
    return IRSTDEvaluationProtocol(fixed_probability_threshold=0.5,
        froc_probability_thresholds=FORMAL_FROC_THRESHOLDS_V2, connectivity=2,
        max_centroid_distance=3.0, min_component_area=1)


def _four_states():
    target = np.zeros((20, 20), dtype=bool)
    previous = np.zeros_like(target)
    diagnostic = np.zeros_like(target)
    for coordinate in [(2, 2), (2, 12), (12, 2), (12, 12)]:
        target[coordinate] = True
    previous[2, 2] = previous[2, 12] = True
    diagnostic[2, 2] = diagnostic[12, 2] = True
    return target, previous, diagnostic


def test_all_four_states_in_ascending_gt_label_order_and_json_safe():
    target, previous, diagnostic = _four_states()
    result = objects.object_transitions(target, previous, diagnostic)
    assert result["total_targets"] == 4
    assert result["transition_counts"] == {state: 1 for state in objects.STATES}
    assert [row["id"] for row in result["targets"]] == [1, 2, 3, 4]
    assert [row["state"] for row in result["targets"]] == list(objects.STATES)
    assert [row["centroid"] for row in result["targets"]] == [[2.0, 2.0], [2.0, 12.0], [12.0, 2.0], [12.0, 12.0]]
    assert all(row["area"] == 1 for row in result["targets"])
    assert result["targets"][0]["bbox"] == [2, 2, 3, 3]
    assert result["targets"][0]["prev_match"]["centroid_distance"] == 0.0
    assert result["targets"][1]["new_match"] is None
    assert result["targets"][2]["prev_match"] is None
    assert result["previous"]["detected_targets"] == 2
    assert result["diagnostic"]["detected_targets"] == 2
    assert result["net_detected_target_change"] == 0
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_exact_three_pixel_distance_is_not_a_hit():
    target = np.zeros((12, 12), dtype=np.uint8)
    previous = np.zeros_like(target)
    diagnostic = np.zeros_like(target)
    target[5, 5] = previous[5, 7] = diagnostic[5, 8] = 1
    result = objects.object_transitions(target, previous, diagnostic)
    assert result["targets"][0]["state"] == "TP→FN"
    assert result["targets"][0]["prev_match"]["centroid_distance"] == 2.0
    assert result["diagnostic"]["false_positive_components"] == 1
    assert result["diagnostic"]["false_alarm_pixels"] == 1


def test_diagonal_pixels_form_one_8_connected_target():
    target = np.zeros((10, 10), dtype=bool)
    target[3, 3] = target[4, 4] = True
    result = objects.object_transitions(target, target, target)
    assert result["total_targets"] == 1
    assert result["targets"][0]["area"] == 2
    assert result["targets"][0]["centroid"] == [3.5, 3.5]
    assert result["transition_counts"]["TP→TP"] == 1


def test_one_prediction_cannot_hit_two_gt_targets():
    target = np.zeros((10, 10), dtype=bool)
    target[2, 2] = target[2, 4] = True
    shared = np.zeros_like(target)
    shared[2, 3] = True
    result = objects.object_transitions(target, shared, target)
    assert result["total_targets"] == 2
    assert result["previous"]["detected_targets"] == 1
    assert result["diagnostic"]["detected_targets"] == 2
    assert result["transition_counts"]["TP→TP"] == 1
    assert result["transition_counts"]["FN→TP"] == 1


def test_false_alarm_pixels_follow_unmatched_components_not_all_pixel_fp():
    target = np.zeros((20, 20), dtype=bool)
    target[4, 4] = True
    previous = np.zeros_like(target)
    previous[4, 6] = True  # No pixel overlap but valid centroid match.
    previous[12:14, 12:14] = True  # Unmatched component with area four.
    result = objects.object_transitions(target, previous, target)
    assert result["previous"]["detected_targets"] == 1
    assert result["previous"]["false_positive_components"] == 1
    assert result["previous"]["false_alarm_pixels"] == 4
    assert result["previous"]["unmatched_prediction_labels"] == [2]
    assert np.logical_and(previous, ~target).sum() == 5


@pytest.mark.parametrize("target_on,previous_on,diagnostic_on", [(False, False, False), (False, True, True), (True, False, False)])
def test_empty_gt_or_predictions_have_complete_zero_safe_counts(target_on, previous_on, diagnostic_on):
    target = np.zeros((8, 8), dtype=bool)
    previous = np.zeros_like(target)
    diagnostic = np.zeros_like(target)
    target[3, 3], previous[3, 3], diagnostic[3, 3] = target_on, previous_on, diagnostic_on
    result = objects.object_transitions(target, previous, diagnostic)
    assert result["total_targets"] == int(target_on)
    assert sum(result["transition_counts"].values()) == int(target_on)
    if not target_on:
        assert result["targets"] == []
        assert result["previous"]["pd"] == result["diagnostic"]["pd"] == 0.0
        assert result["previous"]["false_positive_components"] == int(previous_on)
    else:
        assert result["transition_counts"]["FN→FN"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("seed", range(6))
def test_exact_parity_with_official_target_transitions_and_evaluator(seed):
    rng = np.random.default_rng(seed)
    target = rng.random((18, 19)) > 0.96
    previous = rng.random(target.shape) > 0.93
    diagnostic = rng.random(target.shape) > 0.93
    result = objects.object_transitions(target, previous, diagnostic)
    official = evaluate_target_transitions(previous.astype(np.float32), diagnostic.astype(np.float32), target,
                                            protocol=_protocol())
    assert result["transition_counts"] == official.transition_counts.to_dict()
    assert [row["state"] for row in result["targets"]] == [row.transition for row in official.target_records]
    for name, mask in [("previous", previous), ("diagnostic", diagnostic)]:
        evaluator = UnifiedResearchEvaluatorV2(_protocol())
        evaluator.update_probabilities(mask.astype(np.float32), target)
        fixed = evaluator.compute().fixed
        assert result[name]["detected_targets"] == fixed.detected_targets
        assert result[name]["false_positive_components"] == fixed.false_positive_components
        assert result[name]["false_alarm_pixels"] == fixed.false_alarm_pixels
        assert result[name]["pd"] == fixed.detection_probability


def test_calls_shared_extraction_and_matcher_without_new_matching_rule(monkeypatch):
    target, previous, diagnostic = _four_states()
    extract = objects.components_api.extract_connected_components
    match = objects.matching_api.match_components
    extraction_calls, match_calls = [], []
    def tracked_extract(mask, **kwargs):
        extraction_calls.append(kwargs)
        return extract(mask, **kwargs)
    def tracked_match(predictions, targets, **kwargs):
        match_calls.append(kwargs)
        return match(predictions, targets, **kwargs)
    monkeypatch.setattr(objects.components_api, "extract_connected_components", tracked_extract)
    monkeypatch.setattr(objects.matching_api, "match_components", tracked_match)
    objects.object_transitions(target, previous, diagnostic)
    assert extraction_calls == [{"connectivity": 2, "min_area": 1}] * 3
    assert match_calls == [{"max_centroid_distance": 3.0}] * 2


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int32, np.float32, np.float64])
def test_valid_binary_dtypes_and_noncontiguous_views_not_modified(dtype):
    arrays = [value.astype(dtype).T for value in _four_states()]
    before = [value.copy() for value in arrays]
    result = objects.object_transitions(*arrays)
    assert result == objects.object_transitions(*arrays)
    assert all(np.array_equal(value, old) for value, old in zip(arrays, before))


@pytest.mark.parametrize("bad", [
    np.array([[0.5]], dtype=np.float32), np.array([[255]], dtype=np.uint8),
    np.array([[-1]], dtype=np.int32), np.array([[np.nan]]), np.array([[np.inf]]),
    np.zeros((0, 3)), np.zeros((1, 1, 3)), np.array([["1"]]),
    np.array([[1 + 0j]]), np.array([[1]], dtype=object), [[0, 1]],
    np.ma.array([[1]], mask=[[True]]),
])
def test_no_silent_threshold_conversion_of_invalid_input(bad):
    normal = np.zeros((1, 1), dtype=bool)
    with pytest.raises((TypeError, ValueError)):
        objects.object_transitions(normal, bad, normal)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="identical shapes"):
        objects.object_transitions(np.zeros((2, 3)), np.zeros((3, 2)), np.zeros((2, 3)))


def test_higher_pixel_iou_can_erase_a_detected_target_without_capacity_claim():
    target = np.zeros((40, 40), dtype=bool)
    target[2:12, 2:12] = True
    target[22, 22] = True
    previous = target.copy()
    previous[25:35, 2:12] = True
    diagnostic = target.copy()
    diagnostic[22, 22] = False
    previous_iou = np.logical_and(target, previous).sum() / np.logical_or(target, previous).sum()
    diagnostic_iou = np.logical_and(target, diagnostic).sum() / np.logical_or(target, diagnostic).sum()
    assert diagnostic_iou > previous_iou
    result = objects.object_transitions(target, previous, diagnostic)
    assert result["transition_counts"]["TP→FN"] == 1
    assert result["previous"]["pd"] == 1.0
    assert result["diagnostic"]["pd"] == 0.5
    interpretation = result["interpretation"]
    assert interpretation["pixel_iou_oracle_is_pd_upper_bound"] is False
    assert interpretation["pixel_iou_oracle_is_fa_upper_bound"] is False
    assert interpretation["tp_to_fn_is_proof_of_unprotectable_target"] is False
    assert interpretation["oracle_optimality_verified_by_this_function"] is False
