from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from metrics.irstd_metrics import IRSTDEvaluationProtocol
from metrics.target_transitions import (
    FN_TO_FN,
    FN_TO_TP,
    SCORE_STATUS_CANDIDATE_NOT_FROZEN,
    SOURCE_TO_POST,
    TENT_PRE_TO_POST,
    TP_TO_FN,
    TP_TO_TP,
    TargetScoreContext,
    TargetScoreDefinition,
    TargetTransitionEvaluator,
    evaluate_target_transitions,
)


def _protocol() -> IRSTDEvaluationProtocol:
    return IRSTDEvaluationProtocol(
        fixed_probability_threshold=0.5,
        froc_probability_thresholds=(0.5,),
        connectivity=2,
        max_centroid_distance=3.0,
        min_component_area=1,
    )


def _four_transition_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.zeros((15, 15), dtype=np.uint8)
    coordinates = ((2, 2), (2, 10), (10, 2), (10, 10))
    for row, column in coordinates:
        targets[row, column] = 255

    pre = np.full((15, 15), 0.1, dtype=np.float64)
    pre[2, 2] = 0.9   # TP -> TP
    pre[2, 10] = 0.7  # TP -> FN
    pre[10, 2] = 0.3  # FN -> TP
    pre[10, 10] = 0.2  # FN -> FN

    post = np.full((15, 15), 0.1, dtype=np.float64)
    post[2, 2] = 0.8
    post[2, 10] = 0.4
    post[10, 2] = 0.6
    post[10, 10] = 0.5  # Strict threshold: exactly 0.5 remains background.
    return pre, post, targets


def test_all_four_transitions_scores_geometry_matches_and_aggregates() -> None:
    pre, post, targets = _four_transition_case()
    evaluator = TargetTransitionEvaluator(
        _protocol(), comparison_label=SOURCE_TO_POST
    )

    image = evaluator.update_probabilities(pre, post, targets, image_id="synthetic")
    summary = evaluator.compute()

    assert [record.transition for record in image.target_records] == [
        TP_TO_TP,
        TP_TO_FN,
        FN_TO_TP,
        FN_TO_FN,
    ]
    assert image.transition_counts.to_dict() == {
        TP_TO_TP: 1,
        TP_TO_FN: 1,
        FN_TO_TP: 1,
        FN_TO_FN: 1,
    }
    assert image.target_records[0].gt_area == 1
    assert image.target_records[0].gt_centroid == (2.0, 2.0)
    assert image.target_records[0].gt_bbox == (2, 2, 3, 3)
    assert image.target_records[0].pre_match is not None
    assert image.target_records[0].pre_match.centroid_distance == pytest.approx(0.0)
    assert image.target_records[1].post_match is None
    assert image.target_records[2].pre_match is None

    records = image.target_records
    assert [record.pre_target_score for record in records] == pytest.approx(
        [0.9, 0.7, 0.3, 0.2]
    )
    assert [record.post_target_score for record in records] == pytest.approx(
        [0.8, 0.4, 0.6, 0.5]
    )
    assert records[1].score_change == pytest.approx(-0.3)
    assert records[1].pre_margin_to_threshold == pytest.approx(0.2)
    assert records[1].post_margin_to_threshold == pytest.approx(-0.1)
    assert records[3].post_margin_to_threshold == pytest.approx(0.0)

    assert summary.comparison_label == SOURCE_TO_POST
    assert summary.image_count == 1
    assert summary.total_gt_targets == 4
    assert summary.ater.defined and summary.ater.value == pytest.approx(0.5)
    assert summary.atrr.defined and summary.atrr.value == pytest.approx(0.5)
    assert summary.ntg.defined and summary.ntg.value == pytest.approx(0.0)
    assert summary.net_target_gain_count == 0
    assert summary.ntg.numerator == summary.net_target_gain_count
    assert summary.score_definition.protocol_status == (
        SCORE_STATUS_CANDIDATE_NOT_FROZEN
    )
    json.dumps(image.to_dict(), ensure_ascii=False, sort_keys=True)
    json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True)


def test_ntg_is_normalised_and_absolute_net_gain_is_reported_separately() -> None:
    targets = np.zeros((12, 12), dtype=np.uint8)
    targets[2, 2] = 1
    targets[2, 8] = 1
    targets[8, 2] = 1
    targets[8, 8] = 1
    pre = np.zeros_like(targets, dtype=np.float64)
    post = np.zeros_like(targets, dtype=np.float64)
    post[2, 2] = 0.9
    post[2, 8] = 0.9
    post[8, 2] = 0.9
    evaluator = TargetTransitionEvaluator(_protocol())

    evaluator.update_probabilities(pre, post, targets)
    summary = evaluator.compute()

    assert summary.net_target_gain_count == 3
    assert summary.ntg.numerator == 3
    assert summary.ntg.denominator == 4
    assert summary.ntg.value == pytest.approx(0.75)


def test_zero_denominators_are_null_and_marked_undefined() -> None:
    targets = np.zeros((7, 7), dtype=np.uint8)
    targets[3, 3] = 1

    all_pre_tp = TargetTransitionEvaluator(_protocol())
    probability = np.zeros((7, 7), dtype=np.float64)
    probability[3, 3] = 0.9
    all_pre_tp.update_probabilities(probability, probability, targets)
    tp_summary = all_pre_tp.compute()
    assert tp_summary.ater.defined
    assert tp_summary.ater.value == pytest.approx(0.0)
    assert not tp_summary.atrr.defined
    assert tp_summary.atrr.value is None
    assert tp_summary.atrr.denominator == 0

    all_pre_fn = TargetTransitionEvaluator(_protocol())
    empty_prediction = np.zeros((7, 7), dtype=np.float64)
    all_pre_fn.update_probabilities(empty_prediction, empty_prediction, targets)
    fn_summary = all_pre_fn.compute()
    assert not fn_summary.ater.defined
    assert fn_summary.ater.value is None
    assert fn_summary.atrr.defined
    assert fn_summary.atrr.value == pytest.approx(0.0)

    no_gt = TargetTransitionEvaluator(_protocol())
    no_gt.update_probabilities(
        empty_prediction, empty_prediction, np.zeros_like(targets)
    )
    empty_summary = no_gt.compute()
    assert empty_summary.total_gt_targets == 0
    assert empty_summary.transition_counts.total == 0
    assert empty_summary.net_target_gain_count == 0
    for metric in (empty_summary.ater, empty_summary.atrr, empty_summary.ntg):
        assert not metric.defined
        assert metric.value is None
        assert metric.denominator == 0


def test_strict_threshold_and_strict_centroid_gate_match_frozen_evaluator() -> None:
    targets = np.zeros((8, 8), dtype=np.uint8)
    targets[2, 4] = 1
    pre = np.zeros((8, 8), dtype=np.float64)
    post = np.zeros((8, 8), dtype=np.float64)
    pre[2, 1] = 0.9  # Centroid distance exactly 3: not eligible.
    post[2, 2] = 0.5  # Exactly threshold: not foreground.

    both_fn = evaluate_target_transitions(pre, post, targets, protocol=_protocol())
    assert both_fn.target_records[0].transition == FN_TO_FN
    # The default score dilation also uses a strict distance boundary.
    assert both_fn.target_records[0].pre_target_score == pytest.approx(0.0)
    assert both_fn.target_records[0].post_margin_to_threshold == pytest.approx(0.0)

    post[2, 2] = 0.5001  # Distance 2 and strictly above threshold.
    recovered = evaluate_target_transitions(pre, post, targets, protocol=_protocol())
    assert recovered.target_records[0].transition == FN_TO_TP


def test_matching_is_one_to_one_for_competing_targets_and_predictions() -> None:
    protocol = _protocol()

    two_targets = np.zeros((7, 7), dtype=np.uint8)
    two_targets[3, 2] = 1
    two_targets[3, 4] = 1
    one_prediction = np.zeros((7, 7), dtype=np.float64)
    one_prediction[3, 3] = 0.9
    result = evaluate_target_transitions(
        one_prediction, one_prediction, two_targets, protocol=protocol
    )
    assert sum(record.pre_state == "TP" for record in result.target_records) == 1
    assert sum(record.pre_state == "FN" for record in result.target_records) == 1

    one_target = np.zeros((9, 9), dtype=np.uint8)
    one_target[4, 4] = 1
    two_predictions = np.zeros((9, 9), dtype=np.float64)
    two_predictions[4, 3] = 0.9
    two_predictions[4, 5] = 0.9
    result = evaluate_target_transitions(
        two_predictions, two_predictions, one_target, protocol=protocol
    )
    assert result.target_records[0].pre_state == "TP"
    assert result.pre_prediction_component_count == 2
    assert result.pre_unmatched_prediction_count == 1


def test_score_reducer_is_injectable_but_cannot_change_discrete_states() -> None:
    targets = np.zeros((8, 8), dtype=np.uint8)
    targets[3, 3:5] = 1
    pre = np.zeros((8, 8), dtype=np.float64)
    post = np.zeros((8, 8), dtype=np.float64)
    pre[3, 3] = 0.9
    pre[3, 4] = 0.3
    post[3, 3] = 0.8
    post[3, 4] = 0.2

    default = evaluate_target_transitions(pre, post, targets, protocol=_protocol())

    def gt_component_mean(
        probability_map: np.ndarray, context: TargetScoreContext
    ) -> float:
        return float(probability_map[context.target_component_mask].mean())

    definition = TargetScoreDefinition(
        name="gt_component_mean_probability",
        description="mean probability over the exact GT component",
        protocol_status="explicit_test_definition",
        parameters=(("reduction", "mean"),),
    )
    custom = evaluate_target_transitions(
        pre,
        post,
        targets,
        protocol=_protocol(),
        score_reducer=gt_component_mean,
        score_definition=definition,
    )

    assert custom.transition_counts == default.transition_counts
    assert custom.target_records[0].transition == default.target_records[0].transition
    assert default.target_records[0].pre_target_score == pytest.approx(0.9)
    assert custom.target_records[0].pre_target_score == pytest.approx(0.6)
    assert custom.score_definition == definition
    with pytest.raises(ValueError, match="explicit score_definition"):
        evaluate_target_transitions(
            pre,
            post,
            targets,
            protocol=_protocol(),
            score_reducer=gt_component_mean,
        )


def test_repeated_evaluation_is_deterministic_and_labels_are_optional_metadata() -> None:
    pre, post, targets = _four_transition_case()
    first = evaluate_target_transitions(
        pre,
        post,
        targets,
        protocol=_protocol(),
        comparison_label=TENT_PRE_TO_POST,
        image_id="same-image",
    )
    second = evaluate_target_transitions(
        pre,
        post,
        targets,
        protocol=_protocol(),
        comparison_label=TENT_PRE_TO_POST,
        image_id="same-image",
    )

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.comparison_label == TENT_PRE_TO_POST

    unlabeled = evaluate_target_transitions(pre, post, targets, protocol=_protocol())
    assert unlabeled.comparison_label is None


def test_ground_truth_is_only_materialised_after_both_predictions() -> None:
    access_order: list[str] = []

    class ArrayProbe:
        def __init__(self, name: str, array: np.ndarray) -> None:
            self.name = name
            self.array = array

        def __array__(self, dtype: Any = None) -> np.ndarray:
            access_order.append(self.name)
            return np.asarray(self.array, dtype=dtype)

    pre = np.zeros((5, 5), dtype=np.float64)
    post = np.zeros((5, 5), dtype=np.float64)
    targets = np.zeros((5, 5), dtype=np.uint8)
    targets[2, 2] = 1
    pre_before = pre.copy()
    post_before = post.copy()
    target_before = targets.copy()

    evaluate_target_transitions(
        ArrayProbe("pre", pre),
        ArrayProbe("post", post),
        ArrayProbe("ground_truth", targets),
        protocol=_protocol(),
    )

    assert access_order == ["pre", "post", "ground_truth"]
    assert np.array_equal(pre, pre_before)
    assert np.array_equal(post, post_before)
    assert np.array_equal(targets, target_before)


def test_accumulator_rejects_duplicate_image_ids_and_reset_is_complete() -> None:
    pre, post, targets = _four_transition_case()
    evaluator = TargetTransitionEvaluator(_protocol())
    evaluator.update_probabilities(pre, post, targets, image_id="image-a")
    with pytest.raises(ValueError, match="Duplicate image_id"):
        evaluator.update_probabilities(pre, post, targets, image_id="image-a")

    evaluator.reset()
    assert evaluator.image_results == ()
    assert evaluator.compute().image_count == 0
    evaluator.update_probabilities(pre, post, targets, image_id="image-a")
