from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from metrics.irstd_metrics import IRSTDEvaluationProtocol, UnifiedResearchEvaluator
from metrics.target_transitions import (
    SCORE_STATUS_CANDIDATE_NOT_FROZEN,
    TargetScoreContext,
    TargetScoreDefinition,
)
from metrics.three_stage_transitions import (
    BG_TO_FP,
    COMPARISON_IDS,
    FP_TO_BG,
    FP_TO_FP,
    SCORE_STATUS_FORMAL_FROZEN_V1,
    SNAPSHOT_IDS,
    SOURCE_PRE,
    SOURCE_TO_TENT_POST,
    SOURCE_TO_TENT_PRE,
    TENT_POST,
    TENT_PRE,
    TENT_PRE_TO_TENT_POST,
    ThreeStageTransitionEvaluator,
    ThreeStageTransitionProtocolError,
)


def _protocol() -> IRSTDEvaluationProtocol:
    return IRSTDEvaluationProtocol(
        fixed_probability_threshold=0.5,
        froc_probability_thresholds=(0.0, 0.5, 1.0),
        connectivity=2,
        max_centroid_distance=3.0,
        min_component_area=1,
    )


def _gt_component_max(
    probability_map: np.ndarray, context: TargetScoreContext
) -> float:
    return float(np.max(probability_map[context.target_component_mask]))


def _formal_score_definition() -> TargetScoreDefinition:
    return TargetScoreDefinition(
        name="exact_gt_component_max_probability_v1",
        description="maximum probability on the exact GT component",
        protocol_status=SCORE_STATUS_FORMAL_FROZEN_V1,
        parameters=(
            ("spatial_support", "exact_gt_component"),
            ("reduction", "max"),
        ),
    )


def _evaluator(*, tent_ss: bool = False) -> ThreeStageTransitionEvaluator:
    return ThreeStageTransitionEvaluator(
        _protocol(),
        score_reducer=_gt_component_max,
        score_definition=_formal_score_definition(),
        tent_ss_identity_required=tent_ss,
    )


def _blank(size: int = 17) -> np.ndarray:
    return np.full((size, size), 0.1, dtype=np.float32)


def _by_id(values: Any, field: str) -> dict[str, Any]:
    return {getattr(value, field): value for value in values}


def test_formal_score_and_operating_protocol_are_fail_closed() -> None:
    candidate = TargetScoreDefinition(
        name="candidate",
        description="not frozen",
        protocol_status=SCORE_STATUS_CANDIDATE_NOT_FROZEN,
    )
    with pytest.raises(ThreeStageTransitionProtocolError, match="candidate_not_frozen"):
        ThreeStageTransitionEvaluator(
            _protocol(),
            score_reducer=_gt_component_max,
            score_definition=candidate,
        )

    merely_named = TargetScoreDefinition(
        name="named_but_not_frozen",
        description="does not satisfy the formal versioned contract",
        protocol_status="explicit_test_definition",
    )
    with pytest.raises(ThreeStageTransitionProtocolError, match="formal_frozen_v1"):
        ThreeStageTransitionEvaluator(
            _protocol(),
            score_reducer=_gt_component_max,
            score_definition=merely_named,
        )

    wrong_connectivity = IRSTDEvaluationProtocol(
        fixed_probability_threshold=0.5,
        connectivity=1,
        max_centroid_distance=3.0,
        min_component_area=1,
    )
    with pytest.raises(ThreeStageTransitionProtocolError, match="connectivity"):
        ThreeStageTransitionEvaluator(
            wrong_connectivity,
            score_reducer=_gt_component_max,
            score_definition=_formal_score_definition(),
        )


def test_joint_target_trajectory_uses_fixed_ids_and_same_gt_identity() -> None:
    target = np.zeros((17, 17), dtype=np.uint8)
    coordinates = ((3, 3), (3, 13), (13, 3), (13, 13))
    for row, column in coordinates:
        target[row, column] = 255

    source = _blank()
    source[3, 3] = 0.9
    source[3, 13] = 0.8

    tent_pre = _blank()
    tent_pre[3, 3] = 0.8
    tent_pre[13, 3] = 0.9

    tent_post = _blank()
    tent_post[13, 3] = 0.8
    tent_post[13, 13] = 0.9

    evaluator = _evaluator()
    image = evaluator.update_probabilities(
        source, tent_pre, tent_post, target, image_id="trajectory"
    )
    summary = evaluator.compute()

    assert tuple(
        value.comparison_label for value in image.target_transition_results
    ) == COMPARISON_IDS
    assert tuple(value.comparison_id for value in image.fp_transition_results) == (
        COMPARISON_IDS
    )
    assert tuple(value.snapshot_id for value in image.endpoint_counts) == SNAPSHOT_IDS
    assert [record.gt_target_id for record in image.joint_target_trajectories] == [
        1,
        2,
        3,
        4,
    ]
    assert [
        (record.source_state, record.tent_pre_state, record.tent_post_state)
        for record in image.joint_target_trajectories
    ] == [
        ("TP", "TP", "FN"),
        ("TP", "FN", "FN"),
        ("FN", "TP", "TP"),
        ("FN", "FN", "TP"),
    ]
    first = image.joint_target_trajectories[0]
    assert first.source_to_tent_pre_transition == "TP→TP"
    assert first.tent_pre_to_tent_post_transition == "TP→FN"
    assert first.source_to_tent_post_transition == "TP→FN"
    assert summary.image_count == 1
    assert tuple(
        value.comparison_label for value in summary.target_transition_summaries
    ) == COMPARISON_IDS
    json.dumps(image.to_dict(), ensure_ascii=False, sort_keys=True)
    json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True)


def test_fp_hungarian_transitions_preserve_geometry_and_conservation() -> None:
    target = np.zeros((17, 17), dtype=np.uint8)
    target[8, 8] = 255

    source = _blank()
    source[8, 8] = 0.9
    source[1, 1] = 0.9   # persistent FP
    source[1, 13] = 0.9  # removed FP

    tent_pre = _blank()
    tent_pre[8, 8] = 0.9
    tent_pre[1, 2] = 0.9   # persistent FP, shifted by one pixel
    tent_pre[13, 1] = 0.9  # new FP, removed at post

    tent_post = _blank()
    tent_post[8, 8] = 0.9
    tent_post[1, 2] = 0.9    # persistent FP
    tent_post[13, 13] = 0.9  # new FP

    evaluator = _evaluator()
    image = evaluator.update_probabilities(
        source, tent_pre, tent_post, target, image_id="fp-transitions"
    )
    summary = evaluator.compute()
    fp_results = _by_id(image.fp_transition_results, "comparison_id")
    fp_summaries = _by_id(summary.fp_transition_summaries, "comparison_id")
    for comparison_id in COMPARISON_IDS:
        result = fp_results[comparison_id]
        assert result.counts.to_dict() == {
            FP_TO_FP: 1,
            FP_TO_BG: 1,
            BG_TO_FP: 1,
        }
        assert result.counts.pre_fp == 2
        assert result.counts.post_fp == 2
        assert fp_summaries[comparison_id].counts == result.counts

    persistent = next(
        record
        for record in fp_results[SOURCE_TO_TENT_PRE].records
        if record.transition == FP_TO_FP
    )
    assert persistent.pre_component is not None
    assert persistent.post_component is not None
    assert persistent.pre_component.centroid == (1.0, 1.0)
    assert persistent.post_component.centroid == (1.0, 2.0)
    assert persistent.centroid_distance == pytest.approx(1.0)
    assert next(
        record
        for record in fp_results[SOURCE_TO_TENT_PRE].records
        if record.transition == FP_TO_BG
    ).post_component is None
    assert next(
        record
        for record in fp_results[SOURCE_TO_TENT_PRE].records
        if record.transition == BG_TO_FP
    ).pre_component is None


def test_fp_matching_uses_strict_less_than_three_pixel_gate() -> None:
    target = np.zeros((10, 10), dtype=np.uint8)
    source = _blank(10)
    source[1, 1] = 0.9
    tent_pre = _blank(10)
    tent_pre[1, 4] = 0.9  # centroid distance exactly 3: cannot persist
    tent_post = tent_pre.copy()

    image = _evaluator().update_probabilities(source, tent_pre, tent_post, target)
    result = _by_id(image.fp_transition_results, "comparison_id")[
        SOURCE_TO_TENT_PRE
    ]
    assert result.counts.fp_to_fp == 0
    assert result.counts.fp_to_bg == 1
    assert result.counts.bg_to_fp == 1


def test_endpoint_tp_fp_counts_match_unified_evaluator_fixed_operating_point() -> None:
    target = np.zeros((17, 17), dtype=np.uint8)
    target[5, 5] = 255
    target[11, 11] = 255

    source = _blank()
    source[5, 5] = 0.9
    source[1, 1] = 0.9
    tent_pre = _blank()
    tent_pre[11, 11] = 0.9
    tent_pre[1, 2] = 0.9
    tent_pre[1, 13] = 0.9
    tent_post = _blank()
    tent_post[5, 5] = 0.9
    tent_post[11, 11] = 0.9

    image = _evaluator().update_probabilities(source, tent_pre, tent_post, target)
    endpoints = _by_id(image.endpoint_counts, "snapshot_id")
    maps = {
        SOURCE_PRE: source,
        TENT_PRE: tent_pre,
        TENT_POST: tent_post,
    }
    for snapshot_id, probability in maps.items():
        unified = UnifiedResearchEvaluator(_protocol())
        unified.update_probabilities(probability, target)
        fixed = unified.compute().fixed.target
        endpoint = endpoints[snapshot_id]
        assert endpoint.true_positive_targets == fixed.detected_targets
        assert endpoint.gt_target_count == fixed.total_targets
        assert endpoint.false_positive_components == fixed.false_positive_components
        assert endpoint.false_negative_targets == (
            fixed.total_targets - fixed.detected_targets
        )

    target_results = _by_id(image.target_transition_results, "comparison_label")
    direct = target_results[SOURCE_TO_TENT_POST].transition_counts
    assert direct.pre_tp == endpoints[SOURCE_PRE].true_positive_targets
    assert direct.pre_fn == endpoints[SOURCE_PRE].false_negative_targets
    assert direct.tp_to_tp + direct.fn_to_tp == endpoints[TENT_POST].true_positive_targets


def test_tent_ss_requires_bit_exact_source_pre_and_enforces_identity_path() -> None:
    target = np.zeros((11, 11), dtype=np.uint8)
    target[5, 5] = 255
    source = _blank(11)
    source[5, 5] = 0.9
    tent_pre = source.copy()
    tent_post = _blank(11)

    evaluator = _evaluator(tent_ss=True)
    image = evaluator.update_probabilities(
        source, tent_pre, tent_post, target, image_id="tent-ss"
    )
    summary = evaluator.compute()
    assert image.tent_ss_source_tent_pre_bit_exact is True
    assert summary.tent_ss_identity_verified is True
    target_results = _by_id(image.target_transition_results, "comparison_label")
    identity = target_results[SOURCE_TO_TENT_PRE].transition_counts
    assert identity.tp_to_fn == 0
    assert identity.fn_to_tp == 0
    assert all(
        record.source_state == record.tent_pre_state
        and record.source_target_score == record.tent_pre_target_score
        and record.source_match == record.tent_pre_match
        for record in image.joint_target_trajectories
    )
    assert (
        target_results[TENT_PRE_TO_TENT_POST].transition_counts
        == target_results[SOURCE_TO_TENT_POST].transition_counts
    )

    mismatch = tent_pre.copy()
    mismatch[0, 0] = np.nextafter(
        mismatch[0, 0], np.float32(1.0), dtype=np.float32
    )
    failing = _evaluator(tent_ss=True)
    with pytest.raises(ThreeStageTransitionProtocolError, match="bit-exact"):
        failing.update_probabilities(source, mismatch, tent_post, target)
    assert failing.image_results == ()


def test_all_three_predictions_are_materialised_before_ground_truth() -> None:
    access_order: list[str] = []

    class ArrayProbe:
        def __init__(self, name: str, value: np.ndarray) -> None:
            self.name = name
            self.value = value

        def __array__(self, dtype: Any = None) -> np.ndarray:
            access_order.append(self.name)
            return np.asarray(self.value, dtype=dtype)

    probability = _blank(7)
    target = np.zeros((7, 7), dtype=np.uint8)
    target[3, 3] = 255
    _evaluator().update_probabilities(
        ArrayProbe("source", probability),
        ArrayProbe("tent_pre", probability),
        ArrayProbe("tent_post", probability),
        ArrayProbe("ground_truth", target),
    )
    assert access_order == ["source", "tent_pre", "tent_post", "ground_truth"]


def test_duplicate_ids_and_reset_are_complete() -> None:
    probability = _blank(7)
    target = np.zeros((7, 7), dtype=np.uint8)
    evaluator = _evaluator()
    evaluator.update_probabilities(
        probability, probability, probability, target, image_id="same"
    )
    with pytest.raises(ValueError, match="Duplicate image_id"):
        evaluator.update_probabilities(
            probability, probability, probability, target, image_id="same"
        )
    evaluator.reset()
    assert evaluator.image_results == ()
    evaluator.update_probabilities(
        probability, probability, probability, target, image_id="same"
    )
