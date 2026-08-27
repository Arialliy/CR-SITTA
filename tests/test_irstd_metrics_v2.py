from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from metrics.irstd_metrics import IRSTDEvaluationProtocol
from metrics.irstd_metrics_v2 import (
    FORMAL_FROC_THRESHOLDS_V2,
    NIOU_EMPTY_UNION_CONTRIBUTION_V2,
    STRICT_THRESHOLD_RULE_V2,
    UnifiedResearchEvaluatorV2,
    assert_froc_endpoint_conservation_v2,
    check_froc_endpoint_conservation_v2,
)


def test_default_contract_is_probability_only_strict_and_complete_21_points() -> None:
    probabilities = np.asarray([[0.0, 0.5, 0.500001, 1.0]], dtype=np.float64)
    targets = np.zeros_like(probabilities, dtype=np.uint8)
    evaluator = UnifiedResearchEvaluatorV2()

    evaluator.update_probabilities(probabilities, targets)
    result = evaluator.compute()

    assert result.threshold_rule == STRICT_THRESHOLD_RULE_V2
    assert tuple(point.probability_threshold for point in result.froc) == (
        FORMAL_FROC_THRESHOLDS_V2
    )
    assert len(result.froc) == 21
    assert result.fixed.probability_threshold == 0.5
    # Strict thresholding excludes exactly 0.5 and includes values above it.
    assert result.fixed.pixel.predicted_positive_pixels == 2
    # Probability 1.0 is also excluded by the strict threshold at endpoint 1.
    assert result.froc[-1].pixel.predicted_positive_pixels == 0


def test_fixed_global_iou_pd_and_both_false_alarm_axes_are_explicit() -> None:
    probabilities = np.zeros((5, 5), dtype=np.float64)
    targets = np.zeros((5, 5), dtype=np.uint8)
    probabilities[1, 1] = 0.9
    targets[1, 1] = 1
    probabilities[4, 4] = 0.8
    evaluator = UnifiedResearchEvaluatorV2()

    evaluator.update_probabilities(probabilities, targets)
    result = evaluator.compute()
    fixed = result.fixed

    assert fixed.pixel.intersection_over_union == pytest.approx(0.5)
    assert fixed.detection_probability == pytest.approx(1.0)
    assert fixed.false_positive_components == 1
    assert fixed.fppi == pytest.approx(1.0)  # components / image
    assert fixed.false_alarm_pixels == 1
    assert fixed.false_alarm_pixel_rate == pytest.approx(1.0 / 25.0)
    assert result.froc_axis_semantics.fppi_name == "fppi"
    assert (
        result.froc_axis_semantics.fppi_formula
        == "false_positive_components / image_count"
    )
    assert (
        result.froc_axis_semantics.false_alarm_pixel_rate_formula
        == "false_alarm_pixels / total_image_pixels"
    )


def test_niou_is_arithmetic_mean_with_auditable_empty_union_zero() -> None:
    probabilities = np.zeros((3, 4, 4), dtype=np.float64)
    targets = np.zeros((3, 4, 4), dtype=np.uint8)

    # Image 0: perfect foreground IoU = 1.
    probabilities[0, 1, 1] = 0.9
    targets[0, 1, 1] = 1
    # Image 1: intersection=1, union=2, foreground IoU = 0.5.
    probabilities[1, 1, 1] = 0.9
    probabilities[1, 3, 3] = 0.9
    targets[1, 1, 1] = 1
    # Image 2: empty/empty union, frozen contribution = 0.

    evaluator = UnifiedResearchEvaluatorV2()
    evaluator.update_probabilities(
        probabilities, targets, image_ids=("perfect", "partial", "empty")
    )
    niou = evaluator.compute().normalized_iou

    assert niou.normalized_iou == pytest.approx((1.0 + 0.5 + 0.0) / 3.0)
    assert niou.contribution_sum == pytest.approx(1.5)
    assert niou.averaging_denominator_image_count == 3
    assert niou.nonempty_union_image_count == 2
    assert niou.empty_union_image_count == 1
    assert niou.empty_union_contribution == NIOU_EMPTY_UNION_CONTRIBUTION_V2
    assert [item.image_index for item in niou.per_image_contributions] == [0, 1, 2]
    assert [item.image_id for item in niou.per_image_contributions] == [
        "perfect",
        "partial",
        "empty",
    ]
    assert [item.intersection_pixels for item in niou.per_image_contributions] == [
        1,
        1,
        0,
    ]
    assert [item.union_pixels for item in niou.per_image_contributions] == [1, 2, 0]
    assert [
        item.foreground_iou_contribution for item in niou.per_image_contributions
    ] == [1.0, 0.5, 0.0]
    assert niou.per_image_contributions[-1].empty_union is True


def test_empty_image_has_global_empty_iou_one_but_samplewise_niou_zero() -> None:
    evaluator = UnifiedResearchEvaluatorV2()
    evaluator.update_probabilities(np.zeros((4, 4)), np.zeros((4, 4)))

    result = evaluator.compute()

    # The established global metric keeps its legacy empty/empty convention.
    assert result.fixed.pixel.intersection_over_union == pytest.approx(1.0)
    # V2 explicitly freezes the samplewise foreground-IoU contribution to 0.
    assert result.normalized_iou.normalized_iou == pytest.approx(0.0)
    assert result.fixed.detection_probability == pytest.approx(0.0)
    assert result.fixed.fppi == pytest.approx(0.0)
    assert result.fixed.false_alarm_pixel_rate == pytest.approx(0.0)
    assert result.assert_endpoint_conservation().is_conserved


@pytest.mark.parametrize(
    "bad_probabilities",
    [
        np.full((2, 2), -0.01),
        np.full((2, 2), 1.01),
        np.asarray([[np.nan, 0.0]]),
        np.asarray([[np.inf, 0.0]]),
    ],
)
def test_probability_range_and_finiteness_are_rejected(
    bad_probabilities: np.ndarray,
) -> None:
    evaluator = UnifiedResearchEvaluatorV2()
    with pytest.raises(ValueError):
        evaluator.update_probabilities(
            bad_probabilities, np.zeros_like(bad_probabilities)
        )


def test_shape_type_and_image_id_contracts_are_rejected() -> None:
    evaluator = UnifiedResearchEvaluatorV2()
    with pytest.raises(ValueError, match="must have shape"):
        evaluator.update_probabilities(np.zeros((1, 1, 1, 2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="same normalized shape"):
        evaluator.update_probabilities(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="empty dimensions"):
        evaluator.update_probabilities(np.empty((0, 2, 2)), np.empty((0, 2, 2)))
    with pytest.raises(ValueError, match="numeric or boolean"):
        evaluator.update_probabilities(
            np.asarray([["not-a-probability"]]), np.zeros((1, 1))
        )
    with pytest.raises(ValueError, match="image_ids length"):
        evaluator.update_probabilities(
            np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), image_ids=("only-one",)
        )
    with pytest.raises(ValueError, match="must be a string"):
        evaluator.update_probabilities(
            np.zeros((2, 2)), np.zeros((2, 2)), image_ids=(123,)  # type: ignore[arg-type]
        )


def test_repeated_compute_is_idempotent_and_does_not_duplicate_records() -> None:
    evaluator = UnifiedResearchEvaluatorV2()
    evaluator.update_probabilities(
        np.asarray([[0.0, 0.9], [0.0, 0.0]]),
        np.asarray([[0, 1], [0, 0]]),
        image_ids=("one",),
    )

    first = evaluator.compute()
    second = evaluator.compute()

    assert first == second
    assert len(second.normalized_iou.per_image_contributions) == 1
    assert json.dumps(second.to_dict(), sort_keys=True)


def test_reset_clears_global_and_per_image_state_and_restarts_indices() -> None:
    evaluator = UnifiedResearchEvaluatorV2()
    evaluator.update_probabilities(np.ones((2, 2)), np.ones((2, 2)))
    assert evaluator.compute().fixed.image_count == 1

    evaluator.reset()
    empty = evaluator.compute()

    assert empty.fixed.image_count == 0
    assert empty.fixed.total_image_pixels == 0
    assert empty.normalized_iou.normalized_iou == 0.0
    assert empty.normalized_iou.averaging_denominator_image_count == 0
    assert empty.normalized_iou.per_image_contributions == ()
    assert empty.assert_endpoint_conservation().is_conserved

    evaluator.update_probabilities(np.zeros((2, 2)), np.zeros((2, 2)))
    assert (
        evaluator.compute().normalized_iou.per_image_contributions[0].image_index
        == 0
    )


def test_batched_channel_input_equals_sequential_image_updates() -> None:
    probabilities = np.zeros((3, 1, 5, 5), dtype=np.float32)
    targets = np.zeros((3, 1, 5, 5), dtype=np.uint8)
    probabilities[0, 0, 1, 1] = 0.9
    targets[0, 0, 1, 1] = 1
    probabilities[1, 0, 2, 2] = 0.8
    targets[1, 0, 2, 3] = 1
    probabilities[2, 0, 4, 4] = 0.7
    ids = ("a", "b", "c")

    batched = UnifiedResearchEvaluatorV2()
    batched.update_probabilities(probabilities, targets, image_ids=ids)

    sequential = UnifiedResearchEvaluatorV2()
    for probability, target, image_id in zip(probabilities, targets, ids):
        sequential.update_probabilities(
            probability, target, image_ids=(image_id,)
        )

    assert batched.compute() == sequential.compute()


def test_nonformal_froc_grid_is_rejected_fail_closed() -> None:
    protocol = IRSTDEvaluationProtocol(froc_probability_thresholds=(0.0, 0.5, 1.0))
    with pytest.raises(ValueError, match="exact formal 21-point"):
        UnifiedResearchEvaluatorV2(protocol)


def test_endpoint_conservation_public_helpers_and_tamper_detection() -> None:
    probabilities = np.linspace(0.0, 1.0, 16, dtype=np.float64).reshape(4, 4)
    targets = np.zeros((4, 4), dtype=np.uint8)
    targets[2, 2] = 1
    evaluator = UnifiedResearchEvaluatorV2()
    evaluator.update_probabilities(probabilities, targets)
    result = evaluator.compute()

    report = check_froc_endpoint_conservation_v2(result)
    assert report.is_conserved
    assert report.failed_checks == ()
    assert assert_froc_endpoint_conservation_v2(result) == report

    endpoint = result.froc[-1]
    bad_pixel = replace(endpoint.pixel, predicted_positive_pixels=1)
    bad_endpoint = replace(endpoint, pixel=bad_pixel)
    tampered = replace(result, froc=result.froc[:-1] + (bad_endpoint,))
    bad_report = tampered.check_endpoint_conservation()

    assert not bad_report.is_conserved
    assert not bad_report.threshold_one_has_no_predicted_positive_pixels
    with pytest.raises(AssertionError, match="endpoint/count conservation failed"):
        tampered.assert_endpoint_conservation()


def test_output_intentionally_has_no_auc_field() -> None:
    result = UnifiedResearchEvaluatorV2().compute()

    assert not hasattr(result, "auc")
    assert "auc" not in result.to_dict()
