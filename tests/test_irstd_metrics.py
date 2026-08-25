from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from metrics.irstd_metrics import (
    IRSTDEvaluationProtocol,
    UnifiedResearchEvaluator,
    probabilities_from_logits,
)


def _protocol(
    *,
    fixed: float = 0.5,
    froc: tuple[float, ...] = (0.0, 0.5, 0.7, 1.0),
    connectivity: int = 2,
) -> IRSTDEvaluationProtocol:
    return IRSTDEvaluationProtocol(
        fixed_probability_threshold=fixed,
        froc_probability_thresholds=froc,
        connectivity=connectivity,
        max_centroid_distance=3.0,
    )


def test_logit_zero_maps_to_probability_half_and_extremes_are_finite() -> None:
    probabilities = probabilities_from_logits(np.array([-1000.0, 0.0, 1000.0]))

    assert np.isfinite(probabilities).all()
    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] == pytest.approx(1.0)


def test_perfect_prediction_has_perfect_fixed_metrics() -> None:
    probabilities = np.full((6, 6), 0.1)
    targets = np.zeros((6, 6), dtype=np.uint8)
    probabilities[2:4, 2:4] = 0.9
    targets[2:4, 2:4] = 1
    evaluator = UnifiedResearchEvaluator(_protocol())

    evaluator.update_probabilities(probabilities, targets)
    result = evaluator.compute().fixed

    assert result.pixel.pixel_accuracy == pytest.approx(1.0)
    assert result.pixel.intersection_over_union == pytest.approx(1.0)
    assert result.pixel.precision == pytest.approx(1.0)
    assert result.pixel.recall == pytest.approx(1.0)
    assert result.target.detection_probability == pytest.approx(1.0)
    assert result.target.false_positives_per_image == pytest.approx(0.0)
    assert result.target.false_alarm_pixel_rate == pytest.approx(0.0)


def test_evaluator_enforces_one_to_one_target_matching() -> None:
    probabilities = np.zeros((7, 7), dtype=np.float64)
    targets = np.zeros((7, 7), dtype=np.uint8)
    probabilities[3, 3] = 0.9
    targets[3, 2] = 1
    targets[3, 4] = 1
    evaluator = UnifiedResearchEvaluator(_protocol())

    evaluator.update_probabilities(probabilities, targets)
    target_metrics = evaluator.compute().fixed.target

    assert target_metrics.detected_targets == 1
    assert target_metrics.total_targets == 2
    assert target_metrics.detection_probability == pytest.approx(0.5)


def test_exact_three_pixel_centroid_distance_is_not_a_detection() -> None:
    probabilities = np.zeros((8, 8), dtype=np.float64)
    targets = np.zeros((8, 8), dtype=np.uint8)
    probabilities[2, 1] = 0.9
    targets[2, 4] = 1
    evaluator = UnifiedResearchEvaluator(_protocol())

    evaluator.update_probabilities(probabilities, targets)
    metrics = evaluator.compute().fixed.target

    assert metrics.detected_targets == 0
    assert metrics.false_positive_components == 1


def test_empty_prediction_and_empty_ground_truth_are_finite() -> None:
    evaluator = UnifiedResearchEvaluator(_protocol())
    evaluator.update_probabilities(np.zeros((4, 4)), np.zeros((4, 4)))

    fixed = evaluator.compute().fixed
    values = np.asarray(
        [
            fixed.pixel.pixel_accuracy,
            fixed.pixel.intersection_over_union,
            fixed.pixel.precision,
            fixed.pixel.recall,
            fixed.target.detection_probability,
            fixed.target.false_positives_per_image,
            fixed.target.false_alarm_pixel_rate,
        ]
    )
    assert np.isfinite(values).all()
    assert fixed.pixel.intersection_over_union == pytest.approx(1.0)
    assert fixed.pixel.precision == pytest.approx(1.0)
    assert fixed.pixel.recall == pytest.approx(1.0)
    assert fixed.target.detection_probability == pytest.approx(0.0)


def test_empty_ground_truth_still_counts_false_alarm_area_and_components() -> None:
    probabilities = np.zeros((4, 4), dtype=np.float64)
    probabilities[0, 0] = 0.9
    evaluator = UnifiedResearchEvaluator(_protocol())

    evaluator.update_probabilities(probabilities, np.zeros((4, 4)))
    target = evaluator.compute().fixed.target

    assert target.false_positive_components == 1
    assert target.false_alarm_pixels == 1
    assert target.false_positives_per_image == pytest.approx(1.0)
    assert target.false_alarm_pixel_rate == pytest.approx(1.0 / 16.0)


def test_probability_threshold_scan_has_expected_known_ordering() -> None:
    probabilities = np.zeros((8, 8), dtype=np.float64)
    targets = np.zeros((8, 8), dtype=np.uint8)
    probabilities[1, 1] = 0.9
    targets[1, 1] = 1
    probabilities[6, 6] = 0.6
    evaluator = UnifiedResearchEvaluator(_protocol())

    evaluator.update_probabilities(probabilities, targets)
    froc = evaluator.compute().froc

    assert [entry.probability_threshold for entry in froc] == [0.0, 0.5, 0.7, 1.0]
    assert [entry.pixel.predicted_positive_pixels for entry in froc] == [2, 2, 1, 0]
    assert [entry.target.false_positive_components for entry in froc] == [1, 1, 0, 0]
    assert [entry.target.detection_probability for entry in froc] == [1.0, 1.0, 1.0, 0.0]


def test_batch_and_channel_shapes_accumulate_per_image() -> None:
    logits = np.full((2, 1, 4, 4), -10.0)
    targets = np.zeros((2, 1, 4, 4), dtype=np.uint8)
    logits[:, :, 1, 1] = 10.0
    targets[:, :, 1, 1] = 1
    evaluator = UnifiedResearchEvaluator(_protocol())

    evaluator.update_logits(logits, targets)
    fixed = evaluator.compute().fixed

    assert fixed.target.image_count == 2
    assert fixed.target.total_targets == 2
    assert fixed.target.detected_targets == 2


def test_protocol_is_canonical_and_immutable() -> None:
    protocol = IRSTDEvaluationProtocol(
        froc_probability_thresholds=(1.0, 0.5, 0.5, 0.0)
    )

    assert protocol.froc_probability_thresholds == (0.0, 0.5, 1.0)
    with pytest.raises(FrozenInstanceError):
        protocol.connectivity = 1  # type: ignore[misc]


def test_invalid_probability_inputs_are_rejected() -> None:
    evaluator = UnifiedResearchEvaluator(_protocol())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluator.update_probabilities(np.full((2, 2), 1.1), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="same normalised shape"):
        evaluator.update_probabilities(np.zeros((2, 2)), np.zeros((3, 3)))


def test_result_is_json_serialisable_shape() -> None:
    evaluator = UnifiedResearchEvaluator(_protocol())
    evaluator.update_probabilities(np.zeros((2, 2)), np.zeros((2, 2)))

    result = evaluator.compute().to_dict()

    assert result["fixed"]["probability_threshold"] == 0.5
    assert len(result["froc"]) == 4
