from __future__ import annotations

import numpy as np
import pytest

from metrics.connected_components import (
    ConnectedComponent,
    label_connected_components,
)
from metrics.target_matching import match_components


def _component(row: float, column: float, label: int = 1) -> ConnectedComponent:
    return ConnectedComponent(
        label=label,
        area=1,
        centroid=(row, column),
        bbox=(int(row), int(column), int(row) + 1, int(column) + 1),
    )


def test_each_prediction_and_target_can_only_be_matched_once() -> None:
    predictions = [_component(2.0, 3.0)]
    targets = [_component(2.0, 2.0, 1), _component(2.0, 4.0, 2)]

    result = match_components(predictions, targets)

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 1
    assert len({match.prediction_index for match in result.matches}) == 1
    assert len({match.target_index for match in result.matches}) == 1


def test_matching_maximises_valid_pair_count_before_distance() -> None:
    # P0 can match either target; P1 can only match T0.  A local nearest-first
    # choice for P0 could lose a valid detection, while global assignment does not.
    predictions = [_component(0.0, 0.0, 1), _component(0.0, 4.0, 2)]
    targets = [_component(0.0, 1.1, 1), _component(0.0, -1.0, 2)]

    result = match_components(predictions, targets, max_centroid_distance=3.0)

    assert result.true_positives == 2
    assert {(match.prediction_index, match.target_index) for match in result.matches} == {
        (0, 1),
        (1, 0),
    }


def test_distance_gate_is_strict_at_exactly_three_pixels() -> None:
    result = match_components(
        [_component(0.0, 0.0)],
        [_component(0.0, 3.0)],
        max_centroid_distance=3.0,
    )

    assert result.true_positives == 0
    assert result.unmatched_prediction_indices == (0,)
    assert result.unmatched_target_indices == (0,)


@pytest.mark.parametrize(
    ("prediction_count", "target_count"),
    [(0, 0), (1, 0), (0, 1)],
)
def test_empty_component_sets_are_safe(
    prediction_count: int, target_count: int
) -> None:
    predictions = [_component(0.0, float(index), index + 1) for index in range(prediction_count)]
    targets = [_component(0.0, float(index), index + 1) for index in range(target_count)]

    result = match_components(predictions, targets)

    assert result.true_positives == 0
    assert result.unmatched_prediction_indices == tuple(range(prediction_count))
    assert result.unmatched_target_indices == tuple(range(target_count))


def test_connectivity_is_explicit_and_changes_diagonal_grouping() -> None:
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 1

    four_neighbour = label_connected_components(mask, connectivity=1)
    eight_neighbour = label_connected_components(mask, connectivity=2)

    assert len(four_neighbour.components) == 2
    assert len(eight_neighbour.components) == 1
    assert eight_neighbour.components[0].area == 2


def test_component_filter_relabels_contiguously_and_empty_mask_is_safe() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0, 0] = 1
    mask[3:5, 3:5] = 1

    extraction = label_connected_components(mask, connectivity=2, min_area=2)
    empty = label_connected_components(np.zeros((2, 2)), connectivity=2)

    assert len(extraction.components) == 1
    assert extraction.components[0].label == 1
    assert extraction.components[0].area == 4
    assert set(np.unique(extraction.labels)) == {0, 1}
    assert empty.components == ()
    assert not empty.labels.any()


def test_invalid_connectivity_is_rejected() -> None:
    with pytest.raises(ValueError, match="connectivity"):
        label_connected_components(np.zeros((2, 2)), connectivity=3)
