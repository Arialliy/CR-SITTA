"""One-to-one target matching for IRSTD connected components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from .connected_components import ConnectedComponent


@dataclass(frozen=True)
class TargetMatch:
    """A matched predicted/ground-truth component pair."""

    prediction_index: int
    target_index: int
    centroid_distance: float


@dataclass(frozen=True)
class TargetMatchResult:
    """A complete, one-to-one assignment result."""

    matches: Tuple[TargetMatch, ...]
    unmatched_prediction_indices: Tuple[int, ...]
    unmatched_target_indices: Tuple[int, ...]

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        return len(self.unmatched_prediction_indices)

    @property
    def false_negatives(self) -> int:
        return len(self.unmatched_target_indices)


def centroid_distance_matrix(
    predictions: Sequence[ConnectedComponent],
    targets: Sequence[ConnectedComponent],
) -> NDArray[np.float64]:
    """Return pairwise Euclidean distances in row/column coordinates."""

    if not predictions or not targets:
        return np.empty((len(predictions), len(targets)), dtype=np.float64)
    prediction_centroids = np.asarray(
        [component.centroid for component in predictions], dtype=np.float64
    )
    target_centroids = np.asarray(
        [component.centroid for component in targets], dtype=np.float64
    )
    differences = prediction_centroids[:, None, :] - target_centroids[None, :, :]
    return np.linalg.norm(differences, axis=2)


def match_components(
    predictions: Sequence[ConnectedComponent],
    targets: Sequence[ConnectedComponent],
    *,
    max_centroid_distance: float = 3.0,
) -> TargetMatchResult:
    """Match components one-to-one under a strict centroid-distance gate.

    A pair is eligible exactly when ``distance < max_centroid_distance``.  The
    Hungarian assignment first maximises the number of eligible pairs and then
    minimises their total distance.  Consequently, one prediction cannot hit
    multiple targets and one target cannot be credited multiple times.
    """

    if not np.isfinite(max_centroid_distance) or max_centroid_distance <= 0:
        raise ValueError("max_centroid_distance must be a positive finite value.")

    prediction_count = len(predictions)
    target_count = len(targets)
    if prediction_count == 0 or target_count == 0:
        return TargetMatchResult(
            matches=(),
            unmatched_prediction_indices=tuple(range(prediction_count)),
            unmatched_target_indices=tuple(range(target_count)),
        )

    distances = centroid_distance_matrix(predictions, targets)
    eligible = distances < max_centroid_distance

    # The number of assignments is fixed at min(P, T).  A penalty greater than
    # the largest possible sum of all eligible distances therefore maximises
    # eligible-cardinality before distance is considered.
    assignment_count = min(prediction_count, target_count)
    invalid_penalty = (assignment_count + 1) * (max_centroid_distance + 1.0)
    costs = np.where(eligible, distances, invalid_penalty)
    prediction_rows, target_columns = linear_sum_assignment(costs)

    matches = []
    matched_predictions = set()
    matched_targets = set()
    for prediction_index, target_index in zip(prediction_rows, target_columns):
        if not eligible[prediction_index, target_index]:
            continue
        matched_predictions.add(int(prediction_index))
        matched_targets.add(int(target_index))
        matches.append(
            TargetMatch(
                prediction_index=int(prediction_index),
                target_index=int(target_index),
                centroid_distance=float(distances[prediction_index, target_index]),
            )
        )

    matches.sort(key=lambda match: (match.target_index, match.prediction_index))
    return TargetMatchResult(
        matches=tuple(matches),
        unmatched_prediction_indices=tuple(
            index
            for index in range(prediction_count)
            if index not in matched_predictions
        ),
        unmatched_target_indices=tuple(
            index for index in range(target_count) if index not in matched_targets
        ),
    )


__all__ = [
    "TargetMatch",
    "TargetMatchResult",
    "centroid_distance_matrix",
    "match_components",
]
