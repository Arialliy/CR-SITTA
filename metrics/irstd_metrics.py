"""Unified probability-space metrics for infrared small-target detection.

Unlike the legacy evaluator, this module has one explicit probability
thresholding path for every method.  The immutable protocol records the
connected-component and matching choices so pre/post-adaptation predictions
cannot silently use different rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .connected_components import extract_connected_components
from .target_matching import match_components


DEFAULT_FROC_THRESHOLDS: Tuple[float, ...] = tuple(
    index / 20.0 for index in range(21)
)


def _validate_probability_threshold(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}.")
    return value


@dataclass(frozen=True)
class IRSTDEvaluationProtocol:
    """Frozen post-processing and matching choices for an experiment."""

    fixed_probability_threshold: float = 0.5
    froc_probability_thresholds: Tuple[float, ...] = DEFAULT_FROC_THRESHOLDS
    connectivity: int = 2
    max_centroid_distance: float = 3.0
    min_component_area: int = 1

    def __post_init__(self) -> None:
        fixed_threshold = _validate_probability_threshold(
            self.fixed_probability_threshold, "fixed_probability_threshold"
        )
        if not self.froc_probability_thresholds:
            raise ValueError("froc_probability_thresholds cannot be empty.")
        froc_thresholds = tuple(
            sorted(
                {
                    _validate_probability_threshold(
                        threshold, "froc_probability_thresholds"
                    )
                    for threshold in self.froc_probability_thresholds
                }
            )
        )
        if isinstance(self.connectivity, bool) or self.connectivity not in (1, 2):
            raise ValueError("connectivity must be 1 (4-neighbour) or 2 (8-neighbour).")
        if (
            not np.isfinite(self.max_centroid_distance)
            or self.max_centroid_distance <= 0
        ):
            raise ValueError("max_centroid_distance must be positive and finite.")
        if (
            isinstance(self.min_component_area, bool)
            or not isinstance(self.min_component_area, (int, np.integer))
            or self.min_component_area < 1
        ):
            raise ValueError("min_component_area must be a positive integer.")

        # Canonical values make equality, serialisation, and threshold lookup
        # deterministic even if callers supplied a list or duplicate values.
        object.__setattr__(self, "fixed_probability_threshold", fixed_threshold)
        object.__setattr__(self, "froc_probability_thresholds", froc_thresholds)
        object.__setattr__(self, "connectivity", int(self.connectivity))
        object.__setattr__(self, "max_centroid_distance", float(self.max_centroid_distance))
        object.__setattr__(self, "min_component_area", int(self.min_component_area))


@dataclass(frozen=True)
class PixelMetrics:
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    true_negative_pixels: int
    predicted_positive_pixels: int
    target_positive_pixels: int
    pixel_accuracy: float
    intersection_over_union: float
    precision: float
    recall: float


@dataclass(frozen=True)
class TargetMetrics:
    detected_targets: int
    total_targets: int
    false_positive_components: int
    false_alarm_pixels: int
    image_count: int
    total_image_pixels: int
    detection_probability: float
    false_positives_per_image: float
    false_alarm_pixel_rate: float


@dataclass(frozen=True)
class ThresholdMetrics:
    probability_threshold: float
    pixel: PixelMetrics
    target: TargetMetrics


@dataclass(frozen=True)
class IRSTDEvaluationResult:
    fixed: ThresholdMetrics
    froc: Tuple[ThresholdMetrics, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable nested representation."""

        return asdict(self)


@dataclass
class _ThresholdAccumulator:
    true_positive_pixels: int = 0
    false_positive_pixels: int = 0
    false_negative_pixels: int = 0
    true_negative_pixels: int = 0
    detected_targets: int = 0
    total_targets: int = 0
    false_positive_components: int = 0
    false_alarm_pixels: int = 0
    image_count: int = 0
    total_image_pixels: int = 0


def _to_numpy(value: Any) -> NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_image_batch(value: Any, *, name: str) -> NDArray[Any]:
    """Normalise [H,W], [B,H,W], or [B,1,H,W] to [B,H,W]."""

    array = _to_numpy(value)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        pass
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0, ...]
    else:
        raise ValueError(
            f"{name} must have shape [H,W], [B,H,W], or [B,1,H,W]; "
            f"got {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} cannot contain empty dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return array


def probabilities_from_logits(logits: Any) -> NDArray[np.float64]:
    """Apply a numerically stable sigmoid exactly once."""

    values = _to_numpy(logits).astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("logits contains NaN or Inf values.")
    probabilities = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponentials = np.exp(values[~nonnegative])
    probabilities[~nonnegative] = exponentials / (1.0 + exponentials)
    return probabilities


def _divide_or(numerator: int, denominator: int, empty_value: float) -> float:
    if denominator == 0:
        return float(empty_value)
    return float(numerator / denominator)


class UnifiedResearchEvaluator:
    """Accumulate fixed-threshold pixel/target metrics and a FROC scan.

    ``update_logits`` is the normal entry point and owns the only sigmoid in
    the evaluator.  ``update_probabilities`` is provided for callers that have
    an explicitly probability-valued source.  Thresholding always uses the
    strict rule ``probability > threshold``.

    Empty cases are finite by convention: empty/empty pixel IoU, precision and
    recall are 1, while target detection probability is 0 when the complete
    accumulated set contains no ground-truth targets.
    """

    def __init__(
        self,
        protocol: IRSTDEvaluationProtocol | None = None,
    ) -> None:
        self.protocol = protocol or IRSTDEvaluationProtocol()
        self._thresholds = tuple(
            sorted(
                set(self.protocol.froc_probability_thresholds)
                | {self.protocol.fixed_probability_threshold}
            )
        )
        self.reset()

    def reset(self) -> None:
        self._accumulators: Dict[float, _ThresholdAccumulator] = {
            threshold: _ThresholdAccumulator() for threshold in self._thresholds
        }

    def update(self, logits: Any, targets: Any) -> None:
        """Alias for ``update_logits``; input is always interpreted as logits."""

        self.update_logits(logits, targets)

    def update_logits(self, logits: Any, targets: Any) -> None:
        self.update_probabilities(probabilities_from_logits(logits), targets)

    def update_probabilities(self, probabilities: Any, targets: Any) -> None:
        probability_batch = _as_image_batch(probabilities, name="probabilities").astype(
            np.float64, copy=False
        )
        if (probability_batch < 0.0).any() or (probability_batch > 1.0).any():
            raise ValueError("probabilities must lie in [0, 1].")
        target_batch = _as_image_batch(targets, name="targets") > 0
        if probability_batch.shape != target_batch.shape:
            raise ValueError(
                "probabilities and targets must have the same normalised shape; "
                f"got {probability_batch.shape} and {target_batch.shape}."
            )

        for probabilities_2d, target_2d in zip(probability_batch, target_batch):
            target_components = extract_connected_components(
                target_2d,
                connectivity=self.protocol.connectivity,
                min_area=self.protocol.min_component_area,
            )
            image_pixels = int(target_2d.size)

            for threshold, accumulator in self._accumulators.items():
                prediction_2d = probabilities_2d > threshold
                true_positive = int(np.logical_and(prediction_2d, target_2d).sum())
                false_positive = int(
                    np.logical_and(prediction_2d, np.logical_not(target_2d)).sum()
                )
                false_negative = int(
                    np.logical_and(np.logical_not(prediction_2d), target_2d).sum()
                )
                true_negative = image_pixels - true_positive - false_positive - false_negative

                prediction_components = extract_connected_components(
                    prediction_2d,
                    connectivity=self.protocol.connectivity,
                    min_area=self.protocol.min_component_area,
                )
                matching = match_components(
                    prediction_components,
                    target_components,
                    max_centroid_distance=self.protocol.max_centroid_distance,
                )

                accumulator.true_positive_pixels += true_positive
                accumulator.false_positive_pixels += false_positive
                accumulator.false_negative_pixels += false_negative
                accumulator.true_negative_pixels += true_negative
                accumulator.detected_targets += matching.true_positives
                accumulator.total_targets += len(target_components)
                accumulator.false_positive_components += matching.false_positives
                accumulator.false_alarm_pixels += sum(
                    prediction_components[index].area
                    for index in matching.unmatched_prediction_indices
                )
                accumulator.image_count += 1
                accumulator.total_image_pixels += image_pixels

    def _compute_threshold(self, threshold: float) -> ThresholdMetrics:
        accumulator = self._accumulators[threshold]
        predicted_positive = (
            accumulator.true_positive_pixels + accumulator.false_positive_pixels
        )
        target_positive = (
            accumulator.true_positive_pixels + accumulator.false_negative_pixels
        )
        pixel_total = (
            accumulator.true_positive_pixels
            + accumulator.false_positive_pixels
            + accumulator.false_negative_pixels
            + accumulator.true_negative_pixels
        )
        union = (
            accumulator.true_positive_pixels
            + accumulator.false_positive_pixels
            + accumulator.false_negative_pixels
        )

        pixel = PixelMetrics(
            true_positive_pixels=accumulator.true_positive_pixels,
            false_positive_pixels=accumulator.false_positive_pixels,
            false_negative_pixels=accumulator.false_negative_pixels,
            true_negative_pixels=accumulator.true_negative_pixels,
            predicted_positive_pixels=predicted_positive,
            target_positive_pixels=target_positive,
            pixel_accuracy=_divide_or(
                accumulator.true_positive_pixels + accumulator.true_negative_pixels,
                pixel_total,
                0.0,
            ),
            intersection_over_union=_divide_or(
                accumulator.true_positive_pixels, union, 1.0
            ),
            precision=_divide_or(
                accumulator.true_positive_pixels,
                predicted_positive,
                1.0 if target_positive == 0 else 0.0,
            ),
            recall=_divide_or(
                accumulator.true_positive_pixels, target_positive, 1.0
            ),
        )
        target = TargetMetrics(
            detected_targets=accumulator.detected_targets,
            total_targets=accumulator.total_targets,
            false_positive_components=accumulator.false_positive_components,
            false_alarm_pixels=accumulator.false_alarm_pixels,
            image_count=accumulator.image_count,
            total_image_pixels=accumulator.total_image_pixels,
            detection_probability=_divide_or(
                accumulator.detected_targets, accumulator.total_targets, 0.0
            ),
            false_positives_per_image=_divide_or(
                accumulator.false_positive_components, accumulator.image_count, 0.0
            ),
            false_alarm_pixel_rate=_divide_or(
                accumulator.false_alarm_pixels, accumulator.total_image_pixels, 0.0
            ),
        )
        return ThresholdMetrics(
            probability_threshold=threshold,
            pixel=pixel,
            target=target,
        )

    def compute(self) -> IRSTDEvaluationResult:
        return IRSTDEvaluationResult(
            fixed=self._compute_threshold(
                self.protocol.fixed_probability_threshold
            ),
            froc=tuple(
                self._compute_threshold(threshold)
                for threshold in self.protocol.froc_probability_thresholds
            ),
        )


__all__ = [
    "DEFAULT_FROC_THRESHOLDS",
    "IRSTDEvaluationProtocol",
    "IRSTDEvaluationResult",
    "PixelMetrics",
    "TargetMetrics",
    "ThresholdMetrics",
    "UnifiedResearchEvaluator",
    "probabilities_from_logits",
]
