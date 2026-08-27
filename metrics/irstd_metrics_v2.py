"""Versioned formal evaluator for probability-valued IRSTD predictions.

This module deliberately has a probability-only input contract.  It composes
the established :mod:`metrics.irstd_metrics` implementation for global pixel
and target counts, while making three formal-reporting choices explicit:

* every binary mask uses the strict rule ``probability > threshold``;
* the FROC scan always contains the complete 21-point grid ``0.00:0.05:1.00``;
* the two false-alarm horizontal axes are exported separately as
  ``fppi = false_positive_components / image_count`` and
  ``false_alarm_pixel_rate = false_alarm_pixels / total_image_pixels``.

The normalized IoU (nIoU) reported here is the arithmetic mean of per-image
foreground IoUs at the fixed threshold.  An image whose foreground union is
empty contributes exactly ``0.0``.  Every contribution and both averaging
denominators are retained so this convention can be audited.

No FROC AUC is defined or computed by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .irstd_metrics import (
    DEFAULT_FROC_THRESHOLDS,
    IRSTDEvaluationProtocol,
    PixelMetrics,
    ThresholdMetrics,
    UnifiedResearchEvaluator,
)


FORMAL_FROC_THRESHOLDS_V2: Tuple[float, ...] = DEFAULT_FROC_THRESHOLDS
"""The frozen 21-point formal FROC grid: 0.00, 0.05, ..., 1.00."""

STRICT_THRESHOLD_RULE_V2 = "probability > threshold"
NIOU_EMPTY_UNION_CONTRIBUTION_V2 = 0.0


@dataclass(frozen=True)
class FROCAxisSemanticsV2:
    """Unambiguous names and formulas for the two exported horizontal axes."""

    fppi_name: str = "fppi"
    fppi_formula: str = "false_positive_components / image_count"
    false_alarm_pixel_rate_name: str = "false_alarm_pixel_rate"
    false_alarm_pixel_rate_formula: str = (
        "false_alarm_pixels / total_image_pixels"
    )


@dataclass(frozen=True)
class ThresholdMetricsV2:
    """One fixed-threshold or FROC point with explicit false-alarm axes."""

    probability_threshold: float
    pixel: PixelMetrics
    detected_targets: int
    total_targets: int
    detection_probability: float
    false_positive_components: int
    image_count: int
    fppi: float
    false_alarm_pixels: int
    total_image_pixels: int
    false_alarm_pixel_rate: float


@dataclass(frozen=True)
class PerImageForegroundIoUContributionV2:
    """Auditable contribution of one image to nIoU."""

    image_index: int
    image_id: str | None
    intersection_pixels: int
    union_pixels: int
    foreground_iou_contribution: float
    empty_union: bool


@dataclass(frozen=True)
class NormalizedIoUResultV2:
    """Arithmetic mean of per-image foreground IoU contributions."""

    normalized_iou: float
    contribution_sum: float
    averaging_denominator_image_count: int
    nonempty_union_image_count: int
    empty_union_image_count: int
    empty_union_contribution: float
    per_image_contributions: Tuple[PerImageForegroundIoUContributionV2, ...]


@dataclass(frozen=True)
class UnifiedResearchEvaluationResultV2:
    """Formal fixed-threshold, nIoU, and complete 21-point FROC result."""

    threshold_rule: str
    fixed: ThresholdMetricsV2
    normalized_iou: NormalizedIoUResultV2
    froc_axis_semantics: FROCAxisSemanticsV2
    froc: Tuple[ThresholdMetricsV2, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable nested representation."""

        return asdict(self)

    def check_endpoint_conservation(self) -> "FROCEndpointConservationV2":
        """Return the non-mutating endpoint/count-conservation audit report."""

        return check_froc_endpoint_conservation_v2(self)

    def assert_endpoint_conservation(self) -> "FROCEndpointConservationV2":
        """Raise when the formal FROC endpoint/count invariants do not hold."""

        return assert_froc_endpoint_conservation_v2(self)


@dataclass(frozen=True)
class FROCEndpointConservationV2:
    """Audit result for formal FROC endpoints and conserved denominators.

    Connected-component counts themselves are not required to be monotonic:
    components can split or merge as the threshold changes.  Pixel foreground
    count, however, must be non-increasing on an ascending threshold grid.
    """

    has_exact_formal_21_point_grid: bool
    threshold_zero_present: bool
    threshold_one_present: bool
    threshold_one_has_no_predicted_positive_pixels: bool
    threshold_one_has_no_detected_targets: bool
    threshold_one_has_no_false_positive_components: bool
    threshold_one_has_no_false_alarm_pixels: bool
    target_count_constant: bool
    image_count_constant: bool
    total_image_pixels_constant: bool
    pixel_partition_conserved_at_every_point: bool
    predicted_positive_pixels_nonincreasing: bool
    fixed_point_matches_froc_grid: bool
    axis_formulas_conserved_at_every_point: bool

    @property
    def is_conserved(self) -> bool:
        return all(asdict(self).values())

    @property
    def failed_checks(self) -> Tuple[str, ...]:
        return tuple(name for name, passed in asdict(self).items() if not passed)


def _normalise_image_batch(value: Any, *, name: str) -> NDArray[Any]:
    """Normalize ``[H,W]``, ``[B,H,W]``, or ``[B,1,H,W]`` to ``[B,H,W]``."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
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
    if any(dimension == 0 for dimension in array.shape):
        raise ValueError(f"{name} cannot contain empty dimensions.")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must contain numeric or boolean values.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return array


def _normalise_image_ids(
    image_ids: Sequence[str] | None,
    *,
    batch_size: int,
) -> Tuple[str | None, ...]:
    if image_ids is None:
        return (None,) * batch_size
    ids = tuple(image_ids)
    if len(ids) != batch_size:
        raise ValueError(
            "image_ids length must equal the normalized batch size; "
            f"got {len(ids)} and {batch_size}."
        )
    if any(not isinstance(image_id, str) for image_id in ids):
        raise ValueError("Every image_id must be a string.")
    return ids


def _as_v2_threshold_metrics(point: ThresholdMetrics) -> ThresholdMetricsV2:
    target = point.target
    return ThresholdMetricsV2(
        probability_threshold=point.probability_threshold,
        pixel=point.pixel,
        detected_targets=target.detected_targets,
        total_targets=target.total_targets,
        detection_probability=target.detection_probability,
        false_positive_components=target.false_positive_components,
        image_count=target.image_count,
        fppi=target.false_positives_per_image,
        false_alarm_pixels=target.false_alarm_pixels,
        total_image_pixels=target.total_image_pixels,
        false_alarm_pixel_rate=target.false_alarm_pixel_rate,
    )


def _divide_or_zero(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


class UnifiedResearchEvaluatorV2:
    """Accumulate formal metrics from canonical probabilities only.

    ``update_probabilities`` is the sole update entry point.  The caller must
    apply sigmoid (if needed) before entering this evaluator.  Values outside
    ``[0, 1]`` are rejected, guarding against an accidental logits/probability
    mix-up.  All fixed and FROC masks use strict ``probability > threshold``.
    """

    def __init__(
        self,
        protocol: IRSTDEvaluationProtocol | None = None,
    ) -> None:
        self.protocol = protocol or IRSTDEvaluationProtocol()
        if self.protocol.froc_probability_thresholds != FORMAL_FROC_THRESHOLDS_V2:
            raise ValueError(
                "UnifiedResearchEvaluatorV2 requires the exact formal 21-point "
                "FROC grid (0.00, 0.05, ..., 1.00)."
            )
        self._global_evaluator = UnifiedResearchEvaluator(self.protocol)
        self.reset()

    def reset(self) -> None:
        """Clear global accumulators and all per-image nIoU audit records."""

        self._global_evaluator.reset()
        self._per_image_niou: list[PerImageForegroundIoUContributionV2] = []
        self._next_image_index = 0

    def update_probabilities(
        self,
        probabilities: Any,
        targets: Any,
        *,
        image_ids: Sequence[str] | None = None,
    ) -> None:
        """Accumulate one image or batch of canonical probabilities and masks."""

        probability_batch = _normalise_image_batch(
            probabilities, name="probabilities"
        ).astype(np.float64, copy=False)
        if (probability_batch < 0.0).any() or (probability_batch > 1.0).any():
            raise ValueError("probabilities must lie in [0, 1].")
        target_batch = _normalise_image_batch(targets, name="targets") > 0
        if probability_batch.shape != target_batch.shape:
            raise ValueError(
                "probabilities and targets must have the same normalized shape; "
                f"got {probability_batch.shape} and {target_batch.shape}."
            )
        normalized_ids = _normalise_image_ids(
            image_ids, batch_size=int(probability_batch.shape[0])
        )

        # Delegate global pixel/target/FROC semantics to the established
        # evaluator after this versioned probability contract has validated the
        # input.  No sigmoid is applied here or in the delegated method.
        self._global_evaluator.update_probabilities(probability_batch, target_batch)

        threshold = self.protocol.fixed_probability_threshold
        for probabilities_2d, target_2d, image_id in zip(
            probability_batch, target_batch, normalized_ids
        ):
            prediction_2d = probabilities_2d > threshold
            intersection = int(np.logical_and(prediction_2d, target_2d).sum())
            union = int(np.logical_or(prediction_2d, target_2d).sum())
            contribution = (
                float(intersection / union)
                if union
                else NIOU_EMPTY_UNION_CONTRIBUTION_V2
            )
            self._per_image_niou.append(
                PerImageForegroundIoUContributionV2(
                    image_index=self._next_image_index,
                    image_id=image_id,
                    intersection_pixels=intersection,
                    union_pixels=union,
                    foreground_iou_contribution=contribution,
                    empty_union=union == 0,
                )
            )
            self._next_image_index += 1

    def compute(self) -> UnifiedResearchEvaluationResultV2:
        """Return a deterministic snapshot without mutating evaluator state."""

        global_result = self._global_evaluator.compute()
        contributions = tuple(self._per_image_niou)
        contribution_sum = float(
            sum(item.foreground_iou_contribution for item in contributions)
        )
        image_count = len(contributions)
        empty_count = sum(item.empty_union for item in contributions)
        normalized_iou = (
            contribution_sum / image_count
            if image_count
            else NIOU_EMPTY_UNION_CONTRIBUTION_V2
        )
        result = UnifiedResearchEvaluationResultV2(
            threshold_rule=STRICT_THRESHOLD_RULE_V2,
            fixed=_as_v2_threshold_metrics(global_result.fixed),
            normalized_iou=NormalizedIoUResultV2(
                normalized_iou=float(normalized_iou),
                contribution_sum=contribution_sum,
                averaging_denominator_image_count=image_count,
                nonempty_union_image_count=image_count - empty_count,
                empty_union_image_count=empty_count,
                empty_union_contribution=NIOU_EMPTY_UNION_CONTRIBUTION_V2,
                per_image_contributions=contributions,
            ),
            froc_axis_semantics=FROCAxisSemanticsV2(),
            froc=tuple(_as_v2_threshold_metrics(point) for point in global_result.froc),
        )
        return result


def check_froc_endpoint_conservation_v2(
    result: UnifiedResearchEvaluationResultV2,
) -> FROCEndpointConservationV2:
    """Check formal endpoints, count partitions, and denominator conservation.

    This helper is intentionally non-mutating and does not integrate an AUC.
    It provides a fail-closed audit surface for runners before metrics are
    persisted as formal results.
    """

    points = result.froc
    thresholds = tuple(point.probability_threshold for point in points)
    zero = next((point for point in points if point.probability_threshold == 0.0), None)
    one = next((point for point in points if point.probability_threshold == 1.0), None)

    def _constant(values: Sequence[int]) -> bool:
        return not values or all(value == values[0] for value in values)

    partition_conserved = all(
        (
            point.pixel.true_positive_pixels
            + point.pixel.false_positive_pixels
            + point.pixel.false_negative_pixels
            + point.pixel.true_negative_pixels
        )
        == point.total_image_pixels
        for point in points
    )
    predicted_positive = [point.pixel.predicted_positive_pixels for point in points]
    foreground_nonincreasing = all(
        left >= right
        for left, right in zip(predicted_positive, predicted_positive[1:])
    )
    axes_conserved = all(
        np.isclose(
            point.fppi,
            _divide_or_zero(point.false_positive_components, point.image_count),
            rtol=0.0,
            atol=0.0,
        )
        and np.isclose(
            point.false_alarm_pixel_rate,
            _divide_or_zero(point.false_alarm_pixels, point.total_image_pixels),
            rtol=0.0,
            atol=0.0,
        )
        for point in points
    )
    fixed_grid_point = next(
        (
            point
            for point in points
            if point.probability_threshold == result.fixed.probability_threshold
        ),
        None,
    )

    return FROCEndpointConservationV2(
        has_exact_formal_21_point_grid=thresholds == FORMAL_FROC_THRESHOLDS_V2,
        threshold_zero_present=zero is not None,
        threshold_one_present=one is not None,
        threshold_one_has_no_predicted_positive_pixels=(
            one is not None and one.pixel.predicted_positive_pixels == 0
        ),
        threshold_one_has_no_detected_targets=(
            one is not None and one.detected_targets == 0
        ),
        threshold_one_has_no_false_positive_components=(
            one is not None and one.false_positive_components == 0
        ),
        threshold_one_has_no_false_alarm_pixels=(
            one is not None and one.false_alarm_pixels == 0
        ),
        target_count_constant=_constant([point.total_targets for point in points]),
        image_count_constant=_constant([point.image_count for point in points]),
        total_image_pixels_constant=_constant(
            [point.total_image_pixels for point in points]
        ),
        pixel_partition_conserved_at_every_point=partition_conserved,
        predicted_positive_pixels_nonincreasing=foreground_nonincreasing,
        fixed_point_matches_froc_grid=(
            fixed_grid_point is not None and fixed_grid_point == result.fixed
        ),
        axis_formulas_conserved_at_every_point=axes_conserved,
    )


def assert_froc_endpoint_conservation_v2(
    result: UnifiedResearchEvaluationResultV2,
) -> FROCEndpointConservationV2:
    """Return the conservation report, raising if any invariant fails."""

    report = check_froc_endpoint_conservation_v2(result)
    if not report.is_conserved:
        raise AssertionError(
            "FROC endpoint/count conservation failed: "
            + ", ".join(report.failed_checks)
        )
    return report


__all__ = [
    "FORMAL_FROC_THRESHOLDS_V2",
    "FROCAxisSemanticsV2",
    "FROCEndpointConservationV2",
    "NIOU_EMPTY_UNION_CONTRIBUTION_V2",
    "NormalizedIoUResultV2",
    "PerImageForegroundIoUContributionV2",
    "STRICT_THRESHOLD_RULE_V2",
    "ThresholdMetricsV2",
    "UnifiedResearchEvaluationResultV2",
    "UnifiedResearchEvaluatorV2",
    "assert_froc_endpoint_conservation_v2",
    "check_froc_endpoint_conservation_v2",
]
