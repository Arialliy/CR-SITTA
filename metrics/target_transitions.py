"""Post-hoc target-state transitions for episodic IRSTD evaluation.

This module deliberately accepts *materialised probability maps* rather than a
model, an input image, or an adaptation callback.  Ground truth is therefore
confined to the evaluation boundary and cannot be consumed by the TTA update.

Discrete TP/FN states reuse the exact semantics of :mod:`metrics.irstd_metrics`:

* ground-truth foreground is ``target > 0``;
* predicted foreground is ``probability > fixed_probability_threshold``;
* both masks use the protocol's connectivity and minimum component area; and
* prediction/target components are matched one-to-one with the existing
  strict centroid-distance matcher.

The continuous target score is diagnostic and is not equivalent to the
component-level TP/FN decision.  In particular, one-to-one matching conflicts
and component-centroid movement cannot be represented by a scalar local score.
Consequently its reducer and definition are explicit in every result.  The
provided default is a candidate definition, not a frozen experiment protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .connected_components import ConnectedComponent, label_connected_components
from .irstd_metrics import IRSTDEvaluationProtocol
from .target_matching import TargetMatch, match_components


SOURCE_TO_POST = "Source→post"
TENT_PRE_TO_POST = "tent_pre→post"

TP_TO_TP = "TP→TP"
TP_TO_FN = "TP→FN"
FN_TO_TP = "FN→TP"
FN_TO_FN = "FN→FN"
TRANSITION_LABELS = (TP_TO_TP, TP_TO_FN, FN_TO_TP, FN_TO_FN)

SCORE_STATUS_CANDIDATE_NOT_FROZEN = "candidate_not_frozen"


@dataclass(frozen=True)
class TargetScoreContext:
    """Read-only context passed to a continuous target-score reducer."""

    target_component: ConnectedComponent
    target_component_mask: NDArray[np.bool_]
    protocol: IRSTDEvaluationProtocol


TargetScoreReducer = Callable[
    [NDArray[np.float64], TargetScoreContext], float
]


@dataclass(frozen=True)
class TargetScoreDefinition:
    """Serializable identity for a target-score reducer.

    ``protocol_status`` is intentionally explicit.  The built-in definition is
    marked ``candidate_not_frozen`` because the project guide requests a
    continuous score but does not define its spatial reducer.
    """

    name: str
    description: str
    protocol_status: str = SCORE_STATUS_CANDIDATE_NOT_FROZEN
    parameters: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("name", "description", "protocol_status"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        keys = []
        for pair in self.parameters:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("parameters must contain (name, value) tuples.")
            key, _value = pair
            if not isinstance(key, str) or not key.strip():
                raise ValueError("score parameter names must be non-empty strings.")
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError("score parameter names must be unique.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "protocol_status": self.protocol_status,
            "parameters": {key: value for key, value in self.parameters},
        }


def dilated_gt_neighborhood_max(
    probability_map: NDArray[np.float64],
    context: TargetScoreContext,
) -> float:
    """Maximum probability in a strict Euclidean dilation of one GT target.

    The dilation radius is ``protocol.max_centroid_distance`` and the boundary
    is strict, mirroring the matcher's ``distance < max_centroid_distance``
    gate.  The dilation is measured from target pixels, whereas the discrete
    matcher compares component centroids; this score is therefore correlated
    diagnostic evidence, not an alternative detection rule.
    """

    distance_to_target = ndimage.distance_transform_edt(
        np.logical_not(context.target_component_mask)
    )
    neighborhood = distance_to_target < context.protocol.max_centroid_distance
    return float(np.max(probability_map[neighborhood]))


def default_score_definition(
    protocol: IRSTDEvaluationProtocol,
) -> TargetScoreDefinition:
    """Describe the built-in, explicitly non-frozen score candidate."""

    return TargetScoreDefinition(
        name="dilated_gt_neighborhood_max_probability",
        description=(
            "maximum probability at pixels whose Euclidean distance to the GT "
            "component is strictly less than max_centroid_distance; diagnostic "
            "only and not equivalent to one-to-one centroid matching"
        ),
        protocol_status=SCORE_STATUS_CANDIDATE_NOT_FROZEN,
        parameters=(
            ("distance_boundary", "strict_less_than"),
            ("dilation_radius_pixels", protocol.max_centroid_distance),
            ("reduction", "max"),
            ("spatial_reference", "gt_component_pixels"),
        ),
    )


@dataclass(frozen=True)
class PredictionMatchRecord:
    """Geometry of the prediction assigned to one GT component."""

    prediction_index: int
    prediction_label: int
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    centroid_distance: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetTransitionRecord:
    """Pre/post state and continuous response for one fixed GT target."""

    image_id: Optional[str]
    comparison_label: Optional[str]
    gt_target_id: int
    gt_area: int
    gt_centroid: Tuple[float, float]
    gt_bbox: Tuple[int, int, int, int]
    pre_state: str
    post_state: str
    transition: str
    pre_target_score: float
    post_target_score: float
    score_change: float
    probability_threshold: float
    pre_margin_to_threshold: float
    post_margin_to_threshold: float
    pre_match: Optional[PredictionMatchRecord]
    post_match: Optional[PredictionMatchRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "pre_match": None if self.pre_match is None else self.pre_match.to_dict(),
            "post_match": (
                None if self.post_match is None else self.post_match.to_dict()
            ),
        }


@dataclass(frozen=True)
class TransitionCounts:
    """Absolute counts for the four GT-target state transitions."""

    tp_to_tp: int = 0
    tp_to_fn: int = 0
    fn_to_tp: int = 0
    fn_to_fn: int = 0

    @property
    def total(self) -> int:
        return self.tp_to_tp + self.tp_to_fn + self.fn_to_tp + self.fn_to_fn

    @property
    def pre_tp(self) -> int:
        return self.tp_to_tp + self.tp_to_fn

    @property
    def pre_fn(self) -> int:
        return self.fn_to_tp + self.fn_to_fn

    @property
    def net_target_gain_count(self) -> int:
        return self.fn_to_tp - self.tp_to_fn

    def __add__(self, other: "TransitionCounts") -> "TransitionCounts":
        if not isinstance(other, TransitionCounts):
            return NotImplemented
        return TransitionCounts(
            tp_to_tp=self.tp_to_tp + other.tp_to_tp,
            tp_to_fn=self.tp_to_fn + other.tp_to_fn,
            fn_to_tp=self.fn_to_tp + other.fn_to_tp,
            fn_to_fn=self.fn_to_fn + other.fn_to_fn,
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            TP_TO_TP: self.tp_to_tp,
            TP_TO_FN: self.tp_to_fn,
            FN_TO_TP: self.fn_to_tp,
            FN_TO_FN: self.fn_to_fn,
        }


@dataclass(frozen=True)
class DefinedRatio:
    """A ratio whose zero-denominator state is explicit and JSON-safe."""

    numerator: int
    denominator: int
    value: Optional[float]
    defined: bool

    @classmethod
    def create(cls, numerator: int, denominator: int) -> "DefinedRatio":
        if denominator < 0:
            raise ValueError("ratio denominator cannot be negative.")
        if denominator == 0:
            return cls(
                numerator=int(numerator), denominator=0, value=None, defined=False
            )
        return cls(
            numerator=int(numerator),
            denominator=int(denominator),
            value=float(numerator / denominator),
            defined=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageTargetTransitionResult:
    """All GT-target records and matching diagnostics for one image."""

    image_id: Optional[str]
    comparison_label: Optional[str]
    protocol: IRSTDEvaluationProtocol
    score_definition: TargetScoreDefinition
    transition_counts: TransitionCounts
    target_records: Tuple[TargetTransitionRecord, ...]
    pre_prediction_component_count: int
    post_prediction_component_count: int
    pre_unmatched_prediction_count: int
    post_unmatched_prediction_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "comparison_label": self.comparison_label,
            "protocol": asdict(self.protocol),
            "threshold_rule": "probability_strictly_greater_than_threshold",
            "matching_rule": (
                "one_to_one_hungarian_with_centroid_distance_strictly_less_than_"
                "max_centroid_distance"
            ),
            "score_definition": self.score_definition.to_dict(),
            "transition_counts": self.transition_counts.to_dict(),
            "target_records": [record.to_dict() for record in self.target_records],
            "pre_prediction_component_count": self.pre_prediction_component_count,
            "post_prediction_component_count": self.post_prediction_component_count,
            "pre_unmatched_prediction_count": self.pre_unmatched_prediction_count,
            "post_unmatched_prediction_count": self.post_unmatched_prediction_count,
        }


@dataclass(frozen=True)
class TargetTransitionSummary:
    """Dataset/condition aggregate with explicit metric denominators."""

    comparison_label: Optional[str]
    score_definition: TargetScoreDefinition
    image_count: int
    total_gt_targets: int
    transition_counts: TransitionCounts
    ater: DefinedRatio
    atrr: DefinedRatio
    ntg: DefinedRatio
    net_target_gain_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_label": self.comparison_label,
            "score_definition": self.score_definition.to_dict(),
            "image_count": self.image_count,
            "total_gt_targets": self.total_gt_targets,
            "transition_counts": self.transition_counts.to_dict(),
            "pre_tp_count": self.transition_counts.pre_tp,
            "pre_fn_count": self.transition_counts.pre_fn,
            "ATER": self.ater.to_dict(),
            "ATRR": self.atrr.to_dict(),
            "NTG": self.ntg.to_dict(),
            "net_target_gain_count": self.net_target_gain_count,
        }


def _normalise_optional_label(value: Optional[str], *, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be None or a non-empty string.")
    return value.strip()


def _to_numpy(value: Any) -> NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_single_map(value: Any, *, name: str) -> NDArray[Any]:
    """Normalise [H,W], [1,H,W], or [1,1,H,W] to one copied [H,W] map."""

    array = _to_numpy(value)
    if array.ndim == 2:
        normalised = array
    elif array.ndim == 3 and array.shape[0] == 1:
        normalised = array[0]
    elif array.ndim == 4 and array.shape[:2] == (1, 1):
        normalised = array[0, 0]
    else:
        raise ValueError(
            f"{name} must describe exactly one single-channel image with shape "
            f"[H,W], [1,H,W], or [1,1,H,W]; got {array.shape}."
        )
    if normalised.shape[0] == 0 or normalised.shape[1] == 0:
        raise ValueError(f"{name} cannot contain empty dimensions.")
    if not np.isfinite(normalised).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return np.array(normalised, copy=True)


def _prediction_match_record(
    match: Optional[TargetMatch],
    prediction_components: Tuple[ConnectedComponent, ...],
) -> Optional[PredictionMatchRecord]:
    if match is None:
        return None
    component = prediction_components[match.prediction_index]
    return PredictionMatchRecord(
        prediction_index=match.prediction_index,
        prediction_label=component.label,
        area=component.area,
        centroid=component.centroid,
        bbox=component.bbox,
        centroid_distance=match.centroid_distance,
    )


def _count_records(
    records: Tuple[TargetTransitionRecord, ...],
) -> TransitionCounts:
    counts = {label: 0 for label in TRANSITION_LABELS}
    for record in records:
        counts[record.transition] += 1
    return TransitionCounts(
        tp_to_tp=counts[TP_TO_TP],
        tp_to_fn=counts[TP_TO_FN],
        fn_to_tp=counts[FN_TO_TP],
        fn_to_fn=counts[FN_TO_FN],
    )


def evaluate_target_transitions(
    pre_probabilities: Any,
    post_probabilities: Any,
    ground_truth: Any,
    *,
    protocol: Optional[IRSTDEvaluationProtocol] = None,
    comparison_label: Optional[str] = None,
    image_id: Optional[str] = None,
    score_reducer: Optional[TargetScoreReducer] = None,
    score_definition: Optional[TargetScoreDefinition] = None,
) -> ImageTargetTransitionResult:
    """Evaluate pre/post transitions for every GT target in one image.

    Both probability maps are materialised and validated before ground truth is
    accessed.  This ordering is intentional: this post-hoc evaluator must not
    become an adaptation callback or expose a label to prediction generation.

    A custom ``score_reducer`` must be paired with an explicit serialisable
    ``score_definition``.  If neither is supplied, the built-in non-frozen
    candidate definition is used.  Discrete transition states never depend on
    the score reducer.
    """

    protocol = protocol or IRSTDEvaluationProtocol()
    if not isinstance(protocol, IRSTDEvaluationProtocol):
        raise TypeError("protocol must be an IRSTDEvaluationProtocol instance.")
    comparison_label = _normalise_optional_label(
        comparison_label, name="comparison_label"
    )
    image_id = _normalise_optional_label(image_id, name="image_id")

    if score_reducer is None:
        if score_definition is not None:
            raise ValueError(
                "score_definition cannot be supplied without a custom score_reducer."
            )
        reducer = dilated_gt_neighborhood_max
        definition = default_score_definition(protocol)
    else:
        if not callable(score_reducer):
            raise TypeError("score_reducer must be callable.")
        if score_definition is None:
            raise ValueError(
                "A custom score_reducer requires an explicit score_definition."
            )
        reducer = score_reducer
        definition = score_definition

    # Materialise predictions before crossing the ground-truth boundary.
    pre_map = _as_single_map(pre_probabilities, name="pre_probabilities").astype(
        np.float64, copy=False
    )
    post_map = _as_single_map(post_probabilities, name="post_probabilities").astype(
        np.float64, copy=False
    )
    if (pre_map < 0.0).any() or (pre_map > 1.0).any():
        raise ValueError("pre_probabilities must lie in [0, 1].")
    if (post_map < 0.0).any() or (post_map > 1.0).any():
        raise ValueError("post_probabilities must lie in [0, 1].")

    target_map = _as_single_map(ground_truth, name="ground_truth") > 0
    if pre_map.shape != post_map.shape or pre_map.shape != target_map.shape:
        raise ValueError(
            "pre_probabilities, post_probabilities, and ground_truth must have "
            "the same normalised shape; got "
            f"{pre_map.shape}, {post_map.shape}, and {target_map.shape}."
        )

    target_extraction = label_connected_components(
        target_map,
        connectivity=protocol.connectivity,
        min_area=protocol.min_component_area,
    )
    pre_extraction = label_connected_components(
        pre_map > protocol.fixed_probability_threshold,
        connectivity=protocol.connectivity,
        min_area=protocol.min_component_area,
    )
    post_extraction = label_connected_components(
        post_map > protocol.fixed_probability_threshold,
        connectivity=protocol.connectivity,
        min_area=protocol.min_component_area,
    )
    pre_matching = match_components(
        pre_extraction.components,
        target_extraction.components,
        max_centroid_distance=protocol.max_centroid_distance,
    )
    post_matching = match_components(
        post_extraction.components,
        target_extraction.components,
        max_centroid_distance=protocol.max_centroid_distance,
    )
    pre_by_target = {match.target_index: match for match in pre_matching.matches}
    post_by_target = {match.target_index: match for match in post_matching.matches}

    records = []
    threshold = protocol.fixed_probability_threshold
    for target_index, target_component in enumerate(target_extraction.components):
        target_component_mask = target_extraction.labels == target_component.label
        target_component_mask.setflags(write=False)
        context = TargetScoreContext(
            target_component=target_component,
            target_component_mask=target_component_mask,
            protocol=protocol,
        )
        pre_map.setflags(write=False)
        post_map.setflags(write=False)
        pre_score = float(reducer(pre_map, context))
        post_score = float(reducer(post_map, context))
        if not np.isfinite(pre_score) or not 0.0 <= pre_score <= 1.0:
            raise ValueError(
                "score_reducer must return one finite probability in [0, 1]; "
                f"got pre score {pre_score}."
            )
        if not np.isfinite(post_score) or not 0.0 <= post_score <= 1.0:
            raise ValueError(
                "score_reducer must return one finite probability in [0, 1]; "
                f"got post score {post_score}."
            )

        pre_match = pre_by_target.get(target_index)
        post_match = post_by_target.get(target_index)
        pre_state = "TP" if pre_match is not None else "FN"
        post_state = "TP" if post_match is not None else "FN"
        transition = f"{pre_state}→{post_state}"
        records.append(
            TargetTransitionRecord(
                image_id=image_id,
                comparison_label=comparison_label,
                gt_target_id=target_component.label,
                gt_area=target_component.area,
                gt_centroid=target_component.centroid,
                gt_bbox=target_component.bbox,
                pre_state=pre_state,
                post_state=post_state,
                transition=transition,
                pre_target_score=pre_score,
                post_target_score=post_score,
                score_change=post_score - pre_score,
                probability_threshold=threshold,
                pre_margin_to_threshold=pre_score - threshold,
                post_margin_to_threshold=post_score - threshold,
                pre_match=_prediction_match_record(
                    pre_match, pre_extraction.components
                ),
                post_match=_prediction_match_record(
                    post_match, post_extraction.components
                ),
            )
        )

    target_records = tuple(records)
    return ImageTargetTransitionResult(
        image_id=image_id,
        comparison_label=comparison_label,
        protocol=protocol,
        score_definition=definition,
        transition_counts=_count_records(target_records),
        target_records=target_records,
        pre_prediction_component_count=len(pre_extraction.components),
        post_prediction_component_count=len(post_extraction.components),
        pre_unmatched_prediction_count=len(
            pre_matching.unmatched_prediction_indices
        ),
        post_unmatched_prediction_count=len(
            post_matching.unmatched_prediction_indices
        ),
    )


class TargetTransitionEvaluator:
    """Accumulate single-image target transitions for one comparison."""

    def __init__(
        self,
        protocol: Optional[IRSTDEvaluationProtocol] = None,
        *,
        comparison_label: Optional[str] = None,
        score_reducer: Optional[TargetScoreReducer] = None,
        score_definition: Optional[TargetScoreDefinition] = None,
    ) -> None:
        self.protocol = protocol or IRSTDEvaluationProtocol()
        if not isinstance(self.protocol, IRSTDEvaluationProtocol):
            raise TypeError("protocol must be an IRSTDEvaluationProtocol instance.")
        self.comparison_label = _normalise_optional_label(
            comparison_label, name="comparison_label"
        )
        if score_reducer is None:
            if score_definition is not None:
                raise ValueError(
                    "score_definition cannot be supplied without a custom "
                    "score_reducer."
                )
            self._score_reducer = dilated_gt_neighborhood_max
            self.score_definition = default_score_definition(self.protocol)
        else:
            if not callable(score_reducer):
                raise TypeError("score_reducer must be callable.")
            if score_definition is None:
                raise ValueError(
                    "A custom score_reducer requires an explicit score_definition."
                )
            self._score_reducer = score_reducer
            self.score_definition = score_definition
        self.reset()

    def reset(self) -> None:
        self._image_results: list[ImageTargetTransitionResult] = []
        self._seen_image_ids: set[str] = set()

    @property
    def image_results(self) -> Tuple[ImageTargetTransitionResult, ...]:
        return tuple(self._image_results)

    def update_probabilities(
        self,
        pre_probabilities: Any,
        post_probabilities: Any,
        ground_truth: Any,
        *,
        image_id: Optional[str] = None,
    ) -> ImageTargetTransitionResult:
        normalised_image_id = _normalise_optional_label(image_id, name="image_id")
        if (
            normalised_image_id is not None
            and normalised_image_id in self._seen_image_ids
        ):
            raise ValueError(f"Duplicate image_id: {normalised_image_id!r}.")
        result = evaluate_target_transitions(
            pre_probabilities,
            post_probabilities,
            ground_truth,
            protocol=self.protocol,
            comparison_label=self.comparison_label,
            image_id=normalised_image_id,
            score_reducer=self._score_reducer,
            score_definition=self.score_definition,
        )
        self._image_results.append(result)
        if normalised_image_id is not None:
            self._seen_image_ids.add(normalised_image_id)
        return result

    def compute(self) -> TargetTransitionSummary:
        counts = TransitionCounts()
        for result in self._image_results:
            counts = counts + result.transition_counts
        return TargetTransitionSummary(
            comparison_label=self.comparison_label,
            score_definition=self.score_definition,
            image_count=len(self._image_results),
            total_gt_targets=counts.total,
            transition_counts=counts,
            ater=DefinedRatio.create(counts.tp_to_fn, counts.pre_tp),
            atrr=DefinedRatio.create(counts.fn_to_tp, counts.pre_fn),
            ntg=DefinedRatio.create(counts.net_target_gain_count, counts.total),
            net_target_gain_count=counts.net_target_gain_count,
        )


__all__ = [
    "FN_TO_FN",
    "FN_TO_TP",
    "SCORE_STATUS_CANDIDATE_NOT_FROZEN",
    "SOURCE_TO_POST",
    "TENT_PRE_TO_POST",
    "TP_TO_FN",
    "TP_TO_TP",
    "TRANSITION_LABELS",
    "DefinedRatio",
    "ImageTargetTransitionResult",
    "PredictionMatchRecord",
    "TargetScoreContext",
    "TargetScoreDefinition",
    "TargetScoreReducer",
    "TargetTransitionEvaluator",
    "TargetTransitionRecord",
    "TargetTransitionSummary",
    "TransitionCounts",
    "default_score_definition",
    "dilated_gt_neighborhood_max",
    "evaluate_target_transitions",
]
