"""Versioned, post-hoc three-stage transition evaluation for Binary TENT.

The evaluator consumes three already-materialised probability maps:

``source_pre -> tent_pre -> tent_post``.

Ground truth is materialised only after all three predictions have been copied
and validated.  The module therefore remains an outer evaluation component and
cannot become a label-bearing adaptation callback.

Version 1 intentionally freezes the existing IRSTD operating point: strict
``probability > 0.5``, 8-connected components, minimum component area one, and
one-to-one Hungarian matching under strict centroid distance ``< 3`` pixels.
It also refuses the exploratory ``candidate_not_frozen`` target-score status.
Callers must provide an explicitly frozen score reducer and definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .connected_components import ConnectedComponent, label_connected_components
from .irstd_metrics import IRSTDEvaluationProtocol
from .target_matching import match_components
from .target_transitions import (
    SCORE_STATUS_CANDIDATE_NOT_FROZEN,
    DefinedRatio,
    ImageTargetTransitionResult,
    PredictionMatchRecord,
    TargetScoreDefinition,
    TargetScoreReducer,
    TargetTransitionRecord,
    TargetTransitionSummary,
    TransitionCounts,
    evaluate_target_transitions,
)


THREE_STAGE_PROTOCOL_VERSION = "three_stage_transitions_v1"

SOURCE_PRE = "source_pre"
TENT_PRE = "tent_pre"
TENT_POST = "tent_post"
SNAPSHOT_IDS = (SOURCE_PRE, TENT_PRE, TENT_POST)

SOURCE_TO_TENT_PRE = "source_to_tent_pre"
TENT_PRE_TO_TENT_POST = "tent_pre_to_tent_post"
SOURCE_TO_TENT_POST = "source_to_tent_post"
COMPARISON_IDS = (
    SOURCE_TO_TENT_PRE,
    TENT_PRE_TO_TENT_POST,
    SOURCE_TO_TENT_POST,
)
COMPARISON_ENDPOINTS = {
    SOURCE_TO_TENT_PRE: (SOURCE_PRE, TENT_PRE),
    TENT_PRE_TO_TENT_POST: (TENT_PRE, TENT_POST),
    SOURCE_TO_TENT_POST: (SOURCE_PRE, TENT_POST),
}

FP_TO_FP = "FP→FP"
FP_TO_BG = "FP→BG"
BG_TO_FP = "BG→FP"
FP_TRANSITION_LABELS = (FP_TO_FP, FP_TO_BG, BG_TO_FP)

SCORE_STATUS_FORMAL_FROZEN_V1 = "formal_frozen_v1"


class ThreeStageTransitionProtocolError(ValueError):
    """Raised when a formal three-stage evaluation contract is violated."""


def _to_numpy(value: Any) -> NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_single_map(value: Any, *, name: str) -> NDArray[Any]:
    array = _to_numpy(value)
    if array.ndim == 2:
        normalised = array
    elif array.ndim == 3 and array.shape[0] == 1:
        normalised = array[0]
    elif array.ndim == 4 and array.shape[:2] == (1, 1):
        normalised = array[0, 0]
    else:
        raise ValueError(
            f"{name} must describe one single-channel image with shape [H,W], "
            f"[1,H,W], or [1,1,H,W]; got {array.shape}."
        )
    if normalised.shape[0] == 0 or normalised.shape[1] == 0:
        raise ValueError(f"{name} cannot contain empty dimensions.")
    if not np.isfinite(normalised).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return np.array(normalised, copy=True)


def _as_probability_map(value: Any, *, name: str) -> NDArray[Any]:
    probability = _as_single_map(value, name=name)
    if not np.issubdtype(probability.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype.")
    if (probability < 0.0).any() or (probability > 1.0).any():
        raise ValueError(f"{name} must lie in [0, 1].")
    probability.setflags(write=False)
    return probability


def _as_target_map(value: Any) -> NDArray[np.bool_]:
    target = _as_single_map(value, name="ground_truth") > 0
    target.setflags(write=False)
    return target


def _normalise_image_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image_id must be None or a non-empty string.")
    return value.strip()


def _maps_bit_exact(left: NDArray[Any], right: NDArray[Any]) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype.str == right.dtype.str
        and np.array_equal(left, right)
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _validate_formal_protocol(protocol: IRSTDEvaluationProtocol) -> None:
    if not isinstance(protocol, IRSTDEvaluationProtocol):
        raise TypeError("protocol must be an IRSTDEvaluationProtocol instance.")
    expected = {
        "fixed_probability_threshold": 0.5,
        "connectivity": 2,
        "max_centroid_distance": 3.0,
        "min_component_area": 1,
    }
    observed = {
        "fixed_probability_threshold": protocol.fixed_probability_threshold,
        "connectivity": protocol.connectivity,
        "max_centroid_distance": protocol.max_centroid_distance,
        "min_component_area": protocol.min_component_area,
    }
    if observed != expected:
        raise ThreeStageTransitionProtocolError(
            f"{THREE_STAGE_PROTOCOL_VERSION} requires {expected}, got {observed}."
        )


def _validate_formal_score_definition(
    score_reducer: TargetScoreReducer,
    score_definition: TargetScoreDefinition,
) -> None:
    if not callable(score_reducer):
        raise TypeError("score_reducer must be callable.")
    if not isinstance(score_definition, TargetScoreDefinition):
        raise TypeError("score_definition must be a TargetScoreDefinition.")
    if score_definition.protocol_status == SCORE_STATUS_CANDIDATE_NOT_FROZEN:
        raise ThreeStageTransitionProtocolError(
            "formal three-stage evaluation refuses candidate_not_frozen target "
            "score definitions"
        )
    if score_definition.protocol_status != SCORE_STATUS_FORMAL_FROZEN_V1:
        raise ThreeStageTransitionProtocolError(
            "formal three-stage evaluation requires score protocol_status "
            f"{SCORE_STATUS_FORMAL_FROZEN_V1!r}; got "
            f"{score_definition.protocol_status!r}"
        )


@dataclass(frozen=True)
class FalsePositiveComponentRecord:
    """Geometry of one prediction component unmatched to every GT target."""

    snapshot_id: str
    prediction_index: int
    prediction_label: int
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnapshotEndpointCounts:
    """Fixed-operating-point target and false-positive counts for one snapshot."""

    snapshot_id: str
    gt_target_count: int
    prediction_component_count: int
    true_positive_targets: int
    false_negative_targets: int
    false_positive_components: int
    false_positive_records: Tuple[FalsePositiveComponentRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "gt_target_count": self.gt_target_count,
            "prediction_component_count": self.prediction_component_count,
            "true_positive_targets": self.true_positive_targets,
            "false_negative_targets": self.false_negative_targets,
            "false_positive_components": self.false_positive_components,
            "false_positive_records": [
                record.to_dict() for record in self.false_positive_records
            ],
        }


@dataclass(frozen=True)
class FalsePositiveTransitionRecord:
    """One persistent, removed, or newly induced false-positive component."""

    comparison_id: str
    transition: str
    pre_component: Optional[FalsePositiveComponentRecord]
    post_component: Optional[FalsePositiveComponentRecord]
    centroid_distance: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "transition": self.transition,
            "pre_component": (
                None if self.pre_component is None else self.pre_component.to_dict()
            ),
            "post_component": (
                None if self.post_component is None else self.post_component.to_dict()
            ),
            "centroid_distance": self.centroid_distance,
        }


@dataclass(frozen=True)
class FalsePositiveTransitionCounts:
    fp_to_fp: int = 0
    fp_to_bg: int = 0
    bg_to_fp: int = 0

    @property
    def pre_fp(self) -> int:
        return self.fp_to_fp + self.fp_to_bg

    @property
    def post_fp(self) -> int:
        return self.fp_to_fp + self.bg_to_fp

    def __add__(
        self, other: "FalsePositiveTransitionCounts"
    ) -> "FalsePositiveTransitionCounts":
        if not isinstance(other, FalsePositiveTransitionCounts):
            return NotImplemented
        return FalsePositiveTransitionCounts(
            fp_to_fp=self.fp_to_fp + other.fp_to_fp,
            fp_to_bg=self.fp_to_bg + other.fp_to_bg,
            bg_to_fp=self.bg_to_fp + other.bg_to_fp,
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            FP_TO_FP: self.fp_to_fp,
            FP_TO_BG: self.fp_to_bg,
            BG_TO_FP: self.bg_to_fp,
        }


@dataclass(frozen=True)
class ImageFalsePositiveTransitionResult:
    comparison_id: str
    pre_snapshot_id: str
    post_snapshot_id: str
    counts: FalsePositiveTransitionCounts
    records: Tuple[FalsePositiveTransitionRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "pre_snapshot_id": self.pre_snapshot_id,
            "post_snapshot_id": self.post_snapshot_id,
            "matching_rule": (
                "one_to_one_hungarian_between_gt_unmatched_prediction_components_"
                "with_centroid_distance_strictly_less_than_3_pixels"
            ),
            "counts": self.counts.to_dict(),
            "pre_fp_count": self.counts.pre_fp,
            "post_fp_count": self.counts.post_fp,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class JointTargetTrajectoryRecord:
    """One GT target tracked across all three prediction snapshots."""

    image_id: Optional[str]
    gt_target_id: int
    gt_area: int
    gt_centroid: Tuple[float, float]
    gt_bbox: Tuple[int, int, int, int]
    source_state: str
    tent_pre_state: str
    tent_post_state: str
    source_target_score: float
    tent_pre_target_score: float
    tent_post_target_score: float
    source_margin_to_threshold: float
    tent_pre_margin_to_threshold: float
    tent_post_margin_to_threshold: float
    source_match: Optional[PredictionMatchRecord]
    tent_pre_match: Optional[PredictionMatchRecord]
    tent_post_match: Optional[PredictionMatchRecord]
    source_to_tent_pre_transition: str
    tent_pre_to_tent_post_transition: str
    source_to_tent_post_transition: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["source_match"] = (
            None if self.source_match is None else self.source_match.to_dict()
        )
        payload["tent_pre_match"] = (
            None if self.tent_pre_match is None else self.tent_pre_match.to_dict()
        )
        payload["tent_post_match"] = (
            None if self.tent_post_match is None else self.tent_post_match.to_dict()
        )
        return payload


@dataclass(frozen=True)
class ImageThreeStageTransitionResult:
    image_id: Optional[str]
    protocol: IRSTDEvaluationProtocol
    score_definition: TargetScoreDefinition
    target_transition_results: Tuple[ImageTargetTransitionResult, ...]
    fp_transition_results: Tuple[ImageFalsePositiveTransitionResult, ...]
    endpoint_counts: Tuple[SnapshotEndpointCounts, ...]
    joint_target_trajectories: Tuple[JointTargetTrajectoryRecord, ...]
    tent_ss_identity_required: bool
    tent_ss_source_tent_pre_bit_exact: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": THREE_STAGE_PROTOCOL_VERSION,
            "image_id": self.image_id,
            "protocol": asdict(self.protocol),
            "score_definition": self.score_definition.to_dict(),
            "comparison_ids": list(COMPARISON_IDS),
            "target_transitions": {
                result.comparison_label: result.to_dict()
                for result in self.target_transition_results
            },
            "false_positive_transitions": {
                result.comparison_id: result.to_dict()
                for result in self.fp_transition_results
            },
            "endpoints": {
                result.snapshot_id: result.to_dict()
                for result in self.endpoint_counts
            },
            "joint_target_trajectories": [
                record.to_dict() for record in self.joint_target_trajectories
            ],
            "checks": {
                "formal_score_definition_frozen": True,
                "target_endpoint_conservation": True,
                "false_positive_endpoint_conservation": True,
                "joint_target_ids_and_geometry_consistent": True,
                "tent_ss_identity_required": self.tent_ss_identity_required,
                "tent_ss_source_tent_pre_bit_exact": (
                    self.tent_ss_source_tent_pre_bit_exact
                ),
            },
        }


@dataclass(frozen=True)
class AggregateSnapshotEndpointCounts:
    snapshot_id: str
    image_count: int
    gt_target_count: int
    prediction_component_count: int
    true_positive_targets: int
    false_negative_targets: int
    false_positive_components: int

    def to_dict(self) -> Dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateFalsePositiveTransitionResult:
    comparison_id: str
    image_count: int
    counts: FalsePositiveTransitionCounts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "image_count": self.image_count,
            "counts": self.counts.to_dict(),
            "pre_fp_count": self.counts.pre_fp,
            "post_fp_count": self.counts.post_fp,
        }


@dataclass(frozen=True)
class ThreeStageTransitionSummary:
    protocol: IRSTDEvaluationProtocol
    score_definition: TargetScoreDefinition
    image_count: int
    target_transition_summaries: Tuple[TargetTransitionSummary, ...]
    fp_transition_summaries: Tuple[AggregateFalsePositiveTransitionResult, ...]
    endpoint_counts: Tuple[AggregateSnapshotEndpointCounts, ...]
    tent_ss_identity_required: bool
    tent_ss_identity_verified: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": THREE_STAGE_PROTOCOL_VERSION,
            "protocol": asdict(self.protocol),
            "score_definition": self.score_definition.to_dict(),
            "image_count": self.image_count,
            "target_transitions": {
                result.comparison_label: result.to_dict()
                for result in self.target_transition_summaries
            },
            "false_positive_transitions": {
                result.comparison_id: result.to_dict()
                for result in self.fp_transition_summaries
            },
            "endpoints": {
                result.snapshot_id: result.to_dict()
                for result in self.endpoint_counts
            },
            "checks": {
                "formal_score_definition_frozen": True,
                "target_endpoint_conservation": True,
                "false_positive_endpoint_conservation": True,
                "tent_ss_identity_required": self.tent_ss_identity_required,
                "tent_ss_identity_verified": self.tent_ss_identity_verified,
            },
        }


@dataclass(frozen=True)
class _SnapshotState:
    endpoint: SnapshotEndpointCounts
    false_positive_components: Tuple[ConnectedComponent, ...]


def _false_positive_record(
    *, snapshot_id: str, prediction_index: int, component: ConnectedComponent
) -> FalsePositiveComponentRecord:
    return FalsePositiveComponentRecord(
        snapshot_id=snapshot_id,
        prediction_index=prediction_index,
        prediction_label=component.label,
        area=component.area,
        centroid=component.centroid,
        bbox=component.bbox,
    )


def _snapshot_state(
    probability_map: NDArray[Any],
    target_map: NDArray[np.bool_],
    *,
    snapshot_id: str,
    protocol: IRSTDEvaluationProtocol,
) -> _SnapshotState:
    target_extraction = label_connected_components(
        target_map,
        connectivity=protocol.connectivity,
        min_area=protocol.min_component_area,
    )
    prediction_extraction = label_connected_components(
        probability_map > protocol.fixed_probability_threshold,
        connectivity=protocol.connectivity,
        min_area=protocol.min_component_area,
    )
    matching = match_components(
        prediction_extraction.components,
        target_extraction.components,
        max_centroid_distance=protocol.max_centroid_distance,
    )
    false_positive_components = tuple(
        prediction_extraction.components[index]
        for index in matching.unmatched_prediction_indices
    )
    false_positive_records = tuple(
        _false_positive_record(
            snapshot_id=snapshot_id,
            prediction_index=prediction_index,
            component=prediction_extraction.components[prediction_index],
        )
        for prediction_index in matching.unmatched_prediction_indices
    )
    endpoint = SnapshotEndpointCounts(
        snapshot_id=snapshot_id,
        gt_target_count=len(target_extraction.components),
        prediction_component_count=len(prediction_extraction.components),
        true_positive_targets=matching.true_positives,
        false_negative_targets=matching.false_negatives,
        false_positive_components=matching.false_positives,
        false_positive_records=false_positive_records,
    )
    return _SnapshotState(
        endpoint=endpoint,
        false_positive_components=false_positive_components,
    )


def _evaluate_fp_transitions(
    pre: _SnapshotState,
    post: _SnapshotState,
    *,
    comparison_id: str,
    protocol: IRSTDEvaluationProtocol,
) -> ImageFalsePositiveTransitionResult:
    matching = match_components(
        pre.false_positive_components,
        post.false_positive_components,
        max_centroid_distance=protocol.max_centroid_distance,
    )
    pre_records = pre.endpoint.false_positive_records
    post_records = post.endpoint.false_positive_records
    records = []
    for match in matching.matches:
        records.append(
            FalsePositiveTransitionRecord(
                comparison_id=comparison_id,
                transition=FP_TO_FP,
                pre_component=pre_records[match.prediction_index],
                post_component=post_records[match.target_index],
                centroid_distance=match.centroid_distance,
            )
        )
    for index in matching.unmatched_prediction_indices:
        records.append(
            FalsePositiveTransitionRecord(
                comparison_id=comparison_id,
                transition=FP_TO_BG,
                pre_component=pre_records[index],
                post_component=None,
                centroid_distance=None,
            )
        )
    for index in matching.unmatched_target_indices:
        records.append(
            FalsePositiveTransitionRecord(
                comparison_id=comparison_id,
                transition=BG_TO_FP,
                pre_component=None,
                post_component=post_records[index],
                centroid_distance=None,
            )
        )
    counts = FalsePositiveTransitionCounts(
        fp_to_fp=len(matching.matches),
        fp_to_bg=len(matching.unmatched_prediction_indices),
        bg_to_fp=len(matching.unmatched_target_indices),
    )
    if counts.pre_fp != pre.endpoint.false_positive_components:
        raise ThreeStageTransitionProtocolError(
            f"pre-FP conservation failed for {comparison_id}"
        )
    if counts.post_fp != post.endpoint.false_positive_components:
        raise ThreeStageTransitionProtocolError(
            f"post-FP conservation failed for {comparison_id}"
        )
    return ImageFalsePositiveTransitionResult(
        comparison_id=comparison_id,
        pre_snapshot_id=pre.endpoint.snapshot_id,
        post_snapshot_id=post.endpoint.snapshot_id,
        counts=counts,
        records=tuple(records),
    )


def _assert_target_endpoint_conservation(
    result: ImageTargetTransitionResult,
    pre: SnapshotEndpointCounts,
    post: SnapshotEndpointCounts,
) -> None:
    counts = result.transition_counts
    post_tp = counts.tp_to_tp + counts.fn_to_tp
    post_fn = counts.tp_to_fn + counts.fn_to_fn
    checks = {
        "total_gt_pre": result.transition_counts.total == pre.gt_target_count,
        "total_gt_post": result.transition_counts.total == post.gt_target_count,
        "pre_tp": counts.pre_tp == pre.true_positive_targets,
        "pre_fn": counts.pre_fn == pre.false_negative_targets,
        "post_tp": post_tp == post.true_positive_targets,
        "post_fn": post_fn == post.false_negative_targets,
        "pre_fp": result.pre_unmatched_prediction_count
        == pre.false_positive_components,
        "post_fp": result.post_unmatched_prediction_count
        == post.false_positive_components,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ThreeStageTransitionProtocolError(
            f"target endpoint conservation failed for {result.comparison_label}: "
            + ", ".join(failed)
        )


def _record_by_target_id(
    result: ImageTargetTransitionResult,
) -> Dict[int, TargetTransitionRecord]:
    records = {record.gt_target_id: record for record in result.target_records}
    if len(records) != len(result.target_records):
        raise ThreeStageTransitionProtocolError("duplicate gt_target_id detected")
    return records


def _require_equal(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise ThreeStageTransitionProtocolError(
            f"three-stage target trajectory mismatch for {label}"
        )


def _joint_target_trajectories(
    source_to_pre: ImageTargetTransitionResult,
    pre_to_post: ImageTargetTransitionResult,
    source_to_post: ImageTargetTransitionResult,
) -> Tuple[JointTargetTrajectoryRecord, ...]:
    first = _record_by_target_id(source_to_pre)
    second = _record_by_target_id(pre_to_post)
    direct = _record_by_target_id(source_to_post)
    if set(first) != set(second) or set(first) != set(direct):
        raise ThreeStageTransitionProtocolError(
            "gt_target_id sets differ across three comparisons"
        )

    trajectories = []
    for target_id in sorted(first):
        source_pre = first[target_id]
        pre_post = second[target_id]
        source_post = direct[target_id]
        geometry = (
            source_pre.gt_area,
            source_pre.gt_centroid,
            source_pre.gt_bbox,
        )
        _require_equal(
            (pre_post.gt_area, pre_post.gt_centroid, pre_post.gt_bbox),
            geometry,
            label=f"GT geometry at {target_id} (pre/post)",
        )
        _require_equal(
            (source_post.gt_area, source_post.gt_centroid, source_post.gt_bbox),
            geometry,
            label=f"GT geometry at {target_id} (source/post)",
        )
        for value, expected, label in (
            (source_pre.pre_state, source_post.pre_state, "source state"),
            (source_pre.post_state, pre_post.pre_state, "tent-pre state"),
            (pre_post.post_state, source_post.post_state, "tent-post state"),
            (
                source_pre.pre_target_score,
                source_post.pre_target_score,
                "source score",
            ),
            (
                source_pre.post_target_score,
                pre_post.pre_target_score,
                "tent-pre score",
            ),
            (
                pre_post.post_target_score,
                source_post.post_target_score,
                "tent-post score",
            ),
            (source_pre.pre_match, source_post.pre_match, "source match"),
            (source_pre.post_match, pre_post.pre_match, "tent-pre match"),
            (pre_post.post_match, source_post.post_match, "tent-post match"),
        ):
            _require_equal(value, expected, label=f"{label} at {target_id}")
        trajectories.append(
            JointTargetTrajectoryRecord(
                image_id=source_pre.image_id,
                gt_target_id=target_id,
                gt_area=source_pre.gt_area,
                gt_centroid=source_pre.gt_centroid,
                gt_bbox=source_pre.gt_bbox,
                source_state=source_pre.pre_state,
                tent_pre_state=source_pre.post_state,
                tent_post_state=pre_post.post_state,
                source_target_score=source_pre.pre_target_score,
                tent_pre_target_score=source_pre.post_target_score,
                tent_post_target_score=pre_post.post_target_score,
                source_margin_to_threshold=source_pre.pre_margin_to_threshold,
                tent_pre_margin_to_threshold=source_pre.post_margin_to_threshold,
                tent_post_margin_to_threshold=pre_post.post_margin_to_threshold,
                source_match=source_pre.pre_match,
                tent_pre_match=source_pre.post_match,
                tent_post_match=pre_post.post_match,
                source_to_tent_pre_transition=source_pre.transition,
                tent_pre_to_tent_post_transition=pre_post.transition,
                source_to_tent_post_transition=source_post.transition,
            )
        )
    return tuple(trajectories)


def _same_endpoint_counts(
    left: SnapshotEndpointCounts, right: SnapshotEndpointCounts
) -> bool:
    return (
        left.gt_target_count == right.gt_target_count
        and left.prediction_component_count == right.prediction_component_count
        and left.true_positive_targets == right.true_positive_targets
        and left.false_negative_targets == right.false_negative_targets
        and left.false_positive_components == right.false_positive_components
        and tuple(
            (
                record.prediction_index,
                record.prediction_label,
                record.area,
                record.centroid,
                record.bbox,
            )
            for record in left.false_positive_records
        )
        == tuple(
            (
                record.prediction_index,
                record.prediction_label,
                record.area,
                record.centroid,
                record.bbox,
            )
            for record in right.false_positive_records
        )
    )


class ThreeStageTransitionEvaluator:
    """Accumulate version-1 three-stage target and false-positive transitions."""

    def __init__(
        self,
        protocol: IRSTDEvaluationProtocol,
        *,
        score_reducer: TargetScoreReducer,
        score_definition: TargetScoreDefinition,
        tent_ss_identity_required: bool = False,
    ) -> None:
        _validate_formal_protocol(protocol)
        _validate_formal_score_definition(score_reducer, score_definition)
        if not isinstance(tent_ss_identity_required, bool):
            raise TypeError("tent_ss_identity_required must be bool.")
        self.protocol = protocol
        self.score_reducer = score_reducer
        self.score_definition = score_definition
        self.tent_ss_identity_required = tent_ss_identity_required
        self.reset()

    def reset(self) -> None:
        self._image_results: list[ImageThreeStageTransitionResult] = []
        self._seen_image_ids: set[str] = set()

    @property
    def image_results(self) -> Tuple[ImageThreeStageTransitionResult, ...]:
        return tuple(self._image_results)

    def update_probabilities(
        self,
        source_probabilities: Any,
        tent_pre_probabilities: Any,
        tent_post_probabilities: Any,
        ground_truth: Any,
        *,
        image_id: Optional[str] = None,
    ) -> ImageThreeStageTransitionResult:
        image_id = _normalise_image_id(image_id)
        if image_id is not None and image_id in self._seen_image_ids:
            raise ValueError(f"Duplicate image_id: {image_id!r}.")

        # Prediction materialisation is deliberately complete before GT access.
        source_map = _as_probability_map(
            source_probabilities, name="source_probabilities"
        )
        tent_pre_map = _as_probability_map(
            tent_pre_probabilities, name="tent_pre_probabilities"
        )
        tent_post_map = _as_probability_map(
            tent_post_probabilities, name="tent_post_probabilities"
        )
        if source_map.shape != tent_pre_map.shape or source_map.shape != tent_post_map.shape:
            raise ValueError(
                "source, TENT-pre, and TENT-post probability maps must have the "
                "same normalised shape"
            )
        tent_ss_exact: Optional[bool] = None
        if self.tent_ss_identity_required:
            tent_ss_exact = _maps_bit_exact(source_map, tent_pre_map)
            if not tent_ss_exact:
                raise ThreeStageTransitionProtocolError(
                    "TENT-SS requires source_pre and tent_pre probability maps to "
                    "be bit-exact"
                )

        target_map = _as_target_map(ground_truth)
        if source_map.shape != target_map.shape:
            raise ValueError(
                "all probability maps and ground_truth must have the same "
                f"normalised shape; got {source_map.shape} and {target_map.shape}."
            )

        maps = {
            SOURCE_PRE: source_map,
            TENT_PRE: tent_pre_map,
            TENT_POST: tent_post_map,
        }
        snapshot_states = {
            snapshot_id: _snapshot_state(
                probability_map,
                target_map,
                snapshot_id=snapshot_id,
                protocol=self.protocol,
            )
            for snapshot_id, probability_map in maps.items()
        }

        target_results = []
        fp_results = []
        for comparison_id in COMPARISON_IDS:
            pre_id, post_id = COMPARISON_ENDPOINTS[comparison_id]
            target_result = evaluate_target_transitions(
                maps[pre_id],
                maps[post_id],
                target_map,
                protocol=self.protocol,
                comparison_label=comparison_id,
                image_id=image_id,
                score_reducer=self.score_reducer,
                score_definition=self.score_definition,
            )
            _assert_target_endpoint_conservation(
                target_result,
                snapshot_states[pre_id].endpoint,
                snapshot_states[post_id].endpoint,
            )
            target_results.append(target_result)
            fp_results.append(
                _evaluate_fp_transitions(
                    snapshot_states[pre_id],
                    snapshot_states[post_id],
                    comparison_id=comparison_id,
                    protocol=self.protocol,
                )
            )

        target_by_id = {
            result.comparison_label: result for result in target_results
        }
        trajectories = _joint_target_trajectories(
            target_by_id[SOURCE_TO_TENT_PRE],
            target_by_id[TENT_PRE_TO_TENT_POST],
            target_by_id[SOURCE_TO_TENT_POST],
        )

        endpoints = tuple(
            snapshot_states[snapshot_id].endpoint for snapshot_id in SNAPSHOT_IDS
        )
        if self.tent_ss_identity_required:
            endpoint_by_id = {value.snapshot_id: value for value in endpoints}
            if not _same_endpoint_counts(
                endpoint_by_id[SOURCE_PRE], endpoint_by_id[TENT_PRE]
            ):
                raise ThreeStageTransitionProtocolError(
                    "TENT-SS source and TENT-pre endpoint counts/geometry differ"
                )
            identity_counts = target_by_id[
                SOURCE_TO_TENT_PRE
            ].transition_counts
            if identity_counts.tp_to_fn != 0 or identity_counts.fn_to_tp != 0:
                raise ThreeStageTransitionProtocolError(
                    "TENT-SS identity comparison contains a changed target state"
                )
            if any(
                trajectory.source_state != trajectory.tent_pre_state
                or trajectory.source_target_score
                != trajectory.tent_pre_target_score
                or trajectory.source_match != trajectory.tent_pre_match
                for trajectory in trajectories
            ):
                raise ThreeStageTransitionProtocolError(
                    "TENT-SS source and TENT-pre target trajectories differ"
                )
            if target_by_id[
                TENT_PRE_TO_TENT_POST
            ].transition_counts != target_by_id[
                SOURCE_TO_TENT_POST
            ].transition_counts:
                raise ThreeStageTransitionProtocolError(
                    "TENT-SS direct and gradient-only target transitions differ"
                )

        result = ImageThreeStageTransitionResult(
            image_id=image_id,
            protocol=self.protocol,
            score_definition=self.score_definition,
            target_transition_results=tuple(target_results),
            fp_transition_results=tuple(fp_results),
            endpoint_counts=endpoints,
            joint_target_trajectories=trajectories,
            tent_ss_identity_required=self.tent_ss_identity_required,
            tent_ss_source_tent_pre_bit_exact=tent_ss_exact,
        )
        self._image_results.append(result)
        if image_id is not None:
            self._seen_image_ids.add(image_id)
        return result

    def compute(self) -> ThreeStageTransitionSummary:
        endpoint_totals = []
        for snapshot_id in SNAPSHOT_IDS:
            values = [
                next(
                    endpoint
                    for endpoint in result.endpoint_counts
                    if endpoint.snapshot_id == snapshot_id
                )
                for result in self._image_results
            ]
            endpoint_totals.append(
                AggregateSnapshotEndpointCounts(
                    snapshot_id=snapshot_id,
                    image_count=len(values),
                    gt_target_count=sum(value.gt_target_count for value in values),
                    prediction_component_count=sum(
                        value.prediction_component_count for value in values
                    ),
                    true_positive_targets=sum(
                        value.true_positive_targets for value in values
                    ),
                    false_negative_targets=sum(
                        value.false_negative_targets for value in values
                    ),
                    false_positive_components=sum(
                        value.false_positive_components for value in values
                    ),
                )
            )

        fp_summaries = []
        for comparison_id in COMPARISON_IDS:
            counts = FalsePositiveTransitionCounts()
            for image_result in self._image_results:
                result = next(
                    value
                    for value in image_result.fp_transition_results
                    if value.comparison_id == comparison_id
                )
                counts = counts + result.counts
            fp_summaries.append(
                AggregateFalsePositiveTransitionResult(
                    comparison_id=comparison_id,
                    image_count=len(self._image_results),
                    counts=counts,
                )
            )

        endpoint_by_id = {value.snapshot_id: value for value in endpoint_totals}
        target_summaries_list = []
        for comparison_id in COMPARISON_IDS:
            counts = TransitionCounts()
            for image_result in self._image_results:
                result = next(
                    value
                    for value in image_result.target_transition_results
                    if value.comparison_label == comparison_id
                )
                counts = counts + result.transition_counts
            target_summaries_list.append(
                TargetTransitionSummary(
                    comparison_label=comparison_id,
                    score_definition=self.score_definition,
                    image_count=len(self._image_results),
                    total_gt_targets=counts.total,
                    transition_counts=counts,
                    ater=DefinedRatio.create(counts.tp_to_fn, counts.pre_tp),
                    atrr=DefinedRatio.create(counts.fn_to_tp, counts.pre_fn),
                    ntg=DefinedRatio.create(counts.net_target_gain_count, counts.total),
                    net_target_gain_count=counts.net_target_gain_count,
                )
            )
        target_summaries = tuple(target_summaries_list)
        target_by_id = {
            value.comparison_label: value for value in target_summaries
        }
        fp_by_id = {value.comparison_id: value for value in fp_summaries}
        for comparison_id in COMPARISON_IDS:
            pre_id, post_id = COMPARISON_ENDPOINTS[comparison_id]
            target_counts = target_by_id[comparison_id].transition_counts
            if (
                target_counts.pre_tp
                != endpoint_by_id[pre_id].true_positive_targets
                or target_counts.pre_fn
                != endpoint_by_id[pre_id].false_negative_targets
                or target_counts.tp_to_tp + target_counts.fn_to_tp
                != endpoint_by_id[post_id].true_positive_targets
                or target_counts.tp_to_fn + target_counts.fn_to_fn
                != endpoint_by_id[post_id].false_negative_targets
            ):
                raise ThreeStageTransitionProtocolError(
                    f"aggregate target conservation failed for {comparison_id}"
                )
            fp_counts = fp_by_id[comparison_id].counts
            if (
                fp_counts.pre_fp
                != endpoint_by_id[pre_id].false_positive_components
                or fp_counts.post_fp
                != endpoint_by_id[post_id].false_positive_components
            ):
                raise ThreeStageTransitionProtocolError(
                    f"aggregate FP conservation failed for {comparison_id}"
                )

        identity_verified: Optional[bool]
        if self.tent_ss_identity_required:
            identity_verified = all(
                result.tent_ss_source_tent_pre_bit_exact is True
                for result in self._image_results
            )
            if not identity_verified:
                raise ThreeStageTransitionProtocolError(
                    "aggregate TENT-SS identity verification failed"
                )
        else:
            identity_verified = None

        return ThreeStageTransitionSummary(
            protocol=self.protocol,
            score_definition=self.score_definition,
            image_count=len(self._image_results),
            target_transition_summaries=target_summaries,
            fp_transition_summaries=tuple(fp_summaries),
            endpoint_counts=tuple(endpoint_totals),
            tent_ss_identity_required=self.tent_ss_identity_required,
            tent_ss_identity_verified=identity_verified,
        )


__all__ = [
    "BG_TO_FP",
    "COMPARISON_ENDPOINTS",
    "COMPARISON_IDS",
    "FP_TO_BG",
    "FP_TO_FP",
    "FP_TRANSITION_LABELS",
    "SCORE_STATUS_FORMAL_FROZEN_V1",
    "SNAPSHOT_IDS",
    "SOURCE_PRE",
    "SOURCE_TO_TENT_POST",
    "SOURCE_TO_TENT_PRE",
    "TENT_POST",
    "TENT_PRE",
    "TENT_PRE_TO_TENT_POST",
    "THREE_STAGE_PROTOCOL_VERSION",
    "AggregateFalsePositiveTransitionResult",
    "AggregateSnapshotEndpointCounts",
    "FalsePositiveComponentRecord",
    "FalsePositiveTransitionCounts",
    "FalsePositiveTransitionRecord",
    "ImageFalsePositiveTransitionResult",
    "ImageThreeStageTransitionResult",
    "JointTargetTrajectoryRecord",
    "SnapshotEndpointCounts",
    "ThreeStageTransitionEvaluator",
    "ThreeStageTransitionProtocolError",
    "ThreeStageTransitionSummary",
]
