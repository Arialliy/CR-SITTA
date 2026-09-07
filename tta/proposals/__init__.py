"""Candidate mechanisms evaluated before CR-SITTA parameter adaptation."""

from .gradient_consensus import (
    ConsensusResult,
    GradientConsensusError,
    combine_two_gradients,
)
from .source_multiview_teacher import (
    CONTEXT_TILE_CROP_SIZE,
    CONTEXT_TILE_INPUT_SIZE,
    CONTEXT_TILE_ORIGINS,
    GEOMETRIC_VIEWS,
    AggregationMethod,
    GeometricView,
    LogitsAdapter,
    aggregate_aligned_probabilities,
    apply_geometric_view,
    build_aligned_multiview_teacher,
    build_aligned_view_probabilities,
    extract_context_tiles,
    frozen_probability_forward,
    infer_context_tile_probability,
    inverse_geometric_view,
    stitch_context_tile_probabilities,
)
from .two_sided_safety import (
    CandidateSafetyAudit,
    SafetyDecision,
    TwoSidedSafetyError,
    check_two_sided_safety,
)

TwoSidedSafetyDecision = SafetyDecision

__all__ = [
    "AggregationMethod",
    "CandidateSafetyAudit",
    "ConsensusResult",
    "CONTEXT_TILE_CROP_SIZE",
    "CONTEXT_TILE_INPUT_SIZE",
    "CONTEXT_TILE_ORIGINS",
    "GEOMETRIC_VIEWS",
    "GeometricView",
    "GradientConsensusError",
    "LogitsAdapter",
    "SafetyDecision",
    "TwoSidedSafetyDecision",
    "TwoSidedSafetyError",
    "aggregate_aligned_probabilities",
    "apply_geometric_view",
    "build_aligned_multiview_teacher",
    "build_aligned_view_probabilities",
    "check_two_sided_safety",
    "combine_two_gradients",
    "extract_context_tiles",
    "frozen_probability_forward",
    "infer_context_tile_probability",
    "inverse_geometric_view",
    "stitch_context_tile_probabilities",
]
