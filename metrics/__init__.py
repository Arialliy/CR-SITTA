"""Shared official-reproduction and unified research metrics."""

from .connected_components import (
    ComponentExtraction,
    ConnectedComponent,
    extract_connected_components,
    label_connected_components,
)
from .irstd_metrics import (
    DEFAULT_FROC_THRESHOLDS,
    IRSTDEvaluationProtocol,
    IRSTDEvaluationResult,
    PixelMetrics,
    TargetMetrics,
    ThresholdMetrics,
    UnifiedResearchEvaluator,
    probabilities_from_logits,
)
from .official_metric_adapter import OfficialMetricAdapter, OfficialMetricResult
from .target_matching import (
    TargetMatch,
    TargetMatchResult,
    centroid_distance_matrix,
    match_components,
)

__all__ = [
    "ComponentExtraction",
    "ConnectedComponent",
    "DEFAULT_FROC_THRESHOLDS",
    "IRSTDEvaluationProtocol",
    "IRSTDEvaluationResult",
    "OfficialMetricAdapter",
    "OfficialMetricResult",
    "PixelMetrics",
    "TargetMatch",
    "TargetMatchResult",
    "TargetMetrics",
    "ThresholdMetrics",
    "UnifiedResearchEvaluator",
    "centroid_distance_matrix",
    "extract_connected_components",
    "label_connected_components",
    "match_components",
    "probabilities_from_logits",
]
