"""Label-free, fail-closed objective primitives for CR-SITTA Stage-B."""

from ._validation import StageBObjectiveError
from .balanced_entropy import (
    BalancedEntropyOutput,
    balanced_binary_entropy,
    balanced_entropy,
    bernoulli_entropy_map,
    foreground_background_balanced_entropy,
)
from .feature_statistics import (
    FeatureStatisticsAlignmentOutput,
    SourceFeatureStatistics,
    feature_statistics_alignment,
    feature_statistics_alignment_components,
    feature_statistics_alignment_loss,
    multi_layer_feature_statistics_alignment,
)
from .foreground_mass_guard import (
    ForegroundMassGuardOutput,
    foreground_mass_guard,
    foreground_mass_guard_components,
    foreground_mass_guard_from_logits,
)
from .parameter_anchor import (
    ParameterCollection,
    parameter_anchor,
    parameter_anchor_loss,
)
from .region_balanced_consistency import (
    ConsistencyDivergence,
    MultiViewConsistencyOutput,
    RegionConsistencyOutput,
    RegionLossOutput,
    SourceAnchoredRegionWeights,
    bernoulli_jensen_shannon_map,
    build_source_anchored_region_weights,
    foreground_soft_bce_consistency,
    region_balanced_consistency,
    region_balanced_loss,
    region_balanced_multiview_consistency,
    reliable_background_soft_bce,
    source_anchored_multiview_consistency,
)
from .soft_iou_anchor import (
    ForegroundOverlap,
    foreground_soft_dice_anchor,
    foreground_soft_iou_anchor,
    soft_dice_anchor,
    soft_iou_anchor,
)


__all__ = [
    "BalancedEntropyOutput",
    "ConsistencyDivergence",
    "FeatureStatisticsAlignmentOutput",
    "ForegroundMassGuardOutput",
    "ForegroundOverlap",
    "MultiViewConsistencyOutput",
    "ParameterCollection",
    "RegionConsistencyOutput",
    "RegionLossOutput",
    "SourceAnchoredRegionWeights",
    "SourceFeatureStatistics",
    "StageBObjectiveError",
    "balanced_binary_entropy",
    "balanced_entropy",
    "bernoulli_entropy_map",
    "bernoulli_jensen_shannon_map",
    "build_source_anchored_region_weights",
    "feature_statistics_alignment",
    "feature_statistics_alignment_components",
    "feature_statistics_alignment_loss",
    "foreground_background_balanced_entropy",
    "foreground_mass_guard",
    "foreground_mass_guard_components",
    "foreground_mass_guard_from_logits",
    "foreground_soft_bce_consistency",
    "foreground_soft_dice_anchor",
    "foreground_soft_iou_anchor",
    "multi_layer_feature_statistics_alignment",
    "parameter_anchor",
    "parameter_anchor_loss",
    "region_balanced_consistency",
    "region_balanced_loss",
    "region_balanced_multiview_consistency",
    "reliable_background_soft_bce",
    "soft_dice_anchor",
    "soft_iou_anchor",
    "source_anchored_multiview_consistency",
]
