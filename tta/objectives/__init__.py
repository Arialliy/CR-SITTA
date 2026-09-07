"""Label-free, fail-closed objective primitives for CR-SITTA Stage-B."""

from ._validation import StageBObjectiveError
from .active_bootstrap import (
    ActiveBootstrapOutput,
    active_bootstrap_binary,
    bernoulli_probability_entropy,
)
from .asb_sfr import ASBSFRObjectiveOutput, asb_sfr_proposal_objective
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
from .local_contrast_consistency import (
    LocalContrastConsistencyOutput,
    candidate_local_contrast_consistency,
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
from .router_regularization import (
    RouterRegularizationOutput,
    router_coefficient_l2,
    router_coefficient_total_variation,
    router_regularization,
)
from .soft_iou_anchor import (
    ForegroundOverlap,
    foreground_soft_dice_anchor,
    foreground_soft_iou_anchor,
    soft_dice_anchor,
    soft_iou_anchor,
)


__all__ = [
    "ActiveBootstrapOutput",
    "ASBSFRObjectiveOutput",
    "BalancedEntropyOutput",
    "ConsistencyDivergence",
    "FeatureStatisticsAlignmentOutput",
    "ForegroundMassGuardOutput",
    "ForegroundOverlap",
    "LocalContrastConsistencyOutput",
    "MultiViewConsistencyOutput",
    "ParameterCollection",
    "RegionConsistencyOutput",
    "RegionLossOutput",
    "RouterRegularizationOutput",
    "SourceAnchoredRegionWeights",
    "SourceFeatureStatistics",
    "StageBObjectiveError",
    "active_bootstrap_binary",
    "asb_sfr_proposal_objective",
    "balanced_binary_entropy",
    "balanced_entropy",
    "bernoulli_entropy_map",
    "bernoulli_probability_entropy",
    "bernoulli_jensen_shannon_map",
    "build_source_anchored_region_weights",
    "candidate_local_contrast_consistency",
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
    "router_coefficient_l2",
    "router_coefficient_total_variation",
    "router_regularization",
    "soft_dice_anchor",
    "soft_iou_anchor",
    "source_anchored_multiview_consistency",
]
