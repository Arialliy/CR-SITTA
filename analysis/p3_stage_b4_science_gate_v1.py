"""Pure, exact Stage-B4 R0 science gate for CR-SITTA P3.

The gate consumes only caller-supplied train-side Pilot64 sufficient evidence:
one exact ``4 candidates x 3 datasets x 13 conditions`` cell grid and one
exact 64-episode grid for every cell.  It performs no filesystem, dataset,
model, CUDA, or random-number operation.

Invalid/incomplete evidence raises :class:`StageB4R0ProtocolError`.  A valid
protocol with zero eligible candidates is instead a normal scientific result
(``scientific_no_eligible``).  Every numeric comparison is performed with
``Fraction`` values parsed through ``Decimal``; missing values and imputation
are forbidden.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
from typing import Any, Literal


DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CORRUPTION_FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "low_contrast",
    "stripe_noise",
)
SEVERITIES = ("S1", "S3", "S5")
CONDITIONS = (
    "clean_S0",
    *(
        f"{family}_{severity}"
        for family in CORRUPTION_FAMILIES
        for severity in SEVERITIES
    ),
)
FROZEN_CANDIDATES = (
    "O3_P2",
    "O4_P2",
    "O3_DecoderFiLM",
    "O4_DecoderFiLM",
)
REPLICATE_IDS = ("R0", "R1", "R2")

EPISODES_PER_CELL = 64
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TOTAL_IMAGE_PIXELS_PER_CELL = EPISODES_PER_CELL * IMAGE_HEIGHT * IMAGE_WIDTH
CELLS_PER_CANDIDATE = len(DATASETS) * len(CONDITIONS)
NONCLEAN_CELLS_PER_CANDIDATE = len(DATASETS) * (len(CONDITIONS) - 1)
EPISODES_PER_CANDIDATE = CELLS_PER_CANDIDATE * EPISODES_PER_CELL
NONCLEAN_EPISODES_PER_CANDIDATE = (
    NONCLEAN_CELLS_PER_CANDIDATE * EPISODES_PER_CELL
)

SAFETY_STRATA = (
    "overall",
    "nonclean",
    "clean",
    *(f"dataset:{dataset}" for dataset in DATASETS),
    *(f"corruption_family:{family}" for family in CORRUPTION_FAMILIES),
)

CELL_FIELDS = frozenset(
    {
        "candidate_id",
        "dataset",
        "condition",
        "corruption_family",
        "severity",
        "episode_count",
        "source_counts",
        "adapted_counts",
    }
)

COUNT_FIELDS = frozenset(
    {
        "intersection_pixels",
        "false_positive_pixels",
        "false_negative_pixels",
        "true_negative_pixels",
        "predicted_positive_pixels",
        "target_positive_pixels",
        "detected_targets",
        "total_targets",
        "false_alarm_pixels",
        "total_image_pixels",
        "image_count",
    }
)

EPISODE_FIELDS = frozenset(
    {
        "candidate_id",
        "dataset",
        "condition",
        "episode_index",
        "proxy_gradient_nonzero",
        "task_gradient_nonzero",
        "gradient_cosine",
        "accepted_update",
        "finite",
        "maximum_absolute_logit_delta",
        "proposal_loss_before",
        "proposal_loss_after",
        "threshold_crossing_count",
    }
)

GATE_CONFIG_FIELDS = frozenset(
    {
        "comparison_tolerance",
        "filter_before_ranking",
        "no_eligible_is_normal_scientific_result",
        "maximum_selected_for_r1_r2",
        "per_replicate_performance_and_safety",
        "alignment",
        "activity",
        "gain_attribution",
        "ranking_tie_break_order",
        "r0_no_eligible_action",
        "r0_eligible_action",
        "across_R0_R1_R2_frozen_for_stage_b5",
    }
)

PERFORMANCE_AND_SAFETY_FIELDS = frozenset(
    {
        "nonclean_macro_delta_iou_strictly_greater_than",
        "overall_macro_delta_iou_strictly_greater_than",
        "minimum_positive_nonclean_datasets",
        "worst_nonclean_dataset_delta_iou_minimum",
        "minimum_positive_corruption_families",
        "worst_corruption_family_delta_iou_minimum",
        "clean_macro_delta_iou_minimum",
        "each_clean_dataset_delta_iou_minimum",
        "nonclean_macro_delta_pd_minimum",
        "each_nonclean_dataset_delta_pd_minimum",
        "clean_macro_delta_pd_minimum",
        "fa_delta_maximum_formula",
        "foreground_fraction",
    }
)
FA_FORMULA_FIELDS = frozenset(
    {
        "absolute_allowance_per_million_pixels",
        "source_fa_multiplier",
        "required_strata",
    }
)
FOREGROUND_FIELDS = frozenset(
    {"delta_maximum", "source_multiplier", "epsilon", "required_strata"}
)
ALIGNMENT_FIELDS = frozenset(
    {
        "macro_scope",
        "macro_gradient_cosine_minimum",
        "dataset_statistic",
        "minimum_positive_dataset_medians",
        "both_gradients_nonzero_fraction_minimum",
    }
)
ACTIVITY_FIELDS = frozenset(
    {
        "scope",
        "accepted_update_fraction_minimum",
        "no_update_fraction_maximum",
        "functional_logit_threshold",
        "accepted_functional_logit_changed_fraction_minimum",
        "accepted_proposal_loss_strict_decrease_fraction_minimum",
        "accepted_finite_fraction_minimum",
        "threshold_crossing_episode_fraction_minimum",
    }
)
GAIN_ATTRIBUTION_FIELDS = frozenset(
    {"accepted_and_no_update_subsets_required", "role"}
)
ACROSS_REPLICATE_FIELDS = frozenset(
    {
        "every_replicate_must_pass_all_hard_gates",
        "nonclean_macro_delta_iou_mean_minimum",
        "overall_macro_delta_iou_mean_minimum",
        "maximum_final_selected_candidates",
    }
)

REQUIRED_STRATUM_KINDS = (
    "overall",
    "nonclean",
    "clean",
    "dataset",
    "corruption_family",
)
RANKING_TIE_BREAK_ORDER = (
    "higher_nonclean_macro_delta_iou",
    "higher_worst_dataset_nonclean_delta_iou",
    "higher_overall_macro_delta_iou",
    "higher_clean_macro_delta_iou",
    "lower_nonclean_fa_delta_per_million",
    "lower_nonclean_foreground_fraction_inflation",
    "higher_nonclean_delta_pd",
    "higher_nonclean_accepted_update_fraction",
    "lexicographically_lower_candidate_id",
)


class StageB4R0ProtocolError(ValueError):
    """The frozen roster, topology, schema, or sufficient evidence is invalid."""


StageB4ScienceGateError = StageB4R0ProtocolError


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageB4R0ProtocolError(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise StageB4R0ProtocolError(f"{field} keys must be strings")
    return value


def _exact_mapping(
    value: Any, expected: frozenset[str], *, field: str
) -> Mapping[str, Any]:
    result = _mapping(value, field=field)
    observed = set(result)
    if observed != expected:
        raise StageB4R0ProtocolError(
            f"{field} fields must be exact; "
            f"missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected, key=str)}"
        )
    return result


def _sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise StageB4R0ProtocolError(f"{field} must be a sequence")
    return tuple(value)


def _fraction(value: Any, *, field: str) -> Fraction:
    if isinstance(value, bool):
        raise StageB4R0ProtocolError(f"{field} must be a finite decimal scalar")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise StageB4R0ProtocolError(f"{field} must be finite")
        return Fraction(value)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StageB4R0ProtocolError(f"{field} must be finite")
        return Fraction(Decimal(str(value)))
    if isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise StageB4R0ProtocolError(
                f"{field} must be a finite decimal scalar"
            ) from exc
        if not parsed.is_finite():
            raise StageB4R0ProtocolError(f"{field} must be finite")
        return Fraction(parsed)
    raise StageB4R0ProtocolError(f"{field} must be a finite decimal scalar")


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StageB4R0ProtocolError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise StageB4R0ProtocolError(f"{field} must be a JSON boolean")
    return value


def _mean(values: Sequence[Fraction], *, field: str) -> Fraction:
    if not values:
        raise StageB4R0ProtocolError(f"{field} cannot be empty")
    return sum(values, Fraction(0, 1)) / len(values)


def _median(values: Sequence[Fraction], *, field: str) -> Fraction:
    if not values:
        raise StageB4R0ProtocolError(f"{field} cannot be empty")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _ratio(numerator: int, denominator: int) -> Fraction:
    if denominator == 0:
        return Fraction(0, 1)
    return Fraction(numerator, denominator)


def _fraction_receipt(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_map_receipt(
    values: tuple[tuple[str, Fraction], ...]
) -> dict[str, dict[str, int]]:
    return {key: _fraction_receipt(value) for key, value in values}


def _count_map_receipt(
    values: tuple[tuple[str, int], ...]
) -> dict[str, int]:
    return dict(values)


_FROZEN_NUMERIC_THRESHOLDS = {
    "comparison_tolerance": Fraction(1, 1_000_000_000_000),
    "nonclean_macro_delta_iou_strict_minimum": Fraction(0),
    "overall_macro_delta_iou_strict_minimum": Fraction(0),
    "worst_nonclean_dataset_delta_iou_minimum": Fraction(-2, 1000),
    "worst_nonclean_family_delta_iou_minimum": Fraction(-5, 1000),
    "clean_macro_delta_iou_minimum": Fraction(-2, 1000),
    "clean_dataset_delta_iou_minimum": Fraction(-5, 1000),
    "nonclean_macro_delta_pd_minimum": Fraction(-10, 1000),
    "nonclean_dataset_delta_pd_minimum": Fraction(-20, 1000),
    "clean_macro_delta_pd_minimum": Fraction(-10, 1000),
    "fa_absolute_allowance_per_million": Fraction(10),
    "fa_source_multiplier": Fraction(25, 100),
    "foreground_fraction_delta_maximum": Fraction(1, 1000),
    "foreground_fraction_source_multiplier": Fraction(12, 10),
    "foreground_fraction_epsilon": Fraction(1, 1_000_000),
    "alignment_macro_cosine_minimum": Fraction(5, 100),
    "both_gradients_nonzero_fraction_minimum": Fraction(80, 100),
    "accepted_update_fraction_nonclean_minimum": Fraction(20, 100),
    "no_update_fraction_nonclean_maximum": Fraction(80, 100),
    "functional_logit_delta_threshold": Fraction(1, 1_000_000),
    "accepted_functional_fraction_minimum": Fraction(90, 100),
    "accepted_proposal_loss_decrease_fraction_minimum": Fraction(90, 100),
    "accepted_finite_fraction_minimum": Fraction(1),
    "threshold_crossing_episode_fraction_nonclean_minimum": Fraction(2, 100),
    "across_replicate_nonclean_mean_delta_iou_minimum": Fraction(2, 1000),
    "across_replicate_overall_mean_delta_iou_minimum": Fraction(1, 1000),
}


def default_gate_config() -> dict[str, Any]:
    """Return the exact nested YAML-compatible Stage-B4 R0 gate mapping."""

    return {
        "comparison_tolerance": "0.000000000001",
        "filter_before_ranking": True,
        "no_eligible_is_normal_scientific_result": True,
        "maximum_selected_for_r1_r2": 4,
        "per_replicate_performance_and_safety": {
            "nonclean_macro_delta_iou_strictly_greater_than": "0",
            "overall_macro_delta_iou_strictly_greater_than": "0",
            "minimum_positive_nonclean_datasets": 2,
            "worst_nonclean_dataset_delta_iou_minimum": "-0.002",
            "minimum_positive_corruption_families": 3,
            "worst_corruption_family_delta_iou_minimum": "-0.005",
            "clean_macro_delta_iou_minimum": "-0.002",
            "each_clean_dataset_delta_iou_minimum": "-0.005",
            "nonclean_macro_delta_pd_minimum": "-0.010",
            "each_nonclean_dataset_delta_pd_minimum": "-0.020",
            "clean_macro_delta_pd_minimum": "-0.010",
            "fa_delta_maximum_formula": {
                "absolute_allowance_per_million_pixels": "10",
                "source_fa_multiplier": "0.25",
                "required_strata": list(REQUIRED_STRATUM_KINDS),
            },
            "foreground_fraction": {
                "delta_maximum": "0.001",
                "source_multiplier": "1.2",
                "epsilon": "0.000001",
                "required_strata": list(REQUIRED_STRATUM_KINDS),
            },
        },
        "alignment": {
            "macro_scope": "overall_2496_episodes",
            "macro_gradient_cosine_minimum": "0.05",
            "dataset_statistic": "median_over_valid_13x64_episode_cosines",
            "minimum_positive_dataset_medians": 2,
            "both_gradients_nonzero_fraction_minimum": "0.80",
        },
        "activity": {
            "scope": "nonclean_2304_episodes",
            "accepted_update_fraction_minimum": "0.20",
            "no_update_fraction_maximum": "0.80",
            "functional_logit_threshold": "0.000001",
            "accepted_functional_logit_changed_fraction_minimum": "0.90",
            "accepted_proposal_loss_strict_decrease_fraction_minimum": "0.90",
            "accepted_finite_fraction_minimum": "1.0",
            "threshold_crossing_episode_fraction_minimum": "0.02",
        },
        "gain_attribution": {
            "accepted_and_no_update_subsets_required": True,
            "role": "report_only_not_backtracking_or_candidate_execution",
        },
        "ranking_tie_break_order": list(RANKING_TIE_BREAK_ORDER),
        "r0_no_eligible_action": "stop_without_R1_R2",
        "r0_eligible_action": "run_only_R0_eligible_candidates_in_R1_R2",
        "across_R0_R1_R2_frozen_for_stage_b5": {
            "every_replicate_must_pass_all_hard_gates": True,
            "nonclean_macro_delta_iou_mean_minimum": "0.002",
            "overall_macro_delta_iou_mean_minimum": "0.001",
            "maximum_final_selected_candidates": 3,
        },
    }


@dataclass(frozen=True, slots=True)
class GateConfig:
    comparison_tolerance: Fraction
    maximum_selected_for_r1_r2: int
    nonclean_macro_delta_iou_strict_minimum: Fraction
    overall_macro_delta_iou_strict_minimum: Fraction
    minimum_positive_nonclean_datasets: int
    worst_nonclean_dataset_delta_iou_minimum: Fraction
    minimum_positive_nonclean_families: int
    worst_nonclean_family_delta_iou_minimum: Fraction
    clean_macro_delta_iou_minimum: Fraction
    clean_dataset_delta_iou_minimum: Fraction
    nonclean_macro_delta_pd_minimum: Fraction
    nonclean_dataset_delta_pd_minimum: Fraction
    clean_macro_delta_pd_minimum: Fraction
    fa_absolute_allowance_per_million: Fraction
    fa_source_multiplier: Fraction
    foreground_fraction_delta_maximum: Fraction
    foreground_fraction_source_multiplier: Fraction
    foreground_fraction_epsilon: Fraction
    alignment_macro_cosine_minimum: Fraction
    minimum_positive_alignment_dataset_medians: int
    both_gradients_nonzero_fraction_minimum: Fraction
    accepted_update_fraction_nonclean_minimum: Fraction
    no_update_fraction_nonclean_maximum: Fraction
    functional_logit_delta_threshold: Fraction
    accepted_functional_fraction_minimum: Fraction
    accepted_proposal_loss_decrease_fraction_minimum: Fraction
    accepted_finite_fraction_minimum: Fraction
    threshold_crossing_episode_fraction_nonclean_minimum: Fraction
    filter_before_ranking: bool
    no_eligible_is_normal_scientific_result: bool
    across_replicate_nonclean_mean_delta_iou_minimum: Fraction
    across_replicate_overall_mean_delta_iou_minimum: Fraction
    maximum_final_top_candidates: int
    fa_required_strata: tuple[str, ...]
    foreground_required_strata: tuple[str, ...]
    alignment_macro_scope: str
    alignment_dataset_statistic: str
    activity_scope: str
    gain_attribution_subsets_required: bool
    gain_attribution_role: str
    ranking_tie_break_order: tuple[str, ...]
    r0_no_eligible_action: str
    r0_eligible_action: str
    every_replicate_must_pass_all_hard_gates: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GateConfig":
        raw = _exact_mapping(value, GATE_CONFIG_FIELDS, field="gate_config")
        performance = _exact_mapping(
            raw["per_replicate_performance_and_safety"],
            PERFORMANCE_AND_SAFETY_FIELDS,
            field="gate_config.per_replicate_performance_and_safety",
        )
        fa = _exact_mapping(
            performance["fa_delta_maximum_formula"],
            FA_FORMULA_FIELDS,
            field=(
                "gate_config.per_replicate_performance_and_safety."
                "fa_delta_maximum_formula"
            ),
        )
        foreground = _exact_mapping(
            performance["foreground_fraction"],
            FOREGROUND_FIELDS,
            field=(
                "gate_config.per_replicate_performance_and_safety."
                "foreground_fraction"
            ),
        )
        alignment = _exact_mapping(
            raw["alignment"], ALIGNMENT_FIELDS, field="gate_config.alignment"
        )
        activity = _exact_mapping(
            raw["activity"], ACTIVITY_FIELDS, field="gate_config.activity"
        )
        gain = _exact_mapping(
            raw["gain_attribution"],
            GAIN_ATTRIBUTION_FIELDS,
            field="gate_config.gain_attribution",
        )
        across = _exact_mapping(
            raw["across_R0_R1_R2_frozen_for_stage_b5"],
            ACROSS_REPLICATE_FIELDS,
            field="gate_config.across_R0_R1_R2_frozen_for_stage_b5",
        )
        fa_strata = _sequence(
            fa["required_strata"],
            field=(
                "gate_config.per_replicate_performance_and_safety."
                "fa_delta_maximum_formula.required_strata"
            ),
        )
        foreground_strata = _sequence(
            foreground["required_strata"],
            field=(
                "gate_config.per_replicate_performance_and_safety."
                "foreground_fraction.required_strata"
            ),
        )
        ranking_order = _sequence(
            raw["ranking_tie_break_order"],
            field="gate_config.ranking_tie_break_order",
        )
        result = cls(
            comparison_tolerance=_fraction(
                raw["comparison_tolerance"],
                field="gate_config.comparison_tolerance",
            ),
            maximum_selected_for_r1_r2=_integer(
                raw["maximum_selected_for_r1_r2"],
                field="gate_config.maximum_selected_for_r1_r2",
                minimum=1,
            ),
            minimum_positive_nonclean_datasets=_integer(
                performance["minimum_positive_nonclean_datasets"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "minimum_positive_nonclean_datasets"
                ),
                minimum=1,
            ),
            minimum_positive_nonclean_families=_integer(
                performance["minimum_positive_corruption_families"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "minimum_positive_corruption_families"
                ),
                minimum=1,
            ),
            minimum_positive_alignment_dataset_medians=_integer(
                alignment["minimum_positive_dataset_medians"],
                field="gate_config.alignment.minimum_positive_dataset_medians",
                minimum=1,
            ),
            filter_before_ranking=_boolean(
                raw["filter_before_ranking"],
                field="gate_config.filter_before_ranking",
            ),
            no_eligible_is_normal_scientific_result=_boolean(
                raw["no_eligible_is_normal_scientific_result"],
                field="gate_config.no_eligible_is_normal_scientific_result",
            ),
            maximum_final_top_candidates=_integer(
                across["maximum_final_selected_candidates"],
                field=(
                    "gate_config.across_R0_R1_R2_frozen_for_stage_b5."
                    "maximum_final_selected_candidates"
                ),
                minimum=1,
            ),
            nonclean_macro_delta_iou_strict_minimum=_fraction(
                performance["nonclean_macro_delta_iou_strictly_greater_than"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "nonclean_macro_delta_iou_strictly_greater_than"
                ),
            ),
            overall_macro_delta_iou_strict_minimum=_fraction(
                performance["overall_macro_delta_iou_strictly_greater_than"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "overall_macro_delta_iou_strictly_greater_than"
                ),
            ),
            worst_nonclean_dataset_delta_iou_minimum=_fraction(
                performance["worst_nonclean_dataset_delta_iou_minimum"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "worst_nonclean_dataset_delta_iou_minimum"
                ),
            ),
            worst_nonclean_family_delta_iou_minimum=_fraction(
                performance["worst_corruption_family_delta_iou_minimum"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "worst_corruption_family_delta_iou_minimum"
                ),
            ),
            clean_macro_delta_iou_minimum=_fraction(
                performance["clean_macro_delta_iou_minimum"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "clean_macro_delta_iou_minimum"
                ),
            ),
            clean_dataset_delta_iou_minimum=_fraction(
                performance["each_clean_dataset_delta_iou_minimum"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "each_clean_dataset_delta_iou_minimum"
                ),
            ),
            nonclean_macro_delta_pd_minimum=_fraction(
                performance["nonclean_macro_delta_pd_minimum"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "nonclean_macro_delta_pd_minimum"
                ),
            ),
            nonclean_dataset_delta_pd_minimum=_fraction(
                performance["each_nonclean_dataset_delta_pd_minimum"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "each_nonclean_dataset_delta_pd_minimum"
                ),
            ),
            clean_macro_delta_pd_minimum=_fraction(
                performance["clean_macro_delta_pd_minimum"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "clean_macro_delta_pd_minimum"
                ),
            ),
            fa_absolute_allowance_per_million=_fraction(
                fa["absolute_allowance_per_million_pixels"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "fa_delta_maximum_formula.absolute_allowance_per_million_pixels"
                ),
            ),
            fa_source_multiplier=_fraction(
                fa["source_fa_multiplier"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "fa_delta_maximum_formula.source_fa_multiplier"
                ),
            ),
            foreground_fraction_delta_maximum=_fraction(
                foreground["delta_maximum"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "foreground_fraction.delta_maximum"
                ),
            ),
            foreground_fraction_source_multiplier=_fraction(
                foreground["source_multiplier"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "foreground_fraction.source_multiplier"
                ),
            ),
            foreground_fraction_epsilon=_fraction(
                foreground["epsilon"],
                field=(
                    "gate_config.per_replicate_performance_and_safety."
                    "foreground_fraction.epsilon"
                ),
            ),
            alignment_macro_cosine_minimum=_fraction(
                alignment["macro_gradient_cosine_minimum"],
                field="gate_config.alignment.macro_gradient_cosine_minimum",
            ),
            both_gradients_nonzero_fraction_minimum=_fraction(
                alignment["both_gradients_nonzero_fraction_minimum"],
                field=(
                    "gate_config.alignment."
                    "both_gradients_nonzero_fraction_minimum"
                ),
            ),
            accepted_update_fraction_nonclean_minimum=_fraction(
                activity["accepted_update_fraction_minimum"],
                field="gate_config.activity.accepted_update_fraction_minimum",
            ),
            no_update_fraction_nonclean_maximum=_fraction(
                activity["no_update_fraction_maximum"],
                field="gate_config.activity.no_update_fraction_maximum",
            ),
            functional_logit_delta_threshold=_fraction(
                activity["functional_logit_threshold"],
                field="gate_config.activity.functional_logit_threshold",
            ),
            accepted_functional_fraction_minimum=_fraction(
                activity["accepted_functional_logit_changed_fraction_minimum"],
                field=(
                    "gate_config.activity."
                    "accepted_functional_logit_changed_fraction_minimum"
                ),
            ),
            accepted_proposal_loss_decrease_fraction_minimum=_fraction(
                activity[
                    "accepted_proposal_loss_strict_decrease_fraction_minimum"
                ],
                field=(
                    "gate_config.activity."
                    "accepted_proposal_loss_strict_decrease_fraction_minimum"
                ),
            ),
            accepted_finite_fraction_minimum=_fraction(
                activity["accepted_finite_fraction_minimum"],
                field="gate_config.activity.accepted_finite_fraction_minimum",
            ),
            threshold_crossing_episode_fraction_nonclean_minimum=_fraction(
                activity["threshold_crossing_episode_fraction_minimum"],
                field=(
                    "gate_config.activity."
                    "threshold_crossing_episode_fraction_minimum"
                ),
            ),
            across_replicate_nonclean_mean_delta_iou_minimum=_fraction(
                across["nonclean_macro_delta_iou_mean_minimum"],
                field=(
                    "gate_config.across_R0_R1_R2_frozen_for_stage_b5."
                    "nonclean_macro_delta_iou_mean_minimum"
                ),
            ),
            across_replicate_overall_mean_delta_iou_minimum=_fraction(
                across["overall_macro_delta_iou_mean_minimum"],
                field=(
                    "gate_config.across_R0_R1_R2_frozen_for_stage_b5."
                    "overall_macro_delta_iou_mean_minimum"
                ),
            ),
            fa_required_strata=tuple(fa_strata),
            foreground_required_strata=tuple(foreground_strata),
            alignment_macro_scope=alignment["macro_scope"],
            alignment_dataset_statistic=alignment["dataset_statistic"],
            activity_scope=activity["scope"],
            gain_attribution_subsets_required=_boolean(
                gain["accepted_and_no_update_subsets_required"],
                field=(
                    "gate_config.gain_attribution."
                    "accepted_and_no_update_subsets_required"
                ),
            ),
            gain_attribution_role=gain["role"],
            ranking_tie_break_order=tuple(ranking_order),
            r0_no_eligible_action=raw["r0_no_eligible_action"],
            r0_eligible_action=raw["r0_eligible_action"],
            every_replicate_must_pass_all_hard_gates=_boolean(
                across["every_replicate_must_pass_all_hard_gates"],
                field=(
                    "gate_config.across_R0_R1_R2_frozen_for_stage_b5."
                    "every_replicate_must_pass_all_hard_gates"
                ),
            ),
        )
        result._validate_frozen_contract()
        return result

    def _validate_frozen_contract(self) -> None:
        for name, expected in _FROZEN_NUMERIC_THRESHOLDS.items():
            if getattr(self, name) != expected:
                raise StageB4R0ProtocolError(
                    f"gate_config.{name} must equal frozen value "
                    f"{expected.numerator}/{expected.denominator}"
                )
        exact_integers = {
            "maximum_selected_for_r1_r2": 4,
            "minimum_positive_nonclean_datasets": 2,
            "minimum_positive_nonclean_families": 3,
            "minimum_positive_alignment_dataset_medians": 2,
            "maximum_final_top_candidates": 3,
        }
        for name, expected in exact_integers.items():
            if getattr(self, name) != expected:
                raise StageB4R0ProtocolError(
                    f"gate_config.{name} must equal frozen value {expected}"
                )
        if self.filter_before_ranking is not True:
            raise StageB4R0ProtocolError(
                "gate_config.filter_before_ranking must be true"
            )
        if self.no_eligible_is_normal_scientific_result is not True:
            raise StageB4R0ProtocolError(
                "gate_config.no_eligible_is_normal_scientific_result must be true"
            )
        if self.fa_required_strata != REQUIRED_STRATUM_KINDS:
            raise StageB4R0ProtocolError("gate_config fa required_strata is not frozen")
        if self.foreground_required_strata != REQUIRED_STRATUM_KINDS:
            raise StageB4R0ProtocolError(
                "gate_config foreground required_strata is not frozen"
            )
        exact_strings = {
            "alignment_macro_scope": "overall_2496_episodes",
            "alignment_dataset_statistic": (
                "median_over_valid_13x64_episode_cosines"
            ),
            "activity_scope": "nonclean_2304_episodes",
            "gain_attribution_role": (
                "report_only_not_backtracking_or_candidate_execution"
            ),
            "r0_no_eligible_action": "stop_without_R1_R2",
            "r0_eligible_action": (
                "run_only_R0_eligible_candidates_in_R1_R2"
            ),
        }
        for name, expected in exact_strings.items():
            if getattr(self, name) != expected:
                raise StageB4R0ProtocolError(
                    f"gate_config {name} must equal frozen value {expected!r}"
                )
        if self.gain_attribution_subsets_required is not True:
            raise StageB4R0ProtocolError(
                "gate_config gain attribution subsets must be required"
            )
        if self.ranking_tie_break_order != RANKING_TIE_BREAK_ORDER:
            raise StageB4R0ProtocolError(
                "gate_config.ranking_tie_break_order is not frozen"
            )
        if self.every_replicate_must_pass_all_hard_gates is not True:
            raise StageB4R0ProtocolError(
                "gate_config every replicate hard-gate requirement must be true"
            )

    def to_receipt(self) -> dict[str, Any]:
        number = lambda name: _fraction_receipt(getattr(self, name))
        return {
            "comparison_tolerance": number("comparison_tolerance"),
            "filter_before_ranking": self.filter_before_ranking,
            "no_eligible_is_normal_scientific_result": (
                self.no_eligible_is_normal_scientific_result
            ),
            "maximum_selected_for_r1_r2": self.maximum_selected_for_r1_r2,
            "per_replicate_performance_and_safety": {
                "nonclean_macro_delta_iou_strictly_greater_than": number(
                    "nonclean_macro_delta_iou_strict_minimum"
                ),
                "overall_macro_delta_iou_strictly_greater_than": number(
                    "overall_macro_delta_iou_strict_minimum"
                ),
                "minimum_positive_nonclean_datasets": (
                    self.minimum_positive_nonclean_datasets
                ),
                "worst_nonclean_dataset_delta_iou_minimum": number(
                    "worst_nonclean_dataset_delta_iou_minimum"
                ),
                "minimum_positive_corruption_families": (
                    self.minimum_positive_nonclean_families
                ),
                "worst_corruption_family_delta_iou_minimum": number(
                    "worst_nonclean_family_delta_iou_minimum"
                ),
                "clean_macro_delta_iou_minimum": number(
                    "clean_macro_delta_iou_minimum"
                ),
                "each_clean_dataset_delta_iou_minimum": number(
                    "clean_dataset_delta_iou_minimum"
                ),
                "nonclean_macro_delta_pd_minimum": number(
                    "nonclean_macro_delta_pd_minimum"
                ),
                "each_nonclean_dataset_delta_pd_minimum": number(
                    "nonclean_dataset_delta_pd_minimum"
                ),
                "clean_macro_delta_pd_minimum": number(
                    "clean_macro_delta_pd_minimum"
                ),
                "fa_delta_maximum_formula": {
                    "absolute_allowance_per_million_pixels": number(
                        "fa_absolute_allowance_per_million"
                    ),
                    "source_fa_multiplier": number("fa_source_multiplier"),
                    "required_strata": list(self.fa_required_strata),
                },
                "foreground_fraction": {
                    "delta_maximum": number(
                        "foreground_fraction_delta_maximum"
                    ),
                    "source_multiplier": number(
                        "foreground_fraction_source_multiplier"
                    ),
                    "epsilon": number("foreground_fraction_epsilon"),
                    "required_strata": list(self.foreground_required_strata),
                },
            },
            "alignment": {
                "macro_scope": self.alignment_macro_scope,
                "macro_gradient_cosine_minimum": number(
                    "alignment_macro_cosine_minimum"
                ),
                "dataset_statistic": self.alignment_dataset_statistic,
                "minimum_positive_dataset_medians": (
                    self.minimum_positive_alignment_dataset_medians
                ),
                "both_gradients_nonzero_fraction_minimum": number(
                    "both_gradients_nonzero_fraction_minimum"
                ),
            },
            "activity": {
                "scope": self.activity_scope,
                "accepted_update_fraction_minimum": number(
                    "accepted_update_fraction_nonclean_minimum"
                ),
                "no_update_fraction_maximum": number(
                    "no_update_fraction_nonclean_maximum"
                ),
                "functional_logit_threshold": number(
                    "functional_logit_delta_threshold"
                ),
                "accepted_functional_logit_changed_fraction_minimum": number(
                    "accepted_functional_fraction_minimum"
                ),
                "accepted_proposal_loss_strict_decrease_fraction_minimum": number(
                    "accepted_proposal_loss_decrease_fraction_minimum"
                ),
                "accepted_finite_fraction_minimum": number(
                    "accepted_finite_fraction_minimum"
                ),
                "threshold_crossing_episode_fraction_minimum": number(
                    "threshold_crossing_episode_fraction_nonclean_minimum"
                ),
            },
            "gain_attribution": {
                "accepted_and_no_update_subsets_required": (
                    self.gain_attribution_subsets_required
                ),
                "role": self.gain_attribution_role,
            },
            "ranking_tie_break_order": list(self.ranking_tie_break_order),
            "r0_no_eligible_action": self.r0_no_eligible_action,
            "r0_eligible_action": self.r0_eligible_action,
            "across_R0_R1_R2_frozen_for_stage_b5": {
                "every_replicate_must_pass_all_hard_gates": (
                    self.every_replicate_must_pass_all_hard_gates
                ),
                "nonclean_macro_delta_iou_mean_minimum": number(
                    "across_replicate_nonclean_mean_delta_iou_minimum"
                ),
                "overall_macro_delta_iou_mean_minimum": number(
                    "across_replicate_overall_mean_delta_iou_minimum"
                ),
                "maximum_final_selected_candidates": (
                    self.maximum_final_top_candidates
                ),
            },
        }


def _condition_parts(condition: str) -> tuple[str, str]:
    if condition == "clean_S0":
        return "clean", "S0"
    for family in CORRUPTION_FAMILIES:
        prefix = f"{family}_"
        if condition.startswith(prefix):
            return family, condition[len(prefix) :]
    raise StageB4R0ProtocolError(f"unknown condition {condition!r}")


@dataclass(frozen=True, slots=True)
class _Counts:
    intersection_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    true_negative_pixels: int
    predicted_positive_pixels: int
    target_positive_pixels: int
    detected_targets: int
    total_targets: int
    false_alarm_pixels: int
    total_image_pixels: int
    image_count: int

    @property
    def iou(self) -> Fraction:
        denominator = (
            self.intersection_pixels
            + self.false_positive_pixels
            + self.false_negative_pixels
        )
        # Match metrics.irstd_metrics' frozen empty-union convention exactly:
        # an empty prediction against an empty target has IoU 1, not 0.
        if denominator == 0:
            return Fraction(1, 1)
        return Fraction(self.intersection_pixels, denominator)

    @property
    def pd(self) -> Fraction:
        return _ratio(self.detected_targets, self.total_targets)

    @property
    def fa_per_million(self) -> Fraction:
        return Fraction(
            self.false_alarm_pixels * 1_000_000,
            self.total_image_pixels,
        )

    @property
    def foreground_fraction(self) -> Fraction:
        return Fraction(self.predicted_positive_pixels, self.total_image_pixels)


def _parse_counts(value: Any, *, field: str) -> _Counts:
    raw = _exact_mapping(value, COUNT_FIELDS, field=field)
    values = {
        name: _integer(raw[name], field=f"{field}.{name}")
        for name in COUNT_FIELDS
    }
    counts = _Counts(**values)
    if counts.image_count != EPISODES_PER_CELL:
        raise StageB4R0ProtocolError(f"{field}.image_count must equal 64")
    if counts.total_image_pixels != TOTAL_IMAGE_PIXELS_PER_CELL:
        raise StageB4R0ProtocolError(
            f"{field}.total_image_pixels must equal "
            f"64*256*256={TOTAL_IMAGE_PIXELS_PER_CELL}"
        )
    if (
        counts.intersection_pixels
        + counts.false_positive_pixels
        + counts.false_negative_pixels
        + counts.true_negative_pixels
        != counts.total_image_pixels
    ):
        raise StageB4R0ProtocolError(
            f"{field} pixel confusion counts do not conserve total_image_pixels"
        )
    if counts.predicted_positive_pixels != (
        counts.intersection_pixels + counts.false_positive_pixels
    ):
        raise StageB4R0ProtocolError(
            f"{field}.predicted_positive_pixels is inconsistent"
        )
    if counts.target_positive_pixels != (
        counts.intersection_pixels + counts.false_negative_pixels
    ):
        raise StageB4R0ProtocolError(
            f"{field}.target_positive_pixels is inconsistent"
        )
    if counts.detected_targets > counts.total_targets:
        raise StageB4R0ProtocolError(
            f"{field}.detected_targets exceeds total_targets"
        )
    if counts.false_alarm_pixels > counts.predicted_positive_pixels:
        raise StageB4R0ProtocolError(
            f"{field}.false_alarm_pixels exceeds predicted_positive_pixels"
        )
    return counts


@dataclass(frozen=True, slots=True)
class _Cell:
    candidate_id: str
    dataset: str
    condition: str
    corruption_family: str
    severity: str
    source_counts: _Counts
    adapted_counts: _Counts

    @property
    def source_iou(self) -> Fraction:
        return self.source_counts.iou

    @property
    def adapted_iou(self) -> Fraction:
        return self.adapted_counts.iou

    @property
    def source_pd(self) -> Fraction:
        return self.source_counts.pd

    @property
    def adapted_pd(self) -> Fraction:
        return self.adapted_counts.pd

    @property
    def source_fa_per_million(self) -> Fraction:
        return self.source_counts.fa_per_million

    @property
    def adapted_fa_per_million(self) -> Fraction:
        return self.adapted_counts.fa_per_million

    @property
    def source_foreground_fraction(self) -> Fraction:
        return self.source_counts.foreground_fraction

    @property
    def adapted_foreground_fraction(self) -> Fraction:
        return self.adapted_counts.foreground_fraction

    @property
    def delta_iou(self) -> Fraction:
        return self.adapted_iou - self.source_iou

    @property
    def delta_pd(self) -> Fraction:
        return self.adapted_pd - self.source_pd

    @property
    def delta_fa_per_million(self) -> Fraction:
        return self.adapted_fa_per_million - self.source_fa_per_million


@dataclass(frozen=True, slots=True)
class _Episode:
    candidate_id: str
    dataset: str
    condition: str
    episode_index: int
    proxy_gradient_nonzero: bool
    task_gradient_nonzero: bool
    gradient_cosine: Fraction
    accepted_update: bool
    finite: bool
    maximum_absolute_logit_delta: Fraction
    proposal_loss_before: Fraction
    proposal_loss_after: Fraction
    threshold_crossing_count: int

    @property
    def both_gradients_nonzero(self) -> bool:
        return self.proxy_gradient_nonzero and self.task_gradient_nonzero


def _parse_identity(
    raw: Mapping[str, Any], *, field: str, require_parts: bool
) -> tuple[str, str, str]:
    candidate = raw["candidate_id"]
    dataset = raw["dataset"]
    condition = raw["condition"]
    if candidate not in FROZEN_CANDIDATES:
        raise StageB4R0ProtocolError(
            f"{field}.candidate_id is outside the frozen four-candidate roster"
        )
    if dataset not in DATASETS:
        raise StageB4R0ProtocolError(
            f"{field}.dataset is outside the frozen three-dataset roster"
        )
    if condition not in CONDITIONS:
        raise StageB4R0ProtocolError(
            f"{field}.condition is outside the frozen thirteen-condition roster"
        )
    if require_parts:
        expected_family, expected_severity = _condition_parts(condition)
        if (
            raw["corruption_family"] != expected_family
            or raw["severity"] != expected_severity
        ):
            raise StageB4R0ProtocolError(
                f"{field} corruption_family/severity disagree with condition"
            )
    return candidate, dataset, condition


def _parse_cell(value: Any, *, index: int) -> _Cell:
    field = f"cell_summaries[{index}]"
    raw = _exact_mapping(value, CELL_FIELDS, field=field)
    candidate, dataset, condition = _parse_identity(
        raw, field=field, require_parts=True
    )
    if _integer(raw["episode_count"], field=f"{field}.episode_count") != 64:
        raise StageB4R0ProtocolError(f"{field}.episode_count must equal 64")
    source_counts = _parse_counts(raw["source_counts"], field=f"{field}.source_counts")
    adapted_counts = _parse_counts(
        raw["adapted_counts"], field=f"{field}.adapted_counts"
    )
    if (
        source_counts.total_image_pixels != adapted_counts.total_image_pixels
        or source_counts.image_count != adapted_counts.image_count
        or source_counts.target_positive_pixels
        != adapted_counts.target_positive_pixels
        or source_counts.total_targets != adapted_counts.total_targets
    ):
        raise StageB4R0ProtocolError(
            f"{field} source/adapted target denominators must be identical"
        )
    family, severity = _condition_parts(condition)
    return _Cell(
        candidate_id=candidate,
        dataset=dataset,
        condition=condition,
        corruption_family=family,
        severity=severity,
        source_counts=source_counts,
        adapted_counts=adapted_counts,
    )


def _parse_episode(value: Any, *, index: int) -> _Episode:
    field = f"episode_summaries[{index}]"
    raw = _exact_mapping(value, EPISODE_FIELDS, field=field)
    candidate, dataset, condition = _parse_identity(
        raw, field=field, require_parts=False
    )
    episode_index = _integer(
        raw["episode_index"], field=f"{field}.episode_index"
    )
    if episode_index >= EPISODES_PER_CELL:
        raise StageB4R0ProtocolError(
            f"{field}.episode_index must be in [0, 63]"
        )
    proxy_nonzero = _boolean(
        raw["proxy_gradient_nonzero"],
        field=f"{field}.proxy_gradient_nonzero",
    )
    task_nonzero = _boolean(
        raw["task_gradient_nonzero"],
        field=f"{field}.task_gradient_nonzero",
    )
    cosine = _fraction(raw["gradient_cosine"], field=f"{field}.gradient_cosine")
    if not -1 <= cosine <= 1:
        raise StageB4R0ProtocolError(
            f"{field}.gradient_cosine must be in [-1, 1]"
        )
    if not (proxy_nonzero and task_nonzero) and cosine != 0:
        raise StageB4R0ProtocolError(
            f"{field}.gradient_cosine must be exact zero when either gradient is zero"
        )
    accepted = _boolean(raw["accepted_update"], field=f"{field}.accepted_update")
    finite = _boolean(raw["finite"], field=f"{field}.finite")
    max_logit = _fraction(
        raw["maximum_absolute_logit_delta"],
        field=f"{field}.maximum_absolute_logit_delta",
    )
    if max_logit < 0:
        raise StageB4R0ProtocolError(
            f"{field}.maximum_absolute_logit_delta must be non-negative"
        )
    loss_before = _fraction(
        raw["proposal_loss_before"], field=f"{field}.proposal_loss_before"
    )
    loss_after = _fraction(
        raw["proposal_loss_after"], field=f"{field}.proposal_loss_after"
    )
    crossings = _integer(
        raw["threshold_crossing_count"],
        field=f"{field}.threshold_crossing_count",
    )
    if not accepted and (max_logit != 0 or crossings != 0 or loss_after != loss_before):
        raise StageB4R0ProtocolError(
            f"{field} no-update outcome must have zero logit/crossing change "
            "and unchanged proposal loss"
        )
    return _Episode(
        candidate_id=candidate,
        dataset=dataset,
        condition=condition,
        episode_index=episode_index,
        proxy_gradient_nonzero=proxy_nonzero,
        task_gradient_nonzero=task_nonzero,
        gradient_cosine=cosine,
        accepted_update=accepted,
        finite=finite,
        maximum_absolute_logit_delta=max_logit,
        proposal_loss_before=loss_before,
        proposal_loss_after=loss_after,
        threshold_crossing_count=crossings,
    )


def _is_nonclean_cell(cell: _Cell) -> bool:
    return cell.condition != "clean_S0"


def _is_nonclean_episode(episode: _Episode) -> bool:
    return episode.condition != "clean_S0"


def _stratum_cells(cells: Sequence[_Cell], stratum: str) -> tuple[_Cell, ...]:
    if stratum == "overall":
        return tuple(cells)
    if stratum == "nonclean":
        return tuple(cell for cell in cells if _is_nonclean_cell(cell))
    if stratum == "clean":
        return tuple(cell for cell in cells if not _is_nonclean_cell(cell))
    if stratum.startswith("dataset:"):
        dataset = stratum.partition(":")[2]
        return tuple(cell for cell in cells if cell.dataset == dataset)
    if stratum.startswith("corruption_family:"):
        family = stratum.partition(":")[2]
        return tuple(cell for cell in cells if cell.corruption_family == family)
    raise StageB4R0ProtocolError(f"unknown safety stratum {stratum}")


def _metric_mean(
    cells: Sequence[_Cell], attribute: str, *, field: str
) -> Fraction:
    return _mean([getattr(cell, attribute) for cell in cells], field=field)


def _delta_mean(cells: Sequence[_Cell], attribute: str, *, field: str) -> Fraction:
    return _mean([getattr(cell, attribute) for cell in cells], field=field)


@dataclass(frozen=True, slots=True)
class CandidateAggregate:
    candidate_id: str
    cell_count: int
    nonclean_cell_count: int
    episode_count: int
    nonclean_episode_count: int
    overall_macro_delta_iou: Fraction
    nonclean_macro_delta_iou: Fraction
    dataset_nonclean_delta_iou: tuple[tuple[str, Fraction], ...]
    family_nonclean_delta_iou: tuple[tuple[str, Fraction], ...]
    clean_macro_delta_iou: Fraction
    clean_dataset_delta_iou: tuple[tuple[str, Fraction], ...]
    overall_macro_delta_pd: Fraction
    nonclean_macro_delta_pd: Fraction
    dataset_nonclean_delta_pd: tuple[tuple[str, Fraction], ...]
    clean_macro_delta_pd: Fraction
    source_fa_per_million: tuple[tuple[str, Fraction], ...]
    adapted_fa_per_million: tuple[tuple[str, Fraction], ...]
    fa_delta_per_million: tuple[tuple[str, Fraction], ...]
    source_foreground_fraction: tuple[tuple[str, Fraction], ...]
    adapted_foreground_fraction: tuple[tuple[str, Fraction], ...]
    foreground_fraction_delta: tuple[tuple[str, Fraction], ...]
    alignment_macro_cosine: Fraction
    alignment_valid_episode_count: int
    alignment_dataset_median_cosine: tuple[tuple[str, Fraction], ...]
    alignment_dataset_valid_episode_count: tuple[tuple[str, int], ...]
    both_gradients_nonzero_episode_count: int
    both_gradients_nonzero_fraction: Fraction
    nonclean_accepted_update_episode_count: int
    nonclean_no_update_episode_count: int
    nonclean_accepted_update_fraction: Fraction
    nonclean_no_update_fraction: Fraction
    nonclean_accepted_functional_episode_count: int
    nonclean_accepted_proposal_loss_decrease_episode_count: int
    nonclean_accepted_finite_episode_count: int
    nonclean_accepted_functional_fraction: Fraction
    nonclean_accepted_proposal_loss_decrease_fraction: Fraction
    nonclean_accepted_finite_fraction: Fraction
    nonclean_threshold_crossing_episode_count: int
    nonclean_threshold_crossing_episode_fraction: Fraction

    def to_receipt(self) -> dict[str, Any]:
        fraction_scalars = (
            "overall_macro_delta_iou",
            "nonclean_macro_delta_iou",
            "clean_macro_delta_iou",
            "overall_macro_delta_pd",
            "nonclean_macro_delta_pd",
            "clean_macro_delta_pd",
            "alignment_macro_cosine",
            "both_gradients_nonzero_fraction",
            "nonclean_accepted_update_fraction",
            "nonclean_no_update_fraction",
            "nonclean_accepted_functional_fraction",
            "nonclean_accepted_proposal_loss_decrease_fraction",
            "nonclean_accepted_finite_fraction",
            "nonclean_threshold_crossing_episode_fraction",
        )
        fraction_maps = (
            "dataset_nonclean_delta_iou",
            "family_nonclean_delta_iou",
            "clean_dataset_delta_iou",
            "dataset_nonclean_delta_pd",
            "source_fa_per_million",
            "adapted_fa_per_million",
            "fa_delta_per_million",
            "source_foreground_fraction",
            "adapted_foreground_fraction",
            "foreground_fraction_delta",
            "alignment_dataset_median_cosine",
        )
        count_scalars = (
            "cell_count",
            "nonclean_cell_count",
            "episode_count",
            "nonclean_episode_count",
            "alignment_valid_episode_count",
            "both_gradients_nonzero_episode_count",
            "nonclean_accepted_update_episode_count",
            "nonclean_no_update_episode_count",
            "nonclean_accepted_functional_episode_count",
            "nonclean_accepted_proposal_loss_decrease_episode_count",
            "nonclean_accepted_finite_episode_count",
            "nonclean_threshold_crossing_episode_count",
        )
        return {
            "candidate_id": self.candidate_id,
            **{name: getattr(self, name) for name in count_scalars},
            **{
                name: _fraction_receipt(getattr(self, name))
                for name in fraction_scalars
            },
            **{
                name: _fraction_map_receipt(getattr(self, name))
                for name in fraction_maps
            },
            "alignment_dataset_valid_episode_count": _count_map_receipt(
                self.alignment_dataset_valid_episode_count
            ),
        }


def _aggregate_candidate(
    candidate_id: str,
    cells: Sequence[_Cell],
    episodes: Sequence[_Episode],
    gate: GateConfig,
) -> CandidateAggregate:
    all_cells = tuple(cells)
    nonclean_cells = tuple(cell for cell in cells if _is_nonclean_cell(cell))
    clean_cells = tuple(cell for cell in cells if not _is_nonclean_cell(cell))
    all_episodes = tuple(episodes)
    nonclean_episodes = tuple(
        episode for episode in episodes if _is_nonclean_episode(episode)
    )

    def delta_map(
        keys: Sequence[str], selector: Any, attribute: str
    ) -> tuple[tuple[str, Fraction], ...]:
        return tuple(
            (
                key,
                _delta_mean(
                    selector(key), attribute, field=f"{attribute}.{key}"
                ),
            )
            for key in keys
        )

    dataset_nonclean = lambda dataset: tuple(
        cell
        for cell in nonclean_cells
        if cell.dataset == dataset
    )
    family_nonclean = lambda family: tuple(
        cell
        for cell in nonclean_cells
        if cell.corruption_family == family
    )
    clean_dataset = lambda dataset: tuple(
        cell for cell in clean_cells if cell.dataset == dataset
    )

    source_fa = tuple(
        (
            stratum,
            _metric_mean(
                _stratum_cells(cells, stratum),
                "source_fa_per_million",
                field=f"source_fa_per_million.{stratum}",
            ),
        )
        for stratum in SAFETY_STRATA
    )
    adapted_fa = tuple(
        (
            stratum,
            _metric_mean(
                _stratum_cells(cells, stratum),
                "adapted_fa_per_million",
                field=f"adapted_fa_per_million.{stratum}",
            ),
        )
        for stratum in SAFETY_STRATA
    )
    source_fg = tuple(
        (
            stratum,
            _metric_mean(
                _stratum_cells(cells, stratum),
                "source_foreground_fraction",
                field=f"source_foreground_fraction.{stratum}",
            ),
        )
        for stratum in SAFETY_STRATA
    )
    adapted_fg = tuple(
        (
            stratum,
            _metric_mean(
                _stratum_cells(cells, stratum),
                "adapted_foreground_fraction",
                field=f"adapted_foreground_fraction.{stratum}",
            ),
        )
        for stratum in SAFETY_STRATA
    )
    source_fa_dict = dict(source_fa)
    adapted_fa_dict = dict(adapted_fa)
    source_fg_dict = dict(source_fg)
    adapted_fg_dict = dict(adapted_fg)

    valid_cosines = [
        episode.gradient_cosine
        for episode in all_episodes
        if episode.both_gradients_nonzero
    ]
    dataset_valid_cosines = {
        dataset: [
            episode.gradient_cosine
            for episode in all_episodes
            if episode.dataset == dataset and episode.both_gradients_nonzero
        ]
        for dataset in DATASETS
    }
    # Exact zero is an explicit undefined-alignment sentinel when a stratum has
    # no valid pair.  The corresponding valid count is always emitted, and the
    # sentinel is never treated as positive coverage.
    alignment_macro = (
        _mean(valid_cosines, field="alignment_macro_cosine")
        if valid_cosines
        else Fraction(0)
    )
    dataset_medians = tuple(
        (
            dataset,
            _median(values, field=f"alignment_dataset_median.{dataset}")
            if values
            else Fraction(0),
        )
        for dataset, values in dataset_valid_cosines.items()
    )
    dataset_valid_counts = tuple(
        (dataset, len(values))
        for dataset, values in dataset_valid_cosines.items()
    )

    accepted = [episode for episode in nonclean_episodes if episode.accepted_update]
    accepted_count = len(accepted)
    functional_count = sum(
        episode.maximum_absolute_logit_delta
        > gate.functional_logit_delta_threshold + gate.comparison_tolerance
        for episode in accepted
    )
    loss_decrease_count = sum(
        episode.proposal_loss_after
        < episode.proposal_loss_before - gate.comparison_tolerance
        for episode in accepted
    )
    finite_count = sum(episode.finite for episode in accepted)
    crossing_episode_count = sum(
        episode.threshold_crossing_count > 0 for episode in nonclean_episodes
    )
    return CandidateAggregate(
        candidate_id=candidate_id,
        cell_count=len(all_cells),
        nonclean_cell_count=len(nonclean_cells),
        episode_count=len(all_episodes),
        nonclean_episode_count=len(nonclean_episodes),
        overall_macro_delta_iou=_delta_mean(
            all_cells, "delta_iou", field="overall_macro_delta_iou"
        ),
        nonclean_macro_delta_iou=_delta_mean(
            nonclean_cells, "delta_iou", field="nonclean_macro_delta_iou"
        ),
        dataset_nonclean_delta_iou=delta_map(
            DATASETS, dataset_nonclean, "delta_iou"
        ),
        family_nonclean_delta_iou=delta_map(
            CORRUPTION_FAMILIES, family_nonclean, "delta_iou"
        ),
        clean_macro_delta_iou=_delta_mean(
            clean_cells, "delta_iou", field="clean_macro_delta_iou"
        ),
        clean_dataset_delta_iou=delta_map(
            DATASETS, clean_dataset, "delta_iou"
        ),
        overall_macro_delta_pd=_delta_mean(
            all_cells, "delta_pd", field="overall_macro_delta_pd"
        ),
        nonclean_macro_delta_pd=_delta_mean(
            nonclean_cells, "delta_pd", field="nonclean_macro_delta_pd"
        ),
        dataset_nonclean_delta_pd=delta_map(
            DATASETS, dataset_nonclean, "delta_pd"
        ),
        clean_macro_delta_pd=_delta_mean(
            clean_cells, "delta_pd", field="clean_macro_delta_pd"
        ),
        source_fa_per_million=source_fa,
        adapted_fa_per_million=adapted_fa,
        fa_delta_per_million=tuple(
            (key, adapted_fa_dict[key] - source_fa_dict[key])
            for key in SAFETY_STRATA
        ),
        source_foreground_fraction=source_fg,
        adapted_foreground_fraction=adapted_fg,
        foreground_fraction_delta=tuple(
            (key, adapted_fg_dict[key] - source_fg_dict[key])
            for key in SAFETY_STRATA
        ),
        alignment_macro_cosine=alignment_macro,
        alignment_valid_episode_count=len(valid_cosines),
        alignment_dataset_median_cosine=dataset_medians,
        alignment_dataset_valid_episode_count=dataset_valid_counts,
        both_gradients_nonzero_episode_count=len(valid_cosines),
        both_gradients_nonzero_fraction=_ratio(
            len(valid_cosines), len(all_episodes)
        ),
        nonclean_accepted_update_episode_count=accepted_count,
        nonclean_no_update_episode_count=len(nonclean_episodes) - accepted_count,
        nonclean_accepted_update_fraction=_ratio(
            accepted_count, len(nonclean_episodes)
        ),
        nonclean_no_update_fraction=_ratio(
            len(nonclean_episodes) - accepted_count, len(nonclean_episodes)
        ),
        nonclean_accepted_functional_episode_count=functional_count,
        nonclean_accepted_proposal_loss_decrease_episode_count=(
            loss_decrease_count
        ),
        nonclean_accepted_finite_episode_count=finite_count,
        nonclean_accepted_functional_fraction=_ratio(
            functional_count, accepted_count
        ),
        nonclean_accepted_proposal_loss_decrease_fraction=_ratio(
            loss_decrease_count, accepted_count
        ),
        nonclean_accepted_finite_fraction=_ratio(finite_count, accepted_count),
        nonclean_threshold_crossing_episode_count=crossing_episode_count,
        nonclean_threshold_crossing_episode_fraction=_ratio(
            crossing_episode_count, len(nonclean_episodes)
        ),
    )


def _failed_gates(
    aggregate: CandidateAggregate, gate: GateConfig
) -> tuple[str, ...]:
    failures: list[str] = []
    dataset_iou = dict(aggregate.dataset_nonclean_delta_iou)
    family_iou = dict(aggregate.family_nonclean_delta_iou)
    clean_dataset_iou = dict(aggregate.clean_dataset_delta_iou)
    dataset_pd = dict(aggregate.dataset_nonclean_delta_pd)
    dataset_medians = dict(aggregate.alignment_dataset_median_cosine)
    dataset_valid = dict(aggregate.alignment_dataset_valid_episode_count)

    tolerance = gate.comparison_tolerance
    strictly_above = lambda value, threshold: value > threshold + tolerance
    below_minimum = lambda value, threshold: value < threshold - tolerance
    above_maximum = lambda value, threshold: value > threshold + tolerance

    if not strictly_above(
        aggregate.nonclean_macro_delta_iou,
        gate.nonclean_macro_delta_iou_strict_minimum,
    ):
        failures.append("nonclean_macro_iou_nonpositive")
    if not strictly_above(
        aggregate.overall_macro_delta_iou,
        gate.overall_macro_delta_iou_strict_minimum,
    ):
        failures.append("overall_macro_iou_nonpositive")
    if sum(
        strictly_above(value, Fraction(0)) for value in dataset_iou.values()
    ) < gate.minimum_positive_nonclean_datasets:
        failures.append("positive_dataset_coverage")
    if below_minimum(
        min(dataset_iou.values()),
        gate.worst_nonclean_dataset_delta_iou_minimum,
    ):
        failures.append("worst_dataset_iou")
    if sum(
        strictly_above(value, Fraction(0)) for value in family_iou.values()
    ) < gate.minimum_positive_nonclean_families:
        failures.append("positive_family_coverage")
    if below_minimum(
        min(family_iou.values()), gate.worst_nonclean_family_delta_iou_minimum
    ):
        failures.append("worst_family_iou")
    if below_minimum(
        aggregate.clean_macro_delta_iou, gate.clean_macro_delta_iou_minimum
    ):
        failures.append("clean_macro_iou_safety")
    if below_minimum(
        min(clean_dataset_iou.values()), gate.clean_dataset_delta_iou_minimum
    ):
        failures.append("clean_dataset_iou_safety")
    if below_minimum(
        aggregate.nonclean_macro_delta_pd, gate.nonclean_macro_delta_pd_minimum
    ):
        failures.append("nonclean_pd_safety")
    if below_minimum(
        min(dataset_pd.values()), gate.nonclean_dataset_delta_pd_minimum
    ):
        failures.append("dataset_pd_safety")
    if below_minimum(
        aggregate.clean_macro_delta_pd, gate.clean_macro_delta_pd_minimum
    ):
        failures.append("clean_pd_safety")

    source_fa = dict(aggregate.source_fa_per_million)
    fa_delta = dict(aggregate.fa_delta_per_million)
    source_fg = dict(aggregate.source_foreground_fraction)
    adapted_fg = dict(aggregate.adapted_foreground_fraction)
    fg_delta = dict(aggregate.foreground_fraction_delta)
    for stratum in SAFETY_STRATA:
        if above_maximum(fa_delta[stratum], (
            gate.fa_absolute_allowance_per_million
            + gate.fa_source_multiplier * source_fa[stratum]
        )):
            failures.append(f"fa_inflation:{stratum}")
        if above_maximum(
            fg_delta[stratum], gate.foreground_fraction_delta_maximum
        ):
            failures.append(f"foreground_absolute_inflation:{stratum}")
        if above_maximum(adapted_fg[stratum], (
            gate.foreground_fraction_source_multiplier * source_fg[stratum]
            + gate.foreground_fraction_epsilon
        )):
            failures.append(f"foreground_ratio_inflation:{stratum}")

    if below_minimum(
        aggregate.alignment_macro_cosine, gate.alignment_macro_cosine_minimum
    ):
        failures.append("alignment_macro_cosine")
    if sum(
        dataset_valid[dataset] > 0
        and strictly_above(dataset_medians[dataset], Fraction(0))
        for dataset in DATASETS
    ) < gate.minimum_positive_alignment_dataset_medians:
        failures.append("alignment_dataset_coverage")
    if below_minimum(
        aggregate.both_gradients_nonzero_fraction,
        gate.both_gradients_nonzero_fraction_minimum,
    ):
        failures.append("alignment_nonzero_gradient_fraction")

    if below_minimum(
        aggregate.nonclean_accepted_update_fraction,
        gate.accepted_update_fraction_nonclean_minimum,
    ):
        failures.append("accepted_update_fraction_nonclean")
    if above_maximum(
        aggregate.nonclean_no_update_fraction,
        gate.no_update_fraction_nonclean_maximum,
    ):
        failures.append("no_update_fraction_nonclean")
    if below_minimum(
        aggregate.nonclean_accepted_functional_fraction,
        gate.accepted_functional_fraction_minimum,
    ):
        failures.append("accepted_functional_logit_changed_fraction")
    if below_minimum(
        aggregate.nonclean_accepted_proposal_loss_decrease_fraction,
        gate.accepted_proposal_loss_decrease_fraction_minimum,
    ):
        failures.append("accepted_proposal_loss_decrease_fraction")
    if (
        aggregate.nonclean_accepted_finite_fraction
        != gate.accepted_finite_fraction_minimum
    ):
        failures.append("accepted_finite_fraction")
    if below_minimum(
        aggregate.nonclean_threshold_crossing_episode_fraction,
        gate.threshold_crossing_episode_fraction_nonclean_minimum,
    ):
        failures.append("nonclean_threshold_crossing_episode_fraction")
    return tuple(failures)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_id: str
    aggregate: CandidateAggregate
    eligible: bool
    reason_codes: tuple[str, ...]

    def to_receipt(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "eligible": self.eligible,
            "reason_codes": list(self.reason_codes),
            "aggregate": self.aggregate.to_receipt(),
        }


@dataclass(frozen=True, slots=True)
class CandidateRankingEntry:
    rank: int
    candidate_id: str
    nonclean_macro_delta_iou: Fraction
    overall_macro_delta_iou: Fraction
    worst_dataset_nonclean_delta_iou: Fraction
    worst_family_nonclean_delta_iou: Fraction
    clean_macro_delta_iou: Fraction
    nonclean_macro_delta_pd: Fraction
    nonclean_fa_delta_per_million: Fraction
    nonclean_foreground_fraction_delta: Fraction
    alignment_macro_cosine: Fraction

    def to_receipt(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            **{
                name: _fraction_receipt(getattr(self, name))
                for name in (
                    "nonclean_macro_delta_iou",
                    "overall_macro_delta_iou",
                    "worst_dataset_nonclean_delta_iou",
                    "worst_family_nonclean_delta_iou",
                    "clean_macro_delta_iou",
                    "nonclean_macro_delta_pd",
                    "nonclean_fa_delta_per_million",
                    "nonclean_foreground_fraction_delta",
                    "alignment_macro_cosine",
                )
            },
        }


@dataclass(frozen=True, slots=True)
class StageB4R0ScienceDecision:
    gate: GateConfig
    candidate_evaluations: tuple[CandidateEvaluation, ...]
    eligible_candidate_ids: tuple[str, ...]
    ranking: tuple[CandidateRankingEntry, ...]
    scientific_status: Literal[
        "scientific_pending_R1_R2", "scientific_no_eligible"
    ]
    result_tier: Literal["development"] = "development"
    development_only: Literal[True] = True
    paper_result: Literal[False] = False
    test: Literal[False] = False

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(entry.candidate_id for entry in self.ranking)

    @property
    def r1_r2_allowed(self) -> bool:
        return bool(self.ranking)

    @property
    def next_stage_allowed(self) -> bool:
        return self.r1_r2_allowed

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "receipt_type": "cr_sitta_p3_stage_b4_r0_science_decision_v1",
            "evaluation_phase": "R0",
            "protocol_status": "passed",
            "scientific_status": self.scientific_status,
            "result_tier": self.result_tier,
            "development_only": self.development_only,
            "paper_result": self.paper_result,
            "test": self.test,
            "r1_r2_allowed": self.r1_r2_allowed,
            "next_stage_allowed": self.next_stage_allowed,
            "final_candidate_selection_allowed": False,
            "gate": self.gate.to_receipt(),
            "topology": {
                "candidate_ids": list(FROZEN_CANDIDATES),
                "candidate_count": len(FROZEN_CANDIDATES),
                "datasets": list(DATASETS),
                "conditions": list(CONDITIONS),
                "corruption_families": list(CORRUPTION_FAMILIES),
                "severities": list(SEVERITIES),
                "cells_per_candidate": CELLS_PER_CANDIDATE,
                "nonclean_cells_per_candidate": NONCLEAN_CELLS_PER_CANDIDATE,
                "episodes_per_cell": EPISODES_PER_CELL,
                "episodes_per_candidate": EPISODES_PER_CANDIDATE,
                "nonclean_episodes_per_candidate": (
                    NONCLEAN_EPISODES_PER_CANDIDATE
                ),
            },
            "aggregation_policy": {
                "metric_weighting": "equal_cell_macro",
                "safety_strata": list(SAFETY_STRATA),
                "alignment_macro_scope": (
                    "all_R0_episodes_with_both_gradients_nonzero"
                ),
                "alignment_dataset_median_scope": (
                    "dataset_episodes_with_both_gradients_nonzero"
                ),
                "undefined_alignment_encoding": (
                    "exact_zero_sentinel_with_explicit_valid_count"
                ),
                "activity_scope": "nonclean_episodes",
                "accepted_quality_denominator": "accepted_nonclean_episodes",
                "zero_accepted_quality_fraction": "exact_zero",
                "missing_value_policy": "forbidden_no_imputation",
            },
            "selection_policy": {
                "filter_before_ranking": True,
                "maximum_selected_for_r1_r2": (
                    self.gate.maximum_selected_for_r1_r2
                ),
                "ranking_order": list(self.gate.ranking_tie_break_order),
            },
            "across_replicate_policy": {
                "application_at_r0": "recorded_only_not_applied",
                "replicate_ids": list(REPLICATE_IDS),
                "nonclean_mean_delta_iou_minimum": _fraction_receipt(
                    self.gate.across_replicate_nonclean_mean_delta_iou_minimum
                ),
                "overall_mean_delta_iou_minimum": _fraction_receipt(
                    self.gate.across_replicate_overall_mean_delta_iou_minimum
                ),
                "all_replicates_must_pass_single_replicate_hard_gates": (
                    self.gate.every_replicate_must_pass_all_hard_gates
                ),
                "maximum_final_top_candidates": (
                    self.gate.maximum_final_top_candidates
                ),
            },
            "gain_attribution_policy": {
                "accepted_and_no_update_subsets_required": (
                    self.gate.gain_attribution_subsets_required
                ),
                "role": self.gate.gain_attribution_role,
            },
            "candidate_evaluations": [
                evaluation.to_receipt()
                for evaluation in self.candidate_evaluations
            ],
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "selected_for_r1_r2": list(self.selected_candidate_ids),
            "ranking": [entry.to_receipt() for entry in self.ranking],
            "required_followup_replicates": (
                ["R1", "R2"] if self.r1_r2_allowed else []
            ),
            "exit_semantics": (
                "continue_to_R1_R2"
                if self.r1_r2_allowed
                else "normal_scientific_early_stop_exit_0"
            ),
        }


def _ranking_key(evaluation: CandidateEvaluation) -> tuple[Any, ...]:
    aggregate = evaluation.aggregate
    return (
        -aggregate.nonclean_macro_delta_iou,
        -min(dict(aggregate.dataset_nonclean_delta_iou).values()),
        -aggregate.overall_macro_delta_iou,
        -aggregate.clean_macro_delta_iou,
        dict(aggregate.fa_delta_per_million)["nonclean"],
        dict(aggregate.foreground_fraction_delta)["nonclean"],
        -aggregate.nonclean_macro_delta_pd,
        -aggregate.nonclean_accepted_update_fraction,
        evaluation.candidate_id,
    )


def _ranking_entry(
    rank: int, evaluation: CandidateEvaluation
) -> CandidateRankingEntry:
    aggregate = evaluation.aggregate
    return CandidateRankingEntry(
        rank=rank,
        candidate_id=evaluation.candidate_id,
        nonclean_macro_delta_iou=aggregate.nonclean_macro_delta_iou,
        overall_macro_delta_iou=aggregate.overall_macro_delta_iou,
        worst_dataset_nonclean_delta_iou=min(
            dict(aggregate.dataset_nonclean_delta_iou).values()
        ),
        worst_family_nonclean_delta_iou=min(
            dict(aggregate.family_nonclean_delta_iou).values()
        ),
        clean_macro_delta_iou=aggregate.clean_macro_delta_iou,
        nonclean_macro_delta_pd=aggregate.nonclean_macro_delta_pd,
        nonclean_fa_delta_per_million=dict(
            aggregate.fa_delta_per_million
        )["nonclean"],
        nonclean_foreground_fraction_delta=dict(
            aggregate.foreground_fraction_delta
        )["nonclean"],
        alignment_macro_cosine=aggregate.alignment_macro_cosine,
    )


def _validate_cell_episode_consistency(
    cells: Mapping[tuple[str, str, str], _Cell],
    episodes: Mapping[tuple[str, str, str, int], _Episode],
) -> None:
    expected_indices = set(range(EPISODES_PER_CELL))
    for candidate in FROZEN_CANDIDATES:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                cell_key = (candidate, dataset, condition)
                selected = tuple(
                    episodes[(candidate, dataset, condition, index)]
                    for index in range(EPISODES_PER_CELL)
                )
                observed_indices = {episode.episode_index for episode in selected}
                if observed_indices != expected_indices:
                    raise StageB4R0ProtocolError(
                        f"episode indices are incomplete for {cell_key}"
                    )
                cell = cells[cell_key]
                accepted_count = sum(episode.accepted_update for episode in selected)
                crossing_count = sum(
                    episode.threshold_crossing_count > 0 for episode in selected
                )
                endpoints_equal = (
                    cell.source_iou == cell.adapted_iou
                    and cell.source_pd == cell.adapted_pd
                    and cell.source_fa_per_million == cell.adapted_fa_per_million
                    and cell.source_foreground_fraction
                    == cell.adapted_foreground_fraction
                )
                if accepted_count == 0 and not endpoints_equal:
                    raise StageB4R0ProtocolError(
                        f"{cell_key} has metric change but zero accepted updates"
                    )
                if crossing_count == 0 and not endpoints_equal:
                    raise StageB4R0ProtocolError(
                        f"{cell_key} has thresholded metric change but zero "
                        "threshold-crossing episodes"
                    )


def evaluate_candidates(
    cell_summaries: Sequence[Mapping[str, Any]],
    episode_summaries: Sequence[Mapping[str, Any]],
    gate_config: Mapping[str, Any],
) -> StageB4R0ScienceDecision:
    """Validate, aggregate, hard-filter, then rank the exact Stage-B4 R0 grid."""

    gate = GateConfig.from_mapping(gate_config)
    raw_cells = _sequence(cell_summaries, field="cell_summaries")
    expected_cell_count = len(FROZEN_CANDIDATES) * CELLS_PER_CANDIDATE
    if len(raw_cells) != expected_cell_count:
        raise StageB4R0ProtocolError(
            "cell summary count must equal 4*39; "
            f"expected={expected_cell_count}, observed={len(raw_cells)}"
        )
    parsed_cells = tuple(
        _parse_cell(value, index=index) for index, value in enumerate(raw_cells)
    )
    cells: dict[tuple[str, str, str], _Cell] = {}
    for cell in parsed_cells:
        key = (cell.candidate_id, cell.dataset, cell.condition)
        if key in cells:
            raise StageB4R0ProtocolError(f"duplicate candidate/dataset/condition cell: {key}")
        cells[key] = cell
    expected_cell_keys = {
        (candidate, dataset, condition)
        for candidate in FROZEN_CANDIDATES
        for dataset in DATASETS
        for condition in CONDITIONS
    }
    if set(cells) != expected_cell_keys:
        raise StageB4R0ProtocolError("cell summaries do not have exact 4x3x13 topology")

    raw_episodes = _sequence(episode_summaries, field="episode_summaries")
    expected_episode_count = len(FROZEN_CANDIDATES) * EPISODES_PER_CANDIDATE
    if len(raw_episodes) != expected_episode_count:
        raise StageB4R0ProtocolError(
            "episode summary count must equal 4*2496; "
            f"expected={expected_episode_count}, observed={len(raw_episodes)}"
        )
    parsed_episodes = tuple(
        _parse_episode(value, index=index)
        for index, value in enumerate(raw_episodes)
    )
    episodes: dict[tuple[str, str, str, int], _Episode] = {}
    for episode in parsed_episodes:
        key = (
            episode.candidate_id,
            episode.dataset,
            episode.condition,
            episode.episode_index,
        )
        if key in episodes:
            raise StageB4R0ProtocolError(
                f"duplicate candidate/dataset/condition/episode cell: {key}"
            )
        episodes[key] = episode
    expected_episode_keys = {
        (candidate, dataset, condition, episode_index)
        for candidate in FROZEN_CANDIDATES
        for dataset in DATASETS
        for condition in CONDITIONS
        for episode_index in range(EPISODES_PER_CELL)
    }
    if set(episodes) != expected_episode_keys:
        raise StageB4R0ProtocolError(
            "episode summaries do not have exact 4x3x13x64 topology"
        )

    # All candidates must be compared against the identical frozen Source
    # endpoint for each dataset/condition cell.
    for dataset in DATASETS:
        for condition in CONDITIONS:
            reference = cells[(FROZEN_CANDIDATES[0], dataset, condition)]
            for candidate in FROZEN_CANDIDATES[1:]:
                other = cells[(candidate, dataset, condition)]
                if reference.source_counts != other.source_counts:
                    raise StageB4R0ProtocolError(
                        "Source sufficient counts differ across candidates for "
                        f"{dataset}/{condition}"
                    )

    # Every corruption condition for a dataset covers the same frozen Pilot64
    # images and unchanged GT masks, so all task denominators must be invariant.
    for dataset in DATASETS:
        reference = cells[(FROZEN_CANDIDATES[0], dataset, CONDITIONS[0])]
        denominators = (
            reference.source_counts.total_image_pixels,
            reference.source_counts.image_count,
            reference.source_counts.target_positive_pixels,
            reference.source_counts.total_targets,
        )
        for condition in CONDITIONS[1:]:
            cell = cells[(FROZEN_CANDIDATES[0], dataset, condition)]
            observed = (
                cell.source_counts.total_image_pixels,
                cell.source_counts.image_count,
                cell.source_counts.target_positive_pixels,
                cell.source_counts.total_targets,
            )
            if observed != denominators:
                raise StageB4R0ProtocolError(
                    "Pilot64/GT sufficient-count denominators differ across "
                    f"conditions for {dataset}"
                )

    _validate_cell_episode_consistency(cells, episodes)

    evaluations: list[CandidateEvaluation] = []
    for candidate in FROZEN_CANDIDATES:
        candidate_cells = tuple(
            cells[(candidate, dataset, condition)]
            for dataset in DATASETS
            for condition in CONDITIONS
        )
        candidate_episodes = tuple(
            episodes[(candidate, dataset, condition, episode_index)]
            for dataset in DATASETS
            for condition in CONDITIONS
            for episode_index in range(EPISODES_PER_CELL)
        )
        aggregate = _aggregate_candidate(
            candidate, candidate_cells, candidate_episodes, gate
        )
        reasons = _failed_gates(aggregate, gate)
        evaluations.append(
            CandidateEvaluation(
                candidate_id=candidate,
                aggregate=aggregate,
                eligible=not reasons,
                reason_codes=reasons,
            )
        )

    eligible = sorted(
        (evaluation for evaluation in evaluations if evaluation.eligible),
        key=_ranking_key,
    )
    selected = eligible[: gate.maximum_selected_for_r1_r2]
    ranking = tuple(
        _ranking_entry(rank, evaluation)
        for rank, evaluation in enumerate(selected, start=1)
    )
    return StageB4R0ScienceDecision(
        gate=gate,
        candidate_evaluations=tuple(evaluations),
        eligible_candidate_ids=tuple(
            evaluation.candidate_id for evaluation in eligible
        ),
        ranking=ranking,
        scientific_status=(
            "scientific_pending_R1_R2"
            if selected
            else "scientific_no_eligible"
        ),
    )


def evaluate_stage_b4_r0_science_gate(
    cell_summaries: Sequence[Mapping[str, Any]],
    episode_summaries: Sequence[Mapping[str, Any]],
    gate_config: Mapping[str, Any],
) -> StageB4R0ScienceDecision:
    """Descriptive alias for :func:`evaluate_candidates`."""

    return evaluate_candidates(cell_summaries, episode_summaries, gate_config)


__all__ = [
    "ACROSS_REPLICATE_FIELDS",
    "ACTIVITY_FIELDS",
    "ALIGNMENT_FIELDS",
    "CELL_FIELDS",
    "CELLS_PER_CANDIDATE",
    "CONDITIONS",
    "COUNT_FIELDS",
    "CORRUPTION_FAMILIES",
    "CandidateAggregate",
    "CandidateEvaluation",
    "CandidateRankingEntry",
    "DATASETS",
    "EPISODE_FIELDS",
    "EPISODES_PER_CANDIDATE",
    "EPISODES_PER_CELL",
    "FROZEN_CANDIDATES",
    "FOREGROUND_FIELDS",
    "GAIN_ATTRIBUTION_FIELDS",
    "GATE_CONFIG_FIELDS",
    "GateConfig",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "NONCLEAN_CELLS_PER_CANDIDATE",
    "NONCLEAN_EPISODES_PER_CANDIDATE",
    "PERFORMANCE_AND_SAFETY_FIELDS",
    "RANKING_TIE_BREAK_ORDER",
    "REPLICATE_IDS",
    "SAFETY_STRATA",
    "SEVERITIES",
    "StageB4R0ProtocolError",
    "StageB4R0ScienceDecision",
    "StageB4ScienceGateError",
    "TOTAL_IMAGE_PIXELS_PER_CELL",
    "default_gate_config",
    "evaluate_candidates",
    "evaluate_stage_b4_r0_science_gate",
]
