"""Pure Stage-B3 objective/parameter-space aggregation and science gate.

The module is deliberately CPU-, filesystem-, model-, and dataset-free.  It
consumes JSON-safe summaries for one exact ``3 datasets x 13 conditions``
grid per caller-declared candidate.  Protocol/schema errors raise
``StageB3ScienceGateError``; a complete grid with no scientifically eligible
candidate is a normal ``scientific_no_eligible`` result.

All gate thresholds are supplied through one exact configuration mapping so
the future runner can freeze and hash that mapping before looking at results.
Numeric comparisons use exact rational arithmetic after parsing finite JSON
numbers.  The receipt therefore contains only JSON-safe integer ratios and
never invents, imputes, or rounds missing evidence.
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
CELLS_PER_CANDIDATE = len(DATASETS) * len(CONDITIONS)
NONCLEAN_CELLS_PER_CANDIDATE = len(DATASETS) * (len(CONDITIONS) - 1)
EPISODES_PER_CELL = 16
EPISODES_PER_CANDIDATE = CELLS_PER_CANDIDATE * EPISODES_PER_CELL

FOREGROUND_SAFETY_STRATA = (
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
        "normalized_task_directional_derivative",
        "source_iou",
        "adapted_iou",
        "delta_pd",
        "delta_fa",
        "source_foreground_fraction",
        "adapted_foreground_fraction",
        "valid_alignment_episode_count",
        "episode_count",
        "functional_changed_episode_count",
        "threshold_crossing_count",
    }
)

GATE_CONFIG_FIELDS = frozenset(
    {
        "comparison_tolerance",
        "maximum_selected_candidates",
        "normalized_task_directional_derivative_maximum",
        "minimum_negative_derivative_datasets",
        "minimum_negative_derivative_families",
        "minimum_negative_derivative_severities",
        "nonclean_macro_delta_iou_strictly_greater_than",
        "clean_macro_delta_iou_minimum",
        "foreground_fraction_delta_maximum",
        "foreground_fraction_source_multiplier",
        "foreground_fraction_epsilon",
        "minimum_functional_changed_episodes",
        "minimum_threshold_crossing_count",
        "minimum_positive_delta_iou_severities",
        "filter_before_ranking",
        "no_eligible_is_normal_scientific_result",
        "negative_controls_selectable",
    }
)


class StageB3ScienceGateError(ValueError):
    """The frozen roster, topology, schema, or numeric evidence is invalid."""


ProxyObjectiveSpaceScreenError = StageB3ScienceGateError


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageB3ScienceGateError(f"{field} must be a mapping")
    return value


def _exact_mapping(
    value: Any, expected: frozenset[str], *, field: str
) -> Mapping[str, Any]:
    result = _mapping(value, field=field)
    observed = set(result)
    if observed != expected:
        raise StageB3ScienceGateError(
            f"{field} fields must be exact; "
            f"missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected, key=str)}"
        )
    if any(not isinstance(key, str) for key in result):
        raise StageB3ScienceGateError(f"{field} keys must be strings")
    return result


def _sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise StageB3ScienceGateError(f"{field} must be a sequence")
    return tuple(value)


def _fraction(value: Any, *, field: str) -> Fraction:
    """Parse a finite JSON scalar without retaining binary-float arithmetic."""

    if isinstance(value, bool):
        raise StageB3ScienceGateError(f"{field} must be a finite JSON number")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StageB3ScienceGateError(f"{field} must be finite")
        return Fraction(Decimal(str(value)))
    # Numeric strings are JSON-safe and useful for thresholds loaded from
    # YAML, while Decimal/Fraction inputs are intentionally not JSON-safe.
    if isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise StageB3ScienceGateError(
                f"{field} must be a finite JSON number"
            ) from exc
        if not parsed.is_finite():
            raise StageB3ScienceGateError(f"{field} must be finite")
        return Fraction(parsed)
    raise StageB3ScienceGateError(f"{field} must be a finite JSON number")


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StageB3ScienceGateError(
            f"{field} must be a non-negative JSON integer"
        )
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise StageB3ScienceGateError(f"{field} must be a JSON boolean")
    return value


def _bounded_integer(
    value: Any, *, field: str, minimum: int, maximum: int
) -> int:
    result = _nonnegative_integer(value, field=field)
    if not minimum <= result <= maximum:
        raise StageB3ScienceGateError(
            f"{field} must be in [{minimum}, {maximum}]"
        )
    return result


def _mean(values: Sequence[Fraction], *, field: str) -> Fraction:
    if not values:
        raise StageB3ScienceGateError(f"{field} cannot be empty")
    return sum(values, Fraction(0, 1)) / len(values)


def _fraction_receipt(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_map_receipt(
    values: tuple[tuple[str, Fraction], ...]
) -> dict[str, dict[str, int]]:
    return {key: _fraction_receipt(value) for key, value in values}


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Fully explicit, caller-frozen Stage-B3 thresholds."""

    comparison_tolerance: Fraction
    maximum_selected_candidates: int
    normalized_task_directional_derivative_maximum: Fraction
    minimum_negative_derivative_datasets: int
    minimum_negative_derivative_families: int
    minimum_negative_derivative_severities: int
    nonclean_macro_delta_iou_strictly_greater_than: Fraction
    clean_macro_delta_iou_minimum: Fraction
    foreground_fraction_delta_maximum: Fraction
    foreground_fraction_source_multiplier: Fraction
    foreground_fraction_epsilon: Fraction
    minimum_functional_changed_episodes: int
    minimum_threshold_crossing_count: int
    minimum_positive_delta_iou_severities: int
    filter_before_ranking: bool
    no_eligible_is_normal_scientific_result: bool
    negative_controls_selectable: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GateConfig":
        gate = _exact_mapping(value, GATE_CONFIG_FIELDS, field="gate_config")
        result = cls(
            comparison_tolerance=_fraction(
                gate["comparison_tolerance"],
                field="gate_config.comparison_tolerance",
            ),
            maximum_selected_candidates=_bounded_integer(
                gate["maximum_selected_candidates"],
                field="gate_config.maximum_selected_candidates",
                minimum=1,
                maximum=4,
            ),
            normalized_task_directional_derivative_maximum=_fraction(
                gate["normalized_task_directional_derivative_maximum"],
                field=(
                    "gate_config.normalized_task_directional_derivative_maximum"
                ),
            ),
            minimum_negative_derivative_datasets=_bounded_integer(
                gate["minimum_negative_derivative_datasets"],
                field="gate_config.minimum_negative_derivative_datasets",
                minimum=2,
                maximum=len(DATASETS),
            ),
            minimum_negative_derivative_families=_bounded_integer(
                gate["minimum_negative_derivative_families"],
                field="gate_config.minimum_negative_derivative_families",
                minimum=3,
                maximum=len(CORRUPTION_FAMILIES),
            ),
            minimum_negative_derivative_severities=_bounded_integer(
                gate["minimum_negative_derivative_severities"],
                field="gate_config.minimum_negative_derivative_severities",
                minimum=2,
                maximum=len(SEVERITIES),
            ),
            nonclean_macro_delta_iou_strictly_greater_than=_fraction(
                gate["nonclean_macro_delta_iou_strictly_greater_than"],
                field=(
                    "gate_config.nonclean_macro_delta_iou_strictly_greater_than"
                ),
            ),
            clean_macro_delta_iou_minimum=_fraction(
                gate["clean_macro_delta_iou_minimum"],
                field="gate_config.clean_macro_delta_iou_minimum",
            ),
            foreground_fraction_delta_maximum=_fraction(
                gate["foreground_fraction_delta_maximum"],
                field="gate_config.foreground_fraction_delta_maximum",
            ),
            foreground_fraction_source_multiplier=_fraction(
                gate["foreground_fraction_source_multiplier"],
                field="gate_config.foreground_fraction_source_multiplier",
            ),
            foreground_fraction_epsilon=_fraction(
                gate["foreground_fraction_epsilon"],
                field="gate_config.foreground_fraction_epsilon",
            ),
            minimum_functional_changed_episodes=_nonnegative_integer(
                gate["minimum_functional_changed_episodes"],
                field="gate_config.minimum_functional_changed_episodes",
            ),
            minimum_threshold_crossing_count=_nonnegative_integer(
                gate["minimum_threshold_crossing_count"],
                field="gate_config.minimum_threshold_crossing_count",
            ),
            minimum_positive_delta_iou_severities=_bounded_integer(
                gate["minimum_positive_delta_iou_severities"],
                field="gate_config.minimum_positive_delta_iou_severities",
                minimum=2,
                maximum=len(SEVERITIES),
            ),
            filter_before_ranking=_boolean(
                gate["filter_before_ranking"],
                field="gate_config.filter_before_ranking",
            ),
            no_eligible_is_normal_scientific_result=_boolean(
                gate["no_eligible_is_normal_scientific_result"],
                field="gate_config.no_eligible_is_normal_scientific_result",
            ),
            negative_controls_selectable=_boolean(
                gate["negative_controls_selectable"],
                field="gate_config.negative_controls_selectable",
            ),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if self.comparison_tolerance < 0:
            raise StageB3ScienceGateError(
                "gate_config.comparison_tolerance must be non-negative"
            )
        # A positive derivative boundary or negative IoU boundary would relax
        # the two v5 directional gates.  Stricter frozen values remain legal.
        if self.normalized_task_directional_derivative_maximum > 0:
            raise StageB3ScienceGateError(
                "gate_config.normalized_task_directional_derivative_maximum "
                "must be <= 0"
            )
        if self.nonclean_macro_delta_iou_strictly_greater_than < 0:
            raise StageB3ScienceGateError(
                "gate_config.nonclean_macro_delta_iou_strictly_greater_than "
                "must be >= 0"
            )
        if not -1 <= self.clean_macro_delta_iou_minimum <= 1:
            raise StageB3ScienceGateError(
                "gate_config.clean_macro_delta_iou_minimum must be in [-1, 1]"
            )
        for name in (
            "foreground_fraction_delta_maximum",
            "foreground_fraction_source_multiplier",
            "foreground_fraction_epsilon",
        ):
            if getattr(self, name) < 0:
                raise StageB3ScienceGateError(
                    f"gate_config.{name} must be non-negative"
                )
        if self.foreground_fraction_delta_maximum > 1:
            raise StageB3ScienceGateError(
                "gate_config.foreground_fraction_delta_maximum must be <= 1"
            )
        if self.minimum_functional_changed_episodes < 1:
            raise StageB3ScienceGateError(
                "gate_config.minimum_functional_changed_episodes must be >= 1"
            )
        if self.minimum_threshold_crossing_count < 1:
            raise StageB3ScienceGateError(
                "gate_config.minimum_threshold_crossing_count must be >= 1"
            )
        if self.filter_before_ranking is not True:
            raise StageB3ScienceGateError(
                "gate_config.filter_before_ranking must be true"
            )
        if self.no_eligible_is_normal_scientific_result is not True:
            raise StageB3ScienceGateError(
                "gate_config.no_eligible_is_normal_scientific_result must be true"
            )
        if self.negative_controls_selectable is not False:
            raise StageB3ScienceGateError(
                "gate_config.negative_controls_selectable must be false"
            )

    def to_receipt(self) -> dict[str, Any]:
        numeric_names = (
            "comparison_tolerance",
            "normalized_task_directional_derivative_maximum",
            "nonclean_macro_delta_iou_strictly_greater_than",
            "clean_macro_delta_iou_minimum",
            "foreground_fraction_delta_maximum",
            "foreground_fraction_source_multiplier",
            "foreground_fraction_epsilon",
        )
        integer_names = (
            "maximum_selected_candidates",
            "minimum_negative_derivative_datasets",
            "minimum_negative_derivative_families",
            "minimum_negative_derivative_severities",
            "minimum_functional_changed_episodes",
            "minimum_threshold_crossing_count",
            "minimum_positive_delta_iou_severities",
        )
        return {
            **{
                name: _fraction_receipt(getattr(self, name))
                for name in numeric_names
            },
            **{name: getattr(self, name) for name in integer_names},
            "filter_before_ranking": self.filter_before_ranking,
            "no_eligible_is_normal_scientific_result": (
                self.no_eligible_is_normal_scientific_result
            ),
            "negative_controls_selectable": self.negative_controls_selectable,
        }


@dataclass(frozen=True, slots=True)
class _CellSummary:
    candidate_id: str
    dataset: str
    condition: str
    corruption_family: str
    severity: str
    normalized_task_directional_derivative: Fraction
    source_iou: Fraction
    adapted_iou: Fraction
    delta_pd: Fraction
    delta_fa: Fraction
    source_foreground_fraction: Fraction
    adapted_foreground_fraction: Fraction
    valid_alignment_episode_count: int
    episode_count: int
    functional_changed_episode_count: int
    threshold_crossing_count: int

    @property
    def delta_iou(self) -> Fraction:
        return self.adapted_iou - self.source_iou


def _condition_parts(condition: str) -> tuple[str, str]:
    if condition == "clean_S0":
        return "clean", "S0"
    for family in CORRUPTION_FAMILIES:
        prefix = f"{family}_"
        if condition.startswith(prefix):
            return family, condition[len(prefix) :]
    raise StageB3ScienceGateError(f"unknown condition {condition!r}")


def _parse_cell(value: Any, *, index: int) -> _CellSummary:
    field = f"cell_summaries[{index}]"
    raw = _exact_mapping(value, CELL_FIELDS, field=field)
    candidate_id = raw["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id:
        raise StageB3ScienceGateError(
            f"{field}.candidate_id must be a non-empty string"
        )
    dataset = raw["dataset"]
    condition = raw["condition"]
    family = raw["corruption_family"]
    severity = raw["severity"]
    if dataset not in DATASETS:
        raise StageB3ScienceGateError(
            f"{field}.dataset is not one of the frozen three datasets"
        )
    if condition not in CONDITIONS:
        raise StageB3ScienceGateError(
            f"{field}.condition is not one of the frozen thirteen conditions"
        )
    expected_family, expected_severity = _condition_parts(condition)
    if family != expected_family or severity != expected_severity:
        raise StageB3ScienceGateError(
            f"{field} corruption_family/severity disagree with condition"
        )

    source_iou = _fraction(raw["source_iou"], field=f"{field}.source_iou")
    adapted_iou = _fraction(raw["adapted_iou"], field=f"{field}.adapted_iou")
    delta_pd = _fraction(raw["delta_pd"], field=f"{field}.delta_pd")
    delta_fa = _fraction(raw["delta_fa"], field=f"{field}.delta_fa")
    source_foreground = _fraction(
        raw["source_foreground_fraction"],
        field=f"{field}.source_foreground_fraction",
    )
    adapted_foreground = _fraction(
        raw["adapted_foreground_fraction"],
        field=f"{field}.adapted_foreground_fraction",
    )
    for name, numeric in (
        ("source_iou", source_iou),
        ("adapted_iou", adapted_iou),
        ("source_foreground_fraction", source_foreground),
        ("adapted_foreground_fraction", adapted_foreground),
    ):
        if not 0 <= numeric <= 1:
            raise StageB3ScienceGateError(f"{field}.{name} must be in [0, 1]")
    if not -1 <= delta_pd <= 1:
        raise StageB3ScienceGateError(f"{field}.delta_pd must be in [-1, 1]")

    episode_count = _nonnegative_integer(
        raw["episode_count"], field=f"{field}.episode_count"
    )
    if episode_count != EPISODES_PER_CELL:
        raise StageB3ScienceGateError(
            f"{field}.episode_count must equal {EPISODES_PER_CELL}"
        )
    valid_count = _nonnegative_integer(
        raw["valid_alignment_episode_count"],
        field=f"{field}.valid_alignment_episode_count",
    )
    functional_count = _nonnegative_integer(
        raw["functional_changed_episode_count"],
        field=f"{field}.functional_changed_episode_count",
    )
    if valid_count > episode_count:
        raise StageB3ScienceGateError(
            f"{field}.valid_alignment_episode_count exceeds episode_count"
        )
    if functional_count > episode_count:
        raise StageB3ScienceGateError(
            f"{field}.functional_changed_episode_count exceeds episode_count"
        )
    threshold_count = _nonnegative_integer(
        raw["threshold_crossing_count"],
        field=f"{field}.threshold_crossing_count",
    )
    derivative = _fraction(
        raw["normalized_task_directional_derivative"],
        field=f"{field}.normalized_task_directional_derivative",
    )
    if valid_count == 0 and derivative != 0:
        raise StageB3ScienceGateError(
            f"{field} cannot report a nonzero directional derivative with "
            "zero valid alignment episodes"
        )
    if functional_count == 0:
        if (
            adapted_iou != source_iou
            or delta_pd != 0
            or delta_fa != 0
            or adapted_foreground != source_foreground
            or threshold_count != 0
        ):
            raise StageB3ScienceGateError(
                f"{field} reports endpoint/activity change with zero "
                "functional changed episodes"
            )
    if threshold_count == 0 and (
        adapted_iou != source_iou
        or delta_pd != 0
        or delta_fa != 0
        or adapted_foreground != source_foreground
    ):
        raise StageB3ScienceGateError(
            f"{field} reports thresholded metric change with zero "
            "threshold crossings"
        )
    return _CellSummary(
        candidate_id=candidate_id,
        dataset=dataset,
        condition=condition,
        corruption_family=family,
        severity=severity,
        normalized_task_directional_derivative=derivative,
        source_iou=source_iou,
        adapted_iou=adapted_iou,
        delta_pd=delta_pd,
        delta_fa=delta_fa,
        source_foreground_fraction=source_foreground,
        adapted_foreground_fraction=adapted_foreground,
        valid_alignment_episode_count=valid_count,
        episode_count=episode_count,
        functional_changed_episode_count=functional_count,
        threshold_crossing_count=threshold_count,
    )


def _nonclean(cells: Sequence[_CellSummary]) -> tuple[_CellSummary, ...]:
    return tuple(cell for cell in cells if cell.condition != "clean_S0")


def _clean(cells: Sequence[_CellSummary]) -> tuple[_CellSummary, ...]:
    return tuple(cell for cell in cells if cell.condition == "clean_S0")


def _by_dataset_nonclean(
    cells: Sequence[_CellSummary], dataset: str
) -> tuple[_CellSummary, ...]:
    return tuple(
        cell
        for cell in cells
        if cell.dataset == dataset and cell.condition != "clean_S0"
    )


def _by_family(
    cells: Sequence[_CellSummary], family: str
) -> tuple[_CellSummary, ...]:
    return tuple(cell for cell in cells if cell.corruption_family == family)


def _by_severity(
    cells: Sequence[_CellSummary], severity: str
) -> tuple[_CellSummary, ...]:
    return tuple(cell for cell in cells if cell.severity == severity)


def _metric_mean(
    cells: Sequence[_CellSummary], attribute: str, *, field: str
) -> Fraction:
    return _mean([getattr(cell, attribute) for cell in cells], field=field)


def _delta_iou_mean(
    cells: Sequence[_CellSummary], *, field: str
) -> Fraction:
    return _mean([cell.delta_iou for cell in cells], field=field)


def _ordered_metric_map(
    cells: Sequence[_CellSummary],
    keys: Sequence[str],
    selector: Any,
    attribute: str,
) -> tuple[tuple[str, Fraction], ...]:
    return tuple(
        (
            key,
            _metric_mean(
                selector(cells, key), attribute, field=f"{attribute}.{key}"
            ),
        )
        for key in keys
    )


def _ordered_delta_iou_map(
    cells: Sequence[_CellSummary], keys: Sequence[str], selector: Any
) -> tuple[tuple[str, Fraction], ...]:
    return tuple(
        (key, _delta_iou_mean(selector(cells, key), field=f"delta_iou.{key}"))
        for key in keys
    )


def _foreground_stratum_cells(
    cells: Sequence[_CellSummary], stratum: str
) -> tuple[_CellSummary, ...]:
    if stratum == "overall":
        return tuple(cells)
    if stratum == "nonclean":
        return _nonclean(cells)
    if stratum == "clean":
        return _clean(cells)
    if stratum.startswith("dataset:"):
        dataset = stratum.partition(":")[2]
        return tuple(cell for cell in cells if cell.dataset == dataset)
    if stratum.startswith("corruption_family:"):
        return _by_family(cells, stratum.partition(":")[2])
    raise StageB3ScienceGateError(f"unknown foreground stratum {stratum}")


@dataclass(frozen=True, slots=True)
class CandidateAggregate:
    candidate_id: str
    cell_count: int
    nonclean_cell_count: int
    nonclean_macro_directional_derivative: Fraction
    dataset_nonclean_directional_derivative: tuple[tuple[str, Fraction], ...]
    family_directional_derivative: tuple[tuple[str, Fraction], ...]
    severity_directional_derivative: tuple[tuple[str, Fraction], ...]
    overall_source_iou: Fraction
    overall_adapted_iou: Fraction
    overall_macro_delta_iou: Fraction
    nonclean_source_iou: Fraction
    nonclean_adapted_iou: Fraction
    nonclean_macro_delta_iou: Fraction
    clean_source_iou: Fraction
    clean_adapted_iou: Fraction
    clean_macro_delta_iou: Fraction
    dataset_nonclean_delta_iou: tuple[tuple[str, Fraction], ...]
    family_nonclean_delta_iou: tuple[tuple[str, Fraction], ...]
    severity_nonclean_delta_iou: tuple[tuple[str, Fraction], ...]
    clean_dataset_delta_iou: tuple[tuple[str, Fraction], ...]
    overall_macro_delta_pd: Fraction
    nonclean_macro_delta_pd: Fraction
    clean_macro_delta_pd: Fraction
    dataset_nonclean_delta_pd: tuple[tuple[str, Fraction], ...]
    family_nonclean_delta_pd: tuple[tuple[str, Fraction], ...]
    severity_nonclean_delta_pd: tuple[tuple[str, Fraction], ...]
    overall_macro_delta_fa: Fraction
    nonclean_macro_delta_fa: Fraction
    clean_macro_delta_fa: Fraction
    dataset_nonclean_delta_fa: tuple[tuple[str, Fraction], ...]
    family_nonclean_delta_fa: tuple[tuple[str, Fraction], ...]
    severity_nonclean_delta_fa: tuple[tuple[str, Fraction], ...]
    source_foreground_fraction: tuple[tuple[str, Fraction], ...]
    adapted_foreground_fraction: tuple[tuple[str, Fraction], ...]
    foreground_fraction_delta: tuple[tuple[str, Fraction], ...]
    episode_count: int
    valid_alignment_episode_count: int
    functional_changed_episode_count: int
    threshold_crossing_count: int
    nonclean_episode_count: int
    nonclean_valid_alignment_episode_count: int
    nonclean_functional_changed_episode_count: int
    nonclean_threshold_crossing_count: int
    severity_valid_alignment_episode_count: tuple[tuple[str, int], ...]
    severity_functional_changed_episode_count: tuple[tuple[str, int], ...]
    severity_threshold_crossing_count: tuple[tuple[str, int], ...]

    def to_receipt(self) -> dict[str, Any]:
        scalar_fractions = (
            "nonclean_macro_directional_derivative",
            "overall_source_iou",
            "overall_adapted_iou",
            "overall_macro_delta_iou",
            "nonclean_source_iou",
            "nonclean_adapted_iou",
            "nonclean_macro_delta_iou",
            "clean_source_iou",
            "clean_adapted_iou",
            "clean_macro_delta_iou",
            "overall_macro_delta_pd",
            "nonclean_macro_delta_pd",
            "clean_macro_delta_pd",
            "overall_macro_delta_fa",
            "nonclean_macro_delta_fa",
            "clean_macro_delta_fa",
        )
        fraction_maps = (
            "dataset_nonclean_directional_derivative",
            "family_directional_derivative",
            "severity_directional_derivative",
            "dataset_nonclean_delta_iou",
            "family_nonclean_delta_iou",
            "severity_nonclean_delta_iou",
            "clean_dataset_delta_iou",
            "dataset_nonclean_delta_pd",
            "family_nonclean_delta_pd",
            "severity_nonclean_delta_pd",
            "dataset_nonclean_delta_fa",
            "family_nonclean_delta_fa",
            "severity_nonclean_delta_fa",
            "source_foreground_fraction",
            "adapted_foreground_fraction",
            "foreground_fraction_delta",
        )
        integer_scalars = (
            "cell_count",
            "nonclean_cell_count",
            "episode_count",
            "valid_alignment_episode_count",
            "functional_changed_episode_count",
            "threshold_crossing_count",
            "nonclean_episode_count",
            "nonclean_valid_alignment_episode_count",
            "nonclean_functional_changed_episode_count",
            "nonclean_threshold_crossing_count",
        )
        integer_maps = (
            "severity_valid_alignment_episode_count",
            "severity_functional_changed_episode_count",
            "severity_threshold_crossing_count",
        )
        return {
            "candidate_id": self.candidate_id,
            **{name: getattr(self, name) for name in integer_scalars},
            **{
                name: _fraction_receipt(getattr(self, name))
                for name in scalar_fractions
            },
            **{
                name: _fraction_map_receipt(getattr(self, name))
                for name in fraction_maps
            },
            **{name: dict(getattr(self, name)) for name in integer_maps},
        }


def _aggregate_candidate(
    candidate_id: str, cells: Sequence[_CellSummary]
) -> CandidateAggregate:
    all_cells = tuple(cells)
    nonclean = _nonclean(cells)
    clean = _clean(cells)
    dataset_maps = lambda attribute: _ordered_metric_map(
        cells, DATASETS, _by_dataset_nonclean, attribute
    )
    family_maps = lambda attribute: _ordered_metric_map(
        cells, CORRUPTION_FAMILIES, _by_family, attribute
    )
    severity_maps = lambda attribute: _ordered_metric_map(
        cells, SEVERITIES, _by_severity, attribute
    )
    clean_by_dataset = {
        dataset: tuple(
            cell
            for cell in clean
            if cell.dataset == dataset
        )
        for dataset in DATASETS
    }
    source_fg = tuple(
        (
            stratum,
            _metric_mean(
                _foreground_stratum_cells(cells, stratum),
                "source_foreground_fraction",
                field=f"source_foreground_fraction.{stratum}",
            ),
        )
        for stratum in FOREGROUND_SAFETY_STRATA
    )
    adapted_fg = tuple(
        (
            stratum,
            _metric_mean(
                _foreground_stratum_cells(cells, stratum),
                "adapted_foreground_fraction",
                field=f"adapted_foreground_fraction.{stratum}",
            ),
        )
        for stratum in FOREGROUND_SAFETY_STRATA
    )
    source_fg_dict = dict(source_fg)
    adapted_fg_dict = dict(adapted_fg)
    return CandidateAggregate(
        candidate_id=candidate_id,
        cell_count=len(all_cells),
        nonclean_cell_count=len(nonclean),
        nonclean_macro_directional_derivative=_metric_mean(
            nonclean,
            "normalized_task_directional_derivative",
            field="nonclean_macro_directional_derivative",
        ),
        dataset_nonclean_directional_derivative=dataset_maps(
            "normalized_task_directional_derivative"
        ),
        family_directional_derivative=family_maps(
            "normalized_task_directional_derivative"
        ),
        severity_directional_derivative=severity_maps(
            "normalized_task_directional_derivative"
        ),
        overall_source_iou=_metric_mean(
            all_cells, "source_iou", field="overall_source_iou"
        ),
        overall_adapted_iou=_metric_mean(
            all_cells, "adapted_iou", field="overall_adapted_iou"
        ),
        overall_macro_delta_iou=_delta_iou_mean(
            all_cells, field="overall_delta_iou"
        ),
        nonclean_source_iou=_metric_mean(
            nonclean, "source_iou", field="nonclean_source_iou"
        ),
        nonclean_adapted_iou=_metric_mean(
            nonclean, "adapted_iou", field="nonclean_adapted_iou"
        ),
        nonclean_macro_delta_iou=_delta_iou_mean(
            nonclean, field="nonclean_delta_iou"
        ),
        clean_source_iou=_metric_mean(
            clean, "source_iou", field="clean_source_iou"
        ),
        clean_adapted_iou=_metric_mean(
            clean, "adapted_iou", field="clean_adapted_iou"
        ),
        clean_macro_delta_iou=_delta_iou_mean(clean, field="clean_delta_iou"),
        dataset_nonclean_delta_iou=_ordered_delta_iou_map(
            cells, DATASETS, _by_dataset_nonclean
        ),
        family_nonclean_delta_iou=_ordered_delta_iou_map(
            cells, CORRUPTION_FAMILIES, _by_family
        ),
        severity_nonclean_delta_iou=_ordered_delta_iou_map(
            cells, SEVERITIES, _by_severity
        ),
        clean_dataset_delta_iou=tuple(
            (
                dataset,
                _delta_iou_mean(
                    clean_by_dataset[dataset],
                    field=f"clean_delta_iou.{dataset}",
                ),
            )
            for dataset in DATASETS
        ),
        overall_macro_delta_pd=_metric_mean(
            all_cells, "delta_pd", field="overall_delta_pd"
        ),
        nonclean_macro_delta_pd=_metric_mean(
            nonclean, "delta_pd", field="nonclean_delta_pd"
        ),
        clean_macro_delta_pd=_metric_mean(
            clean, "delta_pd", field="clean_delta_pd"
        ),
        dataset_nonclean_delta_pd=dataset_maps("delta_pd"),
        family_nonclean_delta_pd=family_maps("delta_pd"),
        severity_nonclean_delta_pd=severity_maps("delta_pd"),
        overall_macro_delta_fa=_metric_mean(
            all_cells, "delta_fa", field="overall_delta_fa"
        ),
        nonclean_macro_delta_fa=_metric_mean(
            nonclean, "delta_fa", field="nonclean_delta_fa"
        ),
        clean_macro_delta_fa=_metric_mean(
            clean, "delta_fa", field="clean_delta_fa"
        ),
        dataset_nonclean_delta_fa=dataset_maps("delta_fa"),
        family_nonclean_delta_fa=family_maps("delta_fa"),
        severity_nonclean_delta_fa=severity_maps("delta_fa"),
        source_foreground_fraction=source_fg,
        adapted_foreground_fraction=adapted_fg,
        foreground_fraction_delta=tuple(
            (
                stratum,
                adapted_fg_dict[stratum] - source_fg_dict[stratum],
            )
            for stratum in FOREGROUND_SAFETY_STRATA
        ),
        episode_count=sum(cell.episode_count for cell in cells),
        valid_alignment_episode_count=sum(
            cell.valid_alignment_episode_count for cell in cells
        ),
        functional_changed_episode_count=sum(
            cell.functional_changed_episode_count for cell in cells
        ),
        threshold_crossing_count=sum(
            cell.threshold_crossing_count for cell in cells
        ),
        nonclean_episode_count=sum(cell.episode_count for cell in nonclean),
        nonclean_valid_alignment_episode_count=sum(
            cell.valid_alignment_episode_count for cell in nonclean
        ),
        nonclean_functional_changed_episode_count=sum(
            cell.functional_changed_episode_count for cell in nonclean
        ),
        nonclean_threshold_crossing_count=sum(
            cell.threshold_crossing_count for cell in nonclean
        ),
        severity_valid_alignment_episode_count=tuple(
            (
                severity,
                sum(
                    cell.valid_alignment_episode_count
                    for cell in _by_severity(cells, severity)
                ),
            )
            for severity in SEVERITIES
        ),
        severity_functional_changed_episode_count=tuple(
            (
                severity,
                sum(
                    cell.functional_changed_episode_count
                    for cell in _by_severity(cells, severity)
                ),
            )
            for severity in SEVERITIES
        ),
        severity_threshold_crossing_count=tuple(
            (
                severity,
                sum(
                    cell.threshold_crossing_count
                    for cell in _by_severity(cells, severity)
                ),
            )
            for severity in SEVERITIES
        ),
    )


def _strictly_below(value: Fraction, maximum: Fraction, tolerance: Fraction) -> bool:
    return value < maximum - tolerance


def _strictly_above(value: Fraction, minimum: Fraction, tolerance: Fraction) -> bool:
    return value > minimum + tolerance


def _below_inclusive(value: Fraction, minimum: Fraction, tolerance: Fraction) -> bool:
    return value < minimum - tolerance


def _above_inclusive(value: Fraction, maximum: Fraction, tolerance: Fraction) -> bool:
    return value > maximum + tolerance


def _failed_gates(
    aggregate: CandidateAggregate, gate: GateConfig
) -> tuple[str, ...]:
    failures: list[str] = []
    tolerance = gate.comparison_tolerance
    derivative_datasets = dict(aggregate.dataset_nonclean_directional_derivative)
    derivative_families = dict(aggregate.family_directional_derivative)
    derivative_severities = dict(aggregate.severity_directional_derivative)
    iou_severities = dict(aggregate.severity_nonclean_delta_iou)

    if (
        aggregate.nonclean_valid_alignment_episode_count
        != aggregate.nonclean_episode_count
    ):
        failures.append("nonclean_alignment_evidence_incomplete")
    if not _strictly_below(
        aggregate.nonclean_macro_directional_derivative,
        gate.normalized_task_directional_derivative_maximum,
        tolerance,
    ):
        failures.append("macro_directional_derivative_not_negative")
    if sum(
        _strictly_below(
            value,
            gate.normalized_task_directional_derivative_maximum,
            tolerance,
        )
        for value in derivative_datasets.values()
    ) < gate.minimum_negative_derivative_datasets:
        failures.append("negative_directional_derivative_dataset_coverage")
    if sum(
        _strictly_below(
            value,
            gate.normalized_task_directional_derivative_maximum,
            tolerance,
        )
        for value in derivative_families.values()
    ) < gate.minimum_negative_derivative_families:
        failures.append("negative_directional_derivative_family_coverage")
    if not _strictly_above(
        aggregate.nonclean_macro_delta_iou,
        gate.nonclean_macro_delta_iou_strictly_greater_than,
        tolerance,
    ):
        failures.append("nonclean_macro_delta_iou_not_positive")
    if _below_inclusive(
        aggregate.clean_macro_delta_iou,
        gate.clean_macro_delta_iou_minimum,
        tolerance,
    ):
        failures.append("clean_macro_delta_iou_safety")
    source_fg = dict(aggregate.source_foreground_fraction)
    adapted_fg = dict(aggregate.adapted_foreground_fraction)
    foreground_delta = dict(aggregate.foreground_fraction_delta)
    for stratum in FOREGROUND_SAFETY_STRATA:
        if _above_inclusive(
            foreground_delta[stratum],
            gate.foreground_fraction_delta_maximum,
            tolerance,
        ):
            failures.append(f"foreground_absolute_inflation:{stratum}")
        ratio_maximum = (
            gate.foreground_fraction_source_multiplier * source_fg[stratum]
            + gate.foreground_fraction_epsilon
        )
        if _above_inclusive(adapted_fg[stratum], ratio_maximum, tolerance):
            failures.append(f"foreground_ratio_inflation:{stratum}")

    if (
        aggregate.nonclean_functional_changed_episode_count
        < gate.minimum_functional_changed_episodes
    ):
        failures.append("functional_change_nonzero")
    if (
        aggregate.nonclean_threshold_crossing_count
        < gate.minimum_threshold_crossing_count
    ):
        failures.append("threshold_change_nonzero")
    if sum(
        _strictly_below(
            value,
            gate.normalized_task_directional_derivative_maximum,
            tolerance,
        )
        for value in derivative_severities.values()
    ) < gate.minimum_negative_derivative_severities:
        failures.append("negative_directional_derivative_severity_coverage")
    if sum(
        _strictly_above(
            value,
            gate.nonclean_macro_delta_iou_strictly_greater_than,
            tolerance,
        )
        for value in iou_severities.values()
    ) < gate.minimum_positive_delta_iou_severities:
        failures.append("positive_delta_iou_severity_coverage")
    if aggregate.candidate_id.startswith("O0_"):
        failures.append("negative_control_not_selectable")
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
    nonclean_macro_directional_derivative: Fraction
    minimum_dataset_nonclean_delta_iou: Fraction
    clean_macro_delta_iou: Fraction
    nonclean_foreground_fraction_delta: Fraction
    nonclean_macro_delta_fa: Fraction
    nonclean_macro_delta_pd: Fraction
    nonclean_functional_changed_episode_count: int
    nonclean_threshold_crossing_count: int

    def to_receipt(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "nonclean_macro_delta_iou": _fraction_receipt(
                self.nonclean_macro_delta_iou
            ),
            "nonclean_macro_directional_derivative": _fraction_receipt(
                self.nonclean_macro_directional_derivative
            ),
            "minimum_dataset_nonclean_delta_iou": _fraction_receipt(
                self.minimum_dataset_nonclean_delta_iou
            ),
            "clean_macro_delta_iou": _fraction_receipt(
                self.clean_macro_delta_iou
            ),
            "nonclean_foreground_fraction_delta": _fraction_receipt(
                self.nonclean_foreground_fraction_delta
            ),
            "nonclean_macro_delta_fa": _fraction_receipt(
                self.nonclean_macro_delta_fa
            ),
            "nonclean_macro_delta_pd": _fraction_receipt(
                self.nonclean_macro_delta_pd
            ),
            "nonclean_functional_changed_episode_count": (
                self.nonclean_functional_changed_episode_count
            ),
            "nonclean_threshold_crossing_count": (
                self.nonclean_threshold_crossing_count
            ),
        }


@dataclass(frozen=True, slots=True)
class StageB3ScienceDecision:
    gate: GateConfig
    candidate_evaluations: tuple[CandidateEvaluation, ...]
    eligible_candidate_ids: tuple[str, ...]
    ranking: tuple[CandidateRankingEntry, ...]
    scientific_status: Literal["scientific_passed", "scientific_no_eligible"]
    result_tier: Literal["development"] = "development"
    development_only: Literal[True] = True
    paper_result: Literal[False] = False
    test: Literal[False] = False

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(entry.candidate_id for entry in self.ranking)

    @property
    def stage_b4_allowed(self) -> bool:
        return bool(self.ranking)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "receipt_type": "cr_sitta_p3_stage_b3_science_decision_v1",
            "protocol_status": "passed",
            "scientific_status": self.scientific_status,
            "result_tier": self.result_tier,
            "development_only": self.development_only,
            "paper_result": self.paper_result,
            "test": self.test,
            "stage_b4_allowed": self.stage_b4_allowed,
            "gate": self.gate.to_receipt(),
            "topology": {
                "datasets": list(DATASETS),
                "conditions": list(CONDITIONS),
                "corruption_families": list(CORRUPTION_FAMILIES),
                "severities": list(SEVERITIES),
                "cells_per_candidate": CELLS_PER_CANDIDATE,
                "nonclean_cells_per_candidate": NONCLEAN_CELLS_PER_CANDIDATE,
                "episodes_per_cell": EPISODES_PER_CELL,
                "episodes_per_candidate": EPISODES_PER_CANDIDATE,
                "candidate_count": len(self.candidate_evaluations),
            },
            "aggregation_policy": {
                "macro_weighting": "equal_cell",
                "directional_derivative_scope": "nonclean",
                "dataset_directional_derivative_scope": "nonclean",
                "family_and_severity_scope": "nonclean",
                "actual_delta_iou": "adapted_iou_minus_source_iou",
                "zero_valid_alignment_policy": (
                    "exact_zero_is_recordable_but_not_eligible_nonclean_evidence"
                ),
                "missing_value_policy": "forbidden_no_imputation",
                "nonclean_alignment_completeness_required": True,
                "incomplete_nonclean_alignment_action": (
                    "candidate_science_ineligible_other_candidates_continue"
                ),
            },
            "selection_policy": {
                "filter_before_ranking": True,
                "ranking_order": [
                    "nonclean_macro_delta_iou_desc",
                    "nonclean_macro_directional_derivative_asc",
                    "minimum_dataset_nonclean_delta_iou_desc",
                    "clean_macro_delta_iou_desc",
                    "nonclean_foreground_fraction_delta_asc",
                    "nonclean_macro_delta_fa_asc",
                    "nonclean_macro_delta_pd_desc",
                    "nonclean_functional_changed_episode_count_desc",
                    "nonclean_threshold_crossing_count_desc",
                    "candidate_id_lexical_asc",
                ],
                "maximum_selected_candidates": self.gate.maximum_selected_candidates,
            },
            "candidate_evaluations": [
                evaluation.to_receipt()
                for evaluation in self.candidate_evaluations
            ],
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "selected_for_stage_b4": list(self.selected_candidate_ids),
            "ranking": [entry.to_receipt() for entry in self.ranking],
        }


def _ranking_key(evaluation: CandidateEvaluation) -> tuple[Any, ...]:
    aggregate = evaluation.aggregate
    nonclean_foreground_delta = dict(
        aggregate.foreground_fraction_delta
    )["nonclean"]
    return (
        -aggregate.nonclean_macro_delta_iou,
        aggregate.nonclean_macro_directional_derivative,
        -min(dict(aggregate.dataset_nonclean_delta_iou).values()),
        -aggregate.clean_macro_delta_iou,
        nonclean_foreground_delta,
        aggregate.nonclean_macro_delta_fa,
        -aggregate.nonclean_macro_delta_pd,
        -aggregate.nonclean_functional_changed_episode_count,
        -aggregate.nonclean_threshold_crossing_count,
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
        nonclean_macro_directional_derivative=(
            aggregate.nonclean_macro_directional_derivative
        ),
        minimum_dataset_nonclean_delta_iou=min(
            dict(aggregate.dataset_nonclean_delta_iou).values()
        ),
        clean_macro_delta_iou=aggregate.clean_macro_delta_iou,
        nonclean_foreground_fraction_delta=dict(
            aggregate.foreground_fraction_delta
        )["nonclean"],
        nonclean_macro_delta_fa=aggregate.nonclean_macro_delta_fa,
        nonclean_macro_delta_pd=aggregate.nonclean_macro_delta_pd,
        nonclean_functional_changed_episode_count=(
            aggregate.nonclean_functional_changed_episode_count
        ),
        nonclean_threshold_crossing_count=(
            aggregate.nonclean_threshold_crossing_count
        ),
    )


def evaluate_candidates(
    cell_summaries: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    gate_config: Mapping[str, Any],
) -> StageB3ScienceDecision:
    """Validate, aggregate, hard-filter, then rank a complete Stage-B3 screen."""

    gate = GateConfig.from_mapping(gate_config)
    candidates = _sequence(candidate_ids, field="candidate_ids")
    if not candidates:
        raise StageB3ScienceGateError("candidate_ids cannot be empty")
    if any(not isinstance(candidate, str) or not candidate for candidate in candidates):
        raise StageB3ScienceGateError(
            "candidate_ids must contain non-empty strings"
        )
    if len(set(candidates)) != len(candidates):
        raise StageB3ScienceGateError("candidate_ids must be unique")

    records = _sequence(cell_summaries, field="cell_summaries")
    expected_record_count = len(candidates) * CELLS_PER_CANDIDATE
    if len(records) != expected_record_count:
        raise StageB3ScienceGateError(
            "cell summary count must equal candidate_count*39; "
            f"expected={expected_record_count}, observed={len(records)}"
        )
    parsed = tuple(_parse_cell(record, index=index) for index, record in enumerate(records))
    candidate_set = set(candidates)
    unknown = sorted({cell.candidate_id for cell in parsed} - candidate_set)
    if unknown:
        raise StageB3ScienceGateError(
            f"cell summaries contain candidates outside frozen roster: {unknown}"
        )

    expected_cells = {
        (dataset, condition) for dataset in DATASETS for condition in CONDITIONS
    }
    by_candidate: dict[str, dict[tuple[str, str], _CellSummary]] = {
        candidate: {} for candidate in candidates
    }
    for cell in parsed:
        key = (cell.dataset, cell.condition)
        if key in by_candidate[cell.candidate_id]:
            raise StageB3ScienceGateError(
                "duplicate candidate/dataset/condition cell: "
                f"{cell.candidate_id}/{cell.dataset}/{cell.condition}"
            )
        by_candidate[cell.candidate_id][key] = cell
    for candidate in candidates:
        observed = set(by_candidate[candidate])
        if observed != expected_cells:
            raise StageB3ScienceGateError(
                f"candidate {candidate} does not have exact 3x13 topology; "
                f"missing={sorted(expected_cells - observed)}, "
                f"extra={sorted(observed - expected_cells)}"
            )

    # Source predictions must be the same frozen endpoint for every candidate.
    source_reference: dict[tuple[str, str], tuple[Fraction, Fraction]] = {}
    for candidate in candidates:
        for key in sorted(expected_cells):
            cell = by_candidate[candidate][key]
            endpoint = (cell.source_iou, cell.source_foreground_fraction)
            if key not in source_reference:
                source_reference[key] = endpoint
            elif source_reference[key] != endpoint:
                raise StageB3ScienceGateError(
                    "Source endpoint differs across candidates for "
                    f"{key[0]}/{key[1]}"
                )

    evaluations: list[CandidateEvaluation] = []
    for candidate in candidates:
        ordered_cells = tuple(
            by_candidate[candidate][(dataset, condition)]
            for dataset in DATASETS
            for condition in CONDITIONS
        )
        aggregate = _aggregate_candidate(candidate, ordered_cells)
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
    selected = eligible[: gate.maximum_selected_candidates]
    ranking = tuple(
        _ranking_entry(rank, evaluation)
        for rank, evaluation in enumerate(selected, start=1)
    )
    return StageB3ScienceDecision(
        gate=gate,
        candidate_evaluations=tuple(evaluations),
        eligible_candidate_ids=tuple(
            evaluation.candidate_id for evaluation in eligible
        ),
        ranking=ranking,
        scientific_status=(
            "scientific_passed" if selected else "scientific_no_eligible"
        ),
    )


def evaluate_stage_b3_science_gate(
    cell_summaries: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    gate_config: Mapping[str, Any],
) -> StageB3ScienceDecision:
    """Descriptive alias for :func:`evaluate_candidates`."""

    return evaluate_candidates(cell_summaries, candidate_ids, gate_config)


evaluate_proxy_objective_space_screen = evaluate_stage_b3_science_gate


__all__ = [
    "CELL_FIELDS",
    "CELLS_PER_CANDIDATE",
    "CONDITIONS",
    "CORRUPTION_FAMILIES",
    "CandidateAggregate",
    "CandidateEvaluation",
    "CandidateRankingEntry",
    "DATASETS",
    "EPISODES_PER_CANDIDATE",
    "EPISODES_PER_CELL",
    "FOREGROUND_SAFETY_STRATA",
    "GATE_CONFIG_FIELDS",
    "GateConfig",
    "NONCLEAN_CELLS_PER_CANDIDATE",
    "ProxyObjectiveSpaceScreenError",
    "SEVERITIES",
    "StageB3ScienceDecision",
    "StageB3ScienceGateError",
    "evaluate_candidates",
    "evaluate_proxy_objective_space_screen",
    "evaluate_stage_b3_science_gate",
]
