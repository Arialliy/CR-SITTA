"""Pure P4 non-adaptive-teacher aggregation and scientific gate.

The module deliberately has no filesystem, torch, CUDA, model, or dataset
dependencies.  It consumes one complete 3-dataset x 13-condition cell grid
per caller-declared candidate.  Cell endpoints may contain either integer
sufficient statistics or already-computed metrics, but one evaluation may
not mix the two evidence kinds.

Protocol/schema failures raise :class:`NonadaptiveTeacherGateError`; they are
never converted into a negative scientific result.  A valid grid with no
eligible candidate returns ``scientific_no_eligible``.  Even a positive P4
screen is development evidence only and never authorizes P5 by itself.
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
    *(f"{family}_{severity}" for family in CORRUPTION_FAMILIES for severity in SEVERITIES),
)
CELLS_PER_CANDIDATE = len(DATASETS) * len(CONDITIONS)
NONCLEAN_CELLS_PER_CANDIDATE = len(DATASETS) * (
    len(CONDITIONS) - 1
)
EXPECTED_CANDIDATE_COUNT = 10
IMAGES_PER_CELL = 64
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIXELS_PER_CELL = IMAGES_PER_CELL * IMAGE_HEIGHT * IMAGE_WIDTH
SAFETY_STRATA = (
    "overall",
    "nonclean",
    "clean",
    *(f"dataset:{dataset}" for dataset in DATASETS),
    *(f"corruption_family:{family}" for family in CORRUPTION_FAMILIES),
)

COUNT_FIELD_ORDER = (
    "image_count",
    "intersection_pixels",
    "false_positive_pixels",
    "false_negative_pixels",
    "true_negative_pixels",
    "union_pixels",
    "predicted_positive_pixels",
    "target_positive_pixels",
    "total_image_pixels",
    "detected_targets",
    "total_targets",
    "false_alarm_pixels",
)
COUNT_FIELDS = frozenset(COUNT_FIELD_ORDER)
METRIC_FIELD_ORDER = (
    "global_iou",
    "pd",
    "fa_per_million",
    "foreground_fraction",
)
METRIC_FIELDS = frozenset(METRIC_FIELD_ORDER)


class NonadaptiveTeacherGateError(ValueError):
    """The candidate roster, cell topology, or evidence is invalid."""


def _fraction(value: Any, *, field: str) -> Fraction:
    """Convert a finite JSON-like number without binary-float arithmetic."""

    if isinstance(value, bool):
        raise NonadaptiveTeacherGateError(f"{field} must be a finite number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise NonadaptiveTeacherGateError(f"{field} must be finite")
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonadaptiveTeacherGateError(f"{field} must be finite")
        value = str(value)
    if isinstance(value, str):
        try:
            decimal = Decimal(value)
        except InvalidOperation as exc:
            raise NonadaptiveTeacherGateError(
                f"{field} must be a finite number"
            ) from exc
        if not decimal.is_finite():
            raise NonadaptiveTeacherGateError(f"{field} must be finite")
        return Fraction(decimal)
    raise NonadaptiveTeacherGateError(f"{field} must be a finite number")


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NonadaptiveTeacherGateError(
            f"{field} must be a non-negative integer"
        )
    return value


def _sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise NonadaptiveTeacherGateError(f"{field} must be a sequence")
    return tuple(value)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NonadaptiveTeacherGateError(f"{field} must be a mapping")
    return value


def _mean(values: Sequence[Fraction], *, field: str) -> Fraction:
    if not values:
        raise NonadaptiveTeacherGateError(f"{field} cannot be empty")
    return sum(values, Fraction(0, 1)) / len(values)


def _fraction_receipt(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_map_receipt(
    values: tuple[tuple[str, Fraction], ...],
) -> dict[str, dict[str, int]]:
    return {key: _fraction_receipt(value) for key, value in values}


@dataclass(frozen=True)
class GateConfig:
    """Caller-frozen P4 thresholds.

    Strict gates are used for non-clean and overall macro IoU.  Safety minima
    and maxima are inclusive.  Values are normalized to exact Fractions.
    """

    comparison_tolerance: Fraction
    nonclean_macro_delta_iou_epsilon: Fraction
    overall_macro_delta_iou_threshold: Fraction
    minimum_positive_nonclean_datasets: int
    worst_nonclean_dataset_delta_iou_minimum: Fraction
    clean_macro_delta_iou_minimum: Fraction
    each_clean_dataset_delta_iou_minimum: Fraction
    nonclean_macro_delta_pd_minimum: Fraction
    each_nonclean_dataset_delta_pd_minimum: Fraction
    fa_absolute_allowance_per_million: Fraction
    fa_source_multiplier: Fraction
    foreground_fraction_delta_maximum: Fraction
    foreground_fraction_source_multiplier: Fraction
    foreground_fraction_epsilon: Fraction
    require_integer_counts: bool = True

    def __post_init__(self) -> None:
        fraction_fields = (
            "comparison_tolerance",
            "nonclean_macro_delta_iou_epsilon",
            "overall_macro_delta_iou_threshold",
            "worst_nonclean_dataset_delta_iou_minimum",
            "clean_macro_delta_iou_minimum",
            "each_clean_dataset_delta_iou_minimum",
            "nonclean_macro_delta_pd_minimum",
            "each_nonclean_dataset_delta_pd_minimum",
            "fa_absolute_allowance_per_million",
            "fa_source_multiplier",
            "foreground_fraction_delta_maximum",
            "foreground_fraction_source_multiplier",
            "foreground_fraction_epsilon",
        )
        for name in fraction_fields:
            object.__setattr__(
                self,
                name,
                _fraction(getattr(self, name), field=f"gate.{name}"),
            )
        if (
            isinstance(self.minimum_positive_nonclean_datasets, bool)
            or not isinstance(self.minimum_positive_nonclean_datasets, int)
            or not 1
            <= self.minimum_positive_nonclean_datasets
            <= len(DATASETS)
        ):
            raise NonadaptiveTeacherGateError(
                "gate.minimum_positive_nonclean_datasets must be in [1, 3]"
            )
        if type(self.require_integer_counts) is not bool:
            raise NonadaptiveTeacherGateError(
                "gate.require_integer_counts must be a boolean"
            )
        if self.comparison_tolerance < 0:
            raise NonadaptiveTeacherGateError(
                "gate.comparison_tolerance must be non-negative"
            )
        if self.nonclean_macro_delta_iou_epsilon < 0:
            raise NonadaptiveTeacherGateError(
                "gate.nonclean_macro_delta_iou_epsilon must be non-negative"
            )
        if self.overall_macro_delta_iou_threshold < 0:
            raise NonadaptiveTeacherGateError(
                "gate.overall_macro_delta_iou_threshold must be non-negative"
            )
        bounded_delta_fields = (
            "worst_nonclean_dataset_delta_iou_minimum",
            "clean_macro_delta_iou_minimum",
            "each_clean_dataset_delta_iou_minimum",
            "nonclean_macro_delta_pd_minimum",
            "each_nonclean_dataset_delta_pd_minimum",
            "foreground_fraction_delta_maximum",
        )
        for name in bounded_delta_fields:
            value = getattr(self, name)
            if not -1 <= value <= 1:
                raise NonadaptiveTeacherGateError(
                    f"gate.{name} must be in [-1, 1]"
                )
        nonnegative_fields = (
            "fa_absolute_allowance_per_million",
            "fa_source_multiplier",
            "foreground_fraction_delta_maximum",
            "foreground_fraction_source_multiplier",
            "foreground_fraction_epsilon",
        )
        for name in nonnegative_fields:
            if getattr(self, name) < 0:
                raise NonadaptiveTeacherGateError(
                    f"gate.{name} must be non-negative"
                )

    def to_receipt(self) -> dict[str, Any]:
        return {
            name: _fraction_receipt(getattr(self, name))
            for name in (
                "comparison_tolerance",
                "nonclean_macro_delta_iou_epsilon",
                "overall_macro_delta_iou_threshold",
                "worst_nonclean_dataset_delta_iou_minimum",
                "clean_macro_delta_iou_minimum",
                "each_clean_dataset_delta_iou_minimum",
                "nonclean_macro_delta_pd_minimum",
                "each_nonclean_dataset_delta_pd_minimum",
                "fa_absolute_allowance_per_million",
                "fa_source_multiplier",
                "foreground_fraction_delta_maximum",
                "foreground_fraction_source_multiplier",
                "foreground_fraction_epsilon",
            )
        } | {
            "minimum_positive_nonclean_datasets": (
                self.minimum_positive_nonclean_datasets
            ),
            "require_integer_counts": self.require_integer_counts,
        }


@dataclass(frozen=True)
class SufficientCounts:
    image_count: int
    intersection_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    true_negative_pixels: int
    union_pixels: int
    predicted_positive_pixels: int
    target_positive_pixels: int
    total_image_pixels: int
    detected_targets: int
    total_targets: int
    false_alarm_pixels: int

    def __post_init__(self) -> None:
        for name in COUNT_FIELD_ORDER:
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), field=f"counts.{name}"),
            )
        if self.total_image_pixels <= 0:
            raise NonadaptiveTeacherGateError(
                "counts.total_image_pixels must be positive"
            )
        if self.image_count != IMAGES_PER_CELL:
            raise NonadaptiveTeacherGateError(
                f"counts.image_count must equal {IMAGES_PER_CELL}"
            )
        if self.total_image_pixels != PIXELS_PER_CELL:
            raise NonadaptiveTeacherGateError(
                "counts.total_image_pixels must equal "
                f"{IMAGES_PER_CELL}*{IMAGE_HEIGHT}*{IMAGE_WIDTH}={PIXELS_PER_CELL}"
            )
        if self.predicted_positive_pixels != (
            self.intersection_pixels + self.false_positive_pixels
        ):
            raise NonadaptiveTeacherGateError(
                "counts predicted-positive conservation failed"
            )
        if self.target_positive_pixels != (
            self.intersection_pixels + self.false_negative_pixels
        ):
            raise NonadaptiveTeacherGateError(
                "counts target-positive conservation failed"
            )
        if self.union_pixels != (
            self.intersection_pixels
            + self.false_positive_pixels
            + self.false_negative_pixels
        ):
            raise NonadaptiveTeacherGateError("counts union conservation failed")
        if self.total_image_pixels != (
            self.intersection_pixels
            + self.false_positive_pixels
            + self.false_negative_pixels
            + self.true_negative_pixels
        ):
            raise NonadaptiveTeacherGateError(
                "counts total-pixel conservation failed"
            )
        if self.detected_targets > self.total_targets:
            raise NonadaptiveTeacherGateError(
                "counts detected_targets exceeds total_targets"
            )
        if self.false_alarm_pixels > self.predicted_positive_pixels:
            raise NonadaptiveTeacherGateError(
                "counts false_alarm_pixels exceeds predicted_positive_pixels"
            )

    @classmethod
    def from_mapping(cls, value: Any, *, field: str) -> "SufficientCounts":
        mapping = _mapping(value, field=field)
        missing = sorted(COUNT_FIELDS - set(mapping))
        if missing:
            raise NonadaptiveTeacherGateError(
                f"{field} is missing sufficient-count fields: {missing}"
            )
        try:
            return cls(**{name: mapping[name] for name in COUNT_FIELD_ORDER})
        except NonadaptiveTeacherGateError as exc:
            raise NonadaptiveTeacherGateError(f"{field}: {exc}") from exc

    def metrics(self) -> "EndpointMetrics":
        return EndpointMetrics(
            global_iou=(
                Fraction(self.intersection_pixels, self.union_pixels)
                if self.union_pixels
                else Fraction(1, 1)
            ),
            pd=(
                Fraction(self.detected_targets, self.total_targets)
                if self.total_targets
                else Fraction(0, 1)
            ),
            fa_per_million=Fraction(
                self.false_alarm_pixels * 1_000_000,
                self.total_image_pixels,
            ),
            foreground_fraction=Fraction(
                self.predicted_positive_pixels,
                self.total_image_pixels,
            ),
        )


@dataclass(frozen=True)
class EndpointMetrics:
    global_iou: Fraction
    pd: Fraction
    fa_per_million: Fraction
    foreground_fraction: Fraction

    def __post_init__(self) -> None:
        for name in METRIC_FIELD_ORDER:
            object.__setattr__(
                self,
                name,
                _fraction(getattr(self, name), field=f"metrics.{name}"),
            )
        if not 0 <= self.global_iou <= 1:
            raise NonadaptiveTeacherGateError(
                "metrics.global_iou must be in [0, 1]"
            )
        if not 0 <= self.pd <= 1:
            raise NonadaptiveTeacherGateError("metrics.pd must be in [0, 1]")
        if self.fa_per_million < 0:
            raise NonadaptiveTeacherGateError(
                "metrics.fa_per_million must be non-negative"
            )
        if not 0 <= self.foreground_fraction <= 1:
            raise NonadaptiveTeacherGateError(
                "metrics.foreground_fraction must be in [0, 1]"
            )

    @classmethod
    def from_mapping(cls, value: Any, *, field: str) -> "EndpointMetrics":
        mapping = _mapping(value, field=field)
        iou_keys = tuple(key for key in ("global_iou", "iou") if key in mapping)
        if len(iou_keys) != 1:
            raise NonadaptiveTeacherGateError(
                f"{field} must contain exactly one of global_iou or iou"
            )
        required = {"pd", "fa_per_million", "foreground_fraction"}
        missing = sorted(required - set(mapping))
        if missing:
            raise NonadaptiveTeacherGateError(
                f"{field} is missing metric fields: {missing}"
            )
        try:
            return cls(
                global_iou=mapping[iou_keys[0]],
                pd=mapping["pd"],
                fa_per_million=mapping["fa_per_million"],
                foreground_fraction=mapping["foreground_fraction"],
            )
        except NonadaptiveTeacherGateError as exc:
            raise NonadaptiveTeacherGateError(f"{field}: {exc}") from exc


@dataclass(frozen=True)
class _ParsedCell:
    candidate_id: str
    dataset: str
    condition: str
    evidence_kind: Literal["counts", "metrics"]
    source_raw: SufficientCounts | EndpointMetrics
    teacher_raw: SufficientCounts | EndpointMetrics
    source: EndpointMetrics
    teacher: EndpointMetrics


def _parse_endpoint(
    value: Any, *, field: str
) -> tuple[Literal["counts", "metrics"], SufficientCounts | EndpointMetrics, EndpointMetrics]:
    mapping = _mapping(value, field=field)
    keys = set(mapping)
    has_any_count = bool(keys & COUNT_FIELDS)
    has_all_counts = COUNT_FIELDS <= keys
    metric_core = {"pd", "fa_per_million", "foreground_fraction"}
    has_iou = "global_iou" in mapping or "iou" in mapping
    has_any_metric = bool(keys & metric_core) or has_iou
    has_all_metrics = metric_core <= keys and has_iou

    if has_any_count and not has_all_counts:
        missing = sorted(COUNT_FIELDS - keys)
        raise NonadaptiveTeacherGateError(
            f"{field} has a partial sufficient-count schema; missing={missing}"
        )
    if has_all_counts:
        counts = SufficientCounts.from_mapping(mapping, field=field)
        derived = counts.metrics()
        if has_any_metric:
            if not has_all_metrics:
                raise NonadaptiveTeacherGateError(
                    f"{field} has a partial metric schema beside counts"
                )
            supplied = EndpointMetrics.from_mapping(mapping, field=field)
            if supplied != derived:
                raise NonadaptiveTeacherGateError(
                    f"{field} supplied metrics disagree with sufficient counts"
                )
        return "counts", counts, derived
    if has_all_metrics:
        metrics = EndpointMetrics.from_mapping(mapping, field=field)
        return "metrics", metrics, metrics
    if has_any_metric:
        raise NonadaptiveTeacherGateError(f"{field} has a partial metric schema")
    raise NonadaptiveTeacherGateError(
        f"{field} is neither sufficient counts nor computed metrics"
    )


def _parse_cell(value: Any, *, index: int) -> _ParsedCell:
    field = f"cell_records[{index}]"
    record = _mapping(value, field=field)
    for required in ("candidate_id", "dataset", "condition", "source"):
        if required not in record:
            raise NonadaptiveTeacherGateError(f"{field} is missing {required}")
    teacher_keys = tuple(key for key in ("teacher", "adapted") if key in record)
    if len(teacher_keys) != 1:
        raise NonadaptiveTeacherGateError(
            f"{field} must contain exactly one of teacher or adapted"
        )
    candidate_id = record["candidate_id"]
    dataset = record["dataset"]
    condition = record["condition"]
    if not isinstance(candidate_id, str) or not candidate_id:
        raise NonadaptiveTeacherGateError(
            f"{field}.candidate_id must be a non-empty string"
        )
    if dataset not in DATASETS:
        raise NonadaptiveTeacherGateError(
            f"{field}.dataset is not one of the frozen three datasets"
        )
    if condition not in CONDITIONS:
        raise NonadaptiveTeacherGateError(
            f"{field}.condition is not one of the frozen thirteen conditions"
        )
    source_kind, source_raw, source = _parse_endpoint(
        record["source"], field=f"{field}.source"
    )
    teacher_kind, teacher_raw, teacher = _parse_endpoint(
        record[teacher_keys[0]], field=f"{field}.{teacher_keys[0]}"
    )
    if source_kind != teacher_kind:
        raise NonadaptiveTeacherGateError(
            f"{field} cannot mix count and metric endpoints"
        )
    if source_kind == "counts":
        assert isinstance(source_raw, SufficientCounts)
        assert isinstance(teacher_raw, SufficientCounts)
        if (
            source_raw.target_positive_pixels
            != teacher_raw.target_positive_pixels
            or source_raw.total_image_pixels != teacher_raw.total_image_pixels
            or source_raw.total_targets != teacher_raw.total_targets
        ):
            raise NonadaptiveTeacherGateError(
                f"{field} Source/teacher ground-truth denominators differ"
            )
    return _ParsedCell(
        candidate_id=candidate_id,
        dataset=dataset,
        condition=condition,
        evidence_kind=source_kind,
        source_raw=source_raw,
        teacher_raw=teacher_raw,
        source=source,
        teacher=teacher,
    )


def _family(condition: str) -> str | None:
    if condition == "clean_S0":
        return None
    matches = tuple(
        family
        for family in CORRUPTION_FAMILIES
        if condition.startswith(f"{family}_S")
    )
    if len(matches) != 1:  # CONDITIONS validation should make this unreachable.
        raise NonadaptiveTeacherGateError(
            f"condition has no unique corruption family: {condition}"
        )
    return matches[0]


def _select_cells(cells: Sequence[_ParsedCell], stratum: str) -> tuple[_ParsedCell, ...]:
    selected: list[_ParsedCell] = []
    for cell in cells:
        family = _family(cell.condition)
        include = False
        if stratum == "overall":
            include = True
        elif stratum == "nonclean":
            include = family is not None
        elif stratum == "clean":
            include = family is None
        elif stratum.startswith("dataset:"):
            # YAML freezes this stratum as ``each_dataset`` (not
            # ``each_nonclean_dataset``), so it covers all thirteen cells for
            # the selected dataset.  Clean IoU still has its own additional
            # per-dataset guard.
            include = cell.dataset == stratum.partition(":")[2]
        elif stratum.startswith("corruption_family:"):
            include = family == stratum.partition(":")[2]
        else:
            raise NonadaptiveTeacherGateError(f"unknown safety stratum {stratum}")
        if include:
            selected.append(cell)
    if not selected:
        raise NonadaptiveTeacherGateError(f"empty safety stratum {stratum}")
    return tuple(selected)


def _metric_mean(
    cells: Sequence[_ParsedCell], endpoint: Literal["source", "teacher"], metric: str
) -> Fraction:
    return _mean(
        [getattr(getattr(cell, endpoint), metric) for cell in cells],
        field=f"{endpoint}.{metric}",
    )


def _metric_delta(cells: Sequence[_ParsedCell], metric: str) -> Fraction:
    return _mean(
        [getattr(cell.teacher, metric) - getattr(cell.source, metric) for cell in cells],
        field=f"delta.{metric}",
    )


@dataclass(frozen=True)
class CandidateAggregate:
    candidate_id: str
    cell_count: int
    nonclean_cell_count: int
    overall_macro_delta_iou: Fraction
    nonclean_macro_delta_iou: Fraction
    dataset_nonclean_delta_iou: tuple[tuple[str, Fraction], ...]
    family_nonclean_delta_iou: tuple[tuple[str, Fraction], ...]
    clean_macro_delta_iou: Fraction
    clean_dataset_delta_iou: tuple[tuple[str, Fraction], ...]
    overall_macro_delta_pd: Fraction
    nonclean_macro_delta_pd: Fraction
    dataset_nonclean_delta_pd: tuple[tuple[str, Fraction], ...]
    family_nonclean_delta_pd: tuple[tuple[str, Fraction], ...]
    clean_macro_delta_pd: Fraction
    source_fa_per_million: tuple[tuple[str, Fraction], ...]
    teacher_fa_per_million: tuple[tuple[str, Fraction], ...]
    fa_delta_per_million: tuple[tuple[str, Fraction], ...]
    source_foreground_fraction: tuple[tuple[str, Fraction], ...]
    teacher_foreground_fraction: tuple[tuple[str, Fraction], ...]
    foreground_fraction_delta: tuple[tuple[str, Fraction], ...]

    def to_receipt(self) -> dict[str, Any]:
        scalar_names = (
            "overall_macro_delta_iou",
            "nonclean_macro_delta_iou",
            "clean_macro_delta_iou",
            "overall_macro_delta_pd",
            "nonclean_macro_delta_pd",
            "clean_macro_delta_pd",
        )
        map_names = (
            "dataset_nonclean_delta_iou",
            "family_nonclean_delta_iou",
            "clean_dataset_delta_iou",
            "dataset_nonclean_delta_pd",
            "family_nonclean_delta_pd",
            "source_fa_per_million",
            "teacher_fa_per_million",
            "fa_delta_per_million",
            "source_foreground_fraction",
            "teacher_foreground_fraction",
            "foreground_fraction_delta",
        )
        return {
            "candidate_id": self.candidate_id,
            "cell_count": self.cell_count,
            "nonclean_cell_count": self.nonclean_cell_count,
            **{
                name: _fraction_receipt(getattr(self, name))
                for name in scalar_names
            },
            **{
                name: _fraction_map_receipt(getattr(self, name))
                for name in map_names
            },
        }


def _aggregate_candidate(
    candidate_id: str, cells: Sequence[_ParsedCell]
) -> CandidateAggregate:
    overall = _select_cells(cells, "overall")
    nonclean = _select_cells(cells, "nonclean")
    clean = _select_cells(cells, "clean")
    dataset_nonclean = {
        dataset: _select_cells(cells, f"dataset:{dataset}") for dataset in DATASETS
    }
    family_nonclean = {
        family: _select_cells(cells, f"corruption_family:{family}")
        for family in CORRUPTION_FAMILIES
    }
    clean_by_dataset = {
        dataset: tuple(
            cell
            for cell in cells
            if cell.dataset == dataset and cell.condition == "clean_S0"
        )
        for dataset in DATASETS
    }
    source_fa: list[tuple[str, Fraction]] = []
    teacher_fa: list[tuple[str, Fraction]] = []
    fa_delta: list[tuple[str, Fraction]] = []
    source_fg: list[tuple[str, Fraction]] = []
    teacher_fg: list[tuple[str, Fraction]] = []
    fg_delta: list[tuple[str, Fraction]] = []
    for stratum in SAFETY_STRATA:
        selected = _select_cells(cells, stratum)
        source_fa_value = _metric_mean(selected, "source", "fa_per_million")
        teacher_fa_value = _metric_mean(selected, "teacher", "fa_per_million")
        source_fg_value = _metric_mean(selected, "source", "foreground_fraction")
        teacher_fg_value = _metric_mean(selected, "teacher", "foreground_fraction")
        source_fa.append((stratum, source_fa_value))
        teacher_fa.append((stratum, teacher_fa_value))
        fa_delta.append((stratum, teacher_fa_value - source_fa_value))
        source_fg.append((stratum, source_fg_value))
        teacher_fg.append((stratum, teacher_fg_value))
        fg_delta.append((stratum, teacher_fg_value - source_fg_value))
    return CandidateAggregate(
        candidate_id=candidate_id,
        cell_count=len(overall),
        nonclean_cell_count=len(nonclean),
        overall_macro_delta_iou=_metric_delta(overall, "global_iou"),
        nonclean_macro_delta_iou=_metric_delta(nonclean, "global_iou"),
        dataset_nonclean_delta_iou=tuple(
            (dataset, _metric_delta(dataset_nonclean[dataset], "global_iou"))
            for dataset in DATASETS
        ),
        family_nonclean_delta_iou=tuple(
            (family, _metric_delta(family_nonclean[family], "global_iou"))
            for family in CORRUPTION_FAMILIES
        ),
        clean_macro_delta_iou=_metric_delta(clean, "global_iou"),
        clean_dataset_delta_iou=tuple(
            (dataset, _metric_delta(clean_by_dataset[dataset], "global_iou"))
            for dataset in DATASETS
        ),
        overall_macro_delta_pd=_metric_delta(overall, "pd"),
        nonclean_macro_delta_pd=_metric_delta(nonclean, "pd"),
        dataset_nonclean_delta_pd=tuple(
            (dataset, _metric_delta(dataset_nonclean[dataset], "pd"))
            for dataset in DATASETS
        ),
        family_nonclean_delta_pd=tuple(
            (family, _metric_delta(family_nonclean[family], "pd"))
            for family in CORRUPTION_FAMILIES
        ),
        clean_macro_delta_pd=_metric_delta(clean, "pd"),
        source_fa_per_million=tuple(source_fa),
        teacher_fa_per_million=tuple(teacher_fa),
        fa_delta_per_million=tuple(fa_delta),
        source_foreground_fraction=tuple(source_fg),
        teacher_foreground_fraction=tuple(teacher_fg),
        foreground_fraction_delta=tuple(fg_delta),
    )


def _as_dict(values: tuple[tuple[str, Fraction], ...]) -> dict[str, Fraction]:
    return dict(values)


def _strictly_greater_than(
    value: Fraction, threshold: Fraction, tolerance: Fraction
) -> bool:
    return value > threshold + tolerance


def _below_minimum(
    value: Fraction, minimum: Fraction, tolerance: Fraction
) -> bool:
    return value < minimum - tolerance


def _above_maximum(
    value: Fraction, maximum: Fraction, tolerance: Fraction
) -> bool:
    return value > maximum + tolerance


def _failed_gates(aggregate: CandidateAggregate, gate: GateConfig) -> tuple[str, ...]:
    failures: list[str] = []
    dataset_iou = _as_dict(aggregate.dataset_nonclean_delta_iou)
    clean_dataset_iou = _as_dict(aggregate.clean_dataset_delta_iou)
    dataset_pd = _as_dict(aggregate.dataset_nonclean_delta_pd)
    tolerance = gate.comparison_tolerance
    if not _strictly_greater_than(
        aggregate.nonclean_macro_delta_iou,
        gate.nonclean_macro_delta_iou_epsilon,
        tolerance,
    ):
        failures.append("nonclean_macro_iou_not_above_epsilon")
    if not _strictly_greater_than(
        aggregate.overall_macro_delta_iou,
        gate.overall_macro_delta_iou_threshold,
        tolerance,
    ):
        failures.append("overall_macro_iou_not_above_threshold")
    if sum(
        _strictly_greater_than(value, Fraction(0, 1), tolerance)
        for value in dataset_iou.values()
    ) < (
        gate.minimum_positive_nonclean_datasets
    ):
        failures.append("positive_nonclean_dataset_coverage")
    if _below_minimum(
        min(dataset_iou.values()),
        gate.worst_nonclean_dataset_delta_iou_minimum,
        tolerance,
    ):
        failures.append("worst_nonclean_dataset_iou")
    if _below_minimum(
        aggregate.clean_macro_delta_iou,
        gate.clean_macro_delta_iou_minimum,
        tolerance,
    ):
        failures.append("clean_macro_iou_safety")
    for dataset in DATASETS:
        if _below_minimum(
            clean_dataset_iou[dataset],
            gate.each_clean_dataset_delta_iou_minimum,
            tolerance,
        ):
            failures.append(f"clean_dataset_iou_safety:{dataset}")
    if _below_minimum(
        aggregate.nonclean_macro_delta_pd,
        gate.nonclean_macro_delta_pd_minimum,
        tolerance,
    ):
        failures.append("nonclean_pd_safety")
    for dataset in DATASETS:
        if _below_minimum(
            dataset_pd[dataset],
            gate.each_nonclean_dataset_delta_pd_minimum,
            tolerance,
        ):
            failures.append(f"dataset_nonclean_pd_safety:{dataset}")

    source_fa = _as_dict(aggregate.source_fa_per_million)
    fa_delta = _as_dict(aggregate.fa_delta_per_million)
    source_fg = _as_dict(aggregate.source_foreground_fraction)
    teacher_fg = _as_dict(aggregate.teacher_foreground_fraction)
    fg_delta = _as_dict(aggregate.foreground_fraction_delta)
    for stratum in SAFETY_STRATA:
        fa_maximum = (
            gate.fa_absolute_allowance_per_million
            + gate.fa_source_multiplier * source_fa[stratum]
        )
        if _above_maximum(fa_delta[stratum], fa_maximum, tolerance):
            failures.append(f"fa_inflation:{stratum}")
        if _above_maximum(
            fg_delta[stratum], gate.foreground_fraction_delta_maximum, tolerance
        ):
            failures.append(f"foreground_delta_inflation:{stratum}")
        foreground_maximum = (
            gate.foreground_fraction_source_multiplier * source_fg[stratum]
            + gate.foreground_fraction_epsilon
        )
        if _above_maximum(teacher_fg[stratum], foreground_maximum, tolerance):
            failures.append(f"foreground_ratio_inflation:{stratum}")
    return tuple(failures)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class CandidateRankingEntry:
    rank: int
    candidate_id: str
    nonclean_macro_delta_iou: Fraction
    minimum_dataset_nonclean_delta_iou: Fraction
    overall_macro_delta_iou: Fraction
    clean_macro_delta_iou: Fraction
    nonclean_fa_delta_per_million: Fraction
    nonclean_foreground_fraction_delta: Fraction
    nonclean_macro_delta_pd: Fraction

    def to_receipt(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            **{
                name: _fraction_receipt(getattr(self, name))
                for name in (
                    "nonclean_macro_delta_iou",
                    "minimum_dataset_nonclean_delta_iou",
                    "overall_macro_delta_iou",
                    "clean_macro_delta_iou",
                    "nonclean_fa_delta_per_million",
                    "nonclean_foreground_fraction_delta",
                    "nonclean_macro_delta_pd",
                )
            },
        }


@dataclass(frozen=True)
class TeacherScienceDecision:
    gate: GateConfig
    evidence_kind: Literal["counts", "metrics"]
    integer_count_conservation_verified: bool
    candidate_evaluations: tuple[CandidateEvaluation, ...]
    ranking: tuple[CandidateRankingEntry, ...]
    scientific_status: Literal["scientific_passed", "scientific_no_eligible"]
    result_tier: Literal["development"] = "development"
    development_only: Literal[True] = True
    paper_result: Literal[False] = False
    p5_authorized: Literal[False] = False

    @property
    def eligible_candidate_ids(self) -> tuple[str, ...]:
        return tuple(entry.candidate_id for entry in self.ranking)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "receipt_type": "cr_sitta_p4_nonadaptive_teacher_science_decision",
            "protocol_status": "passed",
            "scientific_status": self.scientific_status,
            "result_tier": self.result_tier,
            "development_only": self.development_only,
            "paper_result": self.paper_result,
            "p5_authorized": self.p5_authorized,
            "gate": self.gate.to_receipt(),
            "evidence_kind": self.evidence_kind,
            "integer_count_conservation_verified": (
                self.integer_count_conservation_verified
            ),
            "topology": {
                "datasets": list(DATASETS),
                "conditions": list(CONDITIONS),
                "expected_candidate_count": EXPECTED_CANDIDATE_COUNT,
                "cells_per_candidate": CELLS_PER_CANDIDATE,
                "nonclean_cells_per_candidate": NONCLEAN_CELLS_PER_CANDIDATE,
                "images_per_cell": IMAGES_PER_CELL,
                "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
                "pixels_per_cell": PIXELS_PER_CELL,
            },
            "candidate_evaluations": [
                evaluation.to_receipt()
                for evaluation in self.candidate_evaluations
            ],
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "ranking": [entry.to_receipt() for entry in self.ranking],
        }


def _ranking_key(
    evaluation: CandidateEvaluation,
) -> tuple[Any, ...]:
    aggregate = evaluation.aggregate
    dataset_values = dict(aggregate.dataset_nonclean_delta_iou).values()
    nonclean_fa = dict(aggregate.fa_delta_per_million)["nonclean"]
    # This order is frozen in nonadaptive_teacher_screen_v1.yaml.  Candidate
    # filtering is complete before this key is ever evaluated.
    return (
        -aggregate.nonclean_macro_delta_iou,
        -aggregate.overall_macro_delta_iou,
        -min(dataset_values),
        nonclean_fa,
        evaluation.candidate_id,
    )


def _ranking_entry(rank: int, evaluation: CandidateEvaluation) -> CandidateRankingEntry:
    aggregate = evaluation.aggregate
    return CandidateRankingEntry(
        rank=rank,
        candidate_id=evaluation.candidate_id,
        nonclean_macro_delta_iou=aggregate.nonclean_macro_delta_iou,
        minimum_dataset_nonclean_delta_iou=min(
            dict(aggregate.dataset_nonclean_delta_iou).values()
        ),
        overall_macro_delta_iou=aggregate.overall_macro_delta_iou,
        clean_macro_delta_iou=aggregate.clean_macro_delta_iou,
        nonclean_fa_delta_per_million=dict(aggregate.fa_delta_per_million)[
            "nonclean"
        ],
        nonclean_foreground_fraction_delta=dict(
            aggregate.foreground_fraction_delta
        )["nonclean"],
        nonclean_macro_delta_pd=aggregate.nonclean_macro_delta_pd,
    )


def evaluate_candidates(
    cell_records: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    gate: GateConfig,
) -> TeacherScienceDecision:
    """Validate, aggregate, filter, and rank one complete P4 screen.

    ``candidate_ids`` is mandatory: it is the frozen ten-candidate roster.
    Missing or extra candidates/cells are protocol errors.  Candidate
    filtering always occurs before the YAML-frozen ranking, whose final
    tie-break is lexicographic candidate ID.
    """

    if not isinstance(gate, GateConfig):
        raise NonadaptiveTeacherGateError("gate must be GateConfig")
    candidates = _sequence(candidate_ids, field="candidate_ids")
    if not candidates:
        raise NonadaptiveTeacherGateError("candidate_ids cannot be empty")
    if any(not isinstance(value, str) or not value for value in candidates):
        raise NonadaptiveTeacherGateError(
            "candidate_ids must contain non-empty strings"
        )
    if len(set(candidates)) != len(candidates):
        raise NonadaptiveTeacherGateError("candidate_ids must be unique")
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise NonadaptiveTeacherGateError(
            "formal P4 candidate_ids must contain exactly "
            f"{EXPECTED_CANDIDATE_COUNT} candidates"
        )
    records = _sequence(cell_records, field="cell_records")
    expected_record_count = len(candidates) * CELLS_PER_CANDIDATE
    if len(records) != expected_record_count:
        raise NonadaptiveTeacherGateError(
            "cell record count must equal candidate_count*39; "
            f"expected={expected_record_count}, observed={len(records)}"
        )
    candidate_set = set(candidates)
    parsed = tuple(_parse_cell(value, index=index) for index, value in enumerate(records))
    unknown_candidates = sorted(
        {cell.candidate_id for cell in parsed} - candidate_set
    )
    if unknown_candidates:
        raise NonadaptiveTeacherGateError(
            f"cell records contain candidates outside frozen roster: {unknown_candidates}"
        )
    evidence_kinds = {cell.evidence_kind for cell in parsed}
    if len(evidence_kinds) != 1:
        raise NonadaptiveTeacherGateError(
            "one P4 evaluation cannot mix count and metric records"
        )
    evidence_kind = next(iter(evidence_kinds))
    if gate.require_integer_counts and evidence_kind != "counts":
        raise NonadaptiveTeacherGateError(
            "formal P4 scientific evaluation requires integer sufficient counts"
        )

    by_candidate: dict[str, dict[tuple[str, str], _ParsedCell]] = {
        candidate: {} for candidate in candidates
    }
    for cell in parsed:
        key = (cell.dataset, cell.condition)
        if key in by_candidate[cell.candidate_id]:
            raise NonadaptiveTeacherGateError(
                "duplicate candidate/dataset/condition cell: "
                f"{cell.candidate_id}/{cell.dataset}/{cell.condition}"
            )
        by_candidate[cell.candidate_id][key] = cell
    expected_keys = {
        (dataset, condition) for dataset in DATASETS for condition in CONDITIONS
    }
    for candidate in candidates:
        observed = set(by_candidate[candidate])
        if observed != expected_keys:
            missing = sorted(expected_keys - observed)
            extra = sorted(observed - expected_keys)
            raise NonadaptiveTeacherGateError(
                f"candidate {candidate} does not have exact 3x13 topology; "
                f"missing={missing}, extra={extra}"
            )

    # Every candidate must compare against exactly the same Source endpoint.
    source_reference: dict[tuple[str, str], SufficientCounts | EndpointMetrics] = {}
    for candidate in candidates:
        for key in sorted(expected_keys):
            source_raw = by_candidate[candidate][key].source_raw
            if key not in source_reference:
                source_reference[key] = source_raw
            elif source_reference[key] != source_raw:
                raise NonadaptiveTeacherGateError(
                    "Source endpoint differs across candidates for "
                    f"{key[0]}/{key[1]}"
                )

    if evidence_kind == "counts":
        reference_candidate = candidates[0]
        for dataset in DATASETS:
            expected_denominators: tuple[int, int, int, int] | None = None
            for condition in CONDITIONS:
                raw = by_candidate[reference_candidate][
                    (dataset, condition)
                ].source_raw
                assert isinstance(raw, SufficientCounts)
                denominators = (
                    raw.target_positive_pixels,
                    raw.total_targets,
                    raw.total_image_pixels,
                    raw.image_count,
                )
                if expected_denominators is None:
                    expected_denominators = denominators
                elif denominators != expected_denominators:
                    raise NonadaptiveTeacherGateError(
                        "ground-truth denominators vary across conditions for "
                        f"dataset {dataset}"
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

    ranked_evaluations = sorted(
        (value for value in evaluations if value.eligible),
        key=_ranking_key,
    )
    ranking = tuple(
        _ranking_entry(index, value)
        for index, value in enumerate(ranked_evaluations, start=1)
    )
    return TeacherScienceDecision(
        gate=gate,
        evidence_kind=evidence_kind,
        integer_count_conservation_verified=evidence_kind == "counts",
        candidate_evaluations=tuple(evaluations),
        ranking=ranking,
        scientific_status=(
            "scientific_passed" if ranking else "scientific_no_eligible"
        ),
    )


def evaluate_nonadaptive_teacher_gate(
    cell_records: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    gate: GateConfig,
) -> TeacherScienceDecision:
    """Descriptive alias for :func:`evaluate_candidates`."""

    return evaluate_candidates(cell_records, candidate_ids, gate)


__all__ = [
    "CELLS_PER_CANDIDATE",
    "CONDITIONS",
    "CORRUPTION_FAMILIES",
    "COUNT_FIELD_ORDER",
    "COUNT_FIELDS",
    "CandidateAggregate",
    "CandidateEvaluation",
    "CandidateRankingEntry",
    "DATASETS",
    "EndpointMetrics",
    "EXPECTED_CANDIDATE_COUNT",
    "GateConfig",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "IMAGES_PER_CELL",
    "METRIC_FIELDS",
    "METRIC_FIELD_ORDER",
    "NONCLEAN_CELLS_PER_CANDIDATE",
    "NonadaptiveTeacherGateError",
    "SAFETY_STRATA",
    "SEVERITIES",
    "PIXELS_PER_CELL",
    "SufficientCounts",
    "TeacherScienceDecision",
    "evaluate_candidates",
    "evaluate_nonadaptive_teacher_gate",
]
