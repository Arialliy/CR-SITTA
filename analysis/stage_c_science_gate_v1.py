"""Pure fail-closed science gate for the CR-SITTA Stage-C0 signal audit.

The gate consumes one train-only fixed-Pilot64 mechanism aggregate.  Its 4,608
non-clean image/condition/probe observations are counted exactly once.  R-E1,
R-D0, and P2 carry separate gradient/alignment summaries over those same
observations; they are not three copies of the activity sample.

This module imports no project, filesystem, CUDA, model, or launcher code and
performs no side effects.  Invalid evidence raises ``StageCProtocolError``.
Complete evidence with no eligible parameter space is the normal scientific
result ``scientific_no_eligible``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
from typing import Any, Literal


SCHEMA_VERSION = "stage_c0_aggregate_evidence_v1"
RECEIPT_SCHEMA_VERSION = "stage_c0_science_receipt_v1"

SCOPE_FIELDS = frozenset(
    {
        "data_role",
        "pilot_role",
        "pilot_image_count_per_dataset",
        "development_only",
        "paper_result",
        "thresholds_frozen_before_run",
        "validation_access_count",
        "test_access_count",
    }
)
MECHANISM_FIELDS = frozenset(
    {
        "mechanism_id",
        "nonclean_probe_episode_count",
        "nonclean_nonfinite_signal_episode_count",
        "nonclean_active_support_episode_count",
        "nonclean_active_pixel_count",
        "nonclean_candidate_proximal_active_episode_count",
        "nonclean_candidate_proximal_active_pixel_count",
        "nonclean_total_pixel_count",
        "signal_summary",
        "space_aggregates",
        "o4_activity",
    }
)
SIGNAL_SUMMARY_FIELDS = frozenset(
    {
        "teacher_student_probability_l1_mean",
        "teacher_student_logit_gap_mean",
        "teacher_student_logit_gap_max",
        "active_pixel_fraction_lf_mean",
        "active_pixel_fraction_hf_mean",
        "active_target_weight_lf_mean",
        "active_background_weight_lf_mean",
        "active_target_weight_hf_mean",
        "active_background_weight_hf_mean",
    }
)
SPACE_FIELDS = frozenset(
    {
        "parameter_space",
        "nonclean_probe_episode_count",
        "nonclean_nonfinite_measurement_episode_count",
        "nonclean_finite_nonzero_proxy_gradient_episode_count",
        "nonclean_threshold_crossing_episode_count",
        "proxy_gradient_norm_mean",
        "task_gradient_norm_mean",
        "candidate_absolute_response_derivative_mean",
        "candidate_local_contrast_derivative_mean",
        "dataset_aggregates",
        "family_aggregates",
        "macro_outer_gradient_cosine",
        "virtual_step_contract",
        "identity",
    }
)
DATASET_FIELDS = frozenset(
    {
        "dataset_id",
        "nonclean_probe_episode_count",
        "both_gradients_nonzero_episode_count",
        "macro_outer_gradient_cosine",
    }
)
FAMILY_FIELDS = frozenset(
    {
        "family_id",
        "nonclean_probe_episode_count",
        "normalized_virtual_task_loss_directional_derivative",
    }
)
VIRTUAL_STEP_FIELDS = frozenset(
    {"radius_mode", "radius_value", "nonzero_epsilon", "direction", "clip_rule"}
)
IDENTITY_FIELDS = frozenset(
    {
        "comparison_count",
        "exact_match_count",
        "mismatch_count",
        "maximum_absolute_output_difference",
    }
)
O4_ACTIVITY_FIELDS = frozenset(
    {"configured", "active_episode_count", "contribution_episode_count"}
)
GATE_CONFIG_FIELDS = frozenset(
    {
        "mechanism_id",
        "dataset_ids",
        "family_ids",
        "space_ids",
        "space_virtual_step_contracts",
        "expected_nonclean_probe_episode_count",
        "expected_nonclean_probe_episode_count_per_dataset",
        "expected_nonclean_probe_episode_count_per_family",
        "expected_identity_comparison_count_per_space",
        "minimum_nonclean_finite_nonzero_proxy_gradient_fraction",
        "minimum_nonclean_active_support_episode_fraction",
        "minimum_candidate_proximal_active_episode_fraction_among_active",
        "dataset_cosine_strictly_greater_than",
        "minimum_positive_dataset_cosines",
        "macro_cosine_strictly_greater_than",
        "virtual_task_derivative_improvement_relation",
        "minimum_improving_families",
        "require_identity_bit_exact",
        "filter_before_reporting_eligible_spaces",
        "zero_signal_spaces_eligible",
    }
)


class StageCProtocolError(ValueError):
    """Stage-C0 configuration/evidence failed its protocol contract."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageCProtocolError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageCProtocolError(f"{field} must be a sequence")
    return value


def _exact(raw: Mapping[str, Any], fields: frozenset[str], name: str) -> None:
    actual = frozenset(raw)
    if actual != fields:
        raise StageCProtocolError(
            f"{name} fields must be exact; missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise StageCProtocolError(f"{field} must be a non-empty string")
    return value


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise StageCProtocolError(f"{field} must be a boolean")
    return value


def _count(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageCProtocolError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise StageCProtocolError(f"{field} must be {qualifier}")
    return value


def _fraction(value: Any, field: str) -> Fraction:
    if isinstance(value, bool):
        raise StageCProtocolError(f"{field} must be a finite real number")
    try:
        if isinstance(value, Mapping):
            _exact(value, frozenset({"numerator", "denominator"}), field)
            numerator = value["numerator"]
            denominator = value["denominator"]
            if isinstance(numerator, bool) or not isinstance(numerator, int):
                raise StageCProtocolError(f"{field}.numerator must be an integer")
            if (
                isinstance(denominator, bool)
                or not isinstance(denominator, int)
                or denominator <= 0
            ):
                raise StageCProtocolError(
                    f"{field}.denominator must be a positive integer"
                )
            result = Fraction(numerator, denominator)
        elif isinstance(value, Fraction):
            result = value
        elif isinstance(value, Decimal):
            if not value.is_finite():
                raise StageCProtocolError(f"{field} must be finite")
            result = Fraction(value)
        elif isinstance(value, int):
            result = Fraction(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise StageCProtocolError(f"{field} must be finite")
            result = Fraction(Decimal(str(value)))
        elif isinstance(value, str):
            decimal = Decimal(value)
            if not decimal.is_finite():
                raise StageCProtocolError(f"{field} must be finite")
            result = Fraction(decimal)
        else:
            raise StageCProtocolError(f"{field} must be a finite real number")
    except StageCProtocolError:
        raise
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise StageCProtocolError(f"{field} must be a finite real number") from exc
    return result


def _nonnegative(value: Any, field: str) -> Fraction:
    result = _fraction(value, field)
    if result < 0:
        raise StageCProtocolError(f"{field} must be non-negative")
    return result


def _unit(value: Any, field: str) -> Fraction:
    result = _fraction(value, field)
    if result < 0 or result > 1:
        raise StageCProtocolError(f"{field} must lie in [0, 1]")
    return result


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


@dataclass(frozen=True, slots=True)
class VirtualStepContract:
    radius_mode: str
    radius_value: Fraction
    nonzero_epsilon: Fraction
    direction: str
    clip_rule: str

    @classmethod
    def from_mapping(cls, value: Any, field: str) -> "VirtualStepContract":
        raw = _mapping(value, field)
        _exact(raw, VIRTUAL_STEP_FIELDS, field)
        result = cls(
            radius_mode=_text(raw["radius_mode"], f"{field}.radius_mode"),
            radius_value=_nonnegative(raw["radius_value"], f"{field}.radius_value"),
            nonzero_epsilon=_nonnegative(
                raw["nonzero_epsilon"], f"{field}.nonzero_epsilon"
            ),
            direction=_text(raw["direction"], f"{field}.direction"),
            clip_rule=_text(raw["clip_rule"], f"{field}.clip_rule"),
        )
        if result.radius_value <= 0 or result.nonzero_epsilon <= 0:
            raise StageCProtocolError(f"{field} radius and epsilon must be positive")
        if result.radius_mode not in {
            "absolute_l2",
            "relative_to_source_parameter_l2",
        }:
            raise StageCProtocolError(f"{field}.radius_mode is unsupported")
        if result.direction != "negative_proxy_gradient":
            raise StageCProtocolError(f"{field}.direction must be negative_proxy_gradient")
        if result.clip_rule != "min_1_radius_over_norm_plus_epsilon":
            raise StageCProtocolError(f"{field}.clip_rule is unsupported")
        return result


@dataclass(frozen=True, slots=True)
class StageCGateConfig:
    mechanism_id: str
    dataset_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    space_ids: tuple[str, ...]
    space_virtual_step_contracts: tuple[tuple[str, VirtualStepContract], ...]
    expected_nonclean_probe_episode_count: int
    expected_nonclean_probe_episode_count_per_dataset: int
    expected_nonclean_probe_episode_count_per_family: int
    expected_identity_comparison_count_per_space: int
    minimum_nonclean_finite_nonzero_proxy_gradient_fraction: Fraction
    minimum_nonclean_active_support_episode_fraction: Fraction
    minimum_candidate_proximal_active_episode_fraction_among_active: Fraction
    dataset_cosine_strictly_greater_than: Fraction
    minimum_positive_dataset_cosines: int
    macro_cosine_strictly_greater_than: Fraction
    virtual_task_derivative_improvement_relation: str
    minimum_improving_families: int
    require_identity_bit_exact: bool
    filter_before_reporting_eligible_spaces: bool
    zero_signal_spaces_eligible: bool

    @property
    def step_contract_by_space(self) -> dict[str, VirtualStepContract]:
        return dict(self.space_virtual_step_contracts)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StageCGateConfig":
        raw = _mapping(value, "gate_config")
        _exact(raw, GATE_CONFIG_FIELDS, "gate_config")
        datasets = tuple(
            _text(item, f"dataset_ids[{index}]")
            for index, item in enumerate(_sequence(raw["dataset_ids"], "dataset_ids"))
        )
        families = tuple(
            _text(item, f"family_ids[{index}]")
            for index, item in enumerate(_sequence(raw["family_ids"], "family_ids"))
        )
        spaces = tuple(
            _text(item, f"space_ids[{index}]")
            for index, item in enumerate(_sequence(raw["space_ids"], "space_ids"))
        )
        if len(datasets) != 3 or len(set(datasets)) != 3:
            raise StageCProtocolError("dataset_ids must contain 3 unique IDs")
        if len(families) != 4 or len(set(families)) != 4:
            raise StageCProtocolError("family_ids must contain 4 unique IDs")
        if spaces != ("R-E1", "R-D0", "P2"):
            raise StageCProtocolError("space_ids must be exactly R-E1, R-D0, P2")
        contract_map = _mapping(
            raw["space_virtual_step_contracts"], "space_virtual_step_contracts"
        )
        if tuple(contract_map) != spaces:
            raise StageCProtocolError("virtual-step contract roster/order differs")
        contracts = tuple(
            (
                space,
                VirtualStepContract.from_mapping(
                    contract_map[space], f"space_virtual_step_contracts.{space}"
                ),
            )
            for space in spaces
        )
        result = cls(
            mechanism_id=_text(raw["mechanism_id"], "mechanism_id"),
            dataset_ids=datasets,
            family_ids=families,
            space_ids=spaces,
            space_virtual_step_contracts=contracts,
            expected_nonclean_probe_episode_count=_count(
                raw["expected_nonclean_probe_episode_count"],
                "expected_nonclean_probe_episode_count",
                positive=True,
            ),
            expected_nonclean_probe_episode_count_per_dataset=_count(
                raw["expected_nonclean_probe_episode_count_per_dataset"],
                "expected_nonclean_probe_episode_count_per_dataset",
                positive=True,
            ),
            expected_nonclean_probe_episode_count_per_family=_count(
                raw["expected_nonclean_probe_episode_count_per_family"],
                "expected_nonclean_probe_episode_count_per_family",
                positive=True,
            ),
            expected_identity_comparison_count_per_space=_count(
                raw["expected_identity_comparison_count_per_space"],
                "expected_identity_comparison_count_per_space",
                positive=True,
            ),
            minimum_nonclean_finite_nonzero_proxy_gradient_fraction=_unit(
                raw["minimum_nonclean_finite_nonzero_proxy_gradient_fraction"],
                "minimum_nonclean_finite_nonzero_proxy_gradient_fraction",
            ),
            minimum_nonclean_active_support_episode_fraction=_unit(
                raw["minimum_nonclean_active_support_episode_fraction"],
                "minimum_nonclean_active_support_episode_fraction",
            ),
            minimum_candidate_proximal_active_episode_fraction_among_active=_unit(
                raw[
                    "minimum_candidate_proximal_active_episode_fraction_among_active"
                ],
                "minimum_candidate_proximal_active_episode_fraction_among_active",
            ),
            dataset_cosine_strictly_greater_than=_fraction(
                raw["dataset_cosine_strictly_greater_than"],
                "dataset_cosine_strictly_greater_than",
            ),
            minimum_positive_dataset_cosines=_count(
                raw["minimum_positive_dataset_cosines"],
                "minimum_positive_dataset_cosines",
                positive=True,
            ),
            macro_cosine_strictly_greater_than=_fraction(
                raw["macro_cosine_strictly_greater_than"],
                "macro_cosine_strictly_greater_than",
            ),
            virtual_task_derivative_improvement_relation=_text(
                raw["virtual_task_derivative_improvement_relation"],
                "virtual_task_derivative_improvement_relation",
            ),
            minimum_improving_families=_count(
                raw["minimum_improving_families"],
                "minimum_improving_families",
                positive=True,
            ),
            require_identity_bit_exact=_boolean(
                raw["require_identity_bit_exact"], "require_identity_bit_exact"
            ),
            filter_before_reporting_eligible_spaces=_boolean(
                raw["filter_before_reporting_eligible_spaces"],
                "filter_before_reporting_eligible_spaces",
            ),
            zero_signal_spaces_eligible=_boolean(
                raw["zero_signal_spaces_eligible"], "zero_signal_spaces_eligible"
            ),
        )
        if result.mechanism_id != "ASB-SFR_C0":
            raise StageCProtocolError("mechanism_id must be ASB-SFR_C0")
        if result.virtual_task_derivative_improvement_relation != "strictly_less_than_zero":
            raise StageCProtocolError("task-loss derivative improvement must mean <0")
        if result.minimum_positive_dataset_cosines > 3:
            raise StageCProtocolError("positive dataset minimum exceeds roster")
        if result.minimum_improving_families > 4:
            raise StageCProtocolError("improving family minimum exceeds roster")
        if 3 * result.expected_nonclean_probe_episode_count_per_dataset != result.expected_nonclean_probe_episode_count:
            raise StageCProtocolError("dataset counts do not sum to unique episodes")
        if 4 * result.expected_nonclean_probe_episode_count_per_family != result.expected_nonclean_probe_episode_count:
            raise StageCProtocolError("family counts do not sum to unique episodes")
        if not result.require_identity_bit_exact:
            raise StageCProtocolError("identity must be bit exact")
        if not result.filter_before_reporting_eligible_spaces:
            raise StageCProtocolError("scientific filtering must precede reporting")
        if result.zero_signal_spaces_eligible:
            raise StageCProtocolError("zero-signal spaces cannot be eligible")
        return result


def default_gate_config() -> StageCGateConfig:
    """Return the preregistered C0 thresholds and diagnostic step contracts."""

    shared_step = {
        "nonzero_epsilon": "1e-12",
        "direction": "negative_proxy_gradient",
        "clip_rule": "min_1_radius_over_norm_plus_epsilon",
    }
    return StageCGateConfig.from_mapping(
        {
            "mechanism_id": "ASB-SFR_C0",
            "dataset_ids": ["IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"],
            "family_ids": [
                "gaussian_noise",
                "gaussian_blur",
                "low_contrast",
                "stripe_noise",
            ],
            "space_ids": ["R-E1", "R-D0", "P2"],
            "space_virtual_step_contracts": {
                "R-E1": {"radius_mode": "absolute_l2", "radius_value": "0.25", **shared_step},
                "R-D0": {"radius_mode": "absolute_l2", "radius_value": "0.25", **shared_step},
                "P2": {"radius_mode": "relative_to_source_parameter_l2", "radius_value": "0.0005", **shared_step},
            },
            "expected_nonclean_probe_episode_count": 4608,
            "expected_nonclean_probe_episode_count_per_dataset": 1536,
            "expected_nonclean_probe_episode_count_per_family": 1152,
            "expected_identity_comparison_count_per_space": 192,
            "minimum_nonclean_finite_nonzero_proxy_gradient_fraction": "0.80",
            "minimum_nonclean_active_support_episode_fraction": "0.30",
            "minimum_candidate_proximal_active_episode_fraction_among_active": "0.05",
            "dataset_cosine_strictly_greater_than": "0",
            "minimum_positive_dataset_cosines": 2,
            "macro_cosine_strictly_greater_than": "0.08",
            "virtual_task_derivative_improvement_relation": "strictly_less_than_zero",
            "minimum_improving_families": 3,
            "require_identity_bit_exact": True,
            "filter_before_reporting_eligible_spaces": True,
            "zero_signal_spaces_eligible": False,
        }
    )


@dataclass(frozen=True, slots=True)
class DatasetAggregate:
    dataset_id: str
    nonclean_probe_episode_count: int
    both_gradients_nonzero_episode_count: int
    macro_outer_gradient_cosine: Fraction


@dataclass(frozen=True, slots=True)
class FamilyAggregate:
    family_id: str
    nonclean_probe_episode_count: int
    normalized_virtual_task_loss_directional_derivative: Fraction

    @property
    def improving(self) -> bool:
        return self.normalized_virtual_task_loss_directional_derivative < 0


@dataclass(frozen=True, slots=True)
class IdentityAggregate:
    comparison_count: int
    exact_match_count: int
    mismatch_count: int
    maximum_absolute_output_difference: Fraction

    @property
    def bit_exact(self) -> bool:
        return (
            self.exact_match_count == self.comparison_count
            and self.mismatch_count == 0
            and self.maximum_absolute_output_difference == 0
        )


@dataclass(frozen=True, slots=True)
class SpaceAggregate:
    parameter_space: str
    nonclean_probe_episode_count: int
    nonclean_nonfinite_measurement_episode_count: int
    nonclean_finite_nonzero_proxy_gradient_episode_count: int
    nonclean_threshold_crossing_episode_count: int
    proxy_gradient_norm_mean: Fraction
    task_gradient_norm_mean: Fraction
    candidate_absolute_response_derivative_mean: Fraction
    candidate_local_contrast_derivative_mean: Fraction
    dataset_aggregates: tuple[DatasetAggregate, ...]
    family_aggregates: tuple[FamilyAggregate, ...]
    macro_outer_gradient_cosine: Fraction
    virtual_step_contract: VirtualStepContract
    identity: IdentityAggregate

    @property
    def finite_nonzero_proxy_fraction(self) -> Fraction:
        return Fraction(
            self.nonclean_finite_nonzero_proxy_gradient_episode_count,
            self.nonclean_probe_episode_count,
        )

    @property
    def threshold_crossing_fraction(self) -> Fraction:
        # Recorded for C3 planning only; it is not a C0 hard gate.
        return Fraction(
            self.nonclean_threshold_crossing_episode_count,
            self.nonclean_probe_episode_count,
        )

    def dataset_count_strictly_above(self, threshold: Fraction) -> int:
        """Count datasets against the preregistered strict cosine threshold."""

        return sum(
            item.macro_outer_gradient_cosine > threshold
            for item in self.dataset_aggregates
        )

    @property
    def improving_family_count(self) -> int:
        return sum(item.improving for item in self.family_aggregates)

    @property
    def has_nonzero_signal(self) -> bool:
        return (
            self.nonclean_finite_nonzero_proxy_gradient_episode_count > 0
            and self.proxy_gradient_norm_mean > 0
        )


@dataclass(frozen=True, slots=True)
class MechanismAggregate:
    mechanism_id: str
    nonclean_probe_episode_count: int
    nonclean_active_support_episode_count: int
    nonclean_candidate_proximal_active_episode_count: int
    space_aggregates: tuple[SpaceAggregate, ...]
    o4_active_episode_count: int
    o4_contribution_episode_count: int

    @property
    def active_fraction(self) -> Fraction:
        return Fraction(
            self.nonclean_active_support_episode_count,
            self.nonclean_probe_episode_count,
        )

    @property
    def proximal_fraction_among_active(self) -> Fraction:
        if self.nonclean_active_support_episode_count == 0:
            return Fraction(0)
        return Fraction(
            self.nonclean_candidate_proximal_active_episode_count,
            self.nonclean_active_support_episode_count,
        )


@dataclass(frozen=True, slots=True)
class StageCReplicateEvidence:
    schema_version: str
    replicate_id: str
    mechanism: MechanismAggregate

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], gate: StageCGateConfig
    ) -> "StageCReplicateEvidence":
        raw = _mapping(value, "evidence")
        _exact(
            raw,
            frozenset({"schema_version", "replicate_id", "scope", "mechanism_aggregate"}),
            "evidence",
        )
        if _text(raw["schema_version"], "schema_version") != SCHEMA_VERSION:
            raise StageCProtocolError("unsupported Stage-C0 evidence schema")
        _parse_scope(raw["scope"])
        return cls(
            schema_version=SCHEMA_VERSION,
            replicate_id=_text(raw["replicate_id"], "replicate_id"),
            mechanism=_parse_mechanism(raw["mechanism_aggregate"], gate),
        )


def _parse_scope(value: Any) -> None:
    raw = _mapping(value, "scope")
    _exact(raw, SCOPE_FIELDS, "scope")
    if _text(raw["data_role"], "scope.data_role") != "train":
        raise StageCProtocolError("C0 data role must be train")
    if _text(raw["pilot_role"], "scope.pilot_role") != "fixed_Pilot64":
        raise StageCProtocolError("C0 must use fixed Pilot64")
    if _count(raw["pilot_image_count_per_dataset"], "scope.pilot count") != 64:
        raise StageCProtocolError("C0 Pilot64 must contain 64 train images per dataset")
    if not _boolean(raw["development_only"], "scope.development_only"):
        raise StageCProtocolError("C0 must be development-only")
    if _boolean(raw["paper_result"], "scope.paper_result"):
        raise StageCProtocolError("C0 is not a paper result")
    if not _boolean(
        raw["thresholds_frozen_before_run"], "scope.thresholds_frozen_before_run"
    ):
        raise StageCProtocolError("C0 thresholds must be frozen before execution")
    if _count(raw["validation_access_count"], "scope.validation_access_count") != 0:
        raise StageCProtocolError("C0 permits no validation access")
    if _count(raw["test_access_count"], "scope.test_access_count") != 0:
        raise StageCProtocolError("C0 permits no test access")


def _parse_signal_summary(value: Any) -> None:
    raw = _mapping(value, "signal_summary")
    _exact(raw, SIGNAL_SUMMARY_FIELDS, "signal_summary")
    parsed = {name: _nonnegative(raw[name], f"signal_summary.{name}") for name in raw}
    for name in ("active_pixel_fraction_lf_mean", "active_pixel_fraction_hf_mean"):
        if parsed[name] > 1:
            raise StageCProtocolError(f"signal_summary.{name} must lie in [0,1]")
    if parsed["teacher_student_logit_gap_max"] < parsed["teacher_student_logit_gap_mean"]:
        raise StageCProtocolError("maximum logit gap cannot be below mean logit gap")


def _parse_dataset(value: Any, gate: StageCGateConfig, field: str) -> DatasetAggregate:
    raw = _mapping(value, field)
    _exact(raw, DATASET_FIELDS, field)
    total = _count(raw["nonclean_probe_episode_count"], f"{field}.episode_count", positive=True)
    both = _count(raw["both_gradients_nonzero_episode_count"], f"{field}.both_count")
    if total != gate.expected_nonclean_probe_episode_count_per_dataset or both > total:
        raise StageCProtocolError(f"{field} counts disagree with frozen topology")
    return DatasetAggregate(
        _text(raw["dataset_id"], f"{field}.dataset_id"),
        total,
        both,
        _fraction(raw["macro_outer_gradient_cosine"], f"{field}.cosine"),
    )


def _parse_family(value: Any, gate: StageCGateConfig, field: str) -> FamilyAggregate:
    raw = _mapping(value, field)
    _exact(raw, FAMILY_FIELDS, field)
    total = _count(raw["nonclean_probe_episode_count"], f"{field}.episode_count", positive=True)
    if total != gate.expected_nonclean_probe_episode_count_per_family:
        raise StageCProtocolError(f"{field} count disagrees with frozen topology")
    return FamilyAggregate(
        _text(raw["family_id"], f"{field}.family_id"),
        total,
        _fraction(
            raw["normalized_virtual_task_loss_directional_derivative"],
            f"{field}.directional_derivative",
        ),
    )


def _parse_identity(value: Any, gate: StageCGateConfig, field: str) -> IdentityAggregate:
    raw = _mapping(value, field)
    _exact(raw, IDENTITY_FIELDS, field)
    result = IdentityAggregate(
        comparison_count=_count(raw["comparison_count"], f"{field}.comparison_count", positive=True),
        exact_match_count=_count(raw["exact_match_count"], f"{field}.exact_match_count"),
        mismatch_count=_count(raw["mismatch_count"], f"{field}.mismatch_count"),
        maximum_absolute_output_difference=_nonnegative(
            raw["maximum_absolute_output_difference"], f"{field}.maximum_difference"
        ),
    )
    if result.comparison_count != gate.expected_identity_comparison_count_per_space:
        raise StageCProtocolError(f"{field} comparison count disagrees with topology")
    if result.exact_match_count + result.mismatch_count != result.comparison_count:
        raise StageCProtocolError(f"{field} counts do not conserve")
    if (result.mismatch_count == 0) != (result.maximum_absolute_output_difference == 0):
        raise StageCProtocolError(f"{field} mismatch count contradicts maximum difference")
    return result


def _parse_space(value: Any, gate: StageCGateConfig, field: str) -> SpaceAggregate:
    raw = _mapping(value, field)
    _exact(raw, SPACE_FIELDS, field)
    name = _text(raw["parameter_space"], f"{field}.parameter_space")
    if name not in gate.space_ids:
        raise StageCProtocolError(f"{field} contains an unfrozen space")
    total = _count(raw["nonclean_probe_episode_count"], f"{field}.episode_count", positive=True)
    if total != gate.expected_nonclean_probe_episode_count:
        raise StageCProtocolError(f"{field} must use the same unique 4608 denominator")
    nonfinite = _count(
        raw["nonclean_nonfinite_measurement_episode_count"], f"{field}.nonfinite_count"
    )
    nonzero = _count(
        raw["nonclean_finite_nonzero_proxy_gradient_episode_count"],
        f"{field}.nonzero_count",
    )
    crossings = _count(
        raw["nonclean_threshold_crossing_episode_count"], f"{field}.crossing_count"
    )
    if nonfinite:
        raise StageCProtocolError(f"{field} contains non-finite measurements")
    if nonzero > total or crossings > total:
        raise StageCProtocolError(f"{field} episode count exceeds denominator")
    proxy_norm = _nonnegative(raw["proxy_gradient_norm_mean"], f"{field}.proxy_norm")
    if (nonzero == 0) != (proxy_norm == 0):
        raise StageCProtocolError(f"{field} nonzero count contradicts proxy norm")
    datasets = tuple(
        _parse_dataset(item, gate, f"{field}.dataset_aggregates[{index}]")
        for index, item in enumerate(
            _sequence(raw["dataset_aggregates"], f"{field}.dataset_aggregates")
        )
    )
    families = tuple(
        _parse_family(item, gate, f"{field}.family_aggregates[{index}]")
        for index, item in enumerate(
            _sequence(raw["family_aggregates"], f"{field}.family_aggregates")
        )
    )
    if tuple(item.dataset_id for item in datasets) != gate.dataset_ids:
        raise StageCProtocolError(f"{field} dataset roster/order differs")
    if tuple(item.family_id for item in families) != gate.family_ids:
        raise StageCProtocolError(f"{field} family roster/order differs")
    if sum(item.nonclean_probe_episode_count for item in datasets) != total:
        raise StageCProtocolError(f"{field} dataset counts do not conserve")
    if sum(item.nonclean_probe_episode_count for item in families) != total:
        raise StageCProtocolError(f"{field} family counts do not conserve")
    macro = _fraction(raw["macro_outer_gradient_cosine"], f"{field}.macro_cosine")
    recomputed = sum(
        (item.macro_outer_gradient_cosine for item in datasets), Fraction(0)
    ) / len(datasets)
    if macro != recomputed:
        raise StageCProtocolError(f"{field} macro cosine disagrees with dataset mean")
    step = VirtualStepContract.from_mapping(
        raw["virtual_step_contract"], f"{field}.virtual_step_contract"
    )
    if step != gate.step_contract_by_space[name]:
        raise StageCProtocolError(f"{field} virtual-step contract differs from preregistration")
    return SpaceAggregate(
        parameter_space=name,
        nonclean_probe_episode_count=total,
        nonclean_nonfinite_measurement_episode_count=nonfinite,
        nonclean_finite_nonzero_proxy_gradient_episode_count=nonzero,
        nonclean_threshold_crossing_episode_count=crossings,
        proxy_gradient_norm_mean=proxy_norm,
        task_gradient_norm_mean=_nonnegative(raw["task_gradient_norm_mean"], f"{field}.task_norm"),
        candidate_absolute_response_derivative_mean=_fraction(
            raw["candidate_absolute_response_derivative_mean"], f"{field}.absolute_derivative"
        ),
        candidate_local_contrast_derivative_mean=_fraction(
            raw["candidate_local_contrast_derivative_mean"], f"{field}.contrast_derivative"
        ),
        dataset_aggregates=datasets,
        family_aggregates=families,
        macro_outer_gradient_cosine=macro,
        virtual_step_contract=step,
        identity=_parse_identity(raw["identity"], gate, f"{field}.identity"),
    )


def _parse_mechanism(value: Any, gate: StageCGateConfig) -> MechanismAggregate:
    raw = _mapping(value, "mechanism_aggregate")
    _exact(raw, MECHANISM_FIELDS, "mechanism_aggregate")
    mechanism_id = _text(raw["mechanism_id"], "mechanism_aggregate.mechanism_id")
    if mechanism_id != gate.mechanism_id:
        raise StageCProtocolError("mechanism ID differs from preregistration")
    total = _count(raw["nonclean_probe_episode_count"], "mechanism episode count", positive=True)
    if total != gate.expected_nonclean_probe_episode_count:
        raise StageCProtocolError("mechanism must have exactly 4608 unique episodes")
    if _count(raw["nonclean_nonfinite_signal_episode_count"], "nonfinite signal count"):
        raise StageCProtocolError("mechanism contains non-finite signal measurements")
    active = _count(raw["nonclean_active_support_episode_count"], "active episode count")
    active_pixels = _count(raw["nonclean_active_pixel_count"], "active pixel count")
    proximal = _count(
        raw["nonclean_candidate_proximal_active_episode_count"], "proximal episode count"
    )
    proximal_pixels = _count(
        raw["nonclean_candidate_proximal_active_pixel_count"], "proximal pixel count"
    )
    total_pixels = _count(raw["nonclean_total_pixel_count"], "total pixel count", positive=True)
    if active > total or proximal > active or active_pixels > total_pixels or proximal_pixels > active_pixels:
        raise StageCProtocolError("mechanism activity counts do not conserve")
    if (active == 0) != (active_pixels == 0):
        raise StageCProtocolError("active episode/pixel counts contradict")
    if (proximal == 0) != (proximal_pixels == 0):
        raise StageCProtocolError("proximal episode/pixel counts contradict")
    _parse_signal_summary(raw["signal_summary"])
    spaces = tuple(
        _parse_space(item, gate, f"space_aggregates[{index}]")
        for index, item in enumerate(_sequence(raw["space_aggregates"], "space_aggregates"))
    )
    if tuple(item.parameter_space for item in spaces) != gate.space_ids:
        raise StageCProtocolError(
            "space roster/order differs; shared activity must not be duplicated as candidates"
        )
    o4_raw = _mapping(raw["o4_activity"], "o4_activity")
    _exact(o4_raw, O4_ACTIVITY_FIELDS, "o4_activity")
    o4_configured = _boolean(o4_raw["configured"], "o4_activity.configured")
    o4_active = _count(o4_raw["active_episode_count"], "o4_activity.active_episode_count")
    o4_contribution = _count(
        o4_raw["contribution_episode_count"], "o4_activity.contribution_episode_count"
    )
    if (
        o4_active > total
        or o4_contribution > o4_active
        or (not o4_configured and o4_active)
        or (o4_active == 0 and o4_contribution)
    ):
        raise StageCProtocolError("inactive or absent O4 cannot claim contribution")
    return MechanismAggregate(
        mechanism_id,
        total,
        active,
        proximal,
        spaces,
        o4_active,
        o4_contribution,
    )


def _shared_failures(
    mechanism: MechanismAggregate, gate: StageCGateConfig
) -> tuple[str, ...]:
    failures: list[str] = []
    if mechanism.active_fraction < gate.minimum_nonclean_active_support_episode_fraction:
        failures.append("nonclean_active_support_episode_fraction")
    if (
        mechanism.proximal_fraction_among_active
        < gate.minimum_candidate_proximal_active_episode_fraction_among_active
    ):
        failures.append("active_support_only_far_background")
    return tuple(failures)


def _space_failures(space: SpaceAggregate, gate: StageCGateConfig) -> tuple[str, ...]:
    failures: list[str] = []
    if not space.has_nonzero_signal:
        failures.append("zero_signal_space")
    if (
        space.finite_nonzero_proxy_fraction
        < gate.minimum_nonclean_finite_nonzero_proxy_gradient_fraction
    ):
        failures.append("nonclean_finite_nonzero_proxy_gradient_fraction")
    if (
        space.dataset_count_strictly_above(
            gate.dataset_cosine_strictly_greater_than
        )
        < gate.minimum_positive_dataset_cosines
    ):
        failures.append("positive_dataset_cosine_coverage")
    # Strict: exactly 0.08 does not pass.
    if space.macro_outer_gradient_cosine <= gate.macro_cosine_strictly_greater_than:
        failures.append("macro_outer_gradient_cosine")
    if space.improving_family_count < gate.minimum_improving_families:
        failures.append("improving_family_coverage")
    if gate.require_identity_bit_exact and not space.identity.bit_exact:
        failures.append("identity_not_bit_exact")
    return tuple(failures)


@dataclass(frozen=True, slots=True)
class SpaceEvaluation:
    parameter_space: str
    eligible: bool
    reason_codes: tuple[str, ...]
    positive_dataset_cosine_count: int
    aggregate: SpaceAggregate

    def to_receipt(self) -> dict[str, Any]:
        return {
            "parameter_space": self.parameter_space,
            "eligible": self.eligible,
            "reason_codes": list(self.reason_codes),
            "finite_nonzero_proxy_fraction": _fraction_json(
                self.aggregate.finite_nonzero_proxy_fraction
            ),
            "positive_dataset_cosine_count": self.positive_dataset_cosine_count,
            "macro_outer_gradient_cosine": _fraction_json(
                self.aggregate.macro_outer_gradient_cosine
            ),
            "improving_family_count": self.aggregate.improving_family_count,
            "threshold_crossing_episode_fraction_report_only": _fraction_json(
                self.aggregate.threshold_crossing_fraction
            ),
            "identity_bit_exact": self.aggregate.identity.bit_exact,
        }


@dataclass(frozen=True, slots=True)
class MechanismEvaluation:
    mechanism_id: str
    shared_reason_codes: tuple[str, ...]
    active_support_episode_fraction: Fraction
    candidate_proximal_active_episode_fraction_among_active: Fraction
    unique_nonclean_probe_episode_count: int
    space_evaluations: tuple[SpaceEvaluation, ...]
    o4_active_episode_count: int
    o4_contribution_episode_count: int

    @property
    def shared_eligible(self) -> bool:
        return not self.shared_reason_codes

    @property
    def eligible(self) -> bool:
        return self.shared_eligible and any(item.eligible for item in self.space_evaluations)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "eligible": self.eligible,
            "shared_eligible": self.shared_eligible,
            "shared_reason_codes": list(self.shared_reason_codes),
            "unique_nonclean_probe_episode_count": self.unique_nonclean_probe_episode_count,
            "active_support_episode_fraction": _fraction_json(
                self.active_support_episode_fraction
            ),
            "candidate_proximal_active_episode_fraction_among_active": _fraction_json(
                self.candidate_proximal_active_episode_fraction_among_active
            ),
            "space_evaluations": [item.to_receipt() for item in self.space_evaluations],
            "o4_active_episode_count": self.o4_active_episode_count,
            "o4_contribution_episode_count": self.o4_contribution_episode_count,
        }


@dataclass(frozen=True, slots=True)
class StageCReceipt:
    replicate_id: str
    mechanism_evaluation: MechanismEvaluation
    eligible_space_ids: tuple[str, ...]
    scientific_status: Literal["scientific_eligible", "scientific_no_eligible"]

    @property
    def protocol_status(self) -> str:
        return "protocol_complete"

    def to_receipt(self) -> dict[str, Any]:
        allowed = self.scientific_status == "scientific_eligible"
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "receipt_type": "stage_c0_train_only_signal_gate",
            "stage": "C0",
            "replicate_id": self.replicate_id,
            "protocol_status": self.protocol_status,
            "scientific_status": self.scientific_status,
            "development_only": True,
            "paper_result": False,
            "data_role": "train_fixed_Pilot64",
            "validation_access_count": 0,
            "test_access_count": 0,
            "mechanism_evaluation": self.mechanism_evaluation.to_receipt(),
            "eligible_space_ids": list(self.eligible_space_ids),
            "stage_c1_allowed": allowed,
            "stage_c_r1_r2_allowed": False,
            "formal_test_allowed": False,
            "exit_semantics": (
                "continue_to_predefined_train_only_C1_screen"
                if allowed
                else "normal_scientific_early_stop_exit_0"
            ),
        }


def evaluate_stage_c0_science_gate(
    evidence: Mapping[str, Any],
    gate_config: StageCGateConfig | Mapping[str, Any] | None = None,
) -> StageCReceipt:
    """Validate and evaluate C0 without ranking candidates or side effects."""

    gate = (
        default_gate_config()
        if gate_config is None
        else gate_config
        if isinstance(gate_config, StageCGateConfig)
        else StageCGateConfig.from_mapping(gate_config)
    )
    parsed = StageCReplicateEvidence.from_mapping(evidence, gate)
    shared = _shared_failures(parsed.mechanism, gate)
    evaluations_list: list[SpaceEvaluation] = []
    for space in parsed.mechanism.space_aggregates:
        space_failures = _space_failures(space, gate)
        evaluations_list.append(
            SpaceEvaluation(
                parameter_space=space.parameter_space,
                eligible=not shared and not space_failures,
                reason_codes=shared + space_failures,
                positive_dataset_cosine_count=space.dataset_count_strictly_above(
                    gate.dataset_cosine_strictly_greater_than
                ),
                aggregate=space,
            )
        )
    evaluations = tuple(evaluations_list)
    # Frozen roster order is preserved.  This is an eligibility report, not a
    # Top-K search or ranking over duplicated mechanism candidates.
    eligible_space_ids = tuple(item.parameter_space for item in evaluations if item.eligible)
    mechanism = MechanismEvaluation(
        mechanism_id=parsed.mechanism.mechanism_id,
        shared_reason_codes=shared,
        active_support_episode_fraction=parsed.mechanism.active_fraction,
        candidate_proximal_active_episode_fraction_among_active=(
            parsed.mechanism.proximal_fraction_among_active
        ),
        unique_nonclean_probe_episode_count=parsed.mechanism.nonclean_probe_episode_count,
        space_evaluations=evaluations,
        o4_active_episode_count=parsed.mechanism.o4_active_episode_count,
        o4_contribution_episode_count=parsed.mechanism.o4_contribution_episode_count,
    )
    return StageCReceipt(
        replicate_id=parsed.replicate_id,
        mechanism_evaluation=mechanism,
        eligible_space_ids=eligible_space_ids,
        scientific_status=(
            "scientific_eligible" if eligible_space_ids else "scientific_no_eligible"
        ),
    )


def evaluate_stage_c_replicate(
    evidence: Mapping[str, Any],
    gate_config: StageCGateConfig | Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return eligible parameter-space IDs in their frozen roster order."""

    return evaluate_stage_c0_science_gate(evidence, gate_config).eligible_space_ids


@dataclass(frozen=True, slots=True)
class StageCAuthorization:
    stage_c1_allowed: bool
    parameter_space_ids: tuple[str, ...]
    reason: str
    stage_c_r1_r2_allowed: bool = False
    formal_test_allowed: bool = False


def authorize_stage_c_followup(receipt: StageCReceipt) -> StageCAuthorization:
    """Authorize only the predefined train-side C1 screen, without side effects."""

    if not isinstance(receipt, StageCReceipt) or receipt.protocol_status != "protocol_complete":
        raise StageCProtocolError("authorization requires a protocol-complete StageCReceipt")
    derived = tuple(
        item.parameter_space
        for item in receipt.mechanism_evaluation.space_evaluations
        if item.eligible
    )
    if derived != receipt.eligible_space_ids:
        raise StageCProtocolError("receipt eligibility is internally inconsistent")
    if receipt.scientific_status == "scientific_no_eligible":
        if receipt.eligible_space_ids:
            raise StageCProtocolError("negative receipt contains eligible spaces")
        return StageCAuthorization(False, (), "scientific_no_eligible")
    if receipt.scientific_status != "scientific_eligible" or not receipt.eligible_space_ids:
        raise StageCProtocolError("positive receipt has no eligible parameter space")
    return StageCAuthorization(
        True,
        receipt.eligible_space_ids,
        "scientific_eligible_for_predefined_train_only_C1",
    )


__all__ = [
    "GATE_CONFIG_FIELDS",
    "MECHANISM_FIELDS",
    "SCHEMA_VERSION",
    "SPACE_FIELDS",
    "MechanismEvaluation",
    "SpaceEvaluation",
    "StageCAuthorization",
    "StageCGateConfig",
    "StageCProtocolError",
    "StageCReceipt",
    "StageCReplicateEvidence",
    "authorize_stage_c_followup",
    "default_gate_config",
    "evaluate_stage_c0_science_gate",
    "evaluate_stage_c_replicate",
]
