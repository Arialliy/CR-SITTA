"""Pure scientific gate for the frozen P3 formal Stage-A protocol.

Protocol integrity is an input precondition and is deliberately not converted
into a negative scientific result.  A complete, valid run with no eligible
candidate returns ``scientific_no_eligible`` normally.  This module performs
no filesystem, model, CUDA, or dataset operations and never authorizes the
later CR-SITTA Stage 2.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import cmp_to_key
import math
from typing import Any, Literal

from analysis.d0_v3_formal_contract import FROZEN_CANDIDATES, FormalCandidate


DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CORRUPTION_FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "low_contrast",
    "stripe_noise",
)
REPLICATE_IDS = ("R0", "R1", "R2")
EPISODES_PER_CANDIDATE_REPLICATE = 3 * 13 * 64
FINE_ALIGNMENT_GROUP_COUNT = 20
ALIGNMENT_OBSERVATIONS_PER_CANDIDATE_REPLICATE = (
    EPISODES_PER_CANDIDATE_REPLICATE * FINE_ALIGNMENT_GROUP_COUNT
)
SAFETY_STRATA = (
    "overall",
    "nonclean",
    "clean",
    *(f"dataset:{dataset}" for dataset in DATASETS),
    *(f"corruption_family:{family}" for family in CORRUPTION_FAMILIES),
)

TOLERANCE = Decimal("1e-12")
NONCLEAN_MEAN_MINIMUM = Decimal("0.002")
OVERALL_MEAN_MINIMUM = Decimal("0.001")
WORST_DATASET_MINIMUM = Decimal("-0.002")
WORST_FAMILY_MINIMUM = Decimal("-0.005")
CLEAN_MACRO_MINIMUM = Decimal("-0.002")
CLEAN_DATASET_MINIMUM = Decimal("-0.005")
NONCLEAN_PD_MINIMUM = Decimal("-0.010")
DATASET_PD_MINIMUM = Decimal("-0.020")
CLEAN_PD_MINIMUM = Decimal("-0.010")
FA_ABSOLUTE_ALLOWANCE = Decimal("10.0")
FA_SOURCE_MULTIPLIER = Decimal("0.25")
FOREGROUND_DELTA_MAXIMUM = Decimal("0.001")
FOREGROUND_SOURCE_MULTIPLIER = Decimal("1.20")
FOREGROUND_EPSILON = Decimal("0.000001")
FUNCTIONAL_LOGIT_THRESHOLD = Decimal("0.000001")
ALIGNMENT_MACRO_COSINE_MINIMUM = Decimal("0.05")


class D0V3ScienceGateError(ValueError):
    """Evidence has an invalid or incomplete scientific-gate schema."""


class D0V3ProtocolGateError(D0V3ScienceGateError):
    """Protocol integrity failed; this is not a scientific negative result."""


def _decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise D0V3ScienceGateError(f"{field} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise D0V3ScienceGateError(f"{field} must be finite")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise D0V3ScienceGateError(f"{field} must be finite") from exc
    if not result.is_finite():
        raise D0V3ScienceGateError(f"{field} must be finite")
    return result


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise D0V3ScienceGateError(f"{field} must be an integer")
    if value < minimum:
        raise D0V3ScienceGateError(f"{field} must be >= {minimum}")
    return value


def _exact_keys(value: Any, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0V3ScienceGateError(f"{field} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected, key=str)
    if missing or unknown:
        raise D0V3ScienceGateError(
            f"{field} fields must be exact; missing={missing}, unknown={unknown}"
        )
    return value


def _ordered_numeric_map(
    value: Any, keys: Sequence[str], *, field: str
) -> tuple[tuple[str, Decimal], ...]:
    mapping = _exact_keys(value, set(keys), field=field)
    return tuple(
        (key, _decimal(mapping[key], field=f"{field}.{key}")) for key in keys
    )


def _as_dict(value: tuple[tuple[str, Decimal], ...]) -> dict[str, Decimal]:
    return dict(value)


def _strictly_greater_than(value: Decimal, threshold: Decimal) -> bool:
    """Apply the frozen comparison tolerance at a strict gate boundary."""

    return value > threshold + TOLERANCE


def _below_minimum(value: Decimal, minimum: Decimal) -> bool:
    """Return whether a value misses an inclusive minimum beyond tolerance."""

    return value < minimum - TOLERANCE


def _above_maximum(value: Decimal, maximum: Decimal) -> bool:
    """Return whether a value exceeds an inclusive maximum beyond tolerance."""

    return value > maximum + TOLERANCE


def _validate_bounded_delta(value: Decimal, *, field: str) -> None:
    if value < -1 or value > 1:
        raise D0V3ScienceGateError(f"{field} must be in [-1, 1]")


@dataclass(frozen=True)
class EpisodeActivity:
    finite_gradient_episodes: int
    parameter_changed_episodes: int
    functional_logit_changed_episodes: int
    threshold_crossing_episodes: int
    metric_sufficient_count_changed_episodes: int
    entropy_decrease_episodes: int
    both_gradients_nonzero_episodes: int
    fine_group_alignment_observation_count: int

    def __post_init__(self) -> None:
        for field in (
            "finite_gradient_episodes",
            "parameter_changed_episodes",
            "functional_logit_changed_episodes",
            "threshold_crossing_episodes",
            "metric_sufficient_count_changed_episodes",
            "entropy_decrease_episodes",
        ):
            count = _integer(getattr(self, field), field=f"activity.{field}")
            if count > EPISODES_PER_CANDIDATE_REPLICATE:
                raise D0V3ScienceGateError(
                    f"activity.{field} exceeds {EPISODES_PER_CANDIDATE_REPLICATE}"
                )
        alignment_count = _integer(
            self.fine_group_alignment_observation_count,
            field="activity.fine_group_alignment_observation_count",
        )
        if alignment_count != ALIGNMENT_OBSERVATIONS_PER_CANDIDATE_REPLICATE:
            raise D0V3ScienceGateError(
                "activity.fine_group_alignment_observation_count must equal "
                f"{ALIGNMENT_OBSERVATIONS_PER_CANDIDATE_REPLICATE}"
            )
        nonzero = _integer(
            self.both_gradients_nonzero_episodes,
            field="activity.both_gradients_nonzero_episodes",
        )
        if nonzero > EPISODES_PER_CANDIDATE_REPLICATE:
            raise D0V3ScienceGateError(
                "activity.both_gradients_nonzero_episodes exceeds episode denominator"
            )


@dataclass(frozen=True)
class ReplicateEvidence:
    candidate: FormalCandidate
    replicate_id: Literal["R0", "R1", "R2"]
    episode_count: int
    nonclean_macro_delta_iou: Decimal
    overall_macro_delta_iou: Decimal
    dataset_nonclean_delta_iou: tuple[tuple[str, Decimal], ...]
    family_delta_iou: tuple[tuple[str, Decimal], ...]
    clean_macro_delta_iou: Decimal
    clean_dataset_delta_iou: tuple[tuple[str, Decimal], ...]
    nonclean_macro_delta_pd: Decimal
    dataset_nonclean_delta_pd: tuple[tuple[str, Decimal], ...]
    clean_macro_delta_pd: Decimal
    source_fa_per_million: tuple[tuple[str, Decimal], ...]
    fa_delta_per_million: tuple[tuple[str, Decimal], ...]
    source_foreground_fraction: tuple[tuple[str, Decimal], ...]
    adapted_foreground_fraction: tuple[tuple[str, Decimal], ...]
    functional_logit_threshold: Decimal
    activity: EpisodeActivity
    alignment_macro_cosine: Decimal
    alignment_dataset_median_cosine: tuple[tuple[str, Decimal], ...]

    def __post_init__(self) -> None:
        if self.candidate not in FROZEN_CANDIDATES:
            raise D0V3ScienceGateError("evidence candidate is not frozen")
        if self.replicate_id not in REPLICATE_IDS:
            raise D0V3ScienceGateError("replicate_id must be R0, R1, or R2")
        if self.episode_count != EPISODES_PER_CANDIDATE_REPLICATE:
            raise D0V3ScienceGateError(
                f"episode_count must equal {EPISODES_PER_CANDIDATE_REPLICATE}"
            )
        scalar_deltas = (
            ("nonclean_macro_delta_iou", self.nonclean_macro_delta_iou),
            ("overall_macro_delta_iou", self.overall_macro_delta_iou),
            ("clean_macro_delta_iou", self.clean_macro_delta_iou),
            ("nonclean_macro_delta_pd", self.nonclean_macro_delta_pd),
            ("clean_macro_delta_pd", self.clean_macro_delta_pd),
        )
        for field, value in scalar_deltas:
            _validate_bounded_delta(_decimal(value, field=field), field=field)
        expected_maps = (
            ("dataset_nonclean_delta_iou", self.dataset_nonclean_delta_iou, DATASETS),
            ("family_delta_iou", self.family_delta_iou, CORRUPTION_FAMILIES),
            ("clean_dataset_delta_iou", self.clean_dataset_delta_iou, DATASETS),
            ("dataset_nonclean_delta_pd", self.dataset_nonclean_delta_pd, DATASETS),
            ("source_fa_per_million", self.source_fa_per_million, SAFETY_STRATA),
            ("fa_delta_per_million", self.fa_delta_per_million, SAFETY_STRATA),
            (
                "source_foreground_fraction",
                self.source_foreground_fraction,
                SAFETY_STRATA,
            ),
            (
                "adapted_foreground_fraction",
                self.adapted_foreground_fraction,
                SAFETY_STRATA,
            ),
            (
                "alignment_dataset_median_cosine",
                self.alignment_dataset_median_cosine,
                DATASETS,
            ),
        )
        for field, items, keys in expected_maps:
            if tuple(key for key, _ in items) != tuple(keys):
                raise D0V3ScienceGateError(f"{field} keys/order must be canonical")
            for key, value in items:
                numeric = _decimal(value, field=f"{field}.{key}")
                if "delta_iou" in field or "delta_pd" in field:
                    _validate_bounded_delta(numeric, field=f"{field}.{key}")
                if field == "source_fa_per_million" and numeric < 0:
                    raise D0V3ScienceGateError(
                        f"source_fa_per_million.{key} must be non-negative"
                    )
                if "foreground_fraction" in field and not 0 <= numeric <= 1:
                    raise D0V3ScienceGateError(
                        f"{field}.{key} must be in [0, 1]"
                    )
                if "cosine" in field and not -1 <= numeric <= 1:
                    raise D0V3ScienceGateError(
                        f"{field}.{key} must be in [-1, 1]"
                    )
        threshold = _decimal(
            self.functional_logit_threshold, field="functional_logit_threshold"
        )
        if threshold != FUNCTIONAL_LOGIT_THRESHOLD:
            raise D0V3ScienceGateError(
                "functional_logit_threshold must equal 1e-6"
            )
        alignment = _decimal(
            self.alignment_macro_cosine, field="alignment_macro_cosine"
        )
        if not -1 <= alignment <= 1:
            raise D0V3ScienceGateError(
                "alignment_macro_cosine must be in [-1, 1]"
            )


_EVIDENCE_TOP_KEYS = {
    "schema_version",
    "artifact_type",
    "candidate",
    "replicate_id",
    "episode_count",
    "metrics",
    "safety",
    "activity",
    "alignment",
}
_METRIC_KEYS = {
    "nonclean_macro_delta_iou",
    "overall_macro_delta_iou",
    "dataset_nonclean_delta_iou",
    "family_delta_iou",
    "clean_macro_delta_iou",
    "clean_dataset_delta_iou",
    "nonclean_macro_delta_pd",
    "dataset_nonclean_delta_pd",
    "clean_macro_delta_pd",
}


def parse_replicate_evidence(value: Mapping[str, Any]) -> ReplicateEvidence:
    """Strictly parse one aggregate candidate/replicate evidence record."""

    root = _exact_keys(value, _EVIDENCE_TOP_KEYS, field="evidence")
    if root["schema_version"] != 3:
        raise D0V3ScienceGateError("evidence.schema_version must equal 3")
    if root["artifact_type"] != "cr_sitta_d0_v3_stage_a_replicate_evidence":
        raise D0V3ScienceGateError("evidence.artifact_type is not frozen")
    candidate_value = _exact_keys(
        root["candidate"],
        {"candidate_id", "optimizer", "learning_rate"},
        field="evidence.candidate",
    )
    candidate_matches = tuple(
        candidate
        for candidate in FROZEN_CANDIDATES
        if candidate.candidate_id == candidate_value["candidate_id"]
        and candidate.optimizer == candidate_value["optimizer"]
        and Decimal(str(candidate.learning_rate))
        == _decimal(candidate_value["learning_rate"], field="candidate.learning_rate")
    )
    if len(candidate_matches) != 1:
        raise D0V3ScienceGateError("candidate tuple is not one frozen candidate")
    metrics = _exact_keys(root["metrics"], _METRIC_KEYS, field="evidence.metrics")
    safety = _exact_keys(
        root["safety"],
        {
            "source_fa_per_million",
            "fa_delta_per_million",
            "source_foreground_fraction",
            "adapted_foreground_fraction",
        },
        field="evidence.safety",
    )
    activity = _exact_keys(
        root["activity"],
        {
            "finite_gradient_episodes",
            "parameter_changed_episodes",
            "functional_logit_threshold",
            "functional_logit_changed_episodes",
            "threshold_crossing_episodes",
            "metric_sufficient_count_changed_episodes",
            "entropy_decrease_episodes",
            "both_gradients_nonzero_episodes",
            "fine_group_alignment_observation_count",
        },
        field="evidence.activity",
    )
    alignment = _exact_keys(
        root["alignment"],
        {"macro_cosine", "dataset_median_cosine"},
        field="evidence.alignment",
    )
    episode_activity = EpisodeActivity(
        finite_gradient_episodes=_integer(
            activity["finite_gradient_episodes"],
            field="activity.finite_gradient_episodes",
        ),
        parameter_changed_episodes=_integer(
            activity["parameter_changed_episodes"],
            field="activity.parameter_changed_episodes",
        ),
        functional_logit_changed_episodes=_integer(
            activity["functional_logit_changed_episodes"],
            field="activity.functional_logit_changed_episodes",
        ),
        threshold_crossing_episodes=_integer(
            activity["threshold_crossing_episodes"],
            field="activity.threshold_crossing_episodes",
        ),
        metric_sufficient_count_changed_episodes=_integer(
            activity["metric_sufficient_count_changed_episodes"],
            field="activity.metric_sufficient_count_changed_episodes",
        ),
        entropy_decrease_episodes=_integer(
            activity["entropy_decrease_episodes"],
            field="activity.entropy_decrease_episodes",
        ),
        both_gradients_nonzero_episodes=_integer(
            activity["both_gradients_nonzero_episodes"],
            field="activity.both_gradients_nonzero_episodes",
        ),
        fine_group_alignment_observation_count=_integer(
            activity["fine_group_alignment_observation_count"],
            field="activity.fine_group_alignment_observation_count",
        ),
    )
    replicate_id = root["replicate_id"]
    if replicate_id not in REPLICATE_IDS:
        raise D0V3ScienceGateError("replicate_id must be R0, R1, or R2")
    return ReplicateEvidence(
        candidate=candidate_matches[0],
        replicate_id=replicate_id,
        episode_count=_integer(root["episode_count"], field="episode_count"),
        nonclean_macro_delta_iou=_decimal(
            metrics["nonclean_macro_delta_iou"],
            field="metrics.nonclean_macro_delta_iou",
        ),
        overall_macro_delta_iou=_decimal(
            metrics["overall_macro_delta_iou"],
            field="metrics.overall_macro_delta_iou",
        ),
        dataset_nonclean_delta_iou=_ordered_numeric_map(
            metrics["dataset_nonclean_delta_iou"],
            DATASETS,
            field="metrics.dataset_nonclean_delta_iou",
        ),
        family_delta_iou=_ordered_numeric_map(
            metrics["family_delta_iou"],
            CORRUPTION_FAMILIES,
            field="metrics.family_delta_iou",
        ),
        clean_macro_delta_iou=_decimal(
            metrics["clean_macro_delta_iou"],
            field="metrics.clean_macro_delta_iou",
        ),
        clean_dataset_delta_iou=_ordered_numeric_map(
            metrics["clean_dataset_delta_iou"],
            DATASETS,
            field="metrics.clean_dataset_delta_iou",
        ),
        nonclean_macro_delta_pd=_decimal(
            metrics["nonclean_macro_delta_pd"],
            field="metrics.nonclean_macro_delta_pd",
        ),
        dataset_nonclean_delta_pd=_ordered_numeric_map(
            metrics["dataset_nonclean_delta_pd"],
            DATASETS,
            field="metrics.dataset_nonclean_delta_pd",
        ),
        clean_macro_delta_pd=_decimal(
            metrics["clean_macro_delta_pd"],
            field="metrics.clean_macro_delta_pd",
        ),
        source_fa_per_million=_ordered_numeric_map(
            safety["source_fa_per_million"],
            SAFETY_STRATA,
            field="safety.source_fa_per_million",
        ),
        fa_delta_per_million=_ordered_numeric_map(
            safety["fa_delta_per_million"],
            SAFETY_STRATA,
            field="safety.fa_delta_per_million",
        ),
        source_foreground_fraction=_ordered_numeric_map(
            safety["source_foreground_fraction"],
            SAFETY_STRATA,
            field="safety.source_foreground_fraction",
        ),
        adapted_foreground_fraction=_ordered_numeric_map(
            safety["adapted_foreground_fraction"],
            SAFETY_STRATA,
            field="safety.adapted_foreground_fraction",
        ),
        functional_logit_threshold=_decimal(
            activity["functional_logit_threshold"],
            field="activity.functional_logit_threshold",
        ),
        activity=episode_activity,
        alignment_macro_cosine=_decimal(
            alignment["macro_cosine"], field="alignment.macro_cosine"
        ),
        alignment_dataset_median_cosine=_ordered_numeric_map(
            alignment["dataset_median_cosine"],
            DATASETS,
            field="alignment.dataset_median_cosine",
        ),
    )


_R0_RECEIPT_KEYS = {
    "schema_version",
    "receipt_type",
    "evaluation_phase",
    "protocol_status",
    "formal_stage_a_protocol_complete",
    "scientific_status",
    "stage2_allowed",
    "eligible_candidates",
    "rejected_candidates",
    "ranking",
    "required_followup_replicates",
}


def _parse_frozen_candidate(value: Any, *, field: str) -> FormalCandidate:
    mapping = _exact_keys(
        value,
        {"candidate_id", "optimizer", "learning_rate"},
        field=field,
    )
    matches = tuple(
        candidate
        for candidate in FROZEN_CANDIDATES
        if candidate.candidate_id == mapping["candidate_id"]
        and candidate.optimizer == mapping["optimizer"]
        and Decimal(str(candidate.learning_rate))
        == _decimal(mapping["learning_rate"], field=f"{field}.learning_rate")
    )
    if len(matches) != 1:
        raise D0V3ProtocolGateError(f"{field} is not one frozen candidate")
    return matches[0]


def _sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise D0V3ProtocolGateError(f"{field} must be a sequence")
    return tuple(value)


def parse_r0_eligibility_receipt(value: Any) -> tuple[FormalCandidate, ...]:
    """Return the exact R0-eligible candidate subset from a canonical receipt.

    R1/R2 aggregation uses this parser rather than accepting a caller-supplied
    list.  It validates the complete R0 candidate partition and the early-stop
    versus follow-up state, so a missing, extra, reordered, or contradictory
    candidate fails as a protocol error.
    """

    root = _exact_keys(value, _R0_RECEIPT_KEYS, field="r0_receipt")
    if root["schema_version"] != 3:
        raise D0V3ProtocolGateError("r0_receipt.schema_version must equal 3")
    if root["receipt_type"] != "cr_sitta_d0_v3_stage_a_science_decision":
        raise D0V3ProtocolGateError("r0_receipt.receipt_type is not frozen")
    if root["evaluation_phase"] != "R0" or root["protocol_status"] != "passed":
        raise D0V3ProtocolGateError("eligibility receipt must be a passed R0 decision")
    if root["stage2_allowed"] is not False:
        raise D0V3ProtocolGateError("R0 eligibility receipt must not authorize Stage 2")
    if _sequence(root["ranking"], field="r0_receipt.ranking"):
        raise D0V3ProtocolGateError("R0 eligibility receipt cannot contain a ranking")

    eligible = tuple(
        _parse_frozen_candidate(item, field=f"r0_receipt.eligible_candidates[{index}]")
        for index, item in enumerate(
            _sequence(
                root["eligible_candidates"],
                field="r0_receipt.eligible_candidates",
            )
        )
    )
    if len(set(eligible)) != len(eligible):
        raise D0V3ProtocolGateError("R0 eligible candidates are duplicated")

    rejected_entries = _sequence(
        root["rejected_candidates"], field="r0_receipt.rejected_candidates"
    )
    rejected: list[FormalCandidate] = []
    for index, raw in enumerate(rejected_entries):
        entry = _exact_keys(
            raw,
            {"candidate", "failed_gates"},
            field=f"r0_receipt.rejected_candidates[{index}]",
        )
        candidate = _parse_frozen_candidate(
            entry["candidate"],
            field=f"r0_receipt.rejected_candidates[{index}].candidate",
        )
        failures = _sequence(
            entry["failed_gates"],
            field=f"r0_receipt.rejected_candidates[{index}].failed_gates",
        )
        if not failures or any(
            not isinstance(reason, str) or not reason for reason in failures
        ):
            raise D0V3ProtocolGateError(
                "each R0-rejected candidate must have non-empty reason codes"
            )
        rejected.append(candidate)
    if len(set(rejected)) != len(rejected):
        raise D0V3ProtocolGateError("R0 rejected candidates are duplicated")

    eligible_set = set(eligible)
    expected_eligible = tuple(
        candidate for candidate in FROZEN_CANDIDATES if candidate in eligible_set
    )
    expected_rejected = tuple(
        candidate for candidate in FROZEN_CANDIDATES if candidate not in eligible_set
    )
    if eligible != expected_eligible or tuple(rejected) != expected_rejected:
        raise D0V3ProtocolGateError(
            "R0 receipt candidate partition/order is not the frozen ten-candidate order"
        )

    followups = _sequence(
        root["required_followup_replicates"],
        field="r0_receipt.required_followup_replicates",
    )
    if eligible:
        if (
            root["formal_stage_a_protocol_complete"] is not False
            or root["scientific_status"] != "scientific_pending_R1_R2"
            or followups != ("R1", "R2")
        ):
            raise D0V3ProtocolGateError(
                "non-empty R0 eligibility receipt must require R1/R2"
            )
    elif (
        root["formal_stage_a_protocol_complete"] is not True
        or root["scientific_status"] != "scientific_no_eligible"
        or followups
    ):
        raise D0V3ProtocolGateError(
            "empty R0 eligibility receipt must be a completed early-stop"
        )
    return eligible


def _rate(numerator: int, denominator: int) -> Decimal:
    return Decimal(numerator) / Decimal(denominator)


def evaluate_replicate_hard_gates(evidence: ReplicateEvidence) -> tuple[str, ...]:
    """Return stable reason codes; an empty tuple means the replicate passes."""

    if not isinstance(evidence, ReplicateEvidence):
        raise D0V3ScienceGateError("evidence must be ReplicateEvidence")
    failures: list[str] = []
    dataset_iou = tuple(_as_dict(evidence.dataset_nonclean_delta_iou).values())
    family_iou = tuple(_as_dict(evidence.family_delta_iou).values())
    clean_dataset_iou = tuple(_as_dict(evidence.clean_dataset_delta_iou).values())
    dataset_pd = tuple(_as_dict(evidence.dataset_nonclean_delta_pd).values())
    alignment_dataset = tuple(
        _as_dict(evidence.alignment_dataset_median_cosine).values()
    )
    if not _strictly_greater_than(evidence.nonclean_macro_delta_iou, Decimal(0)):
        failures.append("nonclean_macro_iou_nonpositive")
    if sum(_strictly_greater_than(value, Decimal(0)) for value in dataset_iou) < 2:
        failures.append("positive_dataset_coverage")
    if _below_minimum(min(dataset_iou), WORST_DATASET_MINIMUM):
        failures.append("worst_dataset_iou")
    if sum(_strictly_greater_than(value, Decimal(0)) for value in family_iou) < 3:
        failures.append("positive_family_coverage")
    if _below_minimum(min(family_iou), WORST_FAMILY_MINIMUM):
        failures.append("worst_family_iou")
    if not _strictly_greater_than(evidence.overall_macro_delta_iou, Decimal(0)):
        failures.append("overall_macro_iou_nonpositive")
    if _below_minimum(evidence.clean_macro_delta_iou, CLEAN_MACRO_MINIMUM):
        failures.append("clean_macro_iou_safety")
    if _below_minimum(min(clean_dataset_iou), CLEAN_DATASET_MINIMUM):
        failures.append("clean_dataset_iou_safety")
    if _below_minimum(evidence.nonclean_macro_delta_pd, NONCLEAN_PD_MINIMUM):
        failures.append("nonclean_pd_safety")
    if _below_minimum(min(dataset_pd), DATASET_PD_MINIMUM):
        failures.append("dataset_pd_safety")
    if _below_minimum(evidence.clean_macro_delta_pd, CLEAN_PD_MINIMUM):
        failures.append("clean_pd_safety")

    source_fa = _as_dict(evidence.source_fa_per_million)
    fa_delta = _as_dict(evidence.fa_delta_per_million)
    if any(
        _above_maximum(
            fa_delta[key],
            FA_ABSOLUTE_ALLOWANCE + FA_SOURCE_MULTIPLIER * source_fa[key],
        )
        for key in SAFETY_STRATA
    ):
        failures.append("fa_inflation")
    source_fg = _as_dict(evidence.source_foreground_fraction)
    adapted_fg = _as_dict(evidence.adapted_foreground_fraction)
    if any(
        _above_maximum(
            adapted_fg[key] - source_fg[key], FOREGROUND_DELTA_MAXIMUM
        )
        or _above_maximum(
            adapted_fg[key],
            FOREGROUND_SOURCE_MULTIPLIER * source_fg[key] + FOREGROUND_EPSILON,
        )
        for key in SAFETY_STRATA
    ):
        failures.append("foreground_inflation")

    activity = evidence.activity
    denominator = EPISODES_PER_CANDIDATE_REPLICATE
    if activity.finite_gradient_episodes != denominator:
        failures.append("finite_gradient_fraction")
    if _below_minimum(
        _rate(activity.parameter_changed_episodes, denominator), Decimal("0.95")
    ):
        failures.append("parameter_changed_fraction")
    if (
        _below_minimum(
            _rate(activity.functional_logit_changed_episodes, denominator),
            Decimal("0.10"),
        )
    ):
        failures.append("functional_logit_changed_fraction")
    if _below_minimum(
        _rate(activity.threshold_crossing_episodes, denominator), Decimal("0.02")
    ):
        failures.append("threshold_crossing_fraction")
    if (
        _below_minimum(
            _rate(activity.metric_sufficient_count_changed_episodes, denominator),
            Decimal("0.02"),
        )
    ):
        failures.append("metric_sufficient_count_changed_fraction")
    if _below_minimum(
        _rate(activity.entropy_decrease_episodes, denominator), Decimal("0.80")
    ):
        failures.append("entropy_decrease_fraction")
    if _below_minimum(
        evidence.alignment_macro_cosine, ALIGNMENT_MACRO_COSINE_MINIMUM
    ):
        failures.append("alignment_macro_cosine")
    if sum(
        _strictly_greater_than(value, Decimal(0)) for value in alignment_dataset
    ) < 2:
        failures.append("alignment_dataset_coverage")
    if (
        _below_minimum(
            _rate(
                activity.both_gradients_nonzero_episodes,
                EPISODES_PER_CANDIDATE_REPLICATE,
            ),
            Decimal("0.80"),
        )
    ):
        failures.append("alignment_nonzero_gradient_fraction")
    return tuple(failures)


@dataclass(frozen=True)
class CandidateGateDecision:
    candidate: FormalCandidate
    eligible: bool
    failed_gates: tuple[str, ...]


@dataclass(frozen=True)
class CandidateRankingSummary:
    candidate: FormalCandidate
    mean_nonclean_delta_iou: Decimal
    worst_replicate_nonclean_delta_iou: Decimal
    minimum_dataset_mean_nonclean_delta_iou: Decimal
    mean_clean_delta_iou: Decimal
    mean_nonclean_fa_delta: Decimal
    mean_nonclean_foreground_inflation: Decimal
    mean_nonclean_delta_pd: Decimal


@dataclass(frozen=True)
class StageAScienceDecision:
    evaluation_phase: Literal["R0", "R0_R1_R2"]
    protocol_status: Literal["passed"]
    formal_stage_a_protocol_complete: bool
    scientific_status: Literal[
        "scientific_pending_R1_R2",
        "scientific_no_eligible",
        "scientific_passed",
    ]
    stage2_allowed: Literal[False]
    eligible_candidates: tuple[FormalCandidate, ...]
    rejected_candidates: tuple[CandidateGateDecision, ...]
    ranking: tuple[CandidateRankingSummary, ...]
    required_followup_replicates: tuple[str, ...]

    def to_receipt(self) -> dict[str, Any]:
        def candidate_value(candidate: FormalCandidate) -> dict[str, Any]:
            return {
                "candidate_id": candidate.candidate_id,
                "optimizer": candidate.optimizer,
                "learning_rate": candidate.learning_rate,
            }

        return {
            "schema_version": 3,
            "receipt_type": "cr_sitta_d0_v3_stage_a_science_decision",
            "evaluation_phase": self.evaluation_phase,
            "protocol_status": self.protocol_status,
            "formal_stage_a_protocol_complete": (
                self.formal_stage_a_protocol_complete
            ),
            "scientific_status": self.scientific_status,
            "stage2_allowed": False,
            "eligible_candidates": [
                candidate_value(candidate) for candidate in self.eligible_candidates
            ],
            "rejected_candidates": [
                {
                    "candidate": candidate_value(decision.candidate),
                    "failed_gates": list(decision.failed_gates),
                }
                for decision in self.rejected_candidates
            ],
            "ranking": [
                {
                    "candidate": candidate_value(summary.candidate),
                    "mean_nonclean_delta_iou": str(
                        summary.mean_nonclean_delta_iou
                    ),
                    "worst_replicate_nonclean_delta_iou": str(
                        summary.worst_replicate_nonclean_delta_iou
                    ),
                    "minimum_dataset_mean_nonclean_delta_iou": str(
                        summary.minimum_dataset_mean_nonclean_delta_iou
                    ),
                    "mean_clean_delta_iou": str(summary.mean_clean_delta_iou),
                    "mean_nonclean_fa_delta": str(
                        summary.mean_nonclean_fa_delta
                    ),
                    "mean_nonclean_foreground_inflation": str(
                        summary.mean_nonclean_foreground_inflation
                    ),
                    "mean_nonclean_delta_pd": str(
                        summary.mean_nonclean_delta_pd
                    ),
                }
                for summary in self.ranking
            ],
            "required_followup_replicates": list(
                self.required_followup_replicates
            ),
        }


def _coerce_evidence(value: Any) -> ReplicateEvidence:
    if isinstance(value, ReplicateEvidence):
        return value
    if isinstance(value, Mapping):
        return parse_replicate_evidence(value)
    raise D0V3ScienceGateError(
        "each evidence record must be ReplicateEvidence or a strict mapping"
    )


def _require_protocol_passed(protocol_status: str) -> None:
    if protocol_status != "passed":
        raise D0V3ProtocolGateError(
            "formal execution protocol did not pass; this is not a scientific result"
        )


def _unique_records(
    values: Iterable[ReplicateEvidence | Mapping[str, Any]],
) -> dict[tuple[FormalCandidate, str], ReplicateEvidence]:
    if isinstance(values, (str, bytes, Mapping)):
        raise D0V3ScienceGateError("evidence must be an iterable of records")
    result: dict[tuple[FormalCandidate, str], ReplicateEvidence] = {}
    for value in values:
        evidence = _coerce_evidence(value)
        key = (evidence.candidate, evidence.replicate_id)
        if key in result:
            raise D0V3ProtocolGateError(
                f"duplicate candidate/replicate evidence: {key}"
            )
        result[key] = evidence
    return result


def _require_all_r0(
    records: Mapping[tuple[FormalCandidate, str], ReplicateEvidence]
) -> None:
    expected = {(candidate, "R0") for candidate in FROZEN_CANDIDATES}
    observed = {key for key in records if key[1] == "R0"}
    if observed != expected:
        missing = sorted(
            candidate.candidate_id for candidate, _ in expected - observed
        )
        extra = sorted(candidate.candidate_id for candidate, _ in observed - expected)
        raise D0V3ProtocolGateError(
            f"R0 must contain all ten frozen candidates; missing={missing}, extra={extra}"
        )


def evaluate_stage_a_r0(
    evidence: Iterable[ReplicateEvidence | Mapping[str, Any]],
    *,
    protocol_status: str,
) -> StageAScienceDecision:
    """Filter R0 only; every passing candidate proceeds to independent R1/R2."""

    _require_protocol_passed(protocol_status)
    records = _unique_records(evidence)
    if any(replicate != "R0" for _, replicate in records):
        raise D0V3ProtocolGateError("R0 decision cannot include R1/R2 evidence")
    _require_all_r0(records)
    eligible: list[FormalCandidate] = []
    rejected: list[CandidateGateDecision] = []
    for candidate in FROZEN_CANDIDATES:
        failures = evaluate_replicate_hard_gates(records[(candidate, "R0")])
        if failures:
            rejected.append(CandidateGateDecision(candidate, False, failures))
        else:
            eligible.append(candidate)
    if not eligible:
        return StageAScienceDecision(
            evaluation_phase="R0",
            protocol_status="passed",
            formal_stage_a_protocol_complete=True,
            scientific_status="scientific_no_eligible",
            stage2_allowed=False,
            eligible_candidates=(),
            rejected_candidates=tuple(rejected),
            ranking=(),
            required_followup_replicates=(),
        )
    return StageAScienceDecision(
        evaluation_phase="R0",
        protocol_status="passed",
        formal_stage_a_protocol_complete=False,
        scientific_status="scientific_pending_R1_R2",
        stage2_allowed=False,
        eligible_candidates=tuple(eligible),
        rejected_candidates=tuple(rejected),
        ranking=(),
        required_followup_replicates=("R1", "R2"),
    )


def _mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        raise D0V3ScienceGateError("cannot average an empty sequence")
    return sum(values, Decimal(0)) / Decimal(len(values))


def _ranking_summary(
    candidate: FormalCandidate,
    replicates: Sequence[ReplicateEvidence],
) -> CandidateRankingSummary:
    dataset_means = tuple(
        _mean(
            tuple(_as_dict(replicate.dataset_nonclean_delta_iou)[dataset] for replicate in replicates)
        )
        for dataset in DATASETS
    )
    return CandidateRankingSummary(
        candidate=candidate,
        mean_nonclean_delta_iou=_mean(
            tuple(replicate.nonclean_macro_delta_iou for replicate in replicates)
        ),
        worst_replicate_nonclean_delta_iou=min(
            replicate.nonclean_macro_delta_iou for replicate in replicates
        ),
        minimum_dataset_mean_nonclean_delta_iou=min(dataset_means),
        mean_clean_delta_iou=_mean(
            tuple(replicate.clean_macro_delta_iou for replicate in replicates)
        ),
        mean_nonclean_fa_delta=_mean(
            tuple(
                _as_dict(replicate.fa_delta_per_million)["nonclean"]
                for replicate in replicates
            )
        ),
        mean_nonclean_foreground_inflation=_mean(
            tuple(
                _as_dict(replicate.adapted_foreground_fraction)["nonclean"]
                - _as_dict(replicate.source_foreground_fraction)["nonclean"]
                for replicate in replicates
            )
        ),
        mean_nonclean_delta_pd=_mean(
            tuple(replicate.nonclean_macro_delta_pd for replicate in replicates)
        ),
    )


def _numeric_compare(left: Decimal, right: Decimal, *, higher: bool) -> int:
    # Ranking must be a transitive total order.  The frozen comparison
    # tolerance is reserved for science-gate boundaries; pairwise approximate
    # equality is non-transitive and makes sorting input-order dependent.
    if left == right:
        return 0
    if higher:
        return -1 if left > right else 1
    return -1 if left < right else 1


def _ranking_compare(
    left: CandidateRankingSummary, right: CandidateRankingSummary
) -> int:
    fields = (
        (left.mean_nonclean_delta_iou, right.mean_nonclean_delta_iou, True),
        (
            left.worst_replicate_nonclean_delta_iou,
            right.worst_replicate_nonclean_delta_iou,
            True,
        ),
        (
            left.minimum_dataset_mean_nonclean_delta_iou,
            right.minimum_dataset_mean_nonclean_delta_iou,
            True,
        ),
        (left.mean_clean_delta_iou, right.mean_clean_delta_iou, True),
        (left.mean_nonclean_fa_delta, right.mean_nonclean_fa_delta, False),
        (
            left.mean_nonclean_foreground_inflation,
            right.mean_nonclean_foreground_inflation,
            False,
        ),
        (left.mean_nonclean_delta_pd, right.mean_nonclean_delta_pd, True),
    )
    for left_value, right_value, higher in fields:
        result = _numeric_compare(left_value, right_value, higher=higher)
        if result:
            return result
    lr_result = _numeric_compare(
        Decimal(str(left.candidate.learning_rate)),
        Decimal(str(right.candidate.learning_rate)),
        higher=False,
    )
    if lr_result:
        return lr_result
    optimizer_order = {"Adam": 0, "SGD": 1}
    left_optimizer = optimizer_order[left.candidate.optimizer]
    right_optimizer = optimizer_order[right.candidate.optimizer]
    if left_optimizer != right_optimizer:
        return -1 if left_optimizer < right_optimizer else 1
    if left.candidate.candidate_id == right.candidate.candidate_id:
        return 0
    return -1 if left.candidate.candidate_id < right.candidate.candidate_id else 1


def rank_eligible_candidates(
    summaries: Iterable[CandidateRankingSummary],
) -> tuple[CandidateRankingSummary, ...]:
    """Apply the complete frozen ranking order as a transitive total order."""

    if isinstance(summaries, (str, bytes, Mapping)):
        raise D0V3ScienceGateError("summaries must be an iterable")
    values = tuple(summaries)
    if any(not isinstance(value, CandidateRankingSummary) for value in values):
        raise D0V3ScienceGateError(
            "all ranking values must be CandidateRankingSummary"
        )
    candidates = tuple(value.candidate for value in values)
    if len(set(candidates)) != len(candidates):
        raise D0V3ScienceGateError("ranking candidates must be unique")
    return tuple(sorted(values, key=cmp_to_key(_ranking_compare)))


def evaluate_stage_a_final(
    evidence: Iterable[ReplicateEvidence | Mapping[str, Any]],
    *,
    protocol_status: str,
) -> StageAScienceDecision:
    """Evaluate the early-stop R0 or exact eligible-only R0/R1/R2 protocol."""

    _require_protocol_passed(protocol_status)
    records = _unique_records(evidence)
    _require_all_r0(records)
    r0_pass = {
        candidate
        for candidate in FROZEN_CANDIDATES
        if not evaluate_replicate_hard_gates(records[(candidate, "R0")])
    }
    if not r0_pass:
        # This is the protocol's normal R0 early-stop terminal.  Do not label
        # a ten-R0-only result as though R1/R2 had been executed.
        return evaluate_stage_a_r0(records.values(), protocol_status="passed")
    for candidate in FROZEN_CANDIDATES:
        observed = {replicate for item, replicate in records if item == candidate}
        expected = {"R0", "R1", "R2"} if candidate in r0_pass else {"R0"}
        if observed != expected:
            raise D0V3ProtocolGateError(
                "eligible-only replicate protocol violated for "
                f"{candidate.candidate_id}; expected={sorted(expected)}, "
                f"observed={sorted(observed)}"
            )

    rejected: list[CandidateGateDecision] = []
    ranking_inputs: list[CandidateRankingSummary] = []
    for candidate in FROZEN_CANDIDATES:
        r0_failures = evaluate_replicate_hard_gates(records[(candidate, "R0")])
        if r0_failures:
            rejected.append(
                CandidateGateDecision(
                    candidate,
                    False,
                    tuple(f"R0:{reason}" for reason in r0_failures),
                )
            )
            continue
        replicates = tuple(records[(candidate, name)] for name in REPLICATE_IDS)
        failures: list[str] = []
        for replicate in replicates:
            failures.extend(
                f"{replicate.replicate_id}:{reason}"
                for reason in evaluate_replicate_hard_gates(replicate)
            )
        nonclean_mean = _mean(
            tuple(replicate.nonclean_macro_delta_iou for replicate in replicates)
        )
        overall_mean = _mean(
            tuple(replicate.overall_macro_delta_iou for replicate in replicates)
        )
        if _below_minimum(nonclean_mean, NONCLEAN_MEAN_MINIMUM):
            failures.append("R0_R1_R2:nonclean_macro_iou_mean")
        if _below_minimum(overall_mean, OVERALL_MEAN_MINIMUM):
            failures.append("R0_R1_R2:overall_macro_iou_mean")
        if failures:
            rejected.append(CandidateGateDecision(candidate, False, tuple(failures)))
        else:
            ranking_inputs.append(_ranking_summary(candidate, replicates))

    ranking = rank_eligible_candidates(ranking_inputs)
    eligible = tuple(summary.candidate for summary in ranking)
    scientific_status: Literal["scientific_no_eligible", "scientific_passed"] = (
        "scientific_passed" if eligible else "scientific_no_eligible"
    )
    return StageAScienceDecision(
        evaluation_phase="R0_R1_R2",
        protocol_status="passed",
        formal_stage_a_protocol_complete=True,
        scientific_status=scientific_status,
        stage2_allowed=False,
        eligible_candidates=eligible,
        rejected_candidates=tuple(rejected),
        ranking=ranking,
        required_followup_replicates=(),
    )


__all__ = [
    "ALIGNMENT_OBSERVATIONS_PER_CANDIDATE_REPLICATE",
    "CORRUPTION_FAMILIES",
    "CandidateGateDecision",
    "CandidateRankingSummary",
    "D0V3ProtocolGateError",
    "D0V3ScienceGateError",
    "DATASETS",
    "EPISODES_PER_CANDIDATE_REPLICATE",
    "EpisodeActivity",
    "FUNCTIONAL_LOGIT_THRESHOLD",
    "REPLICATE_IDS",
    "ReplicateEvidence",
    "SAFETY_STRATA",
    "StageAScienceDecision",
    "evaluate_replicate_hard_gates",
    "evaluate_stage_a_final",
    "evaluate_stage_a_r0",
    "parse_replicate_evidence",
    "parse_r0_eligibility_receipt",
    "rank_eligible_candidates",
]
