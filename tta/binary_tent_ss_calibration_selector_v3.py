"""Scientific-gate selector for Binary Episodic TENT-SS Stage 1.

Version 3 deliberately separates two questions that v2 conflated:

* did the frozen calibration protocol execute correctly? and
* did any candidate satisfy a *fully frozen* scientific utility gate?

Protocol violations still raise :class:`CalibrationSelectionError`.  A valid
run with no scientifically eligible candidate is a normal negative result: it
returns a receipt with ``scientific_status == "failed"`` and never fabricates
a Top-K list.

All task metrics and gate comparisons use :class:`fractions.Fraction` derived
from integer sufficient statistics.  Floating-point summaries supplied by a
caller never influence selection.  The selector performs no filesystem, CUDA,
model, or dataset operations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from tta import binary_tent_ss_calibration_selector_v2 as _v2


SELECTOR_PROTOCOL_ID = "cr-sitta-binary-tent-ss-calibration-selector-v3"
STAGE1_RECEIPT_SCHEMA_VERSION = 3
STAGE1_RECEIPT_TYPE = "stage1_ss_scientific_selection"
STAGE1_RECEIPT_FILENAME = "stage1_ss_scientific_selection_receipt.json"

FORMAL_FROZEN_MODE = "formal_fully_frozen_gate"
RETROSPECTIVE_MODE = "retrospective_negative_replay"
GATE_MODES = (FORMAL_FROZEN_MODE, RETROSPECTIVE_MODE)

DATASETS = _v2.DATASETS
CONDITIONS = _v2.CONDITIONS
IMAGES_PER_CELL = _v2.IMAGES_PER_CELL
CELLS_PER_RUN = _v2.CELLS_PER_RUN
SS_BN_PROTOCOL = _v2.SS_BN_PROTOCOL
ALL_CANDIDATES = _v2.ALL_CANDIDATES
REQUIRED_HARD_GATES = _v2.REQUIRED_HARD_GATES
REQUIRED_PROTOCOL_AUDIT = _v2.REQUIRED_PROTOCOL_AUDIT
Candidate = _v2.Candidate
CandidateRankingMetrics = _v2.CandidateRankingMetrics
CalibrationSelectionError = _v2.CalibrationSelectionError
rank_candidate_metrics = _v2.rank_candidate_metrics

_UNRESOLVED_FORMAL_THRESHOLDS = (
    "positive_coverage.minimum_cells",
    "positive_coverage.minimum_corruption_families",
    "positive_coverage.minimum_datasets",
    "clean_safety.clean_iou_equivalence_margin",
    "operating_point_safety.max_pd_drop",
    "operating_point_safety.max_fa_increase_per_million_pixels",
    "activity.minimum_parameter_update_fraction_above_null",
    "activity.minimum_functional_change_fraction_above_null",
    "objective.minimum_episode_decrease_fraction",
)


def _ensure_fraction(value: Any, label: str) -> Fraction:
    if not isinstance(value, Fraction):
        raise CalibrationSelectionError(f"{label} must be fractions.Fraction")
    return value


def _ensure_unit_fraction(
    value: Any, label: str, *, allow_one: bool = True
) -> Fraction:
    result = _ensure_fraction(value, label)
    upper_ok = result <= 1 if allow_one else result < 1
    if result < 0 or not upper_ok:
        bound = "[0, 1]" if allow_one else "[0, 1)"
        raise CalibrationSelectionError(f"{label} must be in {bound}")
    return result


def _ensure_nonnegative_fraction(value: Any, label: str) -> Fraction:
    result = _ensure_fraction(value, label)
    if result < 0:
        raise CalibrationSelectionError(f"{label} must be non-negative")
    return result


@dataclass(frozen=True)
class ScientificGateSpec:
    """Frozen scientific thresholds, or an explicitly non-authoritative replay.

    ``formal_fully_frozen_gate`` requires every threshold and a SHA-256 binding
    to the external frozen gate manifest.  ``retrospective_negative_replay``
    intentionally leaves those thresholds unresolved and can only certify
    failure of the necessary condition ``macro_delta_iou > 0``.  It can never
    authorize Stage 2.
    """

    profile_id: str
    mode: Literal[
        "formal_fully_frozen_gate", "retrospective_negative_replay"
    ]
    require_macro_iou_positive: bool = True
    min_positive_cells: int | None = None
    min_positive_corruption_families: int | None = None
    min_positive_datasets: int | None = None
    clean_iou_equivalence_margin: Fraction | None = None
    max_pd_drop: Fraction | None = None
    max_fa_increase: Fraction | None = None
    min_parameter_update_fraction: Fraction | None = None
    min_functional_change_fraction: Fraction | None = None
    min_objective_decrease_fraction: Fraction | None = None
    top_k_after_filter: int = 3
    allow_fewer_than_top_k: bool = True
    threshold_source: str | None = None
    frozen_gate_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise CalibrationSelectionError("gate_spec.profile_id must be non-empty")
        if self.mode not in GATE_MODES:
            raise CalibrationSelectionError(
                f"gate_spec.mode must be one of {GATE_MODES}"
            )
        if self.require_macro_iou_positive is not True:
            raise CalibrationSelectionError(
                "v3 requires the strict necessary condition macro_delta_iou > 0"
            )
        if isinstance(self.top_k_after_filter, bool) or not isinstance(
            self.top_k_after_filter, int
        ):
            raise CalibrationSelectionError("top_k_after_filter must be an integer")
        if not 1 <= self.top_k_after_filter <= 3:
            raise CalibrationSelectionError("top_k_after_filter must be in [1, 3]")
        if not isinstance(self.allow_fewer_than_top_k, bool):
            raise CalibrationSelectionError("allow_fewer_than_top_k must be bool")

        threshold_values = (
            self.min_positive_cells,
            self.min_positive_corruption_families,
            self.min_positive_datasets,
            self.clean_iou_equivalence_margin,
            self.max_pd_drop,
            self.max_fa_increase,
            self.min_parameter_update_fraction,
            self.min_functional_change_fraction,
            self.min_objective_decrease_fraction,
        )
        if self.mode == RETROSPECTIVE_MODE:
            if any(value is not None for value in threshold_values):
                raise CalibrationSelectionError(
                    "retrospective replay must not invent unresolved formal thresholds"
                )
            if self.threshold_source is not None:
                raise CalibrationSelectionError(
                    "retrospective replay cannot claim a threshold source"
                )
            if self.frozen_gate_manifest_sha256 is not None:
                raise CalibrationSelectionError(
                    "retrospective replay cannot claim a frozen gate manifest"
                )
            return

        integer_thresholds = (
            (self.min_positive_cells, "min_positive_cells", CELLS_PER_RUN),
            (
                self.min_positive_corruption_families,
                "min_positive_corruption_families",
                len({name for name, _ in CONDITIONS if name != "clean"}),
            ),
            (self.min_positive_datasets, "min_positive_datasets", len(DATASETS)),
        )
        for value, label, maximum in integer_thresholds:
            if isinstance(value, bool) or not isinstance(value, int):
                raise CalibrationSelectionError(
                    f"formal gate {label} must be an explicit integer"
                )
            if not 1 <= value <= maximum:
                raise CalibrationSelectionError(
                    f"formal gate {label} must be in [1, {maximum}]"
                )
        _ensure_unit_fraction(
            self.clean_iou_equivalence_margin,
            "clean_iou_equivalence_margin",
        )
        _ensure_unit_fraction(self.max_pd_drop, "max_pd_drop")
        _ensure_nonnegative_fraction(self.max_fa_increase, "max_fa_increase")
        _ensure_unit_fraction(
            self.min_parameter_update_fraction,
            "min_parameter_update_fraction",
            allow_one=False,
        )
        _ensure_unit_fraction(
            self.min_functional_change_fraction,
            "min_functional_change_fraction",
            allow_one=False,
        )
        objective_floor = _ensure_unit_fraction(
            self.min_objective_decrease_fraction,
            "min_objective_decrease_fraction",
        )
        if objective_floor <= 0:
            raise CalibrationSelectionError(
                "min_objective_decrease_fraction must be strictly positive"
            )
        if not isinstance(self.threshold_source, str) or not self.threshold_source:
            raise CalibrationSelectionError(
                "formal gate requires an explicit frozen threshold_source"
            )
        digest = self.frozen_gate_manifest_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CalibrationSelectionError(
                "formal gate requires a lowercase 64-hex frozen_gate_manifest_sha256"
            )

    @classmethod
    def retrospective_v2_negative_replay(cls) -> "ScientificGateSpec":
        return cls(
            profile_id="current-v2-stage1-negative-replay-only",
            mode=RETROSPECTIVE_MODE,
        )

    @property
    def unresolved_thresholds(self) -> tuple[str, ...]:
        if self.mode == RETROSPECTIVE_MODE:
            return _UNRESOLVED_FORMAL_THRESHOLDS
        return ()


@dataclass(frozen=True)
class CandidateDiagnosticEvidence:
    """Exact count evidence for activity and self-objective decrease gates."""

    candidate: Candidate
    parameter_update_episodes: int
    parameter_update_evaluated_episodes: int
    functional_change_episodes: int
    functional_change_evaluated_episodes: int
    objective_decrease_episodes: int
    objective_evaluated_episodes: int

    def __post_init__(self) -> None:
        if self.candidate not in ALL_CANDIDATES:
            raise CalibrationSelectionError("diagnostic evidence has unknown candidate")
        for numerator_name, denominator_name in (
            ("parameter_update_episodes", "parameter_update_evaluated_episodes"),
            ("functional_change_episodes", "functional_change_evaluated_episodes"),
            ("objective_decrease_episodes", "objective_evaluated_episodes"),
        ):
            numerator = getattr(self, numerator_name)
            denominator = getattr(self, denominator_name)
            if isinstance(numerator, bool) or not isinstance(numerator, int):
                raise CalibrationSelectionError(f"{numerator_name} must be integer")
            if isinstance(denominator, bool) or not isinstance(denominator, int):
                raise CalibrationSelectionError(f"{denominator_name} must be integer")
            if denominator <= 0 or numerator < 0 or numerator > denominator:
                raise CalibrationSelectionError(
                    f"require 0 <= {numerator_name} <= {denominator_name} and denominator > 0"
                )

    @property
    def parameter_update_fraction(self) -> Fraction:
        return Fraction(
            self.parameter_update_episodes,
            self.parameter_update_evaluated_episodes,
        )

    @property
    def functional_change_fraction(self) -> Fraction:
        return Fraction(
            self.functional_change_episodes,
            self.functional_change_evaluated_episodes,
        )

    @property
    def objective_decrease_fraction(self) -> Fraction:
        return Fraction(
            self.objective_decrease_episodes,
            self.objective_evaluated_episodes,
        )


@dataclass(frozen=True)
class CandidateScienceSummary:
    candidate: Candidate
    macro_delta_iou: Fraction
    clean_delta_iou: Fraction
    macro_pd_delta: Fraction
    macro_fa_delta: Fraction
    positive_cells: int
    positive_corruption_families: tuple[str, ...]
    positive_datasets: tuple[str, ...]
    parameter_update_fraction: Fraction | None
    functional_change_fraction: Fraction | None
    objective_decrease_fraction: Fraction | None
    failed_gates: tuple[str, ...]
    not_evaluated_gates: tuple[str, ...]


@dataclass(frozen=True)
class ScientificGateDecision:
    protocol_status: str
    scientific_status: str
    stage2_allowed: bool
    eligible: tuple[CandidateScienceSummary, ...]
    rejected: tuple[CandidateScienceSummary, ...]


def _fraction_receipt(value: Fraction) -> dict[str, Any]:
    return _v2._v1._fraction_receipt(value)


def _fraction_from_receipt(value: Mapping[str, Any], label: str) -> Fraction:
    try:
        numerator = value["numerator"]
        denominator = value["denominator"]
    except KeyError as exc:
        raise CalibrationSelectionError(f"{label} missing exact fraction fields") from exc
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise CalibrationSelectionError(f"{label}.numerator must be integer")
    if isinstance(denominator, bool) or not isinstance(denominator, int):
        raise CalibrationSelectionError(f"{label}.denominator must be integer")
    if denominator <= 0:
        raise CalibrationSelectionError(f"{label}.denominator must be positive")
    return Fraction(numerator, denominator)


def _diagnostic_mapping(
    diagnostics: Iterable[CandidateDiagnosticEvidence] | None,
) -> dict[Candidate, CandidateDiagnosticEvidence]:
    if diagnostics is None:
        return {}
    if isinstance(diagnostics, (str, bytes, Mapping)):
        raise CalibrationSelectionError(
            "diagnostics must be an iterable of CandidateDiagnosticEvidence"
        )
    result: dict[Candidate, CandidateDiagnosticEvidence] = {}
    for index, evidence in enumerate(diagnostics):
        if not isinstance(evidence, CandidateDiagnosticEvidence):
            raise CalibrationSelectionError(
                f"diagnostics[{index}] must be CandidateDiagnosticEvidence"
            )
        if evidence.candidate in result:
            raise CalibrationSelectionError(
                f"duplicate diagnostic candidate {evidence.candidate.to_dict()}"
            )
        result[evidence.candidate] = evidence
    extra = set(result) - set(ALL_CANDIDATES)
    if extra:
        raise CalibrationSelectionError("diagnostics contain unsupported candidates")
    return result


def _positive_coverage(
    summary: _v2._v1._RunSummary,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    positive_cells = 0
    families: set[str] = set()
    datasets: set[str] = set()
    for index, cell in enumerate(summary.cell_receipts):
        derived = cell["derived"]
        delta = _fraction_from_receipt(
            derived["global_iou_delta"],
            f"cell_receipts[{index}].derived.global_iou_delta",
        )
        if delta <= 0:
            continue
        positive_cells += 1
        dataset = cell["dataset"]
        corruption = cell["condition"]["corruption"]
        datasets.add(dataset)
        if corruption != "clean":
            families.add(corruption)
    return (
        positive_cells,
        tuple(
            corruption
            for corruption in dict.fromkeys(name for name, _ in CONDITIONS)
            if corruption != "clean" and corruption in families
        ),
        tuple(dataset for dataset in DATASETS if dataset in datasets),
    )


def _evaluate_candidate_science(
    *,
    summary: _v2._v1._RunSummary,
    evidence: CandidateDiagnosticEvidence | None,
    gate_spec: ScientificGateSpec,
) -> CandidateScienceSummary:
    positive_cells, positive_families, positive_datasets = _positive_coverage(summary)
    failed: list[str] = []
    not_evaluated: list[str] = []

    if summary.macro_delta_iou <= 0:
        failed.append("macro_delta_iou")

    if gate_spec.mode == RETROSPECTIVE_MODE:
        not_evaluated.extend(
            (
                "positive_cell_coverage",
                "positive_corruption_family_coverage",
                "positive_dataset_coverage",
                "clean_iou_safety",
                "pd_fa_joint_safety",
                "parameter_update_activity",
                "functional_change_activity",
                "objective_decrease",
            )
        )
        if not failed:
            failed.append("formal_gate_spec_unresolved")
        return CandidateScienceSummary(
            candidate=summary.candidate,
            macro_delta_iou=summary.macro_delta_iou,
            clean_delta_iou=summary.clean_macro_delta_iou,
            macro_pd_delta=summary.macro_pd_delta,
            macro_fa_delta=summary.macro_fa_increase,
            positive_cells=positive_cells,
            positive_corruption_families=positive_families,
            positive_datasets=positive_datasets,
            parameter_update_fraction=None,
            functional_change_fraction=None,
            objective_decrease_fraction=None,
            failed_gates=tuple(failed),
            not_evaluated_gates=tuple(not_evaluated),
        )

    # ScientificGateSpec.__post_init__ proves these are non-None in formal mode.
    assert gate_spec.min_positive_cells is not None
    assert gate_spec.min_positive_corruption_families is not None
    assert gate_spec.min_positive_datasets is not None
    assert gate_spec.clean_iou_equivalence_margin is not None
    assert gate_spec.max_pd_drop is not None
    assert gate_spec.max_fa_increase is not None
    assert gate_spec.min_parameter_update_fraction is not None
    assert gate_spec.min_functional_change_fraction is not None
    assert gate_spec.min_objective_decrease_fraction is not None

    if positive_cells < gate_spec.min_positive_cells:
        failed.append("positive_cell_coverage")
    if len(positive_families) < gate_spec.min_positive_corruption_families:
        failed.append("positive_corruption_family_coverage")
    if len(positive_datasets) < gate_spec.min_positive_datasets:
        failed.append("positive_dataset_coverage")
    if summary.clean_macro_delta_iou < -gate_spec.clean_iou_equivalence_margin:
        failed.append("clean_iou_safety")
    if (
        summary.macro_pd_delta < -gate_spec.max_pd_drop
        and summary.macro_fa_increase > gate_spec.max_fa_increase
    ):
        failed.append("pd_fa_joint_safety")

    if evidence is None:
        raise CalibrationSelectionError(
            "formal frozen gate requires exact diagnostic evidence for every candidate; "
            f"missing={summary.candidate.to_dict()}"
        )
    parameter_fraction = evidence.parameter_update_fraction
    functional_fraction = evidence.functional_change_fraction
    objective_fraction = evidence.objective_decrease_fraction
    if parameter_fraction <= gate_spec.min_parameter_update_fraction:
        failed.append("parameter_update_activity")
    if functional_fraction <= gate_spec.min_functional_change_fraction:
        failed.append("functional_change_activity")
    if objective_fraction < gate_spec.min_objective_decrease_fraction:
        failed.append("objective_decrease")

    return CandidateScienceSummary(
        candidate=summary.candidate,
        macro_delta_iou=summary.macro_delta_iou,
        clean_delta_iou=summary.clean_macro_delta_iou,
        macro_pd_delta=summary.macro_pd_delta,
        macro_fa_delta=summary.macro_fa_increase,
        positive_cells=positive_cells,
        positive_corruption_families=positive_families,
        positive_datasets=positive_datasets,
        parameter_update_fraction=parameter_fraction,
        functional_change_fraction=functional_fraction,
        objective_decrease_fraction=objective_fraction,
        failed_gates=tuple(failed),
        not_evaluated_gates=(),
    )


def _gate_spec_receipt(gate_spec: ScientificGateSpec) -> dict[str, Any]:
    result: dict[str, Any] = {
        "profile_id": gate_spec.profile_id,
        "mode": gate_spec.mode,
        "require_macro_iou_positive": True,
        "filter_before_ranking": True,
        "top_k_after_filter": gate_spec.top_k_after_filter,
        "allow_fewer_than_top_k": gate_spec.allow_fewer_than_top_k,
        "threshold_resolution_status": (
            "fully_frozen"
            if gate_spec.mode == FORMAL_FROZEN_MODE
            else "unresolved_retrospective_only"
        ),
        "unresolved_thresholds": list(gate_spec.unresolved_thresholds),
        "threshold_source": gate_spec.threshold_source,
        "frozen_gate_manifest_sha256": gate_spec.frozen_gate_manifest_sha256,
        "authorizes_stage2": gate_spec.mode == FORMAL_FROZEN_MODE,
    }
    if gate_spec.mode == FORMAL_FROZEN_MODE:
        assert gate_spec.clean_iou_equivalence_margin is not None
        assert gate_spec.max_pd_drop is not None
        assert gate_spec.max_fa_increase is not None
        assert gate_spec.min_parameter_update_fraction is not None
        assert gate_spec.min_functional_change_fraction is not None
        assert gate_spec.min_objective_decrease_fraction is not None
        result["thresholds"] = {
            "minimum_positive_cells": gate_spec.min_positive_cells,
            "minimum_positive_corruption_families": (
                gate_spec.min_positive_corruption_families
            ),
            "minimum_positive_datasets": gate_spec.min_positive_datasets,
            "clean_iou_equivalence_margin": _fraction_receipt(
                gate_spec.clean_iou_equivalence_margin
            ),
            "max_pd_drop": _fraction_receipt(gate_spec.max_pd_drop),
            "max_fa_increase_per_million_pixels": _fraction_receipt(
                gate_spec.max_fa_increase
            ),
            "minimum_parameter_update_fraction_above_null": _fraction_receipt(
                gate_spec.min_parameter_update_fraction
            ),
            "minimum_functional_change_fraction_above_null": _fraction_receipt(
                gate_spec.min_functional_change_fraction
            ),
            "minimum_objective_decrease_fraction": _fraction_receipt(
                gate_spec.min_objective_decrease_fraction
            ),
        }
    else:
        result["thresholds"] = {
            "macro_global_iou_delta": {
                "rule": "strict_greater_than_zero",
                "role": "necessary_condition_failure_certificate",
            }
        }
    return result


def _science_summary_receipt(summary: CandidateScienceSummary) -> dict[str, Any]:
    activity: dict[str, Any] = {
        "parameter_update_fraction": None,
        "functional_change_fraction": None,
        "objective_decrease_fraction": None,
    }
    if summary.parameter_update_fraction is not None:
        activity["parameter_update_fraction"] = _fraction_receipt(
            summary.parameter_update_fraction
        )
    if summary.functional_change_fraction is not None:
        activity["functional_change_fraction"] = _fraction_receipt(
            summary.functional_change_fraction
        )
    if summary.objective_decrease_fraction is not None:
        activity["objective_decrease_fraction"] = _fraction_receipt(
            summary.objective_decrease_fraction
        )
    return {
        "candidate": summary.candidate.to_dict(),
        "eligible": not summary.failed_gates,
        "failed_gates": list(summary.failed_gates),
        "not_evaluated_gates": list(summary.not_evaluated_gates),
        "metrics": {
            "macro_global_iou_delta": _fraction_receipt(summary.macro_delta_iou),
            "clean_macro_global_iou_delta": _fraction_receipt(
                summary.clean_delta_iou
            ),
            "macro_pd_delta": _fraction_receipt(summary.macro_pd_delta),
            "macro_fa_delta_per_million_pixels": _fraction_receipt(
                summary.macro_fa_delta
            ),
        },
        "positive_coverage": {
            "cells": summary.positive_cells,
            "corruption_families": list(summary.positive_corruption_families),
            "datasets": list(summary.positive_datasets),
        },
        "activity_and_objective": activity,
    }


def _ranking_receipt(
    ranking: Iterable[CandidateRankingMetrics],
) -> list[dict[str, Any]]:
    return [
        {"rank": rank, **_v2._v1._metric_receipt(metric)}
        for rank, metric in enumerate(ranking, start=1)
    ]


def select_stage1_candidates(
    stage1_records: Iterable[Mapping[str, Any]],
    diagnostics: Iterable[CandidateDiagnosticEvidence] | None,
    gate_spec: ScientificGateSpec,
) -> dict[str, Any]:
    """Validate Stage 1, filter by science gates, then rank eligible candidates.

    A valid negative result always returns a receipt.  Only malformed or
    protocol-invalid evidence raises ``CalibrationSelectionError``.
    """

    if not isinstance(gate_spec, ScientificGateSpec):
        raise CalibrationSelectionError("gate_spec must be ScientificGateSpec")
    summaries, full_ranking = _v2._stage1_summaries(stage1_records)
    diagnostic_by_candidate = _diagnostic_mapping(diagnostics)
    if gate_spec.mode == FORMAL_FROZEN_MODE:
        missing = [
            candidate.to_dict()
            for candidate in ALL_CANDIDATES
            if candidate not in diagnostic_by_candidate
        ]
        if missing:
            raise CalibrationSelectionError(
                f"formal gate diagnostic evidence is incomplete; missing={missing}"
            )

    science_by_candidate = {
        candidate: _evaluate_candidate_science(
            summary=summaries[candidate],
            evidence=diagnostic_by_candidate.get(candidate),
            gate_spec=gate_spec,
        )
        for candidate in ALL_CANDIDATES
    }
    eligible_ids = {
        candidate
        for candidate, summary in science_by_candidate.items()
        if not summary.failed_gates
    }

    # This ordering is intentionally filter-then-rank.  Never interpret the
    # first entries of ``full_ranking`` as scientifically qualified.
    eligible_ranking = tuple(
        metric for metric in full_ranking if metric.candidate in eligible_ids
    )
    if (
        not gate_spec.allow_fewer_than_top_k
        and len(eligible_ranking) < gate_spec.top_k_after_filter
    ):
        selected_metrics: tuple[CandidateRankingMetrics, ...] = ()
    else:
        selected_metrics = eligible_ranking[: gate_spec.top_k_after_filter]

    selected = tuple(metric.candidate for metric in selected_metrics)
    scientific_passed = bool(selected) and gate_spec.mode == FORMAL_FROZEN_MODE
    stage2_allowed = scientific_passed
    eligible_summaries = tuple(
        science_by_candidate[metric.candidate] for metric in eligible_ranking
    )
    rejected_summaries = tuple(
        science_by_candidate[metric.candidate]
        for metric in full_ranking
        if metric.candidate not in eligible_ids
    )
    decision = ScientificGateDecision(
        protocol_status="passed",
        scientific_status="passed" if scientific_passed else "failed",
        stage2_allowed=stage2_allowed,
        eligible=eligible_summaries,
        rejected=rejected_summaries,
    )

    receipt = {
        "schema_version": STAGE1_RECEIPT_SCHEMA_VERSION,
        "receipt_type": STAGE1_RECEIPT_TYPE,
        "selector_protocol_id": SELECTOR_PROTOCOL_ID,
        "formal_fully_frozen_gate": gate_spec.mode == FORMAL_FROZEN_MODE,
        "unresolved_gate_thresholds": gate_spec.mode == RETROSPECTIVE_MODE,
        "retrospective_negative_replay": gate_spec.mode == RETROSPECTIVE_MODE,
        "protocol_status": decision.protocol_status,
        "scientific_status": decision.scientific_status,
        "route_decision": (
            "proceed_to_stage2"
            if decision.stage2_allowed
            else "stop_before_stage2"
        ),
        "stage2_allowed": decision.stage2_allowed,
        "stage3_allowed": False,
        "paper_result": False,
        "selection_bn_protocol": "SS",
        "selection_bn_protocol_id": SS_BN_PROTOCOL,
        "scientific_gate": _gate_spec_receipt(gate_spec),
        "validation": {
            "candidate_count": len(ALL_CANDIDATES),
            "candidate_run_count": len(ALL_CANDIDATES),
            "cells_per_run": CELLS_PER_RUN,
            "aggregate_cell_record_count": len(ALL_CANDIDATES) * CELLS_PER_RUN,
            "episode_count": len(ALL_CANDIDATES)
            * CELLS_PER_RUN
            * IMAGES_PER_CELL,
            "protocol_gate_passed": True,
            "all_metric_gate_arithmetic_exact_fraction": True,
            "filter_before_ranking": True,
        },
        "all_candidate_ranking": _ranking_receipt(full_ranking),
        "eligible_ranking": _ranking_receipt(eligible_ranking),
        "eligible_candidates": [
            _science_summary_receipt(value) for value in decision.eligible
        ],
        "rejected_candidates": [
            _science_summary_receipt(value) for value in decision.rejected
        ],
        "selected_for_stage2": [candidate.to_dict() for candidate in selected],
    }
    validate_stage1_scientific_receipt(receipt)
    return receipt


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationSelectionError(f"{label} must be a mapping")
    return value


def _parse_receipt_candidate(value: Any, label: str) -> Candidate:
    candidate = _mapping(value, label)
    if "optimizer" not in candidate or "learning_rate" not in candidate:
        raise CalibrationSelectionError(
            f"{label} requires optimizer and learning_rate"
        )
    return Candidate.from_values(candidate["optimizer"], candidate["learning_rate"])


def _validate_exact_fraction_receipt(value: Any, label: str) -> Fraction:
    receipt = _mapping(value, label)
    fraction = _fraction_from_receipt(receipt, label)
    expected_exact = f"{fraction.numerator}/{fraction.denominator}"
    if receipt.get("exact") != expected_exact:
        raise CalibrationSelectionError(f"{label}.exact is inconsistent")
    return fraction


def _validate_positive_authorization_contract(
    receipt: Mapping[str, Any],
    eligible: tuple[Candidate, ...],
    selected: tuple[Candidate, ...],
) -> None:
    """Reject a hand-written positive receipt lacking frozen gate provenance."""

    if receipt.get("selector_protocol_id") != SELECTOR_PROTOCOL_ID:
        raise CalibrationSelectionError(
            "positive authorization requires the v3 selector_protocol_id"
        )
    gate = _mapping(receipt.get("scientific_gate"), "scientific_gate")
    required_gate_fields = (
        "profile_id",
        "mode",
        "filter_before_ranking",
        "top_k_after_filter",
        "allow_fewer_than_top_k",
        "threshold_resolution_status",
        "unresolved_thresholds",
        "threshold_source",
        "frozen_gate_manifest_sha256",
        "authorizes_stage2",
        "thresholds",
    )
    missing = [key for key in required_gate_fields if key not in gate]
    if missing:
        raise CalibrationSelectionError(
            f"positive scientific_gate missing fields: {missing}"
        )
    if not isinstance(gate["profile_id"], str) or not gate["profile_id"]:
        raise CalibrationSelectionError("scientific_gate.profile_id must be non-empty")
    if gate["mode"] != FORMAL_FROZEN_MODE:
        raise CalibrationSelectionError("positive receipt requires formal gate mode")
    if gate["filter_before_ranking"] is not True:
        raise CalibrationSelectionError("positive receipt must filter before ranking")
    if gate["threshold_resolution_status"] != "fully_frozen":
        raise CalibrationSelectionError("positive gate thresholds are not fully frozen")
    if gate["authorizes_stage2"] is not True:
        raise CalibrationSelectionError("scientific_gate does not authorize Stage 2")
    if gate["unresolved_thresholds"] != []:
        raise CalibrationSelectionError("positive receipt has unresolved thresholds")
    if not isinstance(gate["threshold_source"], str) or not gate["threshold_source"]:
        raise CalibrationSelectionError("positive receipt lacks threshold_source")
    digest = gate["frozen_gate_manifest_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CalibrationSelectionError(
            "positive receipt has invalid frozen_gate_manifest_sha256"
        )
    top_k = gate["top_k_after_filter"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 3:
        raise CalibrationSelectionError("scientific_gate top_k must be in [1, 3]")
    if not isinstance(gate["allow_fewer_than_top_k"], bool):
        raise CalibrationSelectionError("allow_fewer_than_top_k must be bool")

    thresholds = _mapping(gate["thresholds"], "scientific_gate.thresholds")
    required_integer_thresholds = (
        "minimum_positive_cells",
        "minimum_positive_corruption_families",
        "minimum_positive_datasets",
    )
    required_fraction_thresholds = (
        "clean_iou_equivalence_margin",
        "max_pd_drop",
        "max_fa_increase_per_million_pixels",
        "minimum_parameter_update_fraction_above_null",
        "minimum_functional_change_fraction_above_null",
        "minimum_objective_decrease_fraction",
    )
    missing_thresholds = [
        key
        for key in (*required_integer_thresholds, *required_fraction_thresholds)
        if key not in thresholds
    ]
    if missing_thresholds:
        raise CalibrationSelectionError(
            f"positive scientific gate missing thresholds: {missing_thresholds}"
        )
    for key in required_integer_thresholds:
        value = thresholds[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CalibrationSelectionError(
                f"scientific_gate.thresholds.{key} must be positive integer"
            )
    exact_thresholds: dict[str, Fraction] = {}
    for key in required_fraction_thresholds:
        exact_thresholds[key] = _validate_exact_fraction_receipt(
            thresholds[key], f"scientific_gate.thresholds.{key}"
        )
        if exact_thresholds[key] < 0:
            raise CalibrationSelectionError(
                f"scientific_gate.thresholds.{key} must be non-negative"
            )

    eligible_raw = receipt["eligible_candidates"]
    science_metrics: dict[Candidate, CandidateRankingMetrics] = {}
    allowed_families = {name for name, _ in CONDITIONS if name != "clean"}
    for index, raw_entry in enumerate(eligible_raw):
        entry = _mapping(raw_entry, f"eligible_candidates[{index}]")
        candidate = eligible[index]
        if entry.get("eligible") is not True or entry.get("failed_gates") != []:
            raise CalibrationSelectionError(
                "positive eligible entry is not scientifically eligible"
            )
        if entry.get("not_evaluated_gates", []) != []:
            raise CalibrationSelectionError(
                "positive eligible entry has unevaluated gates"
            )
        metrics = _mapping(entry.get("metrics"), f"eligible_candidates[{index}].metrics")
        required_metrics = (
            "macro_global_iou_delta",
            "clean_macro_global_iou_delta",
            "macro_pd_delta",
            "macro_fa_delta_per_million_pixels",
        )
        missing_metrics = [key for key in required_metrics if key not in metrics]
        if missing_metrics:
            raise CalibrationSelectionError(
                f"positive eligible entry missing metrics: {missing_metrics}"
            )
        macro_iou = _validate_exact_fraction_receipt(
            metrics["macro_global_iou_delta"],
            f"eligible_candidates[{index}].metrics.macro_global_iou_delta",
        )
        clean_iou = _validate_exact_fraction_receipt(
            metrics["clean_macro_global_iou_delta"],
            f"eligible_candidates[{index}].metrics.clean_macro_global_iou_delta",
        )
        pd_delta = _validate_exact_fraction_receipt(
            metrics["macro_pd_delta"],
            f"eligible_candidates[{index}].metrics.macro_pd_delta",
        )
        fa_delta = _validate_exact_fraction_receipt(
            metrics["macro_fa_delta_per_million_pixels"],
            f"eligible_candidates[{index}].metrics.macro_fa_delta_per_million_pixels",
        )
        if macro_iou <= 0:
            raise CalibrationSelectionError(
                "positive eligible candidate violates macro_delta_iou > 0"
            )
        if clean_iou < -exact_thresholds["clean_iou_equivalence_margin"]:
            raise CalibrationSelectionError(
                "positive eligible candidate violates clean IoU safety"
            )
        if (
            pd_delta < -exact_thresholds["max_pd_drop"]
            and fa_delta
            > exact_thresholds["max_fa_increase_per_million_pixels"]
        ):
            raise CalibrationSelectionError(
                "positive eligible candidate violates joint Pd/Fa safety"
            )

        coverage = _mapping(
            entry.get("positive_coverage"),
            f"eligible_candidates[{index}].positive_coverage",
        )
        cells = coverage.get("cells")
        families = coverage.get("corruption_families")
        datasets = coverage.get("datasets")
        if (
            isinstance(cells, bool)
            or not isinstance(cells, int)
            or not 0 <= cells <= CELLS_PER_RUN
        ):
            raise CalibrationSelectionError("invalid positive cell coverage")
        if not isinstance(families, list) or any(
            not isinstance(value, str) for value in families
        ):
            raise CalibrationSelectionError("invalid positive corruption families")
        if len(set(families)) != len(families) or not set(families) <= allowed_families:
            raise CalibrationSelectionError("invalid positive corruption family set")
        if not isinstance(datasets, list) or any(
            not isinstance(value, str) for value in datasets
        ):
            raise CalibrationSelectionError("invalid positive datasets")
        if len(set(datasets)) != len(datasets) or not set(datasets) <= set(DATASETS):
            raise CalibrationSelectionError("invalid positive dataset set")
        if cells < thresholds["minimum_positive_cells"]:
            raise CalibrationSelectionError("positive cell coverage gate is not met")
        if len(families) < thresholds["minimum_positive_corruption_families"]:
            raise CalibrationSelectionError(
                "positive corruption-family coverage gate is not met"
            )
        if len(datasets) < thresholds["minimum_positive_datasets"]:
            raise CalibrationSelectionError(
                "positive dataset coverage gate is not met"
            )

        activity = _mapping(
            entry.get("activity_and_objective"),
            f"eligible_candidates[{index}].activity_and_objective",
        )
        activity_keys = (
            "parameter_update_fraction",
            "functional_change_fraction",
            "objective_decrease_fraction",
        )
        missing_activity = [key for key in activity_keys if key not in activity]
        if missing_activity:
            raise CalibrationSelectionError(
                f"positive eligible entry missing activity: {missing_activity}"
            )
        parameter_fraction = _validate_exact_fraction_receipt(
            activity["parameter_update_fraction"],
            f"eligible_candidates[{index}].activity.parameter_update_fraction",
        )
        functional_fraction = _validate_exact_fraction_receipt(
            activity["functional_change_fraction"],
            f"eligible_candidates[{index}].activity.functional_change_fraction",
        )
        objective_fraction = _validate_exact_fraction_receipt(
            activity["objective_decrease_fraction"],
            f"eligible_candidates[{index}].activity.objective_decrease_fraction",
        )
        if not 0 <= parameter_fraction <= 1 or not 0 <= functional_fraction <= 1:
            raise CalibrationSelectionError("activity fractions must be in [0, 1]")
        if not 0 <= objective_fraction <= 1:
            raise CalibrationSelectionError("objective fraction must be in [0, 1]")
        if (
            parameter_fraction
            <= exact_thresholds["minimum_parameter_update_fraction_above_null"]
        ):
            raise CalibrationSelectionError("parameter activity gate is not met")
        if (
            functional_fraction
            <= exact_thresholds["minimum_functional_change_fraction_above_null"]
        ):
            raise CalibrationSelectionError("functional activity gate is not met")
        if (
            objective_fraction
            < exact_thresholds["minimum_objective_decrease_fraction"]
        ):
            raise CalibrationSelectionError("objective-decrease gate is not met")

        science_metrics[candidate] = CandidateRankingMetrics(
            candidate=candidate,
            primary_macro_delta_iou=macro_iou,
            worst_run_macro_delta_iou=macro_iou,
            clean_macro_delta_iou=clean_iou,
            macro_fa_increase_per_million_pixels=fa_delta,
            macro_pd_delta=pd_delta,
        )

    recomputed_ranking = rank_candidate_metrics(science_metrics.values())
    if tuple(metric.candidate for metric in recomputed_ranking) != eligible:
        raise CalibrationSelectionError(
            "eligible_candidates are not in the exact deterministic ranking order"
        )

    ranking_raw = receipt.get("eligible_ranking")
    if not isinstance(ranking_raw, list):
        raise CalibrationSelectionError(
            "positive authorization requires eligible_ranking list"
        )
    ranking_candidates: list[Candidate] = []
    for index, entry in enumerate(ranking_raw):
        item = _mapping(entry, f"eligible_ranking[{index}]")
        if item.get("rank") != index + 1:
            raise CalibrationSelectionError("eligible_ranking ranks are not contiguous")
        if "candidate" not in item:
            raise CalibrationSelectionError(
                f"eligible_ranking[{index}] missing candidate"
            )
        candidate = _parse_receipt_candidate(
            item["candidate"], f"eligible_ranking[{index}].candidate"
        )
        ranking_candidates.append(candidate)
        expected_metric = science_metrics[candidate]
        ranking_fraction_fields = (
            (
                "primary_mean_over_runs_macro_global_iou_delta",
                expected_metric.primary_macro_delta_iou,
            ),
            (
                "worst_run_macro_global_iou_delta",
                expected_metric.worst_run_macro_delta_iou,
            ),
            (
                "clean_mean_over_runs_macro_global_iou_delta",
                expected_metric.clean_macro_delta_iou,
            ),
            (
                "mean_over_runs_macro_fa_increase_per_million_pixels",
                expected_metric.macro_fa_increase_per_million_pixels,
            ),
            ("mean_over_runs_macro_pd_delta", expected_metric.macro_pd_delta),
        )
        for field_name, expected_value in ranking_fraction_fields:
            if field_name not in item:
                raise CalibrationSelectionError(
                    f"eligible_ranking[{index}] missing {field_name}"
                )
            observed = _validate_exact_fraction_receipt(
                item[field_name], f"eligible_ranking[{index}].{field_name}"
            )
            if observed != expected_value:
                raise CalibrationSelectionError(
                    f"eligible_ranking[{index}].{field_name} is inconsistent"
                )
    if tuple(ranking_candidates) != eligible:
        raise CalibrationSelectionError(
            "eligible_candidates order must exactly match eligible_ranking"
        )
    expected_count = min(top_k, len(eligible))
    if not gate["allow_fewer_than_top_k"] and len(eligible) < top_k:
        expected_count = 0
    if len(selected) != expected_count or selected != eligible[: len(selected)]:
        raise CalibrationSelectionError(
            "selected candidates must be the eligible-ranking prefix after filtering"
        )


def validate_stage1_scientific_receipt(
    payload: Mapping[str, Any],
) -> tuple[Candidate, ...]:
    """Fail-closed structural/semantic validation for a Stage-2 launcher.

    File-byte SHA-256 and artifact-manifest checks intentionally remain the
    launcher's responsibility because this pure selector has no filesystem
    access.  This function returns the frozen selected candidates (zero to
    three) after validating the receipt's core authorization semantics.
    """

    receipt = _mapping(payload, "receipt")
    required = (
        "schema_version",
        "receipt_type",
        "protocol_status",
        "scientific_status",
        "stage2_allowed",
        "formal_fully_frozen_gate",
        "unresolved_gate_thresholds",
        "retrospective_negative_replay",
        "eligible_candidates",
        "selected_for_stage2",
    )
    missing = [key for key in required if key not in receipt]
    if missing:
        raise CalibrationSelectionError(f"scientific receipt missing fields: {missing}")
    if receipt["schema_version"] != STAGE1_RECEIPT_SCHEMA_VERSION:
        raise CalibrationSelectionError("unsupported scientific receipt schema_version")
    if receipt["receipt_type"] != STAGE1_RECEIPT_TYPE:
        raise CalibrationSelectionError("unexpected scientific receipt_type")
    protocol_status = receipt["protocol_status"]
    scientific_status = receipt["scientific_status"]
    if protocol_status not in ("passed", "failed"):
        raise CalibrationSelectionError("protocol_status must be passed or failed")
    if scientific_status not in ("passed", "failed"):
        raise CalibrationSelectionError("scientific_status must be passed or failed")
    if protocol_status == "failed" and scientific_status != "failed":
        raise CalibrationSelectionError(
            "protocol failure cannot carry scientific_status=passed"
        )
    allowed = receipt["stage2_allowed"]
    if not isinstance(allowed, bool):
        raise CalibrationSelectionError("stage2_allowed must be bool")
    provenance_values = {
        key: receipt[key]
        for key in (
            "formal_fully_frozen_gate",
            "unresolved_gate_thresholds",
            "retrospective_negative_replay",
        )
    }
    if any(not isinstance(value, bool) for value in provenance_values.values()):
        raise CalibrationSelectionError("gate provenance fields must be bool")
    formal = provenance_values["formal_fully_frozen_gate"]
    unresolved = provenance_values["unresolved_gate_thresholds"]
    retrospective = provenance_values["retrospective_negative_replay"]
    if retrospective:
        if formal or not unresolved or allowed:
            raise CalibrationSelectionError(
                "retrospective receipt must be non-formal, unresolved, and blocked"
            )
    elif not formal or unresolved:
        raise CalibrationSelectionError(
            "non-retrospective receipt must use a fully frozen resolved gate"
        )

    eligible_raw = receipt["eligible_candidates"]
    selected_raw = receipt["selected_for_stage2"]
    if not isinstance(eligible_raw, list):
        raise CalibrationSelectionError("eligible_candidates must be a list")
    if not isinstance(selected_raw, list):
        raise CalibrationSelectionError("selected_for_stage2 must be a list")

    eligible: list[Candidate] = []
    for index, entry in enumerate(eligible_raw):
        decision = _mapping(entry, f"eligible_candidates[{index}]")
        if "candidate" not in decision:
            raise CalibrationSelectionError(
                f"eligible_candidates[{index}] missing candidate"
            )
        candidate = _parse_receipt_candidate(
            decision["candidate"], f"eligible_candidates[{index}].candidate"
        )
        if candidate in eligible:
            raise CalibrationSelectionError("duplicate eligible candidate")
        if "eligible" in decision and decision["eligible"] is not True:
            raise CalibrationSelectionError(
                "eligible_candidates entry must declare eligible=true"
            )
        if "failed_gates" in decision and decision["failed_gates"] != []:
            raise CalibrationSelectionError(
                "eligible_candidates entry must have no failed gates"
            )
        eligible.append(candidate)

    selected = tuple(
        _parse_receipt_candidate(value, f"selected_for_stage2[{index}]")
        for index, value in enumerate(selected_raw)
    )
    if len(set(selected)) != len(selected):
        raise CalibrationSelectionError("duplicate selected candidate")
    if len(selected) > 3:
        raise CalibrationSelectionError("selected_for_stage2 cannot exceed three")
    if any(candidate not in eligible for candidate in selected):
        raise CalibrationSelectionError(
            "selected_for_stage2 must be a subset of eligible_candidates"
        )

    should_allow = (
        protocol_status == "passed"
        and scientific_status == "passed"
        and formal
        and not unresolved
        and not retrospective
        and 1 <= len(selected) <= 3
    )
    if allowed is not should_allow:
        raise CalibrationSelectionError(
            "stage2_allowed contradicts protocol/scientific/selection state"
        )
    if scientific_status == "failed" and selected:
        raise CalibrationSelectionError(
            "scientific failure must not select Stage-2 candidates"
        )
    if allowed:
        # Positive receipts require the full frozen scientific provenance;
        # the three convenience booleans alone can never authorize work.
        for index, entry in enumerate(eligible_raw):
            decision = _mapping(entry, f"eligible_candidates[{index}]")
            if decision.get("eligible") is not True or decision.get("failed_gates") != []:
                raise CalibrationSelectionError(
                    "positive eligible entries require eligible=true and failed_gates=[]"
                )
        _validate_positive_authorization_contract(
            receipt, tuple(eligible), selected
        )
    elif "scientific_gate" in receipt:
        gate = _mapping(receipt["scientific_gate"], "scientific_gate")
        expected_mode = RETROSPECTIVE_MODE if retrospective else FORMAL_FROZEN_MODE
        if gate.get("mode") != expected_mode:
            raise CalibrationSelectionError(
                "scientific_gate.mode contradicts gate provenance fields"
            )
    return selected


__all__ = [
    "ALL_CANDIDATES",
    "CELLS_PER_RUN",
    "CONDITIONS",
    "CalibrationSelectionError",
    "Candidate",
    "CandidateDiagnosticEvidence",
    "CandidateRankingMetrics",
    "CandidateScienceSummary",
    "DATASETS",
    "FORMAL_FROZEN_MODE",
    "IMAGES_PER_CELL",
    "RETROSPECTIVE_MODE",
    "REQUIRED_HARD_GATES",
    "REQUIRED_PROTOCOL_AUDIT",
    "SELECTOR_PROTOCOL_ID",
    "SS_BN_PROTOCOL",
    "STAGE1_RECEIPT_FILENAME",
    "STAGE1_RECEIPT_SCHEMA_VERSION",
    "STAGE1_RECEIPT_TYPE",
    "ScientificGateDecision",
    "ScientificGateSpec",
    "rank_candidate_metrics",
    "select_stage1_candidates",
    "validate_stage1_scientific_receipt",
]
