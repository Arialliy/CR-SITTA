"""Leakage-resistant source-domain selector for Binary Episodic TENT.

This module deliberately contains no model, CUDA, dataset, or file-writing
code.  It consumes one aggregate record for every frozen
``dataset x BN protocol x corruption condition`` cell and returns a JSON-
serialisable selection receipt.  All ranking quantities are recomputed from
integer sufficient statistics; caller-supplied metric values are never used.

The v1 protocol has two phases:

* phase 1 evaluates all ten shared ``(optimizer, learning-rate)`` candidates;
* phase 2 evaluates the phase-1 top three in two additional fresh processes.

Both BN protocols remain evidence branches.  They are macro-averaged and are
never selected or eliminated by this selector.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from itertools import product
from typing import Any


SELECTOR_PROTOCOL_ID = "cr-sitta-binary-tent-source-calibration-selector-v1"

DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
BN_PROTOCOLS = (
    "single_image_spatial_batch_stats",
    "source_running_statistics",
)
CONDITIONS = (
    ("clean", 0),
    ("gaussian_noise", 1),
    ("gaussian_noise", 3),
    ("gaussian_noise", 5),
    ("gaussian_blur", 1),
    ("gaussian_blur", 3),
    ("gaussian_blur", 5),
    ("low_contrast", 1),
    ("low_contrast", 3),
    ("low_contrast", 5),
    ("stripe_noise", 1),
    ("stripe_noise", 3),
    ("stripe_noise", 5),
)
IMAGES_PER_CELL = 64
CELLS_PER_RUN = len(DATASETS) * len(BN_PROTOCOLS) * len(CONDITIONS)

OPTIMIZER_ORDER = ("Adam", "SGD")
LEARNING_RATES = tuple(
    Decimal(value) for value in ("1e-5", "3e-5", "1e-4", "3e-4", "1e-3")
)

REQUIRED_HARD_GATES = (
    "implementation_protocol_valid",
    "exact_reset",
    "label_firewall",
    "finite_values",
    "exactly_one_optimizer_step_per_episode",
    "source_pre_identity_contract",
    "tent_pre_identity_contract",
    "zero_test_opens",
)
REQUIRED_PROTOCOL_AUDIT = (
    "candidate_model_method_optimizer_rebuilt_before_run",
    "global_runtime_seal_valid",
    "fixed_seed_contract_valid",
)

# These quantities may be reported later as scientific diagnostics, but they
# are forbidden at this selector boundary so that they cannot silently enter
# hyperparameter selection.
FORBIDDEN_SELECTOR_FIELDS = frozenset(
    {
        "ater",
        "atrr",
        "ntg",
        "target_erasure",
        "target_erasure_rate",
        "target_recovery",
        "target_recovery_rate",
        "net_target_gain",
        "target_transition",
        "target_transitions",
        "target_transition_counts",
    }
)
FORBIDDEN_SELECTOR_FIELD_PREFIXES = (
    "ater_",
    "ater@",
    "atrr_",
    "atrr@",
    "ntg_",
    "ntg@",
    "target_erasure",
    "target_recovery",
    "target_transition",
    "net_target_gain",
)


class CalibrationSelectionError(ValueError):
    """Raised when calibration evidence violates the frozen v1 contract."""


@dataclass(frozen=True)
class Candidate:
    """A shared optimizer/LR candidate applied to both BN protocols."""

    optimizer: str
    learning_rate: Decimal

    @classmethod
    def from_values(cls, optimizer: Any, learning_rate: Any) -> "Candidate":
        if not isinstance(optimizer, str):
            raise CalibrationSelectionError("candidate optimizer must be a string")
        if optimizer not in OPTIMIZER_ORDER:
            raise CalibrationSelectionError(
                f"unsupported optimizer {optimizer!r}; expected {OPTIMIZER_ORDER}"
            )
        if isinstance(learning_rate, bool):
            raise CalibrationSelectionError("candidate learning rate cannot be bool")
        try:
            normalised_lr = Decimal(str(learning_rate))
        except (InvalidOperation, ValueError) as exc:
            raise CalibrationSelectionError(
                f"invalid candidate learning rate {learning_rate!r}"
            ) from exc
        if not normalised_lr.is_finite() or normalised_lr not in LEARNING_RATES:
            raise CalibrationSelectionError(
                "learning rate must be one of "
                + ", ".join(_decimal_text(value) for value in LEARNING_RATES)
            )
        return cls(optimizer=optimizer, learning_rate=normalised_lr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "learning_rate": float(self.learning_rate),
            "learning_rate_decimal": _decimal_text(self.learning_rate),
        }


ALL_CANDIDATES = tuple(
    Candidate(optimizer, learning_rate)
    for optimizer in OPTIMIZER_ORDER
    for learning_rate in LEARNING_RATES
)


@dataclass(frozen=True)
class CandidateRankingMetrics:
    """The seven frozen deterministic ranking fields for one candidate."""

    candidate: Candidate
    primary_macro_delta_iou: Fraction
    worst_run_macro_delta_iou: Fraction
    clean_macro_delta_iou: Fraction
    macro_fa_increase_per_million_pixels: Fraction
    macro_pd_delta: Fraction


@dataclass(frozen=True)
class _EndpointCounts:
    intersection_pixels: int
    union_pixels: int
    false_alarm_pixels: int
    total_image_pixels: int
    detected_targets: int
    total_targets: int

    @property
    def global_iou(self) -> Fraction:
        return Fraction(self.intersection_pixels, self.union_pixels)

    @property
    def fa_per_million_pixels(self) -> Fraction:
        return Fraction(
            self.false_alarm_pixels * 1_000_000, self.total_image_pixels
        )

    @property
    def pd(self) -> Fraction:
        return Fraction(self.detected_targets, self.total_targets)

    def to_dict(self) -> dict[str, int]:
        return {
            "intersection_pixels": self.intersection_pixels,
            "union_pixels": self.union_pixels,
            "false_alarm_pixels": self.false_alarm_pixels,
            "total_image_pixels": self.total_image_pixels,
            "detected_targets": self.detected_targets,
            "total_targets": self.total_targets,
        }


@dataclass(frozen=True)
class _CellRecord:
    stage: int
    process_id: str
    fresh_process: bool
    candidate: Candidate
    dataset: str
    bn_protocol: str
    corruption: str
    severity: int
    image_count: int
    optimizer_steps_total: int
    test_image_opens: int
    test_label_opens: int
    method_label_accesses: int
    hard_gates: tuple[str, ...]
    protocol_audit: tuple[str, ...]
    tent_pre: _EndpointCounts
    tent_post: _EndpointCounts

    @property
    def cell_key(self) -> tuple[str, str, str, int]:
        return (self.dataset, self.bn_protocol, self.corruption, self.severity)


@dataclass(frozen=True)
class _RunSummary:
    candidate: Candidate
    process_id: str
    macro_delta_iou: Fraction
    clean_macro_delta_iou: Fraction
    macro_fa_increase: Fraction
    macro_pd_delta: Fraction
    cell_receipts: tuple[dict[str, Any], ...]


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "E").replace("E+", "e+").replace("E-", "e-")


def _normalise_field_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _reject_forbidden_fields(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalised = _normalise_field_name(key)
            if normalised in FORBIDDEN_SELECTOR_FIELDS or normalised.startswith(
                FORBIDDEN_SELECTOR_FIELD_PREFIXES
            ):
                raise CalibrationSelectionError(
                    f"forbidden target-transition selection field at {path}.{key}"
                )
            _reject_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationSelectionError(f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationSelectionError(f"{label} must be an integer")
    if value < minimum:
        raise CalibrationSelectionError(f"{label} must be >= {minimum}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationSelectionError(f"{label} must be a non-empty string")
    return value


def _parse_endpoint(value: Any, label: str) -> _EndpointCounts:
    endpoint = _mapping(value, label)
    required = (
        "intersection_pixels",
        "union_pixels",
        "false_alarm_pixels",
        "total_image_pixels",
        "detected_targets",
        "total_targets",
    )
    missing = [key for key in required if key not in endpoint]
    if missing:
        raise CalibrationSelectionError(f"{label} missing fields: {missing}")
    result = _EndpointCounts(
        intersection_pixels=_integer(
            endpoint["intersection_pixels"], f"{label}.intersection_pixels"
        ),
        union_pixels=_integer(
            endpoint["union_pixels"], f"{label}.union_pixels", minimum=1
        ),
        false_alarm_pixels=_integer(
            endpoint["false_alarm_pixels"], f"{label}.false_alarm_pixels"
        ),
        total_image_pixels=_integer(
            endpoint["total_image_pixels"],
            f"{label}.total_image_pixels",
            minimum=1,
        ),
        detected_targets=_integer(
            endpoint["detected_targets"], f"{label}.detected_targets"
        ),
        total_targets=_integer(
            endpoint["total_targets"], f"{label}.total_targets", minimum=1
        ),
    )
    if result.intersection_pixels > result.union_pixels:
        raise CalibrationSelectionError(
            f"{label} intersection_pixels exceeds union_pixels"
        )
    if result.union_pixels > result.total_image_pixels:
        raise CalibrationSelectionError(
            f"{label} union_pixels exceeds total_image_pixels"
        )
    if result.false_alarm_pixels > result.total_image_pixels:
        raise CalibrationSelectionError(
            f"{label} false_alarm_pixels exceeds total_image_pixels"
        )
    if result.detected_targets > result.total_targets:
        raise CalibrationSelectionError(
            f"{label} detected_targets exceeds total_targets"
        )
    return result


def _parse_record(value: Any, index: int, expected_stage: int) -> _CellRecord:
    label = f"records[{index}]"
    record = _mapping(value, label)
    _reject_forbidden_fields(record, label)

    required = (
        "stage",
        "process_id",
        "fresh_process",
        "candidate",
        "dataset",
        "bn_protocol",
        "corruption",
        "severity",
        "image_count",
        "optimizer_steps_total",
        "test_image_opens",
        "test_label_opens",
        "method_label_accesses",
        "hard_gates",
        "protocol_audit",
        "endpoints",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise CalibrationSelectionError(f"{label} missing fields: {missing}")

    stage = _integer(record["stage"], f"{label}.stage", minimum=1)
    if stage != expected_stage:
        raise CalibrationSelectionError(
            f"{label}.stage is {stage}, expected {expected_stage}"
        )
    process_id = _string(record["process_id"], f"{label}.process_id")
    fresh_process = record["fresh_process"]
    if not isinstance(fresh_process, bool):
        raise CalibrationSelectionError(f"{label}.fresh_process must be bool")
    if not fresh_process:
        raise CalibrationSelectionError(
            f"{label} stage-{stage} evidence must come from a fresh process"
        )

    candidate_record = _mapping(record["candidate"], f"{label}.candidate")
    if "optimizer" not in candidate_record or "learning_rate" not in candidate_record:
        raise CalibrationSelectionError(
            f"{label}.candidate requires optimizer and learning_rate"
        )
    candidate = Candidate.from_values(
        candidate_record["optimizer"], candidate_record["learning_rate"]
    )

    dataset = _string(record["dataset"], f"{label}.dataset")
    if dataset not in DATASETS:
        raise CalibrationSelectionError(f"{label} has unsupported dataset {dataset!r}")
    bn_protocol = _string(record["bn_protocol"], f"{label}.bn_protocol")
    if bn_protocol not in BN_PROTOCOLS:
        raise CalibrationSelectionError(
            f"{label} has unsupported BN protocol {bn_protocol!r}"
        )
    corruption = _string(record["corruption"], f"{label}.corruption")
    severity = _integer(record["severity"], f"{label}.severity")
    if (corruption, severity) not in CONDITIONS:
        raise CalibrationSelectionError(
            f"{label} has unsupported condition {(corruption, severity)!r}"
        )

    image_count = _integer(record["image_count"], f"{label}.image_count")
    if image_count != IMAGES_PER_CELL:
        raise CalibrationSelectionError(
            f"{label}.image_count is {image_count}, expected {IMAGES_PER_CELL}"
        )
    optimizer_steps_total = _integer(
        record["optimizer_steps_total"], f"{label}.optimizer_steps_total"
    )
    if optimizer_steps_total != image_count:
        raise CalibrationSelectionError(
            f"{label} must record exactly one optimizer step per image"
        )
    test_image_opens = _integer(
        record["test_image_opens"], f"{label}.test_image_opens"
    )
    test_label_opens = _integer(
        record["test_label_opens"], f"{label}.test_label_opens"
    )
    method_label_accesses = _integer(
        record["method_label_accesses"], f"{label}.method_label_accesses"
    )
    if (test_image_opens, test_label_opens, method_label_accesses) != (0, 0, 0):
        raise CalibrationSelectionError(
            f"{label} violates the test/label access firewall"
        )

    hard_gates = _mapping(record["hard_gates"], f"{label}.hard_gates")
    if set(hard_gates) != set(REQUIRED_HARD_GATES):
        missing_gates = sorted(set(REQUIRED_HARD_GATES) - set(hard_gates))
        extra_gates = sorted(set(hard_gates) - set(REQUIRED_HARD_GATES))
        raise CalibrationSelectionError(
            f"{label}.hard_gates must be exactly the frozen set; "
            f"missing={missing_gates}, extra={extra_gates}"
        )
    failed_gates = [key for key in REQUIRED_HARD_GATES if hard_gates[key] is not True]
    if failed_gates:
        raise CalibrationSelectionError(
            f"{label} failed required hard gates: {failed_gates}"
        )

    protocol_audit = _mapping(
        record["protocol_audit"], f"{label}.protocol_audit"
    )
    if set(protocol_audit) != set(REQUIRED_PROTOCOL_AUDIT):
        missing_audit = sorted(set(REQUIRED_PROTOCOL_AUDIT) - set(protocol_audit))
        extra_audit = sorted(set(protocol_audit) - set(REQUIRED_PROTOCOL_AUDIT))
        raise CalibrationSelectionError(
            f"{label}.protocol_audit must be exactly the frozen set; "
            f"missing={missing_audit}, extra={extra_audit}"
        )
    failed_audit = [
        key for key in REQUIRED_PROTOCOL_AUDIT if protocol_audit[key] is not True
    ]
    if failed_audit:
        raise CalibrationSelectionError(
            f"{label} failed implementation protocol audit: {failed_audit}"
        )

    endpoints = _mapping(record["endpoints"], f"{label}.endpoints")
    if "tent_pre" not in endpoints or "tent_post" not in endpoints:
        raise CalibrationSelectionError(
            f"{label}.endpoints requires tent_pre and tent_post"
        )
    tent_pre = _parse_endpoint(endpoints["tent_pre"], f"{label}.tent_pre")
    tent_post = _parse_endpoint(endpoints["tent_post"], f"{label}.tent_post")
    if tent_pre.total_image_pixels != tent_post.total_image_pixels:
        raise CalibrationSelectionError(
            f"{label} pre/post total_image_pixels denominators differ"
        )
    if tent_pre.total_targets != tent_post.total_targets:
        raise CalibrationSelectionError(
            f"{label} pre/post total_targets denominators differ"
        )

    return _CellRecord(
        stage=stage,
        process_id=process_id,
        fresh_process=fresh_process,
        candidate=candidate,
        dataset=dataset,
        bn_protocol=bn_protocol,
        corruption=corruption,
        severity=severity,
        image_count=image_count,
        optimizer_steps_total=optimizer_steps_total,
        test_image_opens=test_image_opens,
        test_label_opens=test_label_opens,
        method_label_accesses=method_label_accesses,
        hard_gates=REQUIRED_HARD_GATES,
        protocol_audit=REQUIRED_PROTOCOL_AUDIT,
        tent_pre=tent_pre,
        tent_post=tent_post,
    )


EXPECTED_CELL_KEYS = tuple(
    (dataset, protocol, corruption, severity)
    for dataset, protocol, (corruption, severity) in product(
        DATASETS, BN_PROTOCOLS, CONDITIONS
    )
)
EXPECTED_CELL_KEY_SET = frozenset(EXPECTED_CELL_KEYS)


def _parse_records(
    records: Iterable[Mapping[str, Any]], expected_stage: int
) -> tuple[_CellRecord, ...]:
    if isinstance(records, (str, bytes, Mapping)):
        raise CalibrationSelectionError("records must be an iterable of mappings")
    parsed = tuple(
        _parse_record(record, index, expected_stage)
        for index, record in enumerate(records)
    )
    if not parsed:
        raise CalibrationSelectionError("records cannot be empty")
    return parsed


def _group_runs(
    records: Sequence[_CellRecord],
) -> dict[Candidate, dict[str, tuple[_CellRecord, ...]]]:
    mutable: dict[Candidate, dict[str, list[_CellRecord]]] = {}
    seen: set[tuple[Candidate, str, tuple[str, str, str, int]]] = set()
    for record in records:
        unique_key = (record.candidate, record.process_id, record.cell_key)
        if unique_key in seen:
            raise CalibrationSelectionError(
                "duplicate calibration cell for "
                f"candidate={record.candidate.to_dict()}, "
                f"process_id={record.process_id!r}, cell={record.cell_key!r}"
            )
        seen.add(unique_key)
        mutable.setdefault(record.candidate, {}).setdefault(
            record.process_id, []
        ).append(record)
    return {
        candidate: {
            process_id: tuple(values) for process_id, values in processes.items()
        }
        for candidate, processes in mutable.items()
    }


def _validate_complete_run(
    candidate: Candidate, process_id: str, records: Sequence[_CellRecord]
) -> None:
    observed = frozenset(record.cell_key for record in records)
    if observed != EXPECTED_CELL_KEY_SET:
        missing = [key for key in EXPECTED_CELL_KEYS if key not in observed]
        extra = sorted(observed - EXPECTED_CELL_KEY_SET)
        raise CalibrationSelectionError(
            "incomplete calibration run for "
            f"candidate={candidate.to_dict()}, process_id={process_id!r}; "
            f"expected_cells={CELLS_PER_RUN}, observed_cells={len(observed)}, "
            f"first_missing={missing[:3]}, extra={extra[:3]}"
        )


def _fraction_receipt(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
        "value": float(value),
    }


def _mean(values: Sequence[Fraction], label: str) -> Fraction:
    if not values:
        raise CalibrationSelectionError(f"cannot compute empty mean for {label}")
    return sum(values, Fraction(0, 1)) / len(values)


def _summarise_run(
    candidate: Candidate, process_id: str, records: Sequence[_CellRecord]
) -> _RunSummary:
    _validate_complete_run(candidate, process_id, records)
    indexed = {record.cell_key: record for record in records}
    iou_deltas: list[Fraction] = []
    clean_iou_deltas: list[Fraction] = []
    fa_deltas: list[Fraction] = []
    pd_deltas: list[Fraction] = []
    cell_receipts: list[dict[str, Any]] = []

    for cell_key in EXPECTED_CELL_KEYS:
        record = indexed[cell_key]
        iou_delta = record.tent_post.global_iou - record.tent_pre.global_iou
        fa_delta = (
            record.tent_post.fa_per_million_pixels
            - record.tent_pre.fa_per_million_pixels
        )
        pd_delta = record.tent_post.pd - record.tent_pre.pd
        iou_deltas.append(iou_delta)
        fa_deltas.append(fa_delta)
        pd_deltas.append(pd_delta)
        if (record.corruption, record.severity) == ("clean", 0):
            clean_iou_deltas.append(iou_delta)

        cell_receipts.append(
            {
                "dataset": record.dataset,
                "bn_protocol": record.bn_protocol,
                "condition": {
                    "corruption": record.corruption,
                    "severity": record.severity,
                },
                "image_count": record.image_count,
                "optimizer_steps_total": record.optimizer_steps_total,
                "audit_counters": {
                    "test_image_opens": record.test_image_opens,
                    "test_label_opens": record.test_label_opens,
                    "method_label_accesses": record.method_label_accesses,
                },
                "hard_gates": {key: True for key in record.hard_gates},
                "protocol_audit": {
                    key: True for key in record.protocol_audit
                },
                "tent_pre_counts": record.tent_pre.to_dict(),
                "tent_post_counts": record.tent_post.to_dict(),
                "derived": {
                    "tent_pre_global_iou": _fraction_receipt(
                        record.tent_pre.global_iou
                    ),
                    "tent_post_global_iou": _fraction_receipt(
                        record.tent_post.global_iou
                    ),
                    "global_iou_delta": _fraction_receipt(iou_delta),
                    "fa_increase_per_million_pixels": _fraction_receipt(fa_delta),
                    "pd_delta": _fraction_receipt(pd_delta),
                },
            }
        )

    return _RunSummary(
        candidate=candidate,
        process_id=process_id,
        macro_delta_iou=_mean(iou_deltas, "macro IoU delta"),
        clean_macro_delta_iou=_mean(clean_iou_deltas, "clean macro IoU delta"),
        macro_fa_increase=_mean(fa_deltas, "macro FA increase"),
        macro_pd_delta=_mean(pd_deltas, "macro Pd delta"),
        cell_receipts=tuple(cell_receipts),
    )


def _candidate_metrics(
    candidate: Candidate, runs: Sequence[_RunSummary]
) -> CandidateRankingMetrics:
    if not runs:
        raise CalibrationSelectionError("candidate has no run summaries")
    if any(run.candidate != candidate for run in runs):
        raise CalibrationSelectionError("run summary candidate mismatch")
    return CandidateRankingMetrics(
        candidate=candidate,
        primary_macro_delta_iou=_mean(
            [run.macro_delta_iou for run in runs], "run macro IoU delta"
        ),
        worst_run_macro_delta_iou=min(run.macro_delta_iou for run in runs),
        clean_macro_delta_iou=_mean(
            [run.clean_macro_delta_iou for run in runs],
            "run clean macro IoU delta",
        ),
        macro_fa_increase_per_million_pixels=_mean(
            [run.macro_fa_increase for run in runs], "run macro FA increase"
        ),
        macro_pd_delta=_mean(
            [run.macro_pd_delta for run in runs], "run macro Pd delta"
        ),
    )


def _validate_ranking_metrics(metric: CandidateRankingMetrics) -> None:
    if metric.candidate not in ALL_CANDIDATES:
        raise CalibrationSelectionError(
            f"ranking contains unsupported candidate {metric.candidate.to_dict()}"
        )
    for field_name in (
        "primary_macro_delta_iou",
        "worst_run_macro_delta_iou",
        "clean_macro_delta_iou",
        "macro_fa_increase_per_million_pixels",
        "macro_pd_delta",
    ):
        if not isinstance(getattr(metric, field_name), Fraction):
            raise CalibrationSelectionError(
                f"ranking field {field_name} must be fractions.Fraction"
            )


def rank_candidate_metrics(
    metrics: Iterable[CandidateRankingMetrics],
) -> tuple[CandidateRankingMetrics, ...]:
    """Rank candidates with the frozen seven-field deterministic tie-break.

    Order is: larger mean macro IoU delta, larger worst-run macro IoU delta,
    larger clean macro IoU delta, smaller macro FA increase, larger macro Pd
    delta, smaller LR, then the fixed optimizer order ``Adam, SGD``.
    """

    values = tuple(metrics)
    if not values:
        raise CalibrationSelectionError("ranking metrics cannot be empty")
    seen: set[Candidate] = set()
    for metric in values:
        if not isinstance(metric, CandidateRankingMetrics):
            raise CalibrationSelectionError(
                "metrics must contain CandidateRankingMetrics instances"
            )
        _validate_ranking_metrics(metric)
        if metric.candidate in seen:
            raise CalibrationSelectionError(
                f"duplicate ranking candidate {metric.candidate.to_dict()}"
            )
        seen.add(metric.candidate)

    def key(metric: CandidateRankingMetrics) -> tuple[Any, ...]:
        return (
            -metric.primary_macro_delta_iou,
            -metric.worst_run_macro_delta_iou,
            -metric.clean_macro_delta_iou,
            metric.macro_fa_increase_per_million_pixels,
            -metric.macro_pd_delta,
            metric.candidate.learning_rate,
            OPTIMIZER_ORDER.index(metric.candidate.optimizer),
        )

    return tuple(sorted(values, key=key))


def _metric_receipt(metric: CandidateRankingMetrics) -> dict[str, Any]:
    return {
        "candidate": metric.candidate.to_dict(),
        "primary_mean_over_runs_macro_global_iou_delta": _fraction_receipt(
            metric.primary_macro_delta_iou
        ),
        "worst_run_macro_global_iou_delta": _fraction_receipt(
            metric.worst_run_macro_delta_iou
        ),
        "clean_mean_over_runs_macro_global_iou_delta": _fraction_receipt(
            metric.clean_macro_delta_iou
        ),
        "mean_over_runs_macro_fa_increase_per_million_pixels": _fraction_receipt(
            metric.macro_fa_increase_per_million_pixels
        ),
        "mean_over_runs_macro_pd_delta": _fraction_receipt(metric.macro_pd_delta),
    }


def _run_receipt(run: _RunSummary) -> dict[str, Any]:
    return {
        "process_id": run.process_id,
        "cell_count": CELLS_PER_RUN,
        "images_per_cell": IMAGES_PER_CELL,
        "episode_count": CELLS_PER_RUN * IMAGES_PER_CELL,
        "macro_global_iou_delta": _fraction_receipt(run.macro_delta_iou),
        "clean_macro_global_iou_delta": _fraction_receipt(
            run.clean_macro_delta_iou
        ),
        "macro_fa_increase_per_million_pixels": _fraction_receipt(
            run.macro_fa_increase
        ),
        "macro_pd_delta": _fraction_receipt(run.macro_pd_delta),
        "cells_in_frozen_order": list(run.cell_receipts),
    }


def _contract_receipt() -> dict[str, Any]:
    return {
        "selector_protocol_id": SELECTOR_PROTOCOL_ID,
        "paper_result": False,
        "calibration_source": "train-derived 64 images per dataset",
        "independent_validation": False,
        "bn_protocol_is_selected_or_eliminated": False,
        "target_transition_metrics_used_for_selection": False,
        "entropy_decrease_required": False,
        "performance_hard_thresholds": [],
        "cell_definition": "dataset x BN protocol x corruption condition",
        "cell_count_per_run": CELLS_PER_RUN,
        "images_per_cell": IMAGES_PER_CELL,
        "equal_weighting": {
            "within_run": "arithmetic mean of 78 cell deltas; every cell weight=1/78",
            "across_runs": "arithmetic mean of three run-level macro values; every run weight=1/3",
        },
        "stage2_process_contract": {
            "fresh_process_count": 2,
            "same_two_process_ids_each_run_all_top3_candidates": True,
            "process_ids_disjoint_from_stage1": True,
            "candidate_order": "frozen stage1 top3 order",
            "rebuild_model_method_optimizer_before_each_candidate": True,
            "verify_global_runtime_seal_before_and_after_each_candidate": True,
            "reapply_and_verify_fixed_seed_before_each_candidate": True,
        },
        "metric_definitions": {
            "global_iou": "intersection_pixels / union_pixels",
            "cell_global_iou_delta": "tent_post_global_iou - protocol_matched_tent_pre_global_iou",
            "fa_per_million_pixels": "false_alarm_pixels / total_image_pixels * 1,000,000",
            "cell_fa_increase": "tent_post_fa_per_million_pixels - tent_pre_fa_per_million_pixels",
            "pd": "detected_targets / total_targets",
            "cell_pd_delta": "tent_post_pd - tent_pre_pd",
            "clean_macro": "equal mean over 3 datasets x 2 BN protocols for clean severity 0",
        },
        "tie_break_order": [
            "higher primary mean-over-runs macro global-IoU delta",
            "higher worst-run macro global-IoU delta",
            "higher clean mean-over-runs macro global-IoU delta",
            "lower mean-over-runs macro FA increase per million pixels",
            "higher mean-over-runs macro Pd delta",
            "lower learning rate",
            "fixed optimizer order: Adam then SGD",
        ],
        "required_hard_gates": list(REQUIRED_HARD_GATES),
        "implementation_protocol_audit": list(REQUIRED_PROTOCOL_AUDIT),
        "forbidden_selector_fields": sorted(FORBIDDEN_SELECTOR_FIELDS),
    }


def _stage1_summaries(
    stage1_records: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[Candidate, _RunSummary],
    tuple[CandidateRankingMetrics, ...],
]:
    parsed = _parse_records(stage1_records, expected_stage=1)
    grouped = _group_runs(parsed)
    if set(grouped) != set(ALL_CANDIDATES):
        missing = [candidate.to_dict() for candidate in ALL_CANDIDATES if candidate not in grouped]
        extra = [candidate.to_dict() for candidate in grouped if candidate not in ALL_CANDIDATES]
        raise CalibrationSelectionError(
            f"stage 1 must contain all ten candidates; missing={missing}, extra={extra}"
        )

    summaries: dict[Candidate, _RunSummary] = {}
    for candidate in ALL_CANDIDATES:
        processes = grouped[candidate]
        if len(processes) != 1:
            raise CalibrationSelectionError(
                "stage 1 requires exactly one complete process per candidate; "
                f"candidate={candidate.to_dict()}, processes={sorted(processes)}"
            )
        process_id, records = next(iter(processes.items()))
        summaries[candidate] = _summarise_run(candidate, process_id, records)

    observed_episodes = len(parsed) * IMAGES_PER_CELL
    expected_episodes = len(ALL_CANDIDATES) * CELLS_PER_RUN * IMAGES_PER_CELL
    if observed_episodes != expected_episodes:
        raise CalibrationSelectionError(
            f"stage 1 episode count {observed_episodes} != {expected_episodes}"
        )
    ranking = rank_candidate_metrics(
        _candidate_metrics(candidate, [summaries[candidate]])
        for candidate in ALL_CANDIDATES
    )
    return summaries, ranking


def select_stage1_top3(
    stage1_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate all 49,920 phase-1 episodes and return the exact top three."""

    summaries, ranking = _stage1_summaries(stage1_records)
    top3 = tuple(metric.candidate for metric in ranking[:3])
    ranking_receipts = []
    for rank, metric in enumerate(ranking, start=1):
        value = _metric_receipt(metric)
        value.update(
            {
                "rank": rank,
                "run_count": 1,
                "runs": [_run_receipt(summaries[metric.candidate])],
            }
        )
        ranking_receipts.append(value)
    return {
        "schema_version": 1,
        "receipt_type": "stage1_top3",
        "contract": _contract_receipt(),
        "validation": {
            "candidate_count": len(ALL_CANDIDATES),
            "candidate_run_count": len(ALL_CANDIDATES),
            "aggregate_cell_record_count": len(ALL_CANDIDATES) * CELLS_PER_RUN,
            "episode_count": len(ALL_CANDIDATES)
            * CELLS_PER_RUN
            * IMAGES_PER_CELL,
            "all_required_hard_gates_passed": True,
            "all_required_implementation_protocol_audits_passed": True,
            "test_image_opens": 0,
            "test_label_opens": 0,
            "method_label_accesses": 0,
        },
        "ranking": ranking_receipts,
        "top3": [candidate.to_dict() for candidate in top3],
    }


def select_final_candidate(
    stage1_records: Iterable[Mapping[str, Any]],
    stage2_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one shared optimizer/LR from phase 1 plus two fresh repeats."""

    stage1_summaries, stage1_ranking = _stage1_summaries(stage1_records)
    top3 = tuple(metric.candidate for metric in stage1_ranking[:3])

    parsed_stage2 = _parse_records(stage2_records, expected_stage=2)
    grouped_stage2 = _group_runs(parsed_stage2)
    if set(grouped_stage2) != set(top3):
        missing = [candidate.to_dict() for candidate in top3 if candidate not in grouped_stage2]
        extra = [candidate.to_dict() for candidate in grouped_stage2 if candidate not in top3]
        raise CalibrationSelectionError(
            "stage 2 must contain exactly the stage-1 top three candidates; "
            f"missing={missing}, extra={extra}"
        )

    stage1_processes = {summary.process_id for summary in stage1_summaries.values()}
    shared_stage2_processes: frozenset[str] | None = None
    stage2_summaries: dict[Candidate, tuple[_RunSummary, ...]] = {}
    for candidate in top3:
        processes = grouped_stage2[candidate]
        if len(processes) != 2:
            raise CalibrationSelectionError(
                "stage 2 requires exactly two fresh processes per top-three candidate; "
                f"candidate={candidate.to_dict()}, processes={sorted(processes)}"
            )
        process_ids = frozenset(processes)
        if shared_stage2_processes is None:
            shared_stage2_processes = process_ids
        elif process_ids != shared_stage2_processes:
            raise CalibrationSelectionError(
                "the same two fresh stage-2 process ids must each run all top-three "
                f"candidates; expected={sorted(shared_stage2_processes)}, "
                f"observed={sorted(process_ids)}"
            )
        values: list[_RunSummary] = []
        for process_id, records in sorted(processes.items()):
            if process_id in stage1_processes:
                raise CalibrationSelectionError(
                    f"stage-2 process {process_id!r} reuses a stage-1 process id"
                )
            values.append(_summarise_run(candidate, process_id, records))
        stage2_summaries[candidate] = tuple(values)

    expected_stage2_episodes = 3 * 2 * CELLS_PER_RUN * IMAGES_PER_CELL
    observed_stage2_episodes = len(parsed_stage2) * IMAGES_PER_CELL
    if observed_stage2_episodes != expected_stage2_episodes:
        raise CalibrationSelectionError(
            f"stage 2 episode count {observed_stage2_episodes} "
            f"!= {expected_stage2_episodes}"
        )

    all_runs = {
        candidate: (stage1_summaries[candidate], *stage2_summaries[candidate])
        for candidate in top3
    }
    final_ranking = rank_candidate_metrics(
        _candidate_metrics(candidate, all_runs[candidate]) for candidate in top3
    )

    stage1_ranking_receipts = []
    for rank, metric in enumerate(stage1_ranking, start=1):
        value = _metric_receipt(metric)
        value.update(
            {
                "rank": rank,
                "run_count": 1,
                "runs": [_run_receipt(stage1_summaries[metric.candidate])],
            }
        )
        stage1_ranking_receipts.append(value)

    final_ranking_receipts = []
    for rank, metric in enumerate(final_ranking, start=1):
        value = _metric_receipt(metric)
        value.update(
            {
                "rank": rank,
                "run_count": 3,
                "runs": [_run_receipt(run) for run in all_runs[metric.candidate]],
            }
        )
        final_ranking_receipts.append(value)

    selected = final_ranking[0].candidate
    return {
        "schema_version": 1,
        "receipt_type": "final_shared_optimizer_lr_selection",
        "contract": _contract_receipt(),
        "validation": {
            "stage1_candidate_count": len(ALL_CANDIDATES),
            "stage1_episode_count": len(ALL_CANDIDATES)
            * CELLS_PER_RUN
            * IMAGES_PER_CELL,
            "stage2_candidate_count": 3,
            "stage2_fresh_processes_per_candidate": 2,
            "stage2_fresh_process_count": 2,
            "stage2_processes_shared_across_top3": True,
            "stage2_process_ids": sorted(shared_stage2_processes or ()),
            "stage2_episode_count": expected_stage2_episodes,
            "total_calibration_episode_count": (
                len(ALL_CANDIDATES) * CELLS_PER_RUN * IMAGES_PER_CELL
                + expected_stage2_episodes
            ),
            "all_required_hard_gates_passed": True,
            "all_required_implementation_protocol_audits_passed": True,
            "candidate_model_method_optimizer_rebuilds_verified": True,
            "global_runtime_seals_verified": True,
            "fixed_seed_contracts_verified": True,
            "test_image_opens": 0,
            "test_label_opens": 0,
            "method_label_accesses": 0,
        },
        "stage1_ranking": stage1_ranking_receipts,
        "stage1_top3": [candidate.to_dict() for candidate in top3],
        "final_ranking": final_ranking_receipts,
        "selected_candidate": selected.to_dict(),
        "selected_bn_protocol": None,
        "both_bn_protocols_retained_for_formal_evaluation": True,
    }


__all__ = [
    "ALL_CANDIDATES",
    "BN_PROTOCOLS",
    "CELLS_PER_RUN",
    "CONDITIONS",
    "CalibrationSelectionError",
    "Candidate",
    "CandidateRankingMetrics",
    "DATASETS",
    "FORBIDDEN_SELECTOR_FIELDS",
    "FORBIDDEN_SELECTOR_FIELD_PREFIXES",
    "IMAGES_PER_CELL",
    "LEARNING_RATES",
    "OPTIMIZER_ORDER",
    "REQUIRED_HARD_GATES",
    "REQUIRED_PROTOCOL_AUDIT",
    "SELECTOR_PROTOCOL_ID",
    "rank_candidate_metrics",
    "select_final_candidate",
    "select_stage1_top3",
]
