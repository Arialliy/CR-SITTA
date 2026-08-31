"""Pure-CPU SS-only calibration selector for Binary Episodic TENT v2.

Only ``source_running_statistics`` (SS) records may influence selection.  The
selected optimizer and learning rate are nevertheless frozen for later use by
both SS and BS; BS is an application branch, never calibration evidence here.
This module performs no dataset, model, CUDA, or filesystem operations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from tta import binary_tent_calibration_selector as _v1


SELECTOR_PROTOCOL_ID = "cr-sitta-binary-tent-ss-calibration-selector-v2"

DATASETS = _v1.DATASETS
CONDITIONS = _v1.CONDITIONS
IMAGES_PER_CELL = _v1.IMAGES_PER_CELL
SS_BN_PROTOCOL = "source_running_statistics"
BS_BN_PROTOCOL = "single_image_spatial_batch_stats"
BN_PROTOCOLS = (SS_BN_PROTOCOL,)
APPLICATION_PROTOCOLS = ("SS", "BS")
APPLICATION_PROTOCOL_IDS = (SS_BN_PROTOCOL, BS_BN_PROTOCOL)
CELLS_PER_RUN = len(DATASETS) * len(CONDITIONS)

OPTIMIZER_ORDER = _v1.OPTIMIZER_ORDER
LEARNING_RATES = _v1.LEARNING_RATES
REQUIRED_HARD_GATES = _v1.REQUIRED_HARD_GATES
REQUIRED_PROTOCOL_AUDIT = _v1.REQUIRED_PROTOCOL_AUDIT
FORBIDDEN_SELECTOR_FIELDS = _v1.FORBIDDEN_SELECTOR_FIELDS
FORBIDDEN_SELECTOR_FIELD_PREFIXES = _v1.FORBIDDEN_SELECTOR_FIELD_PREFIXES

CalibrationSelectionError = _v1.CalibrationSelectionError
Candidate = _v1.Candidate
CandidateRankingMetrics = _v1.CandidateRankingMetrics
ALL_CANDIDATES = _v1.ALL_CANDIDATES
rank_candidate_metrics = _v1.rank_candidate_metrics

EXPECTED_CELL_KEYS = tuple(
    (dataset, SS_BN_PROTOCOL, corruption, severity)
    for dataset in DATASETS
    for corruption, severity in CONDITIONS
)
EXPECTED_CELL_KEY_SET = frozenset(EXPECTED_CELL_KEYS)


def _parse_ss_records(
    records: Iterable[Mapping[str, Any]], *, expected_stage: int
) -> tuple[_v1._CellRecord, ...]:
    if isinstance(records, (str, bytes, Mapping)):
        raise CalibrationSelectionError("records must be an iterable of mappings")
    materialized = tuple(records)
    for index, raw in enumerate(materialized):
        record = _v1._mapping(raw, f"records[{index}]")
        if (
            "bn_protocol" in record
            and record["bn_protocol"] != SS_BN_PROTOCOL
        ):
            raise CalibrationSelectionError(
                "TENT-SS selector accepts only source_running_statistics; "
                f"records[{index}].bn_protocol={record['bn_protocol']!r}"
            )
    parsed = _v1._parse_records(materialized, expected_stage=expected_stage)
    if any(record.bn_protocol != SS_BN_PROTOCOL for record in parsed):
        raise CalibrationSelectionError(
            "TENT-SS selector evidence must be exclusively SS"
        )
    return parsed


def _validate_complete_ss_run(
    candidate: Candidate,
    process_id: str,
    records: Sequence[_v1._CellRecord],
) -> None:
    observed = frozenset(record.cell_key for record in records)
    if observed != EXPECTED_CELL_KEY_SET:
        missing = [key for key in EXPECTED_CELL_KEYS if key not in observed]
        extra = sorted(observed - EXPECTED_CELL_KEY_SET)
        raise CalibrationSelectionError(
            "incomplete SS calibration run for "
            f"candidate={candidate.to_dict()}, process_id={process_id!r}; "
            f"expected_cells={CELLS_PER_RUN}, observed_cells={len(observed)}, "
            f"first_missing={missing[:3]}, extra={extra[:3]}"
        )


def _summarise_ss_run(
    candidate: Candidate,
    process_id: str,
    records: Sequence[_v1._CellRecord],
) -> _v1._RunSummary:
    _validate_complete_ss_run(candidate, process_id, records)
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
                    "tent_pre_global_iou": _v1._fraction_receipt(
                        record.tent_pre.global_iou
                    ),
                    "tent_post_global_iou": _v1._fraction_receipt(
                        record.tent_post.global_iou
                    ),
                    "global_iou_delta": _v1._fraction_receipt(iou_delta),
                    "fa_increase_per_million_pixels": _v1._fraction_receipt(
                        fa_delta
                    ),
                    "pd_delta": _v1._fraction_receipt(pd_delta),
                },
            }
        )
    return _v1._RunSummary(
        candidate=candidate,
        process_id=process_id,
        macro_delta_iou=_v1._mean(iou_deltas, "SS macro IoU delta"),
        clean_macro_delta_iou=_v1._mean(
            clean_iou_deltas, "SS clean macro IoU delta"
        ),
        macro_fa_increase=_v1._mean(fa_deltas, "SS macro FA increase"),
        macro_pd_delta=_v1._mean(pd_deltas, "SS macro Pd delta"),
        cell_receipts=tuple(cell_receipts),
    )


def _run_receipt(run: _v1._RunSummary) -> dict[str, Any]:
    return {
        "process_id": run.process_id,
        "cell_count": CELLS_PER_RUN,
        "images_per_cell": IMAGES_PER_CELL,
        "episode_count": CELLS_PER_RUN * IMAGES_PER_CELL,
        "macro_global_iou_delta": _v1._fraction_receipt(run.macro_delta_iou),
        "clean_macro_global_iou_delta": _v1._fraction_receipt(
            run.clean_macro_delta_iou
        ),
        "macro_fa_increase_per_million_pixels": _v1._fraction_receipt(
            run.macro_fa_increase
        ),
        "macro_pd_delta": _v1._fraction_receipt(run.macro_pd_delta),
        "cells_in_frozen_order": list(run.cell_receipts),
    }


def _disclosure() -> dict[str, Any]:
    return {
        "selection_bn_protocol": "SS",
        "selection_bn_protocol_id": SS_BN_PROTOCOL,
        "BS_excluded_from_selection": True,
        "application_protocols": list(APPLICATION_PROTOCOLS),
        "application_protocol_ids": list(APPLICATION_PROTOCOL_IDS),
        "best_pd_reuses_same_frozen_hyperparameters_without_tuning": True,
    }


def _contract_receipt() -> dict[str, Any]:
    return {
        "selector_protocol_id": SELECTOR_PROTOCOL_ID,
        "paper_result": False,
        "calibration_source": "train-derived frozen 64 images per dataset",
        "independent_validation": False,
        **_disclosure(),
        "target_transition_metrics_used_for_selection": False,
        "cell_definition": "dataset x SS x corruption condition",
        "cell_count_per_run": CELLS_PER_RUN,
        "images_per_cell": IMAGES_PER_CELL,
        "equal_weighting": {
            "within_run": "arithmetic mean of 39 SS cell deltas; every cell weight=1/39",
            "across_runs": "arithmetic mean of three run-level macro values; every run weight=1/3",
        },
        "episode_contract": {
            "stage1": 24960,
            "stage2": 14976,
            "total": 39936,
        },
        "stage2_process_contract": {
            "fresh_process_count": 2,
            "same_two_process_ids_each_run_all_top3_candidates": True,
            "process_ids_disjoint_from_stage1": True,
            "candidate_order": "frozen stage1 top3 order in each process",
        },
        "metric_definitions": {
            "global_iou": "intersection_pixels / union_pixels",
            "cell_global_iou_delta": "tent_post_global_iou - SS-matched_tent_pre_global_iou",
            "fa_per_million_pixels": "false_alarm_pixels / total_image_pixels * 1,000,000",
            "cell_fa_increase": "tent_post_fa_per_million_pixels - tent_pre_fa_per_million_pixels",
            "pd": "detected_targets / total_targets",
            "cell_pd_delta": "tent_post_pd - tent_pre_pd",
            "clean_macro": "equal mean over the 3 SS clean cells",
        },
        "tie_break_order": list(_v1._contract_receipt()["tie_break_order"]),
        "required_hard_gates": list(REQUIRED_HARD_GATES),
        "implementation_protocol_audit": list(REQUIRED_PROTOCOL_AUDIT),
        "forbidden_selector_fields": sorted(FORBIDDEN_SELECTOR_FIELDS),
    }


def _stage1_summaries(
    stage1_records: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[Candidate, _v1._RunSummary],
    tuple[CandidateRankingMetrics, ...],
]:
    parsed = _parse_ss_records(stage1_records, expected_stage=1)
    grouped = _v1._group_runs(parsed)
    if set(grouped) != set(ALL_CANDIDATES):
        missing = [
            candidate.to_dict()
            for candidate in ALL_CANDIDATES
            if candidate not in grouped
        ]
        extra = [
            candidate.to_dict()
            for candidate in grouped
            if candidate not in ALL_CANDIDATES
        ]
        raise CalibrationSelectionError(
            f"stage 1 must contain all ten candidates; missing={missing}, extra={extra}"
        )
    summaries: dict[Candidate, _v1._RunSummary] = {}
    process_ids: set[str] = set()
    for candidate in ALL_CANDIDATES:
        processes = grouped[candidate]
        if len(processes) != 1:
            raise CalibrationSelectionError(
                "stage 1 requires exactly one fresh process per candidate; "
                f"candidate={candidate.to_dict()}, processes={sorted(processes)}"
            )
        process_id, records = next(iter(processes.items()))
        if process_id in process_ids:
            raise CalibrationSelectionError(
                f"stage 1 process ID {process_id!r} is reused across candidates"
            )
        process_ids.add(process_id)
        summaries[candidate] = _summarise_ss_run(candidate, process_id, records)
    expected_episodes = len(ALL_CANDIDATES) * CELLS_PER_RUN * IMAGES_PER_CELL
    observed_episodes = len(parsed) * IMAGES_PER_CELL
    if observed_episodes != expected_episodes:
        raise CalibrationSelectionError(
            f"stage 1 episode count {observed_episodes} != {expected_episodes}"
        )
    ranking = rank_candidate_metrics(
        _v1._candidate_metrics(candidate, (summaries[candidate],))
        for candidate in ALL_CANDIDATES
    )
    return summaries, ranking


def _assert_stage2_candidate_order(
    parsed: Sequence[_v1._CellRecord], top3: Sequence[Candidate]
) -> None:
    by_process: dict[str, list[_v1._CellRecord]] = {}
    for record in parsed:
        by_process.setdefault(record.process_id, []).append(record)
    if len(by_process) != 2:
        raise CalibrationSelectionError(
            f"stage 2 requires exactly two fresh processes, got {sorted(by_process)}"
        )
    for process_id, records in by_process.items():
        observed_order: list[Candidate] = []
        for record in records:
            if not observed_order or observed_order[-1] != record.candidate:
                observed_order.append(record.candidate)
        if tuple(observed_order) != tuple(top3):
            raise CalibrationSelectionError(
                "each stage-2 process must run the frozen top3 in identical order; "
                f"process_id={process_id!r}, "
                f"expected={[value.to_dict() for value in top3]}, "
                f"observed={[value.to_dict() for value in observed_order]}"
            )


def select_stage1_top3(
    stage1_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate 24,960 SS-only stage-1 episodes and return the top three."""

    summaries, ranking = _stage1_summaries(stage1_records)
    top3 = tuple(metric.candidate for metric in ranking[:3])
    ranking_receipts: list[dict[str, Any]] = []
    for rank, metric in enumerate(ranking, start=1):
        value = _v1._metric_receipt(metric)
        value.update(
            {
                "rank": rank,
                "run_count": 1,
                "runs": [_run_receipt(summaries[metric.candidate])],
            }
        )
        ranking_receipts.append(value)
    return {
        "schema_version": 2,
        "receipt_type": "stage1_ss_top3",
        "contract": _contract_receipt(),
        **_disclosure(),
        "validation": {
            "candidate_count": 10,
            "candidate_run_count": 10,
            "cells_per_run": 39,
            "aggregate_cell_record_count": 390,
            "episode_count": 24960,
            "all_records_are_ss": True,
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
    """Select one frozen optimizer/LR from one plus two fresh SS runs."""

    stage1_summaries, stage1_ranking = _stage1_summaries(stage1_records)
    top3 = tuple(metric.candidate for metric in stage1_ranking[:3])
    parsed_stage2 = _parse_ss_records(stage2_records, expected_stage=2)
    _assert_stage2_candidate_order(parsed_stage2, top3)
    grouped_stage2 = _v1._group_runs(parsed_stage2)
    if set(grouped_stage2) != set(top3):
        missing = [value.to_dict() for value in top3 if value not in grouped_stage2]
        extra = [value.to_dict() for value in grouped_stage2 if value not in top3]
        raise CalibrationSelectionError(
            "stage 2 must contain exactly the stage-1 top three candidates; "
            f"missing={missing}, extra={extra}"
        )
    stage1_processes = {
        summary.process_id for summary in stage1_summaries.values()
    }
    shared_processes: frozenset[str] | None = None
    stage2_summaries: dict[Candidate, tuple[_v1._RunSummary, ...]] = {}
    for candidate in top3:
        processes = grouped_stage2[candidate]
        if len(processes) != 2:
            raise CalibrationSelectionError(
                "stage 2 requires exactly two fresh processes per top3 candidate; "
                f"candidate={candidate.to_dict()}, processes={sorted(processes)}"
            )
        process_ids = frozenset(processes)
        if shared_processes is None:
            shared_processes = process_ids
        elif process_ids != shared_processes:
            raise CalibrationSelectionError(
                "the same two stage-2 process IDs must each run all top3 candidates"
            )
        summaries: list[_v1._RunSummary] = []
        for process_id, records in sorted(processes.items()):
            if process_id in stage1_processes:
                raise CalibrationSelectionError(
                    f"stage-2 process {process_id!r} reuses a stage-1 process ID"
                )
            summaries.append(_summarise_ss_run(candidate, process_id, records))
        stage2_summaries[candidate] = tuple(summaries)
    expected_stage2_episodes = 14976
    observed_stage2_episodes = len(parsed_stage2) * IMAGES_PER_CELL
    if observed_stage2_episodes != expected_stage2_episodes:
        raise CalibrationSelectionError(
            f"stage 2 episode count {observed_stage2_episodes} "
            f"!= {expected_stage2_episodes}"
        )
    all_runs = {
        candidate: (
            stage1_summaries[candidate],
            *stage2_summaries[candidate],
        )
        for candidate in top3
    }
    final_ranking = rank_candidate_metrics(
        _v1._candidate_metrics(candidate, all_runs[candidate])
        for candidate in top3
    )
    stage1_ranking_receipts: list[dict[str, Any]] = []
    for rank, metric in enumerate(stage1_ranking, start=1):
        value = _v1._metric_receipt(metric)
        value.update(
            {
                "rank": rank,
                "run_count": 1,
                "runs": [_run_receipt(stage1_summaries[metric.candidate])],
            }
        )
        stage1_ranking_receipts.append(value)
    final_ranking_receipts: list[dict[str, Any]] = []
    for rank, metric in enumerate(final_ranking, start=1):
        value = _v1._metric_receipt(metric)
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
        "schema_version": 2,
        "receipt_type": "final_ss_shared_optimizer_lr_selection",
        "contract": _contract_receipt(),
        **_disclosure(),
        "validation": {
            "stage1_candidate_count": 10,
            "stage1_cells_per_run": 39,
            "stage1_episode_count": 24960,
            "stage2_candidate_count": 3,
            "stage2_fresh_process_count": 2,
            "stage2_processes_shared_across_top3": True,
            "stage2_process_ids": sorted(shared_processes or ()),
            "stage2_episode_count": 14976,
            "total_calibration_episode_count": 39936,
            "all_records_are_ss": True,
            "test_image_opens": 0,
            "test_label_opens": 0,
            "method_label_accesses": 0,
        },
        "stage1_ranking": stage1_ranking_receipts,
        "stage1_top3": [candidate.to_dict() for candidate in top3],
        "final_ranking": final_ranking_receipts,
        "selected_candidate": selected.to_dict(),
    }


__all__ = [
    "ALL_CANDIDATES",
    "APPLICATION_PROTOCOL_IDS",
    "APPLICATION_PROTOCOLS",
    "BN_PROTOCOLS",
    "BS_BN_PROTOCOL",
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
    "SS_BN_PROTOCOL",
    "rank_candidate_metrics",
    "select_final_candidate",
    "select_stage1_top3",
]
