from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest

from tta.binary_tent_ss_calibration_selector_v2 import (
    ALL_CANDIDATES,
    APPLICATION_PROTOCOL_IDS,
    APPLICATION_PROTOCOLS,
    BS_BN_PROTOCOL,
    CONDITIONS,
    DATASETS,
    REQUIRED_HARD_GATES,
    REQUIRED_PROTOCOL_AUDIT,
    SS_BN_PROTOCOL,
    CalibrationSelectionError,
    CandidateRankingMetrics,
    rank_candidate_metrics,
    select_final_candidate,
    select_stage1_top3,
)


def _counts(intersection: int) -> dict[str, int]:
    return {
        "intersection_pixels": intersection,
        "union_pixels": 100,
        "false_alarm_pixels": 2,
        "total_image_pixels": 64 * 256 * 256,
        "detected_targets": 7,
        "total_targets": 10,
    }


def _run_records(
    candidate,
    *,
    stage: int,
    process_id: str,
    score: int,
    bn_protocol: str = SS_BN_PROTOCOL,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for dataset in DATASETS:
        for corruption, severity in CONDITIONS:
            records.append(
                {
                    "stage": stage,
                    "process_id": process_id,
                    "fresh_process": True,
                    "candidate": candidate.to_dict(),
                    "dataset": dataset,
                    "bn_protocol": bn_protocol,
                    "corruption": corruption,
                    "severity": severity,
                    "image_count": 64,
                    "optimizer_steps_total": 64,
                    "test_image_opens": 0,
                    "test_label_opens": 0,
                    "method_label_accesses": 0,
                    "hard_gates": {
                        key: True for key in REQUIRED_HARD_GATES
                    },
                    "protocol_audit": {
                        key: True for key in REQUIRED_PROTOCOL_AUDIT
                    },
                    "endpoints": {
                        "tent_pre": _counts(10),
                        "tent_post": _counts(10 + score),
                    },
                }
            )
    return records


def _stage1_records() -> list[dict[str, object]]:
    return [
        record
        for index, candidate in enumerate(ALL_CANDIDATES, start=1)
        for record in _run_records(
            candidate,
            stage=1,
            process_id=f"stage1-{index}",
            score=index,
        )
    ]


def _top3(stage1_records):
    receipt = select_stage1_top3(stage1_records)
    candidates_by_key = {
        (value.optimizer, float(value.learning_rate)): value
        for value in ALL_CANDIDATES
    }
    return tuple(
        candidates_by_key[(value["optimizer"], value["learning_rate"])]
        for value in receipt["top3"]
    )


def _stage2_records(stage1_records) -> list[dict[str, object]]:
    top3 = _top3(stage1_records)
    score_by_candidate = {
        candidate: index for index, candidate in enumerate(ALL_CANDIDATES, start=1)
    }
    return [
        record
        for process_id in ("stage2-a", "stage2-b")
        for candidate in top3
        for record in _run_records(
            candidate,
            stage=2,
            process_id=process_id,
            score=score_by_candidate[candidate],
        )
    ]


def test_ss_selector_v2_counts_exact_rationals_and_disclosures() -> None:
    stage1 = _stage1_records()
    stage1_receipt = select_stage1_top3(stage1)
    assert stage1_receipt["validation"] == {
        "candidate_count": 10,
        "candidate_run_count": 10,
        "cells_per_run": 39,
        "aggregate_cell_record_count": 390,
        "episode_count": 24960,
        "all_records_are_ss": True,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
    }
    assert stage1_receipt["ranking"][0][
        "primary_mean_over_runs_macro_global_iou_delta"
    ]["exact"] == "1/10"
    assert stage1_receipt["selection_bn_protocol"] == "SS"
    assert stage1_receipt["selection_bn_protocol_id"] == SS_BN_PROTOCOL
    assert stage1_receipt["BS_excluded_from_selection"] is True
    assert stage1_receipt["application_protocols"] == list(APPLICATION_PROTOCOLS)
    assert stage1_receipt["application_protocol_ids"] == list(
        APPLICATION_PROTOCOL_IDS
    )
    assert (
        stage1_receipt[
            "best_pd_reuses_same_frozen_hyperparameters_without_tuning"
        ]
        is True
    )
    assert stage1_receipt["top3"] == [
        value.to_dict() for value in reversed(ALL_CANDIDATES[-3:])
    ]

    final = select_final_candidate(stage1, _stage2_records(stage1))
    assert final["validation"]["stage1_episode_count"] == 24960
    assert final["validation"]["stage2_episode_count"] == 14976
    assert final["validation"]["total_calibration_episode_count"] == 39936
    assert final["validation"]["stage2_fresh_process_count"] == 2
    assert final["selection_bn_protocol"] == "SS"
    assert final["selection_bn_protocol_id"] == SS_BN_PROTOCOL
    assert final["BS_excluded_from_selection"] is True
    assert final["application_protocols"] == ["SS", "BS"]
    assert final["application_protocol_ids"] == [SS_BN_PROTOCOL, BS_BN_PROTOCOL]
    assert (
        final["best_pd_reuses_same_frozen_hyperparameters_without_tuning"]
        is True
    )
    assert final["selected_candidate"] == ALL_CANDIDATES[-1].to_dict()


def test_stage1_missing_duplicate_and_process_reuse_fail_closed() -> None:
    stage1 = _stage1_records()
    with pytest.raises(CalibrationSelectionError, match="incomplete SS"):
        select_stage1_top3(stage1[:-1])
    with pytest.raises(CalibrationSelectionError, match="duplicate calibration cell"):
        select_stage1_top3([*stage1, deepcopy(stage1[0])])

    reused = deepcopy(stage1)
    first_candidate_records = 39
    reused[first_candidate_records]["process_id"] = "stage1-1"
    for index in range(first_candidate_records, 2 * first_candidate_records):
        reused[index]["process_id"] = "stage1-1"
    with pytest.raises(CalibrationSelectionError, match="reused across candidates"):
        select_stage1_top3(reused)


@pytest.mark.parametrize("protocol", [BS_BN_PROTOCOL, "unexpected_protocol"])
def test_bs_or_extra_protocol_injection_is_rejected(protocol: str) -> None:
    stage1 = _stage1_records()
    stage1[0]["bn_protocol"] = protocol
    with pytest.raises(CalibrationSelectionError, match="accepts only"):
        select_stage1_top3(stage1)


def test_stage2_missing_duplicate_process_reuse_and_order_fail_closed() -> None:
    stage1 = _stage1_records()
    stage2 = _stage2_records(stage1)
    with pytest.raises(CalibrationSelectionError, match="incomplete SS"):
        select_final_candidate(stage1, stage2[:-1])
    with pytest.raises(CalibrationSelectionError, match="duplicate calibration cell"):
        select_final_candidate(stage1, [*stage2, deepcopy(stage2[-1])])

    reused = deepcopy(stage2)
    for record in reused[: 3 * 39]:
        record["process_id"] = "stage1-1"
    with pytest.raises(CalibrationSelectionError, match="reuses a stage-1"):
        select_final_candidate(stage1, reused)

    top3 = _top3(stage1)
    score_by_candidate = {
        candidate: index for index, candidate in enumerate(ALL_CANDIDATES, start=1)
    }
    wrong_order = [
        record
        for process_id, order in (
            ("stage2-a", top3),
            ("stage2-b", tuple(reversed(top3))),
        )
        for candidate in order
        for record in _run_records(
            candidate,
            stage=2,
            process_id=process_id,
            score=score_by_candidate[candidate],
        )
    ]
    with pytest.raises(CalibrationSelectionError, match="identical order"):
        select_final_candidate(stage1, wrong_order)


def test_transition_fields_are_forbidden_recursively() -> None:
    stage1 = _stage1_records()
    stage1[0]["diagnostics"] = {"target_transition_counts": {"lost": 1}}
    with pytest.raises(CalibrationSelectionError, match="forbidden target-transition"):
        select_stage1_top3(stage1)


def test_v1_exact_tie_break_is_preserved() -> None:
    tied = tuple(
        CandidateRankingMetrics(
            candidate=candidate,
            primary_macro_delta_iou=Fraction(0, 1),
            worst_run_macro_delta_iou=Fraction(0, 1),
            clean_macro_delta_iou=Fraction(0, 1),
            macro_fa_increase_per_million_pixels=Fraction(0, 1),
            macro_pd_delta=Fraction(0, 1),
        )
        for candidate in ALL_CANDIDATES
    )
    ranking = rank_candidate_metrics(tied)
    assert ranking[0].candidate.optimizer == "Adam"
    assert str(ranking[0].candidate.learning_rate) == "0.00001"
    assert ranking[1].candidate.optimizer == "SGD"
    assert str(ranking[1].candidate.learning_rate) == "0.00001"
