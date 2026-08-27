from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from tta.binary_tent_calibration_selector import (
    ALL_CANDIDATES,
    BN_PROTOCOLS,
    CELLS_PER_RUN,
    CONDITIONS,
    DATASETS,
    IMAGES_PER_CELL,
    REQUIRED_HARD_GATES,
    REQUIRED_PROTOCOL_AUDIT,
    CalibrationSelectionError,
    Candidate,
    CandidateRankingMetrics,
    rank_candidate_metrics,
    select_final_candidate,
    select_stage1_top3,
)


ROOT = Path(__file__).resolve().parents[1]


def _endpoint(
    *,
    intersection: int,
    union: int = 100,
    false_alarm_pixels: int = 10,
    detected_targets: int = 50,
) -> dict[str, int]:
    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "false_alarm_pixels": false_alarm_pixels,
        "total_image_pixels": 1_000_000,
        "detected_targets": detected_targets,
        "total_targets": 100,
    }


def _record(
    *,
    stage: int,
    process_id: str,
    candidate: Candidate,
    dataset: str,
    protocol: str,
    corruption: str,
    severity: int,
    post_iou_delta_points: int = 0,
    post_fa_delta: int = 0,
    post_pd_delta_points: int = 0,
    union: int = 100,
) -> dict[str, object]:
    pre_intersection = union // 2
    post_intersection = pre_intersection + post_iou_delta_points
    return {
        "stage": stage,
        "process_id": process_id,
        "fresh_process": True,
        "candidate": {
            "optimizer": candidate.optimizer,
            "learning_rate": float(candidate.learning_rate),
        },
        "dataset": dataset,
        "bn_protocol": protocol,
        "corruption": corruption,
        "severity": severity,
        "image_count": IMAGES_PER_CELL,
        "optimizer_steps_total": IMAGES_PER_CELL,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "hard_gates": {key: True for key in REQUIRED_HARD_GATES},
        "protocol_audit": {key: True for key in REQUIRED_PROTOCOL_AUDIT},
        "endpoints": {
            "tent_pre": _endpoint(
                intersection=pre_intersection,
                union=union,
                false_alarm_pixels=10,
                detected_targets=50,
            ),
            "tent_post": _endpoint(
                intersection=post_intersection,
                union=union,
                false_alarm_pixels=10 + post_fa_delta,
                detected_targets=50 + post_pd_delta_points,
            ),
        },
    }


def _full_candidate_run(
    *,
    stage: int,
    process_id: str,
    candidate: Candidate,
    iou_delta_points: int = 0,
    fa_delta: int = 0,
    pd_delta_points: int = 0,
) -> list[dict[str, object]]:
    return [
        _record(
            stage=stage,
            process_id=process_id,
            candidate=candidate,
            dataset=dataset,
            protocol=protocol,
            corruption=corruption,
            severity=severity,
            post_iou_delta_points=iou_delta_points,
            post_fa_delta=fa_delta,
            post_pd_delta_points=pd_delta_points,
        )
        for dataset in DATASETS
        for protocol in BN_PROTOCOLS
        for corruption, severity in CONDITIONS
    ]


def _stage1_records(
    scores: dict[Candidate, int] | None = None,
) -> list[dict[str, object]]:
    scores = scores or {}
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(ALL_CANDIDATES):
        records.extend(
            _full_candidate_run(
                stage=1,
                process_id=f"stage1_candidate_{index:02d}",
                candidate=candidate,
                iou_delta_points=scores.get(candidate, 0),
            )
        )
    return records


def _stage2_records(
    top3: tuple[Candidate, Candidate, Candidate],
    scores_by_candidate_and_repeat: dict[Candidate, tuple[int, int]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate in top3:
        for repeat_index, score in enumerate(scores_by_candidate_and_repeat[candidate]):
            records.extend(
                _full_candidate_run(
                    stage=2,
                    process_id=f"stage2_repeat_{repeat_index + 1:02d}",
                    candidate=candidate,
                    iou_delta_points=score,
                )
            )
    return records


def _candidate_from_receipt(value: dict[str, object]) -> Candidate:
    return Candidate.from_values(value["optimizer"], value["learning_rate"])


def test_versioned_config_freezes_full_grid_and_disclosures() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/binary_tent_source_calibration_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["scope"]["paper_result"] is False
    assert config["scope"]["source_train_derived"] is True
    assert config["scope"]["independent_validation"] is False
    assert config["scope"]["use_test_images"] is False
    assert config["scope"]["use_test_labels"] is False
    assert config["inherited_source_limitation"]["checkpoint_selection"] == (
        "test_selected_during_source_training"
    )
    assert len(config["source_train_subsets"]["datasets"]) == 3
    assert config["corruption_conditions"]["count_per_dataset"] == 13
    assert len(config["corruption_conditions"]["ordered"]) == 13
    assert config["method"]["candidates"]["cross_product_count"] == 10
    assert config["method"]["candidates"]["SGD"]["momentum"] == 0.9
    assert config["method"]["candidates"]["SGD"]["nesterov"] is True
    assert config["selection"]["stage_1"]["episodes"] == 49_920
    assert config["selection"]["stage_2"]["episodes"] == 29_952
    assert config["selection"]["stage_2"]["shared_fresh_process_count"] == 2
    assert (
        config["selection"]["stage_2"][
            "same_two_process_ids_each_run_all_top3_candidates"
        ]
        is True
    )
    assert (
        config["selection"]["stage_2"][
            "rebuild_model_method_optimizer_before_each_candidate"
        ]
        is True
    )
    assert config["selection"]["total_episodes"] == 79_872
    assert config["evaluation"]["target_transition_metrics_used_for_selection"] is False
    assert config["required_hard_gates"]["entropy_decreased_gate"] is False


def test_stage1_uses_equal_cell_weighting_and_selects_exact_top3() -> None:
    records = _stage1_records()
    special = ALL_CANDIDATES[-1]
    special_process = "stage1_candidate_09"
    special_records = [
        record
        for record in records
        if record["process_id"] == special_process
    ]
    assert len(special_records) == CELLS_PER_RUN

    # Only one of 78 cells improves from 0.5 to 1.0.  Its much smaller union
    # cannot reduce its frozen 1/78 macro weight.
    special_records[0]["endpoints"] = {
        "tent_pre": _endpoint(intersection=5, union=10),
        "tent_post": _endpoint(intersection=10, union=10),
    }
    for record in special_records[1:]:
        record["endpoints"] = {
            "tent_pre": _endpoint(intersection=500, union=1000),
            "tent_post": _endpoint(intersection=500, union=1000),
        }

    receipt = select_stage1_top3(records)
    winner = receipt["ranking"][0]
    assert _candidate_from_receipt(winner["candidate"]) == special
    macro = winner["primary_mean_over_runs_macro_global_iou_delta"]
    assert (macro["numerator"], macro["denominator"]) == (1, 156)
    assert receipt["validation"]["episode_count"] == 49_920
    assert len(receipt["top3"]) == 3
    assert all(entry["run_count"] == 1 for entry in receipt["ranking"])
    assert len(winner["runs"][0]["cells_in_frozen_order"]) == CELLS_PER_RUN
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            dict(primary=Fraction(1, 10)),
            dict(primary=Fraction(0)),
            "left",
        ),
        (
            dict(primary=Fraction(1, 10), worst=Fraction(0)),
            dict(primary=Fraction(1, 10), worst=Fraction(-1, 10)),
            "left",
        ),
        (
            dict(primary=Fraction(1, 10), worst=Fraction(0), clean=Fraction(1, 10)),
            dict(primary=Fraction(1, 10), worst=Fraction(0), clean=Fraction(0)),
            "left",
        ),
        (
            dict(primary=Fraction(1, 10), worst=Fraction(0), clean=Fraction(0), fa=Fraction(1)),
            dict(primary=Fraction(1, 10), worst=Fraction(0), clean=Fraction(0), fa=Fraction(2)),
            "left",
        ),
        (
            dict(
                primary=Fraction(1, 10),
                worst=Fraction(0),
                clean=Fraction(0),
                fa=Fraction(1),
                pd=Fraction(1, 10),
            ),
            dict(
                primary=Fraction(1, 10),
                worst=Fraction(0),
                clean=Fraction(0),
                fa=Fraction(1),
                pd=Fraction(0),
            ),
            "left",
        ),
    ],
)
def test_first_five_numeric_tie_break_fields_are_deterministic(
    left: dict[str, Fraction], right: dict[str, Fraction], expected: str
) -> None:
    candidates = (ALL_CANDIDATES[0], ALL_CANDIDATES[1])

    def metric(candidate: Candidate, values: dict[str, Fraction]) -> CandidateRankingMetrics:
        return CandidateRankingMetrics(
            candidate=candidate,
            primary_macro_delta_iou=values.get("primary", Fraction(0)),
            worst_run_macro_delta_iou=values.get("worst", Fraction(0)),
            clean_macro_delta_iou=values.get("clean", Fraction(0)),
            macro_fa_increase_per_million_pixels=values.get("fa", Fraction(0)),
            macro_pd_delta=values.get("pd", Fraction(0)),
        )

    ranked = rank_candidate_metrics(
        [metric(candidates[0], left), metric(candidates[1], right)]
    )
    assert ranked[0].candidate == candidates[0 if expected == "left" else 1]


def test_last_two_tie_breaks_are_lower_lr_then_adam_before_sgd() -> None:
    def tied(candidate: Candidate) -> CandidateRankingMetrics:
        return CandidateRankingMetrics(
            candidate=candidate,
            primary_macro_delta_iou=Fraction(0),
            worst_run_macro_delta_iou=Fraction(0),
            clean_macro_delta_iou=Fraction(0),
            macro_fa_increase_per_million_pixels=Fraction(0),
            macro_pd_delta=Fraction(0),
        )

    lower_lr = Candidate("SGD", Decimal("1e-5"))
    higher_lr = Candidate("Adam", Decimal("3e-5"))
    assert rank_candidate_metrics([tied(higher_lr), tied(lower_lr)])[0].candidate == lower_lr

    adam = Candidate("Adam", Decimal("1e-4"))
    sgd = Candidate("SGD", Decimal("1e-4"))
    assert rank_candidate_metrics([tied(sgd), tied(adam)])[0].candidate == adam


def test_final_selection_uses_stage1_plus_two_fresh_repeats() -> None:
    # Make the first three candidates the phase-1 top three in a known order.
    scores = {candidate: -10 for candidate in ALL_CANDIDATES}
    scores.update(
        {
            ALL_CANDIDATES[0]: 9,
            ALL_CANDIDATES[1]: 8,
            ALL_CANDIDATES[2]: 7,
        }
    )
    stage1 = _stage1_records(scores)
    top3 = (ALL_CANDIDATES[0], ALL_CANDIDATES[1], ALL_CANDIDATES[2])
    stage2 = _stage2_records(
        top3,
        {
            top3[0]: (0, 0),
            top3[1]: (8, 8),
            top3[2]: (7, 7),
        },
    )

    receipt = select_final_candidate(stage1, stage2)
    assert [_candidate_from_receipt(value) for value in receipt["stage1_top3"]] == list(top3)
    assert _candidate_from_receipt(receipt["selected_candidate"]) == top3[1]
    assert receipt["validation"]["stage2_fresh_process_count"] == 2
    assert receipt["validation"]["stage2_processes_shared_across_top3"] is True
    assert receipt["validation"]["stage2_process_ids"] == [
        "stage2_repeat_01",
        "stage2_repeat_02",
    ]
    assert receipt["validation"]["stage2_episode_count"] == 29_952
    assert receipt["validation"]["total_calibration_episode_count"] == 79_872
    assert all(value["run_count"] == 3 for value in receipt["final_ranking"])
    assert receipt["selected_bn_protocol"] is None
    assert receipt["both_bn_protocols_retained_for_formal_evaluation"] is True


def test_final_selection_rejects_missing_or_reused_repeat_process() -> None:
    scores = {candidate: -10 for candidate in ALL_CANDIDATES}
    scores.update({ALL_CANDIDATES[0]: 3, ALL_CANDIDATES[1]: 2, ALL_CANDIDATES[2]: 1})
    stage1 = _stage1_records(scores)
    top3 = (ALL_CANDIDATES[0], ALL_CANDIDATES[1], ALL_CANDIDATES[2])
    stage2 = _stage2_records(top3, {candidate: (1, 1) for candidate in top3})

    missing_repeat = [
        record
        for record in stage2
        if not (
            record["candidate"]["optimizer"] == top3[0].optimizer
            and Decimal(str(record["candidate"]["learning_rate"]))
            == top3[0].learning_rate
            and record["process_id"] == "stage2_repeat_02"
        )
    ]
    with pytest.raises(CalibrationSelectionError, match="exactly two fresh processes"):
        select_final_candidate(stage1, missing_repeat)

    reused = deepcopy(stage2)
    for record in reused:
        if record["process_id"] == "stage2_repeat_01":
            record["process_id"] = "stage1_candidate_00"
    with pytest.raises(CalibrationSelectionError, match="reuses a stage-1 process"):
        select_final_candidate(stage1, reused)

    nonshared = deepcopy(stage2)
    for record in nonshared:
        if (
            record["candidate"]["optimizer"] == top3[0].optimizer
            and Decimal(str(record["candidate"]["learning_rate"]))
            == top3[0].learning_rate
            and record["process_id"] == "stage2_repeat_01"
        ):
            record["process_id"] = "stage2_candidate0_only"
    with pytest.raises(CalibrationSelectionError, match="same two fresh stage-2"):
        select_final_candidate(stage1, nonshared)


def test_selector_forbids_target_transition_metrics_at_any_depth() -> None:
    records = _stage1_records()
    records[0]["diagnostics"] = {"ATER": 0.25}
    with pytest.raises(CalibrationSelectionError, match="forbidden target-transition"):
        select_stage1_top3(records)


def test_selector_rejects_missing_and_duplicate_cells() -> None:
    records = _stage1_records()
    with pytest.raises(CalibrationSelectionError, match="incomplete calibration run"):
        select_stage1_top3(records[:-1])

    records = _stage1_records()
    records.append(deepcopy(records[0]))
    with pytest.raises(CalibrationSelectionError, match="duplicate calibration cell"):
        select_stage1_top3(records)


def test_selector_rejects_failed_missing_and_arbitrary_hard_gates() -> None:
    failed = _stage1_records()
    failed[0]["hard_gates"]["exact_reset"] = False
    with pytest.raises(CalibrationSelectionError, match="failed required hard gates"):
        select_stage1_top3(failed)

    missing = _stage1_records()
    del missing[0]["hard_gates"]["tent_pre_identity_contract"]
    with pytest.raises(CalibrationSelectionError, match="exactly the frozen set"):
        select_stage1_top3(missing)

    arbitrary = _stage1_records()
    arbitrary[0]["hard_gates"]["entropy_decreased"] = True
    with pytest.raises(CalibrationSelectionError, match="exactly the frozen set"):
        select_stage1_top3(arbitrary)

    protocol_audit = _stage1_records()
    protocol_audit[0]["protocol_audit"][
        "candidate_model_method_optimizer_rebuilt_before_run"
    ] = False
    with pytest.raises(CalibrationSelectionError, match="implementation protocol audit"):
        select_stage1_top3(protocol_audit)


def test_selector_rejects_invalid_direct_protocol_counters() -> None:
    records = _stage1_records()
    records[0]["optimizer_steps_total"] = IMAGES_PER_CELL - 1
    with pytest.raises(CalibrationSelectionError, match="exactly one optimizer step"):
        select_stage1_top3(records)

    records = _stage1_records()
    records[0]["test_image_opens"] = 1
    with pytest.raises(CalibrationSelectionError, match="access firewall"):
        select_stage1_top3(records)
