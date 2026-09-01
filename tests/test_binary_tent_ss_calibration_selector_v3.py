from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from tta.binary_tent_ss_calibration_selector_v3 import (
    ALL_CANDIDATES,
    CONDITIONS,
    DATASETS,
    FORMAL_FROZEN_MODE,
    REQUIRED_HARD_GATES,
    REQUIRED_PROTOCOL_AUDIT,
    SS_BN_PROTOCOL,
    STAGE1_RECEIPT_SCHEMA_VERSION,
    STAGE1_RECEIPT_TYPE,
    CalibrationSelectionError,
    CandidateDiagnosticEvidence,
    ScientificGateSpec,
    select_stage1_candidates,
    validate_stage1_scientific_receipt,
)


LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _endpoint(
    *,
    intersection: int,
    false_alarm_pixels: int = 10,
    detected_targets: int = 80,
) -> dict[str, int]:
    return {
        "intersection_pixels": intersection,
        "union_pixels": 100,
        "false_alarm_pixels": false_alarm_pixels,
        "total_image_pixels": 1_000_000,
        "detected_targets": detected_targets,
        "total_targets": 100,
    }


def _stage1_records(
    score_for_cell,
    *,
    fa_delta_for_candidate=None,
    pd_delta_for_candidate=None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(ALL_CANDIDATES):
        fa_delta = (
            0
            if fa_delta_for_candidate is None
            else fa_delta_for_candidate(candidate_index, candidate)
        )
        pd_delta = (
            0
            if pd_delta_for_candidate is None
            else pd_delta_for_candidate(candidate_index, candidate)
        )
        for dataset in DATASETS:
            for corruption, severity in CONDITIONS:
                score = score_for_cell(
                    candidate_index,
                    candidate,
                    dataset,
                    corruption,
                    severity,
                )
                records.append(
                    {
                        "stage": 1,
                        "process_id": f"stage1-{candidate_index}",
                        "fresh_process": True,
                        "candidate": candidate.to_dict(),
                        "dataset": dataset,
                        "bn_protocol": SS_BN_PROTOCOL,
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
                            "tent_pre": _endpoint(intersection=50),
                            "tent_post": _endpoint(
                                intersection=50 + score,
                                false_alarm_pixels=10 + fa_delta,
                                detected_targets=80 + pd_delta,
                            ),
                        },
                    }
                )
    return records


def _formal_spec(**overrides) -> ScientificGateSpec:
    values = {
        "profile_id": "frozen-synthetic-unit-test-gate",
        "mode": FORMAL_FROZEN_MODE,
        "min_positive_cells": 4,
        "min_positive_corruption_families": 2,
        "min_positive_datasets": 2,
        "clean_iou_equivalence_margin": Fraction(1, 100),
        "max_pd_drop": Fraction(1, 100),
        "max_fa_increase": Fraction(1, 1),
        "min_parameter_update_fraction": Fraction(1, 2),
        "min_functional_change_fraction": Fraction(1, 2),
        "min_objective_decrease_fraction": Fraction(1, 2),
        "top_k_after_filter": 3,
        "allow_fewer_than_top_k": True,
        "threshold_source": "synthetic_test_fixture_not_experimental_margin",
        "frozen_gate_manifest_sha256": "a" * 64,
    }
    values.update(overrides)
    return ScientificGateSpec(**values)


def _passing_diagnostics() -> tuple[CandidateDiagnosticEvidence, ...]:
    return tuple(
        CandidateDiagnosticEvidence(
            candidate=candidate,
            parameter_update_episodes=3,
            parameter_update_evaluated_episodes=4,
            functional_change_episodes=3,
            functional_change_evaluated_episodes=4,
            objective_decrease_episodes=3,
            objective_evaluated_episodes=4,
        )
        for candidate in ALL_CANDIDATES
    )


def _decision(receipt, candidate):
    return next(
        entry
        for entry in (
            receipt["eligible_candidates"] + receipt["rejected_candidates"]
        )
        if entry["candidate"]["optimizer"] == candidate.optimizer
        and entry["candidate"]["learning_rate_decimal"]
        == candidate.to_dict()["learning_rate_decimal"]
    )


def test_all_zero_candidates_are_a_legal_negative_scientific_result() -> None:
    records = _stage1_records(lambda *_: 0)
    receipt = select_stage1_candidates(
        records,
        _passing_diagnostics(),
        _formal_spec(),
    )

    assert receipt["protocol_status"] == "passed"
    assert receipt["scientific_status"] == "failed"
    assert receipt["stage2_allowed"] is False
    assert receipt["eligible_candidates"] == []
    assert receipt["selected_for_stage2"] == []
    assert len(receipt["rejected_candidates"]) == 10
    assert all(
        "macro_delta_iou" in entry["failed_gates"]
        for entry in receipt["rejected_candidates"]
    )


def test_all_negative_candidates_do_not_produce_a_top_k() -> None:
    receipt = select_stage1_candidates(
        _stage1_records(lambda *_: -1),
        _passing_diagnostics(),
        _formal_spec(),
    )
    assert receipt["scientific_status"] == "failed"
    assert receipt["stage2_allowed"] is False
    assert receipt["eligible_ranking"] == []
    assert receipt["selected_for_stage2"] == []


def test_positive_macro_with_large_clean_drop_fails_clean_safety() -> None:
    def score(_index, _candidate, _dataset, corruption, _severity):
        # Exact macro delta: (36*1 - 3*10)/(39*100) = 1/650 > 0.
        return -10 if corruption == "clean" else 1

    receipt = select_stage1_candidates(
        _stage1_records(score),
        _passing_diagnostics(),
        _formal_spec(),
    )
    first = _decision(receipt, ALL_CANDIDATES[0])
    assert first["metrics"]["macro_global_iou_delta"]["exact"] == "1/650"
    assert "clean_iou_safety" in first["failed_gates"]
    assert receipt["selected_for_stage2"] == []


def test_positive_macro_with_pd_drop_and_fa_rise_fails_joint_safety() -> None:
    receipt = select_stage1_candidates(
        _stage1_records(
            lambda *_: 1,
            fa_delta_for_candidate=lambda *_: 10,
            pd_delta_for_candidate=lambda *_: -2,
        ),
        _passing_diagnostics(),
        _formal_spec(),
    )
    assert all(
        "pd_fa_joint_safety" in entry["failed_gates"]
        for entry in receipt["rejected_candidates"]
    )
    assert receipt["stage2_allowed"] is False


def test_filter_then_rank_allows_two_without_padding_to_three() -> None:
    def score(candidate_index, _candidate, _dataset, _corruption, _severity):
        return 2 - candidate_index if candidate_index < 2 else -1

    receipt = select_stage1_candidates(
        _stage1_records(score),
        _passing_diagnostics(),
        _formal_spec(),
    )

    assert receipt["scientific_status"] == "passed"
    assert receipt["stage2_allowed"] is True
    assert len(receipt["eligible_candidates"]) == 2
    assert receipt["selected_for_stage2"] == [
        ALL_CANDIDATES[0].to_dict(),
        ALL_CANDIDATES[1].to_dict(),
    ]
    assert validate_stage1_scientific_receipt(receipt) == ALL_CANDIDATES[:2]


def test_formal_gate_requires_complete_exact_diagnostics() -> None:
    with pytest.raises(CalibrationSelectionError, match="diagnostic evidence is incomplete"):
        select_stage1_candidates(
            _stage1_records(lambda *_: 1),
            _passing_diagnostics()[:-1],
            _formal_spec(),
        )


def test_retrospective_spec_rejects_invented_margin() -> None:
    with pytest.raises(CalibrationSelectionError, match="must not invent"):
        ScientificGateSpec(
            profile_id="invalid-retrospective",
            mode="retrospective_negative_replay",
            clean_iou_equivalence_margin=Fraction(0, 1),
        )


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_current_v2_390_records_replay_is_the_required_negative_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/binary_tent_ss_calibration_v3.yaml"
    assert config_path.is_file() and not config_path.is_symlink()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evidence = config["active_profile"]["evidence"]
    records_path = root / evidence["records_path"]
    assert records_path.is_file() and not records_path.is_symlink()
    records_bytes = records_path.read_bytes()
    assert hashlib.sha256(records_bytes).hexdigest() == evidence["records_sha256"]
    records = [
        json.loads(line) for line in records_bytes.decode("utf-8").splitlines()
    ]
    assert len(records) == evidence["expected_cell_records"] == 390

    receipt = select_stage1_candidates(
        records,
        diagnostics=None,
        gate_spec=ScientificGateSpec.retrospective_v2_negative_replay(),
    )
    assert receipt["protocol_status"] == "passed"
    assert receipt["scientific_status"] == "failed"
    assert receipt["stage2_allowed"] is False
    assert receipt["formal_fully_frozen_gate"] is False
    assert receipt["unresolved_gate_thresholds"] is True
    assert receipt["retrospective_negative_replay"] is True
    assert receipt["eligible_candidates"] == []
    assert receipt["selected_for_stage2"] == []
    assert all(
        entry["failed_gates"] == ["macro_delta_iou"]
        for entry in receipt["rejected_candidates"]
    )


def test_v3_config_binds_selector_and_leaves_unknown_formal_margins_unresolved() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/binary_tent_ss_calibration_v3.yaml").read_text()
    )
    selector = root / config["selector_v3"]["path"]
    assert hashlib.sha256(selector.read_bytes()).hexdigest() == config[
        "selector_v3"
    ]["sha256"]
    assert config["active_profile"]["mode"] == "retrospective_negative_replay"
    assert config["active_profile"]["stage2_authorization"] is False
    formal = config["formal_future_profile"]
    assert formal["status"] == "blocked_until_all_thresholds_are_frozen"
    assert formal["formal_fully_frozen_gate"] is False
    assert formal["unresolved_gate_thresholds"] is True
    assert formal["clean_safety"]["clean_iou_equivalence_margin"] is None
    assert formal["operating_point_safety"]["max_pd_drop"] is None
    assert formal["operating_point_safety"][
        "max_fa_increase_per_million_pixels"
    ] is None


def test_protocol_failure_raises_instead_of_becoming_scientific_failure() -> None:
    records = _stage1_records(lambda *_: 0)
    records[0]["test_label_opens"] = 1
    with pytest.raises(CalibrationSelectionError, match="firewall"):
        select_stage1_candidates(
            records,
            _passing_diagnostics(),
            _formal_spec(),
        )


def test_validator_rejects_boolean_only_forged_positive_receipt() -> None:
    candidate = ALL_CANDIDATES[0].to_dict()
    forged = {
        "schema_version": STAGE1_RECEIPT_SCHEMA_VERSION,
        "receipt_type": STAGE1_RECEIPT_TYPE,
        "protocol_status": "passed",
        "scientific_status": "passed",
        "stage2_allowed": True,
        "formal_fully_frozen_gate": True,
        "unresolved_gate_thresholds": False,
        "retrospective_negative_replay": False,
        "eligible_candidates": [
            {"candidate": candidate, "eligible": True, "failed_gates": []}
        ],
        "selected_for_stage2": [candidate],
    }
    with pytest.raises(CalibrationSelectionError, match="selector_protocol_id"):
        validate_stage1_scientific_receipt(forged)


def test_validator_rejects_tampered_positive_eligibility_or_order() -> None:
    def score(candidate_index, _candidate, _dataset, _corruption, _severity):
        return 2 - candidate_index if candidate_index < 2 else -1

    receipt = select_stage1_candidates(
        _stage1_records(score),
        _passing_diagnostics(),
        _formal_spec(),
    )

    failed_gate = deepcopy(receipt)
    failed_gate["eligible_candidates"][0]["failed_gates"] = ["macro_delta_iou"]
    with pytest.raises(CalibrationSelectionError, match="no failed gates"):
        validate_stage1_scientific_receipt(failed_gate)

    wrong_order = deepcopy(receipt)
    wrong_order["selected_for_stage2"].reverse()
    with pytest.raises(CalibrationSelectionError, match="eligible-ranking prefix"):
        validate_stage1_scientific_receipt(wrong_order)

    fake_metric = deepcopy(receipt)
    fake_metric["eligible_candidates"][0]["metrics"][
        "macro_global_iou_delta"
    ] = {
        "numerator": 0,
        "denominator": 1,
        "exact": "0/1",
        "value": 0.0,
    }
    with pytest.raises(CalibrationSelectionError, match="macro_delta_iou"):
        validate_stage1_scientific_receipt(fake_metric)

    fake_manifest = deepcopy(receipt)
    fake_manifest["scientific_gate"]["frozen_gate_manifest_sha256"] = "unfrozen"
    with pytest.raises(CalibrationSelectionError, match="manifest_sha256"):
        validate_stage1_scientific_receipt(fake_manifest)
