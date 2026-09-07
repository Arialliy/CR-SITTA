from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from metrics.result_eligibility import full_tree_identity


REPOSITORY = Path(__file__).resolve().parents[1]
ARCHIVE = REPOSITORY / "results/cr_sitta/stage_c0_r0_negative_v1"
SOURCE = REPOSITORY / "results/cr_sitta/p3_stage_c0_signal_audit_v2"
RECOVERY = REPOSITORY / "results/cr_sitta/p3_stage_c0_aggregate_recovery_v1"
REGISTRY = REPOSITORY / "configs/artifact_eligibility_registry_v4.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
pytestmark = pytest.mark.skipif(
    os.environ.get(LOCAL_ARTIFACT_TEST_ENV) != "1",
    reason=(
        "requires ignored local Stage-C negative-result archives; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_archive_tree_and_append_only_registry_binding() -> None:
    identity = full_tree_identity(ARCHIVE)
    assert identity.sha256 == (
        "e3f167a0cd24b346561002c90a70646b76fb9e65e59e37cdeb8a3f112ba50d0d"
    )
    assert identity.file_count == 8
    assert identity.total_size_bytes == 22380

    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert registry["registry_id"] == "cr-sitta-artifact-eligibility-registry-v4"
    assert registry["parent_registry"] == {
        "registry_id": "cr-sitta-artifact-eligibility-registry-v3",
        "config_path": "configs/artifact_eligibility_registry_v3.yaml",
        "config_sha256": (
            "ef2dc3f8c0bb65b307c3751d2ad0aa298e3a61f95e19808e4a6a3544e42048fb"
        ),
        "artifact_count": 15,
        "append_only": True,
    }
    assert _sha(REPOSITORY / registry["parent_registry"]["config_path"]) == (
        registry["parent_registry"]["config_sha256"]
    )
    artifact = registry["artifacts"][0]
    assert artifact["identity"]["expected_sha256"] == identity.sha256
    assert artifact["identity"]["file_count"] == identity.file_count
    assert artifact["identity"]["total_size_bytes"] == identity.total_size_bytes
    assert artifact["formal_protocol_complete"] is False
    assert artifact["source_stage_c0_protocol_complete"] is True
    assert artifact["source_stage_c0_recovery_verified"] is True


def test_receipt_and_authorization_copies_are_byte_identical() -> None:
    aggregate = SOURCE / "aggregate_phase/R0"
    assert (ARCHIVE / "protocol_receipt.json").read_bytes() == (
        aggregate / "science_decision_receipt.json"
    ).read_bytes()
    assert (ARCHIVE / "stage_c1_authorization.json").read_bytes() == (
        aggregate / "stage_c1_authorization.json"
    ).read_bytes()
    receipt = _json(ARCHIVE / "protocol_receipt.json")
    authorization = _json(ARCHIVE / "stage_c1_authorization.json")
    assert receipt["protocol_status"] == "protocol_complete"
    assert receipt["scientific_status"] == "scientific_no_eligible"
    assert receipt["eligible_space_ids"] == []
    assert authorization == {
        "formal_test_allowed": False,
        "parameter_space_ids": [],
        "reason": "scientific_no_eligible",
        "stage_c1_allowed": False,
        "stage_c_r1_r2_allowed": False,
    }


def test_gate_summary_matches_frozen_receipt() -> None:
    summary = _json(ARCHIVE / "mechanism_space_summary.json")
    receipt = _json(ARCHIVE / "protocol_receipt.json")
    evaluation = receipt["mechanism_evaluation"]
    assert summary["scientific_status"] == "scientific_no_eligible"
    assert summary["eligible_space_ids"] == []
    assert summary["mechanism_shared_gates"]["active_support_episode_fraction"][
        "receipt_fraction"
    ] == evaluation["active_support_episode_fraction"]
    assert summary["mechanism_shared_gates"][
        "candidate_proximal_fraction_among_active"
    ]["count"] == 2453
    expected_reasons = [
        "nonclean_finite_nonzero_proxy_gradient_fraction",
        "macro_outer_gradient_cosine",
    ]
    assert [row["parameter_space"] for row in summary["space_evaluations"]] == [
        "R-E1",
        "R-D0",
        "P2",
    ]
    for row, source_row in zip(
        summary["space_evaluations"], evaluation["space_evaluations"]
    ):
        assert row["eligible"] is False
        assert row["reason_codes"] == expected_reasons
        assert row["reason_codes"] == source_row["reason_codes"]
        assert row["finite_nonzero_proxy_gradient"]["count"] == 3022
        assert row["finite_nonzero_proxy_gradient"]["total"] == 4608
        assert row["finite_nonzero_proxy_gradient"]["passed"] is False
        assert row["macro_outer_gradient_cosine"]["passed"] is False
        assert row["positive_dataset_gate_passed"] is True
        assert row["improving_family_gate_passed"] is True
        assert row["identity_bit_exact"] is True
    assert summary["threshold_crossing_is_gate"] is False


def test_access_boundary_and_scientific_scope_are_fail_closed() -> None:
    negative = _json(ARCHIVE / "NEGATIVE_RESULT.json")
    access = _json(ARCHIVE / "access_audit.json")
    assert negative["candidate_episode_count"] == 4992
    assert negative["outer_episode_count"] == 4992
    assert negative["space_alignment_count"] == 14976
    assert negative["unique_nonclean_probe_episode_count"] == 4608
    assert negative["scientific_status"] == "scientific_no_eligible"
    for field in (
        "stage_c1_allowed",
        "stage_c1_started",
        "stage_c_r1_r2_allowed",
        "stage_c_r1_r2_started",
        "formal_test_allowed",
        "formal_test_started",
        "paper_result",
    ):
        assert negative[field] is False
    assert access["official_split_model"] == "train_test_only"
    assert access["validation_split_created"] is False
    assert access["candidate_phase"]["method_label_access_count"] == 0
    assert access["outer_phase"]["adaptation_gradient_uses_labels"] is False
    assert access["payload_access"] == {
        "validation_image_access_count": 0,
        "validation_mask_access_count": 0,
        "test_image_access_count": 0,
        "test_mask_access_count": 0,
        "test_identifier_metadata_use": "id_only_leakage_guard",
        "test_identifier_metadata_counted_as_payload": False,
    }


def test_recovery_chain_is_exact_read_only_and_non_authorizing() -> None:
    expected = {
        RECOVERY / "PRE_RUN_FREEZE.json": (
            "85df44adba4f103535ce5cccbd6d9ba5de219d2af5d46c3c49e92a703d976d83"
        ),
        RECOVERY / "RECOVERY_VERIFIED_RECEIPT.json": (
            "120a43c8c500d94f2d1a7303ed08db64bbb5e6b4001f8da59b9376d8048aa1ad"
        ),
        SOURCE / "AGGREGATE_POSTVERIFY_ABORT_RECEIPT.json": (
            "81266c924c8cbbbde00b8864f38094d941eff6aeee6b061dc8997457206b978e"
        ),
    }
    for path, digest in expected.items():
        assert _sha(path) == digest
        assert path.stat().st_mode & 0o222 == 0
    receipt = _json(RECOVERY / "RECOVERY_VERIFIED_RECEIPT.json")
    assert receipt["status"] == "recovery_verified"
    assert receipt["original_v2_runner_self_verification_passed"] is False
    result = receipt["scientific_result"]
    assert result["scientific_status"] == "scientific_no_eligible"
    assert result["stage_c1_authorized"] is False
    assert result["stage_c_r1_r2_authorized"] is False
    assert result["formal_test_authorized"] is False
    boundary = receipt["execution_boundary"]
    assert boundary["raw_source_image_payload_opens"] == 0
    assert boundary["raw_source_target_payload_deserializations"] == 0
    assert boundary["validation_payload_opens"] == 0
    assert boundary["test_payload_opens"] == 0
