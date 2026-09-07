from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

import recover_p3_stage_c0_aggregate_v1 as recovery
from analysis.stage_c_science_gate_v1 import StageCAuthorization


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/p3_stage_c0_aggregate_recovery_v1.yaml"
ABORT = (
    REPOSITORY
    / "results/cr_sitta/p3_stage_c0_signal_audit_v2/"
    "AGGREGATE_POSTVERIFY_ABORT_RECEIPT.json"
)
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"
REQUIRES_LOCAL_ARTIFACTS = pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local Stage-C recovery artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)


def _parsed_production_contract() -> recovery.RecoveryContract:
    payload = CONFIG.read_bytes()
    raw = yaml.load(payload.decode("utf-8"), Loader=recovery._UniqueKeyLoader)
    return recovery._parse_contract(
        raw,
        repository=REPOSITORY,
        config_path=CONFIG,
        config_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("forbidden call")


def test_production_config_is_frozen_and_runner_pinned() -> None:
    contract = _parsed_production_contract()
    assert contract.raw["protocol_id"] == recovery.PROTOCOL_ID
    assert contract.raw["freeze"]["state"] == "frozen"
    assert recovery.FROZEN_CONFIG_SHA256 != "TO_BE_FROZEN"
    assert recovery.current_config_sha256(CONFIG) == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()
    assert recovery.load_contract(CONFIG).config_sha256 == recovery.FROZEN_CONFIG_SHA256


@REQUIRES_LOCAL_ARTIFACTS
def test_abort_receipt_is_exact_read_only_and_records_original_failure() -> None:
    assert hashlib.sha256(ABORT.read_bytes()).hexdigest() == recovery.ABORT_RECEIPT_SHA256
    assert ABORT.stat().st_mode & 0o222 == 0
    value = json.loads(ABORT.read_text(encoding="utf-8"))
    assert value["observed_command"]["exit_code"] == 2
    assert (
        value["observed_command"]["terminal_error_message"]
        == recovery.KNOWN_TERMINAL_ERROR
    )
    assert value["v2_verifier_outcome"]["original_v2_runner_self_verification_passed"] is False
    assert value["scientific_boundary"]["stage_c1_authorized"] is False
    assert value["scientific_boundary"]["stage_c1_started"] is False
    assert value["scientific_boundary"]["formal_test_authorized"] is False


@pytest.mark.parametrize(
    ("authorization", "stored"),
    [
        (
            StageCAuthorization(False, (), "scientific_no_eligible"),
            {
                "stage_c1_allowed": False,
                "parameter_space_ids": [],
                "reason": "scientific_no_eligible",
                "stage_c_r1_r2_allowed": False,
                "formal_test_allowed": False,
            },
        ),
        (
            StageCAuthorization(
                True,
                ("R-E1", "P2"),
                "scientific_eligible_for_predefined_train_only_C1",
            ),
            {
                "stage_c1_allowed": True,
                "parameter_space_ids": ["R-E1", "P2"],
                "reason": "scientific_eligible_for_predefined_train_only_C1",
                "stage_c_r1_r2_allowed": False,
                "formal_test_allowed": False,
            },
        ),
    ],
)
def test_container_adapter_preserves_negative_and_positive_authorization_semantics(
    authorization: StageCAuthorization, stored: dict[str, Any]
) -> None:
    producer = asdict(authorization)
    adapted, proof = recovery.canonicalize_authorization_with_proof(
        producer, stored
    )
    assert adapted == stored
    assert proof["canonical_semantics_equal"] is True
    assert proof["producer_semantic_sha256"] == proof["stored_semantic_sha256"]
    assert proof["only_container_difference"] == {
        "json_path": "$.parameter_space_ids",
        "producer_python_type": "tuple",
        "stored_python_type": "list",
    }
    assert all(
        proof[field] is False
        for field in (
            "thresholds_changed",
            "coverage_changed",
            "data_changed",
            "statistics_changed",
        )
    )


def test_container_adapter_rejects_any_value_or_second_container_difference() -> None:
    producer = asdict(StageCAuthorization(False, (), "scientific_no_eligible"))
    wrong_value = recovery._json_native(producer)
    wrong_value["reason"] = "changed"
    with pytest.raises(recovery.StageC0AggregateRecoveryError, match="value/type differs"):
        recovery.canonicalize_authorization_with_proof(producer, wrong_value)

    second = dict(producer)
    second["parameter_space_ids"] = ()
    stored = recovery._json_native(second)
    second["extra"] = ("x",)
    stored["extra"] = ["x"]
    with pytest.raises(
        recovery.StageC0AggregateRecoveryError,
        match="not exactly parameter_space_ids",
    ):
        recovery.canonicalize_authorization_with_proof(second, stored)


def test_in_memory_adapter_is_scoped_and_restores_v2_asdict() -> None:
    authorization = StageCAuthorization(False, (), "scientific_no_eligible")
    original = recovery._v2.asdict
    assert recovery._v2.asdict(authorization)["parameter_space_ids"] == ()
    with recovery._v2_authorization_json_projection():
        assert recovery._v2.asdict(authorization)["parameter_space_ids"] == []
    assert recovery._v2.asdict is original
    assert recovery._v2.asdict(authorization)["parameter_space_ids"] == ()


def _install_collect_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_error: str | None,
    adapted_error: str | None = None,
) -> tuple[recovery.RecoveryContract, list[str]]:
    contract = _parsed_production_contract()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delitem(sys.modules, recovery.TARGET_LOADER_MODULE, raising=False)
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        monkeypatch.setattr(torch_module.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        recovery,
        "validate_live_source_bindings",
        lambda _contract: {"bound": True},
    )
    monkeypatch.setattr(
        recovery,
        "_verify_freeze_receipt",
        lambda _contract, _sources=None: ({"frozen": True}, "f" * 64),
    )
    parent = object()
    monkeypatch.setattr(recovery, "_load_parent_contract", lambda _contract: parent)
    monkeypatch.setattr(
        recovery._v2,
        "_artifact_destination",
        lambda *_args, **_kwargs: Path("aggregate"),
    )
    calls: list[str] = []

    def verify(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        authorization = StageCAuthorization(False, (), "scientific_no_eligible")
        projected = recovery._v2.asdict(authorization)["parameter_space_ids"]
        calls.append(type(projected).__name__)
        if len(calls) == 1:
            if raw_error is None:
                return {
                    "scientific_status": "scientific_no_eligible",
                    "eligible_space_ids": [],
                    "stage_c1_allowed": False,
                    "stage_c_r1_r2_allowed": False,
                    "formal_test_allowed": False,
                }
            raise recovery._v2.StageC0ProtocolError(raw_error)
        if adapted_error is not None:
            raise recovery._v2.StageC0ProtocolError(adapted_error)
        return {
            "scientific_status": "scientific_no_eligible",
            "eligible_space_ids": [],
            "stage_c1_allowed": False,
            "stage_c_r1_r2_allowed": False,
            "formal_test_allowed": False,
        }

    monkeypatch.setattr(recovery._v2, "verify_aggregate_artifact", verify)
    monkeypatch.setattr(
        recovery,
        "_independent_recompute",
        lambda *_args, **_kwargs: {"adapter_proof": {"canonical_semantics_equal": True}},
    )
    return contract, calls


def test_collect_requires_raw_terminal_then_adapted_full_verifier_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, calls = _install_collect_fakes(
        monkeypatch, raw_error=recovery.KNOWN_TERMINAL_ERROR
    )
    original = recovery._v2.asdict
    proof, sources, freeze_sha = recovery.collect_recovery_proof(contract)
    assert calls == ["tuple", "list"]
    assert recovery._v2.asdict is original
    assert proof["unadapted_v2_verifier"]["passed"] is False
    assert proof["unadapted_v2_verifier"]["known_unique_terminal_reached"] is True
    assert proof["adapted_full_v2_verifier"]["passed"] is True
    assert proof["adapted_full_v2_verifier"]["exit_semantics"] == (
        "returned_manifest_without_second_error"
    )
    assert sources == {"bound": True}
    assert freeze_sha == "f" * 64


def test_collect_rejects_an_earlier_or_missing_v2_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _calls = _install_collect_fakes(
        monkeypatch, raw_error="aggregate evidence differs from outer artifacts"
    )
    with pytest.raises(
        recovery.StageC0AggregateRecoveryError, match="stopped before known terminal"
    ):
        recovery.collect_recovery_proof(contract)

    contract, _calls = _install_collect_fakes(monkeypatch, raw_error=None)
    with pytest.raises(
        recovery.StageC0AggregateRecoveryError, match="unexpectedly passed"
    ):
        recovery.collect_recovery_proof(contract)


def test_collect_rejects_a_second_error_after_the_scoped_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, calls = _install_collect_fakes(
        monkeypatch,
        raw_error=recovery.KNOWN_TERMINAL_ERROR,
        adapted_error="hidden second verifier error",
    )
    with pytest.raises(
        recovery.StageC0AggregateRecoveryError,
        match="adapted full v2 verifier did not reach exit 0",
    ):
        recovery.collect_recovery_proof(contract)
    assert calls == ["tuple", "list"]


def test_cpu_and_payload_firewalls_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(
        recovery.StageC0AggregateRecoveryError,
        match="requires CUDA_VISIBLE_DEVICES",
    ):
        recovery._assert_cpu_only("test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setitem(sys.modules, recovery.TARGET_LOADER_MODULE, object())
    with pytest.raises(
        recovery.StageC0AggregateRecoveryError,
        match="target payload loader imported",
    ):
        recovery._assert_payload_firewall("test")
    monkeypatch.delitem(sys.modules, recovery.TARGET_LOADER_MODULE)


def test_freeze_receipt_verifier_rejects_writable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    receipt_path = output / "PRE_RUN_FREEZE.json"
    contract = recovery.RecoveryContract(
        repository=tmp_path,
        config_path=tmp_path / "config.yaml",
        config_sha256="a" * 64,
        raw={
            "freeze": {"pre_run_freeze_receipt": "out/PRE_RUN_FREEZE.json"},
            "output": {
                "root": "out",
                "recovery_verified_receipt": "RECOVERY_VERIFIED_RECEIPT.json",
            },
        },
    )
    expected = {"frozen": True}
    receipt_path.write_bytes(recovery._canonical_json_bytes(expected, newline=True))
    receipt_path.chmod(0o644)
    monkeypatch.setattr(
        recovery, "_expected_freeze_receipt", lambda _contract, _source: expected
    )
    with pytest.raises(
        recovery.StageC0AggregateRecoveryError,
        match="freeze receipt is writable",
    ):
        recovery._verify_freeze_receipt(contract, {"source": "bound"})

def test_freeze_receipt_records_output_absent_and_no_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _parsed_production_contract()
    monkeypatch.setattr(
        recovery,
        "_recovery_code_seal",
        lambda _contract: {"files": [], "bundle_sha256": "a" * 64},
    )
    value = recovery._expected_freeze_receipt(contract, {"source": "bound"})
    assert value["recovery_verified_receipt_absent_at_publication"] is True
    assert value["stage_c1_authorized"] is False
    assert value["stage_c_r1_r2_authorized"] is False
    assert value["formal_test_authorized"] is False
    assert value["raw_source_image_payload_opens"] == 0
    assert value["raw_source_target_payload_deserializations"] == 0


def test_existing_recovery_destination_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "out/RECOVERY_VERIFIED_RECEIPT.json"
    destination.parent.mkdir()
    destination.write_text("owned", encoding="utf-8")
    contract = recovery.RecoveryContract(
        repository=tmp_path,
        config_path=tmp_path / "config.yaml",
        config_sha256="a" * 64,
        raw={
            "output": {
                "root": "out",
                "recovery_verified_receipt": "RECOVERY_VERIFIED_RECEIPT.json",
            }
        },
    )
    monkeypatch.setattr(recovery, "collect_recovery_proof", _forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        recovery.run_recovery(contract)
    assert destination.read_text(encoding="utf-8") == "owned"


def test_parser_exposes_only_fixed_paths_and_commands() -> None:
    for command in (
        "print-config-sha256",
        "verify-config",
        "freeze",
        "validate",
        "recover",
        "verify",
    ):
        parsed = recovery._parser().parse_args([command])
        assert parsed.command == command
        assert not hasattr(parsed, "path")


def test_recovery_source_has_no_raw_payload_or_stage_execution_call() -> None:
    source = Path(recovery.__file__).read_text(encoding="utf-8")
    assert "open_outer_train_targets(" not in source
    assert "load_outer_evaluator_targets_v2(" not in source
    assert "_run_candidate(" not in source
    assert "_run_outer(" not in source
    assert "_run_aggregate(" not in source


@REQUIRES_LOCAL_ARTIFACTS
def test_frozen_v2_inputs_are_unchanged() -> None:
    expected = {
        "configs/p3_stage_c0_signal_audit_v2.yaml": (
            "be10f26359a218687f3d51ffd4aaeea56646c266e7d3862958682b2ccac54ce1"
        ),
        "run_p3_stage_c0_signal_audit_v2.py": (
            "1e3cdf272d0a4d239d47ff98580faa7547c2c81c7fe4fbf6e0eb43db749c44bd"
        ),
        "results/cr_sitta/p3_stage_c0_signal_audit_v2/PRE_RUN_FREEZE.json": (
            "15608a2d69bbd04ea0fb5be9d9e565285fa49aec409b737d1d775629d86d6a24"
        ),
    }
    observed = {
        path: hashlib.sha256((REPOSITORY / path).read_bytes()).hexdigest()
        for path in expected
    }
    assert observed == expected
