from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from analysis import p3_stage_b1_aggregate_recovery_v2_1 as recovery
from scripts import run_p3_stage_b1_aggregate_recovery_v2_1 as runner


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / recovery.CONFIG_RELATIVE_PATH
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"
LEDGER_SHA = recovery.CELL_LEDGER_SHA256
CELL_SEAL = {
    "files": [{"path": "frozen-cell.py", "sha256": "8" * 64}],
    "bundle_sha256": "9" * 64,
}
RECOVERY_SEAL = {
    "files": [{"path": "recovery.py", "sha256": "a" * 64}],
    "bundle_sha256": "b" * 64,
}


def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("forbidden side effect was reached")


def _contract() -> SimpleNamespace:
    return SimpleNamespace(
        config_file_sha256="1" * 64,
        protocol_id="cr-sitta-p3-stage-b1-aggregate-recovery-v2.1",
        output_root="results/recovery-test",
        aggregate_relative_path="aggregate_phase_v2_1/R0",
        incident_relative_path="AGGREGATE_PREPUBLICATION_FAILURE.json",
        raw={"parent_stage_b1_v2": {"cell_count": 39}},
    )


def _parent() -> SimpleNamespace:
    return SimpleNamespace(
        config_file_sha256="2" * 64,
        raw={"mechanism_evidence_flags": {"frozen": "gate"}},
    )


def _context() -> runner._Context:
    return runner._Context(_contract(), _parent(), CELL_SEAL, RECOVERY_SEAL)


def _preflight() -> SimpleNamespace:
    return SimpleNamespace(
        lineage=tuple({"cell": index} for index in range(39)),
        records=tuple({"episode": index} for index in range(39 * 64)),
    )


def _verified(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        manifest_sha256="3" * 64,
        complete_sha256="4" * 64,
        mechanism_evidence_sha256="5" * 64,
        raw_numeric_audit_sha256="6" * 64,
        recovery_receipt_sha256="7" * 64,
        cell_count=39,
        episode_count=2496,
        mechanism_statuses={
            "background_norm_dominance": "supported",
            "background_cancellation": "not_supported",
            "subthreshold_erasure": "not_estimable",
        },
        candidate_selection_performed=False,
        stage_b3_authorized=False,
        p5_authorized=False,
    )


def _cuda_initialized() -> bool:
    torch_module = runner.sys.modules.get("torch")
    cuda = None if torch_module is None else getattr(torch_module, "cuda", None)
    return bool(cuda is not None and cuda.is_initialized())


def _install_unit_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> runner._Context:
    cuda_initialized_before = _cuda_initialized()

    def assert_cuda_state_unchanged(*, boundary: str) -> None:
        del boundary
        assert _cuda_initialized() is cuda_initialized_before

    context = _context()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # These tests exercise mocked orchestration after pytest may legitimately
    # have initialized CUDA.  Preserve the production guard in production and
    # replace it here with the property relevant to the unit tests: the action
    # must not change CUDA initialization state.
    monkeypatch.setattr(runner, "_assert_cpu_only", assert_cuda_state_unchanged)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_context", lambda _path=runner.DEFAULT_CONFIG: context)
    monkeypatch.setattr(
        runner,
        "build_recovery_code_seal",
        lambda *_args, **_kwargs: RECOVERY_SEAL,
    )
    monkeypatch.setattr(
        runner,
        "verify_parent_code_seals",
        lambda *_args, **_kwargs: CELL_SEAL,
    )
    return context


def test_production_parent_gate_requires_only_the_proved_container_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = recovery.load_recovery_contract(CONFIG)
    parent = recovery.load_bound_parent_contract(
        contract, repository_root=REPOSITORY
    )
    frozen = parent.raw["mechanism_evidence_flags"]
    required = frozen["support_accounting"]["required_count_fields"]
    assert all(isinstance(required[key], tuple) for key in (
        "per_cell", "per_joint_cell", "per_stratum", "per_joint_stratum"
    ))

    # Keep the regression focused on gate validation.  The production v1/v2
    # delegated builder validates the entire gate before invoking evaluators.
    evidence = {"status": "not_supported", "per_dataset": {"NUAA-SIRST": {}}}
    monkeypatch.setattr(
        recovery._v2._v1,
        "_evaluate_one_mechanism_flag",
        lambda **_kwargs: evidence,
    )
    monkeypatch.setattr(
        recovery._v2._v1,
        "_evaluate_background_cancellation",
        lambda *_args, **_kwargs: evidence,
    )
    with pytest.raises(
        recovery._v2.P3StageB1AggregateV2Error,
        match="frozen mechanism gate semantics differ",
    ):
        recovery._v2.build_mechanism_evidence(
            (),
            config=parent.raw,
            config_sha256=parent.config_file_sha256,
            mechanism_gate=frozen,
        )

    adapted, proof = recovery.thaw_mechanism_gate_with_proof(frozen)
    assert all(isinstance(adapted["support_accounting"]["required_count_fields"][key], list)
               for key in ("per_cell", "per_joint_cell", "per_stratum", "per_joint_stratum"))
    assert proof["canonical_semantics_equal"] is True
    assert proof["frozen_semantic_sha256"] == proof["thawed_semantic_sha256"]
    assert all(proof[key] is False for key in (
        "thresholds_changed", "coverage_changed", "data_changed", "statistics_changed"
    ))
    result = recovery._v2.build_mechanism_evidence(
        (),
        config=parent.raw,
        config_sha256=parent.config_file_sha256,
        mechanism_gate=adapted,
    )
    assert set(result["P0_flags"]) == {
        "background_norm_dominance", "background_cancellation", "subthreshold_erasure"
    }
    assert result["selection"]["selected_candidates"] == []
    assert result["authorization"]["stage_b3_authorized"] is False


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local Stage-B1 cell artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_validate_is_production_bound_and_writes_nothing() -> None:
    root = REPOSITORY / "results/cr_sitta/p3_stage_b1_aggregate_recovery_v2_1"
    before = None if not root.exists() else (
        root.stat().st_ino,
        root.stat().st_mtime_ns,
        tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))),
    )
    script = Path(runner.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(CONFIG), "validate"],
        cwd=REPOSITORY,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    after = None if not root.exists() else (
        root.stat().st_ino,
        root.stat().st_mtime_ns,
        tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))),
    )
    assert before == after
    assert value["valid"] is True
    assert value["required_cell_count"] == 39
    assert value["required_episode_count"] == 2496
    assert value["cell_ledger_file_count"] == 78
    assert value["cell_manifest_complete_ledger_sha256"] == LEDGER_SHA
    assert value["adapter_semantics_equal"] is True
    assert value["filesystem_created"] is False
    assert value["raw_image_or_target_opened"] is False
    assert value["test_payload_opened"] is False
    assert value["validation_payload_opened"] is False
    assert value["cuda_initialized"] is False
    assert value["stage_b3_authorized"] is False


def test_aggregate_uses_live_preflight_and_atomic_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda_initialized_before = _cuda_initialized()
    context = _install_unit_context(tmp_path, monkeypatch)
    incident = b'{"frozen":"incident"}\n'
    monkeypatch.setattr(runner, "build_incident_payload", lambda _contract: incident)
    monkeypatch.setattr(
        runner,
        "build_cell_manifest_complete_ledger",
        lambda *_args, **_kwargs: (tuple({"path": str(i)} for i in range(78)), LEDGER_SHA),
    )
    collect_calls: list[dict[str, Any]] = []

    def collect(*args: Any, **kwargs: Any) -> Any:
        collect_calls.append({"args": args, **kwargs})
        return _preflight()

    monkeypatch.setattr(runner, "collect_live_parent_preflight", collect)
    payloads = {name: f"payload:{name}\n".encode() for name in runner.MEMBERS}
    build_calls: list[dict[str, Any]] = []

    def build(*args: Any, **kwargs: Any) -> dict[str, bytes]:
        build_calls.append({"args": args, **kwargs})
        return payloads

    monkeypatch.setattr(runner, "build_recovery_payloads", build)
    verify_calls: list[tuple[Path, dict[str, Any]]] = []

    def verify(path: Path, **kwargs: Any) -> Any:
        absolute = Path(path).resolve()
        verify_calls.append((absolute, kwargs))
        return _verified(absolute)

    monkeypatch.setattr(runner, "verify_recovery_aggregate", verify)
    result = runner.run_formal_aggregate(tmp_path / "unused.yaml")
    destination = (
        tmp_path / context.contract.output_root / context.contract.aggregate_relative_path
    )
    incident_path = (
        tmp_path / context.contract.output_root / context.contract.incident_relative_path
    )
    assert Path(result["path"]) == destination
    assert destination.is_dir()
    assert set(path.name for path in destination.iterdir()) == runner.MEMBERS
    assert incident_path.read_bytes() == incident
    assert incident_path.stat().st_mode & 0o222 == 0
    assert collect_calls[0]["expected_cell_code_seal"] == CELL_SEAL
    assert build_calls[0]["incident_sha256"] == hashlib.sha256(incident).hexdigest()
    assert build_calls[0]["cell_ledger_sha256"] == LEDGER_SHA
    assert [kwargs["verify_live_cells"] for _, kwargs in verify_calls] == [False, True]
    assert all(kwargs["expected_cell_code_seal"] == CELL_SEAL for _, kwargs in verify_calls)
    assert result["candidate_selection_performed"] is False
    assert result["stage_b3_authorized"] is False
    assert result["p5_authorized"] is False

    monkeypatch.setattr(runner, "collect_live_parent_preflight", _forbidden)
    with pytest.raises(FileExistsError, match="destination already exists"):
        runner.run_formal_aggregate(tmp_path / "unused.yaml")
    assert _cuda_initialized() is cuda_initialized_before


def test_tampered_existing_incident_fails_before_any_cell_read_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda_initialized_before = _cuda_initialized()
    context = _install_unit_context(tmp_path, monkeypatch)
    expected = b'{"expected":true}\n'
    monkeypatch.setattr(runner, "build_incident_payload", lambda _contract: expected)
    monkeypatch.setattr(runner, "build_cell_manifest_complete_ledger", _forbidden)
    monkeypatch.setattr(runner, "collect_live_parent_preflight", _forbidden)
    incident = tmp_path / context.contract.output_root / context.contract.incident_relative_path
    incident.parent.mkdir(parents=True)
    incident.write_bytes(b'{"tampered":true}\n')

    with pytest.raises(
        runner.P3StageB1AggregateRecoveryRunnerV21Error,
        match="existing recovery incident differs",
    ):
        runner.run_formal_aggregate(tmp_path / "unused.yaml")
    assert not (
        tmp_path / context.contract.output_root / context.contract.aggregate_relative_path
    ).exists()
    assert _cuda_initialized() is cuda_initialized_before


def test_verify_uses_only_the_fixed_path_and_live_rebuilds_all_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda_initialized_before = _cuda_initialized()
    context = _install_unit_context(tmp_path, monkeypatch)
    incident_payload = b'{"incident":"frozen"}\n'
    monkeypatch.setattr(
        runner, "build_incident_payload", lambda _contract: incident_payload
    )
    incident = (
        tmp_path
        / context.contract.output_root
        / context.contract.incident_relative_path
    )
    incident.parent.mkdir(parents=True)
    incident.write_bytes(incident_payload)
    incident.chmod(0o444)
    monkeypatch.setattr(
        runner,
        "build_cell_manifest_complete_ledger",
        lambda *_args, **_kwargs: (
            tuple({"path": str(i)} for i in range(78)),
            LEDGER_SHA,
        ),
    )
    calls: list[tuple[Path, dict[str, Any]]] = []

    def verify(path: Path, **kwargs: Any) -> Any:
        calls.append((Path(path), kwargs))
        return _verified(Path(path))

    monkeypatch.setattr(runner, "verify_recovery_aggregate", verify)
    value = runner.verify_formal_aggregate(tmp_path / "unused.yaml")
    expected = (
        tmp_path
        / context.contract.output_root
        / context.contract.aggregate_relative_path
    )
    assert calls == [(expected, {
        "contract": context.contract,
        "parent": context.parent,
        "repository_root": tmp_path,
        "verify_live_cells": True,
        "expected_cell_code_seal": CELL_SEAL,
    })]
    assert value["path"] == str(expected)
    assert value["incident_sha256"] == hashlib.sha256(incident_payload).hexdigest()
    assert value["stage_b3_authorized"] is False
    assert _cuda_initialized() is cuda_initialized_before


def test_code_seal_drift_fails_before_creating_recovery_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda_initialized_before = _cuda_initialized()
    context = _install_unit_context(tmp_path, monkeypatch)
    incident = b'{"incident":true}\n'
    monkeypatch.setattr(runner, "build_incident_payload", lambda _contract: incident)
    monkeypatch.setattr(
        runner,
        "build_cell_manifest_complete_ledger",
        lambda *_args, **_kwargs: (tuple({"path": str(i)} for i in range(78)), LEDGER_SHA),
    )
    monkeypatch.setattr(
        runner, "collect_live_parent_preflight", lambda *_args, **_kwargs: _preflight()
    )
    monkeypatch.setattr(
        runner,
        "build_recovery_payloads",
        lambda *_args, **_kwargs: {
            name: f"payload:{name}".encode() for name in runner.MEMBERS
        },
    )
    monkeypatch.setattr(
        runner,
        "build_recovery_code_seal",
        lambda *_args, **_kwargs: {
            **RECOVERY_SEAL,
            "bundle_sha256": "c" * 64,
        },
    )
    with pytest.raises(
        runner.P3StageB1AggregateRecoveryRunnerV21Error,
        match="recovery implementation changed",
    ):
        runner.run_formal_aggregate(tmp_path / "unused.yaml")
    assert not (tmp_path / context.contract.output_root).exists()
    assert _cuda_initialized() is cuda_initialized_before


@pytest.mark.parametrize("visible", [None, "0", "0,1", " "])
def test_all_commands_require_explicit_empty_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch, visible: str | None
) -> None:
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(runner, "load_recovery_contract", _forbidden)
    with pytest.raises(
        runner.P3StageB1AggregateRecoveryRunnerV21Error,
        match="requires CUDA_VISIBLE_DEVICES",
    ):
        runner.validate_only(CONFIG)


def test_initialized_cuda_fails_before_contract_or_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch_module = runner.sys.modules["torch"]
    monkeypatch.setattr(torch_module.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(runner, "load_recovery_contract", _forbidden)
    with pytest.raises(
        runner.P3StageB1AggregateRecoveryRunnerV21Error,
        match="initialized CUDA",
    ):
        runner.validate_only(CONFIG)


def test_parser_exposes_only_fixed_recovery_commands() -> None:
    for command in ("validate", "aggregate", "verify"):
        parsed = runner._parser().parse_args([command])
        assert parsed.command == command
        assert not hasattr(parsed, "path")


def test_frozen_v1_v2_aggregate_implementations_are_unchanged() -> None:
    expected = {
        "analysis/p3_stage_b1_aggregate.py":
            "f7d38b7b3a66f00c69f7e2eb8dae3bf38246053a2105441d639affefc7ad7dd7",
        "analysis/p3_stage_b1_aggregate_v2.py":
            "86bb3584924d2b6a8943e756fd36309efd286fd3875fbce6168a410e563ceea3",
        "analysis/p3_stage_b1_contract_v2.py":
            "f304effb541d08bc1b24d8aceefc28f0868088641da41ab87d8ddad15b0a85c5",
        "scripts/run_p3_stage_b_gradient_decomposition_v2.py":
            "07e9f7ef39e3b6a1f4036948df8456c21d7f796afc52301ccab873c4338ac8b4",
    }
    observed = {
        path: hashlib.sha256((REPOSITORY / path).read_bytes()).hexdigest()
        for path in expected
    }
    assert observed == expected
