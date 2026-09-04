from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_p3_stage_b_gradient_decomposition_v2 as runner


CONFIG_SHA = "7" * 64
CELL_SEAL = {
    "files": [{"path": "frozen.py", "sha256": "8" * 64}],
    "bundle_sha256": "9" * 64,
}
OUTPUT_ROOT = "results/cr_sitta/p3_stage_b_gradient_decomposition_v2"


def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("forbidden aggregate side effect was reached")


def _contract() -> SimpleNamespace:
    return SimpleNamespace(
        output_root=OUTPUT_ROOT,
        config_file_sha256=CONFIG_SHA,
        raw={
            "protocol_id": "cr-sitta-p3-stage-b1-gradient-decomposition-v2",
            "mechanism_evidence_flags": {"frozen": "gate"},
        },
    )


def _preflight() -> SimpleNamespace:
    return SimpleNamespace(
        lineage=tuple({"cell": index} for index in range(39)),
        records=tuple({"episode": index} for index in range(39 * 64)),
    )


def _verified(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        manifest_sha256="a" * 64,
        complete_sha256="b" * 64,
        mechanism_evidence_sha256="c" * 64,
        raw_numeric_audit_sha256="d" * 64,
        cell_count=39,
        episode_count=2496,
        stratified_record_count=390,
        summary_record_count=310,
        mechanism_statuses={
            "background_norm_dominance": "supported",
            "background_cancellation": "not_supported",
            "subthreshold_erasure": "not_estimable",
        },
        candidate_selection_performed=False,
        stage_b3_authorized=False,
        p5_authorized=False,
    )


def _install_no_cuda_or_data_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(runner, "guarded_load_outer_targets", _forbidden)
    monkeypatch.setattr(runner, "SourceCalibrationMethodInputDatasetV2", _forbidden)
    monkeypatch.setattr(runner, "build_d0_v3_outer_source_model", _forbidden)


def test_aggregate_cli_is_fixed_cpu_only_live_verified_and_non_authorizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    preflight = _preflight()
    _install_no_cuda_or_data_access(monkeypatch)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "_code_seal", lambda _contract: CELL_SEAL)

    collect_calls: list[dict[str, Any]] = []

    def collect(**kwargs: Any) -> Any:
        collect_calls.append(kwargs)
        return preflight

    monkeypatch.setattr(runner, "collect_stage_b1_preflight", collect)
    payloads = {name: f"payload:{name}\n".encode() for name in runner.AGGREGATE_MEMBERS}
    build_calls: list[tuple[Any, Any]] = []

    def build(value: Any, *, mechanism_gate: Any) -> dict[str, bytes]:
        build_calls.append((value, mechanism_gate))
        return payloads

    monkeypatch.setattr(runner, "build_stage_b1_aggregate_payloads", build)
    verify_calls: list[tuple[Path, dict[str, Any]]] = []

    def verify(path: Path, **kwargs: Any) -> Any:
        absolute = Path(path).resolve()
        verify_calls.append((absolute, kwargs))
        return _verified(absolute)

    monkeypatch.setattr(runner, "verify_stage_b1_aggregate_shard", verify)

    def publish(
        staging: Path,
        destination: Path,
        *,
        expected_members: list[str],
        semantic_verifier: Any,
    ) -> Path:
        assert expected_members == sorted(runner.AGGREGATE_MEMBERS)
        assert set(path.name for path in staging.iterdir()) == runner.AGGREGATE_MEMBERS
        semantic_verifier(staging)
        staging.rename(destination)
        semantic_verifier(destination)
        return destination

    monkeypatch.setattr(runner, "publish_flat_directory_noreplace", publish)

    result = runner.run_formal_aggregate(config_path=tmp_path / "config.yaml")
    destination = tmp_path / OUTPUT_ROOT / "aggregate_phase" / "R0"
    assert Path(result["path"]) == destination
    assert destination.is_dir()
    assert collect_calls == [
        {
            "repository_root": tmp_path,
            "output_root_relative": OUTPUT_ROOT,
            "config": contract.raw,
            "config_sha256": CONFIG_SHA,
            "expected_code_seal": CELL_SEAL,
        }
    ]
    assert build_calls == [(preflight, contract.raw["mechanism_evidence_flags"])]
    assert [kwargs["verify_live_cells"] for _, kwargs in verify_calls] == [False, True]
    assert all(
        kwargs["expected_cell_code_seal"] == CELL_SEAL
        for _, kwargs in verify_calls
    )
    assert result["mechanism_statuses"] == {
        "background_norm_dominance": "supported",
        "background_cancellation": "not_supported",
        "subthreshold_erasure": "not_estimable",
    }
    assert result["candidate_selection_performed"] is False
    assert result["stage_b3_authorized"] is False
    assert result["p5_authorized"] is False


def test_aggregate_refuses_existing_fixed_destination_before_cell_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    _install_no_cuda_or_data_access(monkeypatch)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "_code_seal", _forbidden)
    monkeypatch.setattr(runner, "collect_stage_b1_preflight", _forbidden)
    destination = tmp_path / OUTPUT_ROOT / "aggregate_phase" / "R0"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="aggregate destination exists"):
        runner.run_formal_aggregate(config_path=tmp_path / "config.yaml")


def test_aggregate_code_drift_fails_before_any_publication_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    _install_no_cuda_or_data_access(monkeypatch)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    seals = iter((CELL_SEAL, {**CELL_SEAL, "bundle_sha256": "d" * 64}))
    monkeypatch.setattr(runner, "_code_seal", lambda _contract: next(seals))
    monkeypatch.setattr(
        runner, "collect_stage_b1_preflight", lambda **_kwargs: _preflight()
    )
    monkeypatch.setattr(runner, "build_stage_b1_aggregate_payloads", _forbidden)
    monkeypatch.setattr(runner, "publish_flat_directory_noreplace", _forbidden)

    with pytest.raises(
        runner.P3StageB1RunnerError,
        match="changed during aggregate preflight",
    ):
        runner.run_formal_aggregate(config_path=tmp_path / "config.yaml")
    assert not (tmp_path / OUTPUT_ROOT).exists()


def test_verify_aggregate_rechecks_live_cells_and_reports_three_statuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    _install_no_cuda_or_data_access(monkeypatch)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "_code_seal", lambda _contract: CELL_SEAL)
    calls: list[tuple[Path, dict[str, Any]]] = []

    def verify(path: Path, **kwargs: Any) -> Any:
        calls.append((Path(path), kwargs))
        return _verified(Path(path))

    monkeypatch.setattr(runner, "verify_stage_b1_aggregate_shard", verify)
    result = runner.verify_formal_aggregate(config_path=tmp_path / "config.yaml")

    expected = tmp_path / OUTPUT_ROOT / "aggregate_phase" / "R0"
    assert calls[0][0] == expected
    assert calls[0][1]["verify_live_cells"] is True
    assert calls[0][1]["expected_cell_code_seal"] == CELL_SEAL
    assert len(result["mechanism_statuses"]) == 3
    assert result["stage_b3_authorized"] is False


@pytest.mark.parametrize("operation", ["aggregate", "verify-aggregate"])
def test_aggregate_commands_reject_initialized_cuda_before_contract_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(runner, "_load_contract", _forbidden)
    callable_operation = (
        runner.run_formal_aggregate
        if operation == "aggregate"
        else runner.verify_formal_aggregate
    )
    with pytest.raises(runner.P3StageB1RunnerError, match="must remain CPU-only"):
        callable_operation(config_path=tmp_path / "config.yaml")


def test_parser_exposes_only_fixed_path_aggregate_commands() -> None:
    for command in ("aggregate", "verify-aggregate"):
        parsed = runner._parser().parse_args([command])
        assert parsed.command == command
        assert not hasattr(parsed, "path")
