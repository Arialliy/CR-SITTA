from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_binary_tent_ss_calibration_v3 as runner
import tta.binary_tent_ss_stage1_gate_artifact_v3 as artifact
from tta.binary_tent_ss_calibration_selector_v3 import (
    ALL_CANDIDATES,
    CONDITIONS,
    DATASETS,
    FORMAL_FROZEN_MODE,
    REQUIRED_HARD_GATES,
    REQUIRED_PROTOCOL_AUDIT,
    SELECTOR_PROTOCOL_ID,
    SS_BN_PROTOCOL,
    CandidateDiagnosticEvidence,
    ScientificGateSpec,
    select_stage1_candidates,
)


def _endpoint(*, intersection: int) -> dict[str, int]:
    return {
        "intersection_pixels": intersection,
        "union_pixels": 100,
        "false_alarm_pixels": 10,
        "total_image_pixels": 1_000_000,
        "detected_targets": 80,
        "total_targets": 100,
    }


def _formal_stage1_records(selected_count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(ALL_CANDIDATES):
        improvement = (
            selected_count - candidate_index
            if candidate_index < selected_count
            else -1
        )
        for dataset in DATASETS:
            for corruption, severity in CONDITIONS:
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
                        "hard_gates": {key: True for key in REQUIRED_HARD_GATES},
                        "protocol_audit": {
                            key: True for key in REQUIRED_PROTOCOL_AUDIT
                        },
                        "endpoints": {
                            "tent_pre": _endpoint(intersection=50),
                            "tent_post": _endpoint(
                                intersection=50 + improvement
                            ),
                        },
                    }
                )
    assert len(records) == 390
    return records


def _formal_diagnostics() -> tuple[CandidateDiagnosticEvidence, ...]:
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


def _diagnostic_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate": evidence.candidate.to_dict(),
            "parameter_update_episodes": evidence.parameter_update_episodes,
            "parameter_update_evaluated_episodes": (
                evidence.parameter_update_evaluated_episodes
            ),
            "functional_change_episodes": evidence.functional_change_episodes,
            "functional_change_evaluated_episodes": (
                evidence.functional_change_evaluated_episodes
            ),
            "objective_decrease_episodes": evidence.objective_decrease_episodes,
            "objective_evaluated_episodes": (
                evidence.objective_evaluated_episodes
            ),
        }
        for evidence in _formal_diagnostics()
    ]


def _canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(artifact.canonical_json_bytes(row) for row in rows)


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _file_binding(path: Path, *, line_count: int | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": len(data.splitlines()) if line_count is None else line_count,
    }


@dataclass(frozen=True)
class _FormalFixture:
    binding: artifact.FrozenGateConfigBinding
    root: Path
    config: Path
    selector: Path
    aggregate: Path
    manifest: Path
    completion: Path
    gate: Path
    records: Path
    diagnostics: Path
    receipt: Path


def _make_formal_fixture(
    tmp_path: Path,
    *,
    selected_count: int = 1,
    swap_diagnostic_candidates: bool = False,
    noncanonical_receipt: bool = False,
    bad_records_line_count: bool = False,
) -> _FormalFixture:
    if not 1 <= selected_count <= 3:
        raise ValueError("positive fixture requires one to three selections")
    root = tmp_path / "project"
    selector = root / "tta/binary_tent_ss_calibration_selector_v3.py"
    config = root / "configs/binary_tent_ss_calibration_v3.yaml"
    aggregate = root / "results/formal_stage1/aggregate"
    selector.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    aggregate.mkdir(parents=True)
    shutil.copyfile(ROOT / "tta/binary_tent_ss_calibration_selector_v3.py", selector)

    records = aggregate / "stage1_records.jsonl"
    diagnostics = aggregate / "stage1_diagnostic_evidence.jsonl"
    gate = aggregate / "frozen_scientific_gate_manifest.json"
    receipt = aggregate / artifact.RECEIPT_FILENAME
    manifest = aggregate / artifact.MANIFEST_FILENAME
    completion = aggregate / "COMPLETE.json"

    record_values = _formal_stage1_records(selected_count)
    records.write_bytes(_canonical_jsonl(record_values))
    diagnostic_values = _diagnostic_rows()
    if swap_diagnostic_candidates:
        diagnostic_values[0], diagnostic_values[1] = (
            diagnostic_values[1],
            diagnostic_values[0],
        )
    diagnostics.write_bytes(_canonical_jsonl(diagnostic_values))

    records_sha = hashlib.sha256(records.read_bytes()).hexdigest()
    diagnostics_sha = hashlib.sha256(diagnostics.read_bytes()).hexdigest()
    profile_id = "runner-v3-frozen-synthetic-unit-test-gate"
    threshold_source = "synthetic_test_fixture_not_experimental_margin"
    gate_value = {
        "schema_version": 1,
        "artifact_type": artifact.GATE_MANIFEST_ARTIFACT_TYPE,
        "contract_id": artifact.AUTHORIZATION_CONTRACT_ID,
        "profile_id": profile_id,
        "selector_protocol_id": SELECTOR_PROTOCOL_ID,
        "preregistered_before_evidence": True,
        "threshold_source": threshold_source,
        "candidate_order": [candidate.to_dict() for candidate in ALL_CANDIDATES],
        "top_k_after_filter": 3,
        "allow_fewer_than_top_k": True,
        "thresholds": {
            "minimum_positive_cells": 4,
            "minimum_positive_corruption_families": 2,
            "minimum_positive_datasets": 2,
            "clean_iou_equivalence_margin": _fraction(Fraction(1, 100)),
            "max_pd_drop": _fraction(Fraction(1, 100)),
            "max_fa_increase_per_million_pixels": _fraction(Fraction(1, 1)),
            "minimum_parameter_update_fraction_above_null": _fraction(
                Fraction(1, 2)
            ),
            "minimum_functional_change_fraction_above_null": _fraction(
                Fraction(1, 2)
            ),
            "minimum_objective_decrease_fraction": _fraction(Fraction(1, 2)),
        },
        "evidence": {
            "stage1_records_sha256": records_sha,
            "stage1_cell_record_count": 390,
            "diagnostic_evidence_sha256": diagnostics_sha,
            "diagnostic_candidate_count": 10,
            "evaluated_episodes_per_candidate": 4,
        },
    }
    gate.write_bytes(artifact.canonical_json_bytes(gate_value))
    gate_sha = hashlib.sha256(gate.read_bytes()).hexdigest()
    gate_spec = ScientificGateSpec(
        profile_id=profile_id,
        mode=FORMAL_FROZEN_MODE,
        min_positive_cells=4,
        min_positive_corruption_families=2,
        min_positive_datasets=2,
        clean_iou_equivalence_margin=Fraction(1, 100),
        max_pd_drop=Fraction(1, 100),
        max_fa_increase=Fraction(1, 1),
        min_parameter_update_fraction=Fraction(1, 2),
        min_functional_change_fraction=Fraction(1, 2),
        min_objective_decrease_fraction=Fraction(1, 2),
        top_k_after_filter=3,
        allow_fewer_than_top_k=True,
        threshold_source=threshold_source,
        frozen_gate_manifest_sha256=gate_sha,
    )
    receipt_value = select_stage1_candidates(
        record_values,
        _formal_diagnostics(),
        gate_spec,
    )
    assert len(receipt_value["selected_for_stage2"]) == selected_count
    if noncanonical_receipt:
        import json

        receipt_bytes = (
            json.dumps(receipt_value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")
    else:
        receipt_bytes = artifact.canonical_json_bytes(receipt_value)
    receipt.write_bytes(receipt_bytes)

    selector_sha = hashlib.sha256(selector.read_bytes()).hexdigest()
    manifest_value = {
        "schema_version": 1,
        "artifact_type": artifact.AGGREGATE_ARTIFACT_TYPE,
        "artifact_complete": True,
        "formal_fully_frozen_gate": True,
        "selector_protocol_id": SELECTOR_PROTOCOL_ID,
        "selector_sha256": selector_sha,
        "candidate_order": [candidate.to_dict() for candidate in ALL_CANDIDATES],
        "stage1_cell_record_count": 390,
        "diagnostic_candidate_count": 10,
        "completion_filename": completion.name,
        "files": {
            "frozen_gate_manifest": _file_binding(gate),
            "stage1_records": _file_binding(
                records,
                line_count=389 if bad_records_line_count else None,
            ),
            "diagnostic_evidence": _file_binding(diagnostics),
            "scientific_receipt": _file_binding(receipt),
        },
    }
    manifest.write_bytes(artifact.canonical_json_bytes(manifest_value))
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    completion_value = {
        "schema_version": 1,
        "artifact_type": artifact.COMPLETE_ARTIFACT_TYPE,
        "complete": True,
        "formal_fully_frozen_gate": True,
        "manifest_sha256": manifest_sha,
        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "stage1_cell_record_count": 390,
        "diagnostic_candidate_count": 10,
    }
    completion.write_bytes(artifact.canonical_json_bytes(completion_value))

    config_value = {
        "schema_version": 3,
        "protocol_id": artifact.CONFIG_PROTOCOL_ID,
        "selector_v3": {
            "path": str(selector.relative_to(root)),
            "sha256": selector_sha,
            "protocol_id": SELECTOR_PROTOCOL_ID,
            "pure_cpu": True,
            "exact_rational_arithmetic_from_integer_counts": True,
            "filter_before_ranking": True,
            "top_k_after_filter": 3,
            "allow_fewer_than_top_k": True,
        },
        "stage2_authorization": {
            "schema_version": 1,
            "contract_id": artifact.AUTHORIZATION_CONTRACT_ID,
            "status": artifact.FORMAL_STATUS,
            "aggregate_manifest": {
                "path": str(manifest.relative_to(root)),
                "sha256": manifest_sha,
            },
        },
    }
    config.write_text(
        yaml.safe_dump(config_value, sort_keys=False),
        encoding="utf-8",
    )
    binding = artifact.FrozenGateConfigBinding(
        path=config,
        expected_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        project_root=root,
    )
    return _FormalFixture(
        binding=binding,
        root=root,
        config=config,
        selector=selector,
        aggregate=aggregate,
        manifest=manifest,
        completion=completion,
        gate=gate,
        records=records,
        diagnostics=diagnostics,
        receipt=receipt,
    )


class _RecordingBackend:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.calls: list[str] = []

    def create_stage2_paths(self, authorization):
        self.calls.append("create_stage2_paths")
        stage2 = self.output_root / "stage2"
        stage2.mkdir(parents=True)
        return stage2

    def acquire_gpu_lease(self, authorization):
        self.calls.append("acquire_gpu_lease")
        return "lease"

    def initialize_cuda(self, authorization, lease):
        self.calls.append("initialize_cuda")
        return "cuda"

    def create_process_ids(self, authorization):
        self.calls.append("create_process_ids")
        return tuple(
            f"candidate-{index}"
            for index, _ in enumerate(authorization.selected_for_stage2, start=1)
        )

    def build_worker_commands(
        self, authorization, paths, lease, cuda_runtime, process_ids
    ):
        self.calls.append("build_worker_commands")
        return tuple(("worker", process_id) for process_id in process_ids)

    def launch_workers(
        self,
        authorization,
        paths,
        lease,
        cuda_runtime,
        process_ids,
        commands,
    ):
        self.calls.append("launch_workers")
        return {
            "selected_count": len(authorization.selected_for_stage2),
            "commands": commands,
        }


def _assert_zero_side_effects(backend: _RecordingBackend) -> None:
    assert backend.calls == []
    assert not backend.output_root.exists()


def test_current_production_gate_remains_blocked_and_has_no_side_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    backend = _RecordingBackend(tmp_path / "must-not-exist")
    with pytest.raises(runner.ScientificGateBlockedError, match="unresolved"):
        runner.launch_stage2(backend)
    _assert_zero_side_effects(backend)

    assert runner.main(["authorize-stage2"]) == runner.EXIT_SCIENTIFIC_GATE_BLOCKED
    assert "SCIENTIFIC_GATE_BLOCKED" in capsys.readouterr().err
    _assert_zero_side_effects(backend)


@pytest.mark.parametrize("selected_count", (1, 2, 3))
def test_controlled_positive_fixture_requires_complete_formal_artifact_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_count: int,
) -> None:
    fixture = _make_formal_fixture(tmp_path, selected_count=selected_count)
    monkeypatch.setattr(
        runner, "_production_config_binding", lambda: fixture.binding
    )
    authorization = runner.authorize_stage2()
    assert len(authorization.selected_for_stage2) == selected_count
    assert authorization.config_sha256 == fixture.binding.expected_sha256
    assert authorization.receipt_path == fixture.receipt

    backend = _RecordingBackend(tmp_path / "published")
    result = runner.launch_stage2(backend)
    assert result["selected_count"] == selected_count
    assert backend.calls == [
        "create_stage2_paths",
        "acquire_gpu_lease",
        "initialize_cuda",
        "create_process_ids",
        "build_worker_commands",
        "launch_workers",
    ]


def test_standalone_self_signed_receipt_has_no_authorization_entrypoint(
    tmp_path: Path,
) -> None:
    gate_spec = ScientificGateSpec(
        profile_id="standalone-self-signed-receipt",
        mode=FORMAL_FROZEN_MODE,
        min_positive_cells=4,
        min_positive_corruption_families=2,
        min_positive_datasets=2,
        clean_iou_equivalence_margin=Fraction(1, 100),
        max_pd_drop=Fraction(1, 100),
        max_fa_increase=Fraction(1, 1),
        min_parameter_update_fraction=Fraction(1, 2),
        min_functional_change_fraction=Fraction(1, 2),
        min_objective_decrease_fraction=Fraction(1, 2),
        threshold_source="self_signed_test",
        frozen_gate_manifest_sha256="a" * 64,
    )
    receipt_value = select_stage1_candidates(
        _formal_stage1_records(1), _formal_diagnostics(), gate_spec
    )
    receipt = tmp_path / artifact.RECEIPT_FILENAME
    receipt.write_bytes(artifact.canonical_json_bytes(receipt_value))
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()

    assert not hasattr(runner, "ReceiptBinding")
    with pytest.raises(TypeError):
        runner.authorize_stage2(receipt, digest)  # type: ignore[call-arg]
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            [
                "authorize-stage2",
                "--stage1-receipt",
                str(receipt),
                "--expected-receipt-sha256",
                digest,
            ]
        )


@pytest.mark.parametrize(
    "member",
    (
        "config",
        "selector",
        "manifest",
        "completion",
        "gate",
        "records",
        "diagnostics",
        "receipt",
    ),
)
def test_every_bound_input_tamper_fails_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member: str,
) -> None:
    fixture = _make_formal_fixture(tmp_path)
    monkeypatch.setattr(
        runner, "_production_config_binding", lambda: fixture.binding
    )
    target: Path = getattr(fixture, member)
    target.write_bytes(target.read_bytes() + b" ")
    backend = _RecordingBackend(tmp_path / "must-not-exist")
    with pytest.raises(runner.ReceiptVerificationError):
        runner.launch_stage2(backend)
    _assert_zero_side_effects(backend)


def test_symlinked_aggregate_member_fails_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_formal_fixture(tmp_path)
    monkeypatch.setattr(
        runner, "_production_config_binding", lambda: fixture.binding
    )
    external = tmp_path / "outside-diagnostics.jsonl"
    external.write_bytes(fixture.diagnostics.read_bytes())
    fixture.diagnostics.unlink()
    fixture.diagnostics.symlink_to(external)
    backend = _RecordingBackend(tmp_path / "must-not-exist")
    with pytest.raises(runner.ReceiptVerificationError, match="symlink"):
        runner.launch_stage2(backend)
    _assert_zero_side_effects(backend)


@pytest.mark.parametrize("target_name", ("config", "selector", "records"))
def test_same_byte_path_replacement_during_verification_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    fixture = _make_formal_fixture(tmp_path)
    monkeypatch.setattr(
        runner, "_production_config_binding", lambda: fixture.binding
    )
    target: Path = getattr(fixture, target_name)
    original_selector = artifact.select_stage1_candidates

    def replace_after_recompute(*args, **kwargs):
        result = original_selector(*args, **kwargs)
        replacement = tmp_path / f"replacement-{target.name}"
        replacement.write_bytes(target.read_bytes())
        os.replace(replacement, target)
        return result

    monkeypatch.setattr(artifact, "select_stage1_candidates", replace_after_recompute)
    backend = _RecordingBackend(tmp_path / "must-not-exist")
    with pytest.raises(runner.ReceiptVerificationError, match="changed"):
        runner.launch_stage2(backend)
    _assert_zero_side_effects(backend)


@pytest.mark.parametrize(
    "fixture_options",
    (
        {"swap_diagnostic_candidates": True},
        {"noncanonical_receipt": True},
        {"bad_records_line_count": True},
    ),
)
def test_fully_resealed_but_semantically_invalid_chain_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_options: dict[str, bool],
) -> None:
    fixture = _make_formal_fixture(tmp_path, **fixture_options)
    monkeypatch.setattr(
        runner, "_production_config_binding", lambda: fixture.binding
    )
    backend = _RecordingBackend(tmp_path / "must-not-exist")
    with pytest.raises(runner.ReceiptVerificationError):
        runner.launch_stage2(backend)
    _assert_zero_side_effects(backend)


def test_runner_authorization_layer_is_cpu_only_and_cli_has_no_receipt_flags() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source
    assert "--stage1-receipt" not in source
    assert "--expected-receipt-sha256" not in source
