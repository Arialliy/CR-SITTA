from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts/run_nonadaptive_teacher_screen_v1.py"
SPEC = importlib.util.spec_from_file_location("p4_teacher_runner_for_tests", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _contract(tmp_path: Path):
    contract = runner.load_contract(runner.DEFAULT_CONFIG, verify_files=False)
    return replace(contract, output_root=tmp_path / "p4-output")


def _minimal_manifest(contract: Any, *, phase: str, dataset: str | None, formal: bool = True):
    return {
        "schema_version": 1,
        "artifact_type": f"test_{phase}",
        "protocol_id": runner.EXPECTED_PROTOCOL_ID,
        "phase": phase,
        "dataset": dataset,
        "formal": formal,
        "development_only": True,
        "paper_result": False,
        "p5_authorized": False,
        "config_sha256": contract.config_sha256,
        "lineage": runner._stage_transition_lineage(contract),
        "code_sha256": runner._capture_code_hashes(contract),
    }


def _rng_evidence() -> dict[str, str]:
    return {
        "python_random": "a" * 64,
        "numpy_random": "b" * 64,
        "torch_cpu": "c" * 64,
        "torch_current_cuda": "d" * 64,
    }


def _valid_formal_candidate_manifest(
    staging: Path,
    contract: Any,
    *,
    record_mutator: Any | None = None,
) -> dict[str, Any]:
    dataset = "IRSTD-1K"
    image_ids = [f"pilot_{index:03d}" for index in range(runner.IMAGE_COUNT)]
    fingerprint = "1" * 64
    conditions: list[dict[str, Any]] = []
    for condition_index, (corruption, severity) in enumerate(runner.CONDITIONS):
        key = runner._condition_key(corruption, severity)
        relative = f"conditions/{key}/per_image.jsonl"
        records: list[dict[str, Any]] = []
        for image_index, image_id in enumerate(image_ids):
            record = {
                "index": image_index,
                "image_id": image_id,
                "dataset": dataset,
                "corruption": corruption,
                "severity": severity,
                "seed": 42,
                "model_runtime_fingerprint_sha256_before": fingerprint,
                "model_runtime_fingerprint_sha256_after": fingerprint,
                "model_runtime_fingerprint_unchanged": True,
                "rng_state_sha256_before": _rng_evidence(),
                "rng_state_sha256_after": _rng_evidence(),
                "rng_state_unchanged": True,
                "strict_probability_threshold": 0.5,
                "threshold_rule": "strict_greater_than",
            }
            if record_mutator is not None:
                record_mutator(condition_index, image_index, record)
            records.append(record)
        runner._write_jsonl(staging / relative, records)
        conditions.append(
            {
                "condition": key,
                "corruption": corruption,
                "severity": severity,
                "image_count": runner.IMAGE_COUNT,
                "input_tensor_sequence_sha256": "0" * 64,
                "per_image_path": relative,
            }
        )
    expected_checks = len(runner.CONDITIONS) * runner.IMAGE_COUNT
    execution = {
        "seed": 42,
        "requires_grad": False,
        "optimizer_present": False,
        "model_fully_eval": True,
        "state_sha256_before": "e" * 64,
        "state_sha256_after": "e" * 64,
        "state_bit_exact": True,
        "per_image_lightweight_state_gate": {
            "checks": expected_checks,
            "expected_checks": expected_checks,
            "baseline_fingerprint_sha256": fingerprint,
            "fingerprint_fields": list(runner.MODEL_RUNTIME_FINGERPRINT_FIELDS),
            "module_count": 2,
            "batchnorm_module_count": 1,
            "parameter_count": 2,
            "buffer_count": 3,
            "parameter_versions_unchanged": True,
            "buffer_versions_unchanged": True,
            "module_training_unchanged": True,
            "module_type_and_training_unchanged": True,
            "batchnorm_runtime_attributes_unchanged": True,
            "parameter_requires_grad_unchanged": True,
            "parameter_grad_is_none_unchanged": True,
            "passed": True,
        },
        "per_image_rng_state_gate": {
            "checks": expected_checks,
            "expected_checks": expected_checks,
            "streams": list(runner.RNG_STREAMS),
            "evidence_location": "conditions/*/per_image.jsonl",
            "passed": True,
        },
        "rng_state_sha256_before": _rng_evidence(),
        "rng_state_sha256_after": _rng_evidence(),
        "rng_state_unchanged": True,
        "strict_threshold": "probability > 0.5",
        "runtime_environment": {
            "python": "3.10.0",
            "torch": "test",
            "cuda_runtime": "test",
            "cudnn": 1,
            "device": "cuda:0",
            "gpu_name": "mock-gpu",
            "sfs_extension_module": "MultiScaleDeformableAttention",
            "sfs_extension_file": "/mock/MultiScaleDeformableAttention.so",
            "sfs_extension_sha256": "f" * 64,
        },
    }
    return {
        **_minimal_manifest(
            contract, phase="candidate", dataset=dataset, formal=True
        ),
        "condition_count": len(runner.CONDITIONS),
        "image_count_per_condition": runner.IMAGE_COUNT,
        "candidate_count": len(runner.CANDIDATE_IDS),
        "candidate_ids": list(contract.candidate_ids),
        "base_view_names": list(runner.BASE_VIEW_NAMES),
        "image_ids": image_ids,
        "conditions": conditions,
        "method_boundary": {
            "loader": "SourceCalibrationMethodInputDatasetV2",
            "fields": sorted(runner.METHOD_FIELDS),
            "target_loader_calls": 0,
            "method_label_accesses": 0,
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
        },
        "execution": execution,
    }


def _publish_test_formal_candidate(
    tmp_path: Path,
    contract: Any,
    manifest: dict[str, Any],
    *,
    name: str,
) -> Path:
    staging = tmp_path / f".{name}.staging"
    assert staging.is_dir()
    destination = tmp_path / name
    runner._publish_artifact(staging, destination, manifest=manifest)
    return destination


def test_frozen_config_validates_and_never_authorizes_p5() -> None:
    contract = runner.load_contract(runner.DEFAULT_CONFIG, verify_files=False)
    assert contract.conditions == runner.CONDITIONS
    assert contract.candidate_ids == runner.CANDIDATE_IDS
    assert contract.raw["scope"]["use_validation_payload"] is False
    assert contract.raw["scope"]["use_test_payload"] is False
    assert contract.raw["scope"]["paper_result"] is False
    assert contract.raw["scope"]["p5_authorized"] is False
    assert contract.raw["views"]["rotation"] == {
        "enabled": False,
        "fail_closed": True,
    }
    assert contract.raw["views"]["scale"] == {
        "enabled": False,
        "fail_closed": True,
    }
    assert tuple(contract.raw["views"]["tile5"]["ordered_views"]) == (
        "identity",
        "hflip",
        "vflip",
        "hvflip",
        "tile_reconstruction",
    )
    assert contract.raw["frozen_inputs"]["v4_plan"]["sha256"] == (
        "8fd17c76f3ccdd86d2221337f99f56b26de14dc241fff205676aa7ab3ba4e7b1"
    )
    assert "environment.linux-64.explicit.txt" in contract.raw["critical_code_paths"]
    assert (
        "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py"
        in contract.raw["critical_code_paths"]
    )
    assert (
        "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py"
        in contract.raw["critical_code_paths"]
    )
    assert contract.raw["stage_transition"] == {
        "predecessor_stage": "P3_formal_tent_failure_diagnostics_stage_a",
        "predecessor_formal_protocol_complete": True,
        "predecessor_scientific_status": "scientific_no_eligible",
        "p4_nonadaptive_teacher_screen_authorized": True,
        "authorized_method_class": "nonadaptive_no_parameter_update",
        "parameter_update_authorized": False,
        "stage2_authorized": False,
        "p5_authorized": False,
    }
    assert contract.raw["aggregation"]["disagreement_weight_formula"] == (
        "exp(-(p_m-minus-view_mean)^2/tau)"
    )
    assert contract.raw["aggregation"]["source_anchor_inner_aggregation"] == "mean"


def test_config_rejects_enabling_an_unvalidated_rotation(tmp_path: Path) -> None:
    payload = yaml.safe_load(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["views"]["rotation"]["enabled"] = True
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(runner.P4ProtocolError, match="rotation.enabled"):
        runner.load_contract(config, verify_files=False)


def test_candidate_phase_does_not_reach_outer_target_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path)
    target_calls = 0

    def forbidden_target_loader(*args: Any, **kwargs: Any):
        nonlocal target_calls
        target_calls += 1
        raise AssertionError("candidate phase attempted to load an outer target")

    def fake_candidate_payload(
        staging: Path,
        *,
        contract: Any,
        dataset: str,
        device_name: str,
        image_limit: int,
        formal: bool,
    ):
        del device_name
        (staging / "payload.bin").write_bytes(b"label-free")
        return {
            **_minimal_manifest(
                contract, phase="candidate", dataset=dataset, formal=formal
            ),
            "image_count_per_condition": image_limit,
            "method_boundary": {
                "target_loader_calls": 0,
                "method_label_accesses": 0,
                "validation_payload_opens": 0,
                "test_payload_opens": 0,
            },
        }

    monkeypatch.setattr(runner, "_load_outer_targets", forbidden_target_loader)
    monkeypatch.setattr(runner, "_execute_candidate_payload", fake_candidate_payload)
    result = runner.run_candidate(
        contract,
        dataset="IRSTD-1K",
        device_name="cuda:does-not-exist",
        max_images=1,
    )
    assert result["status"] == "published"
    assert result["formal"] is False
    assert target_calls == 0
    manifest = runner.verify_artifact(
        result["path"],
        contract=contract,
        phase="candidate",
        dataset="IRSTD-1K",
    )
    assert manifest["method_boundary"]["target_loader_calls"] == 0


def test_candidate_aggregation_maps_frozen_names_and_explicit_tau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import tta.proposals.source_multiview_teacher as teacher_module

    contract = runner.load_contract(runner.DEFAULT_CONFIG, verify_files=False)
    calls: list[tuple[str, int, float, float, str]] = []

    def fake_aggregate(
        probabilities: Any,
        method: str,
        *,
        trim_each_side: int,
        tau: float,
        beta: float = 0.5,
        source_anchor_aggregate: str = "mean",
    ):
        calls.append(
            (method, trim_each_side, tau, beta, source_anchor_aggregate)
        )
        return probabilities.mean(dim=0).detach()

    monkeypatch.setattr(
        teacher_module, "aggregate_aligned_probabilities", fake_aggregate
    )
    base = torch.zeros((5, 1, 1, 256, 256), dtype=torch.float32)
    source, uncertainty, candidates = runner._aggregate_candidates(base, contract)
    assert source.shape == (1, 1, 256, 256)
    assert uncertainty.shape == (2, 1, 1, 256, 256)
    assert candidates.shape == (10, 1, 1, 256, 256)
    assert [value[0] for value in calls] == [
        "mean",
        "trimmed_mean",
        "disagreement_weighted_mean",
        "source_anchor",
        "source_anchor",
        "mean",
        "trimmed_mean",
        "disagreement_weighted_mean",
        "source_anchor",
        "source_anchor",
    ]
    assert all(value[1] == 1 and value[2] == 0.01 for value in calls)
    assert [value[3] for value in calls if value[0] == "source_anchor"] == [
        0.25,
        0.5,
        0.25,
        0.5,
    ]


def test_runtime_receipt_seals_loaded_sfs_binary_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension_path = tmp_path / "MultiScaleDeformableAttention.so"
    extension_path.write_bytes(b"mock-extension")
    extension = SimpleNamespace(__file__=str(extension_path))
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: extension)
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: None)),
        cuda=SimpleNamespace(
            get_device_name=lambda _device: pytest.fail("CPU receipt queried a GPU")
        ),
    )
    receipt = runner._runtime_environment_receipt(
        fake_torch, SimpleNamespace(type="cpu")
    )
    assert receipt["torch"] == "test-torch"
    assert receipt["cuda_runtime"] is None
    assert receipt["cudnn"] is None
    assert receipt["gpu_name"] is None
    assert receipt["sfs_extension_file"] == str(extension_path)
    assert receipt["sfs_extension_sha256"] == runner.sha256_file(extension_path)


def test_lightweight_per_image_model_gate_detects_each_state_class() -> None:
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.register_buffer("running", torch.zeros(1))
            self.bn = torch.nn.BatchNorm2d(1, eps=1.0e-5, momentum=0.1).eval()

    def assert_detected(mutate: Any, expected_field: str) -> None:
        model = TinyModel().eval()
        snapshot = runner._capture_lightweight_model_gate(model)
        runner._assert_lightweight_model_gate_unchanged(model, snapshot)
        mutate(model)
        with pytest.raises(runner.P4ProtocolError, match=expected_field):
            runner._assert_lightweight_model_gate_unchanged(model, snapshot)

    def mutate_parameter(model: Any) -> None:
        with torch.no_grad():
            model.weight.add_(1)

    def mutate_buffer(model: Any) -> None:
        model.running.add_(1)

    assert_detected(mutate_parameter, "parameter_versions")
    assert_detected(mutate_buffer, "buffer_versions")
    assert_detected(lambda model: model.train(), "module_training")
    assert_detected(
        lambda model: setattr(model.bn, "eps", 2.0e-5),
        "batchnorm_runtime",
    )
    assert_detected(
        lambda model: model.weight.requires_grad_(True),
        "parameter_requires_grad",
    )
    assert_detected(
        lambda model: setattr(model.weight, "grad", torch.zeros_like(model.weight)),
        "parameter_grad_is_none",
    )


def test_per_image_runtime_gate_detects_actual_rng_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    model = torch.nn.BatchNorm2d(1).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_gate = runner._capture_lightweight_model_gate(model)
    contract = runner.load_contract(runner.DEFAULT_CONFIG, verify_files=False)
    image = torch.zeros((1, 3, 256, 256), dtype=torch.float32)

    def stochastic_teacher(*args: Any, **kwargs: Any):
        del args, kwargs
        torch.rand(1)
        value = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        return value, value, value, value

    monkeypatch.setattr(runner, "_build_teacher_outputs", stochastic_teacher)
    original_rng_state = torch.get_rng_state()
    try:
        with pytest.raises(runner.P4ProtocolError, match="torch_cpu"):
            runner._build_teacher_outputs_with_per_image_runtime_gates(
                object(), image, model, model_gate, contract, torch
            )
    finally:
        torch.set_rng_state(original_rng_state)


def test_per_image_runtime_gate_detects_bn_attribute_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    model = torch.nn.BatchNorm2d(1).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_gate = runner._capture_lightweight_model_gate(model)
    contract = runner.load_contract(runner.DEFAULT_CONFIG, verify_files=False)
    image = torch.zeros((1, 3, 256, 256), dtype=torch.float32)

    def mutating_teacher(*args: Any, **kwargs: Any):
        del args, kwargs
        model.eps = 0.25
        value = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        return value, value, value, value

    monkeypatch.setattr(runner, "_build_teacher_outputs", mutating_teacher)
    with pytest.raises(runner.P4ProtocolError, match="batchnorm_runtime"):
        runner._build_teacher_outputs_with_per_image_runtime_gates(
            object(), image, model, model_gate, contract, torch
        )


def test_rng_state_gate_is_hash_only_and_fail_closed() -> None:
    import torch

    captured = runner._capture_rng_state_sha256(torch, torch.device("cpu"))
    assert set(captured) == {
        "python_random",
        "numpy_random",
        "torch_cpu",
        "torch_current_cuda",
    }
    assert captured["torch_current_cuda"] is None
    assert all(
        value is None or (isinstance(value, str) and len(value) == 64)
        for value in captured.values()
    )
    before = {
        "python_random": "a",
        "numpy_random": "b",
        "torch_cpu": "c",
        "torch_current_cuda": None,
    }
    runner._assert_rng_state_unchanged(before, dict(before))
    changed = dict(before)
    changed["torch_cpu"] = "different"
    with pytest.raises(runner.P4ProtocolError, match="torch_cpu"):
        runner._assert_rng_state_unchanged(before, changed)


def test_incomplete_canonical_candidate_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path)
    destination = runner._artifact_destination(contract, "candidate", "IRSTD-1K")
    destination.mkdir(parents=True)
    (destination / "partial.bin").write_bytes(b"partial")
    executed = False

    def must_not_execute(*args: Any, **kwargs: Any):
        nonlocal executed
        executed = True
        raise AssertionError("candidate payload must not start")

    monkeypatch.setattr(runner, "_execute_candidate_payload", must_not_execute)
    with pytest.raises(runner.ExistingArtifactError, match="refusing to overwrite"):
        runner.run_candidate(
            contract,
            dataset="IRSTD-1K",
            device_name="cuda:0",
        )
    assert executed is False
    assert (destination / "partial.bin").read_bytes() == b"partial"


def test_verify_artifact_rejects_code_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path)
    destination = tmp_path / "candidate-artifact"
    staging = runner._new_staging(destination)
    (staging / "payload.bin").write_bytes(b"immutable")
    runner._publish_artifact(
        staging,
        destination,
        manifest=_minimal_manifest(
            contract, phase="candidate", dataset="IRSTD-1K"
        ),
    )
    recorded = runner._capture_code_hashes(contract)
    drifted = dict(recorded)
    first_path = next(iter(drifted))
    drifted[first_path] = "0" * 64
    monkeypatch.setattr(runner, "_capture_code_hashes", lambda _contract: drifted)
    with pytest.raises(runner.P4ProtocolError, match="code hashes"):
        runner.verify_artifact(
            destination,
            contract=contract,
            phase="candidate",
            dataset="IRSTD-1K",
        )


def test_formal_candidate_semantic_verify_accepts_complete_evidence(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    staging = tmp_path / ".valid-formal.staging"
    manifest = _valid_formal_candidate_manifest(staging, contract)
    destination = _publish_test_formal_candidate(
        tmp_path, contract, manifest, name="valid-formal"
    )
    verified = runner.verify_artifact(
        destination,
        contract=contract,
        phase="candidate",
        dataset="IRSTD-1K",
    )
    assert verified["formal"] is True


def test_formal_candidate_missing_execution_is_rejected(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    staging = tmp_path / ".missing-execution.staging"
    manifest = _valid_formal_candidate_manifest(staging, contract)
    del manifest["execution"]
    destination = _publish_test_formal_candidate(
        tmp_path, contract, manifest, name="missing-execution"
    )
    with pytest.raises(runner.P4ProtocolError, match="execution"):
        runner.verify_artifact(
            destination,
            contract=contract,
            phase="candidate",
            dataset="IRSTD-1K",
        )


def test_formal_candidate_tampered_gate_count_is_rejected(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    staging = tmp_path / ".bad-gate-count.staging"
    manifest = _valid_formal_candidate_manifest(staging, contract)
    manifest["execution"]["per_image_rng_state_gate"]["checks"] = 831
    destination = _publish_test_formal_candidate(
        tmp_path, contract, manifest, name="bad-gate-count"
    )
    with pytest.raises(runner.P4ProtocolError, match="RNG checks"):
        runner.verify_artifact(
            destination,
            contract=contract,
            phase="candidate",
            dataset="IRSTD-1K",
        )


def test_formal_candidate_per_image_rng_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    staging = tmp_path / ".bad-rng.staging"

    def change_rng(
        condition_index: int, image_index: int, record: dict[str, Any]
    ) -> None:
        if condition_index == 0 and image_index == 0:
            record["rng_state_sha256_after"]["torch_cpu"] = "9" * 64

    manifest = _valid_formal_candidate_manifest(
        staging, contract, record_mutator=change_rng
    )
    destination = _publish_test_formal_candidate(
        tmp_path, contract, manifest, name="bad-rng"
    )
    with pytest.raises(runner.P4ProtocolError, match="RNG before/after"):
        runner.verify_artifact(
            destination,
            contract=contract,
            phase="candidate",
            dataset="IRSTD-1K",
        )


def test_formal_candidate_missing_bn_state_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    staging = tmp_path / ".missing-bn-evidence.staging"
    manifest = _valid_formal_candidate_manifest(staging, contract)
    model_gate = manifest["execution"]["per_image_lightweight_state_gate"]
    model_gate["fingerprint_fields"].remove("batchnorm_runtime")
    destination = _publish_test_formal_candidate(
        tmp_path, contract, manifest, name="missing-bn-evidence"
    )
    with pytest.raises(runner.P4ProtocolError, match="fingerprint fields"):
        runner.verify_artifact(
            destination,
            contract=contract,
            phase="candidate",
            dataset="IRSTD-1K",
        )


def test_outer_target_loader_is_unreachable_without_complete_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path)
    candidate = runner._artifact_destination(contract, "candidate", "IRSTD-1K")
    candidate.mkdir(parents=True)
    (candidate / "partial.bin").write_bytes(b"partial")
    target_calls = 0

    def forbidden_target_loader(*args: Any, **kwargs: Any):
        nonlocal target_calls
        target_calls += 1
        raise AssertionError("outer target loader reached too early")

    monkeypatch.setattr(runner, "_load_outer_targets", forbidden_target_loader)
    with pytest.raises(runner.ExistingArtifactError):
        runner.run_outer(contract, dataset="IRSTD-1K")
    assert target_calls == 0
    assert not runner._artifact_destination(contract, "outer", "IRSTD-1K").exists()


def test_aggregate_rejects_missing_outer_grid_before_creating_output(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    with pytest.raises(runner.ExistingArtifactError):
        runner.run_aggregate(contract)
    assert not runner._artifact_destination(contract, "aggregate").exists()


def test_aggregate_grid_rejects_missing_and_duplicate_cells(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    records = [
        {
            "candidate_id": candidate,
            "dataset": dataset,
            "condition": runner._condition_key(*condition),
        }
        for candidate in contract.candidate_ids
        for dataset in runner.DATASETS
        for condition in contract.conditions
    ]
    runner._validate_aggregate_grid(records, contract)
    with pytest.raises(runner.P4ProtocolError, match="incomplete or duplicated"):
        runner._validate_aggregate_grid(records[:-1], contract)
    duplicated = records[:-1] + [records[0], records[0]]
    with pytest.raises(runner.P4ProtocolError, match="incomplete or duplicated"):
        runner._validate_aggregate_grid(duplicated, contract)


def test_gate_config_matches_all_frozen_thresholds() -> None:
    contract = runner.load_contract(runner.DEFAULT_CONFIG, verify_files=False)
    gate = runner._build_gate_config(contract.raw)
    assert str(float(gate.nonclean_macro_delta_iou_epsilon)) == "0.001"
    assert gate.minimum_positive_nonclean_datasets == 2
    assert float(gate.worst_nonclean_dataset_delta_iou_minimum) == -0.002
    assert float(gate.fa_absolute_allowance_per_million) == 10.0
    assert float(gate.fa_source_multiplier) == 0.25
    assert float(gate.foreground_fraction_delta_maximum) == 0.001
    assert float(gate.foreground_fraction_source_multiplier) == 1.2
    assert float(gate.foreground_fraction_epsilon) == 1.0e-6
    assert gate.require_integer_counts is True
    assert tuple(contract.raw["science_gate"]["ranking_tie_break_order"]) == (
        "higher_nonclean_macro_delta_iou",
        "higher_overall_macro_delta_iou",
        "higher_worst_dataset_delta_iou",
        "lower_nonclean_fa_delta",
        "lexical_candidate_id",
    )


@pytest.mark.parametrize(
    ("evidence_kind", "conservation", "message"),
    (
        ("metrics", False, "count-based evidence"),
        ("counts", False, "integer-count conservation"),
    ),
)
def test_formal_aggregate_rejects_noninteger_or_unconserved_gate_evidence(
    evidence_kind: str,
    conservation: bool,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = runner.load_contract(runner.DEFAULT_CONFIG, verify_files=False)

    def fake_gate(records: Any, candidate_ids: Any, gate: Any):
        del records, candidate_ids
        assert gate.require_integer_counts is True
        receipt = {
            "development_only": True,
            "paper_result": False,
            "p5_authorized": False,
            "evidence_kind": evidence_kind,
            "integer_count_conservation_verified": conservation,
            "gate": {"require_integer_counts": True},
        }
        return SimpleNamespace(
            evidence_kind=evidence_kind,
            integer_count_conservation_verified=conservation,
            to_receipt=lambda: receipt,
        )

    import analysis.nonadaptive_teacher_gate as gate_module

    monkeypatch.setattr(
        gate_module, "evaluate_nonadaptive_teacher_gate", fake_gate
    )
    with pytest.raises(runner.P4ProtocolError, match=message):
        runner._evaluate_gate([], contract)


def test_cli_surface_contains_all_required_commands() -> None:
    parser = runner.build_parser()
    help_text = parser.format_help()
    for command in ("validate", "candidate", "outer", "aggregate", "verify"):
        assert command in help_text
