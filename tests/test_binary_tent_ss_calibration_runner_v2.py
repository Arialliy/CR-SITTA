from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_binary_tent_ss_calibration_v2 as runner
from tta.binary_tent_ss_calibration_selector_v2 import (
    ALL_CANDIDATES,
    CONDITIONS,
    DATASETS,
    REQUIRED_HARD_GATES,
    REQUIRED_PROTOCOL_AUDIT,
    SS_BN_PROTOCOL,
    Candidate,
    select_final_candidate,
    select_stage1_top3,
)


SCIENTIFIC = ROOT / "configs" / "binary_tent_ss_calibration_v2.yaml"
EXECUTION = ROOT / "configs" / "binary_tent_ss_calibration_execution_v2.yaml"
EXPECTED_CORE_SHA = "a3175ade39e656d04778ff804b8d1d106c01ea44f8390b6c65a2cb8ed7eb4d20"
EXPECTED_CACHE_SHA = "e225311cce252125eaf3a2b47eeeea4cde60d0363c8c71f48494d70e41c2aa3e"
EXPECTED_INVENTORY_SEAL = "99b487241078828129e1566817290f53c186d5988e42c560a719c9701fd7aa0e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _record(
    *,
    stage: int,
    process_id: str,
    candidate: Candidate,
    dataset: str,
    corruption: str,
    severity: int,
    improvement: int,
) -> dict:
    targets = {"IRSTD-1K": 11, "NUAA-SIRST": 13, "NUDT-SIRST": 17}[dataset]
    pre = {
        "intersection_pixels": 100,
        "union_pixels": 1000,
        "false_alarm_pixels": 20,
        "total_image_pixels": 64 * 256 * 256,
        "detected_targets": 5,
        "total_targets": targets,
    }
    post = {
        **pre,
        "intersection_pixels": 100 + improvement,
        "false_alarm_pixels": 20,
        "detected_targets": min(targets, 5 + improvement // 3),
    }
    return {
        "stage": stage,
        "process_id": process_id,
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
        "protocol_audit": {key: True for key in REQUIRED_PROTOCOL_AUDIT},
        "endpoints": {"tent_pre": pre, "tent_post": post},
    }


def _run_records(
    *, stage: int, process_id: str, candidate: Candidate, improvement: int
) -> list[dict]:
    return [
        _record(
            stage=stage,
            process_id=process_id,
            candidate=candidate,
            dataset=dataset,
            corruption=corruption,
            severity=severity,
            improvement=improvement,
        )
        for dataset in DATASETS
        for corruption, severity in CONDITIONS
    ]


@pytest.fixture(scope="module")
def synthetic_selection() -> tuple[list[dict], list[dict], dict, dict]:
    stage1: list[dict] = []
    for index, candidate in enumerate(ALL_CANDIDATES, start=1):
        stage1.extend(
            _run_records(
                stage=1,
                process_id=f"stage1-{index}",
                candidate=candidate,
                improvement=index,
            )
        )
    stage1_receipt = select_stage1_top3(stage1)
    top3 = [
        Candidate.from_values(value["optimizer"], value["learning_rate"])
        for value in stage1_receipt["top3"]
    ]
    stage2: list[dict] = []
    for slot in (1, 2):
        for rank, candidate in enumerate(top3):
            stage2.extend(
                _run_records(
                    stage=2,
                    process_id=f"stage2-slot-{slot}",
                    candidate=candidate,
                    improvement=20 - rank,
                )
            )
    final_receipt = select_final_candidate(stage1, stage2)
    return stage1, stage2, stage1_receipt, final_receipt


@pytest.fixture(scope="module")
def captured_runtime() -> tuple[runner.RuntimeSeal, dict]:
    """One real opaque-byte validation; deliberately makes np.load fatal."""

    contract = runner.load_contract(EXECUTION)
    output_before = contract.output_root.exists()
    cuda_before = runner.torch.cuda.is_initialized()
    original_load = runner.np.load

    def forbidden_np_load(*args, **kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("validate/capture attempted np.load")

    runner.np.load = forbidden_np_load
    try:
        seal, caches = runner.capture_runtime_seal(contract)
    finally:
        runner.np.load = original_load
    assert runner.torch.cuda.is_initialized() is cuda_before
    assert contract.output_root.exists() is output_before
    return seal, dict(caches)


def test_protocol_hash_chain_and_external_seals() -> None:
    execution = _yaml(EXECUTION)
    assert execution["scientific_protocol"]["sha256"] == _sha(SCIENTIFIC)
    assert execution["cache_protocol"]["sha256"] == EXPECTED_CACHE_SHA
    assert execution["cache_protocol"]["inventory_go_seal_sha256"] == EXPECTED_INVENTORY_SEAL
    assert execution["hardened_safety_core"]["sha256"] == EXPECTED_CORE_SHA
    assert _sha(ROOT / "run_binary_tent_source_calibration.py") == EXPECTED_CORE_SHA


def test_hardened_core_api_is_exact_allow_list() -> None:
    execution = _yaml(EXECUTION)
    assert tuple(execution["hardened_safety_core"]["api_allow_list"]) == runner.CORE_API_ALLOW_LIST
    assert all(hasattr(runner.core, name) for name in runner.CORE_API_ALLOW_LIST)
    assert "_build_fast_runner" not in runner.CORE_API_ALLOW_LIST
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "monkeypatch" in source
    assert "setattr(core" not in source
    assert "core.__dict__" not in source
    tree = ast.parse(source)
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "core"
    }
    assert used <= set(runner.CORE_API_ALLOW_LIST)


def test_contract_loads_with_exact_ss_output_layout() -> None:
    contract = runner.load_contract(EXECUTION)
    assert contract.output_root == ROOT / "results/binary_tent/ss_calibration_v2"
    assert contract.publication_work_root == ROOT / "results/binary_tent/.ss_calibration_v2_publication_work"
    assert not contract.publication_work_root.is_relative_to(contract.output_root)
    assert runner._stage1_receipt_path(contract) == (
        contract.output_root / "stage1/aggregate/stage1_ss_top3_receipt.json"
    )
    assert (
        contract.execution["outputs"]["update_activity_diagnostics_filename"]
        == runner.STRENGTH_DIAGNOSTICS_FILENAME
    )
    assert contract.execution["runtime_seal"]["zero_parameter_update_policy_bound"] is True
    assert (
        contract.execution["runtime_seal"][
            "update_activity_diagnostics_bound_and_ranking_excluded"
        ]
        is True
    )


def test_contract_rejects_execution_bs_or_receipt_role_tamper(tmp_path: Path) -> None:
    execution = _yaml(EXECUTION)
    execution["execution"]["selection_bn_protocol"] = "single_image_spatial_batch_stats"
    tampered = tmp_path / "execution-bs.yaml"
    tampered.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")
    with pytest.raises(runner.CalibrationExecutionError, match="selection_bn_protocol"):
        runner.load_contract(tampered)

    execution = _yaml(EXECUTION)
    execution["runtime_seal"]["stage2_top3_receipt_bound_with_canonical_role"] = "wrong"
    tampered = tmp_path / "execution-receipt.yaml"
    tampered.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")
    with pytest.raises(runner.CalibrationExecutionError, match="canonical_role"):
        runner.load_contract(tampered)


def test_scientific_protocol_is_ss_only_and_does_not_create_validation() -> None:
    scientific = _yaml(SCIENTIFIC)
    scope = scientific["scope"]
    method = scientific["method"]
    assert scope["official_splits_used"] == ["train", "test"]
    assert scope["no_validation_split_created"] is True
    assert scope["use_test_images"] is False
    assert scope["use_test_labels"] is False
    assert method["selection_bn_protocol"] == {
        "display_name": "SS",
        "id": SS_BN_PROTOCOL,
    }
    assert method["diagnostic_detail"] == "global"
    assert _yaml(EXECUTION)["execution"]["diagnostic_detail"] == "global"
    assert method["require_nonzero_parameter_update"] is False
    assert _yaml(EXECUTION)["execution"]["require_nonzero_parameter_update"] is False
    assert method["update_activity_diagnostics"] == runner._update_activity_disclosure()
    assert (
        _yaml(EXECUTION)["execution"]["update_activity_diagnostics"]
        == runner._update_activity_disclosure()
    )
    assert method["BS_excluded_from_selection"] is True
    assert scientific["selection"]["BS_selection_episodes"] == 0
    assert "source_val" not in SCIENTIFIC.read_text(encoding="utf-8")


def test_runtime_seal_binds_v2_runner_and_immutable_v1_dependencies() -> None:
    paths = _yaml(EXECUTION)["runtime_seal"]["critical_code_paths"]
    assert "tta/binary_tent_fast_runner_v2.py" in paths
    assert "tta/binary_tent_fast_runner.py" in paths
    assert "tta/binary_tent.py" in paths
    assert _sha(ROOT / "tta/binary_tent_fast_runner.py") == (
        "4ab23f78df035fb250183cadc5d493a3d10dd24f4f887192137e37a3dfd103d8"
    )
    assert _sha(ROOT / "tta/binary_tent.py") == (
        "0267d96e0fbb41f3429568b0b65b970c9dac35954c98035bc67714467f0eb847"
    )


def test_runtime_seal_refuses_config_that_omits_v2_runner(tmp_path: Path) -> None:
    execution = _yaml(EXECUTION)
    paths = execution["runtime_seal"]["critical_code_paths"]
    paths.remove("tta/binary_tent_fast_runner_v2.py")
    tampered = tmp_path / "execution-missing-v2-runner.yaml"
    tampered.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        runner.CalibrationExecutionError,
        match="critical code path set/order",
    ):
        runner.load_contract(tampered)


def test_independent_pilot_v2_disclosure_is_not_old_round02() -> None:
    limitation = _yaml(SCIENTIFIC)["inherited_source_limitation"]
    assert limitation["same_frozen_train_side_pilot_v2_subset_reused_across_candidates"] is True
    assert limitation["old_round_02_severity_pilot_subset_used"] is False
    assert limitation["best_pd_tuning_episodes"] == 0


def test_exact_stage_counts_in_both_configs() -> None:
    scientific = _yaml(SCIENTIFIC)["selection"]
    execution = _yaml(EXECUTION)["launcher"]
    assert scientific["stage_1"]["cells_per_run"] == 39
    assert scientific["stage_1"]["cell_records"] == 390
    assert scientific["stage_1"]["episodes"] == 24960
    assert scientific["stage_2"]["cells_per_process"] == 117
    assert scientific["stage_2"]["cell_records"] == 234
    assert scientific["stage_2"]["episodes_per_process"] == 7488
    assert scientific["stage_2"]["episodes"] == 14976
    assert scientific["total_episodes"] == 39936
    assert execution["stage1"]["episodes_per_process"] == 2496
    assert execution["stage2"]["episodes_per_process"] == 7488


def test_actual_cache_metadata_anchors_cover_all_three_datasets() -> None:
    artifacts = _yaml(EXECUTION)["cache_protocol"]["artifacts"]
    assert tuple(artifacts) == DATASETS
    for dataset, anchors in artifacts.items():
        root = ROOT / "results/binary_tent/ss_calibration_cache_v2" / dataset
        assert _sha(root / "manifest.json") == anchors["manifest_sha256"]
        assert _sha(root / "method_input_manifest.json") == anchors["method_input_manifest_sha256"]
        assert _sha(root / "COMPLETE.json") == anchors["complete_sha256"]
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["cache_content_sha256"] == anchors["cache_content_sha256"]
        assert manifest["ordered_ids_sha256"] == anchors["ordered_ids_sha256"]
        assert manifest["targets"]["file_sha256"] == anchors["target_file_sha256"]
        assert manifest["targets"]["tensor_sequence_sha256"] == anchors["target_tensor_sequence_sha256"]


def test_real_runtime_capture_binds_payloads_and_both_checkpoint_roles(captured_runtime) -> None:
    seal, caches = captured_runtime
    roles = [binding.role for binding in seal.bindings]
    assert len(caches) == 3
    assert sum(role.startswith("cache_condition_payload:") for role in roles) == 39
    assert sum(role.startswith("cache_target_payload:") for role in roles) == 3
    assert sum(role.startswith("checkpoint_best_miou:") for role in roles) == 3
    assert sum(role.startswith("checkpoint_best_pd_zero_tuning_application:") for role in roles) == 3
    assert seal.cache_lineage["selection_bn_protocol"] == SS_BN_PROTOCOL
    assert seal.cache_lineage["BS_excluded_from_selection"] is True
    assert seal.cache_lineage["diagnostic_detail"] == "global"
    assert seal.cache_lineage["require_nonzero_parameter_update"] is False
    assert seal.cache_lineage["zero_parameter_update_policy"] == runner.ZERO_UPDATE_POLICY
    assert seal.cache_lineage["update_activity"] == runner._update_activity_disclosure()
    assert seal.cache_lineage["best_pd_tuning_episodes"] == 0


def test_validate_only_reports_no_tensor_model_cuda_or_output(monkeypatch, captured_runtime) -> None:
    seal, caches = captured_runtime
    contract = runner.load_contract(EXECUTION)
    before = contract.output_root.exists()
    cuda_before = runner.torch.cuda.is_initialized()
    monkeypatch.setattr(runner, "capture_runtime_seal", lambda ignored: (seal, caches))
    monkeypatch.setattr(
        runner.np,
        "load",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("np.load called")),
    )
    monkeypatch.setattr(
        runner,
        "_build_fast_runner_v2",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("model built")),
    )
    result = runner.validate_only(argparse.Namespace(execution_config=EXECUTION))
    assert result["valid"] is True
    assert result["numpy_load_calls"] == 0
    assert result["model_constructions"] == 0
    assert result["require_nonzero_parameter_update"] is False
    assert result["zero_parameter_update_policy"] == runner.ZERO_UPDATE_POLICY
    assert result["update_activity"] == runner._update_activity_disclosure()
    assert result["cuda_initialized_before"] is cuda_before
    assert result["cuda_initialized_after"] is cuda_before
    assert result["formal_output_writes"] == 0
    assert result["validation_split_created"] is False
    assert contract.output_root.exists() is before


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,2", ("1", "2")), ("0,9", ("0", "9"))],
)
def test_gpu_ids_accept_only_two_canonical_unique_ordinals(raw: str, expected: tuple[str, str]) -> None:
    assert runner._gpu_ids(raw) == expected


@pytest.mark.parametrize("raw", ["1,01", "01,2", "1,1", "1", "1, 2", "+1,2"])
def test_gpu_ids_reject_aliases_duplicates_and_noncanonical_text(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        runner._gpu_ids(raw)


def test_worker_stage1_fails_before_contract_without_inherited_lease(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.core,
        "_verify_inherited_gpu_lease",
        lambda required: (_ for _ in ()).throw(runner.CalibrationExecutionError("no lease")),
    )
    monkeypatch.setattr(
        runner,
        "load_contract",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("lease bypass")),
    )
    args = argparse.Namespace(
        execution_config=EXECUTION,
        optimizer="Adam",
        learning_rate="1e-5",
        process_id="p",
        device="cuda:0",
    )
    with pytest.raises(runner.CalibrationExecutionError, match="no lease"):
        runner.run_stage1_worker(args)


def test_worker_stage2_is_permanently_blocked_before_lease_or_contract(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        runner.core,
        "_verify_inherited_gpu_lease",
        lambda required: calls.append("lease"),
    )
    monkeypatch.setattr(
        runner,
        "load_contract",
        lambda *a, **k: calls.append("contract"),
    )
    args = argparse.Namespace(
        execution_config=EXECUTION,
        top3_receipt=Path("receipt"),
        process_id="p",
        slot_index=1,
        device="cuda:0",
    )
    with pytest.raises(
        runner.LegacyV2Stage2BlockedError, match="permanently blocked"
    ):
        runner.run_stage2_worker(args)
    assert calls == []


def test_execute_candidate_source_has_no_bn_protocol_loop() -> None:
    source = inspect.getsource(runner.execute_candidate)
    assert "for bn_protocol in BN_PROTOCOLS" not in source
    assert "bn_protocol=SS_BN_PROTOCOL" in source
    assert "_build_fast_runner_v2" in source
    assert "core._build_fast_runner" not in source
    assert "SourceCalibrationMethodInputDatasetV2" in source
    assert "load_outer_evaluator_targets_v2" in source
    assert source.index("strength_diagnostics.append") < source.index(
        "load_outer_evaluator_targets_v2("
    )
    assert "_audit_episode_v2" in source


def test_v2_zero_update_audit_keeps_hard_gates_true_with_one_false_result_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardened_gates = {
        key: True for key in REQUIRED_HARD_GATES if key != "zero_test_opens"
    }
    monkeypatch.setattr(
        runner.core,
        "_audit_episode",
        lambda result, *, bn_protocol, expect_full_audit: dict(hardened_gates),
    )
    diagnostics = {
        "changed_bn_affine_tensors_fast_gate": 0,
        "number_updated_bn_affine_tensors": 0,
        "require_nonzero_parameter_update": False,
        "zero_parameter_update_allowed": True,
        "zero_parameter_update_policy": runner.ZERO_UPDATE_POLICY,
        "zero_parameter_update_observed": True,
        "numerically_zero_parameter_delta": True,
        "actual_parameter_delta_nonzero": False,
        "temporary_optimizer_state_parameter_count": 2,
        "gradient_norm": 0.125,
        "step_norm": 0.0,
        "source_parameter_norm": 2.0,
        "relative_step_norm": 0.0,
        "relative_step_norm_epsilon": 1e-12,
        "optimizer_steps": 1,
        "finite": True,
    }
    result = SimpleNamespace(
        diagnostics=diagnostics,
        checks={
            "bn_affine_changed_by_one_step": False,
            "optimizer_changed_by_one_step": True,
        },
    )

    gates = runner._audit_episode_v2(
        result,
        bn_protocol=SS_BN_PROTOCOL,
        expect_full_audit=True,
    )

    assert all(gates.values())
    assert result.checks["bn_affine_changed_by_one_step"] is False
    broken = SimpleNamespace(
        diagnostics=dict(diagnostics),
        checks={**result.checks, "optimizer_changed_by_one_step": False},
    )
    with pytest.raises(runner.CalibrationExecutionError, match="optimizer temporary"):
        runner._audit_episode_v2(
            broken,
            bn_protocol=SS_BN_PROTOCOL,
            expect_full_audit=True,
        )
    bad_policy = SimpleNamespace(
        diagnostics={**diagnostics, "zero_parameter_update_policy": "drifted"},
        checks=dict(result.checks),
    )
    with pytest.raises(runner.CalibrationExecutionError, match="policy diagnostic"):
        runner._audit_episode_v2(
            bad_policy,
            bn_protocol=SS_BN_PROTOCOL,
            expect_full_audit=True,
        )
    bad_relative = SimpleNamespace(
        diagnostics={**diagnostics, "relative_step_norm": 1e-9},
        checks=dict(result.checks),
    )
    with pytest.raises(runner.CalibrationExecutionError, match="diagnostic definition"):
        runner._audit_episode_v2(
            bad_relative,
            bn_protocol=SS_BN_PROTOCOL,
            expect_full_audit=True,
        )


def test_strength_diagnostic_uses_official_probabilities_and_strict_threshold() -> None:
    result = SimpleNamespace(
        logits_tent_pre=runner.torch.zeros((1, 1, 2, 2), dtype=runner.torch.float32),
        logits_tent_post=runner.torch.ones((1, 1, 2, 2), dtype=runner.torch.float32),
        diagnostics={
            "changed_bn_affine_tensors_fast_gate": 2,
            "gradient_norm": 3.0,
            "step_norm": 0.25,
            "relative_step_norm": 0.125,
        },
    )

    value = runner._episode_strength_diagnostic(result)

    expected_delta = 1.0 / (1.0 + runner.np.exp(-1.0)) - 0.5
    assert value["mean_absolute_probability_delta"] == pytest.approx(expected_delta)
    assert value["maximum_absolute_probability_delta"] == pytest.approx(expected_delta)
    assert value["strict_threshold_changed_pixel_ratio"] == 1.0
    assert value["foreground_probability_mass_delta"] == pytest.approx(
        4 * expected_delta
    )


def _strength_for_record(record: dict) -> dict:
    episodes = []
    for index in range(64):
        changed = 0 if index < 7 else 2
        episodes.append(
            {
                "changed_bn_affine_tensor_count": changed,
                "mean_absolute_probability_delta": index / 6400,
                "maximum_absolute_probability_delta": index / 3200,
                "strict_threshold_changed_pixel_ratio": index / 6400,
                "foreground_probability_mass_delta": (index - 32) / 10,
                "gradient_norm": 1.0 + index / 100,
                "step_norm": 0.0 if changed == 0 else index / 1000,
                "relative_step_norm": 0.0 if changed == 0 else index / 2000,
            }
        )
    return runner._cell_strength_diagnostic(
        stage=record["stage"],
        process_id=record["process_id"],
        candidate=Candidate.from_values(
            record["candidate"]["optimizer"],
            record["candidate"]["learning_rate"],
        ),
        dataset=record["dataset"],
        corruption=record["corruption"],
        severity=record["severity"],
        episodes=episodes,
    )


def test_strength_cell_counts_summary_and_tamper_fail_closed() -> None:
    candidate = ALL_CANDIDATES[0]
    record = _run_records(
        stage=1,
        process_id="stage1-strength",
        candidate=candidate,
        improvement=1,
    )[0]
    diagnostic = _strength_for_record(record)

    runner._verify_strength_diagnostic(diagnostic, record, label="strength")
    counts = diagnostic["parameter_delta_episode_counts"]
    assert counts == {"zero": 7, "nonzero": 57, "total": 64}
    assert diagnostic["changed_bn_affine_tensor_count_histogram"] == {
        "0": 7,
        "2": 57,
    }
    summary = runner._combine_strength_diagnostics([diagnostic])
    assert summary["episode_count"] == 64
    assert summary["parameter_delta_episode_counts"] == counts
    assert summary["update_activity_contract"][
        "diagnostics_not_used_for_selection"
    ] is True

    bad_count = copy.deepcopy(diagnostic)
    bad_count["parameter_delta_episode_counts"]["zero"] += 1
    with pytest.raises(runner.CalibrationExecutionError, match="conservation"):
        runner._verify_strength_diagnostic(bad_count, record, label="bad count")
    bad_float = copy.deepcopy(diagnostic)
    bad_float["gradient_norm"]["max"] = float("nan")
    with pytest.raises(runner.CalibrationExecutionError, match="finite"):
        runner._verify_strength_diagnostic(bad_float, record, label="bad float")
    bad_disclosure = copy.deepcopy(diagnostic)
    bad_disclosure["update_activity_contract"][
        "diagnostics_not_used_for_selection"
    ] = False
    with pytest.raises(runner.CalibrationExecutionError, match="contract"):
        runner._verify_strength_diagnostic(
            bad_disclosure, record, label="bad disclosure"
        )


def test_selector_receipts_are_independent_of_separate_strength_artifact(
    synthetic_selection,
) -> None:
    stage1, stage2, stage1_receipt, final_receipt = synthetic_selection
    diagnostic = _strength_for_record(stage1[0])
    mutated = copy.deepcopy(diagnostic)
    mutated["foreground_probability_mass_delta"]["sum"] *= -1000

    assert diagnostic != mutated
    assert select_stage1_top3(stage1) == stage1_receipt
    assert select_final_candidate(stage1, stage2) == final_receipt
    assert all("update_activity_diagnostics" not in record for record in stage1 + stage2)


def test_binary_tent_rejects_legacy_aggregate_diagnostic_detail() -> None:
    from tta.binary_tent import BinaryTentMethod

    parameter = runner.torch.nn.Parameter(runner.torch.ones(()))
    optimizer = runner.torch.optim.SGD([parameter], lr=1e-5)
    with pytest.raises(ValueError, match="global.*per_parameter"):
        BinaryTentMethod(
            optimizer,
            parameter_names=("weight",),
            bn_protocol=SS_BN_PROTOCOL,
            diagnostic_detail="aggregate",
        )


def test_v2_builder_passes_frozen_global_diagnostic_detail(
    monkeypatch, tmp_path: Path
) -> None:
    import tta.binary_tent as binary_tent_module
    import tta.binary_tent_fast_runner_v2 as fast_runner_v2_module
    import tta.model_adapter as adapter_module
    import tta.state_manager as state_module

    observed: dict[str, object] = {}

    class FakeModel:
        def to(self, device):
            observed["model_device"] = device
            return self

    class FakeAdapter:
        def __init__(self, model, *, warm_flag):
            observed["adapter_warm_flag"] = warm_flag

        def set_source_eval_mode(self):
            observed["source_eval_mode"] = True

    class FakeMethodInstance:
        optimizer = object()

    class FakeMethod:
        @classmethod
        def from_adapter(cls, adapter, **kwargs):
            observed["method_kwargs"] = dict(kwargs)
            return FakeMethodInstance()

    class FakeState:
        def __init__(self, model, *, optimizer):
            observed["state_optimizer"] = optimizer
            self.source_fingerprint = SimpleNamespace(full_sha256="f" * 64)

    class FakeFastRunner:
        def __init__(
            self,
            adapter,
            state,
            method,
            *,
            full_audit_cadence,
            require_nonzero_parameter_update,
        ):
            observed["full_audit_cadence"] = full_audit_cadence
            observed["require_nonzero_parameter_update"] = (
                require_nonzero_parameter_update
            )
            self.state = state

    fake_source = ModuleType("test_source")
    fake_source.build_nsfpn_model = lambda: FakeModel()
    fake_source.load_trusted_checkpoint = (
        lambda model, checkpoint: {"trusted": True, "path": str(checkpoint)}
    )
    monkeypatch.setitem(sys.modules, "test_source", fake_source)
    monkeypatch.setattr(binary_tent_module, "BinaryTentMethod", FakeMethod)
    monkeypatch.setattr(
        fast_runner_v2_module, "BinaryTentFastRunnerV2", FakeFastRunner
    )
    monkeypatch.setattr(adapter_module, "IRSTDModelAdapter", FakeAdapter)
    monkeypatch.setattr(state_module, "EpisodicStateManager", FakeState)

    checkpoint = tmp_path / "checkpoint.pth.tar"
    checkpoint.write_bytes(b"trusted-checkpoint")
    contract = SimpleNamespace(
        checkpoints={"IRSTD-1K": checkpoint},
        scientific={
            "method": {
                "entropy_eps": 1.0e-6,
                "diagnostic_detail": "global",
                "require_nonzero_parameter_update": False,
            }
        },
        execution={
            "execution": {
                "diagnostic_detail": "global",
                "require_nonzero_parameter_update": False,
            }
        },
    )
    built, receipt = runner._build_fast_runner_v2(
        contract=contract,
        dataset="IRSTD-1K",
        candidate=ALL_CANDIDATES[0],
        bn_protocol=SS_BN_PROTOCOL,
        device="cpu",
    )
    assert isinstance(built, FakeFastRunner)
    assert observed["method_kwargs"]["diagnostic_detail"] == "global"
    assert observed["full_audit_cadence"] == 64
    assert observed["require_nonzero_parameter_update"] is False
    assert receipt["diagnostic_detail"] == "global"
    assert receipt["require_nonzero_parameter_update"] is False
    assert receipt["zero_parameter_update_policy"] == runner.ZERO_UPDATE_POLICY
    assert receipt["checkpoint_sha256"] == hashlib.sha256(
        b"trusted-checkpoint"
    ).hexdigest()


@pytest.mark.skipif(
    os.environ.get("NS_FPN_RUN_FAILED_BUILDER_FORENSIC_TEST") != "1",
    reason=(
        "requires artifacts from the specific failed builder run; set "
        "NS_FPN_RUN_FAILED_BUILDER_FORENSIC_TEST=1 to opt in"
    ),
)
def test_failed_builder_run_left_zero_formal_episodes_and_durable_locks_reusable() -> None:
    output_root = ROOT / "results/binary_tent/ss_calibration_v2"
    assert not list(output_root.glob("stage1/shards/*"))
    assert not list(output_root.rglob("records.jsonl"))
    durable_controls = (
        ROOT
        / "results/binary_tent/.source_calibration_physical_gpu_leases/physical-gpu-1.lock",
        ROOT
        / "results/binary_tent/.source_calibration_physical_gpu_leases/physical-gpu-2.lock",
        ROOT
        / "results/binary_tent/.ss_calibration_v2_publication_work/.ss-v2-stage1.launcher.lock",
    )
    assert all(path.is_file() and not path.is_symlink() for path in durable_controls)
    for path in durable_controls:
        descriptor = os.open(
            path,
            os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def test_synthetic_selector_counts_and_disclosures(synthetic_selection) -> None:
    stage1, stage2, stage1_receipt, final_receipt = synthetic_selection
    assert len(stage1) == 390
    assert len(stage2) == 234
    assert stage1_receipt["validation"]["episode_count"] == 24960
    assert final_receipt["validation"]["stage2_episode_count"] == 14976
    assert final_receipt["validation"]["total_calibration_episode_count"] == 39936
    for receipt in (stage1_receipt, final_receipt):
        assert receipt["selection_bn_protocol"] == "SS"
        assert receipt["BS_excluded_from_selection"] is True
        assert receipt["application_protocols"] == ["SS", "BS"]
        assert receipt["best_pd_reuses_same_frozen_hyperparameters_without_tuning"] is True


def test_cross_run_endpoint_invariants_are_ss_only(synthetic_selection) -> None:
    stage1, stage2, _, _ = synthetic_selection
    value = runner._verify_cross_run_endpoint_invariants(stage1, stage2)
    assert value["selection_bn_protocol"] == "SS"
    assert value["BS_excluded_from_selection"] is True
    assert value["total_image_pixels_per_cell"] == 64 * 256 * 256
    assert value["stage2_tent_pre_integer_counts_equal_stage1"] is True


def test_cross_run_endpoint_tamper_fails_closed(synthetic_selection) -> None:
    stage1, stage2, _, _ = synthetic_selection
    tampered = copy.deepcopy(stage2)
    tampered[0]["endpoints"]["tent_pre"]["intersection_pixels"] += 1
    with pytest.raises(runner.CalibrationExecutionError, match="TENT-pre identity"):
        runner._verify_cross_run_endpoint_invariants(stage1, tampered)


def test_total_pixel_denominator_tamper_fails_closed(synthetic_selection) -> None:
    stage1, _, _, _ = synthetic_selection
    tampered = copy.deepcopy(stage1)
    tampered[0]["endpoints"]["tent_pre"]["total_image_pixels"] -= 1
    with pytest.raises(runner.CalibrationExecutionError, match="pre pixels"):
        runner._verify_cross_run_endpoint_invariants(tampered)


def test_record_common_rejects_bs_and_transition_fields() -> None:
    candidate = ALL_CANDIDATES[0]
    record = _run_records(
        stage=1, process_id="stage1-1", candidate=candidate, improvement=1
    )[0]
    bs = copy.deepcopy(record)
    bs["bn_protocol"] = "single_image_spatial_batch_stats"
    with pytest.raises(runner.CalibrationExecutionError, match="SS protocol"):
        runner._verify_record_common(
            bs,
            stage=1,
            process_id="stage1-1",
            candidate=candidate,
            label="record",
        )
    transition = copy.deepcopy(record)
    transition["target_erasure"] = 0
    with pytest.raises(runner.CalibrationExecutionError, match="forbidden field"):
        runner._verify_record_common(
            transition,
            stage=1,
            process_id="stage1-1",
            candidate=candidate,
            label="record",
        )
    nested = copy.deepcopy(record)
    nested["extra_audit"] = {"nested": {"target_recovery_rate": 0}}
    with pytest.raises(runner.CalibrationExecutionError, match="forbidden field"):
        runner._verify_record_common(
            nested,
            stage=1,
            process_id="stage1-1",
            candidate=candidate,
            label="record",
        )


def test_application_receipt_binds_best_miou_and_best_pd_without_tuning() -> None:
    contract = runner.load_contract(EXECUTION)
    selected = ALL_CANDIDATES[0].to_dict()
    value = runner._application_receipt(contract, selected)
    assert value["selection_bn_protocol"] == "SS"
    assert value["BS_excluded_from_selection"] is True
    assert value["application_protocols"] == ["SS", "BS"]
    assert value["best_pd_additional_tuning_episodes"] == 0
    assert len(value["checkpoint_bindings"]) == 6
    assert {item["checkpoint_role"] for item in value["checkpoint_bindings"]} == {
        "best_miou",
        "best_pd",
    }
    assert all(item["additional_tuning_episodes"] == 0 for item in value["checkpoint_bindings"])


def test_stage2_process_ids_are_receipt_digest_bound() -> None:
    digest = "a" * 64
    first = runner._stage2_slot_process_id(digest, 1)
    second = runner._stage2_slot_process_id(digest, 2)
    assert first != second
    assert digest in first and digest in second
    with pytest.raises(runner.CalibrationExecutionError):
        runner._stage2_slot_process_id("A" * 64, 1)


def test_parser_exposes_only_explicit_roles_and_does_not_run_by_parsing() -> None:
    parser = runner.build_parser()
    args = parser.parse_args(["validate"])
    assert args.role == "validate"
    assert args.execution_config == runner.DEFAULT_EXECUTION_CONFIG
    args = parser.parse_args(["launch-stage1", "--gpu-ids", "1,2"])
    assert args.gpu_ids == "1,2"


def test_scope_provenance_is_explicit_and_best_pd_zero_tuning() -> None:
    contract = runner.load_contract(EXECUTION)
    value = runner._scope_provenance(contract)
    assert value["scope"] == {"paper_result": False, "source_train_derived": True}
    disclosure = value["calibration_protocol_disclosure"]
    assert disclosure["selection_bn_protocol"] == "SS"
    assert disclosure["BS_excluded_from_selection"] is True
    assert disclosure["best_pd_tuning_episodes"] == 0
    assert disclosure["update_activity"] == runner._update_activity_disclosure()
    assert value["target_access_contract"]["target_integrity_bytes_hashed_before_adaptation"] is True


def test_runner_never_names_a_validation_split_or_old_cache() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "source_val" not in source
    assert "source_calibration_cache_v1" not in source
    assert "materialize_binary_tent_source_calibration_cache" not in source
