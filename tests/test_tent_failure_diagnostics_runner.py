from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts import run_tent_failure_diagnostics as d0


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tent_failure_diagnostics_v1.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _analysis_scope(*, oracle: bool, dataset: str = "IRSTD-1K") -> dict:
    config = d0._load_config(CONFIG)
    entry = config["datasets"].get(dataset, config["datasets"]["IRSTD-1K"])
    return {
        "dataset": dataset,
        "split_name": "train",
        "split_sha256": entry["train_split_sha256"],
        "checkpoint_sha256": entry["checkpoint"]["sha256"],
        "seed": 42,
        "source_train_derived": True,
        "paper_test_result": False,
        "use_test_images": False,
        "use_test_labels": False,
        "oracle_analysis": oracle,
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": 1 if oracle else 0,
        "adaptation_gradient_uses_labels": False,
        "supervised_gradient_role": (
            "outer_oracle_train_labels_only" if oracle else "none"
        ),
    }


def _determinism_audit(scope: str) -> dict:
    config = d0._load_config(CONFIG)
    determinism = config["method"]["determinism"]
    strict = {
        "deterministic_algorithms_enabled": True,
        "warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    during = dict(strict)
    during["deterministic_algorithms_enabled"] = False
    return {
        "schema_version": 1,
        "policy": "strict_forwards_temporary_backward_disable",
        "config_sha256": d0.frozen_determinism_sha256(determinism),
        "scope": scope,
        "loss_device_type": "cuda",
        "temporary_backward_disable_applied": True,
        "state_before_backward_scope": strict,
        "state_during_backward": during,
        "state_after_backward_scope": strict,
        "backward_completed": True,
        "restored_exact": True,
    }


def _geometry_metrics_fixture(*, per_group: bool) -> dict:
    metrics = {
        "scalar_count": 8736,
        "active_gradient_scalar_count": 8736,
        "actual_step_norm": 0.1,
        "cross_backend_cpu_storage_replay_expected_step_norm": 0.1,
        "cross_backend_cpu_storage_replay_cos_actual_expected": 1.0,
        "cross_backend_cpu_storage_replay_actual_to_expected_norm_ratio": 1.0,
        "cross_backend_cpu_storage_replay_residual_norm": 0.0,
        "cross_backend_cpu_storage_replay_max_abs_residual": 0.0,
        "continuous_ideal_expected_step_norm": 0.1,
        "continuous_ideal_cos_actual_expected": 1.0,
        "continuous_ideal_actual_to_expected_norm_ratio": 1.0,
        "continuous_ideal_residual_norm": 0.0,
        "continuous_ideal_max_abs_residual": 0.0,
        "small_gradient_scalar_count": 4368,
        "small_gradient_fraction": 0.5,
        "near_sign_step_threshold": None,
        "near_sign_step_scalar_count": None,
        "near_sign_step_fraction": None,
        "gradient_norm": 0.2,
        "effective_gradient_norm": 0.2,
        "parameter_norm_before": 10.0,
        "relative_actual_step_norm": 0.01,
        "cpu_storage_replay_within_frozen_tolerance": True,
    }
    if per_group:
        metrics["parameter_tensor_count"] = 106
    return metrics


def _record(
    slug: str = "SGD_lr_1em5",
    *,
    dataset: str = "IRSTD-1K",
    image_id: str = "image-0",
    optimizer: str = "SGD",
    learning_rate: float = 1e-5,
) -> dict:
    return {
        "schema_version": 1,
        "dataset": dataset,
        "split_role": "train",
        "image_id": image_id,
        "corruption": "clean",
        "severity": 0,
        "candidate_slug": slug,
        "candidate": {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
        },
        "optimization_entropy_pre": 0.25,
        "source_eval_post_warm_task_loss": 0.5,
        "noop": {
            "schema_version": 1,
            "classification": "threshold_noop",
            "binary_transitions": {"binary_pixel_xor_count": 0},
            "threshold_margin": {
                "strata": {
                    "all": {
                        "pixel_count": 4,
                        "abs_delta_gt_margin_count": 0,
                        "abs_delta_gt_0_1_margin_count": 1,
                        "binary_xor_count": 0,
                    }
                }
            },
            "metric_counts": {
                "identical": True,
                "pre": {
                    "detected_targets": 1,
                    "false_alarm_pixels": 1,
                    "intersection_pixels": 1,
                    "false_positive_pixels": 1,
                    "false_negative_pixels": 1,
                    "total_image_pixels": 4,
                    "total_targets": 2,
                    "union_pixels": 3,
                },
                "post": {
                    "detected_targets": 1,
                    "false_alarm_pixels": 1,
                    "intersection_pixels": 1,
                    "false_positive_pixels": 1,
                    "false_negative_pixels": 1,
                    "total_image_pixels": 4,
                    "total_targets": 2,
                    "union_pixels": 3,
                },
            },
        },
        "optimizer_geometry": {
            "schema_version": 3,
            "analysis_type": "tent_optimizer_first_step_geometry",
            "scope": _analysis_scope(oracle=False, dataset=dataset),
            "empty_optimizer_state_verified": True,
            "current_binary_tent_optimizer_configuration": True,
            "gate_policy": {
                "optimizer_correctness_acceptance_gate": "runtime_same_device_hard_gate",
                "runtime_same_device_hard_gate_is_sole_acceptance_gate": True,
                "cpu_storage_replay_used_for_hard_gate": False,
                "continuous_ideal_used_for_hard_gate": False,
            },
            "references": {
                "cross_backend_cpu_storage_replay": {
                    "name": d0.CROSS_BACKEND_CPU_STORAGE_REPLAY_REFERENCE,
                    "role": "cross_backend_diagnostic_only",
                    "used_for_hard_gate": False,
                    "pytorch_version": d0.PYTORCH_REFERENCE_VERSION,
                    "single_tensor_operation_order": True,
                    "native_storage_dtype": True,
                    "final_parameter_storage_rounding_included": True,
                },
                "continuous_ideal": {
                    "name": d0.CONTINUOUS_IDEAL_REFERENCE,
                    "role": "scientific_geometry_explanation_only",
                    "dtype": "torch.float64",
                    "used_for_hard_gate": False,
                },
            },
            "first_step_formula_role": "continuous_float64_explanation_only",
            "runtime_same_device_hard_gate": {
                "schema_version": 1,
                "reference_name": d0.RUNTIME_HARD_GATE_REFERENCE,
                "pytorch_version_required": d0.PYTORCH_REFERENCE_VERSION,
                "pytorch_version_observed": d0.PYTORCH_REFERENCE_VERSION,
                "pytorch_default_dtype": "torch.float32",
                "pytorch_default_device_type": "cpu",
                "optimizer_class": "SGD",
                "optimizer_config_from_actual_param_group": {
                    "name": "SGD",
                    "learning_rate": 1e-5,
                    "weight_decay": 0.0,
                    "maximize": False,
                    "initial_optimizer_state_empty": True,
                    "momentum": 0.9,
                    "dampening": 0.0,
                    "nesterov": True,
                },
                "implementation_flags_from_actual_param_group": {
                    "foreach": False,
                    "fused": None,
                    "capturable": None,
                    "differentiable": False,
                },
                "defaults_from_actual_optimizer": {
                    "dampening": 0.0,
                    "differentiable": False,
                    "foreach": False,
                    "lr": 1e-5,
                    "maximize": False,
                    "momentum": 0.9,
                    "nesterov": True,
                    "weight_decay": 0.0,
                },
                "single_param_group_verified": True,
                "ordered_parameter_identity_verified": True,
                "parameter_tensor_count": 106,
                "initial_optimizer_state_empty": True,
                "optimizer_step_call_count": 1,
                "logical_step_count_per_parameter": 1,
                "all_parameters_reached_logical_step_one": True,
                "native_step_counter_exposed": False,
                "native_step_counter_one_parameter_count": 0,
                "state_parameter_count": 106,
                "state_tensor_count": 106,
                "bit_exact_state_tensor_count": 106,
                "all_optimizer_state_tensors_bit_exact": True,
                "reference_optimizer_state_bundle_sha256": "e" * 64,
                "actual_optimizer_state_bundle_sha256": "e" * 64,
                "adam_step_dtype": None,
                "adam_step_device_type": None,
                "same_device_reference": True,
                "native_storage_dtype_reference": True,
                "comparison": "torch.equal_per_parameter_tensor",
                "bit_exact_parameter_tensor_count": 106,
                "all_parameters_bit_exact": True,
                "reference_after_bundle_sha256": "c" * 64,
                "actual_after_bundle_sha256": "c" * 64,
            },
            "thresholds": {
                "small_gradient_absolute_threshold": 1e-8,
                "adam_near_sign_step_ratio_threshold": 0.9,
                "verification_rtol": 1e-5,
                "verification_atol": 1e-7,
            },
            "parameter_layout": {
                "parameter_tensor_count": 106,
                "parameter_scalar_count": 8736,
                "parameter_names_sha256": "a" * 64,
                "topology_sha256": "b" * 64,
            },
            "global": _geometry_metrics_fixture(per_group=False),
            "per_group": {
                "decoder_0": _geometry_metrics_fixture(per_group=True)
            },
        },
        "entropy_task_alignment": {
            "analysis_type": "entropy_task_gradient_alignment",
            "scope": _analysis_scope(oracle=True, dataset=dataset),
            "parameter_layout": {
                "parameter_tensor_count": 106,
                "parameter_scalar_count": 8736,
                "parameter_names_sha256": "a" * 64,
                "topology_sha256": "b" * 64,
            },
            "label_isolation": {
                "entropy_gradient_label_free": True,
                "adaptation_step_label_free": True,
                "supervised_gradient_used_by_method": False,
                "supervised_gradient_role": "outer_oracle_train_labels_only",
                "method_label_accesses": 0,
                "outer_evaluator_label_accesses": 1,
            },
            "global": {
                "first_order_task_effect": "predicted_task_loss_decrease"
            },
            "per_group": {
                "decoder_0": {
                    "entropy_supervised_cosine": 0.25,
                    "supervised_dot_adaptation_step": -0.2,
                    "first_order_task_effect": "predicted_task_loss_decrease",
                }
            },
        },
        "determinism": {
            "entropy_backward": _determinism_audit("entropy_backward"),
            "supervised_task_backward": _determinism_audit(
                "supervised_task_backward"
            ),
            "strict_policy_before_optimizer_step": True,
            "strict_policy_before_post_forward": True,
        },
        "scope": {
            "oracle_analysis": True,
            "paper_test_result": False,
            "source_train_derived": True,
            "method_label_accesses": 0,
            "outer_evaluator_label_accesses": 1,
            "use_test_images": False,
            "use_test_labels": False,
        },
    }


def _adam_record() -> dict:
    record = _record(slug="Adam_lr_1em5", optimizer="Adam", learning_rate=1e-5)
    gate = record["optimizer_geometry"]["runtime_same_device_hard_gate"]
    gate["optimizer_class"] = "Adam"
    gate["optimizer_config_from_actual_param_group"] = {
        "name": "Adam", "learning_rate": 1e-5, "weight_decay": 0.0,
        "maximize": False, "initial_optimizer_state_empty": True,
        "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
        "decoupled_weight_decay": False,
    }
    gate["implementation_flags_from_actual_param_group"] = {
        "foreach": False, "fused": False, "capturable": False,
        "differentiable": False,
    }
    gate["defaults_from_actual_optimizer"] = {
        "amsgrad": False, "betas": [0.9, 0.999], "capturable": False,
        "differentiable": False, "eps": 1e-8, "foreach": False,
        "fused": False, "lr": 1e-5, "maximize": False, "weight_decay": 0.0,
    }
    gate["native_step_counter_exposed"] = True
    gate["native_step_counter_one_parameter_count"] = 106
    gate["state_tensor_count"] = 318
    gate["bit_exact_state_tensor_count"] = 318
    gate["adam_step_dtype"] = "torch.float32"
    gate["adam_step_device_type"] = "cpu"
    for metrics in (
        record["optimizer_geometry"]["global"],
        *record["optimizer_geometry"]["per_group"].values(),
    ):
        metrics["near_sign_step_threshold"] = 0.9
        metrics["near_sign_step_scalar_count"] = 4368
        metrics["near_sign_step_fraction"] = 0.5
    return record


def _smoke_provenance(*, source_hashes: dict[str, str]) -> dict:
    config = d0._load_config(CONFIG)
    dataset = "IRSTD-1K"
    entry = config["datasets"][dataset]
    return {
        "schema_version": 1,
        "artifact_type": d0.ARTIFACT_TYPE,
        "dataset": dataset,
        "formal_d0_complete": False,
        "paper_test_result": False,
        "oracle_analysis": True,
        "source_train_derived": True,
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": 1,
        "test_images_opened": 0,
        "test_labels_opened": 0,
        "conditions": ["clean_S0"],
        "candidates": [{"optimizer": "SGD", "learning_rate": 1e-5}],
        "images_per_condition": 1,
        "checkpoint_wrapper": "state_dict",
        "checkpoint_sha256": entry["checkpoint"]["sha256"],
        "train_split_sha256": entry["train_split_sha256"],
        "config_path": str(CONFIG.resolve()),
        "config_sha256": d0.sha256_file(CONFIG),
        "equivalence_receipt_sha256": None,
        "equivalence_receipt": None,
        "global_runtime_seal_sha256": "0" * 64,
        "v2_runtime_seal": {},
        "runtime_audits": [],
        "diagnostic_source_code_sha256": dict(sorted(source_hashes.items())),
        "diagnostic_source_audits": [],
        "frozen_v2_stage1_exact_comparison": None,
        "cache_lineage": {},
        "target_deserialized_only_after_cell_label_free_completion": True,
        "source_eval_task_gradient_contract": dict(config["source_task_gradient"]),
        "determinism_contract": dict(config["method"]["determinism"]),
        "determinism_contract_sha256": d0.frozen_determinism_sha256(
            config["method"]["determinism"]
        ),
        "selection_authorized": False,
        "stage2_authorized": False,
        "stage3_authorized": False,
        "elapsed_seconds": 0.1,
    }


def test_config_is_post_warm_slsiou_on_source_eval_graph() -> None:
    config = d0._load_config(CONFIG)
    assert config["scope"]["no_validation_split"] is True
    assert config["scope"]["paper_result"] is False
    task = config["source_task_gradient"]
    assert task["contract"] == "post_warm_SLSIoU_on_source_eval_graph"
    assert task["human_epoch"] == 1000
    assert task["epoch_index"] == 999
    assert task["warm_epochs"] == 5
    assert task["warm_flag"] is False
    assert task["expected_auxiliary_count"] == 0
    assert task["averaging_denominator"] == 1
    assert task["model_runtime"] == "source_eval"
    assert task["bn_protocol_during_diagnostic"] == "source_running_statistics"
    assert task["full_source_training_runtime_exact"] is False


def test_candidate_and_condition_sets_equal_frozen_v2() -> None:
    config = d0._load_config(CONFIG)
    candidates = d0._config_candidates(config)
    assert len(candidates) == 10
    assert candidates[0].slug == "Adam_lr_1em5"
    assert candidates[-1].slug == "SGD_lr_1em3"
    assert len(d0._config_conditions(config)) == 13


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_full_negative_archive_ledger_is_frozen_and_verified() -> None:
    receipt = d0._verify_frozen_negative_archive(d0._load_config(CONFIG))
    assert receipt["verified"] is True
    assert receipt["paper_result"] is False
    assert receipt["file_count"] == 113
    assert receipt["source_inventory_sha256"] == (
        "bf8954dd2692f23e6b65761e4337f1daa77a64439386b05c6bb8df716800dfbf"
    )
    assert receipt["stage1_records_sha256"] == (
        "ee6a21b29ceee32fa6f9f70c6509436971133f81e815596fd148716807bbe3d3"
    )


def test_summary_reports_noop_and_first_order_counts() -> None:
    summary = d0.summarize_records([_record(), _record()])
    candidate = summary["candidates"]["SGD_lr_1em5"]
    assert candidate["episode_count"] == 2
    assert candidate["classification_counts"] == {"threshold_noop": 2}
    assert candidate["mean_iou_delta"] == 0.0
    assert candidate[
        "cpu_storage_replay_within_frozen_tolerance_episode_count"
    ] == 2
    assert candidate["first_order_task_decrease_episode_count"] == 2
    assert candidate["per_group"]["decoder_0"][
        "predicted_task_loss_decrease_count"
    ] == 2
    assert candidate["threshold_margin"]["all"][
        "fraction_abs_delta_gt_0_1_margin"
    ] == 0.25


def test_atomic_shard_roundtrip_and_tamper_rejection(tmp_path: Path) -> None:
    destination = tmp_path / "shard"
    records = [_record()]
    summary = d0.summarize_records(records)
    source_hashes = {"fake.py": "0" * 64}
    result = d0._publish_shard(
        destination=destination,
        records=records,
        summary=summary,
        provenance=_smoke_provenance(source_hashes=source_hashes),
        formal_scope=False,
        source_hashes=source_hashes,
    )
    assert result["status"] == "verified"
    assert result["formal_d0_complete"] is False
    with pytest.raises(FileExistsError):
        d0._publish_shard(
            destination=destination,
            records=records,
            summary=summary,
            provenance={},
            formal_scope=False,
            source_hashes={},
        )
    (destination / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(d0.DiagnosticRunnerError, match="hash mismatch"):
        d0.verify_shard(destination)


def test_receipt_prepublish_failure_leaves_canonical_path_retryable(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "canonical.json"

    def reject(_staging: Path) -> None:
        raise d0.DiagnosticRunnerError("synthetic staged receipt rejection")

    with pytest.raises(d0.DiagnosticRunnerError, match="synthetic"):
        d0._write_once_json(
            destination,
            {"complete": True},
            prepublish_validator=reject,
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".canonical.json.tmp-*")) == []
    digest = d0._write_once_json(destination, {"complete": True})
    assert digest == d0.sha256_file(destination)


def test_shard_full_staging_verification_failure_is_clean_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "shard"
    records = [_record()]
    source_hashes = {"fake.py": "0" * 64}
    original = d0.verify_shard

    def reject(_path: Path) -> dict:
        raise d0.DiagnosticRunnerError("synthetic staged shard rejection")

    monkeypatch.setattr(d0, "verify_shard", reject)
    with pytest.raises(d0.DiagnosticRunnerError, match="synthetic"):
        d0._publish_shard(
            destination=destination,
            records=records,
            summary=d0.summarize_records(records),
            provenance=_smoke_provenance(source_hashes=source_hashes),
            formal_scope=False,
            source_hashes=source_hashes,
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".shard.tmp-*")) == []

    monkeypatch.setattr(d0, "verify_shard", original)
    assert d0._publish_shard(
        destination=destination,
        records=records,
        summary=d0.summarize_records(records),
        provenance=_smoke_provenance(source_hashes=source_hashes),
        formal_scope=False,
        source_hashes=source_hashes,
    )["status"] == "verified"


def test_aggregate_staging_verification_failure_is_clean_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "aggregate"
    calls: list[tuple[Path, Path | None]] = []

    def reject(
        path: Path, *, expected_canonical_destination: Path | None = None
    ) -> dict:
        calls.append((path, expected_canonical_destination))
        raise d0.DiagnosticRunnerError("synthetic staged aggregate rejection")

    monkeypatch.setattr(d0, "verify_aggregate", reject)
    kwargs = {
        "destination": destination,
        "episode_records": [],
        "cell_records": [],
        "candidate_summary": {"candidates": {}},
        "provenance": {},
        "prepublish_guard": lambda: None,
    }
    with pytest.raises(d0.DiagnosticRunnerError, match="synthetic"):
        d0._publish_aggregate(**kwargs)
    assert not destination.exists()
    assert list(tmp_path.glob(".aggregate.tmp-*")) == []
    assert len(calls) == 1
    assert calls[0][0] != destination
    assert calls[0][1] == destination

    def accept(
        path: Path, *, expected_canonical_destination: Path | None = None
    ) -> dict:
        assert path != destination
        assert expected_canonical_destination == destination
        return {"status": "verified", "path": str(path)}

    monkeypatch.setattr(d0, "verify_aggregate", accept)
    result = d0._publish_aggregate(**kwargs)
    assert result == {"status": "verified", "path": str(destination)}
    assert destination.is_dir()


def test_shard_verifier_rejects_change_during_live_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "shard"
    records = [_record()]
    source_hashes = {"fake.py": "0" * 64}
    d0._publish_shard(
        destination=destination,
        records=records,
        summary=d0.summarize_records(records),
        provenance=_smoke_provenance(source_hashes=source_hashes),
        formal_scope=False,
        source_hashes=source_hashes,
    )
    original = d0._validate_shard_provenance

    def mutate_after_validation(**kwargs):
        result = original(**kwargs)
        summary_path = destination / "summary.json"
        summary_path.write_bytes(summary_path.read_bytes())
        return result

    monkeypatch.setattr(d0, "_validate_shard_provenance", mutate_after_validation)
    with pytest.raises(d0.DiagnosticRunnerError, match="changed during"):
        d0.verify_shard(destination)


def test_formal_scope_fails_before_cuda_without_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched = False

    def forbidden_contract():
        nonlocal touched
        touched = True
        raise AssertionError("contract/CUDA boundary was reached")

    monkeypatch.setattr(d0.frozen_v2, "load_contract", forbidden_contract)
    with pytest.raises(d0.DiagnosticRunnerError, match="equivalence"):
        d0.run_dataset(
            config_path=CONFIG,
            dataset="IRSTD-1K",
            device_name="cuda:0",
            destination=tmp_path / "formal",
            condition_filters=(),
            candidate_filters=(),
            max_images=64,
            equivalence_receipt=None,
        )
    assert touched is False


def test_equivalence_receipt_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = d0._load_config(CONFIG)
    candidates = d0._config_candidates(config)
    receipt = tmp_path / "receipt.json"
    source_hashes = {"fake.py": "0" * 64}
    runtime_seal = "1" * 64
    value = {
        "dataset": "IRSTD-1K",
        "sample": {
            "condition": "clean_S0",
            "image_index": 0,
            "image_id": "XDU102",
        },
        "config_sha256": d0.sha256_file(CONFIG),
        "global_runtime_seal_sha256": runtime_seal,
        "diagnostic_source_code_sha256": source_hashes,
        "determinism_contract": config["method"]["determinism"],
        "candidate_slugs": [candidate.slug for candidate in candidates],
        "candidate_count": 10,
    }
    d0._write_once_json(receipt, value)
    monkeypatch.setattr(
        d0,
        "validate_d0_equivalence_repro_receipt",
        lambda _value, *, revalidate_inputs: d0.sha256_file(receipt),
    )
    assert len(
        d0._verify_equivalence_receipt(
            receipt,
            dataset="IRSTD-1K",
            candidates=candidates,
            config_sha256=d0.sha256_file(CONFIG),
            global_runtime_seal_sha256=runtime_seal,
            diagnostic_source_code_sha256=source_hashes,
            expected_condition="clean_S0",
            expected_image_index=0,
            expected_image_id="XDU102",
        )
    ) == 64
    value["candidate_count"] = 9
    receipt.write_bytes(d0._canonical_json(value) + b"\n")
    with pytest.raises(d0.DiagnosticRunnerError, match="live bindings"):
        d0._verify_equivalence_receipt(
            receipt,
            dataset="IRSTD-1K",
            candidates=candidates,
            config_sha256=d0.sha256_file(CONFIG),
            global_runtime_seal_sha256=runtime_seal,
            diagnostic_source_code_sha256=source_hashes,
            expected_condition="clean_S0",
            expected_image_index=0,
            expected_image_id="XDU102",
        )


def test_named_tensor_bundle_hash_binds_values_and_order() -> None:
    names = ("a", "b")
    values = {
        "a": torch.tensor([1.0], dtype=torch.float32),
        "b": torch.tensor([2.0], dtype=torch.float32),
    }
    digest = d0._named_tensor_bundle_sha256(names, values)
    changed = dict(values)
    changed["b"] = torch.tensor([3.0], dtype=torch.float32)
    assert digest != d0._named_tensor_bundle_sha256(names, changed)
    with pytest.raises(d0.DiagnosticRunnerError, match="topology"):
        d0._named_tensor_bundle_sha256(("b", "a"), values)


def test_first_step_capture_hashes_raw_gradient_and_delta() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    gradient = torch.tensor([0.5, -0.25], dtype=torch.float32)
    before = parameter.detach().clone()
    parameter.grad = gradient.clone()
    original, capture = d0._install_first_step_capture(optimizer, ("weight",))
    try:
        optimizer.step()
    finally:
        optimizer.step = original
    assert capture["gradient_bundle_sha256"] == d0._named_tensor_bundle_sha256(
        ("weight",), {"weight": gradient}
    )
    assert capture["delta_bundle_sha256"] == d0._named_tensor_bundle_sha256(
        ("weight",), {"weight": parameter.detach() - before}
    )


@pytest.mark.parametrize(
    "candidate",
    [d0.Candidate("Adam", 3e-5), d0.Candidate("SGD", 3e-5)],
)
def test_runtime_optimizer_contract_and_native_reference_are_bit_exact(
    candidate: d0.Candidate,
) -> None:
    parameters = [
        nn.Parameter(torch.tensor([2.1058686], dtype=torch.float32)),
        nn.Parameter(torch.tensor([-0.25], dtype=torch.float32)),
    ]
    gradients = [
        torch.tensor([3.7e-11], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
    ]
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.clone()
    optimizer = d0.build_binary_tent_optimizer(
        parameters,
        name=candidate.optimizer,
        learning_rate=candidate.learning_rate,
    )
    spec, receipt = d0._verify_actual_optimizer_pre_step(
        optimizer,
        candidate=candidate,
        parameters=parameters,
    )
    reference = d0.pytorch_first_step_reference(
        parameters_before=tuple(
            (f"p{index}", parameter.detach().clone())
            for index, parameter in enumerate(parameters)
        ),
        gradients=tuple(
            (f"p{index}", gradient)
            for index, gradient in enumerate(gradients)
        ),
        optimizer=spec,
    )
    optimizer.step()
    post = d0._verify_actual_optimizer_post_step(
        optimizer,
        spec=spec,
        names=("p0", "p1"),
        parameters=parameters,
        reference_state=reference.optimizer_state,
    )
    assert receipt["reference_name"] == d0.RUNTIME_HARD_GATE_REFERENCE
    assert receipt["pytorch_version_observed"] == "2.1.2"
    assert receipt["initial_optimizer_state_empty"] is True
    assert post["all_parameters_reached_logical_step_one"] is True
    assert all(
        torch.equal(parameter.detach(), reference.parameters_after[f"p{index}"])
        for index, parameter in enumerate(parameters)
    )


def test_runtime_optimizer_contract_rejects_actual_group_drift() -> None:
    candidate = d0.Candidate("Adam", 3e-5)
    parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = d0.build_binary_tent_optimizer(
        [parameter], name="Adam", learning_rate=candidate.learning_rate
    )
    optimizer.param_groups[0]["foreach"] = True
    with pytest.raises(d0.DiagnosticRunnerError, match="param_group differs"):
        d0._verify_actual_optimizer_pre_step(
            optimizer,
            candidate=candidate,
            parameters=[parameter],
        )


def test_runtime_optimizer_contract_rejects_nonfloat32_default_dtype() -> None:
    candidate = d0.Candidate("Adam", 3e-5)
    parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = d0.build_binary_tent_optimizer(
        [parameter], name="Adam", learning_rate=candidate.learning_rate
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(d0.DiagnosticRunnerError, match="CPU float32"):
            d0._verify_actual_optimizer_pre_step(
                optimizer,
                candidate=candidate,
                parameters=[parameter],
            )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    ("optimizer_name", "field", "tamper"),
    [
        ("Adam", "exp_avg", "value"),
        ("Adam", "exp_avg_sq", "value"),
        ("Adam", "step", "value"),
        ("Adam", "step", "dtype"),
        ("SGD", "momentum_buffer", "value"),
    ],
)
def test_runtime_optimizer_post_step_rejects_state_tampering(
    optimizer_name: str, field: str, tamper: str
) -> None:
    candidate = d0.Candidate(optimizer_name, 3e-5)
    parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    gradient = torch.tensor([0.2], dtype=torch.float32)
    parameter.grad = gradient.clone()
    optimizer = d0.build_binary_tent_optimizer(
        [parameter], name=optimizer_name, learning_rate=candidate.learning_rate
    )
    spec, _receipt = d0._verify_actual_optimizer_pre_step(
        optimizer,
        candidate=candidate,
        parameters=[parameter],
    )
    reference = d0.pytorch_first_step_reference(
        parameters_before=(("p", parameter.detach().clone()),),
        gradients=(("p", gradient),),
        optimizer=spec,
    )
    optimizer.step()
    if tamper == "dtype":
        optimizer.state[parameter][field] = optimizer.state[parameter][field].to(
            torch.float64
        )
    else:
        optimizer.state[parameter][field].add_(1.0)
    with pytest.raises(d0.DiagnosticRunnerError, match="state|step"):
        d0._verify_actual_optimizer_post_step(
            optimizer,
            spec=spec,
            names=("p",),
            parameters=[parameter],
            reference_state=reference.optimizer_state,
        )


def test_label_free_episode_hard_fails_on_single_bit_reference_endpoint_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyAdapter:
        def __init__(self) -> None:
            self.model = nn.Module()
            self.model.register_parameter(
                "weight", nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
            )

        def set_tent_mode(self, *, use_batch_stats: bool) -> None:
            assert use_batch_stats is False

        def set_source_eval_mode(self) -> None:
            self.model.eval()

        def forward_logits(self, image: torch.Tensor) -> torch.Tensor:
            return image[:, :1] * self.model.weight.reshape(1, 1, 1, 1)

    adapter = TinyAdapter()
    parameter = adapter.model.weight
    source_parameters = (parameter.detach().clone(),)
    original_reference = d0.pytorch_first_step_reference

    def one_bit_drift(**kwargs):
        reference = original_reference(**kwargs)
        value = reference.parameters_after["bn.weight"]
        byte_view = value.reshape(-1).view(torch.uint8)
        byte_view[0].bitwise_xor_(1)
        return reference

    def plain_backward(loss, *, scope, config):
        del scope, config
        loss.backward()
        return {"test_stub": True}

    monkeypatch.setattr(d0, "pytorch_first_step_reference", one_bit_drift)
    monkeypatch.setattr(d0, "_backward", plain_backward)
    monkeypatch.setattr(d0, "assert_strict_forward_policy", lambda _config: True)
    with pytest.raises(
        d0.DiagnosticRunnerError,
        match="same-device storage reference mismatch",
    ):
        d0._label_free_episode(
            adapter=adapter,
            parameters=(parameter,),
            names=("bn.weight",),
            source_parameters=source_parameters,
            assignment={"bn.weight": "decoder_0"},
            image=torch.ones((1, 1, 2, 2), dtype=torch.float32),
            metadata={"image_id": "synthetic"},
            candidates=(d0.Candidate("Adam", 1e-5),),
            config=d0._load_config(CONFIG),
            dataset="IRSTD-1K",
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "hash",
        "state_hash",
        "state_count",
        "step_device",
        "extra_field",
        "reference",
    ],
)
def test_record_validator_rejects_runtime_hard_gate_tampering(tamper: str) -> None:
    config = d0._load_config(CONFIG)
    record = _record()
    gate = record["optimizer_geometry"]["runtime_same_device_hard_gate"]
    if tamper == "hash":
        gate["actual_after_bundle_sha256"] = "d" * 64
    elif tamper == "state_hash":
        gate["actual_optimizer_state_bundle_sha256"] = "d" * 64
    elif tamper == "state_count":
        gate["bit_exact_state_tensor_count"] = 105
    elif tamper == "step_device":
        gate["adam_step_device_type"] = "cpu"
    elif tamper == "extra_field":
        gate["unchecked"] = True
    else:
        gate["reference_name"] = "unfrozen"
    with pytest.raises(d0.DiagnosticRunnerError, match=r"hard[_ ]gate"):
        d0.validate_d0_records(
            [record],
            config=config,
            dataset="IRSTD-1K",
            conditions=("clean_S0",),
            candidates=(d0.Candidate("SGD", 1e-5),),
            images_per_condition=1,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "references_root_extra",
        "verification_extra",
        "verification_missing",
        "verification_gate_role",
        "ideal_extra",
        "ideal_missing",
        "threshold_extra",
        "threshold_missing",
        "threshold_value",
        "threshold_bool_type",
    ],
)
def test_record_validator_exactly_binds_references_and_thresholds(
    tamper: str,
) -> None:
    config = d0._load_config(CONFIG)
    record = _record()
    geometry = record["optimizer_geometry"]
    references = geometry["references"]
    thresholds = geometry["thresholds"]
    if tamper == "references_root_extra":
        references["untrusted"] = {}
    elif tamper == "verification_extra":
        references["cross_backend_cpu_storage_replay"]["untrusted"] = True
    elif tamper == "verification_missing":
        del references["cross_backend_cpu_storage_replay"]["single_tensor_operation_order"]
    elif tamper == "verification_gate_role":
        references["cross_backend_cpu_storage_replay"]["used_for_hard_gate"] = True
    elif tamper == "ideal_extra":
        references["continuous_ideal"]["untrusted"] = True
    elif tamper == "ideal_missing":
        del references["continuous_ideal"]["dtype"]
    elif tamper == "threshold_extra":
        thresholds["untrusted"] = 0.0
    elif tamper == "threshold_missing":
        del thresholds["verification_atol"]
    elif tamper == "threshold_value":
        thresholds["verification_atol"] = 2e-7
    else:
        thresholds["adam_near_sign_step_ratio_threshold"] = True
    with pytest.raises(d0.DiagnosticRunnerError, match="reference|threshold"):
        d0.validate_d0_records(
            [record],
            config=config,
            dataset="IRSTD-1K",
            conditions=("clean_S0",),
            candidates=(d0.Candidate("SGD", 1e-5),),
            images_per_condition=1,
        )


def test_record_validator_allows_cross_backend_replay_mismatch_when_runtime_gate_exact() -> None:
    config = d0._load_config(CONFIG)
    record = _record()
    record["optimizer_geometry"]["global"][
        "cpu_storage_replay_within_frozen_tolerance"
    ] = False
    record["optimizer_geometry"]["per_group"]["decoder_0"][
        "cpu_storage_replay_within_frozen_tolerance"
    ] = False
    d0.validate_d0_records(
        [record],
        config=config,
        dataset="IRSTD-1K",
        conditions=("clean_S0",),
        candidates=(d0.Candidate("SGD", 1e-5),),
        images_per_condition=1,
    )


def test_record_validator_rejects_nonfinite_cross_backend_diagnostic_metric() -> None:
    config = d0._load_config(CONFIG)
    record = _record()
    record["optimizer_geometry"]["per_group"]["decoder_0"][
        "actual_step_norm"
    ] = float("nan")
    with pytest.raises(d0.DiagnosticRunnerError, match="must be finite"):
        d0.validate_d0_records(
            [record],
            config=config,
            dataset="IRSTD-1K",
            conditions=("clean_S0",),
            candidates=(d0.Candidate("SGD", 1e-5),),
            images_per_condition=1,
        )


@pytest.mark.parametrize("tamper", ["extra_legacy", "missing", "string", "negative_count"])
def test_record_validator_rejects_geometry_metric_schema_tampering(tamper: str) -> None:
    config = d0._load_config(CONFIG)
    record = _record()
    metrics = record["optimizer_geometry"]["per_group"]["decoder_0"]
    if tamper == "extra_legacy":
        metrics["formula_matches_within_tolerance"] = True
    elif tamper == "missing":
        del metrics["cross_backend_cpu_storage_replay_residual_norm"]
    elif tamper == "string":
        metrics["actual_step_norm"] = "0.1"
    else:
        metrics["active_gradient_scalar_count"] = -1
    with pytest.raises(d0.DiagnosticRunnerError, match="optimizer_geometry"):
        d0.validate_d0_records(
            [record],
            config=config,
            dataset="IRSTD-1K",
            conditions=("clean_S0",),
            candidates=(d0.Candidate("SGD", 1e-5),),
            images_per_condition=1,
        )


@pytest.mark.parametrize("tamper", ["sgd_near", "small_fraction", "adam_near_fraction"])
def test_record_validator_binds_optimizer_dependent_nullable_fractions(tamper: str) -> None:
    config = d0._load_config(CONFIG)
    record = _adam_record() if tamper == "adam_near_fraction" else _record()
    candidate = (
        d0.Candidate("Adam", 1e-5)
        if tamper == "adam_near_fraction"
        else d0.Candidate("SGD", 1e-5)
    )
    metrics = record["optimizer_geometry"]["per_group"]["decoder_0"]
    if tamper == "sgd_near":
        metrics["near_sign_step_threshold"] = 0.9
    elif tamper == "small_fraction":
        metrics["small_gradient_fraction"] = 0.25
    else:
        metrics["near_sign_step_fraction"] = 0.25
    with pytest.raises(d0.DiagnosticRunnerError, match="optimizer_geometry"):
        d0.validate_d0_records(
            [record], config=config, dataset="IRSTD-1K",
            conditions=("clean_S0",), candidates=(candidate,), images_per_condition=1,
        )


def test_record_validator_rejects_per_group_topology_sum_mismatch() -> None:
    config = d0._load_config(CONFIG)
    record = _record()
    record["optimizer_geometry"]["per_group"]["decoder_0"]["scalar_count"] -= 1
    record["optimizer_geometry"]["per_group"]["decoder_0"][
        "active_gradient_scalar_count"
    ] -= 1
    record["optimizer_geometry"]["per_group"]["decoder_0"][
        "small_gradient_scalar_count"
    ] = 0
    record["optimizer_geometry"]["per_group"]["decoder_0"][
        "small_gradient_fraction"
    ] = 0.0
    with pytest.raises(d0.DiagnosticRunnerError, match="parameter topology"):
        d0.validate_d0_records(
            [record], config=config, dataset="IRSTD-1K",
            conditions=("clean_S0",), candidates=(d0.Candidate("SGD", 1e-5),),
            images_per_condition=1,
        )


def test_verify_rejects_extra_member(tmp_path: Path) -> None:
    destination = tmp_path / "shard"
    records = [_record()]
    source_hashes = {"fake.py": "0" * 64}
    d0._publish_shard(
        destination=destination,
        records=records,
        summary=d0.summarize_records(records),
        provenance=_smoke_provenance(source_hashes=source_hashes),
        formal_scope=False,
        source_hashes=source_hashes,
    )
    (destination / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(d0.DiagnosticRunnerError, match="member set"):
        d0.verify_shard(destination)


def test_equivalence_smoke_cli_is_explicit() -> None:
    args = d0.build_parser().parse_args(
        [
            "equivalence-smoke",
            "--dataset",
            "IRSTD-1K",
            "--process-id",
            "repro-1-test",
            "--parent-run-nonce",
            "1" * 64,
            "--child-launch-nonce",
            "2" * 64,
            "--output",
            "/tmp/equivalence.json",
        ]
    )
    assert args.command == "equivalence-smoke"
    assert args.condition == "clean_S0"
    assert args.image_index == 0
    assert args.process_id == "repro-1-test"
    assert args.parent_run_nonce == "1" * 64
    assert args.child_launch_nonce == "2" * 64

    repro = d0.build_parser().parse_args(
        [
            "equivalence-repro",
            "--dataset",
            "IRSTD-1K",
            "--output",
            "/tmp/equivalence-aggregate.json",
        ]
    )
    assert repro.command == "equivalence-repro"

    aggregate = d0.build_parser().parse_args(
        [
            "aggregate",
            "--shard",
            "/tmp/irstd",
            "--shard",
            "/tmp/nuaa",
            "--shard",
            "/tmp/nudt",
            "--output",
            "/tmp/aggregate",
        ]
    )
    assert aggregate.command == "aggregate"
    assert len(aggregate.shard) == 3


def test_equivalence_worker_rejects_missing_parent_nonce_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process_id = "repro-1-test"
    monkeypatch.setenv("CR_SITTA_D0_PROCESS_ID", process_id)
    monkeypatch.delenv("CR_SITTA_D0_PARENT_RUN_NONCE", raising=False)
    monkeypatch.delenv("CR_SITTA_D0_CHILD_LAUNCH_NONCE", raising=False)
    monkeypatch.delenv("CR_SITTA_D0_COMMAND_SHA256", raising=False)
    with pytest.raises(d0.DiagnosticRunnerError, match="parent-issued"):
        d0.run_equivalence_smoke(
            config_path=CONFIG,
            dataset="IRSTD-1K",
            device_name="cuda:0",
            condition_key="clean_S0",
            image_index=0,
            process_id=process_id,
            parent_run_nonce="1" * 64,
            child_launch_nonce="2" * 64,
            output=tmp_path / "never-written.json",
        )
    assert not (tmp_path / "never-written.json").exists()


def test_source_state_gate_covers_runtime_not_only_state_dict() -> None:
    model = nn.Sequential(nn.BatchNorm2d(2))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source = d0._model_state_snapshot(model)
    d0._assert_model_state_exact(model, source)
    model[0].weight.requires_grad_(True)
    with pytest.raises(d0.DiagnosticRunnerError, match="refrozen"):
        d0._assert_model_state_exact(model, source)


def _archive_cell(
    *, dataset: str, pre: dict, post: dict, image_count: int
) -> dict:
    fields = d0.ENDPOINT_COUNT_FIELDS
    return {
        "stage": 1,
        "dataset": dataset,
        "corruption": "clean",
        "severity": 0,
        "candidate": {
            "optimizer": "SGD",
            "learning_rate": 1e-5,
            "learning_rate_decimal": "1e-5",
        },
        "bn_protocol": "source_running_statistics",
        "image_count": image_count,
        "optimizer_steps_total": image_count,
        "method_label_accesses": 0,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "endpoints": {
            "tent_pre": {field: pre[field] for field in fields},
            "tent_post": {field: post[field] for field in fields},
        },
    }


def test_exact_cell_reconstruction_uses_fraction_sufficient_statistics() -> None:
    first = _record(image_id="a")
    second = _record(image_id="b")
    first["noop"]["metric_counts"]["post"].update(
        {
            "intersection_pixels": 2,
            "union_pixels": 3,
            "detected_targets": 2,
            "false_alarm_pixels": 0,
        }
    )
    cells = d0.build_exact_cell_records(
        [first, second],
        expected_datasets=("IRSTD-1K",),
        expected_conditions=("clean_S0",),
        expected_candidates=(d0.Candidate("SGD", 1e-5),),
        expected_images_per_cell=2,
    )
    assert len(cells) == 1
    cell = cells[0]
    assert cell["episode_identity"]["image_ids"] == ["a", "b"]
    assert cell["metrics_exact"]["pre"]["global_iou"]["exact"] == "1/3"
    assert cell["metrics_exact"]["post"]["global_iou"]["exact"] == "1/2"
    assert cell["metrics_exact"]["delta"]["global_iou"]["exact"] == "1/6"
    assert cell["metrics_exact"]["delta"]["pd"]["exact"] == "1/4"
    assert cell["metrics_exact"]["delta"]["fa"]["exact"] == "-1/8"


def test_formal_cell_comparison_is_exact_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = d0.Candidate("SGD", 1e-5)
    monkeypatch.setattr(d0, "FORMAL_DATASETS", ("D",))
    monkeypatch.setattr(d0, "FORMAL_IMAGES_PER_CELL", 2)
    monkeypatch.setattr(d0, "_config_conditions", lambda _config: ("clean_S0",))
    monkeypatch.setattr(d0, "_config_candidates", lambda _config: (candidate,))
    first = _record(dataset="D", image_id="a")
    second = _record(dataset="D", image_id="b")
    cells = d0.build_exact_cell_records(
        [first, second],
        expected_datasets=("D",),
        expected_conditions=("clean_S0",),
        expected_candidates=(candidate,),
        expected_images_per_cell=2,
    )
    archive_path = tmp_path / "stage1_records.jsonl"
    archive = _archive_cell(
        dataset="D",
        pre=cells[0]["endpoint_counts"]["pre"],
        post=cells[0]["endpoint_counts"]["post"],
        image_count=2,
    )
    d0._write_jsonl(archive_path, [archive])
    archive_digest = d0.sha256_file(archive_path)
    archive_verification = {
        "path": str(tmp_path),
        "file_count": 1,
        "source_inventory_sha256": "a" * 64,
        "sha256sums_sha256": "b" * 64,
        "negative_result_sha256": "c" * 64,
        "stage1_records_sha256": archive_digest,
        "paper_result": False,
        "verified": True,
    }
    monkeypatch.setattr(
        d0, "_verify_frozen_negative_archive", lambda _config: archive_verification
    )
    config = {
        "datasets": {"D": {}},
        "comparison": {"stage1_records_sha256": archive_digest},
    }
    receipt = d0.verify_formal_dataset_against_frozen_v2(
        records=[first, second],
        dataset="D",
        config=config,
        archive_records_path=archive_path,
    )
    assert receipt["passed"] is True
    assert receipt["cell_count"] == 1
    assert receipt["diagnostic_record_count"] == 2
    archive["endpoints"]["tent_post"]["intersection_pixels"] += 1
    d0._write_jsonl(archive_path, [archive])
    config["comparison"]["stage1_records_sha256"] = d0.sha256_file(archive_path)
    with pytest.raises(d0.DiagnosticRunnerError, match="differs from frozen"):
        d0.verify_formal_dataset_against_frozen_v2(
            records=[first, second],
            dataset="D",
            config=config,
            archive_records_path=archive_path,
        )


def test_three_dataset_aggregate_is_immutable_hash_sealed_and_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_config = d0._load_config(CONFIG)
    datasets = ("D1", "D2", "D3")
    candidate = d0.Candidate("SGD", 1e-5)
    records = {
        dataset: [_record(dataset=dataset, image_id=f"{dataset}-image")]
        for dataset in datasets
    }
    archive_path = tmp_path / "stage1_records.jsonl"
    archive_records = []
    for dataset in datasets:
        cell = d0.build_exact_cell_records(records[dataset])[0]
        archive_records.append(
            _archive_cell(
                dataset=dataset,
                pre=cell["endpoint_counts"]["pre"],
                post=cell["endpoint_counts"]["post"],
                image_count=1,
            )
        )
    d0._write_jsonl(archive_path, archive_records)
    config = {
        "method": {
            "determinism": canonical_config["method"]["determinism"]
        },
        "evaluation": {
            "optimizer_geometry": canonical_config["evaluation"][
                "optimizer_geometry"
            ]
        },
        "datasets": {
            dataset: {
                "train_split_sha256": records[dataset][0]["optimizer_geometry"][
                    "scope"
                ]["split_sha256"],
                "checkpoint": {
                    "sha256": records[dataset][0]["optimizer_geometry"]["scope"]
                    ["checkpoint_sha256"]
                },
            }
            for dataset in datasets
        },
        "comparison": {
            "frozen_v2_negative_archive": str(tmp_path),
            "stage1_records": archive_path.name,
            "stage1_records_sha256": d0.sha256_file(archive_path),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test-config\n", encoding="utf-8")
    monkeypatch.setattr(d0, "FORMAL_DATASETS", datasets)
    monkeypatch.setattr(d0, "FORMAL_IMAGES_PER_CELL", 1)
    monkeypatch.setattr(d0, "_load_config", lambda _path: config)
    monkeypatch.setattr(d0, "_config_conditions", lambda _config: ("clean_S0",))
    monkeypatch.setattr(d0, "_config_candidates", lambda _config: (candidate,))
    monkeypatch.setattr(
        d0,
        "_archive_records_path_from_config",
        lambda _config: archive_path,
    )
    synthetic_archive_verification = {
        "path": str(tmp_path),
        "file_count": 3,
        "source_inventory_sha256": "d" * 64,
        "sha256sums_sha256": "e" * 64,
        "negative_result_sha256": "f" * 64,
        "stage1_records_sha256": d0.sha256_file(archive_path),
        "paper_result": False,
        "verified": True,
    }
    monkeypatch.setattr(
        d0,
        "_verify_frozen_negative_archive",
        lambda _config: synthetic_archive_verification,
    )
    monkeypatch.setattr(
        d0,
        "_validated_output_destination",
        lambda *, config, destination, role, dataset=None: destination,
    )
    monkeypatch.setattr(
        d0,
        "_ensure_output_parent",
        lambda *, config, role: tmp_path,
    )
    monkeypatch.setattr(d0.frozen_v2, "CONDITIONS", (("clean", 0),))
    monkeypatch.setattr(d0.frozen_v2, "ALL_CANDIDATES", (candidate,))

    def fake_formal_payload(*, path, config, config_sha256, archive_records_path):
        dataset = path.name
        receipt = d0.verify_formal_dataset_against_frozen_v2(
            records=records[dataset],
            dataset=dataset,
            config=config,
            archive_records_path=archive_records_path,
        )
        return dataset, records[dataset], receipt, {
            "path": str(path),
            "artifact_manifest_sha256": "a" * 64,
            "complete_sha256": "b" * 64,
            "episode_records_sha256": "c" * 64,
            "record_count": 1,
            "dataset": dataset,
        }

    monkeypatch.setattr(d0, "_formal_shard_payload", fake_formal_payload)
    output = tmp_path / "aggregate"
    result = d0.aggregate_formal_shards(
        config_path=config_path,
        shard_paths=[tmp_path / dataset for dataset in datasets],
        destination=output,
    )
    assert result["status"] == "verified"
    assert result["dataset_count"] == 3
    assert result["cell_count"] == 3
    assert result["diagnostic_record_count"] == 3
    candidate_summary = json.loads(
        (output / "candidate_summary.json").read_text(encoding="utf-8")
    )["candidates"][candidate.slug]
    assert candidate_summary["per_group"]["decoder_0"][
        "mean_entropy_supervised_cosine"
    ] == 0.25
    assert candidate_summary["per_group"]["decoder_0"][
        "predicted_task_loss_decrease_count"
    ] == 3
    assert candidate_summary["threshold_margin"]["all"][
        "fraction_abs_delta_gt_0_1_margin"
    ] == 0.25
    assert d0.verify_aggregate(output)["status"] == "verified"
    with pytest.raises(FileExistsError):
        d0.aggregate_formal_shards(
            config_path=config_path,
            shard_paths=[tmp_path / dataset for dataset in datasets],
            destination=output,
        )
    (output / "candidate_summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(d0.DiagnosticRunnerError, match="hash mismatch"):
        d0.verify_aggregate(output)
