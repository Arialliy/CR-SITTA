#!/usr/bin/env python3
"""Execute the source-train-derived, SS-only Binary TENT calibration v2.

The calibration evidence is exclusively ``source_running_statistics`` (SS).
The resulting optimizer/LR is later *applied* to both SS and BS, but no BS
episode is accepted by this runner or its selector.  ``validate`` is a
metadata/opaque-byte operation: it never calls ``numpy.load``, constructs a
model, initializes CUDA, creates a validation split, or writes formal output.

Safety-sensitive publication, scheduling, runtime-seal, and GPU-lease
primitives are reused from the hardened v3 runner only after its complete file
SHA256 and an explicit API allow-list have been verified.  No global is
monkeypatched.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXECUTION_CONFIG = (
    PROJECT_ROOT / "configs" / "binary_tent_ss_calibration_execution_v2.yaml"
)
HARDENED_CORE_PATH = PROJECT_ROOT / "run_binary_tent_source_calibration.py"
HARDENED_CORE_SHA256 = (
    "a3175ade39e656d04778ff804b8d1d106c01ea44f8390b6c65a2cb8ed7eb4d20"
)
CACHE_PROTOCOL_SHA256 = (
    "e225311cce252125eaf3a2b47eeeea4cde60d0363c8c71f48494d70e41c2aa3e"
)
CACHE_INVENTORY_GO_SEAL_SHA256 = (
    "99b487241078828129e1566817290f53c186d5988e42c560a719c9701fd7aa0e"
)
SELECTOR_SHA256 = (
    "609cbb8582ad7c497075e95d2896675268678d6f73cb4ebec4441911e6a3ca4e"
)
MATERIALIZER_SHA256 = (
    "5ba2f97289da183ed952ade467833bc0188d5ed8468efceb5667a7dd43450645"
)
EXECUTION_PROTOCOL_ID = "cr-sitta-binary-tent-ss-calibration-execution-v2"
SCIENTIFIC_PROTOCOL_ID = "cr-sitta-binary-tent-ss-calibration-v2"
SCOPE = {"paper_result": False, "source_train_derived": True}
JSON_SEPARATORS = (",", ":")
ZERO_UPDATE_POLICY = (
    "allow_only_with_one_finite_optimizer_step_temporary_state_and_exact_reset"
)
STRENGTH_DIAGNOSTICS_FILENAME = "lr_strength_diagnostics.jsonl"
EXPECTED_CRITICAL_CODE_PATHS = (
    "run_binary_tent_ss_calibration_v2.py",
    "run_binary_tent_source_calibration.py",
    "configs/retrain_fixed_splits.yaml",
    "materialize_binary_tent_ss_calibration_cache_v2.py",
    "tta/binary_tent_ss_calibration_selector_v2.py",
    "tta/binary_tent_calibration_selector.py",
    "tta/binary_tent_fast_runner_v2.py",
    "tta/binary_tent_fast_runner.py",
    "tta/binary_tent_runner.py",
    "tta/binary_tent.py",
    "tta/state_manager.py",
    "tta/model_adapter.py",
    "metrics/irstd_metrics.py",
    "metrics/connected_components.py",
    "metrics/target_matching.py",
    "dataio/corruption_cache.py",
    "dataio/train_side_pilot_protocol.py",
    "dataio/research_dataset.py",
    "corruptions/corruption_protocol.py",
    "corruptions/infrared_corruptions.py",
    "model/MSHNet_NSFPN.py",
    "model/NS_FPN.py",
    "model/diff_cross_attns.py",
    "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
    "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
    ".conda/lib/python3.10/site-packages/"
    "MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so",
    "test_source.py",
)


def _update_activity_disclosure() -> dict[str, Any]:
    return {
        "require_nonzero_parameter_update": False,
        "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
        "optimizer_temporary_state_required": True,
        "diagnostics_not_used_for_selection": True,
        "selector_inputs_exclude_strength_diagnostics": True,
        "artifact_filename": STRENGTH_DIAGNOSTICS_FILENAME,
        "one_record_per_64_image_cell": True,
        "computed_before_outer_target_loader": True,
        "uses_target_tensor": False,
        "probability_transform": "metrics.irstd_metrics.probabilities_from_logits",
        "changed_pixel_rule": "strict_probability_gt_0.5_disagreement",
        "relative_step_norm_definition": "step_norm/(source_parameter_norm+epsilon)",
        "relative_step_norm_epsilon": 1e-12,
    }


def _stdlib_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# This check intentionally precedes importing the safety core.  A drifted core
# cannot execute module-level code under this runner's authority.
if _stdlib_sha256(HARDENED_CORE_PATH) != HARDENED_CORE_SHA256:
    raise RuntimeError(
        "refusing to import drifted hardened safety core: "
        f"{HARDENED_CORE_PATH}"
    )

import run_binary_tent_source_calibration as core  # noqa: E402
from materialize_binary_tent_ss_calibration_cache_v2 import (  # noqa: E402
    METHOD_FIELDS,
    SourceCalibrationMethodInputDatasetV2,
    load_outer_evaluator_targets_v2,
    validate_contract as validate_cache_contract,
    validate_materialization,
)
from metrics.irstd_metrics import (  # noqa: E402
    IRSTDEvaluationProtocol,
    UnifiedResearchEvaluator,
    probabilities_from_logits,
)
from tta.binary_tent_ss_calibration_selector_v2 import (  # noqa: E402
    ALL_CANDIDATES,
    CELLS_PER_RUN,
    CONDITIONS,
    DATASETS,
    FORBIDDEN_SELECTOR_FIELDS,
    FORBIDDEN_SELECTOR_FIELD_PREFIXES,
    IMAGES_PER_CELL,
    REQUIRED_HARD_GATES,
    REQUIRED_PROTOCOL_AUDIT,
    SS_BN_PROTOCOL,
    Candidate,
    select_final_candidate,
    select_stage1_top3,
)


CORE_API_ALLOW_LIST = (
    "BoundFile",
    "CacheContext",
    "CalibrationContract",
    "CalibrationExecutionError",
    "RuntimeSeal",
    "RuntimeSealMonitor",
    "_audit_episode",
    "_bound_file",
    "_configure_cuda_worker",
    "_endpoint_counts",
    "_extend_runtime_seal",
    "_fresh_process_id",
    "_gpu_ids",
    "_load_json",
    "_observed_worker_environment",
    "_publish_directory",
    "_run_parallel",
    "_runtime_seal_from_bindings",
    "_safe_slug",
    "_seed_candidate",
    "_stage2_slot_process_id",
    "_validate_formal_gpu_lease",
    "_verify_embedded_gpu_lease",
    "_verify_inherited_gpu_lease",
    "_with_stage_launcher_lock",
    "build_cell_record",
    "run_label_free_episode",
    "verify_shard",
)
missing_core_api = tuple(name for name in CORE_API_ALLOW_LIST if not hasattr(core, name))
if missing_core_api:
    raise RuntimeError(f"hardened core API allow-list is incomplete: {missing_core_api}")


CalibrationExecutionError = core.CalibrationExecutionError
BoundFile = core.BoundFile
CacheContext = core.CacheContext
CalibrationContract = core.CalibrationContract
RuntimeSeal = core.RuntimeSeal
RuntimeSealMonitor = core.RuntimeSealMonitor


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationExecutionError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CalibrationExecutionError(f"{label} must be a sequence")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise CalibrationExecutionError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _project_path(raw: str | Path) -> Path:
    value = Path(raw).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = yaml.safe_load(handle)
    return dict(_mapping(value, str(path)))


def _load_json(path: Path) -> dict[str, Any]:
    return core._load_json(path)


def _scope_provenance(contract: CalibrationContract) -> dict[str, Any]:
    limitation = dict(
        _mapping(
            contract.scientific["inherited_source_limitation"],
            "inherited source limitation",
        )
    )
    return {
        "scope": dict(SCOPE),
        "inherited_source_limitation": limitation,
        "calibration_protocol_disclosure": {
            "selection_bn_protocol": "SS",
            "selection_bn_protocol_id": SS_BN_PROTOCOL,
            "BS_excluded_from_selection": True,
            "application_protocols": ["SS", "BS"],
            "application_protocol_ids": [
                SS_BN_PROTOCOL,
                "single_image_spatial_batch_stats",
            ],
            "best_pd_reuses_same_frozen_hyperparameters_without_tuning": True,
            "best_pd_tuning_episodes": 0,
            "update_activity": _update_activity_disclosure(),
        },
        "target_access_contract": {
            "target_integrity_bytes_hashed_before_adaptation": True,
            "integrity_hashing_is_opaque_byte_access_not_tensor_access": True,
            "target_tensor_deserialized_or_indexed_only_after_all_cell_episodes_complete": True,
        },
    }


def _target_access_audit(*, tensor_evaluation_completed: bool) -> dict[str, bool]:
    return {
        "target_integrity_bytes_hashed_before_adaptation": True,
        "target_integrity_hashing_did_not_deserialize_or_index_tensor": True,
        "target_tensor_deserialized_or_indexed_before_all_cell_episodes_complete": False,
        "target_tensor_deserialized_and_indexed_after_all_cell_episodes_complete": bool(
            tensor_evaluation_completed
        ),
    }


def _validate_protocols(
    execution: Mapping[str, Any], scientific: Mapping[str, Any]
) -> None:
    _equal(execution.get("schema_version"), 2, "execution schema")
    _equal(
        execution.get("execution_protocol_id"),
        EXECUTION_PROTOCOL_ID,
        "execution protocol ID",
    )
    _equal(scientific.get("schema_version"), 2, "scientific schema")
    _equal(
        scientific.get("protocol_id"), SCIENTIFIC_PROTOCOL_ID, "scientific protocol ID"
    )
    scope = _mapping(scientific.get("scope"), "scientific scope")
    for key, expected in (
        ("paper_result", False),
        ("source_train_derived", True),
        ("independent_validation", False),
        ("no_validation_split_created", True),
        ("official_splits_used", ["train", "test"]),
        ("use_test_images", False),
        ("use_test_labels", False),
        ("method_receives_labels", False),
        ("target_transition_metrics_used_for_selection", False),
    ):
        _equal(scope.get(key), expected, f"scientific scope.{key}")
    method = _mapping(scientific.get("method"), "scientific method")
    _equal(method.get("diagnostic_detail"), "global", "scientific diagnostic detail")
    _equal(
        method.get("require_nonzero_parameter_update"),
        False,
        "scientific zero-update requirement",
    )
    _equal(
        method.get("zero_parameter_update_policy"),
        ZERO_UPDATE_POLICY,
        "scientific zero-update policy",
    )
    _equal(
        dict(
            _mapping(
                method.get("update_activity_diagnostics"),
                "scientific update diagnostics",
            )
        ),
        _update_activity_disclosure(),
        "scientific update diagnostics",
    )
    _equal(
        method.get("selection_bn_protocol"),
        {"display_name": "SS", "id": SS_BN_PROTOCOL},
        "SS selection protocol",
    )
    _equal(method.get("BS_excluded_from_selection"), True, "BS exclusion")
    _equal(
        method.get("application_protocols"),
        {
            "display_names": ["SS", "BS"],
            "ids": [SS_BN_PROTOCOL, "single_image_spatial_batch_stats"],
        },
        "application protocol disclosure",
    )
    _equal(
        method.get("best_pd_reuses_same_frozen_hyperparameters_without_tuning"),
        True,
        "best_pd reuse",
    )
    _equal(
        _mapping(method.get("selector"), "selector").get("sha256"),
        SELECTOR_SHA256,
        "selector SHA anchor",
    )
    candidates = _mapping(method.get("candidates"), "method candidates")
    _equal(candidates.get("optimizer_order"), ["Adam", "SGD"], "optimizer order")
    _equal(
        candidates.get("learning_rates"),
        [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3],
        "learning-rate grid",
    )
    _equal(candidates.get("cross_product_count"), 10, "candidate count")
    _equal(
        candidates.get("shared_across_datasets_and_application_protocols"),
        True,
        "shared candidate policy",
    )
    selection = _mapping(scientific.get("selection"), "selection")
    stage1 = _mapping(selection.get("stage_1"), "selection stage1")
    stage2 = _mapping(selection.get("stage_2"), "selection stage2")
    expected_counts = (
        (stage1.get("cell_records"), 390, "stage1 records"),
        (stage1.get("episodes"), 24960, "stage1 episodes"),
        (stage2.get("cell_records"), 234, "stage2 records"),
        (stage2.get("episodes"), 14976, "stage2 episodes"),
        (stage2.get("episodes_per_process"), 7488, "stage2 process episodes"),
        (selection.get("total_episodes"), 39936, "total episodes"),
        (selection.get("BS_selection_episodes"), 0, "BS selection episodes"),
        (
            selection.get("best_pd_additional_tuning_episodes"),
            0,
            "best_pd tuning episodes",
        ),
    )
    for actual, expected, label in expected_counts:
        _equal(actual, expected, label)
    _equal(
        tuple(tuple(value) for value in scientific["corruption_conditions"]["ordered"]),
        CONDITIONS,
        "condition order",
    )
    evaluation = _mapping(execution.get("evaluation"), "execution evaluation")
    for key, expected in (
        ("fixed_probability_threshold", 0.5),
        ("froc_probability_thresholds", [0.5]),
        ("target_threshold_rule", "strict_greater_than"),
        ("prediction_threshold_rule", "strict_greater_than"),
        ("connectivity", 2),
        ("max_centroid_distance", 3.0),
        ("centroid_distance_eligibility_rule", "strict_less_than"),
        ("min_component_area", 1),
        (
            "matching",
            "one_to_one_minimum_total_centroid_distance_over_eligible_pairs",
        ),
        ("detected_targets", "matched_target_components"),
        ("total_targets", "target_connected_component_count"),
        ("false_alarm_pixels", "sum_area_of_unmatched_prediction_components"),
        ("target_transition_fields_forbidden", True),
    ):
        _equal(evaluation.get(key), expected, f"evaluation.{key}")
    scientific_evaluation = _mapping(
        scientific.get("evaluation"), "scientific evaluation"
    )
    for key in (
        "fixed_probability_threshold",
        "froc_probability_thresholds",
        "target_threshold_rule",
        "prediction_threshold_rule",
        "connectivity",
        "max_centroid_distance",
        "centroid_distance_eligibility_rule",
        "min_component_area",
        "target_components",
        "prediction_components",
        "matching",
        "detected_targets",
        "total_targets",
        "false_alarm_pixels",
    ):
        _equal(
            scientific_evaluation.get(key),
            evaluation.get(key),
            f"scientific/execution evaluation.{key}",
        )
    _equal(scientific_evaluation.get("selector_cell"), "dataset_x_SS_x_condition", "SS selector cell")
    _equal(scientific_evaluation.get("cells_per_run"), 39, "SS cells per run")
    _equal(scientific_evaluation.get("transition_fields_forbidden"), True, "scientific transition firewall")
    execution_values = _mapping(execution.get("execution"), "execution values")
    for key, expected in (
        ("seed", 42),
        ("batch_size", 1),
        ("num_workers", 0),
        ("image_size", 256),
        ("amp_enabled", False),
        ("diagnostic_detail", "global"),
        ("require_nonzero_parameter_update", False),
        ("zero_parameter_update_policy", ZERO_UPDATE_POLICY),
        ("device_type", "cuda"),
        ("cuda_device_order", "PCI_BUS_ID"),
        ("visible_cuda_devices_per_worker", 1),
        ("full_state_sha256_audit_cadence", 64),
        ("force_full_audit_at_every_cell_end", True),
        ("selection_bn_protocol", SS_BN_PROTOCOL),
        ("BS_excluded_from_selection", True),
        ("rebuild_model_method_optimizer_per_dataset", True),
        ("stage2_candidate_order", "frozen_stage1_top3"),
        ("reapply_fixed_seed_before_every_candidate", True),
    ):
        _equal(execution_values.get(key), expected, f"execution.{key}")
    _equal(
        dict(
            _mapping(
                execution_values.get("update_activity_diagnostics"),
                "execution update diagnostics",
            )
        ),
        _update_activity_disclosure(),
        "execution update diagnostics",
    )
    deterministic = _mapping(
        execution_values.get("deterministic_forwards"), "deterministic forwards"
    )
    for key, expected in (
        ("algorithms_enabled", True),
        ("warn_only", False),
        ("cudnn_deterministic", True),
        ("cudnn_benchmark", False),
        ("cublas_workspace_config", ":4096:8"),
    ):
        _equal(deterministic.get(key), expected, f"deterministic.{key}")
    runtime = _mapping(execution.get("runtime_seal"), "runtime seal")
    for key, expected in (
        ("algorithm", "calibration-runtime-seal-v1"),
        ("full_byte_hash_at_process_entry", True),
        ("target_integrity_bytes_hashed_before_adaptation", True),
        (
            "target_tensor_deserialized_or_indexed_only_after_all_cell_episodes_complete",
            True,
        ),
        ("best_pd_checkpoint_bound_without_model_load", True),
        ("best_pd_reuses_same_frozen_hyperparameters_without_tuning", True),
        ("zero_parameter_update_policy_bound", True),
        ("update_activity_diagnostics_bound_and_ranking_excluded", True),
        ("stage2_top3_receipt_bound_with_canonical_role", "stage1_top3_receipt:ss_v2"),
    ):
        _equal(runtime.get(key), expected, f"runtime_seal.{key}")


def load_contract(
    path: str | Path = DEFAULT_EXECUTION_CONFIG,
) -> CalibrationContract:
    execution_path = Path(path).expanduser().resolve()
    execution = _load_yaml(execution_path)
    scientific_link = _mapping(execution.get("scientific_protocol"), "scientific link")
    cache_link = _mapping(execution.get("cache_protocol"), "cache link")
    core_link = _mapping(execution.get("hardened_safety_core"), "hardened core")
    scientific_path = _project_path(str(scientific_link["path"]))
    cache_protocol_path = _project_path(str(cache_link["path"]))
    _equal(_stdlib_sha256(scientific_path), scientific_link["sha256"], "scientific SHA")
    _equal(
        _stdlib_sha256(cache_protocol_path), CACHE_PROTOCOL_SHA256, "cache protocol SHA"
    )
    _equal(cache_link.get("sha256"), CACHE_PROTOCOL_SHA256, "cache SHA anchor")
    _equal(
        cache_link.get("inventory_go_seal_sha256"),
        CACHE_INVENTORY_GO_SEAL_SHA256,
        "cache inventory GO seal",
    )
    _equal(_project_path(str(core_link["path"])), HARDENED_CORE_PATH, "core path")
    _equal(core_link.get("sha256"), HARDENED_CORE_SHA256, "core SHA declaration")
    _equal(_stdlib_sha256(HARDENED_CORE_PATH), HARDENED_CORE_SHA256, "core SHA")
    _equal(core_link.get("import_only_after_sha256_verification"), True, "core gate")
    _equal(core_link.get("monkeypatch_forbidden"), True, "monkeypatch policy")
    _equal(
        tuple(core_link.get("api_allow_list", ())),
        CORE_API_ALLOW_LIST,
        "core API allow-list",
    )
    scientific = _load_yaml(scientific_path)
    cache_protocol = _load_yaml(cache_protocol_path)
    _validate_protocols(execution, scientific)
    scientific_lineage = _mapping(scientific["lineage"], "scientific lineage")
    scientific_cache = _mapping(
        scientific_lineage["cache_protocol"], "scientific cache lineage"
    )
    _equal(
        _project_path(str(scientific_cache["path"])),
        cache_protocol_path,
        "scientific/execution cache path",
    )
    _equal(scientific_cache.get("sha256"), CACHE_PROTOCOL_SHA256, "scientific cache SHA")
    _equal(
        scientific_lineage.get("cache_inventory_go_seal_sha256"),
        CACHE_INVENTORY_GO_SEAL_SHA256,
        "scientific inventory GO seal",
    )
    scientific_core = _mapping(
        scientific_lineage["hardened_safety_core"], "scientific core lineage"
    )
    _equal(_project_path(str(scientific_core["path"])), HARDENED_CORE_PATH, "scientific core path")
    _equal(scientific_core.get("sha256"), HARDENED_CORE_SHA256, "scientific core SHA")
    _equal(scientific_core.get("monkeypatch_forbidden"), True, "scientific monkeypatch policy")
    selector_link = _mapping(scientific["method"]["selector"], "selector link")
    selector_path = _project_path(str(selector_link["path"]))
    _equal(_stdlib_sha256(selector_path), SELECTOR_SHA256, "selector actual SHA")
    materializer_link = _mapping(
        scientific_lineage["materializer"], "materializer link"
    )
    _equal(materializer_link.get("sha256"), MATERIALIZER_SHA256, "materializer SHA declaration")
    materializer_path = _project_path(str(materializer_link["path"]))
    _equal(
        _stdlib_sha256(materializer_path), MATERIALIZER_SHA256, "materializer actual SHA"
    )

    cache_root = _project_path(str(cache_link["root"]))
    _equal(
        cache_root,
        _project_path(str(cache_protocol["cache"]["root"])),
        "cache materialized root",
    )
    outputs = _mapping(execution.get("outputs"), "outputs")
    output_root = _project_path(str(outputs["root"]))
    publication_work_root = _project_path(str(outputs["publication_work_root"]))
    _equal(
        output_root,
        _project_path(str(scientific["outputs"]["root"])),
        "scientific/execution output root",
    )
    _equal(outputs.get("stage1_aggregate_directory"), "stage1/aggregate", "stage1 aggregate")
    _equal(outputs.get("stage2_aggregate_directory"), "final/aggregate", "final aggregate")
    _equal(
        outputs.get("update_activity_diagnostics_filename"),
        STRENGTH_DIAGNOSTICS_FILENAME,
        "update activity diagnostics filename",
    )
    _equal(
        scientific["outputs"].get("update_activity_diagnostics_filename"),
        STRENGTH_DIAGNOSTICS_FILENAME,
        "scientific update activity diagnostics filename",
    )
    for key in ("stage1_aggregate_directory", "stage2_aggregate_directory"):
        destination = (output_root / str(outputs[key])).resolve()
        if not destination.is_relative_to(output_root):
            raise CalibrationExecutionError(f"outputs.{key} escapes output root")
    _equal(
        publication_work_root,
        (output_root.parent / ".ss_calibration_v2_publication_work").resolve(),
        "publication work root",
    )
    if publication_work_root == output_root or publication_work_root.is_relative_to(
        output_root
    ):
        raise CalibrationExecutionError("publication work root must be outside output root")

    runtime = _mapping(execution.get("runtime_seal"), "runtime seal")
    _equal(
        runtime.get("inventory_go_seal_sha256"),
        CACHE_INVENTORY_GO_SEAL_SHA256,
        "runtime inventory seal",
    )
    configured_critical_paths = tuple(
        str(value)
        for value in _sequence(
            runtime.get("critical_code_paths"), "runtime critical code paths"
        )
    )
    _equal(
        configured_critical_paths,
        EXPECTED_CRITICAL_CODE_PATHS,
        "runtime critical code path set/order",
    )
    critical_paths = tuple(
        _project_path(value) for value in configured_critical_paths
    )
    if not critical_paths or len(critical_paths) != len(set(critical_paths)):
        raise CalibrationExecutionError("critical runtime paths must be unique/non-empty")
    for critical_path in critical_paths:
        if not critical_path.is_file() or critical_path.is_symlink():
            raise FileNotFoundError(f"critical runtime path missing/symlink: {critical_path}")

    dataset_contracts = _mapping(
        scientific["source_train_subsets"]["datasets"], "scientific datasets"
    )
    _equal(tuple(dataset_contracts), DATASETS, "dataset order")
    checkpoints: dict[str, Path] = {}
    for dataset in DATASETS:
        dataset_contract = _mapping(dataset_contracts[dataset], dataset)
        calibration = _mapping(
            dataset_contract["calibration_checkpoint"], f"{dataset} calibration checkpoint"
        )
        application = _mapping(
            dataset_contract["inherited_application_checkpoint"],
            f"{dataset} application checkpoint",
        )
        _equal(calibration.get("role"), "best_miou", f"{dataset} calibration role")
        _equal(application.get("role"), "best_pd", f"{dataset} application role")
        _equal(application.get("tuning_episodes"), 0, f"{dataset} best_pd tuning")
        checkpoint = _project_path(str(calibration["path"]))
        best_pd = _project_path(str(application["path"]))
        for role, candidate_path, expected_hash in (
            ("best_miou", checkpoint, calibration["sha256"]),
            ("best_pd", best_pd, application["sha256"]),
        ):
            if not candidate_path.is_file() or candidate_path.is_symlink():
                raise FileNotFoundError(f"{dataset} {role} missing/symlink")
            _equal(
                _stdlib_sha256(candidate_path), expected_hash, f"{dataset} {role} SHA"
            )
        checkpoints[dataset] = checkpoint

    launcher = _mapping(execution.get("launcher"), "launcher")
    for key, expected in (
        ("one_live_worker_per_physical_gpu", True),
        ("cross_launcher_physical_gpu_lease", "crash_recoverable_flock"),
        ("child_inherits_lease_fd_for_worker_lifetime", True),
        ("linux_parent_death_signal", "SIGTERM"),
        ("worker_parent_pid_guard", True),
        ("worker_stdout", "devnull_launcher_emits_single_receipt"),
        ("worker_stderr", "inherited_structured_progress"),
        ("stage_launcher_claim", "crash_recoverable_flock_scan_run_postverify"),
        ("child_workers_inherit_stage_launcher_claim_fd", True),
    ):
        _equal(launcher.get(key), expected, f"launcher.{key}")
    for key in ("failure_termination_timeout_seconds", "lease_wait_heartbeat_seconds"):
        value = launcher.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise CalibrationExecutionError(f"launcher.{key} must be positive")
    stage1_launcher = _mapping(launcher.get("stage1"), "launcher.stage1")
    for key, expected in (
        ("one_candidate_per_fresh_process", True),
        ("maximum_parallel_workers", 2),
        ("cells_per_process", 39),
        ("episodes_per_process", 2496),
        ("safe_resume_current_runtime_seal_only", True),
        ("invalid_or_old_seal_action", "fail_closed_without_delete"),
    ):
        _equal(stage1_launcher.get(key), expected, f"launcher.stage1.{key}")
    stage2_launcher = _mapping(launcher.get("stage2"), "launcher.stage2")
    for key, expected in (
        ("fresh_process_count", 2),
        ("each_process_runs_all_top3_sequentially", True),
        ("cells_per_process", 117),
        ("episodes_per_process", 7488),
        ("safe_resume_current_extended_runtime_seal_only", True),
        ("verify_stage1_aggregate_before_receipt_use", True),
        ("bind_receipt_in_process_entry_runtime_seal", True),
    ):
        _equal(stage2_launcher.get(key), expected, f"launcher.stage2.{key}")
    _equal(
        _project_path(str(launcher["physical_gpu_lease_directory"])),
        (output_root.parent / ".source_calibration_physical_gpu_leases").resolve(),
        "physical GPU lease directory",
    )
    _equal(
        launcher["stage2"]["top3_receipt"],
        "stage1/aggregate/stage1_ss_top3_receipt.json",
        "stage2 receipt path",
    )
    return CalibrationContract(
        execution_path=execution_path,
        execution=execution,
        scientific_path=scientific_path,
        scientific=scientific,
        cache_protocol_path=cache_protocol_path,
        cache_protocol=cache_protocol,
        cache_root=cache_root,
        output_root=output_root,
        publication_work_root=publication_work_root,
        critical_code_paths=critical_paths,
        checkpoints=checkpoints,
    )


def _verify_cache_context(
    contract: CalibrationContract, dataset: str
) -> tuple[CacheContext, tuple[BoundFile, ...], Mapping[str, Any]]:
    context = validate_cache_contract(
        PROJECT_ROOT,
        contract.cache_protocol_path,
        dataset,
        output_override=contract.cache_root / dataset,
    )
    verification = validate_materialization(context)
    _equal(verification.get("valid"), True, f"{dataset} cache validation")
    root = contract.cache_root / dataset
    anchors = _mapping(
        contract.execution["cache_protocol"]["artifacts"][dataset],
        f"{dataset} cache anchors",
    )
    paths = {
        "manifest": root / "manifest.json",
        "method": root / "method_input_manifest.json",
        "complete": root / "COMPLETE.json",
    }
    for role, key in (
        ("manifest", "manifest_sha256"),
        ("method", "method_input_manifest_sha256"),
        ("complete", "complete_sha256"),
    ):
        _equal(_stdlib_sha256(paths[role]), anchors[key], f"{dataset} {role} SHA")
    manifest = _load_json(paths["manifest"])
    method = _load_json(paths["method"])
    complete = _load_json(paths["complete"])
    _equal(manifest.get("protocol_sha256"), CACHE_PROTOCOL_SHA256, f"{dataset} protocol")
    _equal(manifest.get("dataset"), dataset, f"{dataset} manifest dataset")
    _equal(complete.get("dataset"), dataset, f"{dataset} COMPLETE dataset")
    _equal(complete.get("complete"), True, f"{dataset} completion")
    _equal(complete.get("cache_content_sha256"), anchors["cache_content_sha256"], f"{dataset} cache content")
    _equal(manifest.get("cache_content_sha256"), anchors["cache_content_sha256"], f"{dataset} manifest content")
    _equal(manifest.get("ordered_ids_sha256"), anchors["ordered_ids_sha256"], f"{dataset} ordered IDs")
    targets = _mapping(manifest.get("targets"), f"{dataset} targets")
    _equal(targets.get("file_sha256"), anchors["target_file_sha256"], f"{dataset} target file")
    _equal(
        targets.get("tensor_sequence_sha256"),
        anchors["target_tensor_sequence_sha256"],
        f"{dataset} target tensor lineage",
    )
    dataset_science = contract.scientific["source_train_subsets"]["datasets"][dataset]
    for key, manifest_key in (
        ("ordered_ids_sha256", "ordered_ids_sha256"),
        ("train_split_sha256", "train_split_sha256"),
        ("test_split_sha256", "test_split_sha256"),
        ("calibration_ids_file_sha256", "calibration_ids_file_sha256"),
    ):
        _equal(manifest.get(manifest_key), dataset_science[key], f"{dataset} {key}")
    _equal(method.get("outer_manifest_sha256"), anchors["manifest_sha256"], f"{dataset} method outer manifest")
    _equal(method.get("targets_exposed"), False, f"{dataset} method target firewall")
    _equal(tuple(method.get("sample_fields", ())), METHOD_FIELDS, f"{dataset} method fields")
    _equal(set(method.get("files", {})), set(manifest["files"]) - {targets["path"]}, f"{dataset} method image shards")
    condition_by_pair: dict[tuple[str, int], Mapping[str, Any]] = {}
    for index, ((corruption, severity), raw) in enumerate(
        zip(CONDITIONS, manifest["conditions"], strict=True)
    ):
        record = _mapping(raw, f"{dataset} condition {index}")
        _equal((record["corruption"], record["severity"]), (corruption, severity), f"{dataset} condition order")
        condition_by_pair[(corruption, severity)] = record
    bindings: list[BoundFile] = [
        core._bound_file(paths["manifest"], role=f"cache_manifest:{dataset}", expected_sha256=anchors["manifest_sha256"]),
        core._bound_file(paths["method"], role=f"cache_method_input_manifest:{dataset}", expected_sha256=anchors["method_input_manifest_sha256"]),
        core._bound_file(paths["complete"], role=f"cache_complete:{dataset}", expected_sha256=anchors["complete_sha256"]),
    ]
    for relative, file_record in sorted(manifest["files"].items()):
        bindings.append(
            core._bound_file(
                root / relative,
                role=(
                    f"cache_target_payload:{dataset}"
                    if relative == targets["path"]
                    else f"cache_condition_payload:{dataset}:{relative}"
                ),
                expected_sha256=file_record["sha256"],
            )
        )
    lineage = {
        "inventory_go_seal_sha256": CACHE_INVENTORY_GO_SEAL_SHA256,
        "protocol_sha256": CACHE_PROTOCOL_SHA256,
        "manifest_sha256": anchors["manifest_sha256"],
        "method_input_manifest_sha256": anchors["method_input_manifest_sha256"],
        "complete_sha256": anchors["complete_sha256"],
        "cache_content_sha256": anchors["cache_content_sha256"],
        "ordered_ids_sha256": anchors["ordered_ids_sha256"],
        "target_file_sha256": anchors["target_file_sha256"],
        "target_tensor_sequence_sha256": anchors["target_tensor_sequence_sha256"],
        "payload_file_count": len(manifest["files"]),
        "all_payload_hashes_verified": True,
        "method_manifest_semantically_equals_outer_projection": True,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "method_received_labels": False,
    }
    return (
        CacheContext(
            dataset=dataset,
            root=root,
            manifest=manifest,
            complete=complete,
            manifest_sha256=anchors["manifest_sha256"],
            complete_sha256=anchors["complete_sha256"],
            condition_by_pair=condition_by_pair,
        ),
        tuple(bindings),
        lineage,
    )


def capture_runtime_seal(
    contract: CalibrationContract,
) -> tuple[RuntimeSeal, Mapping[str, CacheContext]]:
    bindings: list[BoundFile] = [
        core._bound_file(contract.execution_path, role="execution_config"),
        core._bound_file(contract.scientific_path, role="scientific_config"),
        core._bound_file(contract.cache_protocol_path, role="cache_protocol", expected_sha256=CACHE_PROTOCOL_SHA256),
    ]
    bindings.extend(
        core._bound_file(path, role=f"critical_code:{path.relative_to(PROJECT_ROOT)}")
        for path in contract.critical_code_paths
    )
    dataset_science = contract.scientific["source_train_subsets"]["datasets"]
    for dataset in DATASETS:
        values = dataset_science[dataset]
        for role, section in (
            ("checkpoint_best_miou", values["calibration_checkpoint"]),
            ("checkpoint_best_pd_zero_tuning_application", values["inherited_application_checkpoint"]),
        ):
            bindings.append(
                core._bound_file(
                    _project_path(section["path"]),
                    role=f"{role}:{dataset}",
                    expected_sha256=section["sha256"],
                )
            )
        for role, path_key, hash_key in (
            ("source_train_split", "train_split", "train_split_sha256"),
            ("source_test_split_metadata", "test_split", "test_split_sha256"),
            ("parent_pilot_ids", "calibration_ids", "calibration_ids_file_sha256"),
        ):
            bindings.append(
                core._bound_file(
                    _project_path(values[path_key]),
                    role=f"{role}:{dataset}",
                    expected_sha256=values[hash_key],
                )
            )
    for role, link in (
        ("parent_pilot_protocol", contract.scientific["lineage"]["train_side_pilot_protocol"]),
        ("parent_pilot_manifest", contract.scientific["lineage"]["train_side_pilot_manifest"]),
    ):
        bindings.append(
            core._bound_file(
                _project_path(link["path"]), role=role, expected_sha256=link["sha256"]
            )
        )
    caches: dict[str, CacheContext] = {}
    cache_lineage: dict[str, Any] = {
        "protocol": "SS_selection_only",
        "selection_bn_protocol": SS_BN_PROTOCOL,
        "BS_excluded_from_selection": True,
        "diagnostic_detail": "global",
        "require_nonzero_parameter_update": False,
        "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
        "update_activity": _update_activity_disclosure(),
        "best_pd_reuses_same_frozen_hyperparameters_without_tuning": True,
        "best_pd_tuning_episodes": 0,
        "inventory_go_seal_sha256": CACHE_INVENTORY_GO_SEAL_SHA256,
        "datasets": {},
    }
    for dataset in DATASETS:
        context, cache_bindings, lineage = _verify_cache_context(contract, dataset)
        caches[dataset] = context
        cache_lineage["datasets"][dataset] = lineage
        bindings.extend(cache_bindings)
    seal = core._runtime_seal_from_bindings(bindings, cache_lineage)
    return seal, caches


def _emit_cell_progress(
    *,
    stage: int,
    process_id: str,
    candidate: Candidate,
    dataset: str,
    corruption: str,
    severity: int,
    completed_cells: int,
) -> None:
    event = {
        "event": "binary_tent_ss_calibration_v2_cell_complete",
        "stage": int(stage),
        "process_id": process_id,
        "candidate": candidate.to_dict(),
        "dataset": dataset,
        "bn_protocol": SS_BN_PROTOCOL,
        "corruption": corruption,
        "severity": int(severity),
        "cell_images": IMAGES_PER_CELL,
        "completed_cells_for_candidate": int(completed_cells),
        "total_cells_for_candidate": CELLS_PER_RUN,
        "completed_episodes_for_candidate": int(completed_cells) * IMAGES_PER_CELL,
    }
    print(
        json.dumps(
            event,
            sort_keys=True,
            separators=JSON_SEPARATORS,
            allow_nan=False,
        ),
        file=sys.stderr,
        flush=True,
    )


_STRENGTH_METRICS = (
    "changed_bn_affine_tensor_count",
    "mean_absolute_probability_delta",
    "maximum_absolute_probability_delta",
    "strict_threshold_changed_pixel_ratio",
    "foreground_probability_mass_delta",
    "gradient_norm",
    "step_norm",
    "relative_step_norm",
)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationExecutionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationExecutionError(f"{label} must be finite")
    return result


def _metric_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise CalibrationExecutionError("strength diagnostic metric cannot be empty")
    clean = [_finite_float(value, "strength diagnostic metric") for value in values]
    total = math.fsum(clean)
    return {
        "sum": total,
        "mean": total / len(clean),
        "min": min(clean),
        "max": max(clean),
    }


def _audit_episode_v2(
    result: Any,
    *,
    bn_protocol: str,
    expect_full_audit: bool,
) -> dict[str, bool]:
    """Apply the hardened audit plus the v2 optimizer-state zero-update policy."""

    gates = core._audit_episode(
        result,
        bn_protocol=bn_protocol,
        expect_full_audit=expect_full_audit,
    )
    diagnostics = _mapping(result.diagnostics, "v2 episode diagnostics")
    checks = _mapping(result.checks, "v2 episode checks")
    changed = diagnostics.get("changed_bn_affine_tensors_fast_gate")
    method_changed = diagnostics.get("number_updated_bn_affine_tensors")
    for value, label in (
        (changed, "fast changed BN affine count"),
        (method_changed, "method changed BN affine count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CalibrationExecutionError(f"{label} must be a nonnegative integer")
    _equal(method_changed, changed, "fast/method changed BN affine count")
    _equal(
        diagnostics.get("require_nonzero_parameter_update"),
        False,
        "v2 allow-zero runner policy",
    )
    _equal(
        diagnostics.get("zero_parameter_update_allowed"),
        True,
        "v2 zero-update allowance",
    )
    _equal(
        diagnostics.get("zero_parameter_update_policy"),
        ZERO_UPDATE_POLICY,
        "v2 zero-update policy diagnostic",
    )
    _equal(
        diagnostics.get("zero_parameter_update_observed"),
        changed == 0,
        "v2 zero-update observation",
    )
    _equal(
        diagnostics.get("numerically_zero_parameter_delta"),
        changed == 0,
        "v2 numerical-zero diagnostic",
    )
    _equal(
        diagnostics.get("actual_parameter_delta_nonzero"),
        changed > 0,
        "v2 parameter-delta diagnostic",
    )
    _equal(
        checks.get("bn_affine_changed_by_one_step"),
        changed > 0,
        "v2 BN affine change check",
    )
    _equal(
        checks.get("optimizer_changed_by_one_step"),
        True,
        "v2 optimizer temporary-state check",
    )
    optimizer_state_count = diagnostics.get(
        "temporary_optimizer_state_parameter_count"
    )
    if (
        isinstance(optimizer_state_count, bool)
        or not isinstance(optimizer_state_count, int)
        or optimizer_state_count <= 0
    ):
        raise CalibrationExecutionError(
            "v2 zero-update policy requires temporary optimizer state"
        )
    numeric_diagnostics: dict[str, float] = {}
    for key in (
        "gradient_norm",
        "step_norm",
        "source_parameter_norm",
        "relative_step_norm",
    ):
        value = _finite_float(diagnostics.get(key), f"v2 episode {key}")
        if value < 0.0:
            raise CalibrationExecutionError(f"v2 episode {key} must be nonnegative")
        numeric_diagnostics[key] = value
    _equal(
        diagnostics.get("relative_step_norm_epsilon"),
        1e-12,
        "v2 relative-step epsilon",
    )
    expected_relative_step_norm = numeric_diagnostics["step_norm"] / (
        numeric_diagnostics["source_parameter_norm"] + 1e-12
    )
    _equal(
        numeric_diagnostics["relative_step_norm"],
        expected_relative_step_norm,
        "v2 relative-step diagnostic definition",
    )
    if changed == 0 and numeric_diagnostics["step_norm"] != 0.0:
        raise CalibrationExecutionError(
            "zero changed tensors requires an exactly zero parameter step norm"
        )
    if diagnostics.get("optimizer_steps") != 1 or diagnostics.get("finite") is not True:
        raise CalibrationExecutionError(
            "v2 zero-update policy requires one finite optimizer step"
        )
    if not all(value is True for value in gates.values()):
        raise CalibrationExecutionError("v2 hardened episode gate failed")
    return gates


def _episode_strength_diagnostic(result: Any) -> dict[str, Any]:
    """Derive label-free update strength before any outer target tensor is loaded."""

    diagnostics = _mapping(result.diagnostics, "episode strength diagnostics")
    changed = int(diagnostics["changed_bn_affine_tensors_fast_gate"])
    pre = probabilities_from_logits(result.logits_tent_pre)
    post = probabilities_from_logits(result.logits_tent_post)
    _equal(pre.shape, post.shape, "strength probability shape")
    delta = post - pre
    absolute = np.abs(delta)
    changed_pixels = np.not_equal(pre > 0.5, post > 0.5)
    values = {
        "changed_bn_affine_tensor_count": changed,
        "mean_absolute_probability_delta": float(absolute.mean()),
        "maximum_absolute_probability_delta": float(absolute.max()),
        "strict_threshold_changed_pixel_ratio": float(changed_pixels.mean()),
        "foreground_probability_mass_delta": float(delta.sum()),
        "gradient_norm": _finite_float(diagnostics.get("gradient_norm"), "gradient norm"),
        "step_norm": _finite_float(diagnostics.get("step_norm"), "step norm"),
        "relative_step_norm": _finite_float(
            diagnostics.get("relative_step_norm"), "relative step norm"
        ),
    }
    for key, value in values.items():
        _finite_float(value, f"episode strength {key}")
    return values


def _cell_strength_diagnostic(
    *,
    stage: int,
    process_id: str,
    candidate: Candidate,
    dataset: str,
    corruption: str,
    severity: int,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _equal(len(episodes), IMAGES_PER_CELL, "strength episodes per cell")
    changed_counts = [int(value["changed_bn_affine_tensor_count"]) for value in episodes]
    histogram: dict[str, int] = {}
    for count in changed_counts:
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    zero = sum(count == 0 for count in changed_counts)
    result: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "process_id": process_id,
        "candidate": candidate.to_dict(),
        "dataset": dataset,
        "bn_protocol": SS_BN_PROTOCOL,
        "corruption": corruption,
        "severity": severity,
        "episode_count": IMAGES_PER_CELL,
        "update_activity_contract": _update_activity_disclosure(),
        "parameter_delta_episode_counts": {
            "zero": zero,
            "nonzero": IMAGES_PER_CELL - zero,
            "total": IMAGES_PER_CELL,
        },
        "changed_bn_affine_tensor_count_histogram": dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
    }
    for key in _STRENGTH_METRICS:
        result[key] = _metric_stats(
            [float(value[key]) for value in episodes]
        )
    return result


def _combine_strength_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not diagnostics:
        raise CalibrationExecutionError("strength diagnostics cannot be empty")
    episode_count = sum(int(value["episode_count"]) for value in diagnostics)
    zero = sum(int(value["parameter_delta_episode_counts"]["zero"]) for value in diagnostics)
    nonzero = sum(
        int(value["parameter_delta_episode_counts"]["nonzero"])
        for value in diagnostics
    )
    _equal(zero + nonzero, episode_count, "strength run episode counts")
    metrics: dict[str, dict[str, float]] = {}
    for key in _STRENGTH_METRICS:
        cell_values = [_mapping(value[key], f"strength {key}") for value in diagnostics]
        total = math.fsum(float(value["sum"]) for value in cell_values)
        metrics[key] = {
            "sum": total,
            "mean": total / episode_count,
            "min": min(float(value["min"]) for value in cell_values),
            "max": max(float(value["max"]) for value in cell_values),
        }
    return {
        "update_activity_contract": _update_activity_disclosure(),
        "cell_count": len(diagnostics),
        "episode_count": episode_count,
        "parameter_delta_episode_counts": {
            "zero": zero,
            "nonzero": nonzero,
            "total": episode_count,
        },
        "cells_with_zero_parameter_delta_episodes": sum(
            int(value["parameter_delta_episode_counts"]["zero"]) > 0
            for value in diagnostics
        ),
        "metrics": metrics,
    }


def _build_fast_runner_v2(
    *,
    contract: CalibrationContract,
    dataset: str,
    candidate: Candidate,
    bn_protocol: str,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build one frozen SS runner without the incompatible v3 diagnostic value."""

    # All model/custom-op objects remain lazy so validate-only cannot construct
    # a model or touch a CUDA device.
    import test_source as source_runner
    from tta.binary_tent import (
        BinaryTentMethod,
        CUDA_BACKWARD_TEMPORARILY_DISABLE,
    )
    from tta.binary_tent_fast_runner_v2 import (
        RELATIVE_STEP_NORM_EPSILON as RUNNER_RELATIVE_STEP_NORM_EPSILON,
        ZERO_UPDATE_POLICY as RUNNER_ZERO_UPDATE_POLICY,
        BinaryTentFastRunnerV2,
    )
    from tta.model_adapter import IRSTDModelAdapter
    from tta.state_manager import EpisodicStateManager

    _equal(bn_protocol, SS_BN_PROTOCOL, "v2 builder SS protocol")
    _equal(
        RUNNER_ZERO_UPDATE_POLICY,
        ZERO_UPDATE_POLICY,
        "v2 runner/calibration zero-update policy",
    )
    _equal(
        RUNNER_RELATIVE_STEP_NORM_EPSILON,
        1e-12,
        "v2 runner/calibration relative-step epsilon",
    )
    diagnostic_detail = str(contract.scientific["method"]["diagnostic_detail"])
    _equal(diagnostic_detail, "global", "v2 builder diagnostic detail")
    _equal(
        contract.execution["execution"]["diagnostic_detail"],
        diagnostic_detail,
        "scientific/execution diagnostic detail",
    )
    require_nonzero_parameter_update = contract.scientific["method"][
        "require_nonzero_parameter_update"
    ]
    _equal(
        require_nonzero_parameter_update,
        False,
        "v2 builder zero-update policy",
    )
    _equal(
        contract.execution["execution"]["require_nonzero_parameter_update"],
        require_nonzero_parameter_update,
        "scientific/execution zero-update policy",
    )
    model = source_runner.build_nsfpn_model()
    checkpoint = contract.checkpoints[dataset]
    wrapper = source_runner.load_trusted_checkpoint(model, checkpoint)
    model.to(device)
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name=candidate.optimizer,
        learning_rate=float(candidate.learning_rate),
        bn_protocol=bn_protocol,
        entropy_eps=float(contract.scientific["method"]["entropy_eps"]),
        diagnostic_detail=diagnostic_detail,
        cuda_backward_determinism_policy=CUDA_BACKWARD_TEMPORARILY_DISABLE,
    )
    state = EpisodicStateManager(model, optimizer=method.optimizer)
    fast_runner = BinaryTentFastRunnerV2(
        adapter,
        state,
        method,
        full_audit_cadence=64,
        require_nonzero_parameter_update=require_nonzero_parameter_update,
    )
    return fast_runner, {
        "build_instance_id": uuid.uuid4().hex,
        "model_object_id": id(model),
        "method_object_id": id(method),
        "optimizer_object_id": id(method.optimizer),
        "source_state_sha256": state.source_fingerprint.full_sha256,
        "checkpoint_wrapper": wrapper,
        "checkpoint_sha256": _stdlib_sha256(checkpoint),
        "checkpoint_state_dict_strict_load_verified": True,
        "checkpoint_matches_dataset": dataset,
        "diagnostic_detail": diagnostic_detail,
        "require_nonzero_parameter_update": require_nonzero_parameter_update,
        "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
    }


def execute_candidate(
    *,
    contract: CalibrationContract,
    caches: Mapping[str, CacheContext],
    monitor: RuntimeSealMonitor,
    stage: int,
    process_id: str,
    candidate: Candidate,
    device: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Execute exactly 3 datasets x 13 conditions under SS, never BS."""

    seed_audit = core._seed_candidate(42)
    records: list[dict[str, Any]] = []
    strength_diagnostics: list[dict[str, Any]] = []
    build_receipts: list[dict[str, Any]] = []
    started = time.perf_counter()
    for dataset in DATASETS:
        context = caches[dataset]
        runner, build = _build_fast_runner_v2(
            contract=contract,
            dataset=dataset,
            candidate=candidate,
            bn_protocol=SS_BN_PROTOCOL,
            device=device,
        )
        build_receipts.append(
            {
                "dataset": dataset,
                "bn_protocol": SS_BN_PROTOCOL,
                **build,
                "fresh_model_method_optimizer": True,
            }
        )
        for corruption, severity in CONDITIONS:
            condition = context.condition_by_pair[(corruption, severity)]
            active_image_path = context.root / str(condition["path"])
            target_path = context.root / str(context.manifest["targets"]["path"])
            monitor.assert_unchanged(
                stage=(
                    f"candidate:{candidate_slug(candidate)}:{dataset}:SS:"
                    f"{corruption}_S{severity}:condition_start"
                ),
                active_paths=(active_image_path,),
            )
            method_inputs = SourceCalibrationMethodInputDatasetV2(
                context.root,
                condition_key=f"{corruption}_S{severity}",
                expected_protocol_sha256=CACHE_PROTOCOL_SHA256,
            )
            evaluation_protocol = IRSTDEvaluationProtocol(
                fixed_probability_threshold=0.5,
                froc_probability_thresholds=(0.5,),
                connectivity=2,
                max_centroid_distance=3.0,
                min_component_area=1,
            )
            pre_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
            post_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
            cell_start_episode = runner.completed_episodes
            returned_predictions: list[tuple[Any, Any]] = []
            episode_strength: list[dict[str, Any]] = []
            cell_gate_evidence = {
                key: True for key in REQUIRED_HARD_GATES if key != "zero_test_opens"
            }
            for index in range(len(method_inputs)):
                sample = method_inputs[index]
                _equal(tuple(sample), METHOD_FIELDS, "method sample exact fields")
                image = sample.pop("image").unsqueeze(0)
                _equal(tuple(image.shape), (1, 3, 256, 256), "private image shape")
                result = core.run_label_free_episode(
                    runner,
                    image=image.to(device, non_blocking=False),
                    metadata=dict(sample),
                )
                episode_gates = _audit_episode_v2(
                    result,
                    bn_protocol=SS_BN_PROTOCOL,
                    expect_full_audit=index == IMAGES_PER_CELL - 1,
                )
                for key, passed in episode_gates.items():
                    cell_gate_evidence[key] = (
                        cell_gate_evidence.get(key, True) and passed is True
                    )
                returned_predictions.append(
                    (result.logits_tent_pre, result.logits_tent_post)
                )
                episode_strength.append(_episode_strength_diagnostic(result))
            _equal(len(returned_predictions), IMAGES_PER_CELL, "returned predictions")
            _equal(
                runner.completed_episodes - cell_start_episode,
                IMAGES_PER_CELL,
                "cell episode count",
            )
            # This target-free diagnostic is finalized before the outer target
            # loader is called and is never passed to either selector.
            strength_diagnostics.append(
                _cell_strength_diagnostic(
                    stage=stage,
                    process_id=process_id,
                    candidate=candidate,
                    dataset=dataset,
                    corruption=corruption,
                    severity=severity,
                    episodes=episode_strength,
                )
            )
            complete = context.complete
            counters = {
                "test_image_opens": int(complete["test_images_opened"]),
                "test_label_opens": int(complete["test_masks_opened"]),
                "method_label_accesses": int(bool(complete["method_received_labels"])),
            }
            cell_gate_evidence["zero_test_opens"] = all(
                value == 0 for value in counters.values()
            )
            monitor.assert_unchanged(
                stage=(
                    f"candidate:{candidate_slug(candidate)}:{dataset}:SS:"
                    f"{corruption}_S{severity}:episodes_complete_pre_outer_evaluation"
                ),
                active_paths=(active_image_path, target_path),
            )
            outer_targets = load_outer_evaluator_targets_v2(
                context.root,
                expected_protocol_sha256=CACHE_PROTOCOL_SHA256,
                episodes_complete=True,
            )
            for index, (tent_pre, tent_post) in enumerate(returned_predictions):
                outer_target = np.array(
                    outer_targets[index], dtype=np.float32, copy=True, order="C"
                )
                pre_evaluator.update_logits(tent_pre, outer_target)
                post_evaluator.update_logits(tent_post, outer_target)
            protocol_audit = {
                "candidate_model_method_optimizer_rebuilt_before_run": bool(
                    build_receipts[-1]["fresh_model_method_optimizer"]
                ),
                "global_runtime_seal_valid": bool(monitor.audits)
                and all(value.get("verified") is True for value in monitor.audits),
                "fixed_seed_contract_valid": all(seed_audit.values()),
            }
            records.append(
                core.build_cell_record(
                    stage=stage,
                    process_id=process_id,
                    candidate=candidate,
                    dataset=dataset,
                    bn_protocol=SS_BN_PROTOCOL,
                    corruption=corruption,
                    severity=severity,
                    tent_pre=core._endpoint_counts(pre_evaluator),
                    tent_post=core._endpoint_counts(post_evaluator),
                    hard_gates=cell_gate_evidence,
                    protocol_audit=protocol_audit,
                )
            )
            _emit_cell_progress(
                stage=stage,
                process_id=process_id,
                candidate=candidate,
                dataset=dataset,
                corruption=corruption,
                severity=severity,
                completed_cells=len(records),
            )
            del (
                method_inputs,
                outer_targets,
                returned_predictions,
                episode_strength,
                pre_evaluator,
                post_evaluator,
            )
        _equal(runner.completed_episodes, len(CONDITIONS) * IMAGES_PER_CELL, "SS runner episodes")
        runner.state.assert_source_state()
        del runner
    _equal(len(records), CELLS_PER_RUN, "SS candidate record count")
    _equal(len(strength_diagnostics), CELLS_PER_RUN, "SS strength record count")
    build_ids = [value["build_instance_id"] for value in build_receipts]
    _equal(len(set(build_ids)), len(DATASETS), "fresh dataset build identities")
    return records, strength_diagnostics, {
        "candidate": candidate.to_dict(),
        "selection_bn_protocol": "SS",
        "selection_bn_protocol_id": SS_BN_PROTOCOL,
        "BS_excluded_from_selection": True,
        "diagnostic_detail": "global",
        "require_nonzero_parameter_update": False,
        "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
        "update_activity_diagnostics": _combine_strength_diagnostics(
            strength_diagnostics
        ),
        "cell_record_count": CELLS_PER_RUN,
        "episode_count": CELLS_PER_RUN * IMAGES_PER_CELL,
        "model_method_optimizer_build_count": len(DATASETS),
        "fresh_build_object_id_triples_unique": True,
        "checkpoint_loads": [
            {
                "dataset": value["dataset"],
                "bn_protocol": value["bn_protocol"],
                "checkpoint_sha256": value["checkpoint_sha256"],
                "checkpoint_wrapper": value["checkpoint_wrapper"],
                "checkpoint_state_dict_strict_load_verified": value[
                    "checkpoint_state_dict_strict_load_verified"
                ],
                "checkpoint_matches_dataset": value["checkpoint_matches_dataset"],
                "diagnostic_detail": value["diagnostic_detail"],
                "require_nonzero_parameter_update": value[
                    "require_nonzero_parameter_update"
                ],
                "zero_parameter_update_policy": value[
                    "zero_parameter_update_policy"
                ],
            }
            for value in build_receipts
        ],
        "fixed_seed_audit": seed_audit,
        "runtime_seconds": time.perf_counter() - started,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "target_access_audit": _target_access_audit(
            tensor_evaluation_completed=True
        ),
    }


def _decimal_text(value: Any) -> str:
    return format(value.normalize(), "E").replace("E+", "e+").replace("E-", "e-")


def candidate_slug(candidate: Candidate) -> str:
    lr = _decimal_text(candidate.learning_rate).replace("-", "m").replace("+", "p")
    return f"{candidate.optimizer}_lr_{lr}"


def _candidate_from_args(optimizer: str, learning_rate: str) -> Candidate:
    return Candidate.from_values(optimizer, learning_rate)


def _stage1_shard_path(contract: CalibrationContract, candidate: Candidate) -> Path:
    return contract.output_root / "stage1" / "shards" / candidate_slug(candidate)


def _stage1_aggregate_path(contract: CalibrationContract) -> Path:
    return contract.output_root / "stage1" / "aggregate"


def _stage1_receipt_path(contract: CalibrationContract) -> Path:
    return _stage1_aggregate_path(contract) / "stage1_ss_top3_receipt.json"


def _stage2_shard_path(contract: CalibrationContract, process_id: str) -> Path:
    return contract.output_root / "stage2" / "shards" / core._safe_slug(process_id)


def _stage2_slot_process_id(receipt_sha256: str, slot_index: int) -> str:
    return core._stage2_slot_process_id(receipt_sha256, slot_index)


def _load_top3(path: Path) -> tuple[Candidate, ...]:
    receipt = _load_json(path)
    _equal(receipt.get("receipt_type"), "stage1_ss_top3", "stage1 receipt type")
    _equal(receipt.get("selection_bn_protocol"), "SS", "receipt selection protocol")
    _equal(receipt.get("BS_excluded_from_selection"), True, "receipt BS exclusion")
    values = tuple(
        Candidate.from_values(value["optimizer"], value["learning_rate"])
        for value in _sequence(receipt.get("top3"), "stage1 top3")
    )
    if len(values) != 3 or len(set(values)) != 3:
        raise CalibrationExecutionError("stage1 receipt must contain three unique candidates")
    return values


def _emit_worker_start(
    *, stage: int, process_id: str, gpu_lease: Mapping[str, Any]
) -> None:
    print(
        json.dumps(
            {
                "event": "binary_tent_ss_calibration_v2_worker_start",
                "stage": stage,
                "process_id": process_id,
                "pid": os.getpid(),
                "physical_gpu_id": gpu_lease["physical_gpu_id"],
                "lease_path": gpu_lease["lease_path"],
                "unix_time_ns": time.time_ns(),
            },
            sort_keys=True,
            separators=JSON_SEPARATORS,
        ),
        file=sys.stderr,
        flush=True,
    )


def run_stage1_worker(args: argparse.Namespace) -> dict[str, Any]:
    inherited = core._verify_inherited_gpu_lease(required=True)
    contract = load_contract(args.execution_config)
    gpu_lease = core._validate_formal_gpu_lease(
        contract, inherited, device_text=args.device
    )
    candidate = _candidate_from_args(args.optimizer, args.learning_rate)
    process_id = core._safe_slug(args.process_id)
    _emit_worker_start(stage=1, process_id=process_id, gpu_lease=gpu_lease)
    destination = _stage1_shard_path(contract, candidate)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"stage1 candidate shard exists: {destination}")
    seal, caches = capture_runtime_seal(contract)
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(stage="stage1_process_entry")
    device = core._configure_cuda_worker(contract, args.device)
    worker_environment = core._observed_worker_environment(device)
    records, strength_diagnostics, summary = execute_candidate(
        contract=contract,
        caches=caches,
        monitor=monitor,
        stage=1,
        process_id=process_id,
        candidate=candidate,
        device=device,
    )
    summary.update(
        {
            "stage": 1,
            "process_id": process_id,
            "fresh_process": True,
            "worker_environment": worker_environment,
            "physical_gpu_lease": gpu_lease,
        }
    )
    monitor.assert_unchanged(stage="stage1_pre_publish", full_byte_rehash=True)
    provenance = {
        "schema_version": 2,
        "stage": 1,
        "fresh_process": True,
        "process_id": process_id,
        "pid": os.getpid(),
        "candidate": candidate.to_dict(),
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "runtime_audits": monitor.audits,
        "worker_environment": worker_environment,
        "physical_gpu_lease": gpu_lease,
        "target_access_audit": _target_access_audit(tensor_evaluation_completed=True),
        **_scope_provenance(contract),
    }
    return core._publish_directory(
        final=destination,
        work_root=contract.publication_work_root,
        primary_files={
            "records.jsonl": records,
            STRENGTH_DIAGNOSTICS_FILENAME: strength_diagnostics,
            "run_summary.json": summary,
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_ss_calibration_v2_stage1_candidate_shard",
            "stage": 1,
            "fresh_process": True,
            "process_id": process_id,
            "candidate": candidate.to_dict(),
            "selection_bn_protocol": "SS",
            "BS_excluded_from_selection": True,
            "diagnostic_detail": "global",
            "require_nonzero_parameter_update": False,
            "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
            "update_activity_contract": _update_activity_disclosure(),
            "record_count": 39,
            "episode_count": 2496,
            "worker_environment": worker_environment,
            "physical_gpu_lease": gpu_lease,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            **_scope_provenance(contract),
        },
        completion_metadata={
            "stage": 1,
            "process_id": process_id,
            "candidate": candidate.to_dict(),
            "record_count": 39,
            "episode_count": 2496,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "scope": dict(SCOPE),
            "selection_bn_protocol": "SS",
            "BS_excluded_from_selection": True,
            "diagnostic_detail": "global",
            "require_nonzero_parameter_update": False,
            "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
            "update_activity_contract": _update_activity_disclosure(),
        },
    )


def _bind_verified_stage1_receipt(
    contract: CalibrationContract,
    base_seal: RuntimeSeal,
    *,
    stage1_verified: Mapping[str, Any],
) -> tuple[RuntimeSeal, BoundFile, dict[str, Any]]:
    receipt_path = _stage1_receipt_path(contract)
    file_record = _mapping(stage1_verified["manifest"]["files"][receipt_path.name], "receipt file record")
    binding = core._bound_file(
        receipt_path,
        role="stage1_top3_receipt:ss_v2",
        expected_sha256=str(file_record["sha256"]),
    )
    _equal(binding.bytes, int(file_record["bytes"]), "receipt bytes")
    seal = core._extend_runtime_seal(base_seal, binding)
    reverified = _verify_stage1_aggregate(contract, base_seal)
    _equal(
        reverified["manifest_sha256"],
        stage1_verified["manifest_sha256"],
        "stage1 verify/bind/reverify identity",
    )
    rebound = core._bound_file(
        receipt_path,
        role="stage1_top3_receipt:ss_v2",
        expected_sha256=binding.sha256,
    )
    _equal(rebound, binding, "stage1 receipt stable identity")
    return seal, binding, reverified


def run_stage2_worker(args: argparse.Namespace) -> dict[str, Any]:
    inherited = core._verify_inherited_gpu_lease(required=True)
    contract = load_contract(args.execution_config)
    gpu_lease = core._validate_formal_gpu_lease(
        contract, inherited, device_text=args.device
    )
    process_id = core._safe_slug(args.process_id)
    slot_index = int(args.slot_index)
    _emit_worker_start(stage=2, process_id=process_id, gpu_lease=gpu_lease)
    destination = _stage2_shard_path(contract, process_id)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"stage2 process shard exists: {destination}")
    expected_receipt = _stage1_receipt_path(contract).resolve()
    requested_receipt = Path(args.top3_receipt).expanduser().resolve()
    _equal(requested_receipt, expected_receipt, "canonical stage1 receipt path")
    base_seal, caches = capture_runtime_seal(contract)
    stage1_verified = _verify_stage1_aggregate(contract, base_seal)
    seal, receipt_binding, _ = _bind_verified_stage1_receipt(
        contract, base_seal, stage1_verified=stage1_verified
    )
    _equal(
        process_id,
        _stage2_slot_process_id(receipt_binding.sha256, slot_index),
        "deterministic receipt-bound process ID",
    )
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(
        stage="stage2_process_entry", active_paths=(expected_receipt,)
    )
    top3 = _load_top3(expected_receipt)
    device = core._configure_cuda_worker(contract, args.device)
    worker_environment = core._observed_worker_environment(device)
    records: list[dict[str, Any]] = []
    strength_diagnostics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for rank, candidate in enumerate(top3, start=1):
        monitor.assert_unchanged(stage=f"stage2_candidate_{rank}_pre_rebuild")
        candidate_records, candidate_strength, summary = execute_candidate(
            contract=contract,
            caches=caches,
            monitor=monitor,
            stage=2,
            process_id=process_id,
            candidate=candidate,
            device=device,
        )
        records.extend(candidate_records)
        strength_diagnostics.extend(candidate_strength)
        summaries.append({"frozen_top3_rank": rank, **summary})
        monitor.assert_unchanged(stage=f"stage2_candidate_{rank}_complete")
    _equal(len(records), 117, "stage2 process record count")
    _equal(len(strength_diagnostics), 117, "stage2 process strength record count")
    monitor.assert_unchanged(stage="stage2_pre_publish", full_byte_rehash=True)
    summary = {
        "stage": 2,
        "process_id": process_id,
        "slot_index": slot_index,
        "fresh_process": True,
        "selection_bn_protocol": "SS",
        "BS_excluded_from_selection": True,
        "diagnostic_detail": "global",
        "require_nonzero_parameter_update": False,
        "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
        "update_activity_diagnostics": _combine_strength_diagnostics(
            strength_diagnostics
        ),
        "top3_in_frozen_order": [value.to_dict() for value in top3],
        "candidate_summaries": summaries,
        "record_count": 117,
        "episode_count": 7488,
        "candidate_model_method_optimizer_rebuilt_before_each_candidate": True,
        "worker_environment": worker_environment,
        "physical_gpu_lease": gpu_lease,
        "target_access_audit": _target_access_audit(tensor_evaluation_completed=True),
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
    }
    provenance = {
        "schema_version": 2,
        "stage": 2,
        "fresh_process": True,
        "process_id": process_id,
        "slot_index": slot_index,
        "pid": os.getpid(),
        "top3_receipt": str(expected_receipt),
        "top3_receipt_sha256": receipt_binding.sha256,
        "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "runtime_audits": monitor.audits,
        "worker_environment": worker_environment,
        "physical_gpu_lease": gpu_lease,
        "target_access_audit": _target_access_audit(tensor_evaluation_completed=True),
        **_scope_provenance(contract),
    }
    return core._publish_directory(
        final=destination,
        work_root=contract.publication_work_root,
        primary_files={
            "records.jsonl": records,
            STRENGTH_DIAGNOSTICS_FILENAME: strength_diagnostics,
            "run_summary.json": summary,
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_ss_calibration_v2_stage2_process_shard",
            "stage": 2,
            "fresh_process": True,
            "process_id": process_id,
            "slot_index": slot_index,
            "top3_in_frozen_order": [value.to_dict() for value in top3],
            "selection_bn_protocol": "SS",
            "BS_excluded_from_selection": True,
            "diagnostic_detail": "global",
            "require_nonzero_parameter_update": False,
            "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
            "update_activity_contract": _update_activity_disclosure(),
            "record_count": 117,
            "episode_count": 7488,
            "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
            "top3_receipt_sha256": receipt_binding.sha256,
            "worker_environment": worker_environment,
            "physical_gpu_lease": gpu_lease,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            **_scope_provenance(contract),
        },
        completion_metadata={
            "stage": 2,
            "process_id": process_id,
            "slot_index": slot_index,
            "record_count": 117,
            "episode_count": 7488,
            "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
            "top3_receipt_sha256": receipt_binding.sha256,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "scope": dict(SCOPE),
            "selection_bn_protocol": "SS",
            "BS_excluded_from_selection": True,
            "diagnostic_detail": "global",
            "require_nonzero_parameter_update": False,
            "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
            "update_activity_contract": _update_activity_disclosure(),
        },
    )


def _verify_embedded_scope(
    contract: CalibrationContract, value: Mapping[str, Any], *, label: str
) -> None:
    expected = _scope_provenance(contract)
    for key, expected_value in expected.items():
        _equal(
            dict(_mapping(value.get(key), f"{label}.{key}")),
            expected_value,
            f"{label} {key}",
        )


def _verify_record_common(
    record: Mapping[str, Any],
    *,
    stage: int,
    process_id: str,
    candidate: Candidate,
    label: str,
) -> tuple[str, str, str, int]:
    for actual, expected, field in (
        (record.get("stage"), stage, "stage"),
        (record.get("fresh_process"), True, "fresh process"),
        (record.get("process_id"), process_id, "process ID"),
        (record.get("candidate"), candidate.to_dict(), "candidate"),
        (record.get("bn_protocol"), SS_BN_PROTOCOL, "SS protocol"),
        (record.get("image_count"), IMAGES_PER_CELL, "image count"),
        (record.get("optimizer_steps_total"), IMAGES_PER_CELL, "optimizer steps"),
        (record.get("test_image_opens"), 0, "test image opens"),
        (record.get("test_label_opens"), 0, "test label opens"),
        (record.get("method_label_accesses"), 0, "method label accesses"),
    ):
        _equal(actual, expected, f"{label} {field}")
    hard_gates = _mapping(record.get("hard_gates"), f"{label} hard gates")
    audit = _mapping(record.get("protocol_audit"), f"{label} protocol audit")
    _equal(set(hard_gates), set(REQUIRED_HARD_GATES), f"{label} hard-gate set")
    _equal(set(audit), set(REQUIRED_PROTOCOL_AUDIT), f"{label} audit set")
    if not all(value is True for value in hard_gates.values()):
        raise CalibrationExecutionError(f"{label} contains a failed hard gate")
    if not all(value is True for value in audit.values()):
        raise CalibrationExecutionError(f"{label} contains a failed protocol audit")
    endpoints = _mapping(record.get("endpoints"), f"{label} endpoints")
    _equal(set(endpoints), {"tent_pre", "tent_post"}, f"{label} endpoint set")
    keys = {
        "intersection_pixels",
        "union_pixels",
        "false_alarm_pixels",
        "total_image_pixels",
        "detected_targets",
        "total_targets",
    }
    for endpoint in ("tent_pre", "tent_post"):
        values = _mapping(endpoints[endpoint], f"{label} {endpoint}")
        _equal(set(values), keys, f"{label} {endpoint} fields")
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CalibrationExecutionError(
                    f"{label} {endpoint}.{key} must be nonnegative integer"
                )
        if values["union_pixels"] <= 0 or values["total_targets"] <= 0:
            raise CalibrationExecutionError(f"{label} has nonpositive selector denominator")
    def reject_forbidden(value: Any, trail: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                lowered = key.lower()
                if lowered in FORBIDDEN_SELECTOR_FIELDS or any(
                    lowered.startswith(prefix.lower())
                    for prefix in FORBIDDEN_SELECTOR_FIELD_PREFIXES
                ):
                    raise CalibrationExecutionError(
                        f"{label} contains forbidden field {trail}.{key}"
                    )
                reject_forbidden(child, f"{trail}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child in enumerate(value):
                reject_forbidden(child, f"{trail}[{index}]")

    reject_forbidden(record, label)
    return (
        str(record["dataset"]),
        str(record["bn_protocol"]),
        str(record["corruption"]),
        int(record["severity"]),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise CalibrationExecutionError(
                    f"blank JSONL line at {path}:{line_number}"
                )
            value = json.loads(raw)
            values.append(
                dict(_mapping(value, f"{path.name} line {line_number}"))
            )
    return values


def _verify_metric_stats(
    value: Any,
    *,
    episode_count: int,
    label: str,
    nonnegative: bool,
    upper_bound: float | None = None,
) -> dict[str, float]:
    raw = _mapping(value, label)
    _equal(set(raw), {"sum", "mean", "min", "max"}, f"{label} fields")
    result = {key: _finite_float(raw[key], f"{label}.{key}") for key in raw}
    _equal(result["mean"], result["sum"] / episode_count, f"{label} mean")
    if result["min"] > result["mean"] or result["mean"] > result["max"]:
        raise CalibrationExecutionError(f"{label} min/mean/max order is invalid")
    if nonnegative and result["min"] < 0.0:
        raise CalibrationExecutionError(f"{label} must be nonnegative")
    if upper_bound is not None and result["max"] > upper_bound:
        raise CalibrationExecutionError(
            f"{label} exceeds frozen upper bound {upper_bound}"
        )
    return result


def _verify_strength_diagnostic(
    diagnostic: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    label: str,
) -> None:
    expected_fields = {
        "schema_version",
        "stage",
        "process_id",
        "candidate",
        "dataset",
        "bn_protocol",
        "corruption",
        "severity",
        "episode_count",
        "update_activity_contract",
        "parameter_delta_episode_counts",
        "changed_bn_affine_tensor_count_histogram",
        *_STRENGTH_METRICS,
    }
    _equal(set(diagnostic), expected_fields, f"{label} exact fields")
    for actual, expected, field in (
        (diagnostic.get("schema_version"), 1, "schema"),
        (diagnostic.get("stage"), record.get("stage"), "stage"),
        (diagnostic.get("process_id"), record.get("process_id"), "process ID"),
        (diagnostic.get("candidate"), record.get("candidate"), "candidate"),
        (diagnostic.get("dataset"), record.get("dataset"), "dataset"),
        (diagnostic.get("bn_protocol"), record.get("bn_protocol"), "BN protocol"),
        (diagnostic.get("corruption"), record.get("corruption"), "corruption"),
        (diagnostic.get("severity"), record.get("severity"), "severity"),
        (diagnostic.get("episode_count"), IMAGES_PER_CELL, "episodes"),
        (
            dict(_mapping(diagnostic.get("update_activity_contract"), f"{label} contract")),
            _update_activity_disclosure(),
            "contract",
        ),
    ):
        _equal(actual, expected, f"{label} {field}")
    counts = _mapping(
        diagnostic.get("parameter_delta_episode_counts"), f"{label} delta counts"
    )
    _equal(set(counts), {"zero", "nonzero", "total"}, f"{label} count fields")
    for key in counts:
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CalibrationExecutionError(f"{label} {key} count is invalid")
    _equal(counts["total"], IMAGES_PER_CELL, f"{label} total episodes")
    _equal(
        counts["zero"] + counts["nonzero"],
        IMAGES_PER_CELL,
        f"{label} zero/nonzero conservation",
    )
    histogram = _mapping(
        diagnostic.get("changed_bn_affine_tensor_count_histogram"),
        f"{label} changed-tensor histogram",
    )
    expanded: list[float] = []
    for raw_key, raw_count in histogram.items():
        if not isinstance(raw_key, str) or not raw_key.isascii() or not raw_key.isdigit():
            raise CalibrationExecutionError(f"{label} histogram key is not canonical")
        changed = int(raw_key)
        if str(changed) != raw_key:
            raise CalibrationExecutionError(f"{label} histogram key is not canonical")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
            raise CalibrationExecutionError(f"{label} histogram count is invalid")
        expanded.extend([float(changed)] * raw_count)
    _equal(len(expanded), IMAGES_PER_CELL, f"{label} histogram episode count")
    _equal(sum(value == 0.0 for value in expanded), counts["zero"], f"{label} zero histogram")
    _equal(sum(value > 0.0 for value in expanded), counts["nonzero"], f"{label} nonzero histogram")
    changed_stats = _verify_metric_stats(
        diagnostic["changed_bn_affine_tensor_count"],
        episode_count=IMAGES_PER_CELL,
        label=f"{label} changed BN affine tensors",
        nonnegative=True,
    )
    _equal(changed_stats, _metric_stats(expanded), f"{label} changed tensor statistics")
    for key in _STRENGTH_METRICS[1:]:
        bounded = key in {
            "mean_absolute_probability_delta",
            "maximum_absolute_probability_delta",
            "strict_threshold_changed_pixel_ratio",
        }
        _verify_metric_stats(
            diagnostic[key],
            episode_count=IMAGES_PER_CELL,
            label=f"{label} {key}",
            nonnegative=key != "foreground_probability_mass_delta",
            upper_bound=1.0 if bounded else None,
        )


def _verify_strength_file(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics = _load_jsonl(path)
    _equal(len(diagnostics), len(records), f"{label} diagnostic count")
    for index, (diagnostic, record) in enumerate(
        zip(diagnostics, records, strict=True)
    ):
        _verify_strength_diagnostic(
            diagnostic,
            record,
            label=f"{label} diagnostic {index}",
        )
    return diagnostics, _combine_strength_diagnostics(diagnostics)


def _expected_cells() -> set[tuple[str, str, str, int]]:
    return {
        (dataset, SS_BN_PROTOCOL, corruption, severity)
        for dataset in DATASETS
        for corruption, severity in CONDITIONS
    }


def _verify_stage1_candidate_shard(
    contract: CalibrationContract, seal: RuntimeSeal, candidate: Candidate
) -> dict[str, Any]:
    path = _stage1_shard_path(contract, candidate)
    verified = core.verify_shard(
        path, expected_seal_sha256=seal.global_runtime_seal_sha256
    )
    manifest = verified["manifest"]
    complete = verified["complete"]
    for actual, expected, label in (
        (manifest.get("artifact_type"), "binary_tent_ss_calibration_v2_stage1_candidate_shard", "artifact type"),
        (manifest.get("stage"), 1, "manifest stage"),
        (manifest.get("fresh_process"), True, "manifest fresh process"),
        (manifest.get("candidate"), candidate.to_dict(), "manifest candidate"),
        (manifest.get("selection_bn_protocol"), "SS", "manifest SS"),
        (manifest.get("BS_excluded_from_selection"), True, "manifest BS exclusion"),
        (manifest.get("diagnostic_detail"), "global", "manifest diagnostic detail"),
        (manifest.get("require_nonzero_parameter_update"), False, "manifest zero-update requirement"),
        (manifest.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "manifest zero-update policy"),
        (manifest.get("update_activity_contract"), _update_activity_disclosure(), "manifest update diagnostics"),
        (manifest.get("record_count"), 39, "manifest record count"),
        (manifest.get("episode_count"), 2496, "manifest episode count"),
        (complete.get("stage"), 1, "completion stage"),
        (complete.get("candidate"), candidate.to_dict(), "completion candidate"),
        (complete.get("record_count"), 39, "completion record count"),
        (complete.get("episode_count"), 2496, "completion episode count"),
        (complete.get("selection_bn_protocol"), "SS", "completion SS"),
        (complete.get("BS_excluded_from_selection"), True, "completion BS exclusion"),
        (complete.get("diagnostic_detail"), "global", "completion diagnostic detail"),
        (complete.get("require_nonzero_parameter_update"), False, "completion zero-update requirement"),
        (complete.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "completion zero-update policy"),
        (complete.get("update_activity_contract"), _update_activity_disclosure(), "completion update diagnostics"),
        (complete.get("scope"), dict(SCOPE), "completion scope"),
    ):
        _equal(actual, expected, f"stage1 {label}")
    _equal(
        set(_mapping(manifest.get("files"), "stage1 files")),
        {
            "records.jsonl",
            STRENGTH_DIAGNOSTICS_FILENAME,
            "run_summary.json",
            "provenance.json",
            "runtime_seal.json",
        },
        "stage1 exact files",
    )
    process_id = core._safe_slug(str(manifest["process_id"]))
    _equal(complete.get("process_id"), process_id, "stage1 completion process ID")
    observed: set[tuple[str, str, str, int]] = set()
    for index, record in enumerate(verified["records"]):
        cell = _verify_record_common(
            record,
            stage=1,
            process_id=process_id,
            candidate=candidate,
            label=f"stage1 record {index}",
        )
        if cell in observed:
            raise CalibrationExecutionError(f"duplicate stage1 SS cell: {cell}")
        observed.add(cell)
    _equal(observed, _expected_cells(), "stage1 complete SS cell set")
    _equal(_load_json(path / "runtime_seal.json"), seal.to_dict(), "stage1 runtime seal")
    provenance = _load_json(path / "provenance.json")
    summary = _load_json(path / "run_summary.json")
    strength_diagnostics, strength_summary = _verify_strength_file(
        path / STRENGTH_DIAGNOSTICS_FILENAME,
        verified["records"],
        label="stage1 candidate",
    )
    _verify_embedded_scope(contract, manifest, label="stage1 manifest")
    _verify_embedded_scope(contract, provenance, label="stage1 provenance")
    for actual, expected, label in (
        (provenance.get("stage"), 1, "provenance stage"),
        (provenance.get("fresh_process"), True, "provenance fresh process"),
        (provenance.get("process_id"), process_id, "provenance process ID"),
        (provenance.get("candidate"), candidate.to_dict(), "provenance candidate"),
        (summary.get("stage"), 1, "summary stage"),
        (summary.get("fresh_process"), True, "summary fresh process"),
        (summary.get("process_id"), process_id, "summary process ID"),
        (summary.get("candidate"), candidate.to_dict(), "summary candidate"),
        (summary.get("selection_bn_protocol"), "SS", "summary SS"),
        (summary.get("BS_excluded_from_selection"), True, "summary BS exclusion"),
        (summary.get("diagnostic_detail"), "global", "summary diagnostic detail"),
        (summary.get("require_nonzero_parameter_update"), False, "summary zero-update requirement"),
        (summary.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "summary zero-update policy"),
        (summary.get("update_activity_diagnostics"), strength_summary, "summary update diagnostics"),
        (summary.get("cell_record_count"), 39, "summary records"),
        (summary.get("episode_count"), 2496, "summary episodes"),
        (summary.get("model_method_optimizer_build_count"), 3, "summary builds"),
        (summary.get("target_access_audit"), _target_access_audit(tensor_evaluation_completed=True), "summary target access"),
        (provenance.get("target_access_audit"), _target_access_audit(tensor_evaluation_completed=True), "provenance target access"),
    ):
        _equal(actual, expected, f"stage1 {label}")
    core._verify_embedded_gpu_lease(
        contract=contract,
        manifest=manifest,
        provenance=provenance,
        summary=summary,
        label="SS-v2 stage1",
    )
    verified["strength_diagnostics"] = strength_diagnostics
    verified["strength_summary"] = strength_summary
    return verified


def _endpoint_counts(record: Mapping[str, Any], endpoint: str) -> tuple[int, ...]:
    values = _mapping(record["endpoints"][endpoint], endpoint)
    return tuple(
        int(values[key])
        for key in (
            "intersection_pixels",
            "union_pixels",
            "false_alarm_pixels",
            "total_image_pixels",
            "detected_targets",
            "total_targets",
        )
    )


def _verify_cross_run_endpoint_invariants(
    stage1_records: Sequence[Mapping[str, Any]],
    stage2_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    expected_pixels = IMAGES_PER_CELL * 256 * 256
    pre_by_cell: dict[tuple[str, str, str, int], tuple[int, ...]] = {}
    targets_by_dataset: dict[str, int] = {}

    def inspect(record: Mapping[str, Any], *, stage: int, index: int) -> None:
        pre = _endpoint_counts(record, "tent_pre")
        post = _endpoint_counts(record, "tent_post")
        _equal(pre[3], expected_pixels, f"stage{stage}[{index}] pre pixels")
        _equal(post[3], expected_pixels, f"stage{stage}[{index}] post pixels")
        _equal(pre[5], post[5], f"stage{stage}[{index}] targets")
        dataset = str(record["dataset"])
        if dataset in targets_by_dataset:
            _equal(pre[5], targets_by_dataset[dataset], f"{dataset} target conservation")
        else:
            targets_by_dataset[dataset] = pre[5]
        cell = (
            dataset,
            str(record["bn_protocol"]),
            str(record["corruption"]),
            int(record["severity"]),
        )
        if stage == 1:
            if cell in pre_by_cell:
                _equal(pre, pre_by_cell[cell], f"stage1 TENT-pre identity {cell}")
            else:
                pre_by_cell[cell] = pre
        else:
            if cell not in pre_by_cell:
                raise CalibrationExecutionError(f"stage2 cell lacks stage1 anchor: {cell}")
            _equal(pre, pre_by_cell[cell], f"stage2/stage1 TENT-pre identity {cell}")

    for index, record in enumerate(stage1_records):
        inspect(record, stage=1, index=index)
    _equal(set(pre_by_cell), _expected_cells(), "stage1 endpoint anchor cells")
    for index, record in enumerate(stage2_records):
        inspect(record, stage=2, index=index)
    return {
        "selection_bn_protocol": "SS",
        "BS_excluded_from_selection": True,
        "tent_pre_integer_counts_identical_across_stage1_candidates": True,
        "stage2_tent_pre_integer_counts_equal_stage1": bool(stage2_records),
        "total_image_pixels_per_cell": expected_pixels,
        "total_targets_conserved_within_each_dataset": True,
        "dataset_total_targets": dict(sorted(targets_by_dataset.items())),
    }


def _verify_stage1_aggregate(
    contract: CalibrationContract, seal: RuntimeSeal
) -> dict[str, Any]:
    path = _stage1_aggregate_path(contract)
    verified = core.verify_shard(path, expected_seal_sha256=seal.global_runtime_seal_sha256)
    manifest, complete = verified["manifest"], verified["complete"]
    for actual, expected, label in (
        (manifest.get("artifact_type"), "binary_tent_ss_calibration_v2_stage1_aggregate", "artifact type"),
        (manifest.get("record_count"), 390, "records"),
        (manifest.get("episode_count"), 24960, "episodes"),
        (manifest.get("candidate_count"), 10, "candidates"),
        (manifest.get("fresh_process_count"), 10, "processes"),
        (manifest.get("selection_bn_protocol"), "SS", "SS"),
        (manifest.get("BS_excluded_from_selection"), True, "BS exclusion"),
        (manifest.get("require_nonzero_parameter_update"), False, "zero-update requirement"),
        (manifest.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "zero-update policy"),
        (manifest.get("update_activity_contract"), _update_activity_disclosure(), "update diagnostics"),
        (complete.get("stage"), 1, "completion stage"),
        (complete.get("record_count"), 390, "completion records"),
        (complete.get("episode_count"), 24960, "completion episodes"),
        (complete.get("diagnostics_not_used_for_selection"), True, "completion diagnostic exclusion"),
        (complete.get("require_nonzero_parameter_update"), False, "completion zero-update requirement"),
        (complete.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "completion zero-update policy"),
        (complete.get("scope"), dict(SCOPE), "completion scope"),
    ):
        _equal(actual, expected, f"stage1 aggregate {label}")
    _equal(
        set(manifest["files"]),
        {
            "stage1_ss_top3_receipt.json",
            "stage1_records.jsonl",
            STRENGTH_DIAGNOSTICS_FILENAME,
            "shard_index.json",
            "provenance.json",
            "runtime_seal.json",
        },
        "stage1 aggregate exact files",
    )
    _equal(_load_json(path / "runtime_seal.json"), seal.to_dict(), "stage1 aggregate seal")
    invariants = _verify_cross_run_endpoint_invariants(verified["records"])
    strength_diagnostics, strength_summary = _verify_strength_file(
        path / STRENGTH_DIAGNOSTICS_FILENAME,
        verified["records"],
        label="stage1 aggregate",
    )
    expected_receipt = {
        **select_stage1_top3(verified["records"]),
        "endpoint_invariants": invariants,
        "diagnostics_not_used_for_selection": True,
        "update_activity_summary": strength_summary,
        **_scope_provenance(contract),
    }
    receipt = _load_json(_stage1_receipt_path(contract))
    _equal(receipt, expected_receipt, "stage1 aggregate receipt")
    _equal(complete.get("top3"), receipt["top3"], "stage1 completion top3")
    provenance = _load_json(path / "provenance.json")
    _verify_embedded_scope(contract, manifest, label="stage1 aggregate manifest")
    _verify_embedded_scope(contract, provenance, label="stage1 aggregate provenance")
    _equal(manifest.get("endpoint_invariants"), invariants, "stage1 manifest invariants")
    _equal(provenance.get("endpoint_invariants"), invariants, "stage1 provenance invariants")
    _equal(manifest.get("update_activity_summary"), strength_summary, "stage1 manifest update diagnostics")
    _equal(provenance.get("update_activity_summary"), strength_summary, "stage1 provenance update diagnostics")
    index = _load_json(path / "shard_index.json")
    _equal(set(index), {"shards"}, "stage1 index fields")
    _equal(len(index["shards"]), 10, "stage1 index count")
    source_records: list[dict[str, Any]] = []
    source_strength: list[dict[str, Any]] = []
    process_ids: set[str] = set()
    for candidate, raw_index in zip(ALL_CANDIDATES, index["shards"], strict=True):
        source = _verify_stage1_candidate_shard(contract, seal, candidate)
        process_id = str(source["manifest"]["process_id"])
        if process_id in process_ids:
            raise CalibrationExecutionError("stage1 process ID reused")
        process_ids.add(process_id)
        _equal(
            raw_index,
            {
                "candidate": candidate.to_dict(),
                "process_id": process_id,
                "artifact_manifest_sha256": source["manifest_sha256"],
            },
            "stage1 source shard index",
        )
        source_records.extend(source["records"])
        source_strength.extend(source["strength_diagnostics"])
    _equal(verified["records"], source_records, "stage1 aggregate/source records")
    _equal(strength_diagnostics, source_strength, "stage1 aggregate/source strength diagnostics")
    select_stage1_top3(verified["records"])
    verified["strength_diagnostics"] = strength_diagnostics
    verified["strength_summary"] = strength_summary
    return verified


def aggregate_stage1(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    seal, _ = capture_runtime_seal(contract)
    monitor = RuntimeSealMonitor(seal)
    records: list[dict[str, Any]] = []
    strength_diagnostics: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    process_ids: set[str] = set()
    for candidate in ALL_CANDIDATES:
        verified = _verify_stage1_candidate_shard(contract, seal, candidate)
        process_id = str(verified["manifest"]["process_id"])
        if process_id in process_ids:
            raise CalibrationExecutionError("stage1 candidates reused a process ID")
        process_ids.add(process_id)
        records.extend(verified["records"])
        strength_diagnostics.extend(verified["strength_diagnostics"])
        shards.append(
            {
                "candidate": candidate.to_dict(),
                "process_id": process_id,
                "artifact_manifest_sha256": verified["manifest_sha256"],
            }
        )
    invariants = _verify_cross_run_endpoint_invariants(records)
    strength_summary = _combine_strength_diagnostics(strength_diagnostics)
    receipt = {
        **select_stage1_top3(records),
        "endpoint_invariants": invariants,
        "diagnostics_not_used_for_selection": True,
        "update_activity_summary": strength_summary,
        **_scope_provenance(contract),
    }
    monitor.assert_unchanged(stage="stage1_aggregate_pre_publish", full_byte_rehash=True)
    provenance = {
        "schema_version": 2,
        "artifact_type": "binary_tent_ss_calibration_v2_stage1_aggregate",
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "runtime_audits": monitor.audits,
        "endpoint_invariants": invariants,
        "update_activity_summary": strength_summary,
        **_scope_provenance(contract),
    }
    published = core._publish_directory(
        final=_stage1_aggregate_path(contract),
        work_root=contract.publication_work_root,
        primary_files={
            "stage1_ss_top3_receipt.json": receipt,
            "stage1_records.jsonl": records,
            STRENGTH_DIAGNOSTICS_FILENAME: strength_diagnostics,
            "shard_index.json": {"shards": shards},
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_ss_calibration_v2_stage1_aggregate",
            "record_count": 390,
            "episode_count": 24960,
            "candidate_count": 10,
            "fresh_process_count": 10,
            "selection_bn_protocol": "SS",
            "BS_excluded_from_selection": True,
            "require_nonzero_parameter_update": False,
            "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
            "update_activity_contract": _update_activity_disclosure(),
            "update_activity_summary": strength_summary,
            "endpoint_invariants": invariants,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            **_scope_provenance(contract),
        },
        completion_metadata={
            "stage": 1,
            "record_count": 390,
            "episode_count": 24960,
            "top3": receipt["top3"],
            "selection_bn_protocol": "SS",
            "BS_excluded_from_selection": True,
            "diagnostics_not_used_for_selection": True,
            "require_nonzero_parameter_update": False,
            "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "scope": dict(SCOPE),
        },
    )
    post = _verify_stage1_aggregate(contract, seal)
    return {**published, "post_publish_verified": True, "verified_manifest_sha256": post["manifest_sha256"]}


def _base_seal_from_extended(seal: RuntimeSeal) -> tuple[RuntimeSeal, BoundFile]:
    receipt_bindings = [
        value for value in seal.bindings if value.role == "stage1_top3_receipt:ss_v2"
    ]
    _equal(len(receipt_bindings), 1, "stage2 receipt binding count")
    receipt_binding = receipt_bindings[0]
    base = core._runtime_seal_from_bindings(
        tuple(value for value in seal.bindings if value != receipt_binding),
        seal.cache_lineage,
        seal.runtime_environment,
    )
    return base, receipt_binding


def _verify_stage2_process_shard(
    path: Path,
    *,
    contract: CalibrationContract,
    seal: RuntimeSeal,
    top3: Sequence[Candidate],
    expected_slot_index: int,
) -> dict[str, Any]:
    verified = core.verify_shard(path, expected_seal_sha256=seal.global_runtime_seal_sha256)
    manifest, complete = verified["manifest"], verified["complete"]
    base_seal, receipt_binding = _base_seal_from_extended(seal)
    expected_process_id = _stage2_slot_process_id(
        receipt_binding.sha256, expected_slot_index
    )
    for actual, expected, label in (
        (manifest.get("artifact_type"), "binary_tent_ss_calibration_v2_stage2_process_shard", "artifact type"),
        (manifest.get("stage"), 2, "manifest stage"),
        (manifest.get("fresh_process"), True, "manifest fresh process"),
        (manifest.get("process_id"), expected_process_id, "manifest process ID"),
        (manifest.get("slot_index"), expected_slot_index, "manifest slot"),
        (manifest.get("top3_in_frozen_order"), [value.to_dict() for value in top3], "manifest top3"),
        (manifest.get("selection_bn_protocol"), "SS", "manifest SS"),
        (manifest.get("BS_excluded_from_selection"), True, "manifest BS exclusion"),
        (manifest.get("diagnostic_detail"), "global", "manifest diagnostic detail"),
        (manifest.get("require_nonzero_parameter_update"), False, "manifest zero-update requirement"),
        (manifest.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "manifest zero-update policy"),
        (manifest.get("update_activity_contract"), _update_activity_disclosure(), "manifest update diagnostics"),
        (manifest.get("record_count"), 117, "manifest records"),
        (manifest.get("episode_count"), 7488, "manifest episodes"),
        (manifest.get("stage1_runtime_seal_sha256"), base_seal.global_runtime_seal_sha256, "manifest base seal"),
        (manifest.get("top3_receipt_sha256"), receipt_binding.sha256, "manifest receipt SHA"),
        (complete.get("stage"), 2, "completion stage"),
        (complete.get("process_id"), expected_process_id, "completion process ID"),
        (complete.get("slot_index"), expected_slot_index, "completion slot"),
        (complete.get("record_count"), 117, "completion records"),
        (complete.get("episode_count"), 7488, "completion episodes"),
        (complete.get("diagnostic_detail"), "global", "completion diagnostic detail"),
        (complete.get("require_nonzero_parameter_update"), False, "completion zero-update requirement"),
        (complete.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "completion zero-update policy"),
        (complete.get("update_activity_contract"), _update_activity_disclosure(), "completion update diagnostics"),
        (complete.get("stage1_runtime_seal_sha256"), base_seal.global_runtime_seal_sha256, "completion base seal"),
        (complete.get("top3_receipt_sha256"), receipt_binding.sha256, "completion receipt SHA"),
        (complete.get("scope"), dict(SCOPE), "completion scope"),
    ):
        _equal(actual, expected, f"stage2 {label}")
    _equal(path.name, expected_process_id, "stage2 deterministic shard path")
    _equal(
        set(manifest["files"]),
        {
            "records.jsonl",
            STRENGTH_DIAGNOSTICS_FILENAME,
            "run_summary.json",
            "provenance.json",
            "runtime_seal.json",
        },
        "stage2 exact files",
    )
    _equal(_load_json(path / "runtime_seal.json"), seal.to_dict(), "stage2 runtime seal")
    provenance = _load_json(path / "provenance.json")
    summary = _load_json(path / "run_summary.json")
    strength_diagnostics, strength_summary = _verify_strength_file(
        path / STRENGTH_DIAGNOSTICS_FILENAME,
        verified["records"],
        label="stage2 process",
    )
    _verify_embedded_scope(contract, manifest, label="stage2 manifest")
    _verify_embedded_scope(contract, provenance, label="stage2 provenance")
    for actual, expected, label in (
        (provenance.get("stage"), 2, "provenance stage"),
        (provenance.get("fresh_process"), True, "provenance fresh process"),
        (provenance.get("process_id"), expected_process_id, "provenance process ID"),
        (provenance.get("slot_index"), expected_slot_index, "provenance slot"),
        (provenance.get("top3_receipt"), str(_stage1_receipt_path(contract).resolve()), "provenance receipt path"),
        (provenance.get("top3_receipt_sha256"), receipt_binding.sha256, "provenance receipt SHA"),
        (provenance.get("stage1_runtime_seal_sha256"), base_seal.global_runtime_seal_sha256, "provenance base seal"),
        (summary.get("stage"), 2, "summary stage"),
        (summary.get("fresh_process"), True, "summary fresh process"),
        (summary.get("process_id"), expected_process_id, "summary process ID"),
        (summary.get("slot_index"), expected_slot_index, "summary slot"),
        (summary.get("top3_in_frozen_order"), [value.to_dict() for value in top3], "summary top3"),
        (summary.get("selection_bn_protocol"), "SS", "summary SS"),
        (summary.get("BS_excluded_from_selection"), True, "summary BS exclusion"),
        (summary.get("diagnostic_detail"), "global", "summary diagnostic detail"),
        (summary.get("require_nonzero_parameter_update"), False, "summary zero-update requirement"),
        (summary.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "summary zero-update policy"),
        (summary.get("update_activity_diagnostics"), strength_summary, "summary update diagnostics"),
        (summary.get("record_count"), 117, "summary records"),
        (summary.get("episode_count"), 7488, "summary episodes"),
        (summary.get("candidate_model_method_optimizer_rebuilt_before_each_candidate"), True, "summary rebuild"),
        (summary.get("target_access_audit"), _target_access_audit(tensor_evaluation_completed=True), "summary target access"),
        (provenance.get("target_access_audit"), _target_access_audit(tensor_evaluation_completed=True), "provenance target access"),
    ):
        _equal(actual, expected, f"stage2 {label}")
    core._verify_embedded_gpu_lease(
        contract=contract,
        manifest=manifest,
        provenance=provenance,
        summary=summary,
        label="SS-v2 stage2",
    )
    cells_by_candidate = {candidate: set() for candidate in top3}
    observed_order: list[Candidate] = []
    for index, record in enumerate(verified["records"]):
        candidate = Candidate.from_values(
            record["candidate"]["optimizer"], record["candidate"]["learning_rate"]
        )
        if candidate not in cells_by_candidate:
            raise CalibrationExecutionError("stage2 record candidate outside top3")
        cell = _verify_record_common(
            record,
            stage=2,
            process_id=expected_process_id,
            candidate=candidate,
            label=f"stage2 record {index}",
        )
        if cell in cells_by_candidate[candidate]:
            raise CalibrationExecutionError(f"duplicate stage2 cell: {candidate} {cell}")
        cells_by_candidate[candidate].add(cell)
        if not observed_order or observed_order[-1] != candidate:
            observed_order.append(candidate)
    _equal(tuple(observed_order), tuple(top3), "stage2 frozen candidate order")
    for candidate in top3:
        _equal(cells_by_candidate[candidate], _expected_cells(), f"stage2 complete cells {candidate}")
    verified["strength_diagnostics"] = strength_diagnostics
    verified["strength_summary"] = strength_summary
    return verified


def _application_receipt(
    contract: CalibrationContract, selected_candidate: Mapping[str, Any]
) -> dict[str, Any]:
    dataset_science = contract.scientific["source_train_subsets"]["datasets"]
    checkpoint_bindings: list[dict[str, Any]] = []
    for dataset in DATASETS:
        values = dataset_science[dataset]
        for section_name in ("calibration_checkpoint", "inherited_application_checkpoint"):
            section = values[section_name]
            checkpoint_bindings.append(
                {
                    "dataset": dataset,
                    "checkpoint_role": section["role"],
                    "checkpoint_path": section["path"],
                    "checkpoint_sha256": section["sha256"],
                    "selected_candidate": dict(selected_candidate),
                    "additional_tuning_episodes": 0,
                }
            )
    return {
        "selection_bn_protocol": "SS",
        "BS_excluded_from_selection": True,
        "application_protocols": ["SS", "BS"],
        "application_protocol_ids": [
            SS_BN_PROTOCOL,
            "single_image_spatial_batch_stats",
        ],
        "best_pd_reuses_same_frozen_hyperparameters_without_tuning": True,
        "best_pd_additional_tuning_episodes": 0,
        "BS_selection_episodes": 0,
        "checkpoint_bindings": checkpoint_bindings,
    }


def _verify_final_aggregate(
    contract: CalibrationContract, base_seal: RuntimeSeal, seal: RuntimeSeal
) -> dict[str, Any]:
    stage1 = _verify_stage1_aggregate(contract, base_seal)
    rebound, receipt_binding, stage1 = _bind_verified_stage1_receipt(
        contract, base_seal, stage1_verified=stage1
    )
    _equal(rebound.to_dict(), seal.to_dict(), "final extended runtime seal")
    top3 = _load_top3(_stage1_receipt_path(contract))
    path = contract.output_root / contract.execution["outputs"]["stage2_aggregate_directory"]
    verified = core.verify_shard(path, expected_seal_sha256=seal.global_runtime_seal_sha256)
    manifest, complete = verified["manifest"], verified["complete"]
    for actual, expected, label in (
        (manifest.get("artifact_type"), "binary_tent_ss_calibration_v2_final_aggregate", "artifact type"),
        (manifest.get("record_count"), 234, "records"),
        (manifest.get("episode_count"), 14976, "episodes"),
        (manifest.get("total_episode_count"), 39936, "total episodes"),
        (manifest.get("fresh_stage2_process_count"), 2, "process count"),
        (manifest.get("stage1_runtime_seal_sha256"), base_seal.global_runtime_seal_sha256, "base seal"),
        (manifest.get("top3_receipt_sha256"), receipt_binding.sha256, "receipt SHA"),
        (manifest.get("top3_in_frozen_order"), [value.to_dict() for value in top3], "top3"),
        (manifest.get("selection_bn_protocol"), "SS", "SS"),
        (manifest.get("BS_excluded_from_selection"), True, "BS exclusion"),
        (manifest.get("require_nonzero_parameter_update"), False, "zero-update requirement"),
        (manifest.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "zero-update policy"),
        (manifest.get("update_activity_contract"), _update_activity_disclosure(), "update diagnostics"),
        (complete.get("stage"), 2, "completion stage"),
        (complete.get("record_count"), 234, "completion records"),
        (complete.get("episode_count"), 14976, "completion episodes"),
        (complete.get("total_episode_count"), 39936, "completion total episodes"),
        (complete.get("stage1_runtime_seal_sha256"), base_seal.global_runtime_seal_sha256, "completion base seal"),
        (complete.get("top3_receipt_sha256"), receipt_binding.sha256, "completion receipt SHA"),
        (complete.get("selection_bn_protocol"), "SS", "completion SS"),
        (complete.get("BS_excluded_from_selection"), True, "completion BS exclusion"),
        (complete.get("best_pd_reuses_same_frozen_hyperparameters_without_tuning"), True, "completion best_pd reuse"),
        (complete.get("diagnostics_not_used_for_selection"), True, "completion diagnostic exclusion"),
        (complete.get("require_nonzero_parameter_update"), False, "completion zero-update requirement"),
        (complete.get("zero_parameter_update_policy"), ZERO_UPDATE_POLICY, "completion zero-update policy"),
        (complete.get("scope"), dict(SCOPE), "completion scope"),
    ):
        _equal(actual, expected, f"final {label}")
    _equal(
        set(manifest["files"]),
        {
            "final_ss_selection_receipt.json",
            "stage2_records.jsonl",
            STRENGTH_DIAGNOSTICS_FILENAME,
            "stage2_shard_index.json",
            "provenance.json",
            "runtime_seal.json",
        },
        "final exact files",
    )
    _equal(_load_json(path / "runtime_seal.json"), seal.to_dict(), "final runtime seal")
    provenance = _load_json(path / "provenance.json")
    strength_diagnostics, strength_summary = _verify_strength_file(
        path / STRENGTH_DIAGNOSTICS_FILENAME,
        verified["records"],
        label="final aggregate",
    )
    _verify_embedded_scope(contract, manifest, label="final manifest")
    _verify_embedded_scope(contract, provenance, label="final provenance")
    for actual, expected, label in (
        (provenance.get("artifact_type"), "binary_tent_ss_calibration_v2_final_aggregate", "provenance type"),
        (provenance.get("global_runtime_seal_sha256"), seal.global_runtime_seal_sha256, "provenance seal"),
        (provenance.get("stage1_runtime_seal_sha256"), base_seal.global_runtime_seal_sha256, "provenance base seal"),
        (provenance.get("top3_receipt_sha256"), receipt_binding.sha256, "provenance receipt SHA"),
    ):
        _equal(actual, expected, f"final {label}")
    stage2_root = contract.output_root / "stage2" / "shards"
    expected_paths = {
        slot: _stage2_shard_path(
            contract, _stage2_slot_process_id(receipt_binding.sha256, slot)
        )
        for slot in (1, 2)
    }
    if not stage2_root.is_dir() or stage2_root.is_symlink():
        raise CalibrationExecutionError("stage2 shard root must be a real directory")
    _equal(
        {entry.name for entry in stage2_root.iterdir()},
        {entry.name for entry in expected_paths.values()},
        "final deterministic stage2 shard set",
    )
    stage1_processes = {str(value["process_id"]) for value in stage1["records"]}
    stage2_processes: set[str] = set()
    source_records: list[dict[str, Any]] = []
    source_strength: list[dict[str, Any]] = []
    expected_index: list[dict[str, Any]] = []
    for slot, shard_path in expected_paths.items():
        source = _verify_stage2_process_shard(
            shard_path,
            contract=contract,
            seal=seal,
            top3=top3,
            expected_slot_index=slot,
        )
        process_id = str(source["manifest"]["process_id"])
        if process_id in stage1_processes or process_id in stage2_processes:
            raise CalibrationExecutionError("stage2 process ID duplicated/reused")
        stage2_processes.add(process_id)
        source_records.extend(source["records"])
        source_strength.extend(source["strength_diagnostics"])
        expected_index.append(
            {
                "process_id": process_id,
                "slot_index": slot,
                "artifact_manifest_sha256": source["manifest_sha256"],
            }
        )
    _equal(verified["records"], source_records, "final/source records")
    _equal(strength_diagnostics, source_strength, "final/source strength diagnostics")
    _equal(
        _load_json(path / "stage2_shard_index.json"),
        {"shards": expected_index},
        "final deterministic shard index",
    )
    invariants = _verify_cross_run_endpoint_invariants(stage1["records"], source_records)
    selector_receipt = select_final_candidate(stage1["records"], source_records)
    expected_receipt = {
        **selector_receipt,
        "endpoint_invariants": invariants,
        "diagnostics_not_used_for_selection": True,
        "update_activity_summary": strength_summary,
        "application_receipt": _application_receipt(
            contract, selector_receipt["selected_candidate"]
        ),
        **_scope_provenance(contract),
    }
    receipt = _load_json(path / "final_ss_selection_receipt.json")
    _equal(receipt, expected_receipt, "final selection receipt")
    _equal(complete.get("selected_candidate"), receipt["selected_candidate"], "completion selected candidate")
    _equal(manifest.get("endpoint_invariants"), invariants, "final manifest invariants")
    _equal(provenance.get("endpoint_invariants"), invariants, "final provenance invariants")
    _equal(manifest.get("update_activity_summary"), strength_summary, "final manifest update diagnostics")
    _equal(provenance.get("update_activity_summary"), strength_summary, "final provenance update diagnostics")
    return {
        **verified,
        "selection_receipt": receipt,
        "endpoint_invariants": invariants,
        "strength_diagnostics": strength_diagnostics,
        "strength_summary": strength_summary,
    }


def aggregate_final(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    base_seal, _ = capture_runtime_seal(contract)
    stage1 = _verify_stage1_aggregate(contract, base_seal)
    seal, receipt_binding, stage1 = _bind_verified_stage1_receipt(
        contract, base_seal, stage1_verified=stage1
    )
    monitor = RuntimeSealMonitor(seal)
    receipt_path = _stage1_receipt_path(contract)
    monitor.assert_unchanged(stage="final_aggregate_entry", active_paths=(receipt_path,))
    top3 = _load_top3(receipt_path)
    stage2_root = contract.output_root / "stage2" / "shards"
    expected_paths = {
        slot: _stage2_shard_path(contract, _stage2_slot_process_id(receipt_binding.sha256, slot))
        for slot in (1, 2)
    }
    _equal(
        {entry.name for entry in stage2_root.iterdir()},
        {entry.name for entry in expected_paths.values()},
        "stage2 exact deterministic entries",
    )
    records: list[dict[str, Any]] = []
    strength_diagnostics: list[dict[str, Any]] = []
    shard_index: list[dict[str, Any]] = []
    process_ids: set[str] = set()
    stage1_processes = {str(value["process_id"]) for value in stage1["records"]}
    for slot, path in expected_paths.items():
        verified = _verify_stage2_process_shard(
            path,
            contract=contract,
            seal=seal,
            top3=top3,
            expected_slot_index=slot,
        )
        process_id = str(verified["manifest"]["process_id"])
        if process_id in process_ids or process_id in stage1_processes:
            raise CalibrationExecutionError("stage2 process ID duplicated/reused")
        process_ids.add(process_id)
        records.extend(verified["records"])
        strength_diagnostics.extend(verified["strength_diagnostics"])
        shard_index.append(
            {
                "process_id": process_id,
                "slot_index": slot,
                "artifact_manifest_sha256": verified["manifest_sha256"],
            }
        )
    invariants = _verify_cross_run_endpoint_invariants(stage1["records"], records)
    strength_summary = _combine_strength_diagnostics(strength_diagnostics)
    selector_receipt = select_final_candidate(stage1["records"], records)
    receipt = {
        **selector_receipt,
        "endpoint_invariants": invariants,
        "diagnostics_not_used_for_selection": True,
        "update_activity_summary": strength_summary,
        "application_receipt": _application_receipt(
            contract, selector_receipt["selected_candidate"]
        ),
        **_scope_provenance(contract),
    }
    monitor.assert_unchanged(stage="final_aggregate_pre_publish", full_byte_rehash=True)
    provenance = {
        "schema_version": 2,
        "artifact_type": "binary_tent_ss_calibration_v2_final_aggregate",
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
        "top3_receipt_sha256": receipt_binding.sha256,
        "runtime_audits": monitor.audits,
        "endpoint_invariants": invariants,
        "update_activity_summary": strength_summary,
        **_scope_provenance(contract),
    }
    destination = contract.output_root / contract.execution["outputs"]["stage2_aggregate_directory"]
    published = core._publish_directory(
        final=destination,
        work_root=contract.publication_work_root,
        primary_files={
            "final_ss_selection_receipt.json": receipt,
            "stage2_records.jsonl": records,
            STRENGTH_DIAGNOSTICS_FILENAME: strength_diagnostics,
            "stage2_shard_index.json": {"shards": shard_index},
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_ss_calibration_v2_final_aggregate",
            "record_count": 234,
            "episode_count": 14976,
            "total_episode_count": 39936,
            "fresh_stage2_process_count": 2,
            "selection_bn_protocol": "SS",
            "BS_excluded_from_selection": True,
            "require_nonzero_parameter_update": False,
            "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
            "update_activity_contract": _update_activity_disclosure(),
            "update_activity_summary": strength_summary,
            "top3_in_frozen_order": [value.to_dict() for value in top3],
            "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
            "top3_receipt_sha256": receipt_binding.sha256,
            "endpoint_invariants": invariants,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            **_scope_provenance(contract),
        },
        completion_metadata={
            "stage": 2,
            "record_count": 234,
            "episode_count": 14976,
            "total_episode_count": 39936,
            "selected_candidate": receipt["selected_candidate"],
            "selection_bn_protocol": "SS",
            "BS_excluded_from_selection": True,
            "best_pd_reuses_same_frozen_hyperparameters_without_tuning": True,
            "diagnostics_not_used_for_selection": True,
            "require_nonzero_parameter_update": False,
            "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
            "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
            "top3_receipt_sha256": receipt_binding.sha256,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "scope": dict(SCOPE),
        },
    )
    post = _verify_final_aggregate(contract, base_seal, seal)
    return {**published, "post_publish_verified": True, "verified_manifest_sha256": post["manifest_sha256"]}


def validate_only(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all immutable bytes without tensor/model/CUDA/output access."""

    cuda_initialized_before = bool(torch.cuda.is_initialized())
    contract = load_contract(args.execution_config)
    output_before = (
        contract.output_root.exists(),
        contract.output_root.stat().st_mtime_ns if contract.output_root.exists() else None,
    )
    seal, caches = capture_runtime_seal(contract)
    monitor = RuntimeSealMonitor(seal)
    audit = monitor.assert_unchanged(stage="validate_only", full_byte_rehash=False)
    output_after = (
        contract.output_root.exists(),
        contract.output_root.stat().st_mtime_ns if contract.output_root.exists() else None,
    )
    _equal(output_after, output_before, "validate-only output state")
    cuda_initialized_after = bool(torch.cuda.is_initialized())
    _equal(cuda_initialized_after, cuda_initialized_before, "validate-only CUDA state")
    return {
        "valid": True,
        "role": "validate",
        "execution_protocol_id": EXECUTION_PROTOCOL_ID,
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "selection_bn_protocol": "SS",
        "selection_bn_protocol_id": SS_BN_PROTOCOL,
        "BS_excluded_from_selection": True,
        "diagnostic_detail": "global",
        "require_nonzero_parameter_update": False,
        "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
        "update_activity": _update_activity_disclosure(),
        "application_protocols": ["SS", "BS"],
        "best_pd_reuses_same_frozen_hyperparameters_without_tuning": True,
        "best_pd_additional_tuning_episodes": 0,
        "counts": {
            "cells_per_run": 39,
            "stage1_records": 390,
            "stage1_episodes": 24960,
            "stage2_records": 234,
            "stage2_episodes": 14976,
            "total_episodes": 39936,
        },
        "hardened_safety_core_sha256": HARDENED_CORE_SHA256,
        "cache_protocol_sha256": CACHE_PROTOCOL_SHA256,
        "cache_inventory_go_seal_sha256": CACHE_INVENTORY_GO_SEAL_SHA256,
        "cache_datasets": {
            dataset: {
                "manifest_sha256": context.manifest_sha256,
                "method_input_manifest_sha256": contract.execution[
                    "cache_protocol"
                ]["artifacts"][dataset]["method_input_manifest_sha256"],
                "complete_sha256": context.complete_sha256,
                "cache_content_sha256": context.manifest[
                    "cache_content_sha256"
                ],
                "payload_file_count": len(context.manifest["files"]),
                "ordered_ids_sha256": context.manifest["ordered_ids_sha256"],
                "targets_file_sha256": context.manifest["targets"]["file_sha256"],
            }
            for dataset, context in caches.items()
        },
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "bound_file_count": len(seal.bindings),
        "runtime_audit": audit,
        "metadata_only": True,
        "numpy_load_calls": 0,
        "model_constructions": 0,
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": cuda_initialized_after,
        "formal_output_writes": 0,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "validation_split_created": False,
        **_scope_provenance(contract),
    }


def _gpu_ids(raw: str) -> tuple[str, ...]:
    return core._gpu_ids(raw)


def launch_stage1(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    gpu_ids = _gpu_ids(args.gpu_ids)
    return core._with_stage_launcher_lock(
        contract,
        stage="ss-v2-stage1",
        action=lambda descriptor: _launch_stage1_locked(
            args, contract, gpu_ids, descriptor
        ),
    )


def _launch_stage1_locked(
    args: argparse.Namespace,
    contract: CalibrationContract,
    gpu_ids: tuple[str, ...],
    stage_claim_fd: int,
) -> dict[str, Any]:
    seal, _ = capture_runtime_seal(contract)
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(stage="stage1_launcher_entry")
    root = contract.output_root / "stage1" / "shards"
    expected_names = {candidate_slug(candidate) for candidate in ALL_CANDIDATES}
    if root.exists() or root.is_symlink():
        if not root.is_dir() or root.is_symlink():
            raise CalibrationExecutionError("stage1 shard root must be a real directory")
        unexpected = sorted(entry.name for entry in root.iterdir() if entry.name not in expected_names)
        if unexpected:
            raise CalibrationExecutionError(
                f"unexpected stage1 entries; fail closed without delete: {unexpected}"
            )
    commands: list[list[str]] = []
    skipped: list[dict[str, Any]] = []
    for candidate in ALL_CANDIDATES:
        shard = _stage1_shard_path(contract, candidate)
        if shard.exists() or shard.is_symlink():
            verified = _verify_stage1_candidate_shard(contract, seal, candidate)
            skipped.append(
                {
                    "candidate": candidate.to_dict(),
                    "process_id": verified["manifest"]["process_id"],
                    "artifact_manifest_sha256": verified["manifest_sha256"],
                }
            )
            continue
        commands.append(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker-stage1",
                "--execution-config",
                str(contract.execution_path),
                "--optimizer",
                candidate.optimizer,
                "--learning-rate",
                _decimal_text(candidate.learning_rate),
                "--process-id",
                core._fresh_process_id(f"ss-v2-stage1-{candidate_slug(candidate)}"),
                "--device",
                "cuda:0",
            ]
        )
    assignments = core._run_parallel(
        commands,
        gpu_ids,
        termination_timeout_seconds=float(
            contract.execution["launcher"]["failure_termination_timeout_seconds"]
        ),
        slot_lock_directory=_project_path(
            contract.execution["launcher"]["physical_gpu_lease_directory"]
        ),
        inherited_guard_fds=(stage_claim_fd,),
        lease_wait_heartbeat_seconds=float(
            contract.execution["launcher"]["lease_wait_heartbeat_seconds"]
        ),
    )
    monitor.assert_unchanged(stage="stage1_launcher_workers_complete", full_byte_rehash=True)
    records: list[dict[str, Any]] = []
    process_ids: set[str] = set()
    for candidate in ALL_CANDIDATES:
        verified = _verify_stage1_candidate_shard(contract, seal, candidate)
        process_id = str(verified["manifest"]["process_id"])
        if process_id in process_ids:
            raise CalibrationExecutionError("stage1 safe-resume process ID reused")
        process_ids.add(process_id)
        records.extend(verified["records"])
    invariants = _verify_cross_run_endpoint_invariants(records)
    select_stage1_top3(records)
    return {
        "launched": len(commands),
        "skipped_current_runtime_seal": len(skipped),
        "fresh_processes": len(commands),
        "gpu_ids": list(gpu_ids),
        "assignments": assignments,
        "safe_resume_verified": skipped,
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "endpoint_invariants": invariants,
        "selector_semantics_verified": True,
        "selection_bn_protocol": "SS",
        "BS_excluded_from_selection": True,
    }


def launch_stage2(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    gpu_ids = _gpu_ids(args.gpu_ids)
    return core._with_stage_launcher_lock(
        contract,
        stage="ss-v2-stage2",
        action=lambda descriptor: _launch_stage2_locked(
            args, contract, gpu_ids, descriptor
        ),
    )


def _launch_stage2_locked(
    args: argparse.Namespace,
    contract: CalibrationContract,
    gpu_ids: tuple[str, ...],
    stage_claim_fd: int,
) -> dict[str, Any]:
    expected_receipt = _stage1_receipt_path(contract).resolve()
    if args.top3_receipt is not None:
        _equal(
            Path(args.top3_receipt).expanduser().resolve(),
            expected_receipt,
            "canonical stage2 receipt path",
        )
    base_seal, _ = capture_runtime_seal(contract)
    stage1 = _verify_stage1_aggregate(contract, base_seal)
    seal, receipt_binding, stage1 = _bind_verified_stage1_receipt(
        contract, base_seal, stage1_verified=stage1
    )
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(stage="stage2_launcher_entry", active_paths=(expected_receipt,))
    top3 = _load_top3(expected_receipt)
    expected_ids = {
        slot: _stage2_slot_process_id(receipt_binding.sha256, slot)
        for slot in (1, 2)
    }
    root = contract.output_root / "stage2" / "shards"
    if root.exists() or root.is_symlink():
        if not root.is_dir() or root.is_symlink():
            raise CalibrationExecutionError("stage2 shard root must be a real directory")
        unexpected = sorted(
            entry.name for entry in root.iterdir() if entry.name not in set(expected_ids.values())
        )
        if unexpected:
            raise CalibrationExecutionError(
                f"unexpected stage2 entries; fail closed without delete: {unexpected}"
            )
    commands: list[list[str]] = []
    skipped: list[dict[str, Any]] = []
    for slot, process_id in expected_ids.items():
        shard = _stage2_shard_path(contract, process_id)
        if shard.exists() or shard.is_symlink():
            verified = _verify_stage2_process_shard(
                shard,
                contract=contract,
                seal=seal,
                top3=top3,
                expected_slot_index=slot,
            )
            skipped.append(
                {
                    "slot_index": slot,
                    "process_id": process_id,
                    "artifact_manifest_sha256": verified["manifest_sha256"],
                }
            )
            continue
        commands.append(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker-stage2",
                "--execution-config",
                str(contract.execution_path),
                "--top3-receipt",
                str(expected_receipt),
                "--process-id",
                process_id,
                "--slot-index",
                str(slot),
                "--device",
                "cuda:0",
            ]
        )
    assignments = core._run_parallel(
        commands,
        gpu_ids,
        termination_timeout_seconds=float(
            contract.execution["launcher"]["failure_termination_timeout_seconds"]
        ),
        slot_lock_directory=_project_path(
            contract.execution["launcher"]["physical_gpu_lease_directory"]
        ),
        inherited_guard_fds=(stage_claim_fd,),
        lease_wait_heartbeat_seconds=float(
            contract.execution["launcher"]["lease_wait_heartbeat_seconds"]
        ),
    )
    monitor.assert_unchanged(stage="stage2_launcher_workers_complete", full_byte_rehash=True)
    records: list[dict[str, Any]] = []
    for slot, process_id in expected_ids.items():
        records.extend(
            _verify_stage2_process_shard(
                _stage2_shard_path(contract, process_id),
                contract=contract,
                seal=seal,
                top3=top3,
                expected_slot_index=slot,
            )["records"]
        )
    invariants = _verify_cross_run_endpoint_invariants(stage1["records"], records)
    select_final_candidate(stage1["records"], records)
    return {
        "launched": len(commands),
        "skipped_current_runtime_seal": len(skipped),
        "fresh_processes": len(commands),
        "gpu_ids": list(gpu_ids),
        "assignments": assignments,
        "safe_resume_verified": skipped,
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "top3_receipt": str(expected_receipt),
        "top3_receipt_sha256": receipt_binding.sha256,
        "endpoint_invariants": invariants,
        "selector_semantics_verified": True,
        "selection_bn_protocol": "SS",
        "BS_excluded_from_selection": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)

    def add_config(value: argparse.ArgumentParser) -> None:
        value.add_argument(
            "--execution-config", type=Path, default=DEFAULT_EXECUTION_CONFIG
        )

    validate_parser = subparsers.add_parser("validate")
    add_config(validate_parser)

    stage1_worker = subparsers.add_parser("worker-stage1")
    add_config(stage1_worker)
    stage1_worker.add_argument("--optimizer", required=True)
    stage1_worker.add_argument("--learning-rate", required=True)
    stage1_worker.add_argument("--process-id", required=True)
    stage1_worker.add_argument("--device", default="cuda:0")

    stage2_worker = subparsers.add_parser("worker-stage2")
    add_config(stage2_worker)
    stage2_worker.add_argument("--top3-receipt", type=Path, required=True)
    stage2_worker.add_argument("--process-id", required=True)
    stage2_worker.add_argument("--slot-index", type=int, choices=(1, 2), required=True)
    stage2_worker.add_argument("--device", default="cuda:0")

    aggregate1 = subparsers.add_parser("aggregate-stage1")
    add_config(aggregate1)
    aggregate2 = subparsers.add_parser("aggregate-final")
    add_config(aggregate2)

    launch1 = subparsers.add_parser("launch-stage1")
    add_config(launch1)
    launch1.add_argument("--gpu-ids", default="1,2")
    launch2 = subparsers.add_parser("launch-stage2")
    add_config(launch2)
    launch2.add_argument("--gpu-ids", default="1,2")
    launch2.add_argument("--top3-receipt", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    dispatch = {
        "validate": validate_only,
        "worker-stage1": run_stage1_worker,
        "worker-stage2": run_stage2_worker,
        "aggregate-stage1": aggregate_stage1,
        "aggregate-final": aggregate_final,
        "launch-stage1": launch_stage1,
        "launch-stage2": launch_stage2,
    }
    return dispatch[args.role](args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(args)
    except (CalibrationExecutionError, FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
