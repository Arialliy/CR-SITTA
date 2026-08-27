"""Run and aggregate the train-derived Binary TENT optimizer/LR calibration.

The executable has deliberately narrow process roles:

* ``validate`` verifies protocols, checkpoints, and every completed cache file
  without loading a NumPy tensor, creating an output, or initialising CUDA;
* ``worker-stage1`` evaluates exactly one shared optimizer/LR candidate in one
  fresh process (4,992 episodes);
* ``worker-stage2`` evaluates the frozen stage-1 top three, in receipt order,
  inside one of exactly two additional fresh processes (14,976 episodes);
* ``aggregate-stage1`` and ``aggregate-final`` validate immutable shards and
  call the pure exact-rational selector;
* the two ``launch-*`` roles are lightweight two-GPU subprocess schedulers.

Targets live behind an outer-evaluator reader.  A method call receives only a
private image tensor and safe metadata; the corresponding target slice is not
read until that episode has returned.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any

import numpy as np
import torch
from torch import Tensor
import yaml

from dataio.corruption_cache import (
    ordered_ids_sha256,
    sha256_file,
    verify_cache_artifact,
)
from metrics.irstd_metrics import (
    IRSTDEvaluationProtocol,
    UnifiedResearchEvaluator,
)
from materialize_binary_tent_source_calibration_cache import (
    METHOD_FACING_SAMPLE_FIELDS,
    SourceCalibrationMethodInputDataset,
    _build_method_input_manifest,
    load_outer_evaluator_targets,
)
from tta.binary_tent import (
    BN_PROTOCOL_BATCH_STATS,
    BN_PROTOCOL_SOURCE_STATS,
    CUDA_BACKWARD_TEMPORARILY_DISABLE,
    BinaryTentMethod,
)
from tta.binary_tent_calibration_selector import (
    ALL_CANDIDATES,
    BN_PROTOCOLS,
    CONDITIONS,
    DATASETS,
    IMAGES_PER_CELL,
    REQUIRED_HARD_GATES,
    REQUIRED_PROTOCOL_AUDIT,
    Candidate,
    select_final_candidate,
    select_stage1_top3,
)
from tta.binary_tent_fast_runner import (
    BinaryTentFastEpisodeResult,
    BinaryTentFastRunner,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXECUTION_CONFIG = (
    PROJECT_ROOT / "configs" / "binary_tent_source_calibration_execution_v1.yaml"
)
EXPECTED_EXECUTION_PROTOCOL_ID = (
    "cr-sitta-binary-tent-source-calibration-execution-v1"
)
EXPECTED_SCIENTIFIC_PROTOCOL_ID = "cr-sitta-binary-tent-source-calibration-v1"
EXPECTED_CACHE_PROTOCOL_ID = (
    "cr-sitta-binary-tent-source-calibration-cache-v1"
)
JSON_SEPARATORS = (",", ":")
PROCESS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
AT_FDCWD = -100
RENAME_NOREPLACE = 1
SCOPE = {"paper_result": False, "source_train_derived": True}


class CalibrationExecutionError(RuntimeError):
    """Raised when an execution/provenance contract fails closed."""


@dataclass(frozen=True)
class BoundFile:
    path: str
    role: str
    sha256: str
    bytes: int
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class CacheContext:
    dataset: str
    root: Path
    manifest: Mapping[str, Any]
    complete: Mapping[str, Any]
    manifest_sha256: str
    complete_sha256: str
    condition_by_pair: Mapping[tuple[str, int], Mapping[str, Any]]


@dataclass(frozen=True)
class CalibrationContract:
    execution_path: Path
    execution: Mapping[str, Any]
    scientific_path: Path
    scientific: Mapping[str, Any]
    cache_protocol_path: Path
    cache_protocol: Mapping[str, Any]
    cache_root: Path
    output_root: Path
    critical_code_paths: tuple[Path, ...]
    checkpoints: Mapping[str, Path]


@dataclass(frozen=True)
class RuntimeSeal:
    schema_version: int
    algorithm: str
    bindings: tuple[BoundFile, ...]
    cache_lineage: Mapping[str, Any]
    global_runtime_seal_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "bindings": [asdict(value) for value in self.bindings],
            "cache_lineage": dict(self.cache_lineage),
            "global_runtime_seal_sha256": self.global_runtime_seal_sha256,
        }


def _project_path(raw: str | Path) -> Path:
    value = Path(raw).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise CalibrationExecutionError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"YAML file is missing or a symlink: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(_mapping(value, str(path)))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"JSON file is missing or a symlink: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(_mapping(value, str(path)))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=JSON_SEPARATORS,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _safe_slug(value: str) -> str:
    if not PROCESS_ID_RE.fullmatch(value):
        raise CalibrationExecutionError(
            "process_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
        )
    return value


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "E").replace("E+", "e+").replace("E-", "e-")


def candidate_slug(candidate: Candidate) -> str:
    lr = _decimal_text(candidate.learning_rate).replace("-", "m").replace("+", "p")
    return f"{candidate.optimizer}_lr_{lr}"


def _validate_scientific(scientific: Mapping[str, Any]) -> None:
    _require_equal(scientific.get("schema_version"), 1, "scientific schema")
    _require_equal(
        scientific.get("protocol_id"),
        EXPECTED_SCIENTIFIC_PROTOCOL_ID,
        "scientific protocol_id",
    )
    scope = _mapping(scientific.get("scope"), "scientific.scope")
    for key, expected in (
        ("paper_result", False),
        ("scientific_result_frozen", False),
        ("source_train_derived", True),
        ("independent_validation", False),
        ("use_test_images", False),
        ("use_test_labels", False),
        ("train_labels_outer_evaluator_only", True),
        ("method_receives_labels", False),
        ("target_transition_metrics_used_for_selection", False),
        ("entropy_decrease_required", False),
        ("arbitrary_performance_hard_thresholds", False),
    ):
        _require_equal(scope.get(key), expected, f"scientific.scope.{key}")
    inherited = _mapping(
        scientific.get("inherited_source_limitation"),
        "scientific.inherited_source_limitation",
    )
    _require_equal(
        inherited.get("checkpoint_role"),
        "best_miou",
        "inherited source checkpoint role",
    )
    _require_equal(
        inherited.get("checkpoint_selection"),
        "test_selected_during_source_training",
        "inherited source checkpoint selection",
    )
    if not str(inherited.get("disclosure", "")).strip():
        raise CalibrationExecutionError(
            "scientific inherited_source_limitation.disclosure is required"
        )

    subsets = _mapping(
        scientific.get("source_train_subsets"), "scientific.source_train_subsets"
    )
    _require_equal(subsets.get("subset_size_per_dataset"), 64, "subset size")
    _require_equal(tuple(_mapping(subsets.get("datasets"), "subset datasets")), DATASETS, "dataset order")

    raw_conditions = _sequence(
        _mapping(scientific.get("corruption_conditions"), "corruptions").get("ordered"),
        "scientific conditions",
    )
    observed_conditions = tuple((str(value[0]), int(value[1])) for value in raw_conditions)
    _require_equal(observed_conditions, CONDITIONS, "scientific condition order")

    method = _mapping(scientific.get("method"), "scientific.method")
    _require_equal(method.get("optimizer_steps_per_image"), 1, "optimizer steps")
    _require_equal(method.get("amp_enabled"), False, "AMP")
    _require_equal(
        tuple(_mapping(method.get("bn_protocols"), "bn protocols").get("ordered", ())),
        BN_PROTOCOLS,
        "BN protocol order",
    )
    candidates = _mapping(method.get("candidates"), "method.candidates")
    _require_equal(candidates.get("optimizer_order"), ["Adam", "SGD"], "optimizer order")
    _require_equal(
        tuple(Decimal(str(value)) for value in candidates.get("learning_rates", ())),
        tuple(value.learning_rate for value in ALL_CANDIDATES[:5]),
        "learning-rate order",
    )
    _require_equal(candidates.get("cross_product_count"), 10, "candidate count")
    _require_equal(
        dict(_mapping(candidates.get("Adam"), "Adam contract")),
        {
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        },
        "Adam optimizer contract",
    )
    _require_equal(
        dict(_mapping(candidates.get("SGD"), "SGD contract")),
        {
            "momentum": 0.9,
            "dampening": 0.0,
            "weight_decay": 0.0,
            "nesterov": True,
            "maximize": False,
            "foreach": False,
            "differentiable": False,
        },
        "SGD optimizer contract",
    )
    evaluation = _mapping(scientific.get("evaluation"), "scientific.evaluation")
    _require_equal(evaluation.get("fixed_probability_threshold"), 0.5, "threshold")
    _require_equal(evaluation.get("threshold_rule"), "strict_greater_than", "threshold rule")
    _require_equal(
        tuple(evaluation.get("forbidden_selector_inputs", ())),
        ("ATER", "ATRR", "NTG", "target_erasure", "target_recovery", "target_transitions"),
        "forbidden selector inputs",
    )
    gates = _mapping(scientific.get("required_hard_gates"), "hard gates")
    _require_equal(
        tuple(gates.get("exact_set_no_additional_performance_gates_allowed", ())),
        REQUIRED_HARD_GATES,
        "hard-gate exact set",
    )
    outputs = _mapping(scientific.get("outputs"), "scientific.outputs")
    _require_equal(outputs.get("root"), "results/binary_tent/source_calibration_v1", "output root")
    _require_equal(outputs.get("refuse_overwrite"), True, "refuse overwrite")
    _require_equal(outputs.get("atomic_sibling_staging"), True, "atomic staging")


def _validate_cache_protocol(
    cache_protocol: Mapping[str, Any], scientific: Mapping[str, Any]
) -> None:
    _require_equal(cache_protocol.get("schema_version"), 1, "cache protocol schema")
    _require_equal(
        cache_protocol.get("protocol_id"),
        EXPECTED_CACHE_PROTOCOL_ID,
        "cache protocol_id",
    )
    scope = _mapping(cache_protocol.get("scope"), "cache scope")
    for key, expected in (
        ("paper_result", False),
        ("independent_validation_set", False),
        ("use_test_images", False),
        ("use_test_labels", False),
        ("adaptation_receives_labels", False),
        ("targets_role", "outer_source_side_calibration_evaluator_only"),
        ("target_transition_hyperparameter_selection_allowed", False),
    ):
        _require_equal(scope.get(key), expected, f"cache scope.{key}")
    raw = _mapping(cache_protocol.get("input_protocol"), "cache input protocol")
    conditions = tuple((str(value[0]), int(value[1])) for value in raw.get("ordered_conditions", ()))
    _require_equal(conditions, CONDITIONS, "cache/scientific condition order")
    _require_equal(raw.get("subset_size_per_dataset"), IMAGES_PER_CELL, "cache subset size")
    materialized = _mapping(
        cache_protocol.get("materialized_cache"), "cache materialized_cache"
    )
    targets = _mapping(materialized.get("targets"), "cache target contract")
    _require_equal(
        targets.get("path"),
        "outer_evaluator/targets.npy",
        "outer evaluator target path",
    )
    _require_equal(
        targets.get("outer_evaluator_requires_explicit_episodes_complete"),
        True,
        "delayed outer evaluator access",
    )
    consumers = _mapping(
        materialized.get("official_consumers"), "cache official consumers"
    )
    _require_equal(
        consumers.get("method_facing"),
        "SourceCalibrationMethodInputDataset",
        "official method cache consumer",
    )
    _require_equal(
        consumers.get("method_facing_manifest"),
        "method_input_manifest.json",
        "sanitized method manifest",
    )
    _require_equal(
        consumers.get("outer_evaluator_targets"),
        "load_outer_evaluator_targets",
        "official delayed target loader",
    )
    scientific_datasets = _mapping(
        _mapping(scientific.get("source_train_subsets"), "source subsets").get("datasets"),
        "scientific datasets",
    )
    cache_datasets = _mapping(cache_protocol.get("datasets"), "cache datasets")
    _require_equal(tuple(cache_datasets), DATASETS, "cache dataset order")
    for dataset in DATASETS:
        sci = _mapping(scientific_datasets[dataset], f"scientific {dataset}")
        cache = _mapping(cache_datasets[dataset], f"cache {dataset}")
        for sci_key, cache_key in (
            ("ordered_ids_sha256", "pilot_ordered_ids_sha256"),
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("train_split_sha256", "train_split_sha256"),
        ):
            _require_equal(sci[sci_key], cache[cache_key], f"{dataset} {sci_key}")


def load_contract(path: str | Path = DEFAULT_EXECUTION_CONFIG) -> CalibrationContract:
    execution_path = Path(path).expanduser().resolve()
    execution = _load_yaml(execution_path)
    _require_equal(execution.get("schema_version"), 1, "execution schema")
    _require_equal(
        execution.get("execution_protocol_id"),
        EXPECTED_EXECUTION_PROTOCOL_ID,
        "execution protocol_id",
    )
    scientific_link = _mapping(execution.get("scientific_protocol"), "scientific link")
    cache_link = _mapping(execution.get("cache_protocol"), "cache link")
    scientific_path = _project_path(str(scientific_link["path"]))
    cache_protocol_path = _project_path(str(cache_link["path"]))
    _require_equal(sha256_file(scientific_path), scientific_link["sha256"], "scientific config SHA256")
    _require_equal(sha256_file(cache_protocol_path), cache_link["sha256"], "cache config SHA256")
    scientific = _load_yaml(scientific_path)
    cache_protocol = _load_yaml(cache_protocol_path)
    _validate_scientific(scientific)
    _validate_cache_protocol(cache_protocol, scientific)

    execution_values = _mapping(execution.get("execution"), "execution")
    for key, expected in (
        ("seed", 42),
        ("batch_size", 1),
        ("num_workers", 0),
        ("image_size", 256),
        ("amp_enabled", False),
        ("device_type", "cuda"),
        ("visible_cuda_devices_per_worker", 1),
        ("full_state_sha256_audit_cadence", 64),
        ("force_full_audit_at_every_cell_end", True),
        ("rebuild_model_method_optimizer_per_dataset_protocol", True),
        ("reapply_fixed_seed_before_every_candidate", True),
    ):
        _require_equal(execution_values.get(key), expected, f"execution.{key}")
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
        _require_equal(deterministic.get(key), expected, f"deterministic.{key}")
    evaluation = _mapping(execution.get("evaluation"), "execution evaluation")
    _require_equal(evaluation.get("froc_probability_thresholds"), [0.5], "FROC thresholds")
    _require_equal(evaluation.get("fixed_probability_threshold"), 0.5, "fixed threshold")
    _require_equal(evaluation.get("target_transition_fields_forbidden"), True, "transition firewall")
    _require_equal(evaluation.get("entropy_hard_gate"), False, "entropy hard gate")
    _require_equal(evaluation.get("performance_hard_gate"), False, "performance hard gate")

    runtime = _mapping(execution.get("runtime_seal"), "runtime seal")
    critical = tuple(_project_path(str(value)) for value in runtime.get("critical_code_paths", ()))
    if not critical or len(critical) != len(set(critical)):
        raise CalibrationExecutionError("critical code paths must be unique and non-empty")
    for critical_path in critical:
        if not critical_path.is_file() or critical_path.is_symlink():
            raise FileNotFoundError(f"critical runtime source is missing/symlink: {critical_path}")

    scientific_datasets = _mapping(
        _mapping(scientific["source_train_subsets"], "source subsets")["datasets"],
        "scientific datasets",
    )
    checkpoints = {
        dataset: _project_path(str(scientific_datasets[dataset]["checkpoint"]))
        for dataset in DATASETS
    }
    for dataset, checkpoint in checkpoints.items():
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise FileNotFoundError(f"{dataset} checkpoint missing/symlink: {checkpoint}")
        _require_equal(
            sha256_file(checkpoint),
            scientific_datasets[dataset]["checkpoint_sha256"],
            f"{dataset} checkpoint SHA256",
        )
    cache_root = _project_path(str(cache_link["root"]))
    materialized_cache_root = _project_path(
        str(
            _mapping(
                cache_protocol["materialized_cache"], "cache materialized_cache"
            )["root"]
        )
    )
    _require_equal(
        cache_root,
        materialized_cache_root,
        "execution/cache-protocol materialized cache root",
    )
    outputs = _mapping(execution["outputs"], "execution outputs")
    output_root = _project_path(str(outputs["root"]))
    _require_equal(
        str(output_root.relative_to(PROJECT_ROOT)),
        scientific["outputs"]["root"],
        "execution/scientific output root",
    )
    _require_equal(
        outputs.get("stage1_aggregate_directory"),
        "stage1/aggregate",
        "stage1 aggregate directory",
    )
    launcher = _mapping(execution.get("launcher"), "launcher")
    _require_equal(
        launcher.get("one_live_worker_per_physical_gpu"),
        True,
        "launcher GPU slot exclusivity",
    )
    termination_timeout = launcher.get("failure_termination_timeout_seconds")
    if (
        isinstance(termination_timeout, bool)
        or not isinstance(termination_timeout, (int, float))
        or float(termination_timeout) <= 0
    ):
        raise CalibrationExecutionError(
            "launcher.failure_termination_timeout_seconds must be positive"
        )
    stage1_launcher = _mapping(launcher.get("stage1"), "launcher.stage1")
    _require_equal(
        stage1_launcher.get("safe_resume_current_runtime_seal_only"),
        True,
        "stage1 safe-resume policy",
    )
    _require_equal(
        stage1_launcher.get("invalid_or_old_seal_candidate_shard_action"),
        "fail_closed_without_delete",
        "stage1 invalid-shard policy",
    )
    stage2_launcher = _mapping(launcher.get("stage2"), "launcher.stage2")
    _require_equal(
        stage2_launcher.get("top3_receipt"),
        "stage1/aggregate/stage1_top3_receipt.json",
        "stage2 top3 receipt path",
    )
    _require_equal(
        stage2_launcher.get("verify_stage1_aggregate_before_receipt_use"),
        True,
        "stage2 aggregate verification policy",
    )
    smoke = _mapping(execution.get("gpu_smoke"), "gpu smoke")
    for key, expected in (
        ("publishes_formal_output", False),
        ("fixed_dataset", "IRSTD-1K"),
        ("fixed_condition", ["clean", 0]),
        ("fixed_optimizer", "Adam"),
        ("fixed_learning_rate", 1.0e-5),
        ("default_episodes_per_bn_protocol", 1),
        ("optional_full_cell_episodes_per_bn_protocol", 64),
        ("bn_protocols", list(BN_PROTOCOLS)),
    ):
        _require_equal(smoke.get(key), expected, f"gpu_smoke.{key}")
    _require_equal(
        dict(_mapping(outputs.get("provenance_scope"), "output provenance scope")),
        SCOPE,
        "output provenance scope",
    )
    _require_equal(
        outputs.get("inherited_source_limitation_required"),
        True,
        "output inherited-source limitation policy",
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
        critical_code_paths=critical,
        checkpoints=checkpoints,
    )


def _bound_file(path: Path, *, role: str, expected_sha256: str | None = None) -> BoundFile:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"bound {role} file missing/symlink: {path}")
    observed = sha256_file(path)
    if expected_sha256 is not None:
        _require_equal(observed, expected_sha256, f"{role} SHA256")
    stat = path.stat()
    return BoundFile(
        path=str(path),
        role=role,
        sha256=observed,
        bytes=int(stat.st_size),
        device=int(stat.st_dev),
        inode=int(stat.st_ino),
        mtime_ns=int(stat.st_mtime_ns),
    )


def _verify_parent_pilot_anchor(
    contract: CalibrationContract, dataset: str
) -> tuple[dict[str, Any], tuple[BoundFile, ...]]:
    """Load the configured Pilot chain and bind its external source anchors."""

    dataset_contract = _mapping(
        _mapping(contract.cache_protocol.get("datasets"), "cache datasets").get(
            dataset
        ),
        f"cache dataset {dataset}",
    )
    parent = _mapping(dataset_contract.get("parent_pilot"), f"{dataset} parent Pilot")
    parent_specs = (
        ("artifact", "artifact_sha256", "pilot"),
        ("manifest", "manifest_sha256", "manifest"),
        ("complete", "complete_sha256", "complete"),
    )
    parent_paths: dict[str, Path] = {}
    bindings: list[BoundFile] = []
    for path_key, sha_key, role in parent_specs:
        path = _project_path(str(parent[path_key]))
        parent_paths[role] = path
        bindings.append(
            _bound_file(
                path,
                role=f"parent_pilot_{role}:{dataset}",
                expected_sha256=str(parent[sha_key]),
            )
        )

    pilot = _load_json(parent_paths["pilot"])
    pilot_manifest = _load_json(parent_paths["manifest"])
    pilot_complete = _load_json(parent_paths["complete"])
    _require_equal(pilot.get("dataset"), dataset, f"{dataset} Pilot dataset")
    _require_equal(
        pilot.get("formal_artifact"), True, f"{dataset} Pilot formal artifact"
    )
    _require_equal(
        pilot_complete.get("complete"), True, f"{dataset} Pilot completion"
    )
    _require_equal(
        pilot_complete.get("dataset"), dataset, f"{dataset} Pilot COMPLETE dataset"
    )
    _require_equal(
        pilot_complete.get("pilot_json_sha256"),
        parent["artifact_sha256"],
        f"{dataset} Pilot COMPLETE pilot lineage",
    )
    _require_equal(
        pilot_complete.get("artifact_manifest_sha256"),
        parent["manifest_sha256"],
        f"{dataset} Pilot COMPLETE manifest lineage",
    )
    manifest_files = _mapping(
        pilot_manifest.get("files_sha256"), f"{dataset} Pilot manifest files"
    )
    _require_equal(
        manifest_files.get("pilot.json"),
        parent["artifact_sha256"],
        f"{dataset} Pilot manifest pilot lineage",
    )
    pilot_protocol_sha = str(
        _mapping(
            _mapping(
                contract.cache_protocol.get("provenance_chain"),
                "cache provenance chain",
            ).get("round_02_pilot_protocol"),
            "round-02 Pilot protocol",
        )["sha256"]
    )
    _require_equal(
        pilot.get("protocol_sha256"),
        pilot_protocol_sha,
        f"{dataset} Pilot protocol SHA256",
    )
    _require_equal(
        pilot_manifest.get("protocol_sha256"),
        pilot_protocol_sha,
        f"{dataset} Pilot manifest protocol SHA256",
    )

    train_path = _project_path(str(dataset_contract["train_split"]))
    test_path = _project_path(str(dataset_contract["test_split"]))
    bindings.extend(
        (
            _bound_file(
                train_path,
                role=f"source_train_split:{dataset}",
                expected_sha256=str(dataset_contract["train_split_sha256"]),
            ),
            _bound_file(
                test_path,
                role=f"source_test_split_metadata:{dataset}",
                expected_sha256=str(dataset_contract["test_split_sha256"]),
            ),
        )
    )
    configured_checkpoint = _project_path(str(dataset_contract["checkpoint"]))
    _require_equal(
        configured_checkpoint,
        contract.checkpoints[dataset],
        f"{dataset} configured checkpoint path",
    )
    _require_equal(
        dataset_contract.get("checkpoint_sha256"),
        contract.scientific["source_train_subsets"]["datasets"][dataset][
            "checkpoint_sha256"
        ],
        f"{dataset} configured checkpoint SHA256",
    )
    for observed, expected, label in (
        (pilot.get("checkpoint_sha256"), dataset_contract["checkpoint_sha256"], "checkpoint"),
        (pilot.get("fixed_train_split_sha256"), dataset_contract["train_split_sha256"], "fixed train split"),
        (pilot.get("split_sha256"), dataset_contract["train_split_sha256"], "Pilot split"),
    ):
        _require_equal(observed, expected, f"{dataset} Pilot {label} SHA256")
    _require_equal(
        _project_path(str(pilot["checkpoint"])),
        configured_checkpoint,
        f"{dataset} Pilot checkpoint path",
    )
    _require_equal(
        _project_path(str(pilot["fixed_train_split"])),
        train_path,
        f"{dataset} Pilot fixed train path",
    )
    test_boundary = _mapping(
        pilot.get("fixed_test_boundary"), f"{dataset} Pilot fixed-test boundary"
    )
    _require_equal(
        _project_path(str(test_boundary["split_file"])),
        test_path,
        f"{dataset} Pilot fixed test path",
    )
    _require_equal(
        test_boundary.get("split_sha256"),
        dataset_contract["test_split_sha256"],
        f"{dataset} Pilot fixed test SHA256",
    )
    for key, expected in (
        ("test_dataset_constructed", False),
        ("test_images_opened", 0),
        ("test_masks_opened", 0),
    ):
        _require_equal(
            test_boundary.get(key), expected, f"{dataset} Pilot boundary.{key}"
        )

    selection = _mapping(pilot.get("selection"), f"{dataset} Pilot selection")
    selected_ids = tuple(str(value) for value in selection.get("selected_ids", ()))
    _require_equal(len(selected_ids), IMAGES_PER_CELL, f"{dataset} Pilot ID count")
    _require_equal(
        len(set(selected_ids)), IMAGES_PER_CELL, f"{dataset} Pilot ID uniqueness"
    )
    selected_ids_sha = ordered_ids_sha256(selected_ids)
    _require_equal(
        selected_ids_sha,
        dataset_contract["pilot_ordered_ids_sha256"],
        f"{dataset} Pilot ordered IDs SHA256",
    )
    selected_source_manifests = _mapping(
        selection.get("selected_source_manifests"),
        f"{dataset} Pilot source manifests",
    )
    expected_source_manifests = {
        "algorithm": "sorted-id-image_sha256-mask_sha256-lf-v1",
        "combined_sha256": str(dataset_contract["pilot_source_manifest_sha256"]),
        "images_sha256": str(dataset_contract["pilot_image_manifest_sha256"]),
        "masks_sha256": str(dataset_contract["pilot_mask_manifest_sha256"]),
    }
    _require_equal(
        dict(selected_source_manifests),
        expected_source_manifests,
        f"{dataset} Pilot source-file manifests",
    )

    pilot_by_condition: dict[tuple[str, int], Mapping[str, Any]] = {}
    for raw in _sequence(pilot.get("conditions"), f"{dataset} Pilot conditions"):
        record = _mapping(raw, f"{dataset} Pilot condition")
        key = (str(record["corruption"]), int(record["severity"]))
        if key in pilot_by_condition:
            raise CalibrationExecutionError(
                f"duplicate {dataset} parent Pilot condition: {key}"
            )
        pilot_by_condition[key] = record
    missing = [key for key in CONDITIONS if key not in pilot_by_condition]
    if missing:
        raise CalibrationExecutionError(
            f"{dataset} parent Pilot is missing calibration conditions: {missing}"
        )
    for key in CONDITIONS:
        record = pilot_by_condition[key]
        _require_equal(
            record.get("evaluated_images"),
            IMAGES_PER_CELL,
            f"{dataset} Pilot {key} image count",
        )
        _require_equal(
            record.get("evaluated_ids_sha256"),
            selected_ids_sha,
            f"{dataset} Pilot {key} IDs",
        )
        _require_equal(
            record.get("replay_model_input_tensor_sha256"),
            record.get("model_input_tensor_sha256"),
            f"{dataset} Pilot {key} model-input replay",
        )
        _require_equal(
            record.get("replay_gt_mask_tensor_sha256"),
            record.get("gt_mask_tensor_sha256"),
            f"{dataset} Pilot {key} GT replay",
        )

    return (
        {
            "dataset_contract": dataset_contract,
            "pilot": pilot,
            "pilot_manifest": pilot_manifest,
            "pilot_complete": pilot_complete,
            "pilot_by_condition": pilot_by_condition,
            "selected_ids": selected_ids,
            "selected_ids_sha256": selected_ids_sha,
            "source_manifests": expected_source_manifests,
            "train_path": train_path,
            "test_path": test_path,
            "parent_sha256": {
                role: str(parent[sha_key])
                for _path_key, sha_key, role in parent_specs
            },
        },
        tuple(bindings),
    )


def _verify_cache_context(
    contract: CalibrationContract, dataset: str
) -> tuple[CacheContext, tuple[BoundFile, ...], dict[str, Any]]:
    anchor, anchor_bindings = _verify_parent_pilot_anchor(contract, dataset)
    root = contract.cache_root / dataset
    manifest, audit = verify_cache_artifact(
        root,
        expected_protocol_sha256=sha256_file(contract.cache_protocol_path),
        verify_file_hashes=True,
    )
    complete = _load_json(root / "COMPLETE.json")
    method_manifest_path = root / "method_input_manifest.json"
    method_manifest = _load_json(method_manifest_path)
    dataset_contract = anchor["dataset_contract"]
    scientific_dataset = contract.scientific["source_train_subsets"]["datasets"][dataset]
    _require_equal(manifest.get("dataset"), dataset, f"{dataset} cache dataset")
    _require_equal(len(manifest.get("image_ids", ())), IMAGES_PER_CELL, f"{dataset} image count")
    _require_equal(manifest.get("ordered_ids_sha256"), scientific_dataset["ordered_ids_sha256"], f"{dataset} IDs")
    _require_equal(
        tuple(str(value) for value in manifest.get("image_ids", ())),
        anchor["selected_ids"],
        f"{dataset} cache/Pilot ordered IDs",
    )
    _require_equal(manifest.get("checkpoint_sha256"), scientific_dataset["checkpoint_sha256"], f"{dataset} cache checkpoint")
    _require_equal(
        _project_path(str(manifest["checkpoint"])),
        contract.checkpoints[dataset],
        f"{dataset} cache checkpoint path",
    )
    for key, expected in (
        ("train_split_sha256", dataset_contract["train_split_sha256"]),
        ("test_split_sha256", dataset_contract["test_split_sha256"]),
    ):
        _require_equal(manifest.get(key), expected, f"{dataset} cache {key}")
    _require_equal(
        _project_path(str(manifest["train_split"])),
        anchor["train_path"],
        f"{dataset} cache train split path",
    )
    _require_equal(
        _project_path(str(manifest["test_split_metadata"])),
        anchor["test_path"],
        f"{dataset} cache test split metadata path",
    )
    _require_equal(manifest.get("split_role"), "fixed_train_derived_sha256_ranked_64", f"{dataset} cache role")
    _require_equal(ordered_ids_sha256(tuple(manifest["image_ids"])), manifest["ordered_ids_sha256"], f"{dataset} ID hash")
    _require_equal(
        manifest.get("source_file_manifest_sha256"),
        dataset_contract["pilot_source_manifest_sha256"],
        f"{dataset} cache source-file manifest SHA256",
    )
    _require_equal(
        dict(_mapping(manifest.get("source_manifests"), f"{dataset} source manifests")),
        anchor["source_manifests"],
        f"{dataset} cache/Pilot source-file manifests",
    )
    boundary = _mapping(manifest.get("fixed_test_boundary"), f"{dataset} fixed test boundary")
    for key, expected in (
        ("test_dataset_constructed", False),
        ("test_images_opened", 0),
        ("test_masks_opened", 0),
    ):
        _require_equal(boundary.get(key), expected, f"{dataset} boundary.{key}")
    firewall = _mapping(manifest.get("label_firewall"), f"{dataset} label firewall")
    _require_equal(firewall.get("method_received_labels"), False, f"{dataset} method labels")
    _require_equal(
        firewall.get("targets_written_for_outer_evaluator_only"),
        True,
        f"{dataset} outer targets",
    )
    target = _mapping(manifest.get("targets"), f"{dataset} target cache")
    _require_equal(target.get("method_facing_access"), "forbidden", f"{dataset} target access")
    _require_equal(target.get("role"), "outer_source_side_calibration_evaluator_only", f"{dataset} target role")
    _require_equal(
        target.get("path"),
        "outer_evaluator/targets.npy",
        f"{dataset} delayed target path",
    )
    records = tuple(manifest.get("conditions", ()))
    pairs = tuple((str(value["corruption"]), int(value["severity"])) for value in records)
    _require_equal(pairs, CONDITIONS, f"{dataset} cached conditions")
    condition_by_pair = {
        (str(value["corruption"]), int(value["severity"])): value for value in records
    }
    files = _mapping(manifest.get("files"), f"{dataset} cache files")
    target_path = str(target["path"])
    target_file = _mapping(files.get(target_path), f"{dataset} target file")
    _require_equal(
        target.get("file_sha256"),
        target_file.get("sha256"),
        f"{dataset} target file SHA256",
    )
    expected_gt_hash = str(dataset_contract["expected_gt_tensor_sequence_sha256"])
    for key, expected in (
        ("tensor_sequence_sha256", expected_gt_hash),
        ("parent_pilot_tensor_sequence_sha256", expected_gt_hash),
    ):
        _require_equal(target.get(key), expected, f"{dataset} target {key}")
    for pair in CONDITIONS:
        cache_condition = _mapping(
            condition_by_pair[pair], f"{dataset} cache condition {pair}"
        )
        pilot_condition = _mapping(
            anchor["pilot_by_condition"][pair], f"{dataset} Pilot condition {pair}"
        )
        pilot_input_hash = pilot_condition.get("model_input_tensor_sha256")
        _require_equal(
            cache_condition.get("tensor_sequence_sha256"),
            cache_condition.get("parent_pilot_tensor_sequence_sha256"),
            f"{dataset} {pair} cache/declared-parent tensor hash",
        )
        _require_equal(
            cache_condition.get("parent_pilot_tensor_sequence_sha256"),
            pilot_input_hash,
            f"{dataset} {pair} parent-Pilot model-input tensor hash",
        )
        _require_equal(
            cache_condition.get("gt_mask_tensor_sequence_sha256"),
            expected_gt_hash,
            f"{dataset} {pair} cache GT tensor hash",
        )
        _require_equal(
            pilot_condition.get("gt_mask_tensor_sha256"),
            expected_gt_hash,
            f"{dataset} {pair} Pilot GT tensor hash",
        )

    expected_method_manifest = _build_method_input_manifest(
        manifest, outer_manifest_sha256=str(audit["manifest_sha256"])
    )
    _require_equal(
        method_manifest,
        expected_method_manifest,
        f"{dataset} sanitized/outer manifest semantic projection",
    )
    outer_image_paths = {str(value["path"]) for value in records}
    _require_equal(
        set(_mapping(method_manifest.get("files"), f"{dataset} method files")),
        outer_image_paths,
        f"{dataset} method manifest outer image-shard set",
    )
    _require_equal(
        set(files),
        outer_image_paths | {target_path},
        f"{dataset} outer cache exact image/target shard set",
    )
    _require_equal(
        complete.get("manifest_sha256"),
        audit["manifest_sha256"],
        f"{dataset} COMPLETE outer-manifest lineage",
    )
    _require_equal(
        complete.get("cache_content_sha256"),
        manifest.get("cache_content_sha256"),
        f"{dataset} COMPLETE cache-content lineage",
    )
    bindings: list[BoundFile] = list(anchor_bindings) + [
        _bound_file(root / "manifest.json", role=f"cache_manifest:{dataset}"),
        _bound_file(root / "COMPLETE.json", role=f"cache_complete:{dataset}"),
        _bound_file(
            method_manifest_path,
            role=f"cache_method_input_manifest:{dataset}",
            expected_sha256=str(complete["method_input_manifest_sha256"]),
        ),
    ]
    for relative, record in sorted(files.items()):
        bindings.append(
            _bound_file(
                root / str(relative),
                role=f"cache_payload:{dataset}:{relative}",
                expected_sha256=str(record["sha256"]),
            )
        )
    context = CacheContext(
        dataset=dataset,
        root=root,
        manifest=manifest,
        complete=complete,
        manifest_sha256=str(audit["manifest_sha256"]),
        complete_sha256=sha256_file(root / "COMPLETE.json"),
        condition_by_pair=condition_by_pair,
    )
    lineage = {
        "manifest_sha256": context.manifest_sha256,
        "complete_sha256": context.complete_sha256,
        "cache_content_sha256": manifest["cache_content_sha256"],
        "ordered_ids_sha256": manifest["ordered_ids_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "file_count": len(files),
        "method_input_manifest_sha256": complete[
            "method_input_manifest_sha256"
        ],
        "parent_pilot_sha256": anchor["parent_sha256"],
        "source_file_manifest_sha256": manifest["source_file_manifest_sha256"],
        "gt_tensor_sequence_sha256": expected_gt_hash,
        "train_split_sha256": manifest["train_split_sha256"],
        "test_split_sha256": manifest["test_split_sha256"],
        "condition_tensor_hashes_equal_parent_pilot": True,
        "gt_hashes_equal_parent_pilot": True,
        "sanitized_manifest_semantically_equal_outer_projection": True,
        "all_manifest_file_sha256_verified": True,
    }
    return context, tuple(bindings), lineage


def _runtime_seal_from_bindings(
    bindings: Sequence[BoundFile], cache_lineage: Mapping[str, Any]
) -> RuntimeSeal:
    paths = [value.path for value in bindings]
    if len(paths) != len(set(paths)):
        raise CalibrationExecutionError("runtime seal contains duplicate file bindings")
    ordered = tuple(sorted(bindings, key=lambda value: value.path))
    body = {
        "schema_version": 1,
        "algorithm": "calibration-runtime-seal-v1",
        "bindings": [asdict(value) for value in ordered],
        "cache_lineage": dict(cache_lineage),
    }
    return RuntimeSeal(
        schema_version=1,
        algorithm="calibration-runtime-seal-v1",
        bindings=ordered,
        cache_lineage=dict(cache_lineage),
        global_runtime_seal_sha256=_canonical_json_sha256(body),
    )


def _extend_runtime_seal(
    seal: RuntimeSeal, *additional_bindings: BoundFile
) -> RuntimeSeal:
    """Extend a just-captured entry seal without weakening any base binding."""

    return _runtime_seal_from_bindings(
        (*seal.bindings, *additional_bindings), seal.cache_lineage
    )


def capture_runtime_seal(
    contract: CalibrationContract,
) -> tuple[RuntimeSeal, Mapping[str, CacheContext]]:
    """Fully byte-hash every bound input at process entry."""

    bindings: list[BoundFile] = [
        _bound_file(contract.execution_path, role="execution_config"),
        _bound_file(contract.scientific_path, role="scientific_config"),
        _bound_file(contract.cache_protocol_path, role="cache_protocol"),
    ]
    bindings.extend(
        _bound_file(path, role=f"critical_code:{path.relative_to(PROJECT_ROOT)}")
        for path in contract.critical_code_paths
    )
    scientific_datasets = contract.scientific["source_train_subsets"]["datasets"]
    bindings.extend(
        _bound_file(
            contract.checkpoints[dataset],
            role=f"checkpoint:{dataset}",
            expected_sha256=scientific_datasets[dataset]["checkpoint_sha256"],
        )
        for dataset in DATASETS
    )
    caches: dict[str, CacheContext] = {}
    cache_lineage: dict[str, Any] = {}
    for dataset in DATASETS:
        context, cache_bindings, lineage = _verify_cache_context(contract, dataset)
        caches[dataset] = context
        cache_lineage[dataset] = lineage
        bindings.extend(cache_bindings)
    return _runtime_seal_from_bindings(bindings, cache_lineage), caches


class RuntimeSealMonitor:
    """Reverify one immutable process-entry byte seal throughout a worker."""

    def __init__(self, seal: RuntimeSeal) -> None:
        self.seal = seal
        self._by_path = {value.path: value for value in seal.bindings}
        self.audits: list[dict[str, Any]] = []

    def assert_unchanged(
        self,
        *,
        stage: str,
        active_paths: Sequence[Path] = (),
        full_byte_rehash: bool = False,
    ) -> dict[str, Any]:
        active = {str(path.resolve()) for path in active_paths}
        rehashed = 0
        for binding in self.seal.bindings:
            path = Path(binding.path)
            if not path.is_file() or path.is_symlink():
                raise CalibrationExecutionError(
                    f"runtime seal path disappeared/became symlink at {stage}: {path}"
                )
            stat = path.stat()
            identity = (
                int(stat.st_size),
                int(stat.st_dev),
                int(stat.st_ino),
                int(stat.st_mtime_ns),
            )
            expected = (
                binding.bytes,
                binding.device,
                binding.inode,
                binding.mtime_ns,
            )
            if identity != expected:
                raise CalibrationExecutionError(
                    f"runtime seal file identity drift at {stage}: {path}"
                )
            should_hash = (
                full_byte_rehash
                or binding.role.startswith(
                    (
                        "execution_config",
                        "scientific_config",
                        "cache_protocol",
                        "critical_code",
                        "cache_manifest",
                        "cache_complete",
                        "cache_method_input_manifest",
                        "parent_pilot_",
                        "source_train_split",
                        "source_test_split_metadata",
                        "stage1_top3_receipt",
                    )
                )
                or binding.path in active
            )
            if should_hash:
                rehashed += 1
                if sha256_file(path) != binding.sha256:
                    raise CalibrationExecutionError(
                        f"runtime seal byte drift at {stage}: {path}"
                    )
        audit = {
            "stage": stage,
            "verified": True,
            "global_runtime_seal_sha256": self.seal.global_runtime_seal_sha256,
            "bound_file_count": len(self.seal.bindings),
            "rehashed_file_count": rehashed,
            "all_bound_file_identity_and_metadata_verified": True,
            "full_byte_rehash": bool(full_byte_rehash),
            "active_paths_rehashed": sorted(active),
        }
        self.audits.append(audit)
        return audit


def run_label_free_episode(
    runner: BinaryTentFastRunner,
    *,
    image: Tensor,
    metadata: Mapping[str, Any],
) -> BinaryTentFastEpisodeResult:
    """The sole adaptation boundary: intentionally has no target argument."""

    forbidden = {"mask", "masks", "label", "labels", "target", "targets", "gt"}
    overlap = forbidden & {str(key).lower() for key in metadata}
    if overlap:
        raise CalibrationExecutionError(
            f"unsafe adaptation metadata keys: {sorted(overlap)}"
        )
    return runner.run_one_image(image=image, metadata=metadata)


def _endpoint_counts(evaluator: UnifiedResearchEvaluator) -> dict[str, int]:
    fixed = evaluator.compute().fixed
    pixel = fixed.pixel
    target = fixed.target
    counts = {
        "intersection_pixels": int(pixel.true_positive_pixels),
        "union_pixels": int(
            pixel.true_positive_pixels
            + pixel.false_positive_pixels
            + pixel.false_negative_pixels
        ),
        "false_alarm_pixels": int(target.false_alarm_pixels),
        "total_image_pixels": int(target.total_image_pixels),
        "detected_targets": int(target.detected_targets),
        "total_targets": int(target.total_targets),
    }
    if counts["union_pixels"] <= 0 or counts["total_targets"] <= 0:
        raise CalibrationExecutionError(
            "selector v1 requires positive union and target denominators per cell"
        )
    return counts


def build_cell_record(
    *,
    stage: int,
    process_id: str,
    candidate: Candidate,
    dataset: str,
    bn_protocol: str,
    corruption: str,
    severity: int,
    tent_pre: Mapping[str, int],
    tent_post: Mapping[str, int],
    protocol_audit: Mapping[str, bool],
    image_count: int = IMAGES_PER_CELL,
) -> dict[str, Any]:
    """Build the selector-facing record with exactly the frozen hard gates."""

    _require_equal(set(protocol_audit), set(REQUIRED_PROTOCOL_AUDIT), "protocol audit set")
    if not all(protocol_audit.values()):
        raise CalibrationExecutionError("cannot emit a failed protocol audit")
    return {
        "stage": int(stage),
        "process_id": _safe_slug(process_id),
        "fresh_process": True,
        "candidate": candidate.to_dict(),
        "dataset": dataset,
        "bn_protocol": bn_protocol,
        "corruption": corruption,
        "severity": int(severity),
        "image_count": int(image_count),
        "optimizer_steps_total": int(image_count),
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "hard_gates": {key: True for key in REQUIRED_HARD_GATES},
        "protocol_audit": dict(protocol_audit),
        "endpoints": {
            "tent_pre": {key: int(value) for key, value in tent_pre.items()},
            "tent_post": {key: int(value) for key, value in tent_post.items()},
        },
    }


def _configure_cuda_worker(contract: CalibrationContract, device_text: str) -> torch.device:
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise CalibrationExecutionError("GPU worker requires PYTHONHASHSEED=42 at process entry")
    expected_workspace = contract.execution["execution"]["deterministic_forwards"]["cublas_workspace_config"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_workspace:
        raise CalibrationExecutionError(
            f"GPU worker requires CUBLAS_WORKSPACE_CONFIG={expected_workspace}"
        )
    if not device_text.startswith("cuda"):
        raise CalibrationExecutionError("calibration workers are CUDA-only")
    if not torch.cuda.is_available():
        raise CalibrationExecutionError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise CalibrationExecutionError("a calibration worker must see exactly one CUDA device")
    device = torch.device(device_text)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise CalibrationExecutionError("strict deterministic forward policy was not enabled")
    return device


def _seed_candidate(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "python_seed_applied": True,
        "numpy_seed_applied": True,
        "torch_seed_applied": True,
        "cuda_seed_applied": bool(torch.cuda.is_available()),
    }


def _build_fast_runner(
    *,
    contract: CalibrationContract,
    dataset: str,
    candidate: Candidate,
    bn_protocol: str,
    device: torch.device,
) -> tuple[BinaryTentFastRunner, dict[str, Any]]:
    # Lazy import keeps validate-only away from model/custom-op/CUDA construction.
    import test_source as source_runner

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
        diagnostic_detail="aggregate",
        cuda_backward_determinism_policy=CUDA_BACKWARD_TEMPORARILY_DISABLE,
    )
    state = EpisodicStateManager(model, optimizer=method.optimizer)
    runner = BinaryTentFastRunner(
        adapter,
        state,
        method,
        full_audit_cadence=64,
    )
    return runner, {
        "build_instance_id": uuid.uuid4().hex,
        "model_object_id": id(model),
        "method_object_id": id(method),
        "optimizer_object_id": id(method.optimizer),
        "source_state_sha256": state.source_fingerprint.full_sha256,
        "checkpoint_wrapper": wrapper,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_state_dict_strict_load_verified": True,
        "checkpoint_matches_dataset": dataset,
    }


def _audit_episode(
    result: BinaryTentFastEpisodeResult,
    *,
    bn_protocol: str,
    expect_full_audit: bool,
) -> None:
    diagnostics = result.diagnostics
    checks = result.checks
    required_checks = (
        "safe_label_free_metadata",
        "batch_size_one",
        "source_pre_no_grad",
        "tent_pre_detached",
        "tent_post_no_grad",
        "only_bn_affine_temporarily_trainable",
        "non_bn_parameters_exact_every_gate",
        "all_registered_buffers_exact_every_gate",
        "parameter_module_optimizer_topology_exact_every_gate",
        "post_forward_state_unchanged",
        "all_gradients_cleared",
        "source_runtime_restored_per_image",
        "bn_affine_restored_per_image",
        "optimizer_restored_per_image",
        "input_unchanged",
        "rng_restored",
    )
    if any(checks.get(key) is not True for key in required_checks):
        raise CalibrationExecutionError("Binary TENT fast episode invariant failed")
    if diagnostics.get("bn_protocol") != bn_protocol:
        raise CalibrationExecutionError("Binary TENT BN protocol drift")
    if diagnostics.get("optimizer_steps") != 1:
        raise CalibrationExecutionError("episode did not execute exactly one optimizer step")
    finite_values = (
        result.logits_source_pre,
        result.logits_tent_pre,
        result.logits_tent_post,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise CalibrationExecutionError("episode returned NaN/Inf logits")
    for key in ("entropy_pre", "entropy_post", "gradient_norm", "step_norm"):
        if not math.isfinite(float(diagnostics.get(key, math.nan))):
            raise CalibrationExecutionError(f"non-finite episode diagnostic: {key}")
    # Entropy direction and performance are intentionally not gates.
    policy = {
        "forward": diagnostics.get("forward_deterministic_algorithms_enabled") is True
        and diagnostics.get("forward_deterministic_algorithms_warn_only") is False,
        "backward": diagnostics.get("backward_deterministic_algorithms_enabled") is False
        and diagnostics.get("backward_deterministic_algorithms_warn_only") is False,
        "restored": diagnostics.get("deterministic_policy_restored_after_backward") is True,
        "device": diagnostics.get("episode_device_type") == "cuda",
        "configured": diagnostics.get("cuda_backward_determinism_policy")
        == CUDA_BACKWARD_TEMPORARILY_DISABLE,
    }
    if not all(policy.values()):
        raise CalibrationExecutionError(f"CUDA deterministic policy failed: {policy}")
    if bn_protocol == BN_PROTOCOL_SOURCE_STATS and not result.source_tent_pre_bit_exact:
        raise CalibrationExecutionError("source-stat TENT-pre must equal Source")
    if result.full_audit_performed is not expect_full_audit:
        raise CalibrationExecutionError("full SHA audit cadence drift")
    if expect_full_audit:
        if result.full_audit_reason != "cadence":
            raise CalibrationExecutionError("cell-end full audit was not cadence-driven")
        if result.reset_full_fingerprint is None:
            raise CalibrationExecutionError("cell-end reset fingerprint is missing")


def _emit_cell_progress(
    *,
    stage: int,
    process_id: str,
    candidate: Candidate,
    dataset: str,
    bn_protocol: str,
    corruption: str,
    severity: int,
    completed_cells: int,
) -> None:
    """Emit one machine-readable stderr event after each 64-image cell."""

    event = {
        "event": "binary_tent_source_calibration_cell_complete",
        "stage": int(stage),
        "process_id": process_id,
        "candidate": candidate.to_dict(),
        "dataset": dataset,
        "bn_protocol": bn_protocol,
        "corruption": corruption,
        "severity": int(severity),
        "cell_images": IMAGES_PER_CELL,
        "completed_cells_for_candidate": int(completed_cells),
        "total_cells_for_candidate": len(DATASETS) * len(BN_PROTOCOLS) * len(CONDITIONS),
        "completed_episodes_for_candidate": int(completed_cells) * IMAGES_PER_CELL,
    }
    print(
        json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=JSON_SEPARATORS,
            allow_nan=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def execute_candidate(
    *,
    contract: CalibrationContract,
    caches: Mapping[str, CacheContext],
    monitor: RuntimeSealMonitor,
    stage: int,
    process_id: str,
    candidate: Candidate,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all 78 cells for one shared candidate."""

    seed_audit = _seed_candidate(42)
    records: list[dict[str, Any]] = []
    build_receipts: list[dict[str, Any]] = []
    candidate_started = time.perf_counter()
    for dataset in DATASETS:
        context = caches[dataset]
        for bn_protocol in BN_PROTOCOLS:
            runner, build = _build_fast_runner(
                contract=contract,
                dataset=dataset,
                candidate=candidate,
                bn_protocol=bn_protocol,
                device=device,
            )
            build_receipts.append(
                {
                    "dataset": dataset,
                    "bn_protocol": bn_protocol,
                    **build,
                    "fresh_model_method_optimizer": True,
                }
            )
            for corruption, severity in CONDITIONS:
                condition_record = context.condition_by_pair[(corruption, severity)]
                active_image_path = context.root / str(condition_record["path"])
                target_path = context.root / str(context.manifest["targets"]["path"])
                monitor.assert_unchanged(
                    stage=(
                        f"candidate:{candidate_slug(candidate)}:{dataset}:"
                        f"{bn_protocol}:{corruption}_S{severity}:condition_start"
                    ),
                    # Target bytes are not opened here.  Their already-bound
                    # identity is checked with every other seal entry; a byte
                    # rehash and mmap happen only after all 64 method calls.
                    active_paths=(active_image_path,),
                )
                method_inputs = SourceCalibrationMethodInputDataset(
                    context.root,
                    corruption=corruption,
                    severity=severity,
                    expected_protocol_sha256=str(
                        contract.execution["cache_protocol"]["sha256"]
                    ),
                )
                _require_equal(
                    tuple(method_inputs.method_metadata["sample_fields"]),
                    METHOD_FACING_SAMPLE_FIELDS,
                    "sanitized method-facing schema",
                )
                _require_equal(
                    method_inputs.targets_opened,
                    False,
                    "method consumer target-open flag",
                )
                evaluation_protocol = IRSTDEvaluationProtocol(
                    fixed_probability_threshold=0.5,
                    froc_probability_thresholds=(0.5,),
                )
                pre_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
                post_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
                cell_start_episode = runner.completed_episodes
                returned_predictions: list[tuple[Tensor, Tensor]] = []
                for index in range(len(method_inputs)):
                    sanitized_sample = method_inputs[index]
                    _require_equal(
                        tuple(sanitized_sample),
                        METHOD_FACING_SAMPLE_FIELDS,
                        "method sample exact fields",
                    )
                    image = sanitized_sample.pop("image").unsqueeze(0)
                    if image.shape != (1, 3, 256, 256):
                        raise CalibrationExecutionError(
                            "method-facing private image shape drift"
                        )
                    metadata = dict(sanitized_sample)
                    image = image.to(device, non_blocking=False)
                    result = run_label_free_episode(
                        runner,
                        image=image,
                        metadata=metadata,
                    )
                    expect_full = index == IMAGES_PER_CELL - 1
                    _audit_episode(
                        result,
                        bn_protocol=bn_protocol,
                        expect_full_audit=expect_full,
                    )
                    returned_predictions.append(
                        (result.logits_tent_pre, result.logits_tent_post)
                    )
                _require_equal(len(returned_predictions), 64, "returned predictions")
                _require_equal(
                    runner.completed_episodes - cell_start_episode,
                    64,
                    "cell episode count",
                )
                _require_equal(
                    method_inputs.targets_opened,
                    False,
                    "method consumer remained target-free",
                )
                # Every method-facing episode in this cell has returned.  Only
                # now may the outer evaluator hash/map/index its separate target
                # shard.  Neither the method nor fast runner holds this object.
                monitor.assert_unchanged(
                    stage=(
                        f"candidate:{candidate_slug(candidate)}:{dataset}:"
                        f"{bn_protocol}:{corruption}_S{severity}:"
                        "method_episodes_complete_pre_outer_evaluation"
                    ),
                    active_paths=(active_image_path, target_path),
                )
                outer_targets = load_outer_evaluator_targets(
                    context.root,
                    episodes_complete=True,
                    expected_protocol_sha256=str(
                        contract.execution["cache_protocol"]["sha256"]
                    ),
                )
                for index, (tent_pre, tent_post) in enumerate(returned_predictions):
                    outer_target = np.array(
                        outer_targets[index], dtype=np.float32, copy=True, order="C"
                    )
                    pre_evaluator.update_logits(tent_pre, outer_target)
                    post_evaluator.update_logits(tent_post, outer_target)
                protocol_audit = {
                    "candidate_model_method_optimizer_rebuilt_before_run": True,
                    "global_runtime_seal_valid": True,
                    "fixed_seed_contract_valid": all(seed_audit.values()),
                }
                records.append(
                    build_cell_record(
                        stage=stage,
                        process_id=process_id,
                        candidate=candidate,
                        dataset=dataset,
                        bn_protocol=bn_protocol,
                        corruption=corruption,
                        severity=severity,
                        tent_pre=_endpoint_counts(pre_evaluator),
                        tent_post=_endpoint_counts(post_evaluator),
                        protocol_audit=protocol_audit,
                    )
                )
                _emit_cell_progress(
                    stage=stage,
                    process_id=process_id,
                    candidate=candidate,
                    dataset=dataset,
                    bn_protocol=bn_protocol,
                    corruption=corruption,
                    severity=severity,
                    completed_cells=len(records),
                )
                # No transition/entropy/performance values enter this record.
                del (
                    method_inputs,
                    outer_targets,
                    returned_predictions,
                    pre_evaluator,
                    post_evaluator,
                )
            _require_equal(runner.completed_episodes, 13 * 64, "runner episode count")
            runner.state.assert_source_state()
            del runner
    _require_equal(len(records), 78, "candidate cell record count")
    build_instance_ids = [value["build_instance_id"] for value in build_receipts]
    _require_equal(
        len(set(build_instance_ids)),
        len(build_instance_ids),
        "fresh build instance identities",
    )
    return records, {
        "candidate": candidate.to_dict(),
        "cell_record_count": len(records),
        "episode_count": len(records) * 64,
        "model_method_optimizer_build_count": len(build_receipts),
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
                "checkpoint_matches_dataset": value[
                    "checkpoint_matches_dataset"
                ],
            }
            for value in build_receipts
        ],
        "fixed_seed_audit": seed_audit,
        "runtime_seconds": time.perf_counter() - candidate_started,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "target_access_order": "outer_evaluator_after_episode_return_only",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=JSON_SEPARATORS,
                    allow_nan=False,
                )
                + "\n"
            )
            count += 1
    return count


def _artifact_files(root: Path, excluded: Sequence[str]) -> dict[str, dict[str, Any]]:
    excluded_set = set(excluded)
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        if path.is_symlink():
            raise CalibrationExecutionError(f"artifact file cannot be symlink: {path}")
        result[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing an existing destination."""

    if not sys.platform.startswith("linux"):
        raise CalibrationExecutionError(
            "atomic no-replace directory publication requires Linux renameat2"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CalibrationExecutionError(
            "libc renameat2(RENAME_NOREPLACE) is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"refusing overwrite; destination appeared during publish: {destination}",
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _publish_directory(
    *,
    final: Path,
    primary_files: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    completion_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    final.parent.mkdir(parents=True, exist_ok=True)
    lock = final.with_name(f".{final.name}.publish.lock")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(lock, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"publish lock already exists for destination: {lock}"
        ) from error
    staging = final.with_name(
        f".{final.name}.build-{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    )
    try:
        os.write(
            lock_fd,
            f"pid={os.getpid()} destination={final}\n".encode("utf-8"),
        )
        if final.exists():
            raise FileExistsError(
                f"refusing overwrite of completed destination: {final}"
            )
        staging.mkdir(parents=False)
        for relative, value in primary_files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith(".jsonl"):
                _write_jsonl(path, value)
            else:
                _write_json(path, value)
        files = _artifact_files(
            staging, excluded=("artifact_manifest.json", "COMPLETE.json")
        )
        manifest = {
            "schema_version": 1,
            **dict(manifest_metadata),
            "files": files,
        }
        _write_json(staging / "artifact_manifest.json", manifest)
        completion = {
            "complete": True,
            **dict(completion_metadata),
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
        }
        _write_json(staging / "COMPLETE.json", completion)
        _rename_directory_noreplace(staging, final)
    except BaseException:
        if staging.is_dir() and staging.parent == final.parent:
            shutil.rmtree(staging)
        raise
    finally:
        os.close(lock_fd)
        try:
            lock.unlink()
        except FileNotFoundError as error:
            raise CalibrationExecutionError(
                f"publish lock disappeared before release: {lock}"
            ) from error
    return {
        "destination": str(final),
        "manifest": manifest,
        "completion": completion,
    }


def _candidate_from_args(optimizer: str, learning_rate: str) -> Candidate:
    return Candidate.from_values(optimizer, learning_rate)


def _stage1_shard_path(contract: CalibrationContract, candidate: Candidate) -> Path:
    return contract.output_root / "stage1" / "shards" / candidate_slug(candidate)


def _stage2_shard_path(contract: CalibrationContract, process_id: str) -> Path:
    return contract.output_root / "stage2" / "shards" / _safe_slug(process_id)


def _stage1_aggregate_path(contract: CalibrationContract) -> Path:
    return contract.output_root / "stage1" / "aggregate"


def _stage1_receipt_path(contract: CalibrationContract) -> Path:
    return _stage1_aggregate_path(contract) / "stage1_top3_receipt.json"


def _resolve_top3_receipt(
    contract: CalibrationContract, requested: Path | str | None
) -> Path:
    expected = _stage1_receipt_path(contract).resolve()
    if requested is not None:
        observed = Path(requested).expanduser().resolve()
        _require_equal(
            observed,
            expected,
            "stage2 top3 receipt must be output_root/stage1/aggregate/"
            "stage1_top3_receipt.json",
        )
    return expected


def _scope_provenance(contract: CalibrationContract) -> dict[str, Any]:
    limitation = dict(
        _mapping(
            contract.scientific["inherited_source_limitation"],
            "inherited source limitation",
        )
    )
    limitation.update(
        {
            "same_parent_pilot_64_image_subset_reused": True,
            "calibration_subset_disclosure": (
                "All candidates and both stages reuse the same frozen 64-image "
                "source-train subset inherited from the round-02 corruption Pilot; "
                "this is calibration reuse, not independent validation."
            ),
        }
    )
    return {
        "scope": dict(SCOPE),
        "inherited_source_limitation": limitation,
    }


def run_stage1_worker(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    candidate = _candidate_from_args(args.optimizer, args.learning_rate)
    process_id = _safe_slug(args.process_id)
    destination = _stage1_shard_path(contract, candidate)
    if destination.exists():
        raise FileExistsError(f"stage-1 candidate shard already exists: {destination}")
    seal, caches = capture_runtime_seal(contract)
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(stage="stage1_process_entry", full_byte_rehash=False)
    device = _configure_cuda_worker(contract, args.device)
    records, summary = execute_candidate(
        contract=contract,
        caches=caches,
        monitor=monitor,
        stage=1,
        process_id=process_id,
        candidate=candidate,
        device=device,
    )
    monitor.assert_unchanged(stage="stage1_pre_publish", full_byte_rehash=True)
    provenance = {
        "schema_version": 1,
        "stage": 1,
        "fresh_process": True,
        "process_id": process_id,
        "pid": os.getpid(),
        "candidate": candidate.to_dict(),
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "runtime_audits": monitor.audits,
        "protocol_audit": {key: True for key in REQUIRED_PROTOCOL_AUDIT},
        "hard_gate_set": list(REQUIRED_HARD_GATES),
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "paper_result": False,
        **_scope_provenance(contract),
    }
    return _publish_directory(
        final=destination,
        primary_files={
            "records.jsonl": records,
            "run_summary.json": summary,
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_source_calibration_stage1_candidate_shard",
            "stage": 1,
            "process_id": process_id,
            "candidate": candidate.to_dict(),
            "record_count": 78,
            "episode_count": 4992,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            **_scope_provenance(contract),
        },
        completion_metadata={
            "stage": 1,
            "process_id": process_id,
            "candidate": candidate.to_dict(),
            "record_count": 78,
            "episode_count": 4992,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "scope": dict(SCOPE),
        },
    )


def _load_top3(path: Path) -> tuple[Candidate, ...]:
    receipt = _load_json(path)
    _require_equal(receipt.get("receipt_type"), "stage1_top3", "stage1 receipt type")
    raw = _sequence(receipt.get("top3"), "stage1 top3")
    values = tuple(
        Candidate.from_values(value["optimizer"], value["learning_rate"])
        for value in raw
    )
    if len(values) != 3 or len(set(values)) != 3:
        raise CalibrationExecutionError("stage1 receipt must contain three unique candidates")
    return values


def run_stage2_worker(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    process_id = _safe_slug(args.process_id)
    destination = _stage2_shard_path(contract, process_id)
    if destination.exists():
        raise FileExistsError(f"stage-2 process shard already exists: {destination}")
    receipt_path = _resolve_top3_receipt(
        contract, getattr(args, "top3_receipt", None)
    )
    base_seal, caches = capture_runtime_seal(contract)
    _verify_stage1_aggregate(contract, base_seal)
    receipt_binding = _bound_file(
        receipt_path, role="stage1_top3_receipt"
    )
    seal = _extend_runtime_seal(base_seal, receipt_binding)
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(
        stage="stage2_process_entry", active_paths=(receipt_path,)
    )
    top3 = _load_top3(receipt_path)
    device = _configure_cuda_worker(contract, args.device)
    all_records: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    build_guard: set[tuple[int, int, int]] = set()
    for rank, candidate in enumerate(top3, start=1):
        monitor.assert_unchanged(stage=f"stage2_candidate_{rank}_pre_rebuild")
        records, summary = execute_candidate(
            contract=contract,
            caches=caches,
            monitor=monitor,
            stage=2,
            process_id=process_id,
            candidate=candidate,
            device=device,
        )
        all_records.extend(records)
        candidate_summaries.append({"frozen_top3_rank": rank, **summary})
        # execute_candidate itself verifies six unique fresh object triples;
        # each call returns only after all objects are released/reset.
        build_guard.add((rank, len(records), int(summary["model_method_optimizer_build_count"])))
        monitor.assert_unchanged(stage=f"stage2_candidate_{rank}_complete")
    _require_equal(len(all_records), 234, "stage2 shard record count")
    _require_equal(len(build_guard), 3, "stage2 candidate rebuild count")
    monitor.assert_unchanged(stage="stage2_pre_publish", full_byte_rehash=True)
    summary = {
        "stage": 2,
        "process_id": process_id,
        "fresh_process": True,
        "top3_in_frozen_order": [value.to_dict() for value in top3],
        "candidate_summaries": candidate_summaries,
        "record_count": 234,
        "episode_count": 14976,
        "candidate_model_method_optimizer_rebuilt_before_each_candidate": True,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
    }
    provenance = {
        "schema_version": 1,
        "stage": 2,
        "fresh_process": True,
        "process_id": process_id,
        "pid": os.getpid(),
        "top3_receipt": str(receipt_path),
        "top3_receipt_sha256": receipt_binding.sha256,
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "runtime_audits": monitor.audits,
        "protocol_audit": {key: True for key in REQUIRED_PROTOCOL_AUDIT},
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "paper_result": False,
        **_scope_provenance(contract),
    }
    return _publish_directory(
        final=destination,
        primary_files={
            "records.jsonl": all_records,
            "run_summary.json": summary,
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_source_calibration_stage2_process_shard",
            "stage": 2,
            "process_id": process_id,
            "top3_in_frozen_order": [value.to_dict() for value in top3],
            "record_count": 234,
            "episode_count": 14976,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
            "top3_receipt_sha256": receipt_binding.sha256,
            **_scope_provenance(contract),
        },
        completion_metadata={
            "stage": 2,
            "process_id": process_id,
            "record_count": 234,
            "episode_count": 14976,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "top3_receipt_sha256": receipt_binding.sha256,
            "scope": dict(SCOPE),
        },
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CalibrationExecutionError(f"blank JSONL line: {path}:{line_number}")
            value = json.loads(line)
            records.append(dict(_mapping(value, f"{path}:{line_number}")))
    return records


def verify_shard(path: Path, *, expected_seal_sha256: str) -> dict[str, Any]:
    manifest_path = path / "artifact_manifest.json"
    complete_path = path / "COMPLETE.json"
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    _require_equal(complete.get("complete"), True, "shard completion")
    _require_equal(complete.get("artifact_manifest_sha256"), sha256_file(manifest_path), "shard manifest lineage")
    _require_equal(manifest.get("global_runtime_seal_sha256"), expected_seal_sha256, "shard runtime seal")
    _require_equal(complete.get("global_runtime_seal_sha256"), expected_seal_sha256, "completion runtime seal")
    files = _mapping(manifest.get("files"), "shard manifest files")
    actual = _artifact_files(path, excluded=("artifact_manifest.json", "COMPLETE.json"))
    _require_equal(actual, dict(files), "shard exact file set/hashes")
    record_candidates = (
        path / "records.jsonl",
        path / "stage1_records.jsonl",
        path / "stage2_records.jsonl",
    )
    present = tuple(value for value in record_candidates if value.is_file())
    _require_equal(len(present), 1, "artifact record file count")
    records = _load_jsonl(present[0])
    _require_equal(len(records), int(manifest["record_count"]), "shard record count")
    _require_equal(len(records) * 64, int(manifest["episode_count"]), "shard episode count")
    return {
        "path": str(path),
        "manifest": manifest,
        "complete": complete,
        "records": records,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _verify_embedded_scope(value: Mapping[str, Any], *, label: str) -> None:
    _require_equal(
        dict(_mapping(value.get("scope"), f"{label}.scope")),
        SCOPE,
        f"{label} scope",
    )
    limitation = _mapping(
        value.get("inherited_source_limitation"),
        f"{label}.inherited_source_limitation",
    )
    _require_equal(
        limitation.get("checkpoint_selection"),
        "test_selected_during_source_training",
        f"{label} inherited checkpoint selection",
    )


def _verify_stage1_candidate_shard(
    contract: CalibrationContract,
    seal: RuntimeSeal,
    candidate: Candidate,
) -> dict[str, Any]:
    path = _stage1_shard_path(contract, candidate)
    verified = verify_shard(
        path, expected_seal_sha256=seal.global_runtime_seal_sha256
    )
    manifest = verified["manifest"]
    complete = verified["complete"]
    _require_equal(
        manifest.get("artifact_type"),
        "binary_tent_source_calibration_stage1_candidate_shard",
        "stage1 candidate artifact type",
    )
    for value, expected, label in (
        (manifest.get("stage"), 1, "manifest stage"),
        (complete.get("stage"), 1, "completion stage"),
        (manifest.get("candidate"), candidate.to_dict(), "manifest candidate"),
        (complete.get("candidate"), candidate.to_dict(), "completion candidate"),
        (manifest.get("record_count"), 78, "record count"),
        (manifest.get("episode_count"), 4992, "episode count"),
    ):
        _require_equal(value, expected, f"stage1 {label}")
    _require_equal(
        set(_mapping(manifest.get("files"), "stage1 candidate files")),
        {"records.jsonl", "run_summary.json", "provenance.json", "runtime_seal.json"},
        "stage1 candidate exact primary files",
    )
    process_id = _safe_slug(str(manifest["process_id"]))
    _require_equal(
        complete.get("process_id"), process_id, "stage1 completion process ID"
    )
    observed_cells: set[tuple[str, str, str, int]] = set()
    for record in verified["records"]:
        _require_equal(record.get("stage"), 1, "stage1 record stage")
        _require_equal(
            record.get("candidate"), candidate.to_dict(), "stage1 record candidate"
        )
        _require_equal(
            record.get("process_id"), process_id, "stage1 record process ID"
        )
        _require_equal(
            record.get("image_count"), IMAGES_PER_CELL, "stage1 cell image count"
        )
        _require_equal(
            record.get("optimizer_steps_total"),
            IMAGES_PER_CELL,
            "stage1 cell optimizer steps",
        )
        _require_equal(
            set(_mapping(record.get("hard_gates"), "stage1 hard gates")),
            set(REQUIRED_HARD_GATES),
            "stage1 hard-gate set",
        )
        _require_equal(
            set(_mapping(record.get("protocol_audit"), "stage1 protocol audit")),
            set(REQUIRED_PROTOCOL_AUDIT),
            "stage1 protocol-audit set",
        )
        cell = (
            str(record["dataset"]),
            str(record["bn_protocol"]),
            str(record["corruption"]),
            int(record["severity"]),
        )
        if cell in observed_cells:
            raise CalibrationExecutionError(f"duplicate stage1 cell: {cell}")
        observed_cells.add(cell)
    expected_cells = {
        (dataset, protocol, corruption, severity)
        for dataset in DATASETS
        for protocol in BN_PROTOCOLS
        for corruption, severity in CONDITIONS
    }
    _require_equal(observed_cells, expected_cells, "stage1 complete candidate cells")
    _require_equal(
        _load_json(path / "runtime_seal.json"),
        seal.to_dict(),
        "stage1 embedded runtime seal",
    )
    provenance = _load_json(path / "provenance.json")
    _verify_embedded_scope(provenance, label="stage1 provenance")
    _require_equal(
        provenance.get("global_runtime_seal_sha256"),
        seal.global_runtime_seal_sha256,
        "stage1 provenance runtime seal",
    )
    return verified


def _verify_stage1_aggregate(
    contract: CalibrationContract, seal: RuntimeSeal
) -> dict[str, Any]:
    path = _stage1_aggregate_path(contract)
    verified = verify_shard(
        path, expected_seal_sha256=seal.global_runtime_seal_sha256
    )
    manifest = verified["manifest"]
    complete = verified["complete"]
    _require_equal(
        manifest.get("artifact_type"),
        "binary_tent_source_calibration_stage1_aggregate",
        "stage1 aggregate artifact type",
    )
    for value, expected, label in (
        (manifest.get("record_count"), 780, "record count"),
        (manifest.get("episode_count"), 49920, "episode count"),
        (manifest.get("candidate_count"), 10, "candidate count"),
        (manifest.get("fresh_process_count"), 10, "fresh process count"),
        (complete.get("stage"), 1, "completion stage"),
    ):
        _require_equal(value, expected, f"stage1 aggregate {label}")
    _require_equal(
        set(_mapping(manifest.get("files"), "stage1 aggregate files")),
        {
            "stage1_top3_receipt.json",
            "stage1_records.jsonl",
            "shard_index.json",
            "provenance.json",
            "runtime_seal.json",
        },
        "stage1 aggregate exact primary files",
    )
    expected_receipt = select_stage1_top3(verified["records"])
    observed_receipt = _load_json(_stage1_receipt_path(contract))
    _require_equal(observed_receipt, expected_receipt, "stage1 aggregate receipt")
    _require_equal(
        complete.get("top3"), expected_receipt["top3"], "stage1 completion top3"
    )
    _require_equal(
        _load_json(path / "runtime_seal.json"),
        seal.to_dict(),
        "stage1 aggregate embedded runtime seal",
    )
    provenance = _load_json(path / "provenance.json")
    _verify_embedded_scope(provenance, label="stage1 aggregate provenance")
    _verify_embedded_scope(manifest, label="stage1 aggregate manifest")
    return verified


def _verify_stage2_process_shard(
    path: Path,
    *,
    seal: RuntimeSeal,
    top3: Sequence[Candidate],
) -> dict[str, Any]:
    verified = verify_shard(
        path, expected_seal_sha256=seal.global_runtime_seal_sha256
    )
    manifest = verified["manifest"]
    _require_equal(
        manifest.get("artifact_type"),
        "binary_tent_source_calibration_stage2_process_shard",
        "stage2 process artifact type",
    )
    _require_equal(manifest.get("stage"), 2, "stage2 manifest stage")
    _require_equal(manifest.get("record_count"), 234, "stage2 record count")
    _require_equal(manifest.get("episode_count"), 14976, "stage2 episode count")
    _require_equal(
        set(_mapping(manifest.get("files"), "stage2 process files")),
        {"records.jsonl", "run_summary.json", "provenance.json", "runtime_seal.json"},
        "stage2 process exact primary files",
    )
    _require_equal(
        manifest.get("top3_in_frozen_order"),
        [value.to_dict() for value in top3],
        "stage2 manifest frozen top3",
    )
    _require_equal(
        _load_json(path / "runtime_seal.json"),
        seal.to_dict(),
        "stage2 embedded runtime seal",
    )
    provenance = _load_json(path / "provenance.json")
    _verify_embedded_scope(provenance, label="stage2 provenance")
    _verify_embedded_scope(manifest, label="stage2 manifest")
    receipt_bindings = [
        value for value in seal.bindings if value.role.startswith("stage1_top3_receipt")
    ]
    _require_equal(len(receipt_bindings), 1, "stage2 receipt seal binding count")
    _require_equal(
        provenance.get("top3_receipt_sha256"),
        receipt_bindings[0].sha256,
        "stage2 provenance receipt SHA256",
    )
    return verified


def aggregate_stage1(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    seal, _caches = capture_runtime_seal(contract)
    monitor = RuntimeSealMonitor(seal)
    shards: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    process_ids: set[str] = set()
    for candidate in ALL_CANDIDATES:
        verified = _verify_stage1_candidate_shard(contract, seal, candidate)
        manifest = verified["manifest"]
        _require_equal(manifest.get("candidate"), candidate.to_dict(), "stage1 shard candidate")
        process_id = str(manifest["process_id"])
        if process_id in process_ids:
            raise CalibrationExecutionError("stage1 candidates did not use fresh unique process IDs")
        process_ids.add(process_id)
        records.extend(verified["records"])
        shards.append(
            {
                "candidate": candidate.to_dict(),
                "process_id": process_id,
                "artifact_manifest_sha256": verified["manifest_sha256"],
            }
        )
    receipt = select_stage1_top3(records)
    monitor.assert_unchanged(stage="stage1_aggregate_pre_publish", full_byte_rehash=True)
    destination = _stage1_aggregate_path(contract)
    provenance = {
        "schema_version": 1,
        "artifact_type": "binary_tent_source_calibration_stage1_aggregate",
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "runtime_audits": monitor.audits,
        **_scope_provenance(contract),
    }
    return _publish_directory(
        final=destination,
        primary_files={
            "stage1_top3_receipt.json": receipt,
            "stage1_records.jsonl": records,
            "shard_index.json": {"shards": shards},
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_source_calibration_stage1_aggregate",
            "record_count": 780,
            "episode_count": 49920,
            "candidate_count": 10,
            "fresh_process_count": 10,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            **_scope_provenance(contract),
        },
        completion_metadata={
            "stage": 1,
            "record_count": 780,
            "episode_count": 49920,
            "top3": receipt["top3"],
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "scope": dict(SCOPE),
        },
    )


def aggregate_final(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    base_seal, _caches = capture_runtime_seal(contract)
    stage1_verified = _verify_stage1_aggregate(contract, base_seal)
    receipt_path = _stage1_receipt_path(contract)
    receipt_binding = _bound_file(
        receipt_path, role="stage1_top3_receipt"
    )
    seal = _extend_runtime_seal(base_seal, receipt_binding)
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(
        stage="final_aggregate_process_entry", active_paths=(receipt_path,)
    )
    stage1_records = stage1_verified["records"]
    top3 = _load_top3(receipt_path)
    stage1_processes = {str(value["process_id"]) for value in stage1_records}
    stage2_root = contract.output_root / "stage2" / "shards"
    paths = sorted(value for value in stage2_root.iterdir() if value.is_dir() and not value.name.startswith("."))
    _require_equal(len(paths), 2, "stage2 fresh process shard count")
    stage2_records: list[dict[str, Any]] = []
    process_ids: set[str] = set()
    shard_index: list[dict[str, Any]] = []
    for path in paths:
        verified = _verify_stage2_process_shard(path, seal=seal, top3=top3)
        manifest = verified["manifest"]
        process_id = str(manifest["process_id"])
        if process_id in process_ids or process_id in stage1_processes:
            raise CalibrationExecutionError("stage2 process is duplicate or reused from stage1")
        process_ids.add(process_id)
        observed_order: list[Candidate] = []
        for record in verified["records"]:
            candidate = Candidate.from_values(
                record["candidate"]["optimizer"], record["candidate"]["learning_rate"]
            )
            if not observed_order or observed_order[-1] != candidate:
                observed_order.append(candidate)
        _require_equal(tuple(observed_order), top3, "stage2 frozen top3 execution order")
        stage2_records.extend(verified["records"])
        shard_index.append(
            {
                "process_id": process_id,
                "artifact_manifest_sha256": verified["manifest_sha256"],
            }
        )
    receipt = {
        **select_final_candidate(stage1_records, stage2_records),
        **_scope_provenance(contract),
    }
    monitor.assert_unchanged(stage="final_aggregate_pre_publish", full_byte_rehash=True)
    destination = contract.output_root / str(contract.execution["outputs"]["stage2_aggregate_directory"])
    provenance = {
        "schema_version": 1,
        "artifact_type": "binary_tent_source_calibration_final_aggregate",
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
        "top3_receipt_sha256": receipt_binding.sha256,
        "runtime_audits": monitor.audits,
        **_scope_provenance(contract),
    }
    return _publish_directory(
        final=destination,
        primary_files={
            "final_selection_receipt.json": receipt,
            "stage2_records.jsonl": stage2_records,
            "stage2_shard_index.json": {"shards": shard_index},
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_source_calibration_final_aggregate",
            "record_count": 468,
            "stage2_episode_count": 29952,
            "total_episode_count": 79872,
            "fresh_stage2_process_count": 2,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
            "top3_receipt_sha256": receipt_binding.sha256,
            **_scope_provenance(contract),
        },
        completion_metadata={
            "stage": 2,
            "record_count": 468,
            "stage2_episode_count": 29952,
            "total_episode_count": 79872,
            "selected_candidate": receipt["selected_candidate"],
            "both_bn_protocols_retained": True,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "scope": dict(SCOPE),
        },
    )


def validate_only(args: argparse.Namespace) -> dict[str, Any]:
    cuda_before = bool(torch.cuda.is_initialized())
    contract = load_contract(args.execution_config)
    seal, caches = capture_runtime_seal(contract)
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(stage="validate_only_complete", full_byte_rehash=False)
    cuda_after = bool(torch.cuda.is_initialized())
    if cuda_after != cuda_before:
        raise CalibrationExecutionError("validate-only changed CUDA initialisation state")
    return {
        "validate_only": True,
        "execution_protocol": str(contract.execution_path),
        "scientific_protocol": str(contract.scientific_path),
        "cache_protocol": str(contract.cache_protocol_path),
        "datasets": list(DATASETS),
        "cache_manifest_file_hashes_verified": {
            dataset: caches[dataset].manifest_sha256 for dataset in DATASETS
        },
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "bound_file_count": len(seal.bindings),
        "cache_tensor_arrays_loaded": 0,
        "cache_payloads_opaque_byte_hashed_without_numpy_load": True,
        "parent_pilot_artifact_chains_verified": True,
        "source_train_and_test_split_hashes_verified": True,
        "cache_parent_pilot_tensor_hash_equality_verified": True,
        "sanitized_manifest_semantic_projection_verified": True,
        "model_constructed": False,
        "cuda_initialization_state_unchanged": True,
        "output_created": False,
        "planned_output_root": str(contract.output_root),
        "stage1_candidate_processes": 10,
        "stage1_episodes": 49920,
        "stage2_fresh_processes": 2,
        "stage2_episodes": 29952,
        "total_episodes": 79872,
        "test_image_opens": 0,
        "test_label_opens": 0,
        **_scope_provenance(contract),
    }


def run_gpu_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed IRSTD clean Adam/1e-5 CUDA smoke without publishing."""

    contract = load_contract(args.execution_config)
    seal, caches = capture_runtime_seal(contract)
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(stage="gpu_smoke_process_entry")
    device = _configure_cuda_worker(contract, args.device)
    dataset = "IRSTD-1K"
    candidate = Candidate.from_values("Adam", "1e-5")
    corruption, severity = ("clean", 0)
    episode_count = IMAGES_PER_CELL if bool(args.full_cell_smoke) else 1
    context = caches[dataset]
    condition = context.condition_by_pair[(corruption, severity)]
    image_path = context.root / str(condition["path"])
    receipts: list[dict[str, Any]] = []
    for bn_protocol in BN_PROTOCOLS:
        _seed_candidate(42)
        smoke_runner, build = _build_fast_runner(
            contract=contract,
            dataset=dataset,
            candidate=candidate,
            bn_protocol=bn_protocol,
            device=device,
        )
        method_inputs = SourceCalibrationMethodInputDataset(
            context.root,
            corruption=corruption,
            severity=severity,
            expected_protocol_sha256=str(
                contract.execution["cache_protocol"]["sha256"]
            ),
        )
        monitor.assert_unchanged(
            stage=f"gpu_smoke:{bn_protocol}:start", active_paths=(image_path,)
        )
        for index in range(episode_count):
            sample = method_inputs[index]
            _require_equal(
                tuple(sample), METHOD_FACING_SAMPLE_FIELDS, "smoke sample fields"
            )
            image = sample.pop("image").unsqueeze(0).to(device, non_blocking=False)
            result = run_label_free_episode(
                smoke_runner, image=image, metadata=dict(sample)
            )
            _audit_episode(
                result,
                bn_protocol=bn_protocol,
                expect_full_audit=episode_count == IMAGES_PER_CELL
                and index == IMAGES_PER_CELL - 1,
            )
        _require_equal(
            method_inputs.targets_opened, False, "smoke method remained target-free"
        )
        smoke_runner.state.assert_source_state()
        receipts.append(
            {
                "bn_protocol": bn_protocol,
                "episodes": episode_count,
                "optimizer_steps": episode_count,
                "build_instance_id": build["build_instance_id"],
                "checkpoint_sha256": build["checkpoint_sha256"],
                "completed": True,
            }
        )
        del method_inputs, smoke_runner
    monitor.assert_unchanged(stage="gpu_smoke_complete", full_byte_rehash=True)
    return {
        "gpu_smoke": True,
        "published_output": False,
        "output_created": False,
        "dataset": dataset,
        "condition": {"corruption": corruption, "severity": severity},
        "candidate": candidate.to_dict(),
        "episodes_per_bn_protocol": episode_count,
        "full_64_image_cell_smoke": episode_count == IMAGES_PER_CELL,
        "bn_protocol_receipts": receipts,
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        **_scope_provenance(contract),
    }


def _fresh_process_id(prefix: str) -> str:
    return _safe_slug(f"{prefix}-{uuid.uuid4().hex[:16]}")


def _worker_environment(gpu_id: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "42",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _terminate_kill_and_reap(
    workers: Sequence[subprocess.Popen[Any]], *, timeout_seconds: float
) -> None:
    """Best-effort bounded termination that always waits every child."""

    unique: list[subprocess.Popen[Any]] = []
    seen: set[int] = set()
    for process in workers:
        marker = id(process)
        if marker not in seen:
            seen.add(marker)
            unique.append(process)
    for process in unique:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + float(timeout_seconds)
    for process in unique:
        if process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            break
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    for process in unique:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    for process in unique:
        try:
            process.wait()
        except ChildProcessError:
            # Already reaped by a platform-specific Popen.poll implementation.
            pass


def _run_parallel(
    commands: Sequence[list[str] | tuple[list[str], str]],
    gpu_ids: Sequence[str] | None = None,
    *,
    max_parallel: int | None = None,
    termination_timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.2,
) -> list[dict[str, Any]]:
    """Schedule one live worker per physical-GPU slot and reuse freed slots."""

    if gpu_ids is None:
        legacy = [value for value in commands if isinstance(value, tuple)]
        if len(legacy) != len(commands):
            raise TypeError("gpu_ids are required for unassigned worker commands")
        gpu_ids = tuple(dict.fromkeys(str(value[1]) for value in legacy))
    slots = tuple(str(value) for value in gpu_ids)
    if not slots or len(slots) != len(set(slots)):
        raise CalibrationExecutionError("physical GPU slots must be unique and non-empty")
    if max_parallel is not None and int(max_parallel) != len(slots):
        raise CalibrationExecutionError(
            "max_parallel must equal the number of exclusive physical GPU slots"
        )
    if termination_timeout_seconds <= 0 or poll_interval_seconds < 0:
        raise ValueError("scheduler timeouts must be non-negative and termination positive")
    pending = [value[0] if isinstance(value, tuple) else value for value in commands]
    active: dict[str, tuple[subprocess.Popen[Any], list[str]]] = {}
    assignments: list[dict[str, Any]] = []
    try:
        while pending or active:
            for gpu in slots:
                if not pending:
                    break
                if gpu in active:
                    continue
                command = pending.pop(0)
                process = subprocess.Popen(command, env=_worker_environment(gpu))
                active[gpu] = (process, command)
                assignments.append(
                    {"gpu_id": gpu, "pid": process.pid, "command": list(command)}
                )

            completed_slots: list[str] = []
            failure: tuple[int, list[str], str] | None = None
            for gpu, (process, command) in tuple(active.items()):
                code = process.poll()
                if code is None:
                    continue
                process.wait()
                completed_slots.append(gpu)
                if code != 0 and failure is None:
                    failure = (int(code), command, gpu)
            if failure is not None:
                _terminate_kill_and_reap(
                    [value[0] for value in active.values()],
                    timeout_seconds=termination_timeout_seconds,
                )
                active.clear()
                raise CalibrationExecutionError(
                    f"worker exited {failure[0]} on physical GPU {failure[2]}: "
                    f"{' '.join(failure[1])}"
                )
            for gpu in completed_slots:
                active.pop(gpu)
            if active and not completed_slots and poll_interval_seconds:
                time.sleep(poll_interval_seconds)
    except BaseException:
        if active:
            _terminate_kill_and_reap(
                [value[0] for value in active.values()],
                timeout_seconds=termination_timeout_seconds,
            )
            active.clear()
        raise
    return assignments


def _gpu_ids(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if len(values) != 2 or len(set(values)) != 2 or any(not value.isdigit() for value in values):
        raise argparse.ArgumentTypeError("--gpu-ids requires two distinct physical IDs, e.g. 1,2")
    return values


def launch_stage1(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    gpu_ids = _gpu_ids(args.gpu_ids)
    seal, _caches = capture_runtime_seal(contract)
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(stage="stage1_launcher_entry")
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
        command = [
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
            _fresh_process_id(f"stage1-{candidate_slug(candidate)}"),
            "--device",
            "cuda:0",
        ]
        commands.append(command)
    timeout = float(
        contract.execution["launcher"]["failure_termination_timeout_seconds"]
    )
    assignments = _run_parallel(
        commands,
        gpu_ids,
        termination_timeout_seconds=timeout,
    )
    monitor.assert_unchanged(stage="stage1_launcher_workers_complete", full_byte_rehash=True)
    verified_process_ids: set[str] = set()
    for candidate in ALL_CANDIDATES:
        verified = _verify_stage1_candidate_shard(contract, seal, candidate)
        process_id = str(verified["manifest"]["process_id"])
        if process_id in verified_process_ids:
            raise CalibrationExecutionError(
                "stage1 safe-resume set contains duplicate process IDs"
            )
        verified_process_ids.add(process_id)
    return {
        "launched": len(commands),
        "skipped_current_runtime_seal": len(skipped),
        "fresh_processes": len(commands),
        "gpu_ids": list(gpu_ids),
        "assignments": assignments,
        "safe_resume_verified": skipped,
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
    }


def launch_stage2(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.execution_config)
    gpu_ids = _gpu_ids(args.gpu_ids)
    receipt = _resolve_top3_receipt(contract, getattr(args, "top3_receipt", None))
    base_seal, _caches = capture_runtime_seal(contract)
    _verify_stage1_aggregate(contract, base_seal)
    receipt_binding = _bound_file(
        receipt, role="stage1_top3_receipt"
    )
    seal = _extend_runtime_seal(base_seal, receipt_binding)
    monitor = RuntimeSealMonitor(seal)
    monitor.assert_unchanged(
        stage="stage2_launcher_entry", active_paths=(receipt,)
    )
    top3 = _load_top3(receipt)
    stage2_root = contract.output_root / "stage2" / "shards"
    if stage2_root.exists():
        existing = tuple(stage2_root.iterdir())
        if existing:
            raise FileExistsError(
                f"refusing stage2 launch with existing shard entries: {stage2_root}"
            )
    commands: list[list[str]] = []
    process_ids: list[str] = []
    for index, gpu in enumerate(gpu_ids, start=1):
        process_id = _fresh_process_id(f"stage2-process-{index}")
        process_ids.append(process_id)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker-stage2",
            "--execution-config",
            str(contract.execution_path),
            "--top3-receipt",
            str(receipt),
            "--process-id",
            process_id,
            "--device",
            "cuda:0",
        ]
        commands.append(command)
    timeout = float(
        contract.execution["launcher"]["failure_termination_timeout_seconds"]
    )
    assignments = _run_parallel(
        commands,
        gpu_ids,
        termination_timeout_seconds=timeout,
    )
    monitor.assert_unchanged(stage="stage2_launcher_workers_complete", full_byte_rehash=True)
    for process_id in process_ids:
        _verify_stage2_process_shard(
            _stage2_shard_path(contract, process_id), seal=seal, top3=top3
        )
    return {
        "launched": 2,
        "fresh_processes": 2,
        "gpu_ids": list(gpu_ids),
        "assignments": assignments,
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "top3_receipt": str(receipt),
        "top3_receipt_sha256": receipt_binding.sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--execution-config", type=Path, default=DEFAULT_EXECUTION_CONFIG
    )

    subparsers.add_parser("validate", parents=[common])

    worker1 = subparsers.add_parser("worker-stage1", parents=[common])
    worker1.add_argument("--optimizer", required=True, choices=("Adam", "SGD"))
    worker1.add_argument("--learning-rate", required=True)
    worker1.add_argument("--process-id", required=True)
    worker1.add_argument("--device", default="cuda:0")

    worker2 = subparsers.add_parser("worker-stage2", parents=[common])
    worker2.add_argument("--top3-receipt", type=Path)
    worker2.add_argument("--process-id", required=True)
    worker2.add_argument("--device", default="cuda:0")

    subparsers.add_parser("aggregate-stage1", parents=[common])
    subparsers.add_parser("aggregate-final", parents=[common])

    launcher1 = subparsers.add_parser("launch-stage1", parents=[common])
    launcher1.add_argument("--gpu-ids", default="1,2")
    launcher2 = subparsers.add_parser("launch-stage2", parents=[common])
    launcher2.add_argument("--gpu-ids", default="1,2")
    launcher2.add_argument("--top3-receipt", type=Path)
    smoke = subparsers.add_parser("gpu-smoke", parents=[common])
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument(
        "--full-cell",
        "--full-cell-smoke",
        action="store_true",
        dest="full_cell_smoke",
        help="run all 64 clean IRSTD images for each BS/SS protocol",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    roles = {
        "validate": validate_only,
        "worker-stage1": run_stage1_worker,
        "worker-stage2": run_stage2_worker,
        "aggregate-stage1": aggregate_stage1,
        "aggregate-final": aggregate_final,
        "launch-stage1": launch_stage1,
        "launch-stage2": launch_stage2,
        "gpu-smoke": run_gpu_smoke,
    }
    return roles[args.role](args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundFile",
    "CalibrationContract",
    "CalibrationExecutionError",
    "CacheContext",
    "RuntimeSeal",
    "RuntimeSealMonitor",
    "aggregate_final",
    "aggregate_stage1",
    "build_cell_record",
    "build_parser",
    "candidate_slug",
    "capture_runtime_seal",
    "execute_candidate",
    "launch_stage1",
    "launch_stage2",
    "load_contract",
    "run_label_free_episode",
    "run_gpu_smoke",
    "run_stage1_worker",
    "run_stage2_worker",
    "validate_only",
    "verify_shard",
]
