"""Audit Binary TENT reproducibility across independent CUDA processes.

This is a source-domain, train-derived implementation audit.  It compares the
already-published ``source_implementation_smoke_v1`` canonical run with two or
more newly spawned, sequential Python workers.  Each worker executes exactly
one canonical pass for each Binary TENT BN protocol (32 + 32 episodes).

The automatic output is deliberately a *candidate* numerical envelope.  This
program cannot freeze a scientific tolerance and does not inspect fixed-test
pixels.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid
import warnings

import numpy as np
from PIL import Image
import torch
from torch import nn
import yaml

from dataio.corruption_cache import ordered_ids_sha256
from run_adabn_source_pilot import materialize_label_free_inputs
import run_binary_tent_source_smoke as source_smoke
import test_source as source_runner
from tta.binary_tent import BN_PROTOCOL_BATCH_STATS, BN_PROTOCOL_SOURCE_STATS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "binary_tent_cross_process_repro_candidate_v1.yaml"
)
EXPECTED_PROTOCOL_ID = "cr-sitta-binary-tent-cross-process-repro-candidate-v1"
PROTOCOL_DIRECTORY = {
    BN_PROTOCOL_BATCH_STATS: "batch_stats",
    BN_PROTOCOL_SOURCE_STATS: "source_stats",
}
PREDICTION_NAMES = ("source_pre", "tent_pre", "tent_post")
EVALUATOR_NAMES = (
    "official_nsfpn_operating_point",
    "unified_fixed_probability_0_5",
)
METRIC_NAMES = ("iou", "pd", "fa_per_million_pixels")
PROVENANCE_PATHS = tuple(
    dict.fromkeys(
        (
            *source_smoke.PROVENANCE_PATHS,
            "configs/binary_tent_cross_process_repro_candidate_v1.yaml",
            "run_binary_tent_cross_process_repro.py",
            "tests/test_binary_tent_cross_process_repro.py",
        )
    )
)


@dataclass(frozen=True)
class AuditPaths:
    config: Path
    source_smoke_config: Path
    reference_root: Path
    output_root: Path


@dataclass(frozen=True)
class RunDescriptor:
    run_id: str
    root: Path
    origin: str
    metrics: Mapping[str, Mapping[str, Any]]
    records: Mapping[str, Mapping[str, Mapping[str, Any]]]
    diagnostics: Mapping[str, Mapping[str, Mapping[str, Any]]]
    process: Mapping[str, Any]
    test_images_opened: int
    test_masks_opened: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify contracts and the published reference; launch no worker/GPU.",
    )
    worker = parser.add_argument_group("internal fresh-worker interface")
    worker.add_argument("--worker-id", default=None, help=argparse.SUPPRESS)
    worker.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    worker.add_argument(
        "--expected-parent-pid", type=int, default=None, help=argparse.SUPPRESS
    )
    worker.add_argument(
        "--expected-runtime-seal-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _runtime_seal(config_path: Path) -> dict[str, Any]:
    """Pin runtime/protocol bytes before any audit work begins.

    End-of-run repository provenance alone cannot prove which bytes a long
    process imported.  The entry seal is therefore checked again before any
    worker or parent artifact may be published.
    """

    resolved_config = config_path.expanduser().resolve()
    if not resolved_config.is_file():
        raise FileNotFoundError(f"runtime-seal config is missing: {resolved_config}")
    file_sha256: dict[str, str] = {}
    for relative in PROVENANCE_PATHS:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"runtime-seal file is missing: {path}")
        file_sha256[relative] = source_runner.sha256_file(path)
    return {
        "config_path": str(resolved_config),
        "config_sha256": source_runner.sha256_file(resolved_config),
        "file_sha256": file_sha256,
    }


def _runtime_seal_sha256(seal: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(seal),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_runtime_seal(
    expected: Mapping[str, Any], config_path: Path, *, stage: str
) -> None:
    current = _runtime_seal(config_path)
    if current == dict(expected):
        return
    expected_files = _require_mapping(
        expected.get("file_sha256"), "expected runtime-seal files"
    )
    current_files = _require_mapping(
        current.get("file_sha256"), "current runtime-seal files"
    )
    changed = sorted(
        key
        for key in set(expected_files) | set(current_files)
        if expected_files.get(key) != current_files.get(key)
    )
    if expected.get("config_sha256") != current.get("config_sha256"):
        changed.insert(0, str(config_path.resolve()))
    raise RuntimeError(
        "runtime/protocol files changed after process entry at "
        f"{stage}: {', '.join(changed[:10])}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON mapping: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"JSONL record must be a mapping: {path}:{line_number}")
        records.append(dict(value))
    return records


def load_audit_config(path: str | Path = DEFAULT_CONFIG) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"cross-process config does not exist: {config_path}")
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("cross-process config must be a YAML mapping")
    config = dict(value)
    _require_equal(config.get("schema_version"), 1, "schema_version")
    _require_equal(config.get("protocol_id"), EXPECTED_PROTOCOL_ID, "protocol_id")

    scope = _require_mapping(config.get("scope"), "scope")
    for key, expected in (
        ("paper_result", False),
        ("scientific_result_frozen", False),
        ("tuning_allowed", False),
        ("optimizer_selection_allowed", False),
        ("tolerance_automatically_frozen", False),
        ("use_test_images", False),
        ("use_test_labels", False),
        ("train_labels_outer_evaluator_only", True),
    ):
        _require_equal(scope.get(key), expected, f"scope.{key}")

    source = _require_mapping(config.get("source_smoke_contract"), "source contract")
    _require_equal(source.get("selected_count"), 32, "source selected count")
    _require_equal(
        dict(_require_mapping(source.get("condition"), "source condition")),
        {
            "dataset": "IRSTD-1K",
            "corruption": "gaussian_noise",
            "severity": 3,
            "seed": 42,
        },
        "source condition",
    )

    method = _require_mapping(config.get("method"), "method")
    _require_equal(
        tuple(method.get("protocols", ())),
        (BN_PROTOCOL_BATCH_STATS, BN_PROTOCOL_SOURCE_STATS),
        "method.protocols",
    )
    _require_equal(method.get("optimizer_steps_per_image"), 1, "optimizer steps")
    _require_equal(method.get("amp_enabled"), False, "AMP")

    execution = _require_mapping(config.get("execution"), "execution")
    _require_equal(execution.get("device_type"), "cuda", "device type")
    worker_count = execution.get("worker_count")
    if isinstance(worker_count, bool) or not isinstance(worker_count, int):
        raise TypeError("execution.worker_count must be an integer")
    if worker_count < 2:
        raise ValueError("at least two fresh worker processes are required")
    worker_ids = tuple(execution.get("worker_ids", ()))
    if len(worker_ids) != worker_count or len(set(worker_ids)) != worker_count:
        raise ValueError("worker_ids must be unique and match worker_count")
    if any(not isinstance(value, str) or not value for value in worker_ids):
        raise ValueError("worker_ids must be non-empty strings")
    _require_equal(execution.get("launch_mode"), "sequential_fresh_python_processes", "launch mode")
    _require_equal(execution.get("canonical_passes_per_worker"), 1, "canonical passes")
    _require_equal(execution.get("episodes_per_protocol_per_worker"), 32, "protocol episodes")
    _require_equal(execution.get("episodes_per_worker"), 64, "worker episodes")
    _require_equal(execution.get("historical_reference_processes"), 1, "reference count")
    _require_equal(
        execution.get("minimum_independent_processes"),
        worker_count + 1,
        "independent process count",
    )
    _require_equal(
        execution.get("expected_pairwise_comparisons"),
        math.comb(worker_count + 1, 2),
        "pairwise comparison count",
    )
    _require_equal(execution.get("batch_size"), 1, "batch size")
    _require_equal(execution.get("num_workers"), 0, "data workers")

    comparison = _require_mapping(config.get("comparison"), "comparison")
    for key, expected in (
        ("all_unordered_run_pairs_required", True),
        ("source_pre_probability_bit_exact_required", True),
        ("tent_pre_probability_bit_exact_required", True),
        ("source_pre_mask_bit_exact_required", True),
        ("tent_pre_mask_bit_exact_required", True),
        ("full_state_reset_required_every_episode", True),
        ("tent_post_bit_exact_required", False),
    ):
        _require_equal(comparison.get(key), expected, f"comparison.{key}")

    candidate = _require_mapping(config.get("tolerance_candidate"), "candidate")
    _require_equal(candidate.get("automatic_output"), "candidate_only", "candidate output")
    _require_equal(candidate.get("scientific_tolerance_frozen"), False, "candidate frozen")
    _require_equal(candidate.get("formal_acceptance_gate"), False, "candidate gate")
    multiplier = float(candidate.get("safety_multiplier", 0.0))
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("tolerance candidate safety multiplier must be finite and >= 1")

    outputs = _require_mapping(config.get("outputs"), "outputs")
    _require_equal(outputs.get("atomic_sibling_staging"), True, "atomic staging")
    _require_equal(outputs.get("refuse_overwrite"), True, "overwrite policy")
    return config_path, config


def resolve_audit_paths(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    output_override: Path | None = None,
) -> AuditPaths:
    output = (
        _project_path(config["outputs"]["root"])
        if output_override is None
        else output_override.expanduser().resolve()
    )
    paths = AuditPaths(
        config=config_path,
        source_smoke_config=_project_path(config["source_smoke_contract"]["config"]),
        reference_root=_project_path(config["published_reference"]["root"]),
        output_root=output,
    )
    if not paths.source_smoke_config.is_file():
        raise FileNotFoundError(f"source smoke config is missing: {paths.source_smoke_config}")
    if not paths.reference_root.is_dir():
        raise FileNotFoundError(f"published reference is missing: {paths.reference_root}")
    return paths


def _safe_manifest_path(root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ValueError(f"unsafe artifact manifest path: {relative!r}")
    path = root / candidate_relative
    if path.is_symlink():
        raise ValueError(f"artifact manifest target cannot be a symlink: {path}")
    return path


def verify_artifact_manifest(
    root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_complete_sha256: str | None = None,
    expected_protocol_id: str | None = None,
    expected_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    complete_path = root / "COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(f"artifact is not complete: {root}")
    if expected_manifest_sha256 is not None:
        _require_equal(
            source_runner.sha256_file(manifest_path),
            expected_manifest_sha256,
            f"{root.name} manifest SHA256",
        )
    if expected_complete_sha256 is not None:
        _require_equal(
            source_runner.sha256_file(complete_path),
            expected_complete_sha256,
            f"{root.name} COMPLETE SHA256",
        )
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    _require_equal(complete.get("complete"), True, f"{root.name} complete flag")
    _require_equal(
        complete.get("artifact_manifest_sha256"),
        source_runner.sha256_file(manifest_path),
        f"{root.name} manifest lineage",
    )
    if expected_protocol_id is not None:
        _require_equal(complete.get("protocol_id"), expected_protocol_id, "protocol ID")
        _require_equal(manifest.get("protocol_id"), expected_protocol_id, "manifest protocol ID")
    if expected_protocol_sha256 is not None:
        _require_equal(complete.get("protocol_sha256"), expected_protocol_sha256, "protocol hash")
        _require_equal(manifest.get("protocol_sha256"), expected_protocol_sha256, "manifest protocol hash")
    files = _require_mapping(manifest.get("files"), "artifact manifest files")
    present_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and str(path.relative_to(root))
        not in {"artifact_manifest.json", "COMPLETE.json"}
    }
    _require_equal(
        present_files,
        set(files),
        f"{root.name} manifest exact file set",
    )
    verified = 0
    for relative, raw_record in files.items():
        if not isinstance(relative, str):
            raise TypeError("artifact manifest paths must be strings")
        record = _require_mapping(raw_record, f"manifest entry {relative}")
        path = _safe_manifest_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"manifest file is missing: {path}")
        _require_equal(path.stat().st_size, int(record["bytes"]), f"{relative} bytes")
        _require_equal(source_runner.sha256_file(path), record["sha256"], f"{relative} SHA256")
        verified += 1
    return {"manifest": manifest, "complete": complete, "verified_files": verified}


def validate_contracts(
    config: Mapping[str, Any], paths: AuditPaths
) -> dict[str, Any]:
    source_contract = config["source_smoke_contract"]
    _require_equal(
        source_runner.sha256_file(paths.source_smoke_config),
        source_contract["config_sha256"],
        "source smoke config SHA256",
    )
    smoke_config_path, smoke_config = source_smoke.load_smoke_config(
        paths.source_smoke_config
    )
    smoke_paths = source_smoke.resolve_smoke_paths(smoke_config_path, smoke_config)
    (
        source_config,
        pilot_paths,
        selected_ids,
        selected_files,
        fixed_boundary,
        _,
    ) = source_smoke.validate_frozen_source_reference(smoke_config, smoke_paths)
    _require_equal(len(selected_ids), source_contract["selected_count"], "selected count")
    _require_equal(
        ordered_ids_sha256(selected_ids),
        source_contract["selected_ids_sha256"],
        "selected IDs SHA256",
    )
    frozen = smoke_config["frozen_source_input"]
    _require_equal(
        frozen["input_tensor_sequence_sha256"],
        source_contract["input_tensor_sequence_sha256"],
        "input sequence SHA256",
    )
    _require_equal(
        frozen["mask_tensor_sequence_sha256"],
        source_contract["mask_tensor_sequence_sha256"],
        "mask sequence SHA256",
    )

    reference = config["published_reference"]
    for relative, expected in (
        ("provenance.json", reference["provenance_sha256"]),
        ("aggregate_metrics.json", reference["aggregate_metrics_sha256"]),
    ):
        _require_equal(
            source_runner.sha256_file(paths.reference_root / relative),
            expected,
            f"published reference {relative} SHA256",
        )
    reference_verification = verify_artifact_manifest(
        paths.reference_root,
        expected_manifest_sha256=reference["artifact_manifest_sha256"],
        expected_complete_sha256=reference["complete_sha256"],
        expected_protocol_id=reference["protocol_id"],
        expected_protocol_sha256=reference["protocol_sha256"],
    )
    reference_provenance = _load_json(paths.reference_root / "provenance.json")
    reference_repository = _require_mapping(
        reference_provenance.get("repository"),
        "published reference repository provenance",
    )
    reference_file_hashes = _require_mapping(
        reference_repository.get("file_sha256"),
        "published reference runtime file hashes",
    )
    verified_runtime_files: dict[str, str] = {}
    for relative in source_smoke.PROVENANCE_PATHS:
        expected_hash = reference_file_hashes.get(relative)
        if not isinstance(expected_hash, str):
            raise ValueError(
                f"published reference does not bind runtime file: {relative}"
            )
        current_path = PROJECT_ROOT / relative
        if not current_path.is_file():
            raise FileNotFoundError(f"published runtime file is missing: {current_path}")
        current_hash = source_runner.sha256_file(current_path)
        _require_equal(
            current_hash,
            expected_hash,
            f"published runtime lineage {relative}",
        )
        verified_runtime_files[relative] = current_hash
    reference_runtime = _require_mapping(
        reference_provenance.get("runtime"), "published reference runtime"
    )
    reference_extension = _require_mapping(
        reference_runtime.get("loaded_multiscale_deformable_attention"),
        "published reference extension",
    )
    reference_extension_path = Path(str(reference_extension["file"])).resolve()
    if not reference_extension_path.is_file():
        raise FileNotFoundError(
            f"published reference extension binary is missing: {reference_extension_path}"
        )
    _require_equal(
        reference_extension_path.stat().st_size,
        int(reference_extension["bytes"]),
        "published reference extension bytes",
    )
    _require_equal(
        source_runner.sha256_file(reference_extension_path),
        reference_extension["sha256"],
        "published reference extension SHA256",
    )
    boundary = _require_mapping(
        reference_provenance.get("fixed_test_boundary"),
        "published reference fixed-test boundary",
    )
    for key in ("test_images_opened", "test_masks_opened"):
        _require_equal(boundary.get(key), 0, f"published reference {key}")
    _require_equal(
        reference_provenance.get("selected_ids_sha256"),
        source_contract["selected_ids_sha256"],
        "published selected IDs",
    )
    return {
        "source_config": source_config,
        "pilot_paths": pilot_paths,
        "selected_ids": selected_ids,
        "selected_files": selected_files,
        "fixed_boundary": fixed_boundary,
        "smoke_config": smoke_config,
        "smoke_paths": smoke_paths,
        "reference_verification": reference_verification,
        "reference_provenance": reference_provenance,
        "verified_runtime_files": verified_runtime_files,
        "reference_extension": dict(reference_extension),
    }


def _artifact_files(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        if relative in {"artifact_manifest.json", "COMPLETE.json"}:
            continue
        if path.is_symlink():
            raise ValueError(f"artifact cannot contain a symlink: {path}")
        records[relative] = {
            "sha256": source_runner.sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return records


def _write_yaml_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finalize_artifact(
    *,
    staging: Path,
    final_root: Path,
    protocol_id: str,
    protocol_sha256: str,
    scope: str,
    checks: Mapping[str, Any],
    required_files: Sequence[str],
    extra_complete: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failed = sorted(name for name, value in checks.items() if value is not True)
    if failed:
        raise RuntimeError(f"refusing COMPLETE because hard gates failed: {failed}")
    present = {
        str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()
    }
    missing = sorted(set(required_files) - present)
    if missing:
        raise RuntimeError(f"required artifact files are missing: {missing}")
    files = _artifact_files(staging)
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "scope": scope,
        "paper_result": False,
        "scientific_result_frozen": False,
        "tolerance_status": "candidate_only_not_frozen",
        "files": files,
    }
    manifest_path = staging / "artifact_manifest.json"
    source_runner.write_json_atomic(manifest_path, manifest)
    completion = {
        "complete": True,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "scope": scope,
        "paper_result": False,
        "scientific_result_frozen": False,
        "tolerance_status": "candidate_only_not_frozen",
        "all_required_hard_gates_passed": True,
        "artifact_manifest_sha256": source_runner.sha256_file(manifest_path),
        **dict(extra_complete or {}),
    }
    source_runner.write_json_atomic(staging / "COMPLETE.json", completion)
    if final_root.exists():
        raise FileExistsError(f"output already exists: {final_root}")
    os.replace(staging, final_root)
    return completion


def _worker_protocol(
    *,
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    samples: Sequence[Any],
    targets: Mapping[str, np.ndarray],
    device: torch.device,
    bn_protocol: str,
    destination: Path,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    method_config = config["method"]
    optimizer = method_config["optimizer"]
    execution = config["execution"]
    evaluation = config["evaluation"]
    adapter, method, state, runner, checkpoint = source_smoke._build_protocol_runner(
        source_config=source_config,
        device=device,
        bn_protocol=bn_protocol,
        optimizer_name=str(optimizer["name"]),
        learning_rate=float(optimizer["learning_rate"]),
        entropy_eps=float(method_config["entropy_eps"]),
    )
    bn_count = sum(isinstance(module, nn.BatchNorm2d) for module in adapter.model.modules())
    adaptable, adaptable_names = adapter.collect_adaptable_params()
    adaptable_scalars = sum(value.numel() for value in adaptable)
    _require_equal(bn_count, execution["expected_batchnorm2d_modules"], "BN module count")
    _require_equal(len(adaptable), execution["expected_trainable_tensors"], "BN affine count")
    _require_equal(adaptable_scalars, execution["expected_trainable_scalars"], "BN affine scalars")
    selected_ids = tuple(sample.image_id for sample in samples)
    episodes = source_smoke.execute_episode_pass(
        samples=samples,
        order=selected_ids,
        runner=runner,
        method=method,
        device=device,
        threshold=float(evaluation["fixed_probability_threshold"]),
        expected_trainable_tensors=len(adaptable),
        expected_trainable_scalars=adaptable_scalars,
    )
    state.assert_source_state()
    records, diagnostics, output_counts = source_smoke._save_canonical_predictions(
        protocol_root=destination, canonical=episodes, samples=samples
    )
    for record in records:
        record["optimizer_step_norm"] = record.pop("canonical_optimizer_step_norm")
        record.pop("batch_tent_pre_frozen_adabn_probability_bit_exact", None)
        record.pop("source_pre_frozen_adabn_source_probability_bit_exact", None)
        record.pop("source_stats_tent_pre_source_pre_bit_exact", None)
        record.pop("run_comparisons", None)
    source_runner.write_jsonl_atomic(destination / "per_image.jsonl", records)
    source_runner.write_jsonl_atomic(
        destination / "adaptation_diagnostics.jsonl", diagnostics
    )
    evaluation_protocol = source_smoke.IRSTDEvaluationProtocol(
        fixed_probability_threshold=float(evaluation["fixed_probability_threshold"]),
        connectivity=int(evaluation["connectivity"]),
        max_centroid_distance=float(evaluation["max_centroid_distance"]),
        min_component_area=int(evaluation["min_component_area"]),
    )
    metrics = source_smoke.evaluate_pass_metrics(
        episodes,
        selected_ids=selected_ids,
        targets=targets,
        protocol=evaluation_protocol,
    )
    reset_exact = all(
        value.state["reset_fingerprint"] == value.state["source_fingerprint"]
        and value.diagnostics["episode_invariants"]["reset_exact_source"] is True
        for value in episodes.values()
    )
    checks = {
        "exactly_32_canonical_episodes": len(episodes) == 32,
        "exactly_one_step_per_episode": all(
            value.diagnostics["optimizer_steps"] == 1 for value in episodes.values()
        ),
        "full_reset_exact_every_episode": reset_exact,
        "all_prediction_files_saved": len(output_counts) == 6
        and all(value == 32 for value in output_counts.values()),
        "method_received_no_label": True,
    }
    failed = sorted(name for name, value in checks.items() if value is not True)
    if failed:
        raise RuntimeError(f"worker protocol gates failed ({bn_protocol}): {failed}")
    payload = {
        "schema_version": 1,
        "scope": "source_domain_cross_process_worker",
        "paper_result": False,
        "scientific_result_frozen": False,
        "tolerance_status": "candidate_only_not_frozen",
        "bn_protocol": bn_protocol,
        "canonical_passes": 1,
        "episodes": len(episodes),
        "canonical_metrics_outer_train_diagnostic_only": metrics,
        "saved_output_counts": output_counts,
        "checks": checks,
    }
    source_runner.write_json_atomic(destination / "metrics.json", payload)
    del runner, state, method, adapter
    return {
        "metrics": payload,
        "checkpoint": checkpoint,
        "adaptable_parameter_names": adaptable_names,
    }


def _runtime_summary(device: torch.device) -> dict[str, Any]:
    summary = source_smoke._runtime_device_summary(device)
    summary.update(
        {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "process_uuid": str(uuid.uuid4()),
            "process_started_recorded_at_ns": time.time_ns(),
            "python_executable": sys.executable,
            "argv": list(sys.argv),
        }
    )
    return summary


def run_worker(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    paths: AuditPaths,
    contracts: Mapping[str, Any],
    worker_id: str,
    worker_output: Path,
    expected_parent_pid: int,
    device_name: str,
    runtime_seal: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_runtime_seal(runtime_seal, config_path, stage="worker_start")
    if worker_id not in tuple(config["execution"]["worker_ids"]):
        raise ValueError(f"unregistered worker ID: {worker_id}")
    if os.getppid() != expected_parent_pid:
        raise RuntimeError(
            f"worker parent PID mismatch: expected {expected_parent_pid}, got {os.getppid()}"
        )
    if worker_output.exists():
        raise FileExistsError(f"worker output already exists: {worker_output}")
    worker_output.parent.mkdir(parents=True, exist_ok=True)
    staging = worker_output.with_name(f".{worker_output.name}.build-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"worker staging already exists: {staging}")

    source_config = contracts["source_config"]
    selected_ids = contracts["selected_ids"]
    samples, materialization = materialize_label_free_inputs(
        source_config,
        contracts["pilot_paths"],
        selected_ids,
        contracts["selected_files"],
    )
    source_contract = config["source_smoke_contract"]
    _require_equal(
        materialization["input_tensor_sequence_sha256"],
        source_contract["input_tensor_sequence_sha256"],
        "worker input sequence SHA256",
    )
    targets, target_evaluator = source_smoke.load_outer_train_targets(
        selected_ids=selected_ids,
        selected_files=contracts["selected_files"],
        image_size=256,
        expected_sequence_sha256=source_contract["mask_tensor_sequence_sha256"],
    )
    staging.mkdir(parents=False)
    _write_yaml_atomic(
        staging / "run_config.yaml",
        {
            **deepcopy(dict(config)),
            "runtime_worker": {
                "worker_id": worker_id,
                "worker_output": str(worker_output),
                "device": device_name,
                "parent_pid": expected_parent_pid,
            },
        },
    )

    protocol_results: dict[str, Any] = {}
    execution = config["execution"]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with source_smoke.binary_tent_cuda_policy(
            seed=int(source_contract["condition"]["seed"]),
            workspace_config=str(execution["cublas_workspace_config"]),
        ) as process_policy:
            device = source_runner.resolve_device(device_name)
            if device.type != "cuda":
                raise RuntimeError("cross-process Binary TENT workers are GPU-only")
            if process_policy["cuda_initialized_before_workspace_config"]:
                raise RuntimeError(
                    "CUDA was initialized before the worker policy; a fresh process is required"
                )
            for bn_protocol in config["method"]["protocols"]:
                directory = PROTOCOL_DIRECTORY[bn_protocol]
                protocol_results[directory] = _worker_protocol(
                    config=config,
                    source_config=source_config,
                    samples=samples,
                    targets=targets,
                    device=device,
                    bn_protocol=bn_protocol,
                    destination=staging / directory,
                )
            runtime = _runtime_summary(device)
    process_policy["after"] = asdict(source_smoke._capture_runtime_policy())
    policy_restored = source_smoke.runtime_policy_is_restored(process_policy["before"])
    if not policy_restored:
        raise RuntimeError("worker did not restore process-global CUDA/Torch policy")

    process_record = {
        **runtime,
        "worker_id": worker_id,
        "expected_parent_pid": expected_parent_pid,
        "fresh_process_cuda_uninitialized_before_policy": not process_policy[
            "cuda_initialized_before_workspace_config"
        ],
    }
    _assert_runtime_seal(runtime_seal, config_path, stage="worker_prepublication")
    runtime_seal_hash = _runtime_seal_sha256(runtime_seal)
    provenance = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "scope": "source_domain_cross_process_worker",
        "paper_result": False,
        "scientific_result_frozen": False,
        "tolerance_status": "candidate_only_not_frozen",
        "worker_id": worker_id,
        "process": process_record,
        "config": str(config_path),
        "config_sha256": runtime_seal["config_sha256"],
        "process_entry_runtime_seal": dict(runtime_seal),
        "process_entry_runtime_seal_sha256": runtime_seal_hash,
        "selected_ids": list(selected_ids),
        "selected_ids_sha256": ordered_ids_sha256(selected_ids),
        "materialized_inputs": materialization,
        "outer_train_target_evaluator": target_evaluator,
        "fixed_test_boundary": {
            "test_dataset_constructed": False,
            "test_images_opened": 0,
            "test_masks_opened": 0,
        },
        "process_policy": process_policy,
        "warnings": source_smoke._warning_summary(captured),
        "checkpoint_by_protocol": {
            key: value["checkpoint"] for key, value in protocol_results.items()
        },
        "repository": source_runner.repository_provenance(PROVENANCE_PATHS),
    }
    source_runner.write_json_atomic(staging / "provenance.json", provenance)
    aggregate = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "scope": "source_domain_cross_process_worker",
        "paper_result": False,
        "scientific_result_frozen": False,
        "tolerance_status": "candidate_only_not_frozen",
        "worker_id": worker_id,
        "canonical_passes_per_protocol": 1,
        "selected_train_images": len(selected_ids),
        "episodes_per_protocol": 32,
        "total_episodes": sum(
            value["metrics"]["episodes"] for value in protocol_results.values()
        ),
        "both_protocols": sorted(protocol_results),
        "process": process_record,
        "test_images_opened": 0,
        "test_masks_opened": 0,
    }
    source_runner.write_json_atomic(staging / "aggregate_metrics.json", aggregate)
    checks = {
        "gpu_only": runtime["device"].startswith("cuda"),
        "fresh_process": process_record["fresh_process_cuda_uninitialized_before_policy"],
        "exactly_two_protocols": set(protocol_results) == {"batch_stats", "source_stats"},
        "exactly_64_episodes": aggregate["total_episodes"] == 64,
        "every_episode_reset_exact": all(
            value["metrics"]["checks"]["full_reset_exact_every_episode"] is True
            for value in protocol_results.values()
        ),
        "zero_test_pixel_opens": target_evaluator["test_masks_opened"] == 0,
        "process_policy_restored": policy_restored,
        "candidate_not_frozen": config["tolerance_candidate"][
            "scientific_tolerance_frozen"
        ]
        is False,
    }
    completion = finalize_artifact(
        staging=staging,
        final_root=worker_output,
        protocol_id=config["protocol_id"],
        protocol_sha256=str(runtime_seal["config_sha256"]),
        scope="source_domain_cross_process_worker",
        checks=checks,
        required_files=(
            "run_config.yaml",
            "provenance.json",
            "aggregate_metrics.json",
            "batch_stats/metrics.json",
            "batch_stats/per_image.jsonl",
            "batch_stats/adaptation_diagnostics.jsonl",
            "source_stats/metrics.json",
            "source_stats/per_image.jsonl",
            "source_stats/adaptation_diagnostics.jsonl",
        ),
        extra_complete={
            "worker_id": worker_id,
            "pid": os.getpid(),
            "process_uuid": process_record["process_uuid"],
            "total_episodes": 64,
            "runtime_seal_sha256": runtime_seal_hash,
        },
    )
    return {"worker_id": worker_id, "output": str(worker_output), "completion": completion}


def _index_records(records: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"invalid image ID in {label}")
        if image_id in result:
            raise ValueError(f"duplicate image ID in {label}: {image_id}")
        result[image_id] = record
    return result


def load_run_descriptor(
    *,
    run_id: str,
    root: Path,
    origin: str,
    selected_ids: Sequence[str],
) -> RunDescriptor:
    provenance = _load_json(root / "provenance.json")
    records: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    diagnostics: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    metrics: dict[str, Mapping[str, Any]] = {}
    for protocol_directory in ("batch_stats", "source_stats"):
        indexed_records = _index_records(
            _read_jsonl(root / protocol_directory / "per_image.jsonl"),
            f"{run_id}/{protocol_directory}/per_image",
        )
        indexed_diagnostics = _index_records(
            _read_jsonl(root / protocol_directory / "adaptation_diagnostics.jsonl"),
            f"{run_id}/{protocol_directory}/diagnostics",
        )
        _require_equal(tuple(indexed_records), tuple(selected_ids), f"{run_id} record order")
        _require_equal(tuple(indexed_diagnostics), tuple(selected_ids), f"{run_id} diagnostic order")
        records[protocol_directory] = indexed_records
        diagnostics[protocol_directory] = indexed_diagnostics
        protocol_metrics = _load_json(root / protocol_directory / "metrics.json")
        metrics[protocol_directory] = _require_mapping(
            protocol_metrics.get("canonical_metrics_outer_train_diagnostic_only"),
            f"{run_id}/{protocol_directory} canonical metrics",
        )
    boundary = _require_mapping(provenance.get("fixed_test_boundary"), "fixed-test boundary")
    process = (
        _require_mapping(provenance.get("process"), "worker process")
        if origin == "fresh_worker"
        else {
            "historical_independent_artifact": True,
            "artifact_manifest_sha256": source_runner.sha256_file(
                root / "artifact_manifest.json"
            ),
        }
    )
    return RunDescriptor(
        run_id=run_id,
        root=root,
        origin=origin,
        metrics=metrics,
        records=records,
        diagnostics=diagnostics,
        process=process,
        test_images_opened=int(boundary.get("test_images_opened", -1)),
        test_masks_opened=int(boundary.get("test_masks_opened", -1)),
    )


def _record_step_norm(record: Mapping[str, Any]) -> float:
    key = "optimizer_step_norm"
    if key not in record:
        key = "canonical_optimizer_step_norm"
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError("optimizer step norm must be finite")
    return value


def _load_prediction(
    run: RunDescriptor,
    protocol: str,
    image_id: str,
    prediction_name: str,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = _require_mapping(
        run.records[protocol][image_id]["predictions"][prediction_name],
        "prediction record",
    )
    probability_path = run.root / protocol / str(prediction["probability_map"])
    mask_path = run.root / protocol / str(prediction["prediction_mask"])
    probability = np.load(probability_path, allow_pickle=False)
    if probability.dtype != np.float32 or probability.shape != (256, 256):
        raise RuntimeError(f"invalid probability array contract: {probability_path}")
    if not np.isfinite(probability).all():
        raise RuntimeError(f"non-finite probability array: {probability_path}")
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle.convert("L"), dtype=np.uint8)
    if mask.shape != (256, 256):
        raise RuntimeError(f"invalid mask shape: {mask_path}")
    derived = np.where(probability > threshold, 255, 0).astype(np.uint8)
    if not np.array_equal(mask, derived):
        raise RuntimeError(f"saved mask violates strict probability threshold: {mask_path}")
    return np.ascontiguousarray(probability), np.ascontiguousarray(mask)


def _reset_exact(run: RunDescriptor, protocol: str, image_id: str) -> bool:
    record = run.diagnostics[protocol][image_id]
    source = record.get("source_fingerprint")
    reset = record.get("reset_fingerprint")
    invariants = _require_mapping(
        _require_mapping(record.get("method_diagnostics"), "method diagnostics").get(
            "episode_invariants"
        ),
        "episode invariants",
    )
    return source == reset and invariants.get("reset_exact_source") is True


def compare_run_pair(
    run_a: RunDescriptor,
    run_b: RunDescriptor,
    *,
    selected_ids: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    pair_id = f"{run_a.run_id}__vs__{run_b.run_id}"
    protocols: dict[str, Any] = {}
    for protocol in ("batch_stats", "source_stats"):
        aggregates: dict[str, Any] = {}
        per_image: dict[str, Any] = {}
        for prediction_name in PREDICTION_NAMES:
            exact_probability_images = 0
            exact_mask_images = 0
            probability_abs_sum = 0.0
            probability_values = 0
            probability_max = 0.0
            mask_disagreement_pixels = 0
            mask_values = 0
            for image_id in selected_ids:
                probability_a, mask_a = _load_prediction(
                    run_a, protocol, image_id, prediction_name, threshold=threshold
                )
                probability_b, mask_b = _load_prediction(
                    run_b, protocol, image_id, prediction_name, threshold=threshold
                )
                difference = np.abs(probability_a - probability_b)
                disagreements = int(np.count_nonzero(mask_a != mask_b))
                image_record = per_image.setdefault(image_id, {})
                image_record[prediction_name] = {
                    "probability_bit_exact": bool(np.array_equal(probability_a, probability_b)),
                    "probability_mean_abs_difference": float(difference.mean(dtype=np.float64)),
                    "probability_max_abs_difference": float(difference.max()),
                    "mask_bit_exact": disagreements == 0,
                    "mask_disagreement_pixels": disagreements,
                    "mask_disagreement_rate": float(disagreements / mask_a.size),
                }
                exact_probability_images += int(image_record[prediction_name]["probability_bit_exact"])
                exact_mask_images += int(disagreements == 0)
                probability_abs_sum += float(difference.sum(dtype=np.float64))
                probability_values += int(difference.size)
                probability_max = max(probability_max, float(difference.max()))
                mask_disagreement_pixels += disagreements
                mask_values += int(mask_a.size)
            aggregates[prediction_name] = {
                "probability_bit_exact_images": exact_probability_images,
                "probability_global_mean_abs_difference": probability_abs_sum / probability_values,
                "probability_global_max_abs_difference": probability_max,
                "mask_bit_exact_images": exact_mask_images,
                "mask_disagreement_pixels": mask_disagreement_pixels,
                "mask_disagreement_rate": mask_disagreement_pixels / mask_values,
            }

        step_differences: list[float] = []
        for image_id in selected_ids:
            difference = abs(
                _record_step_norm(run_a.records[protocol][image_id])
                - _record_step_norm(run_b.records[protocol][image_id])
            )
            step_differences.append(difference)
            per_image[image_id]["optimizer_step_norm_absolute_difference"] = difference
        aggregates["optimizer_step_norm"] = {
            "global_max_absolute_difference": max(step_differences),
            "global_mean_absolute_difference": sum(step_differences) / len(step_differences),
        }

        metric_differences: dict[str, Any] = {}
        for prediction_name in PREDICTION_NAMES:
            metric_differences[prediction_name] = {}
            for evaluator in EVALUATOR_NAMES:
                metric_differences[prediction_name][evaluator] = {
                    metric: abs(
                        float(run_a.metrics[protocol][prediction_name][evaluator][metric])
                        - float(run_b.metrics[protocol][prediction_name][evaluator][metric])
                    )
                    for metric in METRIC_NAMES
                }
        reset_a = sum(_reset_exact(run_a, protocol, image_id) for image_id in selected_ids)
        reset_b = sum(_reset_exact(run_b, protocol, image_id) for image_id in selected_ids)
        protocols[protocol] = {
            "aggregate": aggregates,
            "metric_absolute_differences": metric_differences,
            "state_reset": {
                run_a.run_id: {"exact_episodes": reset_a, "total_episodes": len(selected_ids)},
                run_b.run_id: {"exact_episodes": reset_b, "total_episodes": len(selected_ids)},
            },
            "per_image": per_image,
        }
    return {
        "pair_id": pair_id,
        "run_a": run_a.run_id,
        "run_b": run_b.run_id,
        "unordered_pair": True,
        "protocols": protocols,
    }


def all_unordered_pairs(runs: Sequence[RunDescriptor]) -> tuple[tuple[RunDescriptor, RunDescriptor], ...]:
    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run IDs must be unique")
    return tuple(itertools.combinations(runs, 2))


def derive_candidate(
    comparisons: Sequence[Mapping[str, Any]], *, safety_multiplier: float
) -> dict[str, Any]:
    if not comparisons:
        raise ValueError("at least one pairwise comparison is required")
    per_protocol: dict[str, Any] = {}
    global_observed: dict[str, float | int] = {}
    for protocol in ("batch_stats", "source_stats"):
        observed: dict[str, float | int] = {
            "probability_mean_abs": 0.0,
            "probability_max_abs": 0.0,
            "mask_disagreement_pixels": 0,
            "mask_disagreement_rate": 0.0,
            "optimizer_step_norm_abs": 0.0,
        }
        for evaluator in EVALUATOR_NAMES:
            for metric in METRIC_NAMES:
                observed[f"{evaluator}.{metric}_abs"] = 0.0
        for comparison in comparisons:
            evidence = comparison["protocols"][protocol]
            post = evidence["aggregate"]["tent_post"]
            observed["probability_mean_abs"] = max(
                float(observed["probability_mean_abs"]),
                float(post["probability_global_mean_abs_difference"]),
            )
            observed["probability_max_abs"] = max(
                float(observed["probability_max_abs"]),
                float(post["probability_global_max_abs_difference"]),
            )
            observed["mask_disagreement_pixels"] = max(
                int(observed["mask_disagreement_pixels"]),
                int(post["mask_disagreement_pixels"]),
            )
            observed["mask_disagreement_rate"] = max(
                float(observed["mask_disagreement_rate"]),
                float(post["mask_disagreement_rate"]),
            )
            observed["optimizer_step_norm_abs"] = max(
                float(observed["optimizer_step_norm_abs"]),
                float(evidence["aggregate"]["optimizer_step_norm"]["global_max_absolute_difference"]),
            )
            for evaluator in EVALUATOR_NAMES:
                for metric in METRIC_NAMES:
                    key = f"{evaluator}.{metric}_abs"
                    observed[key] = max(
                        float(observed[key]),
                        float(evidence["metric_absolute_differences"]["tent_post"][evaluator][metric]),
                    )
        candidate: dict[str, float | int] = {}
        for key, value in observed.items():
            if key == "mask_disagreement_pixels":
                candidate[key] = int(math.ceil(int(value) * safety_multiplier))
            else:
                candidate[key] = float(value) * safety_multiplier
            prior = global_observed.get(key, 0)
            global_observed[key] = max(prior, value)
        per_protocol[protocol] = {
            "observed_max_over_all_unordered_pairs": observed,
            "heuristic_candidate_envelope": candidate,
        }
    global_candidate = {
        key: (
            int(math.ceil(int(value) * safety_multiplier))
            if key == "mask_disagreement_pixels"
            else float(value) * safety_multiplier
        )
        for key, value in global_observed.items()
    }
    return {
        "status": "candidate_generated_not_frozen",
        "automatic_output": "candidate_only",
        "scientific_tolerance_frozen": False,
        "formal_acceptance_gate": False,
        "used_to_accept_or_reject_tent_post_in_this_audit": False,
        "safety_multiplier": safety_multiplier,
        "derivation": "multiplier_times_max_observed_over_all_unordered_run_pairs",
        "pairwise_comparisons_used": len(comparisons),
        "per_protocol": per_protocol,
        "global_observed_max": global_observed,
        "global_heuristic_candidate_envelope": global_candidate,
        "freeze_decision": {
            "decision": "do_not_freeze_automatically",
            "reason": "train-derived implementation evidence is candidate-only",
            "required_next_action": "explicit_human_scientific_review_and_new_versioned_protocol",
        },
    }


def comparison_hard_gate(comparison: Mapping[str, Any], selected_count: int) -> bool:
    for protocol in ("batch_stats", "source_stats"):
        evidence = comparison["protocols"][protocol]
        for prediction in ("source_pre", "tent_pre"):
            aggregate = evidence["aggregate"][prediction]
            if aggregate["probability_bit_exact_images"] != selected_count:
                return False
            if aggregate["mask_bit_exact_images"] != selected_count:
                return False
        for reset in evidence["state_reset"].values():
            if reset["exact_episodes"] != selected_count:
                return False
    return True


def build_worker_command(
    *,
    config_path: Path,
    device: str,
    worker_id: str,
    worker_output: Path,
    parent_pid: int,
    expected_runtime_seal_sha256: str,
    python_executable: str = sys.executable,
) -> list[str]:
    return [
        python_executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--device",
        device,
        "--worker-id",
        worker_id,
        "--worker-output",
        str(worker_output),
        "--expected-parent-pid",
        str(parent_pid),
        "--expected-runtime-seal-sha256",
        expected_runtime_seal_sha256,
    ]


def run_parent(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    paths: AuditPaths,
    contracts: Mapping[str, Any],
    device: str,
    runtime_seal: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_runtime_seal(runtime_seal, config_path, stage="parent_start")
    runtime_seal_hash = _runtime_seal_sha256(runtime_seal)
    if not device.startswith("cuda"):
        raise RuntimeError("cross-process Binary TENT audit is GPU-only")
    if paths.output_root.exists():
        raise FileExistsError(f"audit output already exists: {paths.output_root}")
    paths.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = paths.output_root.with_name(f".{paths.output_root.name}.build-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"audit staging already exists: {staging}")
    staging.mkdir(parents=False)
    _write_yaml_atomic(
        staging / "run_config.yaml",
        {
            **deepcopy(dict(config)),
            "runtime_parent": {
                "pid": os.getpid(),
                "python_executable": sys.executable,
                "device": device,
                "output_root": str(paths.output_root),
            },
        },
    )

    worker_roots: list[Path] = []
    commands: list[list[str]] = []
    worker_completion: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(config["source_smoke_contract"]["condition"]["seed"])
    environment["CUBLAS_WORKSPACE_CONFIG"] = str(config["execution"]["cublas_workspace_config"])
    for worker_id in config["execution"]["worker_ids"]:
        worker_root = staging / "workers" / worker_id
        command = build_worker_command(
            config_path=config_path,
            device=device,
            worker_id=worker_id,
            worker_output=worker_root,
            parent_pid=os.getpid(),
            expected_runtime_seal_sha256=runtime_seal_hash,
        )
        commands.append(command)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        _assert_runtime_seal(
            runtime_seal,
            config_path,
            stage=f"parent_after_{worker_id}",
        )
        verification = verify_artifact_manifest(
            worker_root,
            expected_protocol_id=config["protocol_id"],
            expected_protocol_sha256=str(runtime_seal["config_sha256"]),
        )
        _require_equal(
            verification["complete"].get("runtime_seal_sha256"),
            runtime_seal_hash,
            f"{worker_id} runtime seal",
        )
        worker_completion.append(verification["complete"])
        worker_roots.append(worker_root)

    selected_ids = contracts["selected_ids"]
    runs: list[RunDescriptor] = [
        load_run_descriptor(
            run_id=config["published_reference"]["run_id"],
            root=paths.reference_root,
            origin="published_historical_reference",
            selected_ids=selected_ids,
        )
    ]
    runs.extend(
        load_run_descriptor(
            run_id=worker_id,
            root=worker_root,
            origin="fresh_worker",
            selected_ids=selected_ids,
        )
        for worker_id, worker_root in zip(config["execution"]["worker_ids"], worker_roots)
    )
    run_pairs = all_unordered_pairs(runs)
    comparisons = [
        compare_run_pair(
            run_a,
            run_b,
            selected_ids=selected_ids,
            threshold=float(config["evaluation"]["fixed_probability_threshold"]),
        )
        for run_a, run_b in run_pairs
    ]
    candidate = derive_candidate(
        comparisons,
        safety_multiplier=float(config["tolerance_candidate"]["safety_multiplier"]),
    )
    source_runner.write_json_atomic(
        staging / "pairwise_comparisons.json",
        {
            "schema_version": 1,
            "run_ids": [run.run_id for run in runs],
            "all_unordered_pairs": [comparison["pair_id"] for comparison in comparisons],
            "comparison_count": len(comparisons),
            "comparisons": comparisons,
        },
    )
    source_runner.write_json_atomic(staging / "tolerance_candidate.json", candidate)

    worker_pids = [int(completion["pid"]) for completion in worker_completion]
    worker_uuids = [str(completion["process_uuid"]) for completion in worker_completion]
    exact_pairwise = all(
        comparison_hard_gate(comparison, len(selected_ids))
        for comparison in comparisons
    )
    all_zero_test = all(
        run.test_images_opened == 0 and run.test_masks_opened == 0 for run in runs
    )
    expected_pair_ids = {
        f"{run_a.run_id}__vs__{run_b.run_id}" for run_a, run_b in run_pairs
    }
    actual_pair_ids = {comparison["pair_id"] for comparison in comparisons}
    reference_extension_hash = contracts["reference_extension"]["sha256"]
    worker_extension_hashes = [
        run.process["loaded_multiscale_deformable_attention"]["sha256"]
        for run in runs
        if run.origin == "fresh_worker"
    ]
    hard_checks = {
        "immutable_published_reference_verified": contracts["reference_verification"]["complete"]["complete"] is True,
        "runtime_code_bytes_match_published_reference": len(
            contracts["verified_runtime_files"]
        )
        == len(source_smoke.PROVENANCE_PATHS)
        and all(
            value == reference_extension_hash for value in worker_extension_hashes
        ),
        "runtime_files_unchanged_during_execution": True,
        "two_or_more_fresh_unique_worker_processes_completed": len(worker_pids) >= 2
        and len(set(worker_pids)) == len(worker_pids)
        and len(set(worker_uuids)) == len(worker_uuids)
        and os.getpid() not in worker_pids,
        "minimum_three_independent_run_artifacts_compared": len(runs)
        == int(config["execution"]["minimum_independent_processes"]),
        "all_unordered_pairs_compared": len(comparisons)
        == int(config["execution"]["expected_pairwise_comparisons"])
        and actual_pair_ids == expected_pair_ids,
        "exactly_64_episodes_per_worker": all(
            int(completion["total_episodes"]) == 64 for completion in worker_completion
        ),
        "source_and_tent_pre_bit_exact_for_every_pair": exact_pairwise,
        "every_run_episode_reset_exact_source_state": all(
            _reset_exact(run, protocol, image_id)
            for run in runs
            for protocol in ("batch_stats", "source_stats")
            for image_id in selected_ids
        ),
        "zero_test_pixel_opens": all_zero_test,
        "tolerance_output_is_candidate_only_not_frozen": candidate[
            "scientific_tolerance_frozen"
        ]
        is False
        and candidate["formal_acceptance_gate"] is False
        and candidate["freeze_decision"]["decision"] == "do_not_freeze_automatically",
    }
    failed = sorted(name for name, value in hard_checks.items() if value is not True)
    if failed:
        raise RuntimeError(f"cross-process audit hard gates failed: {failed}")

    _assert_runtime_seal(runtime_seal, config_path, stage="parent_prepublication")
    provenance = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "scope": "source_domain_cross_process_reproducibility_audit",
        "paper_result": False,
        "scientific_result_frozen": False,
        "tolerance_status": "candidate_only_not_frozen",
        "parent_process": {
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "worker_launch_mode": "sequential_subprocess_run_check_true",
            "commands": commands,
        },
        "published_reference": {
            "root": str(paths.reference_root),
            "role": "immutable_historical_independent_process",
            "complete_sha256": source_runner.sha256_file(paths.reference_root / "COMPLETE.json"),
            "artifact_manifest_sha256": source_runner.sha256_file(paths.reference_root / "artifact_manifest.json"),
        },
        "worker_processes": [dict(run.process) for run in runs if run.origin == "fresh_worker"],
        "run_ids": [run.run_id for run in runs],
        "selected_ids": list(selected_ids),
        "selected_ids_sha256": ordered_ids_sha256(selected_ids),
        "fixed_test_boundary": {
            "test_dataset_constructed": False,
            "test_images_opened": 0,
            "test_masks_opened": 0,
        },
        "process_entry_runtime_seal": dict(runtime_seal),
        "process_entry_runtime_seal_sha256": runtime_seal_hash,
        "repository": source_runner.repository_provenance(PROVENANCE_PATHS),
    }
    source_runner.write_json_atomic(staging / "provenance.json", provenance)
    aggregate = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "scope": "source_domain_cross_process_reproducibility_audit",
        "paper_result": False,
        "scientific_result_frozen": False,
        "tolerance_status": "candidate_only_not_frozen",
        "selected_train_images": len(selected_ids),
        "independent_run_artifacts": len(runs),
        "fresh_worker_processes": len(worker_roots),
        "episodes_per_worker": 64,
        "new_worker_episodes_total": 64 * len(worker_roots),
        "unordered_pairwise_comparisons": len(comparisons),
        "pair_ids": sorted(actual_pair_ids),
        "source_and_tent_pre_bit_exact_all_pairs": exact_pairwise,
        "all_state_resets_exact": hard_checks["every_run_episode_reset_exact_source_state"],
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "candidate_decision": candidate["freeze_decision"],
        "hard_checks": hard_checks,
    }
    source_runner.write_json_atomic(staging / "aggregate_metrics.json", aggregate)
    completion = finalize_artifact(
        staging=staging,
        final_root=paths.output_root,
        protocol_id=config["protocol_id"],
        protocol_sha256=str(runtime_seal["config_sha256"]),
        scope="source_domain_cross_process_reproducibility_audit",
        checks=hard_checks,
        required_files=(
            "run_config.yaml",
            "provenance.json",
            "aggregate_metrics.json",
            "pairwise_comparisons.json",
            "tolerance_candidate.json",
            *(f"workers/{worker_id}/COMPLETE.json" for worker_id in config["execution"]["worker_ids"]),
            *(f"workers/{worker_id}/artifact_manifest.json" for worker_id in config["execution"]["worker_ids"]),
        ),
        extra_complete={
            "independent_run_artifacts": len(runs),
            "unordered_pairwise_comparisons": len(comparisons),
            "tolerance_frozen": False,
            "runtime_seal_sha256": runtime_seal_hash,
        },
    )
    return {
        "published_output_dir": str(paths.output_root),
        "aggregate": aggregate,
        "candidate": candidate,
        "completion": completion,
    }


def validate_only_payload(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    paths: AuditPaths,
    contracts: Mapping[str, Any],
    device: str,
    runtime_seal: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_runtime_seal(runtime_seal, config_path, stage="validate_only")
    runtime_seal_hash = _runtime_seal_sha256(runtime_seal)
    commands = [
        build_worker_command(
            config_path=config_path,
            device=device,
            worker_id=worker_id,
            worker_output=paths.output_root / "workers" / worker_id,
            parent_pid=os.getpid(),
            expected_runtime_seal_sha256=runtime_seal_hash,
        )
        for worker_id in config["execution"]["worker_ids"]
    ]
    return {
        "validate_only": True,
        "protocol_id": config["protocol_id"],
        "published_reference_verified": True,
        "published_reference_files_verified": contracts["reference_verification"]["verified_files"],
        "published_runtime_files_verified": len(contracts["verified_runtime_files"]),
        "published_extension_binary_verified": True,
        "selected_train_images": len(contracts["selected_ids"]),
        "selected_ids_sha256": ordered_ids_sha256(contracts["selected_ids"]),
        "planned_fresh_workers": len(commands),
        "planned_independent_runs_including_reference": len(commands) + 1,
        "planned_unordered_pairwise_comparisons": math.comb(len(commands) + 1, 2),
        "worker_commands": commands,
        "gpu_execution_started": False,
        "worker_processes_started": 0,
        "dataset_pixels_opened": 0,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "output_created": False,
        "tolerance_decision": "candidate_only_not_frozen",
        "runtime_seal_sha256": runtime_seal_hash,
        "runtime_files_unchanged": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested_config_path = Path(args.config).expanduser().resolve()
    runtime_seal = _runtime_seal(requested_config_path)
    config_path, config = load_audit_config(args.config)
    _assert_runtime_seal(runtime_seal, config_path, stage="post_config_validation")
    paths = resolve_audit_paths(
        config_path,
        config,
        output_override=getattr(args, "output_dir", None),
    )
    contracts = validate_contracts(config, paths)
    worker_id = getattr(args, "worker_id", None)
    if worker_id is not None:
        if getattr(args, "validate_only", False):
            raise ValueError("--validate-only cannot be combined with worker mode")
        worker_output = getattr(args, "worker_output", None)
        parent_pid = getattr(args, "expected_parent_pid", None)
        expected_runtime_seal_sha256 = getattr(
            args, "expected_runtime_seal_sha256", None
        )
        if (
            worker_output is None
            or parent_pid is None
            or expected_runtime_seal_sha256 is None
        ):
            raise ValueError(
                "internal worker mode requires output, parent PID, and runtime seal"
            )
        _require_equal(
            _runtime_seal_sha256(runtime_seal),
            expected_runtime_seal_sha256,
            "parent/worker runtime seal",
        )
        return run_worker(
            config_path=config_path,
            config=config,
            paths=paths,
            contracts=contracts,
            worker_id=worker_id,
            worker_output=worker_output.expanduser().resolve(),
            expected_parent_pid=parent_pid,
            device_name=args.device,
            runtime_seal=runtime_seal,
        )
    if getattr(args, "worker_output", None) is not None or getattr(
        args, "expected_parent_pid", None
    ) is not None or getattr(args, "expected_runtime_seal_sha256", None) is not None:
        raise ValueError("worker-only arguments require --worker-id")
    if getattr(args, "validate_only", False):
        return validate_only_payload(
            config_path=config_path,
            config=config,
            paths=paths,
            contracts=contracts,
            device=args.device,
            runtime_seal=runtime_seal,
        )
    return run_parent(
        config_path=config_path,
        config=config,
        paths=paths,
        contracts=contracts,
        device=args.device,
        runtime_seal=runtime_seal,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    if result.get("validate_only"):
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif "worker_id" in result:
        print(f"Worker {result['worker_id']} complete: {result['output']}")
    else:
        print(f"Artifacts: {result['published_output_dir']}")
        print(
            "Cross-process audit complete; tolerance is candidate-only and was not frozen."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuditPaths",
    "DEFAULT_CONFIG",
    "EXPECTED_PROTOCOL_ID",
    "RunDescriptor",
    "all_unordered_pairs",
    "build_worker_command",
    "compare_run_pair",
    "comparison_hard_gate",
    "derive_candidate",
    "finalize_artifact",
    "load_audit_config",
    "resolve_audit_paths",
    "run",
    "validate_contracts",
    "validate_only_payload",
    "verify_artifact_manifest",
]
