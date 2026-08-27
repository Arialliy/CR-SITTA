"""Materialize the frozen train-derived-64 Binary TENT calibration cache.

This is an input-cache layer only.  It never invokes a TTA method, optimizer,
or model forward.  Source-train masks are stored for the outer calibration
evaluator, but no method-facing interface is called and no test pixels are
opened.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from corruptions.corruption_protocol import load_severity_table
from corruptions.infrared_corruptions import apply_corruption
from dataio.corruption_cache import (
    TensorSequenceHasher,
    condition_key,
    ordered_ids_sha256,
    sha256_file,
)
from dataio.research_dataset import IRSTDResearchDataset, read_split_ids
import run_corruption_pilot as pilot_runner
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "configs" / "binary_tent_source_calibration_cache_v1.yaml"
)
DATASET_NAMES = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
EXPECTED_CONDITIONS = (
    ("clean", 0),
    ("gaussian_noise", 1),
    ("gaussian_noise", 3),
    ("gaussian_noise", 5),
    ("gaussian_blur", 1),
    ("gaussian_blur", 3),
    ("gaussian_blur", 5),
    ("low_contrast", 1),
    ("low_contrast", 3),
    ("low_contrast", 5),
    ("stripe_noise", 1),
    ("stripe_noise", 3),
    ("stripe_noise", 5),
)
EXPECTED_PILOT_CONDITIONS = (
    ("clean", 0),
    *((name, severity) for name in (
        "gaussian_noise",
        "gaussian_blur",
        "low_contrast",
        "stripe_noise",
    ) for severity in (1, 2, 3, 4, 5)),
)
TARGET_RELATIVE_PATH = "outer_evaluator/targets.npy"
METHOD_INPUT_MANIFEST_NAME = "method_input_manifest.json"
METHOD_FACING_SAMPLE_FIELDS = (
    "image",
    "image_id",
    "original_size",
    "dataset",
    "corruption",
    "severity",
    "seed",
)
METHOD_FACING_FORBIDDEN_FIELDS = frozenset(
    {"mask", "masks", "target", "targets", "label", "labels", "gt", "ground_truth"}
)
PROVENANCE_PATHS = (
    "configs/binary_tent_source_calibration_cache_v1.yaml",
    "configs/source_corruption_benchmark_fixed_splits.yaml",
    "configs/corruption_pilot_fixed_splits_round_02.yaml",
    "configs/retrain_fixed_splits.yaml",
    "corruptions/corruption_protocol.py",
    "corruptions/infrared_corruptions.py",
    "corruptions/severity_tables.yaml",
    "corruptions/severity_tables_round_02_candidate.yaml",
    "dataio/corruption_cache.py",
    "dataio/research_dataset.py",
    "materialize_binary_tent_source_calibration_cache.py",
    "run_corruption_pilot.py",
    "test_source.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify hashes/lineage without decoding dataset pixels or creating output.",
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


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    _require_equal(value, expected, label)


def _require_int(value: Any, expected: int, label: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer, not {type(value).__name__}")
    _require_equal(value, expected, label)


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer, not {type(value).__name__}")
    return value


def _strict_shape(value: Any, expected: tuple[int, ...], label: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be an integer sequence")
    parts = tuple(_strict_int(part, f"{label}[{index}]") for index, part in enumerate(value))
    _require_equal(parts, expected, label)


def _strict_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError(f"{label} must be a 64-character lowercase SHA256 string")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA256 string")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return dict(value)


def load_protocol(
    path: str | Path = DEFAULT_PROTOCOL,
) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("calibration cache protocol must be a YAML mapping")
    protocol = dict(loaded)
    _require_int(protocol.get("schema_version"), 1, "protocol schema_version")
    _require_equal(
        protocol.get("protocol_id"),
        "cr-sitta-binary-tent-source-calibration-cache-v1",
        "protocol_id",
    )
    scope = _require_mapping(protocol.get("scope"), "scope")
    for key, expected in (
        ("paper_result", False),
        ("independent_validation_set", False),
        ("use_test_images", False),
        ("use_test_labels", False),
        ("test_split_metadata_for_leakage_guard_only", True),
        ("adaptation_interface_invoked", False),
        ("adaptation_receives_labels", False),
        ("target_transition_hyperparameter_selection_allowed", False),
    ):
        _require_bool(scope.get(key), expected, f"scope.{key}")
    datasets = _require_mapping(protocol.get("datasets"), "datasets")
    _require_equal(tuple(datasets), DATASET_NAMES, "dataset order")
    _conditions(protocol)
    return resolved, protocol


def _conditions(protocol: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    input_protocol = _require_mapping(protocol.get("input_protocol"), "input_protocol")
    raw = input_protocol.get("ordered_conditions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("input_protocol.ordered_conditions must be a sequence")
    values_list: list[tuple[str, int]] = []
    for index, item in enumerate(raw):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            raise TypeError(f"condition {index} must be a two-item sequence")
        corruption, severity = item
        if not isinstance(corruption, str) or not corruption:
            raise TypeError(f"condition {index} corruption must be a non-empty string")
        if type(severity) is not int:
            raise TypeError(f"condition {index} severity must be an integer")
        values_list.append((corruption, severity))
    values = tuple(values_list)
    _require_equal(values, EXPECTED_CONDITIONS, "ordered 13-condition contract")
    _require_int(
        input_protocol.get("condition_count_per_dataset"),
        len(EXPECTED_CONDITIONS),
        "condition count",
    )
    _require_int(input_protocol.get("subset_size_per_dataset"), 64, "subset size")
    _require_int(input_protocol.get("seed"), 42, "base seed")
    return values


def _canonical_ids(path: Path) -> tuple[str, ...]:
    return tuple(Path(value).with_suffix("").as_posix() for value in read_split_ids(path))


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is missing or is a symlink: {path}")
    actual = sha256_file(path)
    _require_equal(actual, str(expected), f"{label} SHA256")
    return actual


def capture_runtime_seal(protocol_path: str | Path) -> dict[str, Any]:
    """Bind the launch protocol and every runtime-provenance source byte."""

    resolved_protocol = Path(protocol_path).expanduser().resolve()
    if not resolved_protocol.is_file() or resolved_protocol.is_symlink():
        raise FileNotFoundError(
            f"runtime-seal protocol is missing or is a symlink: {resolved_protocol}"
        )
    provenance_hashes: dict[str, str] = {}
    for relative in PROVENANCE_PATHS:
        path = _project_path(relative)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"runtime-seal provenance file is missing or is a symlink: {path}"
            )
        provenance_hashes[relative] = sha256_file(path)
    payload = {
        "schema_version": 1,
        "algorithm": "canonical-json-sha256-runtime-source-seal-v1",
        "protocol_path": str(resolved_protocol),
        "protocol_sha256": sha256_file(resolved_protocol),
        "provenance_files_sha256": provenance_hashes,
    }
    seal_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "runtime_seal_sha256": seal_hash}


def assert_runtime_seal(expected: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    """Fail closed when protocol/code/config bytes drift after process entry."""

    protocol_path = expected.get("protocol_path")
    if not isinstance(protocol_path, str) or not protocol_path:
        raise ValueError("runtime seal is missing protocol_path")
    observed = capture_runtime_seal(protocol_path)
    if dict(expected) != observed:
        raise RuntimeError(
            f"runtime provenance seal changed at {stage}; refusing mixed-source artifact"
        )
    return {
        "stage": stage,
        "runtime_seal_sha256": observed["runtime_seal_sha256"],
        "verified": True,
    }


def _assert_context_runtime_seal(
    context: Mapping[str, Any], *, stage: str
) -> dict[str, Any]:
    expected = _require_mapping(context.get("runtime_seal"), "runtime_seal")
    return assert_runtime_seal(expected, stage=stage)


def _severity_levels(table: Any) -> dict[str, dict[int, dict[str, float]]]:
    return {
        str(corruption): {
            int(severity): {str(key): float(value) for key, value in parameters.items()}
            for severity, parameters in levels.items()
        }
        for corruption, levels in table.levels.items()
    }


def _verify_global_provenance(protocol: Mapping[str, Any]) -> dict[str, Any]:
    chain = _require_mapping(protocol.get("provenance_chain"), "provenance_chain")
    verified_paths: dict[str, str] = {}
    for name in (
        "source_corruption_protocol",
        "round_02_pilot_protocol",
        "round_02_pilot_report",
        "round_02_candidate_severity_table",
        "frozen_severity_table",
    ):
        record = _require_mapping(chain.get(name), f"provenance_chain.{name}")
        path = _project_path(str(record["path"]))
        verified_paths[name] = _verify_hash(path, str(record["sha256"]), name)

    pipeline_files = _require_mapping(
        chain.get("input_pipeline_files"), "provenance_chain.input_pipeline_files"
    )
    for relative, expected in pipeline_files.items():
        path = _project_path(str(relative))
        verified_paths[f"pipeline:{relative}"] = _verify_hash(
            path, str(expected), f"input pipeline {relative}"
        )

    source = _require_mapping(protocol.get("source"), "source")
    parent_training = _project_path(str(source["parent_training_protocol"]))
    verified_paths["parent_training_protocol"] = _verify_hash(
        parent_training,
        str(source["parent_training_protocol_sha256"]),
        "parent training protocol",
    )

    source_protocol_path = _project_path(
        str(_require_mapping(chain["source_corruption_protocol"], "source protocol")["path"])
    )
    source_protocol = yaml.safe_load(source_protocol_path.read_text(encoding="utf-8"))
    source_protocol = _require_mapping(source_protocol, "source corruption protocol")
    source_corruption = _require_mapping(source_protocol.get("corruption"), "corruption")
    _require_equal(
        tuple((str(item[0]), int(item[1])) for item in source_corruption["ordered_conditions"]),
        EXPECTED_CONDITIONS,
        "source benchmark condition lineage",
    )
    _require_equal(int(source_corruption.get("seed", -1)), 42, "source benchmark seed")
    _require_equal(
        dict(_require_mapping(source_protocol.get("preprocessing"), "source preprocessing")),
        dict(_require_mapping(protocol.get("preprocessing"), "preprocessing")),
        "preprocessing lineage",
    )

    candidate_path = _project_path(
        str(_require_mapping(chain["round_02_candidate_severity_table"], "candidate")["path"])
    )
    frozen_path = _project_path(
        str(_require_mapping(chain["frozen_severity_table"], "frozen table")["path"])
    )
    candidate = load_severity_table(candidate_path)
    frozen = load_severity_table(frozen_path)
    _require_equal(candidate.frozen, False, "round-02 candidate frozen flag")
    _require_equal(candidate.calibration_completed, False, "candidate calibration flag")
    _require_equal(frozen.frozen, True, "frozen severity flag")
    _require_equal(frozen.calibration_completed, True, "frozen calibration flag")
    _require_equal(
        _severity_levels(frozen),
        _severity_levels(candidate),
        "candidate-to-frozen numeric severity lineage",
    )

    report_path = _project_path(
        str(_require_mapping(chain["round_02_pilot_report"], "pilot report")["path"])
    )
    report = _load_json(report_path)
    _require_equal(report.get("decision"), "freeze", "Pilot freeze decision")
    pilot_protocol_hash = str(
        _require_mapping(chain["round_02_pilot_protocol"], "pilot protocol")["sha256"]
    )
    candidate_hash = str(
        _require_mapping(chain["round_02_candidate_severity_table"], "candidate")["sha256"]
    )
    _require_equal(
        tuple(report.get("protocol_sha256", ())),
        (pilot_protocol_hash,),
        "Pilot report protocol lineage",
    )
    _require_equal(
        tuple(report.get("severity_table_sha256", ())),
        (candidate_hash,),
        "Pilot report severity lineage",
    )
    return {
        "verified_file_sha256": verified_paths,
        "candidate_and_frozen_numeric_levels_identical": True,
        "pilot_decision": "freeze",
        "pilot_report": report,
    }


def _verify_parent_pilot(
    *,
    dataset_name: str,
    dataset_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    global_provenance: Mapping[str, Any],
    verify_auxiliary_artifact_files: bool,
) -> dict[str, Any]:
    parent = _require_mapping(dataset_contract.get("parent_pilot"), "parent_pilot")
    pilot_path = _project_path(str(parent["artifact"]))
    manifest_path = _project_path(str(parent["manifest"]))
    complete_path = _project_path(str(parent["complete"]))
    _verify_hash(pilot_path, str(parent["artifact_sha256"]), "parent Pilot")
    _verify_hash(manifest_path, str(parent["manifest_sha256"]), "parent Pilot manifest")
    _verify_hash(complete_path, str(parent["complete_sha256"]), "parent Pilot COMPLETE")

    pilot = _load_json(pilot_path)
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    _require_equal(complete.get("complete"), True, "parent Pilot complete flag")
    _require_equal(complete.get("dataset"), dataset_name, "parent COMPLETE dataset")
    _require_equal(
        complete.get("pilot_json_sha256"),
        parent["artifact_sha256"],
        "parent COMPLETE Pilot lineage",
    )
    _require_equal(
        complete.get("artifact_manifest_sha256"),
        parent["manifest_sha256"],
        "parent COMPLETE manifest lineage",
    )
    _require_equal(pilot.get("dataset"), dataset_name, "parent Pilot dataset")
    _require_equal(pilot.get("formal_artifact"), True, "parent formal_artifact")
    _require_equal(int(pilot.get("calibration_round", -1)), 2, "calibration round")
    _require_equal(int(pilot.get("seed", -1)), 42, "parent Pilot seed")

    chain = _require_mapping(protocol["provenance_chain"], "provenance_chain")
    pilot_protocol_hash = str(
        _require_mapping(chain["round_02_pilot_protocol"], "pilot protocol")["sha256"]
    )
    candidate_hash = str(
        _require_mapping(chain["round_02_candidate_severity_table"], "candidate")["sha256"]
    )
    _require_equal(pilot.get("protocol_sha256"), pilot_protocol_hash, "Pilot protocol SHA")
    _require_equal(manifest.get("protocol_sha256"), pilot_protocol_hash, "manifest protocol SHA")
    _require_equal(
        pilot.get("checks", {}).get("severity_table_sha256_before"),
        candidate_hash,
        "Pilot candidate severity SHA",
    )

    manifest_files = _require_mapping(manifest.get("files_sha256"), "Pilot manifest files")
    _require_equal(
        manifest_files.get("pilot.json"),
        parent["artifact_sha256"],
        "Pilot manifest primary lineage",
    )
    artifact_root = pilot_path.parent
    for relative, expected in manifest_files.items():
        if relative != "pilot.json" and not verify_auxiliary_artifact_files:
            continue
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe Pilot artifact path: {relative}")
        _verify_hash(artifact_root / relative_path, str(expected), f"Pilot artifact {relative}")

    selection = _require_mapping(pilot.get("selection"), "Pilot selection")
    selected_ids = tuple(str(value) for value in selection.get("selected_ids", ()))
    _require_equal(len(selected_ids), 64, "parent selected ID count")
    _require_equal(len(set(selected_ids)), 64, "parent selected ID uniqueness")
    selected_hash = ordered_ids_sha256(selected_ids)
    _require_equal(
        selected_hash,
        str(dataset_contract["pilot_ordered_ids_sha256"]),
        "parent ordered selected IDs",
    )
    source_manifests = _require_mapping(
        selection.get("selected_source_manifests"), "Pilot source manifests"
    )
    for actual_key, expected_key in (
        ("combined_sha256", "pilot_source_manifest_sha256"),
        ("images_sha256", "pilot_image_manifest_sha256"),
        ("masks_sha256", "pilot_mask_manifest_sha256"),
    ):
        _require_equal(
            source_manifests.get(actual_key),
            dataset_contract[expected_key],
            f"Pilot source manifest {actual_key}",
        )

    records = pilot.get("conditions")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("parent Pilot conditions must be a sequence")
    by_condition: dict[tuple[str, int], Mapping[str, Any]] = {}
    for raw in records:
        record = _require_mapping(raw, "Pilot condition")
        key = (str(record["corruption"]), int(record["severity"]))
        if key in by_condition:
            raise ValueError(f"duplicate parent Pilot condition: {key}")
        by_condition[key] = record
    _require_equal(tuple(by_condition), EXPECTED_PILOT_CONDITIONS, "parent Pilot conditions")

    expected_hashes: dict[str, str] = {}
    gt_hashes: set[str] = set()
    for corruption, severity in EXPECTED_CONDITIONS:
        record = by_condition[(corruption, severity)]
        _require_equal(int(record.get("evaluated_images", -1)), 64, "Pilot image count")
        _require_equal(
            record.get("evaluated_ids_sha256"), selected_hash, "Pilot evaluated ID hash"
        )
        _require_equal(record.get("ids_identical_to_selected"), True, "Pilot ID identity")
        _require_equal(record.get("exact_input_reproduction"), True, "Pilot input replay")
        _require_equal(record.get("exact_mask_reproduction"), True, "Pilot mask replay")
        input_hash = str(record.get("model_input_tensor_sha256", ""))
        gt_hash = str(record.get("gt_mask_tensor_sha256", ""))
        if len(input_hash) != 64 or len(gt_hash) != 64:
            raise ValueError(f"invalid Pilot tensor hash for {corruption}/S{severity}")
        _require_equal(
            record.get("replay_model_input_tensor_sha256"),
            input_hash,
            "Pilot input replay hash",
        )
        _require_equal(
            record.get("replay_gt_mask_tensor_sha256"), gt_hash, "Pilot GT replay hash"
        )
        expected_hashes[condition_key(corruption, severity)] = input_hash
        gt_hashes.add(gt_hash)
    _require_equal(len(gt_hashes), 1, "Pilot GT hash across 13 conditions")
    expected_gt_hash = next(iter(gt_hashes))
    _require_equal(
        expected_gt_hash,
        str(dataset_contract["expected_gt_tensor_sequence_sha256"]),
        "configured expected GT tensor hash",
    )

    checks = _require_mapping(pilot.get("checks"), "Pilot checks")
    boundary = _require_mapping(pilot.get("fixed_test_boundary"), "fixed-test boundary")
    for actual, expected, label in (
        (checks.get("fixed_test_id_overlap_count"), 0, "Pilot test overlap"),
        (checks.get("test_image_open_count"), 0, "Pilot test image opens"),
        (checks.get("test_mask_open_count"), 0, "Pilot test mask opens"),
        (boundary.get("test_dataset_constructed"), False, "Pilot test dataset"),
        (boundary.get("test_images_opened"), 0, "Pilot boundary test image opens"),
        (boundary.get("test_masks_opened"), 0, "Pilot boundary test mask opens"),
    ):
        _require_equal(actual, expected, label)

    report = _require_mapping(global_provenance["pilot_report"], "Pilot report")
    report_dataset = _require_mapping(
        _require_mapping(report.get("datasets"), "Pilot report datasets").get(dataset_name),
        f"Pilot report {dataset_name}",
    )
    report_artifact = _require_mapping(report_dataset.get("artifact"), "report artifact")
    _require_equal(report_artifact.get("valid"), True, "Pilot report artifact validity")
    _require_equal(
        report_artifact.get("pilot_json_sha256"),
        parent["artifact_sha256"],
        "Pilot report artifact SHA",
    )
    _require_equal(
        report_artifact.get("artifact_manifest_sha256"),
        parent["manifest_sha256"],
        "Pilot report manifest SHA",
    )
    return {
        "pilot_path": pilot_path,
        "pilot_manifest_path": manifest_path,
        "pilot_complete_path": complete_path,
        "selected_ids": selected_ids,
        "selected_ids_sha256": selected_hash,
        "expected_condition_tensor_sequence_sha256": expected_hashes,
        "expected_gt_tensor_sequence_sha256": expected_gt_hash,
        "source_manifests": dict(source_manifests),
    }


def _validate_cache_layout(protocol: Mapping[str, Any]) -> None:
    cache = _require_mapping(protocol.get("materialized_cache"), "materialized_cache")
    _require_equal(
        cache.get("root"),
        "results/binary_tent/source_calibration_cache_v1",
        "cache result root",
    )
    _require_equal(
        cache.get("cache_format"),
        "nsfpn-materialized-source-calibration-cache-v1",
        "cache format",
    )
    for key, expected in (
        ("immutable_after_completion", True),
        ("overwrite_existing_complete_cache", False),
        ("build_in_sibling_staging_directory", True),
        ("publish_with_atomic_directory_rename", True),
        ("exclusive_sibling_publish_lock", True),
        ("require_complete_sentinel", True),
        ("consumers_must_not_regenerate_corruptions", True),
        ("consumers_use_read_only_mmap_and_private_sample_copy", True),
    ):
        _require_bool(cache.get(key), expected, f"materialized_cache.{key}")
    required_hashes = cache.get("required_hashes")
    if not isinstance(required_hashes, Sequence) or isinstance(
        required_hashes, (str, bytes)
    ):
        raise TypeError("materialized_cache.required_hashes must be a sequence")
    _require_equal(
        tuple(required_hashes),
        (
            "ordered_ids_sha256",
            "source_file_manifest_sha256",
            "tensor_sequence_sha256",
            "file_sha256",
            "manifest_sha256",
            "method_input_manifest_sha256",
        ),
        "required cache hashes",
    )
    images = _require_mapping(cache.get("images"), "cache images")
    targets = _require_mapping(cache.get("targets"), "cache targets")
    for record, shape, label in (
        (images, (64, 3, 256, 256), "images"),
        (targets, (64, 1, 256, 256), "targets"),
    ):
        _require_equal(record.get("format"), "numpy_npy", f"{label} format")
        _require_equal(record.get("dtype"), "little_endian_float32", f"{label} dtype")
        _strict_shape(record.get("shape"), shape, f"{label} shape")
    _require_equal(images.get("contiguous_order"), "C", "image contiguous order")
    _require_bool(images.get("one_file_per_condition"), True, "image shard policy")
    _require_bool(targets.get("one_file_per_dataset"), True, "target shard policy")
    _require_equal(targets.get("method_facing_access"), "forbidden", "label firewall")
    _require_equal(
        targets.get("path"), TARGET_RELATIVE_PATH, "target cache path"
    )
    _require_bool(
        targets.get("outer_evaluator_requires_explicit_episodes_complete"),
        True,
        "delayed target loading",
    )
    consumers = _require_mapping(cache.get("official_consumers"), "official consumers")
    _require_equal(
        consumers.get("method_facing"),
        "SourceCalibrationMethodInputDataset",
        "official method-facing consumer",
    )
    _require_equal(
        consumers.get("method_facing_manifest"),
        METHOD_INPUT_MANIFEST_NAME,
        "official method-facing manifest",
    )
    _require_bool(
        consumers.get("expected_protocol_sha256_required"),
        True,
        "official consumer external protocol anchor",
    )
    raw_fields = consumers.get("method_facing_fields")
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        raise TypeError("official method-facing fields must be a sequence")
    _require_equal(
        tuple(raw_fields), METHOD_FACING_SAMPLE_FIELDS, "method-facing sample fields"
    )
    _require_equal(
        consumers.get("outer_evaluator_targets"),
        "load_outer_evaluator_targets",
        "official outer-evaluator target loader",
    )
    _require_equal(
        consumers.get("generic_cached_corruption_dataset_for_adaptation"),
        "forbidden",
        "generic cache consumer adaptation policy",
    )
    runtime_seal = _require_mapping(cache.get("runtime_seal"), "cache runtime_seal")
    _require_bool(runtime_seal.get("enabled"), True, "runtime seal enabled")
    _require_equal(
        runtime_seal.get("drift_policy"),
        "fail_closed_and_do_not_publish",
        "runtime seal drift policy",
    )
    _require_bool(
        runtime_seal.get("record_in_manifest_and_complete"),
        True,
        "runtime seal artifact recording",
    )


def validate_contract(
    *,
    protocol_path: str | Path,
    dataset_name: str,
    output_override: Path | None = None,
    require_output_absent: bool = True,
    runtime_seal: Mapping[str, Any] | None = None,
    verify_source_file_bytes: bool = False,
) -> dict[str, Any]:
    """Validate all metadata and hashes without constructing a pixel dataset."""

    entry_seal = (
        dict(runtime_seal)
        if runtime_seal is not None
        else capture_runtime_seal(protocol_path)
    )
    assert_runtime_seal(entry_seal, stage="validate_contract_start")
    resolved_protocol, protocol = load_protocol(protocol_path)
    protocol_sha256 = sha256_file(resolved_protocol)
    _require_equal(
        entry_seal["protocol_sha256"],
        protocol_sha256,
        "runtime seal versus loaded protocol SHA256",
    )
    _validate_cache_layout(protocol)
    conditions = _conditions(protocol)
    global_provenance = _verify_global_provenance(protocol)
    datasets = _require_mapping(protocol["datasets"], "datasets")
    if dataset_name not in datasets:
        raise ValueError(f"dataset {dataset_name!r} is absent from protocol")
    dataset_contract = _require_mapping(datasets[dataset_name], dataset_name)
    train_count = _strict_int(dataset_contract.get("train_images"), "train image count")
    test_count = _strict_int(dataset_contract.get("test_images"), "test image count")
    _strict_int(dataset_contract.get("checkpoint_epoch"), "checkpoint epoch")

    dataset_root = _project_path(str(dataset_contract["root"]))
    train_split = _project_path(str(dataset_contract["train_split"]))
    test_split = _project_path(str(dataset_contract["test_split"]))
    checkpoint = _project_path(str(dataset_contract["checkpoint"]))
    _verify_hash(train_split, str(dataset_contract["train_split_sha256"]), "train split")
    _verify_hash(test_split, str(dataset_contract["test_split_sha256"]), "test split metadata")
    _verify_hash(checkpoint, str(dataset_contract["checkpoint_sha256"]), "checkpoint")

    train_ids = _canonical_ids(train_split)
    test_ids = _canonical_ids(test_split)
    _require_equal(len(train_ids), train_count, "train count")
    _require_equal(len(test_ids), test_count, "test count")
    _require_equal(len(set(train_ids)), len(train_ids), "train ID uniqueness")
    _require_equal(len(set(test_ids)), len(test_ids), "test ID uniqueness")
    overlap = tuple(sorted(set(train_ids) & set(test_ids)))
    _require_equal(overlap, (), "fixed train/test ID overlap")

    parent = _verify_parent_pilot(
        dataset_name=dataset_name,
        dataset_contract=dataset_contract,
        protocol=protocol,
        global_provenance=global_provenance,
        verify_auxiliary_artifact_files=verify_source_file_bytes,
    )
    selected_ids = tuple(parent["selected_ids"])
    ranked_ids = tuple(
        record.image_id for record in pilot_runner.sha256_ranked_subset(train_ids, 64)
    )
    _require_equal(selected_ids, ranked_ids, "recomputed SHA-ranked train subset")
    train_set = set(train_ids)
    test_set = set(test_ids)
    _require_equal(
        tuple(value for value in selected_ids if value not in train_set),
        (),
        "selected IDs absent from train",
    )
    _require_equal(
        tuple(value for value in selected_ids if value in test_set),
        (),
        "selected IDs overlapping test",
    )

    selected_files = pilot_runner._resolve_selected_source_paths(
        dataset_root, selected_ids
    )
    if verify_source_file_bytes:
        source_manifests = pilot_runner.source_file_manifests(
            selected_ids, selected_files
        )
        _require_equal(
            source_manifests,
            parent["source_manifests"],
            "selected train source file manifests",
        )
    else:
        source_manifests = dict(parent["source_manifests"])
    image_files_hashed = 64 if verify_source_file_bytes else 0
    mask_files_hashed = 64 if verify_source_file_bytes else 0
    image_bytes_read = (
        sum(selected_files[("image", image_id)].stat().st_size for image_id in selected_ids)
        if verify_source_file_bytes
        else 0
    )
    mask_bytes_read = (
        sum(selected_files[("mask", image_id)].stat().st_size for image_id in selected_ids)
        if verify_source_file_bytes
        else 0
    )

    checkpoint_payload, checkpoint_summary = pilot_runner.validate_checkpoint_contract(
        checkpoint,
        dataset_name=dataset_name,
        dataset_contract=dataset_contract,
        protocol=protocol,
    )
    model = source_runner.build_nsfpn_model()
    model.load_state_dict(checkpoint_payload["state_dict"], strict=True)
    checkpoint_summary["strict_state_dict_load_verified"] = True
    del model
    del checkpoint_payload

    output_root = _project_path(str(protocol["materialized_cache"]["root"]))
    final_output = (
        output_override.expanduser().resolve()
        if output_override is not None
        else output_root / dataset_name
    )
    if require_output_absent and final_output.exists():
        raise FileExistsError(
            f"cache destination already exists; refusing overwrite: {final_output}"
        )

    index_by_id = {image_id: index for index, image_id in enumerate(train_ids)}
    selected_indices = tuple(index_by_id[image_id] for image_id in selected_ids)
    public_validation = {
        "validate_only_pixel_dataset_constructed": False,
        "dataset_pixel_arrays_decoded": 0,
        "dataset_image_files_hashed_for_source_manifest": image_files_hashed,
        "dataset_mask_files_hashed_for_source_manifest": mask_files_hashed,
        "dataset_image_file_bytes_read_for_source_manifest": image_bytes_read,
        "dataset_mask_file_bytes_read_for_source_manifest": mask_bytes_read,
        "selected_source_manifest_byte_verification_deferred": (
            not verify_source_file_bytes
        ),
        "test_dataset_constructed": False,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "fixed_train_split_hash_verified": True,
        "fixed_test_split_metadata_hash_verified": True,
        "fixed_train_test_disjoint": True,
        "selected_ids_in_fixed_train": True,
        "selected_ids_absent_from_fixed_test": True,
        "selected_source_manifests_verified": verify_source_file_bytes,
        "parent_pilot_artifact_chain_verified": True,
        "frozen_severity_provenance_chain_verified": True,
        "checkpoint_provenance_chain_verified": True,
        "condition_tensor_hashes_bound_to_parent_pilot": True,
        "adaptation_interface_invoked": False,
        "method_received_labels": False,
        "runtime_seal_verified": True,
        "checkpoint_state_dict_strict_load_verified": True,
    }
    assert_runtime_seal(entry_seal, stage="validate_contract_complete")
    return {
        "protocol": protocol,
        "protocol_path": resolved_protocol,
        "protocol_sha256": protocol_sha256,
        "dataset_name": dataset_name,
        "dataset_contract": dict(dataset_contract),
        "dataset_root": dataset_root,
        "train_split": train_split,
        "test_split": test_split,
        "checkpoint": checkpoint,
        "checkpoint_summary": checkpoint_summary,
        "conditions": conditions,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "selected_ids": selected_ids,
        "selected_indices": selected_indices,
        "selected_files": selected_files,
        "source_manifests": source_manifests,
        "expected_condition_tensor_sequence_sha256": dict(
            parent["expected_condition_tensor_sequence_sha256"]
        ),
        "expected_gt_tensor_sequence_sha256": parent[
            "expected_gt_tensor_sequence_sha256"
        ],
        "global_provenance": global_provenance,
        "runtime_seal": entry_seal,
        "final_output": final_output,
        "validation": public_validation,
    }


def _write_memmap_atomic(
    destination: Path, *, shape: tuple[int, ...]
) -> tuple[Path, np.memmap]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if partial.exists():
        raise FileExistsError(f"partial cache file already exists: {partial}")
    mapped = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=shape,
        fortran_order=False,
        version=(2, 0),
    )
    return partial, mapped


def _hash_memmap_records(path: Path, image_ids: Sequence[str]) -> str:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape[0] != len(image_ids):
        raise ValueError("materialized tensor first dimension does not match IDs")
    hasher = TensorSequenceHasher()
    for image_id, value in zip(image_ids, values, strict=True):
        hasher.update(image_id, value)
    return hasher.hexdigest()


def _safe_cache_file(root: Path, relative: str, *, label: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe {label} path: {relative}")
    path = root / relative_path
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} file is missing or is a symlink: {path}")
    return path


def _build_method_facing_sample(
    *,
    image: torch.Tensor,
    image_id: str,
    original_size: tuple[int, int],
    dataset_name: str,
    corruption: str,
    severity: int,
    seed: int,
) -> dict[str, Any]:
    """Single source of truth for the adaptation-facing, image-only schema."""

    return {
        "image": image,
        "image_id": image_id,
        "original_size": original_size,
        "dataset": dataset_name,
        "corruption": corruption,
        "severity": severity,
        "seed": seed,
    }


def _label_firewall_contract_evidence() -> dict[str, Any]:
    """Exercise the exact sample builder used by the official consumer."""

    sample = _build_method_facing_sample(
        image=torch.zeros((3, 1, 1), dtype=torch.float32),
        image_id="firewall-self-test",
        original_size=(1, 1),
        dataset_name="firewall-self-test",
        corruption="clean",
        severity=0,
        seed=0,
    )
    fields = tuple(sample)
    _require_equal(fields, METHOD_FACING_SAMPLE_FIELDS, "method-facing sample schema")
    forbidden_hits = tuple(sorted(set(fields) & METHOD_FACING_FORBIDDEN_FIELDS))
    _require_equal(forbidden_hits, (), "method-facing forbidden label fields")
    return {
        "contract_id": "source-calibration-method-image-only-v1",
        "official_consumer": "SourceCalibrationMethodInputDataset",
        "sample_builder": "_build_method_facing_sample",
        "sample_fields": list(fields),
        "forbidden_fields_exposed": list(forbidden_hits),
        "outer_evaluator_target_path_exposed": False,
        "verified": True,
    }


def _build_method_input_manifest(
    manifest: Mapping[str, Any], *, outer_manifest_sha256: str
) -> dict[str, Any]:
    """Create a sanitized manifest with no evaluator target path or label metadata."""

    _strict_sha256(outer_manifest_sha256, "outer manifest SHA256")
    raw_conditions = manifest.get("conditions")
    if not isinstance(raw_conditions, Sequence) or isinstance(
        raw_conditions, (str, bytes)
    ):
        raise TypeError("cache conditions must be a sequence")
    files = _require_mapping(manifest.get("files"), "cache files")
    conditions: list[dict[str, Any]] = []
    image_files: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_conditions):
        record = _require_mapping(raw, f"cache condition {index}")
        corruption = record.get("corruption")
        if not isinstance(corruption, str) or not corruption:
            raise TypeError(f"cache condition {index} corruption must be a string")
        severity = _strict_int(record.get("severity"), f"cache condition {index} severity")
        _require_int(record.get("index"), index, f"cache condition {index} index")
        key = condition_key(corruption, severity)
        _require_equal(record.get("key"), key, f"cache condition {index} key")
        relative = str(record.get("path"))
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 2
            or relative_path.parts[0] != "conditions"
            or relative_path.suffix != ".npy"
        ):
            raise ValueError(f"unsafe method-facing condition path: {relative}")
        _strict_shape(record.get("shape"), (64, 3, 256, 256), f"{key} shape")
        _require_equal(record.get("dtype"), "little_endian_float32", f"{key} dtype")
        _require_equal(record.get("contiguous_order"), "C", f"{key} order")
        file_record = _require_mapping(files.get(relative), f"cache file {relative}")
        byte_count = _strict_int(file_record.get("bytes"), f"{key} bytes")
        if byte_count <= 0:
            raise ValueError(f"{key} byte count must be positive")
        file_hash = _strict_sha256(file_record.get("sha256"), f"{key} file hash")
        _require_equal(file_hash, record.get("file_sha256"), f"{key} file hash")
        tensor_hash = _strict_sha256(
            record.get("tensor_sequence_sha256"), f"{key} tensor sequence hash"
        )
        image_files[relative] = {
            "sha256": file_hash,
            "bytes": byte_count,
        }
        conditions.append(
            {
                "index": index,
                "key": key,
                "corruption": corruption,
                "severity": severity,
                "path": relative,
                "shape": [64, 3, 256, 256],
                "dtype": "little_endian_float32",
                "contiguous_order": "C",
                "tensor_sequence_sha256": tensor_hash,
                "file_sha256": file_hash,
            }
        )
    _require_equal(len(conditions), 13, "method-facing condition count")
    _require_equal(
        tuple((record["corruption"], record["severity"]) for record in conditions),
        EXPECTED_CONDITIONS,
        "method-facing ordered conditions",
    )
    _require_equal(
        len({record["path"] for record in conditions}),
        13,
        "method-facing unique condition paths",
    )
    image_ids = tuple(str(value) for value in manifest.get("image_ids", ()))
    _require_equal(len(image_ids), 64, "method-facing image ID count")
    original_sizes = manifest.get("original_sizes")
    if not isinstance(original_sizes, Sequence) or isinstance(
        original_sizes, (str, bytes)
    ):
        raise TypeError("method-facing original_sizes must be a sequence")
    _require_equal(len(original_sizes), 64, "method-facing original-size count")
    sanitized = {
        "schema_version": 1,
        "manifest_role": "method_facing_image_only",
        "cache_format": manifest["cache_format"],
        "protocol_sha256": manifest["protocol_sha256"],
        "runtime_seal_sha256": manifest["runtime_seal_sha256"],
        "outer_manifest_sha256": outer_manifest_sha256,
        "dataset": manifest["dataset"],
        "seed": _strict_int(manifest.get("seed"), "cache seed"),
        "image_ids": list(image_ids),
        "ordered_ids_sha256": manifest["ordered_ids_sha256"],
        "original_sizes": list(original_sizes),
        "condition_count": len(conditions),
        "conditions": conditions,
        "files": image_files,
        "label_firewall_contract": _label_firewall_contract_evidence(),
    }
    serialized = json.dumps(sanitized, sort_keys=True, separators=(",", ":"))
    if TARGET_RELATIVE_PATH in serialized or '"targets"' in serialized:
        raise RuntimeError("method-facing manifest exposed the outer evaluator target")
    return sanitized


def _load_completed_cache_metadata(
    cache_dir: str | Path, *, expected_protocol_sha256: str | None
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(cache_dir).expanduser().resolve()
    manifest_path = _safe_cache_file(root, "manifest.json", label="cache manifest")
    complete_path = _safe_cache_file(root, "COMPLETE.json", label="cache COMPLETE")
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    _require_bool(complete.get("complete"), True, "cache complete flag")
    _require_equal(
        complete.get("manifest_sha256"),
        sha256_file(manifest_path),
        "cache completion manifest SHA256",
    )
    method_manifest_path = _safe_cache_file(
        root, METHOD_INPUT_MANIFEST_NAME, label="method-facing manifest"
    )
    _require_equal(
        complete.get("method_input_manifest_sha256"),
        sha256_file(method_manifest_path),
        "cache completion method-facing manifest SHA256",
    )
    _require_equal(
        manifest.get("cache_format"),
        "nsfpn-materialized-source-calibration-cache-v1",
        "method-facing cache format",
    )
    if expected_protocol_sha256 is not None:
        _require_equal(
            manifest.get("protocol_sha256"),
            expected_protocol_sha256,
            "method-facing cache protocol SHA256",
        )
    _require_equal(
        complete.get("runtime_seal_sha256"),
        manifest.get("runtime_seal_sha256"),
        "cache runtime seal lineage",
    )
    image_ids = tuple(str(value) for value in manifest.get("image_ids", ()))
    _require_equal(len(image_ids), 64, "method-facing cache ID count")
    _require_equal(len(set(image_ids)), 64, "method-facing cache ID uniqueness")
    _require_equal(
        manifest.get("ordered_ids_sha256"),
        ordered_ids_sha256(image_ids),
        "method-facing ordered IDs SHA256",
    )
    label_firewall = _require_mapping(
        manifest.get("label_firewall"), "cache label_firewall"
    )
    _require_equal(
        label_firewall.get("official_method_facing_consumer"),
        "SourceCalibrationMethodInputDataset",
        "method-facing consumer contract",
    )
    _require_equal(
        label_firewall.get("outer_evaluator_target_loader"),
        "load_outer_evaluator_targets",
        "outer evaluator target loader contract",
    )
    evidence = _require_mapping(
        label_firewall.get("contract_evidence"), "label firewall evidence"
    )
    _require_bool(evidence.get("verified"), True, "label firewall evidence")
    _require_equal(
        complete.get("label_firewall_verified"),
        evidence.get("verified"),
        "completion label firewall evidence",
    )
    _require_bool(
        complete.get("label_firewall_verified"), True, "completion label firewall flag"
    )
    return root, manifest, complete


def _load_method_facing_cache_metadata(
    cache_dir: str | Path, *, expected_protocol_sha256: str | None
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Load only the sanitized image manifest; never parse the outer manifest."""

    root = Path(cache_dir).expanduser().resolve()
    manifest_path = _safe_cache_file(
        root, METHOD_INPUT_MANIFEST_NAME, label="method-facing manifest"
    )
    complete_path = _safe_cache_file(root, "COMPLETE.json", label="cache COMPLETE")
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    _require_bool(complete.get("complete"), True, "cache complete flag")
    _require_equal(
        complete.get("method_input_manifest_sha256"),
        sha256_file(manifest_path),
        "cache completion method-facing manifest SHA256",
    )
    _require_equal(
        manifest.get("outer_manifest_sha256"),
        complete.get("manifest_sha256"),
        "sanitized-to-outer manifest lineage",
    )
    _require_equal(
        manifest.get("manifest_role"),
        "method_facing_image_only",
        "method-facing manifest role",
    )
    _require_equal(
        manifest.get("cache_format"),
        "nsfpn-materialized-source-calibration-cache-v1",
        "method-facing cache format",
    )
    if expected_protocol_sha256 is not None:
        _require_equal(
            manifest.get("protocol_sha256"),
            expected_protocol_sha256,
            "method-facing cache protocol SHA256",
        )
    _require_equal(
        complete.get("runtime_seal_sha256"),
        manifest.get("runtime_seal_sha256"),
        "method-facing runtime seal lineage",
    )
    image_ids = tuple(str(value) for value in manifest.get("image_ids", ()))
    _require_equal(len(image_ids), 64, "method-facing cache ID count")
    _require_equal(len(set(image_ids)), 64, "method-facing cache ID uniqueness")
    _require_equal(
        manifest.get("ordered_ids_sha256"),
        ordered_ids_sha256(image_ids),
        "method-facing ordered IDs SHA256",
    )
    evidence = _require_mapping(
        manifest.get("label_firewall_contract"), "method-facing firewall contract"
    )
    _require_bool(evidence.get("verified"), True, "method-facing firewall verification")
    _require_bool(
        complete.get("label_firewall_verified"),
        True,
        "completion label firewall verification",
    )
    _require_equal(
        complete.get("label_firewall_contract"),
        evidence.get("contract_id"),
        "completion label firewall contract",
    )
    evidence_sha256 = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _require_equal(
        complete.get("label_firewall_evidence_sha256"),
        evidence_sha256,
        "completion label firewall evidence SHA256",
    )
    _require_bool(
        complete.get("adaptation_interface_invoked"),
        False,
        "completion adaptation invocation flag",
    )
    _require_equal(
        tuple(evidence.get("sample_fields", ())),
        METHOD_FACING_SAMPLE_FIELDS,
        "method-facing firewall fields",
    )
    _require_bool(
        evidence.get("outer_evaluator_target_path_exposed"),
        False,
        "method-facing target path exposure",
    )
    return root, manifest, complete


class SourceCalibrationMethodInputDataset(Dataset[dict[str, Any]]):
    """Official image-only calibration-cache consumer for a TTA method.

    The constructor verifies and opens only the requested condition shard.
    It deliberately never hashes, opens, maps, or returns the target shard.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        corruption: str,
        severity: int,
        expected_protocol_sha256: str,
    ) -> None:
        _strict_sha256(
            expected_protocol_sha256, "method-facing expected protocol SHA256"
        )
        root, manifest, _complete = _load_method_facing_cache_metadata(
            cache_dir, expected_protocol_sha256=expected_protocol_sha256
        )
        key = condition_key(corruption, severity)
        raw_conditions = manifest.get("conditions")
        if not isinstance(raw_conditions, Sequence) or isinstance(
            raw_conditions, (str, bytes)
        ):
            raise TypeError("cache conditions must be a sequence")
        conditions = {
            str(_require_mapping(record, "cache condition")["key"]): _require_mapping(
                record, "cache condition"
            )
            for record in raw_conditions
        }
        if key not in conditions:
            raise ValueError(f"condition {key} is absent from calibration cache")
        condition = conditions[key]
        path = _safe_cache_file(root, str(condition["path"]), label=key)
        files = _require_mapping(manifest.get("files"), "cache files")
        file_record = _require_mapping(
            files.get(str(condition["path"])), f"cache file record {key}"
        )
        expected_bytes = _strict_int(file_record.get("bytes"), f"{key} bytes")
        _require_equal(path.stat().st_size, expected_bytes, f"{key} bytes")
        _require_equal(sha256_file(path), file_record["sha256"], f"{key} file SHA256")
        images = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            images.shape != (64, 3, 256, 256)
            or images.dtype.str != "<f4"
            or not images.flags.c_contiguous
            or images.flags.writeable
        ):
            raise ValueError(
                f"method-facing image tensor contract drift: "
                f"{images.shape}/{images.dtype}/{images.flags.c_contiguous}/"
                f"writeable={images.flags.writeable}"
            )
        self.root = root
        self.condition = dict(condition)
        self.images = images
        self.image_ids = tuple(str(value) for value in manifest["image_ids"])
        self.original_sizes = tuple(
            tuple(int(part) for part in value) for value in manifest["original_sizes"]
        )
        _require_equal(len(self.original_sizes), 64, "original-size count")
        self.dataset_name = str(manifest["dataset"])
        self.seed = _strict_int(manifest.get("seed"), "method-facing seed")
        self.targets_opened = False
        self.method_metadata = {
            "dataset": self.dataset_name,
            "condition": {
                "corruption": str(condition["corruption"]),
                "severity": int(condition["severity"]),
            },
            "seed": self.seed,
            "ordered_ids_sha256": manifest["ordered_ids_sha256"],
            "protocol_sha256": manifest["protocol_sha256"],
            "runtime_seal_sha256": manifest["runtime_seal_sha256"],
            "targets_opened": False,
            "sample_fields": METHOD_FACING_SAMPLE_FIELDS,
        }

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = torch.from_numpy(np.array(self.images[index], copy=True))
        if not torch.isfinite(image).all():
            raise ValueError("method-facing cached image contains NaN/Inf")
        # No mask/target/label key exists in the official method-facing sample.
        return _build_method_facing_sample(
            image=image,
            image_id=self.image_ids[index],
            original_size=self.original_sizes[index],
            dataset_name=self.dataset_name,
            corruption=str(self.condition["corruption"]),
            severity=_strict_int(
                self.condition.get("severity"), "method-facing condition severity"
            ),
            seed=self.seed,
        )


def load_outer_evaluator_targets(
    cache_dir: str | Path,
    *,
    episodes_complete: bool,
    expected_protocol_sha256: str,
) -> np.memmap:
    """Explicitly open targets only after every method-facing episode finishes."""

    _require_bool(episodes_complete, True, "episodes_complete")
    _strict_sha256(
        expected_protocol_sha256, "outer-evaluator expected protocol SHA256"
    )
    root, manifest, _complete = _load_completed_cache_metadata(
        cache_dir, expected_protocol_sha256=expected_protocol_sha256
    )
    targets = _require_mapping(manifest.get("targets"), "cache targets")
    _require_equal(
        targets.get("role"),
        "outer_source_side_calibration_evaluator_only",
        "target role",
    )
    _require_equal(
        targets.get("method_facing_access"), "forbidden", "target method access"
    )
    relative = str(targets["path"])
    _require_equal(relative, TARGET_RELATIVE_PATH, "delayed target path")
    path = _safe_cache_file(root, relative, label="outer evaluator targets")
    files = _require_mapping(manifest.get("files"), "cache files")
    file_record = _require_mapping(files.get(relative), "target file record")
    expected_bytes = _strict_int(file_record.get("bytes"), "target bytes")
    _require_equal(path.stat().st_size, expected_bytes, "target bytes")
    _require_equal(sha256_file(path), file_record["sha256"], "target file SHA256")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        values.shape != (64, 1, 256, 256)
        or values.dtype.str != "<f4"
        or not values.flags.c_contiguous
        or values.flags.writeable
    ):
        raise ValueError(
            f"outer-evaluator target tensor contract drift: "
            f"{values.shape}/{values.dtype}/{values.flags.c_contiguous}/"
            f"writeable={values.flags.writeable}"
        )
    return values


def _require_float32_c(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{label} shape mismatch: expected {shape}, got {array.shape}")
    if array.dtype != np.dtype("float32"):
        raise TypeError(f"{label} must be float32, got {array.dtype}")
    array = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    if array.dtype.str != "<f4" or not array.flags.c_contiguous:
        raise RuntimeError(f"{label} is not little-endian contiguous float32")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN/Inf")
    return array


def _materialize_staging(context: Mapping[str, Any], staging: Path) -> dict[str, Any]:
    started = time.perf_counter()
    selected_ids = tuple(str(value) for value in context["selected_ids"])
    selected_indices = tuple(int(value) for value in context["selected_indices"])
    conditions = tuple(context["conditions"])
    protocol = _require_mapping(context["protocol"], "protocol")
    image_size = int(protocol["preprocessing"]["image_resize"]["size"][0])
    _require_equal(image_size, 256, "materialization image size")
    expected_input_hashes = _require_mapping(
        context["expected_condition_tensor_sequence_sha256"], "expected input hashes"
    )
    expected_gt_hash = str(context["expected_gt_tensor_sequence_sha256"])
    seed = int(protocol["input_protocol"]["seed"])
    guard = pilot_runner.PilotIOGuard(context["selected_files"])

    files: dict[str, dict[str, Any]] = {}
    condition_records: list[dict[str, Any]] = []
    original_sizes: list[list[int]] = []
    reference_gt_hash: str | None = None
    target_tensor_hash: str | None = None
    target_relative = TARGET_RELATIVE_PATH
    target_path = staging / target_relative

    for condition_index, (corruption, severity) in enumerate(conditions):
        key = condition_key(str(corruption), int(severity))
        destination = staging / "conditions" / f"{key}.npy"
        image_partial, image_map = _write_memmap_atomic(
            destination, shape=(64, 3, image_size, image_size)
        )
        target_partial: Path | None = None
        target_map: np.memmap | None = None
        if condition_index == 0:
            target_partial, target_map = _write_memmap_atomic(
                target_path, shape=(64, 1, image_size, image_size)
            )

        dataset = IRSTDResearchDataset(
            context["dataset_root"],
            split_file=context["train_split"],
            image_size=image_size,
            dataset_name=str(context["dataset_name"]),
            corruption=str(corruption),
            severity=int(severity),
            seed=seed,
            corruption_transform=None if corruption == "clean" else apply_corruption,
            io_access_guard=guard,
        )
        image_hasher = TensorSequenceHasher()
        mask_hasher = TensorSequenceHasher()
        condition_started = time.perf_counter()
        observed_ids: list[str] = []
        for selected_position, dataset_index in enumerate(selected_indices):
            sample = dataset[dataset_index]
            image_id = str(sample["image_id"])
            _require_equal(
                image_id,
                selected_ids[selected_position],
                f"materialized {key} image ID at {selected_position}",
            )
            image = _require_float32_c(
                sample["image"].detach().cpu().numpy(),
                (3, image_size, image_size),
                f"{key} image {image_id}",
            )
            mask = _require_float32_c(
                sample["mask"].detach().cpu().numpy(),
                (1, image_size, image_size),
                f"{key} target {image_id}",
            )
            image_map[selected_position] = image
            image_hasher.update(image_id, image)
            mask_hasher.update(image_id, mask)
            observed_ids.append(image_id)
            if target_map is not None:
                target_map[selected_position] = mask
                original_sizes.append([int(value) for value in sample["original_size"]])
            # No model, adaptation callback, or method-facing object is invoked.
            del mask

        _require_equal(tuple(observed_ids), selected_ids, f"{key} selected ID order")
        image_map.flush()
        del image_map
        os.replace(image_partial, destination)
        generated_input_hash = image_hasher.hexdigest()
        generated_gt_hash = mask_hasher.hexdigest()
        _require_equal(
            generated_input_hash,
            str(expected_input_hashes[key]),
            f"{key} versus parent Pilot input tensor sequence SHA256",
        )
        _require_equal(
            generated_gt_hash,
            expected_gt_hash,
            f"{key} versus parent Pilot GT tensor sequence SHA256",
        )

        stored_input_hash = _hash_memmap_records(destination, selected_ids)
        _require_equal(
            stored_input_hash,
            generated_input_hash,
            f"stored {key} tensor sequence SHA256",
        )
        file_hash = sha256_file(destination)
        relative = str(destination.relative_to(staging))
        files[relative] = {"sha256": file_hash, "bytes": destination.stat().st_size}

        if reference_gt_hash is None:
            reference_gt_hash = generated_gt_hash
            if target_map is None or target_partial is None:
                raise RuntimeError("first condition did not allocate target cache")
            target_map.flush()
            del target_map
            os.replace(target_partial, target_path)
            target_tensor_hash = _hash_memmap_records(target_path, selected_ids)
            _require_equal(
                target_tensor_hash,
                expected_gt_hash,
                "stored target tensor sequence SHA256",
            )
            target_file_hash = sha256_file(target_path)
            files[target_relative] = {
                "sha256": target_file_hash,
                "bytes": target_path.stat().st_size,
            }
        else:
            _require_equal(generated_gt_hash, reference_gt_hash, f"{key} GT invariance")

        condition_records.append(
            {
                "index": condition_index,
                "key": key,
                "corruption": str(corruption),
                "severity": int(severity),
                "path": relative,
                "shape": [64, 3, 256, 256],
                "dtype": "little_endian_float32",
                "contiguous_order": "C",
                "tensor_sequence_sha256": stored_input_hash,
                "parent_pilot_tensor_sequence_sha256": str(expected_input_hashes[key]),
                "gt_mask_tensor_sequence_sha256": generated_gt_hash,
                "file_sha256": file_hash,
                "stored_tensor_matches_generated": True,
                "matches_parent_round_02_pilot": True,
                "runtime_seconds": time.perf_counter() - condition_started,
            }
        )
        _assert_context_runtime_seal(context, stage=f"after_condition:{key}")

    if target_tensor_hash is None or reference_gt_hash is None:
        raise RuntimeError("target cache was not materialized")
    io_summary = guard.summary()
    expected_opens = 64 * len(conditions)
    for key, expected in (
        ("allowed_unique_ids", 64),
        ("opened_unique_ids", 64),
        ("image_open_count", expected_opens),
        ("mask_open_count", expected_opens),
        ("forbidden_open_count", 0),
    ):
        _require_equal(io_summary.get(key), expected, f"I/O guard {key}")

    content_payload = {
        "protocol_sha256": context["protocol_sha256"],
        "dataset": context["dataset_name"],
        "ordered_ids_sha256": ordered_ids_sha256(selected_ids),
        "targets_sha256": files[target_relative]["sha256"],
        "conditions": [
            [record["key"], record["file_sha256"]] for record in condition_records
        ],
    }
    content_hash = hashlib.sha256(
        json.dumps(content_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "cache_format": protocol["materialized_cache"]["cache_format"],
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(context["protocol_path"]),
        "protocol_sha256": context["protocol_sha256"],
        "repository_provenance": source_runner.repository_provenance(PROVENANCE_PATHS),
        "runtime_seal": dict(context["runtime_seal"]),
        "runtime_seal_sha256": context["runtime_seal"]["runtime_seal_sha256"],
        "dataset": context["dataset_name"],
        "dataset_root": str(context["dataset_root"]),
        "split_role": "fixed_train_derived_sha256_ranked_64",
        "train_split": str(context["train_split"]),
        "train_split_sha256": context["dataset_contract"]["train_split_sha256"],
        "test_split_metadata": str(context["test_split"]),
        "test_split_sha256": context["dataset_contract"]["test_split_sha256"],
        "source_manifests": dict(context["source_manifests"]),
        "source_file_manifest_sha256": context["source_manifests"][
            "combined_sha256"
        ],
        "checkpoint": str(context["checkpoint"]),
        "checkpoint_sha256": context["dataset_contract"]["checkpoint_sha256"],
        "checkpoint_metadata": dict(context["checkpoint_summary"]),
        "seed": seed,
        "image_ids": list(selected_ids),
        "ordered_ids_sha256": ordered_ids_sha256(selected_ids),
        "original_sizes": original_sizes,
        "condition_count": len(condition_records),
        "conditions": condition_records,
        "targets": {
            "path": target_relative,
            "shape": [64, 1, 256, 256],
            "dtype": "little_endian_float32",
            "contiguous_order": "C",
            "tensor_sequence_sha256": target_tensor_hash,
            "parent_pilot_tensor_sequence_sha256": expected_gt_hash,
            "file_sha256": files[target_relative]["sha256"],
            "role": "outer_source_side_calibration_evaluator_only",
            "method_facing_access": "forbidden",
        },
        "files": files,
        "io_guard": io_summary,
        "fixed_test_boundary": {
            "test_split_metadata_read_for_leakage_guard": True,
            "test_dataset_constructed": False,
            "test_images_opened": 0,
            "test_masks_opened": 0,
        },
        "label_firewall": {
            "adaptation_interface_invoked": False,
            "method_received_labels": False,
            "targets_written_for_outer_evaluator_only": True,
            "target_transition_hyperparameter_selection_allowed": False,
            "official_method_facing_consumer": (
                "SourceCalibrationMethodInputDataset"
            ),
            "method_facing_sample_fields": [
                "image",
                "image_id",
                "original_size",
                "dataset",
                "corruption",
                "severity",
                "seed",
            ],
            "outer_evaluator_target_loader": "load_outer_evaluator_targets",
            "outer_evaluator_requires_episodes_complete": True,
            "generic_cached_corruption_dataset_for_adaptation": "forbidden",
            "contract_evidence": _label_firewall_contract_evidence(),
        },
        "checks": {
            "fixed_train_derived_64": True,
            "selected_ids_absent_from_fixed_test": True,
            "all_conditions_same_ordered_ids": True,
            "all_condition_hashes_match_round_02_pilot": True,
            "gt_mask_hash_identical_across_conditions": True,
            "stored_tensors_match_generated_tensors": True,
            "corruption_before_normalization": True,
            "frozen_severity_table_verified": True,
            "test_pixel_and_label_opens_zero": True,
            "consumer_corruption_regeneration_forbidden": True,
        },
        "cache_content_sha256": content_hash,
        "runtime_seconds": time.perf_counter() - started,
    }
    return manifest


def _acquire_publish_lock(final_output: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = final_output.with_name(f".{final_output.name}.publish.lock")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"exclusive cache publish lock already exists: {lock_path}"
        ) from error
    try:
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "destination": str(final_output),
            "protocol_sha256": context["protocol_sha256"],
            "runtime_seal_sha256": context["runtime_seal"]["runtime_seal_sha256"],
        }
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        identity = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return {
        "path": lock_path,
        "device": identity.st_dev,
        "inode": identity.st_ino,
    }


def _assert_publish_lock(lock: Mapping[str, Any]) -> None:
    path = Path(lock["path"])
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_dev != lock["device"]
        or observed.st_ino != lock["inode"]
    ):
        raise RuntimeError("exclusive cache publish lock changed during materialization")


def _release_publish_lock(lock: Mapping[str, Any]) -> None:
    path = Path(lock["path"])
    if not path.exists() and not path.is_symlink():
        return
    _assert_publish_lock(lock)
    path.unlink()


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic directory publication that can never replace a destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable; refusing publish")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            f"cache destination appeared during publish; refusing overwrite: {destination}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported; refusing publish"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def materialize(context: Mapping[str, Any]) -> dict[str, Any]:
    _assert_context_runtime_seal(context, stage="materialize_start")
    final_output = Path(context["final_output"])
    if final_output.exists():
        raise FileExistsError(
            f"cache destination already exists; refusing overwrite: {final_output}"
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    publish_lock = _acquire_publish_lock(final_output, context)
    staging = final_output.with_name(f".{final_output.name}.build-{os.getpid()}")
    try:
        _assert_publish_lock(publish_lock)
        if final_output.exists():
            raise FileExistsError(
                f"cache destination appeared before build; refusing overwrite: {final_output}"
            )
        if staging.exists():
            raise FileExistsError(f"cache staging directory already exists: {staging}")
        staging.mkdir(parents=False)
        manifest = _materialize_staging(context, staging)
        _assert_context_runtime_seal(context, stage="pre_manifest_write")
        manifest_path = staging / "manifest.json"
        source_runner.write_json_atomic(manifest_path, manifest)
        manifest_sha256 = sha256_file(manifest_path)
        method_manifest = _build_method_input_manifest(
            manifest, outer_manifest_sha256=manifest_sha256
        )
        method_manifest_path = staging / METHOD_INPUT_MANIFEST_NAME
        source_runner.write_json_atomic(method_manifest_path, method_manifest)
        firewall_evidence = _require_mapping(
            method_manifest.get("label_firewall_contract"),
            "method-facing firewall evidence",
        )
        firewall_verified = firewall_evidence.get("verified")
        _require_bool(firewall_verified, True, "method-facing firewall self-test")
        firewall_evidence_sha256 = hashlib.sha256(
            json.dumps(
                firewall_evidence, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        completion = {
            "complete": True,
            "dataset": context["dataset_name"],
            "protocol_sha256": context["protocol_sha256"],
            "cache_content_sha256": manifest["cache_content_sha256"],
            "manifest_sha256": manifest_sha256,
            "method_input_manifest_sha256": sha256_file(method_manifest_path),
            "runtime_seal_sha256": context["runtime_seal"][
                "runtime_seal_sha256"
            ],
            "label_firewall_contract": firewall_evidence["contract_id"],
            "label_firewall_evidence_sha256": firewall_evidence_sha256,
            "label_firewall_verified": firewall_verified,
            "adaptation_interface_invoked": False,
            "test_images_opened": 0,
            "test_masks_opened": 0,
        }
        source_runner.write_json_atomic(staging / "COMPLETE.json", completion)
        _assert_context_runtime_seal(context, stage="pre_atomic_publish")
        _assert_publish_lock(publish_lock)
        if final_output.exists():
            raise FileExistsError(
                f"cache destination appeared during build; refusing overwrite: {final_output}"
            )
        _atomic_rename_directory_noreplace(staging, final_output)
    except BaseException:
        if staging.is_dir() and staging.parent == final_output.parent:
            shutil.rmtree(staging)
        raise
    finally:
        _release_publish_lock(publish_lock)
    manifest["published_output_dir"] = str(final_output)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    process_entry_seal = capture_runtime_seal(args.protocol)
    validate_only = bool(args.validate_only)
    context = validate_contract(
        protocol_path=args.protocol,
        dataset_name=args.dataset,
        output_override=args.output_dir,
        require_output_absent=True,
        runtime_seal=process_entry_seal,
        verify_source_file_bytes=not validate_only,
    )
    if validate_only:
        assert_runtime_seal(process_entry_seal, stage="validate_only_return")
        return {
            "validate_only": True,
            "protocol_sha256": context["protocol_sha256"],
            "dataset": context["dataset_name"],
            "selected_count": len(context["selected_ids"]),
            "selected_ids_sha256": ordered_ids_sha256(context["selected_ids"]),
            "condition_count": len(context["conditions"]),
            "expected_condition_tensor_sequence_sha256": dict(
                context["expected_condition_tensor_sequence_sha256"]
            ),
            "expected_gt_tensor_sequence_sha256": context[
                "expected_gt_tensor_sequence_sha256"
            ],
            "planned_output_dir": str(context["final_output"]),
            "output_created": False,
            "runtime_seal": dict(process_entry_seal),
            "runtime_seal_sha256": process_entry_seal["runtime_seal_sha256"],
            "validation": dict(context["validation"]),
        }
    return materialize(context)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    if result.get("validate_only") is True:
        print(
            f"{result['dataset']}: validated {result['selected_count']} fixed "
            f"train-derived IDs and {result['condition_count']} condition hashes; "
            "decoded 0 dataset pixel arrays, read 0 dataset image/mask bytes, "
            "and created no output"
        )
    else:
        print(
            f"{result['dataset']}: materialized {result['condition_count']} conditions "
            f"for {len(result['image_ids'])} fixed train-derived images in "
            f"{result['runtime_seconds']:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
