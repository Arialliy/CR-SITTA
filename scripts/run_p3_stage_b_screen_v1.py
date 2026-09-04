#!/usr/bin/env python3
"""Run the train-only CR-SITTA Stage-B3 objective/space screen.

The method-facing candidate phase can read only the frozen Pilot64 image
cache and the already sealed, label-free multi-view teacher artifact.  The
outer phase is a separate command and opens source-train targets only after a
complete candidate artifact has been verified.  Validation/test payloads are
never accepted by this runner.

Formal artifacts are immutable directories published with rename-no-replace.
An engineering ``smoke`` run always writes below ``engineering_dry_runs`` and
cannot create a formal completion receipt.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

PROTOCOL_ID = "cr-sitta-p3-stage-b3-objective-space-screen-v1"
FROZEN_CONFIG_SHA256 = (
    "6a596b472e6761be17bebab6888402d6f3ef050946edb3f345a92be5f1ad7666"
)
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CONDITIONS = (
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
OBJECTIVES = ("O0", "O1", "O2", "O3", "O4")
SPACES = ("P2", "DecoderFiLM")
CANDIDATES = tuple(
    f"{objective}_{space}"
    for objective in OBJECTIVES
    for space in SPACES
)
PILOT16_COUNT = 16
SOURCE_SHAPE = (1, 256, 256)
THRESHOLD = 0.5
SHA256_HEX = frozenset("0123456789abcdef")


class StageB3ProtocolError(RuntimeError):
    """The frozen Stage-B3 protocol or one of its artifacts is invalid."""


class ExistingArtifactError(StageB3ProtocolError):
    """A formal destination exists but cannot be verified as complete."""


@dataclass(frozen=True)
class ScreenContract:
    repository: Path
    config_path: Path
    config_sha256: str
    raw: Mapping[str, Any]

    @property
    def output_root(self) -> Path:
        return self.repository / str(self.raw["output"]["root"])

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(str(item["id"]) for item in self.raw["candidates"])


def _condition_key(corruption: str, severity: int) -> str:
    return "clean_S0" if corruption == "clean" else f"{corruption}_S{severity}"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    if path.is_symlink() or not path.is_file():
        raise StageB3ProtocolError(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise StageB3ProtocolError(f"{label} must be lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageB3ProtocolError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageB3ProtocolError(f"{label} must be a sequence")
    return value


def _repository_path(repository: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise StageB3ProtocolError(f"{label} must be a non-empty path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise StageB3ProtocolError(f"{label} must be repository-relative")
    resolved = repository / relative
    if not resolved.absolute().is_relative_to(repository):
        raise StageB3ProtocolError(f"{label} escapes repository")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageB3ProtocolError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StageB3ProtocolError(f"JSON root must be a mapping: {path}")
    return value


def _hash_selected_ids(dataset: str, values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            values,
            key=lambda image_id: (
                hashlib.sha256(
                    f"cr-sitta-stage-b3-v1\0{dataset}\0{image_id}".encode()
                ).hexdigest(),
                image_id,
            ),
        )[:PILOT16_COUNT]
    )


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _verify_parent_bindings(repository: Path, config: Mapping[str, Any]) -> None:
    parents = _mapping(config.get("frozen_parent_bindings"), "parent bindings")
    for name, raw_record in parents.items():
        record = _mapping(raw_record, f"parent binding {name}")
        if set(record) != {"path", "sha256"}:
            raise StageB3ProtocolError(f"parent binding {name} fields drifted")
        path = _repository_path(repository, record["path"], f"parent {name}")
        expected = _require_sha256(record["sha256"], f"parent {name} sha256")
        if sha256_file(path) != expected:
            raise StageB3ProtocolError(f"frozen parent changed: {name}")


def _verify_teacher_dataset_binding(
    repository: Path,
    dataset: str,
    record: Mapping[str, Any],
) -> None:
    teacher_root = _repository_path(
        repository, record["teacher_artifact_root"], f"{dataset} teacher root"
    )
    manifest_path = teacher_root / "manifest.json"
    complete_path = teacher_root / "COMPLETE.json"
    if sha256_file(manifest_path) != _require_sha256(
        record["teacher_manifest_sha256"], f"{dataset} teacher manifest"
    ):
        raise StageB3ProtocolError(f"teacher manifest changed: {dataset}")
    if sha256_file(complete_path) != _require_sha256(
        record["teacher_complete_sha256"], f"{dataset} teacher COMPLETE"
    ):
        raise StageB3ProtocolError(f"teacher completion changed: {dataset}")
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    if (
        manifest.get("protocol_id") != "cr-sitta-nonadaptive-teacher-screen-v1"
        or manifest.get("phase") != "candidate"
        or manifest.get("dataset") != dataset
        or manifest.get("formal") is not True
        or manifest.get("paper_result") is not False
        or int(manifest.get("image_count_per_condition", -1)) != 64
        or int(manifest.get("condition_count", -1)) != 13
    ):
        raise StageB3ProtocolError(f"teacher manifest semantics differ: {dataset}")
    if (
        complete.get("complete") is not True
        or complete.get("protocol_id")
        != "cr-sitta-nonadaptive-teacher-screen-v1"
        or complete.get("manifest_sha256") != record["teacher_manifest_sha256"]
    ):
        raise StageB3ProtocolError(f"teacher completion semantics differ: {dataset}")
    all_ids = tuple(str(value) for value in manifest.get("image_ids", ()))
    selected = tuple(str(value) for value in record["selected_image_ids"])
    if len(all_ids) != 64 or len(set(all_ids)) != 64:
        raise StageB3ProtocolError(f"teacher Pilot64 IDs are invalid: {dataset}")
    if selected != _hash_selected_ids(dataset, all_ids):
        raise StageB3ProtocolError(f"Pilot16 hash selection differs: {dataset}")
    if _ordered_ids_sha256(selected) != record["selected_image_ids_sha256"]:
        raise StageB3ProtocolError(f"Pilot16 ordered ID SHA differs: {dataset}")


def _verify_cache_dataset_binding(
    repository: Path,
    dataset: str,
    record: Mapping[str, Any],
    *,
    expected_protocol_sha256: str,
) -> None:
    cache_root = _repository_path(
        repository, record["cache_root"], f"{dataset} cache root"
    )
    manifest_path = cache_root / "manifest.json"
    complete_path = cache_root / "COMPLETE.json"
    expected_manifest_sha256 = _require_sha256(
        record["cache_manifest_sha256"], f"{dataset} cache manifest SHA"
    )
    expected_content_sha256 = _require_sha256(
        record["cache_content_sha256"], f"{dataset} cache content SHA"
    )
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise StageB3ProtocolError(f"frozen cache manifest changed: {dataset}")
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    if (
        manifest.get("protocol_id")
        != "cr-sitta-binary-tent-ss-calibration-cache-v2"
        or manifest.get("protocol_sha256") != expected_protocol_sha256
        or manifest.get("dataset") != dataset
        or manifest.get("split_role") != "train_side_pilot_v2_derived_64"
        or manifest.get("train_split_sha256") != record["train_split_sha256"]
        or manifest.get("cache_content_sha256") != expected_content_sha256
        or manifest.get("seed") != 42
        or len(manifest.get("image_ids", ())) != 64
    ):
        raise StageB3ProtocolError(f"frozen cache semantics differ: {dataset}")
    source_scope = _mapping(
        manifest.get("source_open_scope"), f"{dataset} source-open scope"
    )
    if (
        source_scope.get("test_images") != 0
        or source_scope.get("test_masks") != 0
        or source_scope.get("unique_train_images") != 64
        or source_scope.get("unique_train_masks") != 64
    ):
        raise StageB3ProtocolError(f"cache train/test boundary differs: {dataset}")
    if (
        complete.get("complete") is not True
        or complete.get("dataset") != dataset
        or complete.get("manifest_sha256") != expected_manifest_sha256
        or complete.get("cache_content_sha256") != expected_content_sha256
        or complete.get("protocol_sha256") != expected_protocol_sha256
        or complete.get("method_received_labels") is not False
        or complete.get("test_images_opened") != 0
        or complete.get("test_masks_opened") != 0
    ):
        raise StageB3ProtocolError(f"cache completion semantics differ: {dataset}")


def _validate_contract_semantics(repository: Path, config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("protocol_id") != PROTOCOL_ID:
        raise StageB3ProtocolError("Stage-B3 schema/protocol differs")
    scope = _mapping(config.get("scope"), "scope")
    required_scope = {
        "source_train_derived": True,
        "split_name": "train",
        "image_count_per_condition": 16,
        "dataset_count": 3,
        "condition_count_per_dataset": 13,
        "development_only": True,
        "paper_result": False,
        "paper_test_result": False,
        "use_validation_payload": False,
        "use_test_payload": False,
        "method_label_accesses": 0,
        "adaptation_gradient_uses_labels": False,
        "outer_train_target_access_after_candidate_completion_only": True,
    }
    for key, expected in required_scope.items():
        if scope.get(key) != expected:
            raise StageB3ProtocolError(f"scope.{key} must be exactly {expected!r}")
    transition = _mapping(config.get("stage_transition"), "stage transition")
    for key in (
        "stage_b1_protocol_complete",
        "stage_b1_does_not_authorize_or_block_b3",
        "stage_b2_protocol_complete",
        "teacher_not_claimed_better_than_source",
        "detached_teacher_uncertainty_reuse_allowed",
        "stage_b3_authorized_by_frozen_v5_design",
    ):
        if transition.get(key) is not True:
            raise StageB3ProtocolError(f"stage_transition.{key} must be true")
    if transition.get("stage_b4_allowed_before_b3_gate") is not False:
        raise StageB3ProtocolError("B4 must be forbidden before the B3 gate")
    conditions = tuple(tuple(value) for value in config.get("ordered_conditions", ()))
    if conditions != CONDITIONS:
        raise StageB3ProtocolError("ordered 13-condition grid drifted")
    datasets = _mapping(config.get("datasets"), "datasets")
    if tuple(datasets) != DATASETS:
        raise StageB3ProtocolError("dataset order/set drifted")
    candidates = tuple(str(item["id"]) for item in config.get("candidates", ()))
    if candidates != CANDIDATES:
        raise StageB3ProtocolError("objective/space candidate order drifted")
    teacher = _mapping(config.get("teacher"), "teacher")
    if (
        teacher.get("probability_role") != "sealed_source_identity"
        or teacher.get("probability_array") != "source_probabilities"
        or teacher.get("b2_candidate_selected") is not False
        or teacher.get("b2_relative_best_used_as_pseudo_target") is not False
        or teacher.get("uncertainty_axis_index") != 0
        or teacher.get("detached_required") is not True
        or teacher.get("teacher_scientific_eligibility_claimed") is not False
    ):
        raise StageB3ProtocolError("teacher reuse contract drifted")
    if _mapping(config["objectives"]["O5"], "O5").get("status") != (
        "deferred_not_silently_substituted"
    ):
        raise StageB3ProtocolError("O5 deferral must remain explicit")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    if evaluation.get("probability_threshold") != 0.5 or evaluation.get(
        "threshold_rule"
    ) != "strict_greater_than":
        raise StageB3ProtocolError("fixed threshold semantics drifted")
    output = _mapping(config.get("output"), "output")
    expected_output_paths = {
        "root": "results/cr_sitta/p3_stage_b3_objective_space_screen_v1",
        "candidate_phase": "candidate_phase/R0",
        "outer_phase": "outer_phase/R0",
        "aggregate_phase": "aggregate_phase/R0",
        "engineering_phase": "engineering_dry_runs",
    }
    if any(output.get(key) != value for key, value in expected_output_paths.items()):
        raise StageB3ProtocolError("formal output paths drifted")
    if output.get("atomic_no_replace") is not True or output.get(
        "refuse_overwrite"
    ) is not True:
        raise StageB3ProtocolError("formal outputs must be immutable")
    _verify_parent_bindings(repository, config)
    cache_protocol_sha256 = _require_sha256(
        _mapping(
            config["frozen_parent_bindings"]["cache_protocol"],
            "cache protocol binding",
        )["sha256"],
        "cache protocol SHA",
    )
    for dataset in DATASETS:
        record = _mapping(datasets[dataset], f"dataset {dataset}")
        checkpoint = _repository_path(
            repository, record["checkpoint_path"], f"{dataset} checkpoint"
        )
        if sha256_file(checkpoint) != _require_sha256(
            record["checkpoint_sha256"], f"{dataset} checkpoint sha"
        ):
            raise StageB3ProtocolError(f"checkpoint changed: {dataset}")
        _verify_cache_dataset_binding(
            repository,
            dataset,
            record,
            expected_protocol_sha256=cache_protocol_sha256,
        )
        _verify_teacher_dataset_binding(repository, dataset, record)
    implementation = _mapping(config.get("implementation"), "implementation")
    paths = tuple(implementation.get("critical_code_paths", ()))
    if not paths or len(paths) != len(set(paths)):
        raise StageB3ProtocolError("critical code path list is empty/duplicated")
    for value in paths:
        _repository_path(repository, value, "critical code path")


def load_contract(config_path: Path) -> ScreenContract:
    absolute = config_path if config_path.is_absolute() else REPOSITORY / config_path
    if absolute.is_symlink() or not absolute.is_file():
        raise StageB3ProtocolError(f"config must be a regular file: {absolute}")
    payload = absolute.read_bytes()
    payload_sha256 = _sha256_bytes(payload)
    if payload_sha256 != FROZEN_CONFIG_SHA256:
        raise StageB3ProtocolError(
            "Stage-B3 config bytes differ from the canonical frozen protocol"
        )
    try:
        raw = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise StageB3ProtocolError("Stage-B3 YAML cannot be parsed") from exc
    if not isinstance(raw, dict):
        raise StageB3ProtocolError("Stage-B3 YAML root must be a mapping")
    _validate_contract_semantics(REPOSITORY, raw)
    return ScreenContract(
        repository=REPOSITORY,
        config_path=absolute,
        config_sha256=payload_sha256,
        raw=raw,
    )


def _assert_contract_unchanged(contract: ScreenContract) -> None:
    if sha256_file(contract.config_path) != contract.config_sha256:
        raise StageB3ProtocolError("Stage-B3 config changed during execution")
    _validate_contract_semantics(contract.repository, contract.raw)


def _verify_consumed_dataset_payloads(
    contract: ScreenContract,
    dataset: str,
    *,
    include_outer_target: bool,
) -> None:
    record = contract.raw["datasets"][dataset]
    cache_root = _repository_path(
        contract.repository, record["cache_root"], f"{dataset} cache root"
    )
    cache_manifest = _load_json(cache_root / "manifest.json")
    cache_conditions = {
        str(value["key"]): value for value in cache_manifest.get("conditions", ())
    }
    expected_condition_keys = {_condition_key(*value) for value in CONDITIONS}
    if set(cache_conditions) != expected_condition_keys:
        raise StageB3ProtocolError(f"cache condition topology differs: {dataset}")
    for condition in sorted(expected_condition_keys):
        descriptor = _mapping(
            cache_conditions[condition], f"{dataset}/{condition} cache descriptor"
        )
        path = cache_root / str(descriptor["path"])
        if sha256_file(path) != descriptor.get("file_sha256"):
            raise StageB3ProtocolError(
                f"consumed cache payload changed: {dataset}/{condition}"
            )
    if include_outer_target:
        target = _mapping(cache_manifest.get("targets"), f"{dataset} target")
        if sha256_file(cache_root / str(target["path"])) != target.get("file_sha256"):
            raise StageB3ProtocolError(f"outer target payload changed: {dataset}")

    if include_outer_target:
        return
    teacher_root = _repository_path(
        contract.repository,
        record["teacher_artifact_root"],
        f"{dataset} teacher root",
    )
    teacher_manifest = _load_json(teacher_root / "manifest.json")
    teacher_ledger = _mapping(
        teacher_manifest.get("files"), f"{dataset} teacher ledger"
    )
    teacher_conditions = {
        str(value["condition"]): value
        for value in teacher_manifest.get("conditions", ())
    }
    if set(teacher_conditions) != expected_condition_keys:
        raise StageB3ProtocolError(f"teacher condition topology differs: {dataset}")
    for condition in sorted(expected_condition_keys):
        arrays = _mapping(
            teacher_conditions[condition].get("arrays"),
            f"{dataset}/{condition} teacher arrays",
        )
        for array_name in ("source_probabilities", "view_uncertainty"):
            descriptor = _mapping(
                arrays.get(array_name),
                f"{dataset}/{condition}/{array_name}",
            )
            relative = str(descriptor["path"])
            ledger_record = _mapping(
                teacher_ledger.get(relative),
                f"{dataset} teacher ledger {relative}",
            )
            if sha256_file(teacher_root / relative) != ledger_record.get("sha256"):
                raise StageB3ProtocolError(
                    f"consumed teacher payload changed: {dataset}/{condition}/{array_name}"
                )


def _capture_code_hashes(contract: ScreenContract) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in contract.raw["implementation"]["critical_code_paths"]:
        path = _repository_path(contract.repository, raw, "critical code path")
        result[str(raw)] = sha256_file(path)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for value in values:
            stream.write(_canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _file_ledger(root: Path, *, excluded: Sequence[str] = ()) -> dict[str, Any]:
    skip = frozenset(excluded)
    result: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StageB3ProtocolError(f"artifact contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in skip:
            continue
        result[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _new_staging(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(mode=0o700, exist_ok=False)
    return staging


def _publish(
    staging: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    *,
    pre_rename_guard: Any | None = None,
) -> None:
    files = _file_ledger(staging)
    final_manifest = {**dict(manifest), "files": files}
    final_manifest["payload_tree_sha256"] = _sha256_bytes(
        _canonical_json_bytes(files)
    )
    _write_json(staging / "manifest.json", final_manifest)
    _write_json(
        staging / "COMPLETE.json",
        {
            "schema_version": 1,
            "artifact_type": "cr_sitta_p3_stage_b3_completion",
            "protocol_id": PROTOCOL_ID,
            "complete": True,
            "phase": final_manifest["phase"],
            "dataset": final_manifest.get("dataset"),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "atomic_no_replace": True,
            "development_only": True,
            "paper_result": False,
            "paper_test_result": False,
        },
    )
    from tta.d0_secure_io import publish_directory_noreplace

    publish_directory_noreplace(
        staging,
        destination,
        pre_rename_guard=pre_rename_guard,
    )


def _artifact_destination(
    contract: ScreenContract, phase: str, dataset: str | None = None
) -> Path:
    relative = str(contract.raw["output"][f"{phase}_phase"])
    result = contract.output_root / relative
    return result / dataset if dataset is not None else result


def verify_artifact(
    path: Path,
    *,
    contract: ScreenContract,
    phase: str,
    dataset: str | None,
    expected_formal: bool = True,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise StageB3ProtocolError(f"artifact directory missing/unsafe: {path}")
    manifest_path = path / "manifest.json"
    complete_path = path / "COMPLETE.json"
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    artifact_types = {
        "candidate": "cr_sitta_p3_stage_b3_candidate_dataset",
        "outer": "cr_sitta_p3_stage_b3_outer_dataset",
        "aggregate": "cr_sitta_p3_stage_b3_aggregate",
    }
    if phase not in artifact_types:
        raise StageB3ProtocolError(f"unknown artifact phase: {phase}")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != artifact_types[phase]
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("config_sha256") != contract.config_sha256
        or manifest.get("phase") != phase
        or manifest.get("dataset") != dataset
        or manifest.get("formal") is not expected_formal
        or manifest.get("development_only") is not True
        or manifest.get("paper_result") is not False
        or manifest.get("paper_test_result") is not False
    ):
        raise StageB3ProtocolError(f"artifact manifest semantics differ: {path}")
    if (
        complete.get("schema_version") != 1
        or complete.get("artifact_type")
        != "cr_sitta_p3_stage_b3_completion"
        or complete.get("complete") is not True
        or complete.get("protocol_id") != PROTOCOL_ID
        or complete.get("phase") != phase
        or complete.get("dataset") != dataset
        or complete.get("manifest_sha256") != sha256_file(manifest_path)
        or complete.get("atomic_no_replace") is not True
        or complete.get("development_only") is not True
        or complete.get("paper_result") is not False
        or complete.get("paper_test_result") is not False
    ):
        raise StageB3ProtocolError(f"completion receipt differs: {path}")
    expected_files = _mapping(manifest.get("files"), "artifact files")
    actual_files = _file_ledger(
        path, excluded=("manifest.json", "COMPLETE.json")
    )
    if actual_files != expected_files:
        raise StageB3ProtocolError(f"artifact file ledger differs: {path}")
    if manifest.get("payload_tree_sha256") != _sha256_bytes(
        _canonical_json_bytes(expected_files)
    ):
        raise StageB3ProtocolError(f"artifact tree digest differs: {path}")
    if phase == "candidate":
        boundary = _mapping(manifest.get("method_boundary"), "method boundary")
        if (
            boundary.get("method_label_accesses") != 0
            or boundary.get("outer_target_loader_calls") != 0
            or boundary.get("validation_payload_opens") != 0
            or boundary.get("test_payload_opens") != 0
            or tuple(manifest.get("candidate_ids", ())) != contract.candidate_ids
            or manifest.get("teacher_probability_role")
            != "sealed_source_identity"
            or manifest.get("b2_candidate_selected") is not False
        ):
            raise StageB3ProtocolError(
                f"candidate method-boundary semantics differ: {path}"
            )
        if expected_formal and (
            manifest.get("condition_count") != len(CONDITIONS)
            or manifest.get("image_count_per_condition") != PILOT16_COUNT
            or manifest.get("episode_count")
            != len(CONDITIONS) * PILOT16_COUNT * len(CANDIDATES)
        ):
            raise StageB3ProtocolError(f"formal candidate topology differs: {path}")
    elif phase == "outer":
        if (
            manifest.get("formal") is not True
            or manifest.get("condition_count") != len(CONDITIONS)
            or manifest.get("candidate_count") != len(CANDIDATES)
            or manifest.get("image_count_per_condition") != PILOT16_COUNT
            or manifest.get("cell_summary_count")
            != len(CONDITIONS) * len(CANDIDATES)
            or manifest.get("alignment_record_count")
            != len(CONDITIONS) * PILOT16_COUNT * len(CANDIDATES)
            or manifest.get("outer_target_accesses")
            != len(CONDITIONS) * PILOT16_COUNT
            or manifest.get("method_label_accesses") != 0
            or manifest.get("adaptation_gradient_uses_labels") is not False
            or manifest.get("validation_payload_opens") != 0
            or manifest.get("test_payload_opens") != 0
        ):
            raise StageB3ProtocolError(f"formal outer semantics differ: {path}")
    else:
        receipt = _load_json(path / "science_decision_receipt.json")
        if (
            manifest.get("formal") is not True
            or manifest.get("dataset_count") != len(DATASETS)
            or manifest.get("condition_count_per_dataset") != len(CONDITIONS)
            or manifest.get("candidate_count") != len(CANDIDATES)
            or manifest.get("cell_summary_count")
            != len(DATASETS) * len(CONDITIONS) * len(CANDIDATES)
            or manifest.get("gate_input_count")
            != len(DATASETS) * len(CONDITIONS) * len(CANDIDATES)
            or manifest.get("scientific_status")
            != receipt.get("scientific_status")
            or manifest.get("stage_b4_allowed")
            is not receipt.get("stage_b4_allowed")
            or manifest.get("selected_for_stage_b4")
            != receipt.get("selected_for_stage_b4")
            or receipt.get("paper_result") is not False
            or receipt.get("test") is not False
        ):
            raise StageB3ProtocolError(f"formal aggregate semantics differ: {path}")
    return manifest


def _existing_complete_or_raise(
    destination: Path,
    *,
    contract: ScreenContract,
    phase: str,
    dataset: str | None,
) -> dict[str, Any] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    try:
        return verify_artifact(
            destination, contract=contract, phase=phase, dataset=dataset
        )
    except Exception as exc:
        raise ExistingArtifactError(
            f"refusing to overwrite incomplete/conflicting artifact: {destination}"
        ) from exc


def _method_input_dataset(
    contract: ScreenContract, dataset: str, condition: str
):
    # The target loader is deliberately not imported in the candidate path.
    from materialize_binary_tent_ss_calibration_cache_v2 import (
        SourceCalibrationMethodInputDatasetV2,
    )
    record = contract.raw["datasets"][dataset]
    cache_root = _repository_path(
        contract.repository, record["cache_root"], f"{dataset} cache root"
    )
    return SourceCalibrationMethodInputDatasetV2(
        cache_root,
        condition_key=condition,
        expected_protocol_sha256=contract.raw["frozen_parent_bindings"][
            "cache_protocol"
        ]["sha256"],
    )


def _validate_runtime_environment(device_name: str, *, seed: int) -> None:
    if not isinstance(device_name, str) or not device_name:
        raise StageB3ProtocolError("device name must be a non-empty string")
    if not device_name.startswith("cuda"):
        return
    required = {
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    mismatches = {
        key: {"expected": expected, "observed": os.environ.get(key)}
        for key, expected in required.items()
        if os.environ.get(key) != expected
    }
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(visible, str) or not visible or "," in visible:
        mismatches["CUDA_VISIBLE_DEVICES"] = {
            "expected": "exactly one physical GPU",
            "observed": visible,
        }
    if device_name not in ("cuda", "cuda:0"):
        mismatches["device"] = {
            "expected": "cuda:0 within the sole-visible-GPU process",
            "observed": device_name,
        }
    if mismatches:
        raise StageB3ProtocolError(f"CUDA environment differs: {mismatches}")


def _build_runtime(contract: ScreenContract, dataset: str, device_name: str):
    import torch
    import test_source as source_runner
    from tta.adapters import DecoderFiLM
    from tta.model_adapter import IRSTDModelAdapter
    from tta.state_manager import EpisodicStateManager, StatefulHooks

    seed = int(contract.raw["scope"]["seed"])
    _validate_runtime_environment(device_name, seed=seed)
    source_runner.seed_everything(seed)
    device = source_runner.resolve_device(device_name)
    model = source_runner.build_nsfpn_model()
    record = contract.raw["datasets"][dataset]
    checkpoint = _repository_path(
        contract.repository, record["checkpoint_path"], f"{dataset} checkpoint"
    )
    if sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise StageB3ProtocolError(f"checkpoint changed before model load: {dataset}")
    wrapper = source_runner.load_trusted_checkpoint(model, checkpoint)
    model.to(device)
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    model.zero_grad(set_to_none=True)

    film_config = contract.raw["parameter_spaces"]["DecoderFiLM"]
    film = DecoderFiLM(
        channels=int(film_config["channels"]),
        max_scale_delta=float(film_config["max_scale_delta"]),
        max_bias=float(film_config["max_bias"]),
    ).to(device)

    def snapshot_film() -> dict[str, Any]:
        return {
            "state": {
                name: tensor.detach().clone()
                for name, tensor in film.state_dict().items()
            },
            "training": tuple(
                (name, bool(module.training))
                for name, module in film.named_modules()
            ),
            "requires_grad": tuple(
                (name, bool(parameter.requires_grad))
                for name, parameter in film.named_parameters()
            ),
            "gradients": tuple(
                (
                    name,
                    None if parameter.grad is None else parameter.grad.detach().clone(),
                )
                for name, parameter in film.named_parameters()
            ),
        }

    def restore_film(value: Mapping[str, Any]) -> None:
        film.load_state_dict(value["state"], strict=True)
        modules = dict(film.named_modules())
        for name, training in value["training"]:
            modules[name].training = bool(training)
        parameters = dict(film.named_parameters())
        for name, requires_grad in value["requires_grad"]:
            parameters[name].requires_grad_(bool(requires_grad))
        for name, gradient in value["gradients"]:
            parameters[name].grad = (
                None if gradient is None else gradient.detach().clone()
            )

    state_manager = EpisodicStateManager(
        model,
        extra_stateful={
            "decoder_film": StatefulHooks(
                snapshot=snapshot_film,
                restore=restore_film,
            )
        },
    )
    state_manager.assert_source_state()
    return source_runner, model, adapter, film, state_manager, device, wrapper


def _runtime_environment_receipt(torch: Any, device: Any) -> dict[str, Any]:
    extension = importlib.import_module("MultiScaleDeformableAttention")
    raw_extension_path = getattr(extension, "__file__", None)
    if not isinstance(raw_extension_path, str) or not raw_extension_path:
        raise StageB3ProtocolError("SFS extension has no auditable file path")
    extension_path = Path(os.path.abspath(raw_extension_path))
    if extension_path.is_symlink() or not extension_path.is_file():
        raise StageB3ProtocolError(
            f"SFS extension path is missing/unsafe: {extension_path}"
        )
    return {
        "python": sys.version.split()[0],
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "cuda_runtime": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn": (
            None
            if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version())
        ),
        "device": str(device),
        "gpu_name": (
            str(torch.cuda.get_device_name(device))
            if getattr(device, "type", None) == "cuda"
            else None
        ),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sfs_extension_file": str(extension_path),
        "sfs_extension_sha256": sha256_file(extension_path),
    }


def _configure_parameter_space(
    model: Any,
    adapter: Any,
    film: Any,
    space: str,
) -> tuple[tuple[str, Any], ...]:
    from tta.parameter_groups import PILOT_GROUP_SPECS, collect_adaptable_params

    adapter.set_source_eval_mode()
    model.zero_grad(set_to_none=True)
    film.zero_grad(set_to_none=True)
    if space == "P2":
        parameters, names = collect_adaptable_params(
            model, group_spec=PILOT_GROUP_SPECS["P2"]
        )
        for parameter in parameters:
            parameter.requires_grad_(True)
        named = tuple(zip(names, parameters, strict=True))
        if len(named) != 16 or sum(p.numel() for _, p in named) != 416:
            raise StageB3ProtocolError("P2 parameter topology differs")
        return named
    if space == "DecoderFiLM":
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise StageB3ProtocolError("Source model must remain frozen for FiLM")
        named = tuple((f"decoder_film.{name}", parameter) for name, parameter in film.named_parameters())
        if len(named) != 2 or sum(p.numel() for _, p in named) != 32:
            raise StageB3ProtocolError("DecoderFiLM parameter topology differs")
        return named
    raise StageB3ProtocolError(f"unknown parameter space: {space}")


@contextmanager
def _space_forward_context(model: Any, film: Any, space: str):
    if space == "DecoderFiLM":
        from tta.adapters import decoder0_modulation

        with decoder0_modulation(model, film):
            yield
        return
    if space == "P2":
        yield
        return
    raise StageB3ProtocolError(f"unknown parameter space: {space}")


def _forward_logits(adapter: Any, model: Any, film: Any, image: Any, space: str):
    with _space_forward_context(model, film, space):
        logits = adapter.forward_logits(image)
    if tuple(logits.shape) != (1, 1, 256, 256):
        raise StageB3ProtocolError(f"student logits shape differs: {tuple(logits.shape)}")
    return logits


def _teacher_manifest(contract: ScreenContract, dataset: str) -> dict[str, Any]:
    record = contract.raw["datasets"][dataset]
    root = _repository_path(
        contract.repository, record["teacher_artifact_root"], f"{dataset} teacher"
    )
    return _load_json(root / "manifest.json")


def _teacher_condition_record(
    manifest: Mapping[str, Any], condition: str
) -> Mapping[str, Any]:
    matches = [
        item for item in manifest.get("conditions", ())
        if item.get("condition") == condition
    ]
    if len(matches) != 1:
        raise StageB3ProtocolError(f"teacher condition is not unique: {condition}")
    return matches[0]


def _verified_teacher_array(
    contract: ScreenContract,
    dataset: str,
    manifest: Mapping[str, Any],
    condition_record: Mapping[str, Any],
    array_name: str,
):
    root = _repository_path(
        contract.repository,
        contract.raw["datasets"][dataset]["teacher_artifact_root"],
        f"{dataset} teacher root",
    )
    descriptor = _mapping(
        _mapping(condition_record.get("arrays"), "teacher arrays").get(array_name),
        f"teacher {array_name}",
    )
    relative = str(descriptor.get("path"))
    path = root / relative
    ledger = _mapping(manifest.get("files"), "teacher file ledger")
    file_record = _mapping(ledger.get(relative), f"teacher ledger {relative}")
    if sha256_file(path) != file_record.get("sha256"):
        raise StageB3ProtocolError(f"teacher array changed: {dataset}/{relative}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if bool(value.flags.writeable):
        raise StageB3ProtocolError(f"teacher array is writable: {path}")
    return value


def _selected_indices(
    contract: ScreenContract, dataset: str, teacher_manifest: Mapping[str, Any]
) -> tuple[int, ...]:
    all_ids = tuple(str(value) for value in teacher_manifest["image_ids"])
    index = {image_id: position for position, image_id in enumerate(all_ids)}
    selected = tuple(
        str(value)
        for value in contract.raw["datasets"][dataset]["selected_image_ids"]
    )
    try:
        return tuple(index[value] for value in selected)
    except KeyError as exc:
        raise StageB3ProtocolError("selected Pilot16 ID is absent from Pilot64") from exc


def _raw_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _flatten_named(
    named_parameters: Sequence[tuple[str, Any]],
    tensors: Sequence[Any],
):
    import torch

    if len(named_parameters) != len(tensors):
        raise StageB3ProtocolError("named parameter/gradient lengths differ")
    values = []
    for (name, parameter), tensor in zip(named_parameters, tensors, strict=True):
        if tensor is None or tuple(tensor.shape) != tuple(parameter.shape):
            raise StageB3ProtocolError(f"gradient is missing/malformed: {name}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise StageB3ProtocolError(f"gradient contains NaN/Inf: {name}")
        values.append(tensor.detach().reshape(-1))
    return torch.cat(values)


def _unflatten_like(vector: Any, named_parameters: Sequence[tuple[str, Any]]):
    values = []
    offset = 0
    for _name, parameter in named_parameters:
        count = int(parameter.numel())
        values.append(vector[offset : offset + count].view_as(parameter))
        offset += count
    if offset != int(vector.numel()):
        raise StageB3ProtocolError("proposal vector length differs")
    return tuple(values)


def _normalized_step(
    contract: ScreenContract,
    space: str,
    named_parameters: Sequence[tuple[str, Any]],
    gradient_vector: Any,
):
    import torch

    gradient_norm = torch.linalg.vector_norm(gradient_vector)
    if not bool(torch.isfinite(gradient_norm).item()) or float(gradient_norm.item()) <= 0.0:
        return torch.zeros_like(gradient_vector), float(gradient_norm.item()), 0.0
    space_config = contract.raw["parameter_spaces"][space]
    trust_radius = float(space_config["trust_radius"])
    if space == "P2":
        source = torch.cat(
            [parameter.detach().reshape(-1) for _name, parameter in named_parameters]
        )
        source_norm = torch.linalg.vector_norm(source)
        radius = trust_radius * (source_norm + 1.0e-12)
    else:
        radius = gradient_norm.new_tensor(trust_radius)
    # B3 is a direction-quality screen, so every non-zero gradient receives
    # the same frozen trust-radius norm.  The min-clipped B4 proposal is a
    # different protocol and lives in tta.proposal_step.
    scale = radius / (gradient_norm + 1.0e-12)
    step = -gradient_vector * scale
    step_norm = float(torch.linalg.vector_norm(step).item())
    return step.detach(), float(gradient_norm.item()), step_norm


def _apply_vector(
    named_parameters: Sequence[tuple[str, Any]], vector: Any
) -> None:
    import torch

    values = _unflatten_like(vector, named_parameters)
    with torch.no_grad():
        for (_name, parameter), value in zip(named_parameters, values, strict=True):
            parameter.add_(value)


def _restore_values(
    named_parameters: Sequence[tuple[str, Any]], source_values: Sequence[Any]
) -> None:
    import torch

    with torch.no_grad():
        for (_name, parameter), source in zip(
            named_parameters, source_values, strict=True
        ):
            parameter.copy_(source)


def _build_region_weights(contract: ScreenContract, teacher: Any, uncertainty: Any):
    from tta.views import build_detached_region_weights

    view_path = _repository_path(
        contract.repository,
        contract.raw["view_library"]["config_path"],
        "view library",
    )
    view_config = yaml.safe_load(view_path.read_text(encoding="utf-8"))
    region = view_config["region_weights"]
    return build_detached_region_weights(
        teacher,
        uncertainty,
        tau_background=float(region["tau_background"]),
        tau_foreground=float(region["tau_foreground"]),
        gamma_foreground=float(region["gamma_foreground"]),
        gamma_background=float(region["gamma_background"]),
        temperature_foreground=float(region["temperature_foreground"]),
        temperature_background=float(region["temperature_background"]),
        protection_radius=int(region["protection_radius_pixels"]),
    )


def _objective_loss(
    contract: ScreenContract,
    objective: str,
    logits: Any,
    teacher: Any,
    region_weights: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from tta.objectives import (
        balanced_binary_entropy,
        bernoulli_entropy_map,
        foreground_mass_guard_from_logits,
        foreground_soft_dice_anchor,
        region_balanced_consistency,
        reliable_background_soft_bce,
    )

    eps = float(contract.raw["objectives"]["probability_eps"])
    if objective == "O0":
        loss = bernoulli_entropy_map(
            logits, eps=float(contract.raw["objectives"]["entropy_eps"])
        ).mean()
        terms = {"global_entropy": loss}
    elif objective == "O1":
        output = balanced_binary_entropy(
            logits,
            region_weights.foreground_weight,
            region_weights.background_weight,
            eps=eps,
        )
        loss = output.total
        terms = {
            "balanced_entropy_foreground": output.foreground,
            "balanced_entropy_background": output.background,
        }
    else:
        consistency = region_balanced_consistency(
            logits,
            teacher,
            region_weights.foreground_weight,
            region_weights.background_weight,
            divergence="bce",
            eps=eps,
        )
        if objective == "O2":
            loss = consistency.total
            terms = {
                "multiview_foreground": consistency.foreground,
                "multiview_background": consistency.background,
            }
        elif objective in ("O3", "O4"):
            foreground = foreground_soft_dice_anchor(
                torch.sigmoid(logits),
                teacher,
                region_weights.foreground_weight,
                eps,
            )
            background = reliable_background_soft_bce(
                logits,
                teacher,
                region_weights.background_weight,
                eps=eps,
            )
            weights = contract.raw["objectives"][objective]
            loss = (
                float(weights["lambda_multiview"]) * consistency.total
                + float(weights["lambda_foreground"]) * foreground
                + float(weights["lambda_background"]) * background
            )
            terms = {
                "multiview": consistency.total,
                "foreground_soft_dice": foreground,
                "background_soft_bce": background,
            }
            if objective == "O4":
                mass = foreground_mass_guard_from_logits(
                    logits,
                    teacher,
                    region_weights.background_weight,
                    margin=float(weights["mass_margin"]),
                    eps=eps,
                )
                loss = loss + float(weights["lambda_mass"]) * mass
                terms["foreground_mass_guard"] = mass
        else:
            raise StageB3ProtocolError(f"unknown objective: {objective}")
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise StageB3ProtocolError(f"objective {objective} is not a finite scalar")
    return loss, {
        name: float(value.detach().item())
        for name, value in terms.items()
    }


def _probability_diagnostics(
    source_logits: Any,
    post_logits: Any,
    teacher: Any,
    background_weight: Any,
) -> dict[str, Any]:
    import torch

    source_probability = torch.sigmoid(source_logits)
    post_probability = torch.sigmoid(post_logits)
    delta_logits = post_logits - source_logits
    delta_probability = post_probability - source_probability
    source_mask = source_probability > THRESHOLD
    post_mask = post_probability > THRESHOLD
    background_sum = background_weight.sum()
    if float(background_sum.item()) <= 0.0:
        raise StageB3ProtocolError("reliable-background weight is empty")
    teacher_mass = (teacher * background_weight).sum() / (background_sum + 1.0e-6)
    source_mass = (source_probability * background_weight).sum() / (
        background_sum + 1.0e-6
    )
    post_mass = (post_probability * background_weight).sum() / (
        background_sum + 1.0e-6
    )
    return {
        "max_abs_delta_logit": float(delta_logits.abs().max().item()),
        "mean_abs_delta_logit": float(delta_logits.abs().mean().item()),
        "max_abs_delta_probability": float(delta_probability.abs().max().item()),
        "mean_abs_delta_probability": float(delta_probability.abs().mean().item()),
        "threshold_crossing_count": int(torch.count_nonzero(source_mask != post_mask).item()),
        "source_predicted_positive_pixels": int(torch.count_nonzero(source_mask).item()),
        "post_predicted_positive_pixels": int(torch.count_nonzero(post_mask).item()),
        "source_foreground_fraction": float(source_mask.float().mean().item()),
        "post_foreground_fraction": float(post_mask.float().mean().item()),
        "teacher_reliable_background_mass": float(teacher_mass.item()),
        "source_reliable_background_mass": float(source_mass.item()),
        "post_reliable_background_mass": float(post_mass.item()),
        "functional_logit_change": bool(torch.count_nonzero(delta_logits).item() > 0),
    }


def _candidate_axis(contract: ScreenContract, candidate_id: str) -> int:
    try:
        return contract.candidate_ids.index(candidate_id)
    except ValueError as exc:
        raise StageB3ProtocolError(f"unknown candidate: {candidate_id}") from exc


def _compute_space_episode(
    *,
    contract: ScreenContract,
    model: Any,
    adapter: Any,
    film: Any,
    state_manager: Any,
    image: Any,
    student_image: Any,
    teacher: Any,
    uncertainty: Any,
    source_logits_reference: Any,
    space: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    import torch

    state_manager.reset_to_source()
    state_manager.assert_source_state()
    named_parameters = _configure_parameter_space(model, adapter, film, space)
    source_values = tuple(parameter.detach().clone() for _name, parameter in named_parameters)
    region_weights = _build_region_weights(contract, teacher, uncertainty)
    if float(region_weights.background_weight.sum().item()) <= 0.0:
        raise StageB3ProtocolError("region builder returned empty reliable background")

    # Build one source-view graph for O0 and one perturbed-view graph shared by
    # O1--O4.  All gradients are captured before any in-place virtual step.
    source_logits_graph = _forward_logits(adapter, model, film, image, space)
    student_logits_graph = _forward_logits(
        adapter, model, film, student_image, space
    )
    losses: dict[str, Any] = {}
    loss_terms: dict[str, dict[str, Any]] = {}
    for objective in OBJECTIVES:
        logits = source_logits_graph if objective == "O0" else student_logits_graph
        loss, terms = _objective_loss(
            contract, objective, logits, teacher, region_weights
        )
        losses[objective] = loss
        loss_terms[objective] = terms

    gradient_vectors: dict[str, Any] = {}
    step_vectors: dict[str, Any] = {}
    scalar_geometry: dict[str, tuple[float, float]] = {}
    parameter_tuple = tuple(parameter for _name, parameter in named_parameters)
    for index, objective in enumerate(OBJECTIVES):
        deterministic_before = bool(torch.are_deterministic_algorithms_enabled())
        warn_only_before = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        try:
            if image.device.type == "cuda":
                torch.use_deterministic_algorithms(False)
            gradients = torch.autograd.grad(
                losses[objective],
                parameter_tuple,
                retain_graph=index < len(OBJECTIVES) - 1,
                create_graph=False,
                allow_unused=False,
            )
        finally:
            torch.use_deterministic_algorithms(
                deterministic_before, warn_only=warn_only_before
            )
        flat = _flatten_named(named_parameters, gradients)
        step, gradient_norm, step_norm = _normalized_step(
            contract, space, named_parameters, flat
        )
        gradient_vectors[objective] = flat.detach().cpu().to(dtype=torch.float32)
        step_vectors[objective] = step.detach().cpu().to(dtype=torch.float32)
        scalar_geometry[objective] = (gradient_norm, step_norm)

    outputs: list[dict[str, Any]] = []
    post_probabilities: dict[str, Any] = {}
    for objective in OBJECTIVES:
        _restore_values(named_parameters, source_values)
        step_device = step_vectors[objective].to(device=image.device, dtype=image.dtype)
        _apply_vector(named_parameters, step_device)
        with torch.no_grad():
            post_logits = _forward_logits(adapter, model, film, image, space)
            post_probability = torch.sigmoid(post_logits)
            adapted_student_logits = _forward_logits(
                adapter, model, film, student_image, space
            )
            loss_after, _ = _objective_loss(
                contract,
                objective,
                post_logits if objective == "O0" else adapted_student_logits,
                teacher,
                region_weights,
            )
            diagnostics = _probability_diagnostics(
                source_logits_reference,
                post_logits,
                teacher,
                region_weights.background_weight,
            )
        gradient_norm, step_norm = scalar_geometry[objective]
        outputs.append(
            {
                "candidate_id": f"{objective}_{space}",
                "objective": objective,
                "parameter_space": space,
                "loss_before": float(losses[objective].detach().item()),
                "loss_after": float(loss_after.detach().item()),
                "loss_decreased": bool(
                    float(loss_after.detach().item())
                    < float(losses[objective].detach().item())
                ),
                "loss_terms_before": loss_terms[objective],
                "gradient_norm": gradient_norm,
                "step_norm": step_norm,
                "parameter_tensor_count": len(named_parameters),
                "parameter_scalar_count": sum(
                    int(parameter.numel()) for _name, parameter in named_parameters
                ),
                "foreground_weight_sum": float(
                    region_weights.foreground_weight.sum().item()
                ),
                "background_weight_sum": float(
                    region_weights.background_weight.sum().item()
                ),
                **diagnostics,
            }
        )
        post_probabilities[objective] = post_probability.detach().cpu().to(
            dtype=torch.float32
        )
    _restore_values(named_parameters, source_values)
    state_manager.reset_to_source()
    state_manager.assert_source_state()
    return gradient_vectors, step_vectors, outputs, post_probabilities


def _open_memmap(path: Path, *, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path, mode="w+", dtype="<f4", shape=shape
    )


def _execute_candidate_payload(
    staging: Path,
    *,
    contract: ScreenContract,
    dataset: str,
    device_name: str,
    image_limit: int,
    conditions: Sequence[tuple[str, int]],
    formal: bool,
) -> dict[str, Any]:
    import torch
    from tta.views import validated_student_perturbations

    code_sha256 = _capture_code_hashes(contract)
    (
        source_runner,
        model,
        adapter,
        film,
        state_manager,
        device,
        checkpoint_wrapper,
    ) = _build_runtime(contract, dataset, device_name)
    runtime_environment = _runtime_environment_receipt(torch, device)
    source_state_sha256 = state_manager.source_fingerprint.full_sha256
    teacher_manifest = _teacher_manifest(contract, dataset)
    selected_indices = _selected_indices(contract, dataset, teacher_manifest)[:image_limit]
    selected_ids = tuple(
        str(value)
        for value in contract.raw["datasets"][dataset]["selected_image_ids"]
    )[:image_limit]
    student_perturbation = validated_student_perturbations(
        (contract.raw["view_library"]["student_perturbation"],)
    )[0]

    condition_count = len(conditions)
    source_probabilities = _open_memmap(
        staging / "source_probabilities.npy",
        shape=(condition_count, image_limit, *SOURCE_SHAPE),
    )
    post_probabilities = _open_memmap(
        staging / "post_probabilities.npy",
        shape=(condition_count, image_limit, len(CANDIDATES), *SOURCE_SHAPE),
    )
    proxy_gradients_p2 = _open_memmap(
        staging / "proxy_gradients_P2.npy",
        shape=(condition_count, image_limit, len(OBJECTIVES), 416),
    )
    proposal_steps_p2 = _open_memmap(
        staging / "proposal_steps_P2.npy",
        shape=(condition_count, image_limit, len(OBJECTIVES), 416),
    )
    proxy_gradients_film = _open_memmap(
        staging / "proxy_gradients_DecoderFiLM.npy",
        shape=(condition_count, image_limit, len(OBJECTIVES), 32),
    )
    proposal_steps_film = _open_memmap(
        staging / "proposal_steps_DecoderFiLM.npy",
        shape=(condition_count, image_limit, len(OBJECTIVES), 32),
    )
    diagnostics: list[dict[str, Any]] = []
    source_cross_checks: list[dict[str, Any]] = []
    started = time.perf_counter()

    for condition_index, (corruption, severity) in enumerate(conditions):
        condition = _condition_key(corruption, severity)
        method_dataset = _method_input_dataset(contract, dataset, condition)
        if len(method_dataset) != 64:
            raise StageB3ProtocolError(f"cache does not contain Pilot64: {dataset}/{condition}")
        condition_record = _teacher_condition_record(teacher_manifest, condition)
        teacher_uncertainty = _verified_teacher_array(
            contract,
            dataset,
            teacher_manifest,
            condition_record,
            "view_uncertainty",
        )
        teacher_source = _verified_teacher_array(
            contract,
            dataset,
            teacher_manifest,
            condition_record,
            "source_probabilities",
        )
        expected_shapes = {
            "uncertainty": (64, 2, *SOURCE_SHAPE),
            "source": (64, *SOURCE_SHAPE),
        }
        if tuple(teacher_uncertainty.shape) != expected_shapes["uncertainty"]:
            raise StageB3ProtocolError("teacher uncertainty array shape differs")
        if tuple(teacher_source.shape) != expected_shapes["source"]:
            raise StageB3ProtocolError("teacher source array shape differs")

        for local_index, source_index in enumerate(selected_indices):
            sample = dict(method_dataset[source_index])
            if set(sample) != {
                "image",
                "image_id",
                "original_size",
                "dataset",
                "corruption",
                "severity",
                "seed",
            }:
                raise StageB3ProtocolError("method-facing fields drifted")
            image_id = str(sample["image_id"])
            if image_id != selected_ids[local_index]:
                raise StageB3ProtocolError("Pilot16 image order differs")
            if (
                sample["dataset"] != dataset
                or sample["corruption"] != corruption
                or int(sample["severity"]) != severity
                or int(sample["seed"]) != int(contract.raw["scope"]["seed"])
            ):
                raise StageB3ProtocolError("method-facing sample metadata differs")
            image_cpu = sample["image"]
            if (
                tuple(image_cpu.shape) != (3, 256, 256)
                or image_cpu.dtype != torch.float32
                or not bool(torch.isfinite(image_cpu).all().item())
            ):
                raise StageB3ProtocolError("method-facing image tensor is invalid")
            image_reference = image_cpu.clone()
            image = image_cpu.unsqueeze(0).to(device, non_blocking=False)
            student_image = student_perturbation.forward(image)
            if student_image.shape != image.shape or not bool(
                torch.isfinite(student_image).all().item()
            ):
                raise StageB3ProtocolError("student perturbation output is invalid")

            teacher = torch.from_numpy(
                np.array(
                    teacher_source[source_index],
                    dtype=np.float32,
                    copy=True,
                )
            ).unsqueeze(0).to(device)
            uncertainty = torch.from_numpy(
                np.array(
                    teacher_uncertainty[
                        source_index,
                        int(contract.raw["teacher"]["uncertainty_axis_index"]),
                    ],
                    dtype=np.float32,
                    copy=True,
                )
            ).unsqueeze(0).to(device)
            if teacher.requires_grad or uncertainty.requires_grad:
                raise StageB3ProtocolError("teacher tensors must be detached")

            state_manager.reset_to_source()
            adapter.set_source_eval_mode()
            with torch.no_grad():
                source_logits = adapter.forward_logits(image)
                source_probability = torch.sigmoid(source_logits)
            cached_source = torch.from_numpy(
                np.array(teacher_source[source_index], dtype=np.float32, copy=True)
            ).unsqueeze(0).to(device)
            source_max_abs = float(
                (source_probability - cached_source).abs().max().item()
            )
            if source_max_abs > 1.0e-6:
                raise StageB3ProtocolError(
                    f"runtime Source differs from sealed teacher source: {source_max_abs}"
                )
            source_probabilities[condition_index, local_index] = (
                source_probability[0].cpu().numpy().astype("<f4", copy=False)
            )
            source_cross_checks.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "image_id": image_id,
                    "runtime_vs_teacher_source_max_abs": source_max_abs,
                    "passed": True,
                }
            )

            for space in SPACES:
                (
                    gradients,
                    steps,
                    episode_outputs,
                    episode_post,
                ) = _compute_space_episode(
                    contract=contract,
                    model=model,
                    adapter=adapter,
                    film=film,
                    state_manager=state_manager,
                    image=image,
                    student_image=student_image,
                    teacher=teacher,
                    uncertainty=uncertainty,
                    source_logits_reference=source_logits,
                    space=space,
                )
                for objective_index, objective in enumerate(OBJECTIVES):
                    candidate_id = f"{objective}_{space}"
                    candidate_index = _candidate_axis(contract, candidate_id)
                    post_probabilities[
                        condition_index, local_index, candidate_index
                    ] = episode_post[objective][0].numpy().astype("<f4", copy=False)
                    gradient_np = gradients[objective].numpy().astype("<f4", copy=False)
                    step_np = steps[objective].numpy().astype("<f4", copy=False)
                    if space == "P2":
                        proxy_gradients_p2[
                            condition_index, local_index, objective_index
                        ] = gradient_np
                        proposal_steps_p2[
                            condition_index, local_index, objective_index
                        ] = step_np
                    else:
                        proxy_gradients_film[
                            condition_index, local_index, objective_index
                        ] = gradient_np
                        proposal_steps_film[
                            condition_index, local_index, objective_index
                        ] = step_np
                for output in episode_outputs:
                    diagnostics.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "corruption_family": corruption,
                            "severity": severity,
                            "image_index": local_index,
                            "source_pilot64_index": source_index,
                            "image_id": image_id,
                            "seed": int(sample["seed"]),
                            "input_tensor_sha256": _raw_array_sha256(
                                image_cpu.numpy()
                            ),
                            "teacher_probability_sha256": _raw_array_sha256(
                                teacher[0].cpu().numpy()
                            ),
                            "teacher_uncertainty_sha256": _raw_array_sha256(
                                uncertainty[0].cpu().numpy()
                            ),
                            "method_label_accesses": 0,
                            "validation_payload_opens": 0,
                            "test_payload_opens": 0,
                            **output,
                        }
                    )
            if not torch.equal(image_cpu, image_reference):
                raise StageB3ProtocolError("candidate phase modified its input")
            state_manager.assert_source_state()

    for array in (
        source_probabilities,
        post_probabilities,
        proxy_gradients_p2,
        proposal_steps_p2,
        proxy_gradients_film,
        proposal_steps_film,
    ):
        array.flush()
    del (
        source_probabilities,
        post_probabilities,
        proxy_gradients_p2,
        proposal_steps_p2,
        proxy_gradients_film,
        proposal_steps_film,
    )
    _write_jsonl(staging / "episode_diagnostics.jsonl", diagnostics)
    _write_jsonl(staging / "source_cross_checks.jsonl", source_cross_checks)
    state_manager.assert_source_state()
    if formal:
        _verify_consumed_dataset_payloads(
            contract, dataset, include_outer_target=False
        )
    _assert_contract_unchanged(contract)
    if _capture_code_hashes(contract) != code_sha256:
        raise StageB3ProtocolError("critical code changed during candidate phase")
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_p3_stage_b3_candidate_dataset",
        "protocol_id": PROTOCOL_ID,
        "phase": "candidate",
        "dataset": dataset,
        "formal": formal,
        "development_only": True,
        "paper_result": False,
        "paper_test_result": False,
        "config_path": str(contract.config_path.relative_to(contract.repository)),
        "config_sha256": contract.config_sha256,
        "checkpoint_role": "best_miou",
        "checkpoint_sha256": contract.raw["datasets"][dataset]["checkpoint_sha256"],
        "checkpoint_wrapper": checkpoint_wrapper,
        "runtime_environment": runtime_environment,
        "source_state_sha256": source_state_sha256,
        "source_state_restored": True,
        "conditions": [_condition_key(*value) for value in conditions],
        "condition_count": condition_count,
        "image_ids": list(selected_ids),
        "image_count_per_condition": image_limit,
        "candidate_ids": list(contract.candidate_ids),
        "objective_ids": list(OBJECTIVES),
        "parameter_spaces": list(SPACES),
        "teacher_probability_role": contract.raw["teacher"]["probability_role"],
        "teacher_probability_array": contract.raw["teacher"]["probability_array"],
        "b2_candidate_selected": False,
        "teacher_claimed_eligible": False,
        "student_perturbation": student_perturbation.name,
        "episode_count": condition_count * image_limit * len(CANDIDATES),
        "method_boundary": {
            "method_label_accesses": 0,
            "outer_target_loader_calls": 0,
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
        },
        "arrays": {
            "source_probabilities": [condition_count, image_limit, *SOURCE_SHAPE],
            "post_probabilities": [
                condition_count,
                image_limit,
                len(CANDIDATES),
                *SOURCE_SHAPE,
            ],
            "proxy_gradients_P2": [condition_count, image_limit, 5, 416],
            "proposal_steps_P2": [condition_count, image_limit, 5, 416],
            "proxy_gradients_DecoderFiLM": [condition_count, image_limit, 5, 32],
            "proposal_steps_DecoderFiLM": [condition_count, image_limit, 5, 32],
        },
        "code_sha256": code_sha256,
        "wall_time_seconds": time.perf_counter() - started,
    }


def run_candidate(
    contract: ScreenContract,
    *,
    dataset: str,
    device_name: str,
    max_images: int | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise StageB3ProtocolError(f"unsupported dataset: {dataset}")
    formal = max_images is None and condition is None
    if max_images is not None and not 1 <= max_images <= PILOT16_COUNT:
        raise StageB3ProtocolError("--max-images must lie in [1,16]")
    if condition is None:
        conditions = CONDITIONS
    else:
        matches = tuple(value for value in CONDITIONS if _condition_key(*value) == condition)
        if len(matches) != 1:
            raise StageB3ProtocolError(f"unknown condition: {condition}")
        conditions = matches
    image_limit = PILOT16_COUNT if max_images is None else max_images
    if formal:
        destination = _artifact_destination(contract, "candidate", dataset)
        existing = _existing_complete_or_raise(
            destination,
            contract=contract,
            phase="candidate",
            dataset=dataset,
        )
        if existing is not None:
            return {
                "status": "existing_verified_complete_no_op",
                "path": str(destination),
                "dataset": dataset,
            }
    else:
        destination = (
            contract.output_root
            / str(contract.raw["output"]["engineering_phase"])
            / "candidate"
            / dataset
            / uuid.uuid4().hex
        )
    staging = _new_staging(destination)
    try:
        manifest = _execute_candidate_payload(
            staging,
            contract=contract,
            dataset=dataset,
            device_name=device_name,
            image_limit=image_limit,
            conditions=conditions,
            formal=formal,
        )

        def candidate_publish_guard() -> None:
            _assert_contract_unchanged(contract)
            if formal:
                _verify_consumed_dataset_payloads(
                    contract, dataset, include_outer_target=False
                )
            if _capture_code_hashes(contract) != manifest["code_sha256"]:
                raise StageB3ProtocolError(
                    "critical code changed before candidate publication"
                )

        _publish(
            staging,
            destination,
            manifest,
            pre_rename_guard=candidate_publish_guard,
        )
    except BaseException:
        if staging.exists() and staging.name.startswith("."):
            shutil.rmtree(staging)
        raise
    verify_artifact(
        destination,
        contract=contract,
        phase="candidate",
        dataset=dataset,
        expected_formal=formal,
    )
    return {
        "status": "published",
        "formal": formal,
        "path": str(destination),
        "dataset": dataset,
    }


def _load_outer_targets(contract: ScreenContract, dataset: str):
    # This import and call exist only in the outer subcommand, after the full
    # candidate artifact has been verified by the caller.
    from materialize_binary_tent_ss_calibration_cache_v2 import (
        load_outer_evaluator_targets_v2,
    )

    cache_root = _repository_path(
        contract.repository,
        contract.raw["datasets"][dataset]["cache_root"],
        f"{dataset} cache root",
    )
    return load_outer_evaluator_targets_v2(
        cache_root,
        expected_protocol_sha256=contract.raw["frozen_parent_bindings"][
            "cache_protocol"
        ]["sha256"],
        episodes_complete=True,
    )


def _evaluation_result(probabilities: Any, targets: Any, image_ids: Sequence[str]):
    from metrics.irstd_metrics import IRSTDEvaluationProtocol
    from metrics.irstd_metrics_v2 import (
        FORMAL_FROC_THRESHOLDS_V2,
        UnifiedResearchEvaluatorV2,
    )

    protocol = IRSTDEvaluationProtocol(
        fixed_probability_threshold=0.5,
        froc_probability_thresholds=FORMAL_FROC_THRESHOLDS_V2,
        connectivity=2,
        max_centroid_distance=3.0,
        min_component_area=1,
    )
    evaluator = UnifiedResearchEvaluatorV2(protocol)
    evaluator.update_probabilities(probabilities, targets, image_ids=image_ids)
    result = evaluator.compute()
    result.assert_endpoint_conservation()
    return result


def _endpoint_summary(result: Any) -> dict[str, Any]:
    fixed = result.fixed
    total_pixels = int(fixed.total_image_pixels)
    return {
        "iou": float(fixed.pixel.intersection_over_union),
        "normalized_iou": float(result.normalized_iou.normalized_iou),
        "pd": float(fixed.detection_probability),
        "fa_per_million": float(fixed.false_alarm_pixel_rate * 1_000_000.0),
        "foreground_fraction": (
            float(fixed.pixel.predicted_positive_pixels / total_pixels)
            if total_pixels
            else 0.0
        ),
        "intersection_pixels": int(fixed.pixel.true_positive_pixels),
        "false_positive_pixels": int(fixed.pixel.false_positive_pixels),
        "false_negative_pixels": int(fixed.pixel.false_negative_pixels),
        "true_negative_pixels": int(fixed.pixel.true_negative_pixels),
        "predicted_positive_pixels": int(fixed.pixel.predicted_positive_pixels),
        "target_positive_pixels": int(fixed.pixel.target_positive_pixels),
        "detected_targets": int(fixed.detected_targets),
        "total_targets": int(fixed.total_targets),
        "false_alarm_pixels": int(fixed.false_alarm_pixels),
        "total_image_pixels": total_pixels,
        "image_count": int(fixed.image_count),
    }


def _task_gradient(
    *,
    contract: ScreenContract,
    dataset: str,
    model: Any,
    adapter: Any,
    film: Any,
    state_manager: Any,
    image: Any,
    target: Any,
    space: str,
    outer_accesses: int,
):
    import torch
    from analysis.d0_v2_task_loss import compute_d0_v2_task_loss
    from analysis.source_train_provenance import (
        OUTER_ORACLE_ROLE,
        SourceTrainAnalysisProvenance,
    )

    state_manager.reset_to_source()
    state_manager.assert_source_state()
    named_parameters = _configure_parameter_space(model, adapter, film, space)
    provenance = SourceTrainAnalysisProvenance(
        dataset=dataset,
        split_name="train",
        split_sha256=contract.raw["datasets"][dataset]["train_split_sha256"],
        checkpoint_sha256=contract.raw["datasets"][dataset]["checkpoint_sha256"],
        seed=int(contract.raw["scope"]["seed"]),
        oracle_analysis=True,
        outer_evaluator_label_accesses=outer_accesses,
        supervised_gradient_role=OUTER_ORACLE_ROLE,
    )
    logits = _forward_logits(adapter, model, film, image, space)
    raw_task_config = contract.raw["outer_oracle_task_loss"]
    task_config = {
        key: raw_task_config[key]
        for key in (
            "lambda_bce",
            "lambda_soft_iou",
            "eps",
            "bce_reduction",
            "soft_iou_reduction",
            "empty_target_convention",
        )
    }
    loss, audit = compute_d0_v2_task_loss(
        logits=logits,
        target=target,
        config=task_config,
        provenance=provenance,
    )
    deterministic_before = bool(torch.are_deterministic_algorithms_enabled())
    warn_only_before = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        if image.device.type == "cuda":
            torch.use_deterministic_algorithms(False)
        gradients = torch.autograd.grad(
            loss,
            tuple(parameter for _name, parameter in named_parameters),
            allow_unused=False,
            create_graph=False,
            retain_graph=False,
        )
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_only_before
        )
    flat = _flatten_named(named_parameters, gradients).detach().cpu().to(torch.float64)
    state_manager.reset_to_source()
    state_manager.assert_source_state()
    return flat, audit


def _alignment_record(proxy_gradient: Any, step: Any, task_gradient: Any) -> dict[str, Any]:
    import torch

    proxy = proxy_gradient.detach().cpu().to(torch.float64).reshape(-1)
    direction = step.detach().cpu().to(torch.float64).reshape(-1)
    task = task_gradient.detach().cpu().to(torch.float64).reshape(-1)
    if proxy.shape != direction.shape or proxy.shape != task.shape:
        raise StageB3ProtocolError("proxy/step/task vector shapes differ")
    proxy_norm = float(torch.linalg.vector_norm(proxy).item())
    step_norm = float(torch.linalg.vector_norm(direction).item())
    task_norm = float(torch.linalg.vector_norm(task).item())
    if not all(math.isfinite(value) for value in (proxy_norm, step_norm, task_norm)):
        raise StageB3ProtocolError("alignment vector norm is non-finite")
    valid = proxy_norm > 0.0 and step_norm > 0.0 and task_norm > 0.0
    if valid:
        cosine = float(torch.dot(proxy, task).item() / (proxy_norm * task_norm))
        normalized_derivative = float(
            torch.dot(direction, task).item() / (step_norm * task_norm)
        )
        first_order_delta = float(torch.dot(direction, task).item())
    else:
        cosine = None
        normalized_derivative = None
        first_order_delta = None
    return {
        "proxy_gradient_norm": proxy_norm,
        "proposal_step_norm": step_norm,
        "task_gradient_norm": task_norm,
        "gradient_cosine": cosine,
        "normalized_task_directional_derivative": normalized_derivative,
        "first_order_task_loss_delta": first_order_delta,
        "alignment_valid": valid,
    }


def _candidate_diagnostics_by_key(candidate_root: Path) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    path = candidate_root / "episode_diagnostics.jsonl"
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            key = (
                str(value["condition"]),
                str(value["image_id"]),
                str(value["candidate_id"]),
            )
            if key in result:
                raise StageB3ProtocolError(f"duplicate candidate diagnostic: {key}")
            result[key] = value
    return result


def _execute_outer_payload(
    staging: Path,
    *,
    contract: ScreenContract,
    dataset: str,
    device_name: str,
    candidate_root: Path,
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    code_sha256 = _capture_code_hashes(contract)
    if candidate_manifest.get("formal") is not True:
        raise StageB3ProtocolError("outer phase refuses an engineering candidate")
    if tuple(candidate_manifest.get("conditions", ())) != tuple(
        _condition_key(*value) for value in CONDITIONS
    ):
        raise StageB3ProtocolError("candidate condition topology differs")
    if tuple(candidate_manifest.get("candidate_ids", ())) != contract.candidate_ids:
        raise StageB3ProtocolError("candidate axis differs")
    selected_ids = tuple(str(value) for value in candidate_manifest["image_ids"])
    expected_ids = tuple(
        str(value)
        for value in contract.raw["datasets"][dataset]["selected_image_ids"]
    )
    if selected_ids != expected_ids:
        raise StageB3ProtocolError("candidate Pilot16 IDs differ")

    # This is the first target access in the entire B3 runner path.
    targets = _load_outer_targets(contract, dataset)
    if tuple(targets.shape) != (64, 1, 256, 256):
        raise StageB3ProtocolError("outer target array shape differs")
    teacher_manifest = _teacher_manifest(contract, dataset)
    selected_indices = _selected_indices(contract, dataset, teacher_manifest)
    selected_targets = np.stack(
        [np.asarray(targets[index], dtype=np.float32) for index in selected_indices],
        axis=0,
    )
    if not np.isfinite(selected_targets).all():
        raise StageB3ProtocolError("outer target contains NaN/Inf")

    source_probabilities = np.load(
        candidate_root / "source_probabilities.npy", mmap_mode="r", allow_pickle=False
    )
    post_probabilities = np.load(
        candidate_root / "post_probabilities.npy", mmap_mode="r", allow_pickle=False
    )
    gradients = {
        "P2": np.load(candidate_root / "proxy_gradients_P2.npy", mmap_mode="r", allow_pickle=False),
        "DecoderFiLM": np.load(
            candidate_root / "proxy_gradients_DecoderFiLM.npy", mmap_mode="r", allow_pickle=False
        ),
    }
    steps = {
        "P2": np.load(candidate_root / "proposal_steps_P2.npy", mmap_mode="r", allow_pickle=False),
        "DecoderFiLM": np.load(
            candidate_root / "proposal_steps_DecoderFiLM.npy", mmap_mode="r", allow_pickle=False
        ),
    }
    if tuple(source_probabilities.shape) != (13, 16, 1, 256, 256):
        raise StageB3ProtocolError("candidate source probability shape differs")
    if tuple(post_probabilities.shape) != (13, 16, 10, 1, 256, 256):
        raise StageB3ProtocolError("candidate post probability shape differs")
    if tuple(gradients["P2"].shape) != (13, 16, 5, 416) or tuple(
        steps["P2"].shape
    ) != (13, 16, 5, 416):
        raise StageB3ProtocolError("P2 vector artifact shape differs")
    if tuple(gradients["DecoderFiLM"].shape) != (13, 16, 5, 32) or tuple(
        steps["DecoderFiLM"].shape
    ) != (13, 16, 5, 32):
        raise StageB3ProtocolError("FiLM vector artifact shape differs")

    diagnostics = _candidate_diagnostics_by_key(candidate_root)
    expected_diagnostics = 13 * 16 * 10
    if len(diagnostics) != expected_diagnostics:
        raise StageB3ProtocolError("candidate diagnostic count differs")
    (
        _source_runner,
        model,
        adapter,
        film,
        state_manager,
        device,
        _checkpoint_wrapper,
    ) = _build_runtime(contract, dataset, device_name)
    runtime_environment = _runtime_environment_receipt(torch, device)
    alignment_records: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    outer_accesses = len(CONDITIONS) * len(selected_ids)
    started = time.perf_counter()

    for condition_index, (corruption, severity) in enumerate(CONDITIONS):
        condition = _condition_key(corruption, severity)
        method_dataset = _method_input_dataset(contract, dataset, condition)
        task_gradients: dict[tuple[int, str], Any] = {}
        task_audits: dict[tuple[int, str], Any] = {}
        for local_index, source_index in enumerate(selected_indices):
            sample = dict(method_dataset[source_index])
            if sample["image_id"] != selected_ids[local_index]:
                raise StageB3ProtocolError("outer image ID order differs")
            image = sample["image"].unsqueeze(0).to(device)
            target = torch.from_numpy(
                np.array(selected_targets[local_index], dtype=np.float32, copy=True)
            ).unsqueeze(0).to(device)
            for space in SPACES:
                gradient, audit = _task_gradient(
                    contract=contract,
                    dataset=dataset,
                    model=model,
                    adapter=adapter,
                    film=film,
                    state_manager=state_manager,
                    image=image,
                    target=target,
                    space=space,
                    outer_accesses=outer_accesses,
                )
                task_gradients[(local_index, space)] = gradient
                task_audits[(local_index, space)] = audit

        source_result = _evaluation_result(
            source_probabilities[condition_index], selected_targets, selected_ids
        )
        source_summary = _endpoint_summary(source_result)
        for candidate_index, candidate_id in enumerate(contract.candidate_ids):
            objective, space = candidate_id.split("_", 1)
            objective_index = OBJECTIVES.index(objective)
            candidate_result = _evaluation_result(
                post_probabilities[condition_index, :, candidate_index],
                selected_targets,
                selected_ids,
            )
            candidate_summary = _endpoint_summary(candidate_result)
            derivatives: list[float] = []
            cosines: list[float] = []
            first_order: list[float] = []
            valid_alignment_count = 0
            functional_count = 0
            threshold_crossings = 0
            for local_index, image_id in enumerate(selected_ids):
                proxy = torch.from_numpy(
                    np.array(
                        gradients[space][condition_index, local_index, objective_index],
                        dtype=np.float64,
                        copy=True,
                    )
                )
                step = torch.from_numpy(
                    np.array(
                        steps[space][condition_index, local_index, objective_index],
                        dtype=np.float64,
                        copy=True,
                    )
                )
                alignment = _alignment_record(
                    proxy, step, task_gradients[(local_index, space)]
                )
                diagnostic = diagnostics[(condition, image_id, candidate_id)]
                functional_count += int(bool(diagnostic["functional_logit_change"]))
                threshold_crossings += int(diagnostic["threshold_crossing_count"])
                if alignment["alignment_valid"]:
                    valid_alignment_count += 1
                    derivatives.append(
                        float(alignment["normalized_task_directional_derivative"])
                    )
                    cosines.append(float(alignment["gradient_cosine"]))
                    first_order.append(float(alignment["first_order_task_loss_delta"]))
                else:
                    # The frozen B3 estimator is the mean across all 16
                    # episodes.  A zero-step episode contributes exactly zero
                    # rather than being silently dropped and over-weighting
                    # the remaining images.
                    derivatives.append(0.0)
                alignment_records.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "corruption_family": corruption,
                        "severity": severity,
                        "image_index": local_index,
                        "image_id": image_id,
                        "candidate_id": candidate_id,
                        "objective": objective,
                        "parameter_space": space,
                        "outer_target_role": "source_train_pilot16_outer_oracle_only",
                        "adaptation_gradient_uses_labels": False,
                        "task_loss_total": float(
                            task_audits[(local_index, space)]["components"]["total_loss"]
                        ),
                        **alignment,
                    }
                )
            delta_iou = candidate_summary["iou"] - source_summary["iou"]
            delta_pd = candidate_summary["pd"] - source_summary["pd"]
            delta_fa = (
                candidate_summary["fa_per_million"]
                - source_summary["fa_per_million"]
            )
            cell_summaries.append(
                {
                    "candidate_id": candidate_id,
                    "dataset": dataset,
                    "condition": condition,
                    "corruption_family": corruption,
                    "severity": severity,
                    "episode_count": len(selected_ids),
                    "valid_alignment_episode_count": valid_alignment_count,
                    "normalized_task_directional_derivative": float(
                        sum(derivatives) / len(selected_ids)
                    ),
                    "gradient_cosine": (
                        float(sum(cosines) / len(cosines)) if cosines else None
                    ),
                    "first_order_task_loss_delta": (
                        float(sum(first_order) / len(first_order))
                        if first_order
                        else None
                    ),
                    "source_iou": source_summary["iou"],
                    "adapted_iou": candidate_summary["iou"],
                    "actual_delta_iou": delta_iou,
                    "source_pd": source_summary["pd"],
                    "adapted_pd": candidate_summary["pd"],
                    "delta_pd": delta_pd,
                    "source_fa_per_million": source_summary["fa_per_million"],
                    "adapted_fa_per_million": candidate_summary["fa_per_million"],
                    "delta_fa_per_million": delta_fa,
                    "source_foreground_fraction": source_summary["foreground_fraction"],
                    "adapted_foreground_fraction": candidate_summary["foreground_fraction"],
                    "functional_logit_change": functional_count > 0,
                    "functional_changed_episode_count": functional_count,
                    "threshold_crossing_count": threshold_crossings,
                    "source_counts": source_summary,
                    "adapted_counts": candidate_summary,
                }
            )
    _write_jsonl(staging / "alignment_records.jsonl", alignment_records)
    _write_jsonl(staging / "cell_summaries.jsonl", cell_summaries)
    state_manager.assert_source_state()
    _verify_consumed_dataset_payloads(
        contract, dataset, include_outer_target=True
    )
    _assert_contract_unchanged(contract)
    if _capture_code_hashes(contract) != code_sha256:
        raise StageB3ProtocolError("critical code changed during outer phase")
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_p3_stage_b3_outer_dataset",
        "protocol_id": PROTOCOL_ID,
        "phase": "outer",
        "dataset": dataset,
        "formal": True,
        "development_only": True,
        "paper_result": False,
        "paper_test_result": False,
        "config_sha256": contract.config_sha256,
        "candidate_manifest_sha256": sha256_file(candidate_root / "manifest.json"),
        "candidate_complete_sha256": sha256_file(candidate_root / "COMPLETE.json"),
        "condition_count": 13,
        "candidate_count": 10,
        "image_count_per_condition": 16,
        "cell_summary_count": len(cell_summaries),
        "alignment_record_count": len(alignment_records),
        "outer_target_accesses": outer_accesses,
        "outer_target_role": "source_train_pilot16_outer_oracle_only",
        "runtime_environment": runtime_environment,
        "method_label_accesses": 0,
        "adaptation_gradient_uses_labels": False,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
        "code_sha256": code_sha256,
        "wall_time_seconds": time.perf_counter() - started,
    }


def run_outer(
    contract: ScreenContract, *, dataset: str, device_name: str
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise StageB3ProtocolError(f"unsupported dataset: {dataset}")
    candidate_root = _artifact_destination(contract, "candidate", dataset)
    candidate_manifest = verify_artifact(
        candidate_root,
        contract=contract,
        phase="candidate",
        dataset=dataset,
    )
    if candidate_manifest.get("code_sha256") != _capture_code_hashes(contract):
        raise StageB3ProtocolError(
            f"critical code changed after candidate publication: {dataset}"
        )
    destination = _artifact_destination(contract, "outer", dataset)
    existing = _existing_complete_or_raise(
        destination,
        contract=contract,
        phase="outer",
        dataset=dataset,
    )
    if existing is not None:
        return {
            "status": "existing_verified_complete_no_op",
            "path": str(destination),
            "dataset": dataset,
        }
    staging = _new_staging(destination)
    try:
        manifest = _execute_outer_payload(
            staging,
            contract=contract,
            dataset=dataset,
            device_name=device_name,
            candidate_root=candidate_root,
            candidate_manifest=candidate_manifest,
        )
        verify_artifact(
            candidate_root,
            contract=contract,
            phase="candidate",
            dataset=dataset,
        )
        if (
            manifest.get("candidate_manifest_sha256")
            != sha256_file(candidate_root / "manifest.json")
            or manifest.get("candidate_complete_sha256")
            != sha256_file(candidate_root / "COMPLETE.json")
        ):
            raise StageB3ProtocolError(
                f"candidate lineage changed during outer phase: {dataset}"
            )

        def outer_publish_guard() -> None:
            _assert_contract_unchanged(contract)
            _verify_consumed_dataset_payloads(
                contract, dataset, include_outer_target=True
            )
            verify_artifact(
                candidate_root,
                contract=contract,
                phase="candidate",
                dataset=dataset,
            )
            if (
                _capture_code_hashes(contract) != manifest["code_sha256"]
                or manifest["candidate_manifest_sha256"]
                != sha256_file(candidate_root / "manifest.json")
                or manifest["candidate_complete_sha256"]
                != sha256_file(candidate_root / "COMPLETE.json")
            ):
                raise StageB3ProtocolError(
                    f"outer publication lineage changed: {dataset}"
                )

        _publish(
            staging,
            destination,
            manifest,
            pre_rename_guard=outer_publish_guard,
        )
    except BaseException:
        if staging.exists() and staging.name.startswith("."):
            shutil.rmtree(staging)
        raise
    verify_artifact(
        destination,
        contract=contract,
        phase="outer",
        dataset=dataset,
    )
    return {"status": "published", "path": str(destination), "dataset": dataset}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise StageB3ProtocolError(f"JSONL input is missing/unsafe: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StageB3ProtocolError(
                    f"invalid JSONL record at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise StageB3ProtocolError(
                    f"JSONL record must be a mapping at {path}:{line_number}"
                )
            records.append(value)
    return records


def _gate_cell_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project one rich outer record onto the exact pure-gate schema."""

    valid_count = int(value["valid_alignment_episode_count"])
    derivative = value["normalized_task_directional_derivative"]
    if derivative is None:
        if valid_count != 0:
            raise StageB3ProtocolError(
                "a cell with valid alignments cannot omit its derivative"
            )
        derivative = 0.0
    severity_value = int(value["severity"])
    return {
        "candidate_id": str(value["candidate_id"]),
        "dataset": str(value["dataset"]),
        "condition": str(value["condition"]),
        "corruption_family": str(value["corruption_family"]),
        "severity": f"S{severity_value}",
        "normalized_task_directional_derivative": float(derivative),
        "source_iou": float(value["source_iou"]),
        "adapted_iou": float(value["adapted_iou"]),
        "delta_pd": float(value["delta_pd"]),
        "delta_fa": float(value["delta_fa_per_million"]),
        "source_foreground_fraction": float(
            value["source_foreground_fraction"]
        ),
        "adapted_foreground_fraction": float(
            value["adapted_foreground_fraction"]
        ),
        "valid_alignment_episode_count": valid_count,
        "episode_count": int(value["episode_count"]),
        "functional_changed_episode_count": int(
            value["functional_changed_episode_count"]
        ),
        "threshold_crossing_count": int(value["threshold_crossing_count"]),
    }


def _fraction_receipt_float(value: Mapping[str, Any]) -> float:
    numerator = int(value["numerator"])
    denominator = int(value["denominator"])
    if denominator == 0:
        raise StageB3ProtocolError("science receipt contains a zero denominator")
    return numerator / denominator


def _human_science_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for evaluation in receipt["candidate_evaluations"]:
        aggregate = evaluation["aggregate"]
        candidates.append(
            {
                "candidate_id": evaluation["candidate_id"],
                "eligible": bool(evaluation["eligible"]),
                "reason_codes": list(evaluation["reason_codes"]),
                "nonclean_macro_delta_iou": _fraction_receipt_float(
                    aggregate["nonclean_macro_delta_iou"]
                ),
                "clean_macro_delta_iou": _fraction_receipt_float(
                    aggregate["clean_macro_delta_iou"]
                ),
                "nonclean_macro_directional_derivative": (
                    _fraction_receipt_float(
                        aggregate["nonclean_macro_directional_derivative"]
                    )
                ),
                "nonclean_macro_delta_pd": _fraction_receipt_float(
                    aggregate["nonclean_macro_delta_pd"]
                ),
                "nonclean_macro_delta_fa_per_million": (
                    _fraction_receipt_float(
                        aggregate["nonclean_macro_delta_fa"]
                    )
                ),
                "nonclean_functional_changed_episode_count": int(
                    aggregate["nonclean_functional_changed_episode_count"]
                ),
                "nonclean_threshold_crossing_count": int(
                    aggregate["nonclean_threshold_crossing_count"]
                ),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scope": "source_train_pilot16_development_only",
        "paper_result": False,
        "paper_test_result": False,
        "scientific_status": receipt["scientific_status"],
        "stage_b4_allowed": bool(receipt["stage_b4_allowed"]),
        "selected_for_stage_b4": list(receipt["selected_for_stage_b4"]),
        "candidates": candidates,
    }


def _execute_aggregate_payload(
    staging: Path,
    *,
    contract: ScreenContract,
    outer_artifacts: Mapping[str, tuple[Path, Mapping[str, Any]]],
    candidate_artifacts: Mapping[str, tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    from analysis.proxy_objective_space_screen_v1 import (
        evaluate_stage_b3_science_gate,
    )

    rich_cells: list[dict[str, Any]] = []
    outer_bindings: dict[str, Any] = {}
    current_code_sha256 = _capture_code_hashes(contract)
    for dataset in DATASETS:
        root, manifest = outer_artifacts[dataset]
        candidate_root, candidate_manifest = candidate_artifacts[dataset]
        if manifest.get("code_sha256") != current_code_sha256:
            raise StageB3ProtocolError(
                f"critical code changed after outer publication: {dataset}"
            )
        if candidate_manifest.get("code_sha256") != current_code_sha256:
            raise StageB3ProtocolError(
                f"critical code changed after candidate publication: {dataset}"
            )
        candidate_manifest_sha256 = sha256_file(candidate_root / "manifest.json")
        candidate_complete_sha256 = sha256_file(candidate_root / "COMPLETE.json")
        if (
            manifest.get("candidate_manifest_sha256")
            != candidate_manifest_sha256
            or manifest.get("candidate_complete_sha256")
            != candidate_complete_sha256
        ):
            raise StageB3ProtocolError(
                f"outer-to-candidate lineage differs: {dataset}"
            )
        records = _read_jsonl(root / "cell_summaries.jsonl")
        if len(records) != len(CONDITIONS) * len(CANDIDATES):
            raise StageB3ProtocolError(
                f"outer cell count differs for {dataset}: {len(records)}"
            )
        if any(record.get("dataset") != dataset for record in records):
            raise StageB3ProtocolError(
                f"outer cells contain a wrong dataset label: {dataset}"
            )
        rich_cells.extend(records)
        outer_bindings[dataset] = {
            "path": str(root.relative_to(contract.repository)),
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "complete_sha256": sha256_file(root / "COMPLETE.json"),
            "cell_summaries_sha256": sha256_file(root / "cell_summaries.jsonl"),
            "candidate_path": str(
                candidate_root.relative_to(contract.repository)
            ),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "candidate_complete_sha256": candidate_complete_sha256,
        }

    gate_cells = [_gate_cell_summary(value) for value in rich_cells]
    decision = evaluate_stage_b3_science_gate(
        gate_cells,
        contract.candidate_ids,
        contract.raw["stage_b3_gate"],
    )
    receipt = decision.to_receipt()
    if receipt["paper_result"] is not False or receipt["test"] is not False:
        raise StageB3ProtocolError("science gate emitted a non-development receipt")
    _write_jsonl(staging / "cell_summaries.jsonl", rich_cells)
    _write_jsonl(staging / "science_gate_input.jsonl", gate_cells)
    _write_json(staging / "science_decision_receipt.json", receipt)
    _write_json(staging / "summary.json", _human_science_summary(receipt))
    for dataset in DATASETS:
        candidate_root, _candidate_manifest = candidate_artifacts[dataset]
        outer_root, _outer_manifest = outer_artifacts[dataset]
        verify_artifact(
            candidate_root,
            contract=contract,
            phase="candidate",
            dataset=dataset,
        )
        verify_artifact(
            outer_root,
            contract=contract,
            phase="outer",
            dataset=dataset,
        )
    _assert_contract_unchanged(contract)
    if _capture_code_hashes(contract) != current_code_sha256:
        raise StageB3ProtocolError("critical code changed during aggregate phase")
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_p3_stage_b3_aggregate",
        "protocol_id": PROTOCOL_ID,
        "phase": "aggregate",
        "dataset": None,
        "formal": True,
        "development_only": True,
        "paper_result": False,
        "paper_test_result": False,
        "config_sha256": contract.config_sha256,
        "outer_artifacts": outer_bindings,
        "dataset_count": len(DATASETS),
        "condition_count_per_dataset": len(CONDITIONS),
        "candidate_count": len(CANDIDATES),
        "cell_summary_count": len(rich_cells),
        "gate_input_count": len(gate_cells),
        "scientific_status": decision.scientific_status,
        "stage_b4_allowed": decision.stage_b4_allowed,
        "selected_for_stage_b4": list(decision.selected_candidate_ids),
        "method_label_accesses": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
        "outer_target_role": "source_train_pilot16_outer_oracle_only",
        "code_sha256": current_code_sha256,
    }


def run_aggregate(contract: ScreenContract) -> dict[str, Any]:
    destination = _artifact_destination(contract, "aggregate")
    existing = _existing_complete_or_raise(
        destination,
        contract=contract,
        phase="aggregate",
        dataset=None,
    )
    if existing is not None:
        current_code_sha256 = _capture_code_hashes(contract)
        if existing.get("code_sha256") != current_code_sha256:
            raise StageB3ProtocolError(
                "critical code changed after aggregate publication"
            )
        recorded_outer = _mapping(
            existing.get("outer_artifacts"), "aggregate outer artifacts"
        )
        if set(recorded_outer) != set(DATASETS):
            raise StageB3ProtocolError("aggregate parent dataset roster differs")
        for dataset in DATASETS:
            candidate_root = _artifact_destination(
                contract, "candidate", dataset
            )
            outer_root = _artifact_destination(contract, "outer", dataset)
            candidate_manifest = verify_artifact(
                candidate_root,
                contract=contract,
                phase="candidate",
                dataset=dataset,
            )
            outer_manifest = verify_artifact(
                outer_root,
                contract=contract,
                phase="outer",
                dataset=dataset,
            )
            parent = _mapping(
                recorded_outer[dataset], f"aggregate parent {dataset}"
            )
            if (
                candidate_manifest.get("code_sha256") != current_code_sha256
                or outer_manifest.get("code_sha256") != current_code_sha256
                or parent.get("manifest_sha256")
                != sha256_file(outer_root / "manifest.json")
                or parent.get("complete_sha256")
                != sha256_file(outer_root / "COMPLETE.json")
                or parent.get("cell_summaries_sha256")
                != sha256_file(outer_root / "cell_summaries.jsonl")
                or parent.get("candidate_manifest_sha256")
                != sha256_file(candidate_root / "manifest.json")
                or parent.get("candidate_complete_sha256")
                != sha256_file(candidate_root / "COMPLETE.json")
                or outer_manifest.get("candidate_manifest_sha256")
                != parent.get("candidate_manifest_sha256")
                or outer_manifest.get("candidate_complete_sha256")
                != parent.get("candidate_complete_sha256")
            ):
                raise StageB3ProtocolError(
                    f"existing aggregate lineage differs: {dataset}"
                )
        _assert_contract_unchanged(contract)
        receipt = _load_json(destination / "science_decision_receipt.json")
        from analysis.proxy_objective_space_screen_v1 import (
            evaluate_stage_b3_science_gate,
        )

        gate_cells = _read_jsonl(destination / "science_gate_input.jsonl")
        recomputed_receipt = evaluate_stage_b3_science_gate(
            gate_cells,
            contract.candidate_ids,
            contract.raw["stage_b3_gate"],
        ).to_receipt()
        if receipt != recomputed_receipt:
            raise StageB3ProtocolError(
                "existing aggregate science decision does not recompute exactly"
            )
        if _load_json(destination / "summary.json") != _human_science_summary(
            recomputed_receipt
        ):
            raise StageB3ProtocolError(
                "existing aggregate human summary does not recompute exactly"
            )
        return {
            "status": "existing_verified_complete_no_op",
            "path": str(destination),
            "scientific_status": receipt["scientific_status"],
            "stage_b4_allowed": bool(receipt["stage_b4_allowed"]),
            "selected_for_stage_b4": list(receipt["selected_for_stage_b4"]),
        }

    outer_artifacts: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    candidate_artifacts: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for dataset in DATASETS:
        candidate_root = _artifact_destination(contract, "candidate", dataset)
        candidate_manifest = verify_artifact(
            candidate_root,
            contract=contract,
            phase="candidate",
            dataset=dataset,
        )
        root = _artifact_destination(contract, "outer", dataset)
        manifest = verify_artifact(
            root,
            contract=contract,
            phase="outer",
            dataset=dataset,
        )
        candidate_artifacts[dataset] = (candidate_root, candidate_manifest)
        outer_artifacts[dataset] = (root, manifest)

    staging = _new_staging(destination)
    try:
        manifest = _execute_aggregate_payload(
            staging,
            contract=contract,
            outer_artifacts=outer_artifacts,
            candidate_artifacts=candidate_artifacts,
        )

        def aggregate_publish_guard() -> None:
            _assert_contract_unchanged(contract)
            if _capture_code_hashes(contract) != manifest["code_sha256"]:
                raise StageB3ProtocolError(
                    "critical code changed before aggregate publication"
                )
            for guarded_dataset in DATASETS:
                candidate_root, _candidate = candidate_artifacts[guarded_dataset]
                outer_root, _outer = outer_artifacts[guarded_dataset]
                verify_artifact(
                    candidate_root,
                    contract=contract,
                    phase="candidate",
                    dataset=guarded_dataset,
                )
                verify_artifact(
                    outer_root,
                    contract=contract,
                    phase="outer",
                    dataset=guarded_dataset,
                )
                binding = manifest["outer_artifacts"][guarded_dataset]
                if (
                    binding["manifest_sha256"]
                    != sha256_file(outer_root / "manifest.json")
                    or binding["complete_sha256"]
                    != sha256_file(outer_root / "COMPLETE.json")
                    or binding["cell_summaries_sha256"]
                    != sha256_file(outer_root / "cell_summaries.jsonl")
                    or binding["candidate_manifest_sha256"]
                    != sha256_file(candidate_root / "manifest.json")
                    or binding["candidate_complete_sha256"]
                    != sha256_file(candidate_root / "COMPLETE.json")
                ):
                    raise StageB3ProtocolError(
                        f"aggregate publication lineage changed: {guarded_dataset}"
                    )

        _publish(
            staging,
            destination,
            manifest,
            pre_rename_guard=aggregate_publish_guard,
        )
    except BaseException:
        if staging.exists() and staging.name.startswith("."):
            shutil.rmtree(staging)
        raise
    verify_artifact(
        destination,
        contract=contract,
        phase="aggregate",
        dataset=None,
    )
    receipt = _load_json(destination / "science_decision_receipt.json")
    return {
        "status": "published",
        "path": str(destination),
        "scientific_status": receipt["scientific_status"],
        "stage_b4_allowed": bool(receipt["stage_b4_allowed"]),
        "selected_for_stage_b4": list(receipt["selected_for_stage_b4"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3_stage_b_objective_space_screen_v1.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate every frozen protocol binding")

    smoke = commands.add_parser("smoke", help="run an engineering-only candidate smoke")
    smoke.add_argument("--dataset", choices=DATASETS, required=True)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--max-images", type=int, default=1)
    smoke.add_argument(
        "--condition",
        choices=tuple(_condition_key(*value) for value in CONDITIONS),
        default="clean_S0",
    )

    candidate = commands.add_parser("candidate", help="run one formal candidate dataset")
    candidate.add_argument("--dataset", choices=DATASETS, required=True)
    candidate.add_argument("--device", required=True)

    outer = commands.add_parser("outer", help="run the isolated train-target outer phase")
    outer.add_argument("--dataset", choices=DATASETS, required=True)
    outer.add_argument("--device", required=True)

    commands.add_parser("aggregate", help="aggregate all datasets and apply the B3 gate")
    verify = commands.add_parser("verify", help="rehash one immutable artifact")
    verify.add_argument("--phase", choices=("candidate", "outer", "aggregate"), required=True)
    verify.add_argument("--dataset", choices=DATASETS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = load_contract(args.config)
    if args.command == "validate":
        from analysis.proxy_objective_space_screen_v1 import GateConfig

        GateConfig.from_mapping(contract.raw["stage_b3_gate"])
        result = {
            "status": "valid",
            "protocol_id": PROTOCOL_ID,
            "config_sha256": contract.config_sha256,
            "critical_code_sha256": _capture_code_hashes(contract),
            "method_label_accesses": 0,
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
        }
    elif args.command == "smoke":
        result = run_candidate(
            contract,
            dataset=args.dataset,
            device_name=args.device,
            max_images=args.max_images,
            condition=args.condition,
        )
    elif args.command == "candidate":
        result = run_candidate(
            contract,
            dataset=args.dataset,
            device_name=args.device,
        )
    elif args.command == "outer":
        result = run_outer(
            contract,
            dataset=args.dataset,
            device_name=args.device,
        )
    elif args.command == "aggregate":
        result = run_aggregate(contract)
    elif args.command == "verify":
        if args.phase == "aggregate":
            if args.dataset is not None:
                raise StageB3ProtocolError(
                    "aggregate verification does not accept --dataset"
                )
            dataset = None
        else:
            if args.dataset is None:
                raise StageB3ProtocolError(
                    "candidate/outer verification requires --dataset"
                )
            dataset = args.dataset
        path = _artifact_destination(contract, args.phase, dataset)
        manifest = verify_artifact(
            path,
            contract=contract,
            phase=args.phase,
            dataset=dataset,
        )
        current_code_sha256 = _capture_code_hashes(contract)
        if manifest.get("code_sha256") != current_code_sha256:
            raise StageB3ProtocolError(
                f"critical code changed after {args.phase} publication"
            )
        if args.phase == "candidate":
            assert dataset is not None
            _verify_consumed_dataset_payloads(
                contract, dataset, include_outer_target=False
            )
            _assert_contract_unchanged(contract)
        elif args.phase == "outer":
            assert dataset is not None
            candidate_root = _artifact_destination(
                contract, "candidate", dataset
            )
            verify_artifact(
                candidate_root,
                contract=contract,
                phase="candidate",
                dataset=dataset,
            )
            if (
                manifest.get("candidate_manifest_sha256")
                != sha256_file(candidate_root / "manifest.json")
                or manifest.get("candidate_complete_sha256")
                != sha256_file(candidate_root / "COMPLETE.json")
            ):
                raise StageB3ProtocolError(
                    f"outer-to-candidate lineage differs: {dataset}"
                )
            _verify_consumed_dataset_payloads(
                contract, dataset, include_outer_target=True
            )
            _assert_contract_unchanged(contract)
        else:
            # Destination existence was established above, so this is the
            # read-only existing-artifact path.  It revalidates all parent
            # lineages and recomputes the pure science decision exactly.
            aggregate_check = run_aggregate(contract)
            if aggregate_check["status"] != "existing_verified_complete_no_op":
                raise StageB3ProtocolError(
                    "aggregate verify unexpectedly attempted publication"
                )
        result = {
            "status": "verified_complete",
            "path": str(path),
            "phase": args.phase,
            "dataset": dataset,
            "manifest_sha256": sha256_file(path / "manifest.json"),
            "payload_tree_sha256": manifest["payload_tree_sha256"],
        }
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
