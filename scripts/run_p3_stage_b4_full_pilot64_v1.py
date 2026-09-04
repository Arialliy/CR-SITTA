#!/usr/bin/env python3
"""Run the train-only CR-SITTA Stage-B4 Full-Pilot64 proposal gate.

The candidate command is label-free and consumes only frozen train-side image
caches plus the sealed B2 Source probabilities/uncertainty.  The outer command
is the only command allowed to deserialize train targets, and only after a
complete candidate artifact has passed a full ledger/lineage verification.
Validation and test payloads are never accepted by this runner.

Formal artifacts are immutable, atomically published directories.  Every
publication rechecks the frozen config, critical-code hashes, consumed input
hashes, and parent artifacts immediately before the no-replace rename.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
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

from scripts import run_p3_stage_b_screen_v1 as b3  # noqa: E402


PROTOCOL_ID = "cr-sitta-p3-stage-b4-full-pilot64-proposal-gate-v1"
FROZEN_CONFIG_SHA256 = (
    "dcc2436e8e6443e5731223a49d5e8c74e326cf50f9bd065195990085d8f46614"
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
CANDIDATES = (
    "O3_P2",
    "O4_P2",
    "O3_DecoderFiLM",
    "O4_DecoderFiLM",
)
OBJECTIVES = ("O3", "O4")
SPACES = ("P2", "DecoderFiLM")
PILOT64_COUNT = 64
SOURCE_SHAPE = (1, 256, 256)
SHA256_HEX = frozenset("0123456789abcdef")


class StageB4ProtocolError(RuntimeError):
    """The frozen Stage-B4 contract or an artifact is invalid."""


class ExistingArtifactError(StageB4ProtocolError):
    """A formal destination exists but is incomplete or conflicting."""


@dataclass(frozen=True)
class FullPilotContract:
    repository: Path
    config_path: Path
    config_sha256: str
    raw: Mapping[str, Any]

    @property
    def output_root(self) -> Path:
        return self.repository / str(self.raw["output"]["root"])

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(str(value["id"]) for value in self.raw["candidates"])


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
        raise StageB4ProtocolError(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise StageB4ProtocolError(f"{label} must be lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageB4ProtocolError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageB4ProtocolError(f"{label} must be a sequence")
    return value


def _repository_path(repository: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise StageB4ProtocolError(f"{label} must be a non-empty path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise StageB4ProtocolError(f"{label} must be repository-relative")
    resolved = repository / relative
    if not resolved.absolute().is_relative_to(repository):
        raise StageB4ProtocolError(f"{label} escapes repository")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageB4ProtocolError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StageB4ProtocolError(f"JSON root must be a mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise StageB4ProtocolError(f"JSONL input is missing/unsafe: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StageB4ProtocolError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise StageB4ProtocolError(
                    f"JSONL record must be a mapping: {path}:{line_number}"
                )
            records.append(value)
    return records


def _compact_ordered_ids_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_parent_bindings(repository: Path, raw: Mapping[str, Any]) -> None:
    parents = _mapping(raw.get("frozen_parent_bindings"), "parent bindings")
    for name, item in parents.items():
        record = _mapping(item, f"parent binding {name}")
        if set(record) != {"path", "sha256"}:
            raise StageB4ProtocolError(f"parent binding {name} fields drifted")
        path = _repository_path(repository, record["path"], f"parent {name}")
        expected = _require_sha256(record["sha256"], f"parent {name} sha256")
        if sha256_file(path) != expected:
            raise StageB4ProtocolError(f"frozen parent changed: {name}")


def _teacher_manifest(contract: FullPilotContract, dataset: str) -> dict[str, Any]:
    record = contract.raw["datasets"][dataset]
    root = _repository_path(
        contract.repository, record["teacher_artifact_root"], f"{dataset} teacher"
    )
    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE.json"
    if sha256_file(manifest_path) != record["teacher_manifest_sha256"]:
        raise StageB4ProtocolError(f"teacher manifest changed: {dataset}")
    if sha256_file(complete_path) != record["teacher_complete_sha256"]:
        raise StageB4ProtocolError(f"teacher COMPLETE changed: {dataset}")
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    if (
        manifest.get("protocol_id")
        != "cr-sitta-nonadaptive-teacher-screen-v1"
        or manifest.get("phase") != "candidate"
        or manifest.get("dataset") != dataset
        or manifest.get("formal") is not True
        or manifest.get("paper_result") is not False
        or manifest.get("image_count_per_condition") != PILOT64_COUNT
        or complete.get("complete") is not True
        or complete.get("manifest_sha256") != record["teacher_manifest_sha256"]
    ):
        raise StageB4ProtocolError(f"teacher artifact semantics differ: {dataset}")
    image_ids = tuple(str(value) for value in manifest.get("image_ids", ()))
    if len(image_ids) != PILOT64_COUNT or len(set(image_ids)) != PILOT64_COUNT:
        raise StageB4ProtocolError(f"teacher Pilot64 IDs are invalid: {dataset}")
    if _compact_ordered_ids_sha256(image_ids) != record[
        "ordered_pilot64_image_ids_sha256"
    ]:
        raise StageB4ProtocolError(f"teacher Pilot64 ID hash differs: {dataset}")
    return manifest


def _verify_cache_binding(contract: FullPilotContract, dataset: str) -> None:
    record = contract.raw["datasets"][dataset]
    expected_protocol_sha = contract.raw["frozen_parent_bindings"][
        "cache_protocol"
    ]["sha256"]
    try:
        b3._verify_cache_dataset_binding(
            contract.repository,
            dataset,
            record,
            expected_protocol_sha256=expected_protocol_sha,
        )
    except Exception as exc:
        raise StageB4ProtocolError(f"cache binding differs: {dataset}") from exc
    cache_root = _repository_path(
        contract.repository, record["cache_root"], f"{dataset} cache"
    )
    cache_manifest = _load_json(cache_root / "manifest.json")
    image_ids = tuple(str(value) for value in cache_manifest.get("image_ids", ()))
    teacher_ids = tuple(str(value) for value in _teacher_manifest(contract, dataset)["image_ids"])
    if image_ids != teacher_ids:
        raise StageB4ProtocolError(f"cache/teacher Pilot64 order differs: {dataset}")
    if _compact_ordered_ids_sha256(image_ids) != record[
        "ordered_pilot64_image_ids_sha256"
    ]:
        raise StageB4ProtocolError(f"cache Pilot64 ID hash differs: {dataset}")


def _validate_contract_semantics(repository: Path, raw: Mapping[str, Any]) -> None:
    if raw.get("schema_version") != 1 or raw.get("protocol_id") != PROTOCOL_ID:
        raise StageB4ProtocolError("Stage-B4 schema/protocol differs")
    scope = _mapping(raw.get("scope"), "scope")
    required_scope = {
        "source_train_derived": True,
        "split_name": "train",
        "split_role": "frozen_pilot64",
        "no_validation_split": True,
        "image_count_per_condition": 64,
        "dataset_count": 3,
        "condition_count_per_dataset": 13,
        "candidate_count": 4,
        "episodes_per_candidate": 2496,
        "total_candidate_episodes": 9984,
        "replicate_ids": ["R0"],
        "seed": 42,
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
            raise StageB4ProtocolError(f"scope.{key} must be exactly {expected!r}")
    transition = _mapping(raw.get("stage_transition"), "stage transition")
    if (
        transition.get("stage_b3_protocol_complete") is not True
        or transition.get("stage_b3_scientific_status") != "scientific_passed"
        or transition.get("stage_b3_stage_b4_allowed") is not True
        or transition.get("stage_b5_allowed_before_b4_gate") is not False
        or transition.get("formal_test_allowed_by_b4") is not False
    ):
        raise StageB4ProtocolError("stage-transition authorization differs")
    conditions = tuple(tuple(value) for value in raw.get("ordered_conditions", ()))
    if conditions != CONDITIONS:
        raise StageB4ProtocolError("ordered 13-condition grid drifted")
    datasets = _mapping(raw.get("datasets"), "datasets")
    if tuple(datasets) != DATASETS:
        raise StageB4ProtocolError("dataset order/set drifted")
    candidates = tuple(str(item["id"]) for item in raw.get("candidates", ()))
    if candidates != CANDIDATES:
        raise StageB4ProtocolError("frozen B3-selected candidate roster drifted")
    receipt_binding = raw["frozen_parent_bindings"]["stage_b3_science_decision"]
    receipt = _load_json(_repository_path(repository, receipt_binding["path"], "B3 receipt"))
    if (
        receipt.get("protocol_status") != "passed"
        or receipt.get("scientific_status") != "scientific_passed"
        or receipt.get("stage_b4_allowed") is not True
        or tuple(receipt.get("selected_for_stage_b4", ())) != CANDIDATES
    ):
        raise StageB4ProtocolError("B3 receipt does not authorize exact B4 roster")
    teacher = _mapping(raw.get("teacher"), "teacher")
    if (
        teacher.get("probability_role") != "sealed_source_identity"
        or teacher.get("probability_array") != "source_probabilities"
        or teacher.get("b2_candidate_selected") is not False
        or teacher.get("uncertainty_view_set") != "flip4"
        or teacher.get("uncertainty_axis_index") != 0
        or teacher.get("detached_required") is not True
        or teacher.get("teacher_scientific_eligibility_claimed") is not False
    ):
        raise StageB4ProtocolError("teacher contract drifted")
    proposal = _mapping(raw.get("proposal"), "proposal")
    if (
        proposal.get("algorithm")
        != "explicit_min_clipped_source_anchored_single_step"
        or proposal.get("optimizer_object_forbidden") is not True
        or tuple(proposal.get("backtracking_coefficients", ()))
        != (1.0, 0.5, 0.25, 0.125)
        or proposal.get("armijo_c") != 1.0e-4
        or proposal.get("retry_from_exact_source_snapshot") is not True
        or proposal.get("all_rejected_action") != "exact_no_update"
        or proposal.get("safety_uses_outer_target") is not False
    ):
        raise StageB4ProtocolError("proposal/Armijo contract drifted")
    safety = _mapping(
        proposal.get("per_attempt_label_free_safety"), "proposal safety"
    )
    components = _mapping(safety.get("connected_component_count"), "component safety")
    if (
        safety.get("evaluate_on_original_image") is not True
        or safety.get("all_parameters_logits_probabilities_and_losses_finite")
        is not True
        or safety.get("reliable_background_mass_delta_maximum") != 0.0001
        or safety.get("predicted_positive_fraction_delta_maximum") != 0.001
        or components.get("connectivity_2d") != 8
        or components.get("min_component_area") != 1
        or components.get("absolute_allowance") != 3
        or components.get("source_multiplier") != 2.0
    ):
        raise StageB4ProtocolError("label-free safety thresholds drifted")
    spaces = _mapping(raw.get("parameter_spaces"), "parameter spaces")
    if (
        spaces["P2"].get("relative_radius") != 0.0005
        or spaces["P2"].get("absolute_radius") is not None
        or spaces["DecoderFiLM"].get("relative_radius") is not None
        or spaces["DecoderFiLM"].get("absolute_radius") != 0.25
    ):
        raise StageB4ProtocolError("parameter trust radii drifted")
    evaluation = _mapping(raw.get("evaluation"), "evaluation")
    if (
        evaluation.get("probability_threshold") != 0.5
        or evaluation.get("threshold_rule") != "strict_greater_than"
        or evaluation.get("foreground_connectivity_2d") != 8
        or evaluation.get("min_component_area") != 1
        or evaluation.get("max_centroid_distance_pixels") != 3.0
        or evaluation.get("missing_value_policy") != "forbidden_no_imputation"
    ):
        raise StageB4ProtocolError("evaluation contract drifted")
    output = _mapping(raw.get("output"), "output")
    expected_output = {
        "root": "results/cr_sitta/p3_stage_b4_full_pilot64_proposal_gate_v1",
        "candidate_phase": "candidate_phase/R0",
        "outer_phase": "outer_phase/R0",
        "aggregate_phase": "aggregate_phase/R0",
        "engineering_phase": "engineering_dry_runs",
    }
    if any(output.get(key) != value for key, value in expected_output.items()):
        raise StageB4ProtocolError("formal output paths drifted")
    if output.get("atomic_no_replace") is not True or output.get("refuse_overwrite") is not True:
        raise StageB4ProtocolError("formal outputs must be immutable")
    _verify_parent_bindings(repository, raw)
    for dataset in DATASETS:
        record = _mapping(datasets[dataset], f"dataset {dataset}")
        checkpoint = _repository_path(
            repository, record["checkpoint_path"], f"{dataset} checkpoint"
        )
        if sha256_file(checkpoint) != _require_sha256(
            record["checkpoint_sha256"], f"{dataset} checkpoint SHA"
        ):
            raise StageB4ProtocolError(f"checkpoint changed: {dataset}")
    implementation = _mapping(raw.get("implementation"), "implementation")
    paths = tuple(implementation.get("critical_code_paths", ()))
    if not paths or len(paths) != len(set(paths)):
        raise StageB4ProtocolError("critical code path list is empty/duplicated")
    for value in paths:
        path = _repository_path(repository, value, "critical code path")
        if path.is_symlink() or not path.is_file():
            raise StageB4ProtocolError(f"critical code path missing/unsafe: {value}")
    for dataset in DATASETS:
        _verify_cache_binding(
            FullPilotContract(repository, Path(), "", raw), dataset
        )


def load_contract(config_path: Path) -> FullPilotContract:
    absolute = config_path if config_path.is_absolute() else REPOSITORY / config_path
    if absolute.is_symlink() or not absolute.is_file():
        raise StageB4ProtocolError(f"config must be a regular file: {absolute}")
    payload = absolute.read_bytes()
    payload_sha256 = _sha256_bytes(payload)
    if payload_sha256 != FROZEN_CONFIG_SHA256:
        raise StageB4ProtocolError(
            "Stage-B4 config bytes differ from the canonical frozen protocol"
        )
    try:
        raw = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise StageB4ProtocolError("Stage-B4 YAML cannot be parsed") from exc
    if not isinstance(raw, dict):
        raise StageB4ProtocolError("Stage-B4 YAML root must be a mapping")
    _validate_contract_semantics(REPOSITORY, raw)
    return FullPilotContract(REPOSITORY, absolute, payload_sha256, raw)


def _assert_contract_unchanged(contract: FullPilotContract) -> None:
    if sha256_file(contract.config_path) != contract.config_sha256:
        raise StageB4ProtocolError("Stage-B4 config changed during execution")
    _validate_contract_semantics(contract.repository, contract.raw)


def _capture_code_hashes(contract: FullPilotContract) -> dict[str, str]:
    return {
        str(raw): sha256_file(
            _repository_path(contract.repository, raw, "critical code path")
        )
        for raw in contract.raw["implementation"]["critical_code_paths"]
    }


def _verify_consumed_payloads(
    contract: FullPilotContract,
    dataset: str,
    *,
    include_outer_target: bool,
) -> None:
    try:
        b3._verify_consumed_dataset_payloads(
            contract, dataset, include_outer_target=include_outer_target
        )
    except Exception as exc:
        raise StageB4ProtocolError(
            f"consumed payload changed: {dataset}"
        ) from exc
    _verify_cache_binding(contract, dataset)


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
            raise StageB4ProtocolError(f"artifact contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in skip:
            continue
        result[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
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
    pre_rename_guard: Any,
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
            "artifact_type": "cr_sitta_p3_stage_b4_completion",
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
        staging, destination, pre_rename_guard=pre_rename_guard
    )


def _artifact_destination(
    contract: FullPilotContract, phase: str, dataset: str | None = None
) -> Path:
    result = contract.output_root / str(contract.raw["output"][f"{phase}_phase"])
    return result / dataset if dataset is not None else result


def _open_memmap(path: Path, *, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype="<f4", shape=shape)


def _runtime_manifest_base(
    contract: FullPilotContract,
    *,
    artifact_type: str,
    phase: str,
    dataset: str | None,
    formal: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "dataset": dataset,
        "formal": formal,
        "development_only": True,
        "paper_result": False,
        "paper_test_result": False,
        "config_path": str(contract.config_path.relative_to(contract.repository)),
        "config_sha256": contract.config_sha256,
        "method_label_accesses": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
    }


_RUNTIME_ENVIRONMENT_FIELDS = (
    "python",
    "numpy",
    "torch",
    "cuda_runtime",
    "cudnn",
    "device",
    "gpu_name",
    "deterministic_algorithms",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "pythonhashseed",
    "cublas_workspace_config",
    "cuda_device_order",
    "cuda_visible_devices",
    "sfs_extension_file",
    "sfs_extension_sha256",
)


def _validated_runtime_environment(
    value: Any, label: str, *, verify_extension_file: bool = False
) -> dict[str, Any]:
    environment = _mapping(value, label)
    if set(environment) != set(_RUNTIME_ENVIRONMENT_FIELDS):
        raise StageB4ProtocolError(f"{label} field set differs")
    _require_sha256(environment["sfs_extension_sha256"], f"{label} SFS SHA")
    extension_path = environment["sfs_extension_file"]
    if not isinstance(extension_path, str) or not extension_path:
        raise StageB4ProtocolError(f"{label} SFS path is invalid")
    if verify_extension_file:
        path = Path(extension_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise StageB4ProtocolError(f"{label} SFS file is missing/unsafe")
        if sha256_file(path) != environment["sfs_extension_sha256"]:
            raise StageB4ProtocolError(f"{label} SFS file hash differs")
    return {field: environment[field] for field in _RUNTIME_ENVIRONMENT_FIELDS}


def _assert_runtime_environment_match(
    candidate_environment: Any, outer_environment: Any
) -> None:
    candidate = _validated_runtime_environment(
        candidate_environment, "candidate runtime environment"
    )
    outer = _validated_runtime_environment(
        outer_environment, "outer runtime environment"
    )
    if candidate != outer:
        differences = tuple(
            field
            for field in _RUNTIME_ENVIRONMENT_FIELDS
            if candidate[field] != outer[field]
        )
        raise StageB4ProtocolError(
            f"candidate/outer runtime environments differ: {differences}"
        )


def verify_artifact(
    path: Path,
    *,
    contract: FullPilotContract,
    phase: str,
    dataset: str | None,
    expected_formal: bool = True,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise StageB4ProtocolError(f"artifact directory missing/unsafe: {path}")
    manifest_path = path / "manifest.json"
    complete_path = path / "COMPLETE.json"
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    artifact_types = {
        "candidate": "cr_sitta_p3_stage_b4_candidate_dataset",
        "outer": "cr_sitta_p3_stage_b4_outer_dataset",
        "aggregate": "cr_sitta_p3_stage_b4_aggregate",
    }
    if phase not in artifact_types:
        raise StageB4ProtocolError(f"unknown artifact phase: {phase}")
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
        or manifest.get("method_label_accesses") != 0
        or manifest.get("validation_payload_opens") != 0
        or manifest.get("test_payload_opens") != 0
    ):
        raise StageB4ProtocolError(f"artifact manifest semantics differ: {path}")
    if (
        complete.get("schema_version") != 1
        or complete.get("artifact_type") != "cr_sitta_p3_stage_b4_completion"
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
        raise StageB4ProtocolError(f"completion receipt differs: {path}")
    expected_files = _mapping(manifest.get("files"), "artifact files")
    actual_files = _file_ledger(path, excluded=("manifest.json", "COMPLETE.json"))
    if actual_files != expected_files:
        raise StageB4ProtocolError(f"artifact file ledger differs: {path}")
    if manifest.get("payload_tree_sha256") != _sha256_bytes(
        _canonical_json_bytes(expected_files)
    ):
        raise StageB4ProtocolError(f"artifact tree digest differs: {path}")

    if phase == "candidate":
        _validated_runtime_environment(
            manifest.get("runtime_environment"),
            "candidate runtime environment",
            verify_extension_file=True,
        )
        boundary = _mapping(manifest.get("method_boundary"), "method boundary")
        if (
            boundary.get("method_label_accesses") != 0
            or boundary.get("outer_target_loader_calls") != 0
            or boundary.get("validation_payload_opens") != 0
            or boundary.get("test_payload_opens") != 0
            or tuple(manifest.get("candidate_ids", ())) != CANDIDATES
            or manifest.get("teacher_probability_role")
            != "sealed_source_identity"
            or manifest.get("b2_candidate_selected") is not False
            or manifest.get("source_state_restored") is not True
            or manifest.get("source_teacher_post_output_hashes_recorded") is not True
            or manifest.get("rejected_no_update_bit_exact_source_enforced") is not True
        ):
            raise StageB4ProtocolError(
                f"candidate method-boundary semantics differ: {path}"
            )
        if expected_formal and (
            tuple(manifest.get("conditions", ()))
            != tuple(_condition_key(*value) for value in CONDITIONS)
            or manifest.get("condition_count") != len(CONDITIONS)
            or manifest.get("image_count_per_condition") != PILOT64_COUNT
            or manifest.get("episode_count")
            != len(CONDITIONS) * PILOT64_COUNT * len(CANDIDATES)
        ):
            raise StageB4ProtocolError(f"formal candidate topology differs: {path}")
        if expected_formal:
            assert dataset is not None
            _verify_formal_candidate_payload(
                path, manifest=manifest, dataset=dataset, contract=contract
            )
    elif phase == "outer":
        _validated_runtime_environment(
            manifest.get("runtime_environment"),
            "outer runtime environment",
            verify_extension_file=True,
        )
        if not expected_formal:
            raise StageB4ProtocolError("outer artifacts must always be formal")
        if (
            manifest.get("condition_count") != len(CONDITIONS)
            or manifest.get("candidate_count") != len(CANDIDATES)
            or manifest.get("image_count_per_condition") != PILOT64_COUNT
            or manifest.get("cell_summary_count")
            != len(CONDITIONS) * len(CANDIDATES)
            or manifest.get("episode_summary_count")
            != len(CONDITIONS) * PILOT64_COUNT * len(CANDIDATES)
            or manifest.get("alignment_record_count")
            != len(CONDITIONS) * PILOT64_COUNT * len(CANDIDATES)
            or manifest.get("outer_target_accesses")
            != len(CONDITIONS) * PILOT64_COUNT
            or manifest.get("adaptation_gradient_uses_labels") is not False
            or manifest.get(
                "candidate_output_hash_proofs_verified_before_target_access"
            )
            is not True
            or manifest.get(
                "candidate_runtime_environment_matched_before_target_access"
            )
            is not True
        ):
            raise StageB4ProtocolError(f"formal outer topology differs: {path}")
        assert dataset is not None
        _load_and_validate_outer_records(path, dataset=dataset)
    else:
        if not expected_formal or dataset is not None:
            raise StageB4ProtocolError("aggregate artifact must be formal/global")
        receipt = _load_json(path / "science_decision_receipt.json")
        if (
            manifest.get("dataset_count") != len(DATASETS)
            or manifest.get("condition_count_per_dataset") != len(CONDITIONS)
            or manifest.get("candidate_count") != len(CANDIDATES)
            or manifest.get("cell_summary_count")
            != len(DATASETS) * len(CONDITIONS) * len(CANDIDATES)
            or manifest.get("episode_summary_count")
            != len(DATASETS) * len(CONDITIONS) * PILOT64_COUNT * len(CANDIDATES)
            or manifest.get("scientific_status")
            != receipt.get("scientific_status")
            or manifest.get("stage_b5_allowed")
            is not receipt.get("r1_r2_allowed")
            or manifest.get("selected_for_r1_r2")
            != receipt.get("selected_for_r1_r2")
            or receipt.get("paper_result") is not False
            or receipt.get("test") is not False
        ):
            raise StageB4ProtocolError(f"formal aggregate semantics differ: {path}")
    return manifest


def _existing_complete_or_raise(
    destination: Path,
    *,
    contract: FullPilotContract,
    phase: str,
    dataset: str | None,
) -> dict[str, Any] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    try:
        return verify_artifact(
            destination,
            contract=contract,
            phase=phase,
            dataset=dataset,
        )
    except Exception as exc:
        raise ExistingArtifactError(
            f"refusing to overwrite incomplete/conflicting artifact: {destination}"
        ) from exc


def _serialise_step_result(value: Any) -> dict[str, Any]:
    if not is_dataclass(value):
        raise StageB4ProtocolError("proposal result must be a dataclass")
    raw = asdict(value)
    # Canonical JSON refuses NaN/Inf; those are represented as explicit nulls
    # plus stable reason codes in normal build-rejection records.
    def sanitise(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(key): sanitise(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitise(child) for child in item]
        return item

    result = sanitise(raw)
    if not isinstance(result, dict):  # pragma: no cover - defensive.
        raise StageB4ProtocolError("proposal result serialization failed")
    return result


def _component_count(mask: Any, *, connectivity: int, min_area: int) -> int:
    from metrics.connected_components import extract_connected_components

    array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    array = np.asarray(array)
    if array.shape != (1, 1, 256, 256):
        raise StageB4ProtocolError(f"component mask shape differs: {array.shape}")
    # The config uses the human-readable 8-neighbour value; the metric helper
    # uses scipy's rank-2 convention where 2 denotes 8-neighbour connectivity.
    metric_connectivity = 2 if connectivity == 8 else 1
    return len(
        extract_connected_components(
            array[0, 0], connectivity=metric_connectivity, min_area=min_area
        )
    )


def _episode_diagnostics(
    *,
    contract: FullPilotContract,
    source_logits: Any,
    post_logits: Any,
    teacher: Any,
    foreground_weight: Any,
    background_weight: Any,
) -> dict[str, Any]:
    import torch

    if not bool(torch.isfinite(source_logits).all().item()) or not bool(
        torch.isfinite(post_logits).all().item()
    ):
        raise StageB4ProtocolError("source/post logits contain NaN/Inf")
    diagnostics = b3._probability_diagnostics(
        source_logits, post_logits, teacher, background_weight
    )
    foreground_sum = foreground_weight.sum()
    if float(foreground_sum.item()) > 0.0:
        denominator = foreground_sum + float(contract.raw["proposal"]["epsilon"])
        source_foreground_mass = (
            torch.sigmoid(source_logits) * foreground_weight
        ).sum() / denominator
        post_foreground_mass = (
            torch.sigmoid(post_logits) * foreground_weight
        ).sum() / denominator
        teacher_foreground_mass = (teacher * foreground_weight).sum() / denominator
    else:
        source_foreground_mass = source_logits.new_zeros(())
        post_foreground_mass = post_logits.new_zeros(())
        teacher_foreground_mass = teacher.new_zeros(())
    source_mask = torch.sigmoid(source_logits) > 0.5
    post_mask = torch.sigmoid(post_logits) > 0.5
    component_config = contract.raw["proposal"]["per_attempt_label_free_safety"][
        "connected_component_count"
    ]
    source_components = _component_count(
        source_mask,
        connectivity=int(component_config["connectivity_2d"]),
        min_area=int(component_config["min_component_area"]),
    )
    post_components = _component_count(
        post_mask,
        connectivity=int(component_config["connectivity_2d"]),
        min_area=int(component_config["min_component_area"]),
    )
    maximum_components = max(
        source_components + int(component_config["absolute_allowance"]),
        math.ceil(source_components * float(component_config["source_multiplier"])),
    )
    functional_threshold = float(
        contract.raw["stage_b4_r0_science_gate"]["activity"]
        ["functional_logit_threshold"]
    )
    return {
        **diagnostics,
        "source_connected_component_count": source_components,
        "post_connected_component_count": post_components,
        "connected_component_count_maximum": maximum_components,
        "reliable_foreground_region_empty": bool(
            float(foreground_sum.item()) <= 0.0
        ),
        "teacher_foreground_probability_mass": float(
            teacher_foreground_mass.item()
        ),
        "source_foreground_probability_mass": float(
            source_foreground_mass.item()
        ),
        "post_foreground_probability_mass": float(post_foreground_mass.item()),
        "functional_logit_change_above_threshold": bool(
            diagnostics["max_abs_delta_logit"] > functional_threshold
        ),
    }


def _check_no_update_endpoint(
    *, source_probability: Any, post_probability: Any, accepted_update: bool
) -> bool:
    """Return endpoint equality and fail closed for every rejected proposal."""

    import torch

    bit_exact = bool(torch.equal(post_probability, source_probability))
    if not accepted_update and not bit_exact:
        raise StageB4ProtocolError("no-update endpoint differs from Source")
    return bit_exact


def _vector_norms_by_parameter(
    vector: Any, named_parameters: Sequence[tuple[str, Any]]
) -> dict[str, float]:
    import torch

    offset = 0
    result: dict[str, float] = {}
    for name, parameter in named_parameters:
        count = int(parameter.numel())
        segment = vector[offset : offset + count]
        if int(segment.numel()) != count:
            raise StageB4ProtocolError("proposal vector parameter partition differs")
        result[name] = float(torch.linalg.vector_norm(segment).item())
        offset += count
    if offset != int(vector.numel()) or not all(
        math.isfinite(value) for value in result.values()
    ):
        raise StageB4ProtocolError("proposal vector parameter norms differ")
    return result


def _candidate_episode_from_source(
    *,
    contract: FullPilotContract,
    model: Any,
    adapter: Any,
    film: Any,
    state_manager: Any,
    image: Any,
    student_image: Any,
    teacher: Any,
    uncertainty: Any,
    source_logits: Any,
    candidate_id: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from tta.proposal_runner_v1 import propose_and_backtrack

    objective, space = candidate_id.split("_", 1)
    named_parameters = b3._configure_parameter_space(
        model, adapter, film, space
    )
    region_weights = b3._build_region_weights(
        contract, teacher, uncertainty
    )
    if float(region_weights.background_weight.sum().item()) <= 0.0:
        raise StageB4ProtocolError("reliable-background region is empty")

    def loss_closure():
        logits = b3._forward_logits(
            adapter, model, film, student_image, space
        )
        loss, _terms = b3._objective_loss(
            contract, objective, logits, teacher, region_weights
        )
        return loss

    with torch.no_grad():
        source_student_logits = b3._forward_logits(
            adapter, model, film, student_image, space
        )
        _source_loss, source_loss_terms = b3._objective_loss(
            contract,
            objective,
            source_student_logits,
            teacher,
            region_weights,
        )

    safety_config = contract.raw["proposal"]["per_attempt_label_free_safety"]
    comparison_epsilon = float(contract.raw["proposal"]["epsilon"])

    def safety_closure() -> tuple[bool, str]:
        with torch.no_grad():
            candidate_logits = b3._forward_logits(
                adapter, model, film, image, space
            )
            if not bool(torch.isfinite(candidate_logits).all().item()):
                return False, "nonfinite_logits"
            candidate_probability = torch.sigmoid(candidate_logits)
            if not bool(torch.isfinite(candidate_probability).all().item()):
                return False, "nonfinite_probabilities"
            observed = _episode_diagnostics(
                contract=contract,
                source_logits=source_logits,
                post_logits=candidate_logits,
                teacher=teacher,
                foreground_weight=region_weights.foreground_weight,
                background_weight=region_weights.background_weight,
            )
        failures: list[str] = []
        if (
            observed["post_reliable_background_mass"]
            - observed["source_reliable_background_mass"]
            > float(safety_config["reliable_background_mass_delta_maximum"])
            + comparison_epsilon
        ):
            failures.append("reliable_background_mass")
        if (
            observed["post_foreground_fraction"]
            - observed["source_foreground_fraction"]
            > float(safety_config["predicted_positive_fraction_delta_maximum"])
            + comparison_epsilon
        ):
            failures.append("predicted_positive_fraction")
        if (
            observed["post_connected_component_count"]
            > observed["connected_component_count_maximum"]
        ):
            failures.append("connected_component_count")
        return (not failures), ("passed" if not failures else "+".join(failures))

    space_config = contract.raw["parameter_spaces"][space]
    proposal_config = contract.raw["proposal"]
    execution = propose_and_backtrack(
        named_parameters=named_parameters,
        loss_closure=loss_closure,
        safety_closure=safety_closure,
        relative_radius=space_config["relative_radius"],
        absolute_radius=space_config["absolute_radius"],
        coefficients=tuple(proposal_config["backtracking_coefficients"]),
        armijo_c=float(proposal_config["armijo_c"]),
        epsilon=float(proposal_config["epsilon"]),
        allow_cuda_nondeterministic_backward=image.device.type == "cuda",
    )
    result = execution.result
    with torch.no_grad():
        post_logits = b3._forward_logits(adapter, model, film, image, space)
        post_probability = torch.sigmoid(post_logits)
        post_student_logits = b3._forward_logits(
            adapter, model, film, student_image, space
        )
        _post_loss, post_loss_terms = b3._objective_loss(
            contract,
            objective,
            post_student_logits,
            teacher,
            region_weights,
        )
        diagnostics = _episode_diagnostics(
            contract=contract,
            source_logits=source_logits,
            post_logits=post_logits,
            teacher=teacher,
            foreground_weight=region_weights.foreground_weight,
            background_weight=region_weights.background_weight,
        )
    vector_count = int(space_config["expected_scalar_parameter_count"])
    proxy_gradient = execution.proxy_gradient
    direction = execution.normalized_direction
    if (
        proxy_gradient.ndim != 1
        or direction.ndim != 1
        or proxy_gradient.numel() != vector_count
        or direction.numel() != vector_count
        or not bool(torch.isfinite(proxy_gradient).all().item())
        or not bool(torch.isfinite(direction).all().item())
    ):
        raise StageB4ProtocolError(
            f"proposal vector artifact differs for {candidate_id}"
        )
    serialised_result = _serialise_step_result(result)
    finite = (
        math.isfinite(float(result.loss_before))
        and math.isfinite(float(result.loss_after))
        and not str(result.reason).startswith("nonfinite")
        and bool(torch.isfinite(post_probability).all().item())
    )
    # The exact gate schema requires finite decimal fields.  A non-finite
    # scientific no-step therefore carries explicit finite=false and a zero
    # serialization sentinel; the detailed proposal result retains null plus
    # its stable non-finite reason code.
    loss_before_record = float(result.loss_before) if finite else 0.0
    loss_after_record = float(result.loss_after) if finite else 0.0
    output = {
        "candidate_id": candidate_id,
        "teacher_candidate_id": "sealed_source_identity",
        "objective": objective,
        "parameter_space": space,
        "accepted_update": bool(result.accepted),
        "no_update": not bool(result.accepted),
        "finite": finite,
        "proposal_loss_before": loss_before_record,
        "proposal_loss_after": loss_after_record,
        "proposal_loss_strict_decrease": bool(
            result.loss_after < result.loss_before
        ),
        "loss_terms_before": source_loss_terms,
        "loss_terms_after": post_loss_terms,
        "foreground_weight_sum": float(
            region_weights.foreground_weight.sum().item()
        ),
        "background_weight_sum": float(
            region_weights.background_weight.sum().item()
        ),
        "proxy_gradient_norm_by_group": {
            space: float(result.gradient_norm)
        },
        "proposal_direction_norm_by_group": {
            space: float(result.normalized_step_norm)
        },
        "proxy_gradient_norm_by_parameter": _vector_norms_by_parameter(
            proxy_gradient, named_parameters
        ),
        "proposal_direction_norm_by_parameter": _vector_norms_by_parameter(
            direction, named_parameters
        ),
        "proposal_step": serialised_result,
        **diagnostics,
    }
    return (
        post_probability.detach().cpu().to(dtype=torch.float32),
        proxy_gradient.detach().cpu().to(dtype=torch.float32),
        direction.detach().cpu().to(dtype=torch.float32),
        output,
    )


def _candidate_episode(
    *,
    contract: FullPilotContract,
    model: Any,
    adapter: Any,
    film: Any,
    state_manager: Any,
    image: Any,
    student_image: Any,
    teacher: Any,
    uncertainty: Any,
    source_logits: Any,
    candidate_id: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Execute one proposal and restore Source on every possible exit path."""

    state_manager.reset_to_source()
    state_manager.assert_source_state()
    active_error: BaseException | None = None
    try:
        return _candidate_episode_from_source(
            contract=contract,
            model=model,
            adapter=adapter,
            film=film,
            state_manager=state_manager,
            image=image,
            student_image=student_image,
            teacher=teacher,
            uncertainty=uncertainty,
            source_logits=source_logits,
            candidate_id=candidate_id,
        )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        try:
            state_manager.reset_to_source()
            state_manager.assert_source_state()
        except BaseException as reset_exc:
            if active_error is None:
                raise StageB4ProtocolError(
                    "candidate episode failed to restore exact Source state"
                ) from reset_exc
            active_error.add_note(
                "candidate episode Source restoration also failed: "
                f"{type(reset_exc).__name__}: {reset_exc}"
            )


def _execute_candidate_payload(
    staging: Path,
    *,
    contract: FullPilotContract,
    dataset: str,
    device_name: str,
    image_limit: int,
    conditions: Sequence[tuple[str, int]],
    formal: bool,
) -> dict[str, Any]:
    import torch
    from tta.views import validated_student_perturbations

    code_sha256 = _capture_code_hashes(contract)
    try:
        (
            _source_runner,
            model,
            adapter,
            film,
            state_manager,
            device,
            checkpoint_wrapper,
        ) = b3._build_runtime(contract, dataset, device_name)
        runtime_environment = b3._runtime_environment_receipt(torch, device)
    except Exception as exc:
        raise StageB4ProtocolError("cannot initialize frozen Source runtime") from exc
    source_state_sha256 = state_manager.source_fingerprint.full_sha256
    teacher_manifest = _teacher_manifest(contract, dataset)
    pilot64_ids = tuple(str(value) for value in teacher_manifest["image_ids"])
    image_ids = pilot64_ids[:image_limit]
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
    proxy_gradients = {
        "P2": _open_memmap(
            staging / "proxy_gradients_P2.npy",
            shape=(condition_count, image_limit, len(OBJECTIVES), 416),
        ),
        "DecoderFiLM": _open_memmap(
            staging / "proxy_gradients_DecoderFiLM.npy",
            shape=(condition_count, image_limit, len(OBJECTIVES), 32),
        ),
    }
    proposal_directions = {
        "P2": _open_memmap(
            staging / "proposal_directions_P2.npy",
            shape=(condition_count, image_limit, len(OBJECTIVES), 416),
        ),
        "DecoderFiLM": _open_memmap(
            staging / "proposal_directions_DecoderFiLM.npy",
            shape=(condition_count, image_limit, len(OBJECTIVES), 32),
        ),
    }
    episode_records: list[dict[str, Any]] = []
    source_cross_checks: list[dict[str, Any]] = []
    started = time.perf_counter()

    for condition_index, (corruption, severity) in enumerate(conditions):
        condition = _condition_key(corruption, severity)
        try:
            method_dataset = b3._method_input_dataset(contract, dataset, condition)
            condition_record = b3._teacher_condition_record(
                teacher_manifest, condition
            )
            teacher_uncertainty = b3._verified_teacher_array(
                contract,
                dataset,
                teacher_manifest,
                condition_record,
                "view_uncertainty",
            )
            teacher_source = b3._verified_teacher_array(
                contract,
                dataset,
                teacher_manifest,
                condition_record,
                "source_probabilities",
            )
        except Exception as exc:
            raise StageB4ProtocolError(
                f"cannot open label-free input: {dataset}/{condition}"
            ) from exc
        if len(method_dataset) != PILOT64_COUNT:
            raise StageB4ProtocolError(
                f"cache does not contain Pilot64: {dataset}/{condition}"
            )
        if tuple(teacher_uncertainty.shape) != (64, 2, *SOURCE_SHAPE):
            raise StageB4ProtocolError("teacher uncertainty shape differs")
        if tuple(teacher_source.shape) != (64, *SOURCE_SHAPE):
            raise StageB4ProtocolError("teacher Source shape differs")

        for image_index in range(image_limit):
            sample = dict(method_dataset[image_index])
            if set(sample) != {
                "image",
                "image_id",
                "original_size",
                "dataset",
                "corruption",
                "severity",
                "seed",
            }:
                raise StageB4ProtocolError("method-facing field set drifted")
            image_id = str(sample["image_id"])
            if image_id != image_ids[image_index]:
                raise StageB4ProtocolError("Pilot64 image order differs")
            if (
                sample["dataset"] != dataset
                or sample["corruption"] != corruption
                or int(sample["severity"]) != severity
                or int(sample["seed"]) != int(contract.raw["scope"]["seed"])
            ):
                raise StageB4ProtocolError("method-facing sample metadata differs")
            image_cpu = sample["image"]
            if (
                tuple(image_cpu.shape) != (3, 256, 256)
                or image_cpu.dtype != torch.float32
                or not bool(torch.isfinite(image_cpu).all().item())
            ):
                raise StageB4ProtocolError("method-facing image tensor is invalid")
            image_reference = image_cpu.clone()
            image = image_cpu.unsqueeze(0).to(device, non_blocking=False)
            student_image = student_perturbation.forward(image)
            if tuple(student_image.shape) != tuple(image.shape) or not bool(
                torch.isfinite(student_image).all().item()
            ):
                raise StageB4ProtocolError("student perturbation is invalid")
            teacher = torch.from_numpy(
                np.array(teacher_source[image_index], dtype=np.float32, copy=True)
            ).unsqueeze(0).to(device)
            uncertainty = torch.from_numpy(
                np.array(
                    teacher_uncertainty[
                        image_index,
                        int(contract.raw["teacher"]["uncertainty_axis_index"]),
                    ],
                    dtype=np.float32,
                    copy=True,
                )
            ).unsqueeze(0).to(device)
            if teacher.requires_grad or uncertainty.requires_grad:
                raise StageB4ProtocolError("teacher tensors must be detached")

            state_manager.reset_to_source()
            adapter.set_source_eval_mode()
            with torch.no_grad():
                source_logits = adapter.forward_logits(image)
                source_probability = torch.sigmoid(source_logits)
            source_max_abs = float((source_probability - teacher).abs().max().item())
            if source_max_abs > 1.0e-6:
                raise StageB4ProtocolError(
                    "runtime Source differs from sealed teacher Source: "
                    f"{source_max_abs}"
                )
            source_probabilities[condition_index, image_index] = (
                source_probability[0].cpu().numpy().astype("<f4", copy=False)
            )
            source_output_sha256 = b3._raw_array_sha256(
                source_probability[0].cpu().numpy()
            )
            teacher_output_sha256 = b3._raw_array_sha256(
                teacher[0].cpu().numpy()
            )
            if source_output_sha256 != teacher_output_sha256:
                raise StageB4ProtocolError(
                    "runtime Source and sealed teacher are not bit-exact"
                )
            source_cross_checks.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "image_index": image_index,
                    "image_id": image_id,
                    "runtime_vs_teacher_source_max_abs": source_max_abs,
                    "source_output_sha256": source_output_sha256,
                    "teacher_output_sha256": teacher_output_sha256,
                    "passed": True,
                }
            )

            for candidate_index, candidate_id in enumerate(CANDIDATES):
                objective, space = candidate_id.split("_", 1)
                objective_index = OBJECTIVES.index(objective)
                post, proxy, direction, diagnostics = _candidate_episode(
                    contract=contract,
                    model=model,
                    adapter=adapter,
                    film=film,
                    state_manager=state_manager,
                    image=image,
                    student_image=student_image,
                    teacher=teacher,
                    uncertainty=uncertainty,
                    source_logits=source_logits,
                    candidate_id=candidate_id,
                )
                post_probabilities[
                    condition_index, image_index, candidate_index
                ] = post[0].numpy().astype("<f4", copy=False)
                proxy_gradients[space][
                    condition_index, image_index, objective_index
                ] = proxy.numpy().astype("<f4", copy=False)
                proposal_directions[space][
                    condition_index, image_index, objective_index
                ] = direction.numpy().astype("<f4", copy=False)
                try:
                    no_update_bit_exact_source = _check_no_update_endpoint(
                        source_probability=source_probability.detach().cpu(),
                        post_probability=post,
                        accepted_update=bool(diagnostics["accepted_update"]),
                    )
                except StageB4ProtocolError as exc:
                    raise StageB4ProtocolError(
                        f"no-update endpoint differs from Source: {candidate_id}"
                    ) from exc
                reset_fingerprint = state_manager.assert_source_state()
                episode_records.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "corruption_family": corruption,
                        "severity": severity,
                        "image_index": image_index,
                        "image_id": image_id,
                        "seed": int(sample["seed"]),
                        "candidate_id": candidate_id,
                        "input_tensor_sha256": b3._raw_array_sha256(
                            image_cpu.numpy()
                        ),
                        "teacher_probability_sha256": b3._raw_array_sha256(
                            teacher[0].cpu().numpy()
                        ),
                        "teacher_uncertainty_sha256": b3._raw_array_sha256(
                            uncertainty[0].cpu().numpy()
                        ),
                        "source_state_sha256": source_state_sha256,
                        "post_reset_state_sha256": reset_fingerprint.full_sha256,
                        "source_output_sha256": source_output_sha256,
                        "post_output_sha256": b3._raw_array_sha256(
                            post[0].numpy()
                        ),
                        "no_update_bit_exact_source": no_update_bit_exact_source,
                        "method_label_accesses": 0,
                        "validation_payload_opens": 0,
                        "test_payload_opens": 0,
                        **diagnostics,
                    }
                )
            if not torch.equal(image_cpu, image_reference):
                raise StageB4ProtocolError("candidate phase modified its input")
            state_manager.assert_source_state()

    arrays = (
        source_probabilities,
        post_probabilities,
        *proxy_gradients.values(),
        *proposal_directions.values(),
    )
    for array in arrays:
        array.flush()
    del source_probabilities, post_probabilities, proxy_gradients, proposal_directions
    _write_jsonl(staging / "episode_diagnostics.jsonl", episode_records)
    _write_jsonl(staging / "source_cross_checks.jsonl", source_cross_checks)
    state_manager.assert_source_state()
    if formal:
        _verify_consumed_payloads(
            contract, dataset, include_outer_target=False
        )
    _assert_contract_unchanged(contract)
    if _capture_code_hashes(contract) != code_sha256:
        raise StageB4ProtocolError("critical code changed during candidate phase")
    return {
        **_runtime_manifest_base(
            contract,
            artifact_type="cr_sitta_p3_stage_b4_candidate_dataset",
            phase="candidate",
            dataset=dataset,
            formal=formal,
        ),
        "checkpoint_role": "best_miou",
        "checkpoint_sha256": contract.raw["datasets"][dataset]["checkpoint_sha256"],
        "checkpoint_wrapper": checkpoint_wrapper,
        "runtime_environment": runtime_environment,
        "source_state_sha256": source_state_sha256,
        "source_state_restored": True,
        "conditions": [_condition_key(*value) for value in conditions],
        "condition_count": condition_count,
        "image_ids": list(image_ids),
        "image_count_per_condition": image_limit,
        "candidate_ids": list(CANDIDATES),
        "objective_ids": list(OBJECTIVES),
        "parameter_spaces": list(SPACES),
        "teacher_probability_role": "sealed_source_identity",
        "b2_candidate_selected": False,
        "source_teacher_post_output_hashes_recorded": True,
        "rejected_no_update_bit_exact_source_enforced": True,
        "student_perturbation": student_perturbation.name,
        "episode_count": len(episode_records),
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
            "proxy_gradients_P2": [condition_count, image_limit, 2, 416],
            "proposal_directions_P2": [condition_count, image_limit, 2, 416],
            "proxy_gradients_DecoderFiLM": [condition_count, image_limit, 2, 32],
            "proposal_directions_DecoderFiLM": [
                condition_count,
                image_limit,
                2,
                32,
            ],
        },
        "code_sha256": code_sha256,
        "wall_time_seconds": time.perf_counter() - started,
    }


def run_candidate(
    contract: FullPilotContract,
    *,
    dataset: str,
    device_name: str,
    max_images: int | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise StageB4ProtocolError(f"unsupported dataset: {dataset}")
    formal = max_images is None and condition is None
    if max_images is not None and not 1 <= max_images <= PILOT64_COUNT:
        raise StageB4ProtocolError("--max-images must lie in [1,64]")
    if condition is None:
        conditions = CONDITIONS
    else:
        matches = tuple(
            value for value in CONDITIONS if _condition_key(*value) == condition
        )
        if len(matches) != 1:
            raise StageB4ProtocolError(f"unknown condition: {condition}")
        conditions = matches
    image_limit = PILOT64_COUNT if max_images is None else max_images
    if formal:
        destination = _artifact_destination(contract, "candidate", dataset)
        existing = _existing_complete_or_raise(
            destination,
            contract=contract,
            phase="candidate",
            dataset=dataset,
        )
        if existing is not None:
            if existing.get("code_sha256") != _capture_code_hashes(contract):
                raise StageB4ProtocolError(
                    f"critical code changed after candidate publication: {dataset}"
                )
            _verify_consumed_payloads(
                contract, dataset, include_outer_target=False
            )
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
                _verify_consumed_payloads(
                    contract, dataset, include_outer_target=False
                )
            if _capture_code_hashes(contract) != manifest["code_sha256"]:
                raise StageB4ProtocolError(
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


def _candidate_diagnostics_by_key(
    candidate_root: Path,
    *,
    dataset: str,
    image_ids: Sequence[str],
    expected_source_state_sha256: str,
    seed: int,
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    condition_lookup = {
        _condition_key(corruption, severity): (corruption, severity)
        for corruption, severity in CONDITIONS
    }
    for value in _read_jsonl(candidate_root / "episode_diagnostics.jsonl"):
        accepted = value.get("accepted_update")
        no_update = value.get("no_update")
        bit_exact = value.get("no_update_bit_exact_source")
        if (
            type(accepted) is not bool
            or type(no_update) is not bool
            or no_update is accepted
            or type(bit_exact) is not bool
        ):
            raise StageB4ProtocolError("candidate update-state audit differs")
        source_sha = _require_sha256(
            value.get("source_output_sha256"), "candidate Source output SHA"
        )
        teacher_sha = _require_sha256(
            value.get("teacher_probability_sha256"),
            "candidate teacher output SHA",
        )
        post_sha = _require_sha256(
            value.get("post_output_sha256"), "candidate post output SHA"
        )
        if source_sha != teacher_sha:
            raise StageB4ProtocolError(
                "candidate Source/teacher output hashes differ"
            )
        if not accepted and (not bit_exact or post_sha != source_sha):
            raise StageB4ProtocolError(
                "candidate no-update endpoint is not bit-exact Source"
            )
        condition = str(value.get("condition"))
        candidate_id = str(value.get("candidate_id"))
        image_index = value.get("image_index")
        if (
            condition not in condition_lookup
            or candidate_id not in CANDIDATES
            or type(image_index) is not int
            or not 0 <= image_index < PILOT64_COUNT
        ):
            raise StageB4ProtocolError("candidate diagnostic identity differs")
        corruption, severity = condition_lookup[condition]
        expected_objective, expected_space = candidate_id.split("_", 1)
        if (
            value.get("dataset") != dataset
            or value.get("corruption_family") != corruption
            or value.get("severity") != severity
            or value.get("image_id") != image_ids[image_index]
            or value.get("seed") != seed
            or value.get("teacher_candidate_id") != "sealed_source_identity"
            or value.get("objective") != expected_objective
            or value.get("parameter_space") != expected_space
            or value.get("method_label_accesses") != 0
            or value.get("validation_payload_opens") != 0
            or value.get("test_payload_opens") != 0
        ):
            raise StageB4ProtocolError("candidate diagnostic metadata differs")
        source_state_sha = _require_sha256(
            value.get("source_state_sha256"), "candidate Source-state SHA"
        )
        reset_state_sha = _require_sha256(
            value.get("post_reset_state_sha256"), "candidate reset-state SHA"
        )
        if (
            source_state_sha != expected_source_state_sha256
            or reset_state_sha != expected_source_state_sha256
        ):
            raise StageB4ProtocolError("candidate reset-state identity differs")
        _require_sha256(
            value.get("input_tensor_sha256"), "candidate input tensor SHA"
        )
        _require_sha256(
            value.get("teacher_uncertainty_sha256"),
            "candidate uncertainty SHA",
        )
        key = (
            condition,
            image_index,
            candidate_id,
        )
        if key in result:
            raise StageB4ProtocolError(f"duplicate candidate diagnostic: {key}")
        result[key] = value
    expected_keys = {
        (_condition_key(*condition), image_index, candidate_id)
        for condition in CONDITIONS
        for image_index in range(PILOT64_COUNT)
        for candidate_id in CANDIDATES
    }
    if set(result) != expected_keys:
        raise StageB4ProtocolError("candidate diagnostic Cartesian keyset differs")
    return result


def _verify_candidate_output_hash_proofs(
    *,
    diagnostics: Mapping[tuple[str, int, str], Mapping[str, Any]],
    source_probabilities: Any,
    post_probabilities: Any,
) -> None:
    for condition_index, condition_tuple in enumerate(CONDITIONS):
        condition = _condition_key(*condition_tuple)
        for image_index in range(PILOT64_COUNT):
            source = np.asarray(source_probabilities[condition_index, image_index])
            source_sha = b3._raw_array_sha256(source)
            for candidate_index, candidate_id in enumerate(CANDIDATES):
                value = diagnostics[(condition, image_index, candidate_id)]
                post = np.asarray(
                    post_probabilities[condition_index, image_index, candidate_index]
                )
                if (
                    value["source_output_sha256"] != source_sha
                    or value["post_output_sha256"] != b3._raw_array_sha256(post)
                ):
                    raise StageB4ProtocolError(
                        "candidate output hash proof differs from stored arrays"
                    )
                if not value["accepted_update"] and not np.array_equal(source, post):
                    raise StageB4ProtocolError(
                        "candidate rejected output array is not bit-exact Source"
                    )


def _verify_formal_candidate_payload(
    candidate_root: Path,
    *,
    manifest: Mapping[str, Any],
    dataset: str,
    contract: FullPilotContract,
) -> None:
    image_ids = tuple(str(value) for value in manifest.get("image_ids", ()))
    if len(image_ids) != PILOT64_COUNT or len(set(image_ids)) != PILOT64_COUNT:
        raise StageB4ProtocolError("candidate manifest Pilot64 IDs differ")
    if _compact_ordered_ids_sha256(image_ids) != contract.raw["datasets"][dataset][
        "ordered_pilot64_image_ids_sha256"
    ]:
        raise StageB4ProtocolError("candidate manifest Pilot64 ID hash differs")
    source_probabilities = np.load(
        candidate_root / "source_probabilities.npy", mmap_mode="r", allow_pickle=False
    )
    post_probabilities = np.load(
        candidate_root / "post_probabilities.npy", mmap_mode="r", allow_pickle=False
    )
    if tuple(source_probabilities.shape) != (13, 64, *SOURCE_SHAPE):
        raise StageB4ProtocolError("candidate Source probability shape differs")
    if tuple(post_probabilities.shape) != (13, 64, 4, *SOURCE_SHAPE):
        raise StageB4ProtocolError("candidate post probability shape differs")
    diagnostics = _candidate_diagnostics_by_key(
        candidate_root,
        dataset=dataset,
        image_ids=image_ids,
        expected_source_state_sha256=_require_sha256(
            manifest.get("source_state_sha256"),
            "candidate manifest Source-state SHA",
        ),
        seed=int(contract.raw["scope"]["seed"]),
    )
    _verify_candidate_output_hash_proofs(
        diagnostics=diagnostics,
        source_probabilities=source_probabilities,
        post_probabilities=post_probabilities,
    )


_COUNT_FIELDS = (
    "intersection_pixels",
    "false_positive_pixels",
    "false_negative_pixels",
    "true_negative_pixels",
    "predicted_positive_pixels",
    "target_positive_pixels",
    "detected_targets",
    "total_targets",
    "false_alarm_pixels",
    "total_image_pixels",
    "image_count",
)


def _gate_cell_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(value.get("source_counts"), "source counts")
    adapted = _mapping(value.get("adapted_counts"), "adapted counts")
    result = {
        "candidate_id": str(value["candidate_id"]),
        "dataset": str(value["dataset"]),
        "condition": str(value["condition"]),
        "corruption_family": str(value["corruption_family"]),
        "severity": f"S{int(value['severity'])}",
        "episode_count": int(value["episode_count"]),
        "source_counts": {field: int(source[field]) for field in _COUNT_FIELDS},
        "adapted_counts": {field: int(adapted[field]) for field in _COUNT_FIELDS},
    }
    try:
        from analysis.p3_stage_b4_science_gate_v1 import CELL_FIELDS
    except ImportError:
        return result
    if set(result) != set(CELL_FIELDS):
        raise StageB4ProtocolError("B4 gate cell projection fields differ")
    return result


def _gate_episode_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "candidate_id": str(value["candidate_id"]),
        "dataset": str(value["dataset"]),
        "condition": str(value["condition"]),
        "episode_index": int(value["episode_index"]),
        "proxy_gradient_nonzero": bool(value["proxy_gradient_nonzero"]),
        "task_gradient_nonzero": bool(value["task_gradient_nonzero"]),
        "gradient_cosine": float(value["gradient_cosine"]),
        "accepted_update": bool(value["accepted_update"]),
        "finite": bool(value["finite"]),
        "maximum_absolute_logit_delta": float(
            value["maximum_absolute_logit_delta"]
        ),
        "proposal_loss_before": float(value["proposal_loss_before"]),
        "proposal_loss_after": float(value["proposal_loss_after"]),
        "threshold_crossing_count": int(value["threshold_crossing_count"]),
    }
    try:
        from analysis.p3_stage_b4_science_gate_v1 import EPISODE_FIELDS
    except ImportError:
        return result
    if set(result) != set(EPISODE_FIELDS):
        raise StageB4ProtocolError("B4 gate episode projection fields differ")
    return result


def _subset_endpoint(
    probabilities: Any,
    targets: Any,
    image_ids: Sequence[str],
    indices: Sequence[int],
) -> dict[str, Any] | None:
    if not indices:
        return None
    subset_ids = tuple(image_ids[index] for index in indices)
    result = b3._evaluation_result(
        np.asarray(probabilities)[list(indices)],
        np.asarray(targets)[list(indices)],
        subset_ids,
    )
    return b3._endpoint_summary(result)


def _execute_outer_payload(
    staging: Path,
    *,
    contract: FullPilotContract,
    dataset: str,
    device_name: str,
    candidate_root: Path,
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    code_sha256 = _capture_code_hashes(contract)
    if candidate_manifest.get("formal") is not True:
        raise StageB4ProtocolError("outer phase refuses engineering candidate")
    if tuple(candidate_manifest.get("conditions", ())) != tuple(
        _condition_key(*value) for value in CONDITIONS
    ):
        raise StageB4ProtocolError("candidate condition topology differs")
    if tuple(candidate_manifest.get("candidate_ids", ())) != CANDIDATES:
        raise StageB4ProtocolError("candidate axis differs")
    image_ids = tuple(str(value) for value in candidate_manifest.get("image_ids", ()))
    teacher_ids = tuple(
        str(value) for value in _teacher_manifest(contract, dataset)["image_ids"]
    )
    if image_ids != teacher_ids or len(image_ids) != PILOT64_COUNT:
        raise StageB4ProtocolError("candidate Pilot64 IDs differ")

    source_probabilities = np.load(
        candidate_root / "source_probabilities.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    post_probabilities = np.load(
        candidate_root / "post_probabilities.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    proxy_gradients = {
        "P2": np.load(
            candidate_root / "proxy_gradients_P2.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        "DecoderFiLM": np.load(
            candidate_root / "proxy_gradients_DecoderFiLM.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
    }
    proposal_directions = {
        "P2": np.load(
            candidate_root / "proposal_directions_P2.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        "DecoderFiLM": np.load(
            candidate_root / "proposal_directions_DecoderFiLM.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
    }
    if tuple(source_probabilities.shape) != (13, 64, *SOURCE_SHAPE):
        raise StageB4ProtocolError("candidate Source probability shape differs")
    if tuple(post_probabilities.shape) != (13, 64, 4, *SOURCE_SHAPE):
        raise StageB4ProtocolError("candidate post probability shape differs")
    expected_vectors = {
        "P2": (13, 64, 2, 416),
        "DecoderFiLM": (13, 64, 2, 32),
    }
    for space in SPACES:
        if tuple(proxy_gradients[space].shape) != expected_vectors[space] or tuple(
            proposal_directions[space].shape
        ) != expected_vectors[space]:
            raise StageB4ProtocolError(f"candidate vector shape differs: {space}")
    diagnostics = _candidate_diagnostics_by_key(
        candidate_root,
        dataset=dataset,
        image_ids=image_ids,
        expected_source_state_sha256=_require_sha256(
            candidate_manifest.get("source_state_sha256"),
            "candidate manifest Source-state SHA",
        ),
        seed=int(contract.raw["scope"]["seed"]),
    )
    _verify_candidate_output_hash_proofs(
        diagnostics=diagnostics,
        source_probabilities=source_probabilities,
        post_probabilities=post_probabilities,
    )

    try:
        (
            _source_runner,
            model,
            adapter,
            film,
            state_manager,
            device,
            _checkpoint_wrapper,
        ) = b3._build_runtime(contract, dataset, device_name)
        runtime_environment = b3._runtime_environment_receipt(torch, device)
    except Exception as exc:
        raise StageB4ProtocolError("cannot initialize outer Source runtime") from exc
    _assert_runtime_environment_match(
        candidate_manifest.get("runtime_environment"), runtime_environment
    )

    # This is deliberately the first target deserialization in the B4 runner.
    # Candidate completion, payload topology, no-update endpoint proofs, and
    # the candidate/outer runtime-environment binding have all passed above.
    try:
        targets = b3._load_outer_targets(contract, dataset)
    except Exception as exc:
        raise StageB4ProtocolError("cannot load train-only outer targets") from exc
    if tuple(targets.shape) != (64, 1, 256, 256):
        raise StageB4ProtocolError("outer target shape differs")
    targets_array = np.asarray(targets, dtype=np.float32)
    if not np.isfinite(targets_array).all():
        raise StageB4ProtocolError("outer target contains NaN/Inf")

    alignment_records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    outer_accesses = len(CONDITIONS) * PILOT64_COUNT
    started = time.perf_counter()

    for condition_index, (corruption, severity) in enumerate(CONDITIONS):
        condition = _condition_key(corruption, severity)
        try:
            method_dataset = b3._method_input_dataset(contract, dataset, condition)
        except Exception as exc:
            raise StageB4ProtocolError(
                f"cannot load outer image input: {dataset}/{condition}"
            ) from exc
        task_gradients: dict[tuple[int, str], Any] = {}
        task_audits: dict[tuple[int, str], Any] = {}
        for image_index, image_id in enumerate(image_ids):
            sample = dict(method_dataset[image_index])
            if str(sample["image_id"]) != image_id:
                raise StageB4ProtocolError("outer image ID order differs")
            image = sample["image"].unsqueeze(0).to(device)
            target = torch.from_numpy(
                np.array(targets_array[image_index], dtype=np.float32, copy=True)
            ).unsqueeze(0).to(device)
            for space in SPACES:
                try:
                    gradient, audit = b3._task_gradient(
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
                except Exception as exc:
                    raise StageB4ProtocolError(
                        f"outer task gradient failed: {condition}/{image_id}/{space}"
                    ) from exc
                task_gradients[(image_index, space)] = gradient
                task_audits[(image_index, space)] = audit

        source_result = b3._evaluation_result(
            source_probabilities[condition_index], targets_array, image_ids
        )
        source_summary = b3._endpoint_summary(source_result)
        for candidate_index, candidate_id in enumerate(CANDIDATES):
            objective, space = candidate_id.split("_", 1)
            objective_index = OBJECTIVES.index(objective)
            candidate_result = b3._evaluation_result(
                post_probabilities[condition_index, :, candidate_index],
                targets_array,
                image_ids,
            )
            candidate_summary = b3._endpoint_summary(candidate_result)
            accepted_indices: list[int] = []
            no_update_indices: list[int] = []
            valid_cosines: list[float] = []
            for image_index, image_id in enumerate(image_ids):
                proxy = torch.from_numpy(
                    np.array(
                        proxy_gradients[space][
                            condition_index, image_index, objective_index
                        ],
                        dtype=np.float64,
                        copy=True,
                    )
                )
                direction = torch.from_numpy(
                    np.array(
                        proposal_directions[space][
                            condition_index, image_index, objective_index
                        ],
                        dtype=np.float64,
                        copy=True,
                    )
                )
                task = task_gradients[(image_index, space)]
                proxy_norm = float(torch.linalg.vector_norm(proxy).item())
                task_norm = float(torch.linalg.vector_norm(task).item())
                direction_norm = float(torch.linalg.vector_norm(direction).item())
                if not all(
                    math.isfinite(value)
                    for value in (proxy_norm, task_norm, direction_norm)
                ):
                    raise StageB4ProtocolError("alignment norm is non-finite")
                proxy_nonzero = proxy_norm > 0.0
                task_nonzero = task_norm > 0.0
                cosine = (
                    float(torch.dot(proxy, task).item() / (proxy_norm * task_norm))
                    if proxy_nonzero and task_nonzero
                    else 0.0
                )
                if not math.isfinite(cosine):
                    raise StageB4ProtocolError("gradient cosine is non-finite")
                if proxy_nonzero and task_nonzero:
                    valid_cosines.append(cosine)
                diagnostic = diagnostics[(condition, image_index, candidate_id)]
                accepted = bool(diagnostic["accepted_update"])
                (accepted_indices if accepted else no_update_indices).append(image_index)
                rich_episode = {
                    "candidate_id": candidate_id,
                    "dataset": dataset,
                    "condition": condition,
                    "corruption_family": corruption,
                    "severity": severity,
                    "episode_index": image_index,
                    "image_id": image_id,
                    "proxy_gradient_nonzero": proxy_nonzero,
                    "task_gradient_nonzero": task_nonzero,
                    "gradient_cosine": cosine,
                    "proposal_direction_norm": direction_norm,
                    "accepted_update": accepted,
                    "finite": bool(diagnostic["finite"]),
                    "maximum_absolute_logit_delta": float(
                        diagnostic["max_abs_delta_logit"]
                    ),
                    "proposal_loss_before": float(
                        diagnostic["proposal_loss_before"]
                    ),
                    "proposal_loss_after": float(
                        diagnostic["proposal_loss_after"]
                    ),
                    "threshold_crossing_count": int(
                        diagnostic["threshold_crossing_count"]
                    ),
                    "outer_target_role": "source_train_pilot64_outer_oracle_only",
                    "adaptation_gradient_uses_labels": False,
                    "task_loss_total": float(
                        task_audits[(image_index, space)]["components"]["total_loss"]
                    ),
                }
                alignment_records.append(rich_episode)
                episode_summaries.append(_gate_episode_summary(rich_episode))
            rich_cell = {
                "candidate_id": candidate_id,
                "dataset": dataset,
                "condition": condition,
                "corruption_family": corruption,
                "severity": severity,
                "episode_count": PILOT64_COUNT,
                "valid_alignment_episode_count": len(valid_cosines),
                "gradient_cosine_valid_mean": (
                    float(sum(valid_cosines) / len(valid_cosines))
                    if valid_cosines
                    else 0.0
                ),
                "source_iou": source_summary["iou"],
                "adapted_iou": candidate_summary["iou"],
                "delta_iou": candidate_summary["iou"] - source_summary["iou"],
                "source_pd": source_summary["pd"],
                "adapted_pd": candidate_summary["pd"],
                "delta_pd": candidate_summary["pd"] - source_summary["pd"],
                "source_fa_per_million": source_summary["fa_per_million"],
                "adapted_fa_per_million": candidate_summary["fa_per_million"],
                "delta_fa_per_million": (
                    candidate_summary["fa_per_million"]
                    - source_summary["fa_per_million"]
                ),
                "source_foreground_fraction": source_summary["foreground_fraction"],
                "adapted_foreground_fraction": candidate_summary["foreground_fraction"],
                "source_counts": source_summary,
                "adapted_counts": candidate_summary,
                "accepted_episode_count": len(accepted_indices),
                "no_update_episode_count": len(no_update_indices),
                "gain_attribution": {
                    "accepted": {
                        "episode_count": len(accepted_indices),
                        "source": _subset_endpoint(
                            source_probabilities[condition_index],
                            targets_array,
                            image_ids,
                            accepted_indices,
                        ),
                        "adapted": _subset_endpoint(
                            post_probabilities[
                                condition_index, :, candidate_index
                            ],
                            targets_array,
                            image_ids,
                            accepted_indices,
                        ),
                    },
                    "no_update": {
                        "episode_count": len(no_update_indices),
                        "source": _subset_endpoint(
                            source_probabilities[condition_index],
                            targets_array,
                            image_ids,
                            no_update_indices,
                        ),
                        "adapted": _subset_endpoint(
                            post_probabilities[
                                condition_index, :, candidate_index
                            ],
                            targets_array,
                            image_ids,
                            no_update_indices,
                        ),
                    },
                },
            }
            cell_summaries.append(rich_cell)

    _write_jsonl(staging / "alignment_records.jsonl", alignment_records)
    _write_jsonl(staging / "episode_summaries.jsonl", episode_summaries)
    _write_jsonl(staging / "cell_summaries.jsonl", cell_summaries)
    state_manager.assert_source_state()
    _verify_consumed_payloads(contract, dataset, include_outer_target=True)
    _assert_contract_unchanged(contract)
    if _capture_code_hashes(contract) != code_sha256:
        raise StageB4ProtocolError("critical code changed during outer phase")
    return {
        **_runtime_manifest_base(
            contract,
            artifact_type="cr_sitta_p3_stage_b4_outer_dataset",
            phase="outer",
            dataset=dataset,
            formal=True,
        ),
        "candidate_manifest_sha256": sha256_file(candidate_root / "manifest.json"),
        "candidate_complete_sha256": sha256_file(candidate_root / "COMPLETE.json"),
        "condition_count": len(CONDITIONS),
        "candidate_count": len(CANDIDATES),
        "image_count_per_condition": PILOT64_COUNT,
        "cell_summary_count": len(cell_summaries),
        "episode_summary_count": len(episode_summaries),
        "alignment_record_count": len(alignment_records),
        "outer_target_accesses": outer_accesses,
        "outer_target_role": "source_train_pilot64_outer_oracle_only",
        "adaptation_gradient_uses_labels": False,
        "candidate_output_hash_proofs_verified_before_target_access": True,
        "candidate_runtime_environment_matched_before_target_access": True,
        "runtime_environment": runtime_environment,
        "source_state_restored": True,
        "code_sha256": code_sha256,
        "wall_time_seconds": time.perf_counter() - started,
    }


def run_outer(
    contract: FullPilotContract, *, dataset: str, device_name: str
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise StageB4ProtocolError(f"unsupported dataset: {dataset}")
    candidate_root = _artifact_destination(contract, "candidate", dataset)
    candidate_manifest = verify_artifact(
        candidate_root,
        contract=contract,
        phase="candidate",
        dataset=dataset,
    )
    current_code = _capture_code_hashes(contract)
    if candidate_manifest.get("code_sha256") != current_code:
        raise StageB4ProtocolError(
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
        if (
            existing.get("code_sha256") != current_code
            or existing.get("candidate_manifest_sha256")
            != sha256_file(candidate_root / "manifest.json")
            or existing.get("candidate_complete_sha256")
            != sha256_file(candidate_root / "COMPLETE.json")
        ):
            raise StageB4ProtocolError(
                f"existing outer lineage/code differs: {dataset}"
            )
        _assert_runtime_environment_match(
            candidate_manifest.get("runtime_environment"),
            existing.get("runtime_environment"),
        )
        _verify_consumed_payloads(contract, dataset, include_outer_target=True)
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

        def outer_publish_guard() -> None:
            _assert_contract_unchanged(contract)
            _verify_consumed_payloads(
                contract, dataset, include_outer_target=True
            )
            current_candidate = verify_artifact(
                candidate_root,
                contract=contract,
                phase="candidate",
                dataset=dataset,
            )
            _assert_runtime_environment_match(
                current_candidate.get("runtime_environment"),
                manifest.get("runtime_environment"),
            )
            if (
                _capture_code_hashes(contract) != manifest["code_sha256"]
                or manifest["candidate_manifest_sha256"]
                != sha256_file(candidate_root / "manifest.json")
                or manifest["candidate_complete_sha256"]
                != sha256_file(candidate_root / "COMPLETE.json")
            ):
                raise StageB4ProtocolError(
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


def _human_science_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    evaluations = []
    for raw in receipt.get("candidate_evaluations", ()):  # already exact gate output
        aggregate = _mapping(raw.get("aggregate"), "candidate aggregate")

        def fraction(name: str) -> float:
            value = _mapping(aggregate[name], f"aggregate {name}")
            return float(int(value["numerator"]) / int(value["denominator"]))

        evaluations.append(
            {
                "candidate_id": str(raw["candidate_id"]),
                "eligible": bool(raw["eligible"]),
                "reason_codes": list(raw["reason_codes"]),
                "nonclean_macro_delta_iou": fraction(
                    "nonclean_macro_delta_iou"
                ),
                "overall_macro_delta_iou": fraction("overall_macro_delta_iou"),
                "clean_macro_delta_iou": fraction("clean_macro_delta_iou"),
                "nonclean_macro_delta_pd": fraction("nonclean_macro_delta_pd"),
                "alignment_macro_cosine": fraction("alignment_macro_cosine"),
                "nonclean_accepted_update_fraction": fraction(
                    "nonclean_accepted_update_fraction"
                ),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scope": "source_train_full_pilot64_R0_development_only",
        "paper_result": False,
        "paper_test_result": False,
        "scientific_status": receipt["scientific_status"],
        "stage_b5_allowed": bool(receipt["r1_r2_allowed"]),
        "selected_for_r1_r2": list(receipt["selected_for_r1_r2"]),
        "candidates": evaluations,
    }


_ENDPOINT_FIELDS = frozenset(
    (*_COUNT_FIELDS, "iou", "normalized_iou", "pd", "fa_per_million", "foreground_fraction")
)


def _validated_endpoint_summary(
    value: Any, *, label: str, expected_image_count: int
) -> dict[str, Any]:
    endpoint = _mapping(value, label)
    if set(endpoint) != _ENDPOINT_FIELDS:
        raise StageB4ProtocolError(f"{label} field set differs")
    counts: dict[str, int] = {}
    for field in _COUNT_FIELDS:
        raw = endpoint[field]
        if type(raw) is not int or raw < 0:
            raise StageB4ProtocolError(f"{label}.{field} must be a nonnegative int")
        counts[field] = raw
    if counts["image_count"] != expected_image_count:
        raise StageB4ProtocolError(f"{label} image_count differs")
    if counts["total_image_pixels"] != expected_image_count * 256 * 256:
        raise StageB4ProtocolError(f"{label} total_image_pixels differs")
    if (
        counts["predicted_positive_pixels"]
        != counts["intersection_pixels"] + counts["false_positive_pixels"]
        or counts["target_positive_pixels"]
        != counts["intersection_pixels"] + counts["false_negative_pixels"]
        or counts["total_image_pixels"]
        != counts["intersection_pixels"]
        + counts["false_positive_pixels"]
        + counts["false_negative_pixels"]
        + counts["true_negative_pixels"]
        or counts["detected_targets"] > counts["total_targets"]
        or counts["false_alarm_pixels"] > counts["predicted_positive_pixels"]
    ):
        raise StageB4ProtocolError(f"{label} count conservation differs")

    metrics: dict[str, float] = {}
    for field in _ENDPOINT_FIELDS.difference(_COUNT_FIELDS):
        raw = endpoint[field]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise StageB4ProtocolError(f"{label}.{field} must be numeric")
        observed = float(raw)
        if not math.isfinite(observed):
            raise StageB4ProtocolError(f"{label}.{field} is non-finite")
        metrics[field] = observed
    union = (
        counts["intersection_pixels"]
        + counts["false_positive_pixels"]
        + counts["false_negative_pixels"]
    )
    derived = {
        "iou": counts["intersection_pixels"] / union if union else 1.0,
        "pd": (
            counts["detected_targets"] / counts["total_targets"]
            if counts["total_targets"]
            else 0.0
        ),
        "fa_per_million": (
            counts["false_alarm_pixels"]
            / counts["total_image_pixels"]
            * 1_000_000.0
            if counts["total_image_pixels"]
            else 0.0
        ),
        "foreground_fraction": (
            counts["predicted_positive_pixels"] / counts["total_image_pixels"]
            if counts["total_image_pixels"]
            else 0.0
        ),
    }
    if any(
        not math.isclose(metrics[field], expected, rel_tol=1e-12, abs_tol=1e-12)
        for field, expected in derived.items()
    ):
        raise StageB4ProtocolError(f"{label} derived metric differs")
    if not 0.0 <= metrics["normalized_iou"] <= 1.0:
        raise StageB4ProtocolError(f"{label}.normalized_iou is outside [0,1]")
    return {**metrics, **counts}


def _validate_gain_attribution(
    cell: Mapping[str, Any], *, label: str, accepted_from_episodes: int
) -> None:
    accepted_count = cell.get("accepted_episode_count")
    no_update_count = cell.get("no_update_episode_count")
    if (
        type(accepted_count) is not int
        or type(no_update_count) is not int
        or accepted_count < 0
        or no_update_count < 0
        or accepted_count + no_update_count != PILOT64_COUNT
        or accepted_count != accepted_from_episodes
    ):
        raise StageB4ProtocolError(f"{label} accepted/no-update counts differ")
    attribution = _mapping(cell.get("gain_attribution"), f"{label} gain attribution")
    if set(attribution) != {"accepted", "no_update"}:
        raise StageB4ProtocolError(f"{label} gain-attribution subsets differ")

    subset_endpoints: dict[str, dict[str, dict[str, Any]] | None] = {}
    for subset_name, expected_count in (
        ("accepted", accepted_count),
        ("no_update", no_update_count),
    ):
        subset = _mapping(
            attribution[subset_name], f"{label} gain attribution {subset_name}"
        )
        if set(subset) != {"episode_count", "source", "adapted"}:
            raise StageB4ProtocolError(
                f"{label} gain attribution {subset_name} fields differ"
            )
        if subset.get("episode_count") != expected_count:
            raise StageB4ProtocolError(
                f"{label} gain attribution {subset_name} count differs"
            )
        if expected_count == 0:
            if subset.get("source") is not None or subset.get("adapted") is not None:
                raise StageB4ProtocolError(
                    f"{label} empty gain subset must have null endpoints"
                )
            subset_endpoints[subset_name] = None
            continue
        source = _validated_endpoint_summary(
            subset.get("source"),
            label=f"{label} {subset_name} Source endpoint",
            expected_image_count=expected_count,
        )
        adapted = _validated_endpoint_summary(
            subset.get("adapted"),
            label=f"{label} {subset_name} adapted endpoint",
            expected_image_count=expected_count,
        )
        if any(
            source[field] != adapted[field]
            for field in (
                "target_positive_pixels",
                "total_targets",
                "total_image_pixels",
                "image_count",
            )
        ):
            raise StageB4ProtocolError(f"{label} subset target accounting differs")
        subset_endpoints[subset_name] = {"source": source, "adapted": adapted}

    full_endpoints = {
        "source": _validated_endpoint_summary(
            cell.get("source_counts"),
            label=f"{label} full Source endpoint",
            expected_image_count=PILOT64_COUNT,
        ),
        "adapted": _validated_endpoint_summary(
            cell.get("adapted_counts"),
            label=f"{label} full adapted endpoint",
            expected_image_count=PILOT64_COUNT,
        ),
    }
    if any(
        full_endpoints["source"][field] != full_endpoints["adapted"][field]
        for field in (
            "target_positive_pixels",
            "total_targets",
            "total_image_pixels",
            "image_count",
        )
    ):
        raise StageB4ProtocolError(f"{label} full target accounting differs")
    for endpoint_name in ("source", "adapted"):
        for field in _COUNT_FIELDS:
            partition_total = sum(
                0
                if subset_endpoints[subset_name] is None
                else subset_endpoints[subset_name][endpoint_name][field]
                for subset_name in ("accepted", "no_update")
            )
            if partition_total != full_endpoints[endpoint_name][field]:
                raise StageB4ProtocolError(
                    f"{label} gain partition does not conserve {endpoint_name}.{field}"
                )


def _load_and_validate_outer_records(
    root: Path, *, dataset: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rich_cells = _read_jsonl(root / "cell_summaries.jsonl")
    episodes = _read_jsonl(root / "episode_summaries.jsonl")
    if len(rich_cells) != len(CONDITIONS) * len(CANDIDATES):
        raise StageB4ProtocolError(f"outer cell count differs: {dataset}")
    if len(episodes) != len(CONDITIONS) * PILOT64_COUNT * len(CANDIDATES):
        raise StageB4ProtocolError(f"outer episode count differs: {dataset}")
    if any(value.get("dataset") != dataset for value in rich_cells + episodes):
        raise StageB4ProtocolError(f"outer record dataset differs: {dataset}")
    expected_cell_keys = {
        (_condition_key(*condition), candidate)
        for condition in CONDITIONS
        for candidate in CANDIDATES
    }
    expected_episode_keys = {
        (condition, candidate, episode_index)
        for condition, candidate in expected_cell_keys
        for episode_index in range(PILOT64_COUNT)
    }
    episode_keys: set[tuple[str, str, int]] = set()
    accepted_by_cell = {key: 0 for key in expected_cell_keys}
    for value in episodes:
        key = (
            str(value.get("condition")),
            str(value.get("candidate_id")),
            int(value.get("episode_index", -1)),
        )
        if key in episode_keys:
            raise StageB4ProtocolError(f"duplicate outer episode: {dataset}/{key}")
        episode_keys.add(key)
        if value.get("accepted_update") is True:
            accepted_by_cell[(key[0], key[1])] = (
                accepted_by_cell.get((key[0], key[1]), 0) + 1
            )
    if episode_keys != expected_episode_keys:
        raise StageB4ProtocolError(f"outer episode Cartesian keyset differs: {dataset}")

    cell_keys: set[tuple[str, str]] = set()
    condition_lookup = {
        _condition_key(corruption, severity): (corruption, severity)
        for corruption, severity in CONDITIONS
    }
    for value in rich_cells:
        key = (str(value.get("condition")), str(value.get("candidate_id")))
        if key in cell_keys:
            raise StageB4ProtocolError(f"duplicate outer cell: {dataset}/{key}")
        cell_keys.add(key)
        if key not in expected_cell_keys:
            raise StageB4ProtocolError(f"unexpected outer cell: {dataset}/{key}")
        expected_corruption, expected_severity = condition_lookup[key[0]]
        if (
            value.get("corruption_family") != expected_corruption
            or value.get("severity") != expected_severity
            or value.get("episode_count") != PILOT64_COUNT
        ):
            raise StageB4ProtocolError(f"outer cell identity differs: {dataset}/{key}")
        _validate_gain_attribution(
            value,
            label=f"{dataset}/{key[0]}/{key[1]}",
            accepted_from_episodes=accepted_by_cell[key],
        )
    if cell_keys != expected_cell_keys:
        raise StageB4ProtocolError(f"outer cell Cartesian keyset differs: {dataset}")
    return rich_cells, episodes


def _execute_aggregate_payload(
    staging: Path,
    *,
    contract: FullPilotContract,
    candidate_artifacts: Mapping[str, tuple[Path, Mapping[str, Any]]],
    outer_artifacts: Mapping[str, tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    from analysis.p3_stage_b4_science_gate_v1 import (
        evaluate_stage_b4_r0_science_gate,
    )

    current_code = _capture_code_hashes(contract)
    rich_cells: list[dict[str, Any]] = []
    gate_cells: list[dict[str, Any]] = []
    gate_episodes: list[dict[str, Any]] = []
    parent_bindings: dict[str, Any] = {}
    for dataset in DATASETS:
        candidate_root, candidate_manifest = candidate_artifacts[dataset]
        outer_root, outer_manifest = outer_artifacts[dataset]
        if (
            candidate_manifest.get("code_sha256") != current_code
            or outer_manifest.get("code_sha256") != current_code
        ):
            raise StageB4ProtocolError(
                f"critical code changed after parent publication: {dataset}"
            )
        candidate_manifest_sha = sha256_file(candidate_root / "manifest.json")
        candidate_complete_sha = sha256_file(candidate_root / "COMPLETE.json")
        if (
            outer_manifest.get("candidate_manifest_sha256")
            != candidate_manifest_sha
            or outer_manifest.get("candidate_complete_sha256")
            != candidate_complete_sha
        ):
            raise StageB4ProtocolError(
                f"outer-to-candidate lineage differs: {dataset}"
            )
        _assert_runtime_environment_match(
            candidate_manifest.get("runtime_environment"),
            outer_manifest.get("runtime_environment"),
        )
        dataset_cells, dataset_episodes = _load_and_validate_outer_records(
            outer_root, dataset=dataset
        )
        rich_cells.extend(dataset_cells)
        gate_cells.extend(_gate_cell_summary(value) for value in dataset_cells)
        gate_episodes.extend(_gate_episode_summary(value) for value in dataset_episodes)
        parent_bindings[dataset] = {
            "candidate_path": str(candidate_root.relative_to(contract.repository)),
            "candidate_manifest_sha256": candidate_manifest_sha,
            "candidate_complete_sha256": candidate_complete_sha,
            "outer_path": str(outer_root.relative_to(contract.repository)),
            "outer_manifest_sha256": sha256_file(outer_root / "manifest.json"),
            "outer_complete_sha256": sha256_file(outer_root / "COMPLETE.json"),
            "cell_summaries_sha256": sha256_file(
                outer_root / "cell_summaries.jsonl"
            ),
            "episode_summaries_sha256": sha256_file(
                outer_root / "episode_summaries.jsonl"
            ),
        }

    try:
        decision = evaluate_stage_b4_r0_science_gate(
            gate_cells,
            gate_episodes,
            contract.raw["stage_b4_r0_science_gate"],
        )
    except Exception as exc:
        raise StageB4ProtocolError("Stage-B4 science gate rejected evidence schema") from exc
    receipt = decision.to_receipt()
    if receipt.get("paper_result") is not False or receipt.get("test") is not False:
        raise StageB4ProtocolError("science gate emitted non-development receipt")
    _write_jsonl(staging / "cell_summaries.jsonl", rich_cells)
    _write_jsonl(staging / "science_gate_cells.jsonl", gate_cells)
    _write_jsonl(staging / "science_gate_episodes.jsonl", gate_episodes)
    _write_json(staging / "science_decision_receipt.json", receipt)
    _write_json(staging / "summary.json", _human_science_summary(receipt))
    for dataset in DATASETS:
        candidate_root, _candidate = candidate_artifacts[dataset]
        outer_root, _outer = outer_artifacts[dataset]
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
    if _capture_code_hashes(contract) != current_code:
        raise StageB4ProtocolError("critical code changed during aggregate phase")
    return {
        **_runtime_manifest_base(
            contract,
            artifact_type="cr_sitta_p3_stage_b4_aggregate",
            phase="aggregate",
            dataset=None,
            formal=True,
        ),
        "parent_artifacts": parent_bindings,
        "dataset_count": len(DATASETS),
        "condition_count_per_dataset": len(CONDITIONS),
        "candidate_count": len(CANDIDATES),
        "cell_summary_count": len(gate_cells),
        "episode_summary_count": len(gate_episodes),
        "scientific_status": receipt["scientific_status"],
        "stage_b5_allowed": bool(receipt["r1_r2_allowed"]),
        "selected_for_r1_r2": list(receipt["selected_for_r1_r2"]),
        "required_followup_replicates": list(
            receipt["required_followup_replicates"]
        ),
        "outer_target_role": "source_train_pilot64_outer_oracle_only",
        "code_sha256": current_code,
    }


_AGGREGATE_PARENT_FIELDS = frozenset(
    {
        "candidate_path",
        "candidate_manifest_sha256",
        "candidate_complete_sha256",
        "outer_path",
        "outer_manifest_sha256",
        "outer_complete_sha256",
        "cell_summaries_sha256",
        "episode_summaries_sha256",
    }
)


def _assert_aggregate_parent_paths(
    parent: Mapping[str, Any],
    *,
    repository: Path,
    candidate_root: Path,
    outer_root: Path,
    dataset: str,
) -> None:
    if set(parent) != _AGGREGATE_PARENT_FIELDS:
        raise StageB4ProtocolError(
            f"aggregate parent field set differs: {dataset}"
        )
    expected_paths = {
        "candidate_path": str(candidate_root.relative_to(repository)),
        "outer_path": str(outer_root.relative_to(repository)),
    }
    if any(parent.get(key) != value for key, value in expected_paths.items()):
        raise StageB4ProtocolError(f"aggregate parent path differs: {dataset}")


def _verify_aggregate_lineage(
    contract: FullPilotContract, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current_code = _capture_code_hashes(contract)
    parents = _mapping(manifest.get("parent_artifacts"), "aggregate parents")
    if set(parents) != set(DATASETS):
        raise StageB4ProtocolError("aggregate parent dataset roster differs")
    all_cells: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidate_root = _artifact_destination(contract, "candidate", dataset)
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
        parent = _mapping(parents[dataset], f"aggregate parent {dataset}")
        _assert_aggregate_parent_paths(
            parent,
            repository=contract.repository,
            candidate_root=candidate_root,
            outer_root=outer_root,
            dataset=dataset,
        )
        checks = {
            "candidate_manifest_sha256": sha256_file(candidate_root / "manifest.json"),
            "candidate_complete_sha256": sha256_file(candidate_root / "COMPLETE.json"),
            "outer_manifest_sha256": sha256_file(outer_root / "manifest.json"),
            "outer_complete_sha256": sha256_file(outer_root / "COMPLETE.json"),
            "cell_summaries_sha256": sha256_file(outer_root / "cell_summaries.jsonl"),
            "episode_summaries_sha256": sha256_file(
                outer_root / "episode_summaries.jsonl"
            ),
        }
        if any(parent.get(key) != value for key, value in checks.items()):
            raise StageB4ProtocolError(f"aggregate lineage differs: {dataset}")
        if (
            candidate_manifest.get("code_sha256") != current_code
            or outer_manifest.get("code_sha256") != current_code
            or outer_manifest.get("candidate_manifest_sha256")
            != checks["candidate_manifest_sha256"]
            or outer_manifest.get("candidate_complete_sha256")
            != checks["candidate_complete_sha256"]
        ):
            raise StageB4ProtocolError(f"aggregate code/lineage differs: {dataset}")
        _assert_runtime_environment_match(
            candidate_manifest.get("runtime_environment"),
            outer_manifest.get("runtime_environment"),
        )
        cells, episodes = _load_and_validate_outer_records(
            outer_root, dataset=dataset
        )
        all_cells.extend(_gate_cell_summary(value) for value in cells)
        all_episodes.extend(_gate_episode_summary(value) for value in episodes)
    return all_cells, all_episodes


def run_aggregate(contract: FullPilotContract) -> dict[str, Any]:
    from analysis.p3_stage_b4_science_gate_v1 import (
        evaluate_stage_b4_r0_science_gate,
    )

    destination = _artifact_destination(contract, "aggregate")
    existing = _existing_complete_or_raise(
        destination,
        contract=contract,
        phase="aggregate",
        dataset=None,
    )
    if existing is not None:
        if existing.get("code_sha256") != _capture_code_hashes(contract):
            raise StageB4ProtocolError(
                "critical code changed after aggregate publication"
            )
        cells, episodes = _verify_aggregate_lineage(contract, existing)
        recomputed = evaluate_stage_b4_r0_science_gate(
            cells,
            episodes,
            contract.raw["stage_b4_r0_science_gate"],
        ).to_receipt()
        receipt = _load_json(destination / "science_decision_receipt.json")
        if receipt != recomputed:
            raise StageB4ProtocolError(
                "existing aggregate science decision does not recompute"
            )
        if _load_json(destination / "summary.json") != _human_science_summary(
            recomputed
        ):
            raise StageB4ProtocolError("existing aggregate summary does not recompute")
        return {
            "status": "existing_verified_complete_no_op",
            "path": str(destination),
            "scientific_status": receipt["scientific_status"],
            "stage_b5_allowed": bool(receipt["r1_r2_allowed"]),
            "selected_for_r1_r2": list(receipt["selected_for_r1_r2"]),
        }

    candidate_artifacts: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    outer_artifacts: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for dataset in DATASETS:
        candidate_root = _artifact_destination(contract, "candidate", dataset)
        outer_root = _artifact_destination(contract, "outer", dataset)
        candidate_artifacts[dataset] = (
            candidate_root,
            verify_artifact(
                candidate_root,
                contract=contract,
                phase="candidate",
                dataset=dataset,
            ),
        )
        outer_artifacts[dataset] = (
            outer_root,
            verify_artifact(
                outer_root,
                contract=contract,
                phase="outer",
                dataset=dataset,
            ),
        )
    staging = _new_staging(destination)
    try:
        manifest = _execute_aggregate_payload(
            staging,
            contract=contract,
            candidate_artifacts=candidate_artifacts,
            outer_artifacts=outer_artifacts,
        )

        def aggregate_publish_guard() -> None:
            _assert_contract_unchanged(contract)
            if _capture_code_hashes(contract) != manifest["code_sha256"]:
                raise StageB4ProtocolError(
                    "critical code changed before aggregate publication"
                )
            cells, episodes = _verify_aggregate_lineage(contract, manifest)
            recomputed = evaluate_stage_b4_r0_science_gate(
                cells,
                episodes,
                contract.raw["stage_b4_r0_science_gate"],
            ).to_receipt()
            if recomputed != _load_json(staging / "science_decision_receipt.json"):
                raise StageB4ProtocolError(
                    "science decision changed before aggregate publication"
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
        "stage_b5_allowed": bool(receipt["r1_r2_allowed"]),
        "selected_for_r1_r2": list(receipt["selected_for_r1_r2"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p3_stage_b4_full_pilot64_proposal_gate_v1.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate every frozen B4 binding")
    smoke = commands.add_parser(
        "smoke", help="run an engineering-only label-free candidate smoke"
    )
    smoke.add_argument("--dataset", choices=DATASETS, required=True)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--max-images", type=int, default=1)
    smoke.add_argument(
        "--condition",
        choices=tuple(_condition_key(*value) for value in CONDITIONS),
        default="clean_S0",
    )
    candidate = commands.add_parser(
        "candidate", help="run one formal label-free Pilot64 dataset"
    )
    candidate.add_argument("--dataset", choices=DATASETS, required=True)
    candidate.add_argument("--device", required=True)
    outer = commands.add_parser(
        "outer", help="run isolated train-target outer evaluation"
    )
    outer.add_argument("--dataset", choices=DATASETS, required=True)
    outer.add_argument("--device", required=True)
    commands.add_parser(
        "aggregate", help="verify all parents and apply the pure B4 R0 gate"
    )
    verify = commands.add_parser(
        "verify", help="rehash one immutable B4 artifact and its lineage"
    )
    verify.add_argument(
        "--phase", choices=("candidate", "outer", "aggregate"), required=True
    )
    verify.add_argument("--dataset", choices=DATASETS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = load_contract(args.config)
    if args.command == "validate":
        from analysis.p3_stage_b4_science_gate_v1 import GateConfig

        GateConfig.from_mapping(contract.raw["stage_b4_r0_science_gate"])
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
            contract, dataset=args.dataset, device_name=args.device
        )
    elif args.command == "outer":
        result = run_outer(
            contract, dataset=args.dataset, device_name=args.device
        )
    elif args.command == "aggregate":
        result = run_aggregate(contract)
    elif args.command == "verify":
        if args.phase == "aggregate":
            if args.dataset is not None:
                raise StageB4ProtocolError(
                    "aggregate verification does not accept --dataset"
                )
            dataset = None
        else:
            if args.dataset is None:
                raise StageB4ProtocolError(
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
        current_code = _capture_code_hashes(contract)
        if manifest.get("code_sha256") != current_code:
            raise StageB4ProtocolError(
                f"critical code changed after {args.phase} publication"
            )
        if args.phase == "candidate":
            assert dataset is not None
            _verify_consumed_payloads(
                contract, dataset, include_outer_target=False
            )
        elif args.phase == "outer":
            assert dataset is not None
            candidate_root = _artifact_destination(
                contract, "candidate", dataset
            )
            candidate_manifest = verify_artifact(
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
                raise StageB4ProtocolError(
                    f"outer-to-candidate lineage differs: {dataset}"
                )
            _assert_runtime_environment_match(
                candidate_manifest.get("runtime_environment"),
                manifest.get("runtime_environment"),
            )
            _verify_consumed_payloads(
                contract, dataset, include_outer_target=True
            )
        else:
            check = run_aggregate(contract)
            if check["status"] != "existing_verified_complete_no_op":
                raise StageB4ProtocolError(
                    "aggregate verify unexpectedly attempted publication"
                )
        _assert_contract_unchanged(contract)
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
