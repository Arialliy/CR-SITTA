#!/usr/bin/env python3
"""Run the train-only, two-phase CR-SITTA Stage-C0 signal audit.

``candidate`` can open only the frozen Pilot64 image cache and the sealed
first four geometric-view probabilities.  It creates label-free signal and
proxy-gradient evidence but never imports or deserializes a target payload.

``outer`` first re-hashes and semantically verifies the complete candidate
artifact.  Only a successful preflight token permits the train-target loader
to be imported.  Targets then measure gradient alignment, virtual-step task
direction, and candidate-proximal activity; they never alter a proxy gradient.

Formal result directories are atomically published with no-replace semantics.
Engineering smoke artifacts live under a separate namespace and can never be
accepted as formal evidence.  This runner contains no validation or test data
loader.  Its pure frozen gate may authorize train-only Stage-C1, but never
Stage-C R1/R2 or formal test evaluation.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Final
import uuid
from decimal import Decimal
from fractions import Fraction

import numpy as np
import yaml


REPOSITORY: Final = Path(__file__).resolve().parent
DEFAULT_CONFIG: Final = REPOSITORY / "configs/p3_stage_c0_signal_audit_v1.yaml"
PROTOCOL_ID: Final = "cr-sitta-p3-stage-c0-signal-audit-v1"
# Updated only after the YAML bytes are final.  The YAML intentionally does
# not embed its own digest, avoiding a recursive self-hash fixed-point.
FROZEN_CONFIG_SHA256: Final = (
    "785efe100797e25af776361e048ba3569349a1b925b50993a4f96abdec92358c"
)
DATASETS: Final = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CONDITIONS: Final = (
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
PROBE_IDS: Final = ("lf_mask", "hf_noise")
PARAMETER_SPACES: Final = ("R-E1", "R-D0", "P2")
PILOT_COUNT: Final = 64
IMAGE_SHAPE: Final = (3, 256, 256)
LOGIT_SHAPE: Final = (1, 256, 256)
PACKED_MASK_BYTES: Final = 256 * 256 // 8
SHA256_HEX: Final = frozenset("0123456789abcdef")


class StageC0ProtocolError(RuntimeError):
    """The frozen Stage-C0 contract or one of its artifacts is invalid."""


class ExistingArtifactError(StageC0ProtocolError):
    """A no-replace destination already exists but is not equivalent."""


@dataclass(frozen=True, slots=True)
class StageC0Contract:
    repository: Path
    config_path: Path
    config_sha256: str
    raw: Mapping[str, Any]

    @property
    def output_root(self) -> Path:
        return _repository_path(
            self.repository, self.raw["output"]["root"], "output root"
        )

    @property
    def probe_records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw["deterioration_bank"]["probes"])


@dataclass(frozen=True, slots=True)
class VerifiedCandidateArtifact:
    """Opaque proof that a complete candidate tree passed full preflight."""

    path: Path
    dataset: str
    config_sha256: str
    manifest_sha256: str
    payload_tree_sha256: str
    formal: bool
    condition_count: int
    image_count_per_condition: int
    probe_count: int


def _canonical_json_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise StageC0ProtocolError("value is not canonical-JSON safe") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    if path.is_symlink() or not path.is_file():
        raise StageC0ProtocolError(f"expected regular non-symlink file: {path}")
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
        raise StageC0ProtocolError(f"{label} must be lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageC0ProtocolError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageC0ProtocolError(f"{label} must be a sequence")
    return value


def _repository_path(repository: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise StageC0ProtocolError(f"{label} must be a non-empty path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise StageC0ProtocolError(f"{label} must be repository-relative")
    result = repository / relative
    if not result.absolute().is_relative_to(repository.absolute()):
        raise StageC0ProtocolError(f"{label} escapes repository")
    return result


def _condition_key(corruption: str, severity: int) -> str:
    return "clean_S0" if corruption == "clean" else f"{corruption}_S{severity}"


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StageC0ProtocolError(f"JSON input is missing/unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageC0ProtocolError(f"cannot parse JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StageC0ProtocolError(f"JSON root must be a mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise StageC0ProtocolError(f"JSONL input is missing/unsafe: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StageC0ProtocolError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise StageC0ProtocolError(
                    f"JSONL row must be a mapping at {path}:{line_number}"
                )
            records.append(value)
    return records


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _verify_exact_file(path: Path, expected: Any, label: str) -> None:
    if sha256_file(path) != _require_sha256(expected, f"{label} SHA-256"):
        raise StageC0ProtocolError(f"frozen input changed: {label}")


def _verify_parent_bindings(repository: Path, raw: Mapping[str, Any]) -> None:
    bindings = _mapping(raw.get("frozen_parent_bindings"), "parent bindings")
    for name, value in bindings.items():
        record = _mapping(value, f"parent {name}")
        if set(record) != {"path", "sha256"}:
            raise StageC0ProtocolError(f"parent {name} fields differ")
        _verify_exact_file(
            _repository_path(repository, record["path"], f"parent {name}"),
            record["sha256"],
            f"parent {name}",
        )
    predecessor = _load_json(
        _repository_path(
            repository,
            bindings["stage_b4_science_decision"]["path"],
            "Stage-B4 decision",
        )
    )
    if (
        predecessor.get("protocol_status") != "passed"
        or predecessor.get("scientific_status") != "scientific_no_eligible"
        or predecessor.get("r1_r2_allowed") is not False
        or predecessor.get("test") is not False
    ):
        raise StageC0ProtocolError(
            "Stage-B4 receipt does not bind the expected negative result"
        )


def _verify_dataset_bindings(
    repository: Path, raw: Mapping[str, Any], dataset: str
) -> None:
    record = _mapping(raw["datasets"][dataset], f"dataset {dataset}")
    for field, hash_field in (
        ("train_split", "train_split_sha256"),
        ("pilot_ids", "pilot_ids_file_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
    ):
        path = _repository_path(repository, record[field], f"{dataset} {field}")
        if "trainval" in path.name.lower() or "/test_" in path.as_posix().lower():
            raise StageC0ProtocolError(
                f"Stage-C0 dataset binding is not train-only: {path}"
            )
        _verify_exact_file(path, record[hash_field], f"{dataset} {field}")

    cache = _repository_path(repository, record["cache_root"], f"{dataset} cache")
    for name, field in (
        ("manifest.json", "cache_manifest_sha256"),
        ("method_input_manifest.json", "cache_method_manifest_sha256"),
        ("COMPLETE.json", "cache_complete_sha256"),
    ):
        _verify_exact_file(cache / name, record[field], f"{dataset} cache {name}")
    cache_manifest = _load_json(cache / "manifest.json")
    method_manifest = _load_json(cache / "method_input_manifest.json")
    complete = _load_json(cache / "COMPLETE.json")
    if (
        cache_manifest.get("cache_content_sha256")
        != record["cache_content_sha256"]
        or method_manifest.get("dataset") != dataset
        or method_manifest.get("targets_exposed") is not False
        or method_manifest.get("forbidden_fields")
        != ["ground_truth", "gt", "label", "mask", "target"]
        or complete.get("complete") is not True
        or complete.get("test_images_opened") != 0
        or complete.get("test_masks_opened") != 0
    ):
        raise StageC0ProtocolError(f"cache semantics differ: {dataset}")
    image_ids = tuple(str(value) for value in method_manifest.get("image_ids", ()))
    if (
        len(image_ids) != PILOT_COUNT
        or len(set(image_ids)) != PILOT_COUNT
        or _ordered_ids_sha256(image_ids)
        != record["ordered_pilot64_image_ids_sha256"]
    ):
        raise StageC0ProtocolError(f"cache Pilot64 identity differs: {dataset}")

    teacher = _repository_path(
        repository, record["teacher_root"], f"{dataset} teacher"
    )
    _verify_exact_file(
        teacher / "manifest.json",
        record["teacher_manifest_sha256"],
        f"{dataset} teacher manifest",
    )
    _verify_exact_file(
        teacher / "COMPLETE.json",
        record["teacher_complete_sha256"],
        f"{dataset} teacher completion",
    )
    teacher_manifest = _load_json(teacher / "manifest.json")
    teacher_complete = _load_json(teacher / "COMPLETE.json")
    if (
        teacher_manifest.get("artifact_type")
        != "cr_sitta_nonadaptive_teacher_candidate_dataset"
        or teacher_manifest.get("dataset") != dataset
        or teacher_manifest.get("phase") != "candidate"
        or teacher_manifest.get("formal") is not True
        or teacher_manifest.get("development_only") is not True
        or teacher_manifest.get("paper_result") is not False
        or teacher_manifest.get("base_view_names")[:4]
        != ["identity", "hflip", "vflip", "hvflip"]
        or teacher_manifest.get("image_ids") != list(image_ids)
        or teacher_complete.get("complete") is not True
        or teacher_complete.get("manifest_sha256")
        != record["teacher_manifest_sha256"]
    ):
        raise StageC0ProtocolError(f"teacher artifact semantics differ: {dataset}")


def _validate_contract_semantics(repository: Path, raw: Mapping[str, Any]) -> None:
    if raw.get("schema_version") != 1 or raw.get("protocol_id") != PROTOCOL_ID:
        raise StageC0ProtocolError("schema/protocol differs")
    freeze = _mapping(raw.get("freeze"), "freeze")
    if (
        freeze.get("state") != "frozen"
        or freeze.get("mechanism") != "runner_constant_pins_exact_config_bytes"
        or freeze.get("self_hash_field") != "forbidden_avoids_recursive_hash"
        or freeze.get("pre_run_freeze_receipt")
        != "results/cr_sitta/p3_stage_c0_signal_audit_v1/PRE_RUN_FREEZE.json"
        or freeze.get("freeze_command")
        != "python run_p3_stage_c0_signal_audit_v1.py freeze"
    ):
        raise StageC0ProtocolError("config freeze contract differs")
    scope = _mapping(raw.get("scope"), "scope")
    required_scope = {
        "source_train_derived": True,
        "split_name": "train",
        "split_role": "frozen_pilot64",
        "pilot_image_count_per_dataset": 64,
        "dataset_count": 3,
        "condition_count_per_dataset": 13,
        "probe_count": 2,
        "parameter_space_count": 3,
        "label_free_probe_episode_count_per_dataset": 1664,
        "outer_probe_episode_count_per_dataset": 1664,
        "target_payload_deserialization_count_per_dataset": 1,
        "unique_image_condition_target_uses_per_dataset": 832,
        "probe_episode_target_uses_per_dataset": 1664,
        "replicate_ids": ["R0"],
        "seed": 42,
        "development_only": True,
        "paper_result": False,
        "paper_test_result": False,
        "no_validation_split": True,
        "use_validation_payload": False,
        "use_test_payload": False,
        "method_label_accesses": 0,
        "candidate_outer_target_loader_module_imports": 0,
        "candidate_target_payload_deserializations": 0,
        "adaptation_gradient_uses_labels": False,
        "outer_train_target_access_after_candidate_completion_only": True,
    }
    for key, expected in required_scope.items():
        if scope.get(key) != expected:
            raise StageC0ProtocolError(f"scope.{key} must be {expected!r}")
    transition = _mapping(raw.get("stage_transition"), "stage transition")
    if (
        transition.get("predecessor_protocol_complete") is not True
        or transition.get("predecessor_scientific_status")
        != "scientific_no_eligible"
        or transition.get("predecessor_stage_b5_allowed") is not False
        or transition.get("stage_c0_is_new_development_hypothesis") is not True
        or transition.get("stage_c1_allowed_before_c0_gate") is not False
        or transition.get("formal_test_allowed_by_c0") is not False
    ):
        raise StageC0ProtocolError("stage transition differs")
    if tuple(tuple(value) for value in raw.get("ordered_conditions", ())) != CONDITIONS:
        raise StageC0ProtocolError("condition grid differs")
    if tuple(raw.get("datasets", {})) != DATASETS:
        raise StageC0ProtocolError("dataset order/set differs")
    teacher = _mapping(raw.get("teacher"), "teacher")
    if (
        teacher.get("array_name") != "base_view_probabilities"
        or tuple(teacher.get("accepted_view_names", ()))
        != ("identity", "hflip", "vflip", "hvflip")
        or tuple(teacher.get("accepted_view_indices", ())) != (0, 1, 2, 3)
        or teacher.get("aggregation") != "median"
        or teacher.get("detached_required") is not True
    ):
        raise StageC0ProtocolError("strong-teacher contract differs")
    bank = _mapping(raw.get("deterioration_bank"), "deterioration bank")
    probes = tuple(_mapping(value, "probe") for value in bank.get("probes", ()))
    if (
        tuple(str(value.get("id")) for value in probes) != PROBE_IDS
        or bank.get("every_probe_runs_for_every_condition") is not True
        or bank.get("benchmark_condition_name_available_to_probe") is not False
        or bank.get("benchmark_severity_available_to_probe") is not False
        or tuple(bank.get("seed_derivation_fields", ()))
        != (
            "protocol_id",
            "global_seed",
            "canonical_image_id",
            "probe_id",
            "method_facing_input_tensor_sha256",
        )
    ):
        raise StageC0ProtocolError("deterioration routing contract differs")
    spaces = _mapping(raw.get("parameter_spaces"), "parameter spaces")
    if tuple(spaces.get("ordered", ())) != PARAMETER_SPACES:
        raise StageC0ProtocolError("parameter-space order differs")
    gate = _mapping(raw.get("stage_c0_signal_gate"), "signal gate")
    if (
        gate.get("eligibility_evaluated_before_followup_authorization") is not True
        or gate.get("ranking_at_c0") != "forbidden"
        or gate.get("no_eligible_is_normal_scientific_result") is not True
        or gate.get("formal_test_authorized") is not False
    ):
        raise StageC0ProtocolError("fail-closed signal-gate contract differs")
    output = _mapping(raw.get("output"), "output")
    expected_output = {
        "root": "results/cr_sitta/p3_stage_c0_signal_audit_v1",
        "candidate_phase": "candidate_phase/R0",
        "outer_phase": "outer_phase/R0",
        "aggregate_phase": "aggregate_phase/R0",
        "engineering_phase": "engineering_smoke",
    }
    if any(output.get(key) != value for key, value in expected_output.items()):
        raise StageC0ProtocolError("output namespace differs")
    if output.get("atomic_no_replace") is not True or output.get("refuse_overwrite") is not True:
        raise StageC0ProtocolError("formal output must be immutable no-replace")
    critical = tuple(raw.get("implementation", {}).get("critical_code_paths", ()))
    if not critical or len(critical) != len(set(critical)):
        raise StageC0ProtocolError("critical code path list is invalid")
    for value in critical:
        path = _repository_path(repository, value, "critical code path")
        if path.is_symlink() or not path.is_file():
            raise StageC0ProtocolError(f"critical code path missing: {value}")
    _verify_parent_bindings(repository, raw)
    for dataset in DATASETS:
        _verify_dataset_bindings(repository, raw, dataset)


def current_config_sha256(config_path: Path = DEFAULT_CONFIG) -> str:
    return sha256_file(config_path)


def load_contract(config_path: Path = DEFAULT_CONFIG) -> StageC0Contract:
    absolute = config_path if config_path.is_absolute() else REPOSITORY / config_path
    if absolute.absolute() != DEFAULT_CONFIG.absolute():
        raise StageC0ProtocolError("Stage-C0 frozen config path differs")
    if absolute.is_symlink() or not absolute.is_file():
        raise StageC0ProtocolError(f"config is missing/unsafe: {absolute}")
    payload = absolute.read_bytes()
    digest = _sha256_bytes(payload)
    if FROZEN_CONFIG_SHA256 == "TO_BE_FROZEN":
        raise StageC0ProtocolError(
            "Stage-C0 config is not frozen: pin print-config-sha256 in runner"
        )
    if digest != FROZEN_CONFIG_SHA256:
        raise StageC0ProtocolError("Stage-C0 config bytes differ from frozen hash")
    try:
        raw = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise StageC0ProtocolError("Stage-C0 YAML cannot be parsed") from exc
    if not isinstance(raw, dict):
        raise StageC0ProtocolError("Stage-C0 YAML root must be a mapping")
    _validate_contract_semantics(REPOSITORY, raw)
    return StageC0Contract(REPOSITORY, absolute, digest, raw)


def _assert_contract_unchanged(contract: StageC0Contract) -> None:
    _verify_pre_run_freeze(contract)


def _assert_base_contract_unchanged(contract: StageC0Contract) -> None:
    if sha256_file(contract.config_path) != contract.config_sha256:
        raise StageC0ProtocolError("config changed during execution")
    _validate_contract_semantics(contract.repository, contract.raw)


def _capture_code_hashes(contract: StageC0Contract) -> dict[str, str]:
    return {
        str(value): sha256_file(
            _repository_path(contract.repository, value, "critical code path")
        )
        for value in contract.raw["implementation"]["critical_code_paths"]
    }


def _freeze_receipt_path(contract: StageC0Contract) -> Path:
    return _repository_path(
        contract.repository,
        contract.raw["freeze"]["pre_run_freeze_receipt"],
        "pre-run freeze receipt",
    )


def _formal_phase_directories(contract: StageC0Contract) -> tuple[Path, ...]:
    return tuple(
        contract.output_root / str(contract.raw["output"][f"{phase}_phase"])
        for phase in ("candidate", "outer", "aggregate")
    )


def _assert_formal_phase_directories_absent(contract: StageC0Contract) -> None:
    for path in _formal_phase_directories(contract):
        if path.exists() or path.is_symlink():
            raise StageC0ProtocolError(
                f"pre-run freeze requires absent formal phase directory: {path}"
            )


def _expected_pre_run_freeze_receipt(
    contract: StageC0Contract,
) -> dict[str, Any]:
    gate_config = contract.repository / "configs/p3_stage_c_science_gate_v1.yaml"
    gate_module = contract.repository / "analysis/stage_c_science_gate_v1.py"
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        record = contract.raw["datasets"][dataset]
        datasets[dataset] = {
            "train_split": {
                "path": record["train_split"],
                "sha256": record["train_split_sha256"],
            },
            "pilot_ids": {
                "path": record["pilot_ids"],
                "file_sha256": record["pilot_ids_file_sha256"],
                "ordered_pilot64_image_ids_sha256": record[
                    "ordered_pilot64_image_ids_sha256"
                ],
            },
            "checkpoint": {
                "role": record["checkpoint_role"],
                "path": record["checkpoint_path"],
                "sha256": record["checkpoint_sha256"],
            },
            "cache": {
                "root": record["cache_root"],
                "content_sha256": record["cache_content_sha256"],
                "manifest_sha256": record["cache_manifest_sha256"],
                "method_manifest_sha256": record[
                    "cache_method_manifest_sha256"
                ],
                "complete_sha256": record["cache_complete_sha256"],
            },
            "teacher": {
                "root": record["teacher_root"],
                "manifest_sha256": record["teacher_manifest_sha256"],
                "complete_sha256": record["teacher_complete_sha256"],
            },
        }
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_stage_c0_pre_run_freeze",
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "config_path": str(
                contract.config_path.relative_to(contract.repository)
            ),
            "config_sha256": contract.config_sha256,
        },
        "science_gate": {
            "config_path": str(gate_config.relative_to(contract.repository)),
            "config_sha256": sha256_file(gate_config),
            "module_path": str(gate_module.relative_to(contract.repository)),
            "module_sha256": sha256_file(gate_module),
        },
        "critical_code_sha256": _capture_code_hashes(contract),
        "datasets": datasets,
        "created_before_formal_execution": True,
        "formal_phase_directories_absent_at_publication": True,
        "development_only": True,
        "paper_result": False,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
    }


def _verify_pre_run_freeze(
    contract: StageC0Contract,
) -> tuple[Mapping[str, Any], str]:
    from tta.d0_secure_io import read_stable_regular_file

    _assert_base_contract_unchanged(contract)
    path = _freeze_receipt_path(contract)
    try:
        snapshot = read_stable_regular_file(path)
        raw = json.loads(snapshot.data.decode("utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise StageC0ProtocolError("pre-run freeze receipt is missing/invalid") from exc
    if not isinstance(raw, dict):
        raise StageC0ProtocolError("pre-run freeze receipt root must be a mapping")
    expected = _expected_pre_run_freeze_receipt(contract)
    if raw != expected:
        raise StageC0ProtocolError(
            "pre-run freeze receipt differs from current protocol/code/inputs"
        )
    return raw, snapshot.sha256


def _run_freeze(contract: StageC0Contract) -> Mapping[str, Any]:
    """Atomically preregister current code/input hashes before formal execution."""

    destination = _freeze_receipt_path(contract)
    if destination.exists() or destination.is_symlink():
        return _verify_pre_run_freeze(contract)[0]
    _assert_formal_phase_directories_absent(contract)
    from tta.d0_secure_io import (
        ensure_directory_chain_nofollow,
        publish_file_noreplace,
    )

    relative_parent = destination.parent.relative_to(contract.repository)
    ensure_directory_chain_nofollow(
        contract.repository, tuple(relative_parent.parts)
    )
    temporary = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    expected = _expected_pre_run_freeze_receipt(contract)
    try:
        _write_json(temporary, expected)

        def guard() -> None:
            _assert_base_contract_unchanged(contract)
            _assert_formal_phase_directories_absent(contract)
            if expected != _expected_pre_run_freeze_receipt(contract):
                raise StageC0ProtocolError(
                    "pre-run inputs/code changed before freeze publication"
                )

        publish_file_noreplace(
            temporary, destination, pre_rename_guard=guard
        )
    except BaseException:
        if temporary.exists() and not temporary.is_symlink() and temporary.is_file():
            temporary.unlink()
        raise
    return _verify_pre_run_freeze(contract)[0]


def _file_ledger(root: Path, *, excluded: Sequence[str] = ()) -> dict[str, Any]:
    result: dict[str, Any] = {}
    excluded_set = set(excluded)
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise StageC0ProtocolError(f"artifact contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise StageC0ProtocolError(f"artifact entry is not regular: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
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


def _publish_artifact(
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
            "artifact_type": "cr_sitta_stage_c0_completion",
            "protocol_id": PROTOCOL_ID,
            "complete": True,
            "phase": final_manifest["phase"],
            "dataset": final_manifest.get("dataset"),
            "formal": final_manifest["formal"],
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "atomic_no_replace": True,
            "development_only": True,
            "paper_result": False,
            "paper_test_result": False,
            "pre_run_freeze_receipt_path": final_manifest[
                "pre_run_freeze_receipt_path"
            ],
            "pre_run_freeze_receipt_sha256": final_manifest[
                "pre_run_freeze_receipt_sha256"
            ],
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
        },
    )
    from tta.d0_secure_io import publish_directory_noreplace

    publish_directory_noreplace(
        staging, destination, pre_rename_guard=pre_rename_guard
    )


def _runtime_manifest_base(
    contract: StageC0Contract,
    *,
    artifact_type: str,
    phase: str,
    dataset: str | None,
    formal: bool,
) -> dict[str, Any]:
    _receipt, receipt_sha256 = _verify_pre_run_freeze(contract)
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
        "pre_run_freeze_receipt_path": str(
            _freeze_receipt_path(contract).relative_to(contract.repository)
        ),
        "pre_run_freeze_receipt_sha256": receipt_sha256,
        "method_label_accesses": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
    }


def _artifact_destination(
    contract: StageC0Contract,
    *,
    phase: str,
    dataset: str | None,
    formal: bool,
    smoke_id: str | None,
) -> Path:
    if formal:
        root = contract.output_root / str(contract.raw["output"][f"{phase}_phase"])
        return root / dataset if dataset is not None else root
    if not isinstance(smoke_id, str) or not smoke_id or "/" in smoke_id or ".." in smoke_id:
        raise StageC0ProtocolError("engineering smoke requires a safe --smoke-id")
    root = contract.output_root / str(contract.raw["output"]["engineering_phase"])
    result = root / smoke_id / phase
    return result / dataset if dataset is not None else result


def _staging_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    return staging


def _remove_owned_staging(staging: Path) -> None:
    if staging.name.startswith(".") and ".staging-" in staging.name and staging.exists():
        shutil.rmtree(staging)


def _raw_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def derive_probe_seed(
    *,
    global_seed: int,
    image_id: str,
    probe_id: str,
    input_tensor_sha256: str,
) -> int:
    """Derive randomness without a benchmark condition/severity field."""

    if isinstance(global_seed, bool) or not isinstance(global_seed, int) or global_seed < 0:
        raise StageC0ProtocolError("global_seed must be a non-negative integer")
    if not isinstance(image_id, str) or not image_id:
        raise StageC0ProtocolError("image_id must be non-empty")
    if probe_id not in PROBE_IDS:
        raise StageC0ProtocolError(f"unsupported fixed probe: {probe_id}")
    _require_sha256(input_tensor_sha256, "method-facing input tensor")
    descriptor = {
        "protocol_id": PROTOCOL_ID,
        "global_seed": global_seed,
        "canonical_image_id": image_id,
        "probe_id": probe_id,
        "method_facing_input_tensor_sha256": input_tensor_sha256,
    }
    # manual_seed accepts signed 64-bit values portably across CPU/CUDA.
    return int(_sha256_bytes(_canonical_json_bytes(descriptor))[:16], 16) % (2**63 - 1)


def build_probe_image(
    normalized_image: Any,
    probe: Mapping[str, Any],
    *,
    generator: Any,
):
    """Apply one fixed physical-space probe; no condition label is accepted."""

    from tta.deteriorations.fourier_low_mask import mask_low_frequency_amplitude
    from tta.deteriorations.high_frequency_noise import inject_high_frequency_noise
    from tta.deteriorations.image_space import (
        imagenet_denormalize,
        imagenet_normalize,
    )

    physical = imagenet_denormalize(normalized_image)
    kind = probe.get("kind")
    if kind == "low_frequency_amplitude_mask":
        output = mask_low_frequency_amplitude(
            physical,
            mask_ratio=float(probe["mask_ratio"]),
            keep_probability=float(probe["keep_probability"]),
            generator=generator,
        ).image
    elif kind == "high_frequency_noise":
        output = inject_high_frequency_noise(
            physical,
            target_rms=float(probe["target_rms"]),
            low_cut_ratio=float(probe["low_cut_ratio"]),
            generator=generator,
        )
    else:
        raise StageC0ProtocolError(f"unknown probe kind: {kind!r}")
    return imagenet_normalize(output)


def _verified_teacher_inputs(
    contract: StageC0Contract,
    dataset: str,
    condition: str,
) -> tuple[np.memmap, Mapping[str, Any]]:
    record = contract.raw["datasets"][dataset]
    root = _repository_path(
        contract.repository, record["teacher_root"], f"{dataset} teacher root"
    )
    manifest = _load_json(root / "manifest.json")
    matches = [
        value
        for value in manifest.get("conditions", ())
        if value.get("condition") == condition
    ]
    if len(matches) != 1:
        raise StageC0ProtocolError(f"teacher condition is not unique: {condition}")
    condition_record = _mapping(matches[0], "teacher condition")
    descriptor = _mapping(
        _mapping(condition_record.get("arrays"), "teacher arrays").get(
            "base_view_probabilities"
        ),
        "teacher base-view array",
    )
    relative = descriptor.get("path")
    expected_relative = f"conditions/{condition}/base_view_probabilities.npy"
    if relative != expected_relative:
        raise StageC0ProtocolError("teacher base-view path differs")
    if tuple(descriptor.get("shape", ())) != (64, 5, 1, 256, 256):
        raise StageC0ProtocolError("teacher base-view shape differs")
    if descriptor.get("dtype") != "little_endian_float32":
        raise StageC0ProtocolError("teacher base-view dtype differs")
    if tuple(descriptor.get("view_names", ()))[:4] != (
        "identity",
        "hflip",
        "vflip",
        "hvflip",
    ):
        raise StageC0ProtocolError("teacher geometric view order differs")
    ledger = _mapping(manifest.get("files"), "teacher file ledger")
    file_record = _mapping(ledger.get(expected_relative), "teacher array ledger")
    path = root / expected_relative
    _verify_exact_file(path, file_record.get("sha256"), "teacher base views")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(value, np.memmap)
        or bool(value.flags.writeable)
        or value.dtype.str != "<f4"
        or tuple(value.shape) != (64, 5, 1, 256, 256)
    ):
        raise StageC0ProtocolError("teacher base-view mmap differs")
    return value, condition_record


def _method_input_dataset(
    contract: StageC0Contract, dataset: str, condition: str
):
    # This capability-minimal module has no outer-target loader to import.
    from tta.stage_c0_method_input import StageC0MethodInputDataset

    record = contract.raw["datasets"][dataset]
    cache = _repository_path(
        contract.repository, record["cache_root"], f"{dataset} cache"
    )
    return StageC0MethodInputDataset(
        cache,
        condition_key=condition,
        expected_protocol_sha256=contract.raw["frozen_parent_bindings"][
            "cache_protocol"
        ]["sha256"],
        expected_dataset=dataset,
        expected_complete_sha256=record["cache_complete_sha256"],
        expected_method_manifest_sha256=record["cache_method_manifest_sha256"],
        expected_ordered_ids_sha256=record["ordered_pilot64_image_ids_sha256"],
    )


def _build_runtime(
    contract: StageC0Contract,
    dataset: str,
    device_name: str,
    *,
    include_reference_model: bool,
):
    import torch
    import test_source
    from model.MSHNet_NSFPN_adaptable import MSHNetNSFPNAdaptable
    from tta.adapters.nsfpn_router_adapter import build_default_nsfpn_routers
    from tta.model_adapter import IRSTDModelAdapter
    from tta.parameter_groups import PILOT_GROUP_SPECS, collect_adaptable_params

    seed = int(contract.raw["scope"]["seed"])
    test_source.seed_everything(seed)
    device = test_source.resolve_device(device_name)
    router_config = contract.raw["parameter_spaces"]
    e1, d0 = build_default_nsfpn_routers(
        "R-E1+D0",
        grid_size=tuple(router_config["R-E1"]["grid_size"]),
        max_scale_delta=float(router_config["R-E1"]["max_scale_delta"]),
        max_bias=float(router_config["R-E1"]["max_bias"]),
        seed=int(router_config["router_basis_seed"]),
    )
    model = MSHNetNSFPNAdaptable(3, router_e1=e1, router_d0=d0)
    checkpoint = _repository_path(
        contract.repository,
        contract.raw["datasets"][dataset]["checkpoint_path"],
        f"{dataset} checkpoint",
    )
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - supported older torch fallback.
        payload = torch.load(checkpoint, map_location="cpu")
    state, wrapper = test_source.extract_state_dict(payload)
    model.load_source_state_dict(state)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    named: dict[str, tuple[tuple[str, Any], ...]] = {
        "R-E1": tuple(
            (f"router_e1.{name}", parameter)
            for name, parameter in model.router_e1.named_parameters()
        ),
        "R-D0": tuple(
            (f"router_d0.{name}", parameter)
            for name, parameter in model.router_d0.named_parameters()
        ),
    }
    p2_parameters, p2_names = collect_adaptable_params(
        model,
        group_spec=PILOT_GROUP_SPECS["P2"],
        expected_named_modules_sha256=None,
        expected_bn_affine_inventory_sha256=None,
    )
    named["P2"] = tuple(zip(p2_names, p2_parameters, strict=True))
    for space in PARAMETER_SPACES:
        expected_tensors = int(router_config[space]["expected_parameter_tensor_count"])
        expected_scalars = int(router_config[space]["expected_scalar_parameter_count"])
        if (
            len(named[space]) != expected_tensors
            or sum(int(parameter.numel()) for _, parameter in named[space])
            != expected_scalars
        ):
            raise StageC0ProtocolError(f"{space} parameter topology differs")
        for _name, parameter in named[space]:
            parameter.requires_grad_(True)
    allowed = {id(parameter) for space in PARAMETER_SPACES for _, parameter in named[space]}
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if actual != allowed:
        raise StageC0ProtocolError("trainable tensors escape Stage-C0 spaces")
    adapter = IRSTDModelAdapter(model, warm_flag=False)

    reference_model = None
    reference_adapter = None
    if include_reference_model:
        reference_model = test_source.build_nsfpn_model()
        test_source.load_trusted_checkpoint(reference_model, checkpoint)
        reference_model.to(device).eval()
        reference_adapter = IRSTDModelAdapter(reference_model, warm_flag=False)
        reference_adapter.set_source_eval_mode()
    return (
        torch,
        test_source,
        model,
        adapter,
        named,
        device,
        wrapper,
        reference_model,
        reference_adapter,
    )


def _parameter_layout(named: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    names = [name for name, _ in named]
    shapes = [list(parameter.shape) for _, parameter in named]
    descriptor = {
        "protocol": "cr-sitta-stage-c0-flat-parameter-layout-v1",
        "names": names,
        "shapes": shapes,
        "scalar_count": sum(int(parameter.numel()) for _, parameter in named),
        "dtype": "float32",
    }
    return {**descriptor, "layout_sha256": _sha256_bytes(_canonical_json_bytes(descriptor))}


@contextmanager
def _scoped_cuda_backward_allowance(torch: Any, *, tensor: Any):
    """Temporarily relax deterministic algorithms only for SFS CUDA backward."""

    if not getattr(tensor, "is_cuda", False):
        yield
        return
    enabled = bool(torch.are_deterministic_algorithms_enabled())
    warn_only_getter = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    warn_only = bool(warn_only_getter()) if callable(warn_only_getter) else False
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _flatten_autograd(
    loss: Any,
    named_spaces: Mapping[str, Sequence[tuple[str, Any]]],
    *,
    retain_graph: bool = False,
) -> dict[str, Any]:
    import torch

    from analysis.stage_c0_group_alignment import flatten_gradients

    ordered_parameters = tuple(
        parameter
        for space in PARAMETER_SPACES
        for _name, parameter in named_spaces[space]
    )
    with _scoped_cuda_backward_allowance(torch, tensor=loss):
        gradients = torch.autograd.grad(
            loss,
            ordered_parameters,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
    result: dict[str, Any] = {}
    offset = 0
    for space in PARAMETER_SPACES:
        parameters = tuple(parameter for _name, parameter in named_spaces[space])
        count = len(parameters)
        result[space] = flatten_gradients(
            list(gradients[offset : offset + count]), list(parameters)
        )
        offset += count
    if offset != len(gradients):
        raise StageC0ProtocolError("gradient partition differs")
    return result


def _valid_candidate_masks(teacher: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    pairs = tuple(
        (core, ring)
        for core, ring in zip(
            teacher.candidate_core, teacher.candidate_ring, strict=True
        )
        if bool(core.any().item()) and bool(ring.any().item())
    )
    return (
        tuple(core for core, _ring in pairs),
        tuple(ring for _core, ring in pairs),
    )


def _label_free_objective(
    contract: StageC0Contract,
    *,
    student_logits: Any,
    teacher: Any,
    named_spaces: Mapping[str, Sequence[tuple[str, Any]]],
):
    import torch

    from tta.objectives.asb_sfr import asb_sfr_proposal_objective

    config = contract.raw["objective"]
    cores, rings = _valid_candidate_masks(teacher)
    candidate_weights = (
        student_logits.new_full((len(cores),), 1.0 / len(cores)).detach()
        if cores
        else student_logits.new_empty((0,)).detach()
    )
    router_parameters = {
        name: parameter
        for space in ("R-E1", "R-D0")
        for name, parameter in named_spaces[space]
    }
    output = asb_sfr_proposal_objective(
        student_logits,
        teacher.probability,
        teacher.logits,
        teacher.target_weight,
        teacher.background_weight,
        teacher.stability_mask,
        cores,
        rings,
        candidate_weights,
        router_parameters,
        entropy_margin=float(config["entropy_margin"]),
        foreground_loss_weight=float(config["lambda_foreground"]),
        background_loss_weight=float(config["lambda_background"]),
        candidate_contrast_loss_weight=float(config["lambda_local_contrast"]),
        adapter_l2_weight=float(config["lambda_router_l2"]),
        spatial_tv_weight=float(config["lambda_router_spatial_tv"]),
        min_active_pixels=int(config["min_active_pixels"]),
        contrast_temperature=float(config["local_contrast_temperature"]),
        contrast_ring_weight=float(config["local_contrast_ring_weight"]),
        contrast_huber_delta=float(config["local_contrast_huber_delta"]),
        eps=float(config["probability_eps"]),
    )
    total = output.total
    if total.ndim != 0 or not bool(total.detach().isfinite().item()):
        raise StageC0ProtocolError("label-free objective is not a finite scalar")
    return output


def _pack_active_mask(mask: Any) -> np.ndarray:
    array = mask.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1)
    packed = np.packbits(array, bitorder="little")
    if tuple(packed.shape) != (PACKED_MASK_BYTES,):
        raise StageC0ProtocolError("packed active-mask size differs")
    return packed


def _unpack_active_mask(value: np.ndarray) -> np.ndarray:
    packed = np.asarray(value, dtype=np.uint8).reshape(-1)
    if tuple(packed.shape) != (PACKED_MASK_BYTES,):
        raise StageC0ProtocolError("packed active-mask size differs")
    return np.unpackbits(packed, bitorder="little", count=256 * 256).reshape(
        1, 256, 256
    ).astype(bool, copy=False)


def _runtime_environment(torch: Any, device: Any) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _open_gradient_memmaps(
    staging: Path,
    *,
    condition_count: int,
    image_count: int,
    probe_count: int,
    layouts: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.memmap]:
    result: dict[str, np.memmap] = {}
    for space in PARAMETER_SPACES:
        path = staging / f"proxy_gradients_{space.replace('-', '_')}.npy"
        result[space] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype="<f4",
            shape=(
                condition_count,
                image_count,
                probe_count,
                int(layouts[space]["scalar_count"]),
            ),
        )
    return result


def _candidate_record(
    *,
    dataset: str,
    corruption: str,
    severity: int,
    condition_index: int,
    image_index: int,
    image_id: str,
    probe_index: int,
    probe: Mapping[str, Any],
    input_tensor_sha256: str,
    probe_seed: int,
    probe_input_sha256: str,
    gap: Any,
    support: Any,
    proposal: Any,
    gradients: Mapping[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "cr_sitta_stage_c0_candidate_episode",
        "protocol_id": PROTOCOL_ID,
        "dataset": dataset,
        "condition": _condition_key(corruption, severity),
        "corruption": corruption,
        "severity": severity,
        "condition_index": condition_index,
        "image_index": image_index,
        "image_id": image_id,
        "probe_index": probe_index,
        "probe_id": str(probe["id"]),
        "probe_kind": str(probe["kind"]),
        "input_tensor_sha256": input_tensor_sha256,
        "probe_seed": probe_seed,
        "probe_input_sha256": probe_input_sha256,
        **gap.to_dict(),
        **support.to_dict(),
        "active_foreground_loss": float(proposal.foreground.detach().item()),
        "active_background_loss": float(proposal.background.detach().item()),
        "local_contrast_loss": float(
            proposal.candidate_contrast.detach().item()
        ),
        "candidate_count": int(proposal.candidate_count),
        "candidate_weight_sum": float(
            proposal.candidate_weight_sum.detach().item()
        ),
        "proxy_loss": float(proposal.total.detach().item()),
        "source_point_router_l2_gradient_norm": 0.0,
        "source_point_router_tv_gradient_norm": 0.0,
        "gradient_evidence": {},
        "method_label_accesses": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
    }
    evidence: dict[str, Any] = {}
    for space in PARAMETER_SPACES:
        vector = gradients[space].detach().cpu().numpy().astype("<f4", copy=False)
        norm = float(np.linalg.norm(vector.astype(np.float64, copy=False)))
        evidence[space] = {
            "parameter_space": space,
            "proxy_gradient_norm": norm,
            "finite": bool(np.isfinite(vector).all() and math.isfinite(norm)),
            "nonzero": bool(norm > 1.0e-12),
            "slice_sha256": _raw_array_sha256(vector),
            "slice_index": [condition_index, image_index, probe_index],
        }
    record["gradient_evidence"] = evidence
    return record


def _run_candidate(
    contract: StageC0Contract,
    *,
    dataset: str,
    device_name: str,
    formal: bool,
    smoke_id: str | None,
    smoke_images: int,
) -> Mapping[str, Any]:
    """Create candidate evidence without importing any target loader."""

    if dataset not in DATASETS:
        raise StageC0ProtocolError(f"unsupported dataset: {dataset}")
    _verify_pre_run_freeze(contract)
    condition_grid = CONDITIONS if formal else CONDITIONS[:1]
    image_count = PILOT_COUNT if formal else smoke_images
    if isinstance(image_count, bool) or not 1 <= int(image_count) <= PILOT_COUNT:
        raise StageC0ProtocolError("smoke image count must lie in [1,64]")
    image_count = int(image_count)
    destination = _artifact_destination(
        contract,
        phase="candidate",
        dataset=dataset,
        formal=formal,
        smoke_id=smoke_id,
    )
    if destination.exists() or destination.is_symlink():
        return verify_candidate_artifact(
            destination, contract=contract, dataset=dataset, expected_formal=formal
        )[0]
    staging = _staging_directory(destination)
    try:
        (
            torch,
            test_source,
            model,
            adapter,
            named_spaces,
            device,
            checkpoint_wrapper,
            _reference_model,
            reference_adapter,
        ) = _build_runtime(
            contract,
            dataset,
            device_name,
            include_reference_model=True,
        )
        if reference_adapter is None:
            raise StageC0ProtocolError("reference adapter was not constructed")
        source_state_before = test_source.state_dict_sha256(model.state_dict())
        layouts = {
            space: _parameter_layout(named_spaces[space])
            for space in PARAMETER_SPACES
        }
        _write_json(staging / "parameter_layouts.json", layouts)
        gradients = _open_gradient_memmaps(
            staging,
            condition_count=len(condition_grid),
            image_count=image_count,
            probe_count=len(PROBE_IDS),
            layouts=layouts,
        )
        active_masks = np.lib.format.open_memmap(
            staging / "active_masks_packbits.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(condition_grid), image_count, len(PROBE_IDS), PACKED_MASK_BYTES),
        )
        records: list[dict[str, Any]] = []
        identity_comparison_count = 0
        identity_exact_count = 0
        identity_maximum_difference = 0.0
        teacher_config = contract.raw["teacher"]

        from analysis.stage_c0_active_support import audit_active_support
        from analysis.stage_c0_teacher_student_gap import (
            measure_teacher_student_gap,
        )
        from tta.teachers.strong_view_teacher import (
            build_strong_teacher_from_aligned_probabilities,
        )

        for condition_index, (corruption, severity) in enumerate(condition_grid):
            condition = _condition_key(corruption, severity)
            method = _method_input_dataset(contract, dataset, condition)
            teacher_array, _teacher_condition = _verified_teacher_inputs(
                contract, dataset, condition
            )
            if tuple(method.image_ids) != tuple(
                _load_json(
                    _repository_path(
                        contract.repository,
                        contract.raw["datasets"][dataset]["teacher_root"],
                        "teacher root",
                    )
                    / "manifest.json"
                )["image_ids"]
            ):
                raise StageC0ProtocolError("method/teacher Pilot64 order differs")
            for image_index in range(image_count):
                item = method[image_index]
                image_id = str(item["image_id"])
                image_cpu = item["image"].contiguous()
                if tuple(image_cpu.shape) != IMAGE_SHAPE:
                    raise StageC0ProtocolError("method-facing image shape differs")
                input_hash = _raw_array_sha256(image_cpu.numpy())
                image = image_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
                aligned = torch.from_numpy(
                    np.array(teacher_array[image_index, :4], copy=True)
                ).unsqueeze(1).to(device=device, dtype=torch.float32)
                teacher = build_strong_teacher_from_aligned_probabilities(
                    aligned,
                    tuple(teacher_config["accepted_view_names"]),
                    expected_view_names=tuple(
                        teacher_config["accepted_view_names"]
                    ),
                    aggregation="median",
                    probability_eps=float(teacher_config["probability_eps"]),
                    variance_threshold=float(teacher_config["variance_threshold"]),
                    target_threshold=float(teacher_config["target_threshold"]),
                    background_threshold=float(teacher_config["background_threshold"]),
                    uncertainty_temperature=float(
                        teacher_config["uncertainty_temperature"]
                    ),
                    protection_radius=int(teacher_config["protection_radius"]),
                    candidate_threshold=float(teacher_config["candidate_threshold"]),
                    candidate_min_area=int(teacher_config["candidate_min_area"]),
                    ring_inner_radius=int(teacher_config["ring_inner_radius"]),
                    ring_outer_radius=int(teacher_config["ring_outer_radius"]),
                    local_contrast_radius=int(teacher_config["local_contrast_radius"]),
                )
                with torch.no_grad():
                    source_logits = adapter.forward_logits(image)
                    if tuple(source_logits.shape) != (1, *LOGIT_SHAPE):
                        raise StageC0ProtocolError("Source logit shape differs")
                    if condition == "clean_S0":
                        reference_logits = reference_adapter.forward_logits(image)
                        difference = float(
                            (source_logits - reference_logits).abs().amax().item()
                        )
                        identity_comparison_count += 1
                        identity_maximum_difference = max(
                            identity_maximum_difference, difference
                        )
                        if torch.equal(source_logits, reference_logits):
                            identity_exact_count += 1

                for probe_index, probe in enumerate(contract.probe_records):
                    probe_seed = derive_probe_seed(
                        global_seed=int(contract.raw["scope"]["seed"]),
                        image_id=image_id,
                        probe_id=str(probe["id"]),
                        input_tensor_sha256=input_hash,
                    )
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(probe_seed)
                    probe_image = build_probe_image(
                        image, probe, generator=generator
                    )
                    probe_hash = _raw_array_sha256(
                        probe_image.detach().cpu().numpy()
                    )
                    student_logits = adapter.forward_logits(probe_image)
                    proposal = _label_free_objective(
                        contract,
                        student_logits=student_logits,
                        teacher=teacher,
                        named_spaces=named_spaces,
                    )
                    gradient_vectors = _flatten_autograd(
                        proposal.total, named_spaces, retain_graph=False
                    )
                    for space in PARAMETER_SPACES:
                        vector = gradient_vectors[space]
                        if not bool(torch.isfinite(vector).all().item()):
                            raise StageC0ProtocolError(
                                f"non-finite proxy gradient: {space}"
                            )
                        if (
                            not proposal.has_active_support
                            and bool(torch.count_nonzero(vector).item())
                        ):
                            raise StageC0ProtocolError(
                                "empty active support produced a proxy gradient"
                            )
                        gradients[space][
                            condition_index, image_index, probe_index
                        ] = vector.detach().cpu().numpy().astype("<f4", copy=False)
                    effective_active_mask = (
                        proposal.active_mask
                        if proposal.has_active_support
                        else torch.zeros_like(proposal.active_mask)
                    )
                    active_masks[
                        condition_index, image_index, probe_index
                    ] = _pack_active_mask(effective_active_mask)
                    gap = measure_teacher_student_gap(
                        teacher.probability,
                        student_logits,
                        eps=float(contract.raw["objective"]["probability_eps"]),
                    )
                    support = audit_active_support(
                        effective_active_mask,
                        teacher.target_weight,
                        teacher.background_weight,
                        teacher.guard_union,
                        min_active_pixels=int(
                            contract.raw["objective"]["min_active_pixels"]
                        ),
                    )
                    records.append(
                        _candidate_record(
                            dataset=dataset,
                            corruption=corruption,
                            severity=severity,
                            condition_index=condition_index,
                            image_index=image_index,
                            image_id=image_id,
                            probe_index=probe_index,
                            probe=probe,
                            input_tensor_sha256=input_hash,
                            probe_seed=probe_seed,
                            probe_input_sha256=probe_hash,
                            gap=gap,
                            support=support,
                            proposal=proposal,
                            gradients=gradient_vectors,
                        )
                    )
        for value in (*gradients.values(), active_masks):
            value.flush()
        del gradients, active_masks
        expected_records = len(condition_grid) * image_count * len(PROBE_IDS)
        if len(records) != expected_records:
            raise StageC0ProtocolError("candidate episode count differs")
        _write_jsonl(staging / "candidate_episodes.jsonl", records)
        source_state_after = test_source.state_dict_sha256(model.state_dict())
        if source_state_after != source_state_before:
            raise StageC0ProtocolError("candidate audit mutated model state")
        if identity_comparison_count != image_count:
            raise StageC0ProtocolError("identity comparison count differs")
        manifest = {
            **_runtime_manifest_base(
                contract,
                artifact_type="cr_sitta_stage_c0_candidate_dataset",
                phase="candidate",
                dataset=dataset,
                formal=formal,
            ),
            "replicate_id": "R0",
            "condition_count": len(condition_grid),
            "conditions": [_condition_key(*value) for value in condition_grid],
            "image_count_per_condition": image_count,
            "probe_count": len(PROBE_IDS),
            "probe_ids": list(PROBE_IDS),
            "parameter_spaces": list(PARAMETER_SPACES),
            "episode_count": expected_records,
            "parameter_layouts": layouts,
            "active_mask_encoding": {
                "format": "numpy_packbits",
                "bitorder": "little",
                "unpacked_shape": [1, 256, 256],
                "packed_bytes_per_episode": PACKED_MASK_BYTES,
            },
            "method_boundary": {
                "method_input": "frozen_train_Pilot64_images",
                "teacher_input": "sealed_first_four_geometric_view_probabilities",
                "teacher_aggregation": "median",
                "exact_geometric_view_sequence_validated": True,
                "validated_geometric_view_sequence": list(
                    teacher_config["accepted_view_names"]
                ),
                "outer_target_loader_module_imports": 0,
                "target_payload_deserializations": 0,
                "method_label_accesses": 0,
                "validation_payload_opens": 0,
                "test_payload_opens": 0,
                "probe_condition_routing": False,
                "probe_seed_uses_condition_label": False,
            },
            "identity_adapter": {
                "comparison_scope": "clean_Pilot64_only",
                "comparison_count": identity_comparison_count,
                "exact_match_count": identity_exact_count,
                "mismatch_count": identity_comparison_count - identity_exact_count,
                "maximum_absolute_output_difference": identity_maximum_difference,
            },
            "source_state_sha256_before": source_state_before,
            "source_state_sha256_after": source_state_after,
            "source_state_bit_exact": True,
            "checkpoint_wrapper": checkpoint_wrapper,
            "runtime_environment": _runtime_environment(torch, device),
            "scoped_cuda_backward_allowance": {
                "required_for_sfs_backward": bool(device.type == "cuda"),
                "scope": "torch.autograd.grad_only",
                "forward_determinism_unchanged": True,
                "prior_state_restored_after_success_and_exception": True,
                "global_disable_forbidden": True,
            },
            "critical_code_sha256": _capture_code_hashes(contract),
        }

        def guard() -> None:
            _assert_contract_unchanged(contract)
            _verify_dataset_bindings(contract.repository, contract.raw, dataset)

        _publish_artifact(
            staging, destination, manifest, pre_rename_guard=guard
        )
        return verify_candidate_artifact(
            destination,
            contract=contract,
            dataset=dataset,
            expected_formal=formal,
        )[0]
    except BaseException:
        _remove_owned_staging(staging)
        raise


def _validate_access_zero(record: Mapping[str, Any], *, label: str) -> None:
    for field in (
        "method_label_accesses",
        "validation_payload_opens",
        "test_payload_opens",
    ):
        if record.get(field) != 0:
            raise StageC0ProtocolError(f"{label}.{field} must be exactly zero")


def _validate_layouts(value: Any) -> dict[str, Mapping[str, Any]]:
    layouts = _mapping(value, "parameter layouts")
    if tuple(layouts) != PARAMETER_SPACES:
        raise StageC0ProtocolError("parameter layout order differs")
    result: dict[str, Mapping[str, Any]] = {}
    for space in PARAMETER_SPACES:
        layout = _mapping(layouts[space], f"layout {space}")
        if set(layout) != {
            "protocol",
            "names",
            "shapes",
            "scalar_count",
            "dtype",
            "layout_sha256",
        }:
            raise StageC0ProtocolError(f"layout {space} fields differ")
        descriptor = {key: layout[key] for key in layout if key != "layout_sha256"}
        if (
            layout["protocol"] != "cr-sitta-stage-c0-flat-parameter-layout-v1"
            or layout["dtype"] != "float32"
            or layout["layout_sha256"]
            != _sha256_bytes(_canonical_json_bytes(descriptor))
        ):
            raise StageC0ProtocolError(f"layout {space} digest differs")
        names = tuple(layout["names"])
        shapes = tuple(tuple(item) for item in layout["shapes"])
        if (
            len(names) != len(shapes)
            or len(names)
            != int(
                # Tensor count is frozen in the protocol.
                {"R-E1": 2, "R-D0": 2, "P2": 16}[space]
            )
            or sum(math.prod(shape) for shape in shapes)
            != int(layout["scalar_count"])
            or int(layout["scalar_count"])
            != {"R-E1": 512, "R-D0": 256, "P2": 416}[space]
        ):
            raise StageC0ProtocolError(f"layout {space} topology differs")
        result[space] = layout
    return result


def _validate_completion(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    phase: str,
    dataset: str | None,
    formal: bool,
) -> None:
    complete = _load_json(root / "COMPLETE.json")
    if (
        complete.get("schema_version") != 1
        or complete.get("artifact_type") != "cr_sitta_stage_c0_completion"
        or complete.get("protocol_id") != PROTOCOL_ID
        or complete.get("complete") is not True
        or complete.get("phase") != phase
        or complete.get("dataset") != dataset
        or complete.get("formal") is not formal
        or complete.get("manifest_sha256") != sha256_file(root / "manifest.json")
        or complete.get("atomic_no_replace") is not True
        or complete.get("development_only") is not True
        or complete.get("paper_result") is not False
        or complete.get("paper_test_result") is not False
        or complete.get("pre_run_freeze_receipt_path")
        != manifest.get("pre_run_freeze_receipt_path")
        or complete.get("pre_run_freeze_receipt_sha256")
        != manifest.get("pre_run_freeze_receipt_sha256")
        or complete.get("validation_payload_opens") != 0
        or complete.get("test_payload_opens") != 0
    ):
        raise StageC0ProtocolError(f"completion receipt differs: {root}")
    expected_files = _mapping(manifest.get("files"), "artifact file ledger")
    actual_files = _file_ledger(root, excluded=("manifest.json", "COMPLETE.json"))
    if actual_files != expected_files:
        raise StageC0ProtocolError(f"artifact payload ledger differs: {root}")
    if manifest.get("payload_tree_sha256") != _sha256_bytes(
        _canonical_json_bytes(expected_files)
    ):
        raise StageC0ProtocolError(f"artifact tree digest differs: {root}")


def _validate_manifest_freeze_binding(
    manifest: Mapping[str, Any], *, contract: StageC0Contract
) -> None:
    _receipt, receipt_sha256 = _verify_pre_run_freeze(contract)
    if (
        manifest.get("pre_run_freeze_receipt_path")
        != str(_freeze_receipt_path(contract).relative_to(contract.repository))
        or manifest.get("pre_run_freeze_receipt_sha256") != receipt_sha256
    ):
        raise StageC0ProtocolError("artifact pre-run freeze binding differs")


def verify_candidate_artifact(
    path: Path,
    *,
    contract: StageC0Contract,
    dataset: str,
    expected_formal: bool,
) -> tuple[dict[str, Any], VerifiedCandidateArtifact]:
    """Fully verify a candidate tree before any train target is reachable."""

    if dataset not in DATASETS:
        raise StageC0ProtocolError(f"unsupported dataset: {dataset}")
    if path.is_symlink() or not path.is_dir():
        raise StageC0ProtocolError(f"candidate artifact missing/unsafe: {path}")
    manifest = _load_json(path / "manifest.json")
    _validate_manifest_freeze_binding(manifest, contract=contract)
    _validate_access_zero(manifest, label="candidate manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "cr_sitta_stage_c0_candidate_dataset"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("config_sha256") != contract.config_sha256
        or manifest.get("phase") != "candidate"
        or manifest.get("dataset") != dataset
        or manifest.get("formal") is not expected_formal
        or manifest.get("development_only") is not True
        or manifest.get("paper_result") is not False
        or manifest.get("paper_test_result") is not False
        or manifest.get("replicate_id") != "R0"
        or tuple(manifest.get("probe_ids", ())) != PROBE_IDS
        or tuple(manifest.get("parameter_spaces", ())) != PARAMETER_SPACES
        or manifest.get("probe_count") != len(PROBE_IDS)
        or manifest.get("source_state_bit_exact") is not True
        or manifest.get("source_state_sha256_before")
        != manifest.get("source_state_sha256_after")
    ):
        raise StageC0ProtocolError("candidate manifest semantics differ")
    condition_count = int(manifest.get("condition_count", -1))
    image_count = int(manifest.get("image_count_per_condition", -1))
    conditions = tuple(manifest.get("conditions", ()))
    if expected_formal:
        if (
            condition_count != len(CONDITIONS)
            or image_count != PILOT_COUNT
            or conditions != tuple(_condition_key(*value) for value in CONDITIONS)
        ):
            raise StageC0ProtocolError("formal candidate topology differs")
    elif (
        condition_count != 1
        or not 1 <= image_count <= PILOT_COUNT
        or conditions != ("clean_S0",)
    ):
        raise StageC0ProtocolError("engineering candidate topology differs")
    episode_count = condition_count * image_count * len(PROBE_IDS)
    if manifest.get("episode_count") != episode_count:
        raise StageC0ProtocolError("candidate episode count differs")
    boundary = _mapping(manifest.get("method_boundary"), "candidate method boundary")
    if (
        boundary.get("method_input") != "frozen_train_Pilot64_images"
        or boundary.get("teacher_input")
        != "sealed_first_four_geometric_view_probabilities"
        or boundary.get("teacher_aggregation") != "median"
        or boundary.get("exact_geometric_view_sequence_validated") is not True
        or tuple(boundary.get("validated_geometric_view_sequence", ()))
        != ("identity", "hflip", "vflip", "hvflip")
        or boundary.get("outer_target_loader_module_imports") != 0
        or boundary.get("target_payload_deserializations") != 0
        or boundary.get("method_label_accesses") != 0
        or boundary.get("validation_payload_opens") != 0
        or boundary.get("test_payload_opens") != 0
        or boundary.get("probe_condition_routing") is not False
        or boundary.get("probe_seed_uses_condition_label") is not False
    ):
        raise StageC0ProtocolError("candidate label firewall differs")
    current_code = _capture_code_hashes(contract)
    if manifest.get("critical_code_sha256") != current_code:
        raise StageC0ProtocolError("candidate critical-code seal differs")
    _validate_completion(
        path,
        manifest=manifest,
        phase="candidate",
        dataset=dataset,
        formal=expected_formal,
    )
    layouts = _validate_layouts(_load_json(path / "parameter_layouts.json"))
    if manifest.get("parameter_layouts") != layouts:
        raise StageC0ProtocolError("candidate manifest/layout payload differs")

    gradient_arrays: dict[str, np.memmap] = {}
    for space in PARAMETER_SPACES:
        value = np.load(
            path / f"proxy_gradients_{space.replace('-', '_')}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        expected_shape = (
            condition_count,
            image_count,
            len(PROBE_IDS),
            int(layouts[space]["scalar_count"]),
        )
        if (
            not isinstance(value, np.memmap)
            or bool(value.flags.writeable)
            or value.dtype.str != "<f4"
            or tuple(value.shape) != expected_shape
            or not bool(np.isfinite(value).all())
        ):
            raise StageC0ProtocolError(f"candidate gradient array differs: {space}")
        gradient_arrays[space] = value
    packed = np.load(
        path / "active_masks_packbits.npy", mmap_mode="r", allow_pickle=False
    )
    if (
        not isinstance(packed, np.memmap)
        or bool(packed.flags.writeable)
        or packed.dtype != np.uint8
        or tuple(packed.shape)
        != (condition_count, image_count, len(PROBE_IDS), PACKED_MASK_BYTES)
    ):
        raise StageC0ProtocolError("candidate active-mask array differs")
    records = _read_jsonl(path / "candidate_episodes.jsonl")
    if len(records) != episode_count:
        raise StageC0ProtocolError("candidate JSONL record count differs")
    cursor = 0
    condition_grid = CONDITIONS if expected_formal else CONDITIONS[:1]
    for condition_index, (corruption, severity) in enumerate(condition_grid):
        for image_index in range(image_count):
            for probe_index, probe_id in enumerate(PROBE_IDS):
                record = records[cursor]
                cursor += 1
                _validate_access_zero(record, label="candidate episode")
                if (
                    record.get("schema_version") != 1
                    or record.get("artifact_type")
                    != "cr_sitta_stage_c0_candidate_episode"
                    or record.get("protocol_id") != PROTOCOL_ID
                    or record.get("dataset") != dataset
                    or record.get("condition")
                    != _condition_key(corruption, severity)
                    or record.get("corruption") != corruption
                    or record.get("severity") != severity
                    or record.get("condition_index") != condition_index
                    or record.get("image_index") != image_index
                    or record.get("probe_index") != probe_index
                    or record.get("probe_id") != probe_id
                    or record.get("finite") is not True
                    or record.get("active_pixel_count")
                    != int(
                        _unpack_active_mask(
                            packed[condition_index, image_index, probe_index]
                        ).sum()
                    )
                ):
                    raise StageC0ProtocolError("candidate episode semantics differ")
                _require_sha256(record.get("input_tensor_sha256"), "candidate input")
                _require_sha256(record.get("probe_input_sha256"), "candidate probe")
                evidence = _mapping(
                    record.get("gradient_evidence"), "candidate gradient evidence"
                )
                if tuple(evidence) != PARAMETER_SPACES:
                    raise StageC0ProtocolError("candidate gradient spaces differ")
                for space in PARAMETER_SPACES:
                    item = _mapping(evidence[space], f"candidate gradient {space}")
                    vector = np.asarray(
                        gradient_arrays[space][
                            condition_index, image_index, probe_index
                        ]
                    )
                    norm = float(np.linalg.norm(vector.astype(np.float64, copy=False)))
                    if (
                        item.get("parameter_space") != space
                        or item.get("finite") is not True
                        or item.get("nonzero") is not bool(norm > 1.0e-12)
                        or item.get("slice_index")
                        != [condition_index, image_index, probe_index]
                        or item.get("slice_sha256") != _raw_array_sha256(vector)
                        or not math.isclose(
                            float(item.get("proxy_gradient_norm")),
                            norm,
                            rel_tol=1.0e-6,
                            abs_tol=1.0e-12,
                        )
                    ):
                        raise StageC0ProtocolError(
                            f"candidate gradient proof differs: {space}"
                        )
    identity = _mapping(manifest.get("identity_adapter"), "identity adapter")
    if (
        identity.get("comparison_scope") != "clean_Pilot64_only"
        or identity.get("comparison_count") != image_count
        or identity.get("exact_match_count")
        + identity.get("mismatch_count")
        != image_count
        or identity.get("maximum_absolute_output_difference", -1.0) < 0.0
    ):
        raise StageC0ProtocolError("identity evidence differs")
    token = VerifiedCandidateArtifact(
        path=path,
        dataset=dataset,
        config_sha256=contract.config_sha256,
        manifest_sha256=sha256_file(path / "manifest.json"),
        payload_tree_sha256=str(manifest["payload_tree_sha256"]),
        formal=expected_formal,
        condition_count=condition_count,
        image_count_per_condition=image_count,
        probe_count=len(PROBE_IDS),
    )
    return manifest, token


def open_outer_train_targets(
    contract: StageC0Contract,
    *,
    dataset: str,
    verified_candidate: VerifiedCandidateArtifact,
):
    """Open train targets only after replaying the candidate preflight proof."""

    if not isinstance(verified_candidate, VerifiedCandidateArtifact):
        raise PermissionError("outer targets require a verified candidate token")
    _manifest, replay = verify_candidate_artifact(
        verified_candidate.path,
        contract=contract,
        dataset=dataset,
        expected_formal=verified_candidate.formal,
    )
    if replay != verified_candidate:
        raise PermissionError("candidate artifact changed after preflight")
    # This import is intentionally below complete candidate verification.
    from materialize_binary_tent_ss_calibration_cache_v2 import (
        load_outer_evaluator_targets_v2,
    )

    cache = _repository_path(
        contract.repository,
        contract.raw["datasets"][dataset]["cache_root"],
        f"{dataset} cache",
    )
    return load_outer_evaluator_targets_v2(
        cache,
        expected_protocol_sha256=contract.raw["frozen_parent_bindings"][
            "cache_protocol"
        ]["sha256"],
        episodes_complete=True,
    )


def _candidate_response_scalars(
    logits: Any,
    teacher: Any,
    *,
    temperature: float,
    ring_weight: float,
):
    import torch

    cores, rings = _valid_candidate_masks(teacher)
    if not cores:
        zero = logits.sum() * 0.0
        return zero, zero, 0
    absolute_values = []
    contrast_values = []
    for core, ring in zip(cores, rings, strict=True):
        values = logits[core]
        absolute = temperature * (
            torch.logsumexp(values / temperature, dim=0)
            - torch.log(values.new_tensor(float(values.numel())))
        )
        contrast = absolute - ring_weight * logits[ring].mean()
        absolute_values.append(absolute)
        contrast_values.append(contrast)
    return (
        torch.stack(absolute_values).mean(),
        torch.stack(contrast_values).mean(),
        len(cores),
    )


def _space_radius(
    contract: StageC0Contract,
    *,
    space: str,
    named_parameters: Sequence[tuple[str, Any]],
) -> tuple[str, float, float]:
    import torch

    config = contract.raw["parameter_spaces"][space]
    if space in ("R-E1", "R-D0"):
        return (
            "absolute_l2",
            float(config["normalized_virtual_step_l2_radius"]),
            float(config["normalized_virtual_step_l2_radius"]),
        )
    if space != "P2":
        raise StageC0ProtocolError(f"unknown parameter space: {space}")
    if config.get("normalized_virtual_step_radius_mode") != (
        "relative_to_source_parameter_l2"
    ):
        raise StageC0ProtocolError("P2 radius mode differs")
    relative = float(config["normalized_virtual_step_relative_radius"])
    source = torch.cat(
        [parameter.detach().reshape(-1) for _name, parameter in named_parameters]
    )
    source_norm = float(torch.linalg.vector_norm(source).item())
    absolute = relative * source_norm
    if not math.isfinite(absolute) or absolute <= 0.0:
        raise StageC0ProtocolError("P2 absolute trust radius is invalid")
    return "relative_to_source_parameter_l2", relative, absolute


@contextmanager
def _temporary_parameter_step(
    named_parameters: Sequence[tuple[str, Any]], direction: Any
):
    import torch

    parameters = tuple(parameter for _name, parameter in named_parameters)
    if int(direction.numel()) != sum(int(value.numel()) for value in parameters):
        raise StageC0ProtocolError("virtual-step vector size differs")
    snapshots = tuple(parameter.detach().clone() for parameter in parameters)
    offset = 0
    try:
        with torch.no_grad():
            for parameter in parameters:
                count = int(parameter.numel())
                parameter.add_(direction[offset : offset + count].reshape_as(parameter))
                offset += count
        yield
    finally:
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots, strict=True):
                parameter.copy_(snapshot)


def _gt_proximal_counts(
    packed_active_mask: np.ndarray,
    target: Any,
    *,
    dilation_radius: int = 5,
) -> tuple[int, int, bool]:
    import torch
    import torch.nn.functional as functional

    active = torch.from_numpy(_unpack_active_mask(packed_active_mask)).unsqueeze(0)
    if tuple(target.shape) != (1, 1, 256, 256):
        raise StageC0ProtocolError("outer target shape differs")
    target_binary = (target.detach().cpu() > 0.0).to(torch.float32)
    kernel = 2 * dilation_radius + 1
    proximal = functional.max_pool2d(
        target_binary,
        kernel_size=kernel,
        stride=1,
        padding=dilation_radius,
    ).bool()
    count = int((active & proximal).sum().item())
    active_count = int(active.sum().item())
    return count, active_count, bool(count > 0)


def _outer_episode_record(
    candidate: Mapping[str, Any],
    *,
    proximal_count: int,
    active_count: int,
    space_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    common_keys = (
        "dataset",
        "condition",
        "corruption",
        "severity",
        "condition_index",
        "image_index",
        "image_id",
        "probe_index",
        "probe_id",
        "probe_kind",
        "teacher_student_gap_l1",
        "teacher_student_logit_gap_mean",
        "teacher_student_logit_gap_max",
        "active_episode",
        "active_pixel_count",
        "active_pixel_fraction",
        "active_target_weight",
        "active_background_weight",
    )
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_stage_c0_outer_episode",
        "protocol_id": PROTOCOL_ID,
        **{key: candidate[key] for key in common_keys},
        "candidate_proximal_active_pixel_count": proximal_count,
        "candidate_proximal_active_episode": bool(proximal_count > 0),
        "active_mask_count_reverified": active_count,
        "total_pixel_count": 256 * 256,
        "parameter_space_evidence": dict(space_evidence),
        "method_label_accesses": 0,
        "outer_train_target_accesses": 1,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
    }


def _run_outer(
    contract: StageC0Contract,
    *,
    dataset: str,
    device_name: str,
    formal: bool,
    smoke_id: str | None,
) -> Mapping[str, Any]:
    if dataset not in DATASETS:
        raise StageC0ProtocolError(f"unsupported dataset: {dataset}")
    _verify_pre_run_freeze(contract)
    candidate_path = _artifact_destination(
        contract,
        phase="candidate",
        dataset=dataset,
        formal=formal,
        smoke_id=smoke_id,
    )
    candidate_manifest, candidate_token = verify_candidate_artifact(
        candidate_path,
        contract=contract,
        dataset=dataset,
        expected_formal=formal,
    )
    destination = _artifact_destination(
        contract,
        phase="outer",
        dataset=dataset,
        formal=formal,
        smoke_id=smoke_id,
    )
    if destination.exists() or destination.is_symlink():
        return verify_outer_artifact(
            destination,
            contract=contract,
            dataset=dataset,
            expected_formal=formal,
            candidate_token=candidate_token,
        )
    # No train target is imported/opened above this line.
    targets = open_outer_train_targets(
        contract,
        dataset=dataset,
        verified_candidate=candidate_token,
    )
    staging = _staging_directory(destination)
    try:
        (
            torch,
            test_source,
            model,
            adapter,
            named_spaces,
            device,
            checkpoint_wrapper,
            _reference_model,
            _reference_adapter,
        ) = _build_runtime(
            contract,
            dataset,
            device_name,
            include_reference_model=False,
        )
        source_state_before = test_source.state_dict_sha256(model.state_dict())
        condition_count = candidate_token.condition_count
        image_count = candidate_token.image_count_per_condition
        candidate_records = _read_jsonl(candidate_path / "candidate_episodes.jsonl")
        layouts = _validate_layouts(
            _load_json(candidate_path / "parameter_layouts.json")
        )
        candidate_gradients = {
            space: np.load(
                candidate_path / f"proxy_gradients_{space.replace('-', '_')}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for space in PARAMETER_SPACES
        }
        packed_masks = np.load(
            candidate_path / "active_masks_packbits.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        task_config = contract.raw["outer_oracle_task_loss"]
        epsilon = float(task_config["normalized_virtual_step"]["nonzero_epsilon"])
        threshold = float(
            task_config["normalized_virtual_step"]["threshold_crossing_probability"]
        )
        teacher_config = contract.raw["teacher"]
        objective_config = contract.raw["objective"]
        dataset_record = contract.raw["datasets"][dataset]

        from analysis.d0_v2_task_loss import compute_outer_oracle_task_loss
        from analysis.source_train_provenance import (
            OUTER_ORACLE_ROLE,
            SourceTrainAnalysisProvenance,
        )
        from analysis.stage_c0_group_alignment import (
            analyze_group_alignment,
            clipped_descent_direction,
        )
        from tta.teachers.strong_view_teacher import (
            build_strong_teacher_from_aligned_probabilities,
        )

        provenance = SourceTrainAnalysisProvenance(
            dataset=dataset,
            split_name="train",
            split_sha256=str(dataset_record["train_split_sha256"]),
            checkpoint_sha256=str(dataset_record["checkpoint_sha256"]),
            seed=int(contract.raw["scope"]["seed"]),
            oracle_analysis=True,
            outer_evaluator_label_accesses=1,
            supervised_gradient_role=OUTER_ORACLE_ROLE,
        )
        condition_grid = CONDITIONS if formal else CONDITIONS[:1]
        outer_records: list[dict[str, Any]] = []
        alignment_records: list[dict[str, Any]] = []
        cursor = 0
        for condition_index, (corruption, severity) in enumerate(condition_grid):
            condition = _condition_key(corruption, severity)
            method = _method_input_dataset(contract, dataset, condition)
            teacher_array, _teacher_condition = _verified_teacher_inputs(
                contract, dataset, condition
            )
            for image_index in range(image_count):
                item = method[image_index]
                image = item["image"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                target = torch.from_numpy(
                    np.array(targets[image_index], copy=True)
                ).unsqueeze(0).to(device=device, dtype=torch.float32)
                aligned = torch.from_numpy(
                    np.array(teacher_array[image_index, :4], copy=True)
                ).unsqueeze(1).to(device=device, dtype=torch.float32)
                teacher = build_strong_teacher_from_aligned_probabilities(
                    aligned,
                    tuple(teacher_config["accepted_view_names"]),
                    expected_view_names=tuple(
                        teacher_config["accepted_view_names"]
                    ),
                    aggregation="median",
                    probability_eps=float(teacher_config["probability_eps"]),
                    variance_threshold=float(teacher_config["variance_threshold"]),
                    target_threshold=float(teacher_config["target_threshold"]),
                    background_threshold=float(teacher_config["background_threshold"]),
                    uncertainty_temperature=float(
                        teacher_config["uncertainty_temperature"]
                    ),
                    protection_radius=int(teacher_config["protection_radius"]),
                    candidate_threshold=float(teacher_config["candidate_threshold"]),
                    candidate_min_area=int(teacher_config["candidate_min_area"]),
                    ring_inner_radius=int(teacher_config["ring_inner_radius"]),
                    ring_outer_radius=int(teacher_config["ring_outer_radius"]),
                    local_contrast_radius=int(teacher_config["local_contrast_radius"]),
                )
                source_logits = adapter.forward_logits(image)
                source_probability = torch.sigmoid(source_logits.detach())
                task_loss, _task_audit = compute_outer_oracle_task_loss(
                    logits=source_logits,
                    target=target,
                    config={
                        key: task_config[key]
                        for key in (
                            "lambda_bce",
                            "lambda_soft_iou",
                            "eps",
                            "bce_reduction",
                            "soft_iou_reduction",
                            "empty_target_convention",
                        )
                    },
                    provenance=provenance,
                )
                absolute_response, local_contrast, candidate_count = (
                    _candidate_response_scalars(
                        source_logits,
                        teacher,
                        temperature=float(
                            objective_config["local_contrast_temperature"]
                        ),
                        ring_weight=float(
                            objective_config["local_contrast_ring_weight"]
                        ),
                    )
                )
                task_gradients = _flatten_autograd(
                    task_loss, named_spaces, retain_graph=True
                )
                absolute_gradients = _flatten_autograd(
                    absolute_response, named_spaces, retain_graph=True
                )
                contrast_gradients = _flatten_autograd(
                    local_contrast, named_spaces, retain_graph=False
                )
                for probe_index, probe_id in enumerate(PROBE_IDS):
                    candidate = candidate_records[cursor]
                    cursor += 1
                    if (
                        candidate.get("condition_index") != condition_index
                        or candidate.get("image_index") != image_index
                        or candidate.get("probe_index") != probe_index
                        or candidate.get("probe_id") != probe_id
                    ):
                        raise StageC0ProtocolError(
                            "candidate/outer episode order differs"
                        )
                    proximal_count, active_count, proximal_episode = (
                        _gt_proximal_counts(
                            packed_masks[
                                condition_index, image_index, probe_index
                            ],
                            target,
                            dilation_radius=5,
                        )
                    )
                    if active_count != int(candidate["active_pixel_count"]):
                        raise StageC0ProtocolError(
                            "candidate active-mask count changed before outer"
                        )
                    per_space: dict[str, Mapping[str, Any]] = {}
                    for space in PARAMETER_SPACES:
                        proxy = torch.from_numpy(
                            np.array(
                                candidate_gradients[space][
                                    condition_index, image_index, probe_index
                                ],
                                copy=True,
                            )
                        ).to(device=device, dtype=torch.float32)
                        radius_mode, radius_value, absolute_radius = _space_radius(
                            contract,
                            space=space,
                            named_parameters=named_spaces[space],
                        )
                        alignment = analyze_group_alignment(
                            proxy,
                            task_gradients[space],
                            candidate_absolute_gradient=absolute_gradients[space],
                            candidate_contrast_gradient=contrast_gradients[space],
                            virtual_step_radius=absolute_radius,
                            nonzero_epsilon=epsilon,
                        )
                        direction = clipped_descent_direction(
                            proxy,
                            radius=absolute_radius,
                            nonzero_epsilon=epsilon,
                        )
                        if alignment.virtual_step_norm == 0.0:
                            crossing_count = 0
                        else:
                            with _temporary_parameter_step(
                                named_spaces[space], direction
                            ):
                                with torch.no_grad():
                                    post_logits = adapter.forward_logits(image)
                            crossing_count = int(
                                (
                                    (source_probability > threshold)
                                    ^ (torch.sigmoid(post_logits) > threshold)
                                ).sum().item()
                            )
                            with torch.no_grad():
                                restored = adapter.forward_logits(image)
                            if not torch.equal(restored, source_logits.detach()):
                                raise StageC0ProtocolError(
                                    "virtual step did not restore Source output"
                                )
                        evidence = {
                            **alignment.to_dict(),
                            "parameter_space": space,
                            "radius_mode": radius_mode,
                            "radius_value": radius_value,
                            "absolute_l2_radius": absolute_radius,
                            "nonzero_epsilon": epsilon,
                            "direction_formula": (
                                "-g*min(1,radius/(norm+epsilon))"
                            ),
                            "candidate_count": candidate_count,
                            "threshold_crossing_pixel_count": crossing_count,
                            "threshold_crossing_episode": bool(crossing_count > 0),
                            "threshold_crossing_role": (
                                "report_only_not_stage_c0_hard_gate"
                            ),
                        }
                        per_space[space] = evidence
                        alignment_records.append(
                            {
                                "schema_version": 1,
                                "artifact_type": (
                                    "cr_sitta_stage_c0_outer_space_alignment"
                                ),
                                "protocol_id": PROTOCOL_ID,
                                "dataset": dataset,
                                "condition": condition,
                                "corruption": corruption,
                                "severity": severity,
                                "image_index": image_index,
                                "image_id": str(item["image_id"]),
                                "probe_index": probe_index,
                                "probe_id": probe_id,
                                **evidence,
                                "method_label_accesses": 0,
                                "validation_payload_opens": 0,
                                "test_payload_opens": 0,
                            }
                        )
                    outer_records.append(
                        _outer_episode_record(
                            candidate,
                            proximal_count=proximal_count,
                            active_count=active_count,
                            space_evidence=per_space,
                        )
                    )
        expected_records = condition_count * image_count * len(PROBE_IDS)
        if cursor != expected_records or len(outer_records) != expected_records:
            raise StageC0ProtocolError("outer episode count differs")
        if len(alignment_records) != expected_records * len(PARAMETER_SPACES):
            raise StageC0ProtocolError("outer alignment count differs")
        _write_jsonl(staging / "outer_episodes.jsonl", outer_records)
        _write_jsonl(staging / "space_alignments.jsonl", alignment_records)
        source_state_after = test_source.state_dict_sha256(model.state_dict())
        if source_state_after != source_state_before:
            raise StageC0ProtocolError("outer audit mutated model state")
        manifest = {
            **_runtime_manifest_base(
                contract,
                artifact_type="cr_sitta_stage_c0_outer_dataset",
                phase="outer",
                dataset=dataset,
                formal=formal,
            ),
            "replicate_id": "R0",
            "condition_count": condition_count,
            "conditions": list(candidate_manifest["conditions"]),
            "image_count_per_condition": image_count,
            "probe_count": len(PROBE_IDS),
            "probe_ids": list(PROBE_IDS),
            "parameter_spaces": list(PARAMETER_SPACES),
            "episode_count": expected_records,
            "space_alignment_count": len(alignment_records),
            "candidate_artifact": {
                "path": str(candidate_path.relative_to(contract.repository)),
                "manifest_sha256": candidate_token.manifest_sha256,
                "payload_tree_sha256": candidate_token.payload_tree_sha256,
                "fully_verified_before_target_import": True,
                "fully_verified_before_target_deserialization": True,
            },
            "outer_boundary": {
                "target_split": "train",
                "target_role": "fixed_Pilot64_outer_evaluator_only",
                "target_loader_imports": 1,
                "target_payload_deserializations": 1,
                "unique_image_condition_target_uses": condition_count * image_count,
                "probe_episode_target_uses": expected_records,
                "adaptation_gradient_uses_labels": False,
                "method_label_accesses": 0,
                "validation_payload_opens": 0,
                "test_payload_opens": 0,
                "proximal_definition": "GT_Chebyshev_dilation_radius_5_input_pixels",
                "exact_geometric_view_sequence_validated": True,
                "validated_geometric_view_sequence": list(
                    teacher_config["accepted_view_names"]
                ),
            },
            "identity_adapter": candidate_manifest["identity_adapter"],
            "parameter_layouts": layouts,
            "source_state_sha256_before": source_state_before,
            "source_state_sha256_after": source_state_after,
            "source_state_bit_exact": True,
            "checkpoint_wrapper": checkpoint_wrapper,
            "runtime_environment": _runtime_environment(torch, device),
            "scoped_cuda_backward_allowance": {
                "required_for_sfs_backward": bool(device.type == "cuda"),
                "scope": "torch.autograd.grad_only",
                "prior_state_restored_after_success_and_exception": True,
                "global_disable_forbidden": True,
            },
            "critical_code_sha256": _capture_code_hashes(contract),
        }

        def guard() -> None:
            _assert_contract_unchanged(contract)
            replay_manifest, replay_token = verify_candidate_artifact(
                candidate_path,
                contract=contract,
                dataset=dataset,
                expected_formal=formal,
            )
            if replay_token != candidate_token or replay_manifest != candidate_manifest:
                raise StageC0ProtocolError(
                    "candidate artifact changed before outer publication"
                )

        _publish_artifact(
            staging, destination, manifest, pre_rename_guard=guard
        )
        return verify_outer_artifact(
            destination,
            contract=contract,
            dataset=dataset,
            expected_formal=formal,
            candidate_token=candidate_token,
        )
    except BaseException:
        _remove_owned_staging(staging)
        raise


def _finite_or_none(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageC0ProtocolError(f"{label} must be finite numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise StageC0ProtocolError(f"{label} contains NaN or Inf")
    return result


def _p2_absolute_radius_from_checkpoint(
    contract: StageC0Contract,
    *,
    dataset: str,
    layout: Mapping[str, Any],
) -> float:
    """Recompute the P2 relative radius directly from the sealed Source state."""

    import torch
    import test_source

    checkpoint = _repository_path(
        contract.repository,
        contract.raw["datasets"][dataset]["checkpoint_path"],
        f"{dataset} checkpoint",
    )
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - supported older torch fallback.
        payload = torch.load(checkpoint, map_location="cpu")
    state, _wrapper = test_source.extract_state_dict(payload)
    names = tuple(layout.get("names", ()))
    shapes = tuple(tuple(value) for value in layout.get("shapes", ()))
    if len(names) != len(shapes) or not names:
        raise StageC0ProtocolError("P2 checkpoint radius layout differs")
    pieces = []
    for name, shape in zip(names, shapes, strict=True):
        value = state.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise StageC0ProtocolError(
                f"P2 checkpoint parameter differs: {name}"
            )
        pieces.append(value.detach().reshape(-1))
    source_norm = float(torch.linalg.vector_norm(torch.cat(pieces)).item())
    relative = float(
        contract.raw["parameter_spaces"]["P2"][
            "normalized_virtual_step_relative_radius"
        ]
    )
    result = relative * source_norm
    if not math.isfinite(result) or result <= 0.0:
        raise StageC0ProtocolError("recomputed P2 absolute radius is invalid")
    return result


def verify_outer_artifact(
    path: Path,
    *,
    contract: StageC0Contract,
    dataset: str,
    expected_formal: bool,
    candidate_token: VerifiedCandidateArtifact | None = None,
) -> dict[str, Any]:
    """Verify outer evidence and its immutable candidate-parent binding."""

    if path.is_symlink() or not path.is_dir():
        raise StageC0ProtocolError(f"outer artifact missing/unsafe: {path}")
    manifest = _load_json(path / "manifest.json")
    _validate_manifest_freeze_binding(manifest, contract=contract)
    candidate_binding = _mapping(
        manifest.get("candidate_artifact"), "outer candidate binding"
    )
    if set(candidate_binding) != {
        "path",
        "manifest_sha256",
        "payload_tree_sha256",
        "fully_verified_before_target_import",
        "fully_verified_before_target_deserialization",
    }:
        raise StageC0ProtocolError("outer candidate-binding fields differ")
    bound_candidate_path = _repository_path(
        contract.repository,
        candidate_binding.get("path"),
        "outer candidate path",
    )
    expected_candidate_path = (
        _artifact_destination(
            contract,
            phase="candidate",
            dataset=dataset,
            formal=True,
            smoke_id=None,
        )
        if expected_formal
        else path.parent.parent / "candidate" / dataset
    )
    if bound_candidate_path != expected_candidate_path:
        raise StageC0ProtocolError("outer candidate path is not canonical")
    candidate_manifest, replay_token = verify_candidate_artifact(
        bound_candidate_path,
        contract=contract,
        dataset=dataset,
        expected_formal=expected_formal,
    )
    if candidate_token is not None and replay_token != candidate_token:
        raise StageC0ProtocolError("outer candidate token differs")
    if (
        candidate_binding.get("manifest_sha256") != replay_token.manifest_sha256
        or candidate_binding.get("payload_tree_sha256")
        != replay_token.payload_tree_sha256
        or candidate_binding.get("fully_verified_before_target_import") is not True
        or candidate_binding.get("fully_verified_before_target_deserialization")
        is not True
    ):
        raise StageC0ProtocolError("outer candidate binding differs")
    _validate_access_zero(manifest, label="outer manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "cr_sitta_stage_c0_outer_dataset"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("config_sha256") != contract.config_sha256
        or manifest.get("phase") != "outer"
        or manifest.get("dataset") != dataset
        or manifest.get("formal") is not expected_formal
        or manifest.get("development_only") is not True
        or manifest.get("paper_result") is not False
        or manifest.get("paper_test_result") is not False
        or manifest.get("replicate_id") != "R0"
        or manifest.get("condition_count") != replay_token.condition_count
        or manifest.get("image_count_per_condition")
        != replay_token.image_count_per_condition
        or manifest.get("probe_count") != replay_token.probe_count
        or tuple(manifest.get("probe_ids", ())) != PROBE_IDS
        or tuple(manifest.get("parameter_spaces", ())) != PARAMETER_SPACES
        or manifest.get("source_state_bit_exact") is not True
        or manifest.get("source_state_sha256_before")
        != manifest.get("source_state_sha256_after")
    ):
        raise StageC0ProtocolError("outer manifest semantics differ")
    expected_records = (
        replay_token.condition_count
        * replay_token.image_count_per_condition
        * len(PROBE_IDS)
    )
    if (
        manifest.get("episode_count") != expected_records
        or manifest.get("space_alignment_count")
        != expected_records * len(PARAMETER_SPACES)
        or manifest.get("conditions") != candidate_manifest.get("conditions")
        or manifest.get("identity_adapter")
        != candidate_manifest.get("identity_adapter")
    ):
        raise StageC0ProtocolError("outer topology differs")
    boundary = _mapping(manifest.get("outer_boundary"), "outer boundary")
    if (
        boundary.get("target_split") != "train"
        or boundary.get("target_role")
        != "fixed_Pilot64_outer_evaluator_only"
        or boundary.get("target_loader_imports") != 1
        or boundary.get("target_payload_deserializations") != 1
        or boundary.get("unique_image_condition_target_uses")
        != replay_token.condition_count * replay_token.image_count_per_condition
        or boundary.get("probe_episode_target_uses") != expected_records
        or boundary.get("adaptation_gradient_uses_labels") is not False
        or boundary.get("method_label_accesses") != 0
        or boundary.get("validation_payload_opens") != 0
        or boundary.get("test_payload_opens") != 0
        or boundary.get("proximal_definition")
        != "GT_Chebyshev_dilation_radius_5_input_pixels"
        or boundary.get("exact_geometric_view_sequence_validated") is not True
        or tuple(boundary.get("validated_geometric_view_sequence", ()))
        != ("identity", "hflip", "vflip", "hvflip")
    ):
        raise StageC0ProtocolError("outer label-isolation boundary differs")
    if manifest.get("critical_code_sha256") != _capture_code_hashes(contract):
        raise StageC0ProtocolError("outer critical-code seal differs")
    _validate_completion(
        path,
        manifest=manifest,
        phase="outer",
        dataset=dataset,
        formal=expected_formal,
    )
    outer = _read_jsonl(path / "outer_episodes.jsonl")
    flat = _read_jsonl(path / "space_alignments.jsonl")
    if len(outer) != expected_records or len(flat) != expected_records * 3:
        raise StageC0ProtocolError("outer record count differs")
    if sum(int(record.get("outer_train_target_accesses", -1)) for record in outer) != (
        expected_records
    ):
        raise StageC0ProtocolError("outer target-use count does not conserve")
    layouts = _validate_layouts(
        _load_json(bound_candidate_path / "parameter_layouts.json")
    )
    if manifest.get("parameter_layouts") != layouts:
        raise StageC0ProtocolError("outer parameter layout differs")
    candidate_records = _read_jsonl(
        bound_candidate_path / "candidate_episodes.jsonl"
    )
    candidate_gradients = {
        space: np.load(
            bound_candidate_path
            / f"proxy_gradients_{space.replace('-', '_')}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for space in PARAMETER_SPACES
    }
    expected_absolute_radii = {
        "R-E1": float(
            contract.raw["parameter_spaces"]["R-E1"][
                "normalized_virtual_step_l2_radius"
            ]
        ),
        "R-D0": float(
            contract.raw["parameter_spaces"]["R-D0"][
                "normalized_virtual_step_l2_radius"
            ]
        ),
        "P2": _p2_absolute_radius_from_checkpoint(
            contract, dataset=dataset, layout=layouts["P2"]
        ),
    }
    flat_cursor = 0
    for index, record in enumerate(outer):
        candidate_record = candidate_records[index]
        _validate_access_zero(record, label="outer episode")
        shared_candidate_fields = (
            "dataset",
            "condition",
            "corruption",
            "severity",
            "condition_index",
            "image_index",
            "image_id",
            "probe_index",
            "probe_id",
            "probe_kind",
            "teacher_student_gap_l1",
            "teacher_student_logit_gap_mean",
            "teacher_student_logit_gap_max",
            "active_episode",
            "active_pixel_count",
            "active_pixel_fraction",
            "active_target_weight",
            "active_background_weight",
        )
        if (
            record.get("schema_version") != 1
            or record.get("artifact_type") != "cr_sitta_stage_c0_outer_episode"
            or record.get("protocol_id") != PROTOCOL_ID
            or record.get("dataset") != dataset
            or record.get("outer_train_target_accesses") != 1
            or record.get("total_pixel_count") != 256 * 256
            or record.get("active_mask_count_reverified")
            != record.get("active_pixel_count")
            or not 0
            <= int(record.get("candidate_proximal_active_pixel_count", -1))
            <= int(record.get("active_pixel_count", -1))
            or record.get("candidate_proximal_active_episode")
            is not bool(record.get("candidate_proximal_active_pixel_count") > 0)
            or any(
                record.get(field) != candidate_record.get(field)
                for field in shared_candidate_fields
            )
        ):
            raise StageC0ProtocolError(f"outer episode differs at index {index}")
        spaces = _mapping(
            record.get("parameter_space_evidence"), "outer space evidence"
        )
        if tuple(spaces) != PARAMETER_SPACES:
            raise StageC0ProtocolError("outer space order differs")
        for space in PARAMETER_SPACES:
            evidence = _mapping(spaces[space], f"outer evidence {space}")
            for field in (
                "proxy_gradient_norm",
                "task_gradient_norm",
                "outer_task_gradient_cosine",
                "normalized_virtual_step_task_directional_derivative",
                "candidate_absolute_response_derivative",
                "candidate_local_contrast_derivative",
                "virtual_step_norm",
                "absolute_l2_radius",
            ):
                _finite_or_none(evidence.get(field), label=f"{space}.{field}")
            expected_mode, expected_value, expected_absolute = _space_radius_from_layout(
                contract, space=space, layout=_mapping(
                    manifest["parameter_layouts"][space], f"manifest layout {space}"
                )
            )
            absolute = float(evidence.get("absolute_l2_radius"))
            expected_absolute = expected_absolute_radii[space]
            absolute_ok = math.isclose(
                absolute,
                expected_absolute,
                rel_tol=1.0e-6,
                abs_tol=1.0e-12,
            )
            if (
                evidence.get("parameter_space") != space
                or evidence.get("radius_mode") != expected_mode
                or float(evidence.get("radius_value")) != expected_value
                or float(evidence.get("nonzero_epsilon")) != 1.0e-12
                or evidence.get("direction_formula")
                != "-g*min(1,radius/(norm+epsilon))"
                or evidence.get("threshold_crossing_role")
                != "report_only_not_stage_c0_hard_gate"
                or not absolute_ok
                or int(evidence.get("threshold_crossing_pixel_count", -1)) < 0
                or evidence.get("threshold_crossing_episode")
                is not bool(evidence.get("threshold_crossing_pixel_count") > 0)
            ):
                raise StageC0ProtocolError(f"outer {space} evidence differs")
            proxy = np.asarray(
                candidate_gradients[space][
                    int(record["condition_index"]),
                    int(record["image_index"]),
                    int(record["probe_index"]),
                ],
                dtype=np.float32,
            )
            proxy_norm = float(
                np.linalg.norm(proxy.astype(np.float64, copy=False))
            )
            recorded_proxy_norm = float(evidence["proxy_gradient_norm"])
            task_norm = float(evidence["task_gradient_norm"])
            epsilon = float(evidence["nonzero_epsilon"])
            proxy_nonzero = recorded_proxy_norm > epsilon
            task_nonzero = task_norm > epsilon
            both = proxy_nonzero and task_nonzero
            candidate_proxy_norm = float(
                candidate_record["gradient_evidence"][space][
                    "proxy_gradient_norm"
                ]
            )
            clipped_scale = (
                min(1.0, absolute / (recorded_proxy_norm + epsilon))
                if proxy_nonzero
                else 0.0
            )
            expected_step_norm = recorded_proxy_norm * clipped_scale
            step_norm = float(evidence["virtual_step_norm"])
            cosine = evidence["outer_task_gradient_cosine"]
            task_derivative = evidence[
                "normalized_virtual_step_task_directional_derivative"
            ]
            absolute_derivative = evidence[
                "candidate_absolute_response_derivative"
            ]
            contrast_derivative = evidence[
                "candidate_local_contrast_derivative"
            ]
            crossing_count = int(evidence["threshold_crossing_pixel_count"])
            if (
                evidence.get("finite") is not True
                or evidence.get("both_gradients_nonzero") is not both
                or not math.isclose(
                    recorded_proxy_norm,
                    proxy_norm,
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    recorded_proxy_norm,
                    candidate_proxy_norm,
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-12,
                )
                or task_norm < 0.0
                or not math.isclose(
                    step_norm,
                    expected_step_norm,
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-10,
                )
                or step_norm > absolute + 1.0e-7
                or step_norm > recorded_proxy_norm + 1.0e-7
                or ((cosine is None) != (not both))
                or ((task_derivative is None) != (not both))
                or ((absolute_derivative is None) != (not proxy_nonzero))
                or ((contrast_derivative is None) != (not proxy_nonzero))
                or (
                    cosine is not None
                    and not -1.000001 <= float(cosine) <= 1.000001
                )
                or (
                    task_derivative is not None
                    and abs(float(task_derivative))
                    > task_norm * step_norm + 1.0e-5
                )
                or (
                    both
                    and not math.isclose(
                        float(task_derivative),
                        -float(cosine) * task_norm * step_norm,
                        rel_tol=1.0e-5,
                        abs_tol=1.0e-7,
                    )
                )
                or not 0 <= crossing_count <= 256 * 256
                or (not proxy_nonzero and crossing_count != 0)
                or evidence.get("candidate_count")
                != candidate_record.get("candidate_count")
            ):
                raise StageC0ProtocolError(
                    f"outer {space} mathematical replay differs"
                )
            flat_record = flat[flat_cursor]
            flat_cursor += 1
            _validate_access_zero(flat_record, label="outer alignment")
            expected_flat = {
                "schema_version": 1,
                "artifact_type": "cr_sitta_stage_c0_outer_space_alignment",
                "protocol_id": PROTOCOL_ID,
                "dataset": record["dataset"],
                "condition": record["condition"],
                "corruption": record["corruption"],
                "severity": record["severity"],
                "image_index": record["image_index"],
                "image_id": record["image_id"],
                "probe_index": record["probe_index"],
                "probe_id": record["probe_id"],
                **dict(evidence),
                "method_label_accesses": 0,
                "validation_payload_opens": 0,
                "test_payload_opens": 0,
            }
            if flat_record != expected_flat:
                raise StageC0ProtocolError("flat/nested outer evidence differs")
    return manifest


def _space_radius_from_layout(
    contract: StageC0Contract,
    *,
    space: str,
    layout: Mapping[str, Any],
) -> tuple[str, float, float]:
    """Static part of radius verification (P2 absolute value is runtime-bound)."""

    del layout  # topology has already been checked by _validate_layouts.
    config = contract.raw["parameter_spaces"][space]
    if space in ("R-E1", "R-D0"):
        value = float(config["normalized_virtual_step_l2_radius"])
        return "absolute_l2", value, value
    value = float(config["normalized_virtual_step_relative_radius"])
    return "relative_to_source_parameter_l2", value, math.nan


def _fraction(value: Any) -> Fraction:
    if value is None:
        return Fraction(0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageC0ProtocolError("aggregate input must be numeric or null")
    if not math.isfinite(float(value)):
        raise StageC0ProtocolError("aggregate input contains NaN or Inf")
    return Fraction(Decimal(str(value)))


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _mean(values: Sequence[Any]) -> Fraction:
    if not values:
        return Fraction(0)
    return sum((_fraction(value) for value in values), Fraction(0)) / len(values)


def _family_id(corruption: str) -> str:
    if corruption not in {
        "gaussian_noise",
        "gaussian_blur",
        "low_contrast",
        "stripe_noise",
    }:
        raise StageC0ProtocolError(f"unsupported non-clean family: {corruption}")
    return corruption


def _load_gate_contract():
    from analysis.stage_c_science_gate_v1 import StageCGateConfig

    path = REPOSITORY / "configs/p3_stage_c_science_gate_v1.yaml"
    if path.is_symlink() or not path.is_file():
        raise StageC0ProtocolError("Stage-C science-gate config is missing/unsafe")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise StageC0ProtocolError("cannot parse Stage-C science-gate config") from exc
    gate_raw = _mapping(raw, "science-gate config").get("stage_c0_science_gate")
    return path, StageCGateConfig.from_mapping(_mapping(gate_raw, "stage_c0 gate"))


def build_aggregate_evidence(
    outer_records_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    identity_by_dataset: Mapping[str, Mapping[str, Any]],
    *,
    gate_config: Any,
) -> dict[str, Any]:
    """Summarize complete outer records without applying any science gate."""

    if tuple(outer_records_by_dataset) != DATASETS or tuple(identity_by_dataset) != DATASETS:
        raise StageC0ProtocolError("aggregate dataset roster/order differs")
    nonclean: list[Mapping[str, Any]] = []
    for dataset in DATASETS:
        records = tuple(outer_records_by_dataset[dataset])
        if len(records) != 13 * 64 * 2:
            raise StageC0ProtocolError(f"outer record count differs: {dataset}")
        selected = [record for record in records if record.get("corruption") != "clean"]
        if len(selected) != 12 * 64 * 2:
            raise StageC0ProtocolError(f"non-clean record count differs: {dataset}")
        nonclean.extend(selected)
    if len(nonclean) != 4608:
        raise StageC0ProtocolError("unique non-clean episode count differs")

    nonfinite_signal = 0
    scalar_signal_fields = (
        "teacher_student_gap_l1",
        "teacher_student_logit_gap_mean",
        "teacher_student_logit_gap_max",
        "active_pixel_fraction",
        "active_target_weight",
        "active_background_weight",
    )
    for record in nonclean:
        try:
            for field in scalar_signal_fields:
                if not math.isfinite(float(record[field])):
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            nonfinite_signal += 1
    by_probe = {
        probe: [record for record in nonclean if record.get("probe_id") == probe]
        for probe in PROBE_IDS
    }
    if any(len(values) != 2304 for values in by_probe.values()):
        raise StageC0ProtocolError("LF/HF episode conservation differs")
    signal_summary = {
        "teacher_student_probability_l1_mean": _fraction_json(
            _mean([record["teacher_student_gap_l1"] for record in nonclean])
        ),
        "teacher_student_logit_gap_mean": _fraction_json(
            _mean([record["teacher_student_logit_gap_mean"] for record in nonclean])
        ),
        "teacher_student_logit_gap_max": _fraction_json(
            max(
                (_fraction(record["teacher_student_logit_gap_max"]) for record in nonclean),
                default=Fraction(0),
            )
        ),
        "active_pixel_fraction_lf_mean": _fraction_json(
            _mean([record["active_pixel_fraction"] for record in by_probe["lf_mask"]])
        ),
        "active_pixel_fraction_hf_mean": _fraction_json(
            _mean([record["active_pixel_fraction"] for record in by_probe["hf_noise"]])
        ),
        "active_target_weight_lf_mean": _fraction_json(
            _mean([record["active_target_weight"] for record in by_probe["lf_mask"]])
        ),
        "active_background_weight_lf_mean": _fraction_json(
            _mean([record["active_background_weight"] for record in by_probe["lf_mask"]])
        ),
        "active_target_weight_hf_mean": _fraction_json(
            _mean([record["active_target_weight"] for record in by_probe["hf_noise"]])
        ),
        "active_background_weight_hf_mean": _fraction_json(
            _mean([record["active_background_weight"] for record in by_probe["hf_noise"]])
        ),
    }
    identity = {
        "comparison_count": sum(
            int(identity_by_dataset[dataset]["comparison_count"])
            for dataset in DATASETS
        ),
        "exact_match_count": sum(
            int(identity_by_dataset[dataset]["exact_match_count"])
            for dataset in DATASETS
        ),
        "mismatch_count": sum(
            int(identity_by_dataset[dataset]["mismatch_count"])
            for dataset in DATASETS
        ),
        "maximum_absolute_output_difference": _fraction_json(
            max(
                _fraction(
                    identity_by_dataset[dataset][
                        "maximum_absolute_output_difference"
                    ]
                )
                for dataset in DATASETS
            )
        ),
    }
    spaces = []
    for space in PARAMETER_SPACES:
        def evidence(record: Mapping[str, Any]) -> Mapping[str, Any]:
            return _mapping(record["parameter_space_evidence"][space], space)

        values = [evidence(record) for record in nonclean]
        nonfinite = 0
        for value in values:
            try:
                required = (
                    value["proxy_gradient_norm"],
                    value["task_gradient_norm"],
                    value["virtual_step_norm"],
                    value["absolute_l2_radius"],
                )
                if not all(math.isfinite(float(item)) for item in required):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                nonfinite += 1
        dataset_aggregates = []
        dataset_cosines: list[Fraction] = []
        for dataset in DATASETS:
            subset = [record for record in nonclean if record["dataset"] == dataset]
            valid_cosines = [
                evidence(record)["outer_task_gradient_cosine"]
                for record in subset
                if evidence(record)["both_gradients_nonzero"]
                and evidence(record)["outer_task_gradient_cosine"] is not None
            ]
            macro = _mean(valid_cosines)
            dataset_cosines.append(macro)
            dataset_aggregates.append(
                {
                    "dataset_id": dataset,
                    "nonclean_probe_episode_count": len(subset),
                    "both_gradients_nonzero_episode_count": len(valid_cosines),
                    "macro_outer_gradient_cosine": _fraction_json(macro),
                }
            )
        family_aggregates = []
        for family in gate_config.family_ids:
            subset = [record for record in nonclean if _family_id(record["corruption"]) == family]
            derivative = _mean(
                [
                    evidence(record)[
                        "normalized_virtual_step_task_directional_derivative"
                    ]
                    for record in subset
                ]
            )
            family_aggregates.append(
                {
                    "family_id": family,
                    "nonclean_probe_episode_count": len(subset),
                    "normalized_virtual_task_loss_directional_derivative": (
                        _fraction_json(derivative)
                    ),
                }
            )
        step = gate_config.step_contract_by_space[space]
        spaces.append(
            {
                "parameter_space": space,
                "nonclean_probe_episode_count": len(nonclean),
                "nonclean_nonfinite_measurement_episode_count": nonfinite,
                "nonclean_finite_nonzero_proxy_gradient_episode_count": sum(
                    bool(value["proxy_gradient_norm"] > 1.0e-12)
                    for value in values
                ),
                "nonclean_threshold_crossing_episode_count": sum(
                    bool(value["threshold_crossing_episode"]) for value in values
                ),
                "proxy_gradient_norm_mean": _fraction_json(
                    _mean([value["proxy_gradient_norm"] for value in values])
                ),
                "task_gradient_norm_mean": _fraction_json(
                    _mean([value["task_gradient_norm"] for value in values])
                ),
                "candidate_absolute_response_derivative_mean": _fraction_json(
                    _mean(
                        [
                            value["candidate_absolute_response_derivative"]
                            for value in values
                        ]
                    )
                ),
                "candidate_local_contrast_derivative_mean": _fraction_json(
                    _mean(
                        [
                            value["candidate_local_contrast_derivative"]
                            for value in values
                        ]
                    )
                ),
                "dataset_aggregates": dataset_aggregates,
                "family_aggregates": family_aggregates,
                "macro_outer_gradient_cosine": _fraction_json(
                    sum(dataset_cosines, Fraction(0)) / len(dataset_cosines)
                ),
                "virtual_step_contract": {
                    "radius_mode": step.radius_mode,
                    "radius_value": _fraction_json(step.radius_value),
                    "nonzero_epsilon": _fraction_json(step.nonzero_epsilon),
                    "direction": step.direction,
                    "clip_rule": step.clip_rule,
                },
                "identity": dict(identity),
            }
        )
    return {
        "schema_version": "stage_c0_aggregate_evidence_v1",
        "replicate_id": "R0",
        "scope": {
            "data_role": "train",
            "pilot_role": "fixed_Pilot64",
            "pilot_image_count_per_dataset": 64,
            "development_only": True,
            "paper_result": False,
            "thresholds_frozen_before_run": True,
            "validation_access_count": 0,
            "test_access_count": 0,
        },
        "mechanism_aggregate": {
            "mechanism_id": "ASB-SFR_C0",
            "nonclean_probe_episode_count": len(nonclean),
            "nonclean_nonfinite_signal_episode_count": nonfinite_signal,
            "nonclean_active_support_episode_count": sum(
                bool(record["active_episode"]) for record in nonclean
            ),
            "nonclean_active_pixel_count": sum(
                int(record["active_pixel_count"]) for record in nonclean
            ),
            "nonclean_candidate_proximal_active_episode_count": sum(
                bool(record["candidate_proximal_active_episode"])
                for record in nonclean
            ),
            "nonclean_candidate_proximal_active_pixel_count": sum(
                int(record["candidate_proximal_active_pixel_count"])
                for record in nonclean
            ),
            "nonclean_total_pixel_count": sum(
                int(record["total_pixel_count"]) for record in nonclean
            ),
            "signal_summary": signal_summary,
            "space_aggregates": spaces,
            "o4_activity": {
                "configured": False,
                "active_episode_count": 0,
                "contribution_episode_count": 0,
            },
        },
    }


def _run_aggregate(contract: StageC0Contract) -> Mapping[str, Any]:
    _verify_pre_run_freeze(contract)
    destination = _artifact_destination(
        contract,
        phase="aggregate",
        dataset=None,
        formal=True,
        smoke_id=None,
    )
    if destination.exists() or destination.is_symlink():
        return verify_aggregate_artifact(destination, contract=contract)
    gate_path, gate_config = _load_gate_contract()
    outer_records: dict[str, Sequence[Mapping[str, Any]]] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    outer_bindings: dict[str, Any] = {}
    for dataset in DATASETS:
        candidate_path = _artifact_destination(
            contract,
            phase="candidate",
            dataset=dataset,
            formal=True,
            smoke_id=None,
        )
        candidate_manifest, token = verify_candidate_artifact(
            candidate_path,
            contract=contract,
            dataset=dataset,
            expected_formal=True,
        )
        outer_path = _artifact_destination(
            contract,
            phase="outer",
            dataset=dataset,
            formal=True,
            smoke_id=None,
        )
        outer_manifest = verify_outer_artifact(
            outer_path,
            contract=contract,
            dataset=dataset,
            expected_formal=True,
            candidate_token=token,
        )
        outer_records[dataset] = _read_jsonl(outer_path / "outer_episodes.jsonl")
        identities[dataset] = _mapping(
            candidate_manifest["identity_adapter"], f"{dataset} identity"
        )
        outer_bindings[dataset] = {
            "path": str(outer_path.relative_to(contract.repository)),
            "manifest_sha256": sha256_file(outer_path / "manifest.json"),
            "payload_tree_sha256": outer_manifest["payload_tree_sha256"],
        }
    evidence = build_aggregate_evidence(
        outer_records, identities, gate_config=gate_config
    )
    # Gate logic remains wholly in the pure gate module.
    from analysis.stage_c_science_gate_v1 import (
        authorize_stage_c_followup,
        evaluate_stage_c0_science_gate,
    )

    receipt = evaluate_stage_c0_science_gate(evidence, gate_config)
    authorization = authorize_stage_c_followup(receipt)
    staging = _staging_directory(destination)
    try:
        _write_json(staging / "aggregate_evidence.json", evidence)
        _write_json(staging / "science_decision_receipt.json", receipt.to_receipt())
        _write_json(staging / "stage_c1_authorization.json", asdict(authorization))
        manifest = {
            **_runtime_manifest_base(
                contract,
                artifact_type="cr_sitta_stage_c0_aggregate",
                phase="aggregate",
                dataset=None,
                formal=True,
            ),
            "replicate_id": "R0",
            "dataset_count": len(DATASETS),
            "unique_nonclean_probe_episode_count": 4608,
            "parameter_spaces": list(PARAMETER_SPACES),
            "outer_artifacts": outer_bindings,
            "science_gate": {
                "config_path": str(gate_path.relative_to(contract.repository)),
                "config_sha256": sha256_file(gate_path),
                "module_path": "analysis/stage_c_science_gate_v1.py",
                "module_sha256": sha256_file(
                    contract.repository / "analysis/stage_c_science_gate_v1.py"
                ),
                "gate_called_directly": True,
            },
            "protocol_status": receipt.protocol_status,
            "scientific_status": receipt.scientific_status,
            "eligible_space_ids": list(receipt.eligible_space_ids),
            "stage_c1_allowed": authorization.stage_c1_allowed,
            "stage_c_r1_r2_allowed": False,
            "formal_test_allowed": False,
            "critical_code_sha256": _capture_code_hashes(contract),
        }

        def guard() -> None:
            _assert_contract_unchanged(contract)
            if sha256_file(gate_path) != manifest["science_gate"]["config_sha256"]:
                raise StageC0ProtocolError("science-gate config changed")
            if sha256_file(
                contract.repository / "analysis/stage_c_science_gate_v1.py"
            ) != manifest["science_gate"]["module_sha256"]:
                raise StageC0ProtocolError("science-gate module changed")
            for dataset in DATASETS:
                candidate_path = _artifact_destination(
                    contract,
                    phase="candidate",
                    dataset=dataset,
                    formal=True,
                    smoke_id=None,
                )
                _candidate_manifest, candidate_token = verify_candidate_artifact(
                    candidate_path,
                    contract=contract,
                    dataset=dataset,
                    expected_formal=True,
                )
                outer_path = _repository_path(
                    contract.repository,
                    outer_bindings[dataset]["path"],
                    f"{dataset} outer artifact",
                )
                replay_outer = verify_outer_artifact(
                    outer_path,
                    contract=contract,
                    dataset=dataset,
                    expected_formal=True,
                    candidate_token=candidate_token,
                )
                if (
                    sha256_file(outer_path / "manifest.json")
                    != outer_bindings[dataset]["manifest_sha256"]
                    or replay_outer.get("payload_tree_sha256")
                    != outer_bindings[dataset]["payload_tree_sha256"]
                ):
                    raise StageC0ProtocolError("outer artifact changed")

        _publish_artifact(staging, destination, manifest, pre_rename_guard=guard)
        return verify_aggregate_artifact(destination, contract=contract)
    except BaseException:
        _remove_owned_staging(staging)
        raise


def verify_aggregate_artifact(
    path: Path, *, contract: StageC0Contract
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise StageC0ProtocolError(f"aggregate artifact missing/unsafe: {path}")
    manifest = _load_json(path / "manifest.json")
    _validate_manifest_freeze_binding(manifest, contract=contract)
    _validate_access_zero(manifest, label="aggregate manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "cr_sitta_stage_c0_aggregate"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("config_sha256") != contract.config_sha256
        or manifest.get("phase") != "aggregate"
        or manifest.get("dataset") is not None
        or manifest.get("formal") is not True
        or manifest.get("dataset_count") != 3
        or manifest.get("unique_nonclean_probe_episode_count") != 4608
        or tuple(manifest.get("parameter_spaces", ())) != PARAMETER_SPACES
        or manifest.get("stage_c_r1_r2_allowed") is not False
        or manifest.get("formal_test_allowed") is not False
        or manifest.get("critical_code_sha256") != _capture_code_hashes(contract)
    ):
        raise StageC0ProtocolError("aggregate manifest semantics differ")
    _validate_completion(
        path,
        manifest=manifest,
        phase="aggregate",
        dataset=None,
        formal=True,
    )
    gate_path, gate_config = _load_gate_contract()
    gate_binding = _mapping(manifest.get("science_gate"), "aggregate science gate")
    if (
        gate_binding.get("config_path")
        != str(gate_path.relative_to(contract.repository))
        or gate_binding.get("config_sha256") != sha256_file(gate_path)
        or gate_binding.get("module_path") != "analysis/stage_c_science_gate_v1.py"
        or gate_binding.get("module_sha256")
        != sha256_file(contract.repository / "analysis/stage_c_science_gate_v1.py")
        or gate_binding.get("gate_called_directly") is not True
    ):
        raise StageC0ProtocolError("aggregate science-gate binding differs")

    outer_bindings = _mapping(
        manifest.get("outer_artifacts"), "aggregate outer artifacts"
    )
    if tuple(outer_bindings) != DATASETS:
        raise StageC0ProtocolError("aggregate outer-artifact roster/order differs")
    outer_records: dict[str, Sequence[Mapping[str, Any]]] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    for dataset in DATASETS:
        binding = _mapping(
            outer_bindings[dataset], f"aggregate outer artifact {dataset}"
        )
        if set(binding) != {"path", "manifest_sha256", "payload_tree_sha256"}:
            raise StageC0ProtocolError(
                f"aggregate outer-artifact fields differ: {dataset}"
            )
        candidate_path = _artifact_destination(
            contract,
            phase="candidate",
            dataset=dataset,
            formal=True,
            smoke_id=None,
        )
        candidate_manifest, candidate_token = verify_candidate_artifact(
            candidate_path,
            contract=contract,
            dataset=dataset,
            expected_formal=True,
        )
        expected_outer_path = _artifact_destination(
            contract,
            phase="outer",
            dataset=dataset,
            formal=True,
            smoke_id=None,
        )
        bound_outer_path = _repository_path(
            contract.repository,
            binding.get("path"),
            f"aggregate outer path {dataset}",
        )
        if bound_outer_path != expected_outer_path:
            raise StageC0ProtocolError(
                f"aggregate outer-artifact path differs: {dataset}"
            )
        outer_manifest = verify_outer_artifact(
            bound_outer_path,
            contract=contract,
            dataset=dataset,
            expected_formal=True,
            candidate_token=candidate_token,
        )
        if (
            binding.get("manifest_sha256")
            != sha256_file(bound_outer_path / "manifest.json")
            or binding.get("payload_tree_sha256")
            != outer_manifest.get("payload_tree_sha256")
        ):
            raise StageC0ProtocolError(
                f"aggregate outer-artifact digest differs: {dataset}"
            )
        outer_records[dataset] = _read_jsonl(
            bound_outer_path / "outer_episodes.jsonl"
        )
        identities[dataset] = _mapping(
            candidate_manifest.get("identity_adapter"),
            f"aggregate identity evidence {dataset}",
        )
    from analysis.stage_c_science_gate_v1 import (
        authorize_stage_c_followup,
        evaluate_stage_c0_science_gate,
    )

    evidence = _load_json(path / "aggregate_evidence.json")
    recomputed_evidence = build_aggregate_evidence(
        outer_records,
        identities,
        gate_config=gate_config,
    )
    if evidence != recomputed_evidence:
        raise StageC0ProtocolError("aggregate evidence differs from outer artifacts")
    recomputed = evaluate_stage_c0_science_gate(evidence, gate_config)
    if _load_json(path / "science_decision_receipt.json") != recomputed.to_receipt():
        raise StageC0ProtocolError("aggregate science receipt differs")
    authorization = authorize_stage_c_followup(recomputed)
    if _load_json(path / "stage_c1_authorization.json") != asdict(authorization):
        raise StageC0ProtocolError("aggregate authorization differs")
    if (
        manifest.get("protocol_status") != recomputed.protocol_status
        or manifest.get("scientific_status") != recomputed.scientific_status
        or manifest.get("eligible_space_ids") != list(recomputed.eligible_space_ids)
        or manifest.get("stage_c1_allowed") is not authorization.stage_c1_allowed
    ):
        raise StageC0ProtocolError("aggregate gate projection differs")
    return manifest


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _add_execution_scope(parser: argparse.ArgumentParser, *, candidate: bool) -> None:
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--engineering-smoke",
        action="store_true",
        help="write only under output.engineering_phase; never formal evidence",
    )
    parser.add_argument("--smoke-id")
    if candidate:
        parser.add_argument("--smoke-images", type=int, default=1)


def _scope_from_args(args: argparse.Namespace) -> tuple[bool, str | None]:
    engineering = bool(getattr(args, "engineering_smoke", False))
    smoke_id = getattr(args, "smoke_id", None)
    if engineering and not smoke_id:
        raise StageC0ProtocolError("--engineering-smoke requires --smoke-id")
    if not engineering and smoke_id is not None:
        raise StageC0ProtocolError("--smoke-id is forbidden for formal execution")
    return not engineering, smoke_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train-only CR-SITTA Stage-C0 two-phase signal audit"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "print-config-sha256",
        help="print current YAML digest without treating it as frozen",
    )
    subparsers.add_parser(
        "verify-config", help="verify the runner-pinned frozen protocol and inputs"
    )
    subparsers.add_parser(
        "freeze",
        help="atomically publish the no-replace pre-run code/input freeze receipt",
    )
    candidate = subparsers.add_parser(
        "candidate", help="run label-free Pilot64 signal/proxy capture"
    )
    _add_execution_scope(candidate, candidate=True)
    outer = subparsers.add_parser(
        "outer", help="verify candidate, then run train-target outer audit"
    )
    _add_execution_scope(outer, candidate=False)
    subparsers.add_parser(
        "aggregate",
        help="summarize three formal outers and call the pure Stage-C0 gate",
    )
    verify = subparsers.add_parser("verify", help="replay an existing artifact")
    verify.add_argument("--phase", required=True, choices=("candidate", "outer", "aggregate"))
    verify.add_argument("--dataset", choices=DATASETS)
    verify.add_argument("--engineering-smoke", action="store_true")
    verify.add_argument("--smoke-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPOSITORY / args.config
    if args.command == "print-config-sha256":
        print(current_config_sha256(config_path))
        return 0
    contract = load_contract(config_path)
    if args.command == "freeze":
        result = _run_freeze(contract)
        _print_json(result)
        return 0
    _receipt, receipt_sha256 = _verify_pre_run_freeze(contract)
    if args.command == "verify-config":
        _print_json(
            {
                "protocol_id": PROTOCOL_ID,
                "config_path": str(contract.config_path),
                "config_sha256": contract.config_sha256,
                "pre_run_freeze_receipt": str(_freeze_receipt_path(contract)),
                "pre_run_freeze_receipt_sha256": receipt_sha256,
                "valid": True,
                "data_role": "train_fixed_Pilot64",
                "no_validation_split": True,
                "validation_payload_opens": 0,
                "test_payload_opens": 0,
            }
        )
        return 0
    if args.command == "candidate":
        formal, smoke_id = _scope_from_args(args)
        result = _run_candidate(
            contract,
            dataset=args.dataset,
            device_name=args.device,
            formal=formal,
            smoke_id=smoke_id,
            smoke_images=(64 if formal else args.smoke_images),
        )
        _print_json(result)
        return 0
    if args.command == "outer":
        formal, smoke_id = _scope_from_args(args)
        result = _run_outer(
            contract,
            dataset=args.dataset,
            device_name=args.device,
            formal=formal,
            smoke_id=smoke_id,
        )
        _print_json(result)
        return 0
    if args.command == "aggregate":
        result = _run_aggregate(contract)
        _print_json(result)
        return 0
    if args.command == "verify":
        formal, smoke_id = _scope_from_args(args)
        phase = str(args.phase)
        if phase in ("candidate", "outer") and args.dataset is None:
            raise StageC0ProtocolError(f"verify --phase {phase} requires --dataset")
        if phase == "aggregate" and (args.dataset is not None or not formal):
            raise StageC0ProtocolError("aggregate verification is formal/global only")
        destination = _artifact_destination(
            contract,
            phase=phase,
            dataset=args.dataset,
            formal=formal,
            smoke_id=smoke_id,
        )
        if phase == "candidate":
            result = verify_candidate_artifact(
                destination,
                contract=contract,
                dataset=args.dataset,
                expected_formal=formal,
            )[0]
        elif phase == "outer":
            result = verify_outer_artifact(
                destination,
                contract=contract,
                dataset=args.dataset,
                expected_formal=formal,
            )
        else:
            result = verify_aggregate_artifact(destination, contract=contract)
        _print_json(result)
        return 0
    raise StageC0ProtocolError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StageC0ProtocolError as exc:
        print(f"Stage-C0 protocol error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


__all__ = [
    "DEFAULT_CONFIG",
    "FROZEN_CONFIG_SHA256",
    "PROTOCOL_ID",
    "StageC0Contract",
    "StageC0ProtocolError",
    "VerifiedCandidateArtifact",
    "build_aggregate_evidence",
    "build_probe_image",
    "current_config_sha256",
    "derive_probe_seed",
    "load_contract",
    "main",
    "open_outer_train_targets",
    "verify_aggregate_artifact",
    "verify_candidate_artifact",
    "verify_outer_artifact",
]
