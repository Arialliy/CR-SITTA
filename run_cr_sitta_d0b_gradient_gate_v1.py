#!/usr/bin/env python3
"""Checkpoint-rebound, train-only CR-SITTA D0-B gradient gate.

D0-B deliberately rebuilds every model-dependent Stage-C0 artifact from the
three D0-A epoch-1000 safe checkpoints.  The old Stage-C0 runner is imported
only as hash-bound execution logic for candidate/outer mathematics; none of
its baseline-bound teacher, candidate, gradient, outer, or aggregate artifacts
is an input.  The only reusable payload is the separately verified image-only
Pilot64 corruption cache.

The phase order is fail closed::

    preflight -> freeze -> teacher -> candidate -> outer -> aggregate

All phases are train-only.  The aggregate can authorize only D1
train-internal OOF work and can never authorize formal test evaluation.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Final
import uuid
from fractions import Fraction

import numpy as np
import yaml


REPOSITORY: Final = Path(__file__).resolve().parent
DEFAULT_CONFIG: Final = REPOSITORY / "configs/cr_sitta_d0b_gradient_gate_v1.yaml"
PROTOCOL_ID: Final = "cr-sitta-d0b-checkpoint-rebound-gradient-gate-v1"
LEGACY_C0_PROTOCOL_ID: Final = "cr-sitta-p3-stage-c0-signal-audit-v2"
PROBE_SEED_NAMESPACE: Final = "cr-sitta-p3-stage-c0-signal-audit-v1"
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
PROBABILITY_SHAPE: Final = (1, 256, 256)
BASE_VIEW_NAMES: Final = (
    "identity",
    "hflip",
    "vflip",
    "hvflip",
    "context_tile",
)
SHA256_HEX: Final = frozenset("0123456789abcdef")


class D0BProtocolError(RuntimeError):
    """A D0-B prerequisite, artifact, or frozen invariant failed."""


@dataclass(frozen=True, slots=True)
class D0BContract:
    repository: Path
    config_path: Path
    config_sha256: str
    raw: Mapping[str, Any]

    @property
    def output_root(self) -> Path:
        return _repository_path(self.repository, self.raw["output"]["root"], "output")


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
        raise D0BProtocolError("value is not canonical-JSON safe") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    if path.is_symlink() or not path.is_file():
        raise D0BProtocolError(f"expected regular non-symlink file: {path}")
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
        raise D0BProtocolError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0BProtocolError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise D0BProtocolError(f"{label} must be a sequence")
    return value


def _repository_path(repository: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise D0BProtocolError(f"{label} must be a non-empty repository-relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise D0BProtocolError(f"{label} must be repository-relative")
    result = repository / relative
    if not result.absolute().is_relative_to(repository.absolute()):
        raise D0BProtocolError(f"{label} escapes repository")
    return result


def _condition_key(corruption: str, severity: int) -> str:
    return "clean_S0" if corruption == "clean" else f"{corruption}_S{severity}"


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise D0BProtocolError(f"JSON input missing/unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise D0BProtocolError(f"cannot parse JSON: {path}") from exc
    if not isinstance(value, dict):
        raise D0BProtocolError(f"JSON root must be a mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise D0BProtocolError(f"JSONL input missing/unsafe: {path}")
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise D0BProtocolError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise D0BProtocolError(f"JSONL row is not a mapping at {path}:{line_number}")
            result.append(value)
    return result


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    return _sha256_bytes(
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _verify_exact_file(path: Path, expected: Any, label: str) -> str:
    expected_sha = _require_sha256(expected, f"{label} SHA-256")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise D0BProtocolError(
            f"frozen input changed for {label}: expected {expected_sha}, got {actual}"
        )
    return actual


def _raw_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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
    result: dict[str, Any] = {}
    excluded_set = set(excluded)
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise D0BProtocolError(f"artifact contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise D0BProtocolError(f"artifact entry is not regular: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _validate_contract_semantics(raw: Mapping[str, Any]) -> None:
    if raw.get("schema_version") != 1 or raw.get("protocol_id") != PROTOCOL_ID:
        raise D0BProtocolError("config schema/protocol differs")
    scope = _mapping(raw.get("scope"), "scope")
    expected_scope = {
        "method_name": "CR-SITTA",
        "method_stage": "D0-B",
        "data_role": "train",
        "split_role": "frozen_Pilot64",
        "no_validation_split": True,
        "development_only": True,
        "paper_result": False,
        "use_validation_payload": False,
        "use_test_payload": False,
        "validation_access_count": 0,
        "test_access_count": 0,
        "pilot_image_count_per_dataset": 64,
        "dataset_count": 3,
        "condition_count_per_dataset": 13,
        "probe_count": 2,
        "parameter_space_count": 3,
        "expected_candidate_episode_count_per_dataset": 1664,
        "expected_outer_episode_count_per_dataset": 1664,
        "expected_nonclean_probe_episode_count_total": 4608,
        "seed": 42,
    }
    for field, expected in expected_scope.items():
        if scope.get(field) != expected:
            raise D0BProtocolError(f"scope.{field} must be {expected!r}")
    if tuple(raw.get("datasets", {})) != DATASETS:
        raise D0BProtocolError("dataset roster/order differs")
    if tuple(tuple(item) for item in raw.get("ordered_conditions", ())) != CONDITIONS:
        raise D0BProtocolError("condition roster/order differs")
    if tuple(raw.get("parameter_spaces", {}).get("ordered", ())) != PARAMETER_SPACES:
        raise D0BProtocolError("parameter-space roster/order differs")
    if tuple(item.get("id") for item in raw.get("deterioration_bank", {}).get("probes", ())) != PROBE_IDS:
        raise D0BProtocolError("probe roster/order differs")
    bank = _mapping(raw.get("deterioration_bank"), "deterioration bank")
    if (
        bank.get("benchmark_condition_name_available_to_probe") is not False
        or bank.get("benchmark_severity_available_to_probe") is not False
        or bank.get("probe_seed_namespace") != PROBE_SEED_NAMESPACE
    ):
        raise D0BProtocolError("probe firewall/seed namespace differs")
    policy = _mapping(raw.get("artifact_policy"), "artifact policy")
    if (
        policy.get("may_authorize_only") != "D1_train_internal_OOF"
        or policy.get("formal_test_allowed") is not False
        or policy.get("permitted_reuse", {}).get("artifact")
        != "image_only_13_condition_Pilot64_cache"
    ):
        raise D0BProtocolError("artifact/authorization policy differs")
    required_rebuild = {
        "strong_teacher",
        "candidate_masks_and_region_weights",
        "source_probabilities",
        "proxy_gradients",
        "outer_task_gradients",
        "aggregate_evidence",
    }
    if set(policy.get("checkpoint_dependent_artifacts_must_be_rebuilt", ())) != required_rebuild:
        raise D0BProtocolError("checkpoint-dependent rebuild roster differs")
    forbidden = tuple(str(item) for item in policy.get("forbidden_input_roots", ()))
    if not forbidden or any(not value.startswith("results/") for value in forbidden):
        raise D0BProtocolError("forbidden input roots are invalid")
    output_root = str(raw.get("output", {}).get("root", ""))
    for dataset, record_raw in raw["datasets"].items():
        record = _mapping(record_raw, f"dataset {dataset}")
        if record.get("checkpoint_role") != "epoch_1000_train_only_safe":
            raise D0BProtocolError(f"{dataset} checkpoint role differs")
        checkpoint = str(record.get("checkpoint_path", ""))
        receipt = str(record.get("safe_export_receipt", ""))
        teacher = str(record.get("teacher_root", ""))
        if "best_miou" in checkpoint or "best_pd" in checkpoint:
            raise D0BProtocolError(f"{dataset} binds a test-selected checkpoint")
        if any(checkpoint.startswith(root) or receipt.startswith(root) for root in forbidden):
            raise D0BProtocolError(f"{dataset} checkpoint input uses a forbidden root")
        if teacher != f"{output_root}/teacher_phase/R0/{dataset}":
            raise D0BProtocolError(f"{dataset} teacher root is not D0-B-local")
    teacher = _mapping(raw.get("teacher"), "teacher")
    if (
        teacher.get("source")
        != "freshly_rebuilt_from_each_d0a_epoch1000_safe_checkpoint"
        or tuple(teacher.get("accepted_view_names", ())) != BASE_VIEW_NAMES[:4]
        or tuple(teacher.get("stored_view_names", ())) != BASE_VIEW_NAMES
        or teacher.get("aggregation") != "median"
    ):
        raise D0BProtocolError("teacher rebuild contract differs")
    # Parse the exact inherited gate schema now; no result-dependent threshold
    # movement is accepted later.
    from analysis.stage_c_science_gate_v1 import StageCGateConfig

    gate = StageCGateConfig.from_mapping(_mapping(raw.get("stage_c0_signal_gate"), "gate"))
    if (
        gate.expected_nonclean_probe_episode_count != 4608
        or gate.minimum_nonclean_active_support_episode_fraction != Fraction(3, 10)
        or gate.minimum_candidate_proximal_active_episode_fraction_among_active
        != Fraction(1, 20)
        or gate.macro_cosine_strictly_greater_than != Fraction(2, 25)
        or gate.minimum_positive_dataset_cosines != 2
        or gate.minimum_improving_families != 3
    ):
        raise D0BProtocolError("D0-B scientific thresholds differ from v7")


def load_contract(config_path: Path = DEFAULT_CONFIG) -> D0BContract:
    absolute = config_path.expanduser().absolute()
    if absolute.is_symlink() or not absolute.is_file():
        raise D0BProtocolError(f"config missing/unsafe: {absolute}")
    if not absolute.is_relative_to(REPOSITORY.absolute()):
        raise D0BProtocolError("config must be inside the repository")
    payload = absolute.read_bytes()
    try:
        raw = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise D0BProtocolError("D0-B config cannot be parsed") from exc
    if not isinstance(raw, dict):
        raise D0BProtocolError("D0-B config root must be a mapping")
    _validate_contract_semantics(raw)
    return D0BContract(REPOSITORY, absolute, _sha256_bytes(payload), raw)


def _verify_static_bindings(contract: D0BContract) -> None:
    for group_name in ("lineage", "frozen_formula_bindings"):
        group = _mapping(contract.raw[group_name], group_name)
        for name, value in group.items():
            if not isinstance(value, Mapping) or "path" not in value or "sha256" not in value:
                continue
            # The predecessor negative result is provenance only.  D0-B must
            # never open any checkpoint-dependent predecessor numeric artifact,
            # including merely to re-hash or inspect it at preflight time.
            if group_name == "lineage" and name == "old_c0_negative_result":
                continue
            _verify_exact_file(
                _repository_path(contract.repository, value["path"], f"{group_name}.{name}"),
                value["sha256"],
                f"{group_name}.{name}",
            )
    prior = _mapping(contract.raw["lineage"]["prior_c0_logic"], "prior C0 logic")
    if prior.get("reuse_scope") != "formulas_and_verified_execution_logic_only" or prior.get("numeric_artifact_reuse") != "forbidden":
        raise D0BProtocolError("prior C0 reuse boundary differs")
    pure_gate = _mapping(
        contract.raw["lineage"]["pure_science_gate"], "pure science gate"
    )
    _verify_exact_file(
        _repository_path(
            contract.repository, pure_gate["module_path"], "pure gate module"
        ),
        pure_gate["module_sha256"],
        "pure gate module",
    )
    _verify_exact_file(
        _repository_path(
            contract.repository, pure_gate["config_path"], "pure gate config"
        ),
        pure_gate["config_sha256"],
        "pure gate config",
    )


def _verify_cache_and_split(contract: D0BContract, dataset: str) -> dict[str, Any]:
    record = _mapping(contract.raw["datasets"][dataset], f"dataset {dataset}")
    for field, hash_field in (
        ("train_split", "train_split_sha256"),
        ("pilot_ids", "pilot_ids_file_sha256"),
    ):
        path = _repository_path(contract.repository, record[field], f"{dataset} {field}")
        if "trainval" in path.name.lower() or "/test_" in path.as_posix().lower():
            raise D0BProtocolError(f"{dataset} is not bound to the official train split")
        _verify_exact_file(path, record[hash_field], f"{dataset} {field}")
    pilot_ids = tuple(
        line.strip()
        for line in _repository_path(
            contract.repository, record["pilot_ids"], f"{dataset} Pilot64"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if (
        len(pilot_ids) != PILOT_COUNT
        or len(set(pilot_ids)) != PILOT_COUNT
        or _ordered_ids_sha256(pilot_ids) != record["ordered_pilot64_image_ids_sha256"]
    ):
        raise D0BProtocolError(f"{dataset} Pilot64 identity differs")
    cache = _repository_path(contract.repository, record["cache_root"], f"{dataset} cache")
    for name, hash_field in (
        ("manifest.json", "cache_manifest_sha256"),
        ("method_input_manifest.json", "cache_method_manifest_sha256"),
        ("COMPLETE.json", "cache_complete_sha256"),
    ):
        _verify_exact_file(cache / name, record[hash_field], f"{dataset} cache {name}")
    manifest = _load_json(cache / "manifest.json")
    method = _load_json(cache / "method_input_manifest.json")
    complete = _load_json(cache / "COMPLETE.json")
    if (
        manifest.get("cache_content_sha256") != record["cache_content_sha256"]
        or method.get("dataset") != dataset
        or method.get("targets_exposed") is not False
        or method.get("forbidden_fields") != ["ground_truth", "gt", "label", "mask", "target"]
        or tuple(method.get("image_ids", ())) != pilot_ids
        or complete.get("complete") is not True
        or complete.get("test_images_opened") != 0
        or complete.get("test_masks_opened") != 0
    ):
        raise D0BProtocolError(f"{dataset} image-only cache semantics differ")
    return {
        "train_split": {"path": record["train_split"], "sha256": record["train_split_sha256"]},
        "pilot_ids": {
            "path": record["pilot_ids"],
            "file_sha256": record["pilot_ids_file_sha256"],
            "ordered_ids_sha256": record["ordered_pilot64_image_ids_sha256"],
            "count": PILOT_COUNT,
        },
        "image_only_cache": {
            "root": record["cache_root"],
            "content_sha256": record["cache_content_sha256"],
            "manifest_sha256": record["cache_manifest_sha256"],
            "method_manifest_sha256": record["cache_method_manifest_sha256"],
            "complete_sha256": record["cache_complete_sha256"],
            "targets_exposed": False,
        },
    }


def _verify_safe_export(contract: D0BContract, dataset: str) -> dict[str, Any]:
    record = _mapping(contract.raw["datasets"][dataset], f"dataset {dataset}")
    receipt_path = _repository_path(
        contract.repository, record["safe_export_receipt"], f"{dataset} safe export receipt"
    )
    checkpoint_path = _repository_path(
        contract.repository, record["checkpoint_path"], f"{dataset} safe checkpoint"
    )
    try:
        from export_cr_sitta_d0a_safe_checkpoint import (
            validate_repository_models,
            verify_safe_export,
        )

        receipt = verify_safe_export(
            receipt_path, state_dict_validators=(validate_repository_models,)
        )
    except Exception as exc:
        raise D0BProtocolError(f"{dataset} safe export did not verify: {exc}") from exc
    if receipt.get("dataset") != dataset:
        raise D0BProtocolError(f"{dataset} safe-export dataset identity differs")
    contract_record = _mapping(receipt.get("checkpoint_contract"), "checkpoint contract")
    firewall = _mapping(receipt.get("access_firewall"), "safe-export firewall")
    expected_firewall_fields = {
        "validation_split_reads",
        "validation_image_opens",
        "validation_mask_opens",
        "test_split_reads",
        "test_image_opens",
        "test_mask_opens",
    }
    if (
        contract_record.get("architecture") != "MSHNet_NSFPN"
        or contract_record.get("method_name") != "CR-SITTA"
        or contract_record.get("method_stage") != "D0-A"
        or contract_record.get("epoch") != 1000
        or contract_record.get("test_selected") is not False
        or contract_record.get("selection_rule") != "fixed_final_epoch_train_only"
        or contract_record.get("state_dict_keys") != 505
        or contract_record.get("torch_load_weights_only") is not True
        or contract_record.get("repository_model_loads_verified") is not True
        or set(firewall) != expected_firewall_fields
        or any(int(firewall[field]) != 0 for field in expected_firewall_fields)
    ):
        raise D0BProtocolError(f"{dataset} safe checkpoint semantics differ")
    artifacts = _mapping(receipt.get("artifacts"), "safe-export artifacts")
    required_artifacts = {
        "safe_checkpoint",
        "run_contract",
        "full_train_freeze",
        "protocol_config",
        "train_split",
        "completion_summary",
    }
    if not required_artifacts.issubset(artifacts):
        raise D0BProtocolError(f"{dataset} SAFE_EXPORT lacks required bindings")
    safe = _mapping(artifacts.get("safe_checkpoint"), "safe checkpoint binding")
    if Path(str(safe.get("path"))).resolve() != checkpoint_path.resolve():
        raise D0BProtocolError(f"{dataset} configured checkpoint differs from SAFE_EXPORT")
    checkpoint_sha = _require_sha256(safe.get("sha256"), f"{dataset} checkpoint")
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise D0BProtocolError(f"{dataset} safe checkpoint hash changed")
    full_freeze = _mapping(artifacts["full_train_freeze"], "full train freeze")
    expected_full = _mapping(
        contract.raw["lineage"]["d0a_full_train_freeze"], "expected full train freeze"
    )
    if (
        Path(str(full_freeze.get("path"))).resolve()
        != _repository_path(
            contract.repository, expected_full["path"], "expected full train freeze"
        ).resolve()
        or full_freeze.get("sha256") != expected_full["sha256"]
    ):
        raise D0BProtocolError(f"{dataset} SAFE_EXPORT binds another D0-A freeze")
    train_split = _mapping(artifacts["train_split"], "safe-export train split")
    if (
        Path(str(train_split.get("path"))).resolve()
        != _repository_path(
            contract.repository, record["train_split"], f"{dataset} train split"
        ).resolve()
        or train_split.get("sha256") != record["train_split_sha256"]
    ):
        raise D0BProtocolError(f"{dataset} SAFE_EXPORT binds another train split")
    return {
        "checkpoint": {
            "path": record["checkpoint_path"],
            "sha256": checkpoint_sha,
            "role": record["checkpoint_role"],
            "epoch": 1000,
            "test_selected": False,
        },
        "safe_export_receipt": {
            "path": record["safe_export_receipt"],
            "sha256": sha256_file(receipt_path),
        },
        "safe_export_artifacts": {
            name: {
                "path": str(_mapping(value, name).get("path")),
                "sha256": _require_sha256(_mapping(value, name).get("sha256"), name),
            }
            for name, value in artifacts.items()
        },
    }


def inspect_preflight(contract: D0BContract) -> dict[str, Any]:
    """Return readiness without writing any artifact or initializing CUDA."""

    errors: list[str] = []
    datasets: dict[str, Any] = {}
    try:
        _validate_contract_semantics(contract.raw)
        _verify_static_bindings(contract)
    except Exception as exc:
        errors.append(str(exc))
    for dataset in DATASETS:
        record: dict[str, Any] = {"ready": False}
        try:
            record.update(_verify_cache_and_split(contract, dataset))
            record.update(_verify_safe_export(contract, dataset))
            record["ready"] = True
        except Exception as exc:
            record["error"] = str(exc)
            errors.append(f"{dataset}: {exc}")
        datasets[dataset] = record
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0b_read_only_preflight",
        "protocol_id": PROTOCOL_ID,
        "ready": not errors and all(value["ready"] for value in datasets.values()),
        "config_path": str(contract.config_path.relative_to(contract.repository)),
        "config_sha256": contract.config_sha256,
        "dataset_count": len(DATASETS),
        "datasets": datasets,
        "errors": errors,
        "writes_performed": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
        "formal_test_allowed": False,
    }


def _capture_code_hashes(contract: D0BContract) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in contract.raw["implementation"]["critical_code_paths"]:
        path = _repository_path(contract.repository, value, "critical code path")
        result[str(value)] = sha256_file(path)
    return result


def _freeze_path(contract: D0BContract) -> Path:
    return _repository_path(
        contract.repository, contract.raw["freeze"]["pre_run_receipt"], "freeze receipt"
    )


def _formal_phase_directories(contract: D0BContract) -> tuple[Path, ...]:
    return tuple(
        contract.output_root / str(contract.raw["output"][f"{phase}_phase"])
        for phase in ("teacher", "candidate", "outer", "aggregate")
    )


def _expected_freeze_receipt(contract: D0BContract) -> dict[str, Any]:
    preflight = inspect_preflight(contract)
    if preflight["ready"] is not True:
        raise D0BProtocolError(
            "D0-B preflight is not ready; all three D0-A SAFE_EXPORT artifacts are required"
        )
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0b_pre_run_freeze_v1",
        "protocol_id": PROTOCOL_ID,
        "method_name": "CR-SITTA",
        "method_stage": "D0-B",
        "config_path": str(contract.config_path.relative_to(contract.repository)),
        "config_sha256": contract.config_sha256,
        "critical_code_sha256": _capture_code_hashes(contract),
        "lineage": deepcopy(contract.raw["lineage"]),
        "frozen_formula_bindings": deepcopy(contract.raw["frozen_formula_bindings"]),
        "datasets": preflight["datasets"],
        "checkpoint_dependent_artifacts_rebuilt": list(
            contract.raw["artifact_policy"]["checkpoint_dependent_artifacts_must_be_rebuilt"]
        ),
        "old_checkpoint_dependent_numeric_artifact_inputs": [],
        "permitted_reused_artifact": "image_only_13_condition_Pilot64_cache",
        "formal_phase_directories_absent_at_publication": True,
        "no_validation_split": True,
        "development_only": True,
        "paper_result": False,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
        "formal_test_allowed": False,
    }


def verify_freeze(contract: D0BContract) -> tuple[dict[str, Any], str]:
    if sha256_file(contract.config_path) != contract.config_sha256:
        raise D0BProtocolError("D0-B config changed during execution")
    path = _freeze_path(contract)
    raw = _load_json(path)
    expected = _expected_freeze_receipt(contract)
    if raw != expected:
        raise D0BProtocolError("D0-B freeze receipt differs from current code/inputs")
    return raw, sha256_file(path)


def run_freeze(contract: D0BContract) -> dict[str, Any]:
    destination = _freeze_path(contract)
    if destination.exists() or destination.is_symlink():
        return verify_freeze(contract)[0]
    for path in _formal_phase_directories(contract):
        if path.exists() or path.is_symlink():
            raise D0BProtocolError(f"freeze requires absent formal phase directory: {path}")
    expected = _expected_freeze_receipt(contract)
    from tta.d0_secure_io import ensure_directory_chain_nofollow, publish_file_noreplace

    parent = destination.parent.relative_to(contract.repository)
    ensure_directory_chain_nofollow(contract.repository, tuple(parent.parts))
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        _write_json(staging, expected)

        def guard() -> None:
            if sha256_file(contract.config_path) != contract.config_sha256:
                raise D0BProtocolError("config changed before freeze publication")
            if expected != _expected_freeze_receipt(contract):
                raise D0BProtocolError("D0-B inputs changed before freeze publication")
            for phase in _formal_phase_directories(contract):
                if phase.exists() or phase.is_symlink():
                    raise D0BProtocolError("formal phase appeared before freeze publication")

        publish_file_noreplace(staging, destination, pre_rename_guard=guard)
    finally:
        if staging.exists() and staging.is_file() and not staging.is_symlink():
            staging.unlink()
    return verify_freeze(contract)[0]


def _phase_destination(
    contract: D0BContract, phase: str, dataset: str | None = None
) -> Path:
    if phase not in {"teacher", "candidate", "outer", "aggregate"}:
        raise D0BProtocolError(f"unsupported phase: {phase}")
    root = contract.output_root / str(contract.raw["output"][f"{phase}_phase"])
    return root / dataset if dataset is not None else root


def _staging_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    return staging


def _remove_owned_staging(path: Path) -> None:
    if path.name.startswith(".") and ".staging-" in path.name and path.exists():
        shutil.rmtree(path)


def _publish_directory(
    contract: D0BContract,
    staging: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    *,
    completion_type: str,
    guard: Any,
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
            "artifact_type": completion_type,
            "protocol_id": PROTOCOL_ID,
            "complete": True,
            "phase": final_manifest["phase"],
            "dataset": final_manifest.get("dataset"),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "payload_tree_sha256": final_manifest["payload_tree_sha256"],
            "atomic_no_replace": True,
            "development_only": True,
            "paper_result": False,
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
        },
    )
    from tta.d0_secure_io import publish_directory_noreplace

    publish_directory_noreplace(staging, destination, pre_rename_guard=guard)


def _verify_custom_completion(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    completion_type: str,
    phase: str,
    dataset: str | None,
) -> None:
    complete = _load_json(root / "COMPLETE.json")
    if (
        complete.get("schema_version") != 1
        or complete.get("artifact_type") != completion_type
        or complete.get("protocol_id") != PROTOCOL_ID
        or complete.get("complete") is not True
        or complete.get("phase") != phase
        or complete.get("dataset") != dataset
        or complete.get("manifest_sha256") != sha256_file(root / "manifest.json")
        or complete.get("payload_tree_sha256") != manifest.get("payload_tree_sha256")
        or complete.get("atomic_no_replace") is not True
        or complete.get("development_only") is not True
        or complete.get("paper_result") is not False
        or complete.get("validation_payload_opens") != 0
        or complete.get("test_payload_opens") != 0
    ):
        raise D0BProtocolError(f"completion receipt differs: {root}")
    files = _mapping(manifest.get("files"), "artifact file ledger")
    actual = _file_ledger(root, excluded=("manifest.json", "COMPLETE.json"))
    if dict(files) != actual:
        raise D0BProtocolError(f"artifact file ledger differs: {root}")
    if manifest.get("payload_tree_sha256") != _sha256_bytes(
        _canonical_json_bytes(files)
    ):
        raise D0BProtocolError(f"artifact payload-tree hash differs: {root}")


def _method_input_dataset(contract: D0BContract, dataset: str, condition: str):
    # Capability-minimal: this module has no outer-target loader.
    from tta.stage_c0_method_input import StageC0MethodInputDataset

    record = contract.raw["datasets"][dataset]
    return StageC0MethodInputDataset(
        _repository_path(contract.repository, record["cache_root"], f"{dataset} cache"),
        condition_key=condition,
        expected_protocol_sha256=contract.raw["frozen_formula_bindings"][
            "cache_protocol"
        ]["sha256"],
        expected_dataset=dataset,
        expected_complete_sha256=record["cache_complete_sha256"],
        expected_method_manifest_sha256=record["cache_method_manifest_sha256"],
        expected_ordered_ids_sha256=record["ordered_pilot64_image_ids_sha256"],
    )


def _checkpoint_binding(contract: D0BContract, dataset: str) -> Mapping[str, Any]:
    freeze, _digest = verify_freeze(contract)
    binding = _mapping(freeze["datasets"][dataset], f"freeze dataset {dataset}")
    return _mapping(binding["checkpoint"], f"freeze checkpoint {dataset}")


def _build_source_runtime(contract: D0BContract, dataset: str, device_name: str):
    import torch
    import test_source
    from tta.model_adapter import IRSTDModelAdapter

    binding = _checkpoint_binding(contract, dataset)
    checkpoint = _repository_path(
        contract.repository, binding["path"], f"{dataset} safe checkpoint"
    )
    if sha256_file(checkpoint) != binding["sha256"]:
        raise D0BProtocolError(f"{dataset} checkpoint changed before teacher build")
    test_source.seed_everything(int(contract.raw["scope"]["seed"]))
    device = test_source.resolve_device(device_name)
    model = test_source.build_nsfpn_model()
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch compatibility only.
        payload = torch.load(checkpoint, map_location="cpu")
    state, wrapper = test_source.extract_state_dict(payload)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise D0BProtocolError("safe checkpoint did not strict-load into NS-FPN")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    return torch, test_source, model, adapter, device, wrapper


def _teacher_manifest_base(
    contract: D0BContract, dataset: str, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    _freeze, freeze_sha = verify_freeze(contract)
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0b_rebuilt_strong_teacher_dataset_v1",
        "protocol_id": PROTOCOL_ID,
        "phase": "teacher",
        "dataset": dataset,
        "formal": True,
        "development_only": True,
        "paper_result": False,
        "config_path": str(contract.config_path.relative_to(contract.repository)),
        "config_sha256": contract.config_sha256,
        "pre_run_freeze_path": str(_freeze_path(contract).relative_to(contract.repository)),
        "pre_run_freeze_sha256": freeze_sha,
        "checkpoint_role": checkpoint["role"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "teacher_rebuilt_from_current_checkpoint": True,
        "old_teacher_artifact_inputs": [],
        "base_view_names": list(BASE_VIEW_NAMES),
        "accepted_view_names": list(BASE_VIEW_NAMES[:4]),
        "teacher_aggregation": "median",
        "method_label_accesses": 0,
        "outer_target_loader_imports": 0,
        "target_payload_deserializations": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
        "critical_code_sha256": _capture_code_hashes(contract),
    }


def run_teacher(
    contract: D0BContract, *, dataset: str, device_name: str
) -> Mapping[str, Any]:
    """Rebuild strong-view probabilities from one D0-A safe checkpoint."""

    if dataset not in DATASETS:
        raise D0BProtocolError(f"unsupported dataset: {dataset}")
    verify_freeze(contract)
    _verify_cache_and_split(contract, dataset)
    checkpoint = _checkpoint_binding(contract, dataset)
    destination = _phase_destination(contract, "teacher", dataset)
    if destination.exists() or destination.is_symlink():
        return verify_teacher_artifact(contract, dataset)
    staging = _staging_directory(destination)
    try:
        torch, test_source, model, adapter, device, wrapper = _build_source_runtime(
            contract, dataset, device_name
        )
        from tta.proposals.source_multiview_teacher import (
            build_aligned_view_probabilities,
        )

        source_state_before = test_source.state_dict_sha256(model.state_dict())
        image_ids: tuple[str, ...] | None = None
        conditions: list[dict[str, Any]] = []
        for corruption, severity in CONDITIONS:
            condition = _condition_key(corruption, severity)
            method = _method_input_dataset(contract, dataset, condition)
            if len(method) != PILOT_COUNT:
                raise D0BProtocolError(f"{dataset}/{condition} is not Pilot64")
            current_ids = tuple(str(value) for value in method.image_ids)
            if image_ids is None:
                image_ids = current_ids
            elif current_ids != image_ids:
                raise D0BProtocolError("Pilot64 order differs between conditions")
            condition_dir = staging / "conditions" / condition
            array_path = condition_dir / "base_view_probabilities.npy"
            condition_dir.mkdir(parents=True, exist_ok=True)
            values = np.lib.format.open_memmap(
                array_path,
                mode="w+",
                dtype="<f4",
                shape=(PILOT_COUNT, len(BASE_VIEW_NAMES), *PROBABILITY_SHAPE),
            )
            per_image: list[dict[str, Any]] = []
            for image_index in range(PILOT_COUNT):
                item = method[image_index]
                image_cpu = item["image"].contiguous()
                if tuple(image_cpu.shape) != IMAGE_SHAPE or image_cpu.dtype != torch.float32:
                    raise D0BProtocolError("teacher input must be float32 [3,256,256]")
                input_reference = image_cpu.clone()
                image = image_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.inference_mode():
                    base, names = build_aligned_view_probabilities(
                        adapter, image, include_context_tile=True
                    )
                if tuple(names) != BASE_VIEW_NAMES or tuple(base.shape) != (
                    len(BASE_VIEW_NAMES),
                    1,
                    1,
                    256,
                    256,
                ):
                    raise D0BProtocolError("teacher view topology differs")
                if not torch.equal(image_cpu, input_reference):
                    raise D0BProtocolError("teacher modified method-facing input")
                if base.requires_grad or base.grad_fn is not None:
                    raise D0BProtocolError("teacher output is not detached")
                if not bool(torch.isfinite(base).all().item()) or bool(
                    ((base < 0) | (base > 1)).any().item()
                ):
                    raise D0BProtocolError("teacher probability is invalid")
                base_np = base[:, 0].cpu().numpy().astype("<f4", copy=False)
                values[image_index] = base_np
                per_image.append(
                    {
                        "condition": condition,
                        "image_index": image_index,
                        "image_id": str(item["image_id"]),
                        "input_tensor_sha256": _raw_array_sha256(image_cpu.numpy()),
                        "base_view_probabilities_sha256": _raw_array_sha256(base_np),
                        "checkpoint_sha256": checkpoint["sha256"],
                        "method_label_accesses": 0,
                        "validation_payload_opens": 0,
                        "test_payload_opens": 0,
                    }
                )
            values.flush()
            del values
            per_image_path = condition_dir / "per_image.jsonl"
            _write_jsonl(per_image_path, per_image)
            conditions.append(
                {
                    "condition": condition,
                    "corruption": corruption,
                    "severity": severity,
                    "image_count": PILOT_COUNT,
                    "arrays": {
                        "base_view_probabilities": {
                            "path": array_path.relative_to(staging).as_posix(),
                            "dtype": "little_endian_float32",
                            "shape": [PILOT_COUNT, 5, 1, 256, 256],
                            "view_names": list(BASE_VIEW_NAMES),
                        }
                    },
                    "per_image_path": per_image_path.relative_to(staging).as_posix(),
                }
            )
        assert image_ids is not None
        if (
            len(image_ids) != PILOT_COUNT
            or len(set(image_ids)) != PILOT_COUNT
            or _ordered_ids_sha256(image_ids)
            != contract.raw["datasets"][dataset]["ordered_pilot64_image_ids_sha256"]
        ):
            raise D0BProtocolError("teacher Pilot64 identity differs")
        source_state_after = test_source.state_dict_sha256(model.state_dict())
        if source_state_after != source_state_before:
            raise D0BProtocolError("teacher inference mutated model state")
        manifest = {
            **_teacher_manifest_base(contract, dataset, checkpoint),
            "checkpoint_wrapper": wrapper,
            "condition_count": len(CONDITIONS),
            "image_count_per_condition": PILOT_COUNT,
            "image_ids": list(image_ids),
            "ordered_pilot64_image_ids_sha256": _ordered_ids_sha256(image_ids),
            "conditions": conditions,
            "source_state_sha256_before": source_state_before,
            "source_state_sha256_after": source_state_after,
            "source_state_bit_exact": True,
            "runtime_environment": {
                "python": sys.version.split()[0],
                "numpy": str(np.__version__),
                "torch": str(torch.__version__),
                "device": str(device),
                "gpu_name": (
                    str(torch.cuda.get_device_name(device))
                    if getattr(device, "type", None) == "cuda"
                    else None
                ),
            },
        }

        def guard() -> None:
            verify_freeze(contract)
            _verify_cache_and_split(contract, dataset)
            current = _checkpoint_binding(contract, dataset)
            if dict(current) != dict(checkpoint):
                raise D0BProtocolError("checkpoint binding changed before teacher publish")

        _publish_directory(
            contract,
            staging,
            destination,
            manifest,
            completion_type="cr_sitta_d0b_teacher_completion_v1",
            guard=guard,
        )
        return verify_teacher_artifact(contract, dataset)
    except BaseException:
        _remove_owned_staging(staging)
        raise


def _verify_teacher_per_image_record(
    record: Mapping[str, Any],
    *,
    condition: str,
    image_index: int,
    image_id: str,
    input_tensor_sha256: str,
    probability_sha256: str,
    checkpoint_sha256: str,
) -> None:
    """Verify one label-free teacher provenance row exactly."""

    if (
        record.get("condition") != condition
        or record.get("image_index") != image_index
        or record.get("image_id") != image_id
        or record.get("input_tensor_sha256") != input_tensor_sha256
        or record.get("base_view_probabilities_sha256") != probability_sha256
        or record.get("checkpoint_sha256") != checkpoint_sha256
        or record.get("method_label_accesses") != 0
        or record.get("validation_payload_opens") != 0
        or record.get("test_payload_opens") != 0
    ):
        raise D0BProtocolError(
            f"teacher per-image provenance differs: {condition}/{image_index}"
        )


def verify_teacher_artifact(contract: D0BContract, dataset: str) -> dict[str, Any]:
    verify_freeze(contract)
    root = _phase_destination(contract, "teacher", dataset)
    if root.is_symlink() or not root.is_dir():
        raise D0BProtocolError(f"teacher artifact missing/unsafe: {root}")
    manifest = _load_json(root / "manifest.json")
    checkpoint = _checkpoint_binding(contract, dataset)
    _freeze, freeze_sha = verify_freeze(contract)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type")
        != "cr_sitta_d0b_rebuilt_strong_teacher_dataset_v1"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("phase") != "teacher"
        or manifest.get("dataset") != dataset
        or manifest.get("formal") is not True
        or manifest.get("development_only") is not True
        or manifest.get("paper_result") is not False
        or manifest.get("config_sha256") != contract.config_sha256
        or manifest.get("pre_run_freeze_sha256") != freeze_sha
        or manifest.get("checkpoint_path") != checkpoint["path"]
        or manifest.get("checkpoint_sha256") != checkpoint["sha256"]
        or manifest.get("checkpoint_role") != checkpoint["role"]
        or manifest.get("teacher_rebuilt_from_current_checkpoint") is not True
        or manifest.get("old_teacher_artifact_inputs") != []
        or tuple(manifest.get("base_view_names", ())) != BASE_VIEW_NAMES
        or tuple(manifest.get("accepted_view_names", ())) != BASE_VIEW_NAMES[:4]
        or manifest.get("teacher_aggregation") != "median"
        or manifest.get("condition_count") != len(CONDITIONS)
        or manifest.get("image_count_per_condition") != PILOT_COUNT
        or manifest.get("source_state_bit_exact") is not True
        or manifest.get("source_state_sha256_before")
        != manifest.get("source_state_sha256_after")
        or manifest.get("critical_code_sha256") != _capture_code_hashes(contract)
        or any(
            manifest.get(field) != 0
            for field in (
                "method_label_accesses",
                "outer_target_loader_imports",
                "target_payload_deserializations",
                "validation_payload_opens",
                "test_payload_opens",
            )
        )
    ):
        raise D0BProtocolError("teacher manifest semantics differ")
    ids = tuple(str(value) for value in manifest.get("image_ids", ()))
    if (
        len(ids) != PILOT_COUNT
        or len(set(ids)) != PILOT_COUNT
        or _ordered_ids_sha256(ids)
        != contract.raw["datasets"][dataset]["ordered_pilot64_image_ids_sha256"]
        or manifest.get("ordered_pilot64_image_ids_sha256") != _ordered_ids_sha256(ids)
    ):
        raise D0BProtocolError("teacher Pilot64 identity/count differs")
    conditions = _sequence(manifest.get("conditions"), "teacher conditions")
    if len(conditions) != len(CONDITIONS):
        raise D0BProtocolError("teacher condition count differs")
    ledger = _mapping(manifest.get("files"), "teacher file ledger")
    for expected, raw_condition in zip(CONDITIONS, conditions, strict=True):
        record = _mapping(raw_condition, "teacher condition")
        condition = _condition_key(*expected)
        descriptor = _mapping(
            _mapping(record.get("arrays"), "teacher arrays").get(
                "base_view_probabilities"
            ),
            "teacher base-view descriptor",
        )
        relative = f"conditions/{condition}/base_view_probabilities.npy"
        per_image_relative = f"conditions/{condition}/per_image.jsonl"
        if (
            record.get("condition") != condition
            or record.get("corruption") != expected[0]
            or record.get("severity") != expected[1]
            or record.get("image_count") != PILOT_COUNT
            or descriptor.get("path") != relative
            or descriptor.get("dtype") != "little_endian_float32"
            or tuple(descriptor.get("shape", ())) != (64, 5, 1, 256, 256)
            or tuple(descriptor.get("view_names", ())) != BASE_VIEW_NAMES
            or record.get("per_image_path") != per_image_relative
        ):
            raise D0BProtocolError(f"teacher condition descriptor differs: {condition}")
        file_record = _mapping(ledger.get(relative), "teacher array file record")
        _verify_exact_file(root / relative, file_record.get("sha256"), "teacher array")
        values = np.load(root / relative, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(values, np.memmap)
            or values.flags.writeable
            or values.dtype.str != "<f4"
            or tuple(values.shape) != (64, 5, 1, 256, 256)
            or not bool(np.isfinite(values).all())
            or float(values.min()) < 0.0
            or float(values.max()) > 1.0
        ):
            raise D0BProtocolError(f"teacher probability array differs: {condition}")
        per_image_file = _mapping(
            ledger.get(per_image_relative), "teacher per-image file record"
        )
        _verify_exact_file(
            root / per_image_relative,
            per_image_file.get("sha256"),
            "teacher per-image provenance",
        )
        per_image_records = _read_jsonl(root / per_image_relative)
        if len(per_image_records) != PILOT_COUNT:
            raise D0BProtocolError(
                f"teacher per-image count differs: {condition}"
            )
        method = _method_input_dataset(contract, dataset, condition)
        if tuple(str(value) for value in method.image_ids) != ids:
            raise D0BProtocolError(
                f"teacher method-input order differs: {condition}"
            )
        for image_index, per_image_record in enumerate(per_image_records):
            item = method[image_index]
            image = item["image"].contiguous()
            _verify_teacher_per_image_record(
                per_image_record,
                condition=condition,
                image_index=image_index,
                image_id=str(item["image_id"]),
                input_tensor_sha256=_raw_array_sha256(image.numpy()),
                probability_sha256=_raw_array_sha256(values[image_index]),
                checkpoint_sha256=str(checkpoint["sha256"]),
            )
    _verify_custom_completion(
        root,
        manifest,
        completion_type="cr_sitta_d0b_teacher_completion_v1",
        phase="teacher",
        dataset=dataset,
    )
    return manifest


def _legacy_contract(contract: D0BContract, legacy: Any):
    """Build the old runner's structural contract with D0-B frozen bindings."""

    freeze, _freeze_sha = verify_freeze(contract)
    raw = deepcopy(dict(contract.raw))
    raw["freeze"] = {
        **dict(raw["freeze"]),
        "pre_run_freeze_receipt": raw["freeze"]["pre_run_receipt"],
    }
    raw["frozen_parent_bindings"] = {
        "cache_protocol": deepcopy(raw["frozen_formula_bindings"]["cache_protocol"])
    }
    for dataset in DATASETS:
        bound = _mapping(freeze["datasets"][dataset], f"freeze dataset {dataset}")
        checkpoint = _mapping(bound["checkpoint"], f"freeze checkpoint {dataset}")
        raw["datasets"][dataset]["checkpoint_sha256"] = checkpoint["sha256"]
    return legacy.StageC0Contract(
        contract.repository,
        contract.config_path,
        contract.config_sha256,
        raw,
    )


def _teacher_binding(contract: D0BContract, dataset: str) -> dict[str, Any]:
    manifest = verify_teacher_artifact(contract, dataset)
    root = _phase_destination(contract, "teacher", dataset)
    return {
        "path": str(root.relative_to(contract.repository)),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "payload_tree_sha256": manifest["payload_tree_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "rebuilt_in_d0b": True,
    }


def _verify_dataset_binding_for_execution(
    contract: D0BContract, dataset: str
) -> None:
    _verify_cache_and_split(contract, dataset)
    freeze, _freeze_sha = verify_freeze(contract)
    bound = _mapping(freeze["datasets"][dataset], f"freeze dataset {dataset}")
    checkpoint = _mapping(bound["checkpoint"], f"freeze checkpoint {dataset}")
    checkpoint_path = _repository_path(
        contract.repository, checkpoint["path"], f"{dataset} checkpoint"
    )
    if sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise D0BProtocolError(f"{dataset} checkpoint changed after freeze")
    safe_receipt = _mapping(
        bound["safe_export_receipt"], f"freeze safe-export receipt {dataset}"
    )
    receipt_path = _repository_path(
        contract.repository, safe_receipt["path"], f"{dataset} SAFE_EXPORT"
    )
    if sha256_file(receipt_path) != safe_receipt["sha256"]:
        raise D0BProtocolError(f"{dataset} SAFE_EXPORT changed after freeze")
    verify_teacher_artifact(contract, dataset)


def _legacy_runtime_manifest_base(
    contract: D0BContract,
    *,
    artifact_type: str,
    phase: str,
    dataset: str | None,
    formal: bool,
) -> dict[str, Any]:
    _freeze, freeze_sha = verify_freeze(contract)
    result = {
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
        "implementation_revision": 2,
        "predecessor_protocol_id": LEGACY_C0_PROTOCOL_ID,
        "probe_seed_namespace": PROBE_SEED_NAMESPACE,
        "pre_run_freeze_receipt_path": str(
            _freeze_path(contract).relative_to(contract.repository)
        ),
        "pre_run_freeze_receipt_sha256": freeze_sha,
        "method_stage": "D0-B",
        "old_checkpoint_dependent_numeric_artifact_inputs": [],
        "method_label_accesses": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
    }
    if dataset is not None:
        checkpoint = _checkpoint_binding(contract, dataset)
        result["checkpoint_binding"] = dict(checkpoint)
        result["d0b_teacher_binding"] = _teacher_binding(contract, dataset)
    return result


@contextmanager
def _bound_legacy_c0_logic(contract: D0BContract):
    """Bind hash-frozen C0 math to D0-B inputs for one single-threaded call."""

    legacy = importlib.import_module("run_p3_stage_c0_signal_audit_v2")
    expected_path = _repository_path(
        contract.repository,
        contract.raw["lineage"]["prior_c0_logic"]["path"],
        "prior C0 logic",
    )
    if Path(legacy.__file__).resolve() != expected_path.resolve():
        raise D0BProtocolError("imported prior C0 logic path differs")
    _verify_exact_file(
        expected_path,
        contract.raw["lineage"]["prior_c0_logic"]["sha256"],
        "prior C0 logic",
    )
    legacy_contract = _legacy_contract(contract, legacy)
    names = (
        "PROTOCOL_ID",
        "PREDECESSOR_PROTOCOL_ID",
        "PROBE_SEED_NAMESPACE",
        "_verify_pre_run_freeze",
        "_assert_base_contract_unchanged",
        "_assert_contract_unchanged",
        "_capture_code_hashes",
        "_verify_dataset_bindings",
        "_runtime_manifest_base",
    )
    saved = {name: getattr(legacy, name) for name in names}
    try:
        legacy.PROTOCOL_ID = PROTOCOL_ID
        legacy.PREDECESSOR_PROTOCOL_ID = LEGACY_C0_PROTOCOL_ID
        legacy.PROBE_SEED_NAMESPACE = PROBE_SEED_NAMESPACE
        legacy._verify_pre_run_freeze = lambda _unused: verify_freeze(contract)
        legacy._assert_base_contract_unchanged = lambda _unused: verify_freeze(contract)
        legacy._assert_contract_unchanged = lambda _unused: verify_freeze(contract)
        legacy._capture_code_hashes = lambda _unused: _capture_code_hashes(contract)
        legacy._verify_dataset_bindings = (
            lambda _repository, _raw, dataset: _verify_dataset_binding_for_execution(
                contract, dataset
            )
        )
        legacy._runtime_manifest_base = (
            lambda _unused, *, artifact_type, phase, dataset, formal: (
                _legacy_runtime_manifest_base(
                    contract,
                    artifact_type=artifact_type,
                    phase=phase,
                    dataset=dataset,
                    formal=formal,
                )
            )
        )
        yield legacy, legacy_contract
    finally:
        for name, value in saved.items():
            setattr(legacy, name, value)


def _verify_d0b_projection(
    contract: D0BContract,
    dataset: str,
    manifest: Mapping[str, Any],
) -> None:
    expected_checkpoint = dict(_checkpoint_binding(contract, dataset))
    expected_teacher = _teacher_binding(contract, dataset)
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("method_stage") != "D0-B"
        or manifest.get("checkpoint_binding") != expected_checkpoint
        or manifest.get("d0b_teacher_binding") != expected_teacher
        or manifest.get("old_checkpoint_dependent_numeric_artifact_inputs") != []
        or manifest.get("validation_payload_opens") != 0
        or manifest.get("test_payload_opens") != 0
    ):
        raise D0BProtocolError("legacy C0 artifact lacks the D0-B checkpoint projection")


def run_candidate(
    contract: D0BContract, *, dataset: str, device_name: str
) -> Mapping[str, Any]:
    verify_teacher_artifact(contract, dataset)
    with _bound_legacy_c0_logic(contract) as (legacy, legacy_contract):
        try:
            manifest = legacy._run_candidate(
                legacy_contract,
                dataset=dataset,
                device_name=device_name,
                formal=True,
                smoke_id=None,
                smoke_images=PILOT_COUNT,
            )
        except legacy.StageC0ProtocolError as exc:
            raise D0BProtocolError(str(exc)) from exc
        _verify_d0b_projection(contract, dataset, manifest)
        return manifest


def run_outer(
    contract: D0BContract, *, dataset: str, device_name: str
) -> Mapping[str, Any]:
    verify_teacher_artifact(contract, dataset)
    with _bound_legacy_c0_logic(contract) as (legacy, legacy_contract):
        candidate_path = _phase_destination(contract, "candidate", dataset)
        try:
            candidate_manifest, token = legacy.verify_candidate_artifact(
                candidate_path,
                contract=legacy_contract,
                dataset=dataset,
                expected_formal=True,
            )
            _verify_d0b_projection(contract, dataset, candidate_manifest)
            manifest = legacy._run_outer(
                legacy_contract,
                dataset=dataset,
                device_name=device_name,
                formal=True,
                smoke_id=None,
            )
            replay = legacy.verify_outer_artifact(
                _phase_destination(contract, "outer", dataset),
                contract=legacy_contract,
                dataset=dataset,
                expected_formal=True,
                candidate_token=token,
            )
        except legacy.StageC0ProtocolError as exc:
            raise D0BProtocolError(str(exc)) from exc
        if manifest != replay:
            raise D0BProtocolError("outer run/verification projections differ")
        _verify_d0b_projection(contract, dataset, manifest)
        return manifest


def _aggregate_bindings(contract: D0BContract) -> tuple[
    dict[str, Sequence[Mapping[str, Any]]],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
]:
    outer_records: dict[str, Sequence[Mapping[str, Any]]] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, Any] = {}
    with _bound_legacy_c0_logic(contract) as (legacy, legacy_contract):
        for dataset in DATASETS:
            teacher = _teacher_binding(contract, dataset)
            candidate_path = _phase_destination(contract, "candidate", dataset)
            outer_path = _phase_destination(contract, "outer", dataset)
            try:
                candidate, token = legacy.verify_candidate_artifact(
                    candidate_path,
                    contract=legacy_contract,
                    dataset=dataset,
                    expected_formal=True,
                )
                outer = legacy.verify_outer_artifact(
                    outer_path,
                    contract=legacy_contract,
                    dataset=dataset,
                    expected_formal=True,
                    candidate_token=token,
                )
            except legacy.StageC0ProtocolError as exc:
                raise D0BProtocolError(str(exc)) from exc
            _verify_d0b_projection(contract, dataset, candidate)
            _verify_d0b_projection(contract, dataset, outer)
            outer_records[dataset] = _read_jsonl(outer_path / "outer_episodes.jsonl")
            identities[dataset] = _mapping(candidate["identity_adapter"], "identity")
            bindings[dataset] = {
                "checkpoint": dict(_checkpoint_binding(contract, dataset)),
                "teacher": teacher,
                "candidate": {
                    "path": str(candidate_path.relative_to(contract.repository)),
                    "manifest_sha256": sha256_file(candidate_path / "manifest.json"),
                    "payload_tree_sha256": candidate["payload_tree_sha256"],
                    "episode_count": candidate["episode_count"],
                },
                "outer": {
                    "path": str(outer_path.relative_to(contract.repository)),
                    "manifest_sha256": sha256_file(outer_path / "manifest.json"),
                    "payload_tree_sha256": outer["payload_tree_sha256"],
                    "episode_count": outer["episode_count"],
                },
            }
    return outer_records, identities, bindings


def _d0b_gate_outputs(
    contract: D0BContract,
    outer_records: Mapping[str, Sequence[Mapping[str, Any]]],
    identities: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from analysis.cr_sitta_d0b_gate import authorize_d1, build_d0b_science_receipt
    from analysis.stage_c_science_gate_v1 import (
        StageCGateConfig,
        evaluate_stage_c0_science_gate,
    )

    gate = StageCGateConfig.from_mapping(contract.raw["stage_c0_signal_gate"])
    # This is the exact hash-bound predecessor formula, now fed exclusively by
    # freshly rebuilt D0-B outer records.
    legacy = importlib.import_module("run_p3_stage_c0_signal_audit_v2")
    evidence = legacy.build_aggregate_evidence(
        outer_records, identities, gate_config=gate
    )
    receipt = evaluate_stage_c0_science_gate(evidence, gate)
    science = build_d0b_science_receipt(receipt)
    authorization = authorize_d1(receipt).to_receipt()
    forbidden = set(contract.raw["artifact_policy"]["forbidden_authorization_fields"])
    if forbidden.intersection(science) or forbidden.intersection(authorization):
        raise D0BProtocolError("D0-B output leaked a legacy authorization field")
    if science["formal_test_allowed"] is not False or authorization["formal_test_allowed"] is not False:
        raise D0BProtocolError("D0-B attempted to authorize formal test")
    return evidence, science, authorization


def _verify_aggregate_provenance(
    manifest: Mapping[str, Any], required_rebuilt: Sequence[str]
) -> None:
    """Enforce the non-paper, full-rebuild boundary on aggregate artifacts."""

    if (
        manifest.get("development_only") is not True
        or manifest.get("paper_result") is not False
        or manifest.get("checkpoint_dependent_artifacts_rebuilt")
        != list(required_rebuilt)
        or manifest.get("old_checkpoint_dependent_numeric_artifact_inputs") != []
    ):
        raise D0BProtocolError("D0-B aggregate provenance/rebuild roster differs")


def run_aggregate(contract: D0BContract) -> Mapping[str, Any]:
    verify_freeze(contract)
    destination = _phase_destination(contract, "aggregate")
    if destination.exists() or destination.is_symlink():
        return verify_aggregate_artifact(contract)
    outer_records, identities, bindings = _aggregate_bindings(contract)
    evidence, science, authorization = _d0b_gate_outputs(
        contract, outer_records, identities
    )
    staging = _staging_directory(destination)
    try:
        _write_json(staging / "aggregate_evidence.json", evidence)
        _write_json(staging / "D0B_SCIENCE_DECISION.json", science)
        _write_json(staging / "D1_AUTHORIZATION.json", authorization)
        _freeze, freeze_sha = verify_freeze(contract)
        manifest = {
            "schema_version": 1,
            "artifact_type": "cr_sitta_d0b_checkpoint_rebound_aggregate_v1",
            "protocol_id": PROTOCOL_ID,
            "phase": "aggregate",
            "dataset": None,
            "formal": True,
            "method_name": "CR-SITTA",
            "method_stage": "D0-B",
            "development_only": True,
            "paper_result": False,
            "config_path": str(contract.config_path.relative_to(contract.repository)),
            "config_sha256": contract.config_sha256,
            "pre_run_freeze_path": str(_freeze_path(contract).relative_to(contract.repository)),
            "pre_run_freeze_sha256": freeze_sha,
            "dataset_count": 3,
            "condition_count_per_dataset": 13,
            "pilot_image_count_per_dataset": 64,
            "probe_count": 2,
            "unique_nonclean_probe_episode_count": 4608,
            "parameter_spaces": list(PARAMETER_SPACES),
            "dataset_artifacts": bindings,
            "checkpoint_dependent_artifacts_rebuilt": list(
                contract.raw["artifact_policy"]["checkpoint_dependent_artifacts_must_be_rebuilt"]
            ),
            "old_checkpoint_dependent_numeric_artifact_inputs": [],
            "protocol_status": science["protocol_status"],
            "scientific_status": science["scientific_status"],
            "eligible_parameter_space_ids": science["eligible_parameter_space_ids"],
            "d1_train_internal_oof_allowed": authorization[
                "d1_train_internal_oof_allowed"
            ],
            "formal_test_allowed": False,
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
            "critical_code_sha256": _capture_code_hashes(contract),
        }

        def guard() -> None:
            verify_freeze(contract)
            replay_records, replay_identities, replay_bindings = _aggregate_bindings(
                contract
            )
            if replay_bindings != bindings:
                raise D0BProtocolError("D0-B inputs changed before aggregate publish")
            replay = _d0b_gate_outputs(contract, replay_records, replay_identities)
            if replay != (evidence, science, authorization):
                raise D0BProtocolError("D0-B gate result changed before publication")

        _publish_directory(
            contract,
            staging,
            destination,
            manifest,
            completion_type="cr_sitta_d0b_aggregate_completion_v1",
            guard=guard,
        )
        return verify_aggregate_artifact(contract)
    except BaseException:
        _remove_owned_staging(staging)
        raise


def verify_aggregate_artifact(contract: D0BContract) -> dict[str, Any]:
    verify_freeze(contract)
    root = _phase_destination(contract, "aggregate")
    if root.is_symlink() or not root.is_dir():
        raise D0BProtocolError(f"aggregate artifact missing/unsafe: {root}")
    manifest = _load_json(root / "manifest.json")
    outer_records, identities, bindings = _aggregate_bindings(contract)
    evidence, science, authorization = _d0b_gate_outputs(
        contract, outer_records, identities
    )
    required_rebuilt = tuple(
        contract.raw["artifact_policy"][
            "checkpoint_dependent_artifacts_must_be_rebuilt"
        ]
    )
    _verify_aggregate_provenance(manifest, required_rebuilt)
    _freeze, freeze_sha = verify_freeze(contract)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type")
        != "cr_sitta_d0b_checkpoint_rebound_aggregate_v1"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("phase") != "aggregate"
        or manifest.get("dataset") is not None
        or manifest.get("formal") is not True
        or manifest.get("method_name") != "CR-SITTA"
        or manifest.get("method_stage") != "D0-B"
        or manifest.get("config_sha256") != contract.config_sha256
        or manifest.get("pre_run_freeze_sha256") != freeze_sha
        or manifest.get("dataset_count") != 3
        or manifest.get("condition_count_per_dataset") != 13
        or manifest.get("pilot_image_count_per_dataset") != 64
        or manifest.get("probe_count") != 2
        or manifest.get("unique_nonclean_probe_episode_count") != 4608
        or tuple(manifest.get("parameter_spaces", ())) != PARAMETER_SPACES
        or manifest.get("dataset_artifacts") != bindings
        or manifest.get("old_checkpoint_dependent_numeric_artifact_inputs") != []
        or manifest.get("protocol_status") != science["protocol_status"]
        or manifest.get("scientific_status") != science["scientific_status"]
        or manifest.get("eligible_parameter_space_ids")
        != science["eligible_parameter_space_ids"]
        or manifest.get("d1_train_internal_oof_allowed")
        is not authorization["d1_train_internal_oof_allowed"]
        or manifest.get("formal_test_allowed") is not False
        or manifest.get("validation_payload_opens") != 0
        or manifest.get("test_payload_opens") != 0
        or manifest.get("critical_code_sha256") != _capture_code_hashes(contract)
    ):
        raise D0BProtocolError("D0-B aggregate manifest semantics differ")
    if _load_json(root / "aggregate_evidence.json") != evidence:
        raise D0BProtocolError("D0-B aggregate evidence differs")
    if _load_json(root / "D0B_SCIENCE_DECISION.json") != science:
        raise D0BProtocolError("D0-B science receipt differs")
    if _load_json(root / "D1_AUTHORIZATION.json") != authorization:
        raise D0BProtocolError("D0-B D1 authorization differs")
    forbidden = set(contract.raw["artifact_policy"]["forbidden_authorization_fields"])
    for value in (manifest, science, authorization):
        if forbidden.intersection(value):
            raise D0BProtocolError("aggregate contains a forbidden legacy authorization field")
    _verify_custom_completion(
        root,
        manifest,
        completion_type="cr_sitta_d0b_aggregate_completion_v1",
        phase="aggregate",
        dataset=None,
    )
    return manifest


def _status(contract: D0BContract) -> dict[str, Any]:
    preflight = inspect_preflight(contract)
    phases: dict[str, Any] = {}
    for phase in ("teacher", "candidate", "outer"):
        phases[phase] = {
            dataset: _phase_destination(contract, phase, dataset).is_dir()
            for dataset in DATASETS
        }
    phases["aggregate"] = _phase_destination(contract, "aggregate").is_dir()
    return {
        "protocol_id": PROTOCOL_ID,
        "preflight_ready": preflight["ready"],
        "preflight_errors": preflight["errors"],
        "freeze_exists": _freeze_path(contract).is_file(),
        "phases_exist_unverified": phases,
        "formal_test_allowed": False,
    }


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("print-config-sha256")
    commands.add_parser("preflight")
    commands.add_parser("freeze")
    commands.add_parser("status")
    for name in ("teacher", "candidate", "outer"):
        phase = commands.add_parser(name)
        phase.add_argument("--dataset", required=True, choices=DATASETS)
        phase.add_argument("--device", default="cuda:0")
    commands.add_parser("aggregate")
    commands.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract = load_contract(args.config)
        if args.command == "print-config-sha256":
            print(contract.config_sha256)
            return 0
        if args.command == "preflight":
            result = inspect_preflight(contract)
            _print_json(result)
            return 0 if result["ready"] else 2
        if args.command == "status":
            _print_json(_status(contract))
            return 0
        if args.command == "freeze":
            _print_json(run_freeze(contract))
            return 0
        if args.command == "teacher":
            _print_json(run_teacher(contract, dataset=args.dataset, device_name=args.device))
            return 0
        if args.command == "candidate":
            _print_json(run_candidate(contract, dataset=args.dataset, device_name=args.device))
            return 0
        if args.command == "outer":
            _print_json(run_outer(contract, dataset=args.dataset, device_name=args.device))
            return 0
        if args.command == "aggregate":
            _print_json(run_aggregate(contract))
            return 0
        if args.command == "verify":
            verify_freeze(contract)
            result: dict[str, Any] = {"freeze": "verified", "datasets": {}}
            with _bound_legacy_c0_logic(contract) as (legacy, legacy_contract):
                for dataset in DATASETS:
                    teacher = verify_teacher_artifact(contract, dataset)
                    candidate, token = legacy.verify_candidate_artifact(
                        _phase_destination(contract, "candidate", dataset),
                        contract=legacy_contract,
                        dataset=dataset,
                        expected_formal=True,
                    )
                    outer = legacy.verify_outer_artifact(
                        _phase_destination(contract, "outer", dataset),
                        contract=legacy_contract,
                        dataset=dataset,
                        expected_formal=True,
                        candidate_token=token,
                    )
                    _verify_d0b_projection(contract, dataset, candidate)
                    _verify_d0b_projection(contract, dataset, outer)
                    result["datasets"][dataset] = {
                        "teacher_manifest_sha256": sha256_file(
                            _phase_destination(contract, "teacher", dataset) / "manifest.json"
                        ),
                        "teacher_payload_tree_sha256": teacher["payload_tree_sha256"],
                        "candidate_manifest_sha256": token.manifest_sha256,
                        "outer_manifest_sha256": sha256_file(
                            _phase_destination(contract, "outer", dataset) / "manifest.json"
                        ),
                    }
            result["aggregate"] = verify_aggregate_artifact(contract)
            result["formal_test_allowed"] = False
            _print_json(result)
            return 0
        raise AssertionError(f"unhandled command: {args.command}")
    except (D0BProtocolError, OSError, ValueError, RuntimeError) as exc:
        print(f"D0-B protocol error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
