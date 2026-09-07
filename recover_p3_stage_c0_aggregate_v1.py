#!/usr/bin/env python3
"""Immutable aggregate-only recovery for the Stage-C0 v2 JSON container mismatch.

The completed v2 candidate, outer, and aggregate artifacts are immutable
inputs.  This recovery first requires the unmodified v2 public verifier to
reach its one known terminal error, then applies an in-memory JSON container
projection only to ``StageCAuthorization`` and requires the *full* verifier to
return successfully.  Evidence, science receipt, and authorization are also
rebuilt independently from the completed evidence artifacts.

No candidate/outer computation, model, optimizer, raw image, raw target,
validation, or test payload is reachable from this runner.  It can publish
only a separately versioned, read-only, no-replace recovery receipt.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Final
import uuid

import yaml

import run_p3_stage_c0_signal_audit_v2 as _v2
from analysis.stage_c_science_gate_v1 import (
    StageCAuthorization,
    authorize_stage_c_followup,
    evaluate_stage_c0_science_gate,
)
from tta.d0_secure_io import (
    ensure_directory_chain_nofollow,
    publish_file_noreplace,
    read_stable_regular_file,
)


REPOSITORY: Final = Path(__file__).resolve().parent
DEFAULT_CONFIG: Final = REPOSITORY / "configs/p3_stage_c0_aggregate_recovery_v1.yaml"
PROTOCOL_ID: Final = "cr-sitta-p3-stage-c0-aggregate-recovery-v1"
PARENT_PROTOCOL_ID: Final = "cr-sitta-p3-stage-c0-signal-audit-v2"
FROZEN_CONFIG_SHA256: Final = (
    "8ce80645997ec1ad76caded1e0cadde019cd2ecf7ef67b8f114028e530b91db8"
)
DATASETS: Final = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
KNOWN_TERMINAL_ERROR: Final = "aggregate authorization differs"
TARGET_LOADER_MODULE: Final = "materialize_binary_tent_ss_calibration_cache_v2"
ABORT_RECEIPT_SHA256: Final = (
    "81266c924c8cbbbde00b8864f38094d941eff6aeee6b061dc8997457206b978e"
)
SHA256_HEX: Final = frozenset("0123456789abcdef")


class StageC0AggregateRecoveryError(RuntimeError):
    """The frozen recovery contract, input chain, or output failed closed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise StageC0AggregateRecoveryError(
                "unhashable YAML mapping key is forbidden"
            ) from exc
        if duplicate:
            raise StageC0AggregateRecoveryError(
                f"duplicate YAML key is forbidden: {key!r}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True, slots=True)
class RecoveryContract:
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
    def freeze_receipt_path(self) -> Path:
        return _repository_path(
            self.repository,
            self.raw["freeze"]["pre_run_freeze_receipt"],
            "recovery freeze receipt",
        )

    @property
    def verified_receipt_path(self) -> Path:
        return self.output_root / str(
            self.raw["output"]["recovery_verified_receipt"]
        )


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StageC0AggregateRecoveryError(
            "value is not finite canonical-JSON data"
        ) from exc
    return payload + (b"\n" if newline else b"")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_native(value: Any) -> Any:
    """Recursively project mappings and tuple/list containers to JSON natives."""

    if isinstance(value, Mapping):
        return {str(key): _json_native(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(child) for child in value]
    return value


def _container_differences(
    producer: Any, stored: Any, *, path: str = "$"
) -> list[dict[str, str]]:
    differences: list[dict[str, str]] = []
    if isinstance(producer, Mapping) and isinstance(stored, Mapping):
        if set(producer) != set(stored):
            raise StageC0AggregateRecoveryError(
                f"canonical adapter mapping keys differ at {path}"
            )
        for key in sorted(producer):
            differences.extend(
                _container_differences(
                    producer[key], stored[key], path=f"{path}.{key}"
                )
            )
        return differences
    if isinstance(producer, (tuple, list)) and isinstance(stored, (tuple, list)):
        if len(producer) != len(stored):
            raise StageC0AggregateRecoveryError(
                f"canonical adapter sequence lengths differ at {path}"
            )
        if type(producer) is not type(stored):
            differences.append(
                {
                    "json_path": path,
                    "producer_python_type": type(producer).__name__,
                    "stored_python_type": type(stored).__name__,
                }
            )
        for index, (left, right) in enumerate(zip(producer, stored)):
            differences.extend(
                _container_differences(left, right, path=f"{path}[{index}]")
            )
        return differences
    if type(producer) is not type(stored) or producer != stored:
        raise StageC0AggregateRecoveryError(
            f"canonical adapter value/type differs at {path}"
        )
    return differences


def canonicalize_authorization_with_proof(
    producer: Mapping[str, Any], stored: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Allow exactly the StageCAuthorization tuple-to-JSON-list projection."""

    differences = _container_differences(producer, stored)
    expected = [
        {
            "json_path": "$.parameter_space_ids",
            "producer_python_type": "tuple",
            "stored_python_type": "list",
        }
    ]
    if differences != expected:
        raise StageC0AggregateRecoveryError(
            "authorization difference is not exactly parameter_space_ids tuple->list"
        )
    adapted = _json_native(producer)
    if not isinstance(adapted, dict) or adapted != stored:
        raise StageC0AggregateRecoveryError(
            "authorization JSON-native projection differs from stored authorization"
        )
    before = _canonical_sha256(producer)
    after = _canonical_sha256(adapted)
    stored_sha = _canonical_sha256(stored)
    if before != after or after != stored_sha:
        raise StageC0AggregateRecoveryError(
            "authorization canonical semantics changed during projection"
        )
    return adapted, {
        "adapter": "recursive_mapping_to_dict_and_tuple_or_list_to_list",
        "scope": "StageCAuthorization_serialization_projection_only",
        "only_container_difference": expected[0],
        "canonical_hash_algorithm": (
            "sha256_of_canonical_json_without_trailing_newline"
        ),
        "producer_semantic_sha256": before,
        "adapted_semantic_sha256": after,
        "stored_semantic_sha256": stored_sha,
        "canonical_semantics_equal": True,
        "thresholds_changed": False,
        "coverage_changed": False,
        "data_changed": False,
        "statistics_changed": False,
    }


@contextmanager
def _v2_authorization_json_projection() -> Any:
    """Temporarily JSON-project only v2's StageCAuthorization ``asdict`` result."""

    original = _v2.asdict
    if original is not asdict:
        raise StageC0AggregateRecoveryError(
            "v2 asdict is already modified before the scoped adapter"
        )

    def adapted_asdict(instance: Any) -> Any:
        result = original(instance)
        if isinstance(instance, StageCAuthorization):
            return _json_native(result)
        return result

    _v2.asdict = adapted_asdict
    try:
        yield
    finally:
        _v2.asdict = original
        if _v2.asdict is not original:  # pragma: no cover - defensive.
            raise StageC0AggregateRecoveryError(
                "v2 in-memory authorization adapter was not restored"
            )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageC0AggregateRecoveryError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StageC0AggregateRecoveryError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise StageC0AggregateRecoveryError(
            f"{label} fields differ; missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise StageC0AggregateRecoveryError(f"{label} must be lowercase SHA-256")
    return value


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise StageC0AggregateRecoveryError(f"{label} must be a string")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise StageC0AggregateRecoveryError(
            f"{label} must be canonical project-relative"
        )
    return path.as_posix()


def _repository_path(repository: Path, value: Any, label: str) -> Path:
    relative = _relative(value, label)
    path = Path(os.path.abspath(os.fspath(repository / relative)))
    try:
        path.relative_to(repository)
    except ValueError as exc:  # pragma: no cover - also guarded above.
        raise StageC0AggregateRecoveryError(f"{label} escapes repository") from exc
    return path


def _parse_contract(
    raw_value: Any, *, repository: Path, config_path: Path, config_sha256: str
) -> RecoveryContract:
    raw = _mapping(raw_value, "recovery config")
    _exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "role",
            "freeze",
            "parent_stage_c0_v2",
            "incident",
            "repair",
            "scientific_boundary",
            "execution",
            "output",
            "implementation",
        },
        "recovery config",
    )
    freeze = _mapping(raw["freeze"], "freeze")
    parent = _mapping(raw["parent_stage_c0_v2"], "parent_stage_c0_v2")
    incident = _mapping(raw["incident"], "incident")
    repair = _mapping(raw["repair"], "repair")
    boundary = _mapping(raw["scientific_boundary"], "scientific_boundary")
    execution = _mapping(raw["execution"], "execution")
    output = _mapping(raw["output"], "output")
    implementation = _mapping(raw["implementation"], "implementation")
    _exact_keys(
        freeze,
        {
            "state",
            "mechanism",
            "pre_run_freeze_receipt",
            "print_hash_command",
            "freeze_command",
            "verify_config_command",
            "verify_recovery_command",
        },
        "freeze",
    )
    _exact_keys(
        parent,
        {
            "protocol_id",
            "config",
            "runner",
            "pre_run_freeze",
            "postverify_abort_receipt",
            "science_gate",
            "artifact_ledgers",
        },
        "parent_stage_c0_v2",
    )
    _exact_keys(
        incident,
        {
            "original_command",
            "original_exit_code",
            "terminal_error_type",
            "terminal_error_message",
            "original_v2_runner_self_verification_passed",
            "original_aggregate_must_not_be_described_as_v2_verified",
        },
        "incident",
    )
    _exact_keys(
        repair,
        {
            "failure_class",
            "exact_json_path",
            "producer_python_container_type",
            "stored_json_container_type",
            "allowed_adapter",
            "adapter_scope",
            "canonical_semantic_hash_algorithm",
            "require_canonical_semantic_hash_equality",
            "require_unadapted_v2_verifier_terminal_error_first",
            "require_adapted_full_v2_verifier_success",
            "thresholds_changed",
            "coverage_changed",
            "data_changed",
            "statistics_changed",
        },
        "repair",
    )
    _exact_keys(
        boundary,
        {
            "required_scientific_status",
            "stage_c1_authorized",
            "stage_c1_started",
            "stage_c_r1_r2_authorized",
            "stage_c_r1_r2_started",
            "formal_test_authorized",
            "formal_test_started",
            "no_validation_split",
            "validation_payload_opens",
            "test_payload_opens",
            "raw_source_image_payload_opens",
            "raw_source_target_payload_deserializations",
            "completed_evidence_artifact_reads_only",
        },
        "scientific_boundary",
    )
    _exact_keys(
        execution,
        {
            "device",
            "cuda_visible_devices",
            "cuda_initialized",
            "model_construction",
            "optimizer_construction",
            "candidate_selection",
            "new_candidate_or_outer_execution",
        },
        "execution",
    )
    _exact_keys(
        output,
        {
            "root",
            "recovery_verified_receipt",
            "atomic_no_replace",
            "refuse_overwrite",
            "immutable_mode",
        },
        "output",
    )
    _exact_keys(
        implementation,
        {"critical_code_paths", "path_order_is_frozen"},
        "implementation",
    )
    if (
        raw.get("schema_version") != 1
        or raw.get("protocol_id") != PROTOCOL_ID
        or raw.get("role") != "aggregate_only_postverify_erratum_recovery"
        or freeze.get("state") != "frozen"
        or parent.get("protocol_id") != PARENT_PROTOCOL_ID
        or incident.get("original_exit_code") != 2
        or incident.get("terminal_error_type") != "StageC0ProtocolError"
        or incident.get("terminal_error_message") != KNOWN_TERMINAL_ERROR
        or incident.get("original_v2_runner_self_verification_passed") is not False
        or incident.get("original_aggregate_must_not_be_described_as_v2_verified")
        is not True
        or repair.get("exact_json_path") != "$.parameter_space_ids"
        or repair.get("producer_python_container_type") != "tuple"
        or repair.get("stored_json_container_type") != "list"
        or repair.get("allowed_adapter")
        != "recursive_mapping_to_dict_and_tuple_or_list_to_list"
        or repair.get("adapter_scope")
        != "StageCAuthorization_serialization_projection_only"
        or repair.get("require_canonical_semantic_hash_equality") is not True
        or repair.get("require_unadapted_v2_verifier_terminal_error_first")
        is not True
        or repair.get("require_adapted_full_v2_verifier_success") is not True
        or any(
            repair.get(field) is not False
            for field in (
                "thresholds_changed",
                "coverage_changed",
                "data_changed",
                "statistics_changed",
            )
        )
        or boundary.get("required_scientific_status")
        != "scientific_no_eligible"
        or any(
            boundary.get(field) is not False
            for field in (
                "stage_c1_authorized",
                "stage_c1_started",
                "stage_c_r1_r2_authorized",
                "stage_c_r1_r2_started",
                "formal_test_authorized",
                "formal_test_started",
            )
        )
        or boundary.get("no_validation_split") is not True
        or any(
            boundary.get(field) != 0
            for field in (
                "validation_payload_opens",
                "test_payload_opens",
                "raw_source_image_payload_opens",
                "raw_source_target_payload_deserializations",
            )
        )
        or boundary.get("completed_evidence_artifact_reads_only") is not True
        or execution.get("device") != "cpu"
        or execution.get("cuda_visible_devices") != "empty_string_required"
        or any(
            execution.get(field) != "forbidden"
            for field in (
                "cuda_initialized",
                "model_construction",
                "optimizer_construction",
                "candidate_selection",
                "new_candidate_or_outer_execution",
            )
        )
        or output.get("root")
        != "results/cr_sitta/p3_stage_c0_aggregate_recovery_v1"
        or freeze.get("pre_run_freeze_receipt")
        != "results/cr_sitta/p3_stage_c0_aggregate_recovery_v1/PRE_RUN_FREEZE.json"
        or output.get("atomic_no_replace") is not True
        or output.get("refuse_overwrite") is not True
        or output.get("immutable_mode") != "0444"
        or implementation.get("path_order_is_frozen") is not True
    ):
        raise StageC0AggregateRecoveryError("recovery contract semantics differ")
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise StageC0AggregateRecoveryError("recovery config path differs from frozen path")
    expected_critical = (
        "recover_p3_stage_c0_aggregate_v1.py",
        "run_p3_stage_c0_signal_audit_v2.py",
        "analysis/stage_c_science_gate_v1.py",
        "configs/p3_stage_c_science_gate_v1.yaml",
        "tta/d0_secure_io.py",
    )
    critical = tuple(
        _relative(value, "critical code path")
        for value in _sequence(
            implementation.get("critical_code_paths"), "critical_code_paths"
        )
    )
    if critical != expected_critical:
        raise StageC0AggregateRecoveryError("critical code path roster/order differs")
    for name in ("config", "runner", "pre_run_freeze", "postverify_abort_receipt"):
        binding = _mapping(parent.get(name), f"parent {name}")
        _exact_keys(binding, {"path", "bytes", "sha256"}, f"parent {name}")
        _relative(binding.get("path"), f"parent {name} path")
        _sha256(binding.get("sha256"), f"parent {name} SHA")
        if isinstance(binding.get("bytes"), bool) or not isinstance(
            binding.get("bytes"), int
        ) or int(binding["bytes"]) <= 0:
            raise StageC0AggregateRecoveryError(f"parent {name} bytes invalid")
    if (
        parent["config"]["sha256"] != _v2.FROZEN_CONFIG_SHA256
        or parent["runner"]["sha256"]
        != "1e3cdf272d0a4d239d47ff98580faa7547c2c81c7fe4fbf6e0eb43db749c44bd"
        or parent["pre_run_freeze"]["sha256"]
        != "15608a2d69bbd04ea0fb5be9d9e565285fa49aec409b737d1d775629d86d6a24"
        or parent["postverify_abort_receipt"]["sha256"] != ABORT_RECEIPT_SHA256
    ):
        raise StageC0AggregateRecoveryError("parent primary binding differs")
    ledgers = _mapping(parent.get("artifact_ledgers"), "artifact ledgers")
    _exact_keys(
        ledgers,
        {"serialization", "candidates", "outers", "aggregate"},
        "artifact ledgers",
    )
    science_gate = _mapping(parent.get("science_gate"), "science gate")
    _exact_keys(
        science_gate,
        {"config_path", "config_sha256", "module_path", "module_sha256"},
        "science gate",
    )
    if ledgers.get("serialization") != (
        "sha256_of_canonical_json_without_trailing_newline"
    ):
        raise StageC0AggregateRecoveryError("artifact ledger serialization differs")
    for phase, count in (("candidates", 8), ("outers", 4)):
        records = _mapping(ledgers.get(phase), f"artifact ledgers {phase}")
        if tuple(records) != DATASETS:
            raise StageC0AggregateRecoveryError(
                f"artifact ledger dataset roster/order differs: {phase}"
            )
        for dataset in DATASETS:
            binding = _mapping(records[dataset], f"{phase} {dataset}")
            _exact_keys(
                binding,
                {"path", "file_count", "file_ledger_sha256"},
                f"{phase} {dataset}",
            )
            if binding.get("file_count") != count:
                raise StageC0AggregateRecoveryError(
                    f"artifact ledger file count differs: {phase} {dataset}"
                )
            _relative(binding.get("path"), f"{phase} {dataset} path")
            _sha256(binding.get("file_ledger_sha256"), f"{phase} {dataset} SHA")
    aggregate = _mapping(ledgers.get("aggregate"), "aggregate ledger")
    _exact_keys(
        aggregate,
        {"path", "file_count", "file_ledger_sha256"},
        "aggregate ledger",
    )
    if aggregate.get("file_count") != 5:
        raise StageC0AggregateRecoveryError("aggregate ledger file count differs")
    _relative(aggregate.get("path"), "aggregate path")
    _sha256(aggregate.get("file_ledger_sha256"), "aggregate ledger SHA")
    _repository_path(repository, freeze.get("pre_run_freeze_receipt"), "freeze path")
    _repository_path(repository, output.get("root"), "output root")
    receipt_name = output.get("recovery_verified_receipt")
    if receipt_name != "RECOVERY_VERIFIED_RECEIPT.json":
        raise StageC0AggregateRecoveryError("recovery receipt filename differs")
    return RecoveryContract(repository, config_path, config_sha256, raw)


def current_config_sha256(path: Path = DEFAULT_CONFIG) -> str:
    return read_stable_regular_file(path).sha256


def load_contract(path: Path = DEFAULT_CONFIG) -> RecoveryContract:
    if FROZEN_CONFIG_SHA256 == "TO_BE_FROZEN":
        raise StageC0AggregateRecoveryError(
            "recovery config is not frozen; pin FROZEN_CONFIG_SHA256 first"
        )
    stable = read_stable_regular_file(path)
    if stable.sha256 != FROZEN_CONFIG_SHA256:
        raise StageC0AggregateRecoveryError(
            "recovery config bytes differ from runner-pinned hash"
        )
    try:
        raw = yaml.load(stable.data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise StageC0AggregateRecoveryError("recovery YAML is invalid") from exc
    return _parse_contract(
        raw,
        repository=REPOSITORY,
        config_path=Path(path).resolve(),
        config_sha256=stable.sha256,
    )


def _assert_cpu_only(boundary: str) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "":
        observed = "<unset>" if visible is None else repr(visible)
        raise StageC0AggregateRecoveryError(
            "aggregate recovery requires CUDA_VISIBLE_DEVICES='' "
            f"({boundary}); observed={observed}"
        )
    torch_module = sys.modules.get("torch")
    cuda = None if torch_module is None else getattr(torch_module, "cuda", None)
    if cuda is not None and bool(cuda.is_initialized()):
        raise StageC0AggregateRecoveryError(
            f"aggregate recovery initialized CUDA ({boundary})"
        )


def _assert_payload_firewall(boundary: str) -> None:
    if TARGET_LOADER_MODULE in sys.modules:
        raise StageC0AggregateRecoveryError(
            f"outer target payload loader imported during recovery ({boundary})"
        )


def _file_binding(contract: RecoveryContract, configured: Mapping[str, Any]) -> dict[str, Any]:
    path = _repository_path(contract.repository, configured.get("path"), "bound file")
    stable = read_stable_regular_file(path)
    result = {
        "path": str(path.relative_to(contract.repository)),
        "bytes": len(stable.data),
        "sha256": stable.sha256,
    }
    if result != dict(configured):
        raise StageC0AggregateRecoveryError(f"bound file differs: {result['path']}")
    return result


def _artifact_binding(contract: RecoveryContract, relative: Any) -> dict[str, Any]:
    path = _repository_path(contract.repository, relative, "artifact directory")
    if path.is_symlink() or not path.is_dir():
        raise StageC0AggregateRecoveryError(f"artifact directory missing/unsafe: {path}")
    rows: list[dict[str, Any]] = []
    for member in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if member.is_symlink():
            raise StageC0AggregateRecoveryError(f"artifact contains symlink: {member}")
        if member.is_dir():
            continue
        if not member.is_file():
            raise StageC0AggregateRecoveryError(f"unsafe artifact member: {member}")
        stable = read_stable_regular_file(member)
        rows.append(
            {
                "path": member.relative_to(path).as_posix(),
                "bytes": len(stable.data),
                "sha256": stable.sha256,
            }
        )
    return {
        "path": str(path.relative_to(contract.repository)),
        "file_count": len(rows),
        "file_ledger": rows,
        "file_ledger_serialization": (
            "sha256_of_canonical_json_without_trailing_newline"
        ),
        "file_ledger_sha256": _canonical_sha256(rows),
    }


def _load_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    stable = read_stable_regular_file(path)
    try:
        value = json.loads(stable.data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StageC0AggregateRecoveryError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or stable.data != _canonical_json_bytes(
        value, newline=True
    ):
        raise StageC0AggregateRecoveryError(f"{label} is not canonical JSON")
    return value, stable.sha256


def validate_live_source_bindings(contract: RecoveryContract) -> dict[str, Any]:
    """Verify exact v2 files and all candidate/outer/aggregate file ledgers."""

    parent = _mapping(contract.raw["parent_stage_c0_v2"], "parent")
    primary = {
        name: _file_binding(contract, _mapping(parent[name], f"parent {name}"))
        for name in ("config", "runner", "pre_run_freeze")
    }
    gate = _mapping(parent["science_gate"], "science gate")
    for label, path_key, hash_key in (
        ("science gate config", "config_path", "config_sha256"),
        ("science gate module", "module_path", "module_sha256"),
    ):
        path = _repository_path(contract.repository, gate[path_key], label)
        if read_stable_regular_file(path).sha256 != gate[hash_key]:
            raise StageC0AggregateRecoveryError(f"{label} differs")
    abort_config = _mapping(parent["postverify_abort_receipt"], "abort binding")
    abort_path = _repository_path(
        contract.repository, abort_config["path"], "abort receipt"
    )
    abort, abort_sha = _load_canonical_json(abort_path, "abort receipt")
    if (
        abort_sha != abort_config["sha256"]
        or abort_sha != ABORT_RECEIPT_SHA256
        or len(read_stable_regular_file(abort_path).data) != abort_config["bytes"]
        or abort.get("artifact_type")
        != "cr_sitta_stage_c0_v2_aggregate_postverify_abort_receipt"
        or abort.get("protocol_id") != PARENT_PROTOCOL_ID
        or abort.get("status") != "archived_postpublication_verification_abort"
    ):
        raise StageC0AggregateRecoveryError("abort receipt identity differs")
    abort_sources = _mapping(abort.get("source_bindings"), "abort sources")
    if any(abort_sources.get(name) != primary[name] for name in primary):
        raise StageC0AggregateRecoveryError("abort primary source binding differs")
    if abort_sources.get("science_gate_config") != {
        "path": gate["config_path"],
        "bytes": len(read_stable_regular_file(contract.repository / gate["config_path"]).data),
        "sha256": gate["config_sha256"],
    } or abort_sources.get("science_gate_module") != {
        "path": gate["module_path"],
        "bytes": len(read_stable_regular_file(contract.repository / gate["module_path"]).data),
        "sha256": gate["module_sha256"],
    }:
        raise StageC0AggregateRecoveryError("abort science-gate binding differs")
    ledgers = _mapping(parent["artifact_ledgers"], "configured ledgers")
    live: dict[str, Any] = {"candidates": {}, "outers": {}}
    for configured_name, abort_name in (
        ("candidates", "candidate_artifacts"),
        ("outers", "outer_artifacts"),
    ):
        configured_phase = _mapping(ledgers[configured_name], configured_name)
        abort_phase = _mapping(abort_sources[abort_name], abort_name)
        if tuple(abort_phase) != DATASETS:
            raise StageC0AggregateRecoveryError(
                f"abort dataset roster/order differs: {abort_name}"
            )
        for dataset in DATASETS:
            binding = _artifact_binding(contract, configured_phase[dataset]["path"])
            expected_summary = {
                key: binding[key]
                for key in ("path", "file_count", "file_ledger_sha256")
            }
            if expected_summary != dict(configured_phase[dataset]):
                raise StageC0AggregateRecoveryError(
                    f"configured artifact ledger differs: {configured_name} {dataset}"
                )
            if abort_phase[dataset] != binding:
                raise StageC0AggregateRecoveryError(
                    f"abort all-file ledger differs: {configured_name} {dataset}"
                )
            live[configured_name][dataset] = expected_summary
    aggregate = _artifact_binding(contract, ledgers["aggregate"]["path"])
    aggregate_summary = {
        key: aggregate[key] for key in ("path", "file_count", "file_ledger_sha256")
    }
    if aggregate_summary != dict(ledgers["aggregate"]):
        raise StageC0AggregateRecoveryError("configured aggregate ledger differs")
    if abort_sources.get("original_aggregate_artifact") != aggregate:
        raise StageC0AggregateRecoveryError("abort aggregate all-file ledger differs")
    observed = _mapping(abort.get("observed_command"), "abort command")
    verifier = _mapping(abort.get("v2_verifier_outcome"), "abort verifier")
    science = _mapping(abort.get("scientific_boundary"), "abort boundary")
    if (
        observed.get("command") != contract.raw["incident"]["original_command"]
        or observed.get("exit_code") != 2
        or observed.get("terminal_error_type") != "StageC0ProtocolError"
        or observed.get("terminal_error_message") != KNOWN_TERMINAL_ERROR
        or verifier.get("reached_unique_known_terminal_error") is not True
        or verifier.get("original_v2_runner_self_verification_passed") is not False
        or science.get("scientific_status") != "scientific_no_eligible"
        or any(
            science.get(field) is not False
            for field in (
                "stage_c1_authorized",
                "stage_c1_started",
                "stage_c_r1_r2_authorized",
                "stage_c_r1_r2_started",
                "formal_test_authorized",
                "formal_test_started",
            )
        )
    ):
        raise StageC0AggregateRecoveryError("abort receipt fact boundary differs")
    return {
        "primary_files": primary,
        "science_gate": {
            "config_path": gate["config_path"],
            "config_sha256": gate["config_sha256"],
            "module_path": gate["module_path"],
            "module_sha256": gate["module_sha256"],
        },
        "postverify_abort_receipt": {
            "path": abort_config["path"],
            "sha256": abort_sha,
        },
        "artifact_ledgers": {**live, "aggregate": aggregate_summary},
    }


def _recovery_code_seal(contract: RecoveryContract) -> dict[str, Any]:
    paths = tuple(contract.raw["implementation"]["critical_code_paths"])
    files = [
        {
            "path": path,
            "sha256": read_stable_regular_file(contract.repository / path).sha256,
        }
        for path in paths
    ]
    return {"files": files, "bundle_sha256": _canonical_sha256(files)}


def _expected_freeze_receipt(
    contract: RecoveryContract, source_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_stage_c0_aggregate_recovery_pre_run_freeze",
        "protocol_id": PROTOCOL_ID,
        "config": {
            "path": str(contract.config_path.relative_to(contract.repository)),
            "sha256": contract.config_sha256,
        },
        "recovery_code_seal": _recovery_code_seal(contract),
        "source_bindings": dict(source_bindings),
        "recovery_verified_receipt_absent_at_publication": True,
        "cpu_only": True,
        "raw_source_image_payload_opens": 0,
        "raw_source_target_payload_deserializations": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
        "stage_c1_authorized": False,
        "stage_c_r1_r2_authorized": False,
        "formal_test_authorized": False,
    }


def _verify_freeze_receipt(
    contract: RecoveryContract, source_bindings: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], str]:
    source = (
        validate_live_source_bindings(contract)
        if source_bindings is None
        else source_bindings
    )
    observed, digest = _load_canonical_json(
        contract.freeze_receipt_path, "recovery pre-run freeze receipt"
    )
    if observed != _expected_freeze_receipt(contract, source):
        raise StageC0AggregateRecoveryError(
            "recovery pre-run freeze differs from current code/source bindings"
        )
    if contract.freeze_receipt_path.stat().st_mode & 0o222:
        raise StageC0AggregateRecoveryError(
            "recovery pre-run freeze receipt is writable"
        )
    return observed, digest


def _write_readonly_temporary(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def freeze_recovery(contract: RecoveryContract) -> dict[str, Any]:
    """Atomically freeze recovery code and the exact completed v2 input chain."""

    _assert_cpu_only("freeze entry")
    _assert_payload_firewall("freeze entry")
    source = validate_live_source_bindings(contract)
    destination = contract.freeze_receipt_path
    if destination.exists() or destination.is_symlink():
        receipt, digest = _verify_freeze_receipt(contract, source)
        return {**receipt, "sha256": digest, "existing_exact_receipt": True}
    if contract.verified_receipt_path.exists() or contract.verified_receipt_path.is_symlink():
        raise StageC0AggregateRecoveryError(
            "cannot freeze after a recovery verified receipt exists"
        )
    ensure_directory_chain_nofollow(
        contract.repository, destination.parent.relative_to(contract.repository).parts
    )
    expected = _expected_freeze_receipt(contract, source)
    payload = _canonical_json_bytes(expected, newline=True)
    temporary = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        _write_readonly_temporary(temporary, payload)

        def guard() -> None:
            if contract.verified_receipt_path.exists() or contract.verified_receipt_path.is_symlink():
                raise StageC0AggregateRecoveryError(
                    "recovery output appeared before freeze publication"
                )
            if validate_live_source_bindings(contract) != source:
                raise StageC0AggregateRecoveryError(
                    "v2 source bindings changed before freeze publication"
                )
            if _expected_freeze_receipt(contract, source) != expected:
                raise StageC0AggregateRecoveryError(
                    "recovery implementation changed before freeze publication"
                )

        publish_file_noreplace(temporary, destination, pre_rename_guard=guard)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    receipt, digest = _verify_freeze_receipt(contract, source)
    return {**receipt, "sha256": digest, "existing_exact_receipt": False}


def _load_parent_contract(contract: RecoveryContract) -> _v2.StageC0Contract:
    parent = contract.raw["parent_stage_c0_v2"]
    loaded = _v2.load_contract(contract.repository / parent["config"]["path"])
    if (
        loaded.config_sha256 != parent["config"]["sha256"]
        or _v2.PROTOCOL_ID != parent["protocol_id"]
        or read_stable_regular_file(Path(_v2.__file__)).sha256
        != parent["runner"]["sha256"]
    ):
        raise StageC0AggregateRecoveryError("live v2 contract/runner differs")
    return loaded


def _independent_recompute(
    contract: RecoveryContract, parent: _v2.StageC0Contract
) -> dict[str, Any]:
    gate_path, gate_config = _v2._load_gate_contract()
    outer_records: dict[str, Sequence[Mapping[str, Any]]] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    for dataset in DATASETS:
        candidate_path = _v2._artifact_destination(
            parent,
            phase="candidate",
            dataset=dataset,
            formal=True,
            smoke_id=None,
        )
        candidate_manifest, candidate_token = _v2.verify_candidate_artifact(
            candidate_path,
            contract=parent,
            dataset=dataset,
            expected_formal=True,
        )
        outer_path = _v2._artifact_destination(
            parent,
            phase="outer",
            dataset=dataset,
            formal=True,
            smoke_id=None,
        )
        _v2.verify_outer_artifact(
            outer_path,
            contract=parent,
            dataset=dataset,
            expected_formal=True,
            candidate_token=candidate_token,
        )
        outer_records[dataset] = _v2._read_jsonl(
            outer_path / "outer_episodes.jsonl"
        )
        identities[dataset] = _mapping(
            candidate_manifest["identity_adapter"], f"{dataset} identity"
        )
    aggregate_path = _v2._artifact_destination(
        parent, phase="aggregate", dataset=None, formal=True, smoke_id=None
    )
    stored_evidence = _v2._load_json(aggregate_path / "aggregate_evidence.json")
    recomputed_evidence = _v2.build_aggregate_evidence(
        outer_records, identities, gate_config=gate_config
    )
    if stored_evidence != recomputed_evidence:
        raise StageC0AggregateRecoveryError(
            "aggregate evidence differs under independent recomputation"
        )
    science = evaluate_stage_c0_science_gate(recomputed_evidence, gate_config)
    stored_science = _v2._load_json(
        aggregate_path / "science_decision_receipt.json"
    )
    if stored_science != science.to_receipt():
        raise StageC0AggregateRecoveryError(
            "science receipt differs under independent recomputation"
        )
    authorization_object = authorize_stage_c_followup(science)
    producer_authorization = asdict(authorization_object)
    stored_authorization = _v2._load_json(
        aggregate_path / "stage_c1_authorization.json"
    )
    _adapted, adapter_proof = canonicalize_authorization_with_proof(
        producer_authorization, stored_authorization
    )
    if (
        science.scientific_status != "scientific_no_eligible"
        or science.eligible_space_ids
        or authorization_object.stage_c1_allowed
        or authorization_object.stage_c_r1_r2_allowed
        or authorization_object.formal_test_allowed
        or stored_authorization.get("reason") != "scientific_no_eligible"
    ):
        raise StageC0AggregateRecoveryError(
            "recovery input is not the frozen scientific_no_eligible result"
        )
    return {
        "aggregate_evidence": {
            "path": str(
                (aggregate_path / "aggregate_evidence.json").relative_to(
                    contract.repository
                )
            ),
            "sha256": read_stable_regular_file(
                aggregate_path / "aggregate_evidence.json"
            ).sha256,
            "exact_equal_to_recomputed": True,
        },
        "science_decision_receipt": {
            "path": str(
                (aggregate_path / "science_decision_receipt.json").relative_to(
                    contract.repository
                )
            ),
            "sha256": read_stable_regular_file(
                aggregate_path / "science_decision_receipt.json"
            ).sha256,
            "exact_equal_to_recomputed": True,
        },
        "stage_c1_authorization": {
            "path": str(
                (aggregate_path / "stage_c1_authorization.json").relative_to(
                    contract.repository
                )
            ),
            "sha256": read_stable_regular_file(
                aggregate_path / "stage_c1_authorization.json"
            ).sha256,
            "python_object_exact_equal_before_projection": False,
            "canonical_json_semantics_equal": True,
        },
        "adapter_proof": adapter_proof,
        "science_gate": {
            "config_path": str(gate_path.relative_to(contract.repository)),
            "config_sha256": read_stable_regular_file(gate_path).sha256,
            "scientific_status": science.scientific_status,
            "eligible_space_ids": list(science.eligible_space_ids),
        },
    }


def collect_recovery_proof(
    contract: RecoveryContract,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Replay the raw failure, adapted full verifier, and independent rebuild."""

    _assert_cpu_only("recovery preflight entry")
    _assert_payload_firewall("recovery preflight entry")
    sources = validate_live_source_bindings(contract)
    _freeze, freeze_sha = _verify_freeze_receipt(contract, sources)
    parent = _load_parent_contract(contract)
    if _v2.asdict is not asdict:
        raise StageC0AggregateRecoveryError(
            "unadapted v2 verifier entry is not using dataclasses.asdict"
        )
    aggregate_path = _v2._artifact_destination(
        parent, phase="aggregate", dataset=None, formal=True, smoke_id=None
    )
    try:
        _v2.verify_aggregate_artifact(aggregate_path, contract=parent)
    except _v2.StageC0ProtocolError as exc:
        if str(exc) != KNOWN_TERMINAL_ERROR:
            raise StageC0AggregateRecoveryError(
                f"unadapted v2 verifier stopped before known terminal error: {exc}"
            ) from exc
    else:
        raise StageC0AggregateRecoveryError(
            "unadapted v2 verifier unexpectedly passed; recovery precondition absent"
        )
    _assert_cpu_only("after unadapted v2 verifier")
    _assert_payload_firewall("after unadapted v2 verifier")
    with _v2_authorization_json_projection():
        try:
            adapted_manifest = _v2.verify_aggregate_artifact(
                aggregate_path, contract=parent
            )
        except _v2.StageC0ProtocolError as exc:
            raise StageC0AggregateRecoveryError(
                f"adapted full v2 verifier did not reach exit 0: {exc}"
            ) from exc
    _assert_cpu_only("after adapted full v2 verifier")
    _assert_payload_firewall("after adapted full v2 verifier")
    if (
        adapted_manifest.get("scientific_status") != "scientific_no_eligible"
        or adapted_manifest.get("eligible_space_ids") != []
        or adapted_manifest.get("stage_c1_allowed") is not False
        or adapted_manifest.get("stage_c_r1_r2_allowed") is not False
        or adapted_manifest.get("formal_test_allowed") is not False
    ):
        raise StageC0AggregateRecoveryError(
            "adapted v2 verifier returned an unauthorized scientific projection"
        )
    recomputation = _independent_recompute(contract, parent)
    _assert_payload_firewall("after independent recomputation")
    _assert_cpu_only("recovery preflight return")
    proof = {
        "unadapted_v2_verifier": {
            "called": True,
            "passed": False,
            "terminal_error_type": "StageC0ProtocolError",
            "terminal_error_message": KNOWN_TERMINAL_ERROR,
            "known_unique_terminal_reached": True,
            "all_preceding_v2_checks_passed_by_control_flow": True,
        },
        "adapted_full_v2_verifier": {
            "called": True,
            "passed": True,
            "exit_semantics": "returned_manifest_without_second_error",
            "adapter_restored_after_call": True,
        },
        "independent_recomputation": recomputation,
    }
    return proof, sources, freeze_sha


def _build_verified_receipt(
    contract: RecoveryContract,
    *,
    proof: Mapping[str, Any],
    sources: Mapping[str, Any],
    freeze_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_stage_c0_aggregate_recovery_verified_receipt",
        "protocol_id": PROTOCOL_ID,
        "config": {
            "path": str(contract.config_path.relative_to(contract.repository)),
            "sha256": contract.config_sha256,
        },
        "status": "recovery_verified",
        "role": "aggregate_only_postverify_erratum_recovery",
        "original_v2_runner_self_verification_passed": False,
        "original_aggregate_must_not_be_described_as_v2_verified": True,
        "recovery_pre_run_freeze": {
            "path": str(contract.freeze_receipt_path.relative_to(contract.repository)),
            "sha256": freeze_sha256,
        },
        "source_bindings": dict(sources),
        "verification_proof": dict(proof),
        "scientific_result": {
            "protocol_status": "protocol_complete_under_independent_recovery",
            "scientific_status": "scientific_no_eligible",
            "eligible_space_ids": [],
            "stage_c1_authorized": False,
            "stage_c1_started": False,
            "stage_c_r1_r2_authorized": False,
            "stage_c_r1_r2_started": False,
            "formal_test_authorized": False,
            "formal_test_started": False,
        },
        "scientific_semantics": {
            "thresholds_changed": False,
            "coverage_changed": False,
            "data_changed": False,
            "statistics_changed": False,
        },
        "execution_boundary": {
            "cpu_only": True,
            "cuda_initialized": False,
            "model_construction_count": 0,
            "optimizer_construction_count": 0,
            "candidate_selection_count": 0,
            "new_candidate_execution_count": 0,
            "new_outer_execution_count": 0,
            "raw_source_image_payload_opens": 0,
            "raw_source_target_payload_deserializations": 0,
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
            "completed_evidence_artifact_reads_only": True,
        },
        "recovery_code_seal": _recovery_code_seal(contract),
        "publication": {
            "path": str(contract.verified_receipt_path.relative_to(contract.repository)),
            "atomic_no_replace": True,
            "refuse_overwrite": True,
            "immutable_mode": "0444",
        },
    }


def validate_recovery(contract: RecoveryContract) -> dict[str, Any]:
    proof, sources, freeze_sha = collect_recovery_proof(contract)
    receipt = _build_verified_receipt(
        contract, proof=proof, sources=sources, freeze_sha256=freeze_sha
    )
    return {
        "valid": True,
        "would_publish": str(contract.verified_receipt_path),
        "would_publish_sha256": hashlib.sha256(
            _canonical_json_bytes(receipt, newline=True)
        ).hexdigest(),
        "filesystem_written": False,
        "scientific_status": "scientific_no_eligible",
        "stage_c1_authorized": False,
        "stage_c_r1_r2_authorized": False,
        "formal_test_authorized": False,
        "raw_source_image_or_target_payload_opened": False,
        "validation_or_test_payload_opened": False,
    }


def verify_recovery_receipt(contract: RecoveryContract) -> dict[str, Any]:
    observed, observed_sha = _load_canonical_json(
        contract.verified_receipt_path, "recovery verified receipt"
    )
    proof, sources, freeze_sha = collect_recovery_proof(contract)
    expected = _build_verified_receipt(
        contract, proof=proof, sources=sources, freeze_sha256=freeze_sha
    )
    if observed != expected:
        raise StageC0AggregateRecoveryError(
            "recovery verified receipt differs from live full replay"
        )
    if contract.verified_receipt_path.stat().st_mode & 0o222:
        raise StageC0AggregateRecoveryError(
            "recovery verified receipt is writable"
        )
    return {
        "valid": True,
        "path": str(contract.verified_receipt_path),
        "sha256": observed_sha,
        "status": "recovery_verified",
        "original_v2_runner_self_verification_passed": False,
        "scientific_status": "scientific_no_eligible",
        "stage_c1_authorized": False,
        "stage_c_r1_r2_authorized": False,
        "formal_test_authorized": False,
    }


def run_recovery(contract: RecoveryContract) -> dict[str, Any]:
    """Publish the deterministic recovery receipt without replacing any path."""

    destination = contract.verified_receipt_path
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"recovery receipt already exists: {destination}")
    proof, sources, freeze_sha = collect_recovery_proof(contract)
    receipt = _build_verified_receipt(
        contract, proof=proof, sources=sources, freeze_sha256=freeze_sha
    )
    payload = _canonical_json_bytes(receipt, newline=True)
    ensure_directory_chain_nofollow(
        contract.repository, destination.parent.relative_to(contract.repository).parts
    )
    temporary = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        _write_readonly_temporary(temporary, payload)

        def guard() -> None:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"recovery receipt already exists: {destination}")
            live_sources = validate_live_source_bindings(contract)
            _freeze, live_freeze_sha = _verify_freeze_receipt(
                contract, live_sources
            )
            if live_sources != sources or live_freeze_sha != freeze_sha:
                raise StageC0AggregateRecoveryError(
                    "recovery source/freeze binding changed before publication"
                )
            if _recovery_code_seal(contract) != receipt["recovery_code_seal"]:
                raise StageC0AggregateRecoveryError(
                    "recovery code changed before publication"
                )
            staged = read_stable_regular_file(temporary)
            if staged.data != payload:
                raise StageC0AggregateRecoveryError(
                    "recovery staging bytes changed before publication"
                )

        publish_file_noreplace(temporary, destination, pre_rename_guard=guard)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return verify_recovery_receipt(contract)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("print-config-sha256")
    commands.add_parser("verify-config")
    commands.add_parser("freeze")
    commands.add_parser("validate")
    commands.add_parser("recover")
    commands.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    path = (
        arguments.config
        if arguments.config.is_absolute()
        else REPOSITORY / arguments.config
    )
    if arguments.command == "print-config-sha256":
        print(current_config_sha256(path))
        return 0
    contract = load_contract(path)
    if arguments.command == "verify-config":
        source = validate_live_source_bindings(contract)
        result = {
            "valid": True,
            "protocol_id": PROTOCOL_ID,
            "config_sha256": contract.config_sha256,
            "postverify_abort_receipt_sha256": source[
                "postverify_abort_receipt"
            ]["sha256"],
            "scientific_status": "scientific_no_eligible",
            "stage_c1_authorized": False,
            "stage_c_r1_r2_authorized": False,
            "formal_test_authorized": False,
        }
    elif arguments.command == "freeze":
        result = freeze_recovery(contract)
    elif arguments.command == "validate":
        result = validate_recovery(contract)
    elif arguments.command == "recover":
        result = run_recovery(contract)
    else:
        result = verify_recovery_receipt(contract)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StageC0AggregateRecoveryError, _v2.StageC0ProtocolError) as exc:
        print(f"Stage-C0 aggregate recovery error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


__all__ = [
    "ABORT_RECEIPT_SHA256",
    "DEFAULT_CONFIG",
    "FROZEN_CONFIG_SHA256",
    "KNOWN_TERMINAL_ERROR",
    "PROTOCOL_ID",
    "RecoveryContract",
    "StageC0AggregateRecoveryError",
    "canonicalize_authorization_with_proof",
    "collect_recovery_proof",
    "current_config_sha256",
    "freeze_recovery",
    "load_contract",
    "main",
    "run_recovery",
    "validate_live_source_bindings",
    "validate_recovery",
    "verify_recovery_receipt",
]
