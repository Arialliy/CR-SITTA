"""Immutable aggregate-only recovery for the completed Stage-B1 v2 cells.

The v2 cells are scientific inputs and are never rewritten here.  Recovery
only adapts the recursively frozen mechanism-gate container to ordinary JSON
containers, proves canonical semantic equality, and publishes a separately
versioned aggregate.  Every public verification repeats the v2 public
verifier for all 39 cells.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Final

import yaml

from analysis.d0_v3_label_free_shard import (
    canonical_json_bytes,
    parse_canonical_json,
)
from analysis.p3_stage_b1_contract_v2 import (
    P3StageB1ContractV2,
    load_p3_stage_b1_contract_v2,
)
from analysis import p3_stage_b1_aggregate_v2 as _v2
from tta.d0_secure_io import read_stable_regular_file, snapshot_regular_directory


CONFIG_RELATIVE_PATH: Final = "configs/p3_stage_b1_aggregate_recovery_v2_1.yaml"
CONFIG_FILE_SHA256: Final = (
    "955c3734715b0719081fb4ebe32afe99983b6d300ce5da92d3c348be05ed3710"
)
CONFIG_CANONICAL_MAPPING_SHA256: Final = (
    "75cf02640c876be37635e561eed934fc836eb67f1be34635dc809742b6fed421"
)

SCHEMA_VERSION: Final = 2
ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_aggregate_recovery_v2_1"
COMPLETE_ARTIFACT_TYPE: Final = (
    "cr_sitta_p3_stage_b1_aggregate_recovery_complete_v2_1"
)
INCIDENT_ARTIFACT_TYPE: Final = (
    "cr_sitta_p3_stage_b1_aggregate_prepublication_failure_v2"
)
RECOVERY_RECEIPT_ARTIFACT_TYPE: Final = (
    "cr_sitta_p3_stage_b1_aggregate_recovery_receipt_v2_1"
)
LINEAGE_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_cell_lineage_v2_1"
STRATUM_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_descriptive_stratum_v2_1"
SUMMARY_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_descriptive_summary_v2_1"
MECHANISM_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_mechanism_evidence_v2_1"
RAW_AUDIT_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_raw_vjp_numeric_audit_v2_1"

CELL_LINEAGE_FILENAME: Final = _v2.CELL_LINEAGE_FILENAME
STRATIFIED_STATISTICS_FILENAME: Final = _v2.STRATIFIED_STATISTICS_FILENAME
SUMMARY_STATISTICS_FILENAME: Final = _v2.SUMMARY_STATISTICS_FILENAME
MECHANISM_EVIDENCE_FILENAME: Final = _v2.MECHANISM_EVIDENCE_FILENAME
RAW_NUMERIC_AUDIT_FILENAME: Final = _v2.RAW_NUMERIC_AUDIT_FILENAME
RECOVERY_RECEIPT_FILENAME: Final = "recovery_receipt.json"
MANIFEST_FILENAME: Final = _v2.MANIFEST_FILENAME
COMPLETE_FILENAME: Final = _v2.COMPLETE_FILENAME
MEMBERS: Final = frozenset(
    {
        CELL_LINEAGE_FILENAME,
        STRATIFIED_STATISTICS_FILENAME,
        SUMMARY_STATISTICS_FILENAME,
        MECHANISM_EVIDENCE_FILENAME,
        RAW_NUMERIC_AUDIT_FILENAME,
        RECOVERY_RECEIPT_FILENAME,
        MANIFEST_FILENAME,
        COMPLETE_FILENAME,
    }
)
EXPECTED_CELL_COUNT: Final = 39
EPISODES_PER_CELL: Final = 64
TOTAL_EPISODE_COUNT: Final = EXPECTED_CELL_COUNT * EPISODES_PER_CELL
CELL_LEDGER_MEMBER_NAMES: Final = ("COMPLETE.json", "manifest.json")
CELL_LEDGER_SHA256: Final = (
    "514a58a1198c969cb86b387c257978294e7aeb62c9ddc978d17ccd4f5c2169bc"
)
AUTHORIZATION: Final = dict(_v2.AGGREGATE_AUTHORIZATION)
_STATUS_IDS: Final = frozenset(
    {"background_norm_dominance", "background_cancellation", "subthreshold_erasure"}
)
_STATUS_VALUES: Final = frozenset(
    {"supported", "not_supported", "not_estimable"}
)
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ROOT_FIELDS: Final = {
    "schema_version", "protocol_id", "role", "parent_stage_b1_v2", "repair",
    "related_code_sha256", "execution", "output", "implementation",
}
_REPAIR_FIELDS: Final = {
    "failure_class", "exception_message", "original_command", "original_exit_code",
    "mismatch_fields", "producer_container_type",
    "delegated_checker_expected_container_type", "allowed_adapter", "adapter_scope",
    "canonical_semantic_hash_algorithm", "require_semantic_hash_equality",
    "thresholds_changed", "coverage_changed", "data_changed", "statistics_changed",
    "scientific_status_before_recovery",
}
_RELATED_FIELDS: Final = {
    "parent_contract", "parent_aggregate_v1", "parent_aggregate_v2", "parent_runner_v2",
}
_EXECUTION_FIELDS: Final = {
    "device", "cuda_initialized", "validation_payload_access", "test_payload_access",
    "raw_image_access", "raw_target_access", "optimizer_construction",
    "model_construction", "candidate_selection", "stage_b3_authorized", "p5_authorized",
}
_OUTPUT_FIELDS: Final = {
    "root", "incident_relative_path", "aggregate_relative_path", "atomic_no_replace",
    "immutable",
}
_IMPLEMENTATION_FIELDS: Final = {"critical_code_paths", "path_order_is_frozen"}
_MISMATCH_FIELDS: Final = (
    "support_accounting.required_count_fields.per_cell",
    "support_accounting.required_count_fields.per_joint_cell",
    "support_accounting.required_count_fields.per_stratum",
    "support_accounting.required_count_fields.per_joint_stratum",
)
_ORIGINAL_COMMAND: Final = (
    "CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ./.conda/bin/python "
    "scripts/run_p3_stage_b_gradient_decomposition_v2.py --config "
    "configs/p3_stage_b_gradient_decomposition_v2.yaml aggregate"
)


class P3StageB1AggregateRecoveryV21Error(ValueError):
    """The aggregate-only recovery contract or artifact is invalid."""


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
            raise P3StageB1AggregateRecoveryV21Error(
                "unhashable YAML mapping key is forbidden"
            ) from exc
        if duplicate:
            raise P3StageB1AggregateRecoveryV21Error(
                f"duplicate YAML key is forbidden: {key!r}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True, slots=True)
class P3StageB1AggregateRecoveryV21Contract:
    config_file_sha256: str
    canonical_mapping_sha256: str
    protocol_id: str
    parent_config_path: str
    parent_output_root: str
    output_root: str
    aggregate_relative_path: str
    incident_relative_path: str
    critical_code_paths: tuple[str, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedStageB1AggregateRecoveryV21:
    path: Path
    manifest_sha256: str
    complete_sha256: str
    mechanism_evidence_sha256: str
    raw_numeric_audit_sha256: str
    recovery_receipt_sha256: str
    cell_count: int
    episode_count: int
    mechanism_statuses: Mapping[str, str]
    candidate_selection_performed: bool
    stage_b3_authorized: bool
    p5_authorized: bool


def recursive_thaw(value: Any) -> Any:
    """Convert all mappings and tuple/list containers to JSON-native objects."""

    if isinstance(value, Mapping):
        return {key: recursive_thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [recursive_thaw(child) for child in value]
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def canonical_semantic_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            recursive_thaw(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise P3StageB1AggregateRecoveryV21Error(
            "value is not finite canonical-JSON data"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def thaw_mechanism_gate_with_proof(
    frozen_gate: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    before = canonical_semantic_sha256(frozen_gate)
    thawed = recursive_thaw(frozen_gate)
    if not isinstance(thawed, dict):  # pragma: no cover - guarded by type signature
        raise P3StageB1AggregateRecoveryV21Error("mechanism gate must be a mapping")
    after = canonical_semantic_sha256(thawed)
    if before != after:
        raise P3StageB1AggregateRecoveryV21Error(
            "recursive thaw changed mechanism-gate canonical semantics"
        )
    return thawed, {
        "adapter": "recursive_mapping_to_dict_and_tuple_or_list_to_list",
        "scope": "mechanism_evidence_flags_only",
        "canonical_hash_algorithm": "sha256_of_canonical_json",
        "frozen_semantic_sha256": before,
        "thawed_semantic_sha256": after,
        "canonical_semantics_equal": True,
        "thresholds_changed": False,
        "coverage_changed": False,
        "data_changed": False,
        "statistics_changed": False,
    }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P3StageB1AggregateRecoveryV21Error(f"{label} must be a mapping")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise P3StageB1AggregateRecoveryV21Error(
            f"{label} fields differ; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise P3StageB1AggregateRecoveryV21Error(f"{label} must be lowercase SHA-256")
    return value


def _relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise P3StageB1AggregateRecoveryV21Error(f"{label} must be a string")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise P3StageB1AggregateRecoveryV21Error(
            f"{label} must be canonical project-relative"
        )
    return path.as_posix()


def _parse_contract(value: Any) -> P3StageB1AggregateRecoveryV21Contract:
    root = _mapping(value, label="recovery config")
    _fields(root, _ROOT_FIELDS, label="recovery config")
    parent = _mapping(root["parent_stage_b1_v2"], label="parent_stage_b1_v2")
    _fields(parent, {
        "protocol_id", "config_path", "config_sha256",
        "config_canonical_mapping_sha256", "cell_output_root", "cell_count",
        "episodes_per_cell", "episode_count", "cell_code_seal_bundle_sha256",
        "cell_code_seal_file_count", "cell_manifest_complete_ledger",
        "failed_aggregate_code_seal_bundle_sha256",
    }, label="parent_stage_b1_v2")
    ledger = _mapping(
        parent["cell_manifest_complete_ledger"],
        label="cell_manifest_complete_ledger",
    )
    _fields(ledger, {
        "base", "member_names", "path_format", "row_schema", "ordering",
        "serialization", "file_count", "sha256",
    }, label="cell_manifest_complete_ledger")
    repair = _mapping(root["repair"], label="repair")
    related = _mapping(root["related_code_sha256"], label="related_code_sha256")
    execution = _mapping(root["execution"], label="execution")
    output = _mapping(root["output"], label="output")
    implementation = _mapping(root["implementation"], label="implementation")
    _fields(repair, _REPAIR_FIELDS, label="repair")
    _fields(related, _RELATED_FIELDS, label="related_code_sha256")
    _fields(execution, _EXECUTION_FIELDS, label="execution")
    _fields(output, _OUTPUT_FIELDS, label="output")
    _fields(implementation, _IMPLEMENTATION_FIELDS, label="implementation")
    critical_raw = implementation.get("critical_code_paths")
    if isinstance(critical_raw, (str, bytes)) or not isinstance(critical_raw, Sequence):
        raise P3StageB1AggregateRecoveryV21Error("critical_code_paths must be a sequence")
    critical = tuple(_relative(item, label="critical code path") for item in critical_raw)
    expected_critical = (
        "analysis/p3_stage_b1_aggregate_recovery_v2_1.py",
        "scripts/run_p3_stage_b1_aggregate_recovery_v2_1.py",
        "analysis/p3_stage_b1_aggregate_v2.py",
        "analysis/p3_stage_b1_aggregate.py",
        "analysis/p3_stage_b1_outer_cell_shard_v2.py",
        "analysis/p3_stage_b1_outer_cell_shard.py",
        "analysis/foreground_background_gradient_decomposition_v2.py",
        "analysis/foreground_background_gradient_decomposition_v1.py",
        "analysis/p3_stage_b1_contract_v2.py",
        "analysis/d0_v3_label_free_shard.py",
        "tta/d0_secure_io.py",
        "tta/d0_v3_atomic_shard.py",
    )
    if critical != expected_critical:
        raise P3StageB1AggregateRecoveryV21Error(
            "critical_code_paths differ from the exact frozen ordered paths"
        )
    if (
        root.get("schema_version") != 1
        or root.get("protocol_id") != "cr-sitta-p3-stage-b1-aggregate-recovery-v2.1"
        or root.get("role") != "aggregate_only_erratum_recovery"
        or parent.get("protocol_id") != "cr-sitta-p3-stage-b1-gradient-decomposition-v2"
        or parent.get("cell_count") != EXPECTED_CELL_COUNT
        or parent.get("episodes_per_cell") != EPISODES_PER_CELL
        or parent.get("episode_count") != TOTAL_EPISODE_COUNT
        or parent.get("cell_code_seal_file_count") != 27
        or ledger.get("base") != "parent_stage_b1_v2.cell_output_root"
        or tuple(ledger.get("member_names", ())) != CELL_LEDGER_MEMBER_NAMES
        or ledger.get("path_format") != "output_root_relative_posix"
        or tuple(ledger.get("row_schema", ())) != ("path", "sha256")
        or ledger.get("ordering") != "lexicographic_by_path"
        or ledger.get("serialization") != "canonical_json_without_trailing_newline"
        or ledger.get("file_count") != 78
        or _sha(ledger.get("sha256"), label="cell ledger SHA") != CELL_LEDGER_SHA256
        or repair.get("failure_class") != "immutable_container_adapter_type_mismatch"
        or repair.get("exception_message") != "frozen mechanism gate semantics differ"
        or repair.get("original_command") != _ORIGINAL_COMMAND
        or repair.get("original_exit_code") != 1
        or tuple(repair.get("mismatch_fields", ())) != _MISMATCH_FIELDS
        or repair.get("producer_container_type") != "tuple"
        or repair.get("delegated_checker_expected_container_type") != "list"
        or repair.get("allowed_adapter")
        != "recursive_mapping_to_dict_and_tuple_or_list_to_list"
        or repair.get("adapter_scope") != "mechanism_evidence_flags_only"
        or repair.get("canonical_semantic_hash_algorithm")
        != "sha256_of_canonical_json"
        or repair.get("require_semantic_hash_equality") is not True
        or any(repair.get(field) is not False for field in (
            "thresholds_changed", "coverage_changed", "data_changed",
            "statistics_changed",
        ))
        or repair.get("scientific_status_before_recovery") != "not_evaluated"
        or execution != {
            "device": "cpu", "cuda_initialized": "forbidden",
            "validation_payload_access": "forbidden",
            "test_payload_access": "forbidden", "raw_image_access": "forbidden",
            "raw_target_access": "forbidden", "optimizer_construction": "forbidden",
            "model_construction": "forbidden", "candidate_selection": "forbidden",
            "stage_b3_authorized": False, "p5_authorized": False,
        }
        or output.get("atomic_no_replace") is not True
        or output.get("immutable") is not True
        or output.get("root")
        != "results/cr_sitta/p3_stage_b1_aggregate_recovery_v2_1"
        or output.get("incident_relative_path")
        != "AGGREGATE_PREPUBLICATION_FAILURE.json"
        or output.get("aggregate_relative_path") != "aggregate_phase_v2_1/R0"
        or implementation.get("path_order_is_frozen") is not True
    ):
        raise P3StageB1AggregateRecoveryV21Error(
            "aggregate-recovery frozen identity or safety boundary differs"
        )
    _sha(parent.get("config_sha256"), label="parent config SHA")
    _sha(parent.get("config_canonical_mapping_sha256"), label="parent canonical SHA")
    _sha(parent.get("cell_code_seal_bundle_sha256"), label="cell code seal")
    _sha(parent.get("failed_aggregate_code_seal_bundle_sha256"), label="failed seal")
    for name, digest in related.items():
        _sha(digest, label=f"related code {name}")
    return P3StageB1AggregateRecoveryV21Contract(
        "", canonical_semantic_sha256(root), str(root["protocol_id"]),
        _relative(parent["config_path"], label="parent config path"),
        _relative(parent["cell_output_root"], label="parent output root"),
        _relative(output["root"], label="recovery output root"),
        _relative(output["aggregate_relative_path"], label="aggregate path"),
        _relative(output["incident_relative_path"], label="incident path"),
        critical, _deep_freeze(root),
    )


def load_recovery_contract(
    path: str | os.PathLike[str],
) -> P3StageB1AggregateRecoveryV21Contract:
    stable = read_stable_regular_file(path)
    if stable.sha256 != CONFIG_FILE_SHA256:
        raise P3StageB1AggregateRecoveryV21Error(
            "aggregate-recovery config bytes drifted; "
            f"expected={CONFIG_FILE_SHA256}, observed={stable.sha256}"
        )
    try:
        decoded = stable.data.decode("utf-8")
        value = yaml.load(decoded, Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise P3StageB1AggregateRecoveryV21Error(
            "aggregate-recovery config is invalid UTF-8 YAML"
        ) from exc
    contract = _parse_contract(value)
    if contract.canonical_mapping_sha256 != CONFIG_CANONICAL_MAPPING_SHA256:
        raise P3StageB1AggregateRecoveryV21Error(
            "aggregate-recovery canonical mapping drifted"
        )
    return replace(contract, config_file_sha256=stable.sha256)


def load_bound_parent_contract(
    contract: P3StageB1AggregateRecoveryV21Contract,
    *,
    repository_root: str | os.PathLike[str],
) -> P3StageB1ContractV2:
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    parent = contract.raw["parent_stage_b1_v2"]
    loaded = load_p3_stage_b1_contract_v2(repository / contract.parent_config_path)
    if (
        loaded.config_file_sha256 != parent["config_sha256"]
        or loaded.canonical_mapping_sha256()
        != parent["config_canonical_mapping_sha256"]
        or str(loaded.raw["protocol_id"]) != parent["protocol_id"]
        or str(loaded.output_root) != contract.parent_output_root
    ):
        raise P3StageB1AggregateRecoveryV21Error(
            "bound Stage-B1 v2 parent contract differs"
        )
    return loaded


def _code_seal(
    repository_root: str | os.PathLike[str], paths: Sequence[str]
) -> Mapping[str, Any]:
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    files = [
        {"path": path, "sha256": read_stable_regular_file(repository / path).sha256}
        for path in paths
    ]
    return {
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def build_recovery_code_seal(
    contract: P3StageB1AggregateRecoveryV21Contract,
    *, repository_root: str | os.PathLike[str],
) -> Mapping[str, Any]:
    return _code_seal(repository_root, contract.critical_code_paths)


def verify_parent_code_seals(
    contract: P3StageB1AggregateRecoveryV21Contract,
    parent: P3StageB1ContractV2,
    *, repository_root: str | os.PathLike[str],
) -> Mapping[str, Any]:
    configured = contract.raw["parent_stage_b1_v2"]
    cell = _code_seal(
        repository_root, tuple(parent.raw["implementation"]["critical_code_paths"])
    )
    if (
        len(cell["files"]) != configured["cell_code_seal_file_count"]
        or cell["bundle_sha256"] != configured["cell_code_seal_bundle_sha256"]
    ):
        raise P3StageB1AggregateRecoveryV21Error("frozen v2 cell code seal differs")
    failed = _v2.build_aggregate_code_seal(repository_root)
    if failed["bundle_sha256"] != configured["failed_aggregate_code_seal_bundle_sha256"]:
        raise P3StageB1AggregateRecoveryV21Error(
            "failed v2 aggregate implementation seal differs"
        )
    related = contract.raw["related_code_sha256"]
    expected_related = {
        "parent_contract": "analysis/p3_stage_b1_contract_v2.py",
        "parent_aggregate_v1": "analysis/p3_stage_b1_aggregate.py",
        "parent_aggregate_v2": "analysis/p3_stage_b1_aggregate_v2.py",
        "parent_runner_v2": "scripts/run_p3_stage_b_gradient_decomposition_v2.py",
    }
    for key, path in expected_related.items():
        if read_stable_regular_file(Path(repository_root) / path).sha256 != related[key]:
            raise P3StageB1AggregateRecoveryV21Error(
                f"bound related implementation differs: {key}"
            )
    return cell


def build_cell_manifest_complete_ledger(
    contract: P3StageB1AggregateRecoveryV21Contract,
    parent: P3StageB1ContractV2,
    *, repository_root: str | os.PathLike[str],
) -> tuple[tuple[Mapping[str, str], ...], str]:
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    base = repository / contract.parent_output_root
    rows: list[Mapping[str, str]] = []
    for cell in _v2.fixed_stage_b1_cells_v2(
        repository, contract.parent_output_root, config=parent.raw
    ):
        for name in CELL_LEDGER_MEMBER_NAMES:
            path = cell.path / name
            relative = path.relative_to(base).as_posix()
            rows.append({"path": relative, "sha256": read_stable_regular_file(path).sha256})
    rows.sort(key=lambda row: row["path"])
    digest = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    configured = contract.raw["parent_stage_b1_v2"]["cell_manifest_complete_ledger"]
    if len(rows) != configured["file_count"] or digest != configured["sha256"]:
        raise P3StageB1AggregateRecoveryV21Error(
            "39-cell manifest/COMPLETE ledger differs"
        )
    return tuple(rows), digest


def collect_live_parent_preflight(
    contract: P3StageB1AggregateRecoveryV21Contract,
    parent: P3StageB1ContractV2,
    *, repository_root: str | os.PathLike[str],
    expected_cell_code_seal: Mapping[str, Any],
) -> _v2.StageB1AggregatePreflightV2:
    return _v2.collect_stage_b1_preflight_v2(
        repository_root=repository_root,
        output_root_relative=contract.parent_output_root,
        config=parent.raw,
        config_sha256=parent.config_file_sha256,
        expected_code_seal=expected_cell_code_seal,
    )


def build_incident_payload(
    contract: P3StageB1AggregateRecoveryV21Contract,
) -> bytes:
    repair = contract.raw["repair"]
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": INCIDENT_ARTIFACT_TYPE,
        "protocol_id": contract.protocol_id,
        "config_sha256": contract.config_file_sha256,
        "status": "archived_prepublication_aggregate_failure",
        "scientific_status": "not_evaluated",
        "parent_protocol_id": contract.raw["parent_stage_b1_v2"]["protocol_id"],
        "parent_config_sha256": contract.raw["parent_stage_b1_v2"]["config_sha256"],
        "failed_aggregate_code_seal_bundle_sha256": contract.raw[
            "parent_stage_b1_v2"
        ]["failed_aggregate_code_seal_bundle_sha256"],
        "completed_cell_evidence": {
            "cell_count": EXPECTED_CELL_COUNT,
            "episodes_per_cell": EPISODES_PER_CELL,
            "episode_count": TOTAL_EPISODE_COUNT,
            "cell_manifest_complete_ledger_sha256": contract.raw[
                "parent_stage_b1_v2"
            ]["cell_manifest_complete_ledger"]["sha256"],
            "cell_code_seal_bundle_sha256": contract.raw[
                "parent_stage_b1_v2"
            ]["cell_code_seal_bundle_sha256"],
            "cell_artifacts_modified": False,
        },
        "failure": {
            "class": repair["failure_class"],
            "message": repair["exception_message"],
            "original_command": repair["original_command"],
            "original_exit_code": repair["original_exit_code"],
            "mismatch_fields": list(repair["mismatch_fields"]),
            "producer_container_type": repair["producer_container_type"],
            "delegated_checker_expected_container_type": repair[
                "delegated_checker_expected_container_type"
            ],
        },
        "boundary": {
            "cell_artifacts_modified": False,
            "aggregate_artifact_published": False,
            "mechanism_flags_evaluated": False,
            "thresholds_changed": False,
            "coverage_changed": False,
            "data_changed": False,
            "statistics_changed": False,
            "test_or_validation_accessed": False,
            "stage_b3_authorized": False,
            "p5_authorized": False,
        },
        "authorization": AUTHORIZATION,
    }
    return canonical_json_bytes(value, newline=True)


def verify_incident(
    contract: P3StageB1AggregateRecoveryV21Contract,
    *, repository_root: str | os.PathLike[str],
) -> str:
    expected = build_incident_payload(contract)
    path = (
        Path(os.path.abspath(os.fspath(repository_root)))
        / contract.output_root
        / contract.incident_relative_path
    )
    observed = read_stable_regular_file(path)
    if observed.data != expected:
        raise P3StageB1AggregateRecoveryV21Error(
            "published aggregate-failure incident differs"
        )
    return observed.sha256


def _relabel_rows(
    rows: Sequence[Mapping[str, Any]], *, artifact_type: str
) -> tuple[Mapping[str, Any], ...]:
    output = []
    for row in rows:
        value = dict(row)
        value["schema_version"] = SCHEMA_VERSION
        value["artifact_type"] = artifact_type
        output.append(value)
    return tuple(output)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def build_recovery_payloads(
    contract: P3StageB1AggregateRecoveryV21Contract,
    parent: P3StageB1ContractV2,
    preflight: _v2.StageB1AggregatePreflightV2,
    *,
    repository_root: str | os.PathLike[str],
    incident_sha256: str,
    cell_ledger_sha256: str,
) -> dict[str, bytes]:
    if len(preflight.lineage) != EXPECTED_CELL_COUNT or len(preflight.records) != TOTAL_EPISODE_COUNT:
        raise P3StageB1AggregateRecoveryV21Error("recovery requires exact 39 x 64 preflight")
    if preflight.config_sha256 != parent.config_file_sha256:
        raise P3StageB1AggregateRecoveryV21Error("preflight parent config differs")
    gate, thaw_proof = thaw_mechanism_gate_with_proof(
        parent.raw["mechanism_evidence_flags"]
    )
    lineage = _relabel_rows(preflight.lineage, artifact_type=LINEAGE_ARTIFACT_TYPE)
    strata = _relabel_rows(
        _v2.build_stratified_statistics(preflight.records, config=parent.raw),
        artifact_type=STRATUM_ARTIFACT_TYPE,
    )
    summaries = _relabel_rows(
        _v2.build_summary_statistics(preflight.records, config=parent.raw),
        artifact_type=SUMMARY_ARTIFACT_TYPE,
    )
    mechanism = dict(_v2.build_mechanism_evidence(
        preflight.records, config=parent.raw,
        config_sha256=parent.config_file_sha256, mechanism_gate=gate,
    ))
    mechanism.update({
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MECHANISM_ARTIFACT_TYPE,
        "aggregate_recovery_protocol_id": contract.protocol_id,
        "aggregate_recovery_config_sha256": contract.config_file_sha256,
        "container_adapter_proof": thaw_proof,
    })
    raw_audit = dict(_v2.build_raw_numeric_audit(
        preflight.records, protocol_id=str(parent.raw["protocol_id"]),
        config_sha256=parent.config_file_sha256,
    ))
    raw_audit.update({
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RAW_AUDIT_ARTIFACT_TYPE,
        "aggregate_recovery_protocol_id": contract.protocol_id,
    })
    lineage_bytes = _jsonl(lineage)
    strata_bytes = _jsonl(strata)
    summaries_bytes = _jsonl(summaries)
    mechanism_bytes = canonical_json_bytes(mechanism, newline=True)
    raw_bytes = canonical_json_bytes(raw_audit, newline=True)
    recovery_seal = build_recovery_code_seal(contract, repository_root=repository_root)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RECOVERY_RECEIPT_ARTIFACT_TYPE,
        "protocol_id": contract.protocol_id,
        "config_sha256": contract.config_file_sha256,
        "incident": {"path": contract.incident_relative_path, "sha256": _sha(incident_sha256, label="incident SHA")},
        "parent": {
            "protocol_id": parent.raw["protocol_id"],
            "config_sha256": parent.config_file_sha256,
            "cell_count": EXPECTED_CELL_COUNT,
            "episode_count": TOTAL_EPISODE_COUNT,
            "cell_manifest_complete_ledger_sha256": _sha(cell_ledger_sha256, label="cell ledger SHA"),
            "cell_code_seal_bundle_sha256": contract.raw["parent_stage_b1_v2"]["cell_code_seal_bundle_sha256"],
            "failed_aggregate_code_seal_bundle_sha256": contract.raw["parent_stage_b1_v2"]["failed_aggregate_code_seal_bundle_sha256"],
        },
        "adapter_proof": thaw_proof,
        "scientific_semantics": {
            "thresholds_changed": False, "coverage_changed": False,
            "data_changed": False, "statistics_changed": False,
        },
        "authorization": AUTHORIZATION,
    }
    receipt_bytes = canonical_json_bytes(receipt, newline=True)
    payload: dict[str, bytes] = {
        CELL_LINEAGE_FILENAME: lineage_bytes,
        STRATIFIED_STATISTICS_FILENAME: strata_bytes,
        SUMMARY_STATISTICS_FILENAME: summaries_bytes,
        MECHANISM_EVIDENCE_FILENAME: mechanism_bytes,
        RAW_NUMERIC_AUDIT_FILENAME: raw_bytes,
        RECOVERY_RECEIPT_FILENAME: receipt_bytes,
    }
    references = {
        "cell_lineage": {"path": CELL_LINEAGE_FILENAME, "sha256": hashlib.sha256(lineage_bytes).hexdigest(), "count": len(lineage)},
        "stratified_statistics": {"path": STRATIFIED_STATISTICS_FILENAME, "sha256": hashlib.sha256(strata_bytes).hexdigest(), "count": len(strata)},
        "summary_statistics": {"path": SUMMARY_STATISTICS_FILENAME, "sha256": hashlib.sha256(summaries_bytes).hexdigest(), "count": len(summaries)},
        "mechanism_evidence": {"path": MECHANISM_EVIDENCE_FILENAME, "sha256": hashlib.sha256(mechanism_bytes).hexdigest()},
        "raw_numeric_audit": {"path": RAW_NUMERIC_AUDIT_FILENAME, "sha256": hashlib.sha256(raw_bytes).hexdigest()},
        "recovery_receipt": {"path": RECOVERY_RECEIPT_FILENAME, "sha256": hashlib.sha256(receipt_bytes).hexdigest()},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": contract.protocol_id,
        "config_sha256": contract.config_file_sha256,
        "scope": {
            "role": "aggregate_only_erratum_recovery",
            "output_root": contract.output_root,
            "aggregate_relative_path": contract.aggregate_relative_path,
            "cell_count": EXPECTED_CELL_COUNT,
            "episodes_per_cell": EPISODES_PER_CELL,
            "episode_count": TOTAL_EPISODE_COUNT,
            "source_train_pilot64_only": True,
        },
        "parent": receipt["parent"],
        "incident": receipt["incident"],
        **references,
        "recovery_code_seal": recovery_seal,
        "execution": {
            "cpu_only": True, "public_v2_cell_verifier_count": EXPECTED_CELL_COUNT,
            "raw_image_open_count": 0, "raw_target_open_count": 0,
            "model_build_count": 0, "optimizer_build_count": 0,
            "candidate_selection_count": 0,
        },
        "data_boundary": {
            "completed_v2_cell_artifacts_only": True,
            "validation_payloads_opened": 0, "test_payloads_opened": 0,
            "cell_artifacts_modified": False,
        },
        "authorization": AUTHORIZATION,
    }
    manifest_bytes = canonical_json_bytes(manifest, newline=True)
    payload[MANIFEST_FILENAME] = manifest_bytes
    complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": contract.protocol_id,
        "config_sha256": contract.config_file_sha256,
        "complete": True,
        "cell_count": EXPECTED_CELL_COUNT,
        "episode_count": TOTAL_EPISODE_COUNT,
        "manifest": {"path": MANIFEST_FILENAME, "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "payload_files": [
            {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(payload.items())
        ],
        "atomic_no_replace": True,
        "immutable": True,
        **AUTHORIZATION,
    }
    payload[COMPLETE_FILENAME] = canonical_json_bytes(complete, newline=True)
    if set(payload) != MEMBERS:
        raise P3StageB1AggregateRecoveryV21Error("recovery member set differs")
    return payload


def verify_recovery_aggregate(
    path: str | os.PathLike[str],
    *,
    contract: P3StageB1AggregateRecoveryV21Contract,
    parent: P3StageB1ContractV2,
    repository_root: str | os.PathLike[str],
    verify_live_cells: bool = True,
    expected_cell_code_seal: Mapping[str, Any] | None = None,
) -> VerifiedStageB1AggregateRecoveryV21:
    if verify_live_cells and expected_cell_code_seal is None:
        raise P3StageB1AggregateRecoveryV21Error(
            "public live verify requires the expected cell code seal"
        )
    live_parent = load_bound_parent_contract(contract, repository_root=repository_root)
    if (
        parent.config_file_sha256 != live_parent.config_file_sha256
        or parent.canonical_mapping_sha256()
        != live_parent.canonical_mapping_sha256()
    ):
        raise P3StageB1AggregateRecoveryV21Error(
            "supplied parent contract differs from its live frozen binding"
        )
    live_cell_code_seal = verify_parent_code_seals(
        contract, live_parent, repository_root=repository_root
    )
    if (
        expected_cell_code_seal is not None
        and expected_cell_code_seal != live_cell_code_seal
    ):
        raise P3StageB1AggregateRecoveryV21Error(
            "expected cell code seal differs from live frozen seal"
        )
    root = Path(os.path.abspath(os.fspath(path)))
    snapshot = snapshot_regular_directory(root)
    by_name = {member.path.name: member for member in snapshot.members}
    if set(by_name) != MEMBERS:
        raise P3StageB1AggregateRecoveryV21Error("recovery aggregate member set differs")
    manifest = parse_canonical_json(by_name[MANIFEST_FILENAME].data, label=MANIFEST_FILENAME, newline=True)
    complete = parse_canonical_json(by_name[COMPLETE_FILENAME].data, label=COMPLETE_FILENAME, newline=True)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != ARTIFACT_TYPE
        or manifest.get("protocol_id") != contract.protocol_id
        or manifest.get("config_sha256") != contract.config_file_sha256
        or manifest.get("authorization") != AUTHORIZATION
        or manifest.get("recovery_code_seal")
        != build_recovery_code_seal(contract, repository_root=repository_root)
    ):
        raise P3StageB1AggregateRecoveryV21Error("recovery manifest identity/seal differs")
    for field in (
        "cell_lineage", "stratified_statistics", "summary_statistics",
        "mechanism_evidence", "raw_numeric_audit", "recovery_receipt",
    ):
        reference = _mapping(manifest[field], label=field)
        name = reference.get("path")
        if name not in by_name or reference.get("sha256") != by_name[name].sha256:
            raise P3StageB1AggregateRecoveryV21Error(f"recovery {field} reference differs")
    expected_complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": contract.protocol_id,
        "config_sha256": contract.config_file_sha256,
        "complete": True,
        "cell_count": EXPECTED_CELL_COUNT,
        "episode_count": TOTAL_EPISODE_COUNT,
        "manifest": {"path": MANIFEST_FILENAME, "sha256": by_name[MANIFEST_FILENAME].sha256},
        "payload_files": [
            {"path": name, "sha256": by_name[name].sha256}
            for name in sorted(set(by_name) - {COMPLETE_FILENAME})
        ],
        "atomic_no_replace": True,
        "immutable": True,
        **AUTHORIZATION,
    }
    if complete != expected_complete:
        raise P3StageB1AggregateRecoveryV21Error("recovery COMPLETE differs")
    mechanism = parse_canonical_json(
        by_name[MECHANISM_EVIDENCE_FILENAME].data,
        label=MECHANISM_EVIDENCE_FILENAME, newline=True,
    )
    statuses = {
        key: str(value["status"])
        for key, value in _mapping(mechanism.get("P0_flags"), label="P0 flags").items()
    }
    if set(statuses) != _STATUS_IDS or any(value not in _STATUS_VALUES for value in statuses.values()):
        raise P3StageB1AggregateRecoveryV21Error("mechanism status set differs")
    proof = _mapping(mechanism.get("container_adapter_proof"), label="adapter proof")
    if proof.get("canonical_semantics_equal") is not True or any(
        proof.get(key) is not False
        for key in ("thresholds_changed", "coverage_changed", "data_changed", "statistics_changed")
    ):
        raise P3StageB1AggregateRecoveryV21Error("mechanism adapter proof differs")
    incident_sha = verify_incident(contract, repository_root=repository_root)
    ledger, ledger_sha = build_cell_manifest_complete_ledger(
        contract, live_parent, repository_root=repository_root
    )
    del ledger
    if manifest.get("incident") != {"path": contract.incident_relative_path, "sha256": incident_sha}:
        raise P3StageB1AggregateRecoveryV21Error("recovery incident binding differs")
    if verify_live_cells:
        preflight = collect_live_parent_preflight(
            contract, live_parent, repository_root=repository_root,
            expected_cell_code_seal=live_cell_code_seal,
        )
        expected = build_recovery_payloads(
            contract, live_parent, preflight, repository_root=repository_root,
            incident_sha256=incident_sha, cell_ledger_sha256=ledger_sha,
        )
        for name in MEMBERS:
            if by_name[name].data != expected[name]:
                raise P3StageB1AggregateRecoveryV21Error(
                    f"recovery aggregate differs from live rebuild: {name}"
                )
    return VerifiedStageB1AggregateRecoveryV21(
        root, by_name[MANIFEST_FILENAME].sha256,
        by_name[COMPLETE_FILENAME].sha256,
        by_name[MECHANISM_EVIDENCE_FILENAME].sha256,
        by_name[RAW_NUMERIC_AUDIT_FILENAME].sha256,
        by_name[RECOVERY_RECEIPT_FILENAME].sha256,
        EXPECTED_CELL_COUNT, TOTAL_EPISODE_COUNT, statuses,
        False, False, False,
    )


__all__ = [
    "ARTIFACT_TYPE", "AUTHORIZATION", "CELL_LEDGER_SHA256", "COMPLETE_FILENAME",
    "CONFIG_CANONICAL_MAPPING_SHA256", "CONFIG_FILE_SHA256", "CONFIG_RELATIVE_PATH",
    "EXPECTED_CELL_COUNT", "MEMBERS", "P3StageB1AggregateRecoveryV21Contract",
    "P3StageB1AggregateRecoveryV21Error", "TOTAL_EPISODE_COUNT",
    "VerifiedStageB1AggregateRecoveryV21", "build_cell_manifest_complete_ledger",
    "build_incident_payload", "build_recovery_code_seal", "build_recovery_payloads",
    "canonical_semantic_sha256", "collect_live_parent_preflight",
    "load_bound_parent_contract", "load_recovery_contract", "recursive_thaw",
    "thaw_mechanism_gate_with_proof", "verify_parent_code_seals",
    "verify_incident", "verify_recovery_aggregate",
]
