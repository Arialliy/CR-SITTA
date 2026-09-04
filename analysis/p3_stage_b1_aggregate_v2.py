"""CPU-only aggregate for Stage-B1 v2 cumulative/direct-VJP evidence.

Every cell is first verified by the v2 public verifier.  Descriptive metrics
and mechanism flags consume only its float64 telescoping science reports.
Raw independently reduced region VJPs receive a separate numerical audit and
have no route into eligibility, ranking, or authorization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any, Final

import numpy as np

from analysis.d0_v3_label_free_shard import canonical_json_bytes, parse_canonical_json
from analysis import p3_stage_b1_aggregate as _v1
from analysis.p3_stage_b1_outer_cell_shard_v2 import (
    COMPLETE_FILENAME as CELL_COMPLETE_FILENAME,
    CUMULATIVE_VJPS_FILENAME,
    GROUP_IDS,
    MANIFEST_FILENAME as CELL_MANIFEST_FILENAME,
    VerifiedStageB1CellShardV2,
    canonical_jsonl_bytes,
    parse_canonical_jsonl,
    verify_stage_b1_cell_shard_v2,
)
from tta.d0_secure_io import read_stable_regular_file, snapshot_regular_directory


SCHEMA_VERSION: Final = 2
ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_aggregate_v2"
COMPLETE_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_aggregate_complete_v2"
LINEAGE_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_cell_lineage_v2"
STRATUM_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_descriptive_stratum_v2"
SUMMARY_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_descriptive_summary_v2"
MECHANISM_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_mechanism_evidence_v2"
RAW_AUDIT_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_raw_vjp_numeric_audit_v2"

EXPECTED_CELL_COUNT: Final = 39
EPISODES_PER_CELL: Final = 64
TOTAL_EPISODE_COUNT: Final = EXPECTED_CELL_COUNT * EPISODES_PER_CELL
TARGET_PRESENCE_VALUES: Final = _v1.TARGET_PRESENCE_VALUES

CELL_LINEAGE_FILENAME: Final = "cell_lineage.jsonl"
STRATIFIED_STATISTICS_FILENAME: Final = "stratified_statistics.jsonl"
SUMMARY_STATISTICS_FILENAME: Final = "summary_statistics.jsonl"
MECHANISM_EVIDENCE_FILENAME: Final = "mechanism_evidence_flags.json"
RAW_NUMERIC_AUDIT_FILENAME: Final = "raw_vjp_numeric_audit.json"
MANIFEST_FILENAME: Final = "manifest.json"
COMPLETE_FILENAME: Final = "COMPLETE.json"
MEMBERS: Final = frozenset(
    {
        CELL_LINEAGE_FILENAME,
        STRATIFIED_STATISTICS_FILENAME,
        SUMMARY_STATISTICS_FILENAME,
        MECHANISM_EVIDENCE_FILENAME,
        RAW_NUMERIC_AUDIT_FILENAME,
        MANIFEST_FILENAME,
        COMPLETE_FILENAME,
    }
)

METRIC_PATHS: Final = _v1.METRIC_PATHS
ALL_METRIC_IDS: Final = _v1.ALL_METRIC_IDS
P0_MECHANISM_METRICS: Final = _v1.P0_MECHANISM_METRICS
SMALL_GROUP_METRICS: Final = _v1.SMALL_GROUP_METRICS
AGGREGATE_AUTHORIZATION: Final = dict(_v1.AGGREGATE_AUTHORIZATION)
AGGREGATE_CRITICAL_CODE_PATHS: Final = (
    "analysis/p3_stage_b1_aggregate_v2.py",
    "analysis/p3_stage_b1_aggregate.py",
    "analysis/p3_stage_b1_outer_cell_shard_v2.py",
    "analysis/p3_stage_b1_outer_cell_shard.py",
    "analysis/foreground_background_gradient_decomposition_v2.py",
    "analysis/foreground_background_gradient_decomposition_v1.py",
    "analysis/p3_stage_b1_contract_v2.py",
    "analysis/d0_v3_label_free_shard.py",
    "tta/d0_secure_io.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_LINEAGE_FIELDS = {
    "schema_version",
    "artifact_type",
    "cell_index",
    "dataset",
    "condition",
    "corruption_family",
    "severity",
    "replicate",
    "cell_path",
    "manifest_sha256",
    "complete_sha256",
    "cumulative_vjps_filename",
    "cumulative_vjps_sha256",
    "records_sha256",
    "group_layout_sha256",
    "outer_access_receipt_sha256",
    "ordered_image_ids_sha256",
    "target_identity_sha256",
    "record_count",
    "public_v2_cell_verifier_passed",
    "raw_target_reopened_by_aggregate",
    "raw_vjps_enter_science_metrics",
    "candidate_selection_performed",
    "stage_b3_authorized",
    "p5_authorized",
}


class P3StageB1AggregateV2Error(ValueError):
    """The v2 39-cell evidence grid or immutable aggregate is invalid."""


P3StageB1AggregateError = P3StageB1AggregateV2Error


@dataclass(frozen=True, slots=True)
class StageB1CellPathsV2:
    index: int
    dataset: str
    condition: str
    path: Path


@dataclass(frozen=True, slots=True)
class StageB1AggregatePreflightV2:
    repository_root: Path
    output_root_relative: str
    protocol_id: str
    config_sha256: str
    config: Mapping[str, Any]
    lineage: tuple[Mapping[str, Any], ...]
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VerifiedStageB1AggregateV2:
    path: Path
    manifest_sha256: str
    complete_sha256: str
    mechanism_evidence_sha256: str
    raw_numeric_audit_sha256: str
    cell_count: int
    episode_count: int
    stratified_record_count: int
    summary_record_count: int
    mechanism_statuses: Mapping[str, str]
    candidate_selection_performed: bool
    stage_b3_authorized: bool
    p5_authorized: bool


# Compatibility aliases for the mechanically versioned runner.
StageB1CellPaths = StageB1CellPathsV2
StageB1AggregatePreflight = StageB1AggregatePreflightV2
VerifiedStageB1Aggregate = VerifiedStageB1AggregateV2


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P3StageB1AggregateV2Error(f"{label} must be a mapping")
    return value


def _exact(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    result = _mapping(value, label=label)
    if set(result) != fields:
        raise P3StageB1AggregateV2Error(
            f"{label} fields differ; missing={sorted(fields-set(result))}, "
            f"unknown={sorted(set(result)-fields)}"
        )
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise P3StageB1AggregateV2Error(f"{label} must be lowercase SHA-256")
    return value


def _canonical_output_relative(value: str | os.PathLike[str]) -> str:
    path = Path(os.fspath(value))
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise P3StageB1AggregateV2Error("output root is not canonical relative")
    return path.as_posix()


def _repository_relative(root: Path, path: Path, *, label: str) -> str:
    try:
        relative = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
    except ValueError as exc:
        raise P3StageB1AggregateV2Error(f"{label} is outside repository") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise P3StageB1AggregateV2Error(f"{label} is not canonical")
    return relative.as_posix()


def _config_sequences(config: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    datasets_raw, conditions_raw = config.get("datasets"), config.get("conditions")
    if not isinstance(datasets_raw, Mapping):
        raise P3StageB1AggregateV2Error("config datasets must be a mapping")
    if isinstance(conditions_raw, (str, bytes)) or not isinstance(conditions_raw, Sequence):
        raise P3StageB1AggregateV2Error("config conditions must be a sequence")
    datasets = tuple(str(value) for value in datasets_raw)
    conditions = tuple(str(value) for value in conditions_raw)
    if len(datasets) != 3 or len(conditions) != 13:
        raise P3StageB1AggregateV2Error("B1 v2 config must define exactly 3 x 13 cells")
    return datasets, conditions


def _condition_parts(condition: str) -> tuple[str, str]:
    return ("clean", "S0") if condition == "clean_S0" else condition.rsplit("_", 1)


def fixed_stage_b1_cells_v2(
    repository_root: str | os.PathLike[str],
    output_root_relative: str | os.PathLike[str],
    *,
    config: Mapping[str, Any],
) -> tuple[StageB1CellPathsV2, ...]:
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    output = repository / _canonical_output_relative(output_root_relative)
    datasets, conditions = _config_sequences(config)
    return tuple(
        StageB1CellPathsV2(index, dataset, condition, output / "outer_phase" / "shards" / "R0" / dataset / condition)
        for index, (dataset, condition) in enumerate(
            (dataset, condition) for dataset in datasets for condition in conditions
        )
    )


fixed_stage_b1_cells = fixed_stage_b1_cells_v2


def _lineage_record(
    cell: StageB1CellPathsV2,
    verified: VerifiedStageB1CellShardV2,
    repository_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": LINEAGE_ARTIFACT_TYPE,
        "cell_index": cell.index,
        "dataset": cell.dataset,
        "condition": cell.condition,
        "corruption_family": verified.corruption_family,
        "severity": verified.severity,
        "replicate": "R0",
        "cell_path": _repository_relative(repository_root, cell.path, label="cell path"),
        "manifest_sha256": verified.manifest_sha256,
        "complete_sha256": verified.complete_sha256,
        "cumulative_vjps_filename": CUMULATIVE_VJPS_FILENAME,
        "cumulative_vjps_sha256": verified.cumulative_vjps_sha256,
        "records_sha256": verified.records_sha256,
        "group_layout_sha256": verified.group_layout_sha256,
        "outer_access_receipt_sha256": verified.outer_access_receipt_sha256,
        "ordered_image_ids_sha256": verified.ordered_image_ids_sha256,
        "target_identity_sha256": verified.target_identity_sha256,
        "record_count": verified.record_count,
        "public_v2_cell_verifier_passed": True,
        "raw_target_reopened_by_aggregate": False,
        "raw_vjps_enter_science_metrics": False,
        "candidate_selection_performed": False,
        "stage_b3_authorized": False,
        "p5_authorized": False,
    }


def _validate_target_invariance(
    records: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    conditions: Sequence[str],
) -> None:
    try:
        _v1._validate_target_invariance(records, datasets, conditions)
    except Exception as exc:
        raise P3StageB1AggregateV2Error(str(exc)) from exc


def collect_stage_b1_preflight_v2(
    *,
    repository_root: str | os.PathLike[str],
    output_root_relative: str | os.PathLike[str],
    config: Mapping[str, Any],
    config_sha256: str,
    expected_code_seal: Mapping[str, Any] | None = None,
) -> StageB1AggregatePreflightV2:
    if expected_code_seal is None:
        raise P3StageB1AggregateV2Error(
            "live 39-cell preflight requires the expected cell code seal"
        )
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    output_relative = _canonical_output_relative(output_root_relative)
    digest = _sha256(config_sha256, label="config_sha256")
    datasets, conditions = _config_sequences(config)
    lineage: list[Mapping[str, Any]] = []
    records: list[Mapping[str, Any]] = []
    for cell in fixed_stage_b1_cells_v2(repository, output_relative, config=config):
        verified = verify_stage_b1_cell_shard_v2(
            cell.path,
            repository_root=repository,
            config=config,
            expected_config_sha256=digest,
            verify_live_parents=True,
            expected_code_seal=expected_code_seal,
        )
        if (
            verified.dataset != cell.dataset
            or verified.condition != cell.condition
            or verified.record_count != EPISODES_PER_CELL
        ):
            raise P3StageB1AggregateV2Error("verified v2 cell grid binding differs")
        lineage.append(_lineage_record(cell, verified, repository))
        records.extend(verified.records)
    if len(lineage) != EXPECTED_CELL_COUNT or len(records) != TOTAL_EPISODE_COUNT:
        raise P3StageB1AggregateV2Error("v2 grid is not exact 39 x 64")
    _validate_target_invariance(records, datasets, conditions)
    for dataset in datasets:
        rows = [record for record in lineage if record["dataset"] == dataset]
        for field in ("ordered_image_ids_sha256", "target_identity_sha256"):
            if len({str(record[field]) for record in rows}) != 1:
                raise P3StageB1AggregateV2Error(f"{dataset} {field} drifts")
    return StageB1AggregatePreflightV2(
        repository,
        output_relative,
        str(config["protocol_id"]),
        digest,
        config,
        tuple(lineage),
        tuple(records),
    )


collect_stage_b1_preflight = collect_stage_b1_preflight_v2


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


def build_stratified_statistics(
    records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    try:
        return _relabel_rows(
            _v1.build_stratified_statistics(records, config=config),
            artifact_type=STRATUM_ARTIFACT_TYPE,
        )
    except Exception as exc:
        raise P3StageB1AggregateV2Error(str(exc)) from exc


def build_summary_statistics(
    records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    try:
        return _relabel_rows(
            _v1.build_summary_statistics(records, config=config),
            artifact_type=SUMMARY_ARTIFACT_TYPE,
        )
    except Exception as exc:
        raise P3StageB1AggregateV2Error(str(exc)) from exc


def build_mechanism_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    mechanism_gate: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        value = dict(
            _v1.build_mechanism_evidence(
                records,
                config=config,
                config_sha256=config_sha256,
                mechanism_gate=mechanism_gate,
            )
        )
    except Exception as exc:
        raise P3StageB1AggregateV2Error(str(exc)) from exc
    value["schema_version"] = SCHEMA_VERSION
    value["artifact_type"] = MECHANISM_ARTIFACT_TYPE
    value["numeric_estimator"] = {
        "source": "stored_cumulative_vjps_cpu_float64_telescoping",
        "raw_direct_audit_used": False,
    }
    return value


def build_raw_numeric_audit(
    records: Sequence[Mapping[str, Any]], *, protocol_id: str, config_sha256: str
) -> Mapping[str, Any]:
    if len(records) != TOTAL_EPISODE_COUNT:
        raise P3StageB1AggregateV2Error("raw audit requires exactly 2496 records")
    components: dict[str, Any] = {}
    for component in ("foreground_suprathreshold", "background"):
        rows = []
        for record in records:
            audit = _mapping(
                _mapping(record["gradient_integrity"], label="gradient integrity")[
                    "raw_consistency"
                ],
                label="raw consistency",
            )
            if (
                audit.get("role") != "numeric_diagnostic_only"
                or audit.get("failure_action")
                != "record_only_never_protocol_or_science_gate"
                or audit.get("enters_science_metrics") is not False
                or audit.get("enters_mechanism_flags") is not False
                or audit.get("enters_selection_or_ranking") is not False
            ):
                raise P3StageB1AggregateV2Error("raw audit firewall differs")
            rows.append(_mapping(audit["components"][component], label=component))
        max_abs = np.asarray([float(row["max_abs_error"]) for row in rows], dtype=np.float64)
        relative = np.asarray([float(row["relative_l2_error"]) for row in rows], dtype=np.float64)
        if not np.isfinite(max_abs).all() or not np.isfinite(relative).all():
            raise P3StageB1AggregateV2Error("raw numeric audit contains NaN/Inf")
        within = [row["within_dual_tolerance"] for row in rows]
        if not all(type(value) is bool for value in within):
            raise P3StageB1AggregateV2Error("raw audit tolerance flags are not bool")
        components[component] = {
            "episode_count": len(rows),
            "within_dual_tolerance_count": sum(within),
            "outside_dual_tolerance_count": len(rows) - sum(within),
            "max_max_abs_error": float(np.max(max_abs)),
            "max_relative_l2_error": float(np.max(relative)),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RAW_AUDIT_ARTIFACT_TYPE,
        "protocol_id": protocol_id,
        "config_sha256": _sha256(config_sha256, label="config_sha256"),
        "role": "numeric_diagnostic_only",
        "failure_action": "record_only_never_protocol_or_science_gate",
        "episode_count": TOTAL_EPISODE_COUNT,
        "components": components,
        "science_firewall": {
            "enters_descriptive_science_metrics": False,
            "enters_mechanism_flags": False,
            "enters_coverage": False,
            "enters_selection_or_ranking": False,
            "changes_authorization": False,
        },
        "authorization": AGGREGATE_AUTHORIZATION,
    }


def build_aggregate_code_seal(repository_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(repository_root)))
    files = [
        {"path": path, "sha256": read_stable_regular_file(root / path).sha256}
        for path in AGGREGATE_CRITICAL_CODE_PATHS
    ]
    return {
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def _validate_lineage(
    lineage: Sequence[Mapping[str, Any]], *, preflight: StageB1AggregatePreflightV2
) -> tuple[Mapping[str, Any], ...]:
    cells = fixed_stage_b1_cells_v2(
        preflight.repository_root, preflight.output_root_relative, config=preflight.config
    )
    if isinstance(lineage, (str, bytes, Mapping)) or len(lineage) != len(cells):
        raise P3StageB1AggregateV2Error("lineage must contain exactly 39 rows")
    output: list[Mapping[str, Any]] = []
    paths: set[str] = set()
    for expected, record in zip(cells, lineage, strict=True):
        value = _exact(record, _LINEAGE_FIELDS, label="lineage record")
        family, severity = _condition_parts(expected.condition)
        expected_path = _repository_relative(
            preflight.repository_root, expected.path, label="cell path"
        )
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("artifact_type") != LINEAGE_ARTIFACT_TYPE
            or value.get("cell_index") != expected.index
            or value.get("dataset") != expected.dataset
            or value.get("condition") != expected.condition
            or value.get("corruption_family") != family
            or value.get("severity") != severity
            or value.get("replicate") != "R0"
            or value.get("cell_path") != expected_path
            or value.get("cumulative_vjps_filename") != CUMULATIVE_VJPS_FILENAME
            or value.get("record_count") != EPISODES_PER_CELL
            or value.get("public_v2_cell_verifier_passed") is not True
            or value.get("raw_target_reopened_by_aggregate") is not False
            or value.get("raw_vjps_enter_science_metrics") is not False
            or value.get("candidate_selection_performed") is not False
            or value.get("stage_b3_authorized") is not False
            or value.get("p5_authorized") is not False
        ):
            raise P3StageB1AggregateV2Error("lineage identity/firewall differs")
        for field in (
            "manifest_sha256", "complete_sha256", "cumulative_vjps_sha256",
            "records_sha256", "group_layout_sha256", "outer_access_receipt_sha256",
            "ordered_image_ids_sha256", "target_identity_sha256",
        ):
            _sha256(value.get(field), label=field)
        if expected_path in paths:
            raise P3StageB1AggregateV2Error("duplicate lineage cell path")
        paths.add(expected_path)
        output.append(value)
    datasets = _config_sequences(preflight.config)[0]
    for dataset in datasets:
        rows = [value for value in output if value["dataset"] == dataset]
        for field in ("ordered_image_ids_sha256", "target_identity_sha256"):
            if len({str(value[field]) for value in rows}) != 1:
                raise P3StageB1AggregateV2Error(
                    f"{dataset} lineage target/order SHA drifts"
                )
    return tuple(output)


def build_stage_b1_aggregate_payloads_v2(
    preflight: StageB1AggregatePreflightV2,
    *,
    mechanism_gate: Mapping[str, Any],
) -> dict[str, bytes]:
    if not isinstance(preflight, StageB1AggregatePreflightV2):
        raise P3StageB1AggregateV2Error("preflight has wrong type")
    lineage = _validate_lineage(preflight.lineage, preflight=preflight)
    if len(preflight.records) != TOTAL_EPISODE_COUNT:
        raise P3StageB1AggregateV2Error("aggregate requires exactly 2496 episodes")
    datasets, conditions = _config_sequences(preflight.config)
    _validate_target_invariance(preflight.records, datasets, conditions)
    stratified = build_stratified_statistics(preflight.records, config=preflight.config)
    summaries = build_summary_statistics(preflight.records, config=preflight.config)
    mechanism = build_mechanism_evidence(
        preflight.records,
        config=preflight.config,
        config_sha256=preflight.config_sha256,
        mechanism_gate=mechanism_gate,
    )
    raw_audit = build_raw_numeric_audit(
        preflight.records,
        protocol_id=preflight.protocol_id,
        config_sha256=preflight.config_sha256,
    )
    lineage_bytes = canonical_jsonl_bytes(lineage)
    stratified_bytes = canonical_jsonl_bytes(stratified)
    summary_bytes = canonical_jsonl_bytes(summaries)
    mechanism_bytes = canonical_json_bytes(mechanism, newline=True)
    raw_audit_bytes = canonical_json_bytes(raw_audit, newline=True)
    code_seal = build_aggregate_code_seal(preflight.repository_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": preflight.protocol_id,
        "config_sha256": preflight.config_sha256,
        "scope": {
            "output_root": preflight.output_root_relative,
            "cell_count": EXPECTED_CELL_COUNT,
            "episodes_per_cell": EPISODES_PER_CELL,
            "episode_count": TOTAL_EPISODE_COUNT,
            "cell_order": "dataset_major_condition_minor",
            "source_train_pilot64_only": True,
            "numeric_estimator": "v2_cpu_float64_telescoping",
        },
        "cell_lineage": {
            "path": CELL_LINEAGE_FILENAME,
            "sha256": hashlib.sha256(lineage_bytes).hexdigest(),
            "count": EXPECTED_CELL_COUNT,
            "target_identity_invariance_verified": True,
        },
        "stratified_statistics": {
            "path": STRATIFIED_STATISTICS_FILENAME,
            "sha256": hashlib.sha256(stratified_bytes).hexdigest(),
            "count": len(stratified),
            "accumulation_dtype": "float64",
            "raw_vjps_used": False,
        },
        "summary_statistics": {
            "path": SUMMARY_STATISTICS_FILENAME,
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "count": len(summaries),
            "accumulation_dtype": "float64",
            "raw_vjps_used": False,
        },
        "mechanism_evidence": {
            "path": MECHANISM_EVIDENCE_FILENAME,
            "sha256": hashlib.sha256(mechanism_bytes).hexdigest(),
            "P0_flag_count": 3,
            "small_group_hint_count": 8,
            "raw_vjps_used": False,
            "selection_or_ranking_performed": False,
            "stage_b3_authorized": False,
        },
        "raw_numeric_audit": {
            "path": RAW_NUMERIC_AUDIT_FILENAME,
            "sha256": hashlib.sha256(raw_audit_bytes).hexdigest(),
            "role": "numeric_diagnostic_only",
            "enters_scientific_eligibility": False,
        },
        "aggregate_code_seal": code_seal,
        "execution": {
            "cpu_only": True,
            "public_v2_cell_verifier_count": EXPECTED_CELL_COUNT,
            "episode_parse_count": TOTAL_EPISODE_COUNT,
            "raw_image_open_count": 0,
            "raw_target_open_count": 0,
            "model_build_count": 0,
            "optimizer_build_count": 0,
            "gpu_compute_count": 0,
            "candidate_selection_count": 0,
        },
        "data_boundary": {
            "source_train_completed_v2_cell_artifacts_only": True,
            "raw_train_images_opened": 0,
            "raw_train_targets_opened": 0,
            "validation_payloads_opened": 0,
            "test_split_files_opened": 0,
            "test_images_opened": 0,
            "test_targets_opened": 0,
        },
        "authorization": AGGREGATE_AUTHORIZATION,
    }
    manifest_bytes = canonical_json_bytes(manifest, newline=True)
    payload = {
        CELL_LINEAGE_FILENAME: lineage_bytes,
        STRATIFIED_STATISTICS_FILENAME: stratified_bytes,
        SUMMARY_STATISTICS_FILENAME: summary_bytes,
        MECHANISM_EVIDENCE_FILENAME: mechanism_bytes,
        RAW_NUMERIC_AUDIT_FILENAME: raw_audit_bytes,
        MANIFEST_FILENAME: manifest_bytes,
    }
    complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": preflight.protocol_id,
        "complete": True,
        "config_sha256": preflight.config_sha256,
        "cell_count": EXPECTED_CELL_COUNT,
        "episode_count": TOTAL_EPISODE_COUNT,
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "payload_files": [
            {"path": name, "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in sorted(payload.items())
        ],
        "mechanism_evidence": {
            "path": MECHANISM_EVIDENCE_FILENAME,
            "sha256": hashlib.sha256(mechanism_bytes).hexdigest(),
        },
        "raw_numeric_audit": {
            "path": RAW_NUMERIC_AUDIT_FILENAME,
            "sha256": hashlib.sha256(raw_audit_bytes).hexdigest(),
            "affects_science": False,
        },
        "atomic_no_replace": True,
        "immutable": True,
        **AGGREGATE_AUTHORIZATION,
    }
    payload[COMPLETE_FILENAME] = canonical_json_bytes(complete, newline=True)
    return payload


build_stage_b1_aggregate_payloads = build_stage_b1_aggregate_payloads_v2


def _parse_local(
    by_name: Mapping[str, Any], *, config: Mapping[str, Any], output_relative: str,
    config_sha: str,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    manifest = parse_canonical_json(
        by_name[MANIFEST_FILENAME].data, label=MANIFEST_FILENAME, newline=True
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != ARTIFACT_TYPE
        or manifest.get("protocol_id") != config.get("protocol_id")
        or manifest.get("config_sha256") != config_sha
        or manifest.get("authorization") != AGGREGATE_AUTHORIZATION
    ):
        raise P3StageB1AggregateV2Error("aggregate manifest identity differs")
    expected_scope = {
        "output_root": output_relative,
        "cell_count": EXPECTED_CELL_COUNT,
        "episodes_per_cell": EPISODES_PER_CELL,
        "episode_count": TOTAL_EPISODE_COUNT,
        "cell_order": "dataset_major_condition_minor",
        "source_train_pilot64_only": True,
        "numeric_estimator": "v2_cpu_float64_telescoping",
    }
    if manifest.get("scope") != expected_scope:
        raise P3StageB1AggregateV2Error("aggregate scope differs")
    lineage = parse_canonical_jsonl(
        by_name[CELL_LINEAGE_FILENAME].data,
        count=EXPECTED_CELL_COUNT,
        label=CELL_LINEAGE_FILENAME,
    )
    strata = parse_canonical_jsonl(
        by_name[STRATIFIED_STATISTICS_FILENAME].data,
        count=3 * 13 * len(GROUP_IDS) * 2,
        label=STRATIFIED_STATISTICS_FILENAME,
    )
    summaries = parse_canonical_jsonl(
        by_name[SUMMARY_STATISTICS_FILENAME].data,
        count=int(manifest["summary_statistics"]["count"]),
        label=SUMMARY_STATISTICS_FILENAME,
    )
    mechanism = parse_canonical_json(
        by_name[MECHANISM_EVIDENCE_FILENAME].data,
        label=MECHANISM_EVIDENCE_FILENAME,
        newline=True,
    )
    raw_audit = parse_canonical_json(
        by_name[RAW_NUMERIC_AUDIT_FILENAME].data,
        label=RAW_NUMERIC_AUDIT_FILENAME,
        newline=True,
    )
    references = (
        ("cell_lineage", CELL_LINEAGE_FILENAME, len(lineage)),
        ("stratified_statistics", STRATIFIED_STATISTICS_FILENAME, len(strata)),
        ("summary_statistics", SUMMARY_STATISTICS_FILENAME, len(summaries)),
        ("mechanism_evidence", MECHANISM_EVIDENCE_FILENAME, None),
        ("raw_numeric_audit", RAW_NUMERIC_AUDIT_FILENAME, None),
    )
    for field, filename, count in references:
        reference = _mapping(manifest[field], label=field)
        if reference.get("path") != filename or reference.get("sha256") != by_name[filename].sha256:
            raise P3StageB1AggregateV2Error(f"aggregate {field} reference differs")
        if count is not None and reference.get("count") != count:
            raise P3StageB1AggregateV2Error(f"aggregate {field} count differs")
    if (
        mechanism.get("authorization") != AGGREGATE_AUTHORIZATION
        or mechanism.get("numeric_estimator")
        != {
            "source": "stored_cumulative_vjps_cpu_float64_telescoping",
            "raw_direct_audit_used": False,
        }
        or raw_audit.get("science_firewall")
        != {
            "enters_descriptive_science_metrics": False,
            "enters_mechanism_flags": False,
            "enters_coverage": False,
            "enters_selection_or_ranking": False,
            "changes_authorization": False,
        }
        or raw_audit.get("authorization") != AGGREGATE_AUTHORIZATION
    ):
        raise P3StageB1AggregateV2Error("raw/science aggregate firewall differs")
    return manifest, lineage, strata, summaries, mechanism, raw_audit


def verify_stage_b1_aggregate_shard_v2(
    path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    output_root_relative: str | os.PathLike[str],
    config: Mapping[str, Any],
    expected_config_sha256: str,
    mechanism_gate: Mapping[str, Any],
    verify_live_cells: bool = True,
    expected_cell_code_seal: Mapping[str, Any] | None = None,
) -> VerifiedStageB1AggregateV2:
    if verify_live_cells and expected_cell_code_seal is None:
        raise P3StageB1AggregateV2Error(
            "live aggregate verification requires the expected cell code seal"
        )
    root = Path(os.path.abspath(os.fspath(path)))
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    output_relative = _canonical_output_relative(output_root_relative)
    config_sha = _sha256(expected_config_sha256, label="expected_config_sha256")
    try:
        snapshot = snapshot_regular_directory(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise P3StageB1AggregateV2Error(f"cannot snapshot B1 v2 aggregate: {root}") from exc
    by_name = {member.path.name: member for member in snapshot.members}
    if set(by_name) != MEMBERS:
        raise P3StageB1AggregateV2Error("aggregate member set differs")
    try:
        manifest, lineage, strata, summaries, mechanism, raw_audit = _parse_local(
            by_name, config=config, output_relative=output_relative, config_sha=config_sha
        )
    except P3StageB1AggregateV2Error:
        raise
    except Exception as exc:
        raise P3StageB1AggregateV2Error(
            "aggregate canonical payload verification failed"
        ) from exc
    if manifest.get("aggregate_code_seal") != build_aggregate_code_seal(repository):
        raise P3StageB1AggregateV2Error("aggregate live code seal differs")
    complete = parse_canonical_json(
        by_name[COMPLETE_FILENAME].data, label=COMPLETE_FILENAME, newline=True
    )
    expected_complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": str(config["protocol_id"]),
        "complete": True,
        "config_sha256": config_sha,
        "cell_count": EXPECTED_CELL_COUNT,
        "episode_count": TOTAL_EPISODE_COUNT,
        "manifest": {"path": MANIFEST_FILENAME, "sha256": by_name[MANIFEST_FILENAME].sha256},
        "payload_files": [
            {"path": name, "sha256": by_name[name].sha256}
            for name in sorted(set(by_name) - {COMPLETE_FILENAME})
        ],
        "mechanism_evidence": {
            "path": MECHANISM_EVIDENCE_FILENAME,
            "sha256": by_name[MECHANISM_EVIDENCE_FILENAME].sha256,
        },
        "raw_numeric_audit": {
            "path": RAW_NUMERIC_AUDIT_FILENAME,
            "sha256": by_name[RAW_NUMERIC_AUDIT_FILENAME].sha256,
            "affects_science": False,
        },
        "atomic_no_replace": True,
        "immutable": True,
        **AGGREGATE_AUTHORIZATION,
    }
    if complete != expected_complete:
        raise P3StageB1AggregateV2Error("aggregate COMPLETE differs")
    if verify_live_cells:
        live = collect_stage_b1_preflight_v2(
            repository_root=repository,
            output_root_relative=output_relative,
            config=config,
            config_sha256=config_sha,
            expected_code_seal=expected_cell_code_seal,
        )
        expected = build_stage_b1_aggregate_payloads_v2(
            live, mechanism_gate=mechanism_gate
        )
        for name in MEMBERS:
            if by_name[name].data != expected[name]:
                raise P3StageB1AggregateV2Error(
                    f"aggregate differs from live rebuild: {name}"
                )
    for collection in (strata, summaries):
        for row in collection:
            metrics = _mapping(row.get("metrics"), label="descriptive metrics")
            if set(metrics) != set(ALL_METRIC_IDS):
                raise P3StageB1AggregateV2Error("descriptive metric set differs")
            for statistic in metrics.values():
                for field in ("mean", "median", "q1", "q3", "minimum", "maximum"):
                    value = statistic.get(field)
                    if value is not None and (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise P3StageB1AggregateV2Error("descriptive value is non-finite")
    statuses = {key: str(value["status"]) for key, value in mechanism["P0_flags"].items()}
    return VerifiedStageB1AggregateV2(
        root,
        by_name[MANIFEST_FILENAME].sha256,
        by_name[COMPLETE_FILENAME].sha256,
        by_name[MECHANISM_EVIDENCE_FILENAME].sha256,
        by_name[RAW_NUMERIC_AUDIT_FILENAME].sha256,
        EXPECTED_CELL_COUNT,
        TOTAL_EPISODE_COUNT,
        len(strata),
        len(summaries),
        statuses,
        False,
        False,
        False,
    )


verify_stage_b1_aggregate_shard = verify_stage_b1_aggregate_shard_v2
descriptive_statistics = _v1.descriptive_statistics


__all__ = [
    "AGGREGATE_AUTHORIZATION", "AGGREGATE_CRITICAL_CODE_PATHS", "ALL_METRIC_IDS",
    "ARTIFACT_TYPE", "CELL_LINEAGE_FILENAME", "COMPLETE_ARTIFACT_TYPE",
    "COMPLETE_FILENAME", "EPISODES_PER_CELL", "EXPECTED_CELL_COUNT",
    "LINEAGE_ARTIFACT_TYPE", "MANIFEST_FILENAME", "MECHANISM_ARTIFACT_TYPE",
    "MECHANISM_EVIDENCE_FILENAME", "MEMBERS", "METRIC_PATHS", "P0_MECHANISM_METRICS",
    "P3StageB1AggregateError", "P3StageB1AggregateV2Error", "RAW_AUDIT_ARTIFACT_TYPE",
    "RAW_NUMERIC_AUDIT_FILENAME", "SMALL_GROUP_METRICS", "STRATIFIED_STATISTICS_FILENAME",
    "STRATUM_ARTIFACT_TYPE", "SUMMARY_ARTIFACT_TYPE", "SUMMARY_STATISTICS_FILENAME",
    "StageB1AggregatePreflight", "StageB1AggregatePreflightV2", "StageB1CellPaths",
    "StageB1CellPathsV2", "TARGET_PRESENCE_VALUES", "TOTAL_EPISODE_COUNT",
    "VerifiedStageB1Aggregate", "VerifiedStageB1AggregateV2", "build_aggregate_code_seal",
    "build_mechanism_evidence", "build_raw_numeric_audit",
    "build_stage_b1_aggregate_payloads", "build_stage_b1_aggregate_payloads_v2",
    "build_stratified_statistics", "build_summary_statistics",
    "collect_stage_b1_preflight", "collect_stage_b1_preflight_v2",
    "descriptive_statistics", "fixed_stage_b1_cells", "fixed_stage_b1_cells_v2",
    "verify_stage_b1_aggregate_shard", "verify_stage_b1_aggregate_shard_v2",
]
