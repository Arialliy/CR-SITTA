"""Verified formal Stage-1 scientific-gate artifact chain for Stage 2.

The Stage-2 launcher must not treat a receipt and a digest supplied by the same
caller as authority.  This module starts from an independently trusted frozen
configuration byte identity and follows one closed chain:

``frozen config -> aggregate manifest -> gate/records/diagnostics/receipt``.

Every filesystem input is read without following symlinks.  The aggregate is a
flat, exact-member directory, every JSON input is canonical, the selector is
rerun from the bound Stage-1 evidence, and every input is snapshotted again
before a verified capability is returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Final

import yaml

from tta.binary_tent_ss_calibration_selector_v3 import (
    ALL_CANDIDATES,
    CELLS_PER_RUN,
    FORMAL_FROZEN_MODE,
    SELECTOR_PROTOCOL_ID,
    Candidate,
    CandidateDiagnosticEvidence,
    ScientificGateSpec,
    select_stage1_candidates,
    validate_stage1_scientific_receipt,
)
from tta.d0_secure_io import (
    StableDirectorySnapshot,
    StableFileSnapshot,
    read_stable_regular_file,
    snapshot_regular_directory,
)


CONFIG_PROTOCOL_ID: Final = "cr-sitta-binary-tent-ss-calibration-v3"
AUTHORIZATION_CONTRACT_ID: Final = "cr-sitta-stage1-formal-gate-artifact-v1"
FORMAL_STATUS: Final = "formal_frozen"
BLOCKED_STATUS: Final = "blocked_unresolved_formal_gate"
AGGREGATE_ARTIFACT_TYPE: Final = (
    "binary_tent_ss_stage1_scientific_gate_aggregate_v3"
)
COMPLETE_ARTIFACT_TYPE: Final = (
    "binary_tent_ss_stage1_scientific_gate_aggregate_complete_v3"
)
GATE_MANIFEST_ARTIFACT_TYPE: Final = (
    "binary_tent_ss_frozen_scientific_gate_manifest_v1"
)
RECEIPT_FILENAME: Final = "stage1_ss_scientific_selection_receipt.json"
MANIFEST_FILENAME: Final = "artifact_manifest.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CANONICAL_SEPARATORS = (",", ":")


class Stage1GateArtifactError(ValueError):
    """A trusted configuration or formal Stage-1 artifact failed closed."""


class Stage1GateNotFormalError(Stage1GateArtifactError):
    """The frozen configuration explicitly has no authorizing formal gate."""


@dataclass(frozen=True, slots=True)
class FrozenGateConfigBinding:
    """Independent trust anchor for a frozen authorization configuration."""

    path: Path
    expected_sha256: str
    project_root: Path


@dataclass(frozen=True, slots=True)
class VerifiedFormalStage1Gate:
    """Read-only result of recomputing the complete formal evidence chain."""

    config_path: Path
    config_sha256: str
    aggregate_manifest_path: Path
    aggregate_manifest_sha256: str
    frozen_gate_manifest_path: Path
    frozen_gate_manifest_sha256: str
    stage1_records_path: Path
    stage1_records_sha256: str
    diagnostic_evidence_path: Path
    diagnostic_evidence_sha256: str
    receipt_path: Path
    receipt_sha256: str
    receipt_byte_count: int
    receipt: Mapping[str, Any]
    selected_for_stage2: tuple[Candidate, ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the only accepted durable JSON representation."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=_CANONICAL_SEPARATORS,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage1GateArtifactError("value is not canonical-JSON serializable") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage1GateArtifactError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise Stage1GateArtifactError(f"non-finite JSON number is forbidden: {value}")


def _decode_json(data: bytes, label: str, *, canonical: bool = True) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except Stage1GateArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise Stage1GateArtifactError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise Stage1GateArtifactError(f"JSON root must be an object: {label}")
    if canonical and data != canonical_json_bytes(value):
        raise Stage1GateArtifactError(
            f"JSON is not canonical with exactly one trailing newline: {label}"
        )
    return value


def _decode_jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    if not data or not data.endswith(b"\n"):
        raise Stage1GateArtifactError(f"JSONL must end in exactly one record newline: {label}")
    lines = data.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if line in {b"", b"\n"}:
            raise Stage1GateArtifactError(f"blank JSONL row: {label}[{index}]")
        value = _decode_json(line, f"{label}[{index}]", canonical=True)
        records.append(value)
    return records


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise Stage1GateArtifactError(f"{label} must be a string-key mapping")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    result = _mapping(value, label)
    missing = sorted(expected - set(result))
    unknown = sorted(set(result) - expected)
    if missing or unknown:
        raise Stage1GateArtifactError(
            f"{label} schema is not exact; missing={missing}, unknown={unknown}"
        )
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Stage1GateArtifactError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage1GateArtifactError(f"{label} must be an integer >= {minimum}")
    return value


def _fraction(value: Any, label: str) -> Fraction:
    record = _exact_keys(value, {"numerator", "denominator"}, label)
    numerator = record["numerator"]
    denominator = record["denominator"]
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise Stage1GateArtifactError(f"{label}.numerator must be an integer")
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise Stage1GateArtifactError(f"{label}.denominator must be a positive integer")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise Stage1GateArtifactError(f"{label} must be stored in reduced form")
    return result


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _safe_relative_file(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Stage1GateArtifactError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Stage1GateArtifactError(f"{label} must be a safe project-relative path")
    return _lexical_absolute(project_root / relative)


def _simple_filename(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise Stage1GateArtifactError(f"{label} must be one simple filename")
    return value


def _candidate(value: Any, label: str) -> Candidate:
    record = _exact_keys(
        value,
        {"optimizer", "learning_rate", "learning_rate_decimal"},
        label,
    )
    try:
        candidate = Candidate.from_values(record["optimizer"], record["learning_rate"])
    except (TypeError, ValueError) as exc:
        raise Stage1GateArtifactError(f"invalid candidate: {label}") from exc
    if record["learning_rate_decimal"] != candidate.to_dict()["learning_rate_decimal"]:
        raise Stage1GateArtifactError(
            f"{label}.learning_rate_decimal is inconsistent"
        )
    return candidate


def _candidate_order(value: Any, label: str) -> tuple[Candidate, ...]:
    if not isinstance(value, list):
        raise Stage1GateArtifactError(f"{label} must be a list")
    result = tuple(_candidate(item, f"{label}[{index}]") for index, item in enumerate(value))
    if result != tuple(ALL_CANDIDATES):
        raise Stage1GateArtifactError(f"{label} differs from the ten frozen candidates")
    return result


def _load_yaml(snapshot: StableFileSnapshot) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise Stage1GateArtifactError("frozen authorization config is invalid YAML") from exc
    return _mapping(value, "frozen authorization config")


def _snapshot_member(
    directory: StableDirectorySnapshot, filename: str, label: str
) -> StableFileSnapshot:
    try:
        return directory.member(filename)
    except KeyError as exc:
        raise Stage1GateArtifactError(f"aggregate is missing {label}: {filename}") from exc


def _file_record(value: Any, label: str) -> Mapping[str, Any]:
    result = _exact_keys(
        value, {"filename", "sha256", "size_bytes", "line_count"}, label
    )
    _simple_filename(result["filename"], f"{label}.filename")
    _sha256(result["sha256"], f"{label}.sha256")
    _integer(result["size_bytes"], f"{label}.size_bytes", minimum=1)
    _integer(result["line_count"], f"{label}.line_count", minimum=1)
    return result


def _verify_bound_member(
    directory: StableDirectorySnapshot,
    record: Mapping[str, Any],
    label: str,
) -> StableFileSnapshot:
    filename = _simple_filename(record["filename"], f"{label}.filename")
    snapshot = _snapshot_member(directory, filename, label)
    if (
        snapshot.sha256 != record["sha256"]
        or snapshot.size_bytes != record["size_bytes"]
        or len(snapshot.data.splitlines()) != record["line_count"]
    ):
        raise Stage1GateArtifactError(f"aggregate binding failed: {label}")
    return snapshot


def _gate_spec_from_manifest(
    value: Mapping[str, Any],
    *,
    gate_sha256: str,
    records_sha256: str,
    diagnostics_sha256: str,
) -> tuple[ScientificGateSpec, int]:
    gate = _exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "contract_id",
            "profile_id",
            "selector_protocol_id",
            "preregistered_before_evidence",
            "threshold_source",
            "candidate_order",
            "top_k_after_filter",
            "allow_fewer_than_top_k",
            "thresholds",
            "evidence",
        },
        "frozen gate manifest",
    )
    if (
        gate["schema_version"] != 1
        or gate["artifact_type"] != GATE_MANIFEST_ARTIFACT_TYPE
        or gate["contract_id"] != AUTHORIZATION_CONTRACT_ID
        or gate["selector_protocol_id"] != SELECTOR_PROTOCOL_ID
        or gate["preregistered_before_evidence"] is not True
        or not isinstance(gate["profile_id"], str)
        or not gate["profile_id"]
        or not isinstance(gate["threshold_source"], str)
        or not gate["threshold_source"]
    ):
        raise Stage1GateArtifactError("frozen gate manifest identity/provenance failed")
    _candidate_order(gate["candidate_order"], "frozen gate candidate_order")
    top_k = _integer(gate["top_k_after_filter"], "gate top_k", minimum=1)
    if top_k > 3 or not isinstance(gate["allow_fewer_than_top_k"], bool):
        raise Stage1GateArtifactError("frozen gate ranking policy is invalid")

    thresholds = _exact_keys(
        gate["thresholds"],
        {
            "minimum_positive_cells",
            "minimum_positive_corruption_families",
            "minimum_positive_datasets",
            "clean_iou_equivalence_margin",
            "max_pd_drop",
            "max_fa_increase_per_million_pixels",
            "minimum_parameter_update_fraction_above_null",
            "minimum_functional_change_fraction_above_null",
            "minimum_objective_decrease_fraction",
        },
        "frozen gate thresholds",
    )
    evidence = _exact_keys(
        gate["evidence"],
        {
            "stage1_records_sha256",
            "stage1_cell_record_count",
            "diagnostic_evidence_sha256",
            "diagnostic_candidate_count",
            "evaluated_episodes_per_candidate",
        },
        "frozen gate evidence",
    )
    if (
        evidence["stage1_records_sha256"] != records_sha256
        or evidence["diagnostic_evidence_sha256"] != diagnostics_sha256
        or evidence["stage1_cell_record_count"] != len(ALL_CANDIDATES) * CELLS_PER_RUN
        or evidence["diagnostic_candidate_count"] != len(ALL_CANDIDATES)
    ):
        raise Stage1GateArtifactError("frozen gate evidence hashes/counts differ")
    evaluated = _integer(
        evidence["evaluated_episodes_per_candidate"],
        "evaluated_episodes_per_candidate",
        minimum=1,
    )
    try:
        spec = ScientificGateSpec(
            profile_id=gate["profile_id"],
            mode=FORMAL_FROZEN_MODE,
            min_positive_cells=_integer(
                thresholds["minimum_positive_cells"], "minimum_positive_cells", minimum=1
            ),
            min_positive_corruption_families=_integer(
                thresholds["minimum_positive_corruption_families"],
                "minimum_positive_corruption_families",
                minimum=1,
            ),
            min_positive_datasets=_integer(
                thresholds["minimum_positive_datasets"],
                "minimum_positive_datasets",
                minimum=1,
            ),
            clean_iou_equivalence_margin=_fraction(
                thresholds["clean_iou_equivalence_margin"],
                "clean_iou_equivalence_margin",
            ),
            max_pd_drop=_fraction(thresholds["max_pd_drop"], "max_pd_drop"),
            max_fa_increase=_fraction(
                thresholds["max_fa_increase_per_million_pixels"],
                "max_fa_increase_per_million_pixels",
            ),
            min_parameter_update_fraction=_fraction(
                thresholds["minimum_parameter_update_fraction_above_null"],
                "minimum_parameter_update_fraction_above_null",
            ),
            min_functional_change_fraction=_fraction(
                thresholds["minimum_functional_change_fraction_above_null"],
                "minimum_functional_change_fraction_above_null",
            ),
            min_objective_decrease_fraction=_fraction(
                thresholds["minimum_objective_decrease_fraction"],
                "minimum_objective_decrease_fraction",
            ),
            top_k_after_filter=top_k,
            allow_fewer_than_top_k=gate["allow_fewer_than_top_k"],
            threshold_source=gate["threshold_source"],
            frozen_gate_manifest_sha256=gate_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise Stage1GateArtifactError("frozen gate thresholds are invalid") from exc
    return spec, evaluated


def _diagnostics(
    data: bytes, *, expected_evaluated_episodes: int
) -> tuple[CandidateDiagnosticEvidence, ...]:
    rows = _decode_jsonl(data, "diagnostic evidence")
    if len(rows) != len(ALL_CANDIDATES):
        raise Stage1GateArtifactError("diagnostic evidence must contain ten rows")
    values: list[CandidateDiagnosticEvidence] = []
    expected_keys = {
        "candidate",
        "parameter_update_episodes",
        "parameter_update_evaluated_episodes",
        "functional_change_episodes",
        "functional_change_evaluated_episodes",
        "objective_decrease_episodes",
        "objective_evaluated_episodes",
    }
    for index, row in enumerate(rows):
        record = _exact_keys(row, expected_keys, f"diagnostic evidence[{index}]")
        candidate = _candidate(record["candidate"], f"diagnostic evidence[{index}].candidate")
        if candidate != ALL_CANDIDATES[index]:
            raise Stage1GateArtifactError("diagnostic candidate order/binding differs")
        for field in (
            "parameter_update_evaluated_episodes",
            "functional_change_evaluated_episodes",
            "objective_evaluated_episodes",
        ):
            if record[field] != expected_evaluated_episodes:
                raise Stage1GateArtifactError(
                    f"diagnostic evidence[{index}].{field} differs from gate manifest"
                )
        try:
            values.append(
                CandidateDiagnosticEvidence(
                    candidate=candidate,
                    parameter_update_episodes=record["parameter_update_episodes"],
                    parameter_update_evaluated_episodes=record[
                        "parameter_update_evaluated_episodes"
                    ],
                    functional_change_episodes=record["functional_change_episodes"],
                    functional_change_evaluated_episodes=record[
                        "functional_change_evaluated_episodes"
                    ],
                    objective_decrease_episodes=record["objective_decrease_episodes"],
                    objective_evaluated_episodes=record[
                        "objective_evaluated_episodes"
                    ],
                )
            )
        except (TypeError, ValueError) as exc:
            raise Stage1GateArtifactError(
                f"invalid diagnostic evidence row {index}"
            ) from exc
    return tuple(values)


def verify_formal_stage1_gate(
    binding: FrozenGateConfigBinding,
) -> VerifiedFormalStage1Gate:
    """Recompute and verify a formal Stage-1 gate from an independent config hash."""

    if not isinstance(binding, FrozenGateConfigBinding):
        raise Stage1GateArtifactError("binding must be FrozenGateConfigBinding")
    expected_config_sha = _sha256(binding.expected_sha256, "trusted config SHA-256")
    project_root = _lexical_absolute(binding.project_root)
    config_path = _lexical_absolute(binding.path)
    config_before = read_stable_regular_file(config_path)
    if config_before.sha256 != expected_config_sha:
        raise Stage1GateArtifactError("frozen authorization config SHA-256 mismatch")
    config = _load_yaml(config_before)
    if config.get("schema_version") != 3 or config.get("protocol_id") != CONFIG_PROTOCOL_ID:
        raise Stage1GateArtifactError("unexpected frozen authorization config identity")

    selector_config = _exact_keys(
        config.get("selector_v3"),
        {
            "path",
            "sha256",
            "protocol_id",
            "pure_cpu",
            "exact_rational_arithmetic_from_integer_counts",
            "filter_before_ranking",
            "top_k_after_filter",
            "allow_fewer_than_top_k",
        },
        "selector_v3",
    )
    if (
        selector_config["protocol_id"] != SELECTOR_PROTOCOL_ID
        or selector_config["pure_cpu"] is not True
        or selector_config["exact_rational_arithmetic_from_integer_counts"] is not True
        or selector_config["filter_before_ranking"] is not True
    ):
        raise Stage1GateArtifactError("selector_v3 frozen policy differs")
    selector_path = _safe_relative_file(
        project_root, selector_config["path"], "selector_v3.path"
    )
    selector_before = read_stable_regular_file(selector_path)
    if selector_before.sha256 != _sha256(selector_config["sha256"], "selector_v3.sha256"):
        raise Stage1GateArtifactError("selector_v3 source SHA-256 mismatch")

    authorization = _exact_keys(
        config.get("stage2_authorization"),
        {"schema_version", "contract_id", "status", "aggregate_manifest"},
        "stage2_authorization",
    )
    if authorization["schema_version"] != 1 or authorization["contract_id"] != AUTHORIZATION_CONTRACT_ID:
        raise Stage1GateArtifactError("stage2_authorization contract identity differs")
    if authorization["status"] != FORMAL_STATUS:
        if authorization["status"] == BLOCKED_STATUS and authorization["aggregate_manifest"] is None:
            raise Stage1GateNotFormalError(
                "the frozen v3 configuration has unresolved formal thresholds; Stage 2 is forbidden"
            )
        raise Stage1GateArtifactError("stage2_authorization status is invalid")

    aggregate_binding = _exact_keys(
        authorization["aggregate_manifest"], {"path", "sha256"}, "aggregate_manifest"
    )
    manifest_path = _safe_relative_file(
        project_root, aggregate_binding["path"], "aggregate_manifest.path"
    )
    if manifest_path.name != MANIFEST_FILENAME:
        raise Stage1GateArtifactError("aggregate manifest must use artifact_manifest.json")
    aggregate_before = snapshot_regular_directory(manifest_path.parent)
    manifest_snapshot = _snapshot_member(
        aggregate_before, MANIFEST_FILENAME, "aggregate manifest"
    )
    if manifest_snapshot.sha256 != _sha256(
        aggregate_binding["sha256"], "aggregate_manifest.sha256"
    ):
        raise Stage1GateArtifactError("aggregate manifest SHA-256 mismatch")
    manifest = _decode_json(manifest_snapshot.data, "aggregate manifest")
    manifest = _exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "artifact_complete",
            "formal_fully_frozen_gate",
            "selector_protocol_id",
            "selector_sha256",
            "candidate_order",
            "stage1_cell_record_count",
            "diagnostic_candidate_count",
            "completion_filename",
            "files",
        },
        "aggregate manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_type"] != AGGREGATE_ARTIFACT_TYPE
        or manifest["artifact_complete"] is not True
        or manifest["formal_fully_frozen_gate"] is not True
        or manifest["selector_protocol_id"] != SELECTOR_PROTOCOL_ID
        or manifest["selector_sha256"] != selector_before.sha256
        or manifest["stage1_cell_record_count"] != len(ALL_CANDIDATES) * CELLS_PER_RUN
        or manifest["diagnostic_candidate_count"] != len(ALL_CANDIDATES)
    ):
        raise Stage1GateArtifactError("aggregate manifest protocol/count contract failed")
    _candidate_order(manifest["candidate_order"], "aggregate candidate_order")
    completion_filename = _simple_filename(
        manifest["completion_filename"], "aggregate completion_filename"
    )
    files = _exact_keys(
        manifest["files"],
        {"frozen_gate_manifest", "stage1_records", "diagnostic_evidence", "scientific_receipt"},
        "aggregate manifest.files",
    )
    records_binding = _file_record(files["stage1_records"], "stage1_records")
    diagnostics_binding = _file_record(files["diagnostic_evidence"], "diagnostic_evidence")
    gate_binding = _file_record(files["frozen_gate_manifest"], "frozen_gate_manifest")
    receipt_binding = _file_record(files["scientific_receipt"], "scientific_receipt")
    if receipt_binding["filename"] != RECEIPT_FILENAME:
        raise Stage1GateArtifactError("scientific receipt filename differs")
    expected_names = {
        MANIFEST_FILENAME,
        completion_filename,
        *(record["filename"] for record in (
            records_binding,
            diagnostics_binding,
            gate_binding,
            receipt_binding,
        )),
    }
    if set(aggregate_before.member_names) != expected_names:
        raise Stage1GateArtifactError("formal aggregate member set is not exact")

    records_snapshot = _verify_bound_member(
        aggregate_before, records_binding, "stage1_records"
    )
    diagnostics_snapshot = _verify_bound_member(
        aggregate_before, diagnostics_binding, "diagnostic_evidence"
    )
    gate_snapshot = _verify_bound_member(
        aggregate_before, gate_binding, "frozen_gate_manifest"
    )
    receipt_snapshot = _verify_bound_member(
        aggregate_before, receipt_binding, "scientific_receipt"
    )
    if records_binding["line_count"] != len(ALL_CANDIDATES) * CELLS_PER_RUN:
        raise Stage1GateArtifactError("Stage1 record line count must be exactly 390")
    if diagnostics_binding["line_count"] != len(ALL_CANDIDATES):
        raise Stage1GateArtifactError("diagnostic line count must be exactly ten")
    if gate_binding["line_count"] != 1 or receipt_binding["line_count"] != 1:
        raise Stage1GateArtifactError("gate manifest and receipt must each occupy one line")

    complete_snapshot = _snapshot_member(
        aggregate_before, completion_filename, "aggregate completion"
    )
    complete = _exact_keys(
        _decode_json(complete_snapshot.data, "aggregate completion"),
        {
            "schema_version",
            "artifact_type",
            "complete",
            "formal_fully_frozen_gate",
            "manifest_sha256",
            "receipt_sha256",
            "stage1_cell_record_count",
            "diagnostic_candidate_count",
        },
        "aggregate completion",
    )
    if (
        complete["schema_version"] != 1
        or complete["artifact_type"] != COMPLETE_ARTIFACT_TYPE
        or complete["complete"] is not True
        or complete["formal_fully_frozen_gate"] is not True
        or complete["manifest_sha256"] != manifest_snapshot.sha256
        or complete["receipt_sha256"] != receipt_snapshot.sha256
        or complete["stage1_cell_record_count"] != len(ALL_CANDIDATES) * CELLS_PER_RUN
        or complete["diagnostic_candidate_count"] != len(ALL_CANDIDATES)
    ):
        raise Stage1GateArtifactError("aggregate completion contract failed")

    gate = _decode_json(gate_snapshot.data, "frozen gate manifest")
    gate_spec, evaluated_episodes = _gate_spec_from_manifest(
        gate,
        gate_sha256=gate_snapshot.sha256,
        records_sha256=records_snapshot.sha256,
        diagnostics_sha256=diagnostics_snapshot.sha256,
    )
    if (
        gate_spec.top_k_after_filter != selector_config["top_k_after_filter"]
        or gate_spec.allow_fewer_than_top_k
        is not selector_config["allow_fewer_than_top_k"]
    ):
        raise Stage1GateArtifactError("gate/selector ranking policy differs")
    records = _decode_jsonl(records_snapshot.data, "Stage1 records")
    if len(records) != len(ALL_CANDIDATES) * CELLS_PER_RUN:
        raise Stage1GateArtifactError("Stage1 evidence must contain 390 cell records")
    diagnostics = _diagnostics(
        diagnostics_snapshot.data,
        expected_evaluated_episodes=evaluated_episodes,
    )
    try:
        recomputed = select_stage1_candidates(records, diagnostics, gate_spec)
        selected = tuple(validate_stage1_scientific_receipt(recomputed))
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage1GateArtifactError("formal selector recomputation failed") from exc
    expected_receipt_bytes = canonical_json_bytes(recomputed)
    if receipt_snapshot.data != expected_receipt_bytes:
        raise Stage1GateArtifactError(
            "published scientific receipt is not byte-identical to selector recomputation"
        )
    published_receipt = _decode_json(receipt_snapshot.data, "scientific receipt")
    if published_receipt != recomputed:
        raise Stage1GateArtifactError("published scientific receipt value differs")

    # Close every input read/publish race before returning an authority-bearing
    # object.  Dataclass equality includes inode, timestamps, sizes and bytes.
    config_after = read_stable_regular_file(config_path)
    selector_after = read_stable_regular_file(selector_path)
    aggregate_after = snapshot_regular_directory(manifest_path.parent)
    if config_after != config_before:
        raise Stage1GateArtifactError("frozen authorization config changed during verification")
    if selector_after != selector_before:
        raise Stage1GateArtifactError("selector source changed during verification")
    if aggregate_after != aggregate_before:
        raise Stage1GateArtifactError("formal Stage1 aggregate changed during verification")

    return VerifiedFormalStage1Gate(
        config_path=config_path,
        config_sha256=config_before.sha256,
        aggregate_manifest_path=manifest_path,
        aggregate_manifest_sha256=manifest_snapshot.sha256,
        frozen_gate_manifest_path=manifest_path.parent / gate_binding["filename"],
        frozen_gate_manifest_sha256=gate_snapshot.sha256,
        stage1_records_path=manifest_path.parent / records_binding["filename"],
        stage1_records_sha256=records_snapshot.sha256,
        diagnostic_evidence_path=manifest_path.parent / diagnostics_binding["filename"],
        diagnostic_evidence_sha256=diagnostics_snapshot.sha256,
        receipt_path=manifest_path.parent / receipt_binding["filename"],
        receipt_sha256=receipt_snapshot.sha256,
        receipt_byte_count=receipt_snapshot.size_bytes,
        receipt=published_receipt,
        selected_for_stage2=selected,
    )


__all__ = [
    "AGGREGATE_ARTIFACT_TYPE",
    "AUTHORIZATION_CONTRACT_ID",
    "BLOCKED_STATUS",
    "COMPLETE_ARTIFACT_TYPE",
    "FORMAL_STATUS",
    "FrozenGateConfigBinding",
    "GATE_MANIFEST_ARTIFACT_TYPE",
    "MANIFEST_FILENAME",
    "RECEIPT_FILENAME",
    "Stage1GateArtifactError",
    "Stage1GateNotFormalError",
    "VerifiedFormalStage1Gate",
    "canonical_json_bytes",
    "verify_formal_stage1_gate",
]
