"""CPU-only formal R0 aggregate and science-decision artifact.

This module is the final boundary of the P3 Stage-A R0 phase.  It consumes
only the already completed label-free/outer shards.  It never opens an image,
mask, target, test/validation payload, model, optimizer, or CUDA device.

The public verifier deliberately re-runs the public cell verifiers (including
the label-free live-input seals), rebuilds all ten replicate-evidence records
from the exact 39 x 640 outer records, and recomputes the R0 science decision.
Consequently the aggregate is both byte-canonical and derivationally bound to
the immutable cell evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Final

from analysis.d0_v3_formal_contract import (
    CONDITIONS,
    DATASETS,
    FROZEN_CANDIDATES,
    PROTOCOL_ID,
)
from analysis.d0_v3_label_free_shard import (
    COMPLETE_FILENAME as LABEL_FREE_COMPLETE_FILENAME,
    MANIFEST_FILENAME as LABEL_FREE_MANIFEST_FILENAME,
    PHASE_RECEIPT_FILENAME,
    canonical_json_bytes,
    parse_canonical_json,
    verify_label_free_shard,
)
from analysis.d0_v3_outer_cell_shard import (
    COMPLETE_FILENAME as OUTER_COMPLETE_FILENAME,
    MANIFEST_FILENAME as OUTER_MANIFEST_FILENAME,
    OUTER_ACCESS_RECEIPT_FILENAME,
    OUTER_RECORDS_FILENAME,
    canonical_ordered_json_bytes,
    verify_outer_cell_shard,
)
from analysis.d0_v3_replicate_aggregate import build_replicate_evidence_set
from analysis.d0_v3_science_gate import (
    EPISODES_PER_CANDIDATE_REPLICATE,
    evaluate_stage_a_r0,
    parse_r0_eligibility_receipt,
    parse_replicate_evidence,
)
from tta.d0_secure_io import read_stable_regular_file, snapshot_regular_directory


SCHEMA_VERSION: Final = 3
ARTIFACT_TYPE: Final = "cr_sitta_d0_v3_formal_stage_a_r0_aggregate"
COMPLETE_ARTIFACT_TYPE: Final = (
    "cr_sitta_d0_v3_formal_stage_a_r0_aggregate_complete"
)
LINEAGE_ARTIFACT_TYPE: Final = (
    "cr_sitta_d0_v3_formal_stage_a_r0_cell_lineage"
)
EXPECTED_CELL_COUNT: Final = len(DATASETS) * len(CONDITIONS)
RECORDS_PER_CELL: Final = 64 * len(FROZEN_CANDIDATES)
TOTAL_OUTER_RECORD_COUNT: Final = EXPECTED_CELL_COUNT * RECORDS_PER_CELL

CELL_LINEAGE_FILENAME: Final = "cell_lineage.jsonl"
REPLICATE_EVIDENCE_FILENAME: Final = "r0_replicate_evidence.jsonl"
SCIENCE_DECISION_FILENAME: Final = "science_decision_receipt.json"
MANIFEST_FILENAME: Final = "manifest.json"
COMPLETE_FILENAME: Final = "COMPLETE.json"
MEMBERS: Final = frozenset(
    {
        CELL_LINEAGE_FILENAME,
        REPLICATE_EVIDENCE_FILENAME,
        SCIENCE_DECISION_FILENAME,
        MANIFEST_FILENAME,
        COMPLETE_FILENAME,
    }
)

AGGREGATE_CRITICAL_CODE_PATHS: Final = (
    "scripts/run_d0_v3_formal_stage_a_r0_aggregate.py",
    "analysis/d0_v3_r0_aggregate_shard.py",
    "analysis/d0_v3_replicate_aggregate.py",
    "analysis/d0_v3_science_gate.py",
    "analysis/d0_v3_outer_cell_shard.py",
    "analysis/d0_v3_label_free_shard.py",
    "analysis/d0_v3_formal_contract.py",
    "tta/d0_v3_atomic_shard.py",
    "tta/d0_secure_io.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_LINEAGE_FIELDS = {
    "schema_version",
    "artifact_type",
    "cell_index",
    "dataset",
    "condition",
    "replicate",
    "label_free_shard_path",
    "outer_shard_path",
    "label_free_manifest_sha256",
    "label_free_complete_sha256",
    "label_free_phase_receipt_sha256",
    "outer_manifest_sha256",
    "outer_complete_sha256",
    "outer_access_receipt_sha256",
    "outer_records_sha256",
    "ordered_image_ids_sha256",
    "train_split_sha256",
    "checkpoint_sha256",
    "outer_record_count",
    "live_input_bindings_verified",
    "raw_target_reopened_by_aggregate",
    "stage2_authorized",
}


class D0V3R0AggregateError(ValueError):
    """The fixed R0 grid or aggregate artifact is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class R0CellPaths:
    index: int
    dataset: str
    condition: str
    label_free_path: Path
    outer_path: Path


@dataclass(frozen=True, slots=True)
class R0AggregatePreflight:
    repository_root: Path
    output_root_relative: str
    config_sha256: str
    lineage: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    decision_receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedR0Aggregate:
    path: Path
    manifest_sha256: str
    complete_sha256: str
    science_decision_sha256: str
    cell_count: int
    outer_record_count: int
    evidence_count: int
    eligible_candidate_ids: tuple[str, ...]
    required_followup_replicates: tuple[str, ...]
    formal_stage_a_protocol_complete: bool
    scientific_status: str


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise D0V3R0AggregateError(f"{label} must be lowercase SHA-256")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0V3R0AggregateError(f"{label} must be a mapping")
    return value


def _exact(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label=label)
    if set(mapping) != fields:
        missing = sorted(fields - set(mapping))
        unknown = sorted(set(mapping) - fields)
        raise D0V3R0AggregateError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )
    return mapping


def _repository_relative(root: Path, path: Path, *, label: str) -> str:
    repository = Path(os.path.abspath(os.fspath(root)))
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = absolute.relative_to(repository)
    except ValueError as exc:
        raise D0V3R0AggregateError(f"{label} is outside repository") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise D0V3R0AggregateError(f"{label} is not a canonical repository path")
    return relative.as_posix()


def _canonical_output_relative(value: str | os.PathLike[str]) -> str:
    path = Path(os.fspath(value))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise D0V3R0AggregateError("formal output root is not canonical")
    return path.as_posix()


def fixed_r0_cells(
    repository_root: str | os.PathLike[str],
    output_root_relative: str | os.PathLike[str],
) -> tuple[R0CellPaths, ...]:
    """Return the one permitted R0 input topology in canonical order."""

    repository = Path(os.path.abspath(os.fspath(repository_root)))
    output_relative = _canonical_output_relative(output_root_relative)
    output = repository / output_relative
    cells: list[R0CellPaths] = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            cells.append(
                R0CellPaths(
                    index=len(cells),
                    dataset=dataset,
                    condition=condition,
                    label_free_path=(
                        output
                        / "candidate_phase"
                        / "shards"
                        / "R0"
                        / dataset
                        / condition
                    ),
                    outer_path=(
                        output
                        / "outer_phase"
                        / "shards"
                        / "R0"
                        / dataset
                        / condition
                    ),
                )
            )
    if len(cells) != EXPECTED_CELL_COUNT:
        raise D0V3R0AggregateError("fixed R0 topology is not exactly 39 cells")
    return tuple(cells)


def _strict_ordered_jsonl(data: bytes, *, count: int, label: str) -> list[Mapping[str, Any]]:
    if not data.endswith(b"\n"):
        raise D0V3R0AggregateError(f"{label} is not newline terminated")
    lines = data[:-1].split(b"\n")
    if len(lines) != count or any(not line for line in lines):
        raise D0V3R0AggregateError(
            f"{label} count differs; expected={count}, observed={len(lines)}"
        )

    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise D0V3R0AggregateError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    records: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=unique,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    D0V3R0AggregateError(f"{label}[{index}] contains {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise D0V3R0AggregateError(
                f"{label}[{index}] is not strict UTF-8 JSON"
            ) from exc
        mapping = _mapping(value, label=f"{label}[{index}]")
        if canonical_ordered_json_bytes(mapping) != line:
            raise D0V3R0AggregateError(f"{label}[{index}] is not canonical")
        records.append(mapping)
    return records


def _canonical_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    if isinstance(records, (str, bytes, Mapping)):
        raise D0V3R0AggregateError("JSONL records must be a sequence")
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def build_aggregate_code_seal(
    repository_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Hash the exact aggregate/verifier implementation without importing CUDA."""

    root = Path(os.path.abspath(os.fspath(repository_root)))
    files = [
        {
            "path": relative,
            "sha256": read_stable_regular_file(root / relative).sha256,
        }
        for relative in AGGREGATE_CRITICAL_CODE_PATHS
    ]
    return {
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def _read_verified_cell(
    cell: R0CellPaths,
    *,
    repository_root: Path,
    config_sha256: str,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    """Verify one cell and return only its lineage plus outer records."""

    label = verify_label_free_shard(
        cell.label_free_path,
        expected_config_sha256=config_sha256,
        verify_live_inputs=True,
        repository_root=repository_root,
    )
    outer = verify_outer_cell_shard(
        cell.outer_path,
        repository_root=repository_root,
        label_free_shard_path=cell.label_free_path,
        expected_config_sha256=config_sha256,
    )
    if (
        label.dataset != cell.dataset
        or label.condition != cell.condition
        or label.replicate != "R0"
        or not label.formal
        or label.dry_run
        or label.image_count != 64
        or label.episode_count != RECORDS_PER_CELL
        or outer.dataset != cell.dataset
        or outer.condition != cell.condition
        or outer.replicate != "R0"
        or outer.record_count != RECORDS_PER_CELL
        or outer.label_free_manifest_sha256 != label.manifest_sha256
        or outer.label_free_phase_receipt_sha256 != label.phase_receipt_sha256
        or label.phase_receipt_sha256 is None
    ):
        raise D0V3R0AggregateError(
            f"verified cell binding differs: {cell.dataset}/{cell.condition}"
        )

    label_manifest = read_stable_regular_file(
        cell.label_free_path / LABEL_FREE_MANIFEST_FILENAME
    )
    label_complete = read_stable_regular_file(
        cell.label_free_path / LABEL_FREE_COMPLETE_FILENAME
    )
    phase_receipt = read_stable_regular_file(
        cell.label_free_path / PHASE_RECEIPT_FILENAME
    )
    outer_manifest = read_stable_regular_file(
        cell.outer_path / OUTER_MANIFEST_FILENAME
    )
    outer_complete = read_stable_regular_file(
        cell.outer_path / OUTER_COMPLETE_FILENAME
    )
    outer_access = read_stable_regular_file(
        cell.outer_path / OUTER_ACCESS_RECEIPT_FILENAME
    )
    outer_records = read_stable_regular_file(cell.outer_path / OUTER_RECORDS_FILENAME)
    if (
        label_manifest.sha256 != label.manifest_sha256
        or label_complete.sha256 != label.complete_sha256
        or phase_receipt.sha256 != label.phase_receipt_sha256
        or outer_manifest.sha256 != outer.manifest_sha256
        or outer_complete.sha256 != outer.complete_sha256
        or outer_access.sha256 != outer.outer_access_receipt_sha256
    ):
        raise D0V3R0AggregateError(
            f"cell changed after public verification: {cell.dataset}/{cell.condition}"
        )
    parsed_outer_manifest = parse_canonical_json(
        outer_manifest.data, label="outer manifest", newline=True
    )
    records_binding = _mapping(
        parsed_outer_manifest.get("outer_records"), label="outer manifest records"
    )
    dataset_binding = _mapping(
        parsed_outer_manifest.get("dataset_binding"),
        label="outer manifest dataset binding",
    )
    ordered_ids_sha256 = _sha256(
        parsed_outer_manifest.get("ordered_image_ids_sha256"),
        label="outer ordered image IDs SHA",
    )
    train_split_sha256 = _sha256(
        dataset_binding.get("train_split_sha256"),
        label="outer train split SHA",
    )
    checkpoint_sha256 = _sha256(
        dataset_binding.get("checkpoint_sha256"),
        label="outer checkpoint SHA",
    )
    if (
        records_binding.get("path") != OUTER_RECORDS_FILENAME
        or records_binding.get("sha256") != outer_records.sha256
        or records_binding.get("count") != RECORDS_PER_CELL
    ):
        raise D0V3R0AggregateError("outer record lineage changed after verification")
    records = _strict_ordered_jsonl(
        outer_records.data,
        count=RECORDS_PER_CELL,
        label=f"{cell.dataset}/{cell.condition} outer records",
    )
    lineage = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": LINEAGE_ARTIFACT_TYPE,
        "cell_index": cell.index,
        "dataset": cell.dataset,
        "condition": cell.condition,
        "replicate": "R0",
        "label_free_shard_path": _repository_relative(
            repository_root, cell.label_free_path, label="label-free shard"
        ),
        "outer_shard_path": _repository_relative(
            repository_root, cell.outer_path, label="outer shard"
        ),
        "label_free_manifest_sha256": label.manifest_sha256,
        "label_free_complete_sha256": label.complete_sha256,
        "label_free_phase_receipt_sha256": str(label.phase_receipt_sha256),
        "outer_manifest_sha256": outer.manifest_sha256,
        "outer_complete_sha256": outer.complete_sha256,
        "outer_access_receipt_sha256": outer.outer_access_receipt_sha256,
        "outer_records_sha256": outer_records.sha256,
        "ordered_image_ids_sha256": ordered_ids_sha256,
        "train_split_sha256": train_split_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "outer_record_count": RECORDS_PER_CELL,
        "live_input_bindings_verified": True,
        "raw_target_reopened_by_aggregate": False,
        "stage2_authorized": False,
    }
    return lineage, records


def collect_r0_preflight(
    *,
    repository_root: str | os.PathLike[str],
    output_root_relative: str | os.PathLike[str],
    config_sha256: str,
) -> R0AggregatePreflight:
    """CPU-verify all 39 cells and rebuild the ten exact R0 evidence records.

    This function is read-only.  It creates no aggregate directory, staging
    directory, or marker when a cell is absent or invalid.
    """

    repository = Path(os.path.abspath(os.fspath(repository_root)))
    output_relative = _canonical_output_relative(output_root_relative)
    config_digest = _sha256(config_sha256, label="config_sha256")
    lineage: list[Mapping[str, Any]] = []
    records: list[Mapping[str, Any]] = []
    seen_paths: set[Path] = set()
    for cell in fixed_r0_cells(repository, output_relative):
        for path in (cell.label_free_path, cell.outer_path):
            lexical = Path(os.path.abspath(os.fspath(path)))
            if lexical in seen_paths:
                raise D0V3R0AggregateError(f"duplicate fixed input path: {lexical}")
            seen_paths.add(lexical)
        cell_lineage, cell_records = _read_verified_cell(
            cell,
            repository_root=repository,
            config_sha256=config_digest,
        )
        lineage.append(cell_lineage)
        records.extend(cell_records)
    if len(lineage) != EXPECTED_CELL_COUNT or len(records) != TOTAL_OUTER_RECORD_COUNT:
        raise D0V3R0AggregateError("R0 grid is not the exact 39 x 640 input")
    for dataset in DATASETS:
        dataset_lineage = [value for value in lineage if value["dataset"] == dataset]
        for field in (
            "ordered_image_ids_sha256",
            "train_split_sha256",
            "checkpoint_sha256",
        ):
            if len({str(value[field]) for value in dataset_lineage}) != 1:
                raise D0V3R0AggregateError(
                    f"{dataset} cells do not share one fixed {field}"
                )
    evidence = tuple(build_replicate_evidence_set(records, replicate_id="R0"))
    if len(evidence) != len(FROZEN_CANDIDATES):
        raise D0V3R0AggregateError("R0 evidence is not the frozen ten candidates")
    decision = evaluate_stage_a_r0(evidence, protocol_status="passed").to_receipt()
    # This parser enforces exact candidate partition, early-stop semantics,
    # required R1/R2 topology, and stage2=false.
    parse_r0_eligibility_receipt(decision)
    return R0AggregatePreflight(
        repository_root=repository,
        output_root_relative=output_relative,
        config_sha256=config_digest,
        lineage=tuple(lineage),
        evidence=evidence,
        decision_receipt=decision,
    )


def _candidate_value(candidate: Any) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "optimizer": candidate.optimizer,
        "learning_rate": candidate.learning_rate,
    }


def _validate_lineage(
    records: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
    output_root_relative: str,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(records, (str, bytes, Mapping)) or len(records) != EXPECTED_CELL_COUNT:
        raise D0V3R0AggregateError("cell lineage must contain exactly 39 records")
    expected_cells = fixed_r0_cells(repository_root, output_root_relative)
    observed_paths: set[str] = set()
    result: list[Mapping[str, Any]] = []
    for expected, raw in zip(expected_cells, records, strict=True):
        record = _exact(raw, _LINEAGE_FIELDS, label="cell lineage")
        expected_label_path = _repository_relative(
            repository_root, expected.label_free_path, label="label-free shard"
        )
        expected_outer_path = _repository_relative(
            repository_root, expected.outer_path, label="outer shard"
        )
        if (
            record["schema_version"] != SCHEMA_VERSION
            or record["artifact_type"] != LINEAGE_ARTIFACT_TYPE
            or record["cell_index"] != expected.index
            or record["dataset"] != expected.dataset
            or record["condition"] != expected.condition
            or record["replicate"] != "R0"
            or record["label_free_shard_path"] != expected_label_path
            or record["outer_shard_path"] != expected_outer_path
            or record["outer_record_count"] != RECORDS_PER_CELL
            or record["live_input_bindings_verified"] is not True
            or record["raw_target_reopened_by_aggregate"] is not False
            or record["stage2_authorized"] is not False
        ):
            raise D0V3R0AggregateError("cell lineage identity/order differs")
        for field in (
            "label_free_manifest_sha256",
            "label_free_complete_sha256",
            "label_free_phase_receipt_sha256",
            "outer_manifest_sha256",
            "outer_complete_sha256",
            "outer_access_receipt_sha256",
            "outer_records_sha256",
            "ordered_image_ids_sha256",
            "train_split_sha256",
            "checkpoint_sha256",
        ):
            _sha256(record[field], label=f"cell lineage {field}")
        for field in ("label_free_shard_path", "outer_shard_path"):
            path = str(record[field])
            if path in observed_paths:
                raise D0V3R0AggregateError(f"duplicate cell lineage path: {path}")
            observed_paths.add(path)
        result.append(record)
    if len(observed_paths) != EXPECTED_CELL_COUNT * 2:
        raise D0V3R0AggregateError("cell lineage paths are duplicated")
    for dataset in DATASETS:
        dataset_lineage = [value for value in result if value["dataset"] == dataset]
        for field in (
            "ordered_image_ids_sha256",
            "train_split_sha256",
            "checkpoint_sha256",
        ):
            if len({str(value[field]) for value in dataset_lineage}) != 1:
                raise D0V3R0AggregateError(
                    f"{dataset} lineage does not bind one fixed {field}"
                )
    return tuple(result)


def _validate_evidence(
    values: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(values, (str, bytes, Mapping)) or len(values) != len(FROZEN_CANDIDATES):
        raise D0V3R0AggregateError("R0 evidence must contain exactly ten records")
    result: list[Mapping[str, Any]] = []
    for candidate, value in zip(FROZEN_CANDIDATES, values, strict=True):
        parsed = parse_replicate_evidence(value)
        if parsed.candidate != candidate or parsed.replicate_id != "R0":
            raise D0V3R0AggregateError("R0 evidence candidate/order differs")
        result.append(value)
    return tuple(result)


def _followup_scope(decision: Mapping[str, Any]) -> dict[str, Any]:
    eligible = list(decision["eligible_candidates"])
    required = list(decision["required_followup_replicates"])
    return {
        "policy": (
            "required_eligible_candidates_only"
            if eligible
            else "forbidden_no_r0_eligible_candidate"
        ),
        "eligible_candidates": eligible,
        "required_replicates": required,
        "all_noneligible_candidates_forbidden": True,
        "r1_r2_forbidden": not bool(eligible),
        "stage2_authorized": False,
    }


def build_r0_aggregate_payloads(
    preflight: R0AggregatePreflight,
) -> dict[str, bytes]:
    """Build all five canonical flat-shard members from a verified preflight."""

    if not isinstance(preflight, R0AggregatePreflight):
        raise D0V3R0AggregateError("preflight must be R0AggregatePreflight")
    repository = preflight.repository_root
    output_relative = _canonical_output_relative(preflight.output_root_relative)
    config_sha = _sha256(preflight.config_sha256, label="config_sha256")
    lineage = _validate_lineage(
        preflight.lineage,
        repository_root=repository,
        output_root_relative=output_relative,
    )
    evidence = _validate_evidence(preflight.evidence)
    decision = evaluate_stage_a_r0(evidence, protocol_status="passed").to_receipt()
    parse_r0_eligibility_receipt(decision)
    if dict(preflight.decision_receipt) != decision:
        raise D0V3R0AggregateError("preflight science decision differs on rebuild")

    lineage_bytes = _canonical_jsonl(lineage)
    evidence_bytes = _canonical_jsonl(evidence)
    decision_bytes = canonical_json_bytes(decision, newline=True)
    followup = _followup_scope(decision)
    code_seal = build_aggregate_code_seal(repository)
    evidence_candidate_ids = [candidate.candidate_id for candidate in FROZEN_CANDIDATES]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": PROTOCOL_ID,
        "evaluation_phase": "R0",
        "config_sha256": config_sha,
        "input_scope": {
            "output_root": output_relative,
            "cell_count": EXPECTED_CELL_COUNT,
            "records_per_cell": RECORDS_PER_CELL,
            "outer_record_count": TOTAL_OUTER_RECORD_COUNT,
            "cell_order": "dataset_major_condition_minor",
            "record_order_within_cell": "image_major_candidate_minor",
        },
        "cell_lineage": {
            "path": CELL_LINEAGE_FILENAME,
            "sha256": hashlib.sha256(lineage_bytes).hexdigest(),
            "count": EXPECTED_CELL_COUNT,
            "bundle_sha256": hashlib.sha256(
                canonical_json_bytes(list(lineage))
            ).hexdigest(),
            "public_outer_verifier_passed_count": EXPECTED_CELL_COUNT,
            "live_label_free_binding_passed_count": EXPECTED_CELL_COUNT,
        },
        "replicate_evidence": {
            "path": REPLICATE_EVIDENCE_FILENAME,
            "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            "count": len(FROZEN_CANDIDATES),
            "replicate": "R0",
            "episodes_per_candidate": EPISODES_PER_CANDIDATE_REPLICATE,
            "candidate_ids": evidence_candidate_ids,
        },
        "science_decision": {
            "path": SCIENCE_DECISION_FILENAME,
            "sha256": hashlib.sha256(decision_bytes).hexdigest(),
            "receipt_type": "cr_sitta_d0_v3_stage_a_science_decision",
            "protocol_status": "passed",
            "scientific_status": decision["scientific_status"],
            "formal_stage_a_protocol_complete": decision[
                "formal_stage_a_protocol_complete"
            ],
            "eligible_candidate_ids": [
                value["candidate_id"] for value in decision["eligible_candidates"]
            ],
            "required_followup_replicates": list(
                decision["required_followup_replicates"]
            ),
        },
        "followup_scope": followup,
        "aggregate_code_seal": code_seal,
        "execution": {
            "cpu_only": True,
            "public_outer_cell_verifier_count": EXPECTED_CELL_COUNT,
            "label_free_live_input_verifier_count": EXPECTED_CELL_COUNT,
            "outer_record_parse_count": TOTAL_OUTER_RECORD_COUNT,
            "replicate_evidence_build_count": len(FROZEN_CANDIDATES),
            "science_gate_evaluation_count": 1,
            "model_build_count": 0,
            "optimizer_build_count": 0,
            "gpu_compute_count": 0,
        },
        "data_boundary": {
            "source_train_pilot_outer_records_only": True,
            "raw_train_images_opened_by_aggregate": 0,
            "raw_train_targets_opened_by_aggregate": 0,
            "raw_gt_opened_by_aggregate": 0,
            "test_split_files_opened": 0,
            "test_images_opened": 0,
            "test_masks_opened": 0,
            "test_labels_opened": 0,
            "validation_payloads_opened": 0,
            "outer_target_loader_calls": 0,
            "method_label_accesses": 0,
        },
        "authorization": {
            "source_train_derived": True,
            "r0_scientific_gate_evaluated": True,
            "r0_candidate_filtering_performed": True,
            "scientific_selection_performed": True,
            "paper_result": False,
            "paper_test_result": False,
            "development_test_selected_result": False,
            "stage2_authorized": False,
        },
    }
    manifest_bytes = canonical_json_bytes(manifest, newline=True)
    payload_without_complete = {
        CELL_LINEAGE_FILENAME: lineage_bytes,
        REPLICATE_EVIDENCE_FILENAME: evidence_bytes,
        SCIENCE_DECISION_FILENAME: decision_bytes,
        MANIFEST_FILENAME: manifest_bytes,
    }
    complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "complete": True,
        "r0_grid_complete": True,
        "r0_science_gate_evaluated": True,
        "config_sha256": config_sha,
        "cell_count": EXPECTED_CELL_COUNT,
        "outer_record_count": TOTAL_OUTER_RECORD_COUNT,
        "replicate_evidence_count": len(FROZEN_CANDIDATES),
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "payload_files": [
            {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in sorted(payload_without_complete.items())
        ],
        "science_decision": {
            "path": SCIENCE_DECISION_FILENAME,
            "sha256": hashlib.sha256(decision_bytes).hexdigest(),
            "scientific_status": decision["scientific_status"],
            "formal_stage_a_protocol_complete": decision[
                "formal_stage_a_protocol_complete"
            ],
            "eligible_candidate_ids": [
                value["candidate_id"] for value in decision["eligible_candidates"]
            ],
            "required_followup_replicates": list(
                decision["required_followup_replicates"]
            ),
        },
        "followup_scope": followup,
        "atomic_no_replace": True,
        "immutable": True,
        "paper_result": False,
        "paper_test_result": False,
        "development_test_selected_result": False,
        "stage2_authorized": False,
    }
    return {
        **payload_without_complete,
        COMPLETE_FILENAME: canonical_json_bytes(complete, newline=True),
    }


def _preflight_from_payload(
    *,
    repository_root: Path,
    output_root_relative: str,
    config_sha256: str,
    lineage: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> R0AggregatePreflight:
    validated_lineage = _validate_lineage(
        lineage,
        repository_root=repository_root,
        output_root_relative=output_root_relative,
    )
    validated_evidence = _validate_evidence(evidence)
    decision = evaluate_stage_a_r0(
        validated_evidence, protocol_status="passed"
    ).to_receipt()
    parse_r0_eligibility_receipt(decision)
    return R0AggregatePreflight(
        repository_root=repository_root,
        output_root_relative=output_root_relative,
        config_sha256=config_sha256,
        lineage=validated_lineage,
        evidence=validated_evidence,
        decision_receipt=decision,
    )


def verify_r0_aggregate_shard(
    path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    output_root_relative: str | os.PathLike[str],
    expected_config_sha256: str,
    verify_live_cells: bool = True,
) -> VerifiedR0Aggregate:
    """Public CPU verifier; by default re-derives the artifact from 39 cells."""

    root = Path(os.path.abspath(os.fspath(path)))
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    output_relative = _canonical_output_relative(output_root_relative)
    config_sha = _sha256(expected_config_sha256, label="expected_config_sha256")
    try:
        snapshot = snapshot_regular_directory(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise D0V3R0AggregateError(
            f"cannot securely snapshot R0 aggregate: {root}"
        ) from exc
    by_name = {member.path.name: member for member in snapshot.members}
    if set(by_name) != MEMBERS:
        raise D0V3R0AggregateError("R0 aggregate member set differs")

    lineage = _strict_ordered_jsonl(
        by_name[CELL_LINEAGE_FILENAME].data,
        count=EXPECTED_CELL_COUNT,
        label=CELL_LINEAGE_FILENAME,
    )
    evidence = _strict_ordered_jsonl(
        by_name[REPLICATE_EVIDENCE_FILENAME].data,
        count=len(FROZEN_CANDIDATES),
        label=REPLICATE_EVIDENCE_FILENAME,
    )
    preflight = _preflight_from_payload(
        repository_root=repository,
        output_root_relative=output_relative,
        config_sha256=config_sha,
        lineage=lineage,
        evidence=evidence,
    )
    expected = build_r0_aggregate_payloads(preflight)
    for name in MEMBERS:
        if by_name[name].data != expected[name]:
            raise D0V3R0AggregateError(f"R0 aggregate member differs: {name}")

    if verify_live_cells:
        live = collect_r0_preflight(
            repository_root=repository,
            output_root_relative=output_relative,
            config_sha256=config_sha,
        )
        if (
            tuple(live.lineage) != tuple(preflight.lineage)
            or tuple(live.evidence) != tuple(preflight.evidence)
            or dict(live.decision_receipt) != dict(preflight.decision_receipt)
        ):
            raise D0V3R0AggregateError(
                "live 39-cell rebuild differs from aggregate evidence"
            )

    decision = preflight.decision_receipt
    return VerifiedR0Aggregate(
        path=root,
        manifest_sha256=by_name[MANIFEST_FILENAME].sha256,
        complete_sha256=by_name[COMPLETE_FILENAME].sha256,
        science_decision_sha256=by_name[SCIENCE_DECISION_FILENAME].sha256,
        cell_count=EXPECTED_CELL_COUNT,
        outer_record_count=TOTAL_OUTER_RECORD_COUNT,
        evidence_count=len(FROZEN_CANDIDATES),
        eligible_candidate_ids=tuple(
            value["candidate_id"] for value in decision["eligible_candidates"]
        ),
        required_followup_replicates=tuple(
            decision["required_followup_replicates"]
        ),
        formal_stage_a_protocol_complete=bool(
            decision["formal_stage_a_protocol_complete"]
        ),
        scientific_status=str(decision["scientific_status"]),
    )


__all__ = [
    "AGGREGATE_CRITICAL_CODE_PATHS",
    "ARTIFACT_TYPE",
    "CELL_LINEAGE_FILENAME",
    "COMPLETE_ARTIFACT_TYPE",
    "COMPLETE_FILENAME",
    "D0V3R0AggregateError",
    "EXPECTED_CELL_COUNT",
    "LINEAGE_ARTIFACT_TYPE",
    "MANIFEST_FILENAME",
    "MEMBERS",
    "RECORDS_PER_CELL",
    "REPLICATE_EVIDENCE_FILENAME",
    "R0AggregatePreflight",
    "R0CellPaths",
    "SCIENCE_DECISION_FILENAME",
    "TOTAL_OUTER_RECORD_COUNT",
    "VerifiedR0Aggregate",
    "build_aggregate_code_seal",
    "build_r0_aggregate_payloads",
    "collect_r0_preflight",
    "fixed_r0_cells",
    "verify_r0_aggregate_shard",
]
