#!/usr/bin/env python3
"""Freeze the Binary-TENT-SS v2 Stage-1 negative calibration result.

This is deliberately a copy-only archival tool.  It never edits or removes the
v2 source artifacts, it refuses to archive a published Stage-2/final tree, and
it labels every generated decision artifact as *not* a paper result.

The default mode creates the archive once and atomically publishes it.  A
second identical invocation verifies the existing archive instead of
overwriting it.  ``--dry-run`` performs all source validation without writing;
``--verify-only`` verifies an already-published archive without consulting the
mutable source tree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("results/binary_tent/ss_calibration_v2")
DEFAULT_DESTINATION = Path(
    "results/binary_tent/ss_calibration_v2_negative_archive"
)

EXPECTED_STAGE1_CANDIDATES = 10
EXPECTED_STAGE1_CELLS = 390
EXPECTED_STAGE1_EPISODES = 24_960
EXPECTED_CELLS_PER_CANDIDATE = 39
EXPECTED_EPISODES_PER_CANDIDATE = 2_496

FROZEN_CONFIG_PATHS = (
    Path("configs/binary_tent_ss_calibration_v2.yaml"),
    Path("configs/binary_tent_ss_calibration_execution_v2.yaml"),
    Path("configs/binary_tent_ss_calibration_cache_v2.yaml"),
    Path("configs/binary_tent_ss_calibration_v3.yaml"),
    Path("configs/tta_train_side_calibration_pilot_v2.yaml"),
    Path("configs/retrain_fixed_splits.yaml"),
    Path("corruptions/severity_tables.yaml"),
)

DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
SCIENTIFIC_REPLAY_RECEIPT_FILENAME = (
    "stage1_ss_scientific_selection_receipt.json"
)


class NegativeArchiveError(RuntimeError):
    """Raised when archival cannot proceed without weakening the evidence."""


@dataclass(frozen=True)
class CopyEntry:
    """One byte-exact source-to-archive copy operation."""

    source: Path
    source_project_relative: str
    archive_relative: str
    sha256: str
    size_bytes: int
    category: str

    def public_record(self) -> dict[str, Any]:
        return {
            "archive_path": self.archive_relative,
            "category": self.category,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_path": self.source_project_relative,
        }


@dataclass(frozen=True)
class Stage1Evidence:
    """Validated facts extracted from the immutable v2 Stage-1 tree."""

    record_count: int
    episode_count: int
    candidate_count: int
    aggregate_manifest_sha256: str
    receipt_sha256: str
    runtime_seal_sha256: str
    strictly_positive_macro_iou_candidates: tuple[dict[str, Any], ...]
    nonpositive_macro_iou_candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ScientificReplayEvidence:
    """Optional, independently generated v3 retrospective gate receipt."""

    status: str
    entry: CopyEntry | None
    receipt_type: str | None


@dataclass(frozen=True)
class ArchivePlan:
    """A fully validated and content-addressed archive plan."""

    project_root: Path
    source_root: Path
    destination: Path
    entries: tuple[CopyEntry, ...]
    stage1: Stage1Evidence
    scientific_replay: ScientificReplayEvidence
    source_inventory_sha256: str
    stage2_logs: tuple[CopyEntry, ...]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NegativeArchiveError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise NegativeArchiveError(f"JSON root must be an object: {path}")
    return value


def _resolve_cli_path(project_root: Path, value: Path) -> Path:
    path = (value if value.is_absolute() else project_root / value).absolute()
    if path.is_symlink():
        raise NegativeArchiveError(f"symbolic-link CLI path is forbidden: {path}")
    return path.resolve()


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise NegativeArchiveError(
            f"path is outside the project root: {path}"
        ) from error
    text = relative.as_posix()
    if not text or "\n" in text or "\r" in text:
        raise NegativeArchiveError(f"unsafe project-relative path: {text!r}")
    return text


def _require_regular_file(path: Path, project_root: Path) -> str:
    relative = _project_relative(path, project_root)
    if path.is_symlink():
        raise NegativeArchiveError(f"symbolic-link input is forbidden: {relative}")
    if not path.is_file():
        raise NegativeArchiveError(f"required regular file is missing: {relative}")
    return relative


def _regular_tree_files(directory: Path, project_root: Path) -> tuple[Path, ...]:
    relative_directory = _project_relative(directory, project_root)
    if directory.is_symlink() or not directory.is_dir():
        raise NegativeArchiveError(
            f"required real directory is missing: {relative_directory}"
        )

    files: list[Path] = []
    for current, directory_names, file_names in os.walk(
        directory, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        for name in tuple(directory_names):
            child = current_path / name
            if child.is_symlink():
                raise NegativeArchiveError(
                    "symbolic-link directory is forbidden: "
                    f"{_project_relative(child, project_root)}"
                )
        for name in file_names:
            child = current_path / name
            _require_regular_file(child, project_root)
            files.append(child)
    return tuple(sorted(files, key=lambda item: item.relative_to(directory).as_posix()))


def _verify_listed_artifact_files(directory: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise NegativeArchiveError(f"artifact manifest has no file ledger: {directory}")
    for name, expected in files.items():
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise NegativeArchiveError(
                f"artifact manifest has unsafe member name in {directory}: {name!r}"
            )
        if not isinstance(expected, Mapping):
            raise NegativeArchiveError(
                f"artifact manifest member is not an object: {directory / name}"
            )
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise NegativeArchiveError(f"manifest member is missing: {path}")
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if expected.get("sha256") != actual_hash or expected.get("bytes") != actual_size:
            raise NegativeArchiveError(f"artifact manifest mismatch: {path}")


def _validate_complete_pair(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "artifact_manifest.json"
    complete_path = directory / "COMPLETE.json"
    manifest = _load_json_object(manifest_path)
    complete = _load_json_object(complete_path)
    _verify_listed_artifact_files(directory, manifest)
    if complete.get("complete") is not True:
        raise NegativeArchiveError(f"incomplete Stage-1 artifact: {directory}")
    if complete.get("artifact_manifest_sha256") != sha256_file(manifest_path):
        raise NegativeArchiveError(
            f"COMPLETE.json does not bind artifact_manifest.json: {directory}"
        )
    for label, value in (("manifest", manifest), ("complete", complete)):
        scope = value.get("scope")
        if not isinstance(scope, Mapping) or scope.get("paper_result") is not False:
            raise NegativeArchiveError(
                f"{label} is not explicitly non-paper scope: {directory}"
            )
    return manifest, complete


def _iter_json_lines(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise NegativeArchiveError(f"cannot open JSONL artifact: {path}") from error
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise NegativeArchiveError(
                    f"blank JSONL record at {path}:{line_number}"
                )
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise NegativeArchiveError(
                    f"invalid JSONL record at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise NegativeArchiveError(
                    f"JSONL record is not an object at {path}:{line_number}"
                )
            yield line_number, value


def _validate_record_file(path: Path, *, expected_count: int) -> int:
    count = 0
    episode_count = 0
    unique_cells: set[tuple[str, str, str, int, str]] = set()
    for line_number, record in _iter_json_lines(path):
        count += 1
        if record.get("stage") != 1:
            raise NegativeArchiveError(
                f"non-Stage-1 record at {path}:{line_number}"
            )
        image_count = record.get("image_count")
        if image_count != 64:
            raise NegativeArchiveError(
                f"unexpected image_count at {path}:{line_number}: {image_count!r}"
            )
        episode_count += image_count
        hard_gates = record.get("hard_gates")
        if not isinstance(hard_gates, Mapping) or not hard_gates or not all(
            value is True for value in hard_gates.values()
        ):
            raise NegativeArchiveError(
                f"Stage-1 hard gate failed at {path}:{line_number}"
            )
        if any(
            record.get(field) != 0
            for field in (
                "method_label_accesses",
                "test_image_opens",
                "test_label_opens",
            )
        ):
            raise NegativeArchiveError(
                f"label/test firewall violation at {path}:{line_number}"
            )
        candidate = record.get("candidate")
        if not isinstance(candidate, Mapping):
            raise NegativeArchiveError(f"missing candidate at {path}:{line_number}")
        key = (
            str(candidate.get("optimizer")),
            str(candidate.get("learning_rate_decimal")),
            str(record.get("dataset")),
            int(record.get("severity", -1)),
            str(record.get("corruption")),
        )
        if key in unique_cells:
            raise NegativeArchiveError(
                f"duplicate candidate/cell at {path}:{line_number}: {key}"
            )
        unique_cells.add(key)
    if count != expected_count:
        raise NegativeArchiveError(
            f"unexpected record count in {path}: {count}, expected {expected_count}"
        )
    return episode_count


def _fraction_from_exact_metric(value: Any, *, label: str) -> Fraction:
    if not isinstance(value, Mapping):
        raise NegativeArchiveError(f"missing exact metric object: {label}")
    exact = value.get("exact")
    if not isinstance(exact, str):
        raise NegativeArchiveError(f"missing exact rational string: {label}")
    try:
        parsed = Fraction(exact)
    except (ValueError, ZeroDivisionError) as error:
        raise NegativeArchiveError(f"invalid exact rational metric: {label}") from error
    if value.get("numerator") != parsed.numerator or value.get("denominator") != parsed.denominator:
        raise NegativeArchiveError(f"inconsistent exact rational metric: {label}")
    return parsed


def _candidate_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NegativeArchiveError(f"missing candidate object: {label}")
    required = ("optimizer", "learning_rate", "learning_rate_decimal")
    if any(key not in value for key in required):
        raise NegativeArchiveError(f"incomplete candidate object: {label}")
    return {key: value[key] for key in required}


def validate_stage1_tree(source_root: Path, project_root: Path) -> Stage1Evidence:
    """Validate the complete v2 Stage-1 tree and its minimum utility fact."""

    stage1 = source_root / "stage1"
    _regular_tree_files(stage1, project_root)
    aggregate = stage1 / "aggregate"
    aggregate_manifest, aggregate_complete = _validate_complete_pair(aggregate)

    if aggregate_manifest.get("stage") not in (None, 1):
        raise NegativeArchiveError("aggregate manifest is not Stage 1")
    expected_aggregate = {
        "candidate_count": EXPECTED_STAGE1_CANDIDATES,
        "record_count": EXPECTED_STAGE1_CELLS,
        "episode_count": EXPECTED_STAGE1_EPISODES,
    }
    for field, expected in expected_aggregate.items():
        if aggregate_manifest.get(field) != expected:
            raise NegativeArchiveError(
                f"unexpected aggregate {field}: {aggregate_manifest.get(field)!r}"
            )
        if field != "candidate_count" and aggregate_complete.get(field) != expected:
            raise NegativeArchiveError(
                f"unexpected COMPLETE {field}: {aggregate_complete.get(field)!r}"
            )

    records_path = aggregate / "stage1_records.jsonl"
    episodes = _validate_record_file(
        records_path, expected_count=EXPECTED_STAGE1_CELLS
    )
    if episodes != EXPECTED_STAGE1_EPISODES:
        raise NegativeArchiveError(
            f"Stage-1 records imply {episodes} episodes, expected "
            f"{EXPECTED_STAGE1_EPISODES}"
        )
    diagnostics_count = sum(
        1 for _ in _iter_json_lines(aggregate / "lr_strength_diagnostics.jsonl")
    )
    if diagnostics_count != EXPECTED_STAGE1_CELLS:
        raise NegativeArchiveError(
            f"unexpected aggregate diagnostic count: {diagnostics_count}"
        )

    shard_index = _load_json_object(aggregate / "shard_index.json")
    indexed_shards = shard_index.get("shards")
    if not isinstance(indexed_shards, list) or len(indexed_shards) != EXPECTED_STAGE1_CANDIDATES:
        raise NegativeArchiveError("Stage-1 shard index must bind exactly 10 shards")
    indexed_candidates = {
        (
            str(_candidate_identity(item.get("candidate"), label="shard index")["optimizer"]),
            str(_candidate_identity(item.get("candidate"), label="shard index")["learning_rate_decimal"]),
        )
        for item in indexed_shards
        if isinstance(item, Mapping)
    }
    if len(indexed_candidates) != EXPECTED_STAGE1_CANDIDATES:
        raise NegativeArchiveError("Stage-1 shard index candidates are not unique")

    shards_root = stage1 / "shards"
    shard_directories = sorted(path for path in shards_root.iterdir() if path.is_dir())
    if len(shard_directories) != EXPECTED_STAGE1_CANDIDATES:
        raise NegativeArchiveError(
            f"expected 10 Stage-1 shard directories, found {len(shard_directories)}"
        )
    observed_candidates: set[tuple[str, str]] = set()
    observed_process_ids: set[str] = set()
    for shard in shard_directories:
        manifest, complete = _validate_complete_pair(shard)
        for field, expected in (
            ("stage", 1),
            ("record_count", EXPECTED_CELLS_PER_CANDIDATE),
            ("episode_count", EXPECTED_EPISODES_PER_CANDIDATE),
        ):
            if manifest.get(field) != expected or complete.get(field) != expected:
                raise NegativeArchiveError(
                    f"unexpected shard {field} in {shard.name}"
                )
        candidate = _candidate_identity(complete.get("candidate"), label=shard.name)
        candidate_key = (
            str(candidate["optimizer"]),
            str(candidate["learning_rate_decimal"]),
        )
        if candidate_key in observed_candidates:
            raise NegativeArchiveError(f"duplicate Stage-1 candidate shard: {candidate_key}")
        observed_candidates.add(candidate_key)
        process_id = complete.get("process_id")
        if not isinstance(process_id, str) or not process_id or process_id in observed_process_ids:
            raise NegativeArchiveError(f"invalid/reused Stage-1 process id: {shard}")
        observed_process_ids.add(process_id)
        shard_episodes = _validate_record_file(
            shard / "records.jsonl", expected_count=EXPECTED_CELLS_PER_CANDIDATE
        )
        if shard_episodes != EXPECTED_EPISODES_PER_CANDIDATE:
            raise NegativeArchiveError(f"unexpected shard episode total: {shard}")
        shard_diagnostics = sum(
            1 for _ in _iter_json_lines(shard / "lr_strength_diagnostics.jsonl")
        )
        if shard_diagnostics != EXPECTED_CELLS_PER_CANDIDATE:
            raise NegativeArchiveError(f"unexpected shard diagnostic count: {shard}")
    if observed_candidates != indexed_candidates:
        raise NegativeArchiveError("shard directory candidates do not match shard index")

    receipt_path = aggregate / "stage1_ss_top3_receipt.json"
    receipt = _load_json_object(receipt_path)
    if receipt.get("receipt_type") != "stage1_ss_top3":
        raise NegativeArchiveError("unexpected v2 Stage-1 receipt type")
    scope = receipt.get("scope")
    if not isinstance(scope, Mapping) or scope.get("paper_result") is not False:
        raise NegativeArchiveError("v2 Stage-1 receipt is not non-paper scope")
    ranking = receipt.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != EXPECTED_STAGE1_CANDIDATES:
        raise NegativeArchiveError("v2 Stage-1 receipt must rank exactly 10 candidates")
    positive: list[dict[str, Any]] = []
    nonpositive: list[dict[str, Any]] = []
    ranked_identities: set[tuple[str, str]] = set()
    for index, item in enumerate(ranking):
        if not isinstance(item, Mapping):
            raise NegativeArchiveError(f"invalid ranking item {index}")
        candidate = _candidate_identity(item.get("candidate"), label=f"ranking[{index}]")
        key = (str(candidate["optimizer"]), str(candidate["learning_rate_decimal"]))
        if key in ranked_identities:
            raise NegativeArchiveError(f"duplicate ranking candidate: {key}")
        ranked_identities.add(key)
        metric = _fraction_from_exact_metric(
            item.get("primary_mean_over_runs_macro_global_iou_delta"),
            label=f"ranking[{index}].primary_macro_iou_delta",
        )
        record = {"candidate": candidate, "macro_global_iou_delta_exact": str(metric)}
        (positive if metric > 0 else nonpositive).append(record)
    if ranked_identities != observed_candidates:
        raise NegativeArchiveError("receipt ranking candidates do not match Stage-1 shards")
    if positive:
        raise NegativeArchiveError(
            "current negative archive contract requires no strictly positive "
            "Stage-1 macro-IoU candidate"
        )

    runtime_seal = aggregate_complete.get("global_runtime_seal_sha256")
    if not isinstance(runtime_seal, str) or len(runtime_seal) != 64:
        raise NegativeArchiveError("invalid aggregate runtime-seal SHA-256")
    return Stage1Evidence(
        record_count=EXPECTED_STAGE1_CELLS,
        episode_count=EXPECTED_STAGE1_EPISODES,
        candidate_count=EXPECTED_STAGE1_CANDIDATES,
        aggregate_manifest_sha256=sha256_file(aggregate / "artifact_manifest.json"),
        receipt_sha256=sha256_file(receipt_path),
        runtime_seal_sha256=runtime_seal,
        strictly_positive_macro_iou_candidates=tuple(positive),
        nonpositive_macro_iou_candidates=tuple(nonpositive),
    )


def _require_no_formal_stage2_or_final(source_root: Path, project_root: Path) -> None:
    forbidden = (
        source_root / "stage2" / "shards",
        source_root / "stage2" / "aggregate",
        source_root / "final",
    )
    present = [
        _project_relative(path, project_root)
        for path in forbidden
        if path.exists() or path.is_symlink()
    ]
    if present:
        raise NegativeArchiveError(
            "formal Stage-2/final paths exist; refusing negative archive: "
            + ", ".join(present)
        )


def _make_entry(
    source: Path,
    *,
    project_root: Path,
    archive_relative: str,
    category: str,
) -> CopyEntry:
    source_relative = _require_regular_file(source, project_root)
    archive_path = PurePosixPath(archive_relative)
    if archive_path.is_absolute() or ".." in archive_path.parts or not archive_path.parts:
        raise NegativeArchiveError(f"unsafe archive destination: {archive_relative!r}")
    if "\n" in archive_relative or "\r" in archive_relative:
        raise NegativeArchiveError(f"unsafe archive destination: {archive_relative!r}")
    return CopyEntry(
        source=source,
        source_project_relative=source_relative,
        archive_relative=archive_path.as_posix(),
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
        category=category,
    )


def _source_manifest_paths(project_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    pilot_root = project_root / "configs/tta_train_side_pilot_v2"
    paths.append(pilot_root / "manifest.json")
    paths.extend(pilot_root / f"{dataset}.txt" for dataset in DATASETS)
    for dataset in DATASETS:
        split_root = project_root / "datasets" / dataset / "img_idx"
        paths.extend(
            (
                split_root / f"train_{dataset}.txt",
                split_root / f"test_{dataset}.txt",
            )
        )
        cache_root = (
            project_root
            / "results/binary_tent/ss_calibration_cache_v2"
            / dataset
        )
        paths.extend(
            (
                cache_root / "manifest.json",
                cache_root / "method_input_manifest.json",
                cache_root / "COMPLETE.json",
            )
        )
        paths.append(
            project_root
            / "results/corruption_pilot_fixed_split/round_02"
            / dataset
            / "artifact_manifest.json"
        )
    return tuple(paths)


def _validate_scientific_replay_receipt(path: Path) -> str:
    receipt = _load_json_object(path)
    required = {
        "protocol_status": "passed",
        "scientific_status": "failed",
        "stage2_allowed": False,
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise NegativeArchiveError(
                f"scientific replay receipt has {field}={receipt.get(field)!r}; "
                f"expected {expected!r}"
            )
    for field in ("eligible_candidates", "selected_for_stage2"):
        if receipt.get(field) != []:
            raise NegativeArchiveError(
                f"scientific replay receipt must contain empty {field}"
            )
    route = receipt.get("route_decision")
    if route not in ("stop_before_stage2", "redesign_base_adaptation"):
        raise NegativeArchiveError(
            "scientific replay receipt does not explicitly stop before Stage 2"
        )
    receipt_type = receipt.get("receipt_type")
    if receipt.get("schema_version") != 3:
        raise NegativeArchiveError("scientific replay schema_version must be 3")
    if receipt_type != "stage1_ss_scientific_selection":
        raise NegativeArchiveError("unexpected scientific replay receipt_type")
    if (
        receipt.get("selector_protocol_id")
        != "cr-sitta-binary-tent-ss-calibration-selector-v3"
    ):
        raise NegativeArchiveError("scientific replay selector protocol is invalid")
    if (
        receipt.get("formal_fully_frozen_gate") is not False
        or receipt.get("unresolved_gate_thresholds") is not True
        or receipt.get("retrospective_negative_replay") is not True
        or receipt.get("stage3_allowed") is not False
    ):
        raise NegativeArchiveError(
            "scientific replay provenance must be retrospective and non-authorizing"
        )
    gate = receipt.get("scientific_gate")
    if (
        not isinstance(gate, Mapping)
        or gate.get("mode") != "retrospective_negative_replay"
        or gate.get("threshold_resolution_status")
        != "unresolved_retrospective_only"
        or gate.get("authorizes_stage2") is not False
        or not isinstance(gate.get("unresolved_thresholds"), list)
        or not gate.get("unresolved_thresholds")
    ):
        raise NegativeArchiveError("scientific replay gate declaration is invalid")
    bindings = receipt.get("source_evidence_bindings")
    required_bindings = {
        "v3_config",
        "selector_v3",
        "v2_aggregate_manifest",
        "v2_aggregate_complete",
        "v2_stage1_records",
        "v2_strength_diagnostics",
        "v2_stage1_top3_receipt",
    }
    if not isinstance(bindings, Mapping) or not required_bindings <= set(bindings):
        raise NegativeArchiveError("scientific replay source bindings are incomplete")
    for name in required_bindings:
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            raise NegativeArchiveError(f"invalid scientific replay binding: {name}")
        digest = binding.get("sha256")
        size = binding.get("bytes")
        source_path = binding.get("path")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(source_path, str)
            or not source_path
        ):
            raise NegativeArchiveError(f"invalid scientific replay binding: {name}")
    replay_scope = receipt.get("replay_scope")
    if (
        not isinstance(replay_scope, Mapping)
        or replay_scope.get("paper_result") is not False
        or replay_scope.get("uses_test_images") is not False
        or replay_scope.get("uses_test_labels") is not False
        or replay_scope.get("authorizes_stage2") is not False
    ):
        raise NegativeArchiveError("scientific replay scope is unsafe")
    paper_result = receipt.get("paper_result")
    scope = receipt.get("scope")
    scoped_paper_result = scope.get("paper_result") if isinstance(scope, Mapping) else None
    if paper_result is not False and scoped_paper_result is not False:
        raise NegativeArchiveError(
            "scientific replay receipt is not explicitly marked paper_result=false"
        )
    return receipt_type


def _inventory_digest(entries: Sequence[CopyEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.archive_relative):
        digest.update(
            (
                f"{entry.category}\t{entry.source_project_relative}\t"
                f"{entry.archive_relative}\t{entry.sha256}\t{entry.size_bytes}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def build_archive_plan(
    *,
    project_root: Path = PROJECT_ROOT,
    source_root: Path | None = None,
    destination: Path | None = None,
    stage2_logs: Sequence[Path] = (),
    scientific_gate_receipt: Path | None = None,
) -> ArchivePlan:
    """Validate every input and return a side-effect-free archive plan."""

    project_root = project_root.resolve()
    source_root = (
        _resolve_cli_path(project_root, source_root)
        if source_root is not None
        else (project_root / DEFAULT_SOURCE).resolve()
    )
    destination = (
        _resolve_cli_path(project_root, destination)
        if destination is not None
        else (project_root / DEFAULT_DESTINATION).resolve()
    )
    _project_relative(source_root, project_root)
    _project_relative(destination, project_root)
    if source_root == destination:
        raise NegativeArchiveError("source and destination must differ")
    if source_root in destination.parents or destination in source_root.parents:
        raise NegativeArchiveError("source and destination trees must not overlap")

    stage1_evidence = validate_stage1_tree(source_root, project_root)
    _require_no_formal_stage2_or_final(source_root, project_root)

    entries: list[CopyEntry] = []
    stage1_root = source_root / "stage1"
    for path in _regular_tree_files(stage1_root, project_root):
        relative = path.relative_to(stage1_root).as_posix()
        entries.append(
            _make_entry(
                path,
                project_root=project_root,
                archive_relative=f"stage1/{relative}",
                category="stage1_v2",
            )
        )

    for relative in FROZEN_CONFIG_PATHS:
        source = project_root / relative
        entries.append(
            _make_entry(
                source,
                project_root=project_root,
                archive_relative=f"frozen_configs/{relative.as_posix()}",
                category="frozen_config",
            )
        )

    for source in _source_manifest_paths(project_root):
        source_relative = _require_regular_file(source, project_root)
        entries.append(
            _make_entry(
                source,
                project_root=project_root,
                archive_relative=f"source_manifest/{source_relative}",
                category="source_manifest",
            )
        )

    log_entries: list[CopyEntry] = []
    for path_value in stage2_logs:
        source = _resolve_cli_path(project_root, path_value)
        source_relative = _require_regular_file(source, project_root)
        entry = _make_entry(
            source,
            project_root=project_root,
            archive_relative=(
                "stage2_aborted_partial/raw_logs_only/" + source_relative
            ),
            category="stage2_raw_log_uninterpreted",
        )
        log_entries.append(entry)
        entries.append(entry)

    replay: ScientificReplayEvidence
    if scientific_gate_receipt is None:
        replay = ScientificReplayEvidence(
            status="absent_not_provided", entry=None, receipt_type=None
        )
    else:
        source = _resolve_cli_path(project_root, scientific_gate_receipt)
        _require_regular_file(source, project_root)
        receipt_type = _validate_scientific_replay_receipt(source)
        entry = _make_entry(
            source,
            project_root=project_root,
            archive_relative=(
                "scientific_gate_v3_replay/"
                + SCIENTIFIC_REPLAY_RECEIPT_FILENAME
            ),
            category="scientific_gate_v3_replay",
        )
        entries.append(entry)
        replay = ScientificReplayEvidence(
            status="provided_and_verified", entry=entry, receipt_type=receipt_type
        )

    destinations = [entry.archive_relative for entry in entries]
    if len(destinations) != len(set(destinations)):
        raise NegativeArchiveError("archive plan contains duplicate destination paths")
    entries_tuple = tuple(sorted(entries, key=lambda item: item.archive_relative))
    return ArchivePlan(
        project_root=project_root,
        source_root=source_root,
        destination=destination,
        entries=entries_tuple,
        stage1=stage1_evidence,
        scientific_replay=replay,
        source_inventory_sha256=_inventory_digest(entries_tuple),
        stage2_logs=tuple(sorted(log_entries, key=lambda item: item.archive_relative)),
    )


def _stage2_aborted_payload(plan: ArchivePlan) -> dict[str, Any]:
    raw_log_records = [entry.public_record() for entry in plan.stage2_logs]
    return {
        "artifact_type": "binary_tent_ss_v2_stage2_aborted_incomplete_audit",
        "formal_output_audit": {
            "formal_aggregate_published": False,
            "formal_shards_published": False,
            "source_paths_required_absent": [
                "results/binary_tent/ss_calibration_v2/stage2/shards",
                "results/binary_tent/ss_calibration_v2/stage2/aggregate",
                "results/binary_tent/ss_calibration_v2/final",
            ],
            "verification": "absence_checked_at_archive_preflight",
        },
        "paper_result": False,
        "partial_progress": {
            "claim_intentionally_omitted": True,
            "completed_cells_per_slot": None,
            "status": "not_independently_verifiable_from_archived_files",
        },
        "raw_log_evidence": {
            "file_count": len(raw_log_records),
            "files": raw_log_records,
            "status": (
                "archived_uninterpreted_raw_logs"
                if raw_log_records
                else "not_available_in_archivable_source"
            ),
        },
        "schema_version": 1,
        "stage2_status": "aborted_incomplete",
        "termination_reason": "stage1_scientific_gate_failed",
        "use_for_candidate_ranking": False,
    }


def _negative_result_payload(
    plan: ArchivePlan, *, aborted_sha256: str
) -> dict[str, Any]:
    replay = plan.scientific_replay
    replay_payload: dict[str, Any] = {
        "status": replay.status,
        "receipt": replay.entry.public_record() if replay.entry is not None else None,
        "receipt_type": replay.receipt_type,
    }
    return {
        "archive_scope": {
            "paper_result": False,
            "paper_result_indexing_forbidden": True,
            "source_train_derived": True,
            "use": "negative_calibration_audit_and_method_redesign_only",
        },
        "archive_source_inventory_algorithm": (
            "sorted-category-source-path-archive-path-sha256-size-lf-v1"
        ),
        "archive_source_inventory_sha256": plan.source_inventory_sha256,
        "eligible_candidates": (
            [] if replay.status == "provided_and_verified" else None
        ),
        "reason": (
            "No Stage-1 candidate had strictly positive exact macro global-IoU "
            "delta; a full v3 eligibility claim additionally requires the optional "
            "scientific replay receipt."
        ),
        "result_type": "negative_calibration_result",
        "route_decision": "redesign_base_adaptation",
        "schema_version": 1,
        "scientific_gate_evidence": {
            "retrospective_minimum_utility_check": {
                "criterion": "exact_macro_global_iou_delta_strictly_greater_than_zero",
                "nonpositive_candidate_count": len(
                    plan.stage1.nonpositive_macro_iou_candidates
                ),
                "strictly_positive_candidates": list(
                    plan.stage1.strictly_positive_macro_iou_candidates
                ),
                "source_receipt_sha256": plan.stage1.receipt_sha256,
                "status": "failed",
            },
            "scientific_gate_v3_replay": replay_payload,
        },
        "stage1_cells": plan.stage1.record_count,
        "stage1_episodes": plan.stage1.episode_count,
        "stage1_protocol_status": "passed",
        "stage1_scientific_status": "failed",
        "stage1_source_bindings": {
            "aggregate_manifest_sha256": plan.stage1.aggregate_manifest_sha256,
            "candidate_count": plan.stage1.candidate_count,
            "runtime_seal_sha256": plan.stage1.runtime_seal_sha256,
            "stage1_receipt_sha256": plan.stage1.receipt_sha256,
        },
        "stage2_aborted_audit_sha256": aborted_sha256,
        "stage2_aggregate_published": False,
        "stage2_status": "aborted_incomplete",
        "stage3_allowed": False,
    }


def _copy_registry_payload(
    plan: ArchivePlan, *, category: str, artifact_type: str
) -> dict[str, Any]:
    records = [
        entry.public_record() for entry in plan.entries if entry.category == category
    ]
    return {
        "artifact_type": artifact_type,
        "file_count": len(records),
        "files": records,
        "paper_result": False,
        "schema_version": 1,
    }


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_entry_verified(entry: CopyEntry, archive_root: Path) -> None:
    destination = archive_root / Path(entry.archive_relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with entry.source.open("rb") as source_handle:
            with destination.open("xb") as destination_handle:
                for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                    destination_handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
    except OSError as error:
        raise NegativeArchiveError(
            f"cannot create copy for {entry.source_project_relative}"
        ) from error
    if digest.hexdigest() != entry.sha256 or size != entry.size_bytes:
        raise NegativeArchiveError(
            "source changed while copying: " + entry.source_project_relative
        )


def _current_plan_inventory(plan: ArchivePlan) -> str:
    refreshed: list[CopyEntry] = []
    for entry in plan.entries:
        _require_regular_file(entry.source, plan.project_root)
        refreshed.append(
            CopyEntry(
                source=entry.source,
                source_project_relative=entry.source_project_relative,
                archive_relative=entry.archive_relative,
                sha256=sha256_file(entry.source),
                size_bytes=entry.source.stat().st_size,
                category=entry.category,
            )
        )
    return _inventory_digest(refreshed)


def _all_archive_regular_files(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise NegativeArchiveError(f"archive directory is missing or a symlink: {root}")
    files: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            if (current_path / name).is_symlink():
                raise NegativeArchiveError(
                    f"archive contains symbolic-link directory: {current_path / name}"
                )
        for name in file_names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise NegativeArchiveError(
                    f"archive contains a non-regular file: {path}"
                )
            files.append(path)
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def _write_sha256sums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    members = [
        path
        for path in _all_archive_regular_files(root)
        if path != checksum_path
    ]
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in members
    ]
    _write_bytes(checksum_path, "".join(lines).encode("utf-8"))


def _parse_sha256sums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise NegativeArchiveError(f"cannot read SHA256SUMS: {path}") from error
    if not lines:
        raise NegativeArchiveError("SHA256SUMS is empty")
    result: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise NegativeArchiveError(
                f"invalid SHA256SUMS line {line_number}"
            )
        digest, member = line[:64], line[66:]
        if any(character not in "0123456789abcdef" for character in digest):
            raise NegativeArchiveError(
                f"invalid SHA-256 on SHA256SUMS line {line_number}"
            )
        relative = PurePosixPath(member)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or member == "SHA256SUMS"
            or "\n" in member
            or "\r" in member
        ):
            raise NegativeArchiveError(
                f"unsafe SHA256SUMS member on line {line_number}: {member!r}"
            )
        canonical = relative.as_posix()
        if canonical in result:
            raise NegativeArchiveError(f"duplicate SHA256SUMS member: {canonical}")
        result[canonical] = digest
    return result


def verify_archive(
    destination: Path,
    *,
    expected_source_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the archive is complete, exact, and non-paper scoped."""

    destination = destination.resolve()
    members = _parse_sha256sums(destination / "SHA256SUMS")
    checksum_path = destination / "SHA256SUMS"
    actual_files = {
        path.relative_to(destination).as_posix()
        for path in _all_archive_regular_files(destination)
        if path != checksum_path
    }
    if actual_files != set(members):
        missing = sorted(set(members) - actual_files)
        extra = sorted(actual_files - set(members))
        raise NegativeArchiveError(
            f"archive membership mismatch; missing={missing}, extra={extra}"
        )
    for relative, expected_hash in members.items():
        path = destination / Path(relative)
        if sha256_file(path) != expected_hash:
            raise NegativeArchiveError(f"archive checksum mismatch: {relative}")

    negative = _load_json_object(destination / "NEGATIVE_RESULT.json")
    scope = negative.get("archive_scope")
    if (
        negative.get("result_type") != "negative_calibration_result"
        or not isinstance(scope, Mapping)
        or scope.get("paper_result") is not False
        or scope.get("paper_result_indexing_forbidden") is not True
        or negative.get("stage1_protocol_status") != "passed"
        or negative.get("stage1_scientific_status") != "failed"
        or negative.get("stage2_aggregate_published") is not False
        or negative.get("stage3_allowed") is not False
    ):
        raise NegativeArchiveError("NEGATIVE_RESULT.json safety semantics failed")
    inventory_hash = negative.get("archive_source_inventory_sha256")
    if not isinstance(inventory_hash, str) or len(inventory_hash) != 64:
        raise NegativeArchiveError("NEGATIVE_RESULT.json has invalid source inventory")
    if (
        expected_source_inventory_sha256 is not None
        and inventory_hash != expected_source_inventory_sha256
    ):
        raise NegativeArchiveError(
            "existing archive does not match the currently validated source plan"
        )

    aborted = _load_json_object(
        destination / "stage2_aborted_partial/ABORTED_INCOMPLETE.json"
    )
    raw_logs_directory = destination / "stage2_aborted_partial/raw_logs_only"
    if raw_logs_directory.is_symlink() or not raw_logs_directory.is_dir():
        raise NegativeArchiveError("Stage-2 raw-log quarantine directory is missing")
    if (
        aborted.get("paper_result") is not False
        or aborted.get("stage2_status") != "aborted_incomplete"
        or aborted.get("use_for_candidate_ranking") is not False
    ):
        raise NegativeArchiveError("Stage-2 aborted audit safety semantics failed")
    formal = aborted.get("formal_output_audit")
    partial = aborted.get("partial_progress")
    if (
        not isinstance(formal, Mapping)
        or formal.get("formal_shards_published") is not False
        or formal.get("formal_aggregate_published") is not False
        or not isinstance(partial, Mapping)
        or partial.get("claim_intentionally_omitted") is not True
        or partial.get("completed_cells_per_slot") is not None
    ):
        raise NegativeArchiveError("Stage-2 aborted audit makes an unsafe claim")
    if negative.get("stage2_aborted_audit_sha256") != sha256_file(
        destination / "stage2_aborted_partial/ABORTED_INCOMPLETE.json"
    ):
        raise NegativeArchiveError("NEGATIVE_RESULT does not bind aborted audit")

    validate_stage1_tree(destination, destination)
    replay = (
        negative.get("scientific_gate_evidence", {})
        .get("scientific_gate_v3_replay", {})
    )
    if not isinstance(replay, Mapping):
        raise NegativeArchiveError("invalid scientific replay declaration")
    if replay.get("status") == "provided_and_verified":
        receipt_path = (
            destination
            / "scientific_gate_v3_replay"
            / SCIENTIFIC_REPLAY_RECEIPT_FILENAME
        )
        _validate_scientific_replay_receipt(receipt_path)
        if negative.get("eligible_candidates") != []:
            raise NegativeArchiveError(
                "verified v3 replay must bind an empty eligible-candidate list"
            )
    elif replay.get("status") == "absent_not_provided":
        if replay.get("receipt") is not None or negative.get("eligible_candidates") is not None:
            raise NegativeArchiveError(
                "archive claims v3 eligibility despite absent replay receipt"
            )
        if (destination / "scientific_gate_v3_replay").exists():
            raise NegativeArchiveError("undeclared scientific replay directory exists")
    else:
        raise NegativeArchiveError("unknown scientific replay evidence status")

    return {
        "archive": str(destination),
        "file_count": len(members),
        "paper_result": False,
        "source_inventory_sha256": inventory_hash,
        "status": "verified",
    }


def _materialize_archive(plan: ArchivePlan, temporary_root: Path) -> None:
    for entry in plan.entries:
        _copy_entry_verified(entry, temporary_root)

    (temporary_root / "stage2_aborted_partial/raw_logs_only").mkdir(
        parents=True, exist_ok=True
    )
    aborted_payload = _stage2_aborted_payload(plan)
    aborted_bytes = _json_bytes(aborted_payload)
    _write_bytes(
        temporary_root / "stage2_aborted_partial/ABORTED_INCOMPLETE.json",
        aborted_bytes,
    )
    _write_bytes(
        temporary_root / "frozen_configs/FROZEN_CONFIGS.json",
        _json_bytes(
            _copy_registry_payload(
                plan,
                category="frozen_config",
                artifact_type="binary_tent_ss_v2_frozen_configs",
            )
        ),
    )
    _write_bytes(
        temporary_root / "source_manifest/ARCHIVED_SOURCE_MANIFESTS.json",
        _json_bytes(
            _copy_registry_payload(
                plan,
                category="source_manifest",
                artifact_type="binary_tent_ss_v2_archived_source_manifests",
            )
        ),
    )
    if plan.scientific_replay.entry is not None:
        _write_bytes(
            temporary_root / "scientific_gate_v3_replay/VERIFICATION.json",
            _json_bytes(
                {
                    "artifact_type": "scientific_gate_v3_replay_verification",
                    "paper_result": False,
                    "receipt": plan.scientific_replay.entry.public_record(),
                    "receipt_type": plan.scientific_replay.receipt_type,
                    "schema_version": 1,
                    "verification_status": "passed",
                }
            ),
        )
    negative_payload = _negative_result_payload(
        plan, aborted_sha256=hashlib.sha256(aborted_bytes).hexdigest()
    )
    _write_bytes(
        temporary_root / "NEGATIVE_RESULT.json", _json_bytes(negative_payload)
    )
    _write_sha256sums(temporary_root)


def create_or_verify_archive(plan: ArchivePlan) -> dict[str, Any]:
    """Publish once, or verify an identical existing archive without overwrite."""

    if plan.destination.exists() or plan.destination.is_symlink():
        return verify_archive(
            plan.destination,
            expected_source_inventory_sha256=plan.source_inventory_sha256,
        ) | {"status": "verified_existing"}

    plan.destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = plan.destination.parent / f".{plan.destination.name}.archive.lock"
    lock_descriptor: int | None = None
    try:
        try:
            lock_descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            raise NegativeArchiveError(
                f"archive creation lock already exists: {lock_path}"
            ) from error
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_descriptor)

        if plan.destination.exists() or plan.destination.is_symlink():
            return verify_archive(
                plan.destination,
                expected_source_inventory_sha256=plan.source_inventory_sha256,
            ) | {"status": "verified_existing"}

        with tempfile.TemporaryDirectory(
            prefix=f".{plan.destination.name}.tmp-",
            dir=plan.destination.parent,
        ) as temporary_name:
            temporary_root = Path(temporary_name)
            _materialize_archive(plan, temporary_root)
            if _current_plan_inventory(plan) != plan.source_inventory_sha256:
                raise NegativeArchiveError(
                    "source inputs changed during archival; archive not published"
                )
            verify_archive(
                temporary_root,
                expected_source_inventory_sha256=plan.source_inventory_sha256,
            )
            if plan.destination.exists() or plan.destination.is_symlink():
                raise NegativeArchiveError(
                    "archive destination appeared during publication; refusing overwrite"
                )
            try:
                os.rename(temporary_root, plan.destination)
            except OSError as error:
                raise NegativeArchiveError(
                    f"atomic archive publication failed: {plan.destination}"
                ) from error
        result = verify_archive(
            plan.destination,
            expected_source_inventory_sha256=plan.source_inventory_sha256,
        )
        return result | {"status": "created_and_verified"}
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _dry_run_payload(plan: ArchivePlan) -> dict[str, Any]:
    return {
        "destination": _project_relative(plan.destination, plan.project_root),
        "file_copy_count": len(plan.entries),
        "paper_result": False,
        "scientific_gate_v3_replay": plan.scientific_replay.status,
        "source": _project_relative(plan.source_root, plan.project_root),
        "source_inventory_sha256": plan.source_inventory_sha256,
        "stage1_cells": plan.stage1.record_count,
        "stage1_episodes": plan.stage1.episode_count,
        "stage2_formal_outputs": "absent_verified",
        "stage2_raw_log_count": len(plan.stage2_logs),
        "status": "dry_run_validated_no_writes",
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--stage2-log",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional raw Stage-2 log to archive without parsing or progress claims; "
            "may be repeated."
        ),
    )
    parser.add_argument(
        "--scientific-gate-receipt",
        type=Path,
        help=(
            "Optional verified v3 retrospective failure receipt.  If omitted, the "
            "archive explicitly records that replay evidence is absent."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    destination = _resolve_cli_path(project_root, args.destination)
    try:
        if args.verify_only:
            result = verify_archive(destination)
        else:
            plan = build_archive_plan(
                project_root=project_root,
                source_root=args.source_root,
                destination=args.destination,
                stage2_logs=tuple(args.stage2_log),
                scientific_gate_receipt=args.scientific_gate_receipt,
            )
            result = _dry_run_payload(plan) if args.dry_run else create_or_verify_archive(plan)
    except NegativeArchiveError as error:
        print(f"negative archive refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
