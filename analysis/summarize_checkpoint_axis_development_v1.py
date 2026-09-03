#!/usr/bin/env python3
"""Create a fail-closed development summary for checkpoint-axis v2.

This analyzer is deliberately read-only with respect to benchmark artifacts.  It
accepts only the canonical, COMPLETE artifacts selected by the frozen
``checkpoint_axis_best_pd_v1.yaml`` protocol and accepted by their public
recursive verifiers.  It never looks below private ``.build-*`` directories.

The result is *development evidence*: both checkpoint roles were selected on
test, so every output records ``development_test_selected`` and
``main_paper_table=false``.  A missing role, dataset, condition, completion
sentinel, parity receipt, or verifier pass aborts before an output directory is
created.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import benchmark.adabn_axis_runner_v2 as adabn_axis_runner
import benchmark.checkpoint_axis as checkpoint_axis
import benchmark.source_corruption_axis_runner_v2 as source_axis_runner
from tta.d0_secure_io import (
    ensure_directory_chain_nofollow,
    publish_directory_noreplace,
    read_stable_regular_file,
)


SCHEMA_VERSION = 1
ARTIFACT_CONTRACT = "cr-sitta-checkpoint-axis-development-summary-v1"
DEFAULT_AXIS_CONFIG = PROJECT_ROOT / "configs" / "checkpoint_axis_best_pd_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "checkpoint_axis_v2_summary_v1"

ROLES = ("best_miou", "best_pd")
METHODS = ("Source", "AdaBN")
DATASETS = tuple(checkpoint_axis.SUPPORTED_DATASETS)
CORRUPTIONS = (
    "gaussian_noise",
    "gaussian_blur",
    "low_contrast",
    "stripe_noise",
)
SEVERITIES = (1, 3, 5)
CONDITIONS = (
    ("clean_S0", "clean", 0),
    *tuple(
        (f"{corruption}_S{severity}", corruption, severity)
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    ),
)
CONDITION_INDEX = {key: index for index, (key, _, _) in enumerate(CONDITIONS)}

# These names intentionally preserve the two non-interchangeable metric
# families.  In particular, legacy_mean_iou is not unified_global_iou.
METRIC_FIELDS = (
    "legacy_mean_iou",
    "legacy_pd",
    "legacy_fa_per_million_pixels",
    "unified_global_iou",
    "unified_pd",
    "unified_fa_per_million_pixels",
    "unified_false_positives_per_image",
)
METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "legacy_mean_iou": {
        "family": "legacy_official_reported_operating_point",
        "display_name": "legacy mIoU",
        "unit": "ratio",
        "direction": "higher_is_better",
    },
    "legacy_pd": {
        "family": "legacy_official_reported_operating_point",
        "display_name": "legacy Pd",
        "unit": "ratio",
        "direction": "higher_is_better",
    },
    "legacy_fa_per_million_pixels": {
        "family": "legacy_official_reported_operating_point",
        "display_name": "legacy Fa",
        "unit": "false_alarm_pixels_per_1e6_image_pixels",
        "direction": "lower_is_better",
    },
    "unified_global_iou": {
        "family": "unified_fixed_threshold_0.5",
        "display_name": "unified GlobalIoU",
        "unit": "ratio",
        "direction": "higher_is_better",
    },
    "unified_pd": {
        "family": "unified_fixed_threshold_0.5",
        "display_name": "unified Pd",
        "unit": "ratio",
        "direction": "higher_is_better",
    },
    "unified_fa_per_million_pixels": {
        "family": "unified_fixed_threshold_0.5",
        "display_name": "unified Fa",
        "unit": "false_alarm_pixels_per_1e6_image_pixels",
        "direction": "lower_is_better",
    },
    "unified_false_positives_per_image": {
        "family": "unified_fixed_threshold_0.5",
        "display_name": "unified FPPI",
        "unit": "false_positive_components_per_image",
        "direction": "lower_is_better",
    },
}

TABLE_FILES = {
    "cell": "cell_metrics",
    "dataset": "dataset_summary",
    "severity": "severity_macro",
    "corruption": "corruption_macro",
    "checkpoint": "checkpoint_delta",
    "global": "global_macro",
}
EXPECTED_TABLE_ROWS = {
    "cell": 78,
    "dataset": 18,
    "severity": 24,
    "corruption": 24,
    "checkpoint": 164,
    "global": 20,
}
ELIGIBILITY_FIELDS = {
    "scientific_eligibility_tier": "development_test_selected",
    "development_only": True,
    "main_paper": False,
    "main_paper_table": False,
}
THRESHOLD_CONTRACT = {
    "transform": "sigmoid",
    "rule": "strict_greater_than",
    "value": 0.5,
}
DELTA_CONVENTIONS = {
    "method": "AdaBN_minus_Source",
    "checkpoint": "best_pd_minus_best_miou",
    "note": "delta signs are arithmetic; consult metric direction metadata",
}
MACRO_POLICY = (
    "arithmetic mean with equal weight per dataset-condition cell; no hidden "
    "image-weighting and no pooling of sufficient statistics"
)


class SummaryContractError(RuntimeError):
    """Raised before publication when an input or summary contract is invalid."""


@dataclass(frozen=True, slots=True)
class Metrics:
    legacy_mean_iou: float
    legacy_pd: float
    legacy_fa_per_million_pixels: float
    unified_global_iou: float
    unified_pd: float
    unified_fa_per_million_pixels: float
    unified_false_positives_per_image: float

    def to_dict(self) -> dict[str, float]:
        return {field: float(getattr(self, field)) for field in METRIC_FIELDS}


@dataclass(frozen=True, slots=True)
class MethodCell:
    dataset: str
    checkpoint_role: str
    checkpoint_epoch: int
    checkpoint_sha256: str
    method: str
    condition_index: int
    condition_key: str
    corruption: str
    severity: int
    metrics: Metrics
    metrics_path: str
    metrics_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedInputs:
    cells: tuple[MethodCell, ...]
    lineage: Mapping[str, Any]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryContractError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SummaryContractError(f"{label} must be a sequence")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise SummaryContractError(
            f"{label} drift: expected {expected!r}, got {actual!r}"
        )


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SummaryContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite_number(value: Any, label: str, *, ratio: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SummaryContractError(f"{label} must be finite")
    if ratio and not 0.0 <= result <= 1.0:
        raise SummaryContractError(f"{label} must be in [0, 1]")
    if not ratio and result < 0.0:
        raise SummaryContractError(f"{label} must be non-negative")
    return result


def _metrics_from_flat(record: Mapping[str, Any], label: str) -> Metrics:
    return Metrics(
        legacy_mean_iou=_finite_number(
            record.get("legacy_mean_iou"), f"{label}.legacy_mean_iou", ratio=True
        ),
        legacy_pd=_finite_number(
            record.get("legacy_pd"), f"{label}.legacy_pd", ratio=True
        ),
        legacy_fa_per_million_pixels=_finite_number(
            record.get("legacy_fa_per_million_pixels"),
            f"{label}.legacy_fa_per_million_pixels",
        ),
        unified_global_iou=_finite_number(
            record.get("unified_global_iou"),
            f"{label}.unified_global_iou",
            ratio=True,
        ),
        unified_pd=_finite_number(
            record.get("unified_pd"), f"{label}.unified_pd", ratio=True
        ),
        unified_fa_per_million_pixels=_finite_number(
            record.get("unified_fa_per_million_pixels"),
            f"{label}.unified_fa_per_million_pixels",
        ),
        unified_false_positives_per_image=_finite_number(
            record.get("unified_false_positives_per_image"),
            f"{label}.unified_false_positives_per_image",
        ),
    )


def _metrics_from_clean(record: Mapping[str, Any], label: str) -> Metrics:
    _equal(record.get("schema_version"), 2, f"{label}.schema_version")
    eligibility = _mapping(record.get("scientific_eligibility"), f"{label}.eligibility")
    _equal(eligibility.get("tier"), "development_test_selected", f"{label}.tier")
    _equal(eligibility.get("development_only"), True, f"{label}.development_only")
    _equal(eligibility.get("main_paper_table"), False, f"{label}.main_paper_table")
    _equal(record.get("complete_fixed_test_split"), True, f"{label}.fixed_split")
    checkpoint_metadata = _mapping(
        record.get("checkpoint_metadata"), f"{label}.checkpoint_metadata"
    )
    _equal(checkpoint_metadata.get("selection_split"), "test", f"{label}.selection_split")
    _equal(checkpoint_metadata.get("test_selected"), True, f"{label}.test_selected")
    _equal(checkpoint_metadata.get("development_only"), True, f"{label}.checkpoint_development_only")
    official = _mapping(
        record.get("official_reported_operating_point"), f"{label}.official"
    )
    unified = _mapping(record.get("unified"), f"{label}.unified")
    fixed = _mapping(unified.get("fixed"), f"{label}.unified.fixed")
    _equal(fixed.get("probability_threshold"), 0.5, f"{label}.probability_threshold")
    pixel = _mapping(fixed.get("pixel"), f"{label}.unified.fixed.pixel")
    target = _mapping(fixed.get("target"), f"{label}.unified.fixed.target")
    flattened = {
        "legacy_mean_iou": official.get("miou"),
        "legacy_pd": official.get("pd"),
        "legacy_fa_per_million_pixels": official.get("fa_per_pixel_x1e6"),
        "unified_global_iou": pixel.get("intersection_over_union"),
        "unified_pd": target.get("detection_probability"),
        "unified_fa_per_million_pixels": (
            _finite_number(
                target.get("false_alarm_pixel_rate"),
                f"{label}.unified.fixed.target.false_alarm_pixel_rate",
            )
            * 1_000_000.0
        ),
        "unified_false_positives_per_image": target.get(
            "false_positives_per_image"
        ),
    }
    return _metrics_from_flat(flattened, label)


def _metric_delta(left: Metrics, right: Metrics) -> Metrics:
    """Return ``left - right`` without applying metric-direction semantics."""

    return Metrics(
        **{
            field: float(getattr(left, field) - getattr(right, field))
            for field in METRIC_FIELDS
        }
    )


def _metric_mean(values: Sequence[Metrics], label: str) -> Metrics:
    if not values:
        raise SummaryContractError(f"cannot aggregate an empty metric group: {label}")
    return Metrics(
        **{
            field: math.fsum(float(getattr(value, field)) for value in values)
            / len(values)
            for field in METRIC_FIELDS
        }
    )


def _prefixed(metrics: Metrics, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{field}": value for field, value in metrics.to_dict().items()}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _project_relative(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise SummaryContractError(f"path escapes project root: {path}") from error


def _reject_hidden_path(path: Path, *, results_root: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    root = Path(os.path.abspath(results_root))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise SummaryContractError(f"{label} escapes results root: {absolute}") from error
    if not relative.parts:
        raise SummaryContractError(f"{label} cannot equal results root")
    hidden = [part for part in relative.parts if part.startswith(".")]
    if hidden:
        raise SummaryContractError(
            f"{label} is private/hidden and cannot be summarized: {absolute}"
        )
    return absolute


def _require_canonical_public_root(
    actual: Path,
    expected: Path,
    *,
    results_root: Path,
    label: str,
) -> Path:
    public = _reject_hidden_path(actual, results_root=results_root, label=label)
    expected_absolute = Path(os.path.abspath(expected))
    _equal(public, expected_absolute, f"{label} canonical root")
    return public


def _assert_no_hidden_descendants(root: Path, *, label: str) -> None:
    """Reject private members while pruning them before content is opened."""

    root = Path(os.path.abspath(root))
    # Establish a component-by-component O_NOFOLLOW chain before os.walk is
    # allowed to enumerate names.  The public recursive verifier repeats
    # secure file opens after this preflight.
    ensure_directory_chain_nofollow(root, ())
    if not root.is_dir() or root.is_symlink():
        raise SummaryContractError(f"{label} is not a public real directory: {root}")
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        hidden_directories = sorted(name for name in directory_names if name.startswith("."))
        hidden_files = sorted(name for name in file_names if name.startswith("."))
        # Prune first: only names/stat metadata are observed, never staging bytes.
        directory_names[:] = [name for name in directory_names if not name.startswith(".")]
        if hidden_directories or hidden_files:
            raise SummaryContractError(
                f"{label} contains private members: "
                f"directories={hidden_directories}, files={hidden_files}"
            )
        for name in directory_names:
            child = directory_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise SummaryContractError(f"{label} contains an unsafe directory: {child}")
        for name in file_names:
            child = directory_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise SummaryContractError(f"{label} contains an unsafe file: {child}")


def _configured_root(
    config: Mapping[str, Any], *, artifact_kind: str, role: str
) -> Path:
    runtime = _mapping(config.get("_runtime"), "axis config runtime")
    project_root = Path(str(runtime["project_root"]))
    if role == "best_miou":
        gate = _mapping(config.get("parity_gate"), "parity_gate")
        candidates = _mapping(gate.get("candidate_roots"), "candidate_roots")
        return project_root / str(candidates[artifact_kind])
    roots = _mapping(config.get("output_roots"), "output_roots")
    return project_root / str(roots[artifact_kind]) / role


def _artifact_root(
    config: Mapping[str, Any], *, artifact_kind: str, role: str, dataset: str
) -> Path:
    return _configured_root(config, artifact_kind=artifact_kind, role=role) / dataset


def preflight_required_artifacts(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve all canonical roots and reject missing COMPLETE sentinels early."""

    project_root = Path(str(_mapping(config.get("_runtime"), "runtime")["project_root"]))
    results_root = project_root / "results"
    roots: dict[str, Path] = {}
    missing: list[str] = []
    for role in ROLES:
        for kind in ("clean", "source"):
            for dataset in DATASETS:
                label = f"{role}/{kind}/{dataset}"
                root = _artifact_root(
                    config, artifact_kind=kind, role=role, dataset=dataset
                )
                roots[label] = _reject_hidden_path(
                    root, results_root=results_root, label=label
                )
                complete = root / "COMPLETE.json"
                if not complete.is_file() or complete.is_symlink():
                    missing.append(str(complete))
        label = f"{role}/adabn/global"
        root = _configured_root(config, artifact_kind="adabn", role=role)
        roots[label] = _reject_hidden_path(root, results_root=results_root, label=label)
        complete = root / "COMPLETE.json"
        if not complete.is_file() or complete.is_symlink():
            missing.append(str(complete))
    if missing:
        details = "\n  - ".join(missing)
        raise SummaryContractError(
            "checkpoint-axis summary is incomplete; canonical COMPLETE sentinels "
            f"are missing:\n  - {details}"
        )
    return roots


def _axis(
    config: Mapping[str, Any], *, artifact_kind: str, role: str, dataset: str
) -> checkpoint_axis.CheckpointAxis:
    expected_root = _artifact_root(
        config, artifact_kind=artifact_kind, role=role, dataset=dataset
    )
    output_override = expected_root if role == "best_miou" else None
    axis = checkpoint_axis.resolve_axis(
        config,
        dataset=dataset,
        role=role,  # type: ignore[arg-type]
        artifact_kind=artifact_kind,  # type: ignore[arg-type]
        output_override=output_override,
        verify_files=True,
        verify_parity_gate=True,
    )
    _equal(Path(axis.output_dir), expected_root, f"{role}/{artifact_kind}/{dataset} axis root")
    return axis


def _payload_sha(audit: Mapping[str, Any], relative: str, label: str) -> str:
    payload_tree = _mapping(audit.get("payload_tree"), f"{label}.payload_tree")
    files = _sequence(payload_tree.get("files"), f"{label}.payload_tree.files")
    matches = [
        _mapping(record, f"{label}.payload record")
        for record in files
        if isinstance(record, Mapping) and record.get("path") == relative
    ]
    if len(matches) != 1:
        raise SummaryContractError(
            f"{label} must seal exactly one {relative!r} payload"
        )
    return _sha256(matches[0].get("sha256"), f"{label}/{relative} SHA256")


def verify_and_extract_clean_artifact(
    root: Path,
    *,
    axis: checkpoint_axis.CheckpointAxis,
) -> tuple[Metrics, dict[str, Any]]:
    """Public-verifier-backed extraction used by production and real fixtures."""

    results_root = axis.config_path.parent.parent / "results"
    canonical = _require_canonical_public_root(
        root,
        Path(axis.output_dir),
        results_root=results_root,
        label=f"{axis.role}/clean/{axis.dataset}",
    )
    _assert_no_hidden_descendants(
        canonical, label=f"{axis.role}/clean/{axis.dataset}"
    )
    audit = checkpoint_axis.verify_published_artifact(
        canonical,
        expected_axis=axis,
        required_payloads=("metrics.json",),
    )
    metrics_snapshot = read_stable_regular_file(canonical / "metrics.json")
    _equal(
        metrics_snapshot.sha256,
        _payload_sha(audit, "metrics.json", f"{axis.role}/clean/{axis.dataset}"),
        "clean metrics payload SHA256",
    )
    try:
        record = _mapping(json.loads(metrics_snapshot.data), "clean metrics")
    except json.JSONDecodeError as error:
        raise SummaryContractError(f"invalid clean metrics JSON: {canonical}") from error
    for actual, expected, label in (
        (record.get("artifact_kind"), "clean", "artifact kind"),
        (record.get("dataset"), axis.dataset, "dataset"),
        (record.get("checkpoint_role"), axis.role, "checkpoint role"),
        (record.get("axis_config_sha256"), axis.config_sha256, "axis config SHA256"),
        (record.get("checkpoint_sha256"), axis.checkpoint_sha256, "checkpoint SHA256"),
    ):
        _equal(actual, expected, f"clean {axis.dataset}/{axis.role} {label}")
    checkpoint_metadata = _mapping(
        record.get("checkpoint_metadata"), "clean checkpoint_metadata"
    )
    _equal(
        checkpoint_metadata.get("selection_metric"),
        axis.expected_selection_metric,
        "clean checkpoint selection metric",
    )
    metrics = _metrics_from_clean(record, f"clean/{axis.dataset}/{axis.role}")
    lineage = {
        "artifact_kind": "clean",
        "checkpoint_role": axis.role,
        "dataset": axis.dataset,
        "root": str(canonical),
        "manifest_sha256": _sha256(
            audit.get("manifest_sha256"), "clean manifest SHA256"
        ),
        "complete_sha256": _sha256(
            audit.get("complete_sha256"), "clean COMPLETE SHA256"
        ),
        "metrics_path": str(canonical / "metrics.json"),
        "metrics_sha256": metrics_snapshot.sha256,
        "checkpoint_sha256": axis.checkpoint_sha256,
        "checkpoint_epoch": axis.expected_epoch,
        "public_verifier": "benchmark.checkpoint_axis.verify_published_artifact",
    }
    return metrics, lineage


def _validate_source_benchmark(
    benchmark: Mapping[str, Any], *, axis: checkpoint_axis.CheckpointAxis
) -> Sequence[Any]:
    for actual, expected, label in (
        (benchmark.get("schema_version"), 2, "schema_version"),
        (benchmark.get("artifact_kind"), "source", "artifact_kind"),
        (benchmark.get("method"), "Source", "method"),
        (benchmark.get("dataset"), axis.dataset, "dataset"),
        (benchmark.get("checkpoint_role"), axis.role, "checkpoint_role"),
        (benchmark.get("axis_config_sha256"), axis.config_sha256, "axis_config_sha256"),
        (benchmark.get("checkpoint_sha256"), axis.checkpoint_sha256, "checkpoint_sha256"),
        (benchmark.get("development_only"), True, "development_only"),
        (benchmark.get("main_paper_table"), False, "main_paper_table"),
        (benchmark.get("extra_best_pd_tuning_episodes"), 0, "extra tuning"),
        (benchmark.get("condition_count"), 13, "condition_count"),
    ):
        _equal(actual, expected, f"Source {axis.dataset}/{axis.role} {label}")
    conditions = _sequence(benchmark.get("conditions"), "Source benchmark conditions")
    _equal(len(conditions), len(CONDITIONS), "Source benchmark condition count")
    return conditions


def _source_cells(
    audit: Mapping[str, Any], *, axis: checkpoint_axis.CheckpointAxis
) -> tuple[MethodCell, ...]:
    benchmark = _mapping(audit.get("benchmark"), "verified Source benchmark")
    conditions = _validate_source_benchmark(benchmark, axis=axis)
    manifest = _mapping(audit.get("manifest"), "verified Source manifest")
    condition_files = _mapping(
        manifest.get("condition_files"), "Source manifest.condition_files"
    )
    root = Path(str(audit["root"]))
    cells: list[MethodCell] = []
    for index, (expected_key, expected_corruption, expected_severity) in enumerate(
        CONDITIONS
    ):
        record = _mapping(conditions[index], f"Source condition {index}")
        for actual, expected, label in (
            (record.get("condition_index"), index, "condition_index"),
            (record.get("condition_key"), expected_key, "condition_key"),
            (record.get("corruption"), expected_corruption, "corruption"),
            (record.get("severity"), expected_severity, "severity"),
        ):
            _equal(actual, expected, f"Source {axis.dataset}/{axis.role}/{label}")
        files = _mapping(
            condition_files.get(expected_key), f"Source {expected_key} files"
        )
        metrics_file = _mapping(files.get("metrics"), f"Source {expected_key} metrics file")
        relative = str(metrics_file.get("path"))
        expected_relative = f"conditions/{expected_key}/metrics.json"
        _equal(relative, expected_relative, f"Source {expected_key} metrics path")
        metrics_snapshot = read_stable_regular_file(root / relative)
        _equal(
            metrics_snapshot.sha256,
            _sha256(
                metrics_file.get("sha256"),
                f"Source {expected_key} sealed metrics SHA256",
            ),
            f"Source {expected_key} metrics file SHA256",
        )
        try:
            metrics_record = _mapping(
                json.loads(metrics_snapshot.data), f"Source {expected_key} metrics"
            )
        except json.JSONDecodeError as error:
            raise SummaryContractError(
                f"invalid Source condition metrics JSON: {root / relative}"
            ) from error
        for actual, expected, label in (
            (metrics_record.get("schema_version"), 2, "schema_version"),
            (metrics_record.get("artifact_kind"), "source", "artifact_kind"),
            (metrics_record.get("method"), "Source", "method"),
            (metrics_record.get("dataset"), axis.dataset, "dataset"),
            (metrics_record.get("checkpoint_role"), axis.role, "checkpoint_role"),
            (metrics_record.get("condition_key"), expected_key, "condition_key"),
            (metrics_record.get("corruption"), expected_corruption, "corruption"),
            (metrics_record.get("severity"), expected_severity, "severity"),
            (metrics_record.get("development_only"), True, "development_only"),
            (metrics_record.get("main_paper_table"), False, "main_paper_table"),
            (metrics_record.get("extra_best_pd_tuning_episodes"), 0, "extra tuning"),
        ):
            _equal(
                actual,
                expected,
                f"Source condition metrics {axis.dataset}/{axis.role}/{expected_key}/{label}",
            )
        condition_summary = _mapping(
            metrics_record.get("summary"), f"Source {expected_key} metrics.summary"
        )
        _equal(
            set(condition_summary),
            set(METRIC_FIELDS),
            f"Source {expected_key} metrics.summary keys",
        )
        file_metrics = _metrics_from_flat(
            condition_summary,
            f"Source/{axis.dataset}/{axis.role}/{expected_key}/metrics.json.summary",
        )
        benchmark_metrics = _metrics_from_flat(
            record, f"Source/{axis.dataset}/{axis.role}/{expected_key}/benchmark"
        )
        _equal(
            file_metrics.to_dict(),
            benchmark_metrics.to_dict(),
            f"Source benchmark/condition metrics {axis.dataset}/{axis.role}/{expected_key}",
        )
        cells.append(
            MethodCell(
                dataset=axis.dataset,
                checkpoint_role=axis.role,
                checkpoint_epoch=axis.expected_epoch,
                checkpoint_sha256=axis.checkpoint_sha256,
                method="Source",
                condition_index=index,
                condition_key=expected_key,
                corruption=expected_corruption,
                severity=expected_severity,
                metrics=file_metrics,
                metrics_path=str(root / relative),
                metrics_sha256=metrics_snapshot.sha256,
            )
        )
    return tuple(cells)


def _adabn_cells(
    aggregate_dataset: Mapping[str, Any],
    *,
    axis: checkpoint_axis.CheckpointAxis,
    global_root: Path,
    source_by_key: Mapping[str, MethodCell],
) -> tuple[MethodCell, ...]:
    _equal(aggregate_dataset.get("dataset"), axis.dataset, "AdaBN aggregate dataset")
    conditions = _sequence(
        aggregate_dataset.get("conditions"), "AdaBN aggregate dataset conditions"
    )
    _equal(len(conditions), len(CONDITIONS), "AdaBN condition count")
    cells: list[MethodCell] = []
    for index, (expected_key, expected_corruption, expected_severity) in enumerate(
        CONDITIONS
    ):
        record = _mapping(conditions[index], f"AdaBN condition {index}")
        for actual, expected, label in (
            (record.get("condition_index"), index, "condition_index"),
            (record.get("condition_key"), expected_key, "condition_key"),
            (record.get("corruption"), expected_corruption, "corruption"),
            (record.get("severity"), expected_severity, "severity"),
        ):
            _equal(actual, expected, f"AdaBN {axis.dataset}/{axis.role}/{label}")
        source_cell = source_by_key.get(expected_key)
        if source_cell is None:
            raise SummaryContractError(f"AdaBN source cell is absent: {expected_key}")
        raw_source_summary = _mapping(
            record.get("source_summary"), "AdaBN source_summary"
        )
        _equal(
            set(raw_source_summary),
            set(METRIC_FIELDS),
            f"AdaBN source_summary keys {axis.dataset}/{axis.role}/{expected_key}",
        )
        source_summary = _metrics_from_flat(
            raw_source_summary,
            f"AdaBN/{axis.dataset}/{axis.role}/{expected_key}/source_summary",
        )
        _equal(
            source_summary.to_dict(),
            source_cell.metrics.to_dict(),
            f"AdaBN/Source exact metrics {axis.dataset}/{axis.role}/{expected_key}",
        )
        aggregate_metrics = _metrics_from_flat(
            record, f"AdaBN/{axis.dataset}/{axis.role}/{expected_key}/aggregate"
        )
        # Delta values may be negative, so the non-negative metric parser is
        # intentionally not used here.
        raw_delta = _mapping(record.get("deltas_from_source"), "AdaBN delta")
        _equal(
            set(raw_delta),
            set(METRIC_FIELDS),
            f"AdaBN delta keys {axis.dataset}/{axis.role}/{expected_key}",
        )
        expected_delta = _metric_delta(aggregate_metrics, source_cell.metrics)
        parsed_delta: dict[str, float] = {}
        for field in METRIC_FIELDS:
            value = raw_delta.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SummaryContractError(f"AdaBN delta {field} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise SummaryContractError(f"AdaBN delta {field} must be finite")
            parsed_delta[field] = numeric
        _equal(
            parsed_delta,
            expected_delta.to_dict(),
            f"AdaBN stored delta {axis.dataset}/{axis.role}/{expected_key}",
        )
        relative = str(record.get("metrics"))
        _equal(
            relative,
            f"conditions/{expected_key}/metrics.json",
            f"AdaBN {expected_key} metrics path",
        )
        metrics_path = global_root / axis.dataset / relative
        metrics_snapshot = read_stable_regular_file(metrics_path)
        _equal(
            metrics_snapshot.sha256,
            _sha256(
                record.get("metrics_sha256"),
                f"AdaBN {expected_key} sealed metrics SHA256",
            ),
            f"AdaBN {expected_key} metrics file SHA256",
        )
        try:
            metrics_record = _mapping(
                json.loads(metrics_snapshot.data), f"AdaBN {expected_key} metrics"
            )
        except json.JSONDecodeError as error:
            raise SummaryContractError(
                f"invalid AdaBN condition metrics JSON: {metrics_path}"
            ) from error
        for actual, expected, label in (
            (metrics_record.get("schema_version"), 2, "schema_version"),
            (metrics_record.get("method"), "AdaBN", "method"),
            (metrics_record.get("dataset"), axis.dataset, "dataset"),
            (metrics_record.get("checkpoint_role"), axis.role, "checkpoint_role"),
            (metrics_record.get("checkpoint_sha256"), axis.checkpoint_sha256, "checkpoint_sha256"),
            (metrics_record.get("condition_key"), expected_key, "condition_key"),
            (metrics_record.get("corruption"), expected_corruption, "corruption"),
            (metrics_record.get("severity"), expected_severity, "severity"),
            (metrics_record.get("development_only"), True, "development_only"),
            (metrics_record.get("main_paper_table"), False, "main_paper_table"),
            (metrics_record.get("paper_result"), False, "paper_result"),
            (metrics_record.get("extra_best_pd_tuning_episodes"), 0, "extra tuning"),
        ):
            _equal(
                actual,
                expected,
                f"AdaBN condition metrics {axis.dataset}/{axis.role}/{expected_key}/{label}",
            )
        file_summary = _mapping(
            metrics_record.get("summary"), f"AdaBN {expected_key} metrics.summary"
        )
        _equal(
            set(file_summary),
            set(METRIC_FIELDS),
            f"AdaBN {expected_key} metrics.summary keys",
        )
        file_metrics = _metrics_from_flat(
            file_summary,
            f"AdaBN/{axis.dataset}/{axis.role}/{expected_key}/metrics.json.summary",
        )
        _equal(
            file_metrics.to_dict(),
            aggregate_metrics.to_dict(),
            f"AdaBN aggregate/condition metrics {axis.dataset}/{axis.role}/{expected_key}",
        )
        file_source_summary = _mapping(
            metrics_record.get("source_summary"),
            f"AdaBN {expected_key} metrics.source_summary",
        )
        _equal(
            dict(file_source_summary),
            source_cell.metrics.to_dict(),
            f"AdaBN condition/Source metrics {axis.dataset}/{axis.role}/{expected_key}",
        )
        file_delta = _mapping(
            metrics_record.get("deltas_from_source"),
            f"AdaBN {expected_key} metrics.deltas_from_source",
        )
        _equal(
            dict(file_delta),
            expected_delta.to_dict(),
            f"AdaBN condition delta {axis.dataset}/{axis.role}/{expected_key}",
        )
        cells.append(
            MethodCell(
                dataset=axis.dataset,
                checkpoint_role=axis.role,
                checkpoint_epoch=axis.expected_epoch,
                checkpoint_sha256=axis.checkpoint_sha256,
                method="AdaBN",
                condition_index=index,
                condition_key=expected_key,
                corruption=expected_corruption,
                severity=expected_severity,
                metrics=file_metrics,
                metrics_path=str(metrics_path),
                metrics_sha256=metrics_snapshot.sha256,
            )
        )
    return tuple(cells)


def collect_verified_inputs(
    *, axis_config_path: Path = DEFAULT_AXIS_CONFIG
) -> VerifiedInputs:
    """Verify every canonical artifact, then return immutable summary inputs."""

    config = checkpoint_axis.load_axis_config(
        axis_config_path, project_root=PROJECT_ROOT
    )
    _equal(
        Path(str(config["_runtime"]["config_path"])),
        DEFAULT_AXIS_CONFIG,
        "canonical checkpoint-axis config path",
    )
    roots = preflight_required_artifacts(config)
    parity_receipt = checkpoint_axis.verify_parity_receipt(config=config)
    input_lineage: list[dict[str, Any]] = []
    clean_metrics: dict[tuple[str, str], Metrics] = {}
    source_cells: dict[tuple[str, str], tuple[MethodCell, ...]] = {}
    all_cells: list[MethodCell] = []

    for role in ROLES:
        for dataset in DATASETS:
            clean_axis = _axis(
                config, artifact_kind="clean", role=role, dataset=dataset
            )
            clean_root = roots[f"{role}/clean/{dataset}"]
            metrics, lineage = verify_and_extract_clean_artifact(
                clean_root, axis=clean_axis
            )
            clean_metrics[(role, dataset)] = metrics
            input_lineage.append(lineage)

            source_axis = _axis(
                config, artifact_kind="source", role=role, dataset=dataset
            )
            source_root = _require_canonical_public_root(
                roots[f"{role}/source/{dataset}"],
                Path(source_axis.output_dir),
                results_root=PROJECT_ROOT / "results",
                label=f"{role}/source/{dataset}",
            )
            _assert_no_hidden_descendants(
                source_root, label=f"{role}/source/{dataset}"
            )
            source_audit = source_axis_runner.verify_source_artifact(
                source_root, expected_axis=source_axis
            )
            cells = _source_cells(source_audit, axis=source_axis)
            source_cells[(role, dataset)] = cells
            all_cells.extend(cells)
            _equal(
                cells[0].metrics.to_dict(),
                metrics.to_dict(),
                f"clean export/Source clean exact parity {dataset}/{role}",
            )
            benchmark_path = source_root / "benchmark.json"
            benchmark_snapshot = read_stable_regular_file(benchmark_path)
            _equal(
                benchmark_snapshot.sha256,
                _payload_sha(source_audit, "benchmark.json", "Source audit"),
                f"Source benchmark payload SHA256 {dataset}/{role}",
            )
            input_lineage.append(
                {
                    "artifact_kind": "source",
                    "checkpoint_role": role,
                    "dataset": dataset,
                    "root": str(source_root),
                    "manifest_sha256": source_audit["manifest_sha256"],
                    "complete_sha256": source_audit["complete_sha256"],
                    "benchmark_path": str(benchmark_path),
                    "benchmark_sha256": benchmark_snapshot.sha256,
                    "checkpoint_sha256": source_axis.checkpoint_sha256,
                    "checkpoint_epoch": source_axis.expected_epoch,
                    "public_verifier": (
                        "benchmark.source_corruption_axis_runner_v2."
                        "verify_source_artifact"
                    ),
                }
            )

    for role in ROLES:
        global_root = _require_canonical_public_root(
            roots[f"{role}/adabn/global"],
            _configured_root(config, artifact_kind="adabn", role=role),
            results_root=PROJECT_ROOT / "results",
            label=f"{role}/adabn/global",
        )
        _assert_no_hidden_descendants(
            global_root, label=f"{role}/adabn/global"
        )
        source_override = (
            _configured_root(config, artifact_kind="source", role=role)
            if role == "best_miou"
            else None
        )
        global_audit = adabn_axis_runner.verify_global_artifact(
            global_root,
            axis_config=config,
            axis_config_sha256=str(config["_runtime"]["config_sha256"]),
            role=role,
            formal_development_artifact=(role == "best_pd"),
            parity_only=(role == "best_miou"),
            source_root_override=source_override,
        )
        aggregate = _mapping(global_audit.get("aggregate"), "verified AdaBN aggregate")
        aggregate_datasets = _sequence(
            aggregate.get("datasets"), "AdaBN aggregate datasets"
        )
        _equal(
            tuple(str(_mapping(item, "AdaBN dataset").get("dataset")) for item in aggregate_datasets),
            DATASETS,
            f"AdaBN {role} dataset order",
        )
        by_dataset = {
            str(_mapping(item, "AdaBN dataset")["dataset"]): _mapping(
                item, "AdaBN dataset"
            )
            for item in aggregate_datasets
        }
        for dataset in DATASETS:
            axis = _axis(config, artifact_kind="adabn", role=role, dataset=dataset)
            source_by_key = {
                cell.condition_key: cell
                for cell in source_cells[(role, dataset)]
            }
            all_cells.extend(
                _adabn_cells(
                    by_dataset[dataset],
                    axis=axis,
                    global_root=global_root,
                    source_by_key=source_by_key,
                )
            )
        aggregate_path = global_root / "aggregate_metrics.json"
        aggregate_snapshot = read_stable_regular_file(aggregate_path)
        _equal(
            aggregate_snapshot.sha256,
            global_audit["completion"]["aggregate_metrics_sha256"],
            f"AdaBN {role} aggregate receipt",
        )
        input_lineage.append(
            {
                "artifact_kind": "adabn_global",
                "checkpoint_role": role,
                "dataset": "ALL_DATASETS",
                "root": str(global_root),
                "manifest_sha256": global_audit["manifest_sha256"],
                "complete_sha256": global_audit["completion_sha256"],
                "aggregate_metrics_path": str(aggregate_path),
                "aggregate_metrics_sha256": aggregate_snapshot.sha256,
                "datasets": [
                    {
                        "dataset": dataset,
                        "root": str(global_root / dataset),
                        "benchmark_path": str(
                            global_root / str(by_dataset[dataset]["benchmark"])
                        ),
                        "benchmark_sha256": _sha256(
                            by_dataset[dataset].get("benchmark_sha256"),
                            f"AdaBN {role}/{dataset} benchmark SHA256",
                        ),
                        "artifact_manifest_path": str(
                            global_root
                            / str(by_dataset[dataset]["artifact_manifest"])
                        ),
                        "artifact_manifest_sha256": _sha256(
                            by_dataset[dataset].get("artifact_manifest_sha256"),
                            f"AdaBN {role}/{dataset} manifest SHA256",
                        ),
                        "complete_path": str(
                            global_root / str(by_dataset[dataset]["completion"])
                        ),
                        "complete_sha256": _sha256(
                            by_dataset[dataset].get("completion_sha256"),
                            f"AdaBN {role}/{dataset} COMPLETE SHA256",
                        ),
                    }
                    for dataset in DATASETS
                ],
                "public_verifier": (
                    "benchmark.adabn_axis_runner_v2.verify_global_artifact"
                ),
            }
        )

    lineage = {
        "axis_config": {
            "path": str(DEFAULT_AXIS_CONFIG),
            "sha256": str(config["_runtime"]["config_sha256"]),
        },
        "parity_receipt": {
            "path": str(parity_receipt["_receipt_path"]),
            "sha256": str(parity_receipt["_receipt_sha256"]),
            "status": parity_receipt["status"],
            "passed": parity_receipt["passed"],
            "numeric_tolerance_used": parity_receipt["numeric_tolerance_used"],
            "public_verifier": "benchmark.checkpoint_axis.verify_parity_receipt",
        },
        "artifacts": input_lineage,
    }
    return VerifiedInputs(cells=tuple(all_cells), lineage=lineage)


def _validate_cells(cells: Sequence[MethodCell]) -> dict[tuple[str, str, str, str], MethodCell]:
    by_key: dict[tuple[str, str, str, str], MethodCell] = {}
    for cell in cells:
        if cell.dataset not in DATASETS:
            raise SummaryContractError(f"unexpected dataset: {cell.dataset}")
        if cell.checkpoint_role not in ROLES:
            raise SummaryContractError(f"unexpected checkpoint role: {cell.checkpoint_role}")
        if cell.method not in METHODS:
            raise SummaryContractError(f"unexpected method: {cell.method}")
        expected_index = CONDITION_INDEX.get(cell.condition_key)
        if expected_index is None:
            raise SummaryContractError(f"unexpected condition: {cell.condition_key}")
        expected_key, expected_corruption, expected_severity = CONDITIONS[expected_index]
        _equal(cell.condition_index, expected_index, "cell condition_index")
        _equal(cell.condition_key, expected_key, "cell condition_key")
        _equal(cell.corruption, expected_corruption, "cell corruption")
        _equal(cell.severity, expected_severity, "cell severity")
        if cell.checkpoint_epoch < 1:
            raise SummaryContractError("checkpoint epoch must be positive")
        _sha256(cell.checkpoint_sha256, "cell checkpoint SHA256")
        _sha256(cell.metrics_sha256, "cell metrics SHA256")
        _metrics_from_flat(cell.metrics.to_dict(), "cell metrics")
        key = (cell.checkpoint_role, cell.dataset, cell.condition_key, cell.method)
        if key in by_key:
            raise SummaryContractError(f"duplicate method cell: {key}")
        by_key[key] = cell
    expected = {
        (role, dataset, condition_key, method)
        for role in ROLES
        for dataset in DATASETS
        for condition_key, _, _ in CONDITIONS
        for method in METHODS
    }
    missing = sorted(expected - set(by_key))
    unexpected = sorted(set(by_key) - expected)
    if missing or unexpected:
        raise SummaryContractError(
            f"method-cell lattice is incomplete; missing={missing}, unexpected={unexpected}"
        )
    for role in ROLES:
        for dataset in DATASETS:
            lineage = {
                (
                    by_key[(role, dataset, key, method)].checkpoint_epoch,
                    by_key[(role, dataset, key, method)].checkpoint_sha256,
                )
                for key, _, _ in CONDITIONS
                for method in METHODS
            }
            if len(lineage) != 1:
                raise SummaryContractError(
                    f"checkpoint lineage drifts within {role}/{dataset}: {lineage}"
                )
    return by_key


def _wide_record(
    *,
    identity: Mapping[str, Any],
    source: Metrics,
    adabn: Metrics,
) -> dict[str, Any]:
    return {
        **ELIGIBILITY_FIELDS,
        **dict(identity),
        **_prefixed(source, "source"),
        **_prefixed(adabn, "adabn"),
        **_prefixed(_metric_delta(adabn, source), "delta_adabn_minus_source"),
    }


def _lineage_string(
    by_key: Mapping[tuple[str, str, str, str], MethodCell],
    *,
    role: str,
    datasets: Sequence[str],
) -> tuple[str, str]:
    epochs: list[str] = []
    hashes: list[str] = []
    for dataset in datasets:
        cell = by_key[(role, dataset, "clean_S0", "Source")]
        epochs.append(f"{dataset}={cell.checkpoint_epoch}")
        hashes.append(f"{dataset}={cell.checkpoint_sha256}")
    return ";".join(epochs), ";".join(hashes)


def _aggregate_wide(
    by_key: Mapping[tuple[str, str, str, str], MethodCell],
    *,
    role: str,
    datasets: Sequence[str],
    condition_keys: Sequence[str],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    source_values = [
        by_key[(role, dataset, key, "Source")].metrics
        for dataset in datasets
        for key in condition_keys
    ]
    adabn_values = [
        by_key[(role, dataset, key, "AdaBN")].metrics
        for dataset in datasets
        for key in condition_keys
    ]
    epoch_lineage, checkpoint_lineage = _lineage_string(
        by_key, role=role, datasets=datasets
    )
    return _wide_record(
        identity={
            **dict(identity),
            "checkpoint_role": role,
            "condition_count": len(source_values),
            "checkpoint_epochs": epoch_lineage,
            "checkpoint_sha256s": checkpoint_lineage,
        },
        source=_metric_mean(source_values, "Source aggregate"),
        adabn=_metric_mean(adabn_values, "AdaBN aggregate"),
    )


def _checkpoint_rows(
    tables: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    specifications = {
        "cell": ("dataset", "condition_key", "corruption", "severity", "condition_index"),
        "dataset": ("dataset", "aggregation_group"),
        "severity": ("dataset", "severity_group"),
        "corruption": ("dataset", "corruption"),
        "global": ("aggregation_group",),
    }
    records: list[dict[str, Any]] = []
    for scope, identity_fields in specifications.items():
        grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
        for row in tables[scope]:
            key = tuple(row.get(field) for field in identity_fields)
            role = str(row["checkpoint_role"])
            grouped.setdefault(key, {})[role] = row
        for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
            roles = grouped[key]
            if set(roles) != set(ROLES):
                raise SummaryContractError(
                    f"checkpoint comparison lacks a role: {scope}/{key}"
                )
            miou = roles["best_miou"]
            pd = roles["best_pd"]
            _equal(miou.get("condition_count"), pd.get("condition_count"), f"{scope}/{key} condition count")
            base: dict[str, Any] = {
                **ELIGIBILITY_FIELDS,
                "scope": scope,
                "method": "",
                "dataset": "",
                "aggregation_group": "",
                "severity_group": "",
                "condition_key": "",
                "condition_index": "",
                "corruption": "",
                "severity": "",
                "condition_count": miou["condition_count"],
                "best_miou_checkpoint_epochs": miou["checkpoint_epochs"],
                "best_miou_checkpoint_sha256s": miou["checkpoint_sha256s"],
                "best_pd_checkpoint_epochs": pd["checkpoint_epochs"],
                "best_pd_checkpoint_sha256s": pd["checkpoint_sha256s"],
            }
            for field, value in zip(identity_fields, key):
                base[field] = value
            for method in METHODS:
                prefix = method.lower()
                best_miou_metrics = Metrics(
                    **{
                        field: float(miou[f"{prefix}_{field}"])
                        for field in METRIC_FIELDS
                    }
                )
                best_pd_metrics = Metrics(
                    **{
                        field: float(pd[f"{prefix}_{field}"])
                        for field in METRIC_FIELDS
                    }
                )
                records.append(
                    {
                        **base,
                        "method": method,
                        **_prefixed(best_miou_metrics, "best_miou"),
                        **_prefixed(best_pd_metrics, "best_pd"),
                        **_prefixed(
                            _metric_delta(best_pd_metrics, best_miou_metrics),
                            "delta_best_pd_minus_best_miou",
                        ),
                    }
                )
    return records


def build_tables(cells: Sequence[MethodCell]) -> dict[str, list[dict[str, Any]]]:
    """Build the exact six-table development comparison contract."""

    by_key = _validate_cells(cells)
    tables: dict[str, list[dict[str, Any]]] = {
        "cell": [],
        "dataset": [],
        "severity": [],
        "corruption": [],
        "global": [],
    }

    for role in ROLES:
        for dataset in DATASETS:
            epoch_lineage, checkpoint_lineage = _lineage_string(
                by_key, role=role, datasets=(dataset,)
            )
            for condition_key, corruption, severity in CONDITIONS:
                source = by_key[(role, dataset, condition_key, "Source")]
                adabn = by_key[(role, dataset, condition_key, "AdaBN")]
                tables["cell"].append(
                    _wide_record(
                        identity={
                            "dataset": dataset,
                            "checkpoint_role": role,
                            "condition_index": source.condition_index,
                            "condition_key": condition_key,
                            "corruption": corruption,
                            "severity": severity,
                            "condition_count": 1,
                            "checkpoint_epochs": epoch_lineage,
                            "checkpoint_sha256s": checkpoint_lineage,
                            "source_metrics_path": source.metrics_path,
                            "source_metrics_sha256": source.metrics_sha256,
                            "adabn_metrics_path": adabn.metrics_path,
                            "adabn_metrics_sha256": adabn.metrics_sha256,
                        },
                        source=source.metrics,
                        adabn=adabn.metrics,
                    )
                )

            dataset_groups = (
                ("clean", ("clean_S0",)),
                ("corrupt12", tuple(key for key, _, _ in CONDITIONS if key != "clean_S0")),
                ("all13", tuple(key for key, _, _ in CONDITIONS)),
            )
            for group, keys in dataset_groups:
                tables["dataset"].append(
                    _aggregate_wide(
                        by_key,
                        role=role,
                        datasets=(dataset,),
                        condition_keys=keys,
                        identity={
                            "dataset": dataset,
                            "aggregation_group": group,
                            "aggregation_definition": "equal_weight_per_dataset_condition_cell",
                        },
                    )
                )

            severity_groups = (
                ("clean", ("clean_S0",)),
                *tuple(
                    (
                        f"S{severity}",
                        tuple(
                            key
                            for key, corruption, level in CONDITIONS
                            if corruption != "clean" and level == severity
                        ),
                    )
                    for severity in SEVERITIES
                ),
            )
            for group, keys in severity_groups:
                tables["severity"].append(
                    _aggregate_wide(
                        by_key,
                        role=role,
                        datasets=(dataset,),
                        condition_keys=keys,
                        identity={
                            "dataset": dataset,
                            "severity_group": group,
                            "aggregation_definition": "equal_weight_across_corruptions",
                        },
                    )
                )

            for corruption in CORRUPTIONS:
                keys = tuple(
                    key
                    for key, condition_corruption, _ in CONDITIONS
                    if condition_corruption == corruption
                )
                tables["corruption"].append(
                    _aggregate_wide(
                        by_key,
                        role=role,
                        datasets=(dataset,),
                        condition_keys=keys,
                        identity={
                            "dataset": dataset,
                            "corruption": corruption,
                            "aggregation_definition": "equal_weight_across_S1_S3_S5",
                        },
                    )
                )

        global_groups: list[tuple[str, tuple[str, ...], str]] = [
            ("clean", ("clean_S0",), "equal_weight_across_datasets"),
        ]
        global_groups.extend(
            (
                f"S{severity}",
                tuple(
                    key
                    for key, corruption, level in CONDITIONS
                    if corruption != "clean" and level == severity
                ),
                "equal_weight_across_dataset_condition_cells",
            )
            for severity in SEVERITIES
        )
        global_groups.extend(
            (
                corruption,
                tuple(
                    key
                    for key, condition_corruption, _ in CONDITIONS
                    if condition_corruption == corruption
                ),
                "equal_weight_across_dataset_condition_cells",
            )
            for corruption in CORRUPTIONS
        )
        global_groups.extend(
            (
                ("corrupt36", tuple(key for key, _, _ in CONDITIONS if key != "clean_S0"), "equal_weight_across_36_dataset_condition_cells"),
                ("all39", tuple(key for key, _, _ in CONDITIONS), "equal_weight_across_39_dataset_condition_cells"),
            )
        )
        for group, keys, definition in global_groups:
            tables["global"].append(
                _aggregate_wide(
                    by_key,
                    role=role,
                    datasets=DATASETS,
                    condition_keys=keys,
                    identity={
                        "aggregation_group": group,
                        "aggregation_definition": definition,
                    },
                )
            )

    tables["checkpoint"] = _checkpoint_rows(tables)
    for table, expected in EXPECTED_TABLE_ROWS.items():
        _equal(len(tables[table]), expected, f"{table} table row count")
    return tables


def build_summary_payload(
    *, cells: Sequence[MethodCell], lineage: Mapping[str, Any]
) -> dict[str, Any]:
    tables = build_tables(cells)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_contract": ARTIFACT_CONTRACT,
        "scientific_eligibility": {
            "tier": "development_test_selected",
            "development_only": True,
            "main_paper": False,
            "main_paper_table": False,
            "reason": (
                "best_miou and best_pd checkpoints were selected on the fixed test split"
            ),
        },
        "test_selected": True,
        "development_only": True,
        "main_paper": False,
        "main_paper_table": False,
        "threshold": dict(THRESHOLD_CONTRACT),
        "delta_conventions": dict(DELTA_CONVENTIONS),
        "macro_policy": MACRO_POLICY,
        "metric_definitions": METRIC_DEFINITIONS,
        "table_row_counts": {name: len(rows) for name, rows in tables.items()},
        "input_lineage": dict(lineage),
        "input_lineage_sha256": _json_sha256(lineage),
        "tables": tables,
    }


def _table_fieldnames(table: str, records: Sequence[Mapping[str, Any]]) -> list[str]:
    eligibility = [
        "scientific_eligibility_tier",
        "development_only",
        "main_paper",
        "main_paper_table",
    ]
    preferred = eligibility + {
        "cell": [
            "dataset", "checkpoint_role", "condition_index", "condition_key",
            "corruption", "severity", "condition_count", "checkpoint_epochs",
            "checkpoint_sha256s", "source_metrics_path", "source_metrics_sha256",
            "adabn_metrics_path", "adabn_metrics_sha256",
        ],
        "dataset": [
            "dataset", "checkpoint_role", "aggregation_group",
            "aggregation_definition", "condition_count", "checkpoint_epochs",
            "checkpoint_sha256s",
        ],
        "severity": [
            "dataset", "checkpoint_role", "severity_group",
            "aggregation_definition", "condition_count", "checkpoint_epochs",
            "checkpoint_sha256s",
        ],
        "corruption": [
            "dataset", "checkpoint_role", "corruption", "aggregation_definition",
            "condition_count", "checkpoint_epochs", "checkpoint_sha256s",
        ],
        "global": [
            "checkpoint_role", "aggregation_group", "aggregation_definition",
            "condition_count", "checkpoint_epochs", "checkpoint_sha256s",
        ],
        "checkpoint": [
            "scope", "method", "dataset", "aggregation_group", "condition_key",
            "severity_group", "condition_index", "corruption", "severity",
            "condition_count",
            "best_miou_checkpoint_epochs", "best_miou_checkpoint_sha256s",
            "best_pd_checkpoint_epochs", "best_pd_checkpoint_sha256s",
        ],
    }[table]
    if table == "checkpoint":
        prefixes = ("best_miou", "best_pd", "delta_best_pd_minus_best_miou")
    else:
        prefixes = ("source", "adabn", "delta_adabn_minus_source")
    fields = preferred + [
        f"{prefix}_{metric}" for prefix in prefixes for metric in METRIC_FIELDS
    ]
    expected = set(fields)
    for index, record in enumerate(records):
        actual = set(record)
        if expected != actual:
            raise SummaryContractError(
                f"{table} row {index} schema drift: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
    return fields


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(fieldnames: Sequence[str], records: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def verify_summary_artifact(root: Path) -> dict[str, Any]:
    """Verify a published summary envelope and its six JSON/CSV tables."""

    root = Path(os.path.abspath(root))
    manifest_snapshot = read_stable_regular_file(root / "artifact_manifest.json")
    complete_snapshot = read_stable_regular_file(root / "COMPLETE.json")
    summary_snapshot = read_stable_regular_file(root / "summary.json")
    try:
        manifest = _mapping(json.loads(manifest_snapshot.data), "summary manifest")
        complete = _mapping(json.loads(complete_snapshot.data), "summary COMPLETE")
        summary = _mapping(json.loads(summary_snapshot.data), "summary payload")
    except json.JSONDecodeError as error:
        raise SummaryContractError(f"invalid summary JSON below {root}") from error
    for value, label in ((manifest, "manifest"), (complete, "COMPLETE"), (summary, "summary")):
        _equal(value.get("schema_version"), SCHEMA_VERSION, f"summary {label} schema")
        _equal(value.get("artifact_contract"), ARTIFACT_CONTRACT, f"summary {label} contract")
    _equal(complete.get("complete"), True, "summary COMPLETE.complete")
    _equal(complete.get("manifest_sha256"), manifest_snapshot.sha256, "summary manifest receipt")
    _equal(complete.get("summary_sha256"), summary_snapshot.sha256, "summary payload receipt")
    for value, label in ((manifest, "manifest"), (complete, "COMPLETE"), (summary, "summary")):
        _equal(value.get("development_only"), True, f"summary {label}.development_only")
        _equal(value.get("main_paper"), False, f"summary {label}.main_paper")
        _equal(value.get("main_paper_table"), False, f"summary {label}.main_paper_table")
    eligibility = _mapping(summary.get("scientific_eligibility"), "summary eligibility")
    _equal(
        dict(eligibility),
        {
            "tier": "development_test_selected",
            "development_only": True,
            "main_paper": False,
            "main_paper_table": False,
            "reason": (
                "best_miou and best_pd checkpoints were selected on the fixed test split"
            ),
        },
        "summary eligibility",
    )
    _equal(summary.get("test_selected"), True, "summary test_selected")
    _equal(summary.get("threshold"), THRESHOLD_CONTRACT, "summary threshold")
    _equal(
        summary.get("delta_conventions"),
        DELTA_CONVENTIONS,
        "summary delta conventions",
    )
    _equal(summary.get("macro_policy"), MACRO_POLICY, "summary macro policy")
    _equal(
        summary.get("metric_definitions"),
        METRIC_DEFINITIONS,
        "summary metric definitions",
    )
    lineage = _mapping(summary.get("input_lineage"), "summary input_lineage")
    _equal(
        summary.get("input_lineage_sha256"),
        _json_sha256(lineage),
        "summary input lineage digest",
    )
    for value, label in ((manifest, "manifest"), (complete, "COMPLETE")):
        _equal(
            value.get("input_lineage_sha256"),
            summary.get("input_lineage_sha256"),
            f"summary {label} input lineage receipt",
        )
        _equal(
            value.get("table_row_counts"),
            EXPECTED_TABLE_ROWS,
            f"summary {label} table row counts",
        )
    measured = checkpoint_axis.artifact_tree_ledger(
        root, exclude=("artifact_manifest.json", "COMPLETE.json")
    )
    _equal(manifest.get("payload_tree"), measured, "summary recursive payload tree")
    expected_payloads = {"summary.json"} | {
        f"{stem}.{suffix}"
        for stem in TABLE_FILES.values()
        for suffix in ("json", "csv")
    }
    _equal(
        {str(record["path"]) for record in measured["files"]},
        expected_payloads,
        "summary exact payload filename set",
    )
    _equal(manifest.get("required_tables"), TABLE_FILES, "summary required tables")
    _equal(complete.get("payload_tree_sha256"), measured["sha256"], "summary tree receipt")
    _equal(complete.get("payload_file_count"), measured["file_count"], "summary file count")
    _equal(summary.get("table_row_counts"), EXPECTED_TABLE_ROWS, "summary row counts")
    for table, stem in TABLE_FILES.items():
        table_snapshot = read_stable_regular_file(root / f"{stem}.json")
        try:
            envelope = _mapping(json.loads(table_snapshot.data), f"{table} table")
        except json.JSONDecodeError as error:
            raise SummaryContractError(f"invalid {table} table JSON") from error
        _equal(envelope.get("schema_version"), SCHEMA_VERSION, f"{table} schema")
        _equal(envelope.get("artifact_contract"), ARTIFACT_CONTRACT, f"{table} contract")
        _equal(
            envelope.get("scientific_eligibility_tier"),
            "development_test_selected",
            f"{table} eligibility tier",
        )
        _equal(envelope.get("development_only"), True, f"{table} development_only")
        _equal(envelope.get("main_paper"), False, f"{table} main_paper")
        _equal(envelope.get("main_paper_table"), False, f"{table} main_paper_table")
        _equal(envelope.get("table"), table, f"{table} table name")
        _equal(envelope.get("row_count"), EXPECTED_TABLE_ROWS[table], f"{table} row_count")
        records = _sequence(envelope.get("records"), f"{table} records")
        _equal(len(records), EXPECTED_TABLE_ROWS[table], f"{table} record count")
        normalized_records = [
            dict(_mapping(record, f"{table} record")) for record in records
        ]
        for index, record in enumerate(normalized_records):
            _equal(
                {field: record.get(field) for field in ELIGIBILITY_FIELDS},
                ELIGIBILITY_FIELDS,
                f"{table} row {index} eligibility",
            )
        fieldnames = _table_fieldnames(table, normalized_records)
        csv_snapshot = read_stable_regular_file(root / f"{stem}.csv")
        _equal(
            csv_snapshot.data,
            _csv_bytes(fieldnames, normalized_records),
            f"{table} CSV/JSON exact equivalence",
        )
    return {
        "root": str(root),
        "manifest": dict(manifest),
        "complete": dict(complete),
        "summary": dict(summary),
        "manifest_sha256": manifest_snapshot.sha256,
        "complete_sha256": complete_snapshot.sha256,
        "summary_sha256": summary_snapshot.sha256,
        "payload_tree": measured,
    }


def _publish_summary_bundle(
    *,
    axis_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Collect canonical inputs internally, then write and publish their summary."""

    output = Path(os.path.abspath(output_dir))
    _equal(
        Path(os.path.abspath(axis_config_path)),
        DEFAULT_AXIS_CONFIG,
        "axis config path",
    )
    _equal(output, DEFAULT_OUTPUT, "summary output path")
    verified = collect_verified_inputs(axis_config_path=axis_config_path)
    payload = build_summary_payload(cells=verified.cells, lineage=verified.lineage)
    expected_lineage_sha = str(payload["input_lineage_sha256"])

    def guard() -> None:
        refreshed = collect_verified_inputs(axis_config_path=axis_config_path)
        _equal(
            _json_sha256(refreshed.lineage),
            expected_lineage_sha,
            "live input lineage at publication boundary",
        )
        # Rebuilding also revalidates the exact 156-cell lattice and every
        # stored Source/AdaBN delta before the no-replace rename.
        build_tables(refreshed.cells)

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"summary destination exists; refusing overwrite: {output}")
    if output.name.startswith("."):
        raise SummaryContractError("summary destination cannot be hidden")
    parent = output.parent
    ensure_directory_chain_nofollow(parent, ())
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-", dir=parent))
    published = False
    moved = False
    try:
        tables = _mapping(payload.get("tables"), "summary tables")
        summary = {key: value for key, value in payload.items() if key != "tables"}
        _write_bytes_exclusive(staging / "summary.json", _pretty_json_bytes(summary))
        for table, stem in TABLE_FILES.items():
            records = [dict(_mapping(record, f"{table} record")) for record in _sequence(tables.get(table), f"{table} records")]
            _equal(len(records), EXPECTED_TABLE_ROWS[table], f"{table} row count before publication")
            envelope = {
                "schema_version": SCHEMA_VERSION,
                "artifact_contract": ARTIFACT_CONTRACT,
                "table": table,
                "scientific_eligibility_tier": "development_test_selected",
                "development_only": True,
                "main_paper": False,
                "main_paper_table": False,
                "row_count": len(records),
                "records": records,
            }
            _write_bytes_exclusive(staging / f"{stem}.json", _pretty_json_bytes(envelope))
            fieldnames = _table_fieldnames(table, records)
            _write_bytes_exclusive(
                staging / f"{stem}.csv", _csv_bytes(fieldnames, records)
            )

        ledger = checkpoint_axis.artifact_tree_ledger(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_contract": ARTIFACT_CONTRACT,
            "artifact_kind": "checkpoint_axis_development_summary",
            "development_only": True,
            "main_paper": False,
            "main_paper_table": False,
            "scientific_eligibility_tier": "development_test_selected",
            "input_lineage_sha256": summary["input_lineage_sha256"],
            "required_tables": TABLE_FILES,
            "table_row_counts": EXPECTED_TABLE_ROWS,
            "payload_tree": ledger,
        }
        _write_bytes_exclusive(
            staging / "artifact_manifest.json", _pretty_json_bytes(manifest)
        )
        manifest_snapshot = read_stable_regular_file(staging / "artifact_manifest.json")
        summary_snapshot = read_stable_regular_file(staging / "summary.json")
        complete = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "artifact_contract": ARTIFACT_CONTRACT,
            "artifact_kind": "checkpoint_axis_development_summary",
            "development_only": True,
            "main_paper": False,
            "main_paper_table": False,
            "scientific_eligibility_tier": "development_test_selected",
            "manifest_sha256": manifest_snapshot.sha256,
            "summary_sha256": summary_snapshot.sha256,
            "input_lineage_sha256": summary["input_lineage_sha256"],
            "payload_tree_sha256": ledger["sha256"],
            "payload_file_count": ledger["file_count"],
            "table_row_counts": EXPECTED_TABLE_ROWS,
        }
        _write_bytes_exclusive(staging / "COMPLETE.json", _pretty_json_bytes(complete))
        verify_summary_artifact(staging)

        def final_guard() -> None:
            # The expensive live-input verification happens first.  The staged
            # bytes are then rehashed as the final action immediately before
            # the secure no-replace rename.
            guard()
            verify_summary_artifact(staging)

        publish_directory_noreplace(staging, output, pre_rename_guard=final_guard)
        moved = True
        try:
            audit = verify_summary_artifact(output)
        except BaseException as verification_error:
            # A post-rename failure must not strand an invalid canonical name.
            # Reuse the secure no-replace primitive in reverse, then let the
            # owned-staging cleanup below remove the rejected tree.
            try:
                publish_directory_noreplace(output, staging)
                moved = False
            except BaseException as rollback_error:
                raise SummaryContractError(
                    "summary publication failed verification and exact rollback "
                    f"did not complete cleanly: {output}"
                ) from rollback_error
            raise verification_error
        published = True
        return audit
    finally:
        if not published and not moved and staging.exists():
            # This directory was created above with an exact private prefix and
            # is the only tree this analyzer ever removes.
            if staging.parent != parent or not staging.name.startswith(
                f".{output.name}.build-"
            ):
                raise SummaryContractError(
                    f"refusing to clean an unrecognized staging path: {staging}"
                )
            shutil.rmtree(staging)


def generate_summary(
    *,
    axis_config_path: Path = DEFAULT_AXIS_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Verify all inputs before creating output, then atomically publish."""

    return _publish_summary_bundle(
        axis_config_path=axis_config_path,
        output_dir=output_dir,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis-config",
        type=Path,
        default=DEFAULT_AXIS_CONFIG,
        help="Frozen canonical checkpoint-axis config (no alternate protocol accepted).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Canonical no-replace development summary destination.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audit = generate_summary(
            axis_config_path=args.axis_config, output_dir=args.output_dir
        )
    except Exception as error:
        print(f"SUMMARY_BLOCKED: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "output": audit["root"],
                "development_only": True,
                "main_paper_table": False,
                "table_row_counts": EXPECTED_TABLE_ROWS,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
