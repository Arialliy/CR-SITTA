"""Fail-closed contracts for the test-selected checkpoint development axis.

The v2 runners deliberately keep ``best_miou`` and ``best_pd`` as separate
checkpoint roles.  ``best_pd`` may use its configured role-first destination;
``best_miou`` is a parity anchor and can only be written to an explicit
temporary destination outside every formal checkpoint-axis root.

This module owns the common lineage, path, manifest, and no-replace publication
rules.  It does not own model inference, corruption generation, or adaptation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import yaml

from benchmark.implementation_dependency_seal import (
    verify_implementation_dependency_seal,
)
from tta.d0_secure_io import (
    ensure_directory_chain_nofollow,
    publish_directory_noreplace,
    read_stable_regular_file,
)
from scripts.capture_checkpoint_axis_v2_candidate_producer_observation import (
    _verify_observation_envelope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AXIS_CONFIG = PROJECT_ROOT / "configs" / "checkpoint_axis_best_pd_v1.yaml"
RESULTS_ROOT = PROJECT_ROOT / "results"
SUPPORTED_DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
ARTIFACT_KINDS = ("clean", "source", "adabn")
TREE_ALGORITHM = "sorted-relative-path-tab-sha256-size-lf-v1"
ARTIFACT_CONTRACT = "cr-sitta-checkpoint-axis-artifact-v2"
PARITY_CONTRACT = "cr-sitta-checkpoint-axis-parity-v1"
CANDIDATE_PRODUCER_OBSERVATION_RELATIVE = (
    "results/checkpoint_axis_v2_parity_candidates/best_miou/"
    "CANDIDATE_PRODUCER_OBSERVATION"
)
CANDIDATE_PRODUCER_OBSERVATION_SEAL = MappingProxyType(
    {
        "artifact_manifest_sha256": (
            "a42e320137ff7727c8b7cd0e5dde0cd85d1d2103759aed8fedbc36f181aa63b1"
        ),
        "complete_sha256": (
            "e776dd7addbd5ebade48c792ac8625f4d640db39cf580950eef4eb60a9bd24ba"
        ),
        "observation_sha256": (
            "5920e32d1b7b79c696db96c2c70f4932f8d666391b6aebe3b3a377e8274c69e1"
        ),
        "source_inventory_sha256": (
            "c396c3955951ab35a1ca805e61cac1757b2b660ae8a1e034fcd77f7505d126b2"
        ),
        "payload_tree_sha256": (
            "45f3e1ef9f0dd0d7c37445557945bc0cd824805a8b251c6e0b7fe34a65ba1f50"
        ),
        "payload_file_count": 98,
        "observer_sha256": (
            "e0007328f3f9a2683f8c5f1168d5ce8ee88d54ebad425bf0f0e208b744dda12f"
        ),
    }
)
GUARD_ONLY_ALLOWED_CHANGED_PATHS = (
    "benchmark/checkpoint_axis.py",
    "scripts/verify_checkpoint_axis_v2_parity.py",
)
OBSERVED_NUMERIC_PRODUCER_PATHS = (
    "benchmark/adabn_axis_runner_v2.py",
    "benchmark/source_corruption_axis_runner_v2.py",
    "export_fixed_split_source_axis_v2.py",
    "run_adabn_corruption_checkpoint_axis_v2.py",
    "run_source_corruption_checkpoint_axis_v2.py",
)
OBSERVED_KEY_P1_PATHS = (
    "benchmark/__init__.py",
    "benchmark/adabn_axis_runner_v2.py",
    "benchmark/checkpoint_axis.py",
    "benchmark/source_corruption_axis_runner_v2.py",
    "configs/checkpoint_axis_best_pd_v1.yaml",
    "export_fixed_split_source_axis_v2.py",
    "run_adabn_corruption_checkpoint_axis_v2.py",
    "run_source_corruption_checkpoint_axis_v2.py",
    "scripts/run_best_pd_development_axis_v1.sh",
    "scripts/verify_checkpoint_axis_v2_parity.py",
    "tta/d0_secure_io.py",
)

CheckpointRole: TypeAlias = Literal["best_miou", "best_pd"]
SelectionMetric: TypeAlias = Literal["miou", "pd"]
ArtifactKind: TypeAlias = Literal["clean", "source", "adabn"]


@dataclass(frozen=True, slots=True)
class CheckpointAxis:
    """Fully resolved, immutable checkpoint/dataset/output contract."""

    role: CheckpointRole
    expected_selection_metric: SelectionMetric
    checkpoint_path: Path
    checkpoint_sha256: str
    development_only: bool
    dataset: str
    expected_selection_rule: str
    expected_selection_value: float
    expected_epoch: int
    recorded_test_metrics: Mapping[str, Any]
    checkpoint_selection_split: str
    test_selected: bool
    dataset_root: Path
    split_path: Path
    split_sha256: str
    expected_images: int
    image_size: int
    artifact_kind: ArtifactKind
    output_dir: Path
    protocol_id: str
    config_path: Path
    config_sha256: str
    training_protocol_path: Path
    training_protocol_sha256: str
    training_protocol_id: str
    threshold_transform: str
    threshold_rule: str
    threshold_value: float
    parity_receipt_path: Path | None
    parity_receipt_sha256: str | None

    @property
    def is_parity_anchor(self) -> bool:
        return self.role == "best_miou"

    @property
    def is_formal_development_axis(self) -> bool:
        return self.role == "best_pd"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drift: expected {expected!r}, got {actual!r}")


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _simple_component(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be a simple path component")
    return value


def _absolute_lexical(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_below(path: Path, root: Path, label: str) -> Path:
    absolute = _absolute_lexical(path)
    absolute_root = _absolute_lexical(root)
    try:
        absolute.relative_to(absolute_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes {absolute_root}: {absolute}") from error
    if absolute == absolute_root:
        raise ValueError(f"{label} cannot equal its containment root: {absolute}")
    return absolute


def _project_path(
    raw: Any,
    *,
    project_root: Path,
    label: str,
    must_be_relative: bool = True,
) -> Path:
    if not isinstance(raw, (str, os.PathLike)):
        raise TypeError(f"{label} must be a filesystem path")
    candidate = Path(raw)
    if must_be_relative and candidate.is_absolute():
        raise ValueError(f"{label} must be project-relative")
    if ".." in candidate.parts:
        raise ValueError(f"{label} cannot contain '..'")
    absolute = candidate if candidate.is_absolute() else project_root / candidate
    return _require_below(absolute, project_root, label)


def _require_real_directory(path: Path, *, label: str) -> None:
    absolute = _absolute_lexical(path)
    if not absolute.exists():
        raise FileNotFoundError(f"{label} does not exist: {absolute}")
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        value = os.lstat(current)
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise ValueError(f"{label} contains a symlink or non-directory: {current}")


def _runtime(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("_runtime"), "axis config runtime metadata")


def _config_project_root(config: Mapping[str, Any]) -> Path:
    return Path(str(_runtime(config)["project_root"]))


def load_axis_config(
    path: Path = DEFAULT_AXIS_CONFIG,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Load and structurally validate the frozen checkpoint-axis protocol."""

    project_root = _absolute_lexical(project_root)
    _require_real_directory(project_root, label="project root")
    config_path = _absolute_lexical(path)
    snapshot = read_stable_regular_file(config_path)
    try:
        parsed = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid checkpoint-axis YAML: {config_path}") from error
    payload = dict(_mapping(parsed, "checkpoint-axis config"))

    _equal(payload.get("schema_version"), 1, "axis schema_version")
    protocol_id = payload.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("axis protocol_id must be a non-empty string")
    scope = _mapping(payload.get("scope"), "axis scope")
    _equal(scope.get("checkpoint_selection_split"), "test", "selection split")
    _equal(scope.get("test_selected"), True, "test_selected disclosure")
    _equal(scope.get("development_only"), True, "development_only disclosure")
    _equal(
        scope.get("scientific_eligibility_tier"),
        "development_test_selected",
        "scientific eligibility tier",
    )
    _equal(scope.get("main_paper_table"), False, "main_paper_table")

    calibration = _mapping(payload.get("calibration_reuse"), "calibration_reuse")
    _equal(calibration.get("source_checkpoint_role"), "best_miou", "calibration role")
    _equal(calibration.get("extra_best_pd_tuning_episodes"), 0, "extra tuning")

    threshold = _mapping(payload.get("threshold"), "threshold")
    _equal(threshold.get("transform"), "sigmoid", "threshold transform")
    _equal(threshold.get("rule"), "strict_greater_than", "threshold rule")
    value = threshold.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("threshold value must be numeric")
    _equal(float(value), 0.5, "threshold value")

    conditions = _mapping(payload.get("conditions"), "conditions")
    _equal(conditions.get("count_per_dataset"), 13, "condition count")
    _equal(list(_sequence(conditions.get("ordered_severities"), "severities")), [1, 3, 5], "severities")

    output_roots = _mapping(payload.get("output_roots"), "output_roots")
    _equal(set(output_roots), set(ARTIFACT_KINDS), "output root keys")
    for kind, raw in output_roots.items():
        output_root = _project_path(
            raw, project_root=project_root, label=f"{kind} output root"
        )
        _require_below(output_root, project_root / "results", f"{kind} output root")

    datasets = _mapping(payload.get("datasets"), "datasets")
    _equal(set(datasets), set(SUPPORTED_DATASETS), "dataset set")
    for dataset_name, raw_dataset in datasets.items():
        _simple_component(dataset_name, "dataset name")
        dataset = _mapping(raw_dataset, f"dataset {dataset_name}")
        _project_path(dataset.get("root"), project_root=project_root, label=f"{dataset_name} root")
        _project_path(dataset.get("test_split"), project_root=project_root, label=f"{dataset_name} split")
        _sha256(dataset.get("test_split_sha256"), f"{dataset_name} split SHA256")
        if int(dataset.get("test_images", 0)) < 1:
            raise ValueError(f"{dataset_name} test_images must be positive")
        _equal(int(dataset.get("image_size", 0)), 256, f"{dataset_name} image_size")

    expected_axes = {
        "parity_anchor": ("best_miou", "miou", "maximize_miou_then_pd_then_minimize_fa"),
        "development_axis": ("best_pd", "pd", "maximize_pd_then_minimize_fa_then_miou"),
    }
    for section_name, (role, metric, rule) in expected_axes.items():
        section = _mapping(payload.get(section_name), section_name)
        _equal(section.get("checkpoint_role"), role, f"{section_name} role")
        _equal(section.get("expected_selection_metric"), metric, f"{section_name} metric")
        _equal(section.get("expected_selection_rule"), rule, f"{section_name} rule")
        checkpoints = _mapping(section.get("checkpoints"), f"{section_name} checkpoints")
        _equal(set(checkpoints), set(SUPPORTED_DATASETS), f"{section_name} datasets")
        for dataset_name, raw_checkpoint in checkpoints.items():
            checkpoint = _mapping(raw_checkpoint, f"{section_name}/{dataset_name}")
            _project_path(
                checkpoint.get("path"),
                project_root=project_root,
                label=f"{section_name}/{dataset_name} checkpoint",
            )
            _sha256(checkpoint.get("sha256"), f"{section_name}/{dataset_name} SHA256")
            if int(checkpoint.get("epoch", 0)) < 1:
                raise ValueError(f"{section_name}/{dataset_name} epoch must be positive")
            recorded = _mapping(
                checkpoint.get("recorded_test_metrics"),
                f"{section_name}/{dataset_name} recorded metrics",
            )
            required_metrics = {
                "duration_seconds",
                "epoch",
                "fa_per_pixel_x1e6",
                "images",
                "miou",
                "pd",
            }
            _equal(set(recorded), required_metrics, f"{section_name}/{dataset_name} recorded metric keys")
            _equal(int(recorded["epoch"]), int(checkpoint["epoch"]), f"{section_name}/{dataset_name} recorded epoch")
            _equal(int(recorded["images"]), int(datasets[dataset_name]["test_images"]), f"{section_name}/{dataset_name} recorded images")

    parity_gate = _mapping(payload.get("parity_gate"), "parity_gate")
    _equal(parity_gate.get("required_before_best_pd"), True, "parity requirement")
    _equal(parity_gate.get("receipt_schema_version"), 1, "parity receipt schema")
    _equal(parity_gate.get("checkpoint_role"), "best_miou", "parity role")
    _equal(parity_gate.get("status"), "pending_runtime_receipt", "parity status")
    _equal(
        parity_gate.get("required_receipt_type"),
        "checkpoint_axis_v2_best_miou_exact_parity",
        "parity receipt type",
    )
    _equal(
        parity_gate.get("required_artifact_contract"),
        PARITY_CONTRACT,
        "parity artifact contract",
    )
    _equal(parity_gate.get("required_runtime_status"), "passed", "parity runtime status")
    _equal(parity_gate.get("required_runtime_passed"), True, "parity runtime gate")
    _project_path(
        parity_gate.get("receipt_path"),
        project_root=project_root,
        label="parity receipt",
    )
    candidate_roots = _mapping(
        parity_gate.get("candidate_roots"), "parity candidate_roots"
    )
    _equal(set(candidate_roots), set(ARTIFACT_KINDS), "parity candidate root keys")
    configured_formal_roots = tuple(
        _project_path(
            output_roots[kind],
            project_root=project_root,
            label=f"{kind} formal output root",
        )
        for kind in ARTIFACT_KINDS
    )
    for kind, raw_root in candidate_roots.items():
        candidate_root = _project_path(
            raw_root,
            project_root=project_root,
            label=f"{kind} parity candidate root",
        )
        _require_below(
            candidate_root,
            project_root / "results",
            f"{kind} parity candidate root",
        )
        for protected in configured_formal_roots:
            try:
                candidate_root.relative_to(protected)
            except ValueError:
                continue
            raise ValueError(
                f"{kind} parity candidate root falls inside formal root {protected}"
            )
    references = _mapping(parity_gate.get("references"), "parity references")
    _equal(set(references), set(ARTIFACT_KINDS), "parity reference kinds")
    for kind in ("clean", "source"):
        by_dataset = _mapping(references[kind], f"{kind} parity references")
        _equal(set(by_dataset), set(SUPPORTED_DATASETS), f"{kind} parity reference datasets")
        for dataset_name, raw_reference in by_dataset.items():
            reference = _mapping(raw_reference, f"{kind}/{dataset_name} parity reference")
            _project_path(
                reference.get("root"),
                project_root=project_root,
                label=f"{kind}/{dataset_name} parity reference root",
            )
            _equal(reference.get("tree_algorithm"), TREE_ALGORITHM, f"{kind}/{dataset_name} tree algorithm")
            _sha256(reference.get("tree_sha256"), f"{kind}/{dataset_name} tree SHA256")
            if int(reference.get("tree_file_count", 0)) < 1:
                raise ValueError(f"{kind}/{dataset_name} tree_file_count must be positive")
    adabn_reference = _mapping(references["adabn"], "adabn parity reference")
    adabn_root = _project_path(
        adabn_reference.get("root"),
        project_root=project_root,
        label="adabn parity reference root",
    )
    for field in (
        "aggregate_metrics_file",
        "artifact_manifest_file",
        "complete_file",
    ):
        relative = _relative_artifact_path(
            str(adabn_reference.get(field)), label=f"adabn {field}"
        )
        _require_below(adabn_root.joinpath(*relative.parts), adabn_root, f"adabn {field}")
    for field in (
        "aggregate_metrics_sha256",
        "artifact_manifest_sha256",
        "complete_sha256",
    ):
        _sha256(adabn_reference.get(field), f"adabn {field}")
    adabn_datasets = _mapping(adabn_reference.get("datasets"), "adabn reference datasets")
    _equal(set(adabn_datasets), set(SUPPORTED_DATASETS), "adabn reference dataset set")
    for dataset_name, raw_reference in adabn_datasets.items():
        reference = _mapping(raw_reference, f"adabn/{dataset_name} reference")
        _sha256(reference.get("tree_sha256"), f"adabn/{dataset_name} tree SHA256")
        if int(reference.get("tree_file_count", 0)) < 1:
            raise ValueError(f"adabn/{dataset_name} tree_file_count must be positive")

    training = _mapping(payload.get("training_protocol"), "training_protocol")
    training_path = _project_path(
        training.get("path"), project_root=project_root, label="training protocol"
    )
    training_snapshot = read_stable_regular_file(training_path)
    _equal(training_snapshot.sha256, _sha256(training.get("sha256"), "training protocol SHA256"), "training protocol SHA256")
    try:
        training_payload = _mapping(
            yaml.safe_load(training_snapshot.data.decode("utf-8")),
            "training protocol payload",
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid training protocol YAML: {training_path}") from error
    _equal(training_payload.get("protocol_id"), training.get("protocol_id"), "training protocol ID")
    training_datasets = _mapping(training_payload.get("datasets"), "training datasets")
    training_section = _mapping(training_payload.get("training"), "training section")
    for dataset_name, raw_dataset in datasets.items():
        dataset = _mapping(raw_dataset, dataset_name)
        frozen = _mapping(training_datasets.get(dataset_name), f"training dataset {dataset_name}")
        _equal(dataset.get("root"), frozen.get("root"), f"{dataset_name} root lineage")
        _equal(dataset.get("test_split"), frozen.get("test_split"), f"{dataset_name} split lineage")
        _equal(dataset.get("test_split_sha256"), frozen.get("test_split_sha256"), f"{dataset_name} split hash lineage")
        _equal(dataset.get("test_images"), frozen.get("test_images"), f"{dataset_name} image-count lineage")
        _equal(dataset.get("image_size"), training_section.get("base_size"), f"{dataset_name} preprocessing lineage")

    payload["_runtime"] = {
        "config_path": str(config_path),
        "config_sha256": snapshot.sha256,
        "project_root": str(project_root),
        "training_protocol_path": str(training_path),
        "training_protocol_sha256": training_snapshot.sha256,
    }
    return payload


def _formal_output_roots(config: Mapping[str, Any]) -> tuple[Path, ...]:
    root = _config_project_root(config)
    output_roots = _mapping(config["output_roots"], "output_roots")
    return tuple(
        _project_path(output_roots[kind], project_root=root, label=f"{kind} output root")
        for kind in ARTIFACT_KINDS
    )


def canonical_output_dir(
    config: Mapping[str, Any],
    *,
    artifact_kind: ArtifactKind,
    role: CheckpointRole,
    dataset: str,
    output_override: Path | None = None,
) -> Path:
    """Resolve a safe role-first destination and protect all formal roots."""

    if artifact_kind not in ARTIFACT_KINDS:
        raise ValueError(f"unsupported artifact_kind {artifact_kind!r}")
    if role not in {"best_miou", "best_pd"}:
        raise ValueError(f"unsupported checkpoint role {role!r}")
    _simple_component(dataset, "dataset")
    project_root = _config_project_root(config)
    results_root = project_root / "results"
    formal_root = _project_path(
        _mapping(config["output_roots"], "output_roots")[artifact_kind],
        project_root=project_root,
        label=f"{artifact_kind} output root",
    )

    if role == "best_miou":
        if output_override is None:
            raise ValueError(
                "best_miou is parity-only in v2 and requires an explicit --output-dir"
            )
        output = _require_below(
            output_override if output_override.is_absolute() else project_root / output_override,
            results_root,
            "best_miou parity output",
        )
        for protected in _formal_output_roots(config):
            try:
                output.relative_to(protected)
            except ValueError:
                continue
            raise ValueError(
                "best_miou parity output cannot be inside a formal checkpoint-axis root: "
                f"{output}"
            )
        gate = _mapping(config["parity_gate"], "parity_gate")
        candidate_roots = _mapping(gate["candidate_roots"], "parity candidate_roots")
        expected = _project_path(
            candidate_roots[artifact_kind],
            project_root=project_root,
            label=f"{artifact_kind} parity candidate root",
        ) / dataset
        _equal(output, expected, "best_miou parity destination")
        return output

    canonical = formal_root / role / dataset
    if output_override is None:
        return canonical
    output = _require_below(
        output_override if output_override.is_absolute() else project_root / output_override,
        results_root,
        "best_pd output override",
    )
    for protected in _formal_output_roots(config):
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        raise ValueError(
            "a best_pd output override cannot be placed inside a formal "
            f"checkpoint-axis root: {output}"
        )
    return output


def resolve_axis(
    config: Mapping[str, Any],
    *,
    dataset: str,
    role: CheckpointRole,
    artifact_kind: ArtifactKind,
    output_override: Path | None = None,
    verify_files: bool = True,
    verify_parity_gate: bool = True,
) -> CheckpointAxis:
    """Resolve and, by default, byte-verify one checkpoint-axis cell."""

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected {SUPPORTED_DATASETS}")
    if role not in {"best_miou", "best_pd"}:
        raise ValueError(f"unknown checkpoint role {role!r}")
    section_name = "parity_anchor" if role == "best_miou" else "development_axis"
    section = _mapping(config[section_name], section_name)
    checkpoint = _mapping(
        _mapping(section["checkpoints"], f"{section_name} checkpoints")[dataset],
        f"{section_name}/{dataset}",
    )
    dataset_contract = _mapping(
        _mapping(config["datasets"], "datasets")[dataset], dataset
    )
    scope = _mapping(config["scope"], "scope")
    threshold = _mapping(config["threshold"], "threshold")
    training = _mapping(config["training_protocol"], "training_protocol")
    runtime = _runtime(config)
    project_root = Path(str(runtime["project_root"]))
    output = canonical_output_dir(
        config,
        artifact_kind=artifact_kind,
        role=role,
        dataset=dataset,
        output_override=output_override,
    )
    parity_receipt: dict[str, Any] | None = None
    if role == "best_pd" and verify_parity_gate:
        parity_receipt = verify_parity_receipt(config=config)
    axis = CheckpointAxis(
        role=cast(CheckpointRole, role),
        expected_selection_metric=cast(SelectionMetric, section["expected_selection_metric"]),
        checkpoint_path=_project_path(checkpoint["path"], project_root=project_root, label="checkpoint"),
        checkpoint_sha256=_sha256(checkpoint["sha256"], "checkpoint SHA256"),
        development_only=bool(scope["development_only"]),
        dataset=dataset,
        expected_selection_rule=str(section["expected_selection_rule"]),
        expected_selection_value=float(checkpoint["selection_value"]),
        expected_epoch=int(checkpoint["epoch"]),
        recorded_test_metrics=MappingProxyType(dict(checkpoint["recorded_test_metrics"])),
        checkpoint_selection_split=str(scope["checkpoint_selection_split"]),
        test_selected=bool(scope["test_selected"]),
        dataset_root=_project_path(dataset_contract["root"], project_root=project_root, label="dataset root"),
        split_path=_project_path(dataset_contract["test_split"], project_root=project_root, label="test split"),
        split_sha256=_sha256(dataset_contract["test_split_sha256"], "split SHA256"),
        expected_images=int(dataset_contract["test_images"]),
        image_size=int(dataset_contract["image_size"]),
        artifact_kind=cast(ArtifactKind, artifact_kind),
        output_dir=output,
        protocol_id=str(config["protocol_id"]),
        config_path=Path(str(runtime["config_path"])),
        config_sha256=str(runtime["config_sha256"]),
        training_protocol_path=Path(str(runtime["training_protocol_path"])),
        training_protocol_sha256=str(runtime["training_protocol_sha256"]),
        training_protocol_id=str(training["protocol_id"]),
        threshold_transform=str(threshold["transform"]),
        threshold_rule=str(threshold["rule"]),
        threshold_value=float(threshold["value"]),
        parity_receipt_path=(
            Path(str(parity_receipt["_receipt_path"]))
            if parity_receipt is not None
            else None
        ),
        parity_receipt_sha256=(
            str(parity_receipt["_receipt_sha256"])
            if parity_receipt is not None
            else None
        ),
    )
    if verify_files:
        _require_real_directory(axis.dataset_root, label="dataset root")
        split_snapshot = read_stable_regular_file(axis.split_path)
        _equal(split_snapshot.sha256, axis.split_sha256, "fixed test split SHA256")
        identifiers = split_snapshot.data.decode("utf-8-sig").splitlines()
        identifiers = [value.strip() for value in identifiers]
        if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("fixed test split is empty, duplicated, or contains blank IDs")
        _equal(len(identifiers), axis.expected_images, "fixed test image count")
        verify_checkpoint_file(axis)
    return axis


def verify_checkpoint_file(axis: CheckpointAxis):
    """Return a stable checkpoint snapshot after exact SHA-256 verification."""

    snapshot = read_stable_regular_file(axis.checkpoint_path)
    _equal(snapshot.sha256, axis.checkpoint_sha256, "checkpoint SHA256")
    return snapshot


def validate_checkpoint_payload(
    axis: CheckpointAxis,
    payload: Mapping[str, Any],
) -> None:
    """Require every frozen checkpoint metadata field to match exactly."""

    required = {
        "dataset",
        "epoch",
        "selection_metric",
        "selection_rule",
        "selection_value",
        "test_metrics",
        "test_selected",
        "split_manifest",
        "state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"checkpoint is missing metadata: {missing}")
    _equal(payload["dataset"], axis.dataset, "checkpoint dataset")
    _equal(int(payload["epoch"]), axis.expected_epoch, "checkpoint epoch")
    _equal(payload["selection_metric"], axis.expected_selection_metric, "checkpoint selection metric")
    _equal(payload["selection_rule"], axis.expected_selection_rule, "checkpoint selection rule")
    _equal(float(payload["selection_value"]), axis.expected_selection_value, "checkpoint selection value")
    _equal(payload["test_selected"], True, "checkpoint test_selected")
    _equal(dict(_mapping(payload["test_metrics"], "checkpoint test_metrics")), dict(axis.recorded_test_metrics), "checkpoint recorded test metrics")
    split_manifest = _mapping(payload["split_manifest"], "checkpoint split_manifest")
    _equal(split_manifest.get("test_split_sha256"), axis.split_sha256, "checkpoint test split SHA256")
    _equal(int(split_manifest.get("test_count", -1)), axis.expected_images, "checkpoint test count")
    run_config = _mapping(payload.get("run_config", {}), "checkpoint run_config")
    _equal(run_config.get("protocol_id"), axis.training_protocol_id, "checkpoint training protocol ID")
    _equal(run_config.get("protocol_sha256"), axis.training_protocol_sha256, "checkpoint training protocol SHA256")


def load_and_verify_checkpoint(axis: CheckpointAxis) -> Mapping[str, Any]:
    """Load a trusted Torch payload from the exact stable bytes just hashed."""

    snapshot = verify_checkpoint_file(axis)
    import torch

    stream = io.BytesIO(snapshot.data)
    try:
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    except TypeError:
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a metadata mapping")
    validate_checkpoint_payload(axis, payload)
    return payload


def load_checkpoint_into_model(model: Any, payload: Mapping[str, Any]) -> str:
    """Load an already-byte-verified payload with exact keys and shapes."""

    import torch

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError("checkpoint state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()):
        raise TypeError("checkpoint state_dict must map strings to tensors")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint keys do not exactly match model: missing={missing}, unexpected={unexpected}"
        )
    mismatches = {
        key: (tuple(state_dict[key].shape), tuple(expected[key].shape))
        for key in expected
        if tuple(state_dict[key].shape) != tuple(expected[key].shape)
    }
    if mismatches:
        raise RuntimeError(f"checkpoint tensor shapes do not exactly match model: {mismatches}")
    model.load_state_dict(state_dict, strict=True)
    return "state_dict"


def _relative_artifact_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} is not a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} escapes the artifact root: {value!r}")
    return path


def _regular_tree_members(root: Path) -> tuple[Path, ...]:
    _require_real_directory(root, label="artifact tree")
    members: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(directory_names):
            child = directory_path / name
            value = os.lstat(child)
            if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
                raise ValueError(f"artifact tree contains unsafe directory: {child}")
        for name in sorted(file_names):
            child = directory_path / name
            value = os.lstat(child)
            if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
                raise ValueError(f"artifact tree contains unsafe file: {child}")
            members.append(child)
    return tuple(sorted(members, key=lambda item: item.relative_to(root).as_posix()))


def artifact_tree_ledger(
    root: Path,
    *,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """Hash every regular member and the sorted portable member ledger."""

    excluded = {_relative_artifact_path(value, label="excluded artifact").as_posix() for value in exclude}
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in _regular_tree_members(root):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        snapshot = read_stable_regular_file(path)
        record = {
            "path": relative,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
        }
        records.append(record)
        digest.update(
            f"{relative}\t{snapshot.sha256}\t{snapshot.size_bytes}\n".encode("utf-8")
        )
    return {
        "algorithm": TREE_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def verify_candidate_producer_observation(
    *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Verify the immutable post-run/pre-patch candidate observation.

    The observation is deliberately not promoted to a historical full-runtime
    seal.  It records the already-produced candidate trees and the then-live
    key P1 source bytes, while preserving the explicit limitations written by
    its collector.
    """

    root = _project_path(
        CANDIDATE_PRODUCER_OBSERVATION_RELATIVE,
        project_root=_absolute_lexical(project_root),
        label="candidate producer observation",
    )
    measured = _verify_observation_envelope(root)
    expected = dict(CANDIDATE_PRODUCER_OBSERVATION_SEAL)
    for field, value in expected.items():
        _equal(measured.get(field), value, f"candidate producer observation {field}")
    _equal(
        measured.get("output"),
        str(root),
        "candidate producer observation output path",
    )
    observer_path = _project_path(
        "scripts/capture_checkpoint_axis_v2_candidate_producer_observation.py",
        project_root=_absolute_lexical(project_root),
        label="candidate producer observer",
    )
    _equal(
        read_stable_regular_file(observer_path).sha256,
        expected["observer_sha256"],
        "live candidate producer observer SHA256",
    )
    return {
        "path": CANDIDATE_PRODUCER_OBSERVATION_RELATIVE,
        **expected,
        "capture_timing": "post_run_pre_patch",
        "full_runtime_dependency_sealed": False,
        "implementation_identity_asserted": False,
        "adabn_v2_orchestrator_runtime_bound": False,
    }


def checkpoint_axis_guard_only_patch_audit(
    *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Recompute the honest old-candidate/new-authorization continuity audit.

    Only the parity verifier and this contract module may differ from the
    post-run observation.  The five numerical v2 producer/entrypoint files
    must remain byte-identical.  This is a guard-only continuity statement,
    not a claim that the future best_pd implementation itself was directly
    run in the best_miou parity experiment.
    """

    root_path = _absolute_lexical(project_root)
    observation = verify_candidate_producer_observation(project_root=root_path)
    observation_root = _project_path(
        observation["path"],
        project_root=root_path,
        label="candidate producer observation",
    )
    inventory_snapshot = read_stable_regular_file(
        observation_root / "source_inventory.json"
    )
    _equal(
        inventory_snapshot.sha256,
        observation["source_inventory_sha256"],
        "candidate producer source inventory SHA256",
    )
    try:
        inventory = _mapping(
            json.loads(inventory_snapshot.data), "candidate producer source inventory"
        )
    except json.JSONDecodeError as error:
        raise ValueError("invalid candidate producer source inventory JSON") from error
    _equal(inventory.get("schema_version"), 1, "source inventory schema")
    _equal(
        inventory.get("observation_phase"),
        "post_run_pre_patch",
        "source inventory observation phase",
    )
    _equal(
        inventory.get("full_runtime_dependency_sealed"),
        False,
        "source inventory full-runtime limitation",
    )
    _equal(
        inventory.get("implementation_identity_asserted"),
        False,
        "source inventory implementation-identity limitation",
    )
    records_raw = _sequence(inventory.get("files"), "source inventory files")
    records: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(records_raw):
        record = _mapping(value, f"source inventory file {index}")
        path = record.get("path")
        if not isinstance(path, str) or path in records:
            raise ValueError("source inventory paths must be unique strings")
        _sha256(record.get("sha256"), f"source inventory SHA256 {path}")
        records[path] = record
    _equal(
        tuple(sorted(records)),
        tuple(sorted(OBSERVED_KEY_P1_PATHS)),
        "source inventory key P1 path set",
    )
    comparisons: list[dict[str, Any]] = []
    changed: list[str] = []
    for path in OBSERVED_KEY_P1_PATHS:
        recorded = records[path]
        current = read_stable_regular_file(
            _project_path(path, project_root=root_path, label=f"observed P1 source {path}")
        )
        recorded_sha256 = str(recorded["sha256"])
        status = "unchanged" if current.sha256 == recorded_sha256 else "changed"
        if status == "changed":
            changed.append(path)
        comparisons.append(
            {
                "path": path,
                "observed_post_run_pre_patch_sha256": recorded_sha256,
                "authorized_live_sha256": current.sha256,
                "status": status,
            }
        )
    _equal(
        tuple(sorted(changed)),
        tuple(sorted(GUARD_ONLY_ALLOWED_CHANGED_PATHS)),
        "guard-only changed path set",
    )
    numeric_status = {
        record["path"]: record["status"]
        for record in comparisons
        if record["path"] in OBSERVED_NUMERIC_PRODUCER_PATHS
    }
    _equal(
        numeric_status,
        {path: "unchanged" for path in OBSERVED_NUMERIC_PRODUCER_PATHS},
        "observed numerical producer continuity",
    )
    return {
        "status": "passed",
        "audit_scope": "observed_key_p1_sources_not_full_runtime_dependency_closure",
        "observation_limitations_acknowledged": True,
        "allowed_changed_paths": list(GUARD_ONLY_ALLOWED_CHANGED_PATHS),
        "observed_changed_paths": sorted(changed),
        "observed_numeric_producer_paths": list(OBSERVED_NUMERIC_PRODUCER_PATHS),
        "observed_numeric_producer_files_unchanged": True,
        "comparisons": comparisons,
    }


def get_parity_reference(
    config: Mapping[str, Any],
    *,
    artifact_kind: ArtifactKind,
    dataset: str,
) -> dict[str, Any]:
    """Return a copy of the frozen v1 reference contract for one dataset."""

    references = _mapping(_mapping(config["parity_gate"], "parity_gate")["references"], "parity references")
    kind_reference = _mapping(references.get(artifact_kind), f"{artifact_kind} parity reference")
    if artifact_kind == "adabn":
        result = dict(kind_reference)
        result["dataset"] = dict(_mapping(_mapping(kind_reference["datasets"], "adabn datasets")[dataset], f"adabn/{dataset}"))
        return result
    return dict(_mapping(kind_reference.get(dataset), f"{artifact_kind}/{dataset} parity reference"))


def build_artifact_manifest(
    staging: Path,
    *,
    axis: CheckpointAxis,
    required_payloads: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a recursive manifest for every staged payload file."""

    required = tuple(
        _relative_artifact_path(value, label="required payload").as_posix()
        for value in required_payloads
    )
    ledger = artifact_tree_ledger(
        staging, exclude=("artifact_manifest.json", "COMPLETE.json")
    )
    present = {record["path"] for record in ledger["files"]}
    missing = sorted(set(required) - present)
    if missing:
        raise FileNotFoundError(f"required staged payloads are missing: {missing}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_contract": ARTIFACT_CONTRACT,
        "artifact_kind": axis.artifact_kind,
        "protocol_id": axis.protocol_id,
        "axis_config_sha256": axis.config_sha256,
        "dataset": axis.dataset,
        "checkpoint_role": axis.role,
        "checkpoint_sha256": axis.checkpoint_sha256,
        "checkpoint_epoch": axis.expected_epoch,
        "split_sha256": axis.split_sha256,
        "threshold": {
            "transform": axis.threshold_transform,
            "rule": axis.threshold_rule,
            "value": axis.threshold_value,
        },
        "required_payloads": list(required),
        "payload_tree": ledger,
    }
    if extra:
        overlap = set(manifest) & set(extra)
        if overlap:
            raise ValueError(f"manifest extension overwrites protected fields: {sorted(overlap)}")
        manifest.update(dict(extra))
    return manifest


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_staging(axis: CheckpointAxis) -> Path:
    """Create a private sibling staging directory under the secured results tree."""

    project_results = axis.config_path.parent.parent / "results"
    final = _require_below(axis.output_dir, project_results, "axis output")
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"destination exists; refusing overwrite: {final}")
    relative_parent = final.parent.relative_to(project_results)
    parent = ensure_directory_chain_nofollow(project_results, relative_parent.parts)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.build-", dir=parent))
    return staging


def _validate_config_lineage(axis: CheckpointAxis) -> None:
    _equal(read_stable_regular_file(axis.config_path).sha256, axis.config_sha256, "axis config SHA256")
    _equal(read_stable_regular_file(axis.training_protocol_path).sha256, axis.training_protocol_sha256, "training protocol SHA256")
    _equal(read_stable_regular_file(axis.split_path).sha256, axis.split_sha256, "split SHA256")
    verify_checkpoint_file(axis)


def verify_published_artifact(
    root: Path,
    *,
    expected_axis: CheckpointAxis | None = None,
    required_payloads: Sequence[str] = (),
) -> dict[str, Any]:
    """Recursively rehash and validate a staged or published v2 artifact."""

    root = _absolute_lexical(root)
    manifest_snapshot = read_stable_regular_file(root / "artifact_manifest.json")
    complete_snapshot = read_stable_regular_file(root / "COMPLETE.json")
    try:
        manifest = _mapping(json.loads(manifest_snapshot.data), "artifact manifest")
        complete = _mapping(json.loads(complete_snapshot.data), "artifact COMPLETE")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid artifact metadata below {root}") from error
    _equal(manifest.get("schema_version"), 1, "artifact manifest schema")
    _equal(manifest.get("artifact_contract"), ARTIFACT_CONTRACT, "artifact contract")
    _equal(complete.get("schema_version"), 1, "COMPLETE schema")
    _equal(complete.get("complete"), True, "COMPLETE status")
    _equal(complete.get("artifact_contract"), ARTIFACT_CONTRACT, "COMPLETE contract")
    _equal(complete.get("manifest_sha256"), manifest_snapshot.sha256, "COMPLETE manifest SHA256")

    measured = artifact_tree_ledger(
        root, exclude=("artifact_manifest.json", "COMPLETE.json")
    )
    expected_tree = _mapping(manifest.get("payload_tree"), "manifest payload tree")
    _equal(measured, dict(expected_tree), "recursive payload tree")
    manifest_required = tuple(
        _relative_artifact_path(str(value), label="manifest required payload").as_posix()
        for value in _sequence(manifest.get("required_payloads"), "manifest required payloads")
    )
    requested = tuple(
        _relative_artifact_path(value, label="requested required payload").as_posix()
        for value in required_payloads
    )
    present = {record["path"] for record in measured["files"]}
    missing = sorted((set(manifest_required) | set(requested)) - present)
    if missing:
        raise FileNotFoundError(f"published artifact is missing required payloads: {missing}")

    bound_fields = (
        "artifact_kind",
        "protocol_id",
        "axis_config_sha256",
        "dataset",
        "checkpoint_role",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "split_sha256",
        "payload_tree_sha256",
        "payload_file_count",
    )
    expected_complete = {
        "artifact_kind": manifest["artifact_kind"],
        "protocol_id": manifest["protocol_id"],
        "axis_config_sha256": manifest["axis_config_sha256"],
        "dataset": manifest["dataset"],
        "checkpoint_role": manifest["checkpoint_role"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_epoch": manifest["checkpoint_epoch"],
        "split_sha256": manifest["split_sha256"],
        "payload_tree_sha256": measured["sha256"],
        "payload_file_count": measured["file_count"],
    }
    for field in bound_fields:
        _equal(complete.get(field), expected_complete[field], f"COMPLETE {field}")
    if expected_axis is not None:
        expected = {
            "artifact_kind": expected_axis.artifact_kind,
            "protocol_id": expected_axis.protocol_id,
            "axis_config_sha256": expected_axis.config_sha256,
            "dataset": expected_axis.dataset,
            "checkpoint_role": expected_axis.role,
            "checkpoint_sha256": expected_axis.checkpoint_sha256,
            "checkpoint_epoch": expected_axis.expected_epoch,
            "split_sha256": expected_axis.split_sha256,
        }
        for field, value in expected.items():
            _equal(manifest.get(field), value, f"manifest {field}")
    return {
        "root": str(root),
        "manifest": dict(manifest),
        "complete": dict(complete),
        "manifest_sha256": manifest_snapshot.sha256,
        "complete_sha256": complete_snapshot.sha256,
        "payload_tree": measured,
    }


def finalize_and_publish(
    *,
    staging: Path,
    final: Path,
    axis: CheckpointAxis,
    required_payloads: Sequence[str],
    manifest_extra: Mapping[str, Any] | None = None,
    complete_extra: Mapping[str, Any] | None = None,
    prepublish_guard: Any | None = None,
) -> dict[str, Any]:
    """Write manifest/COMPLETE, verify, and atomically publish without replace."""

    staging = _absolute_lexical(staging)
    final = _absolute_lexical(final)
    _equal(final, _absolute_lexical(axis.output_dir), "canonical publication destination")
    _equal(staging.parent, final.parent, "sibling staging parent")
    if not staging.name.startswith(f".{final.name}.build-"):
        raise ValueError("staging directory does not have the private sibling prefix")
    manifest = build_artifact_manifest(
        staging,
        axis=axis,
        required_payloads=required_payloads,
        extra=manifest_extra,
    )
    _write_json(staging / "artifact_manifest.json", manifest)
    manifest_sha256 = read_stable_regular_file(staging / "artifact_manifest.json").sha256
    complete: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "artifact_contract": ARTIFACT_CONTRACT,
        "artifact_kind": axis.artifact_kind,
        "protocol_id": axis.protocol_id,
        "axis_config_sha256": axis.config_sha256,
        "dataset": axis.dataset,
        "checkpoint_role": axis.role,
        "checkpoint_sha256": axis.checkpoint_sha256,
        "checkpoint_epoch": axis.expected_epoch,
        "split_sha256": axis.split_sha256,
        "manifest_sha256": manifest_sha256,
        "payload_tree_sha256": manifest["payload_tree"]["sha256"],
        "payload_file_count": manifest["payload_tree"]["file_count"],
    }
    if complete_extra:
        overlap = set(complete) & set(complete_extra)
        if overlap:
            raise ValueError(f"COMPLETE extension overwrites protected fields: {sorted(overlap)}")
        complete.update(dict(complete_extra))
    _write_json(staging / "COMPLETE.json", complete)
    verify_published_artifact(
        staging, expected_axis=axis, required_payloads=required_payloads
    )

    def guard() -> None:
        _validate_config_lineage(axis)
        if prepublish_guard is not None:
            prepublish_guard()
        verify_published_artifact(
            staging, expected_axis=axis, required_payloads=required_payloads
        )

    publish_directory_noreplace(staging, final, pre_rename_guard=guard)
    return verify_published_artifact(
        final, expected_axis=axis, required_payloads=required_payloads
    )


def frozen_reference_seal(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact compact reference hashes a global receipt must bind."""

    references = _mapping(
        _mapping(config["parity_gate"], "parity_gate")["references"],
        "parity references",
    )
    clean = _mapping(references["clean"], "clean parity references")
    source = _mapping(references["source"], "source parity references")
    adabn = _mapping(references["adabn"], "adabn parity reference")
    return {
        "clean": {
            dataset: {
                "tree_sha256": clean[dataset]["tree_sha256"],
                "tree_file_count": clean[dataset]["tree_file_count"],
            }
            for dataset in SUPPORTED_DATASETS
        },
        "source": {
            dataset: {
                "tree_sha256": source[dataset]["tree_sha256"],
                "tree_file_count": source[dataset]["tree_file_count"],
                "artifact_manifest_sha256": source[dataset][
                    "artifact_manifest_sha256"
                ],
                "complete_sha256": source[dataset]["complete_sha256"],
            }
            for dataset in SUPPORTED_DATASETS
        },
        "adabn": {
            "aggregate_metrics_file": adabn["aggregate_metrics_file"],
            "aggregate_metrics_sha256": adabn["aggregate_metrics_sha256"],
            "artifact_manifest_file": adabn["artifact_manifest_file"],
            "artifact_manifest_sha256": adabn["artifact_manifest_sha256"],
            "complete_file": adabn["complete_file"],
            "complete_sha256": adabn["complete_sha256"],
            "datasets": {
                dataset: {
                    "tree_sha256": adabn["datasets"][dataset]["tree_sha256"],
                    "tree_file_count": adabn["datasets"][dataset][
                        "tree_file_count"
                    ],
                }
                for dataset in SUPPORTED_DATASETS
            },
        },
    }


def verify_parity_receipt(
    *,
    config: Mapping[str, Any],
    receipt: Path | None = None,
) -> dict[str, Any]:
    """Verify the real, global clean+Source+AdaBN best_miou parity receipt.

    The config's ``pending_runtime_receipt`` value is deliberately not an
    authorization.  Only an existing receipt with ``passed=true`` and exact
    config/reference bindings opens the best_pd gate.
    """

    gate = _mapping(config["parity_gate"], "parity_gate")
    project_root = _config_project_root(config)
    configured_path = _project_path(
        gate["receipt_path"], project_root=project_root, label="parity receipt"
    )
    receipt_path = configured_path if receipt is None else _absolute_lexical(receipt)
    _equal(receipt_path, configured_path, "global parity receipt path")
    if not receipt_path.exists() and not receipt_path.is_symlink():
        raise FileNotFoundError(
            "best_pd is blocked because the global best_miou parity receipt has "
            f"not been published: {receipt_path}"
        )
    snapshot = read_stable_regular_file(receipt_path)
    envelope_root = receipt_path.parent
    manifest_snapshot = read_stable_regular_file(
        envelope_root / "artifact_manifest.json"
    )
    complete_snapshot = read_stable_regular_file(envelope_root / "COMPLETE.json")
    try:
        payload = dict(_mapping(json.loads(snapshot.data), "parity receipt"))
        envelope_manifest = dict(
            _mapping(json.loads(manifest_snapshot.data), "parity artifact manifest")
        )
        envelope_complete = dict(
            _mapping(json.loads(complete_snapshot.data), "parity COMPLETE")
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid parity receipt JSON: {receipt_path}") from error
    _equal(envelope_manifest.get("schema_version"), 1, "parity manifest schema")
    _equal(
        envelope_manifest.get("artifact_contract"),
        PARITY_CONTRACT,
        "parity manifest contract",
    )
    _equal(
        envelope_manifest.get("receipt_type"),
        gate["required_receipt_type"],
        "parity manifest receipt type",
    )
    _equal(
        envelope_manifest.get("axis_config_sha256"),
        _runtime(config)["config_sha256"],
        "parity manifest axis config SHA256",
    )
    _sha256(
        envelope_manifest.get("verifier_sha256"),
        "parity manifest verifier SHA256",
    )
    measured_envelope_tree = artifact_tree_ledger(
        envelope_root, exclude=("artifact_manifest.json", "COMPLETE.json")
    )
    _equal(
        measured_envelope_tree["file_count"],
        1,
        "parity envelope payload file count",
    )
    _equal(
        measured_envelope_tree["files"][0]["path"],
        "PARITY_RECEIPT.json",
        "parity envelope payload name",
    )
    _equal(
        envelope_manifest.get("payload_tree"),
        measured_envelope_tree,
        "parity envelope payload tree",
    )
    _equal(envelope_complete.get("schema_version"), 1, "parity COMPLETE schema")
    _equal(envelope_complete.get("complete"), True, "parity COMPLETE status")
    _equal(
        envelope_complete.get("artifact_contract"),
        PARITY_CONTRACT,
        "parity COMPLETE contract",
    )
    _equal(
        envelope_complete.get("axis_config_sha256"),
        _runtime(config)["config_sha256"],
        "parity COMPLETE axis config SHA256",
    )
    _equal(
        envelope_complete.get("manifest_sha256"),
        manifest_snapshot.sha256,
        "parity COMPLETE manifest SHA256",
    )
    _equal(
        envelope_complete.get("parity_receipt_sha256"),
        snapshot.sha256,
        "parity COMPLETE receipt SHA256",
    )
    _equal(
        envelope_complete.get("payload_tree_sha256"),
        measured_envelope_tree["sha256"],
        "parity COMPLETE payload tree SHA256",
    )
    _equal(
        envelope_complete.get("payload_file_count"),
        measured_envelope_tree["file_count"],
        "parity COMPLETE payload file count",
    )
    expected = {
        "schema_version": gate["receipt_schema_version"],
        "artifact_contract": gate["required_artifact_contract"],
        "status": gate["required_runtime_status"],
        "checkpoint_role": gate["checkpoint_role"],
        "axis_config_sha256": _runtime(config)["config_sha256"],
        "passed": gate["required_runtime_passed"],
        "numeric_tolerance_used": False,
    }
    for key, value in expected.items():
        _equal(payload.get(key), value, f"parity receipt {key}")
    if "receipt_type" in payload:
        _equal(
            payload["receipt_type"],
            gate["required_receipt_type"],
            "parity receipt_type",
        )
    producer = _mapping(payload.get("producer"), "parity producer")
    _equal(
        producer.get("path"),
        "scripts/verify_checkpoint_axis_v2_parity.py",
        "parity producer path",
    )
    producer_sha256 = _sha256(
        producer.get("sha256"), "parity producer SHA256"
    )
    _equal(
        envelope_manifest.get("verifier_sha256"),
        producer_sha256,
        "parity envelope/producer verifier SHA256",
    )
    producer_path = _project_path(
        producer["path"], project_root=project_root, label="parity producer"
    )
    _equal(
        read_stable_regular_file(producer_path).sha256,
        producer_sha256,
        "live parity producer SHA256",
    )
    producer_provenance = _mapping(
        payload.get("candidate_producer_provenance"),
        "candidate producer provenance",
    )
    _equal(
        set(producer_provenance),
        {
            "capture_timing",
            "full_runtime_dependency_sealed",
            "implementation_identity_asserted",
            "adabn_v2_orchestrator_runtime_bound",
            "observation",
        },
        "candidate producer provenance fields",
    )
    _equal(
        producer_provenance.get("capture_timing"),
        "post_run_pre_patch",
        "candidate producer provenance timing",
    )
    for field in (
        "full_runtime_dependency_sealed",
        "implementation_identity_asserted",
        "adabn_v2_orchestrator_runtime_bound",
    ):
        _equal(
            producer_provenance.get(field),
            False,
            f"candidate producer provenance limitation {field}",
        )
    _equal(
        producer_provenance.get("observation"),
        verify_candidate_producer_observation(project_root=project_root),
        "live candidate producer observation",
    )
    parity_assertion = _mapping(
        payload.get("parity_assertion"), "parity assertion"
    )
    _equal(
        dict(parity_assertion),
        {
            "subject": "legacy_best_miou_candidate_scientific_payloads",
            "scientific_payload_bit_exact": True,
            "implementation_identity_asserted": False,
            "applies_to_best_pd_runtime": False,
        },
        "parity assertion semantics",
    )
    receipt_implementation = _mapping(
        payload.get("receipt_verifier_implementation"),
        "receipt verifier implementation",
    )
    _equal(
        set(receipt_implementation),
        {"purpose", "seal"},
        "receipt verifier implementation fields",
    )
    _equal(
        receipt_implementation.get("purpose"),
        "verify_and_publish_exact_output_parity_receipt",
        "receipt verifier implementation purpose",
    )
    authorized_implementation = _mapping(
        payload.get("authorized_best_pd_live_implementation"),
        "authorized best_pd live implementation",
    )
    _equal(
        set(authorized_implementation),
        {
            "purpose",
            "direct_parity_status",
            "guard_only_patch_continuity",
            "guard_only_patch_audit",
            "seal",
        },
        "authorized best_pd live implementation fields",
    )
    _equal(
        authorized_implementation.get("purpose"),
        "authorize_frozen_best_pd_development_axis_without_retuning",
        "authorized best_pd implementation purpose",
    )
    _equal(
        authorized_implementation.get("direct_parity_status"),
        "not_run",
        "authorized best_pd direct parity status",
    )
    _equal(
        authorized_implementation.get("guard_only_patch_continuity"),
        True,
        "authorized best_pd guard-only continuity status",
    )
    _equal(
        authorized_implementation.get("guard_only_patch_audit"),
        checkpoint_axis_guard_only_patch_audit(project_root=project_root),
        "authorized best_pd live guard-only patch audit",
    )
    receipt_seal = _mapping(
        receipt_implementation.get("seal"), "receipt verifier dependency seal"
    )
    authorized_seal = _mapping(
        authorized_implementation.get("seal"),
        "authorized best_pd dependency seal",
    )
    _equal(
        dict(receipt_seal),
        dict(authorized_seal),
        "receipt verifier/authorized best_pd dependency seal",
    )
    live_implementation_seal = verify_implementation_dependency_seal(
        receipt_seal, project_root
    )
    _equal(
        live_implementation_seal,
        dict(receipt_seal),
        "live checkpoint-axis implementation dependency seal",
    )
    implementation_files = {
        str(record["path"]): record
        for record in _sequence(
            live_implementation_seal["files"],
            "live checkpoint-axis implementation files",
        )
    }
    _equal(
        _mapping(
            implementation_files["scripts/verify_checkpoint_axis_v2_parity.py"],
            "live parity signer dependency",
        ).get("sha256"),
        producer_sha256,
        "parity producer/dependency seal SHA256",
    )
    _equal(
        payload.get("frozen_reference_seal"),
        frozen_reference_seal(config),
        "parity frozen reference seal",
    )
    configured_candidate_roots = {
        kind: str(
            _project_path(
                raw,
                project_root=project_root,
                label=f"{kind} parity candidate root",
            )
        )
        for kind, raw in _mapping(
            gate["candidate_roots"], "parity candidate_roots"
        ).items()
    }
    _equal(
        payload.get("candidate_roots"),
        configured_candidate_roots,
        "parity candidate roots",
    )
    for kind in ARTIFACT_KINDS:
        section = _mapping(payload.get(kind), f"parity {kind}")
        _equal(section.get("passed"), True, f"parity {kind} status")
        if kind in {"source", "adabn"}:
            totals = _mapping(section.get("totals"), f"parity {kind} totals")
            _equal(totals.get("conditions"), 39, f"parity {kind} condition count")
        datasets = _mapping(section.get("datasets"), f"parity {kind} datasets")
        _equal(set(datasets), set(SUPPORTED_DATASETS), f"parity {kind} dataset set")
        for dataset, raw_dataset in datasets.items():
            dataset_result = _mapping(raw_dataset, f"parity {kind}/{dataset}")
            seals = _mapping(
                dataset_result.get("candidate_seals"),
                f"parity {kind}/{dataset} candidate seals",
            )
            _sha256(
                seals.get("artifact_manifest.json"),
                f"parity {kind}/{dataset} manifest seal",
            )
            _sha256(
                seals.get("COMPLETE.json"),
                f"parity {kind}/{dataset} COMPLETE seal",
            )
            _sha256(
                seals.get("payload_tree_sha256"),
                f"parity {kind}/{dataset} payload tree seal",
            )
            if int(seals.get("payload_file_count", 0)) < 1:
                raise ValueError(
                    f"parity {kind}/{dataset} payload_file_count must be positive"
                )
            candidate_root = Path(configured_candidate_roots[kind]) / dataset
            verified = verify_published_artifact(candidate_root)
            candidate_manifest = verified["manifest"]
            _equal(
                candidate_manifest.get("artifact_kind"),
                kind,
                f"parity {kind}/{dataset} candidate artifact kind",
            )
            _equal(
                candidate_manifest.get("dataset"),
                dataset,
                f"parity {kind}/{dataset} candidate dataset",
            )
            _equal(
                candidate_manifest.get("checkpoint_role"),
                "best_miou",
                f"parity {kind}/{dataset} candidate role",
            )
            _equal(
                candidate_manifest.get("axis_config_sha256"),
                _runtime(config)["config_sha256"],
                f"parity {kind}/{dataset} candidate config SHA256",
            )
            expected_checkpoint = _mapping(
                _mapping(
                    _mapping(config["parity_anchor"], "parity_anchor")[
                        "checkpoints"
                    ],
                    "parity anchor checkpoints",
                )[dataset],
                f"parity anchor checkpoint {dataset}",
            )
            expected_dataset = _mapping(
                _mapping(config["datasets"], "datasets")[dataset], dataset
            )
            _equal(
                candidate_manifest.get("checkpoint_sha256"),
                expected_checkpoint["sha256"],
                f"parity {kind}/{dataset} candidate checkpoint SHA256",
            )
            _equal(
                candidate_manifest.get("checkpoint_epoch"),
                expected_checkpoint["epoch"],
                f"parity {kind}/{dataset} candidate checkpoint epoch",
            )
            _equal(
                candidate_manifest.get("split_sha256"),
                expected_dataset["test_split_sha256"],
                f"parity {kind}/{dataset} candidate split SHA256",
            )
            _equal(
                seals["artifact_manifest.json"],
                verified["manifest_sha256"],
                f"parity {kind}/{dataset} live manifest seal",
            )
            _equal(
                seals["COMPLETE.json"],
                verified["complete_sha256"],
                f"parity {kind}/{dataset} live COMPLETE seal",
            )
            _equal(
                seals["payload_tree_sha256"],
                verified["payload_tree"]["sha256"],
                f"parity {kind}/{dataset} live payload tree seal",
            )
            _equal(
                seals["payload_file_count"],
                verified["payload_tree"]["file_count"],
                f"parity {kind}/{dataset} live payload file count",
            )
    comparison = _mapping(payload.get("comparison_contract"), "parity comparison contract")
    required_checks = {
        "ordered_ids_exact",
        "float32_probability_arrays_bit_exact",
        "probability_file_sha256_exact",
        "binary_png_bytes_exact",
        "integer_sufficient_statistics_exact",
        "official_metrics_exact",
        "unified_metrics_exact",
    }
    _equal(set(comparison), required_checks, "parity comparison keys")
    failed = sorted(key for key in required_checks if comparison.get(key) is not True)
    if failed:
        raise ValueError(f"parity receipt contains failed comparison checks: {failed}")
    payload["_receipt_path"] = str(receipt_path)
    payload["_receipt_sha256"] = snapshot.sha256
    return payload


def remove_private_staging(staging: Path, *, final: Path) -> None:
    """Remove only an unpublished sibling staging tree created by this module."""

    staging = _absolute_lexical(staging)
    final = _absolute_lexical(final)
    if (
        staging.parent != final.parent
        or not staging.name.startswith(f".{final.name}.build-")
        or not staging.exists()
        or staging.is_symlink()
    ):
        return
    shutil.rmtree(staging)


__all__ = [
    "ARTIFACT_CONTRACT",
    "ARTIFACT_KINDS",
    "CheckpointAxis",
    "CheckpointRole",
    "DEFAULT_AXIS_CONFIG",
    "PARITY_CONTRACT",
    "PROJECT_ROOT",
    "SUPPORTED_DATASETS",
    "TREE_ALGORITHM",
    "artifact_tree_ledger",
    "build_artifact_manifest",
    "canonical_output_dir",
    "finalize_and_publish",
    "frozen_reference_seal",
    "get_parity_reference",
    "load_and_verify_checkpoint",
    "load_axis_config",
    "load_checkpoint_into_model",
    "prepare_staging",
    "remove_private_staging",
    "resolve_axis",
    "validate_checkpoint_payload",
    "verify_checkpoint_file",
    "verify_parity_receipt",
    "verify_published_artifact",
]
