#!/usr/bin/env python3
"""Analyze Binary-TENT-SS no-op levels on provenance-bound train episodes.

Input is a JSON manifest whose episode arrays are precomputed ``.npz`` files.
The analyzer never invokes a method.  It reads each target only in the outer
diagnostic phase and refuses any non-train role, incomplete train provenance,
test pixel access, exposed method target, unhashed input, symlink, or path
outside ``--project-root``.

Minimal manifest shape (all referenced files require ``path`` and ``sha256``)::

    {
      "schema_version": 1,
      "artifact_type": "binary_tent_ss_noop_train_analysis_input",
      "scope": {
        "split_role": "train",
        "source_train_derived": true,
        "oracle_analysis": true,
        "paper_test_result": false,
        "use_test_images": false,
        "use_test_labels": false
      },
      "label_boundary": {
        "method_label_accesses": 0,
        "targets_outer_evaluator_only": true
      },
      "thresholds": { ... every NoOpThresholds field ... },
      "provenance": {
        "train_provenance_complete": true,
        "datasets": {
          "DATASET": {
            "train_split": {"path": "...", "sha256": "..."},
            "pilot_ids": {"path": "...", "sha256": "..."},
            "pilot_manifest": {"path": "...", "sha256": "..."},
            "cache_manifest": {"path": "...", "sha256": "..."},
            "method_input_manifest": {"path": "...", "sha256": "..."},
            "cache_complete": {"path": "...", "sha256": "..."}
          }
        }
      },
      "episodes": [{
        "episode_id": "...", "dataset": "DATASET", "split_role": "train",
        "image_id": "...", "arrays": {"path": "...", "sha256": "..."},
        "logits_pre_key": "logits_pre", "logits_post_key": "logits_post",
        "target_key": "target",
        "parameters": [{"name": "...", "pre_key": "...", "post_key": "..."}]
      }]
    }

Optional ``probability_pre_key`` and ``probability_post_key`` must occur
together; supplied probabilities are checked against sigmoid(logits).
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tta.diagnostics import (
    MARGIN_STRATA,
    NOOP_CLASSIFICATION_ORDER,
    NoOpThresholds,
    analyze_noop_episode,
)


INPUT_ARTIFACT_TYPE = "binary_tent_ss_noop_train_analysis_input"
OUTPUT_ARTIFACT_TYPE = "binary_tent_ss_noop_train_analysis"


class AnalysisInputError(RuntimeError):
    """Raised before publication when provenance or input bytes are unsafe."""


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    project_relative: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.project_relative,
            "sha256": self.sha256,
            "size_bytes": self.path.stat().st_size,
        }


@dataclass(frozen=True)
class DatasetProvenance:
    dataset: str
    train_ids: frozenset[str]
    pilot_ids: tuple[str, ...]
    cache_ids: tuple[str, ...]
    references: tuple[ArtifactRef, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_image_count": len(self.cache_ids),
            "dataset": self.dataset,
            "pilot_image_count": len(self.pilot_ids),
            "references": [reference.to_dict() for reference in self.references],
            "train_image_count": len(self.train_ids),
        }


@dataclass(frozen=True)
class ParameterKeys:
    name: str
    pre_key: str
    post_key: str


@dataclass(frozen=True)
class EpisodeInput:
    episode_id: str
    dataset: str
    image_id: str
    arrays: ArtifactRef
    logits_pre_key: str
    logits_post_key: str
    target_key: str
    probability_pre_key: str | None
    probability_post_key: str | None
    parameters: tuple[ParameterKeys, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class AnalysisRun:
    input_manifest: ArtifactRef
    thresholds: NoOpThresholds
    dataset_provenance: tuple[DatasetProvenance, ...]
    records: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]
    summary: Mapping[str, Any]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisInputError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise AnalysisInputError(f"JSON root must be an object: {path}")
    return value


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise AnalysisInputError(f"path is outside project root: {path}") from error
    text = relative.as_posix()
    if not text or "\n" in text or "\r" in text:
        raise AnalysisInputError(f"unsafe project-relative path: {text!r}")
    return text


def _artifact_ref(
    value: Any,
    *,
    label: str,
    project_root: Path,
) -> ArtifactRef:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise AnalysisInputError(
            f"{label} must contain exactly path and sha256"
        )
    relative_text = value.get("path")
    expected_hash = value.get("sha256")
    if not isinstance(relative_text, str) or not relative_text:
        raise AnalysisInputError(f"{label}.path must be a non-empty string")
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise AnalysisInputError(f"{label}.path must be safe and project-relative")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise AnalysisInputError(f"{label}.sha256 is invalid")
    path = project_root / Path(relative.as_posix())
    _project_relative(path, project_root)
    if path.is_symlink() or not path.is_file():
        raise AnalysisInputError(f"{label} is missing or a symlink: {relative_text}")
    if sha256_file(path) != expected_hash:
        raise AnalysisInputError(f"{label} SHA-256 mismatch: {relative_text}")
    return ArtifactRef(
        path=path,
        project_relative=relative.as_posix(),
        sha256=expected_hash,
    )


def _read_ids(reference: ArtifactRef, *, label: str) -> tuple[str, ...]:
    try:
        raw_lines = reference.path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise AnalysisInputError(f"cannot read {label}") from error
    ids = tuple(line.strip() for line in raw_lines if line.strip())
    if not ids or len(ids) != len(set(ids)):
        raise AnalysisInputError(f"{label} must contain unique non-empty IDs")
    return ids


def _validate_pilot_manifest(
    value: Mapping[str, Any],
    *,
    dataset: str,
    train_split: ArtifactRef,
    pilot_ids: ArtifactRef,
) -> None:
    if value.get("paper_result") is not False or value.get("no_validation_split") is not True:
        raise AnalysisInputError("pilot manifest is not train-side non-paper evidence")
    boundary = value.get("metadata_io_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("test_pixels_opened") != 0:
        raise AnalysisInputError("pilot manifest does not prove zero test-pixel opens")
    datasets = value.get("datasets")
    entry = datasets.get(dataset) if isinstance(datasets, Mapping) else None
    if not isinstance(entry, Mapping):
        raise AnalysisInputError(f"pilot manifest has no dataset {dataset}")
    checks = entry.get("checks")
    required_checks = (
        "output_ids_contained_in_train",
        "output_ids_unique",
        "output_test_overlap_count",
        "train_ids_unique",
        "train_test_overlap_count",
    )
    if not isinstance(checks, Mapping) or any(name not in checks for name in required_checks):
        raise AnalysisInputError(f"pilot manifest checks incomplete for {dataset}")
    if (
        checks["output_ids_contained_in_train"] is not True
        or checks["output_ids_unique"] is not True
        or checks["train_ids_unique"] is not True
        or checks["output_test_overlap_count"] != 0
        or checks["train_test_overlap_count"] != 0
    ):
        raise AnalysisInputError(f"pilot manifest train/test checks failed for {dataset}")
    output = entry.get("output")
    train = entry.get("train_split")
    if (
        not isinstance(output, Mapping)
        or output.get("path") != pilot_ids.project_relative
        or output.get("file_sha256") != pilot_ids.sha256
        or not isinstance(train, Mapping)
        or train.get("path") != train_split.project_relative
        or train.get("sha256") != train_split.sha256
    ):
        raise AnalysisInputError(f"pilot manifest lineage mismatch for {dataset}")


def _validate_cache_lineage(
    *,
    dataset: str,
    cache_manifest_ref: ArtifactRef,
    method_manifest_ref: ArtifactRef,
    complete_ref: ArtifactRef,
    train_split: ArtifactRef,
    pilot_ids: ArtifactRef,
) -> tuple[str, ...]:
    cache = _load_json_object(cache_manifest_ref.path)
    if cache.get("dataset") != dataset:
        raise AnalysisInputError(f"cache dataset mismatch for {dataset}")
    split_role = cache.get("split_role")
    if not isinstance(split_role, str) or not split_role.startswith("train_side"):
        raise AnalysisInputError(f"cache is not train-side for {dataset}")
    if (
        cache.get("train_split") != train_split.project_relative
        or cache.get("train_split_sha256") != train_split.sha256
        or cache.get("calibration_ids_path") != pilot_ids.project_relative
        or cache.get("calibration_ids_file_sha256") != pilot_ids.sha256
    ):
        raise AnalysisInputError(f"cache train lineage mismatch for {dataset}")
    firewall = cache.get("label_firewall")
    open_scope = cache.get("source_open_scope")
    if (
        not isinstance(firewall, Mapping)
        or firewall.get("method_received_labels") is not False
        or firewall.get("targets_for_outer_evaluator_only") is not True
        or not isinstance(open_scope, Mapping)
        or open_scope.get("test_images") != 0
        or open_scope.get("test_masks") != 0
    ):
        raise AnalysisInputError(f"cache label/test firewall failed for {dataset}")
    cache_ids = cache.get("image_ids")
    if (
        not isinstance(cache_ids, list)
        or not cache_ids
        or not all(isinstance(item, str) and item for item in cache_ids)
        or len(cache_ids) != len(set(cache_ids))
    ):
        raise AnalysisInputError(f"cache image IDs are invalid for {dataset}")

    method = _load_json_object(method_manifest_ref.path)
    forbidden = method.get("forbidden_fields")
    if (
        method.get("dataset") != dataset
        or method.get("targets_exposed") is not False
        or not isinstance(forbidden, list)
        or not {"ground_truth", "gt", "label", "mask", "target"}.issubset(
            set(forbidden)
        )
        or method.get("outer_manifest_sha256") != cache_manifest_ref.sha256
    ):
        raise AnalysisInputError(f"method-input label firewall failed for {dataset}")

    complete = _load_json_object(complete_ref.path)
    if (
        complete.get("complete") is not True
        or complete.get("dataset") != dataset
        or complete.get("manifest_sha256") != cache_manifest_ref.sha256
        or complete.get("method_input_manifest_sha256") != method_manifest_ref.sha256
        or complete.get("method_received_labels") is not False
        or complete.get("test_images_opened") != 0
        or complete.get("test_masks_opened") != 0
    ):
        raise AnalysisInputError(f"cache COMPLETE safety checks failed for {dataset}")
    return tuple(cache_ids)


def _parse_dataset_provenance(
    dataset: str,
    value: Any,
    *,
    project_root: Path,
) -> DatasetProvenance:
    required = {
        "train_split",
        "pilot_ids",
        "pilot_manifest",
        "cache_manifest",
        "method_input_manifest",
        "cache_complete",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise AnalysisInputError(
            f"dataset provenance fields must be exact for {dataset}"
        )
    references = {
        name: _artifact_ref(
            value[name], label=f"provenance.datasets[{dataset}].{name}", project_root=project_root
        )
        for name in sorted(required)
    }
    train_split = references["train_split"]
    basename = PurePosixPath(train_split.project_relative).name
    if not basename.startswith("train_") or dataset not in basename:
        raise AnalysisInputError(f"train split path is not canonical for {dataset}")
    train_ids = _read_ids(train_split, label=f"{dataset} train split")
    pilot_ids = _read_ids(references["pilot_ids"], label=f"{dataset} pilot IDs")
    if not set(pilot_ids).issubset(set(train_ids)):
        raise AnalysisInputError(f"pilot IDs are not contained in train for {dataset}")

    pilot_manifest_value = _load_json_object(references["pilot_manifest"].path)
    _validate_pilot_manifest(
        pilot_manifest_value,
        dataset=dataset,
        train_split=train_split,
        pilot_ids=references["pilot_ids"],
    )
    cache_ids = _validate_cache_lineage(
        dataset=dataset,
        cache_manifest_ref=references["cache_manifest"],
        method_manifest_ref=references["method_input_manifest"],
        complete_ref=references["cache_complete"],
        train_split=train_split,
        pilot_ids=references["pilot_ids"],
    )
    if tuple(pilot_ids) != cache_ids:
        raise AnalysisInputError(
            f"cache image order differs from frozen pilot order for {dataset}"
        )
    return DatasetProvenance(
        dataset=dataset,
        train_ids=frozenset(train_ids),
        pilot_ids=pilot_ids,
        cache_ids=cache_ids,
        references=tuple(references[name] for name in sorted(references)),
    )


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisInputError(f"{label} must be a non-empty string")
    return value


def _parse_episode(
    value: Any,
    *,
    index: int,
    project_root: Path,
    provenance: Mapping[str, DatasetProvenance],
) -> EpisodeInput:
    if not isinstance(value, Mapping):
        raise AnalysisInputError(f"episodes[{index}] must be an object")
    if value.get("split_role") != "train":
        raise AnalysisInputError(f"episodes[{index}] is not train role")
    dataset = _nonempty_string(value.get("dataset"), label=f"episodes[{index}].dataset")
    if dataset not in provenance:
        raise AnalysisInputError(f"episodes[{index}] has unknown dataset {dataset}")
    image_id = _nonempty_string(value.get("image_id"), label=f"episodes[{index}].image_id")
    if (
        image_id not in provenance[dataset].train_ids
        or image_id not in provenance[dataset].pilot_ids
        or image_id not in provenance[dataset].cache_ids
    ):
        raise AnalysisInputError(
            f"episodes[{index}] image is not bound to frozen train provenance"
        )
    parameters_value = value.get("parameters")
    if not isinstance(parameters_value, list) or not parameters_value:
        raise AnalysisInputError(f"episodes[{index}].parameters cannot be empty")
    parameters: list[ParameterKeys] = []
    parameter_names: set[str] = set()
    array_keys: set[str] = set()
    for parameter_index, item in enumerate(parameters_value):
        if not isinstance(item, Mapping) or set(item) != {"name", "pre_key", "post_key"}:
            raise AnalysisInputError(
                f"episodes[{index}].parameters[{parameter_index}] fields are invalid"
            )
        name = _nonempty_string(item["name"], label="parameter name")
        pre_key = _nonempty_string(item["pre_key"], label="parameter pre_key")
        post_key = _nonempty_string(item["post_key"], label="parameter post_key")
        if name in parameter_names or pre_key == post_key or pre_key in array_keys or post_key in array_keys:
            raise AnalysisInputError(
                f"episodes[{index}] has duplicate parameter names/array keys"
            )
        parameter_names.add(name)
        array_keys.update((pre_key, post_key))
        parameters.append(ParameterKeys(name=name, pre_key=pre_key, post_key=post_key))

    required_keys = {
        "logits_pre_key",
        "logits_post_key",
        "target_key",
    }
    scalar_keys = {
        name: _nonempty_string(value.get(name), label=f"episodes[{index}].{name}")
        for name in required_keys
    }
    probability_pre = value.get("probability_pre_key")
    probability_post = value.get("probability_post_key")
    if (probability_pre is None) != (probability_post is None):
        raise AnalysisInputError(
            f"episodes[{index}] probability keys must occur together"
        )
    if probability_pre is not None:
        probability_pre = _nonempty_string(probability_pre, label="probability_pre_key")
        probability_post = _nonempty_string(probability_post, label="probability_post_key")
    declared_keys = list(array_keys) + list(scalar_keys.values())
    if probability_pre is not None:
        declared_keys.extend((probability_pre, probability_post))
    if len(declared_keys) != len(set(declared_keys)):
        raise AnalysisInputError(f"episodes[{index}] reuses an NPZ array key")

    reserved = {
        "episode_id",
        "dataset",
        "split_role",
        "image_id",
        "arrays",
        "logits_pre_key",
        "logits_post_key",
        "target_key",
        "probability_pre_key",
        "probability_post_key",
        "parameters",
        "metadata",
    }
    unknown = sorted(set(value) - reserved)
    if unknown:
        raise AnalysisInputError(f"episodes[{index}] has unknown fields: {unknown}")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise AnalysisInputError(f"episodes[{index}].metadata must be an object")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AnalysisInputError(
            f"episodes[{index}].metadata is not finite JSON"
        ) from error
    return EpisodeInput(
        episode_id=_nonempty_string(
            value.get("episode_id"), label=f"episodes[{index}].episode_id"
        ),
        dataset=dataset,
        image_id=image_id,
        arrays=_artifact_ref(
            value.get("arrays"),
            label=f"episodes[{index}].arrays",
            project_root=project_root,
        ),
        logits_pre_key=scalar_keys["logits_pre_key"],
        logits_post_key=scalar_keys["logits_post_key"],
        target_key=scalar_keys["target_key"],
        probability_pre_key=probability_pre,
        probability_post_key=probability_post,
        parameters=tuple(parameters),
        metadata=dict(metadata),
    )


def _output_metadata(*, episode_count: int) -> dict[str, Any]:
    if episode_count <= 0:
        raise AnalysisInputError("analysis requires at least one train episode")
    return {
        "method_label_accesses": 0,
        "oracle_analysis": True,
        "outer_evaluator_label_accesses": episode_count,
        "paper_result": False,
        "paper_test_result": False,
        "source_train_derived": True,
        "split_role": "train",
        "test_image_opens": 0,
        "test_label_opens": 0,
    }


def _load_episode_arrays(
    episode: EpisodeInput,
    *,
    thresholds: NoOpThresholds,
) -> dict[str, Any]:
    try:
        archive = np.load(episode.arrays.path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise AnalysisInputError(
            f"cannot open safe NPZ for episode {episode.episode_id}"
        ) from error
    with archive:
        declared = {
            episode.logits_pre_key,
            episode.logits_post_key,
            episode.target_key,
            *(key for item in episode.parameters for key in (item.pre_key, item.post_key)),
        }
        if episode.probability_pre_key is not None:
            assert episode.probability_post_key is not None
            declared.update(
                (episode.probability_pre_key, episode.probability_post_key)
            )
        if set(archive.files) != declared:
            raise AnalysisInputError(
                f"episode {episode.episode_id} NPZ fields differ from declaration"
            )
        parameter_pre = {
            item.name: np.array(archive[item.pre_key], copy=True)
            for item in episode.parameters
        }
        parameter_post = {
            item.name: np.array(archive[item.post_key], copy=True)
            for item in episode.parameters
        }
        logits_pre = np.array(archive[episode.logits_pre_key], copy=True)
        logits_post = np.array(archive[episode.logits_post_key], copy=True)
        probability_pre = (
            None
            if episode.probability_pre_key is None
            else np.array(archive[episode.probability_pre_key], copy=True)
        )
        probability_post = (
            None
            if episode.probability_post_key is None
            else np.array(archive[episode.probability_post_key], copy=True)
        )
        # Target access deliberately occurs last, in this outer analyzer only.
        target = np.array(archive[episode.target_key], copy=True)
    try:
        return analyze_noop_episode(
            parameter_pre=parameter_pre,
            parameter_post=parameter_post,
            logits_pre=logits_pre,
            logits_post=logits_post,
            probability_pre=probability_pre,
            probability_post=probability_post,
            target=target,
            thresholds=thresholds,
        )
    except (TypeError, ValueError) as error:
        raise AnalysisInputError(
            f"episode diagnostic failed for {episode.episode_id}: {error}"
        ) from error


def _summary(records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    classification_counts = Counter(str(record["classification"]) for record in records)
    margin: dict[str, Any] = {}
    for stratum in MARGIN_STRATA:
        pixel_count = sum(
            int(record["threshold_margin"]["strata"][stratum]["pixel_count"])
            for record in records
        )
        gt_margin = sum(
            int(
                record["threshold_margin"]["strata"][stratum][
                    "abs_delta_gt_margin_count"
                ]
            )
            for record in records
        )
        gt_tenth = sum(
            int(
                record["threshold_margin"]["strata"][stratum][
                    "abs_delta_gt_0_1_margin_count"
                ]
            )
            for record in records
        )
        margin[stratum] = {
            "abs_delta_gt_0_1_margin_count": gt_tenth,
            "abs_delta_gt_margin_count": gt_margin,
            "fraction_abs_delta_gt_0_1_margin": (
                float(gt_tenth / pixel_count) if pixel_count else None
            ),
            "fraction_abs_delta_gt_margin": (
                float(gt_margin / pixel_count) if pixel_count else None
            ),
            "pixel_count": pixel_count,
        }
    return {
        "artifact_type": OUTPUT_ARTIFACT_TYPE + "_summary",
        "binary_transition_totals": {
            name: sum(int(record["binary_transitions"][name]) for record in records)
            for name in (
                "binary_pixel_xor_count",
                "BG_to_FG_pixel_count",
                "FG_to_BG_pixel_count",
            )
        },
        "classification_counts": {
            name: classification_counts.get(name, 0)
            for name in NOOP_CLASSIFICATION_ORDER
        },
        "classification_priority": list(NOOP_CLASSIFICATION_ORDER),
        "episode_count": len(records),
        "margin_aggregate": margin,
        **metadata,
        "schema_version": 1,
    }


def analyze_input_manifest(
    manifest_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> AnalysisRun:
    """Validate train provenance, load episodes, and return in-memory results."""

    project_root = project_root.resolve()
    manifest_path = manifest_path if manifest_path.is_absolute() else project_root / manifest_path
    manifest_path = manifest_path.resolve()
    relative = _project_relative(manifest_path, project_root)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AnalysisInputError(f"input manifest is missing or a symlink: {relative}")
    manifest_ref = ArtifactRef(
        path=manifest_path,
        project_relative=relative,
        sha256=sha256_file(manifest_path),
    )
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != INPUT_ARTIFACT_TYPE:
        raise AnalysisInputError("unsupported analysis input schema/artifact type")
    scope = manifest.get("scope")
    required_scope = {
        "split_role": "train",
        "source_train_derived": True,
        "oracle_analysis": True,
        "paper_test_result": False,
        "use_test_images": False,
        "use_test_labels": False,
    }
    if not isinstance(scope, Mapping) or any(
        scope.get(name) != expected for name, expected in required_scope.items()
    ):
        raise AnalysisInputError("input scope is not strict source-train-only oracle analysis")
    boundary = manifest.get("label_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("method_label_accesses") != 0
        or boundary.get("targets_outer_evaluator_only") is not True
    ):
        raise AnalysisInputError("input label boundary is incomplete")
    try:
        thresholds = NoOpThresholds.from_mapping(manifest.get("thresholds"))
    except (TypeError, ValueError) as error:
        raise AnalysisInputError(f"invalid frozen thresholds: {error}") from error

    provenance_value = manifest.get("provenance")
    if (
        not isinstance(provenance_value, Mapping)
        or provenance_value.get("train_provenance_complete") is not True
        or not isinstance(provenance_value.get("datasets"), Mapping)
        or not provenance_value["datasets"]
    ):
        raise AnalysisInputError("complete train provenance is required")
    dataset_provenance = tuple(
        _parse_dataset_provenance(
            dataset,
            provenance_value["datasets"][dataset],
            project_root=project_root,
        )
        for dataset in sorted(provenance_value["datasets"])
        if isinstance(dataset, str) and dataset
    )
    if len(dataset_provenance) != len(provenance_value["datasets"]):
        raise AnalysisInputError("dataset provenance keys must be non-empty strings")
    provenance_by_dataset = {
        value.dataset: value for value in dataset_provenance
    }

    episodes_value = manifest.get("episodes")
    if not isinstance(episodes_value, list) or not episodes_value:
        raise AnalysisInputError("input manifest requires at least one train episode")
    episodes = tuple(
        _parse_episode(
            value,
            index=index,
            project_root=project_root,
            provenance=provenance_by_dataset,
        )
        for index, value in enumerate(episodes_value)
    )
    episode_ids = [episode.episode_id for episode in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise AnalysisInputError("episode IDs must be unique")

    metadata = _output_metadata(episode_count=len(episodes))
    records: list[dict[str, Any]] = []
    for episode in episodes:
        diagnostic = _load_episode_arrays(episode, thresholds=thresholds)
        records.append(
            {
                "array_artifact": episode.arrays.to_dict(),
                "dataset": episode.dataset,
                "episode_id": episode.episode_id,
                "episode_metadata": dict(episode.metadata),
                "image_id": episode.image_id,
                **diagnostic,
                **metadata,
            }
        )
    summary = _summary(records, metadata)
    return AnalysisRun(
        input_manifest=manifest_ref,
        thresholds=thresholds,
        dataset_provenance=dataset_provenance,
        records=tuple(records),
        metadata=metadata,
        summary=summary,
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for value in values
    ).encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _provenance_payload(run: AnalysisRun) -> dict[str, Any]:
    return {
        "artifact_type": OUTPUT_ARTIFACT_TYPE + "_provenance",
        "datasets": [value.to_dict() for value in run.dataset_provenance],
        "input_manifest": run.input_manifest.to_dict(),
        **run.metadata,
        "schema_version": 1,
        "thresholds": run.thresholds.to_dict(),
    }


def _manifest_payload(root: Path, run: AnalysisRun) -> dict[str, Any]:
    file_names = ("episode_diagnostics.jsonl", "summary.json", "provenance.json")
    return {
        "artifact_type": OUTPUT_ARTIFACT_TYPE + "_artifact_manifest",
        "episode_count": len(run.records),
        "files": {
            name: {
                "sha256": sha256_file(root / name),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in file_names
        },
        **run.metadata,
        "schema_version": 1,
    }


def verify_output(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    if directory.is_symlink() or not directory.is_dir():
        raise AnalysisInputError("analysis output directory is missing or a symlink")
    manifest = _load_json_object(directory / "artifact_manifest.json")
    complete = _load_json_object(directory / "COMPLETE.json")
    required_metadata = _output_metadata(episode_count=int(manifest.get("episode_count", 0)))
    for value, label in ((manifest, "artifact manifest"), (complete, "COMPLETE")):
        if any(value.get(name) != expected for name, expected in required_metadata.items()):
            raise AnalysisInputError(f"{label} mandatory metadata mismatch")
    if complete.get("complete") is not True or complete.get("artifact_manifest_sha256") != sha256_file(
        directory / "artifact_manifest.json"
    ):
        raise AnalysisInputError("analysis COMPLETE binding failed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "episode_diagnostics.jsonl",
        "summary.json",
        "provenance.json",
    }:
        raise AnalysisInputError("analysis artifact manifest member set is invalid")
    expected_members = set(files) | {"artifact_manifest.json", "COMPLETE.json"}
    actual_members = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_members != expected_members or any(path.is_symlink() for path in directory.iterdir()):
        raise AnalysisInputError("analysis output has missing/extra/non-regular members")
    for name, expected in files.items():
        path = directory / name
        if (
            not isinstance(expected, Mapping)
            or expected.get("sha256") != sha256_file(path)
            or expected.get("size_bytes") != path.stat().st_size
        ):
            raise AnalysisInputError(f"analysis output checksum mismatch: {name}")
    return {
        "episode_count": manifest["episode_count"],
        "oracle_analysis": True,
        "paper_test_result": False,
        "status": "verified",
    }


def publish_analysis(run: AnalysisRun, output_directory: Path, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_directory = output_directory if output_directory.is_absolute() else project_root / output_directory
    output_directory = output_directory.resolve()
    _project_relative(output_directory, project_root)
    if output_directory.exists() or output_directory.is_symlink():
        raise AnalysisInputError("analysis output already exists; overwrite is forbidden")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_directory.name}.tmp-", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        _write_exclusive(
            temporary / "episode_diagnostics.jsonl", _jsonl_bytes(run.records)
        )
        _write_exclusive(temporary / "summary.json", _json_bytes(run.summary))
        _write_exclusive(
            temporary / "provenance.json", _json_bytes(_provenance_payload(run))
        )
        manifest = _manifest_payload(temporary, run)
        _write_exclusive(
            temporary / "artifact_manifest.json", _json_bytes(manifest)
        )
        complete = {
            "artifact_manifest_sha256": sha256_file(
                temporary / "artifact_manifest.json"
            ),
            "complete": True,
            "episode_count": len(run.records),
            **run.metadata,
            "schema_version": 1,
        }
        _write_exclusive(temporary / "COMPLETE.json", _json_bytes(complete))
        verify_output(temporary)
        if output_directory.exists() or output_directory.is_symlink():
            raise AnalysisInputError("analysis destination appeared during publish")
        try:
            os.rename(temporary, output_directory)
        except OSError as error:
            raise AnalysisInputError("atomic analysis publication failed") from error
    return verify_output(output_directory) | {"status": "created_and_verified"}


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    try:
        if args.verify_only is not None:
            if args.input_manifest is not None or args.output_dir is not None or args.dry_run:
                raise AnalysisInputError(
                    "--verify-only cannot be combined with analysis arguments"
                )
            output = args.verify_only if args.verify_only.is_absolute() else project_root / args.verify_only
            result = verify_output(output)
        else:
            if args.input_manifest is None:
                raise AnalysisInputError("--input-manifest is required")
            if not args.dry_run and args.output_dir is None:
                raise AnalysisInputError("--output-dir is required unless --dry-run")
            run = analyze_input_manifest(
                args.input_manifest, project_root=project_root
            )
            if args.dry_run:
                result = {
                    "classification_counts": run.summary["classification_counts"],
                    "episode_count": len(run.records),
                    **run.metadata,
                    "status": "dry_run_validated_no_writes",
                }
            else:
                assert args.output_dir is not None
                result = publish_analysis(
                    run, args.output_dir, project_root=project_root
                )
    except AnalysisInputError as error:
        print(f"no-op analysis refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
