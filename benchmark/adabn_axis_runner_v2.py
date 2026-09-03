"""Checkpoint-axis v2 orchestration for the episodic AdaBN failure control.

This module deliberately leaves :mod:`run_adabn_corruption_benchmark` (v1)
untouched.  It reuses only the frozen numerical condition body and publishes a
new, role-first development artifact with an independent provenance chain.

The only canonical formal role in this protocol is ``best_pd``.  A
``best_miou`` run is accepted only as an explicitly named, non-canonical parity
run and must compare bit-for-bit with the sealed v1 AdaBN condition artifacts.
Both roles are test-selected development evidence: ``main_paper_table`` is
always false and no additional tuning episode is permitted.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
import yaml

from benchmark import checkpoint_axis
from benchmark import source_corruption_axis_runner_v2 as source_axis_v2
from dataio.corruption_cache import (
    CachedCorruptionDataset,
    condition_key,
    ordered_ids_sha256,
    sha256_file,
    verify_cache_artifact,
)
from metrics.irstd_metrics import IRSTDEvaluationProtocol
import run_adabn_corruption_benchmark as legacy
import run_source_corruption_benchmark as source_v1
import test_source as source_runner
from tta.adabn_fast_runner import AdaBNFastRunner
from tta.episodic_runner import EpisodicRunner
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager
from tta.d0_secure_io import publish_directory_noreplace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AXIS_CONFIG = PROJECT_ROOT / "configs" / "checkpoint_axis_best_pd_v1.yaml"
FROZEN_ADABN_V1_CONFIG = (
    PROJECT_ROOT / "configs" / "adabn_batch_stats_fixed_splits_v1.yaml"
)
FROZEN_ADABN_V1_CONFIG_SHA256 = (
    "62d53884578996641b5677d8c61686984c415b28656148129ae3019b3f88b163"
)
FROZEN_ADABN_V1_RUNNER = PROJECT_ROOT / "run_adabn_corruption_benchmark.py"
FROZEN_ADABN_V1_RUNNER_SHA256 = (
    "40fe023c74fde6e0f2fac1318fc5156bd165fbb200e3bf7559824325b6b4c612"
)
PROTOCOL_ID = "cr-sitta-adabn-checkpoint-axis-v2"
ARTIFACT_KIND = "adabn"
SCHEMA_VERSION = 2
DATASETS = legacy.DATASETS
CONDITIONS = legacy.CONDITIONS
CONDITION_KEYS = tuple(condition_key(*item) for item in CONDITIONS)
ROLE_BEST_MIOU = "best_miou"
ROLE_BEST_PD = "best_pd"
ROLES = (ROLE_BEST_MIOU, ROLE_BEST_PD)
CONDITION_SENTINEL = "CONDITION_COMPLETE.json"
DATASET_SENTINEL = "COMPLETE.json"
GLOBAL_SENTINEL = "COMPLETE.json"


@dataclass(frozen=True)
class SourceArtifact:
    root: Path
    benchmark: Mapping[str, Any]
    manifest: Mapping[str, Any]
    completion: Mapping[str, Any]
    manifest_sha256: str
    completion_sha256: str
    benchmark_sha256: str
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class DatasetContext:
    axis: Any
    source_axis: Any
    axis_config_path: Path
    axis_config: Mapping[str, Any]
    axis_config_sha256: str
    checkpoint_payload: Mapping[str, Any]
    checkpoint_summary: Mapping[str, Any]
    cache_dir: Path
    cache_manifest: Mapping[str, Any]
    cache_audit: Mapping[str, Any]
    source_artifact: SourceArtifact
    legacy_context: legacy.DatasetContext
    legacy_protocol: Mapping[str, Any]
    legacy_source_protocol: Mapping[str, Any]


@dataclass(frozen=True)
class RunSelection:
    checkpoint_role: Literal["best_miou", "best_pd"]
    datasets: tuple[str, ...]
    conditions: tuple[tuple[str, int], ...]
    output_root: Path
    formal_development_artifact: bool
    parity_only: bool
    max_images: int | None
    aggregate_only: bool
    source_root_override: Path | None


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular JSON file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return dict(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular JSONL file is missing: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"expected JSON object at {path}:{line_number}")
            records.append(dict(value))
    return records


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_bytes(_canonical_json_bytes(value))
    os.replace(temporary, path)


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(
        yaml.safe_dump(dict(value), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _link_file_noreplace(staged_file: Path, final: Path) -> None:
    """Publish one regular file without replacing any existing directory entry."""

    final.parent.mkdir(parents=True, exist_ok=True)
    os.link(staged_file, final, follow_symlinks=False)


def _link_file_recorded(
    staged_file: Path,
    final: Path,
    linked: list[tuple[Path, int, int]],
) -> None:
    """No-replace link one file and record its inode for exact rollback."""

    source_stat = staged_file.stat(follow_symlinks=False)
    _link_file_noreplace(staged_file, final)
    linked.append((final, source_stat.st_dev, source_stat.st_ino))


def _rollback_recorded_links(linked: Sequence[tuple[Path, int, int]]) -> None:
    """Remove only directory entries that still name files linked by this attempt."""

    for path, expected_device, expected_inode in reversed(linked):
        try:
            current = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if current.st_dev == expected_device and current.st_ino == expected_inode:
            path.unlink()


def _write_json_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.stage-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_bytes(_canonical_json_bytes(value))
        _link_file_noreplace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_yaml_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.stage-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(
            yaml.safe_dump(dict(value), sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        _link_file_noreplace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative_path(raw: str) -> Path:
    pure = PurePosixPath(raw)
    if (
        not raw
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or str(pure) != raw
    ):
        raise ValueError(f"unsafe/non-canonical artifact path: {raw!r}")
    return Path(*pure.parts)


def _artifact_files(root: Path, *, excluded: Sequence[str]) -> dict[str, dict[str, Any]]:
    excluded_set = set(excluded)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def verify_recursive_files(
    root: Path,
    files: Mapping[str, Any],
    *,
    allowed_unlisted: Sequence[str],
) -> None:
    """Verify exact recursive file membership, size, hash, and no symlinks."""

    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"artifact root must be a real directory: {root}")
    expected_paths: set[str] = set()
    for relative_raw, record in files.items():
        relative = _safe_relative_path(str(relative_raw))
        relative_text = relative.as_posix()
        if relative_text in expected_paths:
            raise ValueError(f"duplicate artifact path: {relative_text}")
        expected_paths.add(relative_text)
        if not isinstance(record, Mapping):
            raise TypeError(f"invalid file record: {relative_text}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"artifact payload is missing/symlinked: {path}")
        _require_equal(path.stat().st_size, int(record["bytes"]), f"{relative_text} bytes")
        _require_equal(sha256_file(path), record["sha256"], f"{relative_text} SHA256")
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact contains a symlink: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    _require_equal(
        actual_paths,
        expected_paths | set(allowed_unlisted),
        f"{root} exact recursive file set",
    )


def _axis_attr(axis: Any, name: str) -> Any:
    if not hasattr(axis, name):
        raise TypeError(f"CheckpointAxis omits required field {name!r}")
    return getattr(axis, name)


def _development_contract(value: Mapping[str, Any], label: str) -> None:
    _require_equal(value.get("development_only"), True, f"{label}.development_only")
    _require_equal(value.get("main_paper_table"), False, f"{label}.main_paper_table")
    _require_equal(
        value.get("extra_best_pd_tuning_episodes"),
        0,
        f"{label}.extra_best_pd_tuning_episodes",
    )


def _role_publication_contract(
    *,
    role: str,
    formal_development_artifact: bool,
    parity_only: bool,
    max_images: int | None = None,
) -> None:
    """Prevent API callers from downgrading either frozen checkpoint role."""

    if role == ROLE_BEST_PD:
        _require_equal(
            formal_development_artifact, True, "best_pd formal development flag"
        )
        _require_equal(parity_only, False, "best_pd parity-only flag")
        _require_equal(max_images, None, "best_pd complete fixed-test execution")
        return
    _require_equal(role, ROLE_BEST_MIOU, "checkpoint role")
    _require_equal(formal_development_artifact, False, "best_miou formal flag")
    _require_equal(parity_only, True, "best_miou parity-only flag")
    _require_equal(max_images, None, "best_miou complete parity execution")


def _axis_config_sha256(path: Path) -> str:
    return sha256_file(path.resolve())


def _axis_output_dir(
    config: Mapping[str, Any],
    *,
    artifact_kind: str,
    role: str,
    dataset: str,
    output_override: Path | None = None,
) -> Path:
    return checkpoint_axis.canonical_output_dir(
        config,
        artifact_kind=artifact_kind,
        role=role,
        dataset=dataset,
        output_override=output_override,
    )


def _source_dataset_root(
    config: Mapping[str, Any],
    role: str,
    dataset: str,
    source_root_override: Path | None = None,
) -> Path:
    if source_root_override is not None:
        return source_root_override.expanduser().resolve() / dataset
    if role == ROLE_BEST_MIOU:
        project_root = Path(str(config["_runtime"]["project_root"]))
        raw = config["parity_gate"]["candidate_roots"]["source"]
        return (project_root / raw / dataset).resolve()
    errors: list[Exception] = []
    for kind in ("source", "source_corruption"):
        try:
            return _axis_output_dir(
                config, artifact_kind=kind, role=role, dataset=dataset
            )
        except (KeyError, ValueError) as error:
            errors.append(error)
    raise ValueError(f"axis config has no Source corruption output root: {errors}")


def _adabn_dataset_root(
    config: Mapping[str, Any],
    role: str,
    dataset: str,
    output_override: Path | None,
) -> Path:
    return _axis_output_dir(
        config,
        artifact_kind="adabn",
        role=role,
        dataset=dataset,
        output_override=output_override,
    )


def _validate_condition_payload_shape(
    *,
    artifact_root: Path,
    condition_root: Path,
    files: Mapping[str, Any],
    condition_key_value: str,
    expected_count: int | None,
    expected_role: str,
    expected_dataset: str,
) -> tuple[str, ...]:
    prefix = condition_root.relative_to(artifact_root).as_posix()
    probability_relative = f"{prefix}/probabilities_256.npy"
    per_image_relative = f"{prefix}/per_image.jsonl"
    metrics_relative = f"{prefix}/metrics.json"
    for relative in (probability_relative, per_image_relative, metrics_relative):
        if relative not in files:
            raise ValueError(f"manifest omits required condition payload: {relative}")
    probability_path = artifact_root / probability_relative
    probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if probabilities.ndim != 3 or tuple(probabilities.shape[1:]) != (256, 256):
        raise ValueError(f"invalid probability shard shape: {probabilities.shape}")
    if probabilities.dtype.str != "<f4":
        raise ValueError(f"invalid probability shard dtype: {probabilities.dtype}")
    count = int(probabilities.shape[0])
    if expected_count is not None:
        _require_equal(count, expected_count, f"{condition_key_value} image count")
    records = _load_jsonl(artifact_root / per_image_relative)
    _require_equal(len(records), count, f"{condition_key_value} per-image count")
    metrics = _load_json(artifact_root / metrics_relative)
    _require_equal(metrics.get("schema_version"), 2, "Source condition schema")
    _require_equal(metrics.get("artifact_kind"), "source", "Source condition artifact kind")
    _require_equal(metrics.get("method"), "Source", "Source condition method")
    _require_equal(metrics.get("checkpoint_role"), expected_role, "Source condition role")
    _require_equal(metrics.get("dataset"), expected_dataset, "Source condition dataset")
    _development_contract(metrics, f"Source condition {condition_key_value}")
    _require_equal(metrics.get("condition_key"), condition_key_value, "condition metrics key")
    _require_equal(metrics.get("evaluated_images"), count, "condition metrics count")
    shard_sha = sha256_file(probability_path)
    image_ids: list[str] = []
    mask_paths: set[str] = set()
    for index, record in enumerate(records):
        image_id = str(record.get("image_id", ""))
        if not image_id:
            raise ValueError(f"empty image ID at {condition_key_value}:{index}")
        _require_equal(record.get("index"), index, f"{condition_key_value} index")
        _require_equal(
            record.get("probability_shard_index"),
            index,
            f"{condition_key_value} probability index",
        )
        _require_equal(
            record.get("probability_shard_sha256"),
            shard_sha,
            f"{condition_key_value} probability lineage",
        )
        mask_rel_condition = _safe_relative_path(str(record.get("prediction_mask", "")))
        mask_rel_root = f"{prefix}/{mask_rel_condition.as_posix()}"
        if mask_rel_root not in files:
            raise ValueError(f"manifest omits prediction mask: {mask_rel_root}")
        _require_equal(
            record.get("prediction_mask_sha256"),
            files[mask_rel_root]["sha256"],
            f"{condition_key_value} mask lineage",
        )
        mask_paths.add(mask_rel_root)
        image_ids.append(image_id)
    _require_equal(len(mask_paths), count, f"{condition_key_value} unique mask count")
    del probabilities
    return tuple(image_ids)


def verify_source_dataset_artifact(
    root: Path,
    *,
    expected_axis: Any,
    axis_config_sha256: str,
    dataset: str,
) -> SourceArtifact:
    """Recursively verify the role-matched Source v2 dataset artifact."""

    # Preserve the lexical final component so a symlink root cannot disappear
    # through ``resolve()`` before the recursive no-follow audit.
    root = Path(os.path.abspath(root.expanduser()))
    # Consume the Source producer's public verifier, then independently repeat
    # the role/SHA/condition/per-image checks below.  A producer-side success
    # alone is intentionally not treated as downstream trust.
    producer_audit = source_axis_v2.verify_source_artifact(
        root, expected_axis=expected_axis
    )
    _require_equal(
        root,
        Path(_axis_attr(expected_axis, "output_dir")).expanduser().resolve(),
        "Source role-first artifact root",
    )
    manifest_path = root / "artifact_manifest.json"
    completion_path = root / "COMPLETE.json"
    benchmark_path = root / "benchmark.json"
    manifest = _load_json(manifest_path)
    completion = _load_json(completion_path)
    benchmark = _load_json(benchmark_path)
    role = str(_axis_attr(expected_axis, "role"))
    checkpoint_sha = str(_axis_attr(expected_axis, "checkpoint_sha256"))
    for value, label in ((manifest, "Source manifest"), (completion, "Source COMPLETE")):
        _require_equal(value.get("schema_version"), 1, f"{label}.schema_version")
        _require_equal(
            value.get("artifact_contract"),
            checkpoint_axis.ARTIFACT_CONTRACT,
            f"{label}.artifact_contract",
        )
        _require_equal(value.get("artifact_kind"), "source", f"{label}.artifact_kind")
        _require_equal(value.get("method"), "Source", f"{label}.method")
        _require_equal(value.get("checkpoint_role"), role, f"{label}.checkpoint_role")
        _require_equal(value.get("dataset"), dataset, f"{label}.dataset")
        _require_equal(
            value.get("axis_config_sha256"), axis_config_sha256, f"{label}.axis_config_sha256"
        )
        _require_equal(
            value.get("checkpoint_sha256"), checkpoint_sha, f"{label}.checkpoint_sha256"
        )
        _development_contract(value, label)
    _require_equal(benchmark.get("schema_version"), 2, "Source benchmark.schema_version")
    _require_equal(benchmark.get("artifact_kind"), "source", "Source benchmark.artifact_kind")
    _require_equal(benchmark.get("method"), "Source", "Source benchmark.method")
    _require_equal(benchmark.get("checkpoint_role"), role, "Source benchmark.checkpoint_role")
    _require_equal(benchmark.get("dataset"), dataset, "Source benchmark.dataset")
    _require_equal(
        benchmark.get("axis_config_sha256"),
        axis_config_sha256,
        "Source benchmark.axis_config_sha256",
    )
    _require_equal(
        benchmark.get("checkpoint_sha256"),
        checkpoint_sha,
        "Source benchmark.checkpoint_sha256",
    )
    _development_contract(benchmark, "Source benchmark")
    _require_equal(completion.get("complete"), True, "Source COMPLETE.complete")
    _require_equal(completion.get("condition_count"), 13, "Source condition count")
    _require_equal(
        completion.get("manifest_sha256"),
        sha256_file(manifest_path),
        "Source manifest receipt",
    )
    _require_equal(
        completion.get("benchmark_sha256"),
        sha256_file(benchmark_path),
        "Source benchmark receipt",
    )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("Source manifest.files must be a recursive mapping")
    verify_recursive_files(
        root,
        files,
        allowed_unlisted=("artifact_manifest.json", "COMPLETE.json"),
    )
    _require_equal(
        producer_audit["manifest_sha256"],
        sha256_file(manifest_path),
        "Source producer/downstream manifest snapshot",
    )
    parity_gate = manifest.get("best_miou_parity_gate")
    benchmark_parity_gate = benchmark.get("best_miou_parity_gate")
    if not isinstance(parity_gate, Mapping) or not isinstance(
        benchmark_parity_gate, Mapping
    ):
        raise TypeError("Source artifact omits best_miou_parity_gate evidence")
    if _canonical_json_bytes(parity_gate) != _canonical_json_bytes(
        benchmark_parity_gate
    ):
        raise ValueError("Source manifest/benchmark parity gate mismatch")
    expected_receipt_sha = _axis_attr(expected_axis, "parity_receipt_sha256")
    if role == ROLE_BEST_PD:
        if expected_receipt_sha is None:
            raise ValueError("best_pd Source axis omits the frozen parity receipt SHA256")
        _require_equal(parity_gate.get("mode"), "consumer", "Source parity gate mode")
        _require_equal(parity_gate.get("passed"), True, "Source parity gate status")
        _require_equal(
            parity_gate.get("receipt_sha256"),
            expected_receipt_sha,
            "Source manifest parity receipt lineage",
        )
        expected_receipt_path = _axis_attr(expected_axis, "parity_receipt_path")
        if expected_receipt_path is None:
            raise ValueError("best_pd Source axis omits the frozen parity receipt path")
        _require_equal(
            Path(str(parity_gate.get("receipt"))).expanduser().resolve(),
            Path(expected_receipt_path).expanduser().resolve(),
            "Source manifest parity receipt path",
        )
        _require_equal(
            completion.get("parity_receipt_sha256"),
            expected_receipt_sha,
            "Source COMPLETE parity receipt lineage",
        )
    else:
        _require_equal(parity_gate.get("mode"), "producer", "Source parity producer mode")
        _require_equal(
            completion.get("parity_receipt_sha256"),
            None,
            "best_miou Source authorization receipt absence",
        )
    _require_equal(
        completion.get("best_miou_parity_passed"),
        True,
        "Source COMPLETE parity status",
    )
    actual_condition_dirs = {
        path.name
        for path in (root / "conditions").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    _require_equal(actual_condition_dirs, set(CONDITION_KEYS), "Source exact condition set")
    benchmark_conditions = benchmark.get("conditions")
    if not isinstance(benchmark_conditions, Sequence) or isinstance(
        benchmark_conditions, (str, bytes)
    ):
        raise TypeError("Source benchmark.conditions must be a sequence")
    _require_equal(
        tuple(str(item["condition_key"]) for item in benchmark_conditions),
        CONDITION_KEYS,
        "Source benchmark condition order",
    )
    expected_count_raw = benchmark.get("evaluated_images_per_condition")
    if expected_count_raw is None:
        raise ValueError("Source benchmark omits evaluated_images_per_condition")
    expected_count = int(expected_count_raw)
    _require_equal(
        expected_count,
        int(_axis_attr(expected_axis, "expected_images")),
        "Source benchmark fixed-test image count",
    )
    common_ids: tuple[str, ...] | None = None
    for key in CONDITION_KEYS:
        ids = _validate_condition_payload_shape(
            artifact_root=root,
            condition_root=root / "conditions" / key,
            files=files,
            condition_key_value=key,
            expected_count=expected_count,
            expected_role=role,
            expected_dataset=dataset,
        )
        if common_ids is None:
            common_ids = ids
        else:
            _require_equal(ids, common_ids, f"Source {key} ordered image IDs")
    if common_ids is None:
        raise RuntimeError("Source artifact has no conditions")
    _require_equal(
        len(set(common_ids)),
        len(common_ids),
        "Source unique ordered image IDs",
    )
    declared_ids_sha = benchmark.get("ordered_ids_sha256")
    if declared_ids_sha is None:
        raise ValueError("Source benchmark omits ordered_ids_sha256")
    _require_equal(
        ordered_ids_sha256(common_ids), declared_ids_sha, "Source ordered IDs SHA256"
    )
    return SourceArtifact(
        root=root,
        benchmark=benchmark,
        manifest=manifest,
        completion=completion,
        manifest_sha256=sha256_file(manifest_path),
        completion_sha256=sha256_file(completion_path),
        benchmark_sha256=sha256_file(benchmark_path),
        image_ids=common_ids,
    )


def _load_frozen_execution_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    _require_equal(
        sha256_file(FROZEN_ADABN_V1_CONFIG),
        FROZEN_ADABN_V1_CONFIG_SHA256,
        "sealed AdaBN v1 config SHA256",
    )
    _require_equal(
        sha256_file(FROZEN_ADABN_V1_RUNNER),
        FROZEN_ADABN_V1_RUNNER_SHA256,
        "sealed AdaBN v1 runner SHA256",
    )
    loaded = yaml.safe_load(FROZEN_ADABN_V1_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("sealed AdaBN v1 config must be a mapping")
    adabn_protocol = dict(loaded)
    source_path = (PROJECT_ROOT / adabn_protocol["source_benchmark"]["protocol"]).resolve()
    source_loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_loaded, Mapping):
        raise TypeError("sealed Source v1 config must be a mapping")
    return adabn_protocol, dict(source_loaded)


def prepare_dataset_context(
    *,
    axis_config_path: Path,
    axis_config: Mapping[str, Any],
    role: str,
    dataset: str,
    verify_all_cache_file_hashes: bool,
    axis_output_override: Path | None = None,
    source_root_override: Path | None = None,
) -> DatasetContext:
    """Resolve role/checkpoint, verify Source recursively, and bind v1 compute inputs."""

    if role == ROLE_BEST_PD and (
        axis_output_override is not None or source_root_override is not None
    ):
        raise ValueError("best_pd context forbids output and Source root overrides")

    axis = checkpoint_axis.resolve_axis(
        axis_config,
        dataset=dataset,
        role=role,
        artifact_kind="adabn",
        output_override=axis_output_override,
    )
    checkpoint_payload = checkpoint_axis.load_and_verify_checkpoint(axis)
    checkpoint_path = Path(_axis_attr(axis, "checkpoint_path")).resolve()
    checkpoint_sha = str(_axis_attr(axis, "checkpoint_sha256"))
    _require_equal(sha256_file(checkpoint_path), checkpoint_sha, "checkpoint snapshot SHA256")
    expected_selection_metric = {ROLE_BEST_MIOU: "miou", ROLE_BEST_PD: "pd"}[role]
    _require_equal(
        checkpoint_payload.get("selection_metric"),
        expected_selection_metric,
        "checkpoint selection_metric",
    )
    _require_equal(checkpoint_payload.get("dataset"), dataset, "checkpoint dataset")
    _require_equal(checkpoint_payload.get("test_selected"), True, "checkpoint test_selected")
    checkpoint_summary = {
        key: checkpoint_payload.get(key)
        for key in (
            "schema_version",
            "architecture",
            "dataset",
            "epoch",
            "selection_metric",
            "selection_rule",
            "selection_value",
            "test_selected",
        )
    }
    axis_sha = _axis_config_sha256(axis_config_path)
    source_dataset_root = _source_dataset_root(
        axis_config, role, dataset, source_root_override=source_root_override
    )
    source_axis = checkpoint_axis.resolve_axis(
        axis_config,
        dataset=dataset,
        role=role,
        artifact_kind="source",
        output_override=(source_dataset_root if role == ROLE_BEST_MIOU else None),
    )
    source_artifact = verify_source_dataset_artifact(
        source_dataset_root,
        expected_axis=source_axis,
        axis_config_sha256=axis_sha,
        dataset=dataset,
    )
    legacy_protocol, source_protocol = _load_frozen_execution_protocol()
    legacy_contract = dict(legacy_protocol["datasets"][dataset])
    cache_dir = (PROJECT_ROOT / legacy_contract["cache_dir"]).resolve()
    cache_manifest, cache_audit = verify_cache_artifact(
        cache_dir,
        expected_protocol_sha256=legacy_protocol["materialized_cache"][
            "generation_protocol_sha256"
        ],
        verify_file_hashes=verify_all_cache_file_hashes,
    )
    for actual, expected, label in (
        (cache_audit["manifest_sha256"], legacy_contract["cache_manifest_sha256"], "cache manifest"),
        (cache_manifest["cache_content_sha256"], legacy_contract["cache_content_sha256"], "cache content"),
        (tuple(cache_manifest["image_ids"]), source_artifact.image_ids, "cache/Source IDs"),
    ):
        _require_equal(actual, expected, f"{dataset} {label}")
    role_contract = {
        **legacy_contract,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "source_result": str(source_artifact.root),
        "source_benchmark_sha256": source_artifact.benchmark_sha256,
        "source_artifact_manifest_sha256": source_artifact.manifest_sha256,
    }
    legacy_context = legacy.DatasetContext(
        dataset_name=dataset,
        config_contract=role_contract,
        source_protocol=source_protocol,
        source_dataset_contract=dict(source_protocol["datasets"][dataset]),
        cache_dir=cache_dir,
        cache_manifest=cache_manifest,
        cache_audit=cache_audit,
        image_ids=tuple(cache_manifest["image_ids"]),
        checkpoint=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
        checkpoint_summary=checkpoint_summary,
        source_root=source_artifact.root,
        source_benchmark=source_artifact.benchmark,
        source_artifact_manifest=source_artifact.manifest,
    )
    return DatasetContext(
        axis=axis,
        source_axis=source_axis,
        axis_config_path=axis_config_path,
        axis_config=axis_config,
        axis_config_sha256=axis_sha,
        checkpoint_payload=checkpoint_payload,
        checkpoint_summary=checkpoint_summary,
        cache_dir=cache_dir,
        cache_manifest=cache_manifest,
        cache_audit=cache_audit,
        source_artifact=source_artifact,
        legacy_context=legacy_context,
        legacy_protocol=legacy_protocol,
        legacy_source_protocol=source_protocol,
    )


def _publish_directory_noreplace(staging: Path, final: Path) -> None:
    """Atomically publish a staged directory with kernel no-replace semantics."""

    final.parent.mkdir(parents=True, exist_ok=True)
    publish_directory_noreplace(staging, final)


def _staging_sibling(final: Path) -> Path:
    """Reserve a unique sibling pathname without creating the leaf.

    The frozen v1 ``execute_condition`` function owns creation of its
    destination and intentionally rejects an already-existing directory.
    Creating the leaf here would therefore violate the reused numerical
    runner's no-overwrite contract before inference starts.
    """

    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.build-{os.getpid()}-{time.time_ns()}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"condition staging pathname already exists: {staging}")
    return staging


def _best_miou_parity(
    *, staging: Path, dataset: str, key: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    reference = (
        PROJECT_ROOT
        / "results"
        / "adabn"
        / "adabn_batch_stats_v1"
        / dataset
        / "conditions"
        / key
    )
    for name in ("probabilities_256.npy", "metrics.json", "per_image.jsonl"):
        if not (reference / name).is_file():
            raise FileNotFoundError(f"sealed AdaBN v1 parity input is missing: {reference / name}")
    _require_equal(
        sha256_file(staging / "probabilities_256.npy"),
        sha256_file(reference / "probabilities_256.npy"),
        "best_miou probability shard parity",
    )
    new_masks = {
        path.relative_to(staging / "prediction_masks_256").as_posix(): sha256_file(path)
        for path in (staging / "prediction_masks_256").rglob("*.png")
    }
    old_masks = {
        path.relative_to(reference / "prediction_masks_256").as_posix(): sha256_file(path)
        for path in (reference / "prediction_masks_256").rglob("*.png")
    }
    _require_equal(new_masks, old_masks, "best_miou prediction mask parity")
    old_metrics = _load_json(reference / "metrics.json")
    metric_fields = (
        "summary",
        "official",
        "unified",
        "source_summary",
        "deltas_from_source",
        "probability_shard_sha256",
        "adabn_probability_tensor_sequence_sha256",
    )
    for field in metric_fields:
        # Runtime evaluator containers may use tuples while the sealed JSON
        # reference necessarily reloads them as lists.  Compare their exact
        # canonical JSON values so container serialization is normalized but
        # every integer and floating-point value remains tolerance-free.
        if _canonical_json_bytes(metrics.get(field)) != _canonical_json_bytes(
            old_metrics.get(field)
        ):
            raise ValueError(f"best_miou metrics.{field} mismatch")
    new_records = _load_jsonl(staging / "per_image.jsonl")
    old_records = _load_jsonl(reference / "per_image.jsonl")
    _require_equal(
        tuple(record["image_id"] for record in new_records),
        tuple(record["image_id"] for record in old_records),
        "best_miou per-image order parity",
    )
    return {
        "passed": True,
        "reference_root": str(reference),
        "reference_probability_shard_sha256": sha256_file(reference / "probabilities_256.npy"),
        "probability_shard_bit_exact": True,
        "prediction_masks_bit_exact": True,
        "metrics_exact_fields": list(metric_fields),
        "per_image_order_exact": True,
        "numeric_tolerance_used": False,
    }


def _condition_contract_fields(
    *, context: DatasetContext, role: str, dataset: str, key: str
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "axis_config_path": str(context.axis_config_path),
        "axis_config_sha256": context.axis_config_sha256,
        "checkpoint_role": role,
        "checkpoint_path": str(_axis_attr(context.axis, "checkpoint_path")),
        "checkpoint_sha256": str(_axis_attr(context.axis, "checkpoint_sha256")),
        "dataset": dataset,
        "condition_key": key,
        "development_only": True,
        "main_paper_table": False,
        "extra_best_pd_tuning_episodes": 0,
        "parity_receipt_path": (
            str(context.axis.parity_receipt_path)
            if getattr(context.axis, "parity_receipt_path", None) is not None
            else None
        ),
        "parity_receipt_sha256": getattr(
            context.axis, "parity_receipt_sha256", None
        ),
    }


def publish_condition(
    *,
    context: DatasetContext,
    role: str,
    dataset: str,
    corruption: str,
    severity: int,
    device_spec: str,
    final: Path,
    formal_development_artifact: bool,
    parity_only: bool,
    max_images: int | None,
) -> dict[str, Any]:
    """Execute, verify, seal, and no-replace publish one AdaBN condition."""

    key = condition_key(corruption, severity)
    if (corruption, severity) not in CONDITIONS:
        raise ValueError(f"condition is outside the 13-condition contract: {key}")
    _role_publication_contract(
        role=role,
        formal_development_artifact=formal_development_artifact,
        parity_only=parity_only,
        max_images=max_images,
    )
    expected_final = (
        Path(_axis_attr(context.axis, "output_dir"))
        / "conditions"
        / key
    ).expanduser().resolve()
    _require_equal(final.expanduser().resolve(), expected_final, "condition role-first output")
    if formal_development_artifact:
        gate = checkpoint_axis.verify_parity_receipt(config=context.axis_config)
        _require_equal(
            gate["_receipt_sha256"],
            context.axis.parity_receipt_sha256,
            "best_pd global parity receipt snapshot",
        )
    staging = _staging_sibling(final)
    try:
        source_runner.seed_everything(int(context.legacy_protocol["execution"]["seed"]))
        device = source_runner.resolve_device(device_spec)
        if formal_development_artifact and str(device_spec).startswith("cuda"):
            _require_equal(
                torch.cuda.device_count(),
                int(context.legacy_protocol["execution"]["formal_worker_visible_cuda_device_count"]),
                "formal worker visible CUDA count",
            )
        model = source_runner.build_nsfpn_model()
        checkpoint_axis.verify_checkpoint_file(context.axis)
        checkpoint_wrapper = checkpoint_axis.load_checkpoint_into_model(
            model, context.checkpoint_payload
        )
        checkpoint_axis.verify_checkpoint_file(context.axis)
        model.to(device)
        adapter = IRSTDModelAdapter(model, warm_flag=False)
        adapter.set_source_eval_mode()
        state = EpisodicStateManager(model, optimizer=None)
        generic_runner = EpisodicRunner(adapter, state)
        batchnorm_count = sum(isinstance(module, nn.BatchNorm2d) for module in model.modules())
        _require_equal(
            batchnorm_count,
            int(context.legacy_protocol["execution"]["expected_batchnorm2d_modules"]),
            "NS-FPN BatchNorm2d count",
        )
        cached_dataset = CachedCorruptionDataset(
            context.cache_dir,
            corruption=corruption,
            severity=severity,
            manifest=context.cache_manifest,
        )
        available = len(cached_dataset) if max_images is None else min(max_images, len(cached_dataset))
        order_sentinel = legacy._order_isolation_sentinel(
            dataset=cached_dataset,
            runner=generic_runner,
            adapter=adapter,
            expected_batchnorm_count=batchnorm_count,
            available_images=available,
        )
        state.assert_source_state()
        fast_runner = AdaBNFastRunner(
            adapter,
            state,
            full_audit_cadence=int(
                context.legacy_protocol["execution"]["state_audit"][
                    "full_state_sha256_cadence"
                ]["every_n_images"]
            ),
        )
        source_reference = legacy.load_source_condition_reference(
            context.legacy_context, corruption, severity
        )
        evaluation_protocol: IRSTDEvaluationProtocol = source_v1._evaluation_protocol(
            context.legacy_source_protocol
        )
        metrics = legacy.execute_condition(
            context=context.legacy_context,
            corruption=corruption,
            severity=severity,
            dataset=cached_dataset,
            source_reference=source_reference,
            adapter=adapter,
            runner=fast_runner,
            precomputed_order_sentinel=order_sentinel,
            evaluation_protocol=evaluation_protocol,
            destination=staging,
            expected_batchnorm_count=batchnorm_count,
            max_images=max_images,
            formal_artifact=formal_development_artifact,
        )
        state.assert_source_state()
        if fast_runner.aborted:
            raise RuntimeError("AdaBN fast runner aborted")
        contract = _condition_contract_fields(
            context=context, role=role, dataset=dataset, key=key
        )
        metrics.update(
            {
                "schema_version": SCHEMA_VERSION,
                **contract,
                "formal_development_artifact": formal_development_artifact,
                "formal_artifact": formal_development_artifact,
                "paper_result": False,
                "parity_only": parity_only,
                "source_dataset_artifact_manifest_sha256": context.source_artifact.manifest_sha256,
                "source_dataset_complete_sha256": context.source_artifact.completion_sha256,
                "source_dataset_benchmark_sha256": context.source_artifact.benchmark_sha256,
                "delta_role": "report_only_never_used_for_selection_or_tuning",
            }
        )
        _write_json(staging / "metrics.json", metrics)
        parity_receipt: Mapping[str, Any] | None = None
        if role == ROLE_BEST_MIOU:
            parity_receipt = _best_miou_parity(
                staging=staging, dataset=dataset, key=key, metrics=metrics
            )
            _write_json(staging / "BEST_MIOU_PARITY.json", parity_receipt)
        run_config = deepcopy(dict(context.axis_config))
        run_config["runtime"] = {
            **contract,
            "corruption": corruption,
            "severity": severity,
            "device": str(device),
            "max_images": max_images,
            "formal_development_artifact": formal_development_artifact,
            "parity_only": parity_only,
        }
        _write_yaml(staging / "condition_run_config.yaml", run_config)
        provenance = {
            "schema_version": SCHEMA_VERSION,
            **contract,
            "method": "AdaBN",
            "method_role": "single_image_batch_statistics_failure_control",
            "formal_development_artifact": formal_development_artifact,
            "paper_result": False,
            "parity_only": parity_only,
            "checkpoint_wrapper": checkpoint_wrapper,
            "checkpoint_metadata": context.checkpoint_summary,
            "source_artifact": {
                "root": str(context.source_artifact.root),
                "artifact_manifest_sha256": context.source_artifact.manifest_sha256,
                "complete_sha256": context.source_artifact.completion_sha256,
                "benchmark_sha256": context.source_artifact.benchmark_sha256,
            },
            "sealed_v1_compute_lineage": {
                "runner": str(FROZEN_ADABN_V1_RUNNER),
                "runner_sha256": FROZEN_ADABN_V1_RUNNER_SHA256,
                "config": str(FROZEN_ADABN_V1_CONFIG),
                "config_sha256": FROZEN_ADABN_V1_CONFIG_SHA256,
                "reused_function": "execute_condition",
            },
            "method_label_boundary": {
                "method_received_mask": False,
                "labels_used_only_by_external_evaluators_after_logits": True,
            },
            "best_miou_parity": parity_receipt,
        }
        _write_json(staging / "condition_provenance.json", provenance)
        files = _artifact_files(
            staging, excluded=("artifact_manifest.json", CONDITION_SENTINEL)
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            **contract,
            "method": "AdaBN",
            "corruption": corruption,
            "severity": severity,
            "formal_development_artifact": formal_development_artifact,
            "paper_result": False,
            "parity_only": parity_only,
            "source_artifact": {
                "root": str(context.source_artifact.root),
                "artifact_manifest_sha256": context.source_artifact.manifest_sha256,
                "complete_sha256": context.source_artifact.completion_sha256,
                "benchmark_sha256": context.source_artifact.benchmark_sha256,
            },
            "files": files,
        }
        _write_json(staging / "artifact_manifest.json", manifest)
        completion = {
            "complete": True,
            "scope": "condition",
            "schema_version": SCHEMA_VERSION,
            **contract,
            "method": "AdaBN",
            "corruption": corruption,
            "severity": severity,
            "formal_development_artifact": formal_development_artifact,
            "formal_artifact": formal_development_artifact,
            "paper_result": False,
            "parity_only": parity_only,
            "evaluated_images": metrics["evaluated_images"],
            "full_fixed_test_split": metrics["full_fixed_test_split"],
            "metrics_sha256": files["metrics.json"]["sha256"],
            "source_dataset_artifact_manifest_sha256": context.source_artifact.manifest_sha256,
            "source_dataset_complete_sha256": context.source_artifact.completion_sha256,
            "source_dataset_benchmark_sha256": context.source_artifact.benchmark_sha256,
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
            "all_required_gates_passed": True,
        }
        _write_json(staging / CONDITION_SENTINEL, completion)
        verify_condition_artifact(
            staging,
            expected_axis=context.axis,
            axis_config_sha256=context.axis_config_sha256,
            expected_source=context.source_artifact,
            expected_corruption=corruption,
            expected_severity=severity,
            formal_development_artifact=formal_development_artifact,
            parity_only=parity_only,
            allow_staging_name=True,
        )
        _publish_directory_noreplace(staging, final)
        return {"published_output_dir": str(final), "metrics": metrics, "completion": completion}
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_condition_artifact(
    root: Path,
    *,
    expected_axis: Any,
    axis_config_sha256: str,
    expected_source: SourceArtifact | None,
    expected_corruption: str,
    expected_severity: int,
    formal_development_artifact: bool,
    parity_only: bool,
    allow_staging_name: bool = False,
) -> dict[str, Any]:
    root = Path(os.path.abspath(root.expanduser()))
    expected_key = condition_key(expected_corruption, expected_severity)
    expected_final = (
        Path(_axis_attr(expected_axis, "output_dir"))
        / "conditions"
        / expected_key
    ).expanduser().resolve()
    if root.name != expected_key:
        staging_prefix = f".{expected_key}.build-"
        if not allow_staging_name or not root.name.startswith(staging_prefix):
            raise ValueError(
                f"condition directory key mismatch: expected {expected_key!r}, "
                f"got {root.name!r}"
            )
        _require_equal(root.parent.resolve(), expected_final.parent, "condition staging parent")
    else:
        _require_equal(root.resolve(), expected_final, "condition role-first artifact root")
    _require_equal(_axis_attr(expected_axis, "artifact_kind"), "adabn", "condition axis kind")
    manifest = _load_json(root / "artifact_manifest.json")
    completion = _load_json(root / CONDITION_SENTINEL)
    metrics = _load_json(root / "metrics.json")
    role = str(_axis_attr(expected_axis, "role"))
    _role_publication_contract(
        role=role,
        formal_development_artifact=formal_development_artifact,
        parity_only=parity_only,
    )
    checkpoint_sha = str(_axis_attr(expected_axis, "checkpoint_sha256"))
    expected_receipt_sha = _axis_attr(expected_axis, "parity_receipt_sha256")
    expected_receipt_path_raw = _axis_attr(expected_axis, "parity_receipt_path")
    expected_receipt_path = (
        str(expected_receipt_path_raw) if expected_receipt_path_raw is not None else None
    )
    if role == ROLE_BEST_PD and expected_receipt_sha is None:
        raise ValueError("best_pd condition axis omits the global parity receipt")
    for value, label in ((manifest, "manifest"), (completion, "COMPLETE"), (metrics, "metrics")):
        _require_equal(value.get("schema_version"), SCHEMA_VERSION, f"{label}.schema_version")
        _require_equal(value.get("protocol_id"), PROTOCOL_ID, f"{label}.protocol_id")
        _require_equal(value.get("method"), "AdaBN", f"{label}.method")
        _require_equal(value.get("dataset"), _axis_attr(expected_axis, "dataset"), f"{label}.dataset")
        _require_equal(value.get("condition_key"), expected_key, f"{label}.condition_key")
        _require_equal(value.get("corruption"), expected_corruption, f"{label}.corruption")
        _require_equal(value.get("severity"), expected_severity, f"{label}.severity")
        _require_equal(value.get("checkpoint_role"), role, f"{label}.checkpoint_role")
        _require_equal(value.get("checkpoint_sha256"), checkpoint_sha, f"{label}.checkpoint_sha256")
        _require_equal(value.get("axis_config_sha256"), axis_config_sha256, f"{label}.axis_config_sha256")
        _require_equal(
            value.get("parity_receipt_sha256"),
            expected_receipt_sha,
            f"{label}.parity_receipt_sha256",
        )
        _require_equal(
            value.get("parity_receipt_path"),
            expected_receipt_path,
            f"{label}.parity_receipt_path",
        )
        _development_contract(value, label)
        _require_equal(value.get("paper_result"), False, f"{label}.paper_result")
        _require_equal(value.get("parity_only"), parity_only, f"{label}.parity_only")
    _require_equal(completion.get("complete"), True, "condition COMPLETE.complete")
    _require_equal(completion.get("scope"), "condition", "condition COMPLETE.scope")
    _require_equal(
        completion.get("formal_development_artifact"),
        formal_development_artifact,
        "condition formal development flag",
    )
    if formal_development_artifact:
        _require_equal(role, ROLE_BEST_PD, "formal checkpoint role")
    if formal_development_artifact or role == ROLE_BEST_MIOU:
        _require_equal(metrics.get("full_fixed_test_split"), True, "full fixed-test split")
        _require_equal(
            int(metrics.get("evaluated_images", -1)),
            int(_axis_attr(expected_axis, "expected_images")),
            "condition fixed-test image count",
        )
    if role == ROLE_BEST_MIOU:
        _require_equal(parity_only, True, "best_miou parity-only gate")
        parity = _load_json(root / "BEST_MIOU_PARITY.json")
        _require_equal(parity.get("passed"), True, "best_miou parity receipt")
        for field in (
            "probability_shard_bit_exact",
            "prediction_masks_bit_exact",
            "per_image_order_exact",
        ):
            _require_equal(parity.get(field), True, f"best_miou parity {field}")
        _require_equal(
            parity.get("numeric_tolerance_used"),
            False,
            "best_miou parity numeric tolerance",
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("condition manifest.files must be a mapping")
    required_files = {
        "metrics.json",
        "per_image.jsonl",
        "adaptation_diagnostics.jsonl",
        "order_isolation_sentinel.json",
        "probabilities_256.npy",
        "condition_run_config.yaml",
        "condition_provenance.json",
    }
    if role == ROLE_BEST_MIOU:
        required_files.add("BEST_MIOU_PARITY.json")
    missing_files = sorted(required_files - set(files))
    if missing_files:
        raise FileNotFoundError(
            f"condition manifest omits required payloads: {missing_files}"
        )
    verify_recursive_files(
        root,
        files,
        allowed_unlisted=("artifact_manifest.json", CONDITION_SENTINEL),
    )
    _require_equal(
        sha256_file(root / "artifact_manifest.json"),
        completion.get("artifact_manifest_sha256"),
        "condition manifest receipt",
    )
    _require_equal(
        files["metrics.json"]["sha256"], completion.get("metrics_sha256"), "condition metrics receipt"
    )
    probabilities = np.load(root / "probabilities_256.npy", mmap_mode="r", allow_pickle=False)
    expected_count = int(metrics["evaluated_images"])
    _require_equal(probabilities.shape, (expected_count, 256, 256), "AdaBN probability shape")
    _require_equal(probabilities.dtype.str, "<f4", "AdaBN probability dtype")
    records = _load_jsonl(root / "per_image.jsonl")
    diagnostics = _load_jsonl(root / "adaptation_diagnostics.jsonl")
    _require_equal(len(records), expected_count, "AdaBN per-image count")
    _require_equal(len(diagnostics), expected_count, "AdaBN diagnostics count")
    mask_count = len(tuple((root / "prediction_masks_256").rglob("*.png")))
    _require_equal(mask_count, expected_count, "AdaBN prediction mask count")
    _require_equal(
        files["probabilities_256.npy"]["sha256"],
        metrics.get("probability_shard_sha256"),
        "AdaBN probability shard receipt",
    )
    _require_equal(
        completion.get("evaluated_images"),
        metrics.get("evaluated_images"),
        "condition completion image count",
    )
    _require_equal(
        completion.get("full_fixed_test_split"),
        metrics.get("full_fixed_test_split"),
        "condition completion full-split flag",
    )
    _require_equal(
        completion.get("all_required_gates_passed"),
        True,
        "condition completion gates",
    )
    order_sentinel = _load_json(root / "order_isolation_sentinel.json")
    _require_equal(order_sentinel.get("passed"), True, "order isolation sentinel")
    if _canonical_json_bytes(order_sentinel) != _canonical_json_bytes(
        metrics.get("order_isolation_sentinel")
    ):
        raise ValueError("metrics/order isolation sentinel mismatch")
    checks = metrics.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("condition metrics.checks must be a non-empty mapping")
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    if failed_checks:
        raise ValueError(f"condition metrics checks failed: {failed_checks}")
    legacy._verify_condition_internal_semantics(
        condition_root=root,
        metrics=metrics,
        probabilities=probabilities,
        per_image=records,
        diagnostics=diagnostics,
        manifest_files=files,
    )
    del probabilities
    if expected_source is not None:
        _require_equal(
            tuple(str(record.get("image_id", "")) for record in records),
            expected_source.image_ids,
            "AdaBN/Source ordered image IDs",
        )
        source_condition_files = expected_source.manifest.get("condition_files")
        if not isinstance(source_condition_files, Mapping):
            raise TypeError("Source manifest omits condition_files")
        source_condition = source_condition_files.get(expected_key)
        if not isinstance(source_condition, Mapping):
            raise TypeError(f"Source manifest omits condition receipt: {expected_key}")
        source_probability = source_condition.get("probability_shard")
        if not isinstance(source_probability, Mapping):
            raise TypeError(
                f"Source manifest omits probability receipt: {expected_key}"
            )
        source_probability_sha = source_probability.get("sha256")
        for index, record in enumerate(records):
            _require_equal(
                record.get("source_reference_probability_shard_sha256"),
                source_probability_sha,
                f"AdaBN {expected_key}:{index} Source probability lineage",
            )
        for field, expected in (
            ("source_dataset_artifact_manifest_sha256", expected_source.manifest_sha256),
            ("source_dataset_complete_sha256", expected_source.completion_sha256),
            ("source_dataset_benchmark_sha256", expected_source.benchmark_sha256),
        ):
            _require_equal(metrics.get(field), expected, f"metrics.{field}")
            _require_equal(completion.get(field), expected, f"COMPLETE.{field}")
        source_record = manifest.get("source_artifact")
        if not isinstance(source_record, Mapping):
            raise TypeError("condition manifest omits source_artifact")
        _require_equal(
            source_record.get("artifact_manifest_sha256"),
            expected_source.manifest_sha256,
            "condition Source manifest lineage",
        )
        _require_equal(
            source_record.get("complete_sha256"),
            expected_source.completion_sha256,
            "condition Source completion lineage",
        )
        _require_equal(
            source_record.get("benchmark_sha256"),
            expected_source.benchmark_sha256,
            "condition Source benchmark lineage",
        )
        _require_equal(
            Path(str(source_record.get("root"))).expanduser().resolve(),
            expected_source.root.resolve(),
            "condition Source root lineage",
        )
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(root / "artifact_manifest.json"),
        "completion": completion,
        "completion_sha256": sha256_file(root / CONDITION_SENTINEL),
        "metrics": metrics,
    }


def _condition_final(dataset_root: Path, corruption: str, severity: int) -> Path:
    return dataset_root / "conditions" / condition_key(corruption, severity)


def finalize_dataset(
    *,
    context: DatasetContext,
    dataset_root: Path,
    role: str,
    formal_development_artifact: bool,
    parity_only: bool,
) -> dict[str, Any]:
    """Recursively verify all 13 conditions and publish the dataset seal last."""

    _role_publication_contract(
        role=role,
        formal_development_artifact=formal_development_artifact,
        parity_only=parity_only,
    )
    _require_equal(
        dataset_root.expanduser().resolve(),
        Path(_axis_attr(context.axis, "output_dir")).expanduser().resolve(),
        "dataset role-first output root",
    )
    if formal_development_artifact:
        gate = checkpoint_axis.verify_parity_receipt(config=context.axis_config)
        _require_equal(
            gate["_receipt_sha256"],
            context.axis.parity_receipt_sha256,
            "dataset finalization parity receipt snapshot",
        )
    _require_equal(
        sha256_file(context.axis_config_path),
        context.axis_config_sha256,
        "dataset finalization axis config snapshot",
    )
    checkpoint_axis.verify_checkpoint_file(context.axis)
    source_now = verify_source_dataset_artifact(
        context.source_artifact.root,
        expected_axis=context.source_axis,
        axis_config_sha256=context.axis_config_sha256,
        dataset=context.legacy_context.dataset_name,
    )
    for actual, expected, label in (
        (source_now.manifest_sha256, context.source_artifact.manifest_sha256, "Source manifest"),
        (source_now.completion_sha256, context.source_artifact.completion_sha256, "Source COMPLETE"),
        (source_now.benchmark_sha256, context.source_artifact.benchmark_sha256, "Source benchmark"),
    ):
        _require_equal(actual, expected, f"dataset finalization {label} snapshot")
    for name in (
        "benchmark.json",
        "run_config.yaml",
        "provenance.json",
        "artifact_manifest.json",
        DATASET_SENTINEL,
    ):
        if (dataset_root / name).exists() or (dataset_root / name).is_symlink():
            raise FileExistsError(
                f"dataset metadata exists; refusing replacement: {dataset_root / name}"
            )
    condition_entries = tuple((dataset_root / "conditions").iterdir())
    if any(not path.is_dir() or path.is_symlink() for path in condition_entries):
        raise ValueError("AdaBN conditions root contains a non-directory or symlink")
    actual = {path.name for path in condition_entries}
    _require_equal(actual, set(CONDITION_KEYS), "AdaBN exact 13-condition directory set")
    condition_records: list[dict[str, Any]] = []
    child_links: dict[str, dict[str, Any]] = {}
    for corruption, severity in CONDITIONS:
        key = condition_key(corruption, severity)
        verified = verify_condition_artifact(
            dataset_root / "conditions" / key,
            expected_axis=context.axis,
            axis_config_sha256=context.axis_config_sha256,
            expected_source=context.source_artifact,
            expected_corruption=corruption,
            expected_severity=severity,
            formal_development_artifact=formal_development_artifact,
            parity_only=parity_only,
        )
        metrics = verified["metrics"]
        condition_records.append(
            {
                "condition_index": CONDITIONS.index((corruption, severity)),
                "condition_key": key,
                "corruption": corruption,
                "severity": severity,
                **dict(metrics["summary"]),
                "source_summary": dict(metrics["source_summary"]),
                "deltas_from_source": dict(metrics["deltas_from_source"]),
                "metrics": f"conditions/{key}/metrics.json",
                "metrics_sha256": sha256_file(dataset_root / "conditions" / key / "metrics.json"),
            }
        )
        child_links[key] = {
            "artifact_manifest": f"conditions/{key}/artifact_manifest.json",
            "artifact_manifest_sha256": verified["manifest_sha256"],
            "completion": f"conditions/{key}/{CONDITION_SENTINEL}",
            "completion_sha256": verified["completion_sha256"],
        }
    contract = _condition_contract_fields(
        context=context, role=role, dataset=context.legacy_context.dataset_name, key="__dataset__"
    )
    contract.pop("condition_key")
    benchmark = {
        "schema_version": SCHEMA_VERSION,
        **contract,
        "method": "AdaBN",
        "formal_development_artifact": formal_development_artifact,
        "paper_result": False,
        "parity_only": parity_only,
        "condition_count": 13,
        "evaluated_images_per_condition": len(context.source_artifact.image_ids),
        "ordered_ids_sha256": ordered_ids_sha256(context.source_artifact.image_ids),
        "conditions": condition_records,
        "delta_role": "report_only_never_used_for_selection_or_tuning",
    }
    run_config = {
        **deepcopy(dict(context.axis_config)),
        "runtime": {
            **contract,
            "conditions": list(CONDITION_KEYS),
            "formal_development_artifact": formal_development_artifact,
            "parity_only": parity_only,
        },
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        **contract,
        "method": "AdaBN",
        "formal_development_artifact": formal_development_artifact,
        "paper_result": False,
        "parity_only": parity_only,
        "source_artifact": {
            "root": str(context.source_artifact.root),
            "artifact_manifest_sha256": context.source_artifact.manifest_sha256,
            "complete_sha256": context.source_artifact.completion_sha256,
            "benchmark_sha256": context.source_artifact.benchmark_sha256,
        },
        "condition_children": child_links,
    }
    required_payloads = [
        "benchmark.json",
        "run_config.yaml",
        "provenance.json",
        *(
            f"conditions/{key}/{name}"
            for key in CONDITION_KEYS
            for name in (
                "metrics.json",
                "per_image.jsonl",
                "adaptation_diagnostics.jsonl",
                "probabilities_256.npy",
            )
        ),
    ]
    staging = dataset_root.parent / (
        f".{dataset_root.name}.metadata-build-{os.getpid()}-{time.time_ns()}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    linked: list[tuple[Path, int, int]] = []
    try:
        _write_json(staging / "benchmark.json", benchmark)
        _write_yaml(staging / "run_config.yaml", run_config)
        _write_json(staging / "provenance.json", provenance)
        for name in ("benchmark.json", "run_config.yaml", "provenance.json"):
            _link_file_recorded(staging / name, dataset_root / name, linked)

        preliminary_files = _artifact_files(
            dataset_root, excluded=("artifact_manifest.json", DATASET_SENTINEL)
        )
        manifest_extra = {
            "method": "AdaBN",
            "method_protocol_id": PROTOCOL_ID,
            "axis_config_path": str(context.axis_config_path),
            "checkpoint_path": str(_axis_attr(context.axis, "checkpoint_path")),
            "formal_development_artifact": formal_development_artifact,
            "paper_result": False,
            "parity_only": parity_only,
            "development_only": True,
            "main_paper_table": False,
            "extra_best_pd_tuning_episodes": 0,
            "parity_receipt_path": getattr(
                context.axis, "parity_receipt_path", None
            )
            and str(context.axis.parity_receipt_path),
            "parity_receipt_sha256": getattr(
                context.axis, "parity_receipt_sha256", None
            ),
            "condition_count": 13,
            "source_artifact": provenance["source_artifact"],
            "children": child_links,
            "files": preliminary_files,
        }
        manifest = checkpoint_axis.build_artifact_manifest(
            dataset_root,
            axis=context.axis,
            required_payloads=required_payloads,
            extra=manifest_extra,
        )
        _write_json(staging / "artifact_manifest.json", manifest)
        _link_file_recorded(
            staging / "artifact_manifest.json",
            dataset_root / "artifact_manifest.json",
            linked,
        )
        completion_extra = {
            "scope": "dataset",
            "method": "AdaBN",
            "method_protocol_id": PROTOCOL_ID,
            "axis_config_path": str(context.axis_config_path),
            "checkpoint_path": str(_axis_attr(context.axis, "checkpoint_path")),
            "formal_development_artifact": formal_development_artifact,
            "formal_artifact": formal_development_artifact,
            "paper_result": False,
            "parity_only": parity_only,
            "development_only": True,
            "main_paper_table": False,
            "extra_best_pd_tuning_episodes": 0,
            "parity_receipt_path": getattr(
                context.axis, "parity_receipt_path", None
            )
            and str(context.axis.parity_receipt_path),
            "parity_receipt_sha256": getattr(
                context.axis, "parity_receipt_sha256", None
            ),
            "condition_count": 13,
            "global_dataset_condition_count": 13,
            "benchmark_sha256": preliminary_files["benchmark.json"]["sha256"],
            "source_dataset_artifact_manifest_sha256": context.source_artifact.manifest_sha256,
            "source_dataset_complete_sha256": context.source_artifact.completion_sha256,
            "source_dataset_benchmark_sha256": context.source_artifact.benchmark_sha256,
            "all_required_gates_passed": True,
        }
        manifest_sha = sha256_file(staging / "artifact_manifest.json")
        completion = {
            "schema_version": 1,
            "complete": True,
            "artifact_contract": checkpoint_axis.ARTIFACT_CONTRACT,
            "artifact_kind": context.axis.artifact_kind,
            "protocol_id": context.axis.protocol_id,
            "axis_config_sha256": context.axis.config_sha256,
            "dataset": context.axis.dataset,
            "checkpoint_role": context.axis.role,
            "checkpoint_sha256": context.axis.checkpoint_sha256,
            "checkpoint_epoch": context.axis.expected_epoch,
            "split_sha256": context.axis.split_sha256,
            "manifest_sha256": manifest_sha,
            "artifact_manifest_sha256": manifest_sha,
            "payload_tree_sha256": manifest["payload_tree"]["sha256"],
            "payload_file_count": manifest["payload_tree"]["file_count"],
            **completion_extra,
        }
        _write_json(staging / DATASET_SENTINEL, completion)
        # COMPLETE is the final publication operation. Any build/link/post-verify
        # failure removes only this attempt's inode links and leaves all 13
        # independently sealed condition directories untouched.
        _link_file_recorded(
            staging / DATASET_SENTINEL,
            dataset_root / DATASET_SENTINEL,
            linked,
        )
        verified = verify_dataset_artifact(
            dataset_root,
            expected_axis=context.axis,
            axis_config_sha256=context.axis_config_sha256,
            expected_source=context.source_artifact,
            formal_development_artifact=formal_development_artifact,
            parity_only=parity_only,
        )
        shutil.rmtree(staging)
        return verified
    except BaseException:
        _rollback_recorded_links(linked)
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_dataset_artifact(
    root: Path,
    *,
    expected_axis: Any,
    axis_config_sha256: str,
    expected_source: SourceArtifact | None,
    formal_development_artifact: bool,
    parity_only: bool,
) -> dict[str, Any]:
    root = Path(os.path.abspath(root.expanduser()))
    shared_audit = checkpoint_axis.verify_published_artifact(
        root,
        expected_axis=expected_axis,
        required_payloads=(
            "benchmark.json",
            "run_config.yaml",
            "provenance.json",
        ),
    )
    _require_equal(
        root.resolve(),
        Path(_axis_attr(expected_axis, "output_dir")).expanduser().resolve(),
        "AdaBN role-first artifact root",
    )
    manifest = _load_json(root / "artifact_manifest.json")
    completion = _load_json(root / DATASET_SENTINEL)
    benchmark = _load_json(root / "benchmark.json")
    role = str(_axis_attr(expected_axis, "role"))
    _role_publication_contract(
        role=role,
        formal_development_artifact=formal_development_artifact,
        parity_only=parity_only,
    )
    checkpoint_sha = str(_axis_attr(expected_axis, "checkpoint_sha256"))
    expected_receipt_sha = _axis_attr(expected_axis, "parity_receipt_sha256")
    expected_receipt_path_raw = _axis_attr(expected_axis, "parity_receipt_path")
    expected_receipt_path = (
        str(expected_receipt_path_raw) if expected_receipt_path_raw is not None else None
    )
    if role == ROLE_BEST_PD and expected_receipt_sha is None:
        raise ValueError("best_pd dataset axis omits the global parity receipt")
    for value, label in ((manifest, "manifest"), (completion, "COMPLETE")):
        _require_equal(value.get("schema_version"), 1, f"dataset {label} schema")
        _require_equal(
            value.get("artifact_contract"),
            checkpoint_axis.ARTIFACT_CONTRACT,
            f"dataset {label} artifact contract",
        )
        _require_equal(value.get("artifact_kind"), "adabn", f"dataset {label} artifact kind")
        _require_equal(
            value.get("protocol_id"),
            _axis_attr(expected_axis, "protocol_id"),
            f"dataset {label} protocol",
        )
        _require_equal(
            value.get("dataset"),
            _axis_attr(expected_axis, "dataset"),
            f"dataset {label} dataset",
        )
        _require_equal(value.get("method"), "AdaBN", f"dataset {label} method")
        _require_equal(value.get("checkpoint_role"), role, f"dataset {label} role")
        _require_equal(value.get("checkpoint_sha256"), checkpoint_sha, f"dataset {label} checkpoint")
        _require_equal(value.get("axis_config_sha256"), axis_config_sha256, f"dataset {label} config")
        _require_equal(
            value.get("parity_receipt_sha256"),
            expected_receipt_sha,
            f"dataset {label} parity receipt",
        )
        _require_equal(
            value.get("parity_receipt_path"),
            expected_receipt_path,
            f"dataset {label} parity receipt path",
        )
        _require_equal(
            value.get("formal_development_artifact"),
            formal_development_artifact,
            f"dataset {label} formal flag",
        )
        _require_equal(
            value.get("parity_only"), parity_only, f"dataset {label} parity flag"
        )
        _development_contract(value, f"dataset {label}")
        _require_equal(value.get("paper_result"), False, f"dataset {label} paper_result")
    _require_equal(benchmark.get("schema_version"), 2, "dataset benchmark schema")
    _require_equal(benchmark.get("protocol_id"), PROTOCOL_ID, "dataset benchmark protocol")
    _require_equal(benchmark.get("method"), "AdaBN", "dataset benchmark method")
    _require_equal(benchmark.get("checkpoint_role"), role, "dataset benchmark role")
    _require_equal(benchmark.get("checkpoint_sha256"), checkpoint_sha, "dataset benchmark checkpoint")
    _require_equal(benchmark.get("axis_config_sha256"), axis_config_sha256, "dataset benchmark config")
    _require_equal(
        benchmark.get("parity_receipt_sha256"),
        expected_receipt_sha,
        "dataset benchmark parity receipt",
    )
    _require_equal(
        benchmark.get("parity_receipt_path"),
        expected_receipt_path,
        "dataset benchmark parity receipt path",
    )
    _development_contract(benchmark, "dataset benchmark")
    _require_equal(benchmark.get("paper_result"), False, "dataset benchmark paper_result")
    _require_equal(
        benchmark.get("formal_development_artifact"),
        formal_development_artifact,
        "dataset benchmark formal flag",
    )
    _require_equal(benchmark.get("parity_only"), parity_only, "dataset benchmark parity flag")
    _require_equal(completion.get("complete"), True, "dataset complete")
    _require_equal(completion.get("condition_count"), 13, "dataset condition count")
    _require_equal(
        completion.get("all_required_gates_passed"),
        True,
        "dataset completion gates",
    )
    _require_equal(
        completion.get("benchmark_sha256"),
        sha256_file(root / "benchmark.json"),
        "dataset benchmark receipt",
    )
    _require_equal(benchmark.get("condition_count"), 13, "benchmark condition count")
    _require_equal(
        int(benchmark.get("evaluated_images_per_condition", -1)),
        int(_axis_attr(expected_axis, "expected_images")),
        "dataset benchmark fixed-test image count",
    )
    if expected_source is not None:
        _require_equal(
            benchmark.get("ordered_ids_sha256"),
            ordered_ids_sha256(expected_source.image_ids),
            "AdaBN benchmark ordered IDs SHA256",
        )
    _require_equal(
        tuple(record["condition_key"] for record in benchmark["conditions"]),
        CONDITION_KEYS,
        "AdaBN benchmark condition order",
    )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("dataset manifest.files must be a mapping")
    verify_recursive_files(
        root,
        files,
        allowed_unlisted=("artifact_manifest.json", DATASET_SENTINEL),
    )
    _require_equal(
        completion.get("manifest_sha256"),
        sha256_file(root / "artifact_manifest.json"),
        "dataset manifest receipt",
    )
    children = manifest.get("children")
    if not isinstance(children, Mapping) or set(children) != set(CONDITION_KEYS):
        raise ValueError("dataset manifest must bind exactly 13 condition children")
    for corruption, severity in CONDITIONS:
        key = condition_key(corruption, severity)
        child = children[key]
        if not isinstance(child, Mapping):
            raise TypeError(f"dataset child receipt must be a mapping: {key}")
        _require_equal(
            child.get("artifact_manifest"),
            f"conditions/{key}/artifact_manifest.json",
            f"{key} child manifest path",
        )
        _require_equal(
            child.get("completion"),
            f"conditions/{key}/{CONDITION_SENTINEL}",
            f"{key} child completion path",
        )
        verified = verify_condition_artifact(
            root / "conditions" / key,
            expected_axis=expected_axis,
            axis_config_sha256=axis_config_sha256,
            expected_source=expected_source,
            expected_corruption=corruption,
            expected_severity=severity,
            formal_development_artifact=formal_development_artifact,
            parity_only=parity_only,
        )
        _require_equal(
            child["artifact_manifest_sha256"],
            verified["manifest_sha256"],
            f"{key} child manifest receipt",
        )
        _require_equal(
            child["completion_sha256"],
            verified["completion_sha256"],
            f"{key} child completion receipt",
        )
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(root / "artifact_manifest.json"),
        "completion": completion,
        "completion_sha256": sha256_file(root / DATASET_SENTINEL),
        "benchmark": benchmark,
        "shared_audit": shared_audit,
    }


def verify_global_artifact(
    root: Path,
    *,
    axis_config: Mapping[str, Any],
    axis_config_sha256: str,
    role: str,
    formal_development_artifact: bool,
    parity_only: bool,
    source_root_override: Path | None = None,
) -> dict[str, Any]:
    """Verify the global 3 x 13 index and every recursively listed byte."""

    root = Path(os.path.abspath(root.expanduser()))
    _require_equal(
        axis_config.get("_runtime", {}).get("config_sha256"),
        axis_config_sha256,
        "global live axis config SHA256",
    )
    if role == ROLE_BEST_PD:
        _require_equal(formal_development_artifact, True, "best_pd formal global axis")
        _require_equal(parity_only, False, "best_pd parity flag")
        if source_root_override is not None:
            raise ValueError("formal best_pd global verification forbids Source overrides")
        parity_receipt = checkpoint_axis.verify_parity_receipt(config=axis_config)
        expected_receipt_sha = parity_receipt["_receipt_sha256"]
    else:
        _require_equal(role, ROLE_BEST_MIOU, "global checkpoint role")
        _require_equal(formal_development_artifact, False, "best_miou formal flag")
        _require_equal(parity_only, True, "best_miou parity flag")
        expected_receipt_sha = None
    manifest = _load_json(root / "artifact_manifest.json")
    completion = _load_json(root / GLOBAL_SENTINEL)
    aggregate = _load_json(root / "aggregate_metrics.json")
    for value, label in (
        (manifest, "global manifest"),
        (completion, "global COMPLETE"),
        (aggregate, "global aggregate"),
    ):
        _require_equal(value.get("schema_version"), SCHEMA_VERSION, f"{label}.schema_version")
        _require_equal(value.get("protocol_id"), PROTOCOL_ID, f"{label}.protocol_id")
        _require_equal(value.get("method"), "AdaBN", f"{label}.method")
        _require_equal(value.get("checkpoint_role"), role, f"{label}.checkpoint_role")
        _require_equal(
            value.get("axis_config_sha256"),
            axis_config_sha256,
            f"{label}.axis_config_sha256",
        )
        _development_contract(value, label)
        _require_equal(value.get("paper_result"), False, f"{label}.paper_result")
        _require_equal(value.get("parity_only"), parity_only, f"{label}.parity_only")
        _require_equal(
            value.get("formal_development_artifact"),
            formal_development_artifact,
            f"{label}.formal_development_artifact",
        )
        _require_equal(
            value.get("parity_receipt_sha256"),
            expected_receipt_sha,
            f"{label}.parity_receipt_sha256",
        )
    _require_equal(completion.get("complete"), True, "global COMPLETE.complete")
    _require_equal(completion.get("scope"), "global", "global COMPLETE.scope")
    _require_equal(
        completion.get("all_required_gates_passed"), True, "global completion gates"
    )
    aggregate_checks = aggregate.get("checks")
    if not isinstance(aggregate_checks, Mapping) or not aggregate_checks:
        raise ValueError("global aggregate.checks must be a non-empty mapping")
    failed_aggregate_checks = sorted(
        key for key, value in aggregate_checks.items() if value is not True
    )
    if failed_aggregate_checks:
        raise ValueError(f"global aggregate checks failed: {failed_aggregate_checks}")
    for value, label in ((completion, "COMPLETE"), (aggregate, "aggregate")):
        _require_equal(value.get("dataset_count"), 3, f"global {label} dataset count")
        _require_equal(
            value.get("condition_count_per_dataset"),
            13,
            f"global {label} condition count",
        )
        _require_equal(
            value.get("global_dataset_condition_count"),
            39,
            f"global {label} cell count",
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("global manifest.files must be a recursive mapping")
    verify_recursive_files(
        root,
        files,
        allowed_unlisted=("artifact_manifest.json", GLOBAL_SENTINEL),
    )
    _require_equal(
        completion.get("artifact_manifest_sha256"),
        sha256_file(root / "artifact_manifest.json"),
        "global manifest receipt",
    )
    _require_equal(
        completion.get("aggregate_metrics_sha256"),
        sha256_file(root / "aggregate_metrics.json"),
        "global aggregate receipt",
    )
    children = manifest.get("children")
    if not isinstance(children, Mapping) or set(children) != set(DATASETS):
        raise ValueError("global manifest must bind exactly three dataset children")
    aggregate_datasets = aggregate.get("datasets")
    if not isinstance(aggregate_datasets, Sequence) or isinstance(
        aggregate_datasets, (str, bytes)
    ):
        raise TypeError("global aggregate.datasets must be a sequence")
    _require_equal(
        tuple(str(record.get("dataset")) for record in aggregate_datasets),
        DATASETS,
        "global aggregate dataset order",
    )
    aggregate_by_dataset = {
        str(record["dataset"]): record for record in aggregate_datasets
    }
    for dataset in DATASETS:
        child = children[dataset]
        if not isinstance(child, Mapping):
            raise TypeError(f"global dataset child must be a mapping: {dataset}")
        _require_equal(
            child.get("artifact_manifest"),
            f"{dataset}/artifact_manifest.json",
            f"global {dataset} manifest path",
        )
        _require_equal(
            child.get("completion"),
            f"{dataset}/{DATASET_SENTINEL}",
            f"global {dataset} completion path",
        )
        manifest_path = root / _safe_relative_path(str(child["artifact_manifest"]))
        completion_path = root / _safe_relative_path(str(child["completion"]))
        _require_equal(
            sha256_file(manifest_path),
            child["artifact_manifest_sha256"],
            f"global {dataset} manifest child receipt",
        )
        _require_equal(
            sha256_file(completion_path),
            child["completion_sha256"],
            f"global {dataset} completion child receipt",
        )
        dataset_root = root / dataset
        axis = checkpoint_axis.resolve_axis(
            axis_config,
            dataset=dataset,
            role=role,
            artifact_kind="adabn",
            output_override=(dataset_root if role == ROLE_BEST_MIOU else None),
        )
        _require_equal(
            Path(_axis_attr(axis, "output_dir")).expanduser().resolve(),
            dataset_root.resolve(),
            f"global {dataset} role-first output root",
        )
        source_dataset_root = _source_dataset_root(
            axis_config,
            role,
            dataset,
            source_root_override=source_root_override,
        )
        source_axis = checkpoint_axis.resolve_axis(
            axis_config,
            dataset=dataset,
            role=role,
            artifact_kind="source",
            output_override=(
                source_dataset_root if role == ROLE_BEST_MIOU else None
            ),
        )
        source_artifact = verify_source_dataset_artifact(
            source_dataset_root,
            expected_axis=source_axis,
            axis_config_sha256=axis_config_sha256,
            dataset=dataset,
        )
        verified_dataset = verify_dataset_artifact(
            dataset_root,
            expected_axis=axis,
            axis_config_sha256=axis_config_sha256,
            expected_source=source_artifact,
            formal_development_artifact=formal_development_artifact,
            parity_only=parity_only,
        )
        aggregate_record = aggregate_by_dataset[dataset]
        if not isinstance(aggregate_record, Mapping):
            raise TypeError(f"global aggregate dataset record must be a mapping: {dataset}")
        for field, expected in (
            ("benchmark", f"{dataset}/benchmark.json"),
            ("artifact_manifest", f"{dataset}/artifact_manifest.json"),
            ("completion", f"{dataset}/{DATASET_SENTINEL}"),
            ("artifact_manifest_sha256", child["artifact_manifest_sha256"]),
            ("completion_sha256", child["completion_sha256"]),
        ):
            _require_equal(
                aggregate_record.get(field),
                expected,
                f"global aggregate {dataset}.{field}",
            )
        for actual, expected, label in (
            (
                verified_dataset["manifest_sha256"],
                child["artifact_manifest_sha256"],
                "manifest",
            ),
            (
                verified_dataset["completion_sha256"],
                child["completion_sha256"],
                "completion",
            ),
            (
                verified_dataset["completion"]["benchmark_sha256"],
                aggregate_record.get("benchmark_sha256"),
                "benchmark",
            ),
        ):
            _require_equal(actual, expected, f"global {dataset} verified {label} receipt")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(root / "artifact_manifest.json"),
        "completion": completion,
        "completion_sha256": sha256_file(root / GLOBAL_SENTINEL),
        "aggregate": aggregate,
    }


def aggregate_results(
    *,
    axis_config_path: Path,
    axis_config: Mapping[str, Any],
    role: str,
    output_root: Path,
    formal_development_artifact: bool,
    parity_only: bool,
    source_root_override: Path | None = None,
) -> dict[str, Any]:
    _role_publication_contract(
        role=role,
        formal_development_artifact=formal_development_artifact,
        parity_only=parity_only,
    )
    if role == ROLE_BEST_PD:
        if source_root_override is not None:
            raise ValueError("formal best_pd aggregation forbids Source root overrides")
        canonical_root = _adabn_dataset_root(
            axis_config, role, DATASETS[0], None
        ).parent
        _require_equal(
            output_root.expanduser().resolve(),
            canonical_root,
            "best_pd global role-first output root",
        )
    axis_sha = _axis_config_sha256(axis_config_path)
    if (output_root / GLOBAL_SENTINEL).exists() or (
        output_root / GLOBAL_SENTINEL
    ).is_symlink():
        verified_existing = verify_global_artifact(
            output_root,
            axis_config=axis_config,
            axis_config_sha256=axis_sha,
            role=role,
            formal_development_artifact=formal_development_artifact,
            parity_only=parity_only,
            source_root_override=source_root_override,
        )
        return {
            "published_output_dir": str(output_root),
            "aggregate": verified_existing["aggregate"],
            "completion": verified_existing["completion"],
            "verification": verified_existing,
            "resumed_existing_global": True,
        }
    if role == ROLE_BEST_PD:
        parity_receipt = checkpoint_axis.verify_parity_receipt(config=axis_config)
        global_parity_receipt_sha256 = parity_receipt["_receipt_sha256"]
    else:
        global_parity_receipt_sha256 = None
    datasets: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_root = output_root / dataset
        axis = checkpoint_axis.resolve_axis(
            axis_config,
            dataset=dataset,
            role=role,
            artifact_kind="adabn",
            output_override=(dataset_root if role == ROLE_BEST_MIOU else None),
        )
        source_dataset_root = _source_dataset_root(
            axis_config,
            role,
            dataset,
            source_root_override=source_root_override,
        )
        source_axis = checkpoint_axis.resolve_axis(
            axis_config,
            dataset=dataset,
            role=role,
            artifact_kind="source",
            output_override=(source_dataset_root if role == ROLE_BEST_MIOU else None),
        )
        source_artifact = verify_source_dataset_artifact(
            source_dataset_root,
            expected_axis=source_axis,
            axis_config_sha256=axis_sha,
            dataset=dataset,
        )
        if not (dataset_root / DATASET_SENTINEL).is_file():
            context = prepare_dataset_context(
                axis_config_path=axis_config_path,
                axis_config=axis_config,
                role=role,
                dataset=dataset,
                verify_all_cache_file_hashes=formal_development_artifact,
                axis_output_override=(
                    dataset_root if role == ROLE_BEST_MIOU else None
                ),
                source_root_override=source_root_override,
            )
            finalize_dataset(
                context=context,
                dataset_root=dataset_root,
                role=role,
                formal_development_artifact=formal_development_artifact,
                parity_only=parity_only,
            )
        verified = verify_dataset_artifact(
            dataset_root,
            expected_axis=axis,
            axis_config_sha256=axis_sha,
            expected_source=source_artifact,
            formal_development_artifact=formal_development_artifact,
            parity_only=parity_only,
        )
        datasets.append(
            {
                "dataset": dataset,
                "benchmark": f"{dataset}/benchmark.json",
                "benchmark_sha256": verified["completion"]["benchmark_sha256"],
                "artifact_manifest": f"{dataset}/artifact_manifest.json",
                "artifact_manifest_sha256": verified["manifest_sha256"],
                "completion": f"{dataset}/COMPLETE.json",
                "completion_sha256": verified["completion_sha256"],
                "conditions": verified["benchmark"]["conditions"],
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    visible_entries = {path.name for path in output_root.iterdir()}
    _require_equal(
        visible_entries,
        set(DATASETS),
        "global axis exact dataset children before sealing",
    )
    for name in (
        "aggregate_metrics.json",
        "run_config.yaml",
        "provenance.json",
        "artifact_manifest.json",
        GLOBAL_SENTINEL,
    ):
        if (output_root / name).exists() or (output_root / name).is_symlink():
            raise FileExistsError(f"global axis metadata already exists: {output_root / name}")
    staging = output_root.parent / (
        f".{output_root.name}.global-build-{os.getpid()}-{time.time_ns()}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "axis_config_path": str(axis_config_path),
            "axis_config_sha256": axis_sha,
            "checkpoint_role": role,
            "method": "AdaBN",
            "formal_development_artifact": formal_development_artifact,
            "paper_result": False,
            "parity_only": parity_only,
            "development_only": True,
            "main_paper_table": False,
            "extra_best_pd_tuning_episodes": 0,
            "parity_receipt_sha256": global_parity_receipt_sha256,
            "dataset_count": 3,
            "condition_count_per_dataset": 13,
            "global_dataset_condition_count": 39,
            "datasets": datasets,
            "delta_role": "report_only_never_used_for_selection_or_tuning",
            "checks": {
                "exact_three_datasets": True,
                "exact_13_conditions_per_dataset": True,
                "exact_39_dataset_conditions": True,
                "all_source_and_adabn_manifests_recursively_verified": True,
                "zero_additional_best_pd_tuning": True,
            },
        }
        _write_json(staging / "aggregate_metrics.json", aggregate)
        _write_yaml(
            staging / "run_config.yaml",
            {**deepcopy(dict(axis_config)), "runtime": {"mode": "aggregate_only", "checkpoint_role": role}},
        )
        _write_json(
            staging / "provenance.json",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "axis_config_path": str(axis_config_path),
                "axis_config_sha256": axis_sha,
                "checkpoint_role": role,
                "formal_development_artifact": formal_development_artifact,
                "paper_result": False,
                "parity_only": parity_only,
                "development_only": True,
                "main_paper_table": False,
                "extra_best_pd_tuning_episodes": 0,
                "parity_receipt_sha256": global_parity_receipt_sha256,
                "datasets": {
                    item["dataset"]: {
                        "artifact_manifest_sha256": item["artifact_manifest_sha256"],
                        "completion_sha256": item["completion_sha256"],
                    }
                    for item in datasets
                },
            },
        )
        files = _artifact_files(
            output_root,
            excluded=(
                "aggregate_metrics.json",
                "run_config.yaml",
                "provenance.json",
                "artifact_manifest.json",
                GLOBAL_SENTINEL,
            ),
        )
        for name in ("aggregate_metrics.json", "run_config.yaml", "provenance.json"):
            files[name] = {
                "sha256": sha256_file(staging / name),
                "bytes": (staging / name).stat().st_size,
            }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "method": "AdaBN",
            "axis_config_sha256": axis_sha,
            "checkpoint_role": role,
            "formal_development_artifact": formal_development_artifact,
            "paper_result": False,
            "parity_only": parity_only,
            "development_only": True,
            "main_paper_table": False,
            "extra_best_pd_tuning_episodes": 0,
            "parity_receipt_sha256": global_parity_receipt_sha256,
            "children": {
                item["dataset"]: {
                    "artifact_manifest": item["artifact_manifest"],
                    "artifact_manifest_sha256": item["artifact_manifest_sha256"],
                    "completion": item["completion"],
                    "completion_sha256": item["completion_sha256"],
                }
                for item in datasets
            },
            "files": files,
        }
        _write_json(staging / "artifact_manifest.json", manifest)
        completion = {
            "complete": True,
            "scope": "global",
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "method": "AdaBN",
            "axis_config_path": str(axis_config_path),
            "axis_config_sha256": axis_sha,
            "checkpoint_role": role,
            "formal_development_artifact": formal_development_artifact,
            "formal_artifact": formal_development_artifact,
            "paper_result": False,
            "parity_only": parity_only,
            "development_only": True,
            "main_paper_table": False,
            "extra_best_pd_tuning_episodes": 0,
            "parity_receipt_sha256": global_parity_receipt_sha256,
            "dataset_count": 3,
            "condition_count_per_dataset": 13,
            "global_dataset_condition_count": 39,
            "aggregate_metrics_sha256": sha256_file(staging / "aggregate_metrics.json"),
            "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
            "all_required_gates_passed": True,
        }
        _write_json(staging / GLOBAL_SENTINEL, completion)
        # ``output_root`` already contains three independently sealed dataset
        # children, so a directory rename cannot publish the global index.  Use
        # no-replace hard links and publish COMPLETE last.  A failed attempt is
        # rolled back to the exact pre-publication tree.
        linked: list[tuple[Path, int, int]] = []
        try:
            for name in (
                "aggregate_metrics.json",
                "run_config.yaml",
                "provenance.json",
                "artifact_manifest.json",
                GLOBAL_SENTINEL,
            ):
                destination = output_root / name
                _link_file_recorded(staging / name, destination, linked)
            verified_global = verify_global_artifact(
                output_root,
                axis_config=axis_config,
                axis_config_sha256=axis_sha,
                role=role,
                formal_development_artifact=formal_development_artifact,
                parity_only=parity_only,
                source_root_override=source_root_override,
            )
        except BaseException:
            _rollback_recorded_links(linked)
            raise
        shutil.rmtree(staging)
        return {
            "published_output_dir": str(output_root),
            "aggregate": aggregate,
            "completion": completion,
            "verification": verified_global,
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_adabn_artifact(
    root: Path,
    *,
    expected_axis: Any,
    expected_source: SourceArtifact,
) -> dict[str, Any]:
    """Public dataset-level verifier for downstream comparison/receipt code."""

    role = str(_axis_attr(expected_axis, "role"))
    return verify_dataset_artifact(
        root,
        expected_axis=expected_axis,
        axis_config_sha256=str(_axis_attr(expected_axis, "config_sha256")),
        expected_source=expected_source,
        formal_development_artifact=(role == ROLE_BEST_PD),
        parity_only=(role == ROLE_BEST_MIOU),
    )


def run_adabn_axis(
    *,
    axis_config: Mapping[str, Any],
    axis: Any,
    source_artifact: Path,
    output_dir: Path | None = None,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Programmatic full 13-condition entry point for one dataset/role cell."""

    role = str(_axis_attr(axis, "role"))
    dataset = str(_axis_attr(axis, "dataset"))
    axis_config_path = Path(_axis_attr(axis, "config_path"))
    final_root = Path(output_dir or _axis_attr(axis, "output_dir")).expanduser().resolve()
    _require_equal(
        final_root,
        Path(_axis_attr(axis, "output_dir")).expanduser().resolve(),
        "provided CheckpointAxis output_dir",
    )
    canonical_source = _source_dataset_root(axis_config, role, dataset)
    _require_equal(
        source_artifact.expanduser().resolve(),
        canonical_source,
        "role-frozen Source artifact root",
    )
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError(f"AdaBN checkpoint-axis dataset output exists: {final_root}")
    resolved = prepare_dataset_context(
        axis_config_path=axis_config_path,
        axis_config=axis_config,
        role=role,
        dataset=dataset,
        verify_all_cache_file_hashes=True,
        axis_output_override=(final_root if role == ROLE_BEST_MIOU else None),
        source_root_override=source_artifact.expanduser().resolve().parent,
    )
    for field in ("role", "dataset", "checkpoint_sha256", "config_sha256"):
        _require_equal(
            _axis_attr(resolved.axis, field),
            _axis_attr(axis, field),
            f"provided/resolved CheckpointAxis.{field}",
        )
    _require_equal(
        Path(_axis_attr(resolved.axis, "output_dir")).expanduser().resolve(),
        final_root,
        "canonical resolved AdaBN output_dir",
    )
    _require_equal(
        resolved.source_artifact.root,
        source_artifact.expanduser().resolve(),
        "provided Source artifact root",
    )
    formal = role == ROLE_BEST_PD
    parity_only = role == ROLE_BEST_MIOU
    condition_results: list[dict[str, Any]] = []
    for corruption, severity in CONDITIONS:
        condition_results.append(
            publish_condition(
                context=resolved,
                role=role,
                dataset=dataset,
                corruption=corruption,
                severity=severity,
                device_spec=device_name,
                final=_condition_final(final_root, corruption, severity),
                formal_development_artifact=formal,
                parity_only=parity_only,
                max_images=None,
            )
        )
    dataset_result = finalize_dataset(
        context=resolved,
        dataset_root=final_root,
        role=role,
        formal_development_artifact=formal,
        parity_only=parity_only,
    )
    return {
        "published_output_dir": str(final_root),
        "dataset": dataset,
        "checkpoint_role": role,
        "condition_count": 13,
        "conditions": condition_results,
        "artifact_manifest_sha256": dataset_result["manifest_sha256"],
        "complete_sha256": dataset_result["completion_sha256"],
    }


def resolve_selection(args: argparse.Namespace, config: Mapping[str, Any]) -> RunSelection:
    role = str(args.checkpoint_role)
    source_artifact_root = getattr(args, "source_artifact_root", None)
    if role not in ROLES:
        raise ValueError(f"unsupported checkpoint role: {role}")
    parity_only = bool(args.parity_only)
    if role == ROLE_BEST_MIOU:
        if not parity_only or args.output_dir is None:
            raise ValueError(
                "best_miou is allowed only with --parity-only and an explicit --output-dir"
            )
        project_root = Path(str(config["_runtime"]["project_root"]))
        candidate_roots = config["parity_gate"]["candidate_roots"]
        expected_adabn_root = (project_root / candidate_roots["adabn"]).resolve()
        _require_equal(
            args.output_dir.expanduser().resolve(),
            expected_adabn_root,
            "best_miou frozen AdaBN parity candidate root",
        )
        expected_source_root = (project_root / candidate_roots["source"]).resolve()
        if source_artifact_root is not None:
            _require_equal(
                source_artifact_root.expanduser().resolve(),
                expected_source_root,
                "best_miou frozen Source parity candidate root",
            )
        source_artifact_root = expected_source_root
        if args.max_images is not None:
            raise ValueError(
                "best_miou parity requires every fixed-test image; --max-images is forbidden"
            )
    elif parity_only:
        raise ValueError("--parity-only is reserved for best_miou v1/v2 equivalence")
    else:
        if args.output_dir is not None:
            raise ValueError(
                "best_pd is a formal role-first development axis; --output-dir is forbidden"
            )
        if source_artifact_root is not None:
            raise ValueError(
                "formal best_pd must consume the canonical role-matched Source v2 root"
            )
        if args.max_images is not None:
            raise ValueError(
                "best_pd requires the complete fixed test split; --max-images is forbidden"
            )
    formal = role == ROLE_BEST_PD and args.max_images is None and args.output_dir is None
    if args.aggregate_only and (args.dataset or args.condition or args.max_images is not None):
        raise ValueError("--aggregate-only cannot be combined with shard/image filters")
    selected_datasets = (args.dataset,) if args.dataset else DATASETS
    selected_conditions = (
        (legacy.CONDITION_BY_KEY[args.condition],) if args.condition else CONDITIONS
    )
    if args.output_dir is None:
        sample_dataset = selected_datasets[0]
        dataset_root = _adabn_dataset_root(config, role, sample_dataset, None)
        output_root = dataset_root.parent
    else:
        output_root = args.output_dir.expanduser().resolve()
    return RunSelection(
        checkpoint_role=role,  # type: ignore[arg-type]
        datasets=selected_datasets,
        conditions=selected_conditions,
        output_root=output_root,
        formal_development_artifact=formal,
        parity_only=parity_only,
        max_images=args.max_images,
        aggregate_only=bool(args.aggregate_only),
        source_root_override=(
            source_artifact_root.expanduser().resolve()
            if source_artifact_root is not None
            else None
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    axis_config_path = args.axis_config.expanduser().resolve()
    axis_config = checkpoint_axis.load_axis_config(axis_config_path)
    selection = resolve_selection(args, axis_config)
    if selection.aggregate_only:
        return aggregate_results(
            axis_config_path=axis_config_path,
            axis_config=axis_config,
            role=selection.checkpoint_role,
            output_root=selection.output_root,
            formal_development_artifact=selection.formal_development_artifact,
            parity_only=selection.parity_only,
            source_root_override=selection.source_root_override,
        )
    results: list[dict[str, Any]] = []
    for dataset in selection.datasets:
        context = prepare_dataset_context(
            axis_config_path=axis_config_path,
            axis_config=axis_config,
            role=selection.checkpoint_role,
            dataset=dataset,
            verify_all_cache_file_hashes=selection.formal_development_artifact,
            axis_output_override=(
                selection.output_root / dataset
                if selection.checkpoint_role == ROLE_BEST_MIOU
                or args.output_dir is not None
                else None
            ),
            source_root_override=selection.source_root_override,
        )
        dataset_root = selection.output_root / dataset
        if (dataset_root / DATASET_SENTINEL).exists() or (
            dataset_root / DATASET_SENTINEL
        ).is_symlink():
            verified_dataset = verify_dataset_artifact(
                dataset_root,
                expected_axis=context.axis,
                axis_config_sha256=context.axis_config_sha256,
                expected_source=context.source_artifact,
                formal_development_artifact=selection.formal_development_artifact,
                parity_only=selection.parity_only,
            )
            results.append(
                {
                    "published_output_dir": str(dataset_root),
                    "dataset": dataset,
                    "resumed_existing_dataset": True,
                    "artifact_manifest_sha256": verified_dataset["manifest_sha256"],
                    "complete_sha256": verified_dataset["completion_sha256"],
                }
            )
            continue
        for corruption, severity in selection.conditions:
            condition_root = _condition_final(dataset_root, corruption, severity)
            if condition_root.exists() or condition_root.is_symlink():
                verified_condition = verify_condition_artifact(
                    condition_root,
                    expected_axis=context.axis,
                    axis_config_sha256=context.axis_config_sha256,
                    expected_source=context.source_artifact,
                    expected_corruption=corruption,
                    expected_severity=severity,
                    formal_development_artifact=selection.formal_development_artifact,
                    parity_only=selection.parity_only,
                )
                results.append(
                    {
                        "published_output_dir": str(condition_root),
                        "metrics": verified_condition["metrics"],
                        "completion": verified_condition["completion"],
                        "resumed_existing_condition": True,
                    }
                )
            else:
                results.append(
                    publish_condition(
                        context=context,
                        role=selection.checkpoint_role,
                        dataset=dataset,
                        corruption=corruption,
                        severity=severity,
                        device_spec=args.device,
                        final=condition_root,
                        formal_development_artifact=selection.formal_development_artifact,
                        parity_only=selection.parity_only,
                        max_images=selection.max_images,
                    )
                )
        if tuple(selection.conditions) == CONDITIONS:
            finalize_dataset(
                context=context,
                dataset_root=dataset_root,
                role=selection.checkpoint_role,
                formal_development_artifact=selection.formal_development_artifact,
                parity_only=selection.parity_only,
            )
    if tuple(selection.datasets) == DATASETS and tuple(selection.conditions) == CONDITIONS:
        return aggregate_results(
            axis_config_path=axis_config_path,
            axis_config=axis_config,
            role=selection.checkpoint_role,
            output_root=selection.output_root,
            formal_development_artifact=selection.formal_development_artifact,
            parity_only=selection.parity_only,
            source_root_override=selection.source_root_override,
        )
    return {
        "published_output_dir": str(selection.output_root),
        "conditions": results,
        "global_complete_created": False,
    }


def positive_integer(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-config", type=Path, default=DEFAULT_AXIS_CONFIG)
    parser.add_argument("--checkpoint-role", required=True, choices=ROLES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--condition", choices=CONDITION_KEYS)
    parser.add_argument("--max-images", type=positive_integer)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--source-artifact-root",
        type=Path,
        help=(
            "Explicit root containing <dataset>/ Source v2 artifacts. For "
            "best_miou it may only repeat the config-frozen candidate root; "
            "formal best_pd always uses the canonical Source root."
        ),
    )
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(f"Artifacts: {result['published_output_dir']}")
    return 0


__all__ = [
    "ARTIFACT_KIND",
    "CONDITIONS",
    "CONDITION_KEYS",
    "DATASETS",
    "PROTOCOL_ID",
    "SourceArtifact",
    "aggregate_results",
    "build_parser",
    "finalize_dataset",
    "prepare_dataset_context",
    "publish_condition",
    "resolve_selection",
    "run",
    "run_adabn_axis",
    "verify_adabn_artifact",
    "verify_condition_artifact",
    "verify_dataset_artifact",
    "verify_global_artifact",
    "verify_recursive_files",
    "verify_source_dataset_artifact",
]
