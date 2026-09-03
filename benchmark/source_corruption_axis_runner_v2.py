"""Checkpoint-role aware Source corruption benchmark (v2).

This module deliberately imports the frozen v1 runner instead of changing it.
The v1 cache/evaluator helpers remain the numerical implementation of the
13-condition benchmark; v2 adds a checkpoint axis, recursively sealed inputs,
role-first outputs, no-replace publication, a non-authorizing per-dataset
parity audit, and consumption of the one global best-mIoU parity receipt.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import stat
import time
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from benchmark import checkpoint_axis as checkpoint_axis_api
from dataio.corruption_cache import (
    CachedCorruptionDataset,
    TensorSequenceHasher,
    condition_key,
    ordered_ids_sha256,
    sha256_file,
    verify_cache_artifact,
)
from metrics.irstd_metrics import UnifiedResearchEvaluator, probabilities_from_logits
from metrics.official_metric_adapter import OfficialMetricAdapter
import run_source_corruption_benchmark as v1
import test_fixed_split_source as clean_v1
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PROTOCOL = (
    PROJECT_ROOT / "configs/source_corruption_benchmark_fixed_splits.yaml"
)
SOURCE_PROTOCOL_SHA256 = (
    "de0f5434a05486d7dba0179c3cee987c68bb14a55c16042a678b47195644c784"
)
METHOD = "Source"
ARTIFACT_KIND = "source"
SOURCE_PARITY_AUDIT_NAME = "BEST_MIOU_SOURCE_PARITY_AUDIT.json"
SOURCE_PARITY_AUDIT_TYPE = "source_corruption_checkpoint_axis_v2_best_miou_parity_audit"
GLOBAL_PARITY_RECEIPT = (
    PROJECT_ROOT
    / "results/checkpoint_axis_v2_parity/best_miou/PARITY_RECEIPT.json"
)
GLOBAL_PARITY_RECEIPT_TYPE = "checkpoint_axis_v2_best_miou_exact_parity"
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")


def _axis_value(axis: Any, name: str) -> Any:
    """Read an axis field while retaining a useful error across API revisions."""

    if not hasattr(axis, name):
        raise AttributeError(f"CheckpointAxis is missing required field {name!r}")
    return getattr(axis, name)


def _axis_config_sha256(axis: Any) -> str:
    for name in ("config_sha256", "axis_config_sha256", "protocol_sha256"):
        if hasattr(axis, name):
            value = getattr(axis, name)
            if isinstance(value, str) and len(value) == 64:
                return value
    raise AttributeError("CheckpointAxis does not expose its axis config SHA256")


def _axis_dataset(axis: Any) -> str:
    return str(_axis_value(axis, "dataset"))


def _axis_role(axis: Any) -> str:
    return str(_axis_value(axis, "role"))


def _axis_checkpoint_path(axis: Any) -> Path:
    return Path(_axis_value(axis, "checkpoint_path")).expanduser().resolve()


def _axis_checkpoint_sha256(axis: Any) -> str:
    return str(_axis_value(axis, "checkpoint_sha256"))


def _axis_development_only(axis: Any) -> bool:
    value = _axis_value(axis, "development_only")
    if type(value) is not bool:
        raise TypeError("CheckpointAxis.development_only must be bool")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return dict(loaded)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            loaded = json.loads(line)
            if not isinstance(loaded, Mapping):
                raise TypeError(f"expected JSON object at {path}:{line_number}")
            records.append(dict(loaded))
    return records


def _safe_relative(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"unsafe {label}: {raw!r}")
    return path


def _regular_file_nofollow(path: Path, *, label: str) -> os.stat_result:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {label}: {path}") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return status


def _payload_files_nofollow(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"artifact root must be a non-symlink directory: {root}")
    files: set[str] = set()
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"artifact contains a symlink: {relative}")
        if stat.S_ISREG(status.st_mode):
            files.add(relative)
        elif not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"artifact contains a non-regular entry: {relative}")
    return files


def _verify_flat_artifact(
    root: Path,
    *,
    expected_role: str | None = None,
    expected_dataset: str | None = None,
) -> dict[str, Any]:
    """Recursively verify the frozen v1 flat-manifest artifact/tree."""

    # Keep the lexical final component so a symlink root cannot be hidden by
    # ``Path.resolve()`` before the no-follow check below.
    root = Path(os.path.abspath(root.expanduser()))
    manifest_path = root / "artifact_manifest.json"
    complete_path = root / "COMPLETE.json"
    _regular_file_nofollow(manifest_path, label="artifact manifest")
    _regular_file_nofollow(complete_path, label="completion sentinel")
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    if complete.get("complete") is not True:
        raise ValueError("artifact completion sentinel is not complete=true")
    manifest_sha = sha256_file(manifest_path)
    if complete.get("artifact_manifest_sha256") != manifest_sha:
        raise ValueError("completion sentinel does not bind artifact manifest")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("artifact manifest files must be a mapping")
    expected_files = {"artifact_manifest.json", "COMPLETE.json"}
    for raw_relative, raw_record in files.items():
        relative = _safe_relative(raw_relative, label="manifest payload path")
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"manifest file record must be a mapping: {raw_relative}")
        path = root / relative
        status = _regular_file_nofollow(path, label=f"artifact payload {relative}")
        expected_sha = raw_record.get("sha256")
        expected_bytes = raw_record.get("bytes", raw_record.get("size_bytes"))
        if sha256_file(path) != expected_sha:
            raise ValueError(f"artifact payload SHA256 drifted: {relative}")
        if int(status.st_size) != int(expected_bytes):
            raise ValueError(f"artifact payload size drifted: {relative}")
        expected_files.add(str(relative))
    actual_files = _payload_files_nofollow(root)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(f"artifact file tree drifted; missing={missing}, extra={extra}")
    if expected_role is not None:
        if manifest.get("checkpoint_role") != expected_role:
            raise ValueError("artifact manifest checkpoint_role mismatch")
        if complete.get("checkpoint_role") != expected_role:
            raise ValueError("completion checkpoint_role mismatch")
    if expected_dataset is not None:
        if manifest.get("dataset") != expected_dataset:
            raise ValueError("artifact manifest dataset mismatch")
        if complete.get("dataset") != expected_dataset:
            raise ValueError("completion dataset mismatch")
    return {
        "root": str(root),
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "complete": complete,
        "complete_sha256": sha256_file(complete_path),
        "verified_payload_count": len(files),
    }


def verify_source_artifact(
    root: Path,
    *,
    expected_axis: Any | None = None,
) -> dict[str, Any]:
    """Public recursive verifier used by downstream AdaBN checkpoint-axis v2."""

    audit = checkpoint_axis_api.verify_published_artifact(
        Path(root),
        expected_axis=expected_axis,
        required_payloads=("benchmark.json", "run_config.json"),
    )
    manifest = audit["manifest"]
    complete = audit["complete"]
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("not a Source corruption checkpoint-axis v2 artifact")
    if manifest.get("development_only") is not True:
        raise ValueError("Source v2 artifact must be development_only=true")
    if manifest.get("main_paper_table") is not False:
        raise ValueError("Source v2 artifact must be main_paper_table=false")
    if int(manifest.get("extra_best_pd_tuning_episodes", -1)) != 0:
        raise ValueError("Source v2 artifact must record zero best_pd tuning episodes")
    if int(complete.get("condition_count", -1)) != 13:
        raise ValueError("Source v2 completion must bind 13 conditions")
    if int(manifest.get("condition_count", -1)) != 13:
        raise ValueError("Source v2 manifest must bind 13 conditions")
    payload_records = audit["payload_tree"]["files"]
    expected_files = {
        record["path"]: {
            "sha256": record["sha256"],
            "bytes": int(record["size_bytes"]),
        }
        for record in payload_records
    }
    if manifest.get("files") != expected_files:
        raise ValueError("comparison-friendly files mapping does not match recursive ledger")
    condition_files = manifest.get("condition_files")
    if not isinstance(condition_files, Mapping):
        raise TypeError("Source v2 condition_files must be a mapping")
    expected_condition_order = tuple(
        condition_key(corruption, severity)
        for corruption, severity in v1._conditions(
            v1._load_protocol(DEFAULT_SOURCE_PROTOCOL, "IRSTD-1K")[0]
        )
    )
    if set(condition_files) != set(expected_condition_order):
        raise ValueError("Source v2 condition_files do not cover the exact 13 conditions")
    for key, raw_record in condition_files.items():
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"condition file record must be a mapping: {key}")
        expected_names = {"metrics", "per_image", "probability_shard"}
        if set(raw_record) != expected_names:
            raise ValueError(f"condition file mapping drifted: {key}")
        for name, raw_file in raw_record.items():
            if not isinstance(raw_file, Mapping):
                raise TypeError(f"condition file entry must be a mapping: {key}/{name}")
            relative = str(raw_file.get("path"))
            if relative not in expected_files:
                raise ValueError(f"condition file is absent from recursive ledger: {relative}")
            if raw_file.get("sha256") != expected_files[relative]["sha256"]:
                raise ValueError(f"condition file SHA256 mapping drifted: {relative}")
    if expected_axis is not None:
        for actual, expected, label in (
            (manifest.get("axis_config_sha256"), _axis_config_sha256(expected_axis), "axis config"),
            (manifest.get("checkpoint_sha256"), _axis_checkpoint_sha256(expected_axis), "checkpoint"),
            (complete.get("axis_config_sha256"), _axis_config_sha256(expected_axis), "completion axis config"),
            (complete.get("checkpoint_sha256"), _axis_checkpoint_sha256(expected_axis), "completion checkpoint"),
        ):
            if actual != expected:
                raise ValueError(f"{label} SHA256 mismatch")
    benchmark_path = Path(audit["root"]) / "benchmark.json"
    benchmark = _load_json(benchmark_path)
    if benchmark.get("checkpoint_role") != manifest.get("checkpoint_role"):
        raise ValueError("benchmark/manifest checkpoint role mismatch")
    if benchmark.get("dataset") != manifest.get("dataset"):
        raise ValueError("benchmark/manifest dataset mismatch")
    if int(benchmark.get("condition_count", -1)) != 13:
        raise ValueError("benchmark condition count mismatch")
    benchmark_conditions = benchmark.get("conditions")
    if not isinstance(benchmark_conditions, Sequence) or isinstance(
        benchmark_conditions, (str, bytes)
    ):
        raise TypeError("benchmark conditions must be a sequence")
    if tuple(record.get("condition_key") for record in benchmark_conditions) != expected_condition_order:
        raise ValueError("benchmark condition order does not match the frozen 13-condition contract")
    evaluated = int(benchmark.get("evaluated_images_per_condition", -1))
    if evaluated < 1:
        raise ValueError("benchmark evaluated image count must be positive")
    expected_order: list[str] | None = None
    root_path = Path(audit["root"])
    for key in sorted(condition_files):
        file_record = condition_files[key]
        metrics_path = root_path / str(file_record["metrics"]["path"])
        records_path = root_path / str(file_record["per_image"]["path"])
        shard_path = root_path / str(file_record["probability_shard"]["path"])
        metrics = _load_json(metrics_path)
        records = _load_jsonl(records_path)
        if metrics.get("condition_key") != key:
            raise ValueError(f"condition metrics key mismatch: {key}")
        if metrics.get("checkpoint_role") != manifest.get("checkpoint_role"):
            raise ValueError(f"condition checkpoint role mismatch: {key}")
        if int(metrics.get("evaluated_images", -1)) != evaluated or len(records) != evaluated:
            raise ValueError(f"condition evaluated image count mismatch: {key}")
        probability = np.load(shard_path, mmap_mode="r", allow_pickle=False)
        if probability.shape != (evaluated, 256, 256) or probability.dtype != np.dtype("<f4"):
            raise ValueError(f"condition probability shard shape/dtype mismatch: {key}")
        del probability
        order = [str(record.get("image_id")) for record in records]
        if len(set(order)) != evaluated:
            raise ValueError(f"condition image IDs are duplicated: {key}")
        if expected_order is None:
            expected_order = order
        elif order != expected_order:
            raise ValueError(f"condition image ID order drifted: {key}")
        mask_prefix = f"conditions/{key}/prediction_masks_256/"
        masks = [path for path in expected_files if path.startswith(mask_prefix)]
        if len(masks) != evaluated:
            raise ValueError(f"condition prediction mask count mismatch: {key}")
        for index, record in enumerate(records):
            if int(record.get("index", -1)) != index:
                raise ValueError(f"condition per-image index drifted: {key}/{index}")
            mask_relative = f"conditions/{key}/{record.get('prediction_mask')}"
            if mask_relative not in expected_files:
                raise ValueError(f"condition prediction mask is unsealed: {key}/{index}")
            if record.get("prediction_mask_sha256") != expected_files[mask_relative]["sha256"]:
                raise ValueError(f"condition prediction mask hash drifted: {key}/{index}")
            if record.get("probability_shard_sha256") != expected_files[
                str(file_record["probability_shard"]["path"])
            ]["sha256"]:
                raise ValueError(f"condition probability shard binding drifted: {key}/{index}")
    assert expected_order is not None
    if benchmark.get("ordered_ids_sha256") != ordered_ids_sha256(expected_order):
        raise ValueError("benchmark ordered image ID SHA256 mismatch")
    parity = manifest.get("best_miou_parity_gate")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise ValueError("Source v2 parity gate evidence is absent or failed")
    if manifest.get("checkpoint_role") == "best_pd":
        if parity.get("mode") != "consumer" or not parity.get("receipt_sha256"):
            raise ValueError("best_pd Source artifact lacks its global parity receipt seal")
    elif parity.get("authorization_receipt") is not False:
        raise ValueError("best_miou Source audit must not claim authorization")
    audit["benchmark"] = benchmark
    return audit


def _verify_v1_parity_reference(
    root: Path,
    dataset: str,
    *,
    axis_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(os.path.abspath(root.expanduser()))
    if axis_config is not None:
        contract = checkpoint_axis_api.get_parity_reference(
            axis_config, artifact_kind="source", dataset=dataset
        )
        expected_root = Path(axis_config["_runtime"]["project_root"]) / str(
            contract["root"]
        )
        expected_root = Path(os.path.abspath(expected_root))
        if root != expected_root:
            raise ValueError("v1 parity reference is not the config-frozen Source root")
        ledger = checkpoint_axis_api.artifact_tree_ledger(root)
        if ledger["algorithm"] != contract["tree_algorithm"]:
            raise ValueError("v1 parity reference tree algorithm mismatch")
        if ledger["sha256"] != contract["tree_sha256"]:
            raise ValueError("v1 parity reference recursive tree SHA256 drifted")
        if int(ledger["file_count"]) != int(contract["tree_file_count"]):
            raise ValueError("v1 parity reference recursive file count drifted")
    audit = _verify_flat_artifact(root, expected_dataset=dataset)
    manifest = audit["manifest"]
    complete = audit["complete"]
    if manifest.get("protocol_sha256") != SOURCE_PROTOCOL_SHA256:
        raise ValueError("v1 parity reference protocol SHA256 mismatch")
    if complete.get("protocol_sha256") != SOURCE_PROTOCOL_SHA256:
        raise ValueError("v1 parity reference completion protocol mismatch")
    benchmark = _load_json(Path(audit["root"]) / "benchmark.json")
    if benchmark.get("method") != METHOD or benchmark.get("condition_count") != 13:
        raise ValueError("v1 parity reference benchmark contract mismatch")
    if benchmark.get("checkpoint_metadata", {}).get("selection_metric") != "miou":
        raise ValueError("v1 parity reference is not best_miou")
    audit["benchmark"] = benchmark
    if axis_config is not None:
        contract = checkpoint_axis_api.get_parity_reference(
            axis_config, artifact_kind="source", dataset=dataset
        )
        if audit["manifest_sha256"] != contract["artifact_manifest_sha256"]:
            raise ValueError("v1 parity reference manifest config binding drifted")
        if audit["complete_sha256"] != contract["complete_sha256"]:
            raise ValueError("v1 parity reference completion config binding drifted")
        if sha256_file(root / str(contract["summary_file"])) != contract["summary_sha256"]:
            raise ValueError("v1 parity reference summary config binding drifted")
    return audit


def _clean_reference(
    root: Path,
    *,
    axis: Any,
    image_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    required = ("metrics.json", "per_image.jsonl")
    audit = checkpoint_axis_api.verify_published_artifact(
        root, expected_axis=axis, required_payloads=required
    )
    metrics = _load_json(root / "metrics.json")
    records = _load_jsonl(root / "per_image.jsonl")
    if metrics.get("checkpoint_role") != _axis_role(axis):
        raise ValueError("clean artifact checkpoint role mismatch")
    if metrics.get("dataset") != _axis_dataset(axis):
        raise ValueError("clean artifact dataset mismatch")
    if len(records) != len(image_ids):
        raise ValueError("clean artifact record count mismatch")
    for index, (expected_id, record) in enumerate(zip(image_ids, records)):
        if int(record.get("index", -1)) != index or record.get("image_id") != expected_id:
            raise ValueError("clean artifact ID order mismatch")
        for field in ("prediction_mask", "probability_map"):
            relative = _safe_relative(record.get(field), label=f"clean {field}")
            path = root / relative
            _regular_file_nofollow(path, label=f"clean {field}")
            if sha256_file(path) != record.get(f"{field}_sha256"):
                raise ValueError(f"clean artifact {field} record hash mismatch")
    return dict(audit), metrics, records


def _validate_checkpoint_against_source_protocol(
    payload: Mapping[str, Any], *, axis: Any, dataset_contract: Mapping[str, Any]
) -> None:
    expected_metric = {"best_miou": "miou", "best_pd": "pd"}[_axis_role(axis)]
    if payload.get("dataset") != _axis_dataset(axis):
        raise ValueError("checkpoint dataset mismatch")
    if payload.get("selection_metric") != expected_metric:
        raise ValueError("checkpoint selection metric does not match checkpoint role")
    if payload.get("test_selected") is not True:
        raise ValueError("checkpoint must disclose test_selected=true")
    split = payload.get("split_manifest")
    if not isinstance(split, Mapping):
        raise TypeError("checkpoint split_manifest must be a mapping")
    for actual, expected, label in (
        (split.get("test_split_sha256"), dataset_contract["test_split_sha256"], "test split"),
        (int(split.get("test_count", -1)), int(dataset_contract["test_images"]), "test count"),
        (split.get("corpus_manifest_sha256"), dataset_contract["corpus_manifest_sha256"], "corpus manifest"),
        (split.get("overlap_count"), 0, "train/test overlap"),
    ):
        if actual != expected:
            raise ValueError(f"checkpoint {label} mismatch")
    metrics = payload.get("test_metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("checkpoint test_metrics must be a mapping")
    if int(metrics.get("images", -1)) != int(dataset_contract["test_images"]):
        raise ValueError("checkpoint test metric image count mismatch")
    if float(payload.get("selection_value")) != float(metrics[expected_metric]):
        raise ValueError("checkpoint selection value is inconsistent")


def _verify_cache(
    protocol: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
    dataset: str,
    cache_override: Path | None,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], tuple[tuple[str, int], ...], tuple[str, ...]]:
    conditions = v1._conditions(protocol)
    corruption = protocol["corruption"]
    if sha256_file(v1._project_path(corruption["severity_table"])) != corruption["severity_table_sha256"]:
        raise ValueError("frozen severity table SHA256 drifted")
    if sha256_file(v1._project_path(corruption["pilot_report"])) != corruption["pilot_report_sha256"]:
        raise ValueError("frozen Pilot report SHA256 drifted")
    cache_contract = protocol["materialized_cache"]
    if sha256_file(v1._project_path(cache_contract["generation_protocol_archive"])) != cache_contract["generation_protocol_sha256"]:
        raise ValueError("cache generation protocol archive SHA256 drifted")
    default_root = v1._project_path(cache_contract["root"]) / dataset
    cache_dir = (cache_override or default_root).expanduser().resolve()
    expected = dataset_contract.get("materialized_cache")
    if not isinstance(expected, Mapping):
        raise ValueError("source protocol has no frozen dataset cache")
    manifest, audit = verify_cache_artifact(
        cache_dir,
        expected_protocol_sha256=str(expected["generation_protocol_sha256"]),
        verify_file_hashes=True,
    )
    if (
        audit["manifest_sha256"] != expected["manifest_sha256"]
        or manifest["cache_content_sha256"] != expected["content_sha256"]
        or manifest["dataset"] != dataset
        or manifest["split_sha256"] != dataset_contract["test_split_sha256"]
        or manifest["ordered_ids_sha256"] != dataset_contract["ordered_test_ids_sha256"]
        or int(audit["verified_total_bytes"]) != int(expected["bytes"])
    ):
        raise ValueError("materialized cache lineage mismatch")
    actual_conditions = tuple(
        (record["corruption"], int(record["severity"]))
        for record in manifest["conditions"]
    )
    if actual_conditions != conditions:
        raise ValueError("materialized cache condition order mismatch")
    ids = tuple(str(value) for value in manifest["image_ids"])
    if len(ids) != int(dataset_contract["test_images"]):
        raise ValueError("materialized cache image count mismatch")
    if ordered_ids_sha256(ids) != dataset_contract["ordered_test_ids_sha256"]:
        raise ValueError("materialized cache ordered IDs mismatch")
    return cache_dir, manifest, audit, conditions, ids


def _file_ledger(root: Path, *, excluded: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    excluded_set = set(excluded)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in excluded_set:
            continue
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def _metric_core(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary": metrics.get("summary"),
        "official": metrics.get("official"),
        "unified": metrics.get("unified"),
        "evaluated_images": metrics.get("evaluated_images"),
        "probability_shard_sha256": metrics.get("probability_shard_sha256"),
        "probability_tensor_sequence_sha256": metrics.get("probability_tensor_sequence_sha256"),
        "prediction_mask_count": metrics.get("prediction_mask_count"),
    }


def _build_source_parity_audit(
    staging: Path,
    *,
    axis: Any,
    axis_config: Mapping[str, Any],
    reference_root: Path,
    conditions: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    if _axis_role(axis) != "best_miou":
        raise ValueError("a parity receipt can only be issued for best_miou")
    reference = _verify_v1_parity_reference(
        reference_root, _axis_dataset(axis), axis_config=axis_config
    )
    comparisons: list[dict[str, Any]] = []
    all_passed = True
    for corruption, severity in conditions:
        key = condition_key(corruption, severity)
        new_root = staging / "conditions" / key
        old_root = reference_root / "conditions" / key
        new_metrics = _load_json(new_root / "metrics.json")
        old_metrics = _load_json(old_root / "metrics.json")
        new_records = _load_jsonl(new_root / "per_image.jsonl")
        old_records = _load_jsonl(old_root / "per_image.jsonl")
        ids_exact = [record.get("image_id") for record in new_records] == [
            record.get("image_id") for record in old_records
        ]
        record_probability_hashes_exact = [
            record.get("probability_tensor_raw_sha256") for record in new_records
        ] == [record.get("probability_tensor_raw_sha256") for record in old_records]
        mask_hashes_exact = [record.get("prediction_mask_sha256") for record in new_records] == [
            record.get("prediction_mask_sha256") for record in old_records
        ]
        probability_shard_exact = sha256_file(new_root / "probabilities_256.npy") == sha256_file(
            old_root / "probabilities_256.npy"
        )
        metrics_exact = v1._canonical_json_equal(
            _metric_core(new_metrics), _metric_core(old_metrics)
        )
        passed = all(
            (ids_exact, record_probability_hashes_exact, mask_hashes_exact, probability_shard_exact, metrics_exact)
        )
        all_passed &= passed
        comparisons.append(
            {
                "condition_key": key,
                "ordered_image_ids_exact": ids_exact,
                "probability_tensor_hashes_exact": record_probability_hashes_exact,
                "probability_shard_file_hash_exact": probability_shard_exact,
                "binary_mask_file_hashes_exact": mask_hashes_exact,
                "official_unified_and_summary_metrics_exact": metrics_exact,
                "passed": passed,
            }
        )
    audit_record = {
        "schema_version": 2,
        "audit_type": SOURCE_PARITY_AUDIT_TYPE,
        "passed": bool(all_passed and len(comparisons) == 13),
        "checkpoint_role": "best_miou",
        "dataset": _axis_dataset(axis),
        "axis_config_sha256": _axis_config_sha256(axis),
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "checkpoint_sha256": _axis_checkpoint_sha256(axis),
        "development_only": True,
        "main_paper_table": False,
        "numeric_tolerance_used": False,
        "comparison_scope": {
            "condition_count": len(comparisons),
            "all_probability_shards": True,
            "all_probability_tensor_hashes": True,
            "all_binary_mask_file_hashes": True,
            "all_integer_and_float_metrics_exact": True,
            "all_per_image_ids_in_order": True,
            "v1_logit_artifact_available": False,
        },
        "frozen_v1_reference": {
            "root": str(reference_root.resolve()),
            "artifact_manifest_sha256": reference["manifest_sha256"],
            "complete_sha256": reference["complete_sha256"],
        },
        "conditions": comparisons,
    }
    if not audit_record["passed"]:
        raise RuntimeError("best_miou v2/v1 parity gate failed")
    return audit_record


def verify_best_miou_parity_receipt(
    path: Path,
    *,
    axis: Any,
) -> dict[str, Any]:
    """Verify the one globally frozen clean+Source+AdaBN parity receipt.

    An arbitrary caller-supplied receipt is never a trust anchor: the resolved
    path must be the repository's canonical global receipt.  The verifier also
    recursively re-verifies this dataset's referenced Source-v2 producer
    artifact and its two seals.
    """

    if _axis_role(axis) != "best_pd":
        raise ValueError("parity receipt consumption is only valid for best_pd")
    path = Path(os.path.abspath(path.expanduser()))
    expected_path = Path(os.path.abspath(GLOBAL_PARITY_RECEIPT))
    if path != expected_path:
        raise ValueError(f"parity receipt path must be the fixed global path: {expected_path}")
    _regular_file_nofollow(path, label="global parity receipt")
    config = checkpoint_axis_api.load_axis_config(
        Path(_axis_value(axis, "config_path"))
    )
    if config["_runtime"]["config_sha256"] != _axis_config_sha256(axis):
        raise ValueError("parity receipt verifier axis config changed")
    receipt = checkpoint_axis_api.verify_parity_receipt(
        config=config, receipt=path
    )
    for actual, expected, label in (
        (receipt.get("receipt_type"), GLOBAL_PARITY_RECEIPT_TYPE, "receipt type"),
        (receipt.get("passed"), True, "receipt status"),
        (receipt.get("checkpoint_role"), "best_miou", "receipt checkpoint role"),
        (receipt.get("numeric_tolerance_used"), False, "receipt numeric tolerance"),
    ):
        if actual != expected:
            raise ValueError(f"{label} mismatch")
    for section in ("clean", "source", "adabn"):
        value = receipt.get(section)
        if not isinstance(value, Mapping) or value.get("passed") is not True:
            raise ValueError(f"global parity receipt {section} section did not pass")
    source_section = receipt["source"]
    totals = source_section.get("totals")
    if not isinstance(totals, Mapping) or int(totals.get("conditions", -1)) != 39:
        raise ValueError("global parity receipt Source section does not cover 3x13 conditions")
    candidate_roots = receipt.get("candidate_roots")
    if not isinstance(candidate_roots, Mapping) or "source" not in candidate_roots:
        raise ValueError("global parity receipt lacks Source candidate root")
    producer_root = Path(str(candidate_roots["source"])) / _axis_dataset(axis)
    producer_audit = verify_source_artifact(producer_root)
    if producer_audit["manifest"].get("checkpoint_role") != "best_miou":
        raise ValueError("global parity Source producer is not best_miou")
    if producer_audit["manifest"].get("axis_config_sha256") != _axis_config_sha256(axis):
        raise ValueError("global parity Source producer axis config mismatch")
    datasets = source_section.get("datasets")
    if not isinstance(datasets, Mapping) or _axis_dataset(axis) not in datasets:
        raise ValueError("global parity receipt lacks this Source dataset")
    dataset_record = datasets[_axis_dataset(axis)]
    if not isinstance(dataset_record, Mapping):
        raise TypeError("global parity Source dataset record must be a mapping")
    seals = dataset_record.get("candidate_seals")
    if not isinstance(seals, Mapping):
        raise TypeError("global parity Source candidate seals must be a mapping")
    if seals.get("artifact_manifest.json") != producer_audit["manifest_sha256"]:
        raise ValueError("global parity Source producer manifest seal drifted")
    if seals.get("COMPLETE.json") != producer_audit["complete_sha256"]:
        raise ValueError("global parity Source producer completion seal drifted")
    if seals.get("payload_tree_sha256") != producer_audit["payload_tree"]["sha256"]:
        raise ValueError("global parity Source producer payload tree seal drifted")
    if int(seals.get("payload_file_count", -1)) != int(
        producer_audit["payload_tree"]["file_count"]
    ):
        raise ValueError("global parity Source producer payload file count drifted")
    return {"receipt": receipt, "source_artifact": producer_audit}


def _checkpoint_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return clean_v1._checkpoint_summary(payload)


def run_source_corruption_axis(
    *,
    axis_config: Mapping[str, Any],
    axis: Any,
    clean_artifact: Path,
    source_protocol_path: Path = DEFAULT_SOURCE_PROTOCOL,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    device_name: str = "cuda:0",
    parity_reference: Path | None = None,
    parity_receipt: Path | None = None,
) -> dict[str, Any]:
    """Run and atomically publish one dataset/checkpoint Source axis artifact."""

    started = time.perf_counter()
    dataset = _axis_dataset(axis)
    role = _axis_role(axis)
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if role not in {"best_miou", "best_pd"}:
        raise ValueError(f"unsupported checkpoint role: {role}")
    if _axis_development_only(axis) is not True:
        raise ValueError("checkpoint-axis v2 test artifacts must be development_only")
    source_protocol_path = source_protocol_path.expanduser().resolve()
    if sha256_file(source_protocol_path) != SOURCE_PROTOCOL_SHA256:
        raise ValueError("frozen Source corruption v1 protocol SHA256 drifted")
    protocol, dataset_contract = v1._load_protocol(source_protocol_path, dataset)
    cache_root, cache_manifest, cache_audit, conditions, image_ids = _verify_cache(
        protocol, dataset_contract, dataset, cache_dir
    )

    checkpoint_payload = checkpoint_axis_api.load_and_verify_checkpoint(axis)
    _validate_checkpoint_against_source_protocol(
        checkpoint_payload, axis=axis, dataset_contract=dataset_contract
    )
    checkpoint_path = _axis_checkpoint_path(axis)
    if sha256_file(checkpoint_path) != _axis_checkpoint_sha256(axis):
        raise ValueError("checkpoint SHA256 changed after axis verification")

    clean_artifact = Path(os.path.abspath(clean_artifact.expanduser()))
    clean_axis = checkpoint_axis_api.resolve_axis(
        axis_config,
        dataset=dataset,
        role=role,
        artifact_kind="clean",
        output_override=clean_artifact if role == "best_miou" else None,
    )
    if role == "best_pd" and clean_artifact != Path(
        os.path.abspath(_axis_value(clean_axis, "output_dir"))
    ):
        raise ValueError("best_pd clean input must use its canonical role-first artifact")
    clean_audit, clean_metrics, clean_records = _clean_reference(
        clean_artifact, axis=clean_axis, image_ids=image_ids
    )

    parity_gate: dict[str, Any]
    if role == "best_miou":
        if output_dir is None:
            raise ValueError("best_miou v2 is parity-only and requires an explicit output_dir")
        if parity_reference is None:
            raise ValueError("best_miou v2 requires a frozen v1 parity reference")
        if parity_receipt is not None:
            raise ValueError("best_miou produces, rather than consumes, a parity receipt")
        parity_gate = {"mode": "producer", "passed": None}
    else:
        if parity_reference is not None:
            raise ValueError("best_pd must not use a best_miou v1 inference reference")
        if output_dir is not None:
            raise ValueError("formal best_pd Source output must use the canonical role-first root")
        effective_receipt = parity_receipt or GLOBAL_PARITY_RECEIPT
        verified_receipt = verify_best_miou_parity_receipt(effective_receipt, axis=axis)
        receipt_sha256 = str(verified_receipt["receipt"]["_receipt_sha256"])
        axis_receipt_sha256 = _axis_value(axis, "parity_receipt_sha256")
        if axis_receipt_sha256 != receipt_sha256:
            raise ValueError("resolved axis/global parity receipt SHA256 mismatch")
        parity_gate = {
            "mode": "consumer",
            "passed": True,
            "receipt": str(Path(effective_receipt).expanduser().resolve()),
            "receipt_sha256": receipt_sha256,
            "producer_artifact_manifest_sha256": verified_receipt["source_artifact"]["manifest_sha256"],
        }

    final_root = checkpoint_axis_api.canonical_output_dir(
        axis_config,
        artifact_kind=ARTIFACT_KIND,
        role=role,
        dataset=dataset,
        output_override=output_dir,
    )
    final_root = Path(os.path.abspath(final_root.expanduser()))
    if final_root != Path(os.path.abspath(_axis_value(axis, "output_dir"))):
        raise ValueError("resolved axis output differs from Source canonical output")
    staging = checkpoint_axis_api.prepare_staging(axis)

    try:
        source_runner.seed_everything(int(protocol["corruption"]["seed"]))
        device = source_runner.resolve_device(device_name)
        model = source_runner.build_nsfpn_model()
        checkpoint_wrapper = checkpoint_axis_api.load_checkpoint_into_model(
            model, checkpoint_payload
        )
        if sha256_file(checkpoint_path) != _axis_checkpoint_sha256(axis):
            raise ValueError("checkpoint changed while loading model")
        model.to(device)
        adapter = source_runner.IRSTDModelAdapter(model, warm_flag=False)
        adapter.set_source_eval_mode()
        state_hash_before = source_runner.state_dict_sha256(model.state_dict())
        evaluation_protocol = v1._evaluation_protocol(protocol)
        cache_conditions = {record["key"]: record for record in cache_manifest["conditions"]}
        condition_summaries: list[dict[str, Any]] = []
        clean_parity: dict[str, Any] | None = None

        for condition_index, (corruption, severity) in enumerate(conditions):
            condition_started = time.perf_counter()
            key = condition_key(corruption, severity)
            cache_condition = cache_conditions[key]
            condition_dir = staging / "conditions" / key
            condition_dir.mkdir(parents=True)
            dataset_view = CachedCorruptionDataset(
                cache_root,
                corruption=corruption,
                severity=severity,
                manifest=cache_manifest,
            )
            loader = DataLoader(
                dataset_view,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            probability_path = condition_dir / "probabilities_256.npy"
            probability_partial, probability_map = v1._atomic_probability_memmap(
                probability_path, len(dataset_view)
            )
            official_evaluator = OfficialMetricAdapter(image_size=256)
            unified_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
            input_hasher = TensorSequenceHasher()
            target_hasher = TensorSequenceHasher()
            probability_hasher = TensorSequenceHasher()
            records: list[dict[str, Any]] = []
            repeat_exact = False
            clean_probabilities_exact = True
            clean_masks_exact = True
            clean_mask_files_exact = True

            for index, batch in enumerate(loader):
                metadata = source_runner.metadata_from_batch(batch)
                image_id = str(metadata["image_id"])
                if image_id != image_ids[index]:
                    raise RuntimeError("cached loader order drifted")
                image_cpu = batch["image"]
                target_cpu = batch["mask"]
                input_hasher.update(image_id, image_cpu[0])
                target_hasher.update(image_id, target_cpu[0])
                image = image_cpu.to(device, non_blocking=False)
                target_device = target_cpu.to(device, non_blocking=False)
                logits, repeated = source_runner.checked_source_forward(
                    adapter,
                    image,
                    target_device,
                    repeat_exact=index == 0,
                )
                if index == 0:
                    repeat_exact = bool(repeated)
                logits_cpu = logits.detach().cpu()
                source_runner._update_official_evaluator(
                    official_evaluator, logits_cpu, target_cpu
                )
                unified_evaluator.update_logits(logits_cpu, target_cpu)
                probability = probabilities_from_logits(logits_cpu)[0, 0].astype(
                    np.float32, copy=False
                )
                binary_mask = np.where(probability > 0.5, 255, 0).astype(np.uint8)
                probability_map[index] = probability
                probability_hasher.update(image_id, probability)
                mask_relative = Path("prediction_masks_256") / clean_v1._relative_prediction_path(
                    image_id, ".png"
                )
                mask_path = condition_dir / mask_relative
                clean_v1._write_png_atomic(mask_path, binary_mask)
                mask_file_hash = sha256_file(mask_path)

                if condition_index == 0:
                    reference = clean_records[index]
                    reference_probability = np.load(
                        clean_artifact / str(reference["probability_map"]), allow_pickle=False
                    )
                    reference_mask = np.asarray(
                        Image.open(clean_artifact / str(reference["prediction_mask"])).convert("L")
                    )
                    clean_probabilities_exact &= np.array_equal(probability, reference_probability)
                    clean_masks_exact &= np.array_equal(binary_mask, reference_mask)
                    clean_mask_files_exact &= mask_file_hash == reference["prediction_mask_sha256"]
                    if not all((clean_probabilities_exact, clean_masks_exact, clean_mask_files_exact)):
                        raise RuntimeError(f"clean v2 parity diverged at {image_id}")

                target = target_cpu.numpy()[0, 0]
                records.append(
                    {
                        "index": index,
                        **metadata,
                        "checkpoint_role": role,
                        "cache_content_sha256": cache_manifest["cache_content_sha256"],
                        "cache_condition_tensor_sha256": cache_condition["tensor_sequence_sha256"],
                        "probability_shard": "probabilities_256.npy",
                        "probability_shard_index": index,
                        "probability_tensor_raw_sha256": v1._raw_array_sha256(probability),
                        "probability_min": float(probability.min()),
                        "probability_max": float(probability.max()),
                        "prediction_mask": str(mask_relative),
                        "prediction_mask_sha256": mask_file_hash,
                        "pixel_metrics_at_probability_gt_0_5": clean_v1._pixel_record(probability, target),
                    }
                )

            probability_map.flush()
            del probability_map
            os.replace(probability_partial, probability_path)
            if len(records) != len(image_ids):
                raise RuntimeError("condition prediction count drifted")
            if input_hasher.hexdigest() != cache_condition["tensor_sequence_sha256"]:
                raise RuntimeError(f"consumer input tensor hash mismatch for {key}")
            if target_hasher.hexdigest() != cache_manifest["targets"]["tensor_sequence_sha256"]:
                raise RuntimeError(f"consumer target tensor hash mismatch for {key}")
            if source_runner.state_dict_sha256(model.state_dict()) != state_hash_before:
                raise RuntimeError(f"Source model state changed under {key}")
            official = official_evaluator.compute()
            unified = unified_evaluator.compute()
            summary = v1._condition_summary(official, unified)
            if condition_index == 0:
                official_exact = v1._canonical_json_equal(official.to_dict(), clean_metrics["official"])
                unified_exact = v1._canonical_json_equal(unified.to_dict(), clean_metrics["unified"])
                checkpoint_metrics = checkpoint_payload["test_metrics"]
                checkpoint_exact = (
                    summary["legacy_mean_iou"] == float(checkpoint_metrics["miou"])
                    and summary["legacy_pd"] == float(checkpoint_metrics["pd"])
                    and summary["legacy_fa_per_million_pixels"]
                    == float(checkpoint_metrics["fa_per_pixel_x1e6"])
                )
                clean_parity = {
                    "comparison_rule": "bit_exact_arrays_and_exact_json_metrics",
                    "numeric_tolerance_used": False,
                    "probability_arrays_compared": len(records),
                    "probability_values_compared": len(records) * 256 * 256,
                    "all_probability_arrays_exact": clean_probabilities_exact,
                    "all_binary_mask_arrays_exact": clean_masks_exact,
                    "all_binary_mask_file_hashes_exact": clean_mask_files_exact,
                    "official_metrics_exact": official_exact,
                    "unified_metrics_exact": unified_exact,
                    "checkpoint_operating_point_exact": checkpoint_exact,
                    "passed": all(
                        (
                            clean_probabilities_exact,
                            clean_masks_exact,
                            clean_mask_files_exact,
                            official_exact,
                            unified_exact,
                            checkpoint_exact,
                        )
                    ),
                }
                if not clean_parity["passed"]:
                    raise RuntimeError(f"clean checkpoint-axis parity failed: {clean_parity}")

            probability_file_hash = sha256_file(probability_path)
            for record in records:
                record["probability_shard_sha256"] = probability_file_hash
            per_image_path = condition_dir / "per_image.jsonl"
            source_runner.write_jsonl_atomic(per_image_path, records)
            metrics = {
                "schema_version": 2,
                "method": METHOD,
                "artifact_kind": ARTIFACT_KIND,
                "checkpoint_role": role,
                "development_only": True,
                "main_paper_table": False,
                "extra_best_pd_tuning_episodes": 0,
                "dataset": dataset,
                "condition_index": condition_index,
                "condition_key": key,
                "corruption": corruption,
                "severity": severity,
                "evaluated_images": len(records),
                "cache_lineage": {
                    "cache_dir": str(cache_root),
                    "cache_manifest_sha256": cache_audit["manifest_sha256"],
                    "cache_content_sha256": cache_manifest["cache_content_sha256"],
                    "condition_file_sha256": cache_condition["file_sha256"],
                    "condition_tensor_sequence_sha256": cache_condition["tensor_sequence_sha256"],
                    "target_tensor_sequence_sha256": cache_manifest["targets"]["tensor_sequence_sha256"],
                    "ordered_ids_sha256": cache_manifest["ordered_ids_sha256"],
                },
                "summary": summary,
                "official": official.to_dict(),
                "unified": unified.to_dict(),
                "probability_shard": "probabilities_256.npy",
                "probability_shard_sha256": probability_file_hash,
                "probability_tensor_sequence_sha256": probability_hasher.hexdigest(),
                "prediction_mask_count": len(records),
                "checks": {
                    "repeat_logit_exact_first_image": repeat_exact,
                    "input_hash_matches_cache": True,
                    "target_hash_matches_cache": True,
                    "ordered_ids_match_cache": True,
                    "model_state_unchanged": True,
                    "clean_parity": clean_parity if condition_index == 0 else None,
                },
                "runtime_seconds": time.perf_counter() - condition_started,
            }
            metrics_path = condition_dir / "metrics.json"
            source_runner.write_json_atomic(metrics_path, metrics)
            condition_summaries.append(
                {
                    "condition_index": condition_index,
                    "condition_key": key,
                    "corruption": corruption,
                    "severity": severity,
                    **summary,
                    "files": {
                        "metrics": str(metrics_path.relative_to(staging)),
                        "per_image": str(per_image_path.relative_to(staging)),
                        "probability_shard": str(probability_path.relative_to(staging)),
                        "prediction_masks": str((condition_dir / "prediction_masks_256").relative_to(staging)),
                    },
                }
            )

        state_hash_after = source_runner.state_dict_sha256(model.state_dict())
        if state_hash_after != state_hash_before:
            raise RuntimeError("Source model state changed during benchmark")
        if clean_parity is None or not clean_parity["passed"]:
            raise RuntimeError("clean artifact parity was not established")

        benchmark = {
            "schema_version": 2,
            "method": METHOD,
            "artifact_kind": ARTIFACT_KIND,
            "protocol_id": "cr-sitta-source-corruption-checkpoint-axis-v2",
            "source_protocol_path": str(source_protocol_path),
            "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "axis_config_sha256": _axis_config_sha256(axis),
            "checkpoint_role": role,
            "development_only": True,
            "main_paper_table": False,
            "extra_best_pd_tuning_episodes": 0,
            "checkpoint_selection_or_tuning_performed": False,
            "dataset": dataset,
            "condition_count": len(condition_summaries),
            "evaluated_images_per_condition": len(image_ids),
            "sample_condition_count": len(image_ids) * len(condition_summaries),
            "ordered_ids_sha256": ordered_ids_sha256(image_ids),
            "conditions": condition_summaries,
            "cache_audit": cache_audit,
            "cache_content_sha256": cache_manifest["cache_content_sha256"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _axis_checkpoint_sha256(axis),
            "checkpoint_wrapper": checkpoint_wrapper,
            "checkpoint_metadata": _checkpoint_summary(checkpoint_payload),
            "checkpoint_test_selected_disclosure": (
                "This development artifact uses a checkpoint selected by repeated fixed-test evaluation; "
                "it is not eligible for an untouched-test main paper table."
            ),
            "clean_artifact": {
                "root": str(clean_artifact),
                "artifact_manifest_sha256": clean_audit["manifest_sha256"],
                "complete_sha256": clean_audit["complete_sha256"],
            },
            "clean_parity": clean_parity,
            "best_miou_parity_gate": parity_gate,
            "checks": {
                "full_fixed_test_split_all_conditions": True,
                "exact_13_condition_contract": True,
                "corruptions_consumed_only_from_materialized_cache": True,
                "all_prediction_masks_saved": True,
                "all_probability_shards_saved": True,
                "all_per_image_records_saved": True,
                "model_loaded_once": True,
                "model_state_sha256_before": state_hash_before,
                "model_state_sha256_after": state_hash_after,
                "model_state_unchanged": True,
                "zero_additional_checkpoint_or_tta_tuning": True,
            },
            "repository_provenance": source_runner.repository_provenance(
                (
                    "benchmark/checkpoint_axis.py",
                    "benchmark/source_corruption_axis_runner_v2.py",
                    "run_source_corruption_checkpoint_axis_v2.py",
                    "run_source_corruption_benchmark.py",
                    "configs/checkpoint_axis_best_pd_v1.yaml",
                    "configs/source_corruption_benchmark_fixed_splits.yaml",
                    "dataio/corruption_cache.py",
                    "metrics/irstd_metrics.py",
                    "metrics/official_metric_adapter.py",
                    "model/MSHNet_NSFPN.py",
                    "test_fixed_split_source.py",
                    "test_source.py",
                )
            ),
            "runtime_seconds": time.perf_counter() - started,
        }
        source_runner.write_json_atomic(staging / "benchmark.json", benchmark)

        parity_output: dict[str, Any] | None = None
        if role == "best_miou":
            assert parity_reference is not None
            parity_output = _build_source_parity_audit(
                staging,
                axis=axis,
                axis_config=axis_config,
                reference_root=parity_reference.expanduser().resolve(),
                conditions=conditions,
            )
            source_runner.write_json_atomic(staging / SOURCE_PARITY_AUDIT_NAME, parity_output)
            parity_gate = {
                "mode": "producer",
                "passed": True,
                "authorization_receipt": False,
                "audit": SOURCE_PARITY_AUDIT_NAME,
                "frozen_v1_reference": str(parity_reference.expanduser().resolve()),
            }
            benchmark["best_miou_parity_gate"] = dict(parity_gate)
            source_runner.write_json_atomic(staging / "benchmark.json", benchmark)

        source_runner.write_json_atomic(
            staging / "run_config.json",
            {
                "schema_version": 2,
                "axis_config_sha256": _axis_config_sha256(axis),
                "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
                "dataset": dataset,
                "checkpoint_role": role,
                "development_only": True,
                "main_paper_table": False,
                "extra_best_pd_tuning_episodes": 0,
                "threshold": {"transform": "sigmoid", "rule": "strict_greater_than", "value": 0.5},
                "condition_count": 13,
            },
        )
        files = _file_ledger(staging, excluded=("artifact_manifest.json", "COMPLETE.json"))
        condition_files = {
            record["condition_key"]: {
                name: {
                    "path": relative,
                    "sha256": files[relative]["sha256"] if relative in files else None,
                }
                for name, relative in record["files"].items()
                if name != "prediction_masks"
            }
            for record in condition_summaries
        }
        manifest_extra = {
            "method": METHOD,
            "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "development_only": True,
            "main_paper_table": False,
            "extra_best_pd_tuning_episodes": 0,
            "condition_count": 13,
            "condition_files": condition_files,
            "files": files,
            "best_miou_parity_gate": parity_gate,
        }
        complete_extra = {
            "method": METHOD,
            "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "development_only": True,
            "main_paper_table": False,
            "extra_best_pd_tuning_episodes": 0,
            "condition_count": 13,
            "benchmark_sha256": files["benchmark.json"]["sha256"],
            "clean_artifact_manifest_sha256": clean_audit["manifest_sha256"],
            "parity_receipt_sha256": parity_gate.get("receipt_sha256"),
            "best_miou_parity_passed": (
                True if role == "best_miou" else parity_gate["passed"]
            ),
        }
        required_payloads = [
            "benchmark.json",
            "run_config.json",
            *(str(staging_path.relative_to(staging)) for staging_path in sorted(staging.rglob("metrics.json"))),
            *(str(staging_path.relative_to(staging)) for staging_path in sorted(staging.rglob("per_image.jsonl"))),
            *(str(staging_path.relative_to(staging)) for staging_path in sorted(staging.rglob("probabilities_256.npy"))),
        ]

        def prepublish_guard() -> None:
            checkpoint_axis_api.verify_published_artifact(
                clean_artifact,
                expected_axis=clean_axis,
                required_payloads=("metrics.json", "per_image.jsonl"),
            )
            if role == "best_pd":
                verify_best_miou_parity_receipt(
                    Path(parity_gate["receipt"]), axis=axis
                )
            else:
                assert parity_reference is not None
                _verify_v1_parity_reference(
                    parity_reference,
                    dataset,
                    axis_config=axis_config,
                )

        checkpoint_axis_api.finalize_and_publish(
            staging=staging,
            final=final_root,
            axis=axis,
            required_payloads=required_payloads,
            manifest_extra=manifest_extra,
            complete_extra=complete_extra,
            prepublish_guard=prepublish_guard,
        )
    except BaseException:
        checkpoint_axis_api.remove_private_staging(staging, final=final_root)
        raise

    audit = verify_source_artifact(final_root, expected_axis=axis)
    return {
        "published_output_dir": str(final_root),
        "dataset": dataset,
        "checkpoint_role": role,
        "condition_count": 13,
        "evaluated_images_per_condition": len(image_ids),
        "artifact_manifest_sha256": audit["manifest_sha256"],
        "complete_sha256": audit["complete_sha256"],
    }


__all__ = [
    "ARTIFACT_KIND",
    "DATASETS",
    "DEFAULT_SOURCE_PROTOCOL",
    "GLOBAL_PARITY_RECEIPT",
    "GLOBAL_PARITY_RECEIPT_TYPE",
    "SOURCE_PARITY_AUDIT_NAME",
    "SOURCE_PARITY_AUDIT_TYPE",
    "SOURCE_PROTOCOL_SHA256",
    "run_source_corruption_axis",
    "verify_best_miou_parity_receipt",
    "verify_source_artifact",
]
