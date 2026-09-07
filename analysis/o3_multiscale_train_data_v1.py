"""Explicit source-supervised train8 masks, separate from sealed TTA labels.

Only the first eight ordered NUDT-SIRST B4 Pilot64 IDs are admitted.  Their
original train PNGs are checked against the immutable cache's source-file
hashes and transformed exactly as that cache materializer did.  This module
does not open ``outer_evaluator/targets.npy`` or call its outer-only loader.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

import materialize_binary_tent_ss_calibration_cache_v2 as cache_api
from scripts import run_p3_stage_b4_full_pilot64_v1 as b4

DATASET = "NUDT-SIRST"
TRAIN_COUNT = 663
SUPERVISED_COUNT = 8
DATASET_ROOT = "datasets/NUDT-SIRST"
TRAIN_SPLIT = "datasets/NUDT-SIRST/img_idx/train_NUDT-SIRST.txt"


class SourceSupervisedDataError(ValueError):
    """The independent source-supervised train8 boundary was not satisfied."""


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise SourceSupervisedDataError(f"{label} differs")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _verified_context(parent_contract: b4.FullPilotContract, image_ids: Sequence[str]) -> dict[str, Any]:
    if not isinstance(parent_contract, b4.FullPilotContract):
        raise SourceSupervisedDataError("canonical frozen B4 parent contract required")
    canonical = b4.load_contract(parent_contract.config_path)
    for key in ("repository", "config_path", "config_sha256", "raw"):
        _same(getattr(parent_contract, key), getattr(canonical, key), f"parent.{key}")
    if (not isinstance(image_ids, (tuple, list)) or len(image_ids) != SUPERVISED_COUNT
            or any(not isinstance(identifier, str) for identifier in image_ids)):
        raise SourceSupervisedDataError("exactly eight ordered string IDs required")
    teacher = b4._teacher_manifest(canonical, DATASET)
    pilot_ids = tuple(teacher["image_ids"])
    _same(len(pilot_ids), 64, "original Pilot64 count")
    _same(tuple(image_ids), pilot_ids[:SUPERVISED_COUNT], "original Pilot64 first8 IDs")
    if any(Path(identifier).name != identifier or identifier in (".", "..") for identifier in image_ids):
        raise SourceSupervisedDataError("train8 IDs must be plain canonical names")

    spec = canonical.raw["datasets"][DATASET]
    binding = canonical.raw["frozen_parent_bindings"]["cache_protocol"]
    protocol_bytes, protocol_sha = cache_api.secure_io._read_stable_bytes(
        canonical.repository / binding["path"], label="bound cache protocol")
    _same(protocol_sha, binding["sha256"], "cache protocol SHA256")
    protocol = yaml.safe_load(protocol_bytes)
    dataset_spec = protocol["datasets"][DATASET]
    _same(dataset_spec["root"], DATASET_ROOT, "official dataset root")
    _same(dataset_spec["train_split"], TRAIN_SPLIT, "official train split path")
    _same(dataset_spec["train_split_sha256"], spec["train_split_sha256"], "train split lineage")
    _same(protocol["preprocessing"]["mask_resize"],
          {"size": [256, 256], "interpolation": "nearest"}, "mask resize")
    _same(protocol["preprocessing"]["mask_corrupted"], False, "unchanged GT")
    split_bytes, split_sha = cache_api.secure_io._read_stable_bytes(
        canonical.repository / TRAIN_SPLIT, label="official train ID text")
    _same(split_sha, spec["train_split_sha256"], "actual train split SHA256")
    train_ids = split_bytes.decode("utf-8").splitlines()
    if len(train_ids) != TRAIN_COUNT or len(set(train_ids)) != TRAIN_COUNT or "" in train_ids:
        raise SourceSupervisedDataError("official NUDT train count/uniqueness differs")
    if not set(pilot_ids).issubset(set(train_ids)):
        raise SourceSupervisedDataError("frozen Pilot64 is not a subset of official train IDs")

    cache_root = canonical.repository / spec["cache_root"]
    _, manifest, method = cache_api._consumer_metadata(
        cache_root, expected_protocol_sha256=protocol_sha)
    _same(manifest["dataset"], DATASET, "cache dataset")
    _same(tuple(manifest["image_ids"]), pilot_ids, "cache/teacher ordered IDs")
    _same(tuple(method["image_ids"]), pilot_ids, "method/cache ordered IDs")
    _same(manifest["train_split"], TRAIN_SPLIT, "cache train split")
    _same(manifest["train_split_sha256"], split_sha, "cache train split SHA256")
    # Hash all 13 image caches and 26 Source/uncertainty arrays; do not touch the
    # sealed outer target payload, even for a byte-hash audit.
    b4._verify_consumed_payloads(canonical, DATASET, include_outer_target=False)
    return {"parent": canonical, "spec": spec, "cache_manifest": manifest,
            "cache_root": cache_root, "train_split_sha256": split_sha,
            "cache_protocol_sha256": protocol_sha, "image_ids": tuple(image_ids)}


def load_train_targets(
    parent_contract: b4.FullPilotContract, image_ids: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read source-supervised train8 PNG labels, returning float32 [8,1,256,256].

    The caller must already have frozen its *new source-training* protocol.
    There is intentionally no ``episodes_complete`` argument: these are source
    training labels, not labels exposed to the original O3 adaptation episode.
    """
    context = _verified_context(parent_contract, image_ids)
    canonical, manifest = context["parent"], context["cache_manifest"]
    payloads: list[tuple[str, Path, bytes, str]] = []
    for identifier in context["image_ids"]:
        path = canonical.repository / DATASET_ROOT / "masks" / f"{identifier}.png"
        payload, actual_sha = cache_api.secure_io._read_stable_bytes(
            path, label=f"source-supervised train mask {identifier}")
        expected_sha = manifest["source_files"][identifier]["mask_sha256"]
        _same(actual_sha, expected_sha, f"source train mask SHA256 {identifier}")
        payloads.append((identifier, path, payload, actual_sha))

    # Exact original materializer recipe, without reading any sealed GT cache.
    from PIL import Image

    targets: list[np.ndarray] = []
    mask_records: list[dict[str, Any]] = []
    for identifier, path, payload, actual_sha in payloads:
        with Image.open(io.BytesIO(payload)) as handle:
            if handle.format != "PNG":
                raise SourceSupervisedDataError("source mask must be a PNG")
            native_size = list(handle.size)
            mask_01 = np.asarray(
                handle.convert("L").resize((256, 256), resample=Image.Resampling.NEAREST),
                dtype=np.float32,
            ) / 255.0
        target = np.ascontiguousarray(mask_01[None], dtype="<f4")
        if target.shape != (1, 256, 256) or not np.isfinite(target).all():
            raise SourceSupervisedDataError("transformed source target shape/value differs")
        if np.any(target < 0.0) or np.any(target > 1.0):
            raise SourceSupervisedDataError("source target range differs")
        targets.append(target)
        mask_records.append({"image_id": identifier,
            "path": str(path.relative_to(canonical.repository)),
            "file_sha256": actual_sha, "file_bytes": len(payload),
            "native_width_height": native_size,
            "tensor_sha256": hashlib.sha256(target.tobytes()).hexdigest()})
    result = np.stack(targets, axis=0)
    result.flags.writeable = False
    receipt = {
        "schema_version": 1, "dataset": DATASET, "split_name": "train",
        "role": "source_supervised_residual_training_train8",
        "sampling": "first8_in_original_frozen_B4_Pilot64_order",
        "image_ids": list(context["image_ids"]),
        "ordered_image_ids_sha256": _json_hash(list(context["image_ids"])),
        "parent_config_sha256": canonical.config_sha256,
        "checkpoint_sha256": context["spec"]["checkpoint_sha256"],
        "train_split_sha256": context["train_split_sha256"],
        "cache_protocol_sha256": context["cache_protocol_sha256"],
        "cache_manifest_sha256": context["spec"]["cache_manifest_sha256"],
        "masks": mask_records, "target_shape": list(result.shape),
        "target_dtype": str(result.dtype),
        "target_tensor_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "preprocessing": "PNG convert L; nearest resize256; float32 /255; contiguous CHW",
        "preprocessing_reuses_original_materializer_definition": True,
        "cached_target_array_equality_not_claimed": True,
        "train_mask_payload_snapshot_calls": 8, "train_mask_png_decodes": 8,
        "unique_train_mask_files": 8, "train_image_decodes": 0,
        "other_pilot_mask_decodes": 0, "sealed_outer_target_payload_opens": 0,
        "outer_target_loader_calls": 0, "test_split_reads": 0,
        "test_image_decodes": 0, "test_mask_decodes": 0,
        "validation_split_reads": 0, "validation_image_decodes": 0,
        "validation_mask_decodes": 0, "no_validation_split": True,
        "supervised_labels_enter_original_o3_update": False,
        "paper_result": False, "formal_test": False,
        "access_count_definition": "snapshot calls and PNG decodes; low-level repeated hash/inode verification reads are not additional decoded samples",
    }
    return result, receipt


__all__ = ["load_train_targets", "SourceSupervisedDataError", "DATASET", "SUPERVISED_COUNT"]
