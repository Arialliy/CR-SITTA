#!/usr/bin/env python3
"""Build the independent train-side TENT-SS calibration cache v2.

``--validate-only`` is metadata-only and deliberately imports no NumPy, PIL,
Torch, model, or CUDA module.  Pixel dependencies are loaded only inside the
formal materialization function.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
from typing import Any

import yaml


PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
DATAIO_ROOT = PROJECT_ROOT / "dataio"
if str(DATAIO_ROOT) not in sys.path:
    sys.path.insert(0, str(DATAIO_ROOT))

import train_side_pilot_protocol as secure_io  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/binary_tent_ss_calibration_cache_v2.yaml"
DATASET_NAMES = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
PROTOCOL_ID = "cr-sitta-binary-tent-ss-calibration-cache-v2"
CACHE_FORMAT = "nsfpn-binary-tent-ss-calibration-cache-v2"
PILOT_PROTOCOL_SHA256 = "3a35f2107786815ef8e7b053e1c220406f9bf2d095f59bccdc5822e16be9ab96"
PILOT_MANIFEST_SHA256 = "99b1ee706fc57813b473039513566566ba2e6a065663be122c260edf9ef6dfb1"
FROZEN_SEVERITY_SHA256 = "c3218738fd32f2cfa2a24731456d9dc18fa648eb7b0b6e1fbd5175c605ba0536"
CONDITIONS = (
    ("clean", 0),
    ("gaussian_noise", 1),
    ("gaussian_noise", 3),
    ("gaussian_noise", 5),
    ("gaussian_blur", 1),
    ("gaussian_blur", 3),
    ("gaussian_blur", 5),
    ("low_contrast", 1),
    ("low_contrast", 3),
    ("low_contrast", 5),
    ("stripe_noise", 1),
    ("stripe_noise", 3),
    ("stripe_noise", 5),
)
METHOD_FIELDS = (
    "image",
    "image_id",
    "original_size",
    "dataset",
    "corruption",
    "severity",
    "seed",
)
FORBIDDEN_METHOD_FIELDS = frozenset(
    {"mask", "target", "label", "gt", "ground_truth"}
)
PILOT_SCOPE = "source_train_side_method_calibration"
PILOT_OUTPUT_ROOT = "configs/tta_train_side_pilot_v2"
SHA256_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _boolean(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    _equal(value, expected, label)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _safe_path(repository: Path, raw: Any, label: str) -> Path:
    return secure_io._safe_repository_path(repository, raw, label)


def _snapshot(path: Path, expected: Any, label: str) -> tuple[bytes, str]:
    return secure_io._verified_snapshot(path, expected, label)


def _json(payload: bytes, label: str) -> dict[str, Any]:
    return secure_io._parse_json(payload, label)


def _yaml(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} is not valid UTF-8 YAML") from error
    return dict(_mapping(value, label))


def _condition_key(corruption: str, severity: int) -> str:
    return f"{corruption}_S{severity}"


def _parse_conditions(protocol: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    section = _mapping(protocol.get("input_protocol"), "input_protocol")
    _equal(section.get("subset_size_per_dataset"), 64, "subset size")
    _equal(section.get("seed"), 42, "seed")
    raw = section.get("ordered_conditions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("ordered_conditions must be a sequence")
    conditions = tuple((str(item[0]), int(item[1])) for item in raw)
    _equal(conditions, CONDITIONS, "13-condition order")
    return conditions


def load_protocol(
    path: Path, *, repository: Path = PROJECT_ROOT
) -> tuple[dict[str, Any], str]:
    repository = secure_io._absolute_lexical_path(repository)
    actual_path = secure_io._absolute_lexical_path(path)
    payload, digest = secure_io._read_stable_bytes(path, label="cache protocol")
    protocol = _yaml(payload, "cache protocol")
    _equal(protocol.get("schema_version"), 2, "schema_version")
    _equal(protocol.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    _equal(
        protocol.get("protocol_path"),
        "configs/binary_tent_ss_calibration_cache_v2.yaml",
        "protocol_path",
    )
    declared_path = _safe_path(
        repository, protocol.get("protocol_path"), "protocol_path"
    )
    _equal(actual_path, declared_path, "cache protocol actual/declared path")
    _equal(protocol.get("scope"), "source_train_side_method_calibration_cache", "scope")
    for key, expected in (
        ("paper_result", False),
        ("no_validation_split", True),
        ("adaptation_interface_invoked", False),
        ("method_receives_labels", False),
    ):
        _boolean(protocol.get(key), expected, key)
    _equal(protocol.get("test_split_use"), "id_only_leakage_guard", "test_split_use")
    _parse_conditions(protocol)

    preprocessing = _mapping(protocol.get("preprocessing"), "preprocessing")
    expected_preprocessing = {
        "image_color_space": "RGB",
        "image_resize": {"size": [256, 256], "interpolation": "bilinear"},
        "mask_resize": {"size": [256, 256], "interpolation": "nearest"},
        "image_range_before_corruption": [0.0, 1.0],
        "normalization": {
            "name": "ImageNet",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "order": "after_corruption",
        },
        "mask_corrupted": False,
    }
    _equal(dict(preprocessing), expected_preprocessing, "preprocessing")
    cache = _mapping(protocol.get("cache"), "cache")
    for key, expected in (
        ("root", "results/binary_tent/ss_calibration_cache_v2"),
        ("format", CACHE_FORMAT),
        ("image_dtype", "little_endian_float32"),
        ("image_shape", [64, 3, 256, 256]),
        ("targets_path", "outer_evaluator/targets.npy"),
        ("targets_role", "outer_evaluator_only"),
        ("method_facing_targets", "forbidden"),
        ("method_facing_fields", list(METHOD_FIELDS)),
    ):
        _equal(cache.get(key), expected, f"cache.{key}")
    for key, expected in (
        ("atomic_no_replace", True),
        ("overwrite", False),
        ("stale_staging_recovery_requires_explicit_flag", True),
    ):
        _boolean(cache.get(key), expected, f"cache.{key}")
    datasets = _mapping(protocol.get("datasets"), "datasets")
    _equal(tuple(datasets), DATASET_NAMES, "dataset order")
    runtime_seal = _mapping(
        _mapping(protocol.get("lineage"), "lineage").get("runtime_metadata_seal"),
        "lineage.runtime_metadata_seal",
    )
    _equal(
        tuple(runtime_seal),
        ("materializer", "train_side_pilot_protocol_implementation"),
        "runtime metadata seal order",
    )
    expected_runtime_paths = {
        "materializer": "materialize_binary_tent_ss_calibration_cache_v2.py",
        "train_side_pilot_protocol_implementation": (
            "dataio/train_side_pilot_protocol.py"
        ),
    }
    for name, expected_path in expected_runtime_paths.items():
        item = _mapping(runtime_seal.get(name), f"runtime_metadata_seal.{name}")
        _equal(item.get("path"), expected_path, f"runtime_metadata_seal.{name}.path")
        _sha256(item.get("sha256"), f"runtime_metadata_seal.{name}.sha256")
    return protocol, digest


def _validate_pilot_manifest(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    """Fail closed on the independent train-side Pilot's semantic boundary."""

    _equal(manifest.get("schema_version"), 2, "Pilot manifest schema")
    _equal(
        manifest.get("protocol_id"),
        "cr-sitta-tta-train-side-calibration-pilot-v2",
        "Pilot manifest protocol_id",
    )
    _equal(manifest.get("scope"), PILOT_SCOPE, "Pilot manifest scope")
    _boolean(
        manifest.get("no_validation_split"), True, "Pilot no_validation_split"
    )
    _boolean(manifest.get("paper_result"), False, "Pilot paper_result")
    pilot_protocol = _mapping(manifest.get("protocol"), "Pilot manifest protocol")
    _equal(
        pilot_protocol.get("path"),
        "configs/tta_train_side_calibration_pilot_v2.yaml",
        "Pilot manifest protocol path",
    )
    _equal(
        pilot_protocol.get("sha256"),
        PILOT_PROTOCOL_SHA256,
        "Pilot manifest protocol SHA256",
    )
    selection = _mapping(manifest.get("selection"), "Pilot manifest selection")
    _equal(selection.get("subset_size_per_dataset"), 64, "Pilot subset size")
    _equal(
        selection.get("parent_exclusion_role"),
        "round_02_corruption_severity_pilot_64",
        "Pilot parent exclusion role",
    )
    boundary = _mapping(
        manifest.get("metadata_io_boundary"), "Pilot metadata I/O boundary"
    )
    for key, expected in (
        ("images_opened", 0),
        ("masks_opened", 0),
        ("test_pixels_opened", 0),
        ("read_id_text_only", True),
        ("read_parent_pilot_metadata_only", True),
    ):
        _equal(boundary.get(key), expected, f"Pilot metadata boundary.{key}")

    protocol_datasets = _mapping(protocol.get("datasets"), "protocol datasets")
    manifest_datasets = _mapping(
        manifest.get("datasets"), "Pilot manifest datasets"
    )
    _equal(tuple(manifest_datasets), DATASET_NAMES, "Pilot manifest dataset order")
    zero_overlap_keys = (
        "train_test_overlap_count",
        "parent_pilot_output_overlap_count",
        "parent_pilot_test_overlap_count",
        "output_test_overlap_count",
    )
    true_keys = (
        "train_ids_unique",
        "test_ids_unique",
        "parent_pilot_ids_unique",
        "parent_pilot_ids_contained_in_train",
        "output_ids_unique",
        "output_ids_contained_in_train",
    )
    for dataset_name in DATASET_NAMES:
        configured = _mapping(
            protocol_datasets.get(dataset_name), f"protocol {dataset_name}"
        )
        observed = _mapping(
            manifest_datasets.get(dataset_name), f"Pilot manifest {dataset_name}"
        )
        output = _mapping(observed.get("output"), f"Pilot {dataset_name} output")
        expected_output_path = str(configured.get("calibration_ids"))
        _equal(output.get("path"), expected_output_path, f"{dataset_name} Pilot output path")
        output_path = Path(expected_output_path)
        _equal(
            output_path.parent.as_posix(),
            PILOT_OUTPUT_ROOT,
            f"{dataset_name} Pilot output parent",
        )
        _equal(output.get("count"), 64, f"{dataset_name} Pilot output count")
        _equal(
            output.get("file_sha256"),
            configured.get("calibration_ids_file_sha256"),
            f"{dataset_name} Pilot output file SHA256",
        )
        _equal(
            output.get("ordered_ids_sha256"),
            configured.get("calibration_ordered_ids_sha256"),
            f"{dataset_name} Pilot output ordered IDs",
        )
        test_split = _mapping(
            observed.get("test_split"), f"Pilot {dataset_name} test split"
        )
        _equal(
            test_split.get("path"),
            configured.get("test_split"),
            f"{dataset_name} Pilot test path",
        )
        _equal(
            test_split.get("sha256"),
            configured.get("test_split_sha256"),
            f"{dataset_name} Pilot test SHA256",
        )
        _boolean(
            test_split.get("metadata_only"),
            True,
            f"{dataset_name} Pilot test metadata-only",
        )
        checks = _mapping(observed.get("checks"), f"Pilot {dataset_name} checks")
        for key in zero_overlap_keys:
            value = checks.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{dataset_name} Pilot {key} must be an integer")
            _equal(value, 0, f"{dataset_name} Pilot {key}")
        for key in true_keys:
            _boolean(checks.get(key), True, f"{dataset_name} Pilot {key}")


def _verify_lineage(
    repository: Path, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    lineage = _mapping(protocol.get("lineage"), "lineage")
    verified: dict[str, str] = {}
    parsed_manifest: dict[str, Any] | None = None
    for name, expected_hash in (
        ("train_side_pilot_protocol", PILOT_PROTOCOL_SHA256),
        ("train_side_pilot_manifest", PILOT_MANIFEST_SHA256),
        ("frozen_severity_table", FROZEN_SEVERITY_SHA256),
    ):
        item = _mapping(lineage.get(name), f"lineage.{name}")
        _equal(item.get("sha256"), expected_hash, f"lineage.{name}.sha256")
        path = _safe_path(repository, item.get("path"), f"lineage.{name}.path")
        payload, digest = _snapshot(path, expected_hash, name)
        verified[str(item["path"])] = digest
        if name == "train_side_pilot_manifest":
            parsed_manifest = _json(payload, "train-side Pilot manifest")
        elif name == "frozen_severity_table":
            severity = _yaml(payload, "frozen severity table")
            _boolean(severity.get("frozen"), True, "severity frozen")
            calibration = _mapping(severity.get("calibration"), "severity calibration")
            _boolean(calibration.get("completed"), True, "severity calibration completed")
    pipeline = _mapping(lineage.get("pixel_pipeline"), "lineage.pixel_pipeline")
    for name in (
        "corruption_protocol",
        "corruption_implementation",
        "research_dataset_reference",
    ):
        item = _mapping(pipeline.get(name), f"pixel_pipeline.{name}")
        path = _safe_path(repository, item.get("path"), f"pixel_pipeline.{name}.path")
        _, digest = _snapshot(path, item.get("sha256"), f"pixel pipeline {name}")
        verified[str(item["path"])] = digest
    runtime_seal = _mapping(
        lineage.get("runtime_metadata_seal"), "lineage.runtime_metadata_seal"
    )
    for name in (
        "materializer",
        "train_side_pilot_protocol_implementation",
    ):
        item = _mapping(runtime_seal.get(name), f"runtime metadata seal {name}")
        path = _safe_path(repository, item.get("path"), f"runtime metadata seal {name}")
        _, digest = _snapshot(
            path, item.get("sha256"), f"runtime metadata seal {name}"
        )
        verified[str(item["path"])] = digest
    if parsed_manifest is None:
        raise AssertionError("Pilot manifest was not parsed")
    _equal(
        parsed_manifest.get("protocol", {}).get("sha256"),
        PILOT_PROTOCOL_SHA256,
        "Pilot manifest protocol lineage",
    )
    _validate_pilot_manifest(parsed_manifest, protocol)
    return parsed_manifest, verified


def validate_contract(
    repository: Path,
    protocol_path: Path,
    dataset_name: str,
    *,
    output_override: Path | None = None,
    require_output_absent: bool = False,
) -> dict[str, Any]:
    repository = secure_io._absolute_lexical_path(repository)
    protocol_path = secure_io._absolute_lexical_path(protocol_path)
    protocol, protocol_sha256 = load_protocol(protocol_path, repository=repository)
    pilot_manifest, lineage_hashes = _verify_lineage(repository, protocol)
    datasets = _mapping(protocol["datasets"], "datasets")
    if dataset_name not in datasets:
        raise ValueError(f"unknown dataset: {dataset_name}")
    contract = _mapping(datasets[dataset_name], dataset_name)
    snapshots: dict[str, tuple[bytes, str]] = {}
    for role, path_key, hash_key in (
        ("train", "train_split", "train_split_sha256"),
        ("test", "test_split", "test_split_sha256"),
        ("calibration", "calibration_ids", "calibration_ids_file_sha256"),
    ):
        path = _safe_path(repository, contract.get(path_key), f"{dataset_name} {role}")
        snapshots[role] = _snapshot(path, contract.get(hash_key), f"{dataset_name} {role}")
    train_ids = secure_io._parse_canonical_ids(snapshots["train"][0], label="train IDs")
    test_ids = secure_io._parse_canonical_ids(snapshots["test"][0], label="test IDs")
    selected_ids = secure_io._parse_canonical_ids(
        snapshots["calibration"][0], label="calibration IDs"
    )
    _equal(len(selected_ids), 64, "calibration ID count")
    selected_hash = secure_io.ordered_ids_sha256(selected_ids)
    _equal(
        selected_hash,
        contract.get("calibration_ordered_ids_sha256"),
        "calibration ordered ID SHA256",
    )
    train_set, test_set, selected_set = set(train_ids), set(test_ids), set(selected_ids)
    _equal(len(selected_set), 64, "calibration ID uniqueness")
    _equal(train_set & test_set, set(), "train/test overlap")
    _equal(selected_set <= train_set, True, "calibration IDs contained in train")
    _equal(selected_set & test_set, set(), "calibration/test overlap")
    pilot_dataset = _mapping(
        _mapping(pilot_manifest.get("datasets"), "Pilot manifest datasets").get(dataset_name),
        f"Pilot manifest {dataset_name}",
    )
    pilot_output = _mapping(pilot_dataset.get("output"), "Pilot output")
    _equal(pilot_output.get("file_sha256"), snapshots["calibration"][1], "Pilot ID file")
    _equal(pilot_output.get("ordered_ids_sha256"), selected_hash, "Pilot ordered IDs")

    root = _safe_path(repository, contract.get("root"), f"{dataset_name} root")
    cache_root = _safe_path(repository, protocol["cache"]["root"], "cache root")
    final_output = (
        secure_io._absolute_lexical_path(output_override)
        if output_override is not None
        else cache_root / dataset_name
    )
    if require_output_absent and (final_output.exists() or final_output.is_symlink()):
        raise FileExistsError(f"cache output exists; refusing overwrite: {final_output}")
    return {
        "repository": repository,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha256,
        "dataset": dataset_name,
        "dataset_root": root,
        "train_split": str(contract["train_split"]),
        "test_split": str(contract["test_split"]),
        "calibration_ids_path": str(contract["calibration_ids"]),
        "train_split_sha256": snapshots["train"][1],
        "test_split_sha256": snapshots["test"][1],
        "calibration_ids_file_sha256": snapshots["calibration"][1],
        "selected_ids": selected_ids,
        "ordered_ids_sha256": selected_hash,
        "conditions": CONDITIONS,
        "lineage_hashes": lineage_hashes,
        "final_output": final_output,
        "validation": {
            "metadata_only": True,
            "numpy_load_calls": 0,
            "torch_imports": 0,
            "pil_imported": False,
            "model_constructed": False,
            "cuda_calls": 0,
            "output_pixels_opened": 0,
            "test_images_opened": 0,
            "test_masks_opened": 0,
            "selected_ids_contained_in_train": True,
            "selected_ids_absent_from_test": True,
            "old_round_02_cache_read": False,
        },
    }


def _tensor_sequence_hash(records: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for image_id, array in records:
        contiguous = array.astype("<f4", copy=False)
        digest.update(image_id.encode("utf-8") + b"\0")
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _cache_content_sha256(manifest: Mapping[str, Any]) -> str:
    files = _mapping(manifest.get("files"), "cache manifest files")
    content = {
        "protocol_sha256": manifest["protocol_sha256"],
        "dataset": manifest["dataset"],
        "ordered_ids_sha256": manifest["ordered_ids_sha256"],
        "files": {
            name: _mapping(value, f"cache file {name}")["sha256"]
            for name, value in sorted(files.items())
        },
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resolve_pixel_file(directory: Path, image_id: str) -> Path:
    suffix = Path(image_id).suffix
    candidates = (
        [directory / image_id]
        if suffix
        else [directory / f"{image_id}{ext}" for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")]
    )
    matches = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one pixel file for {image_id}: {matches}")
    return matches[0]


def _materialize_staging(context: Mapping[str, Any], staging: Path) -> dict[str, Any]:
    """Pixel path; imports scientific/image packages only after validation."""

    import numpy as np
    from PIL import Image
    from corruptions.corruption_protocol import make_sample_rng
    from corruptions.infrared_corruptions import apply_corruption

    selected_ids = tuple(context["selected_ids"])
    root = Path(context["dataset_root"])
    layouts = [
        (root / "img", root / "label"),
        (root / "images", root / "masks"),
    ]
    layouts = [(images, masks) for images, masks in layouts if images.is_dir() and masks.is_dir()]
    if len(layouts) != 1:
        raise ValueError(f"dataset layout must be unambiguous: {root}")
    images_dir, masks_dir = layouts[0]
    conditions_dir = staging / "conditions"
    targets_dir = staging / "outer_evaluator"
    conditions_dir.mkdir()
    targets_dir.mkdir()
    maps: dict[str, Any] = {}
    for corruption, severity in CONDITIONS:
        key = _condition_key(corruption, severity)
        maps[key] = np.lib.format.open_memmap(
            conditions_dir / f"{key}.npy", mode="w+", dtype="<f4", shape=(64, 3, 256, 256)
        )
    targets = np.lib.format.open_memmap(
        targets_dir / "targets.npy", mode="w+", dtype="<f4", shape=(64, 1, 256, 256)
    )
    image_records: dict[str, list[tuple[str, Any]]] = {
        _condition_key(*condition): [] for condition in CONDITIONS
    }
    target_records: list[tuple[str, Any]] = []
    original_sizes: list[list[int]] = []
    source_files: dict[str, dict[str, str]] = {}
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    for index, image_id in enumerate(selected_ids):
        image_path = _resolve_pixel_file(images_dir, image_id)
        mask_path = _resolve_pixel_file(masks_dir, image_id)
        image_bytes, image_sha = secure_io._read_stable_bytes(image_path, label=f"image {image_id}")
        mask_bytes, mask_sha = secure_io._read_stable_bytes(mask_path, label=f"mask {image_id}")
        source_files[image_id] = {"image_sha256": image_sha, "mask_sha256": mask_sha}
        with Image.open(io.BytesIO(image_bytes)) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            image_01 = np.asarray(
                image.resize((256, 256), resample=Image.Resampling.BILINEAR), dtype=np.float32
            ) / 255.0
        with Image.open(io.BytesIO(mask_bytes)) as handle:
            mask_01 = np.asarray(
                handle.convert("L").resize((256, 256), resample=Image.Resampling.NEAREST),
                dtype=np.float32,
            ) / 255.0
        target = np.ascontiguousarray(mask_01[None], dtype="<f4")
        targets[index] = target
        target_records.append((image_id, target))
        original_sizes.append([height, width])
        for corruption, severity in CONDITIONS:
            key = _condition_key(corruption, severity)
            physical = image_01
            if corruption != "clean":
                physical = apply_corruption(
                    image_01.copy(),
                    corruption,
                    severity,
                    make_sample_rng(image_id, corruption, severity, 42),
                )
            tensor = np.ascontiguousarray(
                ((physical - mean) / std).transpose(2, 0, 1), dtype="<f4"
            )
            maps[key][index] = tensor
            image_records[key].append((image_id, tensor))
    targets.flush()
    del targets
    for value in maps.values():
        value.flush()
    maps.clear()

    files: dict[str, dict[str, Any]] = {}
    condition_records: list[dict[str, Any]] = []
    for index, (corruption, severity) in enumerate(CONDITIONS):
        key = _condition_key(corruption, severity)
        relative = f"conditions/{key}.npy"
        path = staging / relative
        digest = secure_io.sha256_file(path)
        files[relative] = {"sha256": digest, "bytes": path.stat().st_size}
        condition_records.append(
            {
                "index": index,
                "key": key,
                "corruption": corruption,
                "severity": severity,
                "path": relative,
                "shape": [64, 3, 256, 256],
                "dtype": "little_endian_float32",
                "file_sha256": digest,
                "file_bytes": path.stat().st_size,
                "tensor_sequence_sha256": _tensor_sequence_hash(image_records[key]),
            }
        )
    target_path = staging / "outer_evaluator/targets.npy"
    target_digest = secure_io.sha256_file(target_path)
    files["outer_evaluator/targets.npy"] = {
        "sha256": target_digest,
        "bytes": target_path.stat().st_size,
    }
    content = {
        "protocol_sha256": context["protocol_sha256"],
        "dataset": context["dataset"],
        "ordered_ids_sha256": context["ordered_ids_sha256"],
        "files": {name: value["sha256"] for name, value in sorted(files.items())},
    }
    return {
        "schema_version": 2,
        "cache_format": CACHE_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": context["protocol_sha256"],
        "dataset": context["dataset"],
        "split_role": "train_side_pilot_v2_derived_64",
        "train_split": context["train_split"],
        "train_split_sha256": context["train_split_sha256"],
        "test_split_metadata": context["test_split"],
        "test_split_sha256": context["test_split_sha256"],
        "calibration_ids_path": context["calibration_ids_path"],
        "calibration_ids_file_sha256": context["calibration_ids_file_sha256"],
        "image_ids": list(selected_ids),
        "ordered_ids_sha256": context["ordered_ids_sha256"],
        "seed": 42,
        "original_sizes": original_sizes,
        "conditions": condition_records,
        "targets": {
            "path": "outer_evaluator/targets.npy",
            "shape": [64, 1, 256, 256],
            "dtype": "little_endian_float32",
            "file_sha256": target_digest,
            "file_bytes": target_path.stat().st_size,
            "tensor_sequence_sha256": _tensor_sequence_hash(target_records),
            "role": "outer_evaluator_only",
            "method_facing_access": "forbidden",
        },
        "source_files": source_files,
        "source_open_scope": {
            "unique_train_images": 64,
            "unique_train_masks": 64,
            "test_images": 0,
            "test_masks": 0,
        },
        "lineage_files_sha256": dict(context["lineage_hashes"]),
        "files": files,
        "label_firewall": {
            "method_received_labels": False,
            "adaptation_interface_invoked": False,
            "targets_for_outer_evaluator_only": True,
            "method_facing_fields": list(METHOD_FIELDS),
            "forbidden_method_fields": sorted(FORBIDDEN_METHOD_FIELDS),
        },
        "cache_content_sha256": hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _assert_metadata_seal(
    context: Mapping[str, Any], *, stage: str, require_output_absent: bool = True
) -> None:
    fresh = validate_contract(
        Path(context["repository"]),
        Path(context["protocol_path"]),
        str(context["dataset"]),
        output_override=Path(context["final_output"]),
        require_output_absent=require_output_absent,
    )
    for key in (
        "protocol_sha256",
        "train_split_sha256",
        "test_split_sha256",
        "calibration_ids_file_sha256",
        "ordered_ids_sha256",
        "lineage_hashes",
        "selected_ids",
    ):
        if fresh[key] != context[key]:
            raise RuntimeError(
                f"metadata runtime seal changed at {stage}: {key}; refusing publish"
            )


def _method_manifest(manifest: Mapping[str, Any], outer_sha256: str) -> dict[str, Any]:
    outer_files = _mapping(manifest.get("files"), "outer manifest files")
    image_files: dict[str, dict[str, Any]] = {}
    for corruption, severity in CONDITIONS:
        key = _condition_key(corruption, severity)
        relative = f"conditions/{key}.npy"
        record = _mapping(outer_files.get(relative), f"outer image shard {key}")
        image_files[relative] = {
            "sha256": _sha256(record.get("sha256"), f"{key} image shard SHA256"),
            "bytes": int(record["bytes"]),
        }
    return {
        "schema_version": 2,
        "cache_format": CACHE_FORMAT,
        "protocol_sha256": manifest["protocol_sha256"],
        "dataset": manifest["dataset"],
        "outer_manifest_sha256": outer_sha256,
        "image_ids": manifest["image_ids"],
        "ordered_ids_sha256": manifest["ordered_ids_sha256"],
        "original_sizes": manifest.get("original_sizes", [[256, 256]] * 64),
        "seed": 42,
        "conditions": manifest["conditions"],
        "files": image_files,
        "sample_fields": list(METHOD_FIELDS),
        "forbidden_fields": sorted(FORBIDDEN_METHOD_FIELDS),
        "targets_exposed": False,
        "corruption_regeneration_forbidden": True,
        "read_only_mmap_required": True,
    }


def _validate_outer_manifest_structure(outer: Mapping[str, Any]) -> None:
    _equal(outer.get("schema_version"), 2, "outer manifest schema")
    _equal(outer.get("cache_format"), CACHE_FORMAT, "outer cache format")
    _equal(outer.get("protocol_id"), PROTOCOL_ID, "outer protocol_id")
    _sha256(outer.get("protocol_sha256"), "outer protocol SHA256")
    image_ids = outer.get("image_ids")
    if not isinstance(image_ids, Sequence) or isinstance(image_ids, (str, bytes)):
        raise TypeError("outer image_ids must be a sequence")
    _equal(len(image_ids), 64, "outer image ID count")
    _equal(len(set(image_ids)), 64, "outer image ID uniqueness")
    _equal(
        secure_io.ordered_ids_sha256(tuple(str(value) for value in image_ids)),
        outer.get("ordered_ids_sha256"),
        "outer ordered IDs SHA256",
    )
    files = _mapping(outer.get("files"), "outer manifest files")
    expected_image_paths = {
        f"conditions/{_condition_key(corruption, severity)}.npy"
        for corruption, severity in CONDITIONS
    }
    target_relative = "outer_evaluator/targets.npy"
    _equal(
        set(files), expected_image_paths | {target_relative}, "outer exact payload set"
    )
    raw_conditions = outer.get("conditions")
    if not isinstance(raw_conditions, Sequence) or isinstance(
        raw_conditions, (str, bytes)
    ):
        raise TypeError("outer conditions must be a sequence")
    _equal(len(raw_conditions), len(CONDITIONS), "outer condition count")
    for index, ((corruption, severity), raw_record) in enumerate(
        zip(CONDITIONS, raw_conditions, strict=True)
    ):
        record = _mapping(raw_record, f"outer condition {index}")
        key = _condition_key(corruption, severity)
        relative = f"conditions/{key}.npy"
        for actual, expected, label in (
            (record.get("index"), index, "index"),
            (record.get("key"), key, "key"),
            (record.get("corruption"), corruption, "corruption"),
            (record.get("severity"), severity, "severity"),
            (record.get("path"), relative, "path"),
            (record.get("shape"), [64, 3, 256, 256], "shape"),
            (record.get("dtype"), "little_endian_float32", "dtype"),
        ):
            _equal(actual, expected, f"outer {key} {label}")
        file_record = _mapping(files.get(relative), f"outer file {relative}")
        _equal(
            record.get("file_sha256"),
            _sha256(file_record.get("sha256"), f"outer {key} file SHA256"),
            f"outer {key} condition/file SHA256",
        )
        file_bytes = file_record.get("bytes")
        if isinstance(file_bytes, bool) or not isinstance(file_bytes, int) or file_bytes <= 0:
            raise ValueError(f"outer {key} file bytes must be positive integer")
        _equal(record.get("file_bytes"), file_bytes, f"outer {key} file bytes")
    targets = _mapping(outer.get("targets"), "outer targets")
    for actual, expected, label in (
        (targets.get("path"), target_relative, "path"),
        (targets.get("shape"), [64, 1, 256, 256], "shape"),
        (targets.get("dtype"), "little_endian_float32", "dtype"),
        (targets.get("role"), "outer_evaluator_only", "role"),
        (targets.get("method_facing_access"), "forbidden", "firewall"),
    ):
        _equal(actual, expected, f"outer target {label}")
    target_file = _mapping(files.get(target_relative), "outer target file")
    _equal(
        targets.get("file_sha256"),
        _sha256(target_file.get("sha256"), "outer target SHA256"),
        "outer target/file SHA256",
    )
    target_bytes = target_file.get("bytes")
    if isinstance(target_bytes, bool) or not isinstance(target_bytes, int) or target_bytes <= 0:
        raise ValueError("outer target bytes must be positive integer")
    _equal(targets.get("file_bytes"), target_bytes, "outer target/file bytes")
    _equal(
        outer.get("cache_content_sha256"),
        _cache_content_sha256(outer),
        "outer cache content SHA256",
    )


def _consumer_metadata(
    cache_dir: Path, *, expected_protocol_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cache_dir = secure_io._absolute_lexical_path(cache_dir)
    actual_root_entries = secure_io._directory_names_nofollow(
        cache_dir, label="cache directory"
    )
    _equal(
        actual_root_entries,
        {
            "conditions",
            "outer_evaluator",
            "manifest.json",
            "method_input_manifest.json",
            "COMPLETE.json",
        },
        "cache root exact entries",
    )
    complete_payload, _ = secure_io._read_stable_bytes(
        cache_dir / "COMPLETE.json", label="cache COMPLETE"
    )
    complete = _json(complete_payload, "cache COMPLETE")
    _equal(complete.get("complete"), True, "cache complete flag")
    _equal(
        complete.get("protocol_sha256"), expected_protocol_sha256, "consumer protocol SHA256"
    )
    outer_payload, outer_sha = secure_io._read_stable_bytes(
        cache_dir / "manifest.json", label="outer cache manifest"
    )
    _equal(outer_sha, complete.get("manifest_sha256"), "outer manifest SHA256")
    method_payload, method_sha = secure_io._read_stable_bytes(
        cache_dir / "method_input_manifest.json", label="method input manifest"
    )
    _equal(
        method_sha,
        complete.get("method_input_manifest_sha256"),
        "method manifest SHA256",
    )
    outer = _json(outer_payload, "outer cache manifest")
    method = _json(method_payload, "method input manifest")
    _validate_outer_manifest_structure(outer)
    _equal(outer.get("protocol_sha256"), expected_protocol_sha256, "outer protocol SHA256")
    _equal(complete.get("dataset"), outer.get("dataset"), "COMPLETE dataset")
    _equal(
        complete.get("cache_content_sha256"),
        outer.get("cache_content_sha256"),
        "COMPLETE cache content SHA256",
    )
    _equal(method.get("outer_manifest_sha256"), outer_sha, "method-to-outer lineage")
    _equal(method.get("targets_exposed"), False, "method target exposure")
    _equal(tuple(method.get("sample_fields", ())), METHOD_FIELDS, "method sample fields")
    if set(method.get("sample_fields", ())) & FORBIDDEN_METHOD_FIELDS:
        raise ValueError("method manifest exposes a forbidden label field")
    _equal(method, _method_manifest(outer, outer_sha), "method manifest projection")
    return complete, outer, method


def _descriptor_sha256(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _assert_descriptor_path_identity(
    path: Path, descriptor: int, *, label: str
) -> os.stat_result:
    descriptor_stat = os.fstat(descriptor)
    current = secure_io._open_nofollow(path, directory=False, label=label)
    try:
        current_stat = os.fstat(current)
    finally:
        os.close(current)
    _equal(
        (current_stat.st_dev, current_stat.st_ino),
        (descriptor_stat.st_dev, descriptor_stat.st_ino),
        f"{label} pathname inode",
    )
    return descriptor_stat


def _open_verified_payload(
    cache_dir: Path,
    *,
    relative: Any,
    expected_relative: str,
    file_record: Mapping[str, Any],
    label: str,
) -> tuple[int, Path]:
    if relative != expected_relative:
        raise ValueError(
            f"{label} path must be exactly {expected_relative!r}, got {relative!r}"
        )
    path = secure_io._absolute_lexical_path(cache_dir) / expected_relative
    descriptor = secure_io._open_nofollow(path, directory=False, label=label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        expected_bytes = file_record.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
        ):
            raise ValueError(f"{label} configured bytes must be a positive integer")
        _equal(before.st_size, expected_bytes, f"{label} byte size")
        expected_hash = _sha256(file_record.get("sha256"), f"{label} SHA256")
        first_hash = _descriptor_sha256(descriptor)
        middle = os.fstat(descriptor)
        second_hash = _descriptor_sha256(descriptor)
        after = os.fstat(descriptor)
        _equal(
            secure_io._stable_stat_identity(middle),
            secure_io._stable_stat_identity(before),
            f"{label} stable identity during first hash",
        )
        _equal(
            secure_io._stable_stat_identity(after),
            secure_io._stable_stat_identity(before),
            f"{label} stable identity during replay hash",
        )
        _equal(first_hash, expected_hash, f"{label} first SHA256")
        _equal(second_hash, expected_hash, f"{label} replay SHA256")
        _assert_descriptor_path_identity(path, descriptor, label=label)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, path
    except BaseException:
        os.close(descriptor)
        raise


def _numpy_load(path: Path, *, verified_fd: int | None = None):
    import numpy as np

    load_path: str | Path = (
        f"/proc/self/fd/{verified_fd}" if verified_fd is not None else path
    )
    return np.load(load_path, mmap_mode="r", allow_pickle=False)


def _validate_numpy_mmap(
    value: Any, *, expected_shape: tuple[int, ...], label: str
) -> None:
    import numpy as np

    if not isinstance(value, np.memmap):
        raise TypeError(f"{label} must be a NumPy read-only mmap")
    _equal(tuple(value.shape), expected_shape, f"{label} shape")
    _equal(value.dtype.str, "<f4", f"{label} dtype")
    _equal(bool(value.flags.c_contiguous), True, f"{label} C-contiguous")
    _equal(bool(value.flags.writeable), False, f"{label} read-only")
    for index in range(expected_shape[0]):
        if not bool(np.isfinite(value[index]).all()):
            raise ValueError(f"{label} contains NaN/Inf at index {index}")


def _load_verified_numpy_payload(
    cache_dir: Path,
    *,
    relative: Any,
    expected_relative: str,
    file_record: Mapping[str, Any],
    expected_shape: tuple[int, ...],
    label: str,
):
    descriptor, path = _open_verified_payload(
        cache_dir,
        relative=relative,
        expected_relative=expected_relative,
        file_record=file_record,
        label=label,
    )
    try:
        value = _numpy_load(path, verified_fd=descriptor)
        _assert_descriptor_path_identity(path, descriptor, label=label)
    finally:
        os.close(descriptor)
    _validate_numpy_mmap(value, expected_shape=expected_shape, label=label)
    return value


class SourceCalibrationMethodInputDatasetV2:
    """Label-free, read-only-mmap method view of one cached condition."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        condition_key: str,
        expected_protocol_sha256: str,
    ) -> None:
        self.cache_dir = secure_io._absolute_lexical_path(cache_dir)
        _, _, method = _consumer_metadata(
            self.cache_dir, expected_protocol_sha256=expected_protocol_sha256
        )
        expected_conditions = {_condition_key(*value) for value in CONDITIONS}
        if condition_key not in expected_conditions:
            raise ValueError(f"unsupported cached condition {condition_key!r}")
        records = [record for record in method["conditions"] if record["key"] == condition_key]
        if len(records) != 1:
            raise ValueError(f"expected one cached condition {condition_key!r}")
        self.record = records[0]
        self.image_ids = tuple(method["image_ids"])
        self.original_sizes = tuple(tuple(value) for value in method["original_sizes"])
        self.dataset_name = str(method["dataset"])
        self.seed = int(method["seed"])
        expected_relative = f"conditions/{condition_key}.npy"
        method_files = _mapping(method.get("files"), "method image shard files")
        self.images = _load_verified_numpy_payload(
            self.cache_dir,
            relative=self.record.get("path"),
            expected_relative=expected_relative,
            file_record=_mapping(
                method_files.get(expected_relative), f"method file {expected_relative}"
            ),
            expected_shape=(64, 3, 256, 256),
            label=f"method condition {condition_key}",
        )
        _equal(len(self.image_ids), 64, "method cache ID count")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        private_image = self.images[index].copy(order="C")
        return {
            "image": torch.from_numpy(private_image),
            "image_id": self.image_ids[index],
            "original_size": self.original_sizes[index],
            "dataset": self.dataset_name,
            "corruption": self.record["corruption"],
            "severity": int(self.record["severity"]),
            "seed": self.seed,
        }


def load_outer_evaluator_targets_v2(
    cache_dir: Path,
    *,
    expected_protocol_sha256: str,
    episodes_complete: bool,
):
    if type(episodes_complete) is not bool or not episodes_complete:
        raise PermissionError("outer targets require explicitly completed method episodes")
    resolved = secure_io._absolute_lexical_path(cache_dir)
    _, outer, _ = _consumer_metadata(
        resolved, expected_protocol_sha256=expected_protocol_sha256
    )
    targets = _mapping(outer.get("targets"), "outer targets")
    _equal(targets.get("method_facing_access"), "forbidden", "target firewall")
    expected_relative = "outer_evaluator/targets.npy"
    files = _mapping(outer.get("files"), "outer manifest files")
    return _load_verified_numpy_payload(
        resolved,
        relative=targets.get("path"),
        expected_relative=expected_relative,
        file_record=_mapping(files.get(expected_relative), "outer target file"),
        expected_shape=(64, 1, 256, 256),
        label="outer evaluator targets",
    )


def _expected_complete(
    context: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
    method_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "complete": True,
        "dataset": context["dataset"],
        "protocol_sha256": context["protocol_sha256"],
        "cache_content_sha256": manifest["cache_content_sha256"],
        "manifest_sha256": manifest_sha256,
        "method_input_manifest_sha256": method_manifest_sha256,
        "atomic_no_replace": True,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "method_received_labels": False,
    }


def _fsync_directory(path: Path, *, label: str) -> None:
    descriptor = secure_io._open_nofollow(path, directory=True, label=label)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    entries = sorted(root.rglob("*"), key=lambda value: value.as_posix())
    directories: list[Path] = []
    for path in entries:
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"cache staging tree contains a symlink: {path}")
        if stat.S_ISDIR(observed.st_mode):
            directories.append(path)
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"cache staging entry is not regular: {path}")
        descriptor = secure_io._open_nofollow(
            path, directory=False, label="cache staging file"
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(
        directories, key=lambda value: len(value.parts), reverse=True
    ):
        _fsync_directory(directory, label="cache staging directory")
    _fsync_directory(root, label="cache staging root")


def validate_materialization(context: Mapping[str, Any]) -> dict[str, Any]:
    """Fully rehash and semantically verify the published immutable cache."""

    final = secure_io._absolute_lexical_path(Path(context["final_output"]))
    complete, outer, method = _consumer_metadata(
        final, expected_protocol_sha256=str(context["protocol_sha256"])
    )
    _equal(outer.get("dataset"), context["dataset"], "published dataset")
    _equal(outer.get("image_ids"), list(context["selected_ids"]), "published IDs")
    _equal(
        outer.get("ordered_ids_sha256"),
        context["ordered_ids_sha256"],
        "published ordered IDs",
    )
    _equal(
        outer.get("lineage_files_sha256"),
        dict(context["lineage_hashes"]),
        "published runtime/lineage seal",
    )
    conditions_names = secure_io._directory_names_nofollow(
        final / "conditions", label="published conditions directory"
    )
    expected_condition_names = {
        f"{_condition_key(corruption, severity)}.npy"
        for corruption, severity in CONDITIONS
    }
    _equal(conditions_names, expected_condition_names, "published condition files")
    _equal(
        secure_io._directory_names_nofollow(
            final / "outer_evaluator", label="published target directory"
        ),
        {"targets.npy"},
        "published target files",
    )
    files = _mapping(outer.get("files"), "published payload files")
    for relative, raw_record in sorted(files.items()):
        record = _mapping(raw_record, f"published file {relative}")
        descriptor, _ = _open_verified_payload(
            final,
            relative=relative,
            expected_relative=relative,
            file_record=record,
            label=f"published payload {relative}",
        )
        os.close(descriptor)
    manifest_payload, manifest_sha256 = secure_io._read_stable_bytes(
        final / "manifest.json", label="published outer manifest"
    )
    _equal(_json(manifest_payload, "published outer manifest"), outer, "outer replay")
    method_payload, method_sha256 = secure_io._read_stable_bytes(
        final / "method_input_manifest.json", label="published method manifest"
    )
    _equal(_json(method_payload, "published method manifest"), method, "method replay")
    expected_complete = _expected_complete(
        context,
        outer,
        manifest_sha256=manifest_sha256,
        method_manifest_sha256=method_sha256,
    )
    _equal(complete, expected_complete, "published COMPLETE")
    return {
        "valid": True,
        "complete": complete,
        "manifest_sha256": manifest_sha256,
        "method_input_manifest_sha256": method_sha256,
        "verified_payload_count": len(files),
        "full_file_hashes_verified": True,
    }


def _assert_lock_identity(lock: Path, descriptor: int) -> os.stat_result:
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"publish lock descriptor is not regular: {lock}")
    current = secure_io._open_nofollow(
        lock, directory=False, label="publish lock identity"
    )
    try:
        current_stat = os.fstat(current)
    finally:
        os.close(current)
    _equal(
        (current_stat.st_dev, current_stat.st_ino),
        (observed.st_dev, observed.st_ino),
        "publish lock pathname/inode",
    )
    return observed


def _create_publish_lock(lock: Path) -> int:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lock, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"publish lock exists; use explicit stale recovery: {lock}"
        ) from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        _assert_lock_identity(lock, descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        if lock.exists() and not lock.is_symlink():
            lock.unlink()
        raise


def _release_publish_lock(lock: Path, descriptor: int, *, unlink: bool) -> None:
    try:
        if unlink:
            _assert_lock_identity(lock, descriptor)
            lock.unlink()
            _fsync_directory(lock.parent, label="publish lock parent")
    finally:
        os.close(descriptor)


def _read_lock_pid(descriptor: int) -> int:
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = os.read(descriptor, 256)
    if os.read(descriptor, 1):
        raise ValueError("stale publish lock metadata is too large")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("stale publish lock metadata is not ASCII") from error
    match = re.fullmatch(r"pid=([1-9][0-9]*)\n", text)
    if match is None:
        raise ValueError("stale publish lock has invalid PID metadata")
    return int(match.group(1))


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_stale(context: Mapping[str, Any]) -> dict[str, int]:
    final = secure_io._absolute_lexical_path(Path(context["final_output"]))
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"completed/output cache exists; recovery refused: {final}")
    parent = final.parent
    secure_io._directory_names_nofollow(parent, label="stale recovery parent")
    lock = final.with_name(f".{final.name}.publish.lock")
    try:
        descriptor = secure_io._open_nofollow(
            lock, directory=False, label="stale publish lock"
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "stale recovery requires the corresponding publish lock"
        ) from error
    remove_lock = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("publish lock is live/held; recovery refused") from error
        _assert_lock_identity(lock, descriptor)
        pid = _read_lock_pid(descriptor)
        if _pid_is_alive(pid):
            raise RuntimeError(f"publish lock PID {pid} is live; recovery refused")
        removed_staging = 0
        owned_prefix = f".{final.name}.build-{pid}-"
        for name in sorted(
            secure_io._directory_names_nofollow(parent, label="stale recovery parent")
        ):
            if not name.startswith(owned_prefix):
                continue
            path = parent / name
            observed = path.lstat()
            if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"unsafe stale staging entry: {path}")
            shutil.rmtree(path)
            removed_staging += 1
        _assert_lock_identity(lock, descriptor)
        remove_lock = True
    finally:
        _release_publish_lock(lock, descriptor, unlink=remove_lock)
    return {"removed_staging": removed_staging, "removed_lock": 1}


def materialize(context: Mapping[str, Any]) -> dict[str, Any]:
    final = secure_io._absolute_lexical_path(Path(context["final_output"]))
    _assert_metadata_seal(context, stage="materialize_start")
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"cache exists; refusing overwrite: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    secure_io._directory_names_nofollow(
        final.parent, label="cache publication parent"
    )
    lock = final.with_name(f".{final.name}.publish.lock")
    descriptor = _create_publish_lock(lock)
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{final.name}.build-{os.getpid()}-", dir=final.parent
            )
        )
        _assert_lock_identity(lock, descriptor)
        manifest = _materialize_staging(context, staging)
        _assert_metadata_seal(context, stage="after_pixel_materialization")
        _write_json(staging / "manifest.json", manifest)
        manifest_sha = secure_io.sha256_file(staging / "manifest.json")
        method_manifest = _method_manifest(manifest, manifest_sha)
        _write_json(staging / "method_input_manifest.json", method_manifest)
        method_sha = secure_io.sha256_file(staging / "method_input_manifest.json")
        complete = _expected_complete(
            context,
            manifest,
            manifest_sha256=manifest_sha,
            method_manifest_sha256=method_sha,
        )
        _write_json(staging / "COMPLETE.json", complete)
        _assert_metadata_seal(context, stage="pre_atomic_publish")
        _assert_lock_identity(lock, descriptor)
        _fsync_tree(staging)
        secure_io._atomic_rename_directory_noreplace(staging, final)
        _fsync_directory(final.parent, label="cache publication parent")
        _assert_metadata_seal(
            context, stage="post_atomic_publish", require_output_absent=False
        )
        _assert_lock_identity(lock, descriptor)
        validation = validate_materialization(context)
        _equal(validation.get("complete"), complete, "post-publish COMPLETE")
    except BaseException:
        if (
            staging is not None
            and staging.is_dir()
            and staging.parent == final.parent
        ):
            shutil.rmtree(staging)
        raise
    finally:
        _release_publish_lock(lock, descriptor, unlink=True)
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--recover-stale", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = validate_contract(
        PROJECT_ROOT,
        args.protocol,
        args.dataset,
        require_output_absent=not args.validate_only,
    )
    if args.validate_only:
        return {
            "validate_only": True,
            "dataset": args.dataset,
            "protocol_sha256": context["protocol_sha256"],
            "selected_count": len(context["selected_ids"]),
            "ordered_ids_sha256": context["ordered_ids_sha256"],
            "condition_count": len(context["conditions"]),
            "planned_output": str(context["final_output"]),
            "validation": context["validation"],
        }
    if args.recover_stale:
        return {"recovered": True, **recover_stale(context)}
    return materialize(context)


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
