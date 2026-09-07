"""Capability-minimal, label-free reader for Stage-C0 method inputs.

This module deliberately does not import the cache materializer or its outer
evaluator API.  It can read only the cache completion receipt, the projected
method-input manifest, and one explicitly selected condition image shard.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Final

import numpy as np

from dataio import train_side_pilot_protocol as secure_io


CONDITIONS: Final = (
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
METHOD_FIELDS: Final = (
    "image",
    "image_id",
    "original_size",
    "dataset",
    "corruption",
    "severity",
    "seed",
)
FORBIDDEN_FIELDS: Final = ("ground_truth", "gt", "label", "mask", "target")
SHA256_HEX: Final = frozenset("0123456789abcdef")


class StageC0MethodInputError(RuntimeError):
    """The projected label-free cache view violates its sealed contract."""


def _condition_key(corruption: str, severity: int) -> str:
    return "clean_S0" if corruption == "clean" else f"{corruption}_S{severity}"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageC0MethodInputError(f"{label} must be a mapping")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise StageC0MethodInputError(f"{label} must be lowercase SHA-256")
    return value


def _stable_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        payload, digest = secure_io._read_stable_bytes(path, label=label)
        value = json.loads(payload.decode("utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise StageC0MethodInputError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise StageC0MethodInputError(f"{label} root must be a mapping")
    return value, digest


def _ordered_ids_sha256(values: tuple[str, ...]) -> str:
    payload = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _descriptor_sha256(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _open_verified_image_shard(
    cache_dir: Path,
    *,
    relative: Any,
    expected_relative: str,
    file_record: Mapping[str, Any],
    expected_shape: tuple[int, ...],
) -> np.memmap:
    if relative != expected_relative:
        raise StageC0MethodInputError("condition shard path differs")
    path = cache_dir / expected_relative
    try:
        descriptor = secure_io._open_nofollow(
            path, directory=False, label="Stage-C0 method image shard"
        )
    except (OSError, ValueError) as exc:
        raise StageC0MethodInputError("cannot securely open condition shard") from exc
    try:
        before = os.fstat(descriptor)
        expected_bytes = file_record.get("bytes")
        if (
            not stat.S_ISREG(before.st_mode)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
            or before.st_size != expected_bytes
        ):
            raise StageC0MethodInputError("condition shard file layout differs")
        expected_hash = _sha256(file_record.get("sha256"), "condition shard")
        first_hash = _descriptor_sha256(descriptor)
        middle = os.fstat(descriptor)
        second_hash = _descriptor_sha256(descriptor)
        after = os.fstat(descriptor)
        if (
            secure_io._stable_stat_identity(before)
            != secure_io._stable_stat_identity(middle)
            or secure_io._stable_stat_identity(before)
            != secure_io._stable_stat_identity(after)
            or first_hash != expected_hash
            or second_hash != expected_hash
        ):
            raise StageC0MethodInputError("condition shard changed or hash differs")
        current = secure_io._open_nofollow(
            path, directory=False, label="Stage-C0 method image shard replay"
        )
        try:
            current_stat = os.fstat(current)
            if (current_stat.st_dev, current_stat.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise StageC0MethodInputError("condition shard pathname changed")
        finally:
            os.close(current)
        os.lseek(descriptor, 0, os.SEEK_SET)
        value = np.load(
            f"/proc/self/fd/{descriptor}", mmap_mode="r", allow_pickle=False
        )
        if secure_io._stable_stat_identity(os.fstat(descriptor)) != (
            secure_io._stable_stat_identity(before)
        ):
            raise StageC0MethodInputError("condition shard changed during mmap")
    except BaseException:
        os.close(descriptor)
        raise
    os.close(descriptor)
    if (
        not isinstance(value, np.memmap)
        or bool(value.flags.writeable)
        or not bool(value.flags.c_contiguous)
        or value.dtype.str != "<f4"
        or tuple(value.shape) != expected_shape
    ):
        raise StageC0MethodInputError("condition image mmap layout differs")
    for index in range(expected_shape[0]):
        if not bool(np.isfinite(value[index]).all()):
            raise StageC0MethodInputError(
                f"condition image shard is non-finite at index {index}"
            )
    return value


class StageC0MethodInputDataset:
    """Read-only image-only Pilot64 view of one frozen cache condition."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        condition_key: str,
        expected_protocol_sha256: str,
        expected_dataset: str,
        expected_complete_sha256: str,
        expected_method_manifest_sha256: str,
        expected_ordered_ids_sha256: str,
    ) -> None:
        self.cache_dir = secure_io._absolute_lexical_path(cache_dir)
        protocol_sha256 = _sha256(
            expected_protocol_sha256, "expected protocol"
        )
        expected_complete = _sha256(
            expected_complete_sha256, "expected completion receipt"
        )
        expected_method = _sha256(
            expected_method_manifest_sha256, "expected method manifest"
        )
        ordered_ids_hash = _sha256(
            expected_ordered_ids_sha256, "expected ordered IDs"
        )
        if not isinstance(expected_dataset, str) or not expected_dataset:
            raise StageC0MethodInputError("expected dataset must be non-empty")

        complete, complete_sha256 = _stable_json(
            self.cache_dir / "COMPLETE.json", label="cache completion receipt"
        )
        if (
            complete_sha256 != expected_complete
            or complete.get("complete") is not True
            or complete.get("atomic_no_replace") is not True
            or complete.get("protocol_sha256") != protocol_sha256
            or complete.get("dataset") != expected_dataset
            or complete.get("method_received_labels") is not False
            or complete.get("test_images_opened") != 0
            or complete.get("test_masks_opened") != 0
            or complete.get("method_input_manifest_sha256") != expected_method
        ):
            raise StageC0MethodInputError("cache completion semantics differ")

        method, method_sha256 = _stable_json(
            self.cache_dir / "method_input_manifest.json",
            label="method-input manifest",
        )
        if method_sha256 != expected_method:
            raise StageC0MethodInputError("method-input manifest hash differs")
        if (
            method.get("schema_version") != 2
            or method.get("cache_format")
            != "nsfpn-binary-tent-ss-calibration-cache-v2"
            or method.get("protocol_sha256") != protocol_sha256
            or method.get("dataset") != expected_dataset
            or method.get("targets_exposed") is not False
            or method.get("read_only_mmap_required") is not True
            or method.get("corruption_regeneration_forbidden") is not True
            or tuple(method.get("sample_fields", ())) != METHOD_FIELDS
            or tuple(method.get("forbidden_fields", ())) != FORBIDDEN_FIELDS
            or method.get("ordered_ids_sha256") != ordered_ids_hash
            or method.get("outer_manifest_sha256")
            != complete.get("manifest_sha256")
        ):
            raise StageC0MethodInputError("method-input manifest semantics differ")

        self.image_ids = tuple(method.get("image_ids", ()))
        raw_sizes = tuple(method.get("original_sizes", ()))
        if (
            len(self.image_ids) != 64
            or len(set(self.image_ids)) != 64
            or any(not isinstance(value, str) or not value for value in self.image_ids)
            or _ordered_ids_sha256(self.image_ids) != ordered_ids_hash
            or len(raw_sizes) != 64
        ):
            raise StageC0MethodInputError("method Pilot64 identity differs")
        self.original_sizes = tuple(tuple(value) for value in raw_sizes)
        if any(
            len(value) != 2
            or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in value)
            for value in self.original_sizes
        ):
            raise StageC0MethodInputError("original-size layout differs")

        expected_keys = tuple(_condition_key(*value) for value in CONDITIONS)
        conditions = tuple(_mapping(value, "method condition") for value in method.get("conditions", ()))
        if tuple(value.get("key") for value in conditions) != expected_keys:
            raise StageC0MethodInputError("method condition roster/order differs")
        if condition_key not in expected_keys:
            raise StageC0MethodInputError("condition key is outside frozen roster")
        records = tuple(value for value in conditions if value.get("key") == condition_key)
        if len(records) != 1:
            raise StageC0MethodInputError("condition record is not unique")
        self.record = records[0]
        condition_index = expected_keys.index(condition_key)
        corruption, severity = CONDITIONS[condition_index]
        expected_relative = f"conditions/{condition_key}.npy"
        if (
            self.record.get("index") != condition_index
            or self.record.get("corruption") != corruption
            or self.record.get("severity") != severity
            or self.record.get("path") != expected_relative
            or self.record.get("dtype") != "little_endian_float32"
            or tuple(self.record.get("shape", ())) != (64, 3, 256, 256)
        ):
            raise StageC0MethodInputError("selected condition descriptor differs")
        files = _mapping(method.get("files"), "method image-shard ledger")
        if set(files) != {f"conditions/{key}.npy" for key in expected_keys}:
            raise StageC0MethodInputError("method image-shard ledger differs")
        file_record = _mapping(files.get(expected_relative), "selected image shard")
        if (
            file_record.get("bytes") != self.record.get("file_bytes")
            or file_record.get("sha256") != self.record.get("file_sha256")
        ):
            raise StageC0MethodInputError("condition descriptor/ledger differs")
        self.images = _open_verified_image_shard(
            self.cache_dir,
            relative=self.record.get("path"),
            expected_relative=expected_relative,
            file_record=file_record,
            expected_shape=(64, 3, 256, 256),
        )
        self.dataset_name = expected_dataset
        seed = method.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise StageC0MethodInputError("method seed must be an integer")
        self.seed = seed

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError(index)
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


__all__ = [
    "StageC0MethodInputDataset",
    "StageC0MethodInputError",
]
