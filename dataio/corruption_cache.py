"""Immutable materialized inputs shared by every corruption-benchmark method."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from corruptions.corruption_protocol import validate_corruption_request


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(image_ids: Sequence[str]) -> str:
    encoded = json.dumps(
        list(image_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def condition_key(corruption: str, severity: int) -> str:
    corruption, severity = validate_corruption_request(corruption, severity)
    return f"{corruption}_S{severity}"


class TensorSequenceHasher:
    """Hash ordered ``(image_id, dtype, shape, exact bytes)`` tensor records."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.count = 0

    def update(self, image_id: str, value: Any) -> None:
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
        else:
            array = np.ascontiguousarray(np.asarray(value))
        header = json.dumps(
            {
                "image_id": str(image_id),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = array.tobytes(order="C")
        self._digest.update(struct.pack(">Q", len(header)))
        self._digest.update(header)
        self._digest.update(struct.pack(">Q", len(raw)))
        self._digest.update(raw)
        self.count += 1

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return loaded


def verify_cache_artifact(
    cache_dir: str | Path,
    *,
    expected_protocol_sha256: str | None = None,
    verify_file_hashes: bool = True,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    root = Path(cache_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(f"cache is incomplete: {root}")
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    manifest_hash = sha256_file(manifest_path)
    if complete.get("complete") is not True:
        raise ValueError("cache completion sentinel is not true")
    if complete.get("manifest_sha256") != manifest_hash:
        raise ValueError("cache manifest hash does not match completion sentinel")
    if expected_protocol_sha256 is not None:
        if manifest.get("protocol_sha256") != expected_protocol_sha256:
            raise ValueError("cache protocol SHA256 does not match the consumer protocol")
    if manifest.get("ordered_ids_sha256") != ordered_ids_sha256(manifest["image_ids"]):
        raise ValueError("cache ordered ID hash is invalid")

    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("cache manifest contains no files")
    mismatches = []
    total_bytes = 0
    for relative_path, record in files.items():
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe cache manifest path: {relative_path}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"cache file is missing: {path}")
        size = path.stat().st_size
        total_bytes += size
        if size != int(record["bytes"]):
            mismatches.append({"path": relative_path, "reason": "size"})
        if verify_file_hashes and sha256_file(path) != record["sha256"]:
            mismatches.append({"path": relative_path, "reason": "sha256"})
    if mismatches:
        raise ValueError(f"cache file integrity failures: {mismatches}")
    return manifest, {
        "valid": True,
        "manifest_sha256": manifest_hash,
        "verified_file_count": len(files),
        "verified_total_bytes": total_bytes,
        "file_hashes_verified": verify_file_hashes,
    }


class CachedCorruptionDataset(Dataset[dict[str, Any]]):
    """Read one immutable condition without regenerating its corruption."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        corruption: str,
        severity: int,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(cache_dir).expanduser().resolve()
        self.manifest = manifest or _load_json(self.root / "manifest.json")
        self.corruption, self.severity = validate_corruption_request(
            corruption, severity
        )
        key = condition_key(self.corruption, self.severity)
        conditions = {
            str(record["key"]): record for record in self.manifest["conditions"]
        }
        if key not in conditions:
            raise ValueError(f"condition {key} is absent from cache")
        self.condition = conditions[key]
        self.image_ids = tuple(str(value) for value in self.manifest["image_ids"])
        self.original_sizes = tuple(
            tuple(int(part) for part in value)
            for value in self.manifest["original_sizes"]
        )
        self.dataset_name = str(self.manifest["dataset"])
        self.seed = int(self.manifest["seed"])
        self.images = np.load(
            self.root / self.condition["path"], mmap_mode="r", allow_pickle=False
        )
        targets = self.manifest["targets"]
        self.targets = np.load(
            self.root / targets["path"], mmap_mode="r", allow_pickle=False
        )
        expected_images_shape = (len(self.image_ids), 3, 256, 256)
        expected_targets_shape = (len(self.image_ids), 1, 256, 256)
        if (
            self.images.shape != expected_images_shape
            or self.images.dtype.str != "<f4"
            or not self.images.flags.c_contiguous
        ):
            raise ValueError(
                f"cached image tensor contract drift: {self.images.shape}/{self.images.dtype}"
            )
        if (
            self.targets.shape != expected_targets_shape
            or self.targets.dtype.str != "<f4"
            or not self.targets.flags.c_contiguous
        ):
            raise ValueError(
                f"cached target tensor contract drift: {self.targets.shape}/{self.targets.dtype}"
            )
        if len(self.original_sizes) != len(self.image_ids):
            raise ValueError("cache original_sizes length does not match image IDs")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # A private writable copy prevents accidental writes through a read-only
        # NumPy memmap while retaining exact float32 bytes.
        image = torch.from_numpy(np.array(self.images[index], copy=True))
        mask = torch.from_numpy(np.array(self.targets[index], copy=True))
        if not torch.isfinite(image).all() or not torch.isfinite(mask).all():
            raise ValueError("cached image or target contains NaN/Inf")
        return {
            "image": image,
            "mask": mask,
            "image_id": self.image_ids[index],
            "original_size": self.original_sizes[index],
            "dataset": self.dataset_name,
            "corruption": self.corruption,
            "severity": self.severity,
            "seed": self.seed,
        }


__all__ = [
    "CachedCorruptionDataset",
    "TensorSequenceHasher",
    "condition_key",
    "ordered_ids_sha256",
    "sha256_file",
    "verify_cache_artifact",
]
