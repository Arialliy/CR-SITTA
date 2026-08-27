from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dataio.corruption_cache import (
    CachedCorruptionDataset,
    TensorSequenceHasher,
    ordered_ids_sha256,
    sha256_file,
    verify_cache_artifact,
)


def _write_toy_cache(root: Path) -> dict:
    condition_dir = root / "conditions"
    condition_dir.mkdir(parents=True)
    image_ids = ["a", "b"]
    images = np.zeros((2, 3, 256, 256), dtype="<f4")
    images[0, :, 10, 20] = 1.25
    images[1, :, 30, 40] = -0.5
    targets = np.zeros((2, 1, 256, 256), dtype="<f4")
    targets[0, 0, 10, 20] = 1.0
    targets[1, 0, 30, 40] = 1.0
    image_path = condition_dir / "clean_S0.npy"
    target_path = root / "targets.npy"
    np.save(image_path, images, allow_pickle=False)
    np.save(target_path, targets, allow_pickle=False)
    image_hasher = TensorSequenceHasher()
    target_hasher = TensorSequenceHasher()
    for image_id, image, target in zip(image_ids, images, targets):
        image_hasher.update(image_id, image)
        target_hasher.update(image_id, target)
    files = {
        "conditions/clean_S0.npy": {
            "sha256": sha256_file(image_path),
            "bytes": image_path.stat().st_size,
        },
        "targets.npy": {
            "sha256": sha256_file(target_path),
            "bytes": target_path.stat().st_size,
        },
    }
    manifest = {
        "schema_version": 1,
        "cache_format": "nsfpn-materialized-corruption-cache-v1",
        "protocol_sha256": "a" * 64,
        "dataset": "toy",
        "seed": 42,
        "image_ids": image_ids,
        "ordered_ids_sha256": ordered_ids_sha256(image_ids),
        "original_sizes": [[256, 256], [256, 256]],
        "conditions": [
            {
                "key": "clean_S0",
                "corruption": "clean",
                "severity": 0,
                "path": "conditions/clean_S0.npy",
                "tensor_sequence_sha256": image_hasher.hexdigest(),
            }
        ],
        "targets": {
            "path": "targets.npy",
            "tensor_sequence_sha256": target_hasher.hexdigest(),
        },
        "files": files,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "COMPLETE.json").write_text(
        json.dumps({"complete": True, "manifest_sha256": sha256_file(manifest_path)}),
        encoding="utf-8",
    )
    return manifest


def test_cache_verification_and_private_tensor_copy(tmp_path: Path) -> None:
    manifest = _write_toy_cache(tmp_path)
    loaded, audit = verify_cache_artifact(
        tmp_path, expected_protocol_sha256="a" * 64
    )
    assert loaded["ordered_ids_sha256"] == manifest["ordered_ids_sha256"]
    assert audit["valid"] is True
    dataset = CachedCorruptionDataset(
        tmp_path, corruption="clean", severity=0, manifest=loaded
    )
    first = dataset[0]
    assert first["image"].dtype == torch.float32
    first["image"].fill_(99.0)
    assert float(dataset[0]["image"].max()) == pytest.approx(1.25)


def test_cache_tamper_fails_closed(tmp_path: Path) -> None:
    _write_toy_cache(tmp_path)
    path = tmp_path / "conditions" / "clean_S0.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="integrity failures"):
        verify_cache_artifact(tmp_path)


def test_tensor_sequence_hash_binds_ids_shape_dtype_and_bytes() -> None:
    value = np.zeros((3, 2, 2), dtype="<f4")
    first = TensorSequenceHasher()
    first.update("a", value)
    second = TensorSequenceHasher()
    second.update("b", value)
    third = TensorSequenceHasher()
    third.update("a", value.astype(np.float64))
    assert len(first.hexdigest()) == 64
    assert first.hexdigest() != second.hexdigest()
    assert first.hexdigest() != third.hexdigest()
