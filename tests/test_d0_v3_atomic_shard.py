from __future__ import annotations

from pathlib import Path

import pytest

from tta.d0_v3_atomic_shard import (
    D0V3AtomicShardError,
    publish_flat_directory_noreplace,
    snapshot_flat_directory,
)


def _staging(tmp_path: Path, payload: bytes = b"sealed") -> Path:
    path = tmp_path / ".cell.staging"
    path.mkdir(mode=0o700)
    (path / "payload.bin").write_bytes(payload)
    return path


def test_snapshot_flat_directory_rejects_extra_and_nonregular(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    with pytest.raises(D0V3AtomicShardError, match="member set differs"):
        snapshot_flat_directory(staging, expected_members=("other.bin",))
    (staging / "extra").mkdir()
    with pytest.raises(D0V3AtomicShardError, match="member set differs"):
        snapshot_flat_directory(staging, expected_members=("payload.bin",))


def test_guard_mutation_of_descendant_fails_before_publication(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    destination = tmp_path / "canonical"

    def mutating_verifier(path: Path) -> None:
        payload = path / "payload.bin"
        payload.chmod(0o644)
        payload.write_bytes(b"mutated")

    with pytest.raises(D0V3AtomicShardError, match="semantic verifier"):
        publish_flat_directory_noreplace(
            staging,
            destination,
            expected_members=("payload.bin",),
            semantic_verifier=mutating_verifier,
        )
    assert not destination.exists()
    assert staging.is_dir()


def test_snapshot_bound_publication_is_no_replace_and_read_only(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    destination = tmp_path / "canonical"
    calls: list[Path] = []

    def verifier(path: Path) -> None:
        assert (path / "payload.bin").read_bytes() == b"sealed"
        calls.append(path)

    published = publish_flat_directory_noreplace(
        staging,
        destination,
        expected_members=("payload.bin",),
        semantic_verifier=verifier,
    )
    assert published == destination
    assert not staging.exists()
    assert (destination / "payload.bin").read_bytes() == b"sealed"
    assert len(calls) == 2

    second = _staging(tmp_path, b"second")
    with pytest.raises(FileExistsError):
        publish_flat_directory_noreplace(
            second,
            destination,
            expected_members=("payload.bin",),
            semantic_verifier=lambda _path: None,
        )
    assert second.is_dir()
    assert (destination / "payload.bin").read_bytes() == b"sealed"
