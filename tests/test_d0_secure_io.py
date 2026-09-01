from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tta import d0_secure_io as secure_io
from tta.d0_secure_io import (
    SecureIOError,
    ensure_directory_chain_nofollow,
    fsync_directory,
    fsync_tree,
    publish_directory_noreplace,
    publish_file_noreplace,
    read_stable_regular_file,
    snapshot_regular_directory,
)


def test_read_stable_regular_file_binds_bytes_hash_and_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    payload = b'{"complete":true}\n'
    path.write_bytes(payload)

    snapshot = read_stable_regular_file(path)

    observed = path.stat()
    assert snapshot.path == path
    assert snapshot.data == payload
    assert snapshot.sha256 == hashlib.sha256(payload).hexdigest()
    assert snapshot.device == observed.st_dev
    assert snapshot.inode == observed.st_ino
    assert snapshot.size_bytes == len(payload)
    assert snapshot.mtime_ns == observed.st_mtime_ns


def test_stable_file_read_rejects_leaf_and_ancestor_symlinks(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    linked_parent = tmp_path / "linked-parent"
    real_parent.mkdir()
    payload = real_parent / "payload"
    payload.write_bytes(b"trusted")
    leaf_link = real_parent / "leaf-link"
    leaf_link.symlink_to(payload)
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="regular non-symlink"):
        read_stable_regular_file(leaf_link)
    with pytest.raises(ValueError, match="symlink|non-directory"):
        read_stable_regular_file(linked_parent / "payload")


def test_stable_file_read_detects_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large-records.jsonl"
    path.write_bytes(b"a" * (2 * 1024 * 1024))
    original_read = secure_io.os.read
    mutated = False

    def mutate_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        data = original_read(descriptor, count)
        if data and not mutated:
            mutated = True
            with path.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"b")
                stream.flush()
        return data

    monkeypatch.setattr(secure_io.os, "read", mutate_after_first_read)

    with pytest.raises(SecureIOError, match="changed while being read"):
        read_stable_regular_file(path)


def test_snapshot_regular_directory_captures_sorted_regular_members(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "summary.json").write_bytes(b"{}\n")
    (root / "COMPLETE.json").write_bytes(b'{"complete":true}\n')

    snapshot = snapshot_regular_directory(root)

    assert snapshot.path == root
    assert snapshot.member_names == ("COMPLETE.json", "summary.json")
    assert snapshot.member("summary.json").data == b"{}\n"
    assert snapshot.member("COMPLETE.json").sha256 == hashlib.sha256(
        b'{"complete":true}\n'
    ).hexdigest()
    with pytest.raises(KeyError):
        snapshot.member("missing")


def test_snapshot_regular_directory_detects_member_set_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "summary.json").write_bytes(b"{}\n")
    original_listdir = secure_io.os.listdir
    calls = 0

    def add_member_before_second_listing(path: int | str | os.PathLike[str]):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "late-member.json").write_bytes(b"{}\n")
        return original_listdir(path)

    monkeypatch.setattr(secure_io.os, "listdir", add_member_before_second_listing)

    with pytest.raises(SecureIOError, match="members changed while reading"):
        snapshot_regular_directory(root)


def test_snapshot_regular_directory_rejects_root_and_member_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    linked_root = tmp_path / "linked-artifact"
    target = tmp_path / "target"
    root.mkdir()
    target.write_bytes(b"target")
    (root / "member-link").symlink_to(target)
    linked_root.symlink_to(root, target_is_directory=True)

    with pytest.raises(ValueError, match="contains a symlink"):
        snapshot_regular_directory(root)
    with pytest.raises(ValueError, match="symlink|non-directory"):
        snapshot_regular_directory(linked_root)


@pytest.mark.parametrize("member_kind", ("directory", "fifo"))
def test_snapshot_regular_directory_rejects_nonregular_member(
    tmp_path: Path, member_kind: str
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    member = root / "invalid-member"
    if member_kind == "directory":
        member.mkdir()
    else:
        os.mkfifo(member)

    with pytest.raises(ValueError, match="members must be regular files"):
        snapshot_regular_directory(root)


def test_ensure_directory_chain_nofollow_creates_nested_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    root.mkdir()

    result = ensure_directory_chain_nofollow(
        root, ("cr_sitta", "tent_failure_diagnostics_v1", "IRSTD-1K")
    )

    expected = root / "cr_sitta" / "tent_failure_diagnostics_v1" / "IRSTD-1K"
    assert result == expected
    assert result.is_dir() and not result.is_symlink()
    assert all(not path.is_symlink() for path in (result, result.parent))


def test_ensure_directory_chain_nofollow_accepts_existing_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    existing = root / "cr_sitta" / "d0"
    existing.mkdir(parents=True)
    marker = existing / "marker"
    marker.write_bytes(b"keep")

    first = ensure_directory_chain_nofollow(root, ("cr_sitta", "d0"))
    second = ensure_directory_chain_nofollow(root, ())

    assert first == existing
    assert second == root
    assert marker.read_bytes() == b"keep"


def test_ensure_directory_chain_nofollow_rejects_symlink_component(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink directory"):
        ensure_directory_chain_nofollow(root, ("linked", "must-not-exist"))

    assert not (outside / "must-not-exist").exists()


@pytest.mark.parametrize("invalid", ("", ".", "..", "nested/name", "/", "bad\x00name"))
def test_ensure_directory_chain_nofollow_rejects_invalid_component_before_create(
    tmp_path: Path, invalid: str
) -> None:
    root = tmp_path / "results"
    root.mkdir()

    with pytest.raises(ValueError, match="simple names"):
        ensure_directory_chain_nofollow(root, ("would-have-been-created", invalid))

    assert not (root / "would-have-been-created").exists()


def test_publish_file_noreplace_moves_and_fsyncs_regular_file(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "receipt.tmp"
    destination = tmp_path / "receipt.json"
    staging.write_bytes(b'{"complete":true}\n')

    published = publish_file_noreplace(staging, destination)

    assert published == destination
    assert destination.read_bytes() == b'{"complete":true}\n'
    assert destination.is_file() and not destination.is_symlink()
    assert not staging.exists()


def test_publish_file_runs_guard_at_pre_rename_boundary(tmp_path: Path) -> None:
    staging = tmp_path / "receipt.tmp"
    destination = tmp_path / "receipt.json"
    staging.write_bytes(b'{"complete":true}\n')
    observed: list[str] = []

    def guard() -> None:
        assert staging.is_file()
        assert not destination.exists()
        observed.append("guarded")

    publish_file_noreplace(
        staging,
        destination,
        pre_rename_guard=guard,
    )
    assert observed == ["guarded"]
    assert destination.is_file()


def test_publish_file_guard_failure_prevents_publish(tmp_path: Path) -> None:
    staging = tmp_path / "receipt.tmp"
    destination = tmp_path / "receipt.json"
    staging.write_bytes(b'{"complete":true}\n')

    def guard() -> None:
        raise RuntimeError("receipt validation changed")

    with pytest.raises(RuntimeError, match="receipt validation changed"):
        publish_file_noreplace(
            staging,
            destination,
            pre_rename_guard=guard,
        )
    assert staging.is_file()
    assert not destination.exists()


def test_file_postrename_failure_rolls_back_exact_inode_and_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "receipt.tmp"
    destination = tmp_path / "receipt.json"
    staging.write_bytes(b"durable")
    original = secure_io._fsync_renamed_parents
    calls = 0

    def fail_first(source_fd: int, destination_fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic post-rename fsync failure")
        original(source_fd, destination_fd)

    monkeypatch.setattr(secure_io, "_fsync_renamed_parents", fail_first)
    with pytest.raises(OSError, match="synthetic"):
        publish_file_noreplace(staging, destination)
    assert not destination.exists()
    assert staging.read_bytes() == b"durable"

    publish_file_noreplace(staging, destination)
    assert destination.read_bytes() == b"durable"
    assert not staging.exists()


def test_file_destination_is_never_overwritten_and_staging_survives(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "second.tmp"
    destination = tmp_path / "receipt.json"
    staging.write_bytes(b"second")
    destination.write_bytes(b"first")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        publish_file_noreplace(staging, destination)

    assert destination.read_bytes() == b"first"
    assert staging.read_bytes() == b"second"


def test_two_sequential_publishers_simulate_destination_appearance_race(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.tmp"
    second = tmp_path / "second.tmp"
    destination = tmp_path / "winner.json"
    first.write_bytes(b"winner")
    second.write_bytes(b"loser")

    publish_file_noreplace(first, destination)
    with pytest.raises(FileExistsError):
        publish_file_noreplace(second, destination)

    assert destination.read_bytes() == b"winner"
    assert second.read_bytes() == b"loser"


def test_existing_destination_symlink_is_not_replaced(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    destination = tmp_path / "published.json"
    staging = tmp_path / "staging.tmp"
    target.write_bytes(b"target")
    destination.symlink_to(target)
    staging.write_bytes(b"new")

    with pytest.raises(FileExistsError):
        publish_file_noreplace(staging, destination)

    assert destination.is_symlink()
    assert target.read_bytes() == b"target"
    assert staging.read_bytes() == b"new"


def test_publish_file_rejects_symlink_and_nonregular_staging(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.tmp"
    linked = tmp_path / "linked.tmp"
    fifo = tmp_path / "fifo.tmp"
    real.write_bytes(b"payload")
    linked.symlink_to(real)
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular non-symlink"):
        publish_file_noreplace(linked, tmp_path / "from-link")
    with pytest.raises(ValueError, match="regular non-symlink"):
        publish_file_noreplace(fifo, tmp_path / "from-fifo")

    assert not (tmp_path / "from-link").exists()
    assert not (tmp_path / "from-fifo").exists()
    assert real.read_bytes() == b"payload"


def test_publish_directory_noreplace_moves_nested_fsynced_tree(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "artifact.build"
    nested = staging / "nested"
    nested.mkdir(parents=True)
    (staging / "manifest.json").write_bytes(b"{}\n")
    (nested / "records.jsonl").write_bytes(b'{"id":1}\n')
    destination = tmp_path / "artifact"

    published = publish_directory_noreplace(staging, destination)

    assert published == destination
    assert not staging.exists()
    assert (destination / "manifest.json").read_bytes() == b"{}\n"
    assert (destination / "nested" / "records.jsonl").read_bytes() == b'{"id":1}\n'


def test_directory_postrename_failure_rolls_back_exact_tree_and_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "artifact.build"
    destination = tmp_path / "artifact"
    staging.mkdir()
    (staging / "payload").write_bytes(b"durable")
    original = secure_io._fsync_renamed_parents
    calls = 0

    def fail_first(source_fd: int, destination_fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic post-rename fsync failure")
        original(source_fd, destination_fd)

    monkeypatch.setattr(secure_io, "_fsync_renamed_parents", fail_first)
    with pytest.raises(OSError, match="synthetic"):
        publish_directory_noreplace(staging, destination)
    assert not destination.exists()
    assert (staging / "payload").read_bytes() == b"durable"

    publish_directory_noreplace(staging, destination)
    assert (destination / "payload").read_bytes() == b"durable"
    assert not staging.exists()


def test_publish_directory_runs_guard_at_pre_rename_boundary(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "payload.txt").write_text("payload", encoding="utf-8")
    destination = tmp_path / "published"
    observed: list[str] = []

    def guard() -> None:
        assert staging.is_dir()
        assert not destination.exists()
        observed.append("guarded")

    publish_directory_noreplace(
        staging,
        destination,
        pre_rename_guard=guard,
    )
    assert observed == ["guarded"]
    assert destination.is_dir()


def test_publish_directory_guard_failure_prevents_publish(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "payload.txt").write_text("payload", encoding="utf-8")
    destination = tmp_path / "published"

    def guard() -> None:
        raise RuntimeError("lineage changed")

    with pytest.raises(RuntimeError, match="lineage changed"):
        publish_directory_noreplace(
            staging,
            destination,
            pre_rename_guard=guard,
        )
    assert staging.is_dir()
    assert not destination.exists()


def test_directory_destination_is_never_overwritten(tmp_path: Path) -> None:
    staging = tmp_path / "second.build"
    destination = tmp_path / "artifact"
    staging.mkdir()
    destination.mkdir()
    (staging / "value").write_bytes(b"second")
    (destination / "value").write_bytes(b"first")

    with pytest.raises(FileExistsError):
        publish_directory_noreplace(staging, destination)

    assert (destination / "value").read_bytes() == b"first"
    assert (staging / "value").read_bytes() == b"second"


def test_publish_directory_rejects_leaf_and_nested_symlinks(
    tmp_path: Path,
) -> None:
    real_tree = tmp_path / "real-tree"
    linked_tree = tmp_path / "linked-tree"
    real_tree.mkdir()
    (real_tree / "payload").write_bytes(b"ok")
    linked_tree.symlink_to(real_tree, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink directory"):
        publish_directory_noreplace(linked_tree, tmp_path / "leaf-output")

    staging = tmp_path / "nested-link.build"
    staging.mkdir()
    (staging / "payload-link").symlink_to(real_tree / "payload")
    with pytest.raises(ValueError, match="contains a symlink"):
        publish_directory_noreplace(staging, tmp_path / "nested-output")

    assert not (tmp_path / "leaf-output").exists()
    assert not (tmp_path / "nested-output").exists()


def test_publish_directory_rejects_nonregular_member(tmp_path: Path) -> None:
    staging = tmp_path / "fifo-tree.build"
    staging.mkdir()
    os.mkfifo(staging / "payload.fifo")

    with pytest.raises(ValueError, match="regular file or directory"):
        publish_directory_noreplace(staging, tmp_path / "fifo-output")

    assert not (tmp_path / "fifo-output").exists()


def test_fsync_helpers_accept_real_tree_and_reject_symlink_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "payload").write_bytes(b"durable")

    fsync_tree(root)
    fsync_directory(root)

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|non-directory"):
        fsync_tree(linked_root)
    with pytest.raises(ValueError, match="symlink|non-directory"):
        fsync_directory(linked_root)


def test_destination_parent_symlink_is_rejected_before_publish(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging.tmp"
    real_parent = tmp_path / "real-parent"
    linked_parent = tmp_path / "linked-parent"
    staging.write_bytes(b"payload")
    real_parent.mkdir()
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|non-directory"):
        publish_file_noreplace(staging, linked_parent / "output")

    assert staging.read_bytes() == b"payload"
    assert not (real_parent / "output").exists()
