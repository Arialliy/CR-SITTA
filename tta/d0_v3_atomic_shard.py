"""Snapshot-bound publication for flat D0-v3 formal evidence shards.

The sealed :mod:`tta.d0_secure_io` directory primitive guarantees an atomic
``RENAME_NOREPLACE`` but, intentionally, knows nothing about the semantic
payload of a directory.  This additive wrapper binds every regular member
before and after the caller's semantic verifier, again at the narrow
pre-rename boundary, and once more at the canonical name.  It is restricted
to flat directories so the exact member set and every descendant inode can be
checked without path traversal ambiguity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Final

from tta.d0_secure_io import publish_directory_noreplace


_READ_FLAGS: Final = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_DIR_FLAGS: Final = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY


class D0V3AtomicShardError(RuntimeError):
    """A flat formal shard changed or could not be published safely."""


@dataclass(frozen=True, slots=True)
class FlatMemberSnapshot:
    name: str
    sha256: str
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class FlatDirectorySnapshot:
    device: int
    inode: int
    mode: int
    link_count: int
    mtime_ns: int
    ctime_ns: int
    members: tuple[FlatMemberSnapshot, ...]

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(value.name for value in self.members)


def _same_payload(
    left: FlatDirectorySnapshot, right: FlatDirectorySnapshot
) -> bool:
    """Compare stable payload identity while allowing rename-induced root ctime.

    Linux updates a directory inode's ctime when its name is moved.  The inode,
    mode, link count and every descendant identity/hash remain stable, which
    are the properties that bind the published payload.
    """

    return (
        left.device,
        left.inode,
        left.mode,
        left.link_count,
        left.members,
    ) == (
        right.device,
        right.inode,
        right.mode,
        right.link_count,
        right.members,
    )


def _simple_names(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise D0V3AtomicShardError("expected_members must be a sequence")
    names = tuple(value)
    if (
        not names
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\x00" in name
            for name in names
        )
    ):
        raise D0V3AtomicShardError(
            "expected_members must contain unique simple file names"
        )
    return tuple(sorted(names))


def _open_directory_nofollow(path: Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open("/", _DIR_FLAGS)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_fd(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 4 * 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def snapshot_flat_directory(
    path: str | os.PathLike[str],
    *,
    expected_members: Sequence[str],
) -> FlatDirectorySnapshot:
    """Stream-hash one exact flat directory without following symlinks."""

    expected = _simple_names(expected_members)
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        directory_fd = _open_directory_nofollow(absolute)
    except (OSError, ValueError) as exc:
        raise D0V3AtomicShardError(
            f"cannot securely open flat shard directory: {absolute}"
        ) from exc
    try:
        root_before = os.fstat(directory_fd)
        if not stat.S_ISDIR(root_before.st_mode):  # pragma: no cover - O_DIRECTORY
            raise D0V3AtomicShardError("flat shard root is not a directory")
        observed = tuple(sorted(os.listdir(directory_fd)))
        if observed != expected:
            raise D0V3AtomicShardError(
                "flat shard member set differs; "
                f"expected={list(expected)}, observed={list(observed)}"
            )
        members: list[FlatMemberSnapshot] = []
        for name in expected:
            lexical = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(lexical.st_mode):
                raise D0V3AtomicShardError(
                    f"flat shard member is not a regular file: {name}"
                )
            descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
            try:
                before = os.fstat(descriptor)
                if _identity(before) != _identity(lexical):
                    raise D0V3AtomicShardError(
                        f"flat shard member changed while opening: {name}"
                    )
                digest = _hash_fd(descriptor)
                after = os.fstat(descriptor)
                if _identity(before) != _identity(after):
                    raise D0V3AtomicShardError(
                        f"flat shard member changed while hashing: {name}"
                    )
                members.append(
                    FlatMemberSnapshot(
                        name=name,
                        sha256=digest,
                        device=before.st_dev,
                        inode=before.st_ino,
                        mode=before.st_mode,
                        link_count=before.st_nlink,
                        size_bytes=before.st_size,
                        mtime_ns=before.st_mtime_ns,
                        ctime_ns=before.st_ctime_ns,
                    )
                )
            finally:
                os.close(descriptor)
        root_after = os.fstat(directory_fd)
        if _identity(root_before) != _identity(root_after):
            raise D0V3AtomicShardError(
                "flat shard root changed while members were hashed"
            )
        if tuple(sorted(os.listdir(directory_fd))) != expected:
            raise D0V3AtomicShardError("flat shard member set changed while hashing")
        return FlatDirectorySnapshot(
            device=root_before.st_dev,
            inode=root_before.st_ino,
            mode=root_before.st_mode,
            link_count=root_before.st_nlink,
            mtime_ns=root_before.st_mtime_ns,
            ctime_ns=root_before.st_ctime_ns,
            members=tuple(members),
        )
    finally:
        os.close(directory_fd)


def _freeze_flat_directory(path: Path, expected_members: tuple[str, ...]) -> None:
    """Make completed evidence read-only before taking the publication seal."""

    directory_fd = _open_directory_nofollow(path)
    try:
        if tuple(sorted(os.listdir(directory_fd))) != expected_members:
            raise D0V3AtomicShardError("cannot freeze a drifting member set")
        for name in expected_members:
            value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(value.st_mode):
                raise D0V3AtomicShardError(
                    f"cannot freeze non-regular member: {name}"
                )
            os.chmod(name, 0o444, dir_fd=directory_fd, follow_symlinks=False)
            descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.chmod(path, 0o555, follow_symlinks=False)


def publish_flat_directory_noreplace(
    staging: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    expected_members: Sequence[str],
    semantic_verifier: Callable[[Path], object],
) -> Path:
    """Verify, snapshot-bind and atomically publish one immutable flat shard.

    The semantic verifier is deliberately run *between* two complete
    descendant snapshots.  A verifier (or injected test hook) that changes a
    byte, inode, timestamp, permission or member name therefore fails before
    the canonical rename.
    """

    if not callable(semantic_verifier):
        raise D0V3AtomicShardError("semantic_verifier must be callable")
    source = Path(os.path.abspath(os.fspath(staging)))
    target = Path(os.path.abspath(os.fspath(destination)))
    members = _simple_names(expected_members)
    if source == target:
        raise D0V3AtomicShardError("staging and destination must differ")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"formal shard destination already exists: {target}")

    _freeze_flat_directory(source, members)
    sealed = snapshot_flat_directory(source, expected_members=members)
    semantic_verifier(source)
    if not _same_payload(
        snapshot_flat_directory(source, expected_members=members), sealed
    ):
        raise D0V3AtomicShardError(
            "flat shard changed while the semantic verifier ran"
        )

    def immediate_guard() -> None:
        if not _same_payload(
            snapshot_flat_directory(source, expected_members=members), sealed
        ):
            raise D0V3AtomicShardError(
                "flat shard changed at the pre-rename publication boundary"
            )

    published = publish_directory_noreplace(
        source, target, pre_rename_guard=immediate_guard
    )
    try:
        canonical = snapshot_flat_directory(published, expected_members=members)
        if not _same_payload(canonical, sealed):
            raise D0V3AtomicShardError(
                "canonical shard bytes/identity differ after publication"
            )
        semantic_verifier(published)
        if not _same_payload(
            snapshot_flat_directory(published, expected_members=members), sealed
        ):
            raise D0V3AtomicShardError(
                "canonical shard changed while post-publication verification ran"
            )
    except BaseException as publication_error:
        # Preserve a recoverable private name and remove the canonical name.
        # The reverse operation is itself no-replace and refuses to overwrite a
        # staging name that appeared concurrently.
        try:
            publish_directory_noreplace(published, source)
        except BaseException as rollback_error:
            raise D0V3AtomicShardError(
                "formal shard post-publication verification failed and exact "
                "rollback did not complete"
            ) from rollback_error
        raise publication_error
    return published


__all__ = [
    "D0V3AtomicShardError",
    "FlatDirectorySnapshot",
    "FlatMemberSnapshot",
    "publish_flat_directory_noreplace",
    "snapshot_flat_directory",
]
