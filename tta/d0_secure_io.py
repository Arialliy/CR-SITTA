"""Fail-closed durable publication primitives for D0 diagnostic artifacts.

The helpers in this module intentionally target Linux.  Publication uses the
kernel's ``renameat2(RENAME_NOREPLACE)`` operation, so a destination that
appears between validation and publication is never overwritten.  Path
components are opened one by one with ``O_NOFOLLOW`` and staging trees reject
symlinks and non-regular members.

Callers are responsible for creating a private staging file or directory and
an existing destination parent.  On success the staging name has moved to the
destination and the relevant parent directory entries have been fsynced.
"""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import stat
from typing import Callable, Final


_RENAME_NOREPLACE: Final = 1
_DIRECTORY_OPEN_FLAGS: Final = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
)
_FILE_OPEN_FLAGS: Final = (
    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
)


class SecureIOError(RuntimeError):
    """Raised when a secure publication invariant cannot be established."""


@dataclass(frozen=True, slots=True)
class StableFileSnapshot:
    """Bytes and filesystem identity captured by one stable no-follow read."""

    path: Path
    data: bytes
    sha256: str
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class StableDirectorySnapshot:
    """Stable identity and sorted regular-file snapshots for one directory."""

    path: Path
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    members: tuple[StableFileSnapshot, ...]

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(member.path.name for member in self.members)

    def member(self, name: str) -> StableFileSnapshot:
        """Return one member by its simple name or raise ``KeyError``."""

        for value in self.members:
            if value.path.name == name:
                return value
        raise KeyError(name)


def _absolute_lexical_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute path without resolving or following symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    """Open ``path`` component-by-component without following symlinks."""

    absolute = _absolute_lexical_path(path)
    descriptor = os.open("/", _DIRECTORY_OPEN_FLAGS)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        f"{label} contains a symlink or non-directory component: "
                        f"{absolute}"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_parent_nofollow(path: Path, *, label: str) -> tuple[int, str]:
    absolute = _absolute_lexical_path(path)
    if absolute == Path("/") or not absolute.name:
        raise ValueError(f"{label} cannot be the filesystem root: {absolute}")
    descriptor = _open_directory_nofollow(absolute.parent, label=f"{label} parent")
    return descriptor, absolute.name


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _snapshot_identity(snapshot: StableFileSnapshot) -> tuple[int, ...]:
    return (
        snapshot.device,
        snapshot.inode,
        snapshot.mode,
        snapshot.link_count,
        snapshot.size_bytes,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )


def _open_regular_file_at(parent_fd: int, name: str, *, label: str) -> int:
    try:
        descriptor = os.open(name, _FILE_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"{label} must be a regular non-symlink file") from error
        raise
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_at(parent_fd: int, name: str, *, label: str) -> int:
    try:
        return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"{label} must be a real non-symlink directory") from error
        raise


def _read_stable_regular_fd(
    descriptor: int,
    *,
    display_path: Path,
) -> StableFileSnapshot:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"snapshot member must be a regular file: {display_path}")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if _stable_identity(before) != _stable_identity(after):
        raise SecureIOError(f"file changed while being read: {display_path}")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise SecureIOError(f"file size changed while being read: {display_path}")
    return StableFileSnapshot(
        path=display_path,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        mode=before.st_mode,
        link_count=before.st_nlink,
        size_bytes=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
    )


def read_stable_regular_file(
    path: str | os.PathLike[str],
) -> StableFileSnapshot:
    """Read one file without following any ancestor or leaf symlink.

    The full inode/size/time identity must agree before and after the read.  The
    lexical path is then independently reopened and required to identify the
    same inode, closing the common ``Path.resolve()`` verification gap.
    """

    absolute = _absolute_lexical_path(path)
    parent_fd, name = _open_parent_nofollow(absolute, label="snapshot file")
    try:
        descriptor = _open_regular_file_at(
            parent_fd,
            name,
            label=f"snapshot file {absolute}",
        )
        try:
            snapshot = _read_stable_regular_fd(
                descriptor,
                display_path=absolute,
            )
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if _stable_identity(current) != _snapshot_identity(snapshot):
                raise SecureIOError(
                    f"file name changed while being read: {absolute}"
                )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)

    verification_parent_fd, verification_name = _open_parent_nofollow(
        absolute, label="snapshot verification file"
    )
    try:
        verification_fd = _open_regular_file_at(
            verification_parent_fd,
            verification_name,
            label=f"snapshot verification file {absolute}",
        )
        try:
            if _stable_identity(os.fstat(verification_fd)) != _snapshot_identity(
                snapshot
            ):
                raise SecureIOError(
                    f"file path changed after stable read: {absolute}"
                )
        finally:
            os.close(verification_fd)
    finally:
        os.close(verification_parent_fd)
    return snapshot


def snapshot_regular_directory(
    root: str | os.PathLike[str],
) -> StableDirectorySnapshot:
    """Snapshot a flat directory containing only regular non-symlink files.

    The root is opened component-by-component with ``O_NOFOLLOW``.  Member
    names and root identity must remain unchanged for the complete operation;
    each member is bound to its inode, size, timestamps, bytes, and SHA-256.
    """

    absolute = _absolute_lexical_path(root)
    descriptor = _open_directory_nofollow(absolute, label="snapshot directory")
    try:
        before = os.fstat(descriptor)
        names_before = tuple(sorted(os.listdir(descriptor)))
        members: list[StableFileSnapshot] = []
        for name in names_before:
            member_path = absolute / name
            initial = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(initial.st_mode):
                if stat.S_ISLNK(initial.st_mode):
                    raise ValueError(
                        f"snapshot directory contains a symlink: {member_path}"
                    )
                raise ValueError(
                    "snapshot directory members must be regular files: "
                    f"{member_path}"
                )
            member_fd = _open_regular_file_at(
                descriptor,
                name,
                label=f"snapshot member {member_path}",
            )
            try:
                if not _same_inode(initial, os.fstat(member_fd)):
                    raise SecureIOError(
                        f"snapshot member changed while opening: {member_path}"
                    )
                member_snapshot = _read_stable_regular_fd(
                    member_fd,
                    display_path=member_path,
                )
            finally:
                os.close(member_fd)
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if _stable_identity(current) != _snapshot_identity(member_snapshot):
                raise SecureIOError(
                    f"snapshot member name changed while reading: {member_path}"
                )
            members.append(member_snapshot)

        if tuple(sorted(os.listdir(descriptor))) != names_before:
            raise SecureIOError(
                f"snapshot directory members changed while reading: {absolute}"
            )
        after = os.fstat(descriptor)
        if _stable_identity(before) != _stable_identity(after):
            raise SecureIOError(
                f"snapshot directory changed while reading: {absolute}"
            )

        verification_fd = _open_directory_nofollow(
            absolute, label="snapshot directory verification"
        )
        try:
            if _stable_identity(os.fstat(verification_fd)) != _stable_identity(after):
                raise SecureIOError(
                    f"snapshot directory path changed after reading: {absolute}"
                )
        finally:
            os.close(verification_fd)
    finally:
        os.close(descriptor)

    return StableDirectorySnapshot(
        path=absolute,
        device=after.st_dev,
        inode=after.st_ino,
        mode=after.st_mode,
        link_count=after.st_nlink,
        size_bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
        members=tuple(members),
    )


def _rename_noreplace_at(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    *,
    destination: Path,
) -> None:
    """Invoke Linux ``renameat2`` with ``RENAME_NOREPLACE``."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SecureIOError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent_fd,
        os.fsencode(source_name),
        destination_parent_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"destination exists; refusing overwrite: {destination}",
            str(destination),
        )
    if error_number in {errno.ENOSYS, errno.EOPNOTSUPP}:
        raise SecureIOError(
            "atomic no-replace publication is unsupported by this filesystem/kernel"
        )
    if error_number == errno.EINVAL:
        raise SecureIOError(
            "renameat2(RENAME_NOREPLACE) is unsupported or the move is invalid"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _fsync_renamed_parents(source_parent_fd: int, destination_parent_fd: int) -> None:
    """Durably record both sides of a rename, avoiding a redundant fsync."""

    destination_identity = os.fstat(destination_parent_fd)
    source_identity = os.fstat(source_parent_fd)
    os.fsync(destination_parent_fd)
    if not _same_inode(source_identity, destination_identity):
        os.fsync(source_parent_fd)


def _rollback_exact_published_name(
    *,
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    expected_identity: os.stat_result,
    source_display: Path,
    destination_display: Path,
) -> None:
    """Move only the exact just-published inode back to its private name.

    This is used when a post-rename identity or durability check fails.  It
    never removes or moves a destination that no longer names the staging
    inode, preventing a rollback race from touching someone else's artifact.
    """

    try:
        current = os.stat(
            destination_name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not _same_inode(current, expected_identity):
        raise SecureIOError(
            "cannot safely roll back publication because the canonical name no "
            f"longer identifies staging: {destination_display}"
        )
    _rename_noreplace_at(
        destination_parent_fd,
        destination_name,
        source_parent_fd,
        source_name,
        destination=source_display,
    )
    restored = os.stat(
        source_name, dir_fd=source_parent_fd, follow_symlinks=False
    )
    if not _same_inode(restored, expected_identity):
        raise SecureIOError(
            f"publication rollback identity differs: {source_display}"
        )
    try:
        _fsync_renamed_parents(destination_parent_fd, source_parent_fd)
    except BaseException as exc:
        # The canonical name is already absent.  Surface the durability
        # uncertainty while preserving the private staging path for cleanup.
        raise SecureIOError(
            "canonical publication was rolled back, but rollback directory "
            f"fsync failed: {destination_display}"
        ) from exc


def fsync_directory(path: str | os.PathLike[str]) -> None:
    """Fsync one real directory reached without following symlinks."""

    absolute = _absolute_lexical_path(path)
    descriptor = _open_directory_nofollow(absolute, label="directory")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validated_relative_parts(relative_parts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(relative_parts, (str, bytes, os.PathLike)):
        raise ValueError("relative_parts must be a sequence of simple components")
    parts = tuple(relative_parts)
    for component in parts:
        if (
            not isinstance(component, str)
            or not component
            or component in {".", ".."}
            or "/" in component
            or "\x00" in component
        ):
            raise ValueError(
                "directory components must be non-empty simple names without "
                "'.', '..', '/', or NUL"
            )
    return parts


def ensure_directory_chain_nofollow(
    root: str | os.PathLike[str],
    relative_parts: Sequence[str],
) -> Path:
    """Securely ensure a directory chain below an existing real ``root``.

    Every component is created with ``mkdirat`` relative to an already-opened
    directory descriptor and then opened with ``O_NOFOLLOW``.  A parent is
    fsynced immediately after each successful creation.  Existing real
    directories are accepted; symlinks and non-directory components fail
    closed.  All components are validated before the first filesystem change.
    """

    absolute_root = _absolute_lexical_path(root)
    parts = _validated_relative_parts(relative_parts)
    final_path = absolute_root.joinpath(*parts)
    current_fd = _open_directory_nofollow(absolute_root, label="directory root")
    try:
        for component in parts:
            try:
                os.mkdir(component, mode=0o755, dir_fd=current_fd)
                os.fsync(current_fd)
            except FileExistsError:
                pass
            child_fd = _open_directory_at(
                current_fd,
                component,
                label=f"directory component {final_path}",
            )
            os.close(current_fd)
            current_fd = child_fd

        # Reopen the lexical result independently and require it still names
        # the same directory held by the descriptor chain.  This catches a
        # parent/name replacement before returning a misleading path.
        verification_fd = _open_directory_nofollow(
            final_path, label="ensured directory chain"
        )
        try:
            if not _same_inode(os.fstat(current_fd), os.fstat(verification_fd)):
                raise SecureIOError(
                    f"directory chain changed while being created: {final_path}"
                )
        finally:
            os.close(verification_fd)
    finally:
        os.close(current_fd)
    return final_path


def _fsync_tree_fd(directory_fd: int, *, display_path: Path) -> os.stat_result:
    """Fsync a tree bottom-up through already-secured directory descriptors."""

    before = os.fstat(directory_fd)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"tree root must be a directory: {display_path}")
    names_before = sorted(os.listdir(directory_fd))
    for name in names_before:
        member_path = display_path / name
        member = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(member.st_mode):
            raise ValueError(f"staging tree contains a symlink: {member_path}")
        if stat.S_ISREG(member.st_mode):
            file_fd = _open_regular_file_at(
                directory_fd,
                name,
                label=f"staging member {member_path}",
            )
            try:
                opened = os.fstat(file_fd)
                if not _same_inode(member, opened):
                    raise SecureIOError(
                        f"staging member changed while opening: {member_path}"
                    )
                identity_before = _stable_identity(opened)
                os.fsync(file_fd)
                if _stable_identity(os.fstat(file_fd)) != identity_before:
                    raise SecureIOError(
                        f"staging member changed while being fsynced: {member_path}"
                    )
            finally:
                os.close(file_fd)
            continue
        if stat.S_ISDIR(member.st_mode):
            child_fd = _open_directory_at(
                directory_fd,
                name,
                label=f"staging member {member_path}",
            )
            try:
                opened = os.fstat(child_fd)
                if not _same_inode(member, opened):
                    raise SecureIOError(
                        f"staging directory changed while opening: {member_path}"
                    )
                _fsync_tree_fd(child_fd, display_path=member_path)
            finally:
                os.close(child_fd)
            continue
        raise ValueError(
            f"staging tree member must be a regular file or directory: {member_path}"
        )

    if sorted(os.listdir(directory_fd)) != names_before:
        raise SecureIOError(f"staging tree changed while being fsynced: {display_path}")
    os.fsync(directory_fd)
    after = os.fstat(directory_fd)
    if _stable_identity(before) != _stable_identity(after):
        raise SecureIOError(f"staging directory changed while being fsynced: {display_path}")
    return after


def fsync_tree(root: str | os.PathLike[str]) -> None:
    """Fsync every regular file and directory in ``root`` bottom-up.

    The root and every descendant must be a real directory or regular file.
    Symlinks, sockets, devices, and FIFOs are rejected without being followed.
    """

    absolute = _absolute_lexical_path(root)
    descriptor = _open_directory_nofollow(absolute, label="tree root")
    try:
        _fsync_tree_fd(descriptor, display_path=absolute)
    finally:
        os.close(descriptor)


def publish_file_noreplace(
    temporary: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    pre_rename_guard: Callable[[], None] | None = None,
) -> Path:
    """Durably move one regular staging file without replacing a destination.

    ``pre_rename_guard`` runs after the staging file has been fsynced and
    validated, immediately before the no-replace rename.  It lets callers put
    semantic validation at the narrowest publication boundary.
    """

    source = _absolute_lexical_path(temporary)
    target = _absolute_lexical_path(destination)
    source_parent_fd, source_name = _open_parent_nofollow(
        source, label="temporary file"
    )
    try:
        destination_parent_fd, destination_name = _open_parent_nofollow(
            target, label="destination file"
        )
        try:
            source_fd = _open_regular_file_at(
                source_parent_fd,
                source_name,
                label=f"temporary file {source}",
            )
            try:
                source_identity = os.fstat(source_fd)
                os.fsync(source_fd)
                if _stable_identity(os.fstat(source_fd)) != _stable_identity(
                    source_identity
                ):
                    raise SecureIOError(
                        f"temporary file changed while being fsynced: {source}"
                    )
                if pre_rename_guard is not None:
                    pre_rename_guard()
                    if _stable_identity(os.fstat(source_fd)) != _stable_identity(
                        source_identity
                    ):
                        raise SecureIOError(
                            "temporary file changed while the pre-rename guard "
                            f"ran: {source}"
                        )
                renamed = False
                try:
                    _rename_noreplace_at(
                        source_parent_fd,
                        source_name,
                        destination_parent_fd,
                        destination_name,
                        destination=target,
                    )
                    renamed = True
                    published = os.stat(
                        destination_name,
                        dir_fd=destination_parent_fd,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISREG(published.st_mode) or not _same_inode(
                        source_identity, published
                    ):
                        raise SecureIOError(
                            f"published file identity does not match staging: {target}"
                        )
                    _fsync_renamed_parents(
                        source_parent_fd, destination_parent_fd
                    )
                except BaseException as publication_error:
                    if renamed:
                        try:
                            _rollback_exact_published_name(
                                source_parent_fd=source_parent_fd,
                                source_name=source_name,
                                destination_parent_fd=destination_parent_fd,
                                destination_name=destination_name,
                                expected_identity=source_identity,
                                source_display=source,
                                destination_display=target,
                            )
                        except BaseException as rollback_error:
                            raise SecureIOError(
                                "file publication failed and exact rollback did not "
                                f"complete cleanly: {target}"
                            ) from rollback_error
                    raise publication_error
            finally:
                os.close(source_fd)
        finally:
            os.close(destination_parent_fd)
    finally:
        os.close(source_parent_fd)
    return target


def publish_directory_noreplace(
    staging: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    pre_rename_guard: Callable[[], None] | None = None,
) -> Path:
    """Durably move a fully-fsynced staging tree without replacing a target.

    ``pre_rename_guard`` runs after the staging tree has been fully fsynced and
    validated, immediately before the no-replace rename.  It lets callers put
    expensive live-lineage revalidation at the narrowest publication boundary.
    """

    source = _absolute_lexical_path(staging)
    target = _absolute_lexical_path(destination)
    source_parent_fd, source_name = _open_parent_nofollow(
        source, label="staging directory"
    )
    try:
        destination_parent_fd, destination_name = _open_parent_nofollow(
            target, label="destination directory"
        )
        try:
            source_fd = _open_directory_at(
                source_parent_fd,
                source_name,
                label=f"staging directory {source}",
            )
            try:
                source_identity = _fsync_tree_fd(source_fd, display_path=source)
                if _stable_identity(os.fstat(source_fd)) != _stable_identity(
                    source_identity
                ):
                    raise SecureIOError(
                        f"staging directory changed before publication: {source}"
                    )
                if pre_rename_guard is not None:
                    pre_rename_guard()
                    if _stable_identity(os.fstat(source_fd)) != _stable_identity(
                        source_identity
                    ):
                        raise SecureIOError(
                            "staging directory changed while the pre-rename guard "
                            f"ran: {source}"
                        )
                renamed = False
                try:
                    _rename_noreplace_at(
                        source_parent_fd,
                        source_name,
                        destination_parent_fd,
                        destination_name,
                        destination=target,
                    )
                    renamed = True
                    published = os.stat(
                        destination_name,
                        dir_fd=destination_parent_fd,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(published.st_mode) or not _same_inode(
                        source_identity, published
                    ):
                        raise SecureIOError(
                            "published directory identity does not match staging: "
                            f"{target}"
                        )
                    _fsync_renamed_parents(
                        source_parent_fd, destination_parent_fd
                    )
                except BaseException as publication_error:
                    if renamed:
                        try:
                            _rollback_exact_published_name(
                                source_parent_fd=source_parent_fd,
                                source_name=source_name,
                                destination_parent_fd=destination_parent_fd,
                                destination_name=destination_name,
                                expected_identity=source_identity,
                                source_display=source,
                                destination_display=target,
                            )
                        except BaseException as rollback_error:
                            raise SecureIOError(
                                "directory publication failed and exact rollback did "
                                f"not complete cleanly: {target}"
                            ) from rollback_error
                    raise publication_error
            finally:
                os.close(source_fd)
        finally:
            os.close(destination_parent_fd)
    finally:
        os.close(source_parent_fd)
    return target


__all__ = [
    "SecureIOError",
    "StableDirectorySnapshot",
    "StableFileSnapshot",
    "ensure_directory_chain_nofollow",
    "fsync_directory",
    "fsync_tree",
    "publish_directory_noreplace",
    "publish_file_noreplace",
    "read_stable_regular_file",
    "snapshot_regular_directory",
]
