#!/usr/bin/env python3
"""Seal the post-run/pre-production-patch checkpoint-axis v2 observation.

This is a deliberately post-hoc evidence collector.  It snapshots the current
bytes of the key P1 sources, verifies and records the nine already-published
best_miou candidate artifacts, and preserves their embedded provenance.  It
does *not* retroactively claim that those current bytes were the complete
runtime dependency closure used by the candidate producers.

The implementation is standard-library only: it neither imports a model/ML
runtime nor launches inference.  Reads open every path component with
O_NOFOLLOW, require stable inode/size/time identity, and reject non-regular
tree members.  Publication uses a private sibling staging tree followed by
Linux renameat2(RENAME_NOREPLACE); COMPLETE.json is the final staged write.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
CANDIDATE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "checkpoint_axis_v2_parity_candidates"
    / "best_miou"
)
DEFAULT_OUTPUT = CANDIDATE_ROOT / "CANDIDATE_PRODUCER_OBSERVATION"

DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
ARTIFACT_KINDS = ("clean", "source", "adabn")
TREE_ALGORITHM = "sorted-relative-path-tab-sha256-size-lf-v1"
OBSERVATION_CONTRACT = "cr-sitta-candidate-producer-observation-v1"
OBSERVATION_PHASE = "post_run_pre_patch"

# These are intentionally the key P1 surfaces, not a claim of a complete
# Python/import/environment dependency closure.  Both v2 entry points and the
# substantive runner implementations are retained so "runner" is unambiguous.
KEY_P1_SOURCES: tuple[tuple[str, str], ...] = (
    ("benchmark_package", "benchmark/__init__.py"),
    ("checkpoint_axis_contract", "benchmark/checkpoint_axis.py"),
    ("clean_runner", "export_fixed_split_source_axis_v2.py"),
    (
        "source_runner_implementation",
        "benchmark/source_corruption_axis_runner_v2.py",
    ),
    ("source_runner_entrypoint", "run_source_corruption_checkpoint_axis_v2.py"),
    ("adabn_runner_implementation", "benchmark/adabn_axis_runner_v2.py"),
    ("adabn_runner_entrypoint", "run_adabn_corruption_checkpoint_axis_v2.py"),
    ("parity_signer", "scripts/verify_checkpoint_axis_v2_parity.py"),
    ("axis_config", "configs/checkpoint_axis_best_pd_v1.yaml"),
    ("p1_shell_orchestrator", "scripts/run_best_pd_development_axis_v1.sh"),
    ("secure_io_dependency", "tta/d0_secure_io.py"),
)

# These files were introduced only after the nine candidates had completed.
# They must never be laundered into this historical producer observation.
EXCLUDED_POST_RUN_FILES = (
    "benchmark/implementation_dependency_seal.py",
    "tests/test_implementation_dependency_seal.py",
)

LIMITATIONS: dict[str, bool] = {
    "full_runtime_dependency_sealed": False,
    "implementation_identity_asserted": False,
    "adabn_v2_orchestrator_runtime_bound": False,
}

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_RENAME_NOREPLACE = 1


class ObservationError(RuntimeError):
    """Raised when an observation or publication invariant cannot be proved."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Stable content and identity for one regular, non-symlink file."""

    path: Path
    relative_path: str
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mode: int
    link_count: int
    mtime_ns: int
    ctime_ns: int
    data: bytes | None


def _absolute_lexical(path: str | os.PathLike[str]) -> Path:
    """Make a path absolute without resolving or following any symlink."""

    return Path(os.path.abspath(os.fspath(path)))


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


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    absolute = _absolute_lexical(path)
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ObservationError(
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
    absolute = _absolute_lexical(path)
    if absolute == Path("/") or not absolute.name:
        raise ObservationError(f"{label} cannot be the filesystem root")
    return (
        _open_directory_nofollow(absolute.parent, label=f"{label} parent"),
        absolute.name,
    )


def _open_regular_at(parent_fd: int, name: str, *, label: str) -> int:
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ObservationError(
                f"{label} must be a regular non-symlink file"
            ) from error
        raise
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise ObservationError(f"{label} must be a regular non-symlink file")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_at(parent_fd: int, name: str, *, label: str) -> int:
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ObservationError(
                f"{label} must be a real non-symlink directory"
            ) from error
        raise


def _snapshot_open_file(
    descriptor: int,
    *,
    path: Path,
    relative_path: str,
    capture_data: bool,
) -> FileSnapshot:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ObservationError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture_data else None
    total = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        if chunks is not None:
            chunks.append(chunk)
    after = os.fstat(descriptor)
    if _identity(before) != _identity(after):
        raise ObservationError(f"file changed while being read: {path}")
    if total != before.st_size:
        raise ObservationError(f"file size changed while being read: {path}")
    return FileSnapshot(
        path=path,
        relative_path=relative_path,
        sha256=digest.hexdigest(),
        size_bytes=total,
        device=before.st_dev,
        inode=before.st_ino,
        mode=before.st_mode,
        link_count=before.st_nlink,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
        data=b"".join(chunks) if chunks is not None else None,
    )


def _snapshot_regular_file(path: Path, *, capture_data: bool = True) -> FileSnapshot:
    absolute = _absolute_lexical(path)
    parent_fd, name = _open_parent_nofollow(absolute, label="snapshot file")
    try:
        initial = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(initial.st_mode):
            raise ObservationError(f"not a regular non-symlink file: {absolute}")
        descriptor = _open_regular_at(parent_fd, name, label=str(absolute))
        try:
            if not _same_inode(initial, os.fstat(descriptor)):
                raise ObservationError(f"file changed while opening: {absolute}")
            snapshot = _snapshot_open_file(
                descriptor,
                path=absolute,
                relative_path=absolute.name,
                capture_data=capture_data,
            )
        finally:
            os.close(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(current) != (
            snapshot.device,
            snapshot.inode,
            snapshot.mode,
            snapshot.link_count,
            snapshot.size_bytes,
            snapshot.mtime_ns,
            snapshot.ctime_ns,
        ):
            raise ObservationError(f"file name changed while reading: {absolute}")
    finally:
        os.close(parent_fd)

    # Independently reopen the lexical path so an ancestor/name replacement is
    # not hidden by the descriptor retained above.
    verification_parent, verification_name = _open_parent_nofollow(
        absolute, label="snapshot verification file"
    )
    try:
        verification_fd = _open_regular_at(
            verification_parent, verification_name, label=str(absolute)
        )
        try:
            if _identity(os.fstat(verification_fd)) != (
                snapshot.device,
                snapshot.inode,
                snapshot.mode,
                snapshot.link_count,
                snapshot.size_bytes,
                snapshot.mtime_ns,
                snapshot.ctime_ns,
            ):
                raise ObservationError(
                    f"file path changed after stable read: {absolute}"
                )
        finally:
            os.close(verification_fd)
    finally:
        os.close(verification_parent)
    return snapshot


def _snapshot_tree(
    root: Path,
    *,
    capture: Callable[[str], bool] = lambda _relative: False,
) -> tuple[FileSnapshot, ...]:
    """Snapshot a recursive tree through directory descriptors only."""

    absolute = _absolute_lexical(root)
    root_fd = _open_directory_nofollow(absolute, label="snapshot tree")
    members: list[FileSnapshot] = []

    def visit(directory_fd: int, parts: tuple[str, ...]) -> os.stat_result:
        before = os.fstat(directory_fd)
        names_before = tuple(sorted(os.listdir(directory_fd)))
        for name in names_before:
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                raise ObservationError(f"unsafe tree member name below {absolute}")
            relative = "/".join((*parts, name))
            display = absolute.joinpath(*parts, name)
            initial = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISREG(initial.st_mode):
                file_fd = _open_regular_at(directory_fd, name, label=str(display))
                try:
                    if not _same_inode(initial, os.fstat(file_fd)):
                        raise ObservationError(
                            f"tree file changed while opening: {display}"
                        )
                    snapshot = _snapshot_open_file(
                        file_fd,
                        path=display,
                        relative_path=relative,
                        capture_data=capture(relative),
                    )
                finally:
                    os.close(file_fd)
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if _identity(current) != (
                    snapshot.device,
                    snapshot.inode,
                    snapshot.mode,
                    snapshot.link_count,
                    snapshot.size_bytes,
                    snapshot.mtime_ns,
                    snapshot.ctime_ns,
                ):
                    raise ObservationError(
                        f"tree file name changed while reading: {display}"
                    )
                members.append(snapshot)
                continue
            if stat.S_ISDIR(initial.st_mode):
                child_fd = _open_directory_at(
                    directory_fd, name, label=str(display)
                )
                try:
                    if not _same_inode(initial, os.fstat(child_fd)):
                        raise ObservationError(
                            f"tree directory changed while opening: {display}"
                        )
                    child_after = visit(child_fd, (*parts, name))
                finally:
                    os.close(child_fd)
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if _identity(current) != _identity(child_after):
                    raise ObservationError(
                        f"tree directory name changed while reading: {display}"
                    )
                continue
            raise ObservationError(
                f"tree contains a symlink or non-regular member: {display}"
            )
        if tuple(sorted(os.listdir(directory_fd))) != names_before:
            raise ObservationError(f"tree membership changed while reading: {absolute}")
        after = os.fstat(directory_fd)
        if _identity(before) != _identity(after):
            raise ObservationError(f"tree directory changed while reading: {absolute}")
        return after

    try:
        root_after = visit(root_fd, ())
    finally:
        os.close(root_fd)
    verification_fd = _open_directory_nofollow(
        absolute, label="snapshot tree verification"
    )
    try:
        if _identity(os.fstat(verification_fd)) != _identity(root_after):
            raise ObservationError(f"tree root changed after stable read: {absolute}")
    finally:
        os.close(verification_fd)
    return tuple(sorted(members, key=lambda value: value.relative_path))


def _strict_json(data: bytes, *, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ObservationError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ObservationError(f"non-finite JSON value in {label}: {value}")

    try:
        loaded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationError(f"invalid UTF-8 JSON object: {label}") from error
    if not isinstance(loaded, dict):
        raise ObservationError(f"expected JSON object: {label}")
    return loaded


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ObservationError(f"{label} differs")


def _validate_relative_path(raw: str, *, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or "\\" in raw or "\x00" in raw:
        raise ObservationError(f"{label} is not a portable relative path")
    value = PurePosixPath(raw)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise ObservationError(f"{label} escapes its root: {raw!r}")
    return value


def _tree_ledger(
    members: Sequence[FileSnapshot], *, exclude: Sequence[str]
) -> dict[str, Any]:
    excluded = {
        _validate_relative_path(value, label="excluded member").as_posix()
        for value in exclude
    }
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for snapshot in sorted(members, key=lambda value: value.relative_path):
        if snapshot.relative_path in excluded:
            continue
        record = {
            "path": snapshot.relative_path,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
        }
        records.append(record)
        digest.update(
            (
                f"{snapshot.relative_path}\t{snapshot.sha256}\t"
                f"{snapshot.size_bytes}\n"
            ).encode("utf-8")
        )
    return {
        "algorithm": TREE_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def _captured(snapshot: FileSnapshot, *, label: str) -> bytes:
    if snapshot.data is None:
        raise ObservationError(f"required bytes were not captured: {label}")
    return snapshot.data


def _is_provenance_source(kind: str, relative_path: str) -> bool:
    if relative_path == "provenance.json":
        return True
    if relative_path.endswith("/condition_provenance.json"):
        return True
    return kind == "source" and relative_path == "benchmark.json"


def _recorded_source_hashes(
    embedded: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Find explicit runtime/source hash maps inside preserved provenance."""

    found: list[dict[str, Any]] = []

    def visit(value: Any, pointer: str, provenance_file: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_pointer = f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}"
                if key in {"runtime_file_sha256", "file_sha256"} and isinstance(
                    child, Mapping
                ):
                    for source_path, digest in sorted(child.items()):
                        if isinstance(source_path, str) and isinstance(digest, str):
                            found.append(
                                {
                                    "provenance_source": provenance_file,
                                    "json_pointer": child_pointer,
                                    "source_path": source_path,
                                    "recorded_sha256": digest,
                                }
                            )
                visit(child, child_pointer, provenance_file)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{pointer}/{index}", provenance_file)

    for record in embedded:
        visit(record["value"], str(record["json_pointer"]), str(record["source_file"]))
    return tuple(found)


def _verify_candidate(kind: str, dataset: str) -> dict[str, Any]:
    root = CANDIDATE_ROOT / kind / dataset

    def capture(relative: str) -> bool:
        return relative in {"artifact_manifest.json", "COMPLETE.json"} or (
            _is_provenance_source(kind, relative)
        )

    members = _snapshot_tree(root, capture=capture)
    by_path = {value.relative_path: value for value in members}
    for required in ("artifact_manifest.json", "COMPLETE.json"):
        if required not in by_path:
            raise ObservationError(f"{kind}/{dataset} omits {required}")
    manifest_snapshot = by_path["artifact_manifest.json"]
    complete_snapshot = by_path["COMPLETE.json"]
    manifest = _strict_json(
        _captured(manifest_snapshot, label=f"{kind}/{dataset} manifest"),
        label=f"{kind}/{dataset}/artifact_manifest.json",
    )
    complete = _strict_json(
        _captured(complete_snapshot, label=f"{kind}/{dataset} COMPLETE"),
        label=f"{kind}/{dataset}/COMPLETE.json",
    )
    measured = _tree_ledger(
        members, exclude=("artifact_manifest.json", "COMPLETE.json")
    )
    expected_tree = manifest.get("payload_tree")
    if not isinstance(expected_tree, Mapping):
        raise ObservationError(f"{kind}/{dataset} manifest omits payload_tree")
    _require_equal(measured, dict(expected_tree), f"{kind}/{dataset} payload tree")
    for field, expected in (
        ("artifact_kind", kind),
        ("dataset", dataset),
        ("checkpoint_role", "best_miou"),
    ):
        _require_equal(manifest.get(field), expected, f"{kind}/{dataset} manifest {field}")
        _require_equal(complete.get(field), expected, f"{kind}/{dataset} COMPLETE {field}")
    _require_equal(complete.get("complete"), True, f"{kind}/{dataset} completion")
    _require_equal(
        complete.get("manifest_sha256"),
        manifest_snapshot.sha256,
        f"{kind}/{dataset} manifest seal",
    )
    if "artifact_manifest_sha256" in complete:
        _require_equal(
            complete["artifact_manifest_sha256"],
            manifest_snapshot.sha256,
            f"{kind}/{dataset} alternate manifest seal",
        )
    _require_equal(
        complete.get("payload_tree_sha256"),
        measured["sha256"],
        f"{kind}/{dataset} payload tree SHA256",
    )
    _require_equal(
        complete.get("payload_file_count"),
        measured["file_count"],
        f"{kind}/{dataset} payload file count",
    )
    required_payloads = manifest.get("required_payloads")
    if not isinstance(required_payloads, list) or not all(
        isinstance(value, str) for value in required_payloads
    ):
        raise ObservationError(f"{kind}/{dataset} has invalid required_payloads")
    present = {record["path"] for record in measured["files"]}
    missing = sorted(
        _validate_relative_path(value, label="required payload").as_posix()
        for value in required_payloads
        if _validate_relative_path(value, label="required payload").as_posix()
        not in present
    )
    if missing:
        raise ObservationError(f"{kind}/{dataset} missing payloads: {missing}")

    provenance_snapshots = tuple(
        value for value in members if _is_provenance_source(kind, value.relative_path)
    )
    embedded: list[dict[str, Any]] = []
    for snapshot in provenance_snapshots:
        document = _strict_json(
            _captured(snapshot, label=snapshot.relative_path),
            label=f"{kind}/{dataset}/{snapshot.relative_path}",
        )
        if kind == "source" and snapshot.relative_path == "benchmark.json":
            if not isinstance(document.get("repository_provenance"), Mapping):
                raise ObservationError(
                    f"{kind}/{dataset} benchmark omits repository_provenance"
                )
            pointer = "/repository_provenance"
            value: Any = dict(document["repository_provenance"])
        else:
            pointer = ""
            value = document
        embedded.append(
            {
                "source_file": snapshot.relative_path,
                "source_file_sha256": snapshot.sha256,
                "source_file_size_bytes": snapshot.size_bytes,
                "json_pointer": pointer,
                "value": value,
            }
        )
    if not embedded:
        raise ObservationError(f"{kind}/{dataset} has no embedded provenance")

    return {
        "kind": kind,
        "dataset": dataset,
        "root": root.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_snapshot": manifest_snapshot,
        "complete_snapshot": complete_snapshot,
        "manifest": manifest,
        "complete": complete,
        "payload_tree_seal": {
            "algorithm": measured["algorithm"],
            "sha256": measured["sha256"],
            "file_count": measured["file_count"],
        },
        "provenance_snapshots": provenance_snapshots,
        "embedded_provenance": embedded,
        "recorded_source_hashes": _recorded_source_hashes(embedded),
    }


def _simple_parts(relative: str) -> tuple[str, ...]:
    value = _validate_relative_path(relative, label="staging output")
    return tuple(value.parts)


def _write_new_file(staging: Path, relative: str, data: bytes) -> None:
    """Create and fsync one new staging file through no-follow dirfds."""

    parts = _simple_parts(relative)
    directory_fd = _open_directory_nofollow(staging, label="staging root")
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, mode=0o755, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileExistsError:
                pass
            child_fd = _open_directory_at(
                directory_fd, component, label=f"staging component {component}"
            )
            os.close(directory_fd)
            directory_fd = child_fd
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(parts[-1], flags, 0o644, dir_fd=directory_fd)
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ObservationError(f"short staging write: {relative}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _create_private_staging(final: Path) -> Path:
    parent_fd, final_name = _open_parent_nofollow(final, label="observation output")
    try:
        try:
            os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                errno.EEXIST,
                f"destination exists; refusing overwrite: {final}",
                str(final),
            )
        for _attempt in range(128):
            name = f".{final_name}.build-{os.getpid()}-{secrets.token_hex(8)}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            os.fsync(parent_fd)
            child_fd = _open_directory_at(parent_fd, name, label="private staging")
            os.close(child_fd)
            return final.parent / name
        raise ObservationError("could not allocate a unique private staging name")
    finally:
        os.close(parent_fd)


def _fsync_tree_fd(directory_fd: int, *, display: Path) -> os.stat_result:
    before = os.fstat(directory_fd)
    names = tuple(sorted(os.listdir(directory_fd)))
    for name in names:
        child_display = display / name
        initial = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISREG(initial.st_mode):
            file_fd = _open_regular_at(directory_fd, name, label=str(child_display))
            try:
                if not _same_inode(initial, os.fstat(file_fd)):
                    raise ObservationError(
                        f"staging file changed while opening: {child_display}"
                    )
                identity = _identity(os.fstat(file_fd))
                os.fsync(file_fd)
                if _identity(os.fstat(file_fd)) != identity:
                    raise ObservationError(
                        f"staging file changed while fsyncing: {child_display}"
                    )
            finally:
                os.close(file_fd)
            continue
        if stat.S_ISDIR(initial.st_mode):
            child_fd = _open_directory_at(
                directory_fd, name, label=str(child_display)
            )
            try:
                if not _same_inode(initial, os.fstat(child_fd)):
                    raise ObservationError(
                        f"staging directory changed while opening: {child_display}"
                    )
                child_after = _fsync_tree_fd(child_fd, display=child_display)
            finally:
                os.close(child_fd)
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _identity(current) != _identity(child_after):
                raise ObservationError(
                    f"staging directory changed while fsyncing: {child_display}"
                )
            continue
        raise ObservationError(
            f"staging contains a symlink or non-regular member: {child_display}"
        )
    if tuple(sorted(os.listdir(directory_fd))) != names:
        raise ObservationError(f"staging membership changed: {display}")
    os.fsync(directory_fd)
    after = os.fstat(directory_fd)
    if _identity(before) != _identity(after):
        raise ObservationError(f"staging directory changed while fsyncing: {display}")
    return after


def _rename_noreplace(parent_fd: int, source: str, destination: str, *, path: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ObservationError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"destination exists; refusing overwrite: {path}",
            str(path),
        )
    if error_number in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
        raise ObservationError(
            "atomic no-replace directory publication is unsupported"
        )
    raise OSError(error_number, os.strerror(error_number), str(path))


def _publish_directory_noreplace(
    staging: Path,
    final: Path,
    *,
    pre_rename_guard: Callable[[], None],
) -> None:
    if staging.parent != final.parent or not staging.name.startswith(
        f".{final.name}.build-"
    ):
        raise ObservationError("publication requires a private sibling staging tree")
    parent_fd = _open_directory_nofollow(final.parent, label="publication parent")
    try:
        source_fd = _open_directory_at(
            parent_fd, staging.name, label="publication staging"
        )
        try:
            source_identity = _fsync_tree_fd(source_fd, display=staging)
            pre_rename_guard()
            if _identity(os.fstat(source_fd)) != _identity(source_identity):
                raise ObservationError("staging changed while publication guard ran")
            _rename_noreplace(parent_fd, staging.name, final.name, path=final)
            published = os.stat(final.name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(published.st_mode) or not _same_inode(
                source_identity, published
            ):
                raise ObservationError(
                    "published observation identity does not match staging"
                )
            os.fsync(parent_fd)
        finally:
            os.close(source_fd)
    finally:
        os.close(parent_fd)


def _verify_observation_envelope(root: Path) -> dict[str, Any]:
    captured_names = {
        "artifact_manifest.json",
        "COMPLETE.json",
        "OBSERVATION.json",
        "source_inventory.json",
    }
    members = _snapshot_tree(root, capture=lambda relative: relative in captured_names)
    by_path = {value.relative_path: value for value in members}
    for required in captured_names:
        if required not in by_path:
            raise ObservationError(f"observation envelope omits {required}")
    manifest_snapshot = by_path["artifact_manifest.json"]
    complete_snapshot = by_path["COMPLETE.json"]
    observation_snapshot = by_path["OBSERVATION.json"]
    inventory_snapshot = by_path["source_inventory.json"]
    manifest = _strict_json(
        _captured(manifest_snapshot, label="observation manifest"),
        label="artifact_manifest.json",
    )
    complete = _strict_json(
        _captured(complete_snapshot, label="observation COMPLETE"),
        label="COMPLETE.json",
    )
    observation = _strict_json(
        _captured(observation_snapshot, label="observation payload"),
        label="OBSERVATION.json",
    )
    measured = _tree_ledger(
        members, exclude=("artifact_manifest.json", "COMPLETE.json")
    )
    _require_equal(
        manifest.get("artifact_contract"),
        OBSERVATION_CONTRACT,
        "observation contract",
    )
    _require_equal(manifest.get("payload_tree"), measured, "observation payload tree")
    _require_equal(complete.get("complete"), True, "observation completion")
    _require_equal(
        complete.get("manifest_sha256"),
        manifest_snapshot.sha256,
        "observation manifest SHA256",
    )
    _require_equal(
        complete.get("payload_tree_sha256"),
        measured["sha256"],
        "observation payload tree SHA256",
    )
    _require_equal(
        complete.get("payload_file_count"),
        measured["file_count"],
        "observation payload file count",
    )
    _require_equal(
        complete.get("observation_sha256"),
        observation_snapshot.sha256,
        "observation payload SHA256",
    )
    _require_equal(
        complete.get("source_inventory_sha256"),
        inventory_snapshot.sha256,
        "source inventory SHA256",
    )
    for field, expected in LIMITATIONS.items():
        _require_equal(manifest.get(field), expected, f"manifest {field}")
        _require_equal(complete.get(field), expected, f"COMPLETE {field}")
        _require_equal(observation.get(field), expected, f"OBSERVATION {field}")
    _require_equal(
        observation.get("observation_phase"),
        OBSERVATION_PHASE,
        "observation phase",
    )
    return {
        "output": str(_absolute_lexical(root)),
        "artifact_manifest_sha256": manifest_snapshot.sha256,
        "complete_sha256": complete_snapshot.sha256,
        "observation_sha256": observation_snapshot.sha256,
        "source_inventory_sha256": inventory_snapshot.sha256,
        "payload_tree_sha256": measured["sha256"],
        "payload_file_count": measured["file_count"],
        "observer_sha256": complete.get("observer_sha256"),
    }


def _source_inventory(
    source_snapshots: Sequence[tuple[str, str, FileSnapshot]],
    *,
    observer: FileSnapshot,
    captured_at: str,
) -> dict[str, Any]:
    records = []
    for role, relative, snapshot in source_snapshots:
        records.append(
            {
                "role": role,
                "path": relative,
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
                "content_addressed_copy": f"source_blobs/sha256/{snapshot.sha256}",
            }
        )
    return {
        "schema_version": 1,
        "observation_phase": OBSERVATION_PHASE,
        "captured_at_utc": captured_at,
        "hash_algorithm": "sha256",
        "content_address_scheme": "source_blobs/sha256/<sha256>",
        "scope": "key_p1_sources_not_full_runtime_dependency_closure",
        "excluded_post_run_files": list(EXCLUDED_POST_RUN_FILES),
        "excluded_post_run_files_are_candidate_runtime_dependencies": False,
        **LIMITATIONS,
        "observer": {
            "path": "scripts/capture_checkpoint_axis_v2_candidate_producer_observation.py",
            "sha256": observer.sha256,
            "size_bytes": observer.size_bytes,
            "content_addressed_copy": f"source_blobs/sha256/{observer.sha256}",
        },
        "files": records,
    }


def _observation_record(
    candidate: Mapping[str, Any],
    *,
    source_by_path: Mapping[str, FileSnapshot],
) -> dict[str, Any]:
    kind = str(candidate["kind"])
    dataset = str(candidate["dataset"])
    metadata_root = f"candidate_metadata/{kind}/{dataset}"
    bindings: list[dict[str, Any]] = []
    for raw in candidate["recorded_source_hashes"]:
        record = dict(raw)
        observed = source_by_path.get(str(record["source_path"]))
        record["observed_post_run_pre_patch_sha256"] = (
            observed.sha256 if observed is not None else None
        )
        record["matches_observed_snapshot"] = (
            observed is not None
            and observed.sha256 == str(record["recorded_sha256"])
        )
        bindings.append(record)
    provenance = []
    for embedded in candidate["embedded_provenance"]:
        value = dict(embedded)
        value["preserved_source_copy"] = (
            f"{metadata_root}/provenance_sources/{value['source_file']}"
        )
        provenance.append(value)
    manifest_snapshot: FileSnapshot = candidate["manifest_snapshot"]
    complete_snapshot: FileSnapshot = candidate["complete_snapshot"]
    return {
        "schema_version": 1,
        "observation_phase": OBSERVATION_PHASE,
        "artifact_kind": kind,
        "dataset": dataset,
        "candidate_root": candidate["root"],
        "checkpoint_role": "best_miou",
        **LIMITATIONS,
        "artifact_seals": {
            "artifact_manifest": {
                "source_path": f"{candidate['root']}/artifact_manifest.json",
                "preserved_copy": f"{metadata_root}/artifact_manifest.json",
                "sha256": manifest_snapshot.sha256,
                "size_bytes": manifest_snapshot.size_bytes,
            },
            "complete": {
                "source_path": f"{candidate['root']}/COMPLETE.json",
                "preserved_copy": f"{metadata_root}/COMPLETE.json",
                "sha256": complete_snapshot.sha256,
                "size_bytes": complete_snapshot.size_bytes,
            },
            "payload_tree": candidate["payload_tree_seal"],
        },
        "recorded_p1_source_hash_bindings": bindings,
        "embedded_provenance": provenance,
        "interpretation": {
            "artifact_payload_tree_reverified": True,
            "current_key_p1_source_bytes_observed": True,
            "current_source_bytes_prove_historical_runtime_identity": False,
            "adabn_v2_orchestrator_hash_found_in_artifact_provenance": False
            if kind == "adabn"
            else None,
        },
    }


def capture(output: Path) -> dict[str, Any]:
    final = _absolute_lexical(output)
    if final != DEFAULT_OUTPUT:
        raise ObservationError(f"--output must equal the frozen path: {DEFAULT_OUTPUT}")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ObservationError("O_NOFOLLOW is unavailable; refusing observation")

    captured_at = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    observer = _snapshot_regular_file(Path(__file__), capture_data=True)
    source_snapshots: list[tuple[str, str, FileSnapshot]] = []
    for role, relative in KEY_P1_SOURCES:
        source_snapshots.append(
            (
                role,
                relative,
                _snapshot_regular_file(PROJECT_ROOT / relative, capture_data=True),
            )
        )
    source_by_path = {relative: value for _role, relative, value in source_snapshots}

    candidates = [
        _verify_candidate(kind, dataset)
        for kind in ARTIFACT_KINDS
        for dataset in DATASETS
    ]
    initial_candidate_seals = {
        f"{value['kind']}/{value['dataset']}": {
            "manifest_sha256": value["manifest_snapshot"].sha256,
            "complete_sha256": value["complete_snapshot"].sha256,
            "payload_tree": value["payload_tree_seal"],
            "provenance_source_sha256": {
                item.relative_path: item.sha256
                for item in value["provenance_snapshots"]
            },
        }
        for value in candidates
    }

    staging = _create_private_staging(final)
    try:
        written_blobs: set[str] = set()
        for _role, _relative, snapshot in (*source_snapshots,):
            if snapshot.sha256 not in written_blobs:
                _write_new_file(
                    staging,
                    f"source_blobs/sha256/{snapshot.sha256}",
                    _captured(snapshot, label=str(snapshot.path)),
                )
                written_blobs.add(snapshot.sha256)
        if observer.sha256 not in written_blobs:
            _write_new_file(
                staging,
                f"source_blobs/sha256/{observer.sha256}",
                _captured(observer, label="observer script"),
            )
            written_blobs.add(observer.sha256)

        inventory = _source_inventory(
            source_snapshots, observer=observer, captured_at=captured_at
        )
        _write_new_file(staging, "source_inventory.json", _pretty_json(inventory))

        observation_records: list[dict[str, Any]] = []
        for candidate in candidates:
            kind = candidate["kind"]
            dataset = candidate["dataset"]
            prefix = f"candidate_metadata/{kind}/{dataset}"
            manifest_snapshot: FileSnapshot = candidate["manifest_snapshot"]
            complete_snapshot: FileSnapshot = candidate["complete_snapshot"]
            _write_new_file(
                staging,
                f"{prefix}/artifact_manifest.json",
                _captured(manifest_snapshot, label=f"{kind}/{dataset} manifest"),
            )
            _write_new_file(
                staging,
                f"{prefix}/COMPLETE.json",
                _captured(complete_snapshot, label=f"{kind}/{dataset} COMPLETE"),
            )
            _write_new_file(
                staging,
                f"{prefix}/payload_tree_seal.json",
                _pretty_json(candidate["payload_tree_seal"]),
            )
            for provenance_snapshot in candidate["provenance_snapshots"]:
                _write_new_file(
                    staging,
                    f"{prefix}/provenance_sources/"
                    f"{provenance_snapshot.relative_path}",
                    _captured(
                        provenance_snapshot,
                        label=f"{kind}/{dataset} provenance",
                    ),
                )
            record = _observation_record(candidate, source_by_path=source_by_path)
            _write_new_file(
                staging,
                f"{prefix}/observation.json",
                _pretty_json(record),
            )
            observation_records.append(record)

        observation = {
            "schema_version": 1,
            "artifact_contract": OBSERVATION_CONTRACT,
            "observation_type": "checkpoint_axis_v2_candidate_producer_observation",
            "observation_phase": OBSERVATION_PHASE,
            "captured_at_utc": captured_at,
            "candidate_root": CANDIDATE_ROOT.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_role": "best_miou",
            "artifact_kinds": list(ARTIFACT_KINDS),
            "datasets": list(DATASETS),
            "candidate_dataset_artifact_count": len(observation_records),
            "excluded_post_run_files": list(EXCLUDED_POST_RUN_FILES),
            "excluded_post_run_files_are_candidate_runtime_dependencies": False,
            **LIMITATIONS,
            "evidentiary_scope": {
                "post_hoc_current_source_content_observation": True,
                "nine_published_candidate_artifacts_recursively_rehashed": True,
                "candidate_manifest_and_complete_bytes_preserved": True,
                "candidate_embedded_provenance_preserved": True,
                "historical_process_or_environment_attestation": False,
                "gpu_execution_performed_by_observer": False,
                "ml_or_gpu_library_imported_by_observer": False,
            },
            "limitation_reasons": {
                "full_runtime_dependency_sealed": (
                    "The source inventory intentionally preserves key P1 surfaces, "
                    "not the complete interpreter/import/native-library/environment "
                    "dependency closure."
                ),
                "implementation_identity_asserted": (
                    "These source bytes were observed after candidate completion; "
                    "post-hoc equality with selective embedded hashes cannot prove "
                    "the full historical execution identity."
                ),
                "adabn_v2_orchestrator_runtime_bound": (
                    "AdaBN candidate provenance binds the reused sealed v1 compute "
                    "runner, but does not bind the v2 orchestrator implementation "
                    "or its entry point as historical runtime dependencies."
                ),
            },
            "source_inventory": "source_inventory.json",
            "candidates": observation_records,
        }
        _write_new_file(staging, "OBSERVATION.json", _pretty_json(observation))

        # Envelope manifest follows all evidence payload writes.
        payload_members = _snapshot_tree(staging)
        payload_tree = _tree_ledger(
            payload_members, exclude=("artifact_manifest.json", "COMPLETE.json")
        )
        manifest = {
            "schema_version": 1,
            "artifact_contract": OBSERVATION_CONTRACT,
            "observation_phase": OBSERVATION_PHASE,
            "checkpoint_role": "best_miou",
            "candidate_dataset_artifact_count": 9,
            **LIMITATIONS,
            "required_payloads": ["OBSERVATION.json", "source_inventory.json"],
            "payload_tree": payload_tree,
        }
        _write_new_file(staging, "artifact_manifest.json", _pretty_json(manifest))
        manifest_snapshot = _snapshot_regular_file(
            staging / "artifact_manifest.json", capture_data=False
        )
        observation_snapshot = _snapshot_regular_file(
            staging / "OBSERVATION.json", capture_data=False
        )
        inventory_snapshot = _snapshot_regular_file(
            staging / "source_inventory.json", capture_data=False
        )

        # COMPLETE.json is intentionally and mechanically the final staged
        # write.  Everything after this line is read-only validation/fsync or
        # the atomic directory rename.
        complete = {
            "schema_version": 1,
            "complete": True,
            "artifact_contract": OBSERVATION_CONTRACT,
            "observation_phase": OBSERVATION_PHASE,
            "checkpoint_role": "best_miou",
            "candidate_dataset_artifact_count": 9,
            **LIMITATIONS,
            "complete_written_last": True,
            "manifest_sha256": manifest_snapshot.sha256,
            "observation_sha256": observation_snapshot.sha256,
            "source_inventory_sha256": inventory_snapshot.sha256,
            "observer_sha256": observer.sha256,
            "payload_tree_sha256": payload_tree["sha256"],
            "payload_file_count": payload_tree["file_count"],
        }
        _write_new_file(staging, "COMPLETE.json", _pretty_json(complete))
        _verify_observation_envelope(staging)

        def guard() -> None:
            current_observer = _snapshot_regular_file(Path(__file__), capture_data=False)
            _require_equal(
                current_observer.sha256,
                observer.sha256,
                "observer source at publication boundary",
            )
            for _role, relative, initial in source_snapshots:
                current = _snapshot_regular_file(
                    PROJECT_ROOT / relative, capture_data=False
                )
                _require_equal(
                    current.sha256,
                    initial.sha256,
                    f"key P1 source at publication boundary: {relative}",
                )
            live = {
                f"{kind}/{dataset}": _verify_candidate(kind, dataset)
                for kind in ARTIFACT_KINDS
                for dataset in DATASETS
            }
            for key, candidate in live.items():
                current_seal = {
                    "manifest_sha256": candidate["manifest_snapshot"].sha256,
                    "complete_sha256": candidate["complete_snapshot"].sha256,
                    "payload_tree": candidate["payload_tree_seal"],
                    "provenance_source_sha256": {
                        item.relative_path: item.sha256
                        for item in candidate["provenance_snapshots"]
                    },
                }
                _require_equal(
                    current_seal,
                    initial_candidate_seals[key],
                    f"candidate at publication boundary: {key}",
                )
            _verify_observation_envelope(staging)

        _publish_directory_noreplace(staging, final, pre_rename_guard=guard)
        return _verify_observation_envelope(final)
    except BaseException:
        if (
            staging.parent == final.parent
            and staging.name.startswith(f".{final.name}.build-")
            and os.path.lexists(staging)
        ):
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Read-only verification of the already-published frozen output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = _absolute_lexical(args.output)
    result = (
        _verify_observation_envelope(output)
        if args.verify_existing
        else capture(output)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
