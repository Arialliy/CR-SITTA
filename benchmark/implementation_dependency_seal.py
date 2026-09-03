"""Exact, portable seal for checkpoint-axis implementation dependencies.

The seal is intentionally separate from artifact and checkpoint manifests.  It
binds the project-relative implementation files which can affect clean Source,
corrupted Source, or AdaBN checkpoint-axis v2 execution.  Dataset contents,
checkpoints, corruption-cache payloads, and result observations belong to
their existing input/artifact contracts and are deliberately absent here.

Every dependency is opened component-by-component with ``O_NOFOLLOW`` and is
required to be a stable regular file.  A file is hashed twice through the same
descriptor, with complete filesystem identity checks before, between, and
after those reads.  The pathname is then checked again without following a
leaf symlink.  Consequently callers never validate bytes from a resolved
symlink path or from an unstable pathname snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Final


PROJECT_ROOT: Final = Path(__file__).parents[1]
SCHEMA_VERSION: Final = 1
SEAL_TYPE: Final = "nsfpn_checkpoint_axis_implementation_dependencies_v1"
HASH_ALGORITHM: Final = "sha256"
PATH_CONTRACT: Final = "project-relative-exact-ordered-paths-v1"
BUNDLE_ALGORITHM: Final = "sha256-canonical-json-v1"

# Keep this tuple explicit and reviewable.  Its lexicographic order is part of
# the public contract and therefore also part of every bundle digest.
IMPLEMENTATION_DEPENDENCY_PATHS: Final[tuple[str, ...]] = (
    ".conda/lib/python3.10/site-packages/"
    "MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so",
    "SFS_MSDeformAttn/ops/__init__.py",
    "SFS_MSDeformAttn/ops/functions/__init__.py",
    "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
    "SFS_MSDeformAttn/ops/modules/__init__.py",
    "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
    "benchmark/__init__.py",
    "benchmark/adabn_axis_runner_v2.py",
    "benchmark/checkpoint_axis.py",
    "benchmark/implementation_dependency_seal.py",
    "benchmark/source_corruption_axis_runner_v2.py",
    "configs/adabn_batch_stats_fixed_splits_v1.yaml",
    "configs/checkpoint_axis_best_pd_v1.yaml",
    "configs/protocol.yaml",
    "configs/retrain_fixed_splits.yaml",
    "configs/source_corruption_benchmark_fixed_splits.yaml",
    "configs/source_corruption_cache_generation_v1.yaml",
    "corruptions/__init__.py",
    "corruptions/corruption_protocol.py",
    "corruptions/infrared_corruptions.py",
    "corruptions/severity_tables.yaml",
    "dataio/__init__.py",
    "dataio/corruption_cache.py",
    "dataio/research_dataset.py",
    "export_fixed_split_source_axis_v2.py",
    "metrics/__init__.py",
    "metrics/connected_components.py",
    "metrics/irstd_metrics.py",
    "metrics/official_metric_adapter.py",
    "metrics/target_matching.py",
    "model/MSHNet_NSFPN.py",
    "model/NS_FPN.py",
    "model/diff_cross_attns.py",
    "run_adabn_corruption_benchmark.py",
    "run_adabn_corruption_checkpoint_axis_v2.py",
    "run_adabn_source_pilot.py",
    "run_source_corruption_benchmark.py",
    "run_source_corruption_checkpoint_axis_v2.py",
    "scripts/capture_checkpoint_axis_v2_candidate_producer_observation.py",
    "scripts/run_best_pd_development_axis_v1.sh",
    "scripts/verify_checkpoint_axis_v2_parity.py",
    "test_fixed_split_source.py",
    "test_source.py",
    "tta/__init__.py",
    "tta/adabn.py",
    "tta/adabn_fast_runner.py",
    "tta/d0_secure_io.py",
    "tta/episodic_runner.py",
    "tta/model_adapter.py",
    "tta/state_manager.py",
    "utils/metric.py",
)

_RECORD_FIELDS: Final = frozenset(("path", "sha256", "bytes"))
_MATERIAL_FIELDS: Final = frozenset(
    (
        "schema_version",
        "seal_type",
        "hash_algorithm",
        "path_contract",
        "bundle_algorithm",
        "file_count",
        "total_bytes",
        "files",
    )
)
_SEAL_FIELDS: Final = _MATERIAL_FIELDS | {"bundle_sha256"}
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
_DIRECTORY_FLAGS: Final = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
)
_FILE_FLAGS: Final = (
    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
)
_READ_SIZE: Final = 1024 * 1024


class ImplementationDependencySealError(RuntimeError):
    """Raised when an implementation dependency seal fails closed."""


def _validate_fixed_path_contract() -> None:
    if not IMPLEMENTATION_DEPENDENCY_PATHS:
        raise RuntimeError("implementation dependency path contract is empty")
    if IMPLEMENTATION_DEPENDENCY_PATHS != tuple(
        sorted(IMPLEMENTATION_DEPENDENCY_PATHS)
    ):
        raise RuntimeError("implementation dependency paths must be sorted")
    if len(IMPLEMENTATION_DEPENDENCY_PATHS) != len(
        set(IMPLEMENTATION_DEPENDENCY_PATHS)
    ):
        raise RuntimeError("implementation dependency paths must be unique")
    for raw_path in IMPLEMENTATION_DEPENDENCY_PATHS:
        relative = PurePosixPath(raw_path)
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or "\\" in raw_path
            or "\x00" in raw_path
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != raw_path
        ):
            raise RuntimeError(
                f"invalid implementation dependency path: {raw_path!r}"
            )


_validate_fixed_path_contract()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ImplementationDependencySealError(
            "implementation dependency seal is not canonical-JSON serializable"
        ) from error


def _sha256_canonical_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _absolute_lexical_path(path: str | os.PathLike[str]) -> Path:
    """Make ``path`` absolute without resolving or following symlinks."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _open_project_root_nofollow(
    project_root: str | os.PathLike[str],
) -> tuple[Path, int]:
    root = _absolute_lexical_path(project_root)
    if root == Path("/"):
        raise ImplementationDependencySealError(
            "implementation dependency project root cannot be the filesystem root"
        )
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in root.parts[1:]:
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"implementation dependency project root is missing: {root}"
                ) from error
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ImplementationDependencySealError(
                        "implementation dependency project root contains a "
                        f"symlink or non-directory component: {root}"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = child
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode):
            raise ImplementationDependencySealError(
                f"implementation dependency project root is not a directory: {root}"
            )
        return root, descriptor
    except BaseException:
        os.close(descriptor)
        raise


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


def _hash_open_file(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    while True:
        chunk = os.read(descriptor, _READ_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        count += len(chunk)
    return digest.hexdigest(), count


def _open_dependency_parent(
    root_descriptor: int,
    relative_path: str,
) -> tuple[int, str]:
    parts = PurePosixPath(relative_path).parts
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            try:
                child = os.open(
                    component,
                    _DIRECTORY_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"implementation dependency is missing: {relative_path}"
                ) from error
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ImplementationDependencySealError(
                        "implementation dependency contains a symlink or "
                        f"non-directory component: {relative_path}"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _snapshot_dependency(
    root_descriptor: int,
    relative_path: str,
) -> dict[str, Any]:
    parent_descriptor, name = _open_dependency_parent(
        root_descriptor, relative_path
    )
    try:
        try:
            descriptor = os.open(
                name,
                _FILE_FLAGS,
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"implementation dependency is missing: {relative_path}"
            ) from error
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ImplementationDependencySealError(
                    "implementation dependency must be a regular non-symlink "
                    f"file: {relative_path}"
                ) from error
            raise
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ImplementationDependencySealError(
                    "implementation dependency must be a regular non-symlink "
                    f"file: {relative_path}"
                )
            first_sha256, first_bytes = _hash_open_file(descriptor)
            between = os.fstat(descriptor)
            if _stable_identity(before) != _stable_identity(between):
                raise ImplementationDependencySealError(
                    f"implementation dependency changed while being read: {relative_path}"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            second_sha256, second_bytes = _hash_open_file(descriptor)
            after = os.fstat(descriptor)
            if (
                _stable_identity(before) != _stable_identity(after)
                or first_bytes != before.st_size
                or second_bytes != before.st_size
                or first_sha256 != second_sha256
            ):
                raise ImplementationDependencySealError(
                    f"implementation dependency changed while being read: {relative_path}"
                )
            # Re-open the complete relative path from the held project-root
            # descriptor.  Checking only ``name`` against the original parent
            # would miss an atomic replacement of an ancestor directory.
            try:
                current_parent, current_name = _open_dependency_parent(
                    root_descriptor, relative_path
                )
            except FileNotFoundError as error:
                raise ImplementationDependencySealError(
                    "implementation dependency pathname changed while being "
                    f"read: {relative_path}"
                ) from error
            try:
                try:
                    current_descriptor = os.open(
                        current_name,
                        _FILE_FLAGS,
                        dir_fd=current_parent,
                    )
                except (FileNotFoundError, OSError) as error:
                    raise ImplementationDependencySealError(
                        "implementation dependency pathname changed or became a "
                        f"symlink while being read: {relative_path}"
                    ) from error
                try:
                    current = os.fstat(current_descriptor)
                    if (
                        not stat.S_ISREG(current.st_mode)
                        or _stable_identity(current) != _stable_identity(before)
                    ):
                        raise ImplementationDependencySealError(
                            "implementation dependency pathname changed or became a "
                            f"symlink while being read: {relative_path}"
                        )
                finally:
                    os.close(current_descriptor)
            finally:
                os.close(current_parent)
            return {
                "path": relative_path,
                "sha256": first_sha256,
                "bytes": first_bytes,
            }
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _seal_material(files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seal_type": SEAL_TYPE,
        "hash_algorithm": HASH_ALGORITHM,
        "path_contract": PATH_CONTRACT,
        "bundle_algorithm": BUNDLE_ALGORITHM,
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in files),
        "files": files,
    }


def capture_implementation_dependency_seal(
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
) -> dict[str, Any]:
    """Capture the fixed implementation dependency set.

    The returned mapping contains only deterministic, project-portable values:
    ordered project-relative paths, SHA-256 digests, byte counts, derived
    totals, and a canonical bundle digest.  It intentionally has no timestamp
    or absolute path.
    """

    _, root_descriptor = _open_project_root_nofollow(project_root)
    try:
        root_before = os.fstat(root_descriptor)
        files = [
            _snapshot_dependency(root_descriptor, relative_path)
            for relative_path in IMPLEMENTATION_DEPENDENCY_PATHS
        ]
        _, current_root_descriptor = _open_project_root_nofollow(project_root)
        try:
            if not _same_inode(
                root_before, os.fstat(current_root_descriptor)
            ):
                raise ImplementationDependencySealError(
                    "implementation dependency project-root pathname changed "
                    "while the seal was captured"
                )
        finally:
            os.close(current_root_descriptor)
    finally:
        os.close(root_descriptor)
    material = _seal_material(files)
    return {
        **material,
        "bundle_sha256": _sha256_canonical_json(material),
    }


def _field_difference(
    actual: set[Any], expected: frozenset[str]
) -> str:
    missing = sorted(expected - actual, key=repr)
    extra = sorted(actual - expected, key=repr)
    return f"missing={missing!r}, extra={extra!r}"


def _validate_seal_structure(seal: Mapping[str, Any]) -> dict[str, Any]:
    actual_fields = set(seal)
    if actual_fields != _SEAL_FIELDS:
        raise ImplementationDependencySealError(
            "implementation dependency seal fields differ: "
            + _field_difference(actual_fields, _SEAL_FIELDS)
        )
    constants = (
        ("schema_version", SCHEMA_VERSION),
        ("seal_type", SEAL_TYPE),
        ("hash_algorithm", HASH_ALGORITHM),
        ("path_contract", PATH_CONTRACT),
        ("bundle_algorithm", BUNDLE_ALGORITHM),
    )
    for name, expected in constants:
        if seal[name] != expected or type(seal[name]) is not type(expected):
            raise ImplementationDependencySealError(
                f"implementation dependency seal {name} differs"
            )
    files = seal["files"]
    if type(files) is not list:
        raise ImplementationDependencySealError(
            "implementation dependency seal files must be a JSON list"
        )
    if len(files) != len(IMPLEMENTATION_DEPENDENCY_PATHS):
        raise ImplementationDependencySealError(
            "implementation dependency path set differs: "
            f"expected {len(IMPLEMENTATION_DEPENDENCY_PATHS)}, got {len(files)}"
        )
    normalised_files: list[dict[str, Any]] = []
    for index, (record, expected_path) in enumerate(
        zip(files, IMPLEMENTATION_DEPENDENCY_PATHS)
    ):
        if not isinstance(record, Mapping):
            raise ImplementationDependencySealError(
                f"implementation dependency record {index} must be a mapping"
            )
        record_fields = set(record)
        if record_fields != _RECORD_FIELDS:
            raise ImplementationDependencySealError(
                f"implementation dependency record {index} fields differ: "
                + _field_difference(record_fields, _RECORD_FIELDS)
            )
        path = record["path"]
        if type(path) is not str or path != expected_path:
            raise ImplementationDependencySealError(
                "implementation dependency path order/set differs at index "
                f"{index}: expected {expected_path!r}, got {path!r}"
            )
        digest = record["sha256"]
        if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ImplementationDependencySealError(
                f"implementation dependency SHA256 is invalid: {path}"
            )
        byte_count = record["bytes"]
        if type(byte_count) is not int or byte_count < 0:
            raise ImplementationDependencySealError(
                f"implementation dependency byte count is invalid: {path}"
            )
        normalised_files.append(
            {"path": path, "sha256": digest, "bytes": byte_count}
        )
    material = _seal_material(normalised_files)
    for field in ("file_count", "total_bytes"):
        if type(seal[field]) is not int or seal[field] != material[field]:
            raise ImplementationDependencySealError(
                f"implementation dependency seal {field} differs"
            )
    bundle_sha256 = seal["bundle_sha256"]
    if (
        type(bundle_sha256) is not str
        or _SHA256_PATTERN.fullmatch(bundle_sha256) is None
    ):
        raise ImplementationDependencySealError(
            "implementation dependency bundle SHA256 is invalid"
        )
    canonical_bundle_sha256 = _sha256_canonical_json(material)
    if not hmac.compare_digest(bundle_sha256, canonical_bundle_sha256):
        raise ImplementationDependencySealError(
            "implementation dependency canonical bundle SHA256 differs"
        )
    return {**material, "bundle_sha256": bundle_sha256}


def verify_implementation_dependency_seal(
    seal: Mapping[str, Any],
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify seal structure and every current dependency byte-for-byte.

    Missing or extra records, reordered paths, invalid canonical bundle data,
    changed bytes, and any leaf/ancestor symlink fail closed.  On success the
    function returns the freshly captured current seal; it is exactly equal to
    the validated input seal and is convenient for embedding in audit output.
    """

    if not isinstance(seal, Mapping):
        raise ImplementationDependencySealError(
            "implementation dependency seal must be a mapping"
        )
    expected = _validate_seal_structure(seal)
    current = capture_implementation_dependency_seal(project_root)
    for expected_record, current_record in zip(
        expected["files"], current["files"]
    ):
        path = expected_record["path"]
        if expected_record["bytes"] != current_record["bytes"]:
            raise ImplementationDependencySealError(
                f"implementation dependency byte count drifted: {path}"
            )
        if not hmac.compare_digest(
            expected_record["sha256"], current_record["sha256"]
        ):
            raise ImplementationDependencySealError(
                f"implementation dependency SHA256 drifted: {path}"
            )
    if not hmac.compare_digest(
        expected["bundle_sha256"], current["bundle_sha256"]
    ):
        raise ImplementationDependencySealError(
            "implementation dependency bundle SHA256 drifted"
        )
    return current


__all__ = [
    "BUNDLE_ALGORITHM",
    "HASH_ALGORITHM",
    "IMPLEMENTATION_DEPENDENCY_PATHS",
    "ImplementationDependencySealError",
    "PATH_CONTRACT",
    "PROJECT_ROOT",
    "SCHEMA_VERSION",
    "SEAL_TYPE",
    "capture_implementation_dependency_seal",
    "verify_implementation_dependency_seal",
]
