"""Metadata-only protocol for the train-side TTA calibration Pilot v2.

This module deliberately has no dataset/image dependencies.  Its readable
inputs are limited to the protocol YAML, the fixed train/test ID text files,
and the round-02 Pilot JSON metadata chain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any

import yaml


DATASET_NAMES = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
PROTOCOL_ID = "cr-sitta-tta-train-side-calibration-pilot-v2"
SCOPE = "source_train_side_method_calibration"
SELECTION_RULE = (
    "exclude_round_02_corruption_severity_pilot_then_"
    "ascending_sha256_of_utf8_canonical_image_id_then_id"
)
SUBSET_SIZE = 64
PROTOCOL_RELATIVE_PATH = "configs/tta_train_side_calibration_pilot_v2.yaml"
OUTPUT_RELATIVE_ROOT = "configs/tta_train_side_pilot_v2"
OUTPUT_MANIFEST = f"{OUTPUT_RELATIVE_ROOT}/manifest.json"


def _absolute_lexical_path(path: Path) -> Path:
    """Make a path absolute without resolving or otherwise following symlinks."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _open_nofollow(path: Path, *, directory: bool, label: str) -> int:
    """Open an absolute path component-by-component without following symlinks."""

    absolute = _absolute_lexical_path(path)
    if not absolute.is_absolute():  # defensive; abspath above guarantees this
        raise ValueError(f"{label} path must be absolute: {absolute}")
    parts = absolute.parts[1:]
    if not parts:
        raise ValueError(f"{label} cannot be the filesystem root")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", directory_flags)
    try:
        for index, component in enumerate(parts):
            is_last = index == len(parts) - 1
            flags = directory_flags if (not is_last or directory) else file_flags
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        f"{label} contains a symlink or non-directory component: {path}"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_bytes(path: Path, *, label: str) -> tuple[bytes, str]:
    """Return one stable regular-file snapshot and its hash.

    Hashing and parsing callers consume these same bytes.  The before/after
    ``fstat`` seal detects in-place replacement or mutation during the read.
    """

    try:
        descriptor = _open_nofollow(path, directory=False, label=label)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stable_stat_identity(before) != _stable_stat_identity(after):
            raise RuntimeError(f"{label} changed while its stable snapshot was read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RuntimeError(f"{label} size changed while its stable snapshot was read")
        os.lseek(descriptor, 0, os.SEEK_SET)
        replay_chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            replay_chunks.append(chunk)
        replay = b"".join(replay_chunks)
        final = os.fstat(descriptor)
        if (
            _stable_stat_identity(before) != _stable_stat_identity(final)
            or replay != payload
        ):
            raise RuntimeError(f"{label} changed while its stable snapshot was read")
        # Re-open the name while the original descriptor is still held.  This
        # catches atomic pathname swaps that leave the first inode unchanged.
        current_descriptor = _open_nofollow(path, directory=False, label=label)
        try:
            current = os.fstat(current_descriptor)
            if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError(
                    f"{label} pathname changed while its stable snapshot was read"
                )
        finally:
            os.close(current_descriptor)
        return payload, hashlib.sha256(payload).hexdigest()
    finally:
        os.close(descriptor)


def _directory_names_nofollow(path: Path, *, label: str) -> set[str]:
    descriptor = _open_nofollow(path, directory=True, label=label)
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode):
            raise ValueError(f"{label} must be a directory: {path}")
        return set(os.listdir(descriptor))
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    return _read_stable_bytes(path, label="SHA256 input")[1]


def ordered_ids_sha256(image_ids: Sequence[str]) -> str:
    payload = json.dumps(
        list(image_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serialize_ids(image_ids: Sequence[str]) -> bytes:
    return "".join(f"{image_id}\n" for image_id in image_ids).encode("utf-8")


def canonical_image_id(raw: str, *, label: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError(f"{label} contains an empty image ID")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} contains an unsafe image ID: {value!r}")
    canonical = path.with_suffix("").as_posix()
    if canonical in {"", "."}:
        raise ValueError(f"{label} contains an invalid image ID: {value!r}")
    return canonical


def _parse_canonical_ids(payload: bytes, *, label: str) -> tuple[str, ...]:
    try:
        raw_values = payload.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    if not raw_values:
        raise ValueError(f"{label} is empty")
    values = tuple(
        canonical_image_id(raw, label=f"{label} line {index}")
        for index, raw in enumerate(raw_values, start=1)
    )
    if len(values) != len(set(values)):
        raise ValueError(f"{label} has duplicate canonical image IDs")
    return values


def read_canonical_ids(path: Path, *, label: str) -> tuple[str, ...]:
    """Read one stable ID snapshot without consulting dataset pixels."""

    payload, _ = _read_stable_bytes(path, label=label)
    return _parse_canonical_ids(payload, label=label)


def select_train_side_ids(
    train_ids: Sequence[str],
    excluded_parent_ids: Sequence[str],
    *,
    limit: int = SUBSET_SIZE,
) -> tuple[str, ...]:
    if type(limit) is not int or limit < 1:
        raise ValueError("selection limit must be a positive integer")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("train IDs must be unique")
    if len(excluded_parent_ids) != len(set(excluded_parent_ids)):
        raise ValueError("parent Pilot IDs must be unique")
    excluded = set(excluded_parent_ids)
    candidates = [image_id for image_id in train_ids if image_id not in excluded]
    if len(candidates) < limit:
        raise ValueError(
            f"only {len(candidates)} eligible train IDs remain; {limit} required"
        )
    candidates.sort(
        key=lambda image_id: (
            hashlib.sha256(image_id.encode("utf-8")).hexdigest(),
            image_id,
        )
    )
    return tuple(candidates[:limit])


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    _require_equal(value, expected, label)


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _safe_repository_path(repository: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise TypeError(f"{label} must be a non-empty repository-relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be repository-relative: {raw!r}")
    repository = _absolute_lexical_path(repository)
    candidate = Path(os.path.normpath(os.fspath(repository / relative)))
    if os.path.commonpath((os.fspath(repository), os.fspath(candidate))) != os.fspath(
        repository
    ):
        raise ValueError(f"{label} escapes the repository: {raw!r}")
    return candidate


def _parse_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    return dict(_require_mapping(loaded, label))


def _verified_snapshot(path: Path, expected: Any, label: str) -> tuple[bytes, str]:
    expected_hash = _require_sha256(expected, f"{label} configured SHA256")
    payload, actual = _read_stable_bytes(path, label=label)
    _require_equal(actual, expected_hash, f"{label} SHA256")
    return payload, actual


def load_protocol(protocol_path: Path) -> tuple[dict[str, Any], str]:
    payload, protocol_sha256 = _read_stable_bytes(protocol_path, label="protocol")
    try:
        loaded = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("protocol is not valid UTF-8 YAML") from error
    protocol = dict(_require_mapping(loaded, "protocol"))
    _require_equal(protocol.get("schema_version"), 2, "schema_version")
    _require_equal(protocol.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    _require_equal(
        protocol.get("protocol_path"), PROTOCOL_RELATIVE_PATH, "protocol_path"
    )
    _require_equal(protocol.get("scope"), SCOPE, "scope")
    _require_bool(protocol.get("no_validation_split"), True, "no_validation_split")
    _require_bool(protocol.get("paper_result"), False, "paper_result")

    boundary = _require_mapping(protocol.get("data_boundary"), "data_boundary")
    for key, expected in (
        ("allowed_split_roles", ["train", "test_metadata_only"]),
        ("test_split_use", "id_only_leakage_guard"),
    ):
        _require_equal(boundary.get(key), expected, f"data_boundary.{key}")
    for key in (
        "create_train_core",
        "create_source_val",
        "open_images",
        "open_masks",
        "modify_existing_split_files",
    ):
        _require_bool(boundary.get(key), False, f"data_boundary.{key}")

    selection = _require_mapping(protocol.get("selection"), "selection")
    _require_equal(selection.get("subset_size_per_dataset"), SUBSET_SIZE, "subset size")
    _require_equal(
        selection.get("parent_exclusion"),
        "round_02_corruption_severity_pilot_64",
        "selection parent exclusion",
    )
    _require_equal(selection.get("rule"), SELECTION_RULE, "selection rule")
    _require_equal(
        selection.get("canonical_id"),
        "posix_relative_identifier_without_final_suffix",
        "selection canonical ID",
    )
    _require_equal(
        selection.get("primary_sort_key"),
        "sha256_utf8_canonical_image_id_ascending",
        "selection primary sort key",
    )
    _require_equal(
        selection.get("tie_break"),
        "canonical_image_id_ascending",
        "selection tie break",
    )

    output = _require_mapping(protocol.get("output"), "output")
    for key, expected in (
        ("root", OUTPUT_RELATIVE_ROOT),
        ("format", "one_utf8_lf_id_per_line"),
        ("manifest", OUTPUT_MANIFEST),
    ):
        _require_equal(output.get(key), expected, f"output.{key}")
    _require_bool(
        output.get("atomic_directory_publish"),
        True,
        "output.atomic_directory_publish",
    )
    _require_bool(output.get("overwrite"), False, "output.overwrite")

    checkpoint = _require_mapping(protocol.get("checkpoint_anchor"), "checkpoint_anchor")
    _require_equal(checkpoint.get("role"), "best_miou", "checkpoint role")
    _require_equal(checkpoint.get("selection"), "test_selected", "checkpoint selection")
    _require_equal(
        checkpoint.get("best_pd_policy"),
        "reuse_same_frozen_hyperparameters_without_additional_tuning",
        "best_pd policy",
    )
    disclosure = checkpoint.get("disclosure")
    if not isinstance(disclosure, str) or "test" not in disclosure.casefold():
        raise ValueError("checkpoint disclosure must explicitly mention test selection")

    datasets = _require_mapping(protocol.get("datasets"), "datasets")
    _require_equal(tuple(datasets), DATASET_NAMES, "dataset order")
    return protocol, protocol_sha256


def _verify_parent_pilot(
    repository: Path,
    dataset_name: str,
    contract: Mapping[str, Any],
    *,
    train_sha256: str,
) -> dict[str, Any]:
    parent = _require_mapping(contract.get("round_02_parent_pilot"), "parent Pilot")
    pilot_path = _safe_repository_path(repository, parent.get("artifact"), "Pilot artifact")
    manifest_path = _safe_repository_path(repository, parent.get("manifest"), "Pilot manifest")
    complete_path = _safe_repository_path(repository, parent.get("complete"), "Pilot COMPLETE")
    pilot_payload, artifact_sha256 = _verified_snapshot(
        pilot_path, parent.get("artifact_sha256"), "parent Pilot artifact"
    )
    manifest_payload, manifest_sha256 = _verified_snapshot(
        manifest_path, parent.get("manifest_sha256"), "parent Pilot manifest"
    )
    complete_payload, complete_sha256 = _verified_snapshot(
        complete_path, parent.get("complete_sha256"), "parent Pilot COMPLETE"
    )
    pilot = _parse_json(pilot_payload, "parent Pilot artifact")
    manifest = _parse_json(manifest_payload, "parent Pilot manifest")
    complete = _parse_json(complete_payload, "parent Pilot COMPLETE")

    _require_equal(pilot.get("dataset"), dataset_name, "parent Pilot dataset")
    _require_equal(pilot.get("formal_artifact"), True, "parent Pilot formal flag")
    _require_equal(pilot.get("calibration_round"), 2, "parent Pilot round")
    _require_equal(pilot.get("split_sha256"), train_sha256, "parent Pilot train split")
    _require_equal(
        pilot.get("protocol_sha256"),
        _require_sha256(parent.get("protocol_sha256"), "parent Pilot protocol SHA256"),
        "parent Pilot protocol SHA256",
    )
    _require_equal(manifest.get("dataset"), dataset_name, "parent manifest dataset")
    _require_equal(
        _require_mapping(manifest.get("files_sha256"), "parent manifest files").get(
            "pilot.json"
        ),
        artifact_sha256,
        "parent manifest artifact lineage",
    )
    _require_equal(complete.get("complete"), True, "parent COMPLETE flag")
    _require_equal(complete.get("dataset"), dataset_name, "parent COMPLETE dataset")
    _require_equal(
        complete.get("pilot_json_sha256"), artifact_sha256, "parent COMPLETE artifact"
    )
    _require_equal(
        complete.get("artifact_manifest_sha256"),
        manifest_sha256,
        "parent COMPLETE manifest",
    )

    selection = _require_mapping(pilot.get("selection"), "parent Pilot selection")
    raw_ids = selection.get("selected_ids")
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise TypeError("parent Pilot selected_ids must be a sequence")
    selected_ids = tuple(
        canonical_image_id(str(value), label="parent Pilot selected_ids")
        for value in raw_ids
    )
    _require_equal(len(selected_ids), SUBSET_SIZE, "parent Pilot selected count")
    _require_equal(len(set(selected_ids)), SUBSET_SIZE, "parent Pilot ID uniqueness")
    selected_sha256 = ordered_ids_sha256(selected_ids)
    _require_equal(
        selected_sha256,
        _require_sha256(parent.get("ordered_ids_sha256"), "parent ordered ID SHA256"),
        "parent ordered ID SHA256",
    )
    _require_equal(
        selection.get("ordered_selected_ids_sha256"),
        selected_sha256,
        "parent artifact ordered ID SHA256",
    )
    return {
        "artifact": str(parent["artifact"]),
        "artifact_sha256": artifact_sha256,
        "manifest": str(parent["manifest"]),
        "manifest_sha256": manifest_sha256,
        "complete": str(parent["complete"]),
        "complete_sha256": complete_sha256,
        "protocol_sha256": str(parent["protocol_sha256"]),
        "ordered_ids_sha256": selected_sha256,
        "selected_ids": selected_ids,
    }


def build_contract(repository: Path, protocol_path: Path) -> dict[str, Any]:
    """Verify all input metadata and derive the exact output IDs."""

    repository = _absolute_lexical_path(repository)
    protocol_path = _absolute_lexical_path(protocol_path)
    protocol, protocol_sha256 = load_protocol(protocol_path)
    output = _require_mapping(protocol.get("output"), "output")
    output_root = _safe_repository_path(repository, output.get("root"), "output.root")
    if output_root.parent != repository / "configs":
        raise ValueError("output.root must be a direct child of configs/")

    reports: dict[str, Any] = {}
    datasets = _require_mapping(protocol["datasets"], "datasets")
    for dataset_name in DATASET_NAMES:
        contract = _require_mapping(datasets[dataset_name], f"datasets.{dataset_name}")
        train_path = _safe_repository_path(
            repository, contract.get("train_split"), f"{dataset_name} train split"
        )
        test_path = _safe_repository_path(
            repository, contract.get("test_split"), f"{dataset_name} test split"
        )
        train_payload, train_sha256 = _verified_snapshot(
            train_path, contract.get("train_split_sha256"), f"{dataset_name} train split"
        )
        test_payload, test_sha256 = _verified_snapshot(
            test_path, contract.get("test_split_sha256"), f"{dataset_name} test split"
        )
        train_ids = _parse_canonical_ids(
            train_payload, label=f"{dataset_name} train split"
        )
        test_ids = _parse_canonical_ids(
            test_payload, label=f"{dataset_name} test split"
        )
        _require_equal(len(train_ids), contract.get("train_images"), "train image count")
        _require_equal(len(test_ids), contract.get("test_images"), "test image count")
        train_set, test_set = set(train_ids), set(test_ids)
        overlap = train_set & test_set
        if overlap:
            raise ValueError(f"{dataset_name} train/test overlap: {sorted(overlap)[:5]}")

        parent = _verify_parent_pilot(
            repository,
            dataset_name,
            contract,
            train_sha256=train_sha256,
        )
        parent_ids = tuple(parent.pop("selected_ids"))
        parent_set = set(parent_ids)
        if not parent_set <= train_set:
            raise ValueError(f"{dataset_name} parent Pilot IDs are not contained in train")
        if parent_set & test_set:
            raise ValueError(f"{dataset_name} parent Pilot IDs overlap test")

        selected_ids = select_train_side_ids(train_ids, parent_ids)
        selected_set = set(selected_ids)
        if len(selected_set) != SUBSET_SIZE:
            raise ValueError(f"{dataset_name} output IDs are not unique")
        if not selected_set <= train_set:
            raise ValueError(f"{dataset_name} output IDs are not contained in train")
        if selected_set & parent_set:
            raise ValueError(f"{dataset_name} output IDs overlap parent Pilot")
        if selected_set & test_set:
            raise ValueError(f"{dataset_name} output IDs overlap test")

        selected_sha256 = ordered_ids_sha256(selected_ids)
        selected_bytes = serialize_ids(selected_ids)
        selected_file_sha256 = hashlib.sha256(selected_bytes).hexdigest()
        _require_equal(
            selected_sha256,
            _require_sha256(contract.get("output_ordered_ids_sha256"), "output ID SHA256"),
            f"{dataset_name} output ordered ID SHA256",
        )
        _require_equal(
            selected_file_sha256,
            _require_sha256(contract.get("output_file_sha256"), "output file SHA256"),
            f"{dataset_name} output file SHA256",
        )
        output_file = contract.get("output_file")
        if output_file != f"{dataset_name}.txt":
            raise ValueError(f"{dataset_name} output_file must be {dataset_name}.txt")
        reports[dataset_name] = {
            "train": {
                "path": str(contract["train_split"]),
                "sha256": train_sha256,
                "count": len(train_ids),
            },
            "test": {
                "path": str(contract["test_split"]),
                "sha256": test_sha256,
                "count": len(test_ids),
                "metadata_only": True,
            },
            "parent_pilot": parent,
            "output": {
                "path": f"{output['root']}/{output_file}",
                "file_name": output_file,
                "count": len(selected_ids),
                "ordered_ids_sha256": selected_sha256,
                "file_sha256": selected_file_sha256,
                "ids": selected_ids,
                "bytes": selected_bytes,
            },
            "checks": {
                "train_ids_unique": True,
                "test_ids_unique": True,
                "parent_pilot_ids_unique": True,
                "output_ids_unique": True,
                "train_test_overlap_count": 0,
                "parent_pilot_ids_contained_in_train": True,
                "output_ids_contained_in_train": True,
                "parent_pilot_output_overlap_count": 0,
                "parent_pilot_test_overlap_count": 0,
                "output_test_overlap_count": 0,
            },
        }

    return {
        "repository": repository,
        "protocol_path": protocol_path,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "output_root": output_root,
        "datasets": reports,
    }


def expected_manifest(context: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _require_mapping(context["protocol"], "protocol")
    datasets: dict[str, Any] = {}
    for dataset_name in DATASET_NAMES:
        report = context["datasets"][dataset_name]
        output = report["output"]
        datasets[dataset_name] = {
            "train_split": report["train"],
            "test_split": report["test"],
            "round_02_parent_pilot": report["parent_pilot"],
            "output": {
                key: output[key]
                for key in ("path", "count", "ordered_ids_sha256", "file_sha256")
            },
            "checks": report["checks"],
        }
    return {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "scope": SCOPE,
        "no_validation_split": True,
        "paper_result": False,
        "protocol": {
            "path": str(protocol.get("protocol_path")),
            "sha256": context["protocol_sha256"],
        },
        "selection": {
            "subset_size_per_dataset": SUBSET_SIZE,
            "rule": SELECTION_RULE,
            "parent_exclusion_role": "round_02_corruption_severity_pilot_64",
        },
        "checkpoint_anchor": dict(protocol["checkpoint_anchor"]),
        "metadata_io_boundary": {
            "read_id_text_only": True,
            "read_parent_pilot_metadata_only": True,
            "images_opened": 0,
            "masks_opened": 0,
            "test_pixels_opened": 0,
        },
        "datasets": datasets,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"destination exists; refusing overwrite: {destination}")
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise RuntimeError("atomic no-replace directory publication is unsupported")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def materialize(context: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish all ID files and the manifest, never overwriting."""

    destination = Path(context["output_root"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination exists; refusing overwrite: {destination}")
    parent = destination.parent
    try:
        _directory_names_nofollow(parent, label="output parent directory")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"output parent does not exist: {parent}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.build-", dir=parent))
    try:
        for dataset_name in DATASET_NAMES:
            output = context["datasets"][dataset_name]["output"]
            path = staging / output["file_name"]
            with path.open("xb") as stream:
                stream.write(output["bytes"])
                stream.flush()
                os.fsync(stream.fileno())
        manifest = expected_manifest(context)
        _write_json(staging / "manifest.json", manifest)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"destination appeared; refusing overwrite: {destination}")
        _atomic_rename_directory_noreplace(staging, destination)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if staging.is_dir() and staging.parent == parent:
            shutil.rmtree(staging)
        raise
    return validate_materialization(context)


def validate_materialization(context: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only validation of the already-published config artifacts."""

    destination = Path(context["output_root"])
    try:
        actual_names = _directory_names_nofollow(
            destination, label="materialized output directory"
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"materialized output is missing or unsafe: {destination}"
        ) from error
    expected_names = {f"{dataset_name}.txt" for dataset_name in DATASET_NAMES} | {
        "manifest.json"
    }
    _require_equal(actual_names, expected_names, "materialized file set")

    dataset_summary: dict[str, Any] = {}
    for dataset_name in DATASET_NAMES:
        expected = context["datasets"][dataset_name]["output"]
        path = destination / expected["file_name"]
        payload, actual_file_sha256 = _read_stable_bytes(
            path, label=f"{dataset_name} materialized IDs"
        )
        actual_ids = _parse_canonical_ids(
            payload, label=f"{dataset_name} materialized IDs"
        )
        _require_equal(actual_ids, expected["ids"], f"{dataset_name} materialized IDs")
        _require_equal(
            actual_file_sha256,
            expected["file_sha256"],
            f"{dataset_name} materialized file SHA256",
        )
        dataset_summary[dataset_name] = {
            "count": len(actual_ids),
            "ordered_ids_sha256": ordered_ids_sha256(actual_ids),
            "file_sha256": actual_file_sha256,
        }

    manifest_path = destination / "manifest.json"
    manifest_payload, manifest_sha256 = _read_stable_bytes(
        manifest_path, label="materialized manifest"
    )
    manifest = _parse_json(manifest_payload, "materialized manifest")
    _require_equal(manifest, expected_manifest(context), "materialized manifest")
    return {
        "valid": True,
        "read_only_validation": True,
        "scope": SCOPE,
        "protocol_sha256": context["protocol_sha256"],
        "output_root": str(destination),
        "manifest_sha256": manifest_sha256,
        "datasets": dataset_summary,
        "images_opened": 0,
        "masks_opened": 0,
    }
