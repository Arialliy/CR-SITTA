"""Frozen, CPU-only contract for P3 Stage-B1 gradient decomposition.

Loading this contract is deliberately metadata-only: it imports neither
PyTorch nor CUDA, constructs no model or optimizer, opens no dataset payload,
and creates no directory.  The complete YAML byte stream and its canonical
parsed mapping are independently SHA-256 pinned.  Duplicate YAML keys and any
missing, unknown, reordered, type-drifted, or value-drifted field fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Final

import yaml


PROTOCOL_ID: Final = "cr-sitta-p3-stage-b1-gradient-decomposition-v1"
CONFIG_RELATIVE_PATH: Final = "configs/p3_stage_b_gradient_decomposition_v1.yaml"
CONFIG_FILE_SHA256: Final = (
    "67a5c462ca092a7472badbbdca50a7b7fe9a2d2a43efea17c9bf3afb920e6a5f"
)
CONFIG_CANONICAL_MAPPING_SHA256: Final = (
    "ce1aac1127001c89dc7c53adf5d38dc6038bebf40d8fc0b572547fe7e831dedf"
)

DATASETS: Final = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CONDITIONS: Final = (
    "clean_S0",
    "gaussian_noise_S1",
    "gaussian_noise_S3",
    "gaussian_noise_S5",
    "gaussian_blur_S1",
    "gaussian_blur_S3",
    "gaussian_blur_S5",
    "low_contrast_S1",
    "low_contrast_S3",
    "low_contrast_S5",
    "stripe_noise_S1",
    "stripe_noise_S3",
    "stripe_noise_S5",
)
REPLICATE_IDS: Final = ("R0",)
PARAMETER_GROUP_SCALAR_COUNTS: Final = (
    ("P0", 8736),
    ("P1", 96),
    ("P2", 416),
    ("P3", 2080),
    ("P4", 2592),
)
PILOT64_ORDERED_ID_SHA256: Final = (
    (
        "IRSTD-1K",
        "0ee4400c021dcfb639d813bc20582060a3fff8bb635292e48abf820ce90abb13",
    ),
    (
        "NUAA-SIRST",
        "a48e5e3f0ec3e6911a74f953e96aa7ea362b2f2f32c37626099afd61f4935baa",
    ),
    (
        "NUDT-SIRST",
        "5c6c8b441eab5277611fa1007354692df557db311e2d7969e34ebeaba3ae00e4",
    ),
)
LIVE_PARENT_AGGREGATE_BINDING_KEYS: Final = (
    "d0_formal_stage_a_aggregate_complete",
    "d0_formal_stage_a_aggregate_manifest",
    "d0_formal_stage_a_science_decision",
)


class P3StageB1ContractError(ValueError):
    """The requested mapping or live input is not the frozen B1 contract."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise P3StageB1ContractError(
                "unhashable YAML mapping key is forbidden"
            ) from exc
        if duplicate:
            raise P3StageB1ContractError(
                f"duplicate YAML key is forbidden: {key!r}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(child) for child in value]
    return value


def _canonical_mapping_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            _thaw(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise P3StageB1ContractError(
            "B1 config must contain only finite JSON-compatible values"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


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


def _absolute_lexical_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_directory_nofollow(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    absolute = _absolute_lexical_path(path)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise P3StageB1ContractError(
                        f"contract path contains a symlink or non-directory: {absolute}"
                    ) from exc
                raise P3StageB1ContractError(
                    f"cannot open contract directory: {absolute}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_stable_regular_file(path: str | Path) -> tuple[bytes, str]:
    """Read one regular file without following ancestor or leaf symlinks."""

    absolute = _absolute_lexical_path(path)
    if absolute == Path("/") or not absolute.name:
        raise P3StageB1ContractError("contract input cannot be the filesystem root")
    parent_fd = _open_directory_nofollow(absolute.parent)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        try:
            descriptor = os.open(absolute.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise P3StageB1ContractError(
                f"cannot securely read regular file: {absolute}"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise P3StageB1ContractError(
                    f"contract input is not a regular file: {absolute}"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            data = b"".join(chunks)
            if _identity(before) != _identity(after) or len(data) != before.st_size:
                raise P3StageB1ContractError(
                    f"contract input changed while being read: {absolute}"
                )
            current = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
            if _identity(current) != _identity(before):
                raise P3StageB1ContractError(
                    f"contract file name changed while being read: {absolute}"
                )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)
    return data, hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, order=True)
class VerifiedFileBinding:
    key: str
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class P3StageB1Contract:
    schema_version: int
    protocol_id: str
    config_file_sha256: str | None
    datasets: tuple[str, ...]
    conditions: tuple[str, ...]
    replicate_ids: tuple[str, ...]
    parameter_group_scalar_counts: tuple[tuple[str, int], ...]
    pilot64_ordered_id_sha256: tuple[tuple[str, str], ...]
    output_root: str
    raw: Mapping[str, Any]

    @property
    def paper_result(self) -> bool:
        return bool(self.raw["scope"]["paper_result"])

    @property
    def stage_b3_authorized(self) -> bool:
        return bool(self.raw["scope"]["stage_b3_authorized"])

    def canonical_mapping_sha256(self) -> str:
        return _canonical_mapping_sha256(self.raw)


def parse_p3_stage_b1_contract(value: Mapping[str, Any]) -> P3StageB1Contract:
    """Parse exactly the frozen, byte-independent canonical B1 mapping."""

    if not isinstance(value, Mapping):
        raise P3StageB1ContractError("P3 Stage-B1 config must be a mapping")
    observed = _canonical_mapping_sha256(value)
    if observed != CONFIG_CANONICAL_MAPPING_SHA256:
        raise P3StageB1ContractError(
            "P3 Stage-B1 config schema/value drifted (missing, unknown, reordered, "
            f"type-changed, or value-changed field); expected="
            f"{CONFIG_CANONICAL_MAPPING_SHA256}, observed={observed}"
        )
    frozen = _deep_freeze(value)
    return P3StageB1Contract(
        schema_version=1,
        protocol_id=PROTOCOL_ID,
        config_file_sha256=None,
        datasets=DATASETS,
        conditions=CONDITIONS,
        replicate_ids=REPLICATE_IDS,
        parameter_group_scalar_counts=PARAMETER_GROUP_SCALAR_COUNTS,
        pilot64_ordered_id_sha256=PILOT64_ORDERED_ID_SHA256,
        output_root=str(frozen["output"]["root"]),
        raw=frozen,
    )


def load_p3_stage_b1_contract(path: str | Path) -> P3StageB1Contract:
    """Load the exact frozen YAML without creating directories or payloads."""

    data, digest = _read_stable_regular_file(path)
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise P3StageB1ContractError("P3 Stage-B1 YAML must be UTF-8") from exc
    try:
        value = yaml.load(decoded, Loader=_UniqueKeyLoader)
    except P3StageB1ContractError:
        raise
    except yaml.YAMLError as exc:
        raise P3StageB1ContractError("P3 Stage-B1 YAML is invalid") from exc
    contract = parse_p3_stage_b1_contract(value)
    if digest != CONFIG_FILE_SHA256:
        raise P3StageB1ContractError(
            "P3 Stage-B1 YAML bytes drifted; "
            f"expected={CONFIG_FILE_SHA256}, observed={digest}"
        )
    return replace(contract, config_file_sha256=digest)


def _assert_live_contract(contract: P3StageB1Contract) -> None:
    if not isinstance(contract, P3StageB1Contract):
        raise P3StageB1ContractError("contract must be P3StageB1Contract")
    if contract.canonical_mapping_sha256() != CONFIG_CANONICAL_MAPPING_SHA256:
        raise P3StageB1ContractError("in-memory P3 Stage-B1 contract drifted")


def _binding_from_mapping(
    key: str, value: Mapping[str, Any]
) -> VerifiedFileBinding:
    relative_path = str(value["path"])
    expected_sha256 = str(value["sha256"])
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise P3StageB1ContractError(
            f"frozen binding path is not canonical project-relative: {relative_path}"
        )
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise P3StageB1ContractError(f"invalid SHA-256 in frozen binding: {key}")
    return VerifiedFileBinding(key, relative_path, expected_sha256)


def _verify_bindings(
    contract: P3StageB1Contract,
    *,
    repository_root: str | Path,
    binding_keys: tuple[str, ...],
) -> tuple[VerifiedFileBinding, ...]:
    _assert_live_contract(contract)
    root = _absolute_lexical_path(repository_root)
    parent = contract.raw["frozen_parent_bindings"]
    verified: list[VerifiedFileBinding] = []
    for key in binding_keys:
        binding = _binding_from_mapping(key, parent[key])
        _, observed = _read_stable_regular_file(root / binding.relative_path)
        if observed != binding.sha256:
            raise P3StageB1ContractError(
                f"frozen live binding drifted: {binding.relative_path}; "
                f"expected={binding.sha256}, observed={observed}"
            )
        verified.append(binding)
    return tuple(verified)


def verify_live_parent_aggregate(
    contract: P3StageB1Contract,
    *,
    repository_root: str | Path,
) -> tuple[VerifiedFileBinding, ...]:
    """Hash-verify the live Stage-A R0 COMPLETE, manifest, and decision receipt."""

    return _verify_bindings(
        contract,
        repository_root=repository_root,
        binding_keys=LIVE_PARENT_AGGREGATE_BINDING_KEYS,
    )


def verify_all_frozen_file_bindings(
    contract: P3StageB1Contract,
    *,
    repository_root: str | Path,
) -> tuple[VerifiedFileBinding, ...]:
    """Hash-verify the design document and every frozen config/artifact input."""

    _assert_live_contract(contract)
    root = _absolute_lexical_path(repository_root)
    bindings = contract.raw["frozen_parent_bindings"]
    verified = list(
        _verify_bindings(
            contract,
            repository_root=root,
            binding_keys=tuple(bindings.keys()),
        )
    )
    design = _binding_from_mapping(
        "stage_b_v5_document", contract.raw["frozen_design_binding"]["document"]
    )
    _, observed = _read_stable_regular_file(root / design.relative_path)
    if observed != design.sha256:
        raise P3StageB1ContractError(
            f"frozen live binding drifted: {design.relative_path}; "
            f"expected={design.sha256}, observed={observed}"
        )
    return (design, *verified)


__all__ = [
    "CONDITIONS",
    "CONFIG_CANONICAL_MAPPING_SHA256",
    "CONFIG_FILE_SHA256",
    "CONFIG_RELATIVE_PATH",
    "DATASETS",
    "LIVE_PARENT_AGGREGATE_BINDING_KEYS",
    "PARAMETER_GROUP_SCALAR_COUNTS",
    "PILOT64_ORDERED_ID_SHA256",
    "PROTOCOL_ID",
    "P3StageB1Contract",
    "P3StageB1ContractError",
    "REPLICATE_IDS",
    "VerifiedFileBinding",
    "load_p3_stage_b1_contract",
    "parse_p3_stage_b1_contract",
    "verify_all_frozen_file_bindings",
    "verify_live_parent_aggregate",
]
