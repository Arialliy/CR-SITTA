"""Frozen, CPU-only contract for P3 formal Stage-A.

The v2 engineering-smoke configuration and artifact are immutable parents.
This module adds a new formal protocol without importing CUDA, constructing a
model, touching a dataset, or creating an output directory.  Both the YAML
bytes and its canonical parsed mapping are digest-pinned.  Consequently every
missing, unknown, reordered sequence, type-drifted, or value-drifted field
fails closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any

import yaml


PROTOCOL_ID = "cr-sitta-d0-v3-formal-stage-a"
CONFIG_RELATIVE_PATH = "configs/tent_failure_diagnostics_v3_formal_stage_a.yaml"
CONFIG_FILE_SHA256 = "f0ed056520f840d1432b90b6426f55803afa1867b98108712dc242371dd1c05a"
CONFIG_CANONICAL_MAPPING_SHA256 = (
    "2a586df9de8cf99cca99f41ba350718ba34f46803ecfcd390ee5a6e569af9b89"
)
BASE_ENGINEERING_CONFIG_SHA256 = (
    "a839af3599111854196548cfa0b05ab5b3b4ba3f60ab0829ee5a827ce871728d"
)
ENGINEERING_SMOKE_AGGREGATE_SHA256 = (
    "9a46508c5c10478f9d2ef3b0a359435db90d93a1a3cc3a9662c3764c46ec27c2"
)

DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CONDITIONS = (
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
FINE_ALIGNMENT_GROUP_IDS = (
    "encoder_0",
    "encoder_1",
    "encoder_2",
    "encoder_3",
    "middle",
    "fpn_lateral_0",
    "fpn_lateral_1",
    "fpn_lateral_2",
    "fpn_lateral_3",
    "fpn_output_0",
    "fpn_output_1",
    "fpn_output_2",
    "fpn_output_3",
    "fpn_sfs_0",
    "fpn_sfs_1",
    "fpn_sfs_2",
    "decoder_3",
    "decoder_2",
    "decoder_1",
    "decoder_0",
)


class D0V3FormalContractError(ValueError):
    """The requested formal contract is not the frozen P3 Stage-A contract."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise D0V3FormalContractError(f"duplicate YAML key is forbidden: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, order=True)
class FormalCandidate:
    candidate_id: str
    optimizer: str
    learning_rate: float


FROZEN_CANDIDATES = (
    FormalCandidate("Adam_lr_1em5", "Adam", 1.0e-5),
    FormalCandidate("Adam_lr_3em5", "Adam", 3.0e-5),
    FormalCandidate("Adam_lr_1em4", "Adam", 1.0e-4),
    FormalCandidate("Adam_lr_3em4", "Adam", 3.0e-4),
    FormalCandidate("Adam_lr_1em3", "Adam", 1.0e-3),
    FormalCandidate("SGD_lr_1em5", "SGD", 1.0e-5),
    FormalCandidate("SGD_lr_3em5", "SGD", 3.0e-5),
    FormalCandidate("SGD_lr_1em4", "SGD", 1.0e-4),
    FormalCandidate("SGD_lr_3em4", "SGD", 3.0e-4),
    FormalCandidate("SGD_lr_1em3", "SGD", 1.0e-3),
)


def _canonical_mapping_sha256(value: Any) -> str:
    def thaw(child: Any) -> Any:
        if isinstance(child, Mapping):
            return {key: thaw(item) for key, item in child.items()}
        if isinstance(child, tuple):
            return [thaw(item) for item in child]
        if isinstance(child, list):
            return [thaw(item) for item in child]
        return child

    try:
        payload = json.dumps(
            thaw(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise D0V3FormalContractError(
            "formal config must contain only finite JSON-compatible values"
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


def _read_regular_file_no_follow(path: str | Path) -> tuple[bytes, str]:
    """Read one stable regular file without following a final symlink."""

    resolved = os.fspath(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise D0V3FormalContractError(
            f"cannot securely read regular file: {resolved}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise D0V3FormalContractError(
                f"contract input is not a regular file: {resolved}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    data = b"".join(chunks)
    if identity_before != identity_after or len(data) != before.st_size:
        raise D0V3FormalContractError(
            f"contract input changed while being read: {resolved}"
        )
    return data, hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class D0V3FormalContract:
    schema_version: int
    protocol_id: str
    config_file_sha256: str | None
    candidates: tuple[FormalCandidate, ...]
    datasets: tuple[str, ...]
    conditions: tuple[str, ...]
    fine_alignment_group_ids: tuple[str, ...]
    output_root: str
    raw: Mapping[str, Any]

    @property
    def stage2_authorized(self) -> bool:
        return bool(self.raw["status_contract"]["stage2_authorized"])

    @property
    def formal_protocol_complete_initial(self) -> bool:
        return bool(self.raw["scope"]["formal_protocol_complete_initial"])

    @property
    def scientific_status_initial(self) -> str:
        return str(self.raw["scope"]["scientific_status_initial"])

    def canonical_mapping_sha256(self) -> str:
        return _canonical_mapping_sha256(self.raw)


def parse_d0_v3_formal_contract(value: Mapping[str, Any]) -> D0V3FormalContract:
    """Parse only the byte-independent canonical frozen mapping.

    A cryptographic identity over the complete nested mapping is stricter than
    a permissive field parser: even a single unknown nested key, YAML scalar
    type change, list reordering, or changed disclosure is rejected.
    """

    if not isinstance(value, Mapping):
        raise D0V3FormalContractError("D0-v3 formal config must be a mapping")
    observed = _canonical_mapping_sha256(value)
    if observed != CONFIG_CANONICAL_MAPPING_SHA256:
        raise D0V3FormalContractError(
            "D0-v3 formal config schema/value drifted (missing, unknown, "
            "reordered, type-changed, or value-changed field); "
            f"expected={CONFIG_CANONICAL_MAPPING_SHA256}, observed={observed}"
        )
    frozen = _deep_freeze(value)
    return D0V3FormalContract(
        schema_version=3,
        protocol_id=PROTOCOL_ID,
        config_file_sha256=None,
        candidates=FROZEN_CANDIDATES,
        datasets=DATASETS,
        conditions=CONDITIONS,
        fine_alignment_group_ids=FINE_ALIGNMENT_GROUP_IDS,
        output_root=str(frozen["output"]["root"]),
        raw=frozen,
    )


def load_d0_v3_formal_contract(path: str | Path) -> D0V3FormalContract:
    """Load the exact frozen YAML without constructing any output path."""

    data, digest = _read_regular_file_no_follow(path)
    if digest != CONFIG_FILE_SHA256:
        raise D0V3FormalContractError(
            "D0-v3 formal YAML bytes drifted; "
            f"expected={CONFIG_FILE_SHA256}, observed={digest}"
        )
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise D0V3FormalContractError("D0-v3 formal YAML must be UTF-8") from exc
    try:
        value = yaml.load(decoded, Loader=_UniqueKeyLoader)
    except D0V3FormalContractError:
        raise
    except yaml.YAMLError as exc:
        raise D0V3FormalContractError("D0-v3 formal YAML is invalid") from exc
    contract = parse_d0_v3_formal_contract(value)
    return replace(contract, config_file_sha256=digest)


def verify_frozen_parent_bindings(
    contract: D0V3FormalContract,
    *,
    repository_root: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Verify the v2 config and completed engineering-smoke aggregate read-only."""

    if not isinstance(contract, D0V3FormalContract):
        raise D0V3FormalContractError("contract must be D0V3FormalContract")
    if contract.canonical_mapping_sha256() != CONFIG_CANONICAL_MAPPING_SHA256:
        raise D0V3FormalContractError("in-memory D0-v3 formal contract drifted")
    root = Path(repository_root)
    bindings = (
        (
            "configs/tent_failure_diagnostics_v2_independent_candidates.yaml",
            BASE_ENGINEERING_CONFIG_SHA256,
        ),
        (
            "results/cr_sitta/tent_failure_diagnostics_v2_independent_candidates/"
            "engineering_smoke/aggregate.json",
            ENGINEERING_SMOKE_AGGREGATE_SHA256,
        ),
    )
    verified: list[tuple[str, str]] = []
    for relative, expected in bindings:
        _, observed = _read_regular_file_no_follow(root / relative)
        if observed != expected:
            raise D0V3FormalContractError(
                f"frozen parent binding drifted: {relative}; "
                f"expected={expected}, observed={observed}"
            )
        verified.append((relative, observed))
    return tuple(verified)


__all__ = [
    "BASE_ENGINEERING_CONFIG_SHA256",
    "CONDITIONS",
    "CONFIG_CANONICAL_MAPPING_SHA256",
    "CONFIG_FILE_SHA256",
    "CONFIG_RELATIVE_PATH",
    "DATASETS",
    "D0V3FormalContract",
    "D0V3FormalContractError",
    "ENGINEERING_SMOKE_AGGREGATE_SHA256",
    "FINE_ALIGNMENT_GROUP_IDS",
    "FROZEN_CANDIDATES",
    "FormalCandidate",
    "PROTOCOL_ID",
    "load_d0_v3_formal_contract",
    "parse_d0_v3_formal_contract",
    "verify_frozen_parent_bindings",
]
