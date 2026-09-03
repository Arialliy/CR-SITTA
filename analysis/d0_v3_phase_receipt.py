"""Cryptographic phase firewall between formal P3 candidates and outer targets.

The existing cache loader has a legacy ``episodes_complete: bool`` switch.  A
boolean is not evidence that the formal label-free phase actually completed,
so formal P3 code must enter the loader through :func:`guarded_load_outer_targets`.
The guard accepts only canonical immutable bytes or a stable no-follow file and
requires one exact receipt containing all 64 x 10 candidate episodes.

This module is additive and deliberately does not modify the sealed D0-v1 or
engineering-smoke implementation.  A receipt proves ordering and observed
access counters; it does not authorize Stage 2 or turn train-only diagnostics
into a paper result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

from analysis.d0_v2_independent_candidate_contract import FROZEN_CANDIDATES
from materialize_binary_tent_ss_calibration_cache_v2 import (
    load_outer_evaluator_targets_v2,
)
from tta.d0_secure_io import read_stable_regular_file


SCHEMA_VERSION = 3
PROTOCOL_ID = "cr-sitta-p3-formal-candidate-outer-phase-firewall-v1"
LABEL_FREE_ARTIFACT_TYPE = "cr_sitta_p3_label_free_cell_phase_receipt"
OUTER_ACCESS_ARTIFACT_TYPE = "cr_sitta_p3_outer_target_access_receipt"
IMAGE_COUNT = 64
CANDIDATE_COUNT = 10
EPISODE_COUNT = IMAGE_COUNT * CANDIDATE_COUNT
EPISODE_ORDER = "image_major_candidate_minor"
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

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ROOT_FIELDS = {
    "schema_version",
    "artifact_type",
    "protocol_id",
    "cell_binding",
    "pilot",
    "candidate_grid",
    "episodes",
    "seals",
    "candidate_phase_data_boundary",
    "completion",
    "authorization",
}
_CELL_FIELDS = {"dataset", "condition", "replicate", "split_name", "split_role"}
_PILOT_FIELDS = {
    "image_count",
    "ordered_image_ids",
    "ordered_image_ids_sha256",
}
_CANDIDATE_GRID_FIELDS = {
    "candidate_count",
    "candidates",
    "candidate_order_sha256",
}
_CANDIDATE_FIELDS = {"index", "optimizer", "learning_rate", "slug"}
_EPISODE_SECTION_FIELDS = {
    "expected_count",
    "complete_count",
    "order",
    "records",
    "episode_receipt_hashes_sha256",
    "episode_ledger_sha256",
}
_EPISODE_FIELDS = {
    "image_index",
    "image_id",
    "candidate_index",
    "candidate_slug",
    "complete",
    "episode_receipt_sha256",
}
_SEAL_FIELDS = {
    "cache_protocol_sha256",
    "pilot_ids_sha256",
    "cache_method_manifest_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "code_files",
    "code_bundle_sha256",
}
_CODE_FILE_FIELDS = {"path", "sha256"}
_BOUNDARY_FIELDS = {
    "train_target_payload_bytes_opened",
    "train_target_payload_deserialization_count",
    "train_target_indexing_count",
    "candidate_phase_outer_target_loader_calls",
    "test_split_files_opened",
    "test_images_opened",
    "test_masks_opened",
    "test_labels_opened",
    "method_label_accesses",
    "target_payload_deserialized_during_candidate_phase",
}
_COMPLETION_FIELDS = {
    "label_free_phase_complete",
    "ordered_images_complete",
    "candidate_grid_complete",
    "episode_receipts_complete",
    "complete_episode_count",
}
_AUTHORIZATION_FIELDS = {
    "source_train_derived",
    "paper_result",
    "paper_test_result",
    "scientific_selection_performed",
    "stage2_authorized",
}


class D0V3PhaseReceiptError(ValueError):
    """A formal P3 phase receipt is incomplete, unbound, or unsafe."""


@dataclass(frozen=True, slots=True)
class ValidatedLabelFreeCellReceipt:
    """Immutable validated view of one canonical 64 x 10 cell receipt."""

    dataset: str
    condition: str
    replicate: int
    ordered_image_ids: tuple[str, ...]
    pilot_ids_sha256: str
    cache_protocol_sha256: str
    cache_method_manifest_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    code_files: tuple[tuple[str, str], ...]
    episode_receipt_sha256s: tuple[str, ...]
    canonical_bytes: bytes
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of the validated receipt."""

        value = json.loads(self.canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise AssertionError("validated receipt root stopped being a mapping")
        return value


@dataclass(frozen=True, slots=True)
class GuardedOuterTargetLoad:
    """Outer targets plus a deterministic non-adaptation access receipt."""

    targets: Any
    access_receipt_bytes: bytes
    access_receipt_sha256: str
    label_free_receipt_sha256: str

    def access_receipt(self) -> dict[str, Any]:
        value = json.loads(self.access_receipt_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise AssertionError("outer access receipt root stopped being a mapping")
        return value


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
        raise D0V3PhaseReceiptError("receipt must contain only finite JSON values") from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0V3PhaseReceiptError(f"{field} must be a mapping")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise D0V3PhaseReceiptError(f"{field} must be a sequence")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, field: str
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise D0V3PhaseReceiptError(
            f"{field} fields must be exact; missing={missing}, unknown={unknown}"
        )


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise D0V3PhaseReceiptError(f"{field} must be a non-empty trimmed string")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise D0V3PhaseReceiptError(f"{field} contains a forbidden control character")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise D0V3PhaseReceiptError(f"{field} must be lowercase 64-hex SHA-256")
    return value


def _require_int(value: Any, *, field: str, exact: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise D0V3PhaseReceiptError(f"{field} must be an integer")
    if exact is not None and value != exact:
        raise D0V3PhaseReceiptError(f"{field} must be exactly {exact}")
    if value < 0:
        raise D0V3PhaseReceiptError(f"{field} must be non-negative")
    return value


def _require_bool(value: Any, *, field: str, expected: bool) -> bool:
    if value is not expected:
        raise D0V3PhaseReceiptError(f"{field} must be exactly {expected}")
    return expected


def _ordered_ids_sha256(image_ids: Sequence[str]) -> str:
    # This intentionally matches dataio.train_side_pilot_protocol exactly.
    payload = json.dumps(
        list(image_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_path(value: Any, *, field: str) -> str:
    path = _require_string(value, field=field)
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or path != parsed.as_posix()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise D0V3PhaseReceiptError(
            f"{field} must be a canonical repository-relative POSIX path"
        )
    return path


def zero_candidate_phase_data_boundary() -> dict[str, Any]:
    """Create the exact zero-access observation required before target loading."""

    return {
        "train_target_payload_bytes_opened": 0,
        "train_target_payload_deserialization_count": 0,
        "train_target_indexing_count": 0,
        "candidate_phase_outer_target_loader_calls": 0,
        "test_split_files_opened": 0,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "test_labels_opened": 0,
        "method_label_accesses": 0,
        "target_payload_deserialized_during_candidate_phase": False,
    }


def _candidate_records() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "optimizer": candidate.optimizer,
            "learning_rate": candidate.learning_rate,
            "slug": candidate.slug,
        }
        for index, candidate in enumerate(FROZEN_CANDIDATES)
    ]


def _canonical_code_files(value: Mapping[str, str]) -> list[dict[str, str]]:
    mapping = _require_mapping(value, field="code_seals")
    if not mapping:
        raise D0V3PhaseReceiptError("code_seals must bind at least one code file")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_path, raw_sha256 in mapping.items():
        path = _code_path(raw_path, field="code_seals path")
        if path in seen:
            raise D0V3PhaseReceiptError(f"duplicate code seal path {path!r}")
        seen.add(path)
        records.append(
            {
                "path": path,
                "sha256": _require_sha256(
                    raw_sha256, field=f"code_seals[{path!r}]"
                ),
            }
        )
    return sorted(records, key=lambda record: record["path"])


def _validate_boundary(value: Any) -> dict[str, Any]:
    boundary = _require_mapping(value, field="candidate_phase_data_boundary")
    _require_exact_keys(
        boundary, _BOUNDARY_FIELDS, field="candidate_phase_data_boundary"
    )
    for field in sorted(_BOUNDARY_FIELDS - {"target_payload_deserialized_during_candidate_phase"}):
        _require_int(boundary[field], field=field, exact=0)
    _require_bool(
        boundary["target_payload_deserialized_during_candidate_phase"],
        field="target_payload_deserialized_during_candidate_phase",
        expected=False,
    )
    return dict(boundary)


def _normalize_episode_records(
    records: Sequence[Mapping[str, Any]], ordered_image_ids: Sequence[str]
) -> list[dict[str, Any]]:
    sequence = _require_sequence(records, field="episode_records")
    if len(sequence) != EPISODE_COUNT:
        raise D0V3PhaseReceiptError(
            f"episode_records must contain exactly {EPISODE_COUNT} records"
        )
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    candidates = _candidate_records()
    for position, raw in enumerate(sequence):
        record = _require_mapping(raw, field=f"episode_records[{position}]")
        _require_exact_keys(record, _EPISODE_FIELDS, field=f"episode_records[{position}]")
        image_index = _require_int(
            record["image_index"], field=f"episode_records[{position}].image_index"
        )
        candidate_index = _require_int(
            record["candidate_index"],
            field=f"episode_records[{position}].candidate_index",
        )
        if image_index >= IMAGE_COUNT or candidate_index >= CANDIDATE_COUNT:
            raise D0V3PhaseReceiptError(
                f"episode_records[{position}] index is outside the 64 x 10 grid"
            )
        image_id = _require_string(
            record["image_id"], field=f"episode_records[{position}].image_id"
        )
        if image_id != ordered_image_ids[image_index]:
            raise D0V3PhaseReceiptError(
                f"episode_records[{position}] image_id does not match Pilot order"
            )
        candidate_slug = _require_string(
            record["candidate_slug"],
            field=f"episode_records[{position}].candidate_slug",
        )
        if candidate_slug != candidates[candidate_index]["slug"]:
            raise D0V3PhaseReceiptError(
                f"episode_records[{position}] candidate_slug/index mismatch"
            )
        _require_bool(
            record["complete"],
            field=f"episode_records[{position}].complete",
            expected=True,
        )
        episode_sha = _require_sha256(
            record["episode_receipt_sha256"],
            field=f"episode_records[{position}].episode_receipt_sha256",
        )
        key = (image_index, candidate_index)
        if key in by_key:
            raise D0V3PhaseReceiptError(f"duplicate episode grid key {key}")
        by_key[key] = {
            "image_index": image_index,
            "image_id": image_id,
            "candidate_index": candidate_index,
            "candidate_slug": candidate_slug,
            "complete": True,
            "episode_receipt_sha256": episode_sha,
        }
    expected_keys = {
        (image_index, candidate_index)
        for image_index in range(IMAGE_COUNT)
        for candidate_index in range(CANDIDATE_COUNT)
    }
    if set(by_key) != expected_keys:
        raise D0V3PhaseReceiptError("episode_records do not cover the exact 64 x 10 grid")
    ordered = [by_key[key] for key in sorted(expected_keys)]
    hashes = [record["episode_receipt_sha256"] for record in ordered]
    if len(set(hashes)) != EPISODE_COUNT:
        raise D0V3PhaseReceiptError(
            "all 640 episode receipt hashes must be unique and independently bound"
        )
    return ordered


def build_label_free_cell_receipt(
    *,
    dataset: str,
    condition: str,
    replicate: int,
    ordered_image_ids: Sequence[str],
    episode_records: Sequence[Mapping[str, Any]],
    cache_protocol_sha256: str,
    cache_method_manifest_sha256: str,
    checkpoint_sha256: str,
    config_sha256: str,
    code_seals: Mapping[str, str],
    candidate_phase_data_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and self-validate one deterministic label-free cell receipt.

    ``episode_records`` may arrive in any order, but must contain each image x
    candidate key exactly once.  The persisted receipt is always image-major,
    candidate-minor and therefore has a single canonical byte representation.
    """

    dataset_value = _require_string(dataset, field="dataset")
    if dataset_value not in DATASETS:
        raise D0V3PhaseReceiptError(f"unsupported formal P3 dataset {dataset_value!r}")
    condition_value = _require_string(condition, field="condition")
    if condition_value not in CONDITIONS:
        raise D0V3PhaseReceiptError(
            f"unsupported formal P3 condition {condition_value!r}"
        )
    replicate_value = _require_int(replicate, field="replicate")
    raw_ids = _require_sequence(ordered_image_ids, field="ordered_image_ids")
    image_ids = [
        _require_string(value, field=f"ordered_image_ids[{index}]")
        for index, value in enumerate(raw_ids)
    ]
    if len(image_ids) != IMAGE_COUNT or len(set(image_ids)) != IMAGE_COUNT:
        raise D0V3PhaseReceiptError(
            "ordered_image_ids must contain exactly 64 unique Pilot IDs"
        )
    pilot_ids_sha256 = _ordered_ids_sha256(image_ids)
    candidates = _candidate_records()
    episodes = _normalize_episode_records(episode_records, image_ids)
    code_files = _canonical_code_files(code_seals)
    boundary = _validate_boundary(candidate_phase_data_boundary)
    episode_hashes = [record["episode_receipt_sha256"] for record in episodes]
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": LABEL_FREE_ARTIFACT_TYPE,
        "protocol_id": PROTOCOL_ID,
        "cell_binding": {
            "dataset": dataset_value,
            "condition": condition_value,
            "replicate": replicate_value,
            "split_name": "train",
            "split_role": "frozen_pilot64",
        },
        "pilot": {
            "image_count": IMAGE_COUNT,
            "ordered_image_ids": image_ids,
            "ordered_image_ids_sha256": pilot_ids_sha256,
        },
        "candidate_grid": {
            "candidate_count": CANDIDATE_COUNT,
            "candidates": candidates,
            "candidate_order_sha256": _canonical_sha256(candidates),
        },
        "episodes": {
            "expected_count": EPISODE_COUNT,
            "complete_count": EPISODE_COUNT,
            "order": EPISODE_ORDER,
            "records": episodes,
            "episode_receipt_hashes_sha256": _canonical_sha256(episode_hashes),
            "episode_ledger_sha256": _canonical_sha256(episodes),
        },
        "seals": {
            "cache_protocol_sha256": _require_sha256(
                cache_protocol_sha256, field="cache_protocol_sha256"
            ),
            "pilot_ids_sha256": pilot_ids_sha256,
            "cache_method_manifest_sha256": _require_sha256(
                cache_method_manifest_sha256,
                field="cache_method_manifest_sha256",
            ),
            "checkpoint_sha256": _require_sha256(
                checkpoint_sha256, field="checkpoint_sha256"
            ),
            "config_sha256": _require_sha256(
                config_sha256, field="config_sha256"
            ),
            "code_files": code_files,
            "code_bundle_sha256": _canonical_sha256(code_files),
        },
        "candidate_phase_data_boundary": boundary,
        "completion": {
            "label_free_phase_complete": True,
            "ordered_images_complete": True,
            "candidate_grid_complete": True,
            "episode_receipts_complete": True,
            "complete_episode_count": EPISODE_COUNT,
        },
        "authorization": {
            "source_train_derived": True,
            "paper_result": False,
            "paper_test_result": False,
            "scientific_selection_performed": False,
            "stage2_authorized": False,
        },
    }
    return validate_label_free_cell_receipt(receipt).to_dict()


def _receipt_mapping(value: Mapping[str, Any] | bytes) -> tuple[Mapping[str, Any], bytes]:
    if isinstance(value, bytes):
        try:
            decoded = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise D0V3PhaseReceiptError("receipt bytes must be UTF-8 JSON") from error
        mapping = _require_mapping(decoded, field="receipt")
        canonical = _canonical_json_bytes(mapping)
        if value != canonical:
            raise D0V3PhaseReceiptError(
                "receipt bytes are not the exact canonical representation"
            )
        return mapping, canonical
    mapping = _require_mapping(value, field="receipt")
    return mapping, _canonical_json_bytes(mapping)


def validate_label_free_cell_receipt(
    value: Mapping[str, Any] | bytes,
    *,
    expected_dataset: str | None = None,
    expected_condition: str | None = None,
    expected_replicate: int | None = None,
    expected_cache_protocol_sha256: str | None = None,
    expected_cache_method_manifest_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_code_seals: Mapping[str, str] | None = None,
    expected_ordered_image_ids: Sequence[str] | None = None,
    expected_receipt_sha256: str | None = None,
) -> ValidatedLabelFreeCellReceipt:
    """Strictly validate the full receipt and optional live-run bindings."""

    root, canonical = _receipt_mapping(value)
    _require_exact_keys(root, _ROOT_FIELDS, field="receipt")
    _require_int(root["schema_version"], field="schema_version", exact=SCHEMA_VERSION)
    if root["artifact_type"] != LABEL_FREE_ARTIFACT_TYPE:
        raise D0V3PhaseReceiptError("artifact_type is not a P3 label-free receipt")
    if root["protocol_id"] != PROTOCOL_ID:
        raise D0V3PhaseReceiptError("protocol_id is not the P3 phase firewall")

    cell = _require_mapping(root["cell_binding"], field="cell_binding")
    _require_exact_keys(cell, _CELL_FIELDS, field="cell_binding")
    dataset = _require_string(cell["dataset"], field="cell_binding.dataset")
    condition = _require_string(cell["condition"], field="cell_binding.condition")
    replicate = _require_int(cell["replicate"], field="cell_binding.replicate")
    if dataset not in DATASETS:
        raise D0V3PhaseReceiptError(f"unsupported formal P3 dataset {dataset!r}")
    if condition not in CONDITIONS:
        raise D0V3PhaseReceiptError(f"unsupported formal P3 condition {condition!r}")
    if cell["split_name"] != "train" or cell["split_role"] != "frozen_pilot64":
        raise D0V3PhaseReceiptError("cell must bind only the frozen train Pilot64")

    pilot = _require_mapping(root["pilot"], field="pilot")
    _require_exact_keys(pilot, _PILOT_FIELDS, field="pilot")
    _require_int(pilot["image_count"], field="pilot.image_count", exact=IMAGE_COUNT)
    raw_ids = _require_sequence(pilot["ordered_image_ids"], field="ordered_image_ids")
    image_ids = tuple(
        _require_string(value, field=f"ordered_image_ids[{index}]")
        for index, value in enumerate(raw_ids)
    )
    if len(image_ids) != IMAGE_COUNT or len(set(image_ids)) != IMAGE_COUNT:
        raise D0V3PhaseReceiptError(
            "ordered_image_ids must contain exactly 64 unique Pilot IDs"
        )
    pilot_ids_sha256 = _require_sha256(
        pilot["ordered_image_ids_sha256"], field="ordered_image_ids_sha256"
    )
    if pilot_ids_sha256 != _ordered_ids_sha256(image_ids):
        raise D0V3PhaseReceiptError("ordered Pilot ID SHA-256 does not match the IDs")

    grid = _require_mapping(root["candidate_grid"], field="candidate_grid")
    _require_exact_keys(grid, _CANDIDATE_GRID_FIELDS, field="candidate_grid")
    _require_int(
        grid["candidate_count"], field="candidate_grid.candidate_count", exact=CANDIDATE_COUNT
    )
    raw_candidates = _require_sequence(grid["candidates"], field="candidates")
    candidates: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _require_mapping(raw_candidate, field=f"candidates[{index}]")
        _require_exact_keys(candidate, _CANDIDATE_FIELDS, field=f"candidates[{index}]")
        _require_int(candidate["index"], field=f"candidates[{index}].index", exact=index)
        if not isinstance(candidate["learning_rate"], float):
            raise D0V3PhaseReceiptError(
                f"candidates[{index}].learning_rate must remain a JSON float"
            )
        candidates.append(dict(candidate))
    if candidates != _candidate_records():
        raise D0V3PhaseReceiptError("candidate grid differs from the frozen ten candidates")
    candidate_order_sha = _require_sha256(
        grid["candidate_order_sha256"], field="candidate_order_sha256"
    )
    if candidate_order_sha != _canonical_sha256(candidates):
        raise D0V3PhaseReceiptError("candidate_order_sha256 mismatch")

    episode_section = _require_mapping(root["episodes"], field="episodes")
    _require_exact_keys(episode_section, _EPISODE_SECTION_FIELDS, field="episodes")
    _require_int(
        episode_section["expected_count"],
        field="episodes.expected_count",
        exact=EPISODE_COUNT,
    )
    _require_int(
        episode_section["complete_count"],
        field="episodes.complete_count",
        exact=EPISODE_COUNT,
    )
    if episode_section["order"] != EPISODE_ORDER:
        raise D0V3PhaseReceiptError(f"episodes.order must be {EPISODE_ORDER}")
    raw_episodes = _require_sequence(episode_section["records"], field="episodes.records")
    normalized_episodes = _normalize_episode_records(raw_episodes, image_ids)
    if list(raw_episodes) != normalized_episodes:
        raise D0V3PhaseReceiptError(
            "episode records must be stored in canonical image-major candidate-minor order"
        )
    episode_hashes = [
        record["episode_receipt_sha256"] for record in normalized_episodes
    ]
    expected_hash_list_sha = _require_sha256(
        episode_section["episode_receipt_hashes_sha256"],
        field="episode_receipt_hashes_sha256",
    )
    if expected_hash_list_sha != _canonical_sha256(episode_hashes):
        raise D0V3PhaseReceiptError("episode receipt hash-list seal mismatch")
    ledger_sha = _require_sha256(
        episode_section["episode_ledger_sha256"], field="episode_ledger_sha256"
    )
    if ledger_sha != _canonical_sha256(normalized_episodes):
        raise D0V3PhaseReceiptError("episode ledger seal mismatch")

    seals = _require_mapping(root["seals"], field="seals")
    _require_exact_keys(seals, _SEAL_FIELDS, field="seals")
    cache_protocol_sha256 = _require_sha256(
        seals["cache_protocol_sha256"], field="cache_protocol_sha256"
    )
    sealed_pilot_sha = _require_sha256(
        seals["pilot_ids_sha256"], field="seals.pilot_ids_sha256"
    )
    if sealed_pilot_sha != pilot_ids_sha256:
        raise D0V3PhaseReceiptError("pilot_ids_sha256 is not bound to ordered Pilot IDs")
    cache_method_sha256 = _require_sha256(
        seals["cache_method_manifest_sha256"],
        field="cache_method_manifest_sha256",
    )
    checkpoint_sha256 = _require_sha256(
        seals["checkpoint_sha256"], field="checkpoint_sha256"
    )
    config_sha256 = _require_sha256(seals["config_sha256"], field="config_sha256")
    raw_code_files = _require_sequence(seals["code_files"], field="code_files")
    if not raw_code_files:
        raise D0V3PhaseReceiptError("code_files must bind at least one code file")
    code_files: list[dict[str, str]] = []
    for index, raw_code_file in enumerate(raw_code_files):
        code_file = _require_mapping(raw_code_file, field=f"code_files[{index}]")
        _require_exact_keys(code_file, _CODE_FILE_FIELDS, field=f"code_files[{index}]")
        code_files.append(
            {
                "path": _code_path(code_file["path"], field=f"code_files[{index}].path"),
                "sha256": _require_sha256(
                    code_file["sha256"], field=f"code_files[{index}].sha256"
                ),
            }
        )
    if code_files != sorted(code_files, key=lambda record: record["path"]):
        raise D0V3PhaseReceiptError("code_files must be sorted by canonical path")
    if len({record["path"] for record in code_files}) != len(code_files):
        raise D0V3PhaseReceiptError("code_files contains duplicate paths")
    code_bundle_sha = _require_sha256(
        seals["code_bundle_sha256"], field="code_bundle_sha256"
    )
    if code_bundle_sha != _canonical_sha256(code_files):
        raise D0V3PhaseReceiptError("code_bundle_sha256 mismatch")

    _validate_boundary(root["candidate_phase_data_boundary"])

    completion = _require_mapping(root["completion"], field="completion")
    _require_exact_keys(completion, _COMPLETION_FIELDS, field="completion")
    for field in (
        "label_free_phase_complete",
        "ordered_images_complete",
        "candidate_grid_complete",
        "episode_receipts_complete",
    ):
        _require_bool(completion[field], field=f"completion.{field}", expected=True)
    _require_int(
        completion["complete_episode_count"],
        field="completion.complete_episode_count",
        exact=EPISODE_COUNT,
    )

    authorization = _require_mapping(root["authorization"], field="authorization")
    _require_exact_keys(authorization, _AUTHORIZATION_FIELDS, field="authorization")
    _require_bool(
        authorization["source_train_derived"],
        field="authorization.source_train_derived",
        expected=True,
    )
    for field in (
        "paper_result",
        "paper_test_result",
        "scientific_selection_performed",
        "stage2_authorized",
    ):
        _require_bool(authorization[field], field=f"authorization.{field}", expected=False)

    receipt_sha256 = hashlib.sha256(canonical).hexdigest()
    if expected_dataset is not None and dataset != expected_dataset:
        raise D0V3PhaseReceiptError(
            f"receipt dataset mismatch: expected {expected_dataset!r}, got {dataset!r}"
        )
    if expected_condition is not None and condition != expected_condition:
        raise D0V3PhaseReceiptError(
            f"receipt condition mismatch: expected {expected_condition!r}, got {condition!r}"
        )
    if expected_replicate is not None:
        expected_replicate_value = _require_int(
            expected_replicate, field="expected_replicate"
        )
        if replicate != expected_replicate_value:
            raise D0V3PhaseReceiptError(
                f"receipt replicate mismatch: expected {expected_replicate_value}, got {replicate}"
            )
    for actual, expected, field in (
        (cache_protocol_sha256, expected_cache_protocol_sha256, "cache_protocol_sha256"),
        (
            cache_method_sha256,
            expected_cache_method_manifest_sha256,
            "cache_method_manifest_sha256",
        ),
        (checkpoint_sha256, expected_checkpoint_sha256, "checkpoint_sha256"),
        (config_sha256, expected_config_sha256, "config_sha256"),
        (receipt_sha256, expected_receipt_sha256, "receipt_sha256"),
    ):
        if expected is not None:
            expected_sha = _require_sha256(expected, field=f"expected_{field}")
            if actual != expected_sha:
                raise D0V3PhaseReceiptError(
                    f"{field} mismatch: expected {expected_sha}, got {actual}"
                )
    if expected_ordered_image_ids is not None:
        expected_ids = tuple(
            _require_string(value, field=f"expected_ordered_image_ids[{index}]")
            for index, value in enumerate(
                _require_sequence(
                    expected_ordered_image_ids, field="expected_ordered_image_ids"
                )
            )
        )
        if image_ids != expected_ids:
            raise D0V3PhaseReceiptError("receipt Pilot ID order differs from expected order")
    if expected_code_seals is not None:
        if code_files != _canonical_code_files(expected_code_seals):
            raise D0V3PhaseReceiptError("receipt code seals differ from live expected seals")

    return ValidatedLabelFreeCellReceipt(
        dataset=dataset,
        condition=condition,
        replicate=replicate,
        ordered_image_ids=image_ids,
        pilot_ids_sha256=pilot_ids_sha256,
        cache_protocol_sha256=cache_protocol_sha256,
        cache_method_manifest_sha256=cache_method_sha256,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        code_files=tuple((record["path"], record["sha256"]) for record in code_files),
        episode_receipt_sha256s=tuple(episode_hashes),
        canonical_bytes=canonical,
        sha256=receipt_sha256,
    )


def canonical_label_free_cell_receipt_bytes(
    value: Mapping[str, Any] | bytes,
) -> bytes:
    """Return the unique canonical bytes after full receipt validation."""

    return validate_label_free_cell_receipt(value).canonical_bytes


def label_free_cell_receipt_sha256(value: Mapping[str, Any] | bytes) -> str:
    """Return SHA-256 of the unique canonical receipt bytes."""

    return validate_label_free_cell_receipt(value).sha256


def _stable_receipt_bytes(source: bytes | str | os.PathLike[str]) -> bytes:
    if isinstance(source, bytes):
        # ``bytes`` is immutable; copy through bytes() for a uniform owned value.
        return bytes(source)
    if isinstance(source, (str, os.PathLike)):
        return read_stable_regular_file(source).data
    raise D0V3PhaseReceiptError(
        "receipt_source must be canonical immutable bytes or a stable file path"
    )


def _verify_live_method_manifest(
    cache_root: Path,
    *,
    receipt: ValidatedLabelFreeCellReceipt,
) -> None:
    snapshot = read_stable_regular_file(cache_root / "method_input_manifest.json")
    if snapshot.sha256 != receipt.cache_method_manifest_sha256:
        raise D0V3PhaseReceiptError(
            "live cache method manifest SHA-256 differs from the phase receipt"
        )
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0V3PhaseReceiptError("live cache method manifest is not UTF-8 JSON") from error
    method = _require_mapping(value, field="live cache method manifest")
    if method.get("protocol_sha256") != receipt.cache_protocol_sha256:
        raise D0V3PhaseReceiptError("live method manifest protocol differs from receipt")
    if method.get("dataset") != receipt.dataset:
        raise D0V3PhaseReceiptError("live method manifest dataset differs from receipt")
    raw_ids = _require_sequence(method.get("image_ids"), field="live method image_ids")
    live_ids = tuple(str(value) for value in raw_ids)
    if live_ids != receipt.ordered_image_ids:
        raise D0V3PhaseReceiptError("live method Pilot ID order differs from receipt")
    if method.get("ordered_ids_sha256") != receipt.pilot_ids_sha256:
        raise D0V3PhaseReceiptError("live method Pilot ID seal differs from receipt")
    raw_conditions = _require_sequence(
        method.get("conditions"), field="live method conditions"
    )
    matching = [
        record
        for record in raw_conditions
        if isinstance(record, Mapping) and record.get("key") == receipt.condition
    ]
    if len(matching) != 1:
        raise D0V3PhaseReceiptError(
            "live method manifest does not contain exactly the receipt condition"
        )
    if method.get("targets_exposed") is not False:
        raise D0V3PhaseReceiptError("live method manifest exposes outer targets")


def guarded_load_outer_targets(
    cache_root: Path | str | os.PathLike[str],
    protocol_sha: str,
    receipt_source: bytes | str | os.PathLike[str],
    *,
    dataset: str,
    condition: str,
    replicate: int,
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_code_seals: Mapping[str, str] | None = None,
    expected_receipt_sha256: str | None = None,
) -> GuardedOuterTargetLoad:
    """Load outer targets only after a stable, exact 640-episode receipt.

    There is intentionally no ``episodes_complete`` argument.  The legacy
    boolean is supplied internally only after receipt validation, live cache
    lineage verification, and exact current-cell binding all succeed.
    """

    protocol_sha256 = _require_sha256(protocol_sha, field="protocol_sha")
    payload = _stable_receipt_bytes(receipt_source)
    receipt = validate_label_free_cell_receipt(
        payload,
        expected_dataset=dataset,
        expected_condition=condition,
        expected_replicate=replicate,
        expected_cache_protocol_sha256=protocol_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_code_seals=expected_code_seals,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    lexical_cache_root = Path(os.path.abspath(os.fspath(cache_root)))
    _verify_live_method_manifest(lexical_cache_root, receipt=receipt)

    # This is the only target-loader call in the module.  It is deliberately
    # unreachable until every cryptographic and live-cache gate above passes.
    targets = load_outer_evaluator_targets_v2(
        lexical_cache_root,
        expected_protocol_sha256=protocol_sha256,
        episodes_complete=True,
    )
    shape = getattr(targets, "shape", None)
    observed_shape = tuple(shape) if shape is not None else None
    if observed_shape != (IMAGE_COUNT, 1, 256, 256):
        raise D0V3PhaseReceiptError(
            "outer target loader returned an unexpected target tensor shape"
        )
    access_receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": OUTER_ACCESS_ARTIFACT_TYPE,
        "protocol_id": PROTOCOL_ID,
        "cell_binding": {
            "dataset": receipt.dataset,
            "condition": receipt.condition,
            "replicate": receipt.replicate,
        },
        "phase_evidence": {
            "label_free_cell_receipt_sha256": receipt.sha256,
            "complete_candidate_episode_count": EPISODE_COUNT,
            "candidate_phase_target_access_count": 0,
            "cache_protocol_sha256": receipt.cache_protocol_sha256,
            "cache_method_manifest_sha256": receipt.cache_method_manifest_sha256,
        },
        "outer_access": {
            "loader": (
                "materialize_binary_tent_ss_calibration_cache_v2."
                "load_outer_evaluator_targets_v2"
            ),
            "loader_call_count": 1,
            "target_count": IMAGE_COUNT,
            "target_shape": [IMAGE_COUNT, 1, 256, 256],
            "role": "train_only_outer_evaluator_after_label_free_phase",
            "used_by_adaptation": False,
            "method_facing_access": "forbidden",
            "adaptation_target_indexing_count": 0,
        },
        "authorization": {
            "paper_result": False,
            "paper_test_result": False,
            "scientific_selection_performed": False,
            "stage2_authorized": False,
        },
    }
    access_bytes = _canonical_json_bytes(access_receipt)
    return GuardedOuterTargetLoad(
        targets=targets,
        access_receipt_bytes=access_bytes,
        access_receipt_sha256=hashlib.sha256(access_bytes).hexdigest(),
        label_free_receipt_sha256=receipt.sha256,
    )
