"""Immutable Stage-B1 v2 cumulative/direct-VJP cell artifacts.

Five float32 VJPs are retained as primary evidence.  Scientific quantities are
always reconstructed by this CPU verifier in float64 from the cumulative
slots; the two independently differentiated region VJPs are numerical audits
only and can never enter a mechanism flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Final

import numpy as np

from analysis.d0_v3_label_free_shard import (
    array_slice_sha256,
    canonical_json_bytes,
    parse_canonical_json,
)
from analysis import p3_stage_b1_outer_cell_shard as _v1
from tta.d0_secure_io import read_stable_regular_file, snapshot_regular_directory


SCHEMA_VERSION: Final = 2
ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_outer_cell_shard_v2"
COMPLETE_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_outer_cell_complete_v2"
RECORD_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_outer_episode_v2"
GROUP_LAYOUT_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_coarse_group_layout_v2"

FORMAL_IMAGE_COUNT: Final = 64
SCALAR_PARAMETER_COUNT: Final = 8736
GROUP_IDS: Final = _v1.GROUP_IDS
GROUP_SCALAR_COUNTS: Final = _v1.GROUP_SCALAR_COUNTS

DIRECT_VJP_ORDER: Final = (
    "full",
    "foreground_total",
    "foreground_subthreshold",
    "raw_foreground_suprathreshold",
    "raw_background",
)
# Compatibility names deliberately point at the five-slot v2 evidence, not at
# the three-region v1 layout.
BASIS_ORDER: Final = DIRECT_VJP_ORDER
ANALYZER_BASIS_NAMES: Final = (
    "full_entropy_mean_direct",
    "foreground_entropy_add_direct",
    "foreground_subthreshold_entropy_add_direct",
    "foreground_suprathreshold_entropy_add_raw_audit",
    "background_entropy_add_raw_audit",
)
SCIENCE_BASIS_ORDER: Final = (
    "foreground_subthreshold",
    "foreground_suprathreshold",
    "background",
)
RAW_AUDIT_COMPONENTS: Final = (
    "foreground_suprathreshold",
    "background",
)

CUMULATIVE_VJPS_FILENAME: Final = "cumulative_and_raw_entropy_vjps.npy"
# Explicit compatibility alias for callers that treat the evidence array as a basis.
GRADIENT_DECOMPOSITION_BASIS_FILENAME: Final = CUMULATIVE_VJPS_FILENAME
REGION_GRADIENT_BASIS_FILENAME: Final = CUMULATIVE_VJPS_FILENAME
EPISODE_RECORDS_FILENAME: Final = "episode_records.jsonl"
COARSE_GROUP_LAYOUT_FILENAME: Final = "coarse_group_layout.json"
OUTER_ACCESS_RECEIPT_FILENAME: Final = "outer_access_receipt.json"
MANIFEST_FILENAME: Final = "manifest.json"
COMPLETE_FILENAME: Final = "COMPLETE.json"
MEMBERS: Final = frozenset(
    {
        CUMULATIVE_VJPS_FILENAME,
        EPISODE_RECORDS_FILENAME,
        COARSE_GROUP_LAYOUT_FILENAME,
        OUTER_ACCESS_RECEIPT_FILENAME,
        MANIFEST_FILENAME,
        COMPLETE_FILENAME,
    }
)

CELL_DATA_BOUNDARY: Final = dict(_v1.CELL_DATA_BOUNDARY)
CELL_AUTHORIZATION: Final = dict(_v1.CELL_AUTHORIZATION)
MANIFEST_DATA_BOUNDARY: Final = dict(_v1.MANIFEST_DATA_BOUNDARY)
FORMAL_RUNTIME: Final = dict(_v1.FORMAL_RUNTIME)
FORMAL_EXECUTION_COUNTS: Final = {
    "source_model_build_count": 1,
    "source_forward_count": FORMAL_IMAGE_COUNT,
    "direct_vjp_count": FORMAL_IMAGE_COUNT * len(DIRECT_VJP_ORDER),
    "optimizer_build_count": 0,
    "optimizer_step_count": 0,
    "model_weight_update_count": 0,
    "checkpoint_write_count": 0,
    "state_reset_passed_count": FORMAL_IMAGE_COUNT,
    "rng_reset_passed_count": FORMAL_IMAGE_COUNT,
    "validation_payload_access_count": 0,
    "test_payload_access_count": 0,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_TARGET_FIELDS = {
    "target_present",
    "total_pixel_count",
    "foreground_pixel_count",
    "background_pixel_count",
    "foreground_subthreshold_pixel_count",
    "foreground_suprathreshold_pixel_count",
    "target_value_sum",
    "target_slice_sha256",
    "partition_disjoint",
    "partition_exhaustive",
}
_SOURCE_INTEGRITY_FIELDS = {
    "source_logits_bit_exact",
    "parent_source_logits_slice_sha256",
    "recomputed_source_logits_slice_sha256",
    "source_state_before_sha256",
    "source_state_after_sha256",
    "state_restored",
    "rng_before_sha256",
    "rng_after_sha256",
    "rng_restored",
}
_GRADIENT_INTEGRITY_FIELDS = {
    "cumulative_basis_order",
    "direct_vjp_count_per_image",
    "basis_slice_sha256",
    "finite",
    "derived_closure",
    "raw_consistency",
    "parent_candidate_slice_count",
    "parent_direct_full_max_abs_error",
    "parent_direct_full_max_relative_l2_error",
    "max_abs_tolerance",
    "relative_l2_tolerance",
    "parent_direct_full_consistency_passed",
}
_RECORD_FIELDS = {
    "schema_version",
    "artifact_type",
    "protocol_id",
    "config_sha256",
    "dataset",
    "condition",
    "corruption_family",
    "severity",
    "replicate",
    "image_index",
    "image_id",
    "target",
    "source_integrity",
    "gradient_integrity",
    "groups",
    "data_boundary",
    "authorization",
}


class P3StageB1OuterCellShardV2Error(ValueError):
    """A v2 cell violates its immutable numerical or protocol contract."""


P3StageB1OuterCellShardError = P3StageB1OuterCellShardV2Error


@dataclass(frozen=True, slots=True)
class VerifiedStageB1CellShardV2:
    path: Path
    dataset: str
    condition: str
    corruption_family: str
    severity: str
    replicate: str
    image_count: int
    record_count: int
    ordered_image_ids_sha256: str
    target_identity_sha256: str
    manifest_sha256: str
    complete_sha256: str
    cumulative_vjps_sha256: str
    records_sha256: str
    group_layout_sha256: str
    outer_access_receipt_sha256: str
    records: tuple[Mapping[str, Any], ...]

    @property
    def basis_sha256(self) -> str:
        """Compatibility name used by the aggregate lineage."""

        return self.cumulative_vjps_sha256


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P3StageB1OuterCellShardV2Error(f"{label} must be a mapping")
    return value


def _exact(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    result = _mapping(value, label=label)
    if set(result) != fields:
        raise P3StageB1OuterCellShardV2Error(
            f"{label} fields differ; missing={sorted(fields-set(result))}, "
            f"unknown={sorted(set(result)-fields)}"
        )
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise P3StageB1OuterCellShardV2Error(f"{label} must be lowercase SHA-256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise P3StageB1OuterCellShardV2Error(f"{label} must be integer >= {minimum}")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P3StageB1OuterCellShardV2Error(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise P3StageB1OuterCellShardV2Error(f"{label} must be finite numeric")
    return result


def _numeric_close(observed: Any, expected: float, *, label: str) -> None:
    value = _finite(observed, label=label)
    if not math.isclose(value, expected, rel_tol=2.0e-10, abs_tol=2.0e-12):
        raise P3StageB1OuterCellShardV2Error(
            f"{label} differs; observed={value}, expected={expected}"
        )


def _condition_parts(condition: str) -> tuple[str, str]:
    if condition == "clean_S0":
        return "clean", "S0"
    if "_" not in condition:
        raise P3StageB1OuterCellShardV2Error("condition is malformed")
    return condition.rsplit("_", 1)


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def _load_npy(data: bytes, *, label: str) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise P3StageB1OuterCellShardV2Error(f"{label} is not safe NPY") from exc
    if not isinstance(value, np.ndarray):
        raise P3StageB1OuterCellShardV2Error(f"{label} is not ndarray")
    return value


def canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def parse_canonical_jsonl(data: bytes, *, count: int, label: str) -> list[Mapping[str, Any]]:
    try:
        return _v1.parse_canonical_jsonl(data, count=count, label=label)
    except Exception as exc:
        raise P3StageB1OuterCellShardV2Error(str(exc)) from exc


def derive_science_basis(direct_vjps: np.ndarray) -> np.ndarray:
    """Return [subthreshold, suprathreshold, background] in CPU float64."""

    direct = np.asarray(direct_vjps)
    if direct.ndim != 2 or direct.shape[0] != len(DIRECT_VJP_ORDER):
        raise P3StageB1OuterCellShardV2Error("one direct-VJP slice must have shape [5,D]")
    if direct.shape[1] <= 0 or not np.issubdtype(direct.dtype, np.floating):
        raise P3StageB1OuterCellShardV2Error("direct VJPs must be floating point")
    if not np.isfinite(direct).all():
        raise P3StageB1OuterCellShardV2Error("direct VJPs contain NaN/Inf")
    value = direct.astype(np.float64, copy=False)
    full, foreground, subthreshold = value[:3]
    suprathreshold = foreground - subthreshold
    background = full - foreground
    return np.ascontiguousarray(
        np.stack((subthreshold, suprathreshold, background), axis=0),
        dtype=np.float64,
    )


def _residual_metrics(reference: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    reference64 = np.asarray(reference, dtype=np.float64)
    observed64 = np.asarray(observed, dtype=np.float64)
    residual = observed64 - reference64
    reference_l2 = float(np.linalg.norm(reference64))
    observed_l2 = float(np.linalg.norm(observed64))
    l2 = float(np.linalg.norm(residual))
    values = {
        "max_abs_error": float(np.max(np.abs(residual))),
        "l2_error": l2,
        "relative_l2_error": l2 / max(reference_l2, 1.0e-12),
        "reference_l2": reference_l2,
        "observed_l2": observed_l2,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise P3StageB1OuterCellShardV2Error("residual audit is non-finite")
    return values


def build_gradient_integrity(
    direct_vjps: np.ndarray,
    *,
    parent_entropy_gradients: np.ndarray,
    max_abs_tolerance: float,
    relative_l2_tolerance: float,
) -> dict[str, Any]:
    """Build the exact independently reproducible integrity report for one image."""

    direct = np.asarray(direct_vjps)
    if direct.shape != (len(DIRECT_VJP_ORDER), SCALAR_PARAMETER_COUNT):
        raise P3StageB1OuterCellShardV2Error("direct VJP slice shape differs")
    if direct.dtype.str != "<f4" or not direct.flags.c_contiguous:
        raise P3StageB1OuterCellShardV2Error("direct VJP slice must be C-order <f4")
    science = derive_science_basis(direct)
    subthreshold, suprathreshold, background = science
    full = direct[0].astype(np.float64, copy=False)
    foreground = direct[1].astype(np.float64, copy=False)
    closure = _residual_metrics(full, subthreshold + suprathreshold + background)
    closure_passed = (
        closure["max_abs_error"] <= max_abs_tolerance
        and closure["relative_l2_error"] <= relative_l2_tolerance
    )
    if not closure_passed:
        raise P3StageB1OuterCellShardV2Error("float64 telescoping closure failed")

    raw_specs = (
        ("foreground_suprathreshold", direct[3], suprathreshold),
        ("background", direct[4], background),
    )
    raw_components: dict[str, Any] = {}
    for name, raw, derived in raw_specs:
        metrics = _residual_metrics(raw, derived)
        raw_components[name] = {
            **metrics,
            "max_abs_tolerance": float(max_abs_tolerance),
            "relative_l2_tolerance": float(relative_l2_tolerance),
            "within_dual_tolerance": (
                metrics["max_abs_error"] <= max_abs_tolerance
                and metrics["relative_l2_error"] <= relative_l2_tolerance
            ),
        }

    parent = np.asarray(parent_entropy_gradients)
    if parent.shape != (10, SCALAR_PARAMETER_COUNT):
        raise P3StageB1OuterCellShardV2Error("parent entropy slice shape differs")
    if not np.issubdtype(parent.dtype, np.floating) or not np.isfinite(parent).all():
        raise P3StageB1OuterCellShardV2Error("parent entropy slice is invalid")
    parent_errors = [
        _residual_metrics(candidate, full)
        for candidate in parent.astype(np.float64, copy=False)
    ]
    parent_abs = max(value["max_abs_error"] for value in parent_errors)
    parent_relative = max(value["relative_l2_error"] for value in parent_errors)
    parent_passed = (
        parent_abs <= max_abs_tolerance
        and parent_relative <= relative_l2_tolerance
    )
    return {
        "cumulative_basis_order": list(DIRECT_VJP_ORDER),
        "direct_vjp_count_per_image": len(DIRECT_VJP_ORDER),
        "basis_slice_sha256": array_slice_sha256(direct),
        "finite": True,
        "derived_closure": {
            "arithmetic_dtype": "float64",
            "science_basis_order": list(SCIENCE_BASIS_ORDER),
            "foreground_suprathreshold_formula": (
                "foreground_total-foreground_subthreshold"
            ),
            "background_formula": "full-foreground_total",
            "full_reconstruction_formula": (
                "foreground_subthreshold+derived_foreground_suprathreshold+"
                "derived_background"
            ),
            **closure,
            "max_abs_tolerance": float(max_abs_tolerance),
            "relative_l2_tolerance": float(relative_l2_tolerance),
            "within_dual_tolerance": True,
        },
        "raw_consistency": {
            "role": "numeric_diagnostic_only",
            "failure_action": "record_only_never_protocol_or_science_gate",
            "enters_science_metrics": False,
            "enters_mechanism_flags": False,
            "enters_selection_or_ranking": False,
            "components": raw_components,
        },
        "parent_candidate_slice_count": 10,
        "parent_direct_full_max_abs_error": parent_abs,
        "parent_direct_full_max_relative_l2_error": parent_relative,
        "max_abs_tolerance": float(max_abs_tolerance),
        "relative_l2_tolerance": float(relative_l2_tolerance),
        "parent_direct_full_consistency_passed": parent_passed,
    }


def build_coarse_group_layout(
    *,
    protocol_id: str,
    source_parameter_layout: Mapping[str, Any],
    group_parameter_names: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    try:
        layout = _v1.build_coarse_group_layout(
            protocol_id=protocol_id,
            source_parameter_layout=source_parameter_layout,
            group_parameter_names=group_parameter_names,
        )
    except Exception as exc:
        raise P3StageB1OuterCellShardV2Error(str(exc)) from exc
    layout["schema_version"] = SCHEMA_VERSION
    layout["artifact_type"] = GROUP_LAYOUT_ARTIFACT_TYPE
    return layout


def validate_coarse_group_layout(
    value: Mapping[str, Any], *, protocol_id: str, expected_layout_sha256: str
) -> tuple[Mapping[str, Any], dict[str, np.ndarray]]:
    layout = dict(_mapping(value, label="coarse group layout"))
    if (
        layout.get("schema_version") != SCHEMA_VERSION
        or layout.get("artifact_type") != GROUP_LAYOUT_ARTIFACT_TYPE
    ):
        raise P3StageB1OuterCellShardV2Error("coarse group layout identity differs")
    legacy = dict(layout)
    legacy["schema_version"] = _v1.SCHEMA_VERSION
    legacy["artifact_type"] = _v1.GROUP_LAYOUT_ARTIFACT_TYPE
    try:
        _, indices = _v1.validate_coarse_group_layout(
            legacy,
            protocol_id=protocol_id,
            expected_layout_sha256=expected_layout_sha256,
        )
    except Exception as exc:
        raise P3StageB1OuterCellShardV2Error(str(exc)) from exc
    return layout, indices


def _validate_code_seal(value: Any) -> Mapping[str, Any]:
    try:
        return _v1._validate_code_seal(value)
    except Exception as exc:
        raise P3StageB1OuterCellShardV2Error(str(exc)) from exc


def _validate_execution(value: Any) -> Mapping[str, Any]:
    execution = _mapping(value, label="execution")
    if set(execution) != {"runtime", *FORMAL_EXECUTION_COUNTS}:
        raise P3StageB1OuterCellShardV2Error("execution fields differ")
    if execution["runtime"] != FORMAL_RUNTIME:
        raise P3StageB1OuterCellShardV2Error("formal runtime differs")
    for field, expected in FORMAL_EXECUTION_COUNTS.items():
        observed = execution[field]
        if type(observed) is not type(expected) or observed != expected:
            raise P3StageB1OuterCellShardV2Error(
                f"formal execution counts differ: {field}"
            )
    return execution


def build_stage_b1_cell_payloads_v2(
    *,
    protocol_id: str,
    config_sha256: str,
    cell: Mapping[str, Any],
    ordered_image_ids: Sequence[str],
    dataset_binding: Mapping[str, Any],
    parent_lineage: Mapping[str, Any],
    parameter_layout: Mapping[str, Any],
    cumulative_and_raw_entropy_vjps: np.ndarray | None = None,
    episode_records: Sequence[Mapping[str, Any]],
    coarse_group_layout: Mapping[str, Any],
    outer_access_receipt_bytes: bytes,
    code_seal: Mapping[str, Any],
    execution: Mapping[str, Any],
    region_gradient_basis: np.ndarray | None = None,
    data_boundary: Mapping[str, Any] = MANIFEST_DATA_BOUNDARY,
    authorization: Mapping[str, Any] = CELL_AUTHORIZATION,
) -> dict[str, bytes]:
    """Build canonical bytes; publication is performed separately, no-replace."""

    digest = _sha256(config_sha256, label="config_sha256")
    cell_map = _exact(
        cell,
        {"dataset", "condition", "corruption_family", "severity", "replicate"},
        label="cell",
    )
    family, severity = _condition_parts(str(cell_map["condition"]))
    if (
        cell_map["replicate"] != "R0"
        or cell_map["corruption_family"] != family
        or cell_map["severity"] != severity
    ):
        raise P3StageB1OuterCellShardV2Error("cell identity differs")
    image_ids = list(ordered_image_ids)
    if (
        len(image_ids) != FORMAL_IMAGE_COUNT
        or len(set(image_ids)) != FORMAL_IMAGE_COUNT
        or not all(isinstance(value, str) and value for value in image_ids)
    ):
        raise P3StageB1OuterCellShardV2Error("ordered image IDs must be 64 unique strings")
    if (cumulative_and_raw_entropy_vjps is None) == (region_gradient_basis is None):
        raise P3StageB1OuterCellShardV2Error(
            "provide exactly one cumulative_and_raw_entropy_vjps/region_gradient_basis"
        )
    direct = np.asarray(
        cumulative_and_raw_entropy_vjps
        if cumulative_and_raw_entropy_vjps is not None
        else region_gradient_basis
    )
    expected_shape = (FORMAL_IMAGE_COUNT, len(DIRECT_VJP_ORDER), SCALAR_PARAMETER_COUNT)
    if (
        direct.shape != expected_shape
        or direct.dtype.str != "<f4"
        or not direct.flags.c_contiguous
        or not np.isfinite(direct).all()
    ):
        raise P3StageB1OuterCellShardV2Error("cumulative/direct VJP array schema differs")
    if len(episode_records) != FORMAL_IMAGE_COUNT:
        raise P3StageB1OuterCellShardV2Error("episode record count differs")
    validate_coarse_group_layout(
        coarse_group_layout,
        protocol_id=protocol_id,
        expected_layout_sha256=str(parameter_layout.get("source_layout_sha256")),
    )
    if not isinstance(outer_access_receipt_bytes, bytes) or not outer_access_receipt_bytes:
        raise P3StageB1OuterCellShardV2Error("outer access receipt bytes are missing")
    try:
        json.loads(
            outer_access_receipt_bytes.decode("utf-8"),
            object_pairs_hook=lambda pairs: _v1._unique_object(
                pairs, "outer access receipt"
            ),
            parse_constant=lambda token: (_ for _ in ()).throw(
                P3StageB1OuterCellShardV2Error(
                    f"outer access receipt contains {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P3StageB1OuterCellShardV2Error("outer access receipt is not JSON") from exc
    _validate_code_seal(code_seal)
    _validate_execution(execution)
    if dict(data_boundary) != MANIFEST_DATA_BOUNDARY:
        raise P3StageB1OuterCellShardV2Error("manifest data boundary differs")
    if dict(authorization) != CELL_AUTHORIZATION:
        raise P3StageB1OuterCellShardV2Error("manifest authorization differs")

    direct_bytes = _npy_bytes(direct)
    records_bytes = canonical_jsonl_bytes(episode_records)
    layout_bytes = canonical_json_bytes(coarse_group_layout, newline=True)
    payload = {
        CUMULATIVE_VJPS_FILENAME: direct_bytes,
        EPISODE_RECORDS_FILENAME: records_bytes,
        COARSE_GROUP_LAYOUT_FILENAME: layout_bytes,
        OUTER_ACCESS_RECEIPT_FILENAME: outer_access_receipt_bytes,
    }
    ids_sha = hashlib.sha256(canonical_json_bytes(image_ids)).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": protocol_id,
        "config_sha256": digest,
        "cell": dict(cell_map),
        "mode": {
            "formal": True,
            "dry_run": False,
            "image_count": FORMAL_IMAGE_COUNT,
            "record_count": FORMAL_IMAGE_COUNT,
            "record_order": "image_index_ascending",
        },
        "ordered_image_ids": image_ids,
        "ordered_image_ids_sha256": ids_sha,
        "dataset_binding": dict(dataset_binding),
        "parent_lineage": dict(parent_lineage),
        "parameter_layout": dict(parameter_layout),
        "direct_vjps": {
            "path": CUMULATIVE_VJPS_FILENAME,
            "sha256": hashlib.sha256(direct_bytes).hexdigest(),
            "shape": list(expected_shape),
            "dtype": "<f4",
            "c_order": True,
            "finite": True,
            "lossless": True,
            "cumulative_basis_order": list(DIRECT_VJP_ORDER),
            "science_derivation_dtype": "float64",
            "raw_slots_role": "numeric_diagnostic_only",
        },
        "records": {
            "path": EPISODE_RECORDS_FILENAME,
            "sha256": hashlib.sha256(records_bytes).hexdigest(),
            "count": FORMAL_IMAGE_COUNT,
            "order": "image_index_ascending",
        },
        "outer_access_receipt": {
            "path": OUTER_ACCESS_RECEIPT_FILENAME,
            "sha256": hashlib.sha256(outer_access_receipt_bytes).hexdigest(),
            "copied_bit_exact_from_parent": True,
        },
        "code_seal": dict(code_seal),
        "execution": dict(execution),
        "data_boundary": dict(data_boundary),
        "authorization": dict(authorization),
    }
    manifest_bytes = canonical_json_bytes(manifest, newline=True)
    payload[MANIFEST_FILENAME] = manifest_bytes
    complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": protocol_id,
        "complete": True,
        "config_sha256": digest,
        "cell": dict(cell_map),
        "image_count": FORMAL_IMAGE_COUNT,
        "record_count": FORMAL_IMAGE_COUNT,
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "payload_files": [
            {"path": name, "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in sorted(payload.items())
        ],
        "atomic_no_replace": True,
        "immutable": True,
        **CELL_AUTHORIZATION,
    }
    payload[COMPLETE_FILENAME] = canonical_json_bytes(complete, newline=True)
    return payload


# Short compatibility spelling for the v2 runner.
build_stage_b1_cell_payloads = build_stage_b1_cell_payloads_v2


def _config_sequences(config: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        return _v1._config_sequences(config)
    except Exception as exc:
        raise P3StageB1OuterCellShardV2Error(str(exc)) from exc


def _tolerances(config: Mapping[str, Any]) -> tuple[float, float]:
    section = _mapping(config.get("entropy_gradient_consistency"), label="entropy consistency")
    return (
        _finite(section.get("max_abs_tolerance"), label="max_abs_tolerance"),
        _finite(section.get("relative_l2_tolerance"), label="relative_l2_tolerance"),
    )


def _validate_gradient_integrity(
    raw: Any,
    *,
    direct: np.ndarray,
    parent_entropy: np.ndarray | None,
    max_abs_tolerance: float,
    relative_l2_tolerance: float,
) -> None:
    observed = _exact(raw, _GRADIENT_INTEGRITY_FIELDS, label="gradient_integrity")
    if parent_entropy is not None:
        expected = build_gradient_integrity(
            direct,
            parent_entropy_gradients=parent_entropy,
            max_abs_tolerance=max_abs_tolerance,
            relative_l2_tolerance=relative_l2_tolerance,
        )
        try:
            _v1._compare_json_metric_tree(observed, expected, label="gradient_integrity")
        except Exception as exc:
            raise P3StageB1OuterCellShardV2Error(str(exc)) from exc
        if expected["parent_direct_full_consistency_passed"] is not True:
            raise P3StageB1OuterCellShardV2Error(
                "parent direct-full dual tolerance failed"
            )
        return
    if (
        observed["cumulative_basis_order"] != list(DIRECT_VJP_ORDER)
        or observed["direct_vjp_count_per_image"] != len(DIRECT_VJP_ORDER)
        or observed["basis_slice_sha256"] != array_slice_sha256(direct)
        or observed["finite"] is not True
        or observed["parent_candidate_slice_count"] != 10
        or observed["parent_direct_full_consistency_passed"] is not True
    ):
        raise P3StageB1OuterCellShardV2Error("gradient integrity identity differs")
    _numeric_close(observed["max_abs_tolerance"], max_abs_tolerance, label="max abs tolerance")
    _numeric_close(
        observed["relative_l2_tolerance"],
        relative_l2_tolerance,
        label="relative l2 tolerance",
    )
    # Recompute every local closure/audit even without access to the parent arrays.
    zero_parent = np.repeat(direct[0:1], 10, axis=0)
    local_expected = build_gradient_integrity(
        direct,
        parent_entropy_gradients=zero_parent,
        max_abs_tolerance=max_abs_tolerance,
        relative_l2_tolerance=relative_l2_tolerance,
    )
    for field in ("derived_closure", "raw_consistency"):
        try:
            _v1._compare_json_metric_tree(
                observed[field], local_expected[field], label=f"gradient_integrity.{field}"
            )
        except Exception as exc:
            raise P3StageB1OuterCellShardV2Error(str(exc)) from exc
    _finite(observed["parent_direct_full_max_abs_error"], label="parent max abs")
    _finite(observed["parent_direct_full_max_relative_l2_error"], label="parent max relative")


def _validate_record(
    raw: Any,
    *,
    position: int,
    manifest: Mapping[str, Any],
    direct: np.ndarray,
    group_layout: Mapping[str, Any],
    group_indices: Mapping[str, np.ndarray],
    parent_entropy: np.ndarray | None,
    parent_task: np.ndarray | None,
    candidate_source_logits: np.ndarray | None,
    outer_source_logits: np.ndarray | None,
    max_abs_tolerance: float,
    relative_l2_tolerance: float,
) -> Mapping[str, Any]:
    record = _exact(raw, _RECORD_FIELDS, label=f"records[{position}]")
    cell = manifest["cell"]
    if (
        record["schema_version"] != SCHEMA_VERSION
        or record["artifact_type"] != RECORD_ARTIFACT_TYPE
        or record["protocol_id"] != manifest["protocol_id"]
        or record["config_sha256"] != manifest["config_sha256"]
        or record["dataset"] != cell["dataset"]
        or record["condition"] != cell["condition"]
        or record["corruption_family"] != cell["corruption_family"]
        or record["severity"] != cell["severity"]
        or record["replicate"] != "R0"
        or record["image_index"] != position
        or record["image_id"] != manifest["ordered_image_ids"][position]
        or record["data_boundary"] != CELL_DATA_BOUNDARY
        or record["authorization"] != CELL_AUTHORIZATION
    ):
        raise P3StageB1OuterCellShardV2Error(f"record identity differs at {position}")
    target = _exact(record["target"], _TARGET_FIELDS, label=f"records[{position}].target")
    total = _integer(target["total_pixel_count"], label="total pixels", minimum=1)
    fg = _integer(target["foreground_pixel_count"], label="foreground pixels")
    bg = _integer(target["background_pixel_count"], label="background pixels")
    sub = _integer(target["foreground_subthreshold_pixel_count"], label="subthreshold pixels")
    supra = _integer(target["foreground_suprathreshold_pixel_count"], label="suprathreshold pixels")
    target_sum = _finite(target["target_value_sum"], label="target value sum")
    if (
        target["target_present"] is not (fg > 0)
        or fg + bg != total
        or sub + supra != fg
        or not 0.0 <= target_sum <= total
        or target["partition_disjoint"] is not True
        or target["partition_exhaustive"] is not True
    ):
        raise P3StageB1OuterCellShardV2Error("target partition/count fields differ")
    _sha256(target["target_slice_sha256"], label="target slice SHA")
    source = _exact(
        record["source_integrity"],
        _SOURCE_INTEGRITY_FIELDS,
        label=f"records[{position}].source_integrity",
    )
    for field in _SOURCE_INTEGRITY_FIELDS - {
        "source_logits_bit_exact", "state_restored", "rng_restored"
    }:
        if field.endswith("sha256"):
            _sha256(source[field], label=field)
    if (
        source["source_logits_bit_exact"] is not True
        or source["state_restored"] is not True
        or source["rng_restored"] is not True
        or source["source_state_before_sha256"] != source["source_state_after_sha256"]
        or source["rng_before_sha256"] != source["rng_after_sha256"]
    ):
        raise P3StageB1OuterCellShardV2Error("Source/RNG integrity differs")
    if candidate_source_logits is not None and outer_source_logits is not None:
        candidate_sha = array_slice_sha256(candidate_source_logits[position])
        outer_sha = array_slice_sha256(outer_source_logits[position])
        if (
            not np.array_equal(candidate_source_logits[position], outer_source_logits[position])
            or source["parent_source_logits_slice_sha256"] != candidate_sha
            or source["recomputed_source_logits_slice_sha256"] != outer_sha
            or candidate_sha != outer_sha
        ):
            raise P3StageB1OuterCellShardV2Error("Source logit slice integrity differs")
    _validate_gradient_integrity(
        record["gradient_integrity"],
        direct=direct[position],
        parent_entropy=None if parent_entropy is None else parent_entropy[position],
        max_abs_tolerance=max_abs_tolerance,
        relative_l2_tolerance=relative_l2_tolerance,
    )
    groups = _mapping(record["groups"], label=f"records[{position}].groups")
    if set(groups) != set(GROUP_IDS):
        raise P3StageB1OuterCellShardV2Error("record group keys differ")
    science = derive_science_basis(direct[position])
    task = None if parent_task is None else parent_task[position].astype(np.float64, copy=False)
    layout_groups = _mapping(group_layout["groups"], label="layout groups")
    for group_id in GROUP_IDS:
        try:
            _v1._validate_group_report(
                groups[group_id],
                group_id=group_id,
                indices=group_indices[group_id],
                basis=science,
                task=task,
                target=target,
                layout_group=layout_groups[group_id],
            )
        except Exception as exc:
            raise P3StageB1OuterCellShardV2Error(str(exc)) from exc
    return record


def verify_stage_b1_cell_shard_v2(
    path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    config: Mapping[str, Any],
    expected_config_sha256: str,
    verify_live_parents: bool = True,
    expected_code_seal: Mapping[str, Any] | None = None,
) -> VerifiedStageB1CellShardV2:
    """Verify membership, hashes, boundaries and all derived episode metrics."""

    if verify_live_parents and expected_code_seal is None:
        raise P3StageB1OuterCellShardV2Error(
            "live parent verification requires the expected code seal"
        )

    root = Path(os.path.abspath(os.fspath(path)))
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    config_sha = _sha256(expected_config_sha256, label="expected_config_sha256")
    protocol_id = str(config.get("protocol_id"))
    datasets, conditions = _config_sequences(config)
    try:
        snapshot = snapshot_regular_directory(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise P3StageB1OuterCellShardV2Error(f"cannot snapshot B1 v2 cell: {root}") from exc
    by_name = {member.path.name: member for member in snapshot.members}
    if set(by_name) != MEMBERS:
        raise P3StageB1OuterCellShardV2Error("B1 v2 cell member set differs")
    manifest = parse_canonical_json(
        by_name[MANIFEST_FILENAME].data, label=MANIFEST_FILENAME, newline=True
    )
    expected_manifest_fields = {
        "schema_version", "artifact_type", "protocol_id", "config_sha256", "cell",
        "mode", "ordered_image_ids", "ordered_image_ids_sha256", "dataset_binding",
        "parent_lineage", "parameter_layout", "direct_vjps", "records",
        "outer_access_receipt", "code_seal", "execution", "data_boundary",
        "authorization",
    }
    if set(manifest) != expected_manifest_fields or (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["artifact_type"] != ARTIFACT_TYPE
        or manifest["protocol_id"] != protocol_id
        or manifest["config_sha256"] != config_sha
    ):
        raise P3StageB1OuterCellShardV2Error("manifest identity differs")
    cell = _exact(
        manifest["cell"],
        {"dataset", "condition", "corruption_family", "severity", "replicate"},
        label="manifest.cell",
    )
    dataset, condition = str(cell["dataset"]), str(cell["condition"])
    family, severity = _condition_parts(condition)
    if (
        dataset not in datasets
        or condition not in conditions
        or cell["corruption_family"] != family
        or cell["severity"] != severity
        or cell["replicate"] != "R0"
    ):
        raise P3StageB1OuterCellShardV2Error("cell grid binding differs")
    if manifest["mode"] != {
        "formal": True, "dry_run": False, "image_count": 64, "record_count": 64,
        "record_order": "image_index_ascending",
    }:
        raise P3StageB1OuterCellShardV2Error("formal mode differs")
    image_ids = manifest["ordered_image_ids"]
    if (
        not isinstance(image_ids, list)
        or len(image_ids) != FORMAL_IMAGE_COUNT
        or len(set(image_ids)) != FORMAL_IMAGE_COUNT
        or not all(isinstance(value, str) and value for value in image_ids)
    ):
        raise P3StageB1OuterCellShardV2Error("ordered image IDs differ")
    ids_sha = hashlib.sha256(canonical_json_bytes(image_ids)).hexdigest()
    dataset_config = _mapping(config["datasets"][dataset], label="dataset config")
    if (
        manifest["ordered_image_ids_sha256"] != ids_sha
        or ids_sha != dataset_config.get("ordered_pilot64_image_ids_sha256")
    ):
        raise P3StageB1OuterCellShardV2Error("ordered image ID SHA differs")
    if manifest["dataset_binding"] != {
        "split_name": "train",
        "train_split_sha256": dataset_config["train_split_sha256"],
        "checkpoint_role": dataset_config["checkpoint_role"],
        "checkpoint_path": dataset_config["checkpoint_path"],
        "checkpoint_sha256": dataset_config["checkpoint_sha256"],
    }:
        raise P3StageB1OuterCellShardV2Error("dataset binding differs")
    parent_lineage = _mapping(manifest["parent_lineage"], label="parent lineage")
    expected_lineage_fields = {
        "candidate_shard_path", "outer_shard_path", "candidate_manifest_sha256",
        "candidate_complete_sha256", "candidate_phase_receipt_sha256",
        "outer_manifest_sha256", "outer_complete_sha256", "outer_access_receipt_sha256",
        "parent_entropy_gradients_sha256", "parent_task_gradients_sha256",
    }
    _exact(parent_lineage, expected_lineage_fields, label="parent lineage")
    for field in expected_lineage_fields - {"candidate_shard_path", "outer_shard_path"}:
        _sha256(parent_lineage[field], label=field)
    parent_config = _mapping(config["parent_cell_artifacts"], label="parent artifacts")
    expected_candidate = str(parent_config["candidate_shard_template"]).format(
        dataset=dataset, condition=condition
    )
    expected_outer = str(parent_config["outer_shard_template"]).format(
        dataset=dataset, condition=condition
    )
    if (
        parent_lineage["candidate_shard_path"] != expected_candidate
        or parent_lineage["outer_shard_path"] != expected_outer
    ):
        raise P3StageB1OuterCellShardV2Error("parent paths differ from config")
    parameter_layout = _mapping(manifest["parameter_layout"], label="parameter layout")
    expected_layout_fields = {
        "parent_source_path", "parent_source_file_sha256", "source_layout_sha256",
        "source_parameter_tensor_count", "source_scalar_parameter_count",
        "coarse_group_layout_path", "coarse_group_layout_sha256", "group_order",
        "group_scalar_counts",
    }
    _exact(parameter_layout, expected_layout_fields, label="parameter layout")
    frozen_layout = _mapping(parent_config["parameter_layout"], label="frozen layout")
    if (
        parameter_layout["source_layout_sha256"] != frozen_layout["layout_sha256"]
        or parameter_layout["source_parameter_tensor_count"]
        != frozen_layout["parameter_tensor_count"]
        or parameter_layout["source_scalar_parameter_count"] != SCALAR_PARAMETER_COUNT
        or parameter_layout["coarse_group_layout_path"] != COARSE_GROUP_LAYOUT_FILENAME
        or parameter_layout["coarse_group_layout_sha256"]
        != by_name[COARSE_GROUP_LAYOUT_FILENAME].sha256
        or parameter_layout["group_order"] != list(GROUP_IDS)
        or parameter_layout["group_scalar_counts"] != GROUP_SCALAR_COUNTS
        or parameter_layout["parent_source_path"]
        != f"{expected_candidate}/{frozen_layout['filename']}"
    ):
        raise P3StageB1OuterCellShardV2Error("manifest parameter layout differs")
    _sha256(parameter_layout["parent_source_file_sha256"], label="parent layout SHA")
    group_layout = parse_canonical_json(
        by_name[COARSE_GROUP_LAYOUT_FILENAME].data,
        label=COARSE_GROUP_LAYOUT_FILENAME,
        newline=True,
    )
    group_layout, group_indices = validate_coarse_group_layout(
        group_layout,
        protocol_id=protocol_id,
        expected_layout_sha256=str(parameter_layout["source_layout_sha256"]),
    )
    direct = _load_npy(
        by_name[CUMULATIVE_VJPS_FILENAME].data, label=CUMULATIVE_VJPS_FILENAME
    )
    expected_shape = (64, len(DIRECT_VJP_ORDER), SCALAR_PARAMETER_COUNT)
    if (
        direct.shape != expected_shape
        or direct.dtype.str != "<f4"
        or not direct.flags.c_contiguous
        or not np.isfinite(direct).all()
    ):
        raise P3StageB1OuterCellShardV2Error("cumulative/direct VJP schema differs")
    if manifest["direct_vjps"] != {
        "path": CUMULATIVE_VJPS_FILENAME,
        "sha256": by_name[CUMULATIVE_VJPS_FILENAME].sha256,
        "shape": list(expected_shape),
        "dtype": "<f4",
        "c_order": True,
        "finite": True,
        "lossless": True,
        "cumulative_basis_order": list(DIRECT_VJP_ORDER),
        "science_derivation_dtype": "float64",
        "raw_slots_role": "numeric_diagnostic_only",
    }:
        raise P3StageB1OuterCellShardV2Error("direct VJP manifest differs")
    records = parse_canonical_jsonl(
        by_name[EPISODE_RECORDS_FILENAME].data,
        count=64,
        label=EPISODE_RECORDS_FILENAME,
    )
    if manifest["records"] != {
        "path": EPISODE_RECORDS_FILENAME,
        "sha256": by_name[EPISODE_RECORDS_FILENAME].sha256,
        "count": 64,
        "order": "image_index_ascending",
    }:
        raise P3StageB1OuterCellShardV2Error("record manifest differs")
    receipt_reference = {
        "path": OUTER_ACCESS_RECEIPT_FILENAME,
        "sha256": by_name[OUTER_ACCESS_RECEIPT_FILENAME].sha256,
        "copied_bit_exact_from_parent": True,
    }
    if (
        manifest["outer_access_receipt"] != receipt_reference
        or receipt_reference["sha256"] != parent_lineage["outer_access_receipt_sha256"]
    ):
        raise P3StageB1OuterCellShardV2Error("outer receipt binding differs")
    code_seal = _validate_code_seal(manifest["code_seal"])
    if expected_code_seal is not None and code_seal != expected_code_seal:
        raise P3StageB1OuterCellShardV2Error("code seal differs from expected")
    if (
        manifest["data_boundary"] != MANIFEST_DATA_BOUNDARY
        or manifest["authorization"] != CELL_AUTHORIZATION
    ):
        raise P3StageB1OuterCellShardV2Error("manifest boundary/authorization differs")
    _validate_execution(manifest["execution"])
    max_abs_tolerance, relative_l2_tolerance = _tolerances(config)
    parent_entropy = parent_task = candidate_logits = outer_logits = None
    if verify_live_parents:
        try:
            (
                parent_entropy,
                parent_task,
                candidate_logits,
                outer_logits,
                parent_receipt,
            ) = _v1._live_parent_context(
                repository_root=repository, lineage=parent_lineage
            )
        except Exception as exc:
            raise P3StageB1OuterCellShardV2Error(str(exc)) from exc
        if parent_receipt != by_name[OUTER_ACCESS_RECEIPT_FILENAME].data:
            raise P3StageB1OuterCellShardV2Error("outer receipt is not parent bit-exact")
        parent_layout_snapshot = read_stable_regular_file(
            repository / parameter_layout["parent_source_path"]
        )
        if parent_layout_snapshot.sha256 != parameter_layout["parent_source_file_sha256"]:
            raise P3StageB1OuterCellShardV2Error("live parent layout SHA differs")
    else:
        try:
            json.loads(
                by_name[OUTER_ACCESS_RECEIPT_FILENAME].data.decode("utf-8"),
                object_pairs_hook=lambda pairs: _v1._unique_object(
                    pairs, "outer access receipt"
                ),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    P3StageB1OuterCellShardV2Error(
                        f"outer access receipt contains {token}"
                    )
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise P3StageB1OuterCellShardV2Error("outer receipt is invalid JSON") from exc
    validated_records = tuple(
        _validate_record(
            record,
            position=index,
            manifest=manifest,
            direct=direct,
            group_layout=group_layout,
            group_indices=group_indices,
            parent_entropy=parent_entropy,
            parent_task=parent_task,
            candidate_source_logits=candidate_logits,
            outer_source_logits=outer_logits,
            max_abs_tolerance=max_abs_tolerance,
            relative_l2_tolerance=relative_l2_tolerance,
        )
        for index, record in enumerate(records)
    )
    target_identities = [
        {
            "image_index": record["image_index"],
            "image_id": record["image_id"],
            "target_present": record["target"]["target_present"],
            "total_pixel_count": record["target"]["total_pixel_count"],
            "foreground_pixel_count": record["target"]["foreground_pixel_count"],
            "background_pixel_count": record["target"]["background_pixel_count"],
            "target_value_sum": record["target"]["target_value_sum"],
            "target_slice_sha256": record["target"]["target_slice_sha256"],
        }
        for record in validated_records
    ]
    target_identity_sha = hashlib.sha256(canonical_json_bytes(target_identities)).hexdigest()
    complete = parse_canonical_json(
        by_name[COMPLETE_FILENAME].data, label=COMPLETE_FILENAME, newline=True
    )
    expected_complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": protocol_id,
        "complete": True,
        "config_sha256": config_sha,
        "cell": dict(cell),
        "image_count": 64,
        "record_count": 64,
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": by_name[MANIFEST_FILENAME].sha256,
        },
        "payload_files": [
            {"path": name, "sha256": by_name[name].sha256}
            for name in sorted(set(by_name) - {COMPLETE_FILENAME})
        ],
        "atomic_no_replace": True,
        "immutable": True,
        **CELL_AUTHORIZATION,
    }
    if complete != expected_complete:
        raise P3StageB1OuterCellShardV2Error("COMPLETE does not rebuild from payload")
    return VerifiedStageB1CellShardV2(
        path=root,
        dataset=dataset,
        condition=condition,
        corruption_family=family,
        severity=severity,
        replicate="R0",
        image_count=64,
        record_count=64,
        ordered_image_ids_sha256=ids_sha,
        target_identity_sha256=target_identity_sha,
        manifest_sha256=by_name[MANIFEST_FILENAME].sha256,
        complete_sha256=by_name[COMPLETE_FILENAME].sha256,
        cumulative_vjps_sha256=by_name[CUMULATIVE_VJPS_FILENAME].sha256,
        records_sha256=by_name[EPISODE_RECORDS_FILENAME].sha256,
        group_layout_sha256=by_name[COARSE_GROUP_LAYOUT_FILENAME].sha256,
        outer_access_receipt_sha256=by_name[OUTER_ACCESS_RECEIPT_FILENAME].sha256,
        records=validated_records,
    )


verify_stage_b1_cell_shard = verify_stage_b1_cell_shard_v2


__all__ = [
    "ANALYZER_BASIS_NAMES", "ARTIFACT_TYPE", "BASIS_ORDER", "CELL_AUTHORIZATION", "CELL_DATA_BOUNDARY",
    "COARSE_GROUP_LAYOUT_FILENAME", "COMPLETE_ARTIFACT_TYPE", "COMPLETE_FILENAME",
    "CUMULATIVE_VJPS_FILENAME", "DIRECT_VJP_ORDER", "EPISODE_RECORDS_FILENAME",
    "FORMAL_EXECUTION_COUNTS", "FORMAL_IMAGE_COUNT", "FORMAL_RUNTIME",
    "GRADIENT_DECOMPOSITION_BASIS_FILENAME", "GROUP_IDS", "GROUP_SCALAR_COUNTS",
    "GROUP_LAYOUT_ARTIFACT_TYPE", "MANIFEST_DATA_BOUNDARY", "MANIFEST_FILENAME",
    "MEMBERS", "OUTER_ACCESS_RECEIPT_FILENAME", "P3StageB1OuterCellShardError", "P3StageB1OuterCellShardV2Error",
    "RAW_AUDIT_COMPONENTS", "RECORD_ARTIFACT_TYPE", "REGION_GRADIENT_BASIS_FILENAME", "SCALAR_PARAMETER_COUNT",
    "SCHEMA_VERSION", "SCIENCE_BASIS_ORDER", "VerifiedStageB1CellShardV2",
    "build_coarse_group_layout", "build_gradient_integrity",
    "build_stage_b1_cell_payloads", "build_stage_b1_cell_payloads_v2",
    "canonical_jsonl_bytes", "derive_science_basis", "parse_canonical_jsonl",
    "validate_coarse_group_layout", "verify_stage_b1_cell_shard",
    "verify_stage_b1_cell_shard_v2",
]
