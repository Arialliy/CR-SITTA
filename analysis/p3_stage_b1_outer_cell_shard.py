"""Immutable CPU verifier for one P3 Stage-B1 outer-oracle cell.

The shard contains only the three additive entropy-gradient basis vectors and
JSON metadata.  It is deliberately independent of the model runner: public
verification needs NumPy plus the already sealed Stage-A parent artifacts, and
never constructs a model, optimizer, CUDA context, validation loader, or test
loader.

The three stored vectors are, in order, foreground-below-threshold,
foreground-above-threshold, and background.  Full and foreground gradients
are derivable sums and therefore are not stored a second time.
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
    PHASE_RECEIPT_FILENAME as PARENT_PHASE_RECEIPT_FILENAME,
    array_slice_sha256,
    canonical_json_bytes,
    parse_canonical_json,
)
from tta.d0_secure_io import read_stable_regular_file, snapshot_regular_directory


SCHEMA_VERSION: Final = 1
ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_outer_cell_shard"
COMPLETE_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_outer_cell_complete"
RECORD_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_outer_episode"
GROUP_LAYOUT_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_coarse_group_layout"

FORMAL_IMAGE_COUNT: Final = 64
SCALAR_PARAMETER_COUNT: Final = 8736
GROUP_IDS: Final = ("P0", "P1", "P2", "P3", "P4")
GROUP_SCALAR_COUNTS: Final = {
    "P0": 8736,
    "P1": 96,
    "P2": 416,
    "P3": 2080,
    "P4": 2592,
}
BASIS_ORDER: Final = (
    "foreground_subthreshold",
    "foreground_suprathreshold",
    "background",
)
ANALYZER_BASIS_NAMES: Final = (
    "foreground_subthreshold_add",
    "foreground_suprathreshold_add",
    "background_add",
)
CONDITIONAL_COMPONENTS: Final = (
    "full_entropy_mean",
    "foreground_entropy_mean",
    "background_entropy_mean",
    "foreground_subthreshold_entropy_mean",
    "foreground_suprathreshold_entropy_mean",
)
ADDITIVE_ALIGNMENT_COMPONENTS: Final = (
    "full_entropy_mean",
    "foreground_entropy_add",
    "background_entropy_add",
    "foreground_subthreshold_entropy_add",
    "foreground_suprathreshold_entropy_add",
)
ADDITIVE_COMPONENTS: Final = (
    "foreground_subthreshold_add",
    "foreground_suprathreshold_add",
    "background_add",
    "foreground_add",
    "full_add",
)

REGION_GRADIENT_BASIS_FILENAME: Final = "region_gradient_basis.npy"
EPISODE_RECORDS_FILENAME: Final = "episode_records.jsonl"
COARSE_GROUP_LAYOUT_FILENAME: Final = "coarse_group_layout.json"
OUTER_ACCESS_RECEIPT_FILENAME: Final = "outer_access_receipt.json"
MANIFEST_FILENAME: Final = "manifest.json"
COMPLETE_FILENAME: Final = "COMPLETE.json"
MEMBERS: Final = frozenset(
    {
        REGION_GRADIENT_BASIS_FILENAME,
        EPISODE_RECORDS_FILENAME,
        COARSE_GROUP_LAYOUT_FILENAME,
        OUTER_ACCESS_RECEIPT_FILENAME,
        MANIFEST_FILENAME,
        COMPLETE_FILENAME,
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
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
    "basis_order",
    "basis_slice_sha256",
    "finite",
    "parent_candidate_slice_count",
    "parent_max_abs_error",
    "parent_max_relative_l2_error",
    "max_abs_tolerance",
    "relative_l2_tolerance",
    "parent_consistency_passed",
}
_RUNTIME_FIELDS = {
    "seed",
    "device",
    "deterministic_algorithms",
    "deterministic_warn_only",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "cublas_workspace_config",
    "visible_cuda_device_count",
    "gradient_backward_policy",
}
_EXECUTION_FIELDS = {
    "runtime",
    "source_model_build_count",
    "source_forward_count",
    "region_backward_count",
    "optimizer_build_count",
    "optimizer_step_count",
    "model_weight_update_count",
    "checkpoint_write_count",
    "state_reset_passed_count",
    "rng_reset_passed_count",
    "validation_payload_access_count",
    "test_payload_access_count",
}
FORMAL_RUNTIME: Final = {
    "seed": 42,
    "device": "cuda:0",
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cublas_workspace_config": ":4096:8",
    "visible_cuda_device_count": 1,
    "gradient_backward_policy": (
        "temporarily_disable_strict_determinism_then_restore"
    ),
}
FORMAL_EXECUTION_COUNTS: Final = {
    "source_model_build_count": 1,
    "source_forward_count": FORMAL_IMAGE_COUNT,
    "region_backward_count": FORMAL_IMAGE_COUNT * len(BASIS_ORDER),
    "optimizer_build_count": 0,
    "optimizer_step_count": 0,
    "model_weight_update_count": 0,
    "checkpoint_write_count": 0,
    "state_reset_passed_count": FORMAL_IMAGE_COUNT,
    "rng_reset_passed_count": FORMAL_IMAGE_COUNT,
    "validation_payload_access_count": 0,
    "test_payload_access_count": 0,
}
CELL_DATA_BOUNDARY: Final = {
    "source_train_outer_oracle_only": True,
    "method_label_accesses": 0,
    "outer_target_accesses": 1,
    "used_by_adaptation": False,
    "validation_payload_accesses": 0,
    "test_payload_accesses": 0,
    "optimizer_constructed": False,
    "model_parameters_updated": False,
}
CELL_AUTHORIZATION: Final = {
    "paper_result": False,
    "paper_test_result": False,
    "performance_claim": False,
    "candidate_selection_performed": False,
    "stage_b3_authorized": False,
    "p5_authorized": False,
}
MANIFEST_DATA_BOUNDARY: Final = {
    "source_train_pilot64_outer_oracle_only": True,
    "method_label_accesses": 0,
    "outer_target_loader_call_count": 1,
    "outer_target_indexing_count": FORMAL_IMAGE_COUNT,
    "used_by_adaptation": False,
    "validation_payload_accesses": 0,
    "test_payload_accesses": 0,
    "test_split_files_opened": 0,
    "test_images_opened": 0,
    "test_masks_opened": 0,
}


class P3StageB1OuterCellShardError(ValueError):
    """A B1 cell is incomplete, non-canonical, unbound, or unsafe."""


@dataclass(frozen=True, slots=True)
class VerifiedStageB1CellShard:
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
    basis_sha256: str
    records_sha256: str
    group_layout_sha256: str
    outer_access_receipt_sha256: str
    records: tuple[Mapping[str, Any], ...]


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P3StageB1OuterCellShardError(f"{label} must be a mapping")
    return value


def _exact(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    result = _mapping(value, label=label)
    if set(result) != fields:
        missing = sorted(fields - set(result))
        unknown = sorted(set(result) - fields)
        raise P3StageB1OuterCellShardError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise P3StageB1OuterCellShardError(f"{label} must be lowercase SHA-256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise P3StageB1OuterCellShardError(
            f"{label} must be an integer >= {minimum}"
        )
    return int(value)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P3StageB1OuterCellShardError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise P3StageB1OuterCellShardError(f"{label} must be finite")
    return result


def _optional_finite(value: Any, *, label: str) -> float | None:
    return None if value is None else _finite(value, label=label)


def _validate_finite_json_tree(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _finite(value, label=label)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise P3StageB1OuterCellShardError(
                    f"{label} has a non-string JSON key"
                )
            _validate_finite_json_tree(child, label=f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _validate_finite_json_tree(child, label=f"{label}[{index}]")
        return
    raise P3StageB1OuterCellShardError(f"{label} is not finite JSON")


def _compare_json_metric_tree(observed: Any, expected: Any, *, label: str) -> None:
    if expected is None or isinstance(expected, (int, float)) and not isinstance(
        expected, bool
    ):
        _numeric_close(observed, None if expected is None else float(expected), label=label)
        return
    if isinstance(expected, Mapping):
        value = _mapping(observed, label=label)
        if set(value) != set(expected):
            raise P3StageB1OuterCellShardError(f"{label} fields differ")
        for key, child in expected.items():
            _compare_json_metric_tree(value[key], child, label=f"{label}.{key}")
        return
    if observed != expected:
        raise P3StageB1OuterCellShardError(
            f"{label} differs; observed={observed!r}, expected={expected!r}"
        )


def _canonical_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise P3StageB1OuterCellShardError(f"{label} must be a relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise P3StageB1OuterCellShardError(f"{label} is not canonical relative")
    return path.as_posix()


def _repository_path(repository_root: Path, value: Any, *, label: str) -> Path:
    relative = _canonical_relative(value, label=label)
    root = Path(os.path.abspath(os.fspath(repository_root)))
    path = Path(os.path.abspath(os.fspath(root / relative)))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise P3StageB1OuterCellShardError(f"{label} escapes repository") from exc
    return path


def _condition_parts(condition: str) -> tuple[str, str]:
    if condition == "clean_S0":
        return "clean", "S0"
    try:
        family, severity = condition.rsplit("_", 1)
    except ValueError as exc:
        raise P3StageB1OuterCellShardError("condition is malformed") from exc
    if not family or severity not in {"S1", "S3", "S5"}:
        raise P3StageB1OuterCellShardError("condition family/severity is malformed")
    return family, severity


def _unique_object(pairs: Sequence[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise P3StageB1OuterCellShardError(
                f"{label} contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _parse_json_line(line: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            line.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_object(pairs, label),
            parse_constant=lambda token: (_ for _ in ()).throw(
                P3StageB1OuterCellShardError(f"{label} contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P3StageB1OuterCellShardError(f"{label} is not strict JSON") from exc
    result = _mapping(value, label=label)
    if canonical_json_bytes(result) != line:
        raise P3StageB1OuterCellShardError(f"{label} is not canonical JSON")
    return result


def parse_canonical_jsonl(data: bytes, *, count: int, label: str) -> list[Mapping[str, Any]]:
    if not data or not data.endswith(b"\n"):
        raise P3StageB1OuterCellShardError(
            f"{label} must be non-empty newline-terminated JSONL"
        )
    lines = data[:-1].split(b"\n")
    if len(lines) != count or any(not line for line in lines):
        raise P3StageB1OuterCellShardError(
            f"{label} count differs; expected={count}, observed={len(lines)}"
        )
    return [
        _parse_json_line(line, label=f"{label}[{index}]")
        for index, line in enumerate(lines)
    ]


def canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Sequence):
        raise P3StageB1OuterCellShardError("records must be a sequence")
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.ascontiguousarray(value, dtype="<f4"), allow_pickle=False)
    return output.getvalue()


def _load_npy(data: bytes, *, label: str) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise P3StageB1OuterCellShardError(f"{label} is not safe NPY") from exc
    if not isinstance(value, np.ndarray):
        raise P3StageB1OuterCellShardError(f"{label} is not an ndarray")
    return value


def _config_sequences(config: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    value = _mapping(config, label="config")
    datasets_raw = value.get("datasets")
    conditions_raw = value.get("conditions")
    if isinstance(datasets_raw, Mapping):
        datasets = tuple(str(key) for key in datasets_raw)
    elif isinstance(datasets_raw, Sequence) and not isinstance(
        datasets_raw, (str, bytes)
    ):
        datasets = tuple(str(item) for item in datasets_raw)
    else:
        raise P3StageB1OuterCellShardError("config datasets are missing")
    if isinstance(conditions_raw, Sequence) and not isinstance(
        conditions_raw, (str, bytes)
    ):
        conditions = tuple(str(item) for item in conditions_raw)
    else:
        raise P3StageB1OuterCellShardError("config conditions are missing")
    if not datasets or not conditions or len(set(datasets)) != len(datasets) or len(
        set(conditions)
    ) != len(conditions):
        raise P3StageB1OuterCellShardError("config dataset/condition grid is invalid")
    return datasets, conditions


def _layout_descriptor(source_parameter_layout: Mapping[str, Any]) -> tuple[
    tuple[str, ...], tuple[int, ...], tuple[int, ...], str
]:
    layout = _mapping(source_parameter_layout, label="source_parameter_layout")
    names_raw = layout.get("names")
    offsets_raw = layout.get("offsets")
    shapes_raw = layout.get("shapes")
    if any(
        isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
        for value in (names_raw, offsets_raw, shapes_raw)
    ):
        raise P3StageB1OuterCellShardError("source parameter layout is malformed")
    names = tuple(names_raw)
    offsets = tuple(offsets_raw)
    if (
        not names
        or len(names) != len(offsets)
        or len(names) != len(shapes_raw)
        or len(set(names)) != len(names)
        or not all(isinstance(value, str) and value for value in names)
    ):
        raise P3StageB1OuterCellShardError("source parameter layout topology differs")
    cursor = 0
    ends: list[int] = []
    for index, (offset, shape) in enumerate(zip(offsets, shapes_raw, strict=True)):
        if isinstance(offset, bool) or not isinstance(offset, int) or offset != cursor:
            raise P3StageB1OuterCellShardError("source layout offsets are not canonical")
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence) or not shape:
            raise P3StageB1OuterCellShardError("source layout shape is invalid")
        size = 1
        for dimension in shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise P3StageB1OuterCellShardError("source layout dimension is invalid")
            size *= dimension
        cursor += size
        ends.append(cursor)
    if cursor != SCALAR_PARAMETER_COUNT or layout.get("scalar_count") != cursor:
        raise P3StageB1OuterCellShardError("source layout scalar count differs")
    layout_sha = _sha256(layout.get("layout_sha256"), label="layout_sha256")
    return names, tuple(int(value) for value in offsets), tuple(ends), layout_sha


def build_coarse_group_layout(
    *,
    protocol_id: str,
    source_parameter_layout: Mapping[str, Any],
    group_parameter_names: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Build the compact P0--P4 span map used by the CPU verifier."""

    if not isinstance(protocol_id, str) or not protocol_id:
        raise P3StageB1OuterCellShardError("protocol_id must be non-empty")
    names, offsets, ends, layout_sha = _layout_descriptor(source_parameter_layout)
    if not isinstance(group_parameter_names, Mapping) or tuple(
        group_parameter_names
    ) != GROUP_IDS:
        raise P3StageB1OuterCellShardError("group order must be exactly P0--P4")
    spans_by_name = dict(zip(names, zip(offsets, ends, strict=True), strict=True))
    groups: dict[str, Any] = {}
    for group_id in GROUP_IDS:
        raw = group_parameter_names[group_id]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise P3StageB1OuterCellShardError(f"{group_id} names must be a sequence")
        selected = tuple(raw)
        canonical = tuple(name for name in names if name in set(selected))
        if selected != canonical or len(set(selected)) != len(selected):
            raise P3StageB1OuterCellShardError(
                f"{group_id} names must be unique and in source-layout order"
            )
        spans = [
            {"parameter_name": name, "start": spans_by_name[name][0], "end": spans_by_name[name][1]}
            for name in selected
        ]
        scalar_count = sum(value["end"] - value["start"] for value in spans)
        if scalar_count != GROUP_SCALAR_COUNTS[group_id]:
            raise P3StageB1OuterCellShardError(
                f"{group_id} scalar count differs; expected={GROUP_SCALAR_COUNTS[group_id]}"
            )
        groups[group_id] = {
            "parameter_tensor_count": len(selected),
            "parameter_scalar_count": scalar_count,
            "spans": spans,
        }
    if tuple(group_parameter_names["P0"]) != names:
        raise P3StageB1OuterCellShardError("P0 must equal the complete source layout")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": GROUP_LAYOUT_ARTIFACT_TYPE,
        "protocol_id": protocol_id,
        "source_layout_sha256": layout_sha,
        "source_parameter_tensor_count": len(names),
        "source_scalar_parameter_count": SCALAR_PARAMETER_COUNT,
        "group_order": list(GROUP_IDS),
        "groups": groups,
    }


def validate_coarse_group_layout(
    value: Mapping[str, Any],
    *,
    protocol_id: str,
    expected_layout_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, np.ndarray]]:
    layout = _exact(
        value,
        {
            "schema_version",
            "artifact_type",
            "protocol_id",
            "source_layout_sha256",
            "source_parameter_tensor_count",
            "source_scalar_parameter_count",
            "group_order",
            "groups",
        },
        label="coarse group layout",
    )
    if (
        layout["schema_version"] != SCHEMA_VERSION
        or layout["artifact_type"] != GROUP_LAYOUT_ARTIFACT_TYPE
        or layout["protocol_id"] != protocol_id
        or layout["source_layout_sha256"] != expected_layout_sha256
        or layout["source_scalar_parameter_count"] != SCALAR_PARAMETER_COUNT
        or layout["group_order"] != list(GROUP_IDS)
    ):
        raise P3StageB1OuterCellShardError("coarse group layout identity differs")
    tensor_total = _integer(
        layout["source_parameter_tensor_count"],
        label="source_parameter_tensor_count",
        minimum=1,
    )
    groups = _mapping(layout["groups"], label="coarse groups")
    if set(groups) != set(GROUP_IDS):
        raise P3StageB1OuterCellShardError("coarse group keys/order differ")
    indices: dict[str, np.ndarray] = {}
    names_by_group: dict[str, tuple[str, ...]] = {}
    for group_id in GROUP_IDS:
        group = _exact(
            groups[group_id],
            {"parameter_tensor_count", "parameter_scalar_count", "spans"},
            label=f"coarse group {group_id}",
        )
        spans = group["spans"]
        if isinstance(spans, (str, bytes, Mapping)) or not isinstance(spans, Sequence):
            raise P3StageB1OuterCellShardError(f"{group_id} spans must be a sequence")
        flat: list[int] = []
        names: list[str] = []
        previous_end = -1
        for position, raw in enumerate(spans):
            span = _exact(
                raw,
                {"parameter_name", "start", "end"},
                label=f"{group_id}.spans[{position}]",
            )
            name = span["parameter_name"]
            start = _integer(span["start"], label=f"{group_id} span start")
            end = _integer(span["end"], label=f"{group_id} span end", minimum=1)
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or start >= end
                or end > SCALAR_PARAMETER_COUNT
                or start < previous_end
            ):
                raise P3StageB1OuterCellShardError(f"{group_id} span is invalid")
            previous_end = end
            names.append(name)
            flat.extend(range(start, end))
        if (
            len(spans) != group["parameter_tensor_count"]
            or len(flat) != group["parameter_scalar_count"]
            or len(flat) != GROUP_SCALAR_COUNTS[group_id]
            or len(set(flat)) != len(flat)
        ):
            raise P3StageB1OuterCellShardError(f"{group_id} span counts differ")
        indices[group_id] = np.asarray(flat, dtype=np.int64)
        names_by_group[group_id] = tuple(names)
    if (
        len(names_by_group["P0"]) != tensor_total
        or not np.array_equal(indices["P0"], np.arange(SCALAR_PARAMETER_COUNT))
    ):
        raise P3StageB1OuterCellShardError("P0 is not the complete flat layout")
    for smaller, larger in zip(GROUP_IDS[1:-1], GROUP_IDS[2:], strict=True):
        if not set(names_by_group[smaller]).issubset(names_by_group[larger]):
            raise P3StageB1OuterCellShardError(
                f"nested group invariant fails: {smaller} not subset of {larger}"
            )
    if not set(names_by_group["P4"]).issubset(names_by_group["P0"]):
        raise P3StageB1OuterCellShardError("P4 is not a subset of P0")
    return layout, indices


def _validate_code_seal(value: Any) -> Mapping[str, Any]:
    seal = _exact(value, {"files", "bundle_sha256"}, label="code_seal")
    files = seal["files"]
    if isinstance(files, (str, bytes, Mapping)) or not isinstance(files, Sequence) or not files:
        raise P3StageB1OuterCellShardError("code_seal.files must be non-empty")
    normalized: list[Mapping[str, Any]] = []
    paths: set[str] = set()
    for index, raw in enumerate(files):
        item = _exact(raw, {"path", "sha256"}, label=f"code_seal.files[{index}]")
        path = _canonical_relative(item["path"], label="code seal path")
        if path in paths:
            raise P3StageB1OuterCellShardError("code seal paths are duplicated")
        paths.add(path)
        _sha256(item["sha256"], label="code seal file SHA")
        normalized.append(item)
    expected = hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()
    if seal["bundle_sha256"] != expected:
        raise P3StageB1OuterCellShardError("code seal bundle SHA differs")
    return seal


def _validate_formal_execution(value: Any) -> Mapping[str, Any]:
    execution = _exact(value, _EXECUTION_FIELDS, label="manifest.execution")
    runtime = _exact(
        execution["runtime"], _RUNTIME_FIELDS, label="manifest.execution.runtime"
    )
    for field, expected in FORMAL_RUNTIME.items():
        observed = runtime[field]
        if type(observed) is not type(expected) or observed != expected:
            raise P3StageB1OuterCellShardError(
                "formal execution runtime differs from frozen CUDA runtime; "
                f"field={field}"
            )
    observed_counts = {
        key: execution[key] for key in FORMAL_EXECUTION_COUNTS
    }
    for field, expected in FORMAL_EXECUTION_COUNTS.items():
        observed = observed_counts[field]
        if type(observed) is not type(expected) or observed != expected:
            raise P3StageB1OuterCellShardError(
                "formal execution counts differ from frozen 64-image protocol; "
                f"field={field}"
            )
    return execution


def build_stage_b1_cell_payloads(
    *,
    protocol_id: str,
    config_sha256: str,
    cell: Mapping[str, Any],
    ordered_image_ids: Sequence[str],
    dataset_binding: Mapping[str, Any],
    parent_lineage: Mapping[str, Any],
    parameter_layout: Mapping[str, Any],
    region_gradient_basis: np.ndarray,
    episode_records: Sequence[Mapping[str, Any]],
    coarse_group_layout: Mapping[str, Any],
    outer_access_receipt_bytes: bytes,
    code_seal: Mapping[str, Any],
    execution: Mapping[str, Any],
    data_boundary: Mapping[str, Any] = MANIFEST_DATA_BOUNDARY,
    authorization: Mapping[str, Any] = CELL_AUTHORIZATION,
) -> dict[str, bytes]:
    """Build canonical cell members; publication remains an atomic no-replace rename."""

    config_digest = _sha256(config_sha256, label="config_sha256")
    cell_map = _mapping(cell, label="cell")
    if set(cell_map) != {"dataset", "condition", "corruption_family", "severity", "replicate"}:
        raise P3StageB1OuterCellShardError("cell fields differ")
    if cell_map["replicate"] != "R0":
        raise P3StageB1OuterCellShardError("only R0 is permitted")
    expected_family, expected_severity = _condition_parts(str(cell_map["condition"]))
    if (
        cell_map["corruption_family"] != expected_family
        or cell_map["severity"] != expected_severity
    ):
        raise P3StageB1OuterCellShardError("cell family/severity differs")
    image_ids = list(ordered_image_ids)
    if (
        len(image_ids) != FORMAL_IMAGE_COUNT
        or len(set(image_ids)) != FORMAL_IMAGE_COUNT
        or not all(isinstance(value, str) and value for value in image_ids)
    ):
        raise P3StageB1OuterCellShardError("ordered_image_ids must be 64 unique IDs")
    basis = np.asarray(region_gradient_basis)
    if (
        basis.shape != (FORMAL_IMAGE_COUNT, len(BASIS_ORDER), SCALAR_PARAMETER_COUNT)
        or basis.dtype.str != "<f4"
        or not basis.flags.c_contiguous
        or not np.isfinite(basis).all()
    ):
        raise P3StageB1OuterCellShardError("region gradient basis schema differs")
    if len(episode_records) != FORMAL_IMAGE_COUNT:
        raise P3StageB1OuterCellShardError("episode record count differs")
    layout_sha = str(parameter_layout.get("source_layout_sha256"))
    validate_coarse_group_layout(
        coarse_group_layout,
        protocol_id=protocol_id,
        expected_layout_sha256=layout_sha,
    )
    if not isinstance(outer_access_receipt_bytes, bytes) or not outer_access_receipt_bytes:
        raise P3StageB1OuterCellShardError("outer access receipt bytes are missing")
    # Require strict canonical JSON, but preserve the parent's exact bytes.
    try:
        json.loads(
            outer_access_receipt_bytes.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_object(pairs, "outer access receipt"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                P3StageB1OuterCellShardError(f"outer access receipt contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P3StageB1OuterCellShardError("outer access receipt is not JSON") from exc
    _validate_code_seal(code_seal)
    basis_bytes = _npy_bytes(basis)
    records_bytes = canonical_jsonl_bytes(episode_records)
    layout_bytes = canonical_json_bytes(coarse_group_layout, newline=True)
    payload = {
        REGION_GRADIENT_BASIS_FILENAME: basis_bytes,
        EPISODE_RECORDS_FILENAME: records_bytes,
        COARSE_GROUP_LAYOUT_FILENAME: layout_bytes,
        OUTER_ACCESS_RECEIPT_FILENAME: outer_access_receipt_bytes,
    }
    ids_sha = hashlib.sha256(canonical_json_bytes(image_ids)).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": protocol_id,
        "config_sha256": config_digest,
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
        "array": {
            "path": REGION_GRADIENT_BASIS_FILENAME,
            "sha256": hashlib.sha256(basis_bytes).hexdigest(),
            "shape": [FORMAL_IMAGE_COUNT, len(BASIS_ORDER), SCALAR_PARAMETER_COUNT],
            "dtype": "<f4",
            "c_order": True,
            "finite": True,
            "lossless": True,
            "basis_order": list(BASIS_ORDER),
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
    complete_without_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": protocol_id,
        "complete": True,
        "config_sha256": config_digest,
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
        "paper_result": False,
        "paper_test_result": False,
        "candidate_selection_performed": False,
        "stage_b3_authorized": False,
        "p5_authorized": False,
    }
    payload[COMPLETE_FILENAME] = canonical_json_bytes(
        complete_without_payload, newline=True
    )
    return payload


def _numeric_close(observed: Any, expected: float | None, *, label: str) -> None:
    if expected is None:
        if observed is not None:
            raise P3StageB1OuterCellShardError(f"{label} must be null")
        return
    value = _finite(observed, label=label)
    if not math.isclose(value, expected, rel_tol=2.0e-10, abs_tol=2.0e-12):
        raise P3StageB1OuterCellShardError(
            f"{label} differs; observed={value}, expected={expected}"
        )


def _alignment_metrics(
    entropy: np.ndarray | None, task: np.ndarray
) -> dict[str, Any]:
    task_norm = float(np.linalg.norm(task))
    if entropy is None:
        return {
            "estimable": False,
            "not_estimable_reason": None,
            "entropy_gradient_norm": None,
            "task_gradient_norm": task_norm,
            "entropy_task_dot": None,
            "entropy_task_cosine": None,
            "task_projection": None,
            "unit_descent_task_change": None,
            "direction": "undefined",
            "cosine_status": "not_estimable_region_absent",
        }
    entropy_norm = float(np.linalg.norm(entropy))
    dot = float(np.dot(entropy, task))
    task_projection = None if task_norm == 0.0 else dot / task_norm
    unit_change = None if entropy_norm == 0.0 else -dot / entropy_norm
    if entropy_norm == 0.0 and task_norm == 0.0:
        cosine, status = None, "not_estimable_both_gradients_zero"
    elif entropy_norm == 0.0:
        cosine, status = None, "not_estimable_entropy_gradient_zero"
    elif task_norm == 0.0:
        cosine, status = None, "not_estimable_task_gradient_zero"
    else:
        cosine = float(np.clip(dot / (entropy_norm * task_norm), -1.0, 1.0))
        status = "estimable"
    if entropy_norm == 0.0 or task_norm == 0.0:
        direction = "undefined"
    elif unit_change is not None and unit_change < 0.0:
        direction = "beneficial"
    elif unit_change is not None and unit_change > 0.0:
        direction = "harmful"
    else:
        direction = "neutral"
    return {
        "estimable": True,
        "not_estimable_reason": None,
        "entropy_gradient_norm": entropy_norm,
        "task_gradient_norm": task_norm,
        "entropy_task_dot": dot,
        "entropy_task_cosine": cosine,
        "task_projection": task_projection,
        "unit_descent_task_change": unit_change,
        "direction": direction,
        "cosine_status": status,
    }


def _ratio(
    numerator: float | None,
    denominator: float | None,
    *,
    absent_reason: str | None,
) -> dict[str, Any]:
    if numerator is None or denominator is None:
        return {
            "value": None,
            "status": "not_estimable_region_absent",
            "not_estimable_reason": absent_reason,
        }
    if denominator == 0.0:
        return {
            "value": None,
            "status": "not_estimable_denominator_zero",
            "not_estimable_reason": "foreground_gradient_zero",
        }
    return {
        "value": numerator / denominator,
        "status": "estimable",
        "not_estimable_reason": None,
    }


def _cross_region_metrics(
    foreground_add: np.ndarray | None,
    background_add: np.ndarray | None,
    foreground_conditional: np.ndarray | None,
    background_conditional: np.ndarray | None,
    task: np.ndarray,
) -> dict[str, Any]:
    if foreground_add is None or background_add is None:
        reasons = []
        if foreground_add is None:
            reasons.append("empty_foreground")
        if background_add is None:
            reasons.append("empty_background")
        pair = {
            "estimable": False,
            "not_estimable_reason": "+".join(reasons),
            "foreground_gradient_norm": None,
            "background_gradient_norm": None,
            "foreground_background_dot": None,
            "foreground_background_cosine": None,
            "cosine_status": "not_estimable_region_absent",
        }
        fg_norm = bg_norm = None
        fg_dot = bg_dot = None
    else:
        fg_norm = float(np.linalg.norm(foreground_add))
        bg_norm = float(np.linalg.norm(background_add))
        dot = float(np.dot(foreground_add, background_add))
        if fg_norm == 0.0 and bg_norm == 0.0:
            cosine, status = None, "not_estimable_both_gradients_zero"
        elif fg_norm == 0.0:
            cosine, status = None, "not_estimable_foreground_gradient_zero"
        elif bg_norm == 0.0:
            cosine, status = None, "not_estimable_background_gradient_zero"
        else:
            cosine = float(np.clip(dot / (fg_norm * bg_norm), -1.0, 1.0))
            status = "estimable"
        pair = {
            "estimable": True,
            "not_estimable_reason": None,
            "foreground_gradient_norm": fg_norm,
            "background_gradient_norm": bg_norm,
            "foreground_background_dot": dot,
            "foreground_background_cosine": cosine,
            "cosine_status": status,
        }
        fg_dot = float(np.dot(foreground_add, task))
        bg_dot = float(np.dot(background_add, task))
    absent_reason = (
        "empty_foreground_or_background"
        if foreground_add is None or background_add is None
        else None
    )
    additive_ratio = _ratio(bg_norm, fg_norm, absent_reason=absent_reason)
    fg_cond_norm = (
        None if foreground_conditional is None else float(np.linalg.norm(foreground_conditional))
    )
    bg_cond_norm = (
        None if background_conditional is None else float(np.linalg.norm(background_conditional))
    )
    conditional_ratio = _ratio(
        bg_cond_norm,
        fg_cond_norm,
        absent_reason=(
            "empty_foreground_or_background"
            if foreground_conditional is None or background_conditional is None
            else None
        ),
    )
    full_dot = (
        float(np.dot(foreground_add + background_add, task))
        if foreground_add is not None and background_add is not None
        else float(np.dot(
            (np.zeros_like(task) if foreground_add is None else foreground_add)
            + (np.zeros_like(task) if background_add is None else background_add),
            task,
        ))
    )
    denominator = (
        None if fg_dot is None or bg_dot is None else abs(fg_dot) + abs(bg_dot)
    )
    task_norm = float(np.linalg.norm(task))
    if denominator is None:
        cancellation = {
            "value": None,
            "status": "not_estimable_region_absent",
            "not_estimable_reason": (
                "empty_foreground" if foreground_add is None else "empty_background"
            ),
        }
    elif task_norm == 0.0:
        cancellation = {
            "value": None,
            "status": "not_estimable_task_gradient_zero",
            "not_estimable_reason": "task_gradient_zero",
        }
    elif denominator == 0.0:
        cancellation = {
            "value": None,
            "status": "not_estimable_projection_denominator_zero",
            "not_estimable_reason": "foreground_and_background_task_dots_zero",
        }
    else:
        cancellation = {
            "value": min(1.0, max(0.0, 1.0 - abs(full_dot) / denominator)),
            "status": "estimable",
            "not_estimable_reason": None,
        }
    cancellation.update(
        {
            "formula": (
                "1-abs(full_task_dot)/(abs(foreground_task_dot)+"
                "abs(background_task_dot))"
            ),
            "foreground_task_dot": fg_dot,
            "background_task_dot": bg_dot,
            "full_task_dot": full_dot,
        }
    )
    return {
        "foreground_background_additive_alignment": pair,
        "background_to_foreground_additive_norm_ratio": additive_ratio,
        "background_to_foreground_conditional_norm_ratio": conditional_ratio,
        "projection_cancellation_ratio": cancellation,
    }


def _validate_group_report(
    raw: Any,
    *,
    group_id: str,
    indices: np.ndarray,
    basis: np.ndarray,
    task: np.ndarray | None,
    target: Mapping[str, Any],
    layout_group: Mapping[str, Any],
) -> None:
    group = _exact(
        raw,
        {
            "parameter_tensor_count",
            "parameter_scalar_count",
            "task_gradient_norm",
            "additive_entropy_task_alignment",
            "conditional_entropy_task_alignment",
            "additive_gradient_norms",
            "cross_region",
        },
        label=f"groups.{group_id}",
    )
    if (
        group["parameter_tensor_count"] != layout_group["parameter_tensor_count"]
        or group["parameter_scalar_count"] != layout_group["parameter_scalar_count"]
    ):
        raise P3StageB1OuterCellShardError(f"{group_id} report layout counts differ")
    # If live parent task evidence is intentionally not requested, retain strict
    # schema/finiteness checking but skip derivational dot/cosine recomputation.
    selected_basis = basis[:, indices].astype(np.float64, copy=False)
    fg_sub, fg_supra, background = selected_basis
    foreground_add = fg_sub + fg_supra
    full_add = foreground_add + background
    additive = {
        "foreground_subthreshold_add": fg_sub,
        "foreground_suprathreshold_add": fg_supra,
        "background_add": background,
        "foreground_add": foreground_add,
        "full_add": full_add,
    }
    additive_raw = _mapping(group["additive_gradient_norms"], label=f"{group_id}.additive")
    if set(additive_raw) != set(ADDITIVE_COMPONENTS):
        raise P3StageB1OuterCellShardError(f"{group_id} additive component order differs")
    for name, vector in additive.items():
        present = {
            "foreground_subthreshold_add": int(target["foreground_subthreshold_pixel_count"]) > 0,
            "foreground_suprathreshold_add": int(target["foreground_suprathreshold_pixel_count"]) > 0,
            "background_add": int(target["background_pixel_count"]) > 0,
            "foreground_add": int(target["foreground_pixel_count"]) > 0,
            "full_add": True,
        }[name]
        _numeric_close(
            additive_raw[name],
            float(np.linalg.norm(vector)) if present else None,
            label=f"{group_id}.{name}.norm",
        )
    target_total = int(target["total_pixel_count"])
    count_by_component = {
        "full_entropy_mean": target_total,
        "foreground_entropy_mean": int(target["foreground_pixel_count"]),
        "background_entropy_mean": int(target["background_pixel_count"]),
        "foreground_subthreshold_entropy_mean": int(
            target["foreground_subthreshold_pixel_count"]
        ),
        "foreground_suprathreshold_entropy_mean": int(
            target["foreground_suprathreshold_pixel_count"]
        ),
    }
    additive_by_component = {
        "full_entropy_mean": full_add,
        "foreground_entropy_mean": foreground_add,
        "background_entropy_mean": background,
        "foreground_subthreshold_entropy_mean": fg_sub,
        "foreground_suprathreshold_entropy_mean": fg_supra,
    }
    selected_task = None if task is None else task[indices].astype(np.float64, copy=False)
    alignment_fields = {
        "estimable",
        "not_estimable_reason",
        "entropy_gradient_norm",
        "task_gradient_norm",
        "entropy_task_dot",
        "entropy_task_cosine",
        "task_projection",
        "unit_descent_task_change",
        "direction",
        "cosine_status",
    }

    def validate_alignment_set(
        raw_set: Any,
        *,
        expected_names: tuple[str, ...],
        vectors: Mapping[str, np.ndarray],
        counts: Mapping[str, int],
        conditional: bool,
        label: str,
    ) -> None:
        alignments = _mapping(raw_set, label=label)
        if set(alignments) != set(expected_names):
            raise P3StageB1OuterCellShardError(f"{label} component order differs")
        absent_reasons = {
            "foreground_entropy_mean": "empty_foreground",
            "background_entropy_mean": "empty_background",
            "foreground_subthreshold_entropy_mean": "empty_foreground_subthreshold",
            "foreground_suprathreshold_entropy_mean": "empty_foreground_suprathreshold",
            "foreground_entropy_add": "empty_foreground",
            "background_entropy_add": "empty_background",
            "foreground_subthreshold_entropy_add": "empty_foreground_subthreshold",
            "foreground_suprathreshold_entropy_add": "empty_foreground_suprathreshold",
        }
        for name in expected_names:
            alignment = _exact(
                alignments[name], alignment_fields, label=f"{label}.{name}"
            )
            region_count = counts[name]
            absent = region_count == 0
            if alignment["estimable"] is not (not absent):
                raise P3StageB1OuterCellShardError(
                    f"{label}.{name} estimable flag differs"
                )
            reason = absent_reasons.get(name) if absent else None
            if alignment["not_estimable_reason"] != reason:
                raise P3StageB1OuterCellShardError(
                    f"{label}.{name} absent reason differs"
                )
            vector = None if absent else vectors[name]
            if vector is not None and conditional:
                vector = vector * (target_total / region_count)
            if selected_task is None:
                for field in (
                    "entropy_gradient_norm",
                    "task_gradient_norm",
                    "entropy_task_dot",
                    "entropy_task_cosine",
                    "task_projection",
                    "unit_descent_task_change",
                ):
                    _optional_finite(
                        alignment[field], label=f"{label}.{name}.{field}"
                    )
                if alignment["direction"] not in {
                    "beneficial", "harmful", "neutral", "undefined"
                } or not isinstance(alignment["cosine_status"], str):
                    raise P3StageB1OuterCellShardError(
                        f"{label}.{name} direction/status differs"
                    )
            else:
                expected = _alignment_metrics(vector, selected_task)
                expected["not_estimable_reason"] = reason
                for field in (
                    "entropy_gradient_norm",
                    "task_gradient_norm",
                    "entropy_task_dot",
                    "entropy_task_cosine",
                    "task_projection",
                    "unit_descent_task_change",
                ):
                    _numeric_close(
                        alignment[field],
                        expected[field],
                        label=f"{label}.{name}.{field}",
                    )
                for field in (
                    "estimable", "not_estimable_reason", "direction", "cosine_status"
                ):
                    if alignment[field] != expected[field]:
                        raise P3StageB1OuterCellShardError(
                            f"{label}.{name}.{field} differs"
                        )

    validate_alignment_set(
        group["conditional_entropy_task_alignment"],
        expected_names=CONDITIONAL_COMPONENTS,
        vectors=additive_by_component,
        counts=count_by_component,
        conditional=True,
        label=f"{group_id}.conditional",
    )
    additive_alignment_vectors = {
        "full_entropy_mean": full_add,
        "foreground_entropy_add": foreground_add,
        "background_entropy_add": background,
        "foreground_subthreshold_entropy_add": fg_sub,
        "foreground_suprathreshold_entropy_add": fg_supra,
    }
    additive_alignment_counts = {
        "full_entropy_mean": target_total,
        "foreground_entropy_add": int(target["foreground_pixel_count"]),
        "background_entropy_add": int(target["background_pixel_count"]),
        "foreground_subthreshold_entropy_add": int(
            target["foreground_subthreshold_pixel_count"]
        ),
        "foreground_suprathreshold_entropy_add": int(
            target["foreground_suprathreshold_pixel_count"]
        ),
    }
    validate_alignment_set(
        group["additive_entropy_task_alignment"],
        expected_names=ADDITIVE_ALIGNMENT_COMPONENTS,
        vectors=additive_alignment_vectors,
        counts=additive_alignment_counts,
        conditional=False,
        label=f"{group_id}.additive_alignment",
    )
    if selected_task is not None:
        _numeric_close(group["task_gradient_norm"], float(np.linalg.norm(selected_task)), label=f"{group_id}.task_gradient_norm")
    else:
        _finite(group["task_gradient_norm"], label=f"{group_id}.task_gradient_norm")
    cross_raw = _mapping(group["cross_region"], label=f"{group_id}.cross")
    expected_cross = None
    if selected_task is not None:
        fg_present = int(target["foreground_pixel_count"]) > 0
        bg_present = int(target["background_pixel_count"]) > 0
        foreground_conditional = (
            foreground_add * (target_total / int(target["foreground_pixel_count"]))
            if fg_present
            else None
        )
        background_conditional = (
            background * (target_total / int(target["background_pixel_count"]))
            if bg_present
            else None
        )
        expected_cross = _cross_region_metrics(
            foreground_add if fg_present else None,
            background if bg_present else None,
            foreground_conditional,
            background_conditional,
            selected_task,
        )
    expected_cross_fields = {
        "foreground_background_additive_alignment",
        "background_to_foreground_additive_norm_ratio",
        "background_to_foreground_conditional_norm_ratio",
        "projection_cancellation_ratio",
    }
    if set(cross_raw) != expected_cross_fields:
        raise P3StageB1OuterCellShardError(f"{group_id} cross metric fields differ")
    if expected_cross is None:
        # Live-parent-free verification still rejects NaN/Inf recursively.
        _validate_finite_json_tree(cross_raw, label=f"{group_id}.cross")
    else:
        _compare_json_metric_tree(
            cross_raw, expected_cross, label=f"{group_id}.cross"
        )


def _validate_record(
    raw: Any,
    *,
    position: int,
    manifest: Mapping[str, Any],
    basis: np.ndarray,
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
    image_ids = manifest["ordered_image_ids"]
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
        or record["image_id"] != image_ids[position]
        or record["data_boundary"] != CELL_DATA_BOUNDARY
        or record["authorization"] != CELL_AUTHORIZATION
    ):
        raise P3StageB1OuterCellShardError(f"record identity differs at {position}")
    target = _exact(record["target"], _TARGET_FIELDS, label=f"records[{position}].target")
    total = _integer(target["total_pixel_count"], label="total pixels", minimum=1)
    foreground = _integer(target["foreground_pixel_count"], label="foreground pixels")
    background = _integer(target["background_pixel_count"], label="background pixels")
    sub = _integer(target["foreground_subthreshold_pixel_count"], label="subthreshold pixels")
    supra = _integer(target["foreground_suprathreshold_pixel_count"], label="suprathreshold pixels")
    target_sum = _finite(target["target_value_sum"], label="target value sum")
    if (
        target["target_present"] is not (foreground > 0)
        or foreground + background != total
        or sub + supra != foreground
        or target_sum < 0.0
        or target_sum > total
        or target["partition_disjoint"] is not True
        or target["partition_exhaustive"] is not True
    ):
        raise P3StageB1OuterCellShardError("target partition/count fields differ")
    _sha256(target["target_slice_sha256"], label="target_slice_sha256")
    source = _exact(
        record["source_integrity"],
        _SOURCE_INTEGRITY_FIELDS,
        label=f"records[{position}].source_integrity",
    )
    for field in (
        "parent_source_logits_slice_sha256",
        "recomputed_source_logits_slice_sha256",
        "source_state_before_sha256",
        "source_state_after_sha256",
        "rng_before_sha256",
        "rng_after_sha256",
    ):
        _sha256(source[field], label=field)
    if (
        source["source_logits_bit_exact"] is not True
        or source["state_restored"] is not True
        or source["rng_restored"] is not True
        or source["source_state_before_sha256"] != source["source_state_after_sha256"]
        or source["rng_before_sha256"] != source["rng_after_sha256"]
    ):
        raise P3StageB1OuterCellShardError("Source/RNG integrity differs")
    if candidate_source_logits is not None and outer_source_logits is not None:
        candidate_sha = array_slice_sha256(candidate_source_logits[position])
        outer_sha = array_slice_sha256(outer_source_logits[position])
        if (
            not np.array_equal(candidate_source_logits[position], outer_source_logits[position])
            or source["parent_source_logits_slice_sha256"] != candidate_sha
            or source["recomputed_source_logits_slice_sha256"] != outer_sha
            or candidate_sha != outer_sha
        ):
            raise P3StageB1OuterCellShardError("Source logit slice integrity differs")
    integrity = _exact(
        record["gradient_integrity"],
        _GRADIENT_INTEGRITY_FIELDS,
        label=f"records[{position}].gradient_integrity",
    )
    if (
        integrity["basis_order"] != list(BASIS_ORDER)
        or integrity["basis_slice_sha256"] != array_slice_sha256(basis[position])
        or integrity["finite"] is not True
        or integrity["parent_candidate_slice_count"] != 10
        or integrity["parent_consistency_passed"] is not True
    ):
        raise P3StageB1OuterCellShardError("gradient integrity identity differs")
    _numeric_close(integrity["max_abs_tolerance"], max_abs_tolerance, label="max_abs_tolerance")
    _numeric_close(integrity["relative_l2_tolerance"], relative_l2_tolerance, label="relative_l2_tolerance")
    full = basis[position].astype(np.float64).sum(axis=0)
    if parent_entropy is not None:
        absolute_errors: list[float] = []
        relative_errors: list[float] = []
        for candidate in parent_entropy[position].astype(np.float64, copy=False):
            residual = full - candidate
            absolute_errors.append(float(np.max(np.abs(residual))))
            relative_errors.append(
                float(np.linalg.norm(residual) / max(float(np.linalg.norm(candidate)), 1.0e-12))
            )
        expected_abs = max(absolute_errors)
        expected_relative = max(relative_errors)
        _numeric_close(integrity["parent_max_abs_error"], expected_abs, label="parent max abs")
        _numeric_close(integrity["parent_max_relative_l2_error"], expected_relative, label="parent max relative l2")
        if expected_abs > max_abs_tolerance or expected_relative > relative_l2_tolerance:
            raise P3StageB1OuterCellShardError("parent entropy consistency tolerance fails")
    else:
        _finite(integrity["parent_max_abs_error"], label="parent_max_abs_error")
        _finite(integrity["parent_max_relative_l2_error"], label="parent_max_relative_l2_error")
    groups = _mapping(record["groups"], label=f"records[{position}].groups")
    if set(groups) != set(GROUP_IDS):
        raise P3StageB1OuterCellShardError("record group keys/order differ")
    task = None if parent_task is None else parent_task[position].astype(np.float64, copy=False)
    layout_groups = _mapping(group_layout["groups"], label="layout groups")
    for group_id in GROUP_IDS:
        _validate_group_report(
            groups[group_id],
            group_id=group_id,
            indices=group_indices[group_id],
            basis=basis[position],
            task=task,
            target=target,
            layout_group=layout_groups[group_id],
        )
    return record


def _live_parent_context(
    *, repository_root: Path, lineage: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bytes]:
    fields = {
        "candidate_shard_path",
        "outer_shard_path",
        "candidate_manifest_sha256",
        "candidate_complete_sha256",
        "candidate_phase_receipt_sha256",
        "outer_manifest_sha256",
        "outer_complete_sha256",
        "outer_access_receipt_sha256",
        "parent_entropy_gradients_sha256",
        "parent_task_gradients_sha256",
    }
    parent = _exact(lineage, fields, label="parent_lineage")
    candidate = _repository_path(
        repository_root, parent["candidate_shard_path"], label="candidate_shard_path"
    )
    outer = _repository_path(
        repository_root, parent["outer_shard_path"], label="outer_shard_path"
    )
    bindings = (
        (candidate / "manifest.json", "candidate_manifest_sha256"),
        (candidate / "COMPLETE.json", "candidate_complete_sha256"),
        (candidate / PARENT_PHASE_RECEIPT_FILENAME, "candidate_phase_receipt_sha256"),
        (outer / "manifest.json", "outer_manifest_sha256"),
        (outer / "COMPLETE.json", "outer_complete_sha256"),
        (outer / OUTER_ACCESS_RECEIPT_FILENAME, "outer_access_receipt_sha256"),
        (candidate / "entropy_gradients.npy", "parent_entropy_gradients_sha256"),
        (outer / "supervised_gradients.npy", "parent_task_gradients_sha256"),
    )
    snapshots: dict[str, Any] = {}
    for path, field in bindings:
        snapshot = read_stable_regular_file(path)
        if snapshot.sha256 != _sha256(parent[field], label=field):
            raise P3StageB1OuterCellShardError(f"live parent hash differs: {field}")
        snapshots[field] = snapshot
    def load(path: Path, label: str) -> np.ndarray:
        return _load_npy(read_stable_regular_file(path).data, label=label)
    entropy = _load_npy(
        snapshots["parent_entropy_gradients_sha256"].data, label="parent entropy gradients"
    )
    task = _load_npy(
        snapshots["parent_task_gradients_sha256"].data, label="parent task gradients"
    )
    candidate_logits = load(candidate / "source_logits_pre.npy", "candidate Source logits")
    outer_logits = load(outer / "outer_source_logits.npy", "outer Source logits")
    for value, shape, label in (
        (entropy, (64, 10, SCALAR_PARAMETER_COUNT), "parent entropy gradients"),
        (task, (64, SCALAR_PARAMETER_COUNT), "parent task gradients"),
        (candidate_logits, (64, 1, 256, 256), "candidate Source logits"),
        (outer_logits, (64, 1, 256, 256), "outer Source logits"),
    ):
        if value.shape != shape or value.dtype.str != "<f4" or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise P3StageB1OuterCellShardError(f"{label} schema differs")
    return entropy, task, candidate_logits, outer_logits, snapshots[
        "outer_access_receipt_sha256"
    ].data


def verify_stage_b1_cell_shard(
    path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    config: Mapping[str, Any],
    expected_config_sha256: str,
    verify_live_parents: bool = True,
    expected_code_seal: Mapping[str, Any] | None = None,
) -> VerifiedStageB1CellShard:
    """Verify exact membership, hashes, lineage, arrays, records, and denial flags."""

    root = Path(os.path.abspath(os.fspath(path)))
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    protocol_id = str(config.get("protocol_id"))
    datasets, conditions = _config_sequences(config)
    config_sha = _sha256(expected_config_sha256, label="expected_config_sha256")
    try:
        snapshot = snapshot_regular_directory(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise P3StageB1OuterCellShardError(f"cannot snapshot B1 cell: {root}") from exc
    by_name = {member.path.name: member for member in snapshot.members}
    if set(by_name) != MEMBERS:
        raise P3StageB1OuterCellShardError("B1 cell member set differs")
    manifest = parse_canonical_json(
        by_name[MANIFEST_FILENAME].data, label=MANIFEST_FILENAME, newline=True
    )
    manifest_fields = {
        "schema_version", "artifact_type", "protocol_id", "config_sha256", "cell",
        "mode", "ordered_image_ids", "ordered_image_ids_sha256", "dataset_binding",
        "parent_lineage", "parameter_layout", "array", "records",
        "outer_access_receipt", "code_seal", "execution", "data_boundary", "authorization",
    }
    if set(manifest) != manifest_fields:
        raise P3StageB1OuterCellShardError("manifest fields differ")
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["artifact_type"] != ARTIFACT_TYPE
        or manifest["protocol_id"] != protocol_id
        or manifest["config_sha256"] != config_sha
    ):
        raise P3StageB1OuterCellShardError("manifest identity differs")
    cell = _exact(
        manifest["cell"],
        {"dataset", "condition", "corruption_family", "severity", "replicate"},
        label="manifest.cell",
    )
    dataset, condition = str(cell["dataset"]), str(cell["condition"])
    family, severity = _condition_parts(condition)
    if (
        dataset not in datasets or condition not in conditions or cell["corruption_family"] != family
        or cell["severity"] != severity or cell["replicate"] != "R0"
    ):
        raise P3StageB1OuterCellShardError("cell grid binding differs")
    expected_mode = {
        "formal": True, "dry_run": False, "image_count": 64, "record_count": 64,
        "record_order": "image_index_ascending",
    }
    if manifest["mode"] != expected_mode:
        raise P3StageB1OuterCellShardError("formal mode differs")
    image_ids = manifest["ordered_image_ids"]
    if (
        not isinstance(image_ids, list) or len(image_ids) != 64 or len(set(image_ids)) != 64
        or not all(isinstance(value, str) and value for value in image_ids)
    ):
        raise P3StageB1OuterCellShardError("ordered image IDs differ")
    ids_sha = hashlib.sha256(canonical_json_bytes(image_ids)).hexdigest()
    if manifest["ordered_image_ids_sha256"] != ids_sha:
        raise P3StageB1OuterCellShardError("ordered image ID SHA differs")
    dataset_config = _mapping(_mapping(config["datasets"], label="datasets")[dataset], label="dataset config")
    if ids_sha != dataset_config.get("ordered_pilot64_image_ids_sha256"):
        raise P3StageB1OuterCellShardError("ordered image IDs differ from frozen config")
    expected_dataset_binding = {
        "split_name": "train",
        "train_split_sha256": dataset_config["train_split_sha256"],
        "checkpoint_role": dataset_config["checkpoint_role"],
        "checkpoint_path": dataset_config["checkpoint_path"],
        "checkpoint_sha256": dataset_config["checkpoint_sha256"],
    }
    if manifest["dataset_binding"] != expected_dataset_binding:
        raise P3StageB1OuterCellShardError("dataset binding differs")
    parent_lineage = _exact(
        manifest["parent_lineage"],
        {
            "candidate_shard_path", "outer_shard_path", "candidate_manifest_sha256",
            "candidate_complete_sha256", "candidate_phase_receipt_sha256",
            "outer_manifest_sha256", "outer_complete_sha256", "outer_access_receipt_sha256",
            "parent_entropy_gradients_sha256", "parent_task_gradients_sha256",
        },
        label="parent_lineage",
    )
    for field in set(parent_lineage) - {"candidate_shard_path", "outer_shard_path"}:
        _sha256(parent_lineage[field], label=field)
    parent_cfg = _mapping(config["parent_cell_artifacts"], label="parent_cell_artifacts")
    expected_candidate_path = str(parent_cfg["candidate_shard_template"]).format(dataset=dataset, condition=condition)
    expected_outer_path = str(parent_cfg["outer_shard_template"]).format(dataset=dataset, condition=condition)
    if (
        parent_lineage["candidate_shard_path"] != expected_candidate_path
        or parent_lineage["outer_shard_path"] != expected_outer_path
    ):
        raise P3StageB1OuterCellShardError("parent cell paths differ from config")
    parameter_layout = _exact(
        manifest["parameter_layout"],
        {
            "parent_source_path", "parent_source_file_sha256", "source_layout_sha256",
            "source_parameter_tensor_count", "source_scalar_parameter_count",
            "coarse_group_layout_path", "coarse_group_layout_sha256", "group_order",
            "group_scalar_counts",
        },
        label="parameter_layout",
    )
    frozen_layout = _mapping(parent_cfg["parameter_layout"], label="frozen parameter layout")
    if (
        parameter_layout["source_layout_sha256"] != frozen_layout["layout_sha256"]
        or parameter_layout["source_parameter_tensor_count"] != frozen_layout["parameter_tensor_count"]
        or parameter_layout["source_scalar_parameter_count"] != SCALAR_PARAMETER_COUNT
        or parameter_layout["coarse_group_layout_path"] != COARSE_GROUP_LAYOUT_FILENAME
        or parameter_layout["coarse_group_layout_sha256"] != by_name[COARSE_GROUP_LAYOUT_FILENAME].sha256
        or parameter_layout["group_order"] != list(GROUP_IDS)
        or parameter_layout["group_scalar_counts"] != GROUP_SCALAR_COUNTS
    ):
        raise P3StageB1OuterCellShardError("manifest parameter layout differs")
    _sha256(parameter_layout["parent_source_file_sha256"], label="parent layout file SHA")
    expected_parent_layout_path = f"{expected_candidate_path}/{frozen_layout['filename']}"
    if parameter_layout["parent_source_path"] != expected_parent_layout_path:
        raise P3StageB1OuterCellShardError("parent parameter layout path differs")
    group_layout = parse_canonical_json(
        by_name[COARSE_GROUP_LAYOUT_FILENAME].data,
        label=COARSE_GROUP_LAYOUT_FILENAME,
        newline=True,
    )
    group_layout, group_indices = validate_coarse_group_layout(
        group_layout, protocol_id=protocol_id,
        expected_layout_sha256=str(parameter_layout["source_layout_sha256"]),
    )
    basis = _load_npy(by_name[REGION_GRADIENT_BASIS_FILENAME].data, label=REGION_GRADIENT_BASIS_FILENAME)
    if (
        basis.shape != (64, 3, SCALAR_PARAMETER_COUNT) or basis.dtype.str != "<f4"
        or not basis.flags.c_contiguous or not np.isfinite(basis).all()
    ):
        raise P3StageB1OuterCellShardError("region gradient basis schema differs")
    expected_array = {
        "path": REGION_GRADIENT_BASIS_FILENAME,
        "sha256": by_name[REGION_GRADIENT_BASIS_FILENAME].sha256,
        "shape": [64, 3, SCALAR_PARAMETER_COUNT], "dtype": "<f4", "c_order": True,
        "finite": True, "lossless": True, "basis_order": list(BASIS_ORDER),
    }
    if manifest["array"] != expected_array:
        raise P3StageB1OuterCellShardError("array manifest differs")
    records = parse_canonical_jsonl(
        by_name[EPISODE_RECORDS_FILENAME].data, count=64, label=EPISODE_RECORDS_FILENAME
    )
    if manifest["records"] != {
        "path": EPISODE_RECORDS_FILENAME, "sha256": by_name[EPISODE_RECORDS_FILENAME].sha256,
        "count": 64, "order": "image_index_ascending",
    }:
        raise P3StageB1OuterCellShardError("record manifest differs")
    receipt_ref = {
        "path": OUTER_ACCESS_RECEIPT_FILENAME,
        "sha256": by_name[OUTER_ACCESS_RECEIPT_FILENAME].sha256,
        "copied_bit_exact_from_parent": True,
    }
    if manifest["outer_access_receipt"] != receipt_ref or receipt_ref["sha256"] != parent_lineage["outer_access_receipt_sha256"]:
        raise P3StageB1OuterCellShardError("outer access receipt binding differs")
    code_seal = _validate_code_seal(manifest["code_seal"])
    if expected_code_seal is not None and code_seal != expected_code_seal:
        raise P3StageB1OuterCellShardError("code seal differs from expected live seal")
    if manifest["data_boundary"] != MANIFEST_DATA_BOUNDARY or manifest["authorization"] != CELL_AUTHORIZATION:
        raise P3StageB1OuterCellShardError("manifest boundary/authorization differs")
    _validate_formal_execution(manifest["execution"])
    max_abs_tolerance = float(_mapping(config["entropy_gradient_consistency"], label="entropy consistency")["max_abs_tolerance"])
    relative_l2_tolerance = float(config["entropy_gradient_consistency"]["relative_l2_tolerance"])
    parent_entropy = parent_task = candidate_logits = outer_logits = None
    if verify_live_parents:
        parent_entropy, parent_task, candidate_logits, outer_logits, parent_receipt = _live_parent_context(
            repository_root=repository, lineage=parent_lineage
        )
        if parent_receipt != by_name[OUTER_ACCESS_RECEIPT_FILENAME].data:
            raise P3StageB1OuterCellShardError("copied outer access receipt is not bit exact")
        parent_layout_snapshot = read_stable_regular_file(repository / parameter_layout["parent_source_path"])
        if parent_layout_snapshot.sha256 != parameter_layout["parent_source_file_sha256"]:
            raise P3StageB1OuterCellShardError("live parent parameter layout SHA differs")
    else:
        # Still require strict JSON and no duplicate keys in the copied receipt.
        receipt_data = by_name[OUTER_ACCESS_RECEIPT_FILENAME].data
        try:
            json.loads(
                receipt_data.decode("utf-8"),
                object_pairs_hook=lambda pairs: _unique_object(pairs, "outer access receipt"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    P3StageB1OuterCellShardError(f"outer access receipt contains {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise P3StageB1OuterCellShardError("outer access receipt is not JSON") from exc
    validated_records = tuple(
        _validate_record(
            record, position=index, manifest=manifest, basis=basis,
            group_layout=group_layout, group_indices=group_indices,
            parent_entropy=parent_entropy, parent_task=parent_task,
            candidate_source_logits=candidate_logits, outer_source_logits=outer_logits,
            max_abs_tolerance=max_abs_tolerance,
            relative_l2_tolerance=relative_l2_tolerance,
        )
        for index, record in enumerate(records)
    )
    target_identities = [
        {
            "image_index": record["image_index"], "image_id": record["image_id"],
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
        "schema_version": SCHEMA_VERSION, "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": protocol_id, "complete": True, "config_sha256": config_sha,
        "cell": dict(cell), "image_count": 64, "record_count": 64,
        "manifest": {"path": MANIFEST_FILENAME, "sha256": by_name[MANIFEST_FILENAME].sha256},
        "payload_files": [
            {"path": name, "sha256": by_name[name].sha256}
            for name in sorted(set(by_name) - {COMPLETE_FILENAME})
        ],
        "atomic_no_replace": True, "immutable": True, "paper_result": False,
        "paper_test_result": False, "candidate_selection_performed": False,
        "stage_b3_authorized": False, "p5_authorized": False,
    }
    if complete != expected_complete:
        raise P3StageB1OuterCellShardError("COMPLETE does not rebuild from payload")
    return VerifiedStageB1CellShard(
        path=root, dataset=dataset, condition=condition, corruption_family=family,
        severity=severity, replicate="R0", image_count=64, record_count=64,
        ordered_image_ids_sha256=ids_sha, target_identity_sha256=target_identity_sha,
        manifest_sha256=by_name[MANIFEST_FILENAME].sha256,
        complete_sha256=by_name[COMPLETE_FILENAME].sha256,
        basis_sha256=by_name[REGION_GRADIENT_BASIS_FILENAME].sha256,
        records_sha256=by_name[EPISODE_RECORDS_FILENAME].sha256,
        group_layout_sha256=by_name[COARSE_GROUP_LAYOUT_FILENAME].sha256,
        outer_access_receipt_sha256=by_name[OUTER_ACCESS_RECEIPT_FILENAME].sha256,
        records=validated_records,
    )


__all__ = [
    "ADDITIVE_COMPONENTS", "ANALYZER_BASIS_NAMES", "ARTIFACT_TYPE", "BASIS_ORDER",
    "CELL_AUTHORIZATION", "CELL_DATA_BOUNDARY", "COARSE_GROUP_LAYOUT_FILENAME",
    "COMPLETE_ARTIFACT_TYPE", "COMPLETE_FILENAME", "CONDITIONAL_COMPONENTS",
    "EPISODE_RECORDS_FILENAME", "FORMAL_EXECUTION_COUNTS", "FORMAL_IMAGE_COUNT",
    "FORMAL_RUNTIME", "GROUP_IDS",
    "GROUP_LAYOUT_ARTIFACT_TYPE", "GROUP_SCALAR_COUNTS", "MANIFEST_DATA_BOUNDARY",
    "MANIFEST_FILENAME", "MEMBERS", "OUTER_ACCESS_RECEIPT_FILENAME",
    "P3StageB1OuterCellShardError", "RECORD_ARTIFACT_TYPE",
    "REGION_GRADIENT_BASIS_FILENAME", "SCALAR_PARAMETER_COUNT", "SCHEMA_VERSION",
    "VerifiedStageB1CellShard", "build_coarse_group_layout",
    "build_stage_b1_cell_payloads", "canonical_jsonl_bytes",
    "parse_canonical_jsonl", "validate_coarse_group_layout",
    "verify_stage_b1_cell_shard",
]
