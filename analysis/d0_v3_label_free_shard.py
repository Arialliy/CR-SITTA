"""CPU-only artifact contract for one D0-v3 Stage-A label-free shard.

One formal R0 shard contains the exact 64-image by 10-candidate grid.  A
one-image artifact is permitted only as an explicitly non-formal engineering
dry-run.  This module never imports a model, initializes CUDA, or opens an
outer target.  It validates the lossless numeric payload, the nested v2
independent-candidate evidence, the native first-step evidence, and the
authorization/data-boundary fields from bytes on disk.
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
import torch

from analysis.d0_v2_independent_candidate_contract import (
    FROZEN_CANDIDATES,
    parse_independent_candidate_receipt,
    validate_independent_candidate_cell,
)
from analysis.d0_v3_formal_contract import (
    CONDITIONS,
    DATASETS,
    FROZEN_CANDIDATES as FORMAL_CANDIDATES,
    PROTOCOL_ID as FORMAL_PROTOCOL_ID,
)
from analysis.d0_v3_outer_analyzer import FlatParameterLayout
from analysis.d0_v3_phase_receipt import (
    canonical_label_free_cell_receipt_bytes,
    validate_label_free_cell_receipt,
)
from tta.d0_secure_io import read_stable_regular_file, snapshot_regular_directory
from tta.d0_v2_native_step import NAMED_TENSOR_HASH_CONTRACT
from tta.d0_v2_parameter_groups import FROZEN_D0_V2_FINE_INVENTORY_SHA256


SCHEMA_VERSION: Final = 3
ARTIFACT_TYPE: Final = "cr_sitta_d0_v3_formal_stage_a_label_free_shard"
EPISODE_ARTIFACT_TYPE: Final = (
    "cr_sitta_d0_v3_formal_stage_a_label_free_episode"
)
COMPLETE_ARTIFACT_TYPE: Final = (
    "cr_sitta_d0_v3_formal_stage_a_label_free_complete"
)
EPISODE_ORDER: Final = "image_major_candidate_minor"
PARAMETER_TENSOR_COUNT: Final = 106
PARAMETER_SCALAR_COUNT: Final = 8736
FORMAL_IMAGE_COUNT: Final = 64
CANDIDATE_COUNT: Final = 10
LOGIT_HEIGHT: Final = 256
LOGIT_WIDTH: Final = 256
FLOAT_DTYPE: Final = "<f4"
SLICE_HASH_CONTRACT: Final = "cr-sitta-d0-v3-array-slice-sha256-v1"
INPUT_TENSOR_HASH_CONTRACT: Final = "cr-sitta-d0-v2-strided-tensor-v1"

LAYOUT_FILENAME: Final = "parameter_layout.json"
SOURCE_LOGITS_FILENAME: Final = "source_logits_pre.npy"
POST_LOGITS_FILENAME: Final = "tent_logits_post.npy"
SOURCE_PARAMETERS_FILENAME: Final = "source_parameters.npy"
PARAMETERS_AFTER_FILENAME: Final = "parameters_after.npy"
ENTROPY_GRADIENTS_FILENAME: Final = "entropy_gradients.npy"
EPISODES_FILENAME: Final = "episode_receipts.jsonl"
PHASE_RECEIPT_FILENAME: Final = "label_free_phase_receipt.json"
MANIFEST_FILENAME: Final = "manifest.json"
COMPLETE_FILENAME: Final = "COMPLETE.json"

ARRAY_FILENAMES: Final = (
    SOURCE_LOGITS_FILENAME,
    POST_LOGITS_FILENAME,
    SOURCE_PARAMETERS_FILENAME,
    PARAMETERS_AFTER_FILENAME,
    ENTROPY_GRADIENTS_FILENAME,
)

REQUIRED_CRITICAL_CODE_PATHS: Final = (
    "scripts/run_d0_v3_formal_stage_a_label_free.py",
    "analysis/d0_v3_label_free_shard.py",
    "analysis/d0_v3_formal_contract.py",
    "analysis/d0_v3_phase_receipt.py",
    "analysis/d0_v3_outer_analyzer.py",
    "analysis/d0_v2_independent_candidate_contract.py",
    "analysis/d0_v2_protocol_contract.py",
    "analysis/analyze_tent_optimizer_geometry.py",
    "analysis/d0_determinism_runtime.py",
    "analysis/source_train_provenance.py",
    "tta/d0_v3_atomic_shard.py",
    "tta/d0_v3_formal_capture.py",
    "tta/d0_v2_candidate_worker.py",
    "tta/d0_v2_native_step.py",
    "tta/d0_v2_parameter_groups.py",
    "tta/binary_tent.py",
    "tta/binary_tent_fast_runner.py",
    "tta/binary_tent_fast_runner_v2.py",
    "tta/episodic_runner.py",
    "tta/state_manager.py",
    "tta/model_adapter.py",
    "tta/parameter_groups.py",
    "tta/parameter_vector.py",
    "model/MSHNet_NSFPN.py",
    "model/NS_FPN.py",
    "model/diff_cross_attns.py",
    "materialize_binary_tent_ss_calibration_cache_v2.py",
    "test_source.py",
)
_INPUT_SEAL_BASE_ROLES: Final = frozenset(
    {
        "formal_config",
        "parent_engineering_config",
        "engineering_smoke_aggregate",
        "cache_execution_protocol",
        "cache_protocol",
        "source_train_split",
        "frozen_pilot_ids",
        "source_checkpoint",
        "cache_manifest",
        "cache_method_manifest",
        "cache_complete",
        "method_condition",
    }
)
_INPUT_SEAL_ZERO_FIELDS: Final = {
    "target_payload_bytes_opened": 0,
    "target_payload_deserialized": False,
    "validation_payload_opens": 0,
    "test_split_files_opened": 0,
    "test_images_opened": 0,
    "test_masks_opened": 0,
    "test_labels_opened": 0,
}

FORMAL_MEMBERS: Final = frozenset(
    {
        *ARRAY_FILENAMES,
        LAYOUT_FILENAME,
        EPISODES_FILENAME,
        PHASE_RECEIPT_FILENAME,
        MANIFEST_FILENAME,
        COMPLETE_FILENAME,
    }
)
DRY_RUN_MEMBERS: Final = FORMAL_MEMBERS - {PHASE_RECEIPT_FILENAME}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_EPISODE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "cell_binding",
        "process_identity",
        "tensor_slices",
        "independent_candidate_receipt",
        "native_first_step",
        "optimizer_geometry",
        "execution_gates",
        "data_boundary",
        "authorization",
    }
)
_CELL_KEYS = frozenset(
    {
        "config_sha256",
        "dataset",
        "condition",
        "replicate",
        "image_index",
        "image_id",
        "candidate_index",
        "candidate_slug",
        "split_name",
        "split_role",
    }
)
_PROCESS_KEYS = frozenset(
    {
        "logical_process_id",
        "os_process_id",
        "process_start_time_ticks",
        "parent_run_nonce",
        "child_launch_nonce",
        "command_sha256",
    }
)
_PAYLOAD_KEYS = frozenset(
    {
        "file",
        "file_sha256",
        "array_index",
        "slice_shape",
        "dtype",
        "slice_sha256",
        "named_bundle_sha256",
    }
)
_TENSOR_SLICE_KEYS = frozenset(
    {
        "source_logits_pre",
        "tent_logits_post",
        "source_parameters",
        "parameters_after",
        "entropy_gradients",
    }
)
_EXECUTION_GATES = {
    "fresh_model_method_optimizer": True,
    "own_forward_backward": True,
    "gradient_reuse": False,
    "only_selected_trainable": True,
    "optimizer_state_empty_before_step": True,
    "native_same_device_endpoint_exact": True,
    "native_optimizer_state_exact": True,
    "source_tent_pre_bit_exact": True,
    "full_audit_forced": True,
    "exact_source_reset": True,
    "formal_native_hash_parity": True,
    "live_gradient_storage_owned": True,
    "finite": True,
}
_DATA_BOUNDARY = {
    "source_train_derived": True,
    "split_name": "train",
    "split_role": "frozen_pilot64",
    "no_validation_split": True,
    "method_label_accesses": 0,
    "outer_evaluator_label_accesses": 0,
    "train_target_payload_bytes_opened": 0,
    "train_target_payload_deserialization_count": 0,
    "validation_payload_opens": 0,
    "test_split_files_opened": 0,
    "test_images_opened": 0,
    "test_masks_opened": 0,
    "test_labels_opened": 0,
}
_AUTHORIZATION = {
    "paper_result": False,
    "paper_test_result": False,
    "scientific_gate_status": "not_evaluated",
    "scientific_selection_performed": False,
    "formal_protocol_complete": False,
    "stage2_authorized": False,
}


class D0V3LabelFreeShardError(ValueError):
    """A label-free shard is incomplete, non-canonical, or unsafe."""


@dataclass(frozen=True, slots=True)
class VerifiedLabelFreeShard:
    path: Path
    dataset: str
    condition: str
    replicate: str
    formal: bool
    dry_run: bool
    image_count: int
    candidate_count: int
    episode_count: int
    manifest_sha256: str
    complete_sha256: str
    phase_receipt_sha256: str | None


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise D0V3LabelFreeShardError(
            "formal shard value is not canonical-JSON safe"
        ) from exc
    return payload + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0V3LabelFreeShardError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise D0V3LabelFreeShardError(
            f"{label} fields must be exact; missing={missing}, unknown={unknown}"
        )


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise D0V3LabelFreeShardError(
            f"{label} must be lowercase 64-hex SHA-256"
        )
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise D0V3LabelFreeShardError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D0V3LabelFreeShardError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise D0V3LabelFreeShardError(f"{label} must be finite")
    return result


def parse_canonical_json(data: bytes, *, label: str, newline: bool) -> Mapping[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_json_object(pairs, label),
            parse_constant=lambda token: (_ for _ in ()).throw(
                D0V3LabelFreeShardError(f"{label} contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D0V3LabelFreeShardError(f"{label} is not strict UTF-8 JSON") from exc
    mapping = _require_mapping(value, label=label)
    if canonical_json_bytes(mapping, newline=newline) != data:
        raise D0V3LabelFreeShardError(f"{label} bytes are not canonical")
    return mapping


def _unique_json_object(
    pairs: Sequence[tuple[str, Any]], label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise D0V3LabelFreeShardError(
                f"{label} contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def array_slice_sha256(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.str != FLOAT_DTYPE or not array.flags.c_contiguous:
        raise D0V3LabelFreeShardError(
            "array slice must be C-contiguous little-endian float32"
        )
    if not np.isfinite(array).all():
        raise D0V3LabelFreeShardError("array slice contains NaN/Inf")
    digest = hashlib.sha256()
    digest.update(SLICE_HASH_CONTRACT.encode("ascii") + b"\0")
    for component in (
        FLOAT_DTYPE.encode("ascii"),
        ",".join(str(value) for value in array.shape).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return digest.hexdigest()


def raw_float32_sha256(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.str != FLOAT_DTYPE or not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=np.dtype(FLOAT_DTYPE))
    if not np.isfinite(array).all():
        raise D0V3LabelFreeShardError("raw float32 hash received NaN/Inf")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def input_tensor_sha256(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.str != FLOAT_DTYPE or not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=np.dtype(FLOAT_DTYPE))
    if not np.isfinite(array).all():
        raise D0V3LabelFreeShardError("input tensor hash received NaN/Inf")
    digest = hashlib.sha256()
    digest.update(INPUT_TENSOR_HASH_CONTRACT.encode("ascii") + b"\0")
    for component in (
        b"torch.float32",
        ",".join(str(value) for value in array.shape).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return digest.hexdigest()


def named_bundle_sha256(flat: Any, layout: FlatParameterLayout) -> str:
    """Rebuild the exact v2 named-tensor bundle hash from one flat vector."""

    if not isinstance(layout, FlatParameterLayout):
        raise D0V3LabelFreeShardError("layout must be FlatParameterLayout")
    vector = np.asarray(flat)
    if (
        vector.dtype.str != FLOAT_DTYPE
        or vector.shape != (layout.scalar_count,)
        or not vector.flags.c_contiguous
        or not np.isfinite(vector).all()
    ):
        raise D0V3LabelFreeShardError(
            "named bundle vector must be finite flat little-endian float32"
        )
    digest = hashlib.sha256()
    digest.update(NAMED_TENSOR_HASH_CONTRACT.encode("ascii") + b"\0")
    for index, (name, shape, offset) in enumerate(
        zip(layout.names, layout.shapes, layout.offsets, strict=True)
    ):
        end = (
            layout.offsets[index + 1]
            if index + 1 < len(layout.offsets)
            else layout.scalar_count
        )
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        tensor = vector[offset:end].reshape(shape)
        for component in (
            b"torch.float32",
            ",".join(str(value) for value in shape).encode("ascii"),
            tensor.tobytes(order="C"),
        ):
            digest.update(len(component).to_bytes(8, "big"))
            digest.update(component)
    return digest.hexdigest()


def validate_parameter_layout(value: Mapping[str, Any]) -> FlatParameterLayout:
    try:
        layout = FlatParameterLayout.from_mapping(value)
    except (TypeError, ValueError) as exc:
        raise D0V3LabelFreeShardError("parameter layout is invalid") from exc
    if (
        len(layout.names) != PARAMETER_TENSOR_COUNT
        or layout.scalar_count != PARAMETER_SCALAR_COUNT
    ):
        raise D0V3LabelFreeShardError(
            "formal all-BN layout must contain exactly 106 tensors/8736 scalars"
        )
    return layout


def expected_array_specs(image_count: int) -> dict[str, tuple[int, ...]]:
    count = _integer(image_count, label="image_count", minimum=1)
    return {
        SOURCE_LOGITS_FILENAME: (count, 1, LOGIT_HEIGHT, LOGIT_WIDTH),
        POST_LOGITS_FILENAME: (
            count,
            CANDIDATE_COUNT,
            1,
            LOGIT_HEIGHT,
            LOGIT_WIDTH,
        ),
        SOURCE_PARAMETERS_FILENAME: (
            count,
            CANDIDATE_COUNT,
            PARAMETER_SCALAR_COUNT,
        ),
        PARAMETERS_AFTER_FILENAME: (
            count,
            CANDIDATE_COUNT,
            PARAMETER_SCALAR_COUNT,
        ),
        ENTROPY_GRADIENTS_FILENAME: (
            count,
            CANDIDATE_COUNT,
            PARAMETER_SCALAR_COUNT,
        ),
    }


def _npy_from_stable_bytes(data: bytes, *, label: str) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise D0V3LabelFreeShardError(f"{label} is not a safe NPY payload") from exc
    if not isinstance(value, np.ndarray):
        raise D0V3LabelFreeShardError(f"{label} did not decode to ndarray")
    return value


def validate_method_facing_seal_path(role: Any, relative: Any) -> str:
    """Accept code/train-Pilot inputs and reject target/test payload paths.

    A Python producer named ``test_source.py`` is executable code, not a test
    split.  Rejection is therefore based on explicit payload roles, path
    components and split filenames rather than a broad ``test*`` prefix.
    """

    if not isinstance(role, str) or not role:
        raise D0V3LabelFreeShardError("input seal role must be non-empty")
    if not isinstance(relative, str) or not relative:
        raise D0V3LabelFreeShardError("input seal path must be non-empty")
    relative_path = Path(relative)
    lower_parts = tuple(part.lower() for part in relative_path.parts)
    forbidden_test_parts = {
        "test",
        "test_images",
        "test_masks",
        "test_labels",
        "test_predictions",
    }
    forbidden_test_file = (
        bool(lower_parts)
        and lower_parts[-1]
        in {"test.txt", "test.csv", "test.json", "test.jsonl"}
    )
    forbidden_role = role.lower().startswith(
        (
            "test_split",
            "test_image",
            "test_mask",
            "test_label",
            "test_prediction",
        )
    )
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative.endswith("outer_evaluator/targets.npy")
        or any(part in forbidden_test_parts for part in lower_parts)
        or forbidden_test_file
        or forbidden_role
    ):
        raise D0V3LabelFreeShardError("input seal contains forbidden payload path")
    return relative_path.as_posix()


def _validate_input_seal(
    value: Any,
    *,
    config_sha256: str,
    dataset_binding: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    seal = _require_mapping(value, label="manifest.input_seal")
    expected_fields = {"files", "seal_sha256", *_INPUT_SEAL_ZERO_FIELDS}
    if set(seal) != expected_fields:
        raise D0V3LabelFreeShardError("input seal fields must be exact")
    for field, expected in _INPUT_SEAL_ZERO_FIELDS.items():
        if seal[field] is not expected and seal[field] != expected:
            raise D0V3LabelFreeShardError(f"input seal boundary differs: {field}")
    raw_files = seal["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise D0V3LabelFreeShardError("input seal files are missing")
    records: list[dict[str, str]] = []
    by_role: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(raw_files):
        record = _require_mapping(raw, label=f"input_seal.files[{index}]")
        if set(record) != {"role", "path", "sha256"}:
            raise D0V3LabelFreeShardError("input seal file record fields differ")
        role = record["role"]
        path = validate_method_facing_seal_path(role, record["path"])
        digest = _sha256(record["sha256"], label=f"input seal {role} SHA")
        if role in by_role:
            raise D0V3LabelFreeShardError(f"duplicate input seal role: {role}")
        normalized = {"role": role, "path": path, "sha256": digest}
        records.append(normalized)
        by_role[role] = normalized
    if records != [by_role[key] for key in sorted(by_role)]:
        raise D0V3LabelFreeShardError("input seal records must be role-sorted")
    required_roles = _INPUT_SEAL_BASE_ROLES | {
        f"critical_code:{path}" for path in REQUIRED_CRITICAL_CODE_PATHS
    }
    if set(by_role) != required_roles:
        raise D0V3LabelFreeShardError(
            "input seal required role set differs; "
            f"missing={sorted(required_roles - set(by_role))}, "
            f"unknown={sorted(set(by_role) - required_roles)}"
        )
    if seal["seal_sha256"] != canonical_sha256(records):
        raise D0V3LabelFreeShardError("input seal aggregate SHA differs")
    if by_role["formal_config"]["sha256"] != config_sha256:
        raise D0V3LabelFreeShardError("input seal formal config SHA differs")
    if set(dataset_binding) != {
        "train_split_sha256",
        "checkpoint_role",
        "checkpoint_path",
        "checkpoint_sha256",
    }:
        raise D0V3LabelFreeShardError("manifest dataset binding fields differ")
    split_sha = _sha256(
        dataset_binding["train_split_sha256"], label="dataset train split SHA"
    )
    checkpoint_sha = _sha256(
        dataset_binding["checkpoint_sha256"], label="dataset checkpoint SHA"
    )
    checkpoint_path = validate_method_facing_seal_path(
        "source_checkpoint", dataset_binding["checkpoint_path"]
    )
    if dataset_binding["checkpoint_role"] != "best_miou":
        raise D0V3LabelFreeShardError("dataset checkpoint role must be best_miou")
    if (
        by_role["source_train_split"]["sha256"] != split_sha
        or by_role["source_checkpoint"]["sha256"] != checkpoint_sha
        or by_role["source_checkpoint"]["path"] != checkpoint_path
    ):
        raise D0V3LabelFreeShardError(
            "dataset split/checkpoint binding differs from input seal"
        )
    code_seals = {
        path: by_role[f"critical_code:{path}"]["sha256"]
        for path in REQUIRED_CRITICAL_CODE_PATHS
    }
    return records, code_seals


def _verify_native(
    native: Mapping[str, Any], *, layout: FlatParameterLayout, candidate_index: int
) -> None:
    expected_root = {
        "schema_version",
        "protocol_id",
        "reference_name",
        "pytorch_version_required",
        "pytorch_version_observed",
        "optimizer_name",
        "learning_rate",
        "parameter_names",
        "device",
        "runtime_optimizer",
        "hash_contracts",
        "hashes",
        "counts",
        "step_norm_l2",
        "gates",
        "authorization",
    }
    if set(native) != expected_root:
        raise D0V3LabelFreeShardError("native first-step root schema differs")
    candidate = FROZEN_CANDIDATES[candidate_index]
    if (
        native["optimizer_name"] != candidate.optimizer
        or native["learning_rate"] != candidate.learning_rate
        or tuple(native["parameter_names"]) != layout.names
        or native["hash_contracts"].get("named_tensors")
        != NAMED_TENSOR_HASH_CONTRACT
    ):
        raise D0V3LabelFreeShardError("native candidate/layout binding differs")
    counts = _require_mapping(native["counts"], label="native.counts")
    required_counts = {
        "optimizer_step_call_count": 1,
        "parameter_tensor_count": PARAMETER_TENSOR_COUNT,
        "gradient_tensor_count": PARAMETER_TENSOR_COUNT,
        "scalar_parameter_count": PARAMETER_SCALAR_COUNT,
        "native_reference_parameter_tensor_count": PARAMETER_TENSOR_COUNT,
        "optimizer_state_parameter_count": PARAMETER_TENSOR_COUNT,
        "bit_exact_parameter_tensor_count": PARAMETER_TENSOR_COUNT,
    }
    if any(counts.get(key) != expected for key, expected in required_counts.items()):
        raise D0V3LabelFreeShardError("native first-step counts differ")
    gates = _require_mapping(native["gates"], label="native.gates")
    if not gates or any(value is not True for value in gates.values()):
        raise D0V3LabelFreeShardError("native first-step gate failed")
    if native["authorization"] != {
        "engineering_observation_only": True,
        "scientific_gate_status": "unresolved",
        "scientific_selection_performed": False,
        "stage2_authorized": False,
    }:
        raise D0V3LabelFreeShardError("native authorization drifted")
    _finite_float(native["step_norm_l2"], label="native.step_norm_l2")


def _verify_payload_record(
    value: Any,
    *,
    key: str,
    expected_file: str,
    expected_file_sha256: str,
    expected_index: list[int],
    expected_shape: tuple[int, ...],
    array: np.ndarray,
    layout: FlatParameterLayout,
) -> np.ndarray:
    record = _require_mapping(value, label=f"tensor_slices.{key}")
    _exact_keys(record, _PAYLOAD_KEYS, f"tensor_slices.{key}")
    if (
        record["file"] != expected_file
        or record["file_sha256"] != expected_file_sha256
        or record["array_index"] != expected_index
        or record["slice_shape"] != list(expected_shape)
        or record["dtype"] != FLOAT_DTYPE
    ):
        raise D0V3LabelFreeShardError(f"tensor slice binding differs: {key}")
    selector = tuple(expected_index)
    selected = np.ascontiguousarray(array[selector], dtype=np.dtype(FLOAT_DTYPE))
    if selected.shape != expected_shape:
        raise D0V3LabelFreeShardError(f"tensor slice shape differs: {key}")
    observed_slice_sha = array_slice_sha256(selected)
    if record["slice_sha256"] != observed_slice_sha:
        raise D0V3LabelFreeShardError(f"tensor slice SHA differs: {key}")
    named = record["named_bundle_sha256"]
    if key in {"source_logits_pre", "tent_logits_post"}:
        if named is not None:
            raise D0V3LabelFreeShardError(f"logit slice must not claim named hash: {key}")
    else:
        expected_named = named_bundle_sha256(selected, layout)
        if named != expected_named:
            raise D0V3LabelFreeShardError(f"named tensor hash differs: {key}")
    return selected


def _verify_geometry(
    value: Any, *, dataset: str, split_sha256: str, checkpoint_sha256: str,
    candidate_index: int
) -> None:
    geometry = _require_mapping(value, label="optimizer_geometry")
    candidate = FROZEN_CANDIDATES[candidate_index]
    if (
        geometry.get("schema_version") != 3
        or geometry.get("analysis_type") != "tent_optimizer_first_step_geometry"
        or geometry.get("optimizer", {}).get("name") != candidate.optimizer
        or geometry.get("optimizer", {}).get("learning_rate")
        != candidate.learning_rate
    ):
        raise D0V3LabelFreeShardError("optimizer geometry binding differs")
    scope = geometry.get("scope")
    if not isinstance(scope, Mapping) or any(
        (
            scope.get("dataset") != dataset,
            scope.get("split_name") != "train",
            scope.get("split_sha256") != split_sha256,
            scope.get("checkpoint_sha256") != checkpoint_sha256,
            scope.get("oracle_analysis") is not False,
            scope.get("method_label_accesses") != 0,
            scope.get("outer_evaluator_label_accesses") != 0,
            scope.get("adaptation_gradient_uses_labels") is not False,
            scope.get("use_test_images") is not False,
            scope.get("use_test_labels") is not False,
        )
    ):
        raise D0V3LabelFreeShardError("optimizer geometry is not label-free train-side")
    policy = geometry.get("gate_policy")
    if not isinstance(policy, Mapping) or policy.get(
        "runtime_same_device_hard_gate_is_sole_acceptance_gate"
    ) is not True:
        raise D0V3LabelFreeShardError("optimizer geometry hard-gate policy differs")


def _parse_episode_lines(data: bytes) -> list[Mapping[str, Any]]:
    if not data or not data.endswith(b"\n"):
        raise D0V3LabelFreeShardError("episode JSONL must end with exactly one newline")
    lines = data.splitlines(keepends=True)
    result: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.endswith(b"\n") or line == b"\n":
            raise D0V3LabelFreeShardError("episode JSONL contains an empty/partial line")
        result.append(
            parse_canonical_json(
                line, label=f"episode_receipts[{index}]", newline=True
            )
        )
    return result


def _verify_episode(
    value: Mapping[str, Any],
    *,
    image_index: int,
    candidate_index: int,
    image_id: str,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    array_sha256s: Mapping[str, str],
    layout: FlatParameterLayout,
) -> tuple[Mapping[str, Any], str]:
    _exact_keys(value, _EPISODE_KEYS, "episode")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["artifact_type"] != EPISODE_ARTIFACT_TYPE
        or value["protocol_id"] != FORMAL_PROTOCOL_ID
    ):
        raise D0V3LabelFreeShardError("episode protocol header differs")
    cell = _require_mapping(value["cell_binding"], label="episode.cell_binding")
    _exact_keys(cell, _CELL_KEYS, "episode.cell_binding")
    candidate = FROZEN_CANDIDATES[candidate_index]
    expected_cell = {
        "config_sha256": manifest["config_sha256"],
        "dataset": manifest["cell"]["dataset"],
        "condition": manifest["cell"]["condition"],
        "replicate": manifest["cell"]["replicate"],
        "image_index": image_index,
        "image_id": image_id,
        "candidate_index": candidate_index,
        "candidate_slug": candidate.slug,
        "split_name": "train",
        "split_role": "frozen_pilot64",
    }
    if dict(cell) != expected_cell:
        raise D0V3LabelFreeShardError("episode cell binding differs")
    process = _require_mapping(value["process_identity"], label="process_identity")
    _exact_keys(process, _PROCESS_KEYS, "process_identity")
    if dict(process) != manifest["process_identity"]:
        raise D0V3LabelFreeShardError("episode process identity differs")
    _integer(process["os_process_id"], label="os_process_id", minimum=1)
    _integer(
        process["process_start_time_ticks"],
        label="process_start_time_ticks",
        minimum=1,
    )
    for field in ("parent_run_nonce", "child_launch_nonce", "command_sha256"):
        _sha256(process[field], label=field)

    slices = _require_mapping(value["tensor_slices"], label="tensor_slices")
    _exact_keys(slices, _TENSOR_SLICE_KEYS, "tensor_slices")
    source_logits = _verify_payload_record(
        slices["source_logits_pre"],
        key="source_logits_pre",
        expected_file=SOURCE_LOGITS_FILENAME,
        expected_file_sha256=array_sha256s[SOURCE_LOGITS_FILENAME],
        expected_index=[image_index],
        expected_shape=(1, LOGIT_HEIGHT, LOGIT_WIDTH),
        array=arrays[SOURCE_LOGITS_FILENAME],
        layout=layout,
    )
    post_logits = _verify_payload_record(
        slices["tent_logits_post"],
        key="tent_logits_post",
        expected_file=POST_LOGITS_FILENAME,
        expected_file_sha256=array_sha256s[POST_LOGITS_FILENAME],
        expected_index=[image_index, candidate_index],
        expected_shape=(1, LOGIT_HEIGHT, LOGIT_WIDTH),
        array=arrays[POST_LOGITS_FILENAME],
        layout=layout,
    )
    source_parameters = _verify_payload_record(
        slices["source_parameters"],
        key="source_parameters",
        expected_file=SOURCE_PARAMETERS_FILENAME,
        expected_file_sha256=array_sha256s[SOURCE_PARAMETERS_FILENAME],
        expected_index=[image_index, candidate_index],
        expected_shape=(PARAMETER_SCALAR_COUNT,),
        array=arrays[SOURCE_PARAMETERS_FILENAME],
        layout=layout,
    )
    parameters_after = _verify_payload_record(
        slices["parameters_after"],
        key="parameters_after",
        expected_file=PARAMETERS_AFTER_FILENAME,
        expected_file_sha256=array_sha256s[PARAMETERS_AFTER_FILENAME],
        expected_index=[image_index, candidate_index],
        expected_shape=(PARAMETER_SCALAR_COUNT,),
        array=arrays[PARAMETERS_AFTER_FILENAME],
        layout=layout,
    )
    gradients = _verify_payload_record(
        slices["entropy_gradients"],
        key="entropy_gradients",
        expected_file=ENTROPY_GRADIENTS_FILENAME,
        expected_file_sha256=array_sha256s[ENTROPY_GRADIENTS_FILENAME],
        expected_index=[image_index, candidate_index],
        expected_shape=(PARAMETER_SCALAR_COUNT,),
        array=arrays[ENTROPY_GRADIENTS_FILENAME],
        layout=layout,
    )

    nested = parse_independent_candidate_receipt(
        _require_mapping(
            value["independent_candidate_receipt"],
            label="independent_candidate_receipt",
        )
    )
    if (
        nested.candidate_index != candidate_index
        or nested.dataset != cell["dataset"]
        or nested.condition != cell["condition"]
        or nested.sample_index != image_index
        or nested.sample_id != image_id
        or nested.config_sha256 != manifest["config_sha256"]
        or nested.split_sha256 != manifest["dataset_binding"]["train_split_sha256"]
        or nested.checkpoint_sha256
        != manifest["dataset_binding"]["checkpoint_sha256"]
        or nested.pre_logits_sha256 != raw_float32_sha256(source_logits)
        or nested.post_logits_sha256 != raw_float32_sha256(post_logits)
        or nested.entropy_gradient_bundle_sha256
        != named_bundle_sha256(gradients, layout)
    ):
        raise D0V3LabelFreeShardError("nested independent receipt differs from payload")

    native = _require_mapping(value["native_first_step"], label="native_first_step")
    _verify_native(native, layout=layout, candidate_index=candidate_index)
    hashes = native["hashes"]
    derived_step = np.ascontiguousarray(
        parameters_after - source_parameters, dtype=np.dtype(FLOAT_DTYPE)
    )
    expected_native_hashes = {
        "parameter_before_bundle_sha256": named_bundle_sha256(
            source_parameters, layout
        ),
        "gradient_bundle_sha256": named_bundle_sha256(gradients, layout),
        "actual_parameter_after_bundle_sha256": named_bundle_sha256(
            parameters_after, layout
        ),
        "parameter_delta_bundle_sha256": named_bundle_sha256(
            derived_step, layout
        ),
    }
    if any(hashes.get(key) != expected for key, expected in expected_native_hashes.items()):
        raise D0V3LabelFreeShardError("native hash does not match lossless payload")
    if (
        hashes.get("actual_parameter_after_bundle_sha256")
        != hashes.get("reference_parameter_after_bundle_sha256")
        or hashes.get("actual_optimizer_state_bundle_sha256")
        != hashes.get("reference_optimizer_state_bundle_sha256")
        or nested.parameter_delta_bundle_sha256
        != expected_native_hashes["parameter_delta_bundle_sha256"]
        or not math.isclose(
            nested.step_norm_l2,
            float(native["step_norm_l2"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise D0V3LabelFreeShardError("native/nested first-step evidence differs")

    _verify_geometry(
        value["optimizer_geometry"],
        dataset=str(cell["dataset"]),
        split_sha256=manifest["dataset_binding"]["train_split_sha256"],
        checkpoint_sha256=manifest["dataset_binding"]["checkpoint_sha256"],
        candidate_index=candidate_index,
    )
    if value["execution_gates"] != _EXECUTION_GATES:
        raise D0V3LabelFreeShardError("episode execution gates differ")
    if value["data_boundary"] != _DATA_BOUNDARY:
        raise D0V3LabelFreeShardError("episode data boundary differs")
    if value["authorization"] != _AUTHORIZATION:
        raise D0V3LabelFreeShardError("episode authorization differs")
    return value["independent_candidate_receipt"], hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()


def _manifest_schema(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "config_sha256",
        "cell",
        "mode",
        "process_identity",
        "dataset_binding",
        "ordered_image_ids",
        "ordered_image_ids_sha256",
        "candidate_slugs",
        "parameter_layout",
        "fine_inventory_sha256",
        "input_seal",
        "arrays",
        "episode_receipts",
        "phase_receipt",
        "data_boundary",
        "authorization",
    }
    if set(value) != expected:
        raise D0V3LabelFreeShardError("manifest fields differ")


def verify_label_free_shard(
    path: str | os.PathLike[str],
    *,
    expected_config_sha256: str | None = None,
    verify_live_inputs: bool = False,
    repository_root: str | os.PathLike[str] | None = None,
) -> VerifiedLabelFreeShard:
    """Fully verify one immutable flat shard without CUDA or target access."""

    root = Path(os.path.abspath(os.fspath(path)))
    try:
        snapshot = snapshot_regular_directory(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise D0V3LabelFreeShardError(
            f"cannot securely snapshot label-free shard: {root}"
        ) from exc
    by_name = {value.path.name: value for value in snapshot.members}
    if MANIFEST_FILENAME not in by_name:
        raise D0V3LabelFreeShardError("manifest.json is missing")
    manifest = parse_canonical_json(
        by_name[MANIFEST_FILENAME].data, label="manifest.json", newline=True
    )
    _manifest_schema(manifest)
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["artifact_type"] != ARTIFACT_TYPE
        or manifest["protocol_id"] != FORMAL_PROTOCOL_ID
    ):
        raise D0V3LabelFreeShardError("manifest protocol header differs")
    config_sha256 = _sha256(manifest["config_sha256"], label="config_sha256")
    if expected_config_sha256 is not None and config_sha256 != _sha256(
        expected_config_sha256, label="expected_config_sha256"
    ):
        raise D0V3LabelFreeShardError("manifest config SHA differs")
    cell = _require_mapping(manifest["cell"], label="manifest.cell")
    if set(cell) != {"dataset", "condition", "replicate"}:
        raise D0V3LabelFreeShardError("manifest cell fields differ")
    dataset = cell["dataset"]
    condition = cell["condition"]
    replicate = cell["replicate"]
    if dataset not in DATASETS or condition not in CONDITIONS or replicate not in {
        "R0",
        "R1",
        "R2",
    }:
        raise D0V3LabelFreeShardError("manifest cell is outside frozen grid")
    mode = _require_mapping(manifest["mode"], label="manifest.mode")
    if set(mode) != {
        "formal",
        "dry_run",
        "image_count",
        "candidate_count",
        "episode_count",
        "episode_order",
    }:
        raise D0V3LabelFreeShardError("manifest mode fields differ")
    formal = mode["formal"] is True
    dry_run = mode["dry_run"] is True
    image_count = _integer(mode["image_count"], label="image_count", minimum=1)
    if formal == dry_run or (
        formal and (replicate != "R0" or image_count != FORMAL_IMAGE_COUNT)
    ) or (dry_run and image_count != 1):
        raise D0V3LabelFreeShardError("formal/dry-run mode contract differs")
    if (
        mode["candidate_count"] != CANDIDATE_COUNT
        or mode["episode_count"] != image_count * CANDIDATE_COUNT
        or mode["episode_order"] != EPISODE_ORDER
    ):
        raise D0V3LabelFreeShardError("manifest grid size/order differs")
    expected_members = FORMAL_MEMBERS if formal else DRY_RUN_MEMBERS
    if set(by_name) != expected_members:
        raise D0V3LabelFreeShardError("flat shard member set differs from mode")
    if manifest["authorization"] != _AUTHORIZATION:
        raise D0V3LabelFreeShardError("manifest authorization differs")
    if manifest["data_boundary"] != _DATA_BOUNDARY:
        raise D0V3LabelFreeShardError("manifest data boundary differs")
    if manifest["candidate_slugs"] != [value.slug for value in FROZEN_CANDIDATES]:
        raise D0V3LabelFreeShardError("manifest candidate order differs")
    if any(
        (left.optimizer, left.learning_rate, left.slug)
        != (right.optimizer, right.learning_rate, right.candidate_id)
        for left, right in zip(FROZEN_CANDIDATES, FORMAL_CANDIDATES, strict=True)
    ):
        raise D0V3LabelFreeShardError("v2/v3 frozen candidate grids disagree")
    dataset_binding = _require_mapping(
        manifest["dataset_binding"], label="manifest.dataset_binding"
    )
    input_records, code_seals = _validate_input_seal(
        manifest["input_seal"],
        config_sha256=config_sha256,
        dataset_binding=dataset_binding,
    )
    input_by_role = {record["role"]: record for record in input_records}
    if manifest["fine_inventory_sha256"] != FROZEN_D0_V2_FINE_INVENTORY_SHA256:
        raise D0V3LabelFreeShardError("fine-group inventory SHA differs")

    layout_mapping = parse_canonical_json(
        by_name[LAYOUT_FILENAME].data,
        label=LAYOUT_FILENAME,
        newline=True,
    )
    layout = validate_parameter_layout(layout_mapping)
    if manifest["parameter_layout"] != {
        "path": LAYOUT_FILENAME,
        "sha256": by_name[LAYOUT_FILENAME].sha256,
        "layout_sha256": layout.layout_sha256,
        "parameter_tensor_count": PARAMETER_TENSOR_COUNT,
        "scalar_parameter_count": PARAMETER_SCALAR_COUNT,
    }:
        raise D0V3LabelFreeShardError("manifest parameter layout binding differs")

    expected_specs = expected_array_specs(image_count)
    arrays_section = _require_mapping(manifest["arrays"], label="manifest.arrays")
    if set(arrays_section) != set(ARRAY_FILENAMES):
        raise D0V3LabelFreeShardError("manifest arrays member set differs")
    arrays: dict[str, np.ndarray] = {}
    array_hashes: dict[str, str] = {}
    for filename, shape in expected_specs.items():
        member = by_name[filename]
        record = arrays_section[filename]
        if record != {
            "path": filename,
            "sha256": member.sha256,
            "shape": list(shape),
            "dtype": FLOAT_DTYPE,
            "c_order": True,
            "finite": True,
            "lossless": True,
        }:
            raise D0V3LabelFreeShardError(f"manifest array record differs: {filename}")
        array = _npy_from_stable_bytes(member.data, label=filename)
        if (
            array.shape != shape
            or array.dtype.str != FLOAT_DTYPE
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
        ):
            raise D0V3LabelFreeShardError(f"array schema/content differs: {filename}")
        arrays[filename] = array
        array_hashes[filename] = member.sha256

    image_ids = manifest["ordered_image_ids"]
    if (
        not isinstance(image_ids, list)
        or len(image_ids) != image_count
        or len(set(image_ids)) != image_count
        or not all(isinstance(value, str) and value for value in image_ids)
    ):
        raise D0V3LabelFreeShardError("ordered image IDs differ")
    expected_ids_sha = hashlib.sha256(
        json.dumps(image_ids, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if manifest["ordered_image_ids_sha256"] != expected_ids_sha:
        raise D0V3LabelFreeShardError("ordered image ID SHA differs")

    episodes = _parse_episode_lines(by_name[EPISODES_FILENAME].data)
    expected_episode_count = image_count * CANDIDATE_COUNT
    if len(episodes) != expected_episode_count:
        raise D0V3LabelFreeShardError("episode receipt count differs")
    independent_by_image: list[list[Mapping[str, Any]]] = [
        [] for _ in range(image_count)
    ]
    episode_hashes: list[str] = []
    identity_values: dict[str, set[str]] = {
        key: set()
        for key in (
            "model_instance_id",
            "method_instance_id",
            "optimizer_instance_id",
            "autograd_graph_id",
            "backward_execution_id",
            "gradient_buffer_owner_id",
        )
    }
    for position, episode in enumerate(episodes):
        image_index, candidate_index = divmod(position, CANDIDATE_COUNT)
        nested, receipt_sha = _verify_episode(
            episode,
            image_index=image_index,
            candidate_index=candidate_index,
            image_id=image_ids[image_index],
            manifest=manifest,
            arrays=arrays,
            array_sha256s=array_hashes,
            layout=layout,
        )
        independent_by_image[image_index].append(nested)
        episode_hashes.append(receipt_sha)
        identity = nested["execution_identity"]
        for key in identity_values:
            token = identity[key]
            if token in identity_values[key]:
                raise D0V3LabelFreeShardError(
                    f"execution identity reused across cell: {key}"
                )
            identity_values[key].add(token)
    if len(set(episode_hashes)) != expected_episode_count:
        raise D0V3LabelFreeShardError("episode receipt hashes must be unique")
    for receipts in independent_by_image:
        validate_independent_candidate_cell(receipts)
    if manifest["episode_receipts"] != {
        "path": EPISODES_FILENAME,
        "sha256": by_name[EPISODES_FILENAME].sha256,
        "count": expected_episode_count,
        "order": EPISODE_ORDER,
        "episode_receipt_hashes_sha256": canonical_sha256(episode_hashes),
    }:
        raise D0V3LabelFreeShardError("manifest episode receipt binding differs")

    phase_sha: str | None
    if formal:
        phase_member = by_name[PHASE_RECEIPT_FILENAME]
        phase = validate_label_free_cell_receipt(
            phase_member.data,
            expected_dataset=dataset,
            expected_condition=condition,
            expected_replicate=0,
            expected_config_sha256=config_sha256,
            expected_checkpoint_sha256=manifest["dataset_binding"][
                "checkpoint_sha256"
            ],
            expected_ordered_image_ids=image_ids,
            expected_cache_method_manifest_sha256=input_by_role[
                "cache_method_manifest"
            ]["sha256"],
            expected_code_seals=code_seals,
        )
        if tuple(episode_hashes) != phase.episode_receipt_sha256s:
            raise D0V3LabelFreeShardError("phase receipt episode hashes differ")
        if canonical_label_free_cell_receipt_bytes(phase_member.data) != phase_member.data:
            raise D0V3LabelFreeShardError("phase receipt bytes are non-canonical")
        phase_sha = phase_member.sha256
        expected_phase = {
            "path": PHASE_RECEIPT_FILENAME,
            "sha256": phase_sha,
            "label_free_phase_complete": True,
            "outer_target_access_authorized_for_this_cell": True,
            "formal_protocol_complete": False,
            "stage2_authorized": False,
        }
    else:
        phase_sha = None
        expected_phase = None
    if manifest["phase_receipt"] != expected_phase:
        raise D0V3LabelFreeShardError("manifest phase receipt binding differs")

    if verify_live_inputs:
        if repository_root is None:
            raise D0V3LabelFreeShardError(
                "repository_root is required for live input verification"
            )
        repo = Path(os.path.abspath(os.fspath(repository_root)))
        observed: list[dict[str, str]] = []
        for record in input_records:
            relative = record["path"]
            canonical_relative = validate_method_facing_seal_path(
                record.get("role"), relative
            )
            stable = read_stable_regular_file(repo / canonical_relative)
            if stable.sha256 != record["sha256"]:
                raise D0V3LabelFreeShardError("live input seal SHA differs")
            observed.append(dict(record))
        if manifest["input_seal"].get("seal_sha256") != canonical_sha256(observed):
            raise D0V3LabelFreeShardError("input seal aggregate SHA differs")

    complete = parse_canonical_json(
        by_name[COMPLETE_FILENAME].data, label=COMPLETE_FILENAME, newline=True
    )
    expected_payload_files = [
        {"path": name, "sha256": by_name[name].sha256}
        for name in sorted(set(by_name) - {COMPLETE_FILENAME})
    ]
    expected_complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "complete": True,
        "candidate_phase_complete": True,
        "formal": formal,
        "dry_run": dry_run,
        "config_sha256": config_sha256,
        "dataset": dataset,
        "condition": condition,
        "replicate": replicate,
        "image_count": image_count,
        "candidate_count": CANDIDATE_COUNT,
        "episode_count": expected_episode_count,
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": by_name[MANIFEST_FILENAME].sha256,
        },
        "payload_files": expected_payload_files,
        "atomic_no_replace": True,
        "paper_result": False,
        "formal_protocol_complete": False,
        "scientific_gate_status": "not_evaluated",
        "stage2_authorized": False,
    }
    if complete != expected_complete:
        raise D0V3LabelFreeShardError("COMPLETE receipt does not rebuild from payload")
    return VerifiedLabelFreeShard(
        path=root,
        dataset=dataset,
        condition=condition,
        replicate=replicate,
        formal=formal,
        dry_run=dry_run,
        image_count=image_count,
        candidate_count=CANDIDATE_COUNT,
        episode_count=expected_episode_count,
        manifest_sha256=by_name[MANIFEST_FILENAME].sha256,
        complete_sha256=by_name[COMPLETE_FILENAME].sha256,
        phase_receipt_sha256=phase_sha,
    )


__all__ = [
    "ARRAY_FILENAMES",
    "ARTIFACT_TYPE",
    "CANDIDATE_COUNT",
    "COMPLETE_ARTIFACT_TYPE",
    "COMPLETE_FILENAME",
    "DRY_RUN_MEMBERS",
    "D0V3LabelFreeShardError",
    "ENTROPY_GRADIENTS_FILENAME",
    "EPISODES_FILENAME",
    "EPISODE_ARTIFACT_TYPE",
    "FLOAT_DTYPE",
    "FORMAL_IMAGE_COUNT",
    "FORMAL_MEMBERS",
    "LAYOUT_FILENAME",
    "MANIFEST_FILENAME",
    "PARAMETERS_AFTER_FILENAME",
    "PARAMETER_SCALAR_COUNT",
    "PARAMETER_TENSOR_COUNT",
    "PHASE_RECEIPT_FILENAME",
    "POST_LOGITS_FILENAME",
    "REQUIRED_CRITICAL_CODE_PATHS",
    "SOURCE_LOGITS_FILENAME",
    "SOURCE_PARAMETERS_FILENAME",
    "VerifiedLabelFreeShard",
    "array_slice_sha256",
    "canonical_json_bytes",
    "canonical_sha256",
    "expected_array_specs",
    "input_tensor_sha256",
    "named_bundle_sha256",
    "raw_float32_sha256",
    "validate_parameter_layout",
    "validate_method_facing_seal_path",
    "verify_label_free_shard",
]
