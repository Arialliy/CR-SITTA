"""CPU-only three-process engineering-smoke contract for D0-v2.

The D0-v2 candidate contract proves that every optimizer/LR candidate owns a
fresh model, method, optimizer, autograd graph and gradient buffers inside one
cell.  This module adds the next *engineering* boundary without changing the
sealed D0-v1 implementation:

* one process receipt binds a parent/child nonce, command digest and Linux
  process identity (PID plus ``/proc/<pid>/stat`` start-time ticks) to exactly
  ten independently executed candidate receipts and their recomputed cell
  aggregate;
* :func:`aggregate_three` accepts exactly three unique fresh-process receipts;
* config/sample/runtime/determinism/source/input/selected-parameter bindings
  and pre-adaptation logits must agree across processes;
* candidate schemas and integer structure gates must agree across processes;
* CUDA backward-dependent post-logit, gradient, parameter-delta and floating
  step-norm values are deliberately *not* required to be bit-exact.

Passing this contract is only an ``engineering_smoke`` result.  Every emitted
receipt explicitly keeps ``paper_result=false``, ``formal_p3_complete=false``
and ``stage2_authorized=false``.  No filesystem output or GPU work is performed
by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Final

from analysis.d0_v2_independent_candidate_contract import (
    FROZEN_CANDIDATES,
    IndependentCandidateCellReceipt,
    IndependentCandidateReceipt,
    validate_independent_candidate_cell,
)


SCHEMA_VERSION: Final = 2
PROTOCOL_ID: Final = "cr-sitta-d0-v2-independent-candidate-smoke-repro"
PROCESS_ARTIFACT_TYPE: Final = (
    "cr_sitta_d0_v2_independent_candidate_engineering_smoke_process"
)
AGGREGATE_ARTIFACT_TYPE: Final = (
    "cr_sitta_d0_v2_independent_candidate_engineering_smoke_three_process"
)
PROCESS_COUNT: Final = 3
CANDIDATE_COUNT: Final = 10
SCIENTIFIC_GATE_STATUS: Final = "unresolved"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROCESS_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,191}\Z")

_PROCESS_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "engineering_smoke",
        "paper_result",
        "formal_p3_complete",
        "stage2_authorized",
        "passed",
        "fresh_subprocess",
        "parent_run_nonce",
        "child_launch_nonce",
        "command_sha256",
        "process_identity",
        "candidate_count",
        "candidate_slugs",
        "candidate_receipts",
        "cell_aggregate",
        "cell_aggregate_sha256",
        "data_boundary",
        "authorization",
    }
)
_PROCESS_IDENTITY_KEYS = frozenset(
    {"process_id", "os_process_id", "process_start_time_ticks"}
)
_AGGREGATE_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "engineering_smoke",
        "paper_result",
        "formal_p3_complete",
        "stage2_authorized",
        "passed",
        "parent_run_nonce",
        "fresh_process_count",
        "process_ids",
        "process_instances",
        "process_receipt_sha256s",
        "process_receipts",
        "candidate_count",
        "candidate_slugs",
        "candidate_receipts_per_process",
        "total_candidate_receipt_count",
        "shared_cell_binding",
        "shared_pre_logits_sha256",
        "candidate_integer_profiles",
        "cross_process_policy",
        "cross_process_gates",
        "data_boundary",
        "authorization",
    }
)

_SHARED_CELL_FIELDS: Final = (
    "config_sha256",
    "dataset",
    "condition",
    "sample_index",
    "sample_id",
    "split_sha256",
    "checkpoint_sha256",
    "source_state_sha256",
    "runtime_sha256",
    "determinism_sha256",
    "input_sha256",
    "selected_parameter_names_sha256",
)
_CANDIDATE_INTEGER_FIELDS: Final = (
    "gradient_tensor_count",
    "changed_parameter_tensor_count",
    "optimizer_state_entry_count_after_step",
    "native_reference_parameter_tensor_count",
    "native_reference_optimizer_state_tensor_count",
)
_GLOBAL_EXECUTION_ID_FIELDS: Final = (
    "model_instance_id",
    "method_instance_id",
    "optimizer_instance_id",
    "autograd_graph_id",
    "backward_execution_id",
    "gradient_buffer_owner_id",
)

_DATA_BOUNDARY: Final = {
    "source_train_derived": True,
    "paper_test_result": False,
    "no_validation_split": True,
    "use_validation": False,
    "use_test_images": False,
    "use_test_labels": False,
    "method_label_accesses": 0,
    "target_payload_deserialized_during_candidates": False,
    "train_target_payload_bytes_opened": 0,
    "test_split_files_opened": 0,
    "test_images_opened": 0,
    "test_masks_opened": 0,
    "test_labels_opened": 0,
}
_AUTHORIZATION: Final = {
    "scientific_gate_status": SCIENTIFIC_GATE_STATUS,
    "scientific_selection_performed": False,
}
_CROSS_PROCESS_POLICY: Final = {
    "required_exact_shared_cell_fields": list(_SHARED_CELL_FIELDS),
    "source_state_sha256_contract": (
        "cr-sitta-d0-v2-source-state-excluding-candidate-optimizer-v1"
    ),
    "source_state_components": (
        "model_runtime_topology_gradients_extras"
    ),
    "candidate_optimizer_excluded_from_shared_source_hash": True,
    "candidate_optimizer_exact_reset_required_per_candidate": True,
    "required_exact_pre_logits_sha256": True,
    "required_exact_candidate_integer_fields": list(_CANDIDATE_INTEGER_FIELDS),
    "cuda_backward_bit_exact_required": False,
    "post_logits_sha256_cross_process_exact_required": False,
    "entropy_gradient_bundle_sha256_cross_process_exact_required": False,
    "parameter_delta_bundle_sha256_cross_process_exact_required": False,
    "step_norm_l2_cross_process_exact_required": False,
}


class D0V2SmokeReproContractError(ValueError):
    """A D0-v2 engineering-smoke receipt failed closed."""


@dataclass(frozen=True, slots=True)
class FreshSubprocessIdentity:
    """Logical and Linux identity for one child process."""

    process_id: str
    os_process_id: int
    process_start_time_ticks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_id": self.process_id,
            "os_process_id": self.os_process_id,
            "process_start_time_ticks": self.process_start_time_ticks,
        }

    @classmethod
    def capture_current(cls, process_id: str) -> "FreshSubprocessIdentity":
        """Capture the current Linux process identity for a future runner."""

        pid = os.getpid()
        return cls(
            process_id=_require_process_id(process_id, "process_id"),
            os_process_id=pid,
            process_start_time_ticks=read_proc_process_start_time_ticks(pid),
        )


def read_proc_process_start_time_ticks(os_process_id: int) -> int:
    """Read Linux ``/proc/<pid>/stat`` field 22 without assuming comm spacing."""

    pid = _require_integer(os_process_id, "os_process_id", minimum=1)
    path = Path("/proc") / str(pid) / "stat"
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise D0V2SmokeReproContractError(
            f"cannot read Linux process identity from {path}"
        ) from exc
    closing_paren = payload.rfind(")")
    if closing_paren < 0:
        raise D0V2SmokeReproContractError(f"malformed Linux process stat: {path}")
    # Tokens after the command name start at field 3 (state); starttime is
    # field 22, hence zero-based token index 19 in this suffix.
    suffix = payload[closing_paren + 1 :].split()
    if len(suffix) <= 19:
        raise D0V2SmokeReproContractError(f"malformed Linux process stat: {path}")
    try:
        ticks = int(suffix[19], 10)
    except ValueError as exc:
        raise D0V2SmokeReproContractError(
            f"malformed Linux process start ticks: {path}"
        ) from exc
    return _require_integer(ticks, "process_start_time_ticks", minimum=1)


def _require_exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise D0V2SmokeReproContractError(f"{label} must be a string-key mapping")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise D0V2SmokeReproContractError(
            f"{label} schema is not exact; missing={missing}, unknown={unknown}"
        )
    return value


def _require_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise D0V2SmokeReproContractError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _require_exact_integer(value: Any, expected: int, label: str) -> int:
    observed = _require_integer(value, label)
    if observed != expected:
        raise D0V2SmokeReproContractError(f"{label} must equal {expected}")
    return observed


def _require_exact_bool(value: Any, expected: bool, label: str) -> bool:
    if value is not expected:
        raise D0V2SmokeReproContractError(f"{label} must be exactly {expected}")
    return expected


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise D0V2SmokeReproContractError(
            f"{label} must be a lowercase 64-hex SHA-256"
        )
    return value


def _require_process_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _PROCESS_ID_RE.fullmatch(value) is None:
        raise D0V2SmokeReproContractError(f"{label} must be a safe logical process ID")
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
    except (TypeError, ValueError) as exc:
        raise D0V2SmokeReproContractError(
            "receipt must be canonical-JSON serializable"
        ) from exc


def _canonical_receipt_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value) + b"\n").hexdigest()


def _validate_static_mapping(
    value: Any, expected: Mapping[str, Any], label: str
) -> None:
    if not isinstance(value, Mapping):
        raise D0V2SmokeReproContractError(f"{label} must be a mapping")
    if _canonical_json_bytes(value) != _canonical_json_bytes(expected):
        raise D0V2SmokeReproContractError(f"{label} differs from the frozen contract")


def _coerce_process_identity(
    value: FreshSubprocessIdentity | Mapping[str, Any],
) -> FreshSubprocessIdentity:
    if isinstance(value, FreshSubprocessIdentity):
        identity = value
    else:
        mapping = _require_exact_keys(value, _PROCESS_IDENTITY_KEYS, "process_identity")
        identity = FreshSubprocessIdentity(
            process_id=_require_process_id(mapping["process_id"], "process_id"),
            os_process_id=_require_integer(
                mapping["os_process_id"], "os_process_id", minimum=1
            ),
            process_start_time_ticks=_require_integer(
                mapping["process_start_time_ticks"],
                "process_start_time_ticks",
                minimum=1,
            ),
        )
    _require_process_id(identity.process_id, "process_id")
    _require_integer(identity.os_process_id, "os_process_id", minimum=1)
    _require_integer(
        identity.process_start_time_ticks,
        "process_start_time_ticks",
        minimum=1,
    )
    return identity


def _validated_cell(
    candidate_receipts: Sequence[Mapping[str, Any] | IndependentCandidateReceipt],
) -> IndependentCandidateCellReceipt:
    if (
        isinstance(candidate_receipts, (str, bytes, bytearray, Mapping))
        or not isinstance(candidate_receipts, Sequence)
    ):
        raise D0V2SmokeReproContractError(
            "candidate_receipts must be a sequence of exactly ten receipts"
        )
    try:
        return validate_independent_candidate_cell(candidate_receipts)
    except (TypeError, ValueError) as exc:
        raise D0V2SmokeReproContractError(
            "independent candidate cell did not pass the D0-v2 contract"
        ) from exc


def _build_process_receipt(
    cell: IndependentCandidateCellReceipt,
    *,
    parent_run_nonce: str,
    child_launch_nonce: str,
    command_sha256: str,
    identity: FreshSubprocessIdentity,
) -> dict[str, Any]:
    candidate_receipts = [value.to_dict() for value in cell.candidate_receipts]
    cell_aggregate = cell.to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PROCESS_ARTIFACT_TYPE,
        "protocol_id": PROTOCOL_ID,
        "engineering_smoke": True,
        "paper_result": False,
        "formal_p3_complete": False,
        "stage2_authorized": False,
        "passed": True,
        "fresh_subprocess": True,
        "parent_run_nonce": parent_run_nonce,
        "child_launch_nonce": child_launch_nonce,
        "command_sha256": command_sha256,
        "process_identity": identity.to_dict(),
        "candidate_count": CANDIDATE_COUNT,
        "candidate_slugs": [candidate.slug for candidate in FROZEN_CANDIDATES],
        "candidate_receipts": candidate_receipts,
        "cell_aggregate": cell_aggregate,
        "cell_aggregate_sha256": _canonical_receipt_sha256(cell_aggregate),
        "data_boundary": dict(_DATA_BOUNDARY),
        "authorization": dict(_AUTHORIZATION),
    }


def build_d0_v2_smoke_process_receipt(
    candidate_receipts: Sequence[Mapping[str, Any] | IndependentCandidateReceipt],
    *,
    parent_run_nonce: str,
    child_launch_nonce: str,
    command_sha256: str,
    process_identity: FreshSubprocessIdentity | Mapping[str, Any],
) -> dict[str, Any]:
    """Build one canonical engineering-smoke process receipt.

    A child runner should call :meth:`FreshSubprocessIdentity.capture_current`
    inside the child and pass that identity here after all ten candidate
    receipts have been produced.
    """

    parent_nonce = _require_sha256(parent_run_nonce, "parent_run_nonce")
    child_nonce = _require_sha256(child_launch_nonce, "child_launch_nonce")
    command = _require_sha256(command_sha256, "command_sha256")
    identity = _coerce_process_identity(process_identity)
    cell = _validated_cell(candidate_receipts)
    return _build_process_receipt(
        cell,
        parent_run_nonce=parent_nonce,
        child_launch_nonce=child_nonce,
        command_sha256=command,
        identity=identity,
    )


def validate_d0_v2_smoke_process_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one exact process receipt."""

    root = _require_exact_keys(receipt, _PROCESS_ROOT_KEYS, "process receipt")
    _require_exact_integer(root["schema_version"], SCHEMA_VERSION, "schema_version")
    if root["artifact_type"] != PROCESS_ARTIFACT_TYPE:
        raise D0V2SmokeReproContractError("process artifact_type is not D0-v2 smoke")
    if root["protocol_id"] != PROTOCOL_ID:
        raise D0V2SmokeReproContractError("process protocol_id is not D0-v2 smoke")
    for field, expected in (
        ("engineering_smoke", True),
        ("paper_result", False),
        ("formal_p3_complete", False),
        ("stage2_authorized", False),
        ("passed", True),
        ("fresh_subprocess", True),
    ):
        _require_exact_bool(root[field], expected, field)
    parent_nonce = _require_sha256(root["parent_run_nonce"], "parent_run_nonce")
    child_nonce = _require_sha256(root["child_launch_nonce"], "child_launch_nonce")
    command = _require_sha256(root["command_sha256"], "command_sha256")
    identity = _coerce_process_identity(root["process_identity"])
    _require_exact_integer(root["candidate_count"], CANDIDATE_COUNT, "candidate_count")
    expected_slugs = [candidate.slug for candidate in FROZEN_CANDIDATES]
    if root["candidate_slugs"] != expected_slugs:
        raise D0V2SmokeReproContractError("candidate_slugs differ from frozen grid")
    cell = _validated_cell(root["candidate_receipts"])
    expected_cell = cell.to_dict()
    if _canonical_json_bytes(root["cell_aggregate"]) != _canonical_json_bytes(
        expected_cell
    ):
        raise D0V2SmokeReproContractError(
            "cell_aggregate does not recompute from candidate_receipts"
        )
    cell_sha = _require_sha256(root["cell_aggregate_sha256"], "cell_aggregate_sha256")
    if cell_sha != _canonical_receipt_sha256(expected_cell):
        raise D0V2SmokeReproContractError("cell_aggregate_sha256 binding mismatch")
    _validate_static_mapping(root["data_boundary"], _DATA_BOUNDARY, "data_boundary")
    _validate_static_mapping(root["authorization"], _AUTHORIZATION, "authorization")

    expected = _build_process_receipt(
        cell,
        parent_run_nonce=parent_nonce,
        child_launch_nonce=child_nonce,
        command_sha256=command,
        identity=identity,
    )
    if _canonical_json_bytes(root) != _canonical_json_bytes(expected):
        raise D0V2SmokeReproContractError("process receipt is not canonical or exact")
    return expected


def canonical_d0_v2_smoke_process_receipt_bytes(
    receipt: Mapping[str, Any],
) -> bytes:
    """Return validated canonical process-receipt JSON plus one newline."""

    return _canonical_json_bytes(validate_d0_v2_smoke_process_receipt(receipt)) + b"\n"


def d0_v2_smoke_process_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash one validated process receipt's canonical bytes."""

    return hashlib.sha256(
        canonical_d0_v2_smoke_process_receipt_bytes(receipt)
    ).hexdigest()


def _cell_from_process_receipt(
    receipt: Mapping[str, Any],
) -> IndependentCandidateCellReceipt:
    return _validated_cell(receipt["candidate_receipts"])


def _integer_profiles(cell: IndependentCandidateCellReceipt) -> list[dict[str, Any]]:
    return [
        {
            "candidate_index": value.candidate_index,
            "candidate_slug": value.candidate.slug,
            "gradient_tensor_count": value.gradient_tensor_count,
            "changed_parameter_tensor_count": value.changed_parameter_tensor_count,
            "optimizer_state_entry_count_after_step": (
                value.optimizer_state_entry_count_after_step
            ),
            "native_reference_parameter_tensor_count": (
                value.native_reference_parameter_tensor_count
            ),
            "native_reference_optimizer_state_tensor_count": (
                value.native_reference_optimizer_state_tensor_count
            ),
        }
        for value in cell.candidate_receipts
    ]


def _assert_shared_process_contract(
    receipts: Sequence[Mapping[str, Any]],
    cells: Sequence[IndependentCandidateCellReceipt],
) -> None:
    reference = cells[0]
    for field in _SHARED_CELL_FIELDS:
        expected = getattr(reference, field)
        if any(getattr(cell, field) != expected for cell in cells[1:]):
            raise D0V2SmokeReproContractError(
                f"fresh processes disagree on shared {field}"
            )
    expected_pre = reference.candidate_receipts[0].pre_logits_sha256
    if any(
        cell.candidate_receipts[0].pre_logits_sha256 != expected_pre
        for cell in cells[1:]
    ):
        raise D0V2SmokeReproContractError(
            "fresh processes disagree on shared pre_logits_sha256"
        )

    expected_profile = _integer_profiles(reference)
    if any(_integer_profiles(cell) != expected_profile for cell in cells[1:]):
        raise D0V2SmokeReproContractError(
            "fresh processes disagree on candidate integer structure gates"
        )

    # These opaque execution identities are not numeric-equality gates.  Their
    # uniqueness prevents copying one process cell and merely changing its PID.
    for field in _GLOBAL_EXECUTION_ID_FIELDS:
        values = [
            getattr(candidate, field)
            for cell in cells
            for candidate in cell.candidate_receipts
        ]
        if len(set(values)) != PROCESS_COUNT * CANDIDATE_COUNT:
            raise D0V2SmokeReproContractError(
                f"cross-process {field} reuse is forbidden"
            )

    parent_nonce = receipts[0]["parent_run_nonce"]
    if any(value["parent_run_nonce"] != parent_nonce for value in receipts[1:]):
        raise D0V2SmokeReproContractError(
            "fresh processes disagree on parent_run_nonce"
        )


def aggregate_three(
    process_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exactly three fresh-process receipts and build the aggregate.

    The return value is deliberately non-authorizing.  Cross-process CUDA
    backward/post/gradient/delta hashes and floating step norms may differ.
    """

    if (
        isinstance(process_receipts, (str, bytes, bytearray, Mapping))
        or not isinstance(process_receipts, Sequence)
        or len(process_receipts) != PROCESS_COUNT
    ):
        raise D0V2SmokeReproContractError(
            "engineering smoke requires exactly three process receipts"
        )
    normalized = [
        validate_d0_v2_smoke_process_receipt(value)
        for value in process_receipts
    ]
    normalized.sort(key=lambda value: value["process_identity"]["process_id"])

    process_ids = [value["process_identity"]["process_id"] for value in normalized]
    if len(set(process_ids)) != PROCESS_COUNT:
        raise D0V2SmokeReproContractError(
            "engineering smoke requires three unique logical process IDs"
        )
    process_instances = [
        (
            value["process_identity"]["os_process_id"],
            value["process_identity"]["process_start_time_ticks"],
        )
        for value in normalized
    ]
    if len(set(process_instances)) != PROCESS_COUNT:
        raise D0V2SmokeReproContractError(
            "engineering smoke requires three unique fresh Linux process identities"
        )
    child_nonces = [value["child_launch_nonce"] for value in normalized]
    if len(set(child_nonces)) != PROCESS_COUNT:
        raise D0V2SmokeReproContractError(
            "engineering smoke requires three unique child_launch_nonce values"
        )

    cells = [_cell_from_process_receipt(value) for value in normalized]
    _assert_shared_process_contract(normalized, cells)
    reference = cells[0]
    reference_cell_binding = reference.to_dict()["cell_binding"]
    pre_logits_sha256 = reference.candidate_receipts[0].pre_logits_sha256
    receipt_hashes = [d0_v2_smoke_process_receipt_sha256(value) for value in normalized]
    if len(set(receipt_hashes)) != PROCESS_COUNT:
        raise D0V2SmokeReproContractError(
            "engineering smoke process receipt hashes must be unique"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": AGGREGATE_ARTIFACT_TYPE,
        "protocol_id": PROTOCOL_ID,
        "engineering_smoke": True,
        "paper_result": False,
        "formal_p3_complete": False,
        "stage2_authorized": False,
        "passed": True,
        "parent_run_nonce": normalized[0]["parent_run_nonce"],
        "fresh_process_count": PROCESS_COUNT,
        "process_ids": process_ids,
        "process_instances": [value["process_identity"] for value in normalized],
        "process_receipt_sha256s": receipt_hashes,
        "process_receipts": normalized,
        "candidate_count": CANDIDATE_COUNT,
        "candidate_slugs": [candidate.slug for candidate in FROZEN_CANDIDATES],
        "candidate_receipts_per_process": CANDIDATE_COUNT,
        "total_candidate_receipt_count": PROCESS_COUNT * CANDIDATE_COUNT,
        "shared_cell_binding": reference_cell_binding,
        "shared_pre_logits_sha256": pre_logits_sha256,
        "candidate_integer_profiles": _integer_profiles(reference),
        "cross_process_policy": dict(_CROSS_PROCESS_POLICY),
        "cross_process_gates": {
            "fresh_process_identities_unique": True,
            "logical_process_ids_unique": True,
            "child_launch_nonces_unique": True,
            "all_process_cells_passed": True,
            "shared_cell_binding_exact": True,
            "shared_source_state_exact": True,
            "shared_pre_logits_exact": True,
            "candidate_grid_structure_exact": True,
            "same_device_native_step_endpoint_exact_all_candidates": True,
            "same_device_native_optimizer_state_exact_all_candidates": True,
            "candidate_execution_identity_reuse_absent": True,
            "candidate_integer_structure_gates_exact": True,
        },
        "data_boundary": dict(_DATA_BOUNDARY),
        "authorization": dict(_AUTHORIZATION),
    }


def validate_d0_v2_smoke_repro_aggregate(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an aggregate by rebuilding it from its embedded receipts."""

    root = _require_exact_keys(receipt, _AGGREGATE_ROOT_KEYS, "smoke aggregate")
    _require_exact_integer(root["schema_version"], SCHEMA_VERSION, "schema_version")
    if root["artifact_type"] != AGGREGATE_ARTIFACT_TYPE:
        raise D0V2SmokeReproContractError("aggregate artifact_type is not D0-v2 smoke")
    if root["protocol_id"] != PROTOCOL_ID:
        raise D0V2SmokeReproContractError("aggregate protocol_id is not D0-v2 smoke")
    for field, expected in (
        ("engineering_smoke", True),
        ("paper_result", False),
        ("formal_p3_complete", False),
        ("stage2_authorized", False),
        ("passed", True),
    ):
        _require_exact_bool(root[field], expected, field)
    expected = aggregate_three(root["process_receipts"])
    if _canonical_json_bytes(root) != _canonical_json_bytes(expected):
        raise D0V2SmokeReproContractError(
            "smoke aggregate does not recompute exactly from process_receipts"
        )
    return expected


def canonical_d0_v2_smoke_repro_aggregate_bytes(
    receipt: Mapping[str, Any],
) -> bytes:
    """Return validated canonical aggregate JSON plus one newline."""

    return _canonical_json_bytes(validate_d0_v2_smoke_repro_aggregate(receipt)) + b"\n"


def d0_v2_smoke_repro_aggregate_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash one validated aggregate's canonical bytes."""

    return hashlib.sha256(
        canonical_d0_v2_smoke_repro_aggregate_bytes(receipt)
    ).hexdigest()


__all__ = [
    "AGGREGATE_ARTIFACT_TYPE",
    "CANDIDATE_COUNT",
    "D0V2SmokeReproContractError",
    "FreshSubprocessIdentity",
    "PROCESS_ARTIFACT_TYPE",
    "PROCESS_COUNT",
    "PROTOCOL_ID",
    "aggregate_three",
    "build_d0_v2_smoke_process_receipt",
    "canonical_d0_v2_smoke_process_receipt_bytes",
    "canonical_d0_v2_smoke_repro_aggregate_bytes",
    "d0_v2_smoke_process_receipt_sha256",
    "d0_v2_smoke_repro_aggregate_sha256",
    "read_proc_process_start_time_ticks",
    "validate_d0_v2_smoke_process_receipt",
    "validate_d0_v2_smoke_repro_aggregate",
]
