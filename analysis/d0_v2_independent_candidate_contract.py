"""Fail-closed D0-v2 contract for independent candidate execution.

This module is deliberately independent of the sealed D0-v1 runner.  It
defines two complementary gates:

* :class:`IndependentCandidateExecutionLedger` keeps strong references to the
  live model, method, optimizer, autograd-graph token and gradient buffers for
  every candidate.  Reusing any of those objects across candidates is rejected
  before a cell can be declared complete.
* :func:`validate_independent_candidate_cell` validates the persisted receipts
  for the frozen ten-candidate optimizer/LR grid.  Numeric hashes are allowed
  to coincide naturally, but provenance/ownership identifiers must be unique.

Passing this engineering contract is not a scientific selection result.  The
canonical aggregate always records ``scientific_gate_status=unresolved`` and
``stage2_authorized=false``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any


PROTOCOL_ID = "cr-sitta-d0-v2-independent-candidates"
RECEIPT_ARTIFACT_TYPE = "cr_sitta_d0_v2_independent_candidate_receipt"
AGGREGATE_ARTIFACT_TYPE = "cr_sitta_d0_v2_independent_candidate_cell"
SCIENTIFIC_GATE_STATUS = "unresolved"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,191}\Z")


class D0V2IndependentCandidateError(ValueError):
    """Independent execution or receipt evidence is incomplete or unsafe."""


@dataclass(frozen=True, order=True)
class CandidateSpec:
    optimizer: str
    learning_rate: float

    @property
    def slug(self) -> str:
        value = format(self.learning_rate, ".0e").replace("e-0", "e-")
        return f"{self.optimizer}_lr_{value.replace('-', 'm').replace('+', 'p')}"


FROZEN_CANDIDATES = tuple(
    CandidateSpec(optimizer, learning_rate)
    for optimizer in ("Adam", "SGD")
    for learning_rate in (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3)
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise D0V2IndependentCandidateError(
            f"{context} fields must be exact; missing={missing}, unknown={unknown}"
        )


def _require_bool(value: Any, expected: bool, *, field: str) -> bool:
    if value is not expected:
        raise D0V2IndependentCandidateError(f"{field} must be exactly {expected}")
    return expected


def _require_int(
    value: Any, *, field: str, minimum: int = 0, exact: int | None = None
) -> int:
    if not _is_int(value):
        raise D0V2IndependentCandidateError(f"{field} must be an integer")
    if exact is not None and value != exact:
        raise D0V2IndependentCandidateError(f"{field} must be exactly {exact}")
    if value < minimum:
        raise D0V2IndependentCandidateError(f"{field} must be >= {minimum}")
    return value


def _require_float(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D0V2IndependentCandidateError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise D0V2IndependentCandidateError(
            f"{field} must be finite and >= {minimum}"
        )
    return result


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise D0V2IndependentCandidateError(f"{field} must be a non-empty string")
    return value


def _require_identity(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or IDENTITY_RE.fullmatch(value) is None:
        raise D0V2IndependentCandidateError(
            f"{field} must be a stable non-empty identity token"
        )
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise D0V2IndependentCandidateError(
            f"{field} must be lowercase 64-hex SHA-256"
        )
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class IndependentCandidateReceipt:
    candidate_index: int
    candidate: CandidateSpec
    config_sha256: str
    dataset: str
    condition: str
    sample_index: int
    sample_id: str
    split_sha256: str
    checkpoint_sha256: str
    source_state_sha256: str
    runtime_sha256: str
    determinism_sha256: str
    input_sha256: str
    selected_parameter_names_sha256: str
    pre_logits_sha256: str
    post_logits_sha256: str
    entropy_gradient_bundle_sha256: str
    parameter_delta_bundle_sha256: str
    model_instance_id: str
    method_instance_id: str
    optimizer_instance_id: str
    autograd_graph_id: str
    backward_execution_id: str
    gradient_buffer_owner_id: str
    gradient_tensor_count: int
    changed_parameter_tensor_count: int
    optimizer_state_entry_count_after_step: int
    native_reference_parameter_tensor_count: int
    native_reference_optimizer_state_tensor_count: int
    step_norm_l2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "artifact_type": RECEIPT_ARTIFACT_TYPE,
            "protocol_id": PROTOCOL_ID,
            "candidate_index": self.candidate_index,
            "candidate": {
                "optimizer": self.candidate.optimizer,
                "learning_rate": self.candidate.learning_rate,
                "slug": self.candidate.slug,
            },
            "cell_binding": {
                "config_sha256": self.config_sha256,
                "dataset": self.dataset,
                "condition": self.condition,
                "sample_index": self.sample_index,
                "sample_id": self.sample_id,
                "split_name": "train",
                "split_role": "frozen_pilot64",
                "split_sha256": self.split_sha256,
                "checkpoint_sha256": self.checkpoint_sha256,
                "source_state_sha256": self.source_state_sha256,
                "runtime_sha256": self.runtime_sha256,
                "determinism_sha256": self.determinism_sha256,
                "input_sha256": self.input_sha256,
                "selected_parameter_names_sha256": (
                    self.selected_parameter_names_sha256
                ),
            },
            "execution_identity": {
                "model_instance_id": self.model_instance_id,
                "method_instance_id": self.method_instance_id,
                "optimizer_instance_id": self.optimizer_instance_id,
                "autograd_graph_id": self.autograd_graph_id,
                "backward_execution_id": self.backward_execution_id,
                "gradient_buffer_owner_id": self.gradient_buffer_owner_id,
            },
            "execution_gates": {
                "fresh_model_built": True,
                "fresh_method_built": True,
                "fresh_optimizer_built": True,
                "own_forward_backward": True,
                "gradient_reuse": False,
                "shared_autograd_graph_across_candidates": False,
                "shared_gradient_buffers_across_candidates": False,
                "optimizer_state_entry_count_before_step": 0,
                "pre_backward_gradient_tensor_count": 0,
                "entropy_backward_pass_count": 1,
                "optimizer_step_count": 1,
                "source_parameter_reset_exact": True,
                "source_runtime_reset_exact": True,
                "source_rng_reset_exact": True,
                "selected_parameter_state_equal_source_before_forward": True,
                "non_selected_parameter_state_equal_source_after_step": True,
                "bn_running_statistics_equal_source_after_episode": True,
                "gradients_cleared_after_episode": True,
                "optimizer_state_discarded_after_episode": True,
                "same_device_native_step_endpoint_exact": True,
                "same_device_native_optimizer_state_exact": True,
                "amp_enabled": False,
                "finite": True,
            },
            "numeric_evidence": {
                "pre_logits_sha256": self.pre_logits_sha256,
                "post_logits_sha256": self.post_logits_sha256,
                "entropy_gradient_bundle_sha256": (
                    self.entropy_gradient_bundle_sha256
                ),
                "parameter_delta_bundle_sha256": (
                    self.parameter_delta_bundle_sha256
                ),
                "gradient_tensor_count": self.gradient_tensor_count,
                "changed_parameter_tensor_count": (
                    self.changed_parameter_tensor_count
                ),
                "optimizer_state_entry_count_after_step": (
                    self.optimizer_state_entry_count_after_step
                ),
                "native_reference_parameter_tensor_count": (
                    self.native_reference_parameter_tensor_count
                ),
                "native_reference_optimizer_state_tensor_count": (
                    self.native_reference_optimizer_state_tensor_count
                ),
                "step_norm_l2": self.step_norm_l2,
            },
            "data_boundary": {
                "source_train_derived": True,
                "paper_test_result": False,
                "no_validation_split": True,
                "use_validation": False,
                "use_test_images": False,
                "use_test_labels": False,
                "method_label_accesses": 0,
                "target_payload_deserialized_during_candidate": False,
                "train_target_payload_bytes_opened": 0,
                "test_split_files_opened": 0,
                "test_images_opened": 0,
                "test_masks_opened": 0,
                "test_labels_opened": 0,
            },
            "authorization": {
                "engineering_gate_passed": True,
                "scientific_gate_status": SCIENTIFIC_GATE_STATUS,
                "scientific_selection_performed": False,
                "stage2_authorized": False,
            },
        }


_ROOT_FIELDS = {
    "schema_version",
    "artifact_type",
    "protocol_id",
    "candidate_index",
    "candidate",
    "cell_binding",
    "execution_identity",
    "execution_gates",
    "numeric_evidence",
    "data_boundary",
    "authorization",
}
_CANDIDATE_FIELDS = {"optimizer", "learning_rate", "slug"}
_CELL_FIELDS = {
    "config_sha256",
    "dataset",
    "condition",
    "sample_index",
    "sample_id",
    "split_name",
    "split_role",
    "split_sha256",
    "checkpoint_sha256",
    "source_state_sha256",
    "runtime_sha256",
    "determinism_sha256",
    "input_sha256",
    "selected_parameter_names_sha256",
}
_IDENTITY_FIELDS = {
    "model_instance_id",
    "method_instance_id",
    "optimizer_instance_id",
    "autograd_graph_id",
    "backward_execution_id",
    "gradient_buffer_owner_id",
}
_GATE_FIELDS = {
    "fresh_model_built",
    "fresh_method_built",
    "fresh_optimizer_built",
    "own_forward_backward",
    "gradient_reuse",
    "shared_autograd_graph_across_candidates",
    "shared_gradient_buffers_across_candidates",
    "optimizer_state_entry_count_before_step",
    "pre_backward_gradient_tensor_count",
    "entropy_backward_pass_count",
    "optimizer_step_count",
    "source_parameter_reset_exact",
    "source_runtime_reset_exact",
    "source_rng_reset_exact",
    "selected_parameter_state_equal_source_before_forward",
    "non_selected_parameter_state_equal_source_after_step",
    "bn_running_statistics_equal_source_after_episode",
    "gradients_cleared_after_episode",
    "optimizer_state_discarded_after_episode",
    "same_device_native_step_endpoint_exact",
    "same_device_native_optimizer_state_exact",
    "amp_enabled",
    "finite",
}
_NUMERIC_FIELDS = {
    "pre_logits_sha256",
    "post_logits_sha256",
    "entropy_gradient_bundle_sha256",
    "parameter_delta_bundle_sha256",
    "gradient_tensor_count",
    "changed_parameter_tensor_count",
    "optimizer_state_entry_count_after_step",
    "native_reference_parameter_tensor_count",
    "native_reference_optimizer_state_tensor_count",
    "step_norm_l2",
}
_BOUNDARY_FIELDS = {
    "source_train_derived",
    "paper_test_result",
    "no_validation_split",
    "use_validation",
    "use_test_images",
    "use_test_labels",
    "method_label_accesses",
    "target_payload_deserialized_during_candidate",
    "train_target_payload_bytes_opened",
    "test_split_files_opened",
    "test_images_opened",
    "test_masks_opened",
    "test_labels_opened",
}
_AUTHORIZATION_FIELDS = {
    "engineering_gate_passed",
    "scientific_gate_status",
    "scientific_selection_performed",
    "stage2_authorized",
}


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0V2IndependentCandidateError(f"{field} must be a mapping")
    return value


def parse_independent_candidate_receipt(
    value: Mapping[str, Any],
) -> IndependentCandidateReceipt:
    """Parse one exact receipt and reject every unknown or unsafe value."""

    root = _mapping(value, field="receipt")
    _require_exact_keys(root, _ROOT_FIELDS, context="receipt")
    _require_int(root["schema_version"], field="schema_version", exact=2)
    if root["artifact_type"] != RECEIPT_ARTIFACT_TYPE:
        raise D0V2IndependentCandidateError("artifact_type is not D0-v2")
    if root["protocol_id"] != PROTOCOL_ID:
        raise D0V2IndependentCandidateError("protocol_id is not D0-v2")

    candidate_index = _require_int(
        root["candidate_index"], field="candidate_index", minimum=0
    )
    if candidate_index >= len(FROZEN_CANDIDATES):
        raise D0V2IndependentCandidateError("candidate_index is outside frozen grid")
    expected_candidate = FROZEN_CANDIDATES[candidate_index]
    candidate = _mapping(root["candidate"], field="candidate")
    _require_exact_keys(candidate, _CANDIDATE_FIELDS, context="candidate")
    if (
        candidate["optimizer"] != expected_candidate.optimizer
        or not isinstance(candidate["learning_rate"], float)
        or candidate["learning_rate"] != expected_candidate.learning_rate
        or candidate["slug"] != expected_candidate.slug
    ):
        raise D0V2IndependentCandidateError(
            f"candidate_index {candidate_index} does not match frozen candidate"
        )

    binding = _mapping(root["cell_binding"], field="cell_binding")
    _require_exact_keys(binding, _CELL_FIELDS, context="cell_binding")
    if binding["split_name"] != "train" or binding["split_role"] != "frozen_pilot64":
        raise D0V2IndependentCandidateError(
            "cell_binding must use only the frozen train-side Pilot64"
        )
    config_sha256 = _require_sha256(
        binding["config_sha256"], field="config_sha256"
    )
    dataset = _require_string(binding["dataset"], field="dataset")
    condition = _require_string(binding["condition"], field="condition")
    sample_index = _require_int(
        binding["sample_index"], field="sample_index", minimum=0
    )
    sample_id = _require_string(binding["sample_id"], field="sample_id")
    sha_binding = {
        field: _require_sha256(binding[field], field=field)
        for field in (
            "split_sha256",
            "checkpoint_sha256",
            "source_state_sha256",
            "runtime_sha256",
            "determinism_sha256",
            "input_sha256",
            "selected_parameter_names_sha256",
        )
    }

    identities = _mapping(root["execution_identity"], field="execution_identity")
    _require_exact_keys(identities, _IDENTITY_FIELDS, context="execution_identity")
    identity_values = {
        field: _require_identity(identities[field], field=field)
        for field in sorted(_IDENTITY_FIELDS)
    }

    gates = _mapping(root["execution_gates"], field="execution_gates")
    _require_exact_keys(gates, _GATE_FIELDS, context="execution_gates")
    for field in (
        "fresh_model_built",
        "fresh_method_built",
        "fresh_optimizer_built",
        "own_forward_backward",
        "source_parameter_reset_exact",
        "source_runtime_reset_exact",
        "source_rng_reset_exact",
        "selected_parameter_state_equal_source_before_forward",
        "non_selected_parameter_state_equal_source_after_step",
        "bn_running_statistics_equal_source_after_episode",
        "gradients_cleared_after_episode",
        "optimizer_state_discarded_after_episode",
        "same_device_native_step_endpoint_exact",
        "same_device_native_optimizer_state_exact",
        "finite",
    ):
        _require_bool(gates[field], True, field=field)
    for field in (
        "gradient_reuse",
        "shared_autograd_graph_across_candidates",
        "shared_gradient_buffers_across_candidates",
        "amp_enabled",
    ):
        _require_bool(gates[field], False, field=field)
    for field, expected in (
        ("optimizer_state_entry_count_before_step", 0),
        ("pre_backward_gradient_tensor_count", 0),
        ("entropy_backward_pass_count", 1),
        ("optimizer_step_count", 1),
    ):
        _require_int(gates[field], field=field, exact=expected)

    numeric = _mapping(root["numeric_evidence"], field="numeric_evidence")
    _require_exact_keys(numeric, _NUMERIC_FIELDS, context="numeric_evidence")
    numeric_hashes = {
        field: _require_sha256(numeric[field], field=field)
        for field in (
            "pre_logits_sha256",
            "post_logits_sha256",
            "entropy_gradient_bundle_sha256",
            "parameter_delta_bundle_sha256",
        )
    }
    gradient_tensor_count = _require_int(
        numeric["gradient_tensor_count"], field="gradient_tensor_count", minimum=1
    )
    changed_count = _require_int(
        numeric["changed_parameter_tensor_count"],
        field="changed_parameter_tensor_count",
        minimum=0,
    )
    if changed_count > gradient_tensor_count:
        raise D0V2IndependentCandidateError(
            "changed_parameter_tensor_count cannot exceed gradient_tensor_count"
        )
    state_after = _require_int(
        numeric["optimizer_state_entry_count_after_step"],
        field="optimizer_state_entry_count_after_step",
        minimum=1,
    )
    if state_after != gradient_tensor_count:
        raise D0V2IndependentCandidateError(
            "optimizer_state_entry_count_after_step must equal "
            "gradient_tensor_count for the frozen Adam/SGD configurations"
        )
    native_parameter_count = _require_int(
        numeric["native_reference_parameter_tensor_count"],
        field="native_reference_parameter_tensor_count",
        minimum=1,
    )
    if native_parameter_count != gradient_tensor_count:
        raise D0V2IndependentCandidateError(
            "native_reference_parameter_tensor_count must equal "
            "gradient_tensor_count"
        )
    native_state_tensor_count = _require_int(
        numeric["native_reference_optimizer_state_tensor_count"],
        field="native_reference_optimizer_state_tensor_count",
        minimum=1,
    )
    step_norm = _require_float(
        numeric["step_norm_l2"], field="step_norm_l2", minimum=0.0
    )

    boundary = _mapping(root["data_boundary"], field="data_boundary")
    _require_exact_keys(boundary, _BOUNDARY_FIELDS, context="data_boundary")
    for field in ("source_train_derived", "no_validation_split"):
        _require_bool(boundary[field], True, field=field)
    for field in (
        "paper_test_result",
        "use_validation",
        "use_test_images",
        "use_test_labels",
        "target_payload_deserialized_during_candidate",
    ):
        _require_bool(boundary[field], False, field=field)
    for field in (
        "method_label_accesses",
        "train_target_payload_bytes_opened",
        "test_split_files_opened",
        "test_images_opened",
        "test_masks_opened",
        "test_labels_opened",
    ):
        _require_int(boundary[field], field=field, exact=0)

    authorization = _mapping(root["authorization"], field="authorization")
    _require_exact_keys(
        authorization, _AUTHORIZATION_FIELDS, context="authorization"
    )
    _require_bool(
        authorization["engineering_gate_passed"],
        True,
        field="engineering_gate_passed",
    )
    if authorization["scientific_gate_status"] != SCIENTIFIC_GATE_STATUS:
        raise D0V2IndependentCandidateError(
            "scientific_gate_status must remain unresolved"
        )
    _require_bool(
        authorization["scientific_selection_performed"],
        False,
        field="scientific_selection_performed",
    )
    _require_bool(
        authorization["stage2_authorized"], False, field="stage2_authorized"
    )

    return IndependentCandidateReceipt(
        candidate_index=candidate_index,
        candidate=expected_candidate,
        config_sha256=config_sha256,
        dataset=dataset,
        condition=condition,
        sample_index=sample_index,
        sample_id=sample_id,
        split_sha256=sha_binding["split_sha256"],
        checkpoint_sha256=sha_binding["checkpoint_sha256"],
        source_state_sha256=sha_binding["source_state_sha256"],
        runtime_sha256=sha_binding["runtime_sha256"],
        determinism_sha256=sha_binding["determinism_sha256"],
        input_sha256=sha_binding["input_sha256"],
        selected_parameter_names_sha256=sha_binding[
            "selected_parameter_names_sha256"
        ],
        pre_logits_sha256=numeric_hashes["pre_logits_sha256"],
        post_logits_sha256=numeric_hashes["post_logits_sha256"],
        entropy_gradient_bundle_sha256=numeric_hashes[
            "entropy_gradient_bundle_sha256"
        ],
        parameter_delta_bundle_sha256=numeric_hashes[
            "parameter_delta_bundle_sha256"
        ],
        model_instance_id=identity_values["model_instance_id"],
        method_instance_id=identity_values["method_instance_id"],
        optimizer_instance_id=identity_values["optimizer_instance_id"],
        autograd_graph_id=identity_values["autograd_graph_id"],
        backward_execution_id=identity_values["backward_execution_id"],
        gradient_buffer_owner_id=identity_values["gradient_buffer_owner_id"],
        gradient_tensor_count=gradient_tensor_count,
        changed_parameter_tensor_count=changed_count,
        optimizer_state_entry_count_after_step=state_after,
        native_reference_parameter_tensor_count=native_parameter_count,
        native_reference_optimizer_state_tensor_count=native_state_tensor_count,
        step_norm_l2=step_norm,
    )


@dataclass(frozen=True)
class IndependentCandidateCellReceipt:
    config_sha256: str
    dataset: str
    condition: str
    sample_index: int
    sample_id: str
    split_sha256: str
    checkpoint_sha256: str
    source_state_sha256: str
    runtime_sha256: str
    determinism_sha256: str
    input_sha256: str
    selected_parameter_names_sha256: str
    candidate_receipts: tuple[IndependentCandidateReceipt, ...]

    def to_dict(self) -> dict[str, Any]:
        receipt_dicts = [receipt.to_dict() for receipt in self.candidate_receipts]
        receipt_hashes = [_canonical_sha256(value) for value in receipt_dicts]
        return {
            "schema_version": 2,
            "artifact_type": AGGREGATE_ARTIFACT_TYPE,
            "protocol_id": PROTOCOL_ID,
            "cell_binding": {
                "config_sha256": self.config_sha256,
                "dataset": self.dataset,
                "condition": self.condition,
                "sample_index": self.sample_index,
                "sample_id": self.sample_id,
                "split_name": "train",
                "split_role": "frozen_pilot64",
                "split_sha256": self.split_sha256,
                "checkpoint_sha256": self.checkpoint_sha256,
                "source_state_sha256": self.source_state_sha256,
                "runtime_sha256": self.runtime_sha256,
                "determinism_sha256": self.determinism_sha256,
                "input_sha256": self.input_sha256,
                "selected_parameter_names_sha256": (
                    self.selected_parameter_names_sha256
                ),
            },
            "candidate_count": len(self.candidate_receipts),
            "candidate_slugs": [
                receipt.candidate.slug for receipt in self.candidate_receipts
            ],
            "candidate_receipt_sha256s": receipt_hashes,
            "independent_execution": {
                "fresh_model_count": len(
                    {value.model_instance_id for value in self.candidate_receipts}
                ),
                "fresh_method_count": len(
                    {value.method_instance_id for value in self.candidate_receipts}
                ),
                "fresh_optimizer_count": len(
                    {value.optimizer_instance_id for value in self.candidate_receipts}
                ),
                "own_autograd_graph_count": len(
                    {value.autograd_graph_id for value in self.candidate_receipts}
                ),
                "own_backward_execution_count": len(
                    {value.backward_execution_id for value in self.candidate_receipts}
                ),
                "own_gradient_buffer_owner_count": len(
                    {
                        value.gradient_buffer_owner_id
                        for value in self.candidate_receipts
                    }
                ),
                "gradient_reuse": False,
                "shared_autograd_graph_across_candidates": False,
                "shared_gradient_buffers_across_candidates": False,
                "engineering_gate_passed": True,
            },
            "data_boundary": {
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
            },
            "authorization": {
                "scientific_gate_status": SCIENTIFIC_GATE_STATUS,
                "scientific_selection_performed": False,
                "stage2_authorized": False,
            },
        }


_SHARED_FIELDS = (
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
    "pre_logits_sha256",
)
_UNIQUE_ID_FIELDS = (
    "model_instance_id",
    "method_instance_id",
    "optimizer_instance_id",
    "autograd_graph_id",
    "backward_execution_id",
    "gradient_buffer_owner_id",
)


def validate_independent_candidate_cell(
    receipts: Sequence[Mapping[str, Any] | IndependentCandidateReceipt],
) -> IndependentCandidateCellReceipt:
    """Validate exactly ten independently executed candidate receipts."""

    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise D0V2IndependentCandidateError("receipts must be a sequence")
    parsed = tuple(
        parse_independent_candidate_receipt(value.to_dict())
        if isinstance(value, IndependentCandidateReceipt)
        else parse_independent_candidate_receipt(value)
        for value in receipts
    )
    if len(parsed) != len(FROZEN_CANDIDATES):
        raise D0V2IndependentCandidateError(
            f"cell must contain exactly {len(FROZEN_CANDIDATES)} receipts"
        )
    by_index = {value.candidate_index: value for value in parsed}
    if len(by_index) != len(parsed) or set(by_index) != set(range(len(parsed))):
        raise D0V2IndependentCandidateError(
            "cell must contain each frozen candidate_index exactly once"
        )
    ordered = tuple(by_index[index] for index in range(len(FROZEN_CANDIDATES)))

    first = ordered[0]
    for field in _SHARED_FIELDS:
        expected = getattr(first, field)
        if any(getattr(value, field) != expected for value in ordered[1:]):
            raise D0V2IndependentCandidateError(
                f"candidate receipts disagree on shared {field}"
            )
    for field in _UNIQUE_ID_FIELDS:
        identities = [getattr(value, field) for value in ordered]
        if len(set(identities)) != len(identities):
            raise D0V2IndependentCandidateError(
                f"cross-candidate {field} reuse is forbidden"
            )

    return IndependentCandidateCellReceipt(
        config_sha256=first.config_sha256,
        dataset=first.dataset,
        condition=first.condition,
        sample_index=first.sample_index,
        sample_id=first.sample_id,
        split_sha256=first.split_sha256,
        checkpoint_sha256=first.checkpoint_sha256,
        source_state_sha256=first.source_state_sha256,
        runtime_sha256=first.runtime_sha256,
        determinism_sha256=first.determinism_sha256,
        input_sha256=first.input_sha256,
        selected_parameter_names_sha256=first.selected_parameter_names_sha256,
        candidate_receipts=ordered,
    )


@dataclass(frozen=True)
class _CandidateObjectClaim:
    model: object
    method: object
    optimizer: object


@dataclass(frozen=True)
class _BackwardObjectClaim:
    autograd_graph: object
    gradient_buffers: tuple[object, ...]


class IndependentCandidateExecutionLedger:
    """Live-object ownership guard proving no candidate reuses a gradient.

    Strong references are retained until :meth:`assert_complete` so Python
    cannot recycle object identities between candidates during a cell.
    """

    def __init__(self) -> None:
        self._candidate_claims: dict[str, _CandidateObjectClaim] = {}
        self._backward_claims: dict[str, _BackwardObjectClaim] = {}
        self._object_owners: dict[int, tuple[str, str]] = {}

    @staticmethod
    def _candidate(candidate: CandidateSpec) -> CandidateSpec:
        if candidate not in FROZEN_CANDIDATES:
            raise D0V2IndependentCandidateError("candidate is outside frozen grid")
        return candidate

    def _claim_object(self, candidate_slug: str, role: str, value: object) -> None:
        identity = id(value)
        previous = self._object_owners.get(identity)
        if previous is not None:
            raise D0V2IndependentCandidateError(
                f"live object reuse is forbidden: role={role}, "
                f"candidate={candidate_slug}, previous_owner={previous}"
            )
        self._object_owners[identity] = (candidate_slug, role)

    def claim_candidate_objects(
        self,
        candidate: CandidateSpec,
        *,
        model: object,
        method: object,
        optimizer: object,
        optimizer_state_entry_count: int,
    ) -> None:
        candidate = self._candidate(candidate)
        slug = candidate.slug
        if slug in self._candidate_claims:
            raise D0V2IndependentCandidateError(
                f"candidate objects already claimed for {slug}"
            )
        if not _is_int(optimizer_state_entry_count) or optimizer_state_entry_count != 0:
            raise D0V2IndependentCandidateError(
                "every fresh candidate optimizer must begin with empty state"
            )
        for role, value in (
            ("model", model),
            ("method", method),
            ("optimizer", optimizer),
        ):
            self._claim_object(slug, role, value)
        self._candidate_claims[slug] = _CandidateObjectClaim(
            model=model, method=method, optimizer=optimizer
        )

    def claim_candidate_backward(
        self,
        candidate: CandidateSpec,
        *,
        autograd_graph: object,
        gradient_buffers: Sequence[object],
    ) -> None:
        candidate = self._candidate(candidate)
        slug = candidate.slug
        if slug not in self._candidate_claims:
            raise D0V2IndependentCandidateError(
                f"candidate objects must be claimed before backward: {slug}"
            )
        if slug in self._backward_claims:
            raise D0V2IndependentCandidateError(
                f"candidate backward already claimed for {slug}"
            )
        if isinstance(gradient_buffers, (str, bytes)) or not isinstance(
            gradient_buffers, Sequence
        ):
            raise D0V2IndependentCandidateError(
                "gradient_buffers must be a non-empty sequence"
            )
        buffers = tuple(gradient_buffers)
        if not buffers:
            raise D0V2IndependentCandidateError(
                "candidate backward must own at least one gradient buffer"
            )
        self._claim_object(slug, "autograd_graph", autograd_graph)
        for index, buffer in enumerate(buffers):
            self._claim_object(slug, f"gradient_buffer[{index}]", buffer)
        self._backward_claims[slug] = _BackwardObjectClaim(
            autograd_graph=autograd_graph,
            gradient_buffers=buffers,
        )

    def assert_complete(self) -> dict[str, Any]:
        expected = {candidate.slug for candidate in FROZEN_CANDIDATES}
        candidate_claims = set(self._candidate_claims)
        backward_claims = set(self._backward_claims)
        if candidate_claims != expected or backward_claims != expected:
            raise D0V2IndependentCandidateError(
                "independent execution ledger is incomplete; "
                f"missing_candidate_objects={sorted(expected - candidate_claims)}, "
                f"missing_backwards={sorted(expected - backward_claims)}"
            )
        return {
            "candidate_count": len(expected),
            "fresh_model_count": len(expected),
            "fresh_method_count": len(expected),
            "fresh_optimizer_count": len(expected),
            "own_autograd_graph_count": len(expected),
            "own_backward_execution_count": len(expected),
            "gradient_reuse": False,
            "engineering_gate_passed": True,
            "scientific_gate_status": SCIENTIFIC_GATE_STATUS,
            "stage2_authorized": False,
        }


__all__ = [
    "AGGREGATE_ARTIFACT_TYPE",
    "CandidateSpec",
    "D0V2IndependentCandidateError",
    "FROZEN_CANDIDATES",
    "IndependentCandidateCellReceipt",
    "IndependentCandidateExecutionLedger",
    "IndependentCandidateReceipt",
    "PROTOCOL_ID",
    "RECEIPT_ARTIFACT_TYPE",
    "SCIENTIFIC_GATE_STATUS",
    "parse_independent_candidate_receipt",
    "validate_independent_candidate_cell",
]
