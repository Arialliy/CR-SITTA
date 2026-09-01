"""Fail-closed three-process reproducibility contract for D0 equivalence.

One D0 equivalence receipt proves that the shared-gradient implementation and
the frozen historical implementation agree *inside one process*.  It does not
by itself prove that CUDA execution is reproducible across fresh processes.
This module closes that gap without importing the GPU runner:

* exactly three canonically encoded single-process receipts are read through
  stable, no-follow filesystem snapshots;
* every receipt is bound to its caller-supplied path, SHA-256 and process ID;
* the supported runner binds all children to one parent-issued 256-bit nonce
  and records a strict launch transcript for each child;
* process IDs, receipt paths and receipt digests must all be unique;
* the frozen dataset/sample, config, source, runtime and determinism contracts
  must be identical;
* all ten frozen candidates must agree for every one of the three process
  pairs on pre/post logits hashes, gradient/delta hashes, changed-tensor count
  and step norm (absolute tolerance ``1e-12``).

The builder returns a canonical aggregate receipt only when every invariant
passes.  There is deliberately no "best effort" or partially-passed output.
The module is CPU-only; the receipts carry the GPU evidence, while aggregation
performs JSON, hash and exact-value checks only.  This is an execution contract
for the supported runner, not a cryptographic attestation against a local
principal that can rewrite the runner and every artifact.  In particular,
simultaneously rewriting an artifact and its local checksum ledger is outside
the stated threat model; no external trust root or signing key is claimed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Final

from tta.d0_secure_io import read_stable_regular_file


SINGLE_PROCESS_ARTIFACT_TYPE: Final = (
    "cr_sitta_tent_shared_gradient_equivalence_v1"
)
AGGREGATE_ARTIFACT_TYPE: Final = (
    "cr_sitta_d0_equivalence_three_fresh_process_reproducibility_v1"
)
LOGITS_TENSOR_SHA256_CONTRACT: Final = (
    "cr_sitta_strided_tensor_dtype_shape_contiguous_bytes_sha256_v1"
)
PROCESS_COUNT: Final = 3
CANDIDATE_COUNT: Final = 10
STEP_NORM_ABS_TOLERANCE: Final = 1.0e-12
SUPPORTED_RUNNER_ATTESTATION_SCOPE: Final = (
    "supported_runner_execution_contract_without_external_trust_root_"
    "local_artifact_and_checksum_corewrite_out_of_scope"
)

_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROCESS_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")

_FROZEN_CANDIDATES: Final = (
    ("Adam", 1.0e-5, "Adam_lr_1em5"),
    ("Adam", 3.0e-5, "Adam_lr_3em5"),
    ("Adam", 1.0e-4, "Adam_lr_1em4"),
    ("Adam", 3.0e-4, "Adam_lr_3em4"),
    ("Adam", 1.0e-3, "Adam_lr_1em3"),
    ("SGD", 1.0e-5, "SGD_lr_1em5"),
    ("SGD", 3.0e-5, "SGD_lr_3em5"),
    ("SGD", 1.0e-4, "SGD_lr_1em4"),
    ("SGD", 3.0e-4, "SGD_lr_3em4"),
    ("SGD", 1.0e-3, "SGD_lr_1em3"),
)
_FROZEN_SLUGS: Final = tuple(value[2] for value in _FROZEN_CANDIDATES)
_FROZEN_SAMPLES: Final = {
    "IRSTD-1K": ("clean_S0", 0, "XDU102"),
    "NUAA-SIRST": ("clean_S0", 0, "Misc_421"),
    "NUDT-SIRST": ("clean_S0", 0, "000891"),
}
_FROZEN_DETERMINISM: Final = {
    "policy": "strict_forwards_temporary_backward_disable",
    "strict_forward": {
        "deterministic_algorithms_enabled": True,
        "warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    },
    "temporary_backward_disable_scopes": [
        "entropy_backward",
        "supervised_task_backward",
    ],
    "restore_strict_policy_before_optimizer_step": True,
    "restore_strict_policy_before_post_forward": True,
    "restore_strict_policy_after_backward_exception": True,
}

_BINDING_KEYS: Final = frozenset(
    {
        "path",
        "sha256",
        "process_id",
        "os_process_id",
        "process_start_time_ticks",
    }
)
_SINGLE_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_version",
        "artifact_type",
        "paper_test_result",
        "source_train_derived",
        "oracle_analysis",
        "fresh_process",
        "parent_run_nonce",
        "child_launch_nonce",
        "command_sha256",
        "process_id",
        "os_process_id",
        "process_start_time_ticks",
        "dataset",
        "condition",
        "image_index",
        "image_id",
        "config_sha256",
        "global_runtime_seal_sha256",
        "diagnostic_source_code_sha256",
        "determinism_contract",
        "logits_tensor_sha256_contract",
        "shared_pre_logits_tensor_sha256",
        "runtime_audits",
        "diagnostic_source_audits",
        "candidate_slugs",
        "candidate_count",
        "method_label_accesses",
        "outer_evaluator_label_accesses",
        "test_images_opened",
        "test_labels_opened",
        "target_payload_deserialized",
        "pre_logits_all_bit_exact",
        "post_logits_all_bit_exact",
        "step_norm_all_within_1e_minus_12",
        "changed_parameter_tensor_counts_all_equal",
        "entropy_gradient_hashes_all_equal",
        "parameter_delta_hashes_all_equal",
        "historical_resets_all_exact_source",
        "cache_zero_test_opens_verified",
        "comparisons",
        "passed",
    }
)
_COMPARISON_KEYS: Final = frozenset(
    {
        "candidate",
        "candidate_slug",
        "pre_logits_bit_exact",
        "post_logits_bit_exact",
        "post_logits_max_abs_difference",
        "shared_post_logits_tensor_sha256",
        "shared_step_norm",
        "historical_step_norm",
        "step_norm_abs_difference",
        "shared_changed_parameter_tensors",
        "historical_changed_parameter_tensors",
        "changed_parameter_tensor_count_equal",
        "shared_entropy_gradient_bundle_sha256",
        "historical_entropy_gradient_bundle_sha256",
        "entropy_gradient_bundle_sha256_equal",
        "shared_parameter_delta_bundle_sha256",
        "historical_parameter_delta_bundle_sha256",
        "parameter_delta_bundle_sha256_equal",
        "historical_reset_exact_source",
    }
)
_RUNTIME_AUDIT_KEYS: Final = frozenset(
    {
        "stage",
        "verified",
        "global_runtime_seal_sha256",
        "bound_file_count",
        "rehashed_file_count",
        "all_bound_file_identity_and_metadata_verified",
        "full_byte_rehash",
        "active_paths_rehashed",
    }
)
_SOURCE_AUDIT_KEYS: Final = frozenset(
    {
        "stage",
        "verified",
        "bound_file_count",
        "all_files_rehashed",
    }
)

_AGGREGATE_KEYS: Final = frozenset(
    {
        "schema_version",
        "artifact_type",
        "passed",
        "paper_test_result",
        "source_train_derived",
        "oracle_analysis",
        "dataset",
        "sample",
        "config_sha256",
        "global_runtime_seal_sha256",
        "diagnostic_source_code_sha256",
        "determinism_contract",
        "logits_tensor_sha256_contract",
        "attestation_scope",
        "parent_run",
        "fresh_process_count",
        "process_ids",
        "process_instances",
        "input_receipts",
        "candidate_count",
        "candidate_slugs",
        "pair_count",
        "pairwise_comparisons",
        "pre_logits_all_pairs_bit_exact",
        "post_logits_all_pairs_bit_exact",
        "entropy_gradient_hashes_all_pairs_exact",
        "parameter_delta_hashes_all_pairs_exact",
        "changed_parameter_tensor_counts_all_pairs_exact",
        "step_norms_all_pairs_within_1e_minus_12",
    }
)
_AGGREGATE_INPUT_KEYS: Final = frozenset(
    {
        "path",
        "sha256",
        "process_id",
        "os_process_id",
        "process_start_time_ticks",
        "fresh_process_verified",
    }
)
_PARENT_RUN_KEYS: Final = frozenset(
    {
        "parent_run_nonce",
        "parent_os_process_id",
        "parent_process_start_time_ticks",
        "launches",
    }
)
_LAUNCH_KEYS: Final = frozenset(
    {
        "process_id",
        "parent_run_nonce",
        "child_launch_nonce",
        "command_sha256",
        "os_process_id",
        "process_start_time_ticks",
        "returncode",
        "stdout_sha256",
        "stderr_sha256",
        "receipt_path",
        "receipt_sha256",
    }
)
_PROCESS_INSTANCE_KEYS: Final = frozenset(
    {"process_id", "os_process_id", "process_start_time_ticks"}
)
_PAIR_KEYS: Final = frozenset(
    {
        "process_ids",
        "process_instances",
        "pre_logits_bit_exact",
        "pre_logits_tensor_sha256",
        "candidate_count",
        "comparisons",
        "post_logits_all_bit_exact",
        "entropy_gradient_hashes_all_exact",
        "parameter_delta_hashes_all_exact",
        "changed_parameter_tensor_counts_all_exact",
        "step_norms_all_within_1e_minus_12",
        "passed",
    }
)
_PAIR_COMPARISON_KEYS: Final = frozenset(
    {
        "candidate_slug",
        "post_logits_bit_exact",
        "post_logits_tensor_sha256",
        "entropy_gradient_bundle_sha256_exact",
        "entropy_gradient_bundle_sha256",
        "parameter_delta_bundle_sha256_exact",
        "parameter_delta_bundle_sha256",
        "changed_parameter_tensor_count_exact",
        "changed_parameter_tensor_count",
        "max_step_norm_abs_difference",
        "step_norm_abs_difference_within_1e_minus_12",
        "passed",
    }
)


class D0EquivalenceReproContractError(ValueError):
    """A receipt or cross-process invariant failed closed."""


@dataclass(frozen=True, slots=True)
class EquivalenceReceiptBinding:
    """Caller-supplied immutable binding for one fresh-process receipt."""

    path: Path
    sha256: str
    process_id: str
    os_process_id: int
    process_start_time_ticks: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EquivalenceReceiptBinding":
        _require_exact_keys(value, _BINDING_KEYS, "receipt binding")
        path = value["path"]
        if not isinstance(path, str) or not path:
            raise D0EquivalenceReproContractError(
                "receipt binding.path must be a non-empty string"
            )
        sha256 = _require_sha256(value["sha256"], "receipt binding.sha256")
        process_id = _require_process_id(
            value["process_id"], "receipt binding.process_id"
        )
        os_process_id = _require_integer(
            value["os_process_id"], "receipt binding.os_process_id", minimum=1
        )
        process_start_time_ticks = _require_integer(
            value["process_start_time_ticks"],
            "receipt binding.process_start_time_ticks",
            minimum=1,
        )
        return cls(
            Path(path),
            sha256,
            process_id,
            os_process_id,
            process_start_time_ticks,
        )

    def normalized(self) -> "EquivalenceReceiptBinding":
        return EquivalenceReceiptBinding(
            Path(os.path.abspath(os.fspath(self.path))),
            _require_sha256(self.sha256, "receipt binding.sha256"),
            _require_process_id(self.process_id, "receipt binding.process_id"),
            _require_integer(
                self.os_process_id, "receipt binding.os_process_id", minimum=1
            ),
            _require_integer(
                self.process_start_time_ticks,
                "receipt binding.process_start_time_ticks",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class _ValidatedSingleReceipt:
    binding: EquivalenceReceiptBinding
    receipt: dict[str, Any]

    @property
    def comparisons(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.receipt["comparisons"])


def _require_exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise D0EquivalenceReproContractError(f"{label} must be a string-key mapping")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise D0EquivalenceReproContractError(
            f"{label} schema is not exact; missing={missing}, unknown={unknown}"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _LOWER_SHA256.fullmatch(value) is None:
        raise D0EquivalenceReproContractError(
            f"{label} must be a lowercase 64-hex SHA-256"
        )
    return value


def _require_process_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _PROCESS_ID.fullmatch(value) is None:
        raise D0EquivalenceReproContractError(f"{label} is not a safe process ID")
    return value


def _require_nonce(value: Any, label: str) -> str:
    """Require one parent-issued 256-bit nonce in canonical lowercase hex."""

    return _require_sha256(value, label)


def _require_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise D0EquivalenceReproContractError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _require_exact_integer(value: Any, expected: int, label: str) -> int:
    observed = _require_integer(value, label)
    if observed != expected:
        raise D0EquivalenceReproContractError(
            f"{label} must equal {expected}"
        )
    return observed


def _require_finite_nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise D0EquivalenceReproContractError(
            f"{label} must be a finite non-negative number"
        )
    return float(value)


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
        raise D0EquivalenceReproContractError(
            "receipt is not canonical-JSON serializable"
        ) from exc


def canonical_d0_equivalence_repro_receipt_bytes(
    receipt: Mapping[str, Any],
) -> bytes:
    """Encode one aggregate receipt canonically, including its final newline."""

    _validate_aggregate_structure(receipt)
    return _canonical_json_bytes(receipt) + b"\n"


def d0_equivalence_repro_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Return the digest of the canonical aggregate receipt bytes."""

    return hashlib.sha256(
        canonical_d0_equivalence_repro_receipt_bytes(receipt)
    ).hexdigest()


def _decode_canonical_json(data: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise D0EquivalenceReproContractError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        text = data.decode("utf-8")
        value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except D0EquivalenceReproContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D0EquivalenceReproContractError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise D0EquivalenceReproContractError(f"{label} root must be an object")
    if data != _canonical_json_bytes(value) + b"\n":
        raise D0EquivalenceReproContractError(
            f"{label} is not canonical JSON with one trailing newline"
        )
    return value


def _validate_source_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise D0EquivalenceReproContractError(
            "diagnostic_source_code_sha256 must be a non-empty mapping"
        )
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not key
            or key.startswith("/")
            or "\\" in key
            or any(part in {"", ".", ".."} for part in key.split("/"))
        ):
            raise D0EquivalenceReproContractError(
                "diagnostic source hash keys must be safe relative POSIX paths"
            )
        result[key] = _require_sha256(
            digest, f"diagnostic_source_code_sha256[{key!r}]"
        )
    return dict(sorted(result.items()))


def _validate_runtime_audits(
    value: Any, *, runtime_seal_sha256: str
) -> None:
    if not isinstance(value, list) or len(value) < 2:
        raise D0EquivalenceReproContractError(
            "runtime_audits must contain at least entry and pre-receipt audits"
        )
    for index, audit_value in enumerate(value):
        audit = _require_exact_keys(
            audit_value, _RUNTIME_AUDIT_KEYS, f"runtime_audits[{index}]"
        )
        if (
            not isinstance(audit["stage"], str)
            or not audit["stage"]
            or audit["verified"] is not True
            or audit["global_runtime_seal_sha256"] != runtime_seal_sha256
            or audit["all_bound_file_identity_and_metadata_verified"] is not True
            or not isinstance(audit["full_byte_rehash"], bool)
        ):
            raise D0EquivalenceReproContractError(
                f"runtime_audits[{index}] did not verify the bound runtime"
            )
        bound = _require_integer(
            audit["bound_file_count"],
            f"runtime_audits[{index}].bound_file_count",
            minimum=1,
        )
        rehashed = _require_integer(
            audit["rehashed_file_count"],
            f"runtime_audits[{index}].rehashed_file_count",
        )
        if rehashed > bound or (
            audit["full_byte_rehash"] is True and rehashed != bound
        ):
            raise D0EquivalenceReproContractError(
                f"runtime_audits[{index}] rehash count is inconsistent"
            )
        active = audit["active_paths_rehashed"]
        if (
            not isinstance(active, list)
            or any(not isinstance(path, str) or not path for path in active)
            or len(active) != len(set(active))
        ):
            raise D0EquivalenceReproContractError(
                f"runtime_audits[{index}].active_paths_rehashed is invalid"
            )
    if value[-1]["full_byte_rehash"] is not True:
        raise D0EquivalenceReproContractError(
            "final runtime audit must perform a full byte rehash"
        )


def _validate_source_audits(value: Any, *, source_file_count: int) -> None:
    if not isinstance(value, list) or len(value) < 2:
        raise D0EquivalenceReproContractError(
            "diagnostic_source_audits must contain at least two audits"
        )
    for index, audit_value in enumerate(value):
        audit = _require_exact_keys(
            audit_value, _SOURCE_AUDIT_KEYS, f"diagnostic_source_audits[{index}]"
        )
        if (
            not isinstance(audit["stage"], str)
            or not audit["stage"]
            or audit["verified"] is not True
            or audit["all_files_rehashed"] is not True
            or audit["bound_file_count"] != source_file_count
        ):
            raise D0EquivalenceReproContractError(
                f"diagnostic_source_audits[{index}] did not verify all source files"
            )


def _validate_determinism(value: Any) -> dict[str, Any]:
    if value != _FROZEN_DETERMINISM:
        raise D0EquivalenceReproContractError(
            "determinism_contract differs from the frozen D0 policy"
        )
    # Equality alone would accept exotic Mapping subclasses.  A canonical
    # JSON round-trip fixes the representation and rejects non-JSON values.
    normalized = json.loads(_canonical_json_bytes(value).decode("utf-8"))
    if normalized != _FROZEN_DETERMINISM:
        raise D0EquivalenceReproContractError(
            "determinism_contract is not canonically representable"
        )
    return normalized


def _validate_candidate_comparison(
    value: Any,
    *,
    index: int,
    expected: tuple[str, float, str],
) -> None:
    comparison = _require_exact_keys(
        value, _COMPARISON_KEYS, f"comparisons[{index}]"
    )
    optimizer, learning_rate, slug = expected
    candidate = comparison["candidate"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "optimizer",
        "learning_rate",
    }:
        raise D0EquivalenceReproContractError(
            f"comparisons[{index}].candidate schema is not exact"
        )
    if (
        candidate["optimizer"] != optimizer
        or isinstance(candidate["learning_rate"], bool)
        or not isinstance(candidate["learning_rate"], (int, float))
        or float(candidate["learning_rate"]) != learning_rate
        or comparison["candidate_slug"] != slug
    ):
        raise D0EquivalenceReproContractError(
            f"comparisons[{index}] differs from frozen candidate {slug}"
        )

    shared_step = _require_finite_nonnegative(
        comparison["shared_step_norm"], f"comparisons[{index}].shared_step_norm"
    )
    historical_step = _require_finite_nonnegative(
        comparison["historical_step_norm"],
        f"comparisons[{index}].historical_step_norm",
    )
    reported_difference = _require_finite_nonnegative(
        comparison["step_norm_abs_difference"],
        f"comparisons[{index}].step_norm_abs_difference",
    )
    if reported_difference != abs(shared_step - historical_step):
        raise D0EquivalenceReproContractError(
            f"comparisons[{index}] step norm difference does not recompute"
        )
    if reported_difference > STEP_NORM_ABS_TOLERANCE:
        raise D0EquivalenceReproContractError(
            f"comparisons[{index}] step norm exceeds the 1e-12 tolerance"
        )

    shared_changed = _require_integer(
        comparison["shared_changed_parameter_tensors"],
        f"comparisons[{index}].shared_changed_parameter_tensors",
    )
    historical_changed = _require_integer(
        comparison["historical_changed_parameter_tensors"],
        f"comparisons[{index}].historical_changed_parameter_tensors",
    )
    hash_fields = (
        "shared_post_logits_tensor_sha256",
        "shared_entropy_gradient_bundle_sha256",
        "historical_entropy_gradient_bundle_sha256",
        "shared_parameter_delta_bundle_sha256",
        "historical_parameter_delta_bundle_sha256",
    )
    for field in hash_fields:
        _require_sha256(comparison[field], f"comparisons[{index}].{field}")

    if (
        comparison["pre_logits_bit_exact"] is not True
        or comparison["post_logits_bit_exact"] is not True
        or _require_finite_nonnegative(
            comparison["post_logits_max_abs_difference"],
            f"comparisons[{index}].post_logits_max_abs_difference",
        )
        != 0.0
        or shared_changed != historical_changed
        or comparison["changed_parameter_tensor_count_equal"] is not True
        or comparison["shared_entropy_gradient_bundle_sha256"]
        != comparison["historical_entropy_gradient_bundle_sha256"]
        or comparison["entropy_gradient_bundle_sha256_equal"] is not True
        or comparison["shared_parameter_delta_bundle_sha256"]
        != comparison["historical_parameter_delta_bundle_sha256"]
        or comparison["parameter_delta_bundle_sha256_equal"] is not True
        or comparison["historical_reset_exact_source"] is not True
    ):
        raise D0EquivalenceReproContractError(
            f"comparisons[{index}] did not pass its single-process exact gates"
        )


def _validate_single_receipt(
    receipt: dict[str, Any], binding: EquivalenceReceiptBinding
) -> None:
    _require_exact_keys(receipt, _SINGLE_RECEIPT_KEYS, "single-process receipt")
    _require_exact_integer(receipt["schema_version"], 1, "schema_version")
    _require_exact_integer(receipt["image_index"], 0, "image_index")
    for field in (
        "method_label_accesses",
        "outer_evaluator_label_accesses",
        "test_images_opened",
        "test_labels_opened",
    ):
        _require_exact_integer(receipt[field], 0, field)
    _require_exact_integer(
        receipt["candidate_count"], CANDIDATE_COUNT, "candidate_count"
    )
    if (
        receipt["artifact_type"] != SINGLE_PROCESS_ARTIFACT_TYPE
        or receipt["passed"] is not True
        or receipt["paper_test_result"] is not False
        or receipt["source_train_derived"] is not True
        or receipt["oracle_analysis"] is not False
        or receipt["fresh_process"] is not True
        or receipt["process_id"] != binding.process_id
        or receipt["os_process_id"] != binding.os_process_id
        or receipt["process_start_time_ticks"]
        != binding.process_start_time_ticks
        or receipt["target_payload_deserialized"] is not False
        or receipt["pre_logits_all_bit_exact"] is not True
        or receipt["post_logits_all_bit_exact"] is not True
        or receipt["step_norm_all_within_1e_minus_12"] is not True
        or receipt["changed_parameter_tensor_counts_all_equal"] is not True
        or receipt["entropy_gradient_hashes_all_equal"] is not True
        or receipt["parameter_delta_hashes_all_equal"] is not True
        or receipt["historical_resets_all_exact_source"] is not True
        or receipt["cache_zero_test_opens_verified"] is not True
    ):
        raise D0EquivalenceReproContractError(
            "single-process receipt did not pass its exact protocol gates"
        )
    _require_integer(receipt["os_process_id"], "os_process_id", minimum=1)
    _require_integer(
        receipt["process_start_time_ticks"],
        "process_start_time_ticks",
        minimum=1,
    )
    _require_nonce(receipt["parent_run_nonce"], "parent_run_nonce")
    _require_nonce(receipt["child_launch_nonce"], "child_launch_nonce")
    _require_sha256(receipt["command_sha256"], "command_sha256")

    dataset = receipt["dataset"]
    if dataset not in _FROZEN_SAMPLES:
        raise D0EquivalenceReproContractError("receipt dataset is not frozen for D0")
    if (
        receipt["condition"],
        receipt["image_index"],
        receipt["image_id"],
    ) != _FROZEN_SAMPLES[dataset]:
        raise D0EquivalenceReproContractError(
            "receipt sample differs from the frozen dataset sample"
        )
    _require_sha256(receipt["config_sha256"], "config_sha256")
    runtime_sha = _require_sha256(
        receipt["global_runtime_seal_sha256"], "global_runtime_seal_sha256"
    )
    source_hashes = _validate_source_hashes(
        receipt["diagnostic_source_code_sha256"]
    )
    _validate_determinism(receipt["determinism_contract"])
    if receipt["logits_tensor_sha256_contract"] != LOGITS_TENSOR_SHA256_CONTRACT:
        raise D0EquivalenceReproContractError(
            "logits_tensor_sha256_contract differs from the frozen hash contract"
        )
    _require_sha256(
        receipt["shared_pre_logits_tensor_sha256"],
        "shared_pre_logits_tensor_sha256",
    )
    _validate_runtime_audits(
        receipt["runtime_audits"], runtime_seal_sha256=runtime_sha
    )
    _validate_source_audits(
        receipt["diagnostic_source_audits"],
        source_file_count=len(source_hashes),
    )
    if (
        receipt["candidate_slugs"] != list(_FROZEN_SLUGS)
        or not isinstance(receipt["comparisons"], list)
        or len(receipt["comparisons"]) != CANDIDATE_COUNT
    ):
        raise D0EquivalenceReproContractError(
            "single-process receipt does not contain the ten frozen candidates"
        )
    for index, (comparison, candidate) in enumerate(
        zip(receipt["comparisons"], _FROZEN_CANDIDATES, strict=True)
    ):
        _validate_candidate_comparison(
            comparison, index=index, expected=candidate
        )


def _coerce_binding(
    value: EquivalenceReceiptBinding | Mapping[str, Any], index: int
) -> EquivalenceReceiptBinding:
    if isinstance(value, EquivalenceReceiptBinding):
        binding = value
    elif isinstance(value, Mapping):
        binding = EquivalenceReceiptBinding.from_mapping(value)
    else:
        raise D0EquivalenceReproContractError(
            f"receipt binding[{index}] has an unsupported type"
        )
    if not isinstance(binding.path, Path):
        raise D0EquivalenceReproContractError(
            f"receipt binding[{index}].path must be pathlib.Path"
        )
    return binding.normalized()


def _load_single_receipt(
    binding: EquivalenceReceiptBinding,
) -> _ValidatedSingleReceipt:
    try:
        snapshot = read_stable_regular_file(binding.path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise D0EquivalenceReproContractError(
            f"cannot stably read receipt: {binding.path}"
        ) from exc
    if snapshot.sha256 != binding.sha256:
        raise D0EquivalenceReproContractError(
            f"receipt SHA-256 binding mismatch: {binding.path}"
        )
    receipt = _decode_canonical_json(
        snapshot.data, f"single-process receipt {binding.path}"
    )
    _validate_single_receipt(receipt, binding)
    return _ValidatedSingleReceipt(binding=binding, receipt=receipt)


def _assert_same_receipt_identity(
    receipts: Sequence[_ValidatedSingleReceipt],
) -> None:
    fields = (
        "parent_run_nonce",
        "dataset",
        "condition",
        "image_index",
        "image_id",
        "config_sha256",
        "global_runtime_seal_sha256",
        "diagnostic_source_code_sha256",
        "determinism_contract",
        "logits_tensor_sha256_contract",
        "candidate_slugs",
        "candidate_count",
    )
    reference = receipts[0].receipt
    for item in receipts[1:]:
        for field in fields:
            if item.receipt[field] != reference[field]:
                raise D0EquivalenceReproContractError(
                    f"fresh-process receipts disagree on {field}"
                )


def _normalize_parent_run(
    value: Any,
    *,
    bindings: Sequence[EquivalenceReceiptBinding],
    receipts: Sequence[_ValidatedSingleReceipt] | None = None,
) -> dict[str, Any]:
    """Validate the supported runner's parent-observed launch transcript."""

    parent = _require_exact_keys(value, _PARENT_RUN_KEYS, "parent_run")
    parent_nonce = _require_nonce(
        parent["parent_run_nonce"], "parent_run.parent_run_nonce"
    )
    parent_pid = _require_integer(
        parent["parent_os_process_id"],
        "parent_run.parent_os_process_id",
        minimum=1,
    )
    parent_ticks = _require_integer(
        parent["parent_process_start_time_ticks"],
        "parent_run.parent_process_start_time_ticks",
        minimum=1,
    )
    launches_value = parent["launches"]
    if not isinstance(launches_value, list) or len(launches_value) != PROCESS_COUNT:
        raise D0EquivalenceReproContractError(
            "parent_run.launches must contain exactly three child transcripts"
        )
    binding_by_id = {binding.process_id: binding for binding in bindings}
    receipt_by_id = (
        None
        if receipts is None
        else {receipt.binding.process_id: receipt.receipt for receipt in receipts}
    )
    normalized: list[dict[str, Any]] = []
    child_nonces: set[str] = set()
    command_hashes: set[str] = set()
    child_instances: set[tuple[int, int]] = set()
    for index, launch_value in enumerate(launches_value):
        launch = _require_exact_keys(
            launch_value, _LAUNCH_KEYS, f"parent_run.launches[{index}]"
        )
        process_id = _require_process_id(
            launch["process_id"], f"parent_run.launches[{index}].process_id"
        )
        if process_id not in binding_by_id:
            raise D0EquivalenceReproContractError(
                f"parent launch is not bound to an input receipt: {process_id}"
            )
        binding = binding_by_id[process_id]
        launch_parent_nonce = _require_nonce(
            launch["parent_run_nonce"],
            f"parent_run.launches[{index}].parent_run_nonce",
        )
        child_nonce = _require_nonce(
            launch["child_launch_nonce"],
            f"parent_run.launches[{index}].child_launch_nonce",
        )
        command_sha = _require_sha256(
            launch["command_sha256"],
            f"parent_run.launches[{index}].command_sha256",
        )
        os_process_id = _require_integer(
            launch["os_process_id"],
            f"parent_run.launches[{index}].os_process_id",
            minimum=1,
        )
        start_ticks = _require_integer(
            launch["process_start_time_ticks"],
            f"parent_run.launches[{index}].process_start_time_ticks",
            minimum=1,
        )
        returncode = _require_integer(
            launch["returncode"],
            f"parent_run.launches[{index}].returncode",
        )
        stdout_sha = _require_sha256(
            launch["stdout_sha256"],
            f"parent_run.launches[{index}].stdout_sha256",
        )
        stderr_sha = _require_sha256(
            launch["stderr_sha256"],
            f"parent_run.launches[{index}].stderr_sha256",
        )
        receipt_sha = _require_sha256(
            launch["receipt_sha256"],
            f"parent_run.launches[{index}].receipt_sha256",
        )
        receipt_path = launch["receipt_path"]
        if not isinstance(receipt_path, str) or not receipt_path:
            raise D0EquivalenceReproContractError(
                f"parent_run.launches[{index}].receipt_path is invalid"
            )
        absolute_receipt_path = os.path.abspath(receipt_path)
        if receipt_path != absolute_receipt_path:
            raise D0EquivalenceReproContractError(
                f"parent_run.launches[{index}].receipt_path is not canonical"
            )
        if (
            launch_parent_nonce != parent_nonce
            or returncode != 0
            or os_process_id != binding.os_process_id
            or start_ticks != binding.process_start_time_ticks
            or receipt_path != str(binding.path)
            or receipt_sha != binding.sha256
        ):
            raise D0EquivalenceReproContractError(
                f"parent launch transcript differs from binding: {process_id}"
            )
        if receipt_by_id is not None:
            receipt = receipt_by_id[process_id]
            if (
                receipt["parent_run_nonce"] != parent_nonce
                or receipt["child_launch_nonce"] != child_nonce
                or receipt["command_sha256"] != command_sha
                or receipt["os_process_id"] != os_process_id
                or receipt["process_start_time_ticks"] != start_ticks
            ):
                raise D0EquivalenceReproContractError(
                    f"worker receipt differs from parent launch transcript: {process_id}"
                )
        child_nonces.add(child_nonce)
        command_hashes.add(command_sha)
        child_instances.add((os_process_id, start_ticks))
        normalized.append(
            {
                "process_id": process_id,
                "parent_run_nonce": parent_nonce,
                "child_launch_nonce": child_nonce,
                "command_sha256": command_sha,
                "os_process_id": os_process_id,
                "process_start_time_ticks": start_ticks,
                "returncode": 0,
                "stdout_sha256": stdout_sha,
                "stderr_sha256": stderr_sha,
                "receipt_path": receipt_path,
                "receipt_sha256": receipt_sha,
            }
        )
    normalized.sort(key=lambda item: item["process_id"])
    if [item["process_id"] for item in normalized] != sorted(binding_by_id):
        raise D0EquivalenceReproContractError(
            "parent launch transcript does not cover the three receipt bindings"
        )
    if (
        len(child_nonces) != PROCESS_COUNT
        or len(command_hashes) != PROCESS_COUNT
        or len(child_instances) != PROCESS_COUNT
        or (parent_pid, parent_ticks) in child_instances
    ):
        raise D0EquivalenceReproContractError(
            "parent/child nonce, command, or process-instance transcript is not unique"
        )
    return {
        "parent_run_nonce": parent_nonce,
        "parent_os_process_id": parent_pid,
        "parent_process_start_time_ticks": parent_ticks,
        "launches": normalized,
    }


def _build_pair_receipt(
    left: _ValidatedSingleReceipt,
    right: _ValidatedSingleReceipt,
) -> dict[str, Any]:
    left_receipt = left.receipt
    right_receipt = right.receipt
    if (
        left_receipt["shared_pre_logits_tensor_sha256"]
        != right_receipt["shared_pre_logits_tensor_sha256"]
    ):
        raise D0EquivalenceReproContractError(
            "fresh-process pair has non-bit-exact pre logits"
        )

    comparisons: list[dict[str, Any]] = []
    for index, (left_value, right_value) in enumerate(
        zip(left.comparisons, right.comparisons, strict=True)
    ):
        slug = _FROZEN_SLUGS[index]
        post_hashes = {
            left_value["shared_post_logits_tensor_sha256"],
            right_value["shared_post_logits_tensor_sha256"],
        }
        gradient_hashes = {
            left_value["shared_entropy_gradient_bundle_sha256"],
            left_value["historical_entropy_gradient_bundle_sha256"],
            right_value["shared_entropy_gradient_bundle_sha256"],
            right_value["historical_entropy_gradient_bundle_sha256"],
        }
        delta_hashes = {
            left_value["shared_parameter_delta_bundle_sha256"],
            left_value["historical_parameter_delta_bundle_sha256"],
            right_value["shared_parameter_delta_bundle_sha256"],
            right_value["historical_parameter_delta_bundle_sha256"],
        }
        changed_counts = {
            left_value["shared_changed_parameter_tensors"],
            left_value["historical_changed_parameter_tensors"],
            right_value["shared_changed_parameter_tensors"],
            right_value["historical_changed_parameter_tensors"],
        }
        step_norms = tuple(
            float(value)
            for value in (
                left_value["shared_step_norm"],
                left_value["historical_step_norm"],
                right_value["shared_step_norm"],
                right_value["historical_step_norm"],
            )
        )
        maximum_step_difference = max(step_norms) - min(step_norms)
        if len(post_hashes) != 1:
            raise D0EquivalenceReproContractError(
                f"fresh-process pair has non-bit-exact post logits: {slug}"
            )
        if len(gradient_hashes) != 1:
            raise D0EquivalenceReproContractError(
                f"fresh-process pair has different entropy gradients: {slug}"
            )
        if len(delta_hashes) != 1:
            raise D0EquivalenceReproContractError(
                f"fresh-process pair has different parameter deltas: {slug}"
            )
        if len(changed_counts) != 1:
            raise D0EquivalenceReproContractError(
                f"fresh-process pair has different changed tensor counts: {slug}"
            )
        if maximum_step_difference > STEP_NORM_ABS_TOLERANCE:
            raise D0EquivalenceReproContractError(
                f"fresh-process pair step norm differs by more than 1e-12: {slug}"
            )
        comparisons.append(
            {
                "candidate_slug": slug,
                "post_logits_bit_exact": True,
                "post_logits_tensor_sha256": next(iter(post_hashes)),
                "entropy_gradient_bundle_sha256_exact": True,
                "entropy_gradient_bundle_sha256": next(iter(gradient_hashes)),
                "parameter_delta_bundle_sha256_exact": True,
                "parameter_delta_bundle_sha256": next(iter(delta_hashes)),
                "changed_parameter_tensor_count_exact": True,
                "changed_parameter_tensor_count": next(iter(changed_counts)),
                "max_step_norm_abs_difference": maximum_step_difference,
                "step_norm_abs_difference_within_1e_minus_12": True,
                "passed": True,
            }
        )
    return {
        "process_ids": [left.binding.process_id, right.binding.process_id],
        "process_instances": [
            {
                "process_id": value.binding.process_id,
                "os_process_id": value.binding.os_process_id,
                "process_start_time_ticks": (
                    value.binding.process_start_time_ticks
                ),
            }
            for value in (left, right)
        ],
        "pre_logits_bit_exact": True,
        "pre_logits_tensor_sha256": left_receipt[
            "shared_pre_logits_tensor_sha256"
        ],
        "candidate_count": CANDIDATE_COUNT,
        "comparisons": comparisons,
        "post_logits_all_bit_exact": True,
        "entropy_gradient_hashes_all_exact": True,
        "parameter_delta_hashes_all_exact": True,
        "changed_parameter_tensor_counts_all_exact": True,
        "step_norms_all_within_1e_minus_12": True,
        "passed": True,
    }


def build_d0_equivalence_repro_receipt(
    bindings: Sequence[EquivalenceReceiptBinding | Mapping[str, Any]],
    *,
    parent_run: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate three fresh-process receipts and return one canonical aggregate.

    The returned mapping is ordered deterministically by ``process_id``.  Use
    :func:`canonical_d0_equivalence_repro_receipt_bytes` for durable bytes and
    :func:`d0_equivalence_repro_receipt_sha256` for its content digest.
    """

    if isinstance(bindings, (str, bytes, bytearray, Mapping)) or len(bindings) != 3:
        raise D0EquivalenceReproContractError(
            "D0 equivalence reproducibility requires exactly three bindings"
        )
    normalized = tuple(
        _coerce_binding(value, index) for index, value in enumerate(bindings)
    )
    process_ids = [value.process_id for value in normalized]
    paths = [str(value.path) for value in normalized]
    digests = [value.sha256 for value in normalized]
    process_instances = [
        (value.os_process_id, value.process_start_time_ticks)
        for value in normalized
    ]
    if len(set(process_ids)) != PROCESS_COUNT:
        raise D0EquivalenceReproContractError(
            "D0 equivalence requires three unique fresh process IDs"
        )
    if len(set(paths)) != PROCESS_COUNT:
        raise D0EquivalenceReproContractError(
            "D0 equivalence requires three unique receipt paths"
        )
    if len(set(digests)) != PROCESS_COUNT:
        raise D0EquivalenceReproContractError(
            "D0 equivalence requires three unique receipt SHA-256 bindings"
        )
    if len(set(process_instances)) != PROCESS_COUNT:
        raise D0EquivalenceReproContractError(
            "D0 equivalence requires three unique Linux process instances "
            "(os_process_id, process_start_time_ticks)"
        )

    receipts = tuple(
        sorted(
            (_load_single_receipt(value) for value in normalized),
            key=lambda value: value.binding.process_id,
        )
    )
    _assert_same_receipt_identity(receipts)
    normalized_parent_run = _normalize_parent_run(
        parent_run, bindings=normalized, receipts=receipts
    )
    pairs = [
        _build_pair_receipt(left, right)
        for left, right in combinations(receipts, 2)
    ]
    reference = receipts[0].receipt
    aggregate = {
        "schema_version": 1,
        "artifact_type": AGGREGATE_ARTIFACT_TYPE,
        "passed": True,
        "paper_test_result": False,
        "source_train_derived": True,
        "oracle_analysis": False,
        "dataset": reference["dataset"],
        "sample": {
            "condition": reference["condition"],
            "image_index": reference["image_index"],
            "image_id": reference["image_id"],
        },
        "config_sha256": reference["config_sha256"],
        "global_runtime_seal_sha256": reference[
            "global_runtime_seal_sha256"
        ],
        "diagnostic_source_code_sha256": dict(
            sorted(reference["diagnostic_source_code_sha256"].items())
        ),
        "determinism_contract": reference["determinism_contract"],
        "logits_tensor_sha256_contract": LOGITS_TENSOR_SHA256_CONTRACT,
        "attestation_scope": SUPPORTED_RUNNER_ATTESTATION_SCOPE,
        "parent_run": normalized_parent_run,
        "fresh_process_count": PROCESS_COUNT,
        "process_ids": [value.binding.process_id for value in receipts],
        "process_instances": [
            {
                "process_id": value.binding.process_id,
                "os_process_id": value.binding.os_process_id,
                "process_start_time_ticks": (
                    value.binding.process_start_time_ticks
                ),
            }
            for value in receipts
        ],
        "input_receipts": [
            {
                "path": str(value.binding.path),
                "sha256": value.binding.sha256,
                "process_id": value.binding.process_id,
                "os_process_id": value.binding.os_process_id,
                "process_start_time_ticks": (
                    value.binding.process_start_time_ticks
                ),
                "fresh_process_verified": True,
            }
            for value in receipts
        ],
        "candidate_count": CANDIDATE_COUNT,
        "candidate_slugs": list(_FROZEN_SLUGS),
        "pair_count": 3,
        "pairwise_comparisons": pairs,
        "pre_logits_all_pairs_bit_exact": True,
        "post_logits_all_pairs_bit_exact": True,
        "entropy_gradient_hashes_all_pairs_exact": True,
        "parameter_delta_hashes_all_pairs_exact": True,
        "changed_parameter_tensor_counts_all_pairs_exact": True,
        "step_norms_all_pairs_within_1e_minus_12": True,
    }
    _validate_aggregate_structure(aggregate)
    return aggregate


def _validate_aggregate_structure(receipt: Mapping[str, Any]) -> None:
    aggregate = _require_exact_keys(
        receipt, _AGGREGATE_KEYS, "aggregate reproducibility receipt"
    )
    _require_exact_integer(aggregate["schema_version"], 1, "aggregate.schema_version")
    _require_exact_integer(
        aggregate["fresh_process_count"],
        PROCESS_COUNT,
        "aggregate.fresh_process_count",
    )
    _require_exact_integer(
        aggregate["candidate_count"],
        CANDIDATE_COUNT,
        "aggregate.candidate_count",
    )
    _require_exact_integer(aggregate["pair_count"], 3, "aggregate.pair_count")
    if (
        aggregate["artifact_type"] != AGGREGATE_ARTIFACT_TYPE
        or aggregate["passed"] is not True
        or aggregate["paper_test_result"] is not False
        or aggregate["source_train_derived"] is not True
        or aggregate["oracle_analysis"] is not False
        or aggregate["candidate_slugs"] != list(_FROZEN_SLUGS)
        or aggregate["pre_logits_all_pairs_bit_exact"] is not True
        or aggregate["post_logits_all_pairs_bit_exact"] is not True
        or aggregate["entropy_gradient_hashes_all_pairs_exact"] is not True
        or aggregate["parameter_delta_hashes_all_pairs_exact"] is not True
        or aggregate["changed_parameter_tensor_counts_all_pairs_exact"] is not True
        or aggregate["step_norms_all_pairs_within_1e_minus_12"] is not True
        or aggregate["logits_tensor_sha256_contract"]
        != LOGITS_TENSOR_SHA256_CONTRACT
        or aggregate["attestation_scope"] != SUPPORTED_RUNNER_ATTESTATION_SCOPE
    ):
        raise D0EquivalenceReproContractError(
            "aggregate reproducibility receipt did not pass exact gates"
        )
    dataset = aggregate["dataset"]
    sample = aggregate["sample"]
    if dataset not in _FROZEN_SAMPLES or not isinstance(sample, Mapping) or set(
        sample
    ) != {"condition", "image_index", "image_id"}:
        raise D0EquivalenceReproContractError("aggregate sample schema is invalid")
    if (
        sample["condition"],
        sample["image_index"],
        sample["image_id"],
    ) != _FROZEN_SAMPLES[dataset]:
        raise D0EquivalenceReproContractError("aggregate sample is not frozen")
    _require_exact_integer(
        sample["image_index"], 0, "aggregate.sample.image_index"
    )
    _require_sha256(aggregate["config_sha256"], "aggregate config_sha256")
    _require_sha256(
        aggregate["global_runtime_seal_sha256"],
        "aggregate global_runtime_seal_sha256",
    )
    _validate_source_hashes(aggregate["diagnostic_source_code_sha256"])
    _validate_determinism(aggregate["determinism_contract"])

    process_ids = aggregate["process_ids"]
    process_instances = aggregate["process_instances"]
    inputs = aggregate["input_receipts"]
    if (
        not isinstance(process_ids, list)
        or process_ids != sorted(process_ids)
        or len(process_ids) != PROCESS_COUNT
        or len(set(process_ids)) != PROCESS_COUNT
        or not isinstance(process_instances, list)
        or len(process_instances) != PROCESS_COUNT
        or not isinstance(inputs, list)
        or len(inputs) != PROCESS_COUNT
    ):
        raise D0EquivalenceReproContractError(
            "aggregate fresh-process bindings are not canonical"
        )
    observed_paths: set[str] = set()
    observed_hashes: set[str] = set()
    observed_process_instances: set[tuple[int, int]] = set()
    for index, (process_id, instance_value, input_value) in enumerate(
        zip(process_ids, process_instances, inputs, strict=True)
    ):
        instance = _require_exact_keys(
            instance_value,
            _PROCESS_INSTANCE_KEYS,
            f"process_instances[{index}]",
        )
        input_receipt = _require_exact_keys(
            input_value, _AGGREGATE_INPUT_KEYS, f"input_receipts[{index}]"
        )
        os_process_id = _require_integer(
            input_receipt["os_process_id"],
            f"input_receipts[{index}].os_process_id",
            minimum=1,
        )
        start_ticks = _require_integer(
            input_receipt["process_start_time_ticks"],
            f"input_receipts[{index}].process_start_time_ticks",
            minimum=1,
        )
        if (
            input_receipt["process_id"] != process_id
            or instance["process_id"] != process_id
            or instance["os_process_id"] != os_process_id
            or instance["process_start_time_ticks"] != start_ticks
            or input_receipt["fresh_process_verified"] is not True
            or not isinstance(input_receipt["path"], str)
            or not input_receipt["path"]
            or not os.path.isabs(input_receipt["path"])
            or os.path.abspath(input_receipt["path"])
            != input_receipt["path"]
        ):
            raise D0EquivalenceReproContractError(
                f"input_receipts[{index}] binding is invalid"
            )
        _require_process_id(process_id, f"process_ids[{index}]")
        _require_sha256(
            input_receipt["sha256"], f"input_receipts[{index}].sha256"
        )
        observed_paths.add(input_receipt["path"])
        observed_hashes.add(input_receipt["sha256"])
        observed_process_instances.add((os_process_id, start_ticks))
    if (
        len(observed_paths) != PROCESS_COUNT
        or len(observed_hashes) != PROCESS_COUNT
        or len(observed_process_instances) != PROCESS_COUNT
    ):
        raise D0EquivalenceReproContractError(
            "aggregate input paths/digests/Linux process instances are not unique"
        )
    aggregate_bindings = tuple(
        EquivalenceReceiptBinding(
            path=Path(value["path"]),
            sha256=value["sha256"],
            process_id=value["process_id"],
            os_process_id=value["os_process_id"],
            process_start_time_ticks=value["process_start_time_ticks"],
        ).normalized()
        for value in inputs
    )
    normalized_parent = _normalize_parent_run(
        aggregate["parent_run"], bindings=aggregate_bindings
    )
    if normalized_parent != aggregate["parent_run"]:
        raise D0EquivalenceReproContractError(
            "aggregate parent launch transcript is not canonical"
        )

    pairs = aggregate["pairwise_comparisons"]
    expected_pairs = [list(value) for value in combinations(process_ids, 2)]
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise D0EquivalenceReproContractError(
            "aggregate must contain exactly three process pairs"
        )
    observed_pre_hashes: set[str] = set()
    observed_candidate_evidence: list[dict[str, set[Any]]] = [
        {
            "post": set(),
            "gradient": set(),
            "delta": set(),
            "changed": set(),
        }
        for _ in _FROZEN_SLUGS
    ]
    for pair_index, (pair_value, expected_process_ids) in enumerate(
        zip(pairs, expected_pairs, strict=True)
    ):
        pair = _require_exact_keys(
            pair_value, _PAIR_KEYS, f"pairwise_comparisons[{pair_index}]"
        )
        _require_exact_integer(
            pair["candidate_count"],
            CANDIDATE_COUNT,
            f"pairwise_comparisons[{pair_index}].candidate_count",
        )
        if (
            pair["process_ids"] != expected_process_ids
            or pair["process_instances"]
            != [
                process_instances[process_ids.index(process_id)]
                for process_id in expected_process_ids
            ]
            or pair["pre_logits_bit_exact"] is not True
            or pair["post_logits_all_bit_exact"] is not True
            or pair["entropy_gradient_hashes_all_exact"] is not True
            or pair["parameter_delta_hashes_all_exact"] is not True
            or pair["changed_parameter_tensor_counts_all_exact"] is not True
            or pair["step_norms_all_within_1e_minus_12"] is not True
            or pair["passed"] is not True
        ):
            raise D0EquivalenceReproContractError(
                f"pairwise_comparisons[{pair_index}] did not pass exact gates"
            )
        _require_sha256(
            pair["pre_logits_tensor_sha256"],
            f"pairwise_comparisons[{pair_index}].pre_logits_tensor_sha256",
        )
        observed_pre_hashes.add(pair["pre_logits_tensor_sha256"])
        comparisons_value = pair["comparisons"]
        if (
            not isinstance(comparisons_value, list)
            or len(comparisons_value) != CANDIDATE_COUNT
        ):
            raise D0EquivalenceReproContractError(
                f"pairwise_comparisons[{pair_index}] lacks ten candidates"
            )
        for candidate_index, (candidate_value, slug) in enumerate(
            zip(comparisons_value, _FROZEN_SLUGS, strict=True)
        ):
            candidate = _require_exact_keys(
                candidate_value,
                _PAIR_COMPARISON_KEYS,
                f"pairwise_comparisons[{pair_index}].comparisons[{candidate_index}]",
            )
            if (
                candidate["candidate_slug"] != slug
                or candidate["post_logits_bit_exact"] is not True
                or candidate["entropy_gradient_bundle_sha256_exact"] is not True
                or candidate["parameter_delta_bundle_sha256_exact"] is not True
                or candidate["changed_parameter_tensor_count_exact"] is not True
                or candidate["step_norm_abs_difference_within_1e_minus_12"]
                is not True
                or candidate["passed"] is not True
            ):
                raise D0EquivalenceReproContractError(
                    f"aggregate pair candidate did not pass: {slug}"
                )
            for field in (
                "post_logits_tensor_sha256",
                "entropy_gradient_bundle_sha256",
                "parameter_delta_bundle_sha256",
            ):
                _require_sha256(
                    candidate[field],
                    f"aggregate pair candidate {slug}.{field}",
                )
            _require_integer(
                candidate["changed_parameter_tensor_count"],
                f"aggregate pair candidate {slug}.changed_parameter_tensor_count",
            )
            evidence = observed_candidate_evidence[candidate_index]
            evidence["post"].add(candidate["post_logits_tensor_sha256"])
            evidence["gradient"].add(
                candidate["entropy_gradient_bundle_sha256"]
            )
            evidence["delta"].add(candidate["parameter_delta_bundle_sha256"])
            evidence["changed"].add(candidate["changed_parameter_tensor_count"])
            if _require_finite_nonnegative(
                candidate["max_step_norm_abs_difference"],
                f"aggregate pair candidate {slug}.max_step_norm_abs_difference",
            ) > STEP_NORM_ABS_TOLERANCE:
                raise D0EquivalenceReproContractError(
                    f"aggregate pair candidate step norm exceeds tolerance: {slug}"
                )
    if len(observed_pre_hashes) != 1:
        raise D0EquivalenceReproContractError(
            "aggregate process pairs disagree on the pre-logits hash"
        )
    for slug, evidence in zip(
        _FROZEN_SLUGS, observed_candidate_evidence, strict=True
    ):
        if any(len(values) != 1 for values in evidence.values()):
            raise D0EquivalenceReproContractError(
                f"aggregate process pairs disagree on candidate evidence: {slug}"
            )


def validate_d0_equivalence_repro_receipt(
    receipt: Mapping[str, Any], *, revalidate_inputs: bool = True
) -> str:
    """Validate an aggregate and optionally rebuild it from its bound inputs.

    Returns the canonical aggregate SHA-256.  With the default
    ``revalidate_inputs=True``, all three input files are read again and the
    aggregate must equal a freshly rebuilt receipt byte-for-byte in value.
    """

    _validate_aggregate_structure(receipt)
    if revalidate_inputs:
        bindings = [
            EquivalenceReceiptBinding(
                path=Path(value["path"]),
                sha256=value["sha256"],
                process_id=value["process_id"],
                os_process_id=value["os_process_id"],
                process_start_time_ticks=value["process_start_time_ticks"],
            )
            for value in receipt["input_receipts"]
        ]
        rebuilt = build_d0_equivalence_repro_receipt(
            bindings, parent_run=receipt["parent_run"]
        )
        if rebuilt != receipt:
            raise D0EquivalenceReproContractError(
                "aggregate receipt does not recompute from its bound inputs"
            )
    return d0_equivalence_repro_receipt_sha256(receipt)


__all__ = [
    "AGGREGATE_ARTIFACT_TYPE",
    "CANDIDATE_COUNT",
    "D0EquivalenceReproContractError",
    "EquivalenceReceiptBinding",
    "LOGITS_TENSOR_SHA256_CONTRACT",
    "PROCESS_COUNT",
    "SINGLE_PROCESS_ARTIFACT_TYPE",
    "STEP_NORM_ABS_TOLERANCE",
    "SUPPORTED_RUNNER_ATTESTATION_SCOPE",
    "build_d0_equivalence_repro_receipt",
    "canonical_d0_equivalence_repro_receipt_bytes",
    "d0_equivalence_repro_receipt_sha256",
    "validate_d0_equivalence_repro_receipt",
]
