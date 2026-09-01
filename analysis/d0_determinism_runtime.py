"""Runtime enforcement and audit for the frozen Phase-D0 determinism policy.

The D0 protocol requires strict deterministic forward passes, while allowing
only two named CUDA backward scopes to temporarily disable PyTorch's
deterministic-algorithm guard.  This module keeps that policy executable and
auditable instead of treating the YAML fields as comments.

The implementation is intentionally independent of CUDA kernels.  A CPU-only
test can exercise the CUDA control-flow branch with a small loss-like object
whose ``is_cuda`` attribute is true.  Real callers pass an ordinary scalar
``torch.Tensor`` loss.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch


class D0DeterminismRuntimeError(RuntimeError):
    """The frozen D0 determinism contract or its runtime state was violated."""


class D0BackwardExecutionError(D0DeterminismRuntimeError):
    """A scoped backward failed after the deterministic state was restored."""

    def __init__(self, *, scope: str, audit: Mapping[str, Any]) -> None:
        super().__init__(
            f"D0 backward failed in scope {scope!r}; runtime policy was restored"
        )
        # ``validate_audit`` has already checked this exact-key JSON-safe value.
        self.audit = dict(audit)


_SCHEMA_VERSION = 1
_POLICY = "strict_forwards_temporary_backward_disable"
_SCOPES = ("entropy_backward", "supervised_task_backward")
_STRICT_FORWARD = {
    "deterministic_algorithms_enabled": True,
    "warn_only": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
_FROZEN_CONFIG = {
    "policy": _POLICY,
    "strict_forward": _STRICT_FORWARD,
    "temporary_backward_disable_scopes": list(_SCOPES),
    "restore_strict_policy_before_optimizer_step": True,
    "restore_strict_policy_before_post_forward": True,
    "restore_strict_policy_after_backward_exception": True,
}
_STATE_KEYS = frozenset(_STRICT_FORWARD)
_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "policy",
        "config_sha256",
        "scope",
        "loss_device_type",
        "temporary_backward_disable_applied",
        "state_before_backward_scope",
        "state_during_backward",
        "state_after_backward_scope",
        "backward_completed",
        "restored_exact",
    }
)


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D0DeterminismRuntimeError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise D0DeterminismRuntimeError(f"{path} keys must all be strings")
    return value


def _assert_exact(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        observed = _require_mapping(actual, path)
        missing = sorted(set(expected) - set(observed))
        unknown = sorted(set(observed) - set(expected))
        if missing or unknown:
            raise D0DeterminismRuntimeError(
                f"{path} fields are not exact; missing={missing}, unknown={unknown}"
            )
        for key, expected_value in expected.items():
            _assert_exact(observed[key], expected_value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, Sequence) or isinstance(
            actual, (str, bytes, bytearray)
        ):
            raise D0DeterminismRuntimeError(f"{path} must be a sequence")
        if len(actual) != len(expected):
            raise D0DeterminismRuntimeError(
                f"{path} length drifted: expected {len(expected)}, got {len(actual)}"
            )
        for index, (observed, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact(observed, expected_value, f"{path}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise D0DeterminismRuntimeError(
            f"{path} drifted: expected {expected!r}, got {actual!r}"
        )


def validate_frozen_determinism_mapping(value: Mapping[str, Any]) -> None:
    """Fail closed unless ``value`` is exactly the frozen D0 YAML mapping."""

    _assert_exact(value, _FROZEN_CONFIG, "method.determinism")


def _canonical_config(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_frozen_determinism_mapping(value)
    # Return a fresh JSON-native object so caller-owned mappings cannot affect
    # canonicalization after validation.
    return {
        "policy": _POLICY,
        "strict_forward": dict(_STRICT_FORWARD),
        "temporary_backward_disable_scopes": list(_SCOPES),
        "restore_strict_policy_before_optimizer_step": True,
        "restore_strict_policy_before_post_forward": True,
        "restore_strict_policy_after_backward_exception": True,
    }


def frozen_determinism_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 binding for a valid frozen mapping."""

    payload = json.dumps(
        _canonical_config(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_runtime_determinism() -> dict[str, bool]:
    """Capture every runtime switch governed by the D0 forward policy."""

    return {
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _validate_state(value: Any, path: str) -> dict[str, bool]:
    observed = _require_mapping(value, path)
    missing = sorted(_STATE_KEYS - set(observed))
    unknown = sorted(set(observed) - _STATE_KEYS)
    if missing or unknown:
        raise D0DeterminismRuntimeError(
            f"{path} fields are not exact; missing={missing}, unknown={unknown}"
        )
    result: dict[str, bool] = {}
    for key in _STRICT_FORWARD:
        item = observed[key]
        if type(item) is not bool:
            raise D0DeterminismRuntimeError(f"{path}.{key} must be a bool")
        result[key] = item
    return result


def assert_strict_forward_policy(
    determinism_config: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate the YAML policy and assert the live process is strictly set."""

    validate_frozen_determinism_mapping(determinism_config)
    state = capture_runtime_determinism()
    if state != _STRICT_FORWARD:
        raise D0DeterminismRuntimeError(
            "D0 strict forward runtime drifted: "
            f"expected={_STRICT_FORWARD}, observed={state}"
        )
    return state


def _restore_runtime_determinism(state: Mapping[str, bool]) -> None:
    expected = _validate_state(state, "restore_state")
    current = capture_runtime_determinism()
    if (
        current["deterministic_algorithms_enabled"]
        != expected["deterministic_algorithms_enabled"]
        or current["warn_only"] != expected["warn_only"]
    ):
        torch.use_deterministic_algorithms(
            expected["deterministic_algorithms_enabled"],
            warn_only=expected["warn_only"],
        )
    if current["cudnn_deterministic"] != expected["cudnn_deterministic"]:
        torch.backends.cudnn.deterministic = expected["cudnn_deterministic"]
    if current["cudnn_benchmark"] != expected["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = expected["cudnn_benchmark"]


def _validate_scope(scope: Any) -> str:
    if type(scope) is not str or scope not in _SCOPES:
        raise D0DeterminismRuntimeError(
            f"unknown D0 backward scope {scope!r}; expected one of {_SCOPES}"
        )
    return scope


def validate_audit(
    audit: Mapping[str, Any],
    *,
    determinism_config: Mapping[str, Any],
    expected_scope: str | None = None,
    expected_loss_device_type: str | None = None,
    expected_backward_completed: bool | None = True,
) -> None:
    """Validate an exact-key, JSON-safe audit against the frozen policy."""

    validate_frozen_determinism_mapping(determinism_config)
    observed = _require_mapping(audit, "audit")
    missing = sorted(_AUDIT_KEYS - set(observed))
    unknown = sorted(set(observed) - _AUDIT_KEYS)
    if missing or unknown:
        raise D0DeterminismRuntimeError(
            f"audit fields are not exact; missing={missing}, unknown={unknown}"
        )
    if type(observed["schema_version"]) is not int or observed["schema_version"] != 1:
        raise D0DeterminismRuntimeError("audit.schema_version must be exactly 1")
    if observed["policy"] != _POLICY:
        raise D0DeterminismRuntimeError("audit.policy drifted")
    if observed["config_sha256"] != frozen_determinism_sha256(
        determinism_config
    ):
        raise D0DeterminismRuntimeError("audit.config_sha256 drifted")
    scope = _validate_scope(observed["scope"])
    if expected_scope is not None and scope != _validate_scope(expected_scope):
        raise D0DeterminismRuntimeError(
            f"audit.scope mismatch: expected {expected_scope!r}, got {scope!r}"
        )
    device_type = observed["loss_device_type"]
    if type(device_type) is not str or device_type not in {"cpu", "cuda"}:
        raise D0DeterminismRuntimeError(
            "audit.loss_device_type must be exactly 'cpu' or 'cuda'"
        )
    if (
        expected_loss_device_type is not None
        and device_type != expected_loss_device_type
    ):
        raise D0DeterminismRuntimeError(
            "audit.loss_device_type mismatch: "
            f"expected {expected_loss_device_type!r}, got {device_type!r}"
        )
    for key in (
        "temporary_backward_disable_applied",
        "backward_completed",
        "restored_exact",
    ):
        if type(observed[key]) is not bool:
            raise D0DeterminismRuntimeError(f"audit.{key} must be a bool")
    disable_expected = device_type == "cuda"
    if observed["temporary_backward_disable_applied"] is not disable_expected:
        raise D0DeterminismRuntimeError(
            "audit temporary-disable decision does not match loss device"
        )
    before = _validate_state(
        observed["state_before_backward_scope"],
        "audit.state_before_backward_scope",
    )
    during = _validate_state(
        observed["state_during_backward"], "audit.state_during_backward"
    )
    after = _validate_state(
        observed["state_after_backward_scope"],
        "audit.state_after_backward_scope",
    )
    if before != _STRICT_FORWARD:
        raise D0DeterminismRuntimeError("audit did not start under strict policy")
    expected_during = dict(_STRICT_FORWARD)
    if disable_expected:
        expected_during["deterministic_algorithms_enabled"] = False
    if during != expected_during:
        raise D0DeterminismRuntimeError(
            f"audit backward state drifted: expected={expected_during}, got={during}"
        )
    if after != before or observed["restored_exact"] is not True:
        raise D0DeterminismRuntimeError(
            "audit did not prove exact deterministic-policy restoration"
        )
    completed = observed["backward_completed"]
    if (
        expected_backward_completed is not None
        and completed is not expected_backward_completed
    ):
        raise D0DeterminismRuntimeError(
            "audit.backward_completed mismatch: "
            f"expected {expected_backward_completed}, got {completed}"
        )
    # This also proves that the full nested structure is JSON-native.
    try:
        json.dumps(observed, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise D0DeterminismRuntimeError("audit is not JSON-safe") from exc


def backward_with_d0_determinism(
    loss: Any,
    *,
    scope: str,
    determinism_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one backward under the frozen D0 policy and return its audit.

    CUDA losses temporarily disable deterministic algorithms only around the
    backward invocation.  CPU losses remain under the strict forward policy.
    The exact pre-scope state is restored on success and on ordinary
    exceptions.  Failures are wrapped in :class:`D0BackwardExecutionError`;
    the original exception remains available as ``__cause__`` and the failed
    exact-key audit is available as ``.audit``.
    """

    scope = _validate_scope(scope)
    before = assert_strict_forward_policy(determinism_config)
    if not callable(getattr(loss, "backward", None)):
        raise D0DeterminismRuntimeError("loss.backward must be callable")
    is_cuda = getattr(loss, "is_cuda", None)
    if type(is_cuda) is not bool:
        raise D0DeterminismRuntimeError("loss.is_cuda must be a bool")
    device_type = "cuda" if is_cuda else "cpu"
    failure: Exception | None = None
    completed = False
    during: dict[str, bool] | None = None
    try:
        if is_cuda:
            torch.use_deterministic_algorithms(False, warn_only=False)
        during = capture_runtime_determinism()
        loss.backward()
        completed = True
    except Exception as exc:  # restoration is the contract; preserve as cause
        failure = exc
    finally:
        _restore_runtime_determinism(before)
    after = capture_runtime_determinism()
    if during is None:  # defensive: the branch above always captures first
        raise D0DeterminismRuntimeError("backward runtime state was not captured")
    audit: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "policy": _POLICY,
        "config_sha256": frozen_determinism_sha256(determinism_config),
        "scope": scope,
        "loss_device_type": device_type,
        "temporary_backward_disable_applied": is_cuda,
        "state_before_backward_scope": before,
        "state_during_backward": during,
        "state_after_backward_scope": after,
        "backward_completed": completed,
        "restored_exact": after == before,
    }
    validate_audit(
        audit,
        determinism_config=determinism_config,
        expected_scope=scope,
        expected_loss_device_type=device_type,
        expected_backward_completed=completed,
    )
    if failure is not None:
        raise D0BackwardExecutionError(scope=scope, audit=audit) from failure
    return audit


__all__ = [
    "D0BackwardExecutionError",
    "D0DeterminismRuntimeError",
    "assert_strict_forward_policy",
    "backward_with_d0_determinism",
    "capture_runtime_determinism",
    "frozen_determinism_sha256",
    "validate_audit",
    "validate_frozen_determinism_mapping",
]
