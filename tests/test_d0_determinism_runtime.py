from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from analysis.d0_determinism_runtime import (
    D0BackwardExecutionError,
    D0DeterminismRuntimeError,
    assert_strict_forward_policy,
    backward_with_d0_determinism,
    capture_runtime_determinism,
    frozen_determinism_sha256,
    validate_audit,
    validate_frozen_determinism_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "tent_failure_diagnostics_v1.yaml"


def _config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return value["method"]["determinism"]


def _set_state(state: dict[str, bool]) -> None:
    torch.use_deterministic_algorithms(
        state["deterministic_algorithms_enabled"],
        warn_only=state["warn_only"],
    )
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]


@pytest.fixture(autouse=True)
def _restore_global_determinism() -> Any:
    original = capture_runtime_determinism()
    try:
        yield
    finally:
        _set_state(original)


@pytest.fixture
def strict_runtime() -> dict[str, Any]:
    config = _config()
    _set_state(
        {
            "deterministic_algorithms_enabled": True,
            "warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }
    )
    return config


class _LossProbe:
    def __init__(self, *, is_cuda: bool, failure: Exception | None = None) -> None:
        self.is_cuda = is_cuda
        self.failure = failure
        self.calls = 0
        self.state_seen: dict[str, bool] | None = None

    def backward(self) -> None:
        self.calls += 1
        self.state_seen = capture_runtime_determinism()
        if self.failure is not None:
            raise self.failure


def test_checked_in_config_is_exact_and_has_stable_sha256() -> None:
    config = _config()
    validate_frozen_determinism_mapping(config)
    first = frozen_determinism_sha256(config)
    second = frozen_determinism_sha256(copy.deepcopy(config))
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("policy"),
        lambda value: value.update({"policy": "best_effort"}),
        lambda value: value["strict_forward"].update({"warn_only": True}),
        lambda value: value["strict_forward"].update({"extra": False}),
        lambda value: value.update(
            {
                "temporary_backward_disable_scopes": [
                    "supervised_task_backward",
                    "entropy_backward",
                ]
            }
        ),
        lambda value: value.update(
            {"restore_strict_policy_before_optimizer_step": 1}
        ),
    ],
)
def test_config_drift_fails_closed(mutation: Any) -> None:
    config = _config()
    mutation(config)
    with pytest.raises(D0DeterminismRuntimeError):
        validate_frozen_determinism_mapping(config)


def test_assert_strict_forward_policy_returns_exact_runtime_state(
    strict_runtime: dict[str, Any],
) -> None:
    state = assert_strict_forward_policy(strict_runtime)
    assert state == {
        "deterministic_algorithms_enabled": True,
        "warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deterministic_algorithms_enabled", False),
        ("warn_only", True),
        ("cudnn_deterministic", False),
        ("cudnn_benchmark", True),
    ],
)
def test_assert_strict_forward_policy_rejects_each_runtime_drift(
    strict_runtime: dict[str, Any], field: str, value: bool
) -> None:
    state = capture_runtime_determinism()
    state[field] = value
    _set_state(state)
    with pytest.raises(D0DeterminismRuntimeError, match="runtime drifted"):
        assert_strict_forward_policy(strict_runtime)


def test_real_cpu_backward_stays_strict_and_returns_json_safe_audit(
    strict_runtime: dict[str, Any],
) -> None:
    parameter = torch.tensor(3.0, requires_grad=True)
    loss = parameter.square()
    audit = backward_with_d0_determinism(
        loss,
        scope="entropy_backward",
        determinism_config=strict_runtime,
    )
    assert parameter.grad is not None
    assert parameter.grad.item() == 6.0
    assert audit["loss_device_type"] == "cpu"
    assert audit["temporary_backward_disable_applied"] is False
    assert audit["state_during_backward"][
        "deterministic_algorithms_enabled"
    ] is True
    assert audit["backward_completed"] is True
    assert audit["restored_exact"] is True
    validate_audit(
        audit,
        determinism_config=strict_runtime,
        expected_scope="entropy_backward",
        expected_loss_device_type="cpu",
    )
    json.dumps(audit, sort_keys=True)


def test_cpu_branch_does_not_toggle_deterministic_algorithms(
    strict_runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = torch.use_deterministic_algorithms
    calls: list[tuple[bool, bool]] = []

    def spy(mode: bool, *, warn_only: bool = False) -> None:
        calls.append((mode, warn_only))
        original(mode, warn_only=warn_only)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", spy)
    probe = _LossProbe(is_cuda=False)
    audit = backward_with_d0_determinism(
        probe,
        scope="supervised_task_backward",
        determinism_config=strict_runtime,
    )
    assert probe.calls == 1
    assert probe.state_seen == audit["state_before_backward_scope"]
    assert calls == []


def test_simulated_cuda_branch_disables_only_algorithm_guard_and_restores(
    strict_runtime: dict[str, Any],
) -> None:
    probe = _LossProbe(is_cuda=True)
    audit = backward_with_d0_determinism(
        probe,
        scope="entropy_backward",
        determinism_config=strict_runtime,
    )
    assert probe.calls == 1
    assert probe.state_seen == {
        "deterministic_algorithms_enabled": False,
        "warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    assert audit["temporary_backward_disable_applied"] is True
    assert audit["state_during_backward"] == probe.state_seen
    assert capture_runtime_determinism() == audit["state_before_backward_scope"]
    validate_audit(
        audit,
        determinism_config=strict_runtime,
        expected_scope="entropy_backward",
        expected_loss_device_type="cuda",
    )


@pytest.mark.parametrize("is_cuda", [False, True])
def test_backward_exception_is_wrapped_with_failed_audit_after_exact_restore(
    strict_runtime: dict[str, Any], is_cuda: bool
) -> None:
    original_failure = ValueError("synthetic backward failure")
    probe = _LossProbe(is_cuda=is_cuda, failure=original_failure)
    before = capture_runtime_determinism()
    with pytest.raises(D0BackwardExecutionError) as caught:
        backward_with_d0_determinism(
            probe,
            scope="supervised_task_backward",
            determinism_config=strict_runtime,
        )
    assert caught.value.__cause__ is original_failure
    assert capture_runtime_determinism() == before
    audit = caught.value.audit
    assert audit["backward_completed"] is False
    assert audit["restored_exact"] is True
    validate_audit(
        audit,
        determinism_config=strict_runtime,
        expected_scope="supervised_task_backward",
        expected_loss_device_type="cuda" if is_cuda else "cpu",
        expected_backward_completed=False,
    )


@pytest.mark.parametrize("scope", ["unknown", "", None, 1])
def test_unknown_scope_fails_before_backward(
    strict_runtime: dict[str, Any], scope: Any
) -> None:
    probe = _LossProbe(is_cuda=True)
    with pytest.raises(D0DeterminismRuntimeError, match="unknown D0 backward scope"):
        backward_with_d0_determinism(
            probe,
            scope=scope,
            determinism_config=strict_runtime,
        )
    assert probe.calls == 0
    assert_strict_forward_policy(strict_runtime)


def test_invalid_loss_fails_before_any_policy_toggle(
    strict_runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = torch.use_deterministic_algorithms
    calls: list[tuple[bool, bool]] = []

    def spy(mode: bool, *, warn_only: bool = False) -> None:
        calls.append((mode, warn_only))
        original(mode, warn_only=warn_only)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", spy)
    with pytest.raises(D0DeterminismRuntimeError, match="loss.backward"):
        backward_with_d0_determinism(
            object(),
            scope="entropy_backward",
            determinism_config=strict_runtime,
        )
    assert calls == []


def test_config_drift_fails_before_backward(strict_runtime: dict[str, Any]) -> None:
    strict_runtime["strict_forward"]["cudnn_benchmark"] = True
    probe = _LossProbe(is_cuda=True)
    with pytest.raises(D0DeterminismRuntimeError):
        backward_with_d0_determinism(
            probe,
            scope="entropy_backward",
            determinism_config=strict_runtime,
        )
    assert probe.calls == 0


def test_validate_audit_rejects_unknown_missing_and_nested_drift(
    strict_runtime: dict[str, Any],
) -> None:
    base = backward_with_d0_determinism(
        _LossProbe(is_cuda=False),
        scope="entropy_backward",
        determinism_config=strict_runtime,
    )
    variants = []
    unknown = copy.deepcopy(base)
    unknown["note"] = "informational"
    variants.append(unknown)
    missing = copy.deepcopy(base)
    missing.pop("restored_exact")
    variants.append(missing)
    nested = copy.deepcopy(base)
    nested["state_during_backward"]["extra"] = False
    variants.append(nested)
    wrong_device = copy.deepcopy(base)
    wrong_device["loss_device_type"] = "cuda"
    variants.append(wrong_device)
    bad_restore = copy.deepcopy(base)
    bad_restore["restored_exact"] = False
    variants.append(bad_restore)
    bad_sha = copy.deepcopy(base)
    bad_sha["config_sha256"] = "0" * 64
    variants.append(bad_sha)
    for audit in variants:
        with pytest.raises(D0DeterminismRuntimeError):
            validate_audit(audit, determinism_config=strict_runtime)


def test_validate_audit_rejects_scope_or_completion_expectation_mismatch(
    strict_runtime: dict[str, Any],
) -> None:
    audit = backward_with_d0_determinism(
        _LossProbe(is_cuda=False),
        scope="entropy_backward",
        determinism_config=strict_runtime,
    )
    with pytest.raises(D0DeterminismRuntimeError, match="scope mismatch"):
        validate_audit(
            audit,
            determinism_config=strict_runtime,
            expected_scope="supervised_task_backward",
        )
    with pytest.raises(D0DeterminismRuntimeError, match="backward_completed"):
        validate_audit(
            audit,
            determinism_config=strict_runtime,
            expected_backward_completed=False,
        )
