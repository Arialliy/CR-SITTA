from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest
import torch
from torch import nn

import tta.proposal_runner_v1 as proposal_runner
from tta.proposal_step import ProposalStepError, SafetyDecision


def _bytes(value: torch.Tensor) -> bytes:
    return (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def _hook_count(parameter: nn.Parameter) -> int:
    hooks = getattr(parameter, "_backward_hooks", None)
    return 0 if hooks is None else len(hooks)


def test_accepted_execution_keeps_endpoint_and_exact_ordered_vectors() -> None:
    first = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float32))
    second = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    first.grad = torch.tensor([19.0, -7.0], dtype=torch.float32)
    first_source = first.detach().clone()
    second_source = second.detach().clone()
    first_grad = _bytes(first.grad)
    epsilon = 1e-12

    # The graph expression is deliberately reversed relative to the requested
    # topology.  Hook firing order must not determine vector order.
    execution = proposal_runner.propose_and_backtrack(
        named_parameters=(
            ("first", first),
            ("second", second),
        ),
        loss_closure=lambda: second.square().sum() + first.square().sum(),
        safety_closure=lambda: SafetyDecision(True, "all_guards_passed"),
        relative_radius=0.1,
        absolute_radius=None,
        coefficients=(1.0, 0.5, 0.25, 0.125),
        armijo_c=1e-4,
        epsilon=epsilon,
    )

    result = execution.result
    expected_gradient = torch.tensor([6.0, 8.0, 2.0], dtype=torch.float64)
    source_norm = math.hypot(5.0, 1.0)
    gradient_norm = math.hypot(10.0, 2.0)
    trust_radius = 0.1 * (source_norm + epsilon)
    scale = min(1.0, trust_radius / (gradient_norm + epsilon))
    first_direction = -expected_gradient[:2].to(torch.float32) * scale
    second_direction = -expected_gradient[2:].to(torch.float64) * scale
    expected_direction = torch.cat(
        (first_direction.to(torch.float64), second_direction)
    )

    assert result.accepted is True
    assert result.parameter_names == ("first", "second")
    assert result.scalar_parameter_count == 3
    assert execution.proxy_gradient.device.type == "cpu"
    assert execution.proxy_gradient.dtype == torch.float64
    assert execution.proxy_gradient.ndim == 1
    assert torch.equal(execution.proxy_gradient, expected_gradient)
    assert torch.equal(execution.normalized_direction, expected_direction)
    assert torch.linalg.vector_norm(execution.proxy_gradient).item() == pytest.approx(
        result.gradient_norm, rel=1e-12, abs=1e-12
    )
    assert torch.linalg.vector_norm(
        execution.normalized_direction
    ).item() == pytest.approx(
        result.normalized_step_norm, rel=1e-12, abs=1e-12
    )
    assert torch.dot(
        execution.proxy_gradient, execution.normalized_direction
    ).item() == pytest.approx(
        result.directional_derivative, rel=1e-12, abs=1e-12
    )
    assert _bytes(first) == _bytes(
        torch.add(first_source, first_direction, alpha=result.coefficient)
    )
    assert _bytes(second) == _bytes(
        torch.add(second_source, second_direction, alpha=result.coefficient)
    )
    assert first.grad is not None and _bytes(first.grad) == first_grad
    assert second.grad is None
    assert _hook_count(first) == _hook_count(second) == 0

    local_gradient = execution.proxy_gradient
    local_direction = execution.normalized_direction
    local_gradient.fill_(999.0)
    local_direction.zero_()
    assert torch.equal(execution.proxy_gradient, expected_gradient)
    assert torch.equal(execution.normalized_direction, expected_direction)
    with pytest.raises(FrozenInstanceError):
        execution.result = result  # type: ignore[misc]


def test_all_rejected_returns_vectors_and_restores_exact_source() -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    parameter.grad = torch.tensor([17.0], dtype=torch.float64)
    source_value = _bytes(parameter)
    source_gradient = _bytes(parameter.grad)

    execution = proposal_runner.propose_and_backtrack(
        named_parameters=(("decoder.scale", parameter),),
        loss_closure=lambda: parameter.sum(),
        safety_closure=lambda: (False, "mass_guard"),
        relative_radius=0.5,
        absolute_radius=None,
        coefficients=(1.0, 0.5, 0.25, 0.125),
        armijo_c=1e-4,
        epsilon=1e-12,
    )

    assert execution.result.accepted is False
    assert execution.result.reason == "all_candidates_rejected"
    assert [attempt.reason for attempt in execution.result.attempts] == [
        "safety_rejected",
        "safety_rejected",
        "safety_rejected",
        "safety_rejected",
    ]
    assert torch.equal(
        execution.proxy_gradient, torch.tensor([1.0], dtype=torch.float64)
    )
    expected_scale = execution.result.trust_radius / (
        execution.result.gradient_norm + 1e-12
    )
    assert torch.equal(
        execution.normalized_direction,
        torch.tensor([-expected_scale], dtype=torch.float64),
    )
    assert _bytes(parameter) == source_value
    assert parameter.grad is not None
    assert _bytes(parameter.grad) == source_gradient
    assert _hook_count(parameter) == 0


def test_zero_gradient_is_a_finite_no_update_execution() -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    source_value = _bytes(parameter)

    execution = proposal_runner.propose_and_backtrack(
        named_parameters=(("decoder.bias", parameter),),
        loss_closure=lambda: (parameter * 0.0).sum(),
        safety_closure=lambda: pytest.fail("safety must not run"),
        relative_radius=0.1,
        absolute_radius=None,
        coefficients=(1.0, 0.5, 0.25, 0.125),
        armijo_c=1e-4,
        epsilon=1e-12,
    )

    assert execution.result.accepted is False
    assert execution.result.reason == "zero_gradient"
    assert execution.result.attempts == ()
    assert torch.equal(
        execution.proxy_gradient, torch.zeros(1, dtype=torch.float64)
    )
    assert torch.equal(
        execution.normalized_direction, torch.zeros(1, dtype=torch.float64)
    )
    assert _bytes(parameter) == source_value
    assert parameter.grad is None
    assert _hook_count(parameter) == 0


def test_capture_hook_exception_removes_only_temporary_hook_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    source_value = _bytes(parameter)
    permanent = parameter.register_hook(lambda gradient: gradient)
    hooks_before = _hook_count(parameter)

    def broken_factory(**_kwargs):
        def broken(_gradient: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("capture hook failed")

        return broken

    monkeypatch.setattr(
        proposal_runner, "_captured_gradient_hook", broken_factory
    )
    try:
        with pytest.raises(ProposalStepError, match="autograd.grad failed"):
            proposal_runner.propose_and_backtrack(
                named_parameters=(("p", parameter),),
                loss_closure=lambda: parameter.square().sum(),
                safety_closure=lambda: True,
                relative_radius=0.1,
                absolute_radius=None,
                coefficients=(1.0, 0.5, 0.25, 0.125),
                armijo_c=1e-4,
                epsilon=1e-12,
            )
        assert _hook_count(parameter) == hooks_before
        assert _bytes(parameter) == source_value
        assert parameter.grad is None
    finally:
        permanent.remove()


def test_autograd_gradient_capture_never_writes_grad_slots() -> None:
    first = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    second = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    first.grad = torch.tensor([-123.0], dtype=torch.float64)
    first_gradient = _bytes(first.grad)

    execution = proposal_runner.propose_and_backtrack(
        named_parameters=(("first", first), ("second", second)),
        loss_closure=lambda: first.square().sum() + second.square().sum(),
        safety_closure=lambda: True,
        relative_radius=0.01,
        absolute_radius=None,
        coefficients=(1.0, 0.5, 0.25, 0.125),
        armijo_c=1e-4,
        epsilon=1e-12,
    )

    assert execution.result.accepted
    assert first.grad is not None and _bytes(first.grad) == first_gradient
    assert second.grad is None


def _determinism_state() -> tuple[bool, bool]:
    return (
        bool(torch.are_deterministic_algorithms_enabled()),
        bool(torch.is_deterministic_algorithms_warn_only_enabled()),
    )


def test_cuda_override_is_only_active_during_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    policy_before_test = _determinism_state()
    events: list[tuple[str, tuple[bool, bool]]] = []

    monkeypatch.setattr(
        proposal_runner, "_has_cuda_parameters", lambda _parameters: True
    )

    class ObservedSquare(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
            events.append(("objective_forward", _determinism_state()))
            setattr(ctx, "saved_value", value)
            return value.square().sum()

        @staticmethod
        def backward(
            ctx: object, upstream: torch.Tensor
        ) -> torch.Tensor:
            events.append(("objective_backward", _determinism_state()))
            value = getattr(ctx, "saved_value")
            return upstream * 2.0 * value

    loss_call_count = 0

    def observed_loss() -> torch.Tensor:
        nonlocal loss_call_count
        label = "source_closure" if loss_call_count == 0 else "candidate_closure"
        loss_call_count += 1
        events.append((label, _determinism_state()))
        return ObservedSquare.apply(parameter)

    def observed_safety() -> bool:
        events.append(("safety", _determinism_state()))
        return True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        execution = proposal_runner.propose_and_backtrack(
            named_parameters=(("p", parameter),),
            loss_closure=observed_loss,
            safety_closure=observed_safety,
            relative_radius=0.1,
            absolute_radius=None,
            coefficients=(1.0, 0.5, 0.25, 0.125),
            armijo_c=1e-4,
            epsilon=1e-12,
            allow_cuda_nondeterministic_backward=True,
        )
        events.append(("post", _determinism_state()))

        assert execution.result.accepted is True
        assert events == [
            ("source_closure", (True, True)),
            ("objective_forward", (True, True)),
            ("objective_backward", (False, False)),
            ("candidate_closure", (True, True)),
            ("objective_forward", (True, True)),
            ("safety", (True, True)),
            ("post", (True, True)),
        ]
        assert _hook_count(parameter) == 0
    finally:
        torch.use_deterministic_algorithms(
            policy_before_test[0], warn_only=policy_before_test[1]
        )


def test_cuda_override_restores_policy_and_source_after_backward_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    source_value = _bytes(parameter)
    policy_before_test = _determinism_state()
    events: list[tuple[str, tuple[bool, bool]]] = []

    monkeypatch.setattr(
        proposal_runner, "_has_cuda_parameters", lambda _parameters: True
    )

    class FailingBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
            del ctx
            events.append(("source_forward", _determinism_state()))
            return value.square().sum()

        @staticmethod
        def backward(
            ctx: object, upstream: torch.Tensor
        ) -> torch.Tensor:
            del ctx, upstream
            events.append(("source_backward", _determinism_state()))
            raise RuntimeError("deliberate backward failure")

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.raises(ProposalStepError, match="autograd.grad failed"):
            proposal_runner.propose_and_backtrack(
                named_parameters=(("p", parameter),),
                loss_closure=lambda: FailingBackward.apply(parameter),
                safety_closure=lambda: True,
                relative_radius=0.1,
                absolute_radius=None,
                coefficients=(1.0, 0.5, 0.25, 0.125),
                armijo_c=1e-4,
                epsilon=1e-12,
                allow_cuda_nondeterministic_backward=True,
            )

        assert events == [
            ("source_forward", (True, True)),
            ("source_backward", (False, False)),
        ]
        assert _determinism_state() == (True, True)
        assert _bytes(parameter) == source_value
        assert parameter.grad is None
        assert _hook_count(parameter) == 0
    finally:
        torch.use_deterministic_algorithms(
            policy_before_test[0], warn_only=policy_before_test[1]
        )


def test_cuda_override_restores_after_zero_gradient_without_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    source_value = _bytes(parameter)
    policy_before_test = _determinism_state()
    events: list[tuple[str, tuple[bool, bool]]] = []

    monkeypatch.setattr(
        proposal_runner, "_has_cuda_parameters", lambda _parameters: True
    )

    class ObservedZeroGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
            events.append(("source_forward", _determinism_state()))
            setattr(ctx, "saved_value", value)
            return value.square().sum()

        @staticmethod
        def backward(
            ctx: object, upstream: torch.Tensor
        ) -> torch.Tensor:
            del upstream
            events.append(("source_backward", _determinism_state()))
            value = getattr(ctx, "saved_value")
            return torch.zeros_like(value)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        execution = proposal_runner.propose_and_backtrack(
            named_parameters=(("p", parameter),),
            loss_closure=lambda: ObservedZeroGradient.apply(parameter),
            safety_closure=lambda: pytest.fail("safety must not run"),
            relative_radius=0.1,
            absolute_radius=None,
            coefficients=(1.0, 0.5, 0.25, 0.125),
            armijo_c=1e-4,
            epsilon=1e-12,
            allow_cuda_nondeterministic_backward=True,
        )
        events.append(("post", _determinism_state()))

        assert execution.result.reason == "zero_gradient"
        assert events == [
            ("source_forward", (True, True)),
            ("source_backward", (False, False)),
            ("post", (True, True)),
        ]
        assert _bytes(parameter) == source_value
        assert parameter.grad is None
        assert _hook_count(parameter) == 0
    finally:
        torch.use_deterministic_algorithms(
            policy_before_test[0], warn_only=policy_before_test[1]
        )
