from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from tta.proposal_step import (
    DEFAULT_BACKTRACKING_COEFFICIENTS,
    ProposalBuildRejected,
    ProposalStepError,
    SafetyDecision,
    applied_proposal,
    apply_proposal_from_source,
    build_normalized_proposal,
    export_proposal_vectors,
    proposal_source_is_exact,
    propose_and_backtrack,
    restore_proposal_source,
)


def _bytes(value: torch.Tensor) -> bytes:
    return (
        value.detach()
        .cpu()
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def test_relative_trust_radius_accepts_explicit_descent_without_optimizer() -> None:
    parameter = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float32))

    result = propose_and_backtrack(
        named_parameters=(("decoder.weight", parameter),),
        loss_closure=lambda: parameter.square().sum(),
        safety_closure=lambda: SafetyDecision(True, "guards_passed"),
        relative_radius=0.1,
        absolute_radius=None,
    )

    assert result.accepted is True
    assert result.backtracking_index == 0
    assert result.coefficient == 1.0
    assert result.radius_mode == "relative"
    assert result.source_parameter_norm == pytest.approx(5.0)
    assert result.gradient_norm == pytest.approx(10.0)
    assert result.raw_step_norm == pytest.approx(10.0)
    assert result.trust_radius == pytest.approx(0.5)
    assert result.normalized_step_norm == pytest.approx(0.5, rel=1e-6)
    assert result.accepted_step_norm == pytest.approx(0.5, rel=1e-6)
    assert result.directional_derivative < 0.0
    assert result.loss_after < result.loss_before
    assert result.parameter_names == ("decoder.weight",)
    assert result.scalar_parameter_count == 2
    assert result.attempts[0].safety_reason == "guards_passed"
    assert torch.allclose(parameter, torch.tensor([2.7, 3.6]))

    with pytest.raises(FrozenInstanceError):
        result.coefficient = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.attempts[0].accepted = False  # type: ignore[misc]


def test_zero_initialized_adapter_requires_and_uses_absolute_radius() -> None:
    parameter = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    target = torch.tensor([1.0, -2.0], dtype=torch.float64)

    result = propose_and_backtrack(
        named_parameters={"adapter.raw": parameter},
        loss_closure=lambda: (parameter - target).square().sum(),
        safety_closure=lambda: True,
        relative_radius=None,
        absolute_radius=0.25,
    )

    assert result.accepted
    assert result.radius_mode == "absolute"
    assert result.source_parameter_norm == 0.0
    assert result.trust_radius == pytest.approx(0.25)
    assert result.normalized_step_norm == pytest.approx(0.25)
    assert torch.linalg.vector_norm(parameter).item() == pytest.approx(0.25)


def test_radius_mode_is_bound_to_actual_source_initialization() -> None:
    nonzero = nn.Parameter(torch.ones(1))
    zero = nn.Parameter(torch.zeros(1))

    with pytest.raises(ProposalStepError, match="non-zero Source"):
        build_normalized_proposal(
            named_parameters=(("p", nonzero),),
            loss_closure=lambda: nonzero.square().sum(),
            relative_radius=None,
            absolute_radius=0.1,
        )
    with pytest.raises(ProposalStepError, match="zero-initialized"):
        build_normalized_proposal(
            named_parameters=(("p", zero),),
            loss_closure=lambda: (zero - 1.0).square().sum(),
            relative_radius=0.1,
            absolute_radius=None,
        )
    with pytest.raises(ProposalStepError, match="exactly one"):
        build_normalized_proposal(
            named_parameters=(("p", nonzero),),
            loss_closure=lambda: nonzero.square().sum(),
            relative_radius=0.1,
            absolute_radius=0.1,
        )


def test_safety_backtracking_retries_from_source_not_previous_candidate() -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    observed_candidates: list[float] = []

    def safety() -> tuple[bool, str]:
        value = float(parameter.detach().item())
        observed_candidates.append(value)
        return value >= 1.25, "mass_guard"

    result = propose_and_backtrack(
        named_parameters=(("p", parameter),),
        loss_closure=lambda: parameter.sum(),
        safety_closure=safety,
        relative_radius=0.5,
        absolute_radius=None,
    )

    assert result.accepted
    assert result.backtracking_index == 1
    assert result.coefficient == 0.5
    assert observed_candidates == pytest.approx([1.0, 1.5])
    assert parameter.item() == pytest.approx(1.5)
    assert [attempt.reason for attempt in result.attempts] == [
        "safety_rejected",
        "accepted",
    ]
    assert result.attempts[0].safety_passed is False
    assert result.attempts[1].safety_passed is True


def test_armijo_rejection_restores_source_before_smaller_retry() -> None:
    parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))

    result = propose_and_backtrack(
        named_parameters=(("p", parameter),),
        loss_closure=lambda: parameter.square().sum(),
        safety_closure=lambda: True,
        relative_radius=2.0,
        absolute_radius=None,
    )

    assert result.accepted
    assert result.backtracking_index == 1
    assert result.attempts[0].reason == "armijo_rejected"
    assert result.attempts[0].safety_passed is None
    assert result.attempts[1].reason == "accepted"
    assert parameter.item() == pytest.approx(0.0, abs=1e-12)


def test_all_safety_rejections_are_exact_no_update_including_grad_state() -> None:
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    parameter.grad = torch.tensor([17.0], dtype=torch.float32)
    source_value = _bytes(parameter)
    source_gradient = _bytes(parameter.grad)
    candidates: list[float] = []

    def reject() -> SafetyDecision:
        candidates.append(float(parameter.detach().item()))
        return SafetyDecision(False, "foreground_mass")

    result = propose_and_backtrack(
        named_parameters=(("p", parameter),),
        loss_closure=lambda: parameter.sum(),
        safety_closure=reject,
        relative_radius=0.5,
        absolute_radius=None,
    )

    assert result.accepted is False
    assert result.backtracking_index is None
    assert result.coefficient == 0.0
    assert result.accepted_step_norm == 0.0
    assert result.loss_after == result.loss_before
    assert result.reason == "all_candidates_rejected"
    assert tuple(attempt.coefficient for attempt in result.attempts) == (
        DEFAULT_BACKTRACKING_COEFFICIENTS
    )
    assert candidates == pytest.approx([1.0, 1.5, 1.75, 1.875])
    assert all(attempt.reason == "safety_rejected" for attempt in result.attempts)
    assert _bytes(parameter) == source_value
    assert parameter.grad is not None
    assert _bytes(parameter.grad) == source_gradient


@pytest.mark.parametrize(
    ("initial_value", "loss_builder", "relative_radius", "absolute_radius", "reason"),
    [
        (
            2.0,
            lambda parameter: (parameter * 0.0).sum(),
            0.1,
            None,
            "zero_gradient",
        ),
        (
            0.0,
            lambda parameter: torch.sqrt(parameter).sum(),
            None,
            0.1,
            "nonfinite_gradient",
        ),
    ],
)
def test_unusable_gradient_returns_normal_no_update(
    initial_value: float,
    loss_builder,
    relative_radius: float | None,
    absolute_radius: float | None,
    reason: str,
) -> None:
    parameter = nn.Parameter(torch.tensor([initial_value], dtype=torch.float32))
    source = _bytes(parameter)

    result = propose_and_backtrack(
        named_parameters=(("p", parameter),),
        loss_closure=lambda: loss_builder(parameter),
        safety_closure=lambda: pytest.fail("safety must not run"),
        relative_radius=relative_radius,
        absolute_radius=absolute_radius,
    )

    assert not result.accepted
    assert result.reason == reason
    assert result.attempts == ()
    assert _bytes(parameter) == source


def test_named_parameter_topology_is_strict_and_fully_connected() -> None:
    parameter = nn.Parameter(torch.ones(1))
    other = nn.Parameter(torch.ones(1))

    with pytest.raises(ProposalStepError, match="must not be empty"):
        propose_and_backtrack(
            named_parameters=(),
            loss_closure=lambda: parameter.sum(),
            safety_closure=lambda: True,
            relative_radius=0.1,
            absolute_radius=None,
        )
    with pytest.raises(ProposalStepError, match="names must be unique"):
        propose_and_backtrack(
            named_parameters=(("p", parameter), ("p", other)),
            loss_closure=lambda: parameter.sum() + other.sum(),
            safety_closure=lambda: True,
            relative_radius=0.1,
            absolute_radius=None,
        )
    with pytest.raises(ProposalStepError, match="objects must be unique"):
        propose_and_backtrack(
            named_parameters=(("p", parameter), ("q", parameter)),
            loss_closure=lambda: parameter.sum(),
            safety_closure=lambda: True,
            relative_radius=0.1,
            absolute_radius=None,
        )
    source_parameter = _bytes(parameter)
    source_other = _bytes(other)
    with pytest.raises(ProposalStepError, match="disconnected.*q"):
        propose_and_backtrack(
            named_parameters=(("p", parameter), ("q", other)),
            loss_closure=lambda: parameter.sum(),
            safety_closure=lambda: True,
            relative_radius=0.1,
            absolute_radius=None,
        )
    assert _bytes(parameter) == source_parameter
    assert _bytes(other) == source_other


def test_b3_lower_level_virtual_step_and_exception_path_restore_exactly() -> None:
    parameter = nn.Parameter(torch.zeros(2, dtype=torch.float32))
    parameter.grad = torch.tensor([4.0, -3.0])
    source_value = _bytes(parameter)
    source_gradient = _bytes(parameter.grad)
    proposal = build_normalized_proposal(
        named_parameters=(("adapter.raw", parameter),),
        loss_closure=lambda: (parameter - 1.0).square().sum(),
        relative_radius=None,
        absolute_radius=0.2,
    )

    assert proposal_source_is_exact(proposal)
    with pytest.raises(RuntimeError, match="diagnostic failed"):
        with applied_proposal(proposal) as applied:
            assert applied.coefficient == 1.0
            assert applied.actual_step_norm == pytest.approx(0.2)
            assert not proposal_source_is_exact(proposal)
            raise RuntimeError("diagnostic failed")
    assert proposal_source_is_exact(proposal)
    assert _bytes(parameter) == source_value
    assert parameter.grad is not None
    assert _bytes(parameter.grad) == source_gradient

    first = apply_proposal_from_source(proposal, coefficient=1.0)
    endpoint = parameter.detach().clone()
    second = apply_proposal_from_source(proposal, coefficient=0.5)
    assert first == pytest.approx(0.2)
    assert second == pytest.approx(0.1)
    assert torch.allclose(parameter, endpoint * 0.5)
    restore_proposal_source(proposal)
    assert proposal_source_is_exact(proposal)


def test_candidate_closure_failure_is_fail_closed_to_exact_source() -> None:
    parameter = nn.Parameter(torch.tensor([2.0]))
    source = _bytes(parameter)
    calls = 0

    def loss() -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("candidate forward failed")
        return parameter.sum()

    with pytest.raises(RuntimeError, match="candidate forward failed"):
        propose_and_backtrack(
            named_parameters=(("p", parameter),),
            loss_closure=loss,
            safety_closure=lambda: True,
            relative_radius=0.1,
            absolute_radius=None,
        )
    assert _bytes(parameter) == source


def test_public_builder_exposes_normal_rejection_for_b3() -> None:
    parameter = nn.Parameter(torch.zeros(1))
    with pytest.raises(ProposalBuildRejected) as captured:
        build_normalized_proposal(
            named_parameters=(("adapter.raw", parameter),),
            loss_closure=lambda: (parameter * 0.0).sum(),
            relative_radius=None,
            absolute_radius=0.1,
        )
    assert captured.value.reason == "zero_gradient"
    assert parameter.item() == 0.0


def test_scalar_parameter_and_backtracking_coefficient_contract() -> None:
    parameter = nn.Parameter(torch.tensor(2.0))
    result = propose_and_backtrack(
        named_parameters=(("scalar", parameter),),
        loss_closure=lambda: parameter.square(),
        safety_closure=lambda: True,
        relative_radius=0.1,
        absolute_radius=None,
    )
    assert result.accepted

    restore_value = nn.Parameter(torch.tensor(2.0))
    with pytest.raises(ProposalStepError, match="must not exceed one"):
        propose_and_backtrack(
            named_parameters=(("scalar", restore_value),),
            loss_closure=lambda: restore_value.square(),
            safety_closure=lambda: True,
            relative_radius=0.1,
            absolute_radius=None,
            coefficients=(1.5, 1.0),
        )


def test_b3_vector_export_is_ordered_cpu_float64_and_read_only() -> None:
    first = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    second = nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    proposal = build_normalized_proposal(
        named_parameters=(("z_first", first), ("a_second", second)),
        loss_closure=lambda: first.square().sum() + second.square().sum(),
        relative_radius=0.1,
        absolute_radius=None,
    )

    gradient, direction = export_proposal_vectors(proposal)

    assert proposal.parameter_names == ("z_first", "a_second")
    assert gradient.device.type == direction.device.type == "cpu"
    assert gradient.dtype == direction.dtype == torch.float64
    assert gradient.shape == direction.shape == (3,)
    assert gradient.tolist() == pytest.approx([2.0, -4.0, 6.0])
    assert direction.tolist() == pytest.approx([-0.1, 0.2, -0.3])
    assert proposal_source_is_exact(proposal)

    gradient.fill_(999.0)
    direction.zero_()
    exported_again = export_proposal_vectors(proposal)
    assert exported_again[0].tolist() == pytest.approx([2.0, -4.0, 6.0])
    assert exported_again[1].tolist() == pytest.approx([-0.1, 0.2, -0.3])
    assert exported_again[0].data_ptr() != gradient.data_ptr()
    assert exported_again[1].data_ptr() != direction.data_ptr()
    assert proposal_source_is_exact(proposal)

    with pytest.raises(TypeError, match="NormalizedProposal"):
        export_proposal_vectors(object())  # type: ignore[arg-type]
