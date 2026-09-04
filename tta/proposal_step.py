"""Explicit normalized proposal steps for Stage-B adaptation.

The Stage-B proposal is deliberately not an optimizer.  A proposal is built
once from a scalar, label-free loss at the immutable Source endpoint, clipped
to either a relative (non-zero Source parameters) or absolute (zero-initialized
adapter parameters) trust radius, and can then be applied from that same
Source endpoint.  This split supports both uses required by the protocol:

* B3 can apply one normalized virtual step inside :func:`applied_proposal` and
  is guaranteed to return to Source, even when its diagnostic raises; and
* B4 can use :func:`propose_and_backtrack` for Source-anchored Armijo retries
  with an injected label-free safety check.

Only the explicitly supplied named parameters are changed.  ``autograd.grad``
is used instead of ``Tensor.backward`` so pre-existing ``.grad`` state is not
consumed or populated.  Rejected candidates are restored byte-for-byte before
the next coefficient is tried, and exhaustion is an exact no-update.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Iterator, TypeAlias

import torch
from torch import Tensor, nn


DEFAULT_BACKTRACKING_COEFFICIENTS = (1.0, 0.5, 0.25, 0.125)

NamedParameters: TypeAlias = (
    Mapping[str, nn.Parameter] | Iterable[tuple[str, nn.Parameter]]
)
LossClosure: TypeAlias = Callable[[], Tensor]


class ProposalStepError(RuntimeError):
    """The explicit proposal protocol was structurally violated."""


class ProposalBuildRejected(ProposalStepError):
    """A finite, non-zero normalized direction cannot be built safely.

    This is a normal fail-closed outcome for B3 callers.  The high-level
    :func:`propose_and_backtrack` function converts it into an immutable
    ``ProposalStepResult`` with ``accepted=False``.
    """

    def __init__(
        self,
        reason: str,
        *,
        loss_before: float,
        gradient_norm: float,
        source_parameter_norm: float,
        trust_radius: float,
        radius_mode: str,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.loss_before = loss_before
        self.gradient_norm = gradient_norm
        self.source_parameter_norm = source_parameter_norm
        self.trust_radius = trust_radius
        self.radius_mode = radius_mode


class ProposalCandidateRejected(ProposalStepError):
    """One coefficient cannot produce a finite parameter candidate."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SafetyDecision:
    """Immutable result returned by a label-free candidate safety closure."""

    passed: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("SafetyDecision.passed must be bool")
        if not isinstance(self.reason, str):
            raise TypeError("SafetyDecision.reason must be a string")


SafetyClosureResult: TypeAlias = SafetyDecision | bool | tuple[bool, str]
SafetyClosure: TypeAlias = Callable[[], SafetyClosureResult]


@dataclass(frozen=True)
class ProposalAttemptResult:
    """One immutable Source-anchored backtracking observation."""

    index: int
    coefficient: float
    proposed_step_norm: float
    loss_after: float | None
    armijo_rhs: float
    parameters_finite: bool
    loss_finite: bool
    armijo_passed: bool
    safety_passed: bool | None
    safety_reason: str
    accepted: bool
    reason: str


@dataclass(frozen=True)
class ProposalStepResult:
    """Detailed immutable outcome of one explicit proposal/backtracking step."""

    accepted: bool
    backtracking_index: int | None
    coefficient: float
    gradient_norm: float
    raw_step_norm: float
    normalized_step_norm: float
    accepted_step_norm: float
    source_parameter_norm: float
    trust_radius: float
    radius_mode: str
    directional_derivative: float
    loss_before: float
    loss_after: float
    parameter_names: tuple[str, ...]
    scalar_parameter_count: int
    attempts: tuple[ProposalAttemptResult, ...]
    reason: str


@dataclass(frozen=True)
class AppliedProposal:
    """Metadata yielded while a temporary B3-style virtual step is active."""

    coefficient: float
    actual_step_norm: float


@dataclass(frozen=True)
class _ParameterState:
    name: str
    parameter: nn.Parameter = field(repr=False, compare=False)
    source_value: Tensor = field(repr=False, compare=False)
    source_gradient: Tensor | None = field(repr=False, compare=False)
    proposal_gradient: Tensor = field(repr=False, compare=False)
    direction: Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class NormalizedProposal:
    """A normalized direction bound to an exact named Source snapshot.

    Tensor payloads are intentionally private.  Use
    :func:`apply_proposal_from_source`, :func:`restore_proposal_source`, or
    :func:`applied_proposal`; direct tensor mutation would violate the
    protocol contract.
    """

    parameter_names: tuple[str, ...]
    scalar_parameter_count: int
    gradient_norm: float
    raw_step_norm: float
    normalized_step_norm: float
    source_parameter_norm: float
    trust_radius: float
    radius_mode: str
    directional_derivative: float
    loss_before: float
    _states: tuple[_ParameterState, ...] = field(repr=False, compare=False)


def _finite_positive_float(value: Real, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProposalStepError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ProposalStepError(
            f"{field_name} must be finite and strictly positive"
        )
    return result


def _materialize_named_parameters(
    named_parameters: NamedParameters,
) -> tuple[tuple[str, nn.Parameter], ...]:
    if isinstance(named_parameters, Mapping):
        materialized = tuple(
            (name, named_parameters[name]) for name in sorted(named_parameters)
        )
    else:
        if isinstance(named_parameters, (str, bytes, bytearray)):
            raise ProposalStepError(
                "named_parameters must contain (name, nn.Parameter) pairs"
            )
        try:
            materialized = tuple(named_parameters)
        except TypeError as exc:
            raise ProposalStepError(
                "named_parameters must be a mapping or iterable of pairs"
            ) from exc

    if not materialized:
        raise ProposalStepError("named_parameters must not be empty")

    names: list[str] = []
    parameter_ids: list[int] = []
    for index, item in enumerate(materialized):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ProposalStepError(
                f"named_parameters[{index}] must be a (name, parameter) pair"
            )
        name, parameter = item
        if not isinstance(name, str) or not name or "\0" in name:
            raise ProposalStepError(
                f"named_parameters[{index}] has an invalid name"
            )
        if not isinstance(parameter, nn.Parameter):
            raise ProposalStepError(
                f"named_parameters[{index}] must contain nn.Parameter"
            )
        if (
            parameter.layout != torch.strided
            or parameter.is_sparse
            or not torch.is_floating_point(parameter)
            or parameter.is_complex()
            or parameter.numel() <= 0
        ):
            raise ProposalStepError(
                f"parameter {name!r} must be a non-empty real strided tensor"
            )
        if not parameter.is_leaf:
            raise ProposalStepError(f"parameter {name!r} must be a leaf tensor")
        if not parameter.requires_grad:
            raise ProposalStepError(
                f"parameter {name!r} must have requires_grad=True"
            )
        if not _tensor_is_finite(parameter.detach()):
            raise ProposalStepError(f"parameter {name!r} contains NaN or Inf")
        names.append(name)
        parameter_ids.append(id(parameter))

    if len(set(names)) != len(names):
        raise ProposalStepError("named parameter names must be unique")
    if len(set(parameter_ids)) != len(parameter_ids):
        raise ProposalStepError("named parameter objects must be unique")
    return materialized


def _tensor_is_finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _tensor_bit_exact(current: Tensor, expected: Tensor) -> bool:
    if (
        current.shape != expected.shape
        or current.dtype != expected.dtype
        or current.device != expected.device
        or current.layout != expected.layout
    ):
        return False
    current_bytes = (
        current.detach()
        .resolve_conj()
        .resolve_neg()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
    )
    expected_bytes = (
        expected.detach()
        .resolve_conj()
        .resolve_neg()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
    )
    return bool(torch.eq(current_bytes, expected_bytes).all().item())


def _global_norm(values: Iterable[Tensor]) -> float:
    component_norms = [
        float(
            torch.linalg.vector_norm(
                value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            ).item()
        )
        for value in values
    ]
    return math.hypot(*component_norms)


def _directional_derivative(
    gradients: tuple[Tensor, ...], directions: tuple[Tensor, ...]
) -> float:
    components = [
        float(
            torch.sum(
                gradient.detach().to(device="cpu", dtype=torch.float64)
                * direction.detach().to(device="cpu", dtype=torch.float64)
            ).item()
        )
        for gradient, direction in zip(gradients, directions, strict=True)
    ]
    return math.fsum(components)


def _snapshot_gradient(parameter: nn.Parameter) -> Tensor | None:
    if parameter.grad is None:
        return None
    if (
        parameter.grad.layout != torch.strided
        or parameter.grad.is_sparse
        or not torch.is_floating_point(parameter.grad)
        or parameter.grad.is_complex()
        or parameter.grad.shape != parameter.shape
        or parameter.grad.dtype != parameter.dtype
        or parameter.grad.device != parameter.device
    ):
        raise ProposalStepError(
            "pre-existing parameter gradients must match their dense parameter state"
        )
    if not _tensor_is_finite(parameter.grad):
        raise ProposalStepError("pre-existing parameter gradients contain NaN or Inf")
    return parameter.grad.detach().clone()


def _restore_gradient(parameter: nn.Parameter, source: Tensor | None) -> None:
    if source is None:
        parameter.grad = None
        return
    if (
        parameter.grad is None
        or parameter.grad.shape != source.shape
        or parameter.grad.dtype != source.dtype
        or parameter.grad.device != source.device
        or parameter.grad.layout != source.layout
    ):
        parameter.grad = source.detach().clone()
        return
    with torch.no_grad():
        parameter.grad.copy_(source)


def _source_state_is_exact(states: tuple[_ParameterState, ...]) -> bool:
    for state in states:
        parameter = state.parameter
        if not parameter.requires_grad:
            return False
        if not _tensor_bit_exact(parameter, state.source_value):
            return False
        if state.source_gradient is None:
            if parameter.grad is not None:
                return False
        elif parameter.grad is None or not _tensor_bit_exact(
            parameter.grad, state.source_gradient
        ):
            return False
    return True


def _restore_states(states: tuple[_ParameterState, ...]) -> None:
    try:
        with torch.no_grad():
            for state in states:
                if not state.parameter.requires_grad:
                    state.parameter.requires_grad_(True)
                state.parameter.copy_(state.source_value)
                _restore_gradient(state.parameter, state.source_gradient)
    except Exception as exc:
        raise ProposalStepError("failed to restore the exact Source snapshot") from exc
    if not _source_state_is_exact(states):
        raise ProposalStepError("restored parameters differ from the Source snapshot")


def _assert_source_exact(states: tuple[_ParameterState, ...], *, where: str) -> None:
    if not _source_state_is_exact(states):
        raise ProposalStepError(f"adaptable state changed {where}")


def _candidate_is_exact(
    states: tuple[_ParameterState, ...], coefficient: float
) -> bool:
    for state in states:
        expected = torch.add(
            state.source_value, state.direction, alpha=coefficient
        )
        if not _tensor_bit_exact(state.parameter, expected):
            return False
        if state.source_gradient is None:
            if state.parameter.grad is not None:
                return False
        elif state.parameter.grad is None or not _tensor_bit_exact(
            state.parameter.grad, state.source_gradient
        ):
            return False
    return True


def _validate_loss_tensor(value: object, *, field_name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ProposalStepError(f"{field_name} must return a torch.Tensor")
    if value.ndim != 0:
        raise ProposalStepError(f"{field_name} must return a scalar tensor")
    if not torch.is_floating_point(value) or value.is_complex():
        raise ProposalStepError(
            f"{field_name} must return a real floating-point tensor"
        )
    return value


def _radius_contract(
    *,
    source_parameter_norm: float,
    relative_radius: float | None,
    absolute_radius: float | None,
    epsilon: float,
) -> tuple[str, float]:
    if (relative_radius is None) == (absolute_radius is None):
        raise ProposalStepError(
            "provide exactly one of relative_radius or absolute_radius"
        )
    if source_parameter_norm > 0.0:
        if relative_radius is None:
            raise ProposalStepError(
                "non-zero Source parameters require relative_radius"
            )
        relative = _finite_positive_float(
            relative_radius, field_name="relative_radius"
        )
        radius = relative * (source_parameter_norm + epsilon)
        mode = "relative"
    else:
        if absolute_radius is None:
            raise ProposalStepError(
                "zero-initialized parameters require absolute_radius"
            )
        radius = _finite_positive_float(
            absolute_radius, field_name="absolute_radius"
        )
        mode = "absolute"
    if not math.isfinite(radius) or radius <= 0.0:
        raise ProposalStepError("effective trust radius must be finite and positive")
    return mode, radius


def _build_rejection(
    reason: str,
    *,
    loss_before: float,
    gradient_norm: float,
    source_parameter_norm: float,
    trust_radius: float,
    radius_mode: str,
) -> ProposalBuildRejected:
    return ProposalBuildRejected(
        reason,
        loss_before=loss_before,
        gradient_norm=gradient_norm,
        source_parameter_norm=source_parameter_norm,
        trust_radius=trust_radius,
        radius_mode=radius_mode,
    )


def build_normalized_proposal(
    *,
    named_parameters: NamedParameters,
    loss_closure: LossClosure,
    relative_radius: float | None,
    absolute_radius: float | None,
    epsilon: float = 1e-12,
) -> NormalizedProposal:
    """Build a finite, non-zero clipped descent direction at Source.

    The supplied names and parameter objects form an exact, non-empty
    topology.  Every parameter must participate in ``loss_closure``; a
    disconnected parameter is a protocol error rather than an implicit zero
    gradient.  A normal scientific no-step (non-finite loss/gradient, exact
    zero gradient, or numerically zero normalized direction) raises
    :class:`ProposalBuildRejected` after restoring Source exactly.
    """

    if not callable(loss_closure):
        raise ProposalStepError("loss_closure must be callable")
    epsilon = _finite_positive_float(epsilon, field_name="epsilon")
    materialized = _materialize_named_parameters(named_parameters)
    source_values = tuple(
        parameter.detach().clone() for _, parameter in materialized
    )
    source_gradients = tuple(
        _snapshot_gradient(parameter) for _, parameter in materialized
    )
    empty_directions = tuple(torch.zeros_like(value) for value in source_values)
    provisional_states = tuple(
        _ParameterState(
            name, parameter, source, gradient, proposal_gradient, direction
        )
        for (
            (name, parameter),
            source,
            gradient,
            proposal_gradient,
            direction,
        ) in zip(
            materialized,
            source_values,
            source_gradients,
            empty_directions,
            empty_directions,
            strict=True,
        )
    )
    source_norm = _global_norm(source_values)
    if not math.isfinite(source_norm):
        raise ProposalStepError("Source parameter norm is non-finite")
    radius_mode, trust_radius = _radius_contract(
        source_parameter_norm=source_norm,
        relative_radius=relative_radius,
        absolute_radius=absolute_radius,
        epsilon=epsilon,
    )

    try:
        loss_tensor = _validate_loss_tensor(
            loss_closure(), field_name="loss_closure"
        )
        _assert_source_exact(
            provisional_states, where="while evaluating the Source loss"
        )
        loss_before = float(loss_tensor.detach().cpu().item())
        if not math.isfinite(loss_before):
            raise _build_rejection(
                "nonfinite_loss_before",
                loss_before=loss_before,
                gradient_norm=math.nan,
                source_parameter_norm=source_norm,
                trust_radius=trust_radius,
                radius_mode=radius_mode,
            )
        if not loss_tensor.requires_grad:
            raise ProposalStepError(
                "loss_closure result must require grad at the Source endpoint"
            )
        try:
            raw_gradients = torch.autograd.grad(
                loss_tensor,
                tuple(parameter for _, parameter in materialized),
                allow_unused=True,
                create_graph=False,
                retain_graph=False,
            )
        except Exception as exc:
            raise ProposalStepError("autograd.grad failed for proposal loss") from exc
        _assert_source_exact(
            provisional_states, where="while computing the Source gradient"
        )
        if any(gradient is None for gradient in raw_gradients):
            missing = tuple(
                name
                for (name, _), gradient in zip(
                    materialized, raw_gradients, strict=True
                )
                if gradient is None
            )
            raise ProposalStepError(
                "loss_closure is disconnected from named parameters: "
                + ", ".join(missing)
            )
        gradients = tuple(
            gradient.detach().clone()
            for gradient in raw_gradients
            if gradient is not None
        )
        if any(not _tensor_is_finite(gradient) for gradient in gradients):
            gradient_norm = _global_norm(gradients)
            raise _build_rejection(
                "nonfinite_gradient",
                loss_before=loss_before,
                gradient_norm=gradient_norm,
                source_parameter_norm=source_norm,
                trust_radius=trust_radius,
                radius_mode=radius_mode,
            )
        gradient_norm = _global_norm(gradients)
        if not math.isfinite(gradient_norm):
            raise _build_rejection(
                "nonfinite_gradient_norm",
                loss_before=loss_before,
                gradient_norm=gradient_norm,
                source_parameter_norm=source_norm,
                trust_radius=trust_radius,
                radius_mode=radius_mode,
            )
        if gradient_norm == 0.0:
            raise _build_rejection(
                "zero_gradient",
                loss_before=loss_before,
                gradient_norm=0.0,
                source_parameter_norm=source_norm,
                trust_radius=trust_radius,
                radius_mode=radius_mode,
            )

        scale = min(1.0, trust_radius / (gradient_norm + epsilon))
        directions = tuple(-gradient * scale for gradient in gradients)
        if any(not _tensor_is_finite(direction) for direction in directions):
            raise _build_rejection(
                "nonfinite_normalized_step",
                loss_before=loss_before,
                gradient_norm=gradient_norm,
                source_parameter_norm=source_norm,
                trust_radius=trust_radius,
                radius_mode=radius_mode,
            )
        normalized_step_norm = _global_norm(directions)
        if normalized_step_norm == 0.0:
            raise _build_rejection(
                "zero_normalized_step",
                loss_before=loss_before,
                gradient_norm=gradient_norm,
                source_parameter_norm=source_norm,
                trust_radius=trust_radius,
                radius_mode=radius_mode,
            )
        directional_derivative = _directional_derivative(
            gradients, directions
        )
        if not math.isfinite(directional_derivative):
            raise _build_rejection(
                "nonfinite_directional_derivative",
                loss_before=loss_before,
                gradient_norm=gradient_norm,
                source_parameter_norm=source_norm,
                trust_radius=trust_radius,
                radius_mode=radius_mode,
            )
        if directional_derivative >= 0.0:
            raise _build_rejection(
                "non_descent_direction",
                loss_before=loss_before,
                gradient_norm=gradient_norm,
                source_parameter_norm=source_norm,
                trust_radius=trust_radius,
                radius_mode=radius_mode,
            )

        states = tuple(
            _ParameterState(
                name,
                parameter,
                source,
                source_gradient,
                gradient,
                direction,
            )
            for (
                (name, parameter),
                source,
                source_gradient,
                gradient,
                direction,
            ) in zip(
                materialized,
                source_values,
                source_gradients,
                gradients,
                directions,
                strict=True,
            )
        )
        _assert_source_exact(states, where="while sealing the proposal")
        return NormalizedProposal(
            parameter_names=tuple(name for name, _ in materialized),
            scalar_parameter_count=sum(
                parameter.numel() for _, parameter in materialized
            ),
            gradient_norm=gradient_norm,
            raw_step_norm=gradient_norm,
            normalized_step_norm=normalized_step_norm,
            source_parameter_norm=source_norm,
            trust_radius=trust_radius,
            radius_mode=radius_mode,
            directional_derivative=directional_derivative,
            loss_before=loss_before,
            _states=states,
        )
    except Exception:
        _restore_states(provisional_states)
        raise


def restore_proposal_source(proposal: NormalizedProposal) -> None:
    """Restore the proposal's named parameters and ``.grad`` state exactly."""

    if not isinstance(proposal, NormalizedProposal):
        raise TypeError("proposal must be NormalizedProposal")
    _restore_states(proposal._states)


def proposal_source_is_exact(proposal: NormalizedProposal) -> bool:
    """Return whether all bound values and gradients equal sealed Source bytes."""

    if not isinstance(proposal, NormalizedProposal):
        raise TypeError("proposal must be NormalizedProposal")
    return _source_state_is_exact(proposal._states)


def export_proposal_vectors(
    proposal: NormalizedProposal,
) -> tuple[Tensor, Tensor]:
    """Return independent CPU-float64 gradient and direction vectors.

    Tensor slices are concatenated in the exact sealed
    ``proposal.parameter_names`` order.  The returned tensors never alias the
    live parameters or the proposal's private payload, so a B3 analyzer may
    serialize or mutate its local copies without changing adaptation state.
    """

    if not isinstance(proposal, NormalizedProposal):
        raise TypeError("proposal must be NormalizedProposal")
    if tuple(state.name for state in proposal._states) != proposal.parameter_names:
        raise ProposalStepError("proposal parameter topology is inconsistent")
    gradients = torch.cat(
        tuple(
            state.proposal_gradient.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1)
            for state in proposal._states
        )
    ).contiguous()
    directions = torch.cat(
        tuple(
            state.direction.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1)
            for state in proposal._states
        )
    ).contiguous()
    if (
        gradients.numel() != proposal.scalar_parameter_count
        or directions.numel() != proposal.scalar_parameter_count
    ):
        raise ProposalStepError("proposal vector length differs from sealed topology")
    if not _tensor_is_finite(gradients) or not _tensor_is_finite(directions):
        raise ProposalStepError("proposal vectors contain NaN or Inf")
    return gradients.clone(), directions.clone()


def apply_proposal_from_source(
    proposal: NormalizedProposal, *, coefficient: float = 1.0
) -> float:
    """Apply ``Source + coefficient * direction`` and return actual L2 delta.

    The Source snapshot is restored before every application.  Thus repeated
    calls never compound candidate increments.
    """

    if not isinstance(proposal, NormalizedProposal):
        raise TypeError("proposal must be NormalizedProposal")
    coefficient = _finite_positive_float(
        coefficient, field_name="coefficient"
    )
    _restore_states(proposal._states)
    expected_values: list[Tensor] = []
    try:
        with torch.no_grad():
            for state in proposal._states:
                candidate = torch.add(
                    state.source_value, state.direction, alpha=coefficient
                )
                expected_values.append(candidate)
                if not _tensor_is_finite(candidate):
                    raise ProposalCandidateRejected(
                        f"nonfinite_candidate_parameter:{state.name}"
                    )
            for state, candidate in zip(
                proposal._states, expected_values, strict=True
            ):
                state.parameter.copy_(candidate)
        if not _candidate_is_exact(proposal._states, coefficient):
            raise ProposalStepError(
                "applied candidate differs from Source plus proposal direction"
            )
        return _global_norm(
            state.parameter.detach().to(device="cpu", dtype=torch.float64)
            - state.source_value.detach().to(device="cpu", dtype=torch.float64)
            for state in proposal._states
        )
    except Exception:
        _restore_states(proposal._states)
        raise


@contextmanager
def applied_proposal(
    proposal: NormalizedProposal, *, coefficient: float = 1.0
) -> Iterator[AppliedProposal]:
    """Temporarily apply one B3 normalized virtual step, then restore Source."""

    actual_step_norm = apply_proposal_from_source(
        proposal, coefficient=coefficient
    )
    try:
        yield AppliedProposal(
            coefficient=float(coefficient), actual_step_norm=actual_step_norm
        )
    finally:
        restore_proposal_source(proposal)


def _normalise_coefficients(
    coefficients: Iterable[Real],
) -> tuple[float, ...]:
    if isinstance(coefficients, (str, bytes, bytearray)):
        raise ProposalStepError("coefficients must be an iterable of numbers")
    try:
        materialized = tuple(coefficients)
    except TypeError as exc:
        raise ProposalStepError("coefficients must be iterable") from exc
    if not materialized:
        raise ProposalStepError("coefficients must not be empty")
    values = tuple(
        _finite_positive_float(value, field_name=f"coefficients[{index}]")
        for index, value in enumerate(materialized)
    )
    if any(value > 1.0 for value in values):
        raise ProposalStepError(
            "backtracking coefficients must not exceed one trust-radius step"
        )
    if any(current >= previous for previous, current in zip(values, values[1:])):
        raise ProposalStepError(
            "coefficients must be strictly decreasing Source-based retries"
        )
    return values


def _safety_decision(value: SafetyClosureResult) -> SafetyDecision:
    if isinstance(value, SafetyDecision):
        return value
    if type(value) is bool:
        return SafetyDecision(
            passed=value, reason="passed" if value else "safety_rejected"
        )
    if isinstance(value, tuple) and len(value) == 2:
        passed, reason = value
        if type(passed) is bool and isinstance(reason, str):
            return SafetyDecision(passed=passed, reason=reason)
    raise ProposalStepError(
        "safety_closure must return bool, SafetyDecision, or (bool, reason)"
    )


def _result_from_build_rejection(
    rejection: ProposalBuildRejected,
    *,
    names: tuple[str, ...],
    scalar_count: int,
) -> ProposalStepResult:
    raw_norm = rejection.gradient_norm
    if not math.isfinite(raw_norm):
        raw_norm = rejection.gradient_norm
    return ProposalStepResult(
        accepted=False,
        backtracking_index=None,
        coefficient=0.0,
        gradient_norm=rejection.gradient_norm,
        raw_step_norm=raw_norm,
        normalized_step_norm=0.0,
        accepted_step_norm=0.0,
        source_parameter_norm=rejection.source_parameter_norm,
        trust_radius=rejection.trust_radius,
        radius_mode=rejection.radius_mode,
        directional_derivative=0.0,
        loss_before=rejection.loss_before,
        loss_after=rejection.loss_before,
        parameter_names=names,
        scalar_parameter_count=scalar_count,
        attempts=(),
        reason=rejection.reason,
    )


def propose_and_backtrack(
    *,
    named_parameters: NamedParameters,
    loss_closure: LossClosure,
    safety_closure: SafetyClosure,
    relative_radius: float | None,
    absolute_radius: float | None,
    coefficients: Iterable[Real] = DEFAULT_BACKTRACKING_COEFFICIENTS,
    armijo_c: float = 1e-4,
    epsilon: float = 1e-12,
) -> ProposalStepResult:
    """Build, backtrack, and possibly retain one explicit Stage-B proposal.

    ``safety_closure`` is invoked only after a candidate has a finite loss and
    passes Armijo.  It must use label-free signals (for example finite logits,
    reliable-background foreground mass, component count, and positive
    fraction guards).  Returning ``False`` rejects that candidate.

    The accepted candidate remains applied for the caller to evaluate.  The
    caller's episodic state manager is responsible for the eventual episode
    reset.  Every non-accepted return leaves the named state exactly at Source.
    """

    if not callable(safety_closure):
        raise ProposalStepError("safety_closure must be callable")
    coefficients = _normalise_coefficients(coefficients)
    armijo_c = _finite_positive_float(armijo_c, field_name="armijo_c")
    if armijo_c >= 1.0:
        raise ProposalStepError("armijo_c must be strictly less than one")

    # Materialize generators exactly once.  The builder re-validates the same
    # topology; the early materialization also supplies metadata for a normal
    # build rejection without touching the closure a second time.
    materialized = _materialize_named_parameters(named_parameters)
    names = tuple(name for name, _ in materialized)
    scalar_count = sum(parameter.numel() for _, parameter in materialized)
    try:
        proposal = build_normalized_proposal(
            named_parameters=materialized,
            loss_closure=loss_closure,
            relative_radius=relative_radius,
            absolute_radius=absolute_radius,
            epsilon=epsilon,
        )
    except ProposalBuildRejected as rejection:
        return _result_from_build_rejection(
            rejection, names=names, scalar_count=scalar_count
        )

    attempts: list[ProposalAttemptResult] = []
    accepted = False
    try:
        for index, coefficient in enumerate(coefficients):
            armijo_rhs = proposal.loss_before + (
                armijo_c
                * coefficient
                * proposal.directional_derivative
            )
            try:
                step_norm = apply_proposal_from_source(
                    proposal, coefficient=coefficient
                )
            except ProposalCandidateRejected as rejection:
                attempts.append(
                    ProposalAttemptResult(
                        index=index,
                        coefficient=coefficient,
                        proposed_step_norm=math.nan,
                        loss_after=None,
                        armijo_rhs=armijo_rhs,
                        parameters_finite=False,
                        loss_finite=False,
                        armijo_passed=False,
                        safety_passed=None,
                        safety_reason="",
                        accepted=False,
                        reason=rejection.reason,
                    )
                )
                continue
            if step_norm == 0.0:
                attempts.append(
                    ProposalAttemptResult(
                        index=index,
                        coefficient=coefficient,
                        proposed_step_norm=0.0,
                        loss_after=None,
                        armijo_rhs=armijo_rhs,
                        parameters_finite=True,
                        loss_finite=False,
                        armijo_passed=False,
                        safety_passed=None,
                        safety_reason="",
                        accepted=False,
                        reason="zero_effective_step",
                    )
                )
                restore_proposal_source(proposal)
                continue

            with torch.no_grad():
                candidate_loss_tensor = _validate_loss_tensor(
                    loss_closure(), field_name="loss_closure"
                )
            if not _candidate_is_exact(proposal._states, coefficient):
                raise ProposalStepError(
                    "adaptable state changed while evaluating candidate loss"
                )
            candidate_loss = float(candidate_loss_tensor.detach().cpu().item())
            if not math.isfinite(candidate_loss):
                attempts.append(
                    ProposalAttemptResult(
                        index=index,
                        coefficient=coefficient,
                        proposed_step_norm=step_norm,
                        loss_after=candidate_loss,
                        armijo_rhs=armijo_rhs,
                        parameters_finite=True,
                        loss_finite=False,
                        armijo_passed=False,
                        safety_passed=None,
                        safety_reason="",
                        accepted=False,
                        reason="nonfinite_candidate_loss",
                    )
                )
                restore_proposal_source(proposal)
                continue
            if not math.isfinite(armijo_rhs):
                attempts.append(
                    ProposalAttemptResult(
                        index=index,
                        coefficient=coefficient,
                        proposed_step_norm=step_norm,
                        loss_after=candidate_loss,
                        armijo_rhs=armijo_rhs,
                        parameters_finite=True,
                        loss_finite=True,
                        armijo_passed=False,
                        safety_passed=None,
                        safety_reason="",
                        accepted=False,
                        reason="nonfinite_armijo_bound",
                    )
                )
                restore_proposal_source(proposal)
                continue
            armijo_passed = candidate_loss <= armijo_rhs
            if not armijo_passed:
                attempts.append(
                    ProposalAttemptResult(
                        index=index,
                        coefficient=coefficient,
                        proposed_step_norm=step_norm,
                        loss_after=candidate_loss,
                        armijo_rhs=armijo_rhs,
                        parameters_finite=True,
                        loss_finite=True,
                        armijo_passed=False,
                        safety_passed=None,
                        safety_reason="",
                        accepted=False,
                        reason="armijo_rejected",
                    )
                )
                restore_proposal_source(proposal)
                continue

            with torch.no_grad():
                safety = _safety_decision(safety_closure())
            if not _candidate_is_exact(proposal._states, coefficient):
                raise ProposalStepError(
                    "adaptable state changed while evaluating candidate safety"
                )
            if not safety.passed:
                attempts.append(
                    ProposalAttemptResult(
                        index=index,
                        coefficient=coefficient,
                        proposed_step_norm=step_norm,
                        loss_after=candidate_loss,
                        armijo_rhs=armijo_rhs,
                        parameters_finite=True,
                        loss_finite=True,
                        armijo_passed=True,
                        safety_passed=False,
                        safety_reason=safety.reason,
                        accepted=False,
                        reason="safety_rejected",
                    )
                )
                restore_proposal_source(proposal)
                continue

            attempts.append(
                ProposalAttemptResult(
                    index=index,
                    coefficient=coefficient,
                    proposed_step_norm=step_norm,
                    loss_after=candidate_loss,
                    armijo_rhs=armijo_rhs,
                    parameters_finite=True,
                    loss_finite=True,
                    armijo_passed=True,
                    safety_passed=True,
                    safety_reason=safety.reason,
                    accepted=True,
                    reason="accepted",
                )
            )
            accepted = True
            return ProposalStepResult(
                accepted=True,
                backtracking_index=index,
                coefficient=coefficient,
                gradient_norm=proposal.gradient_norm,
                raw_step_norm=proposal.raw_step_norm,
                normalized_step_norm=proposal.normalized_step_norm,
                accepted_step_norm=step_norm,
                source_parameter_norm=proposal.source_parameter_norm,
                trust_radius=proposal.trust_radius,
                radius_mode=proposal.radius_mode,
                directional_derivative=proposal.directional_derivative,
                loss_before=proposal.loss_before,
                loss_after=candidate_loss,
                parameter_names=proposal.parameter_names,
                scalar_parameter_count=proposal.scalar_parameter_count,
                attempts=tuple(attempts),
                reason="accepted",
            )

        restore_proposal_source(proposal)
        return ProposalStepResult(
            accepted=False,
            backtracking_index=None,
            coefficient=0.0,
            gradient_norm=proposal.gradient_norm,
            raw_step_norm=proposal.raw_step_norm,
            normalized_step_norm=proposal.normalized_step_norm,
            accepted_step_norm=0.0,
            source_parameter_norm=proposal.source_parameter_norm,
            trust_radius=proposal.trust_radius,
            radius_mode=proposal.radius_mode,
            directional_derivative=proposal.directional_derivative,
            loss_before=proposal.loss_before,
            loss_after=proposal.loss_before,
            parameter_names=proposal.parameter_names,
            scalar_parameter_count=proposal.scalar_parameter_count,
            attempts=tuple(attempts),
            reason="all_candidates_rejected",
        )
    except Exception:
        restore_proposal_source(proposal)
        raise
    finally:
        if not accepted and not proposal_source_is_exact(proposal):
            restore_proposal_source(proposal)


__all__ = [
    "AppliedProposal",
    "DEFAULT_BACKTRACKING_COEFFICIENTS",
    "NormalizedProposal",
    "ProposalAttemptResult",
    "ProposalBuildRejected",
    "ProposalCandidateRejected",
    "ProposalStepError",
    "ProposalStepResult",
    "SafetyDecision",
    "applied_proposal",
    "apply_proposal_from_source",
    "build_normalized_proposal",
    "export_proposal_vectors",
    "proposal_source_is_exact",
    "propose_and_backtrack",
    "restore_proposal_source",
]
