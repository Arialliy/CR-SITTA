"""Audited Stage-B4 proposal execution with loss-gradient capture.

The low-level :mod:`tta.proposal_step` module owns the proposal, Armijo
backtracking, and exact Source restoration semantics.  This module adds the
piece needed by the label-isolated outer evaluator: a lossless, ordered copy
of the *same* proxy gradient consumed by that low-level proposal call.

Temporary leaf hooks observe the single ``torch.autograd.grad`` performed by
``proposal_step.propose_and_backtrack``.  They return gradients unchanged and
are removed on every exit path.  The returned vectors are independent,
contiguous CPU ``float64`` snapshots in the exact named-parameter order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import TypeAlias

import torch
from torch import Tensor, nn

from .proposal_step import (
    DEFAULT_BACKTRACKING_COEFFICIENTS,
    ProposalStepError,
    ProposalStepResult,
    SafetyDecision,
    propose_and_backtrack as _propose_and_backtrack,
)


NamedParameters: TypeAlias = (
    Mapping[str, nn.Parameter] | Iterable[tuple[str, nn.Parameter]]
)
LossClosure: TypeAlias = Callable[[], Tensor]
SafetyClosureResult: TypeAlias = SafetyDecision | bool | tuple[bool, str]
SafetyClosure: TypeAlias = Callable[[], SafetyClosureResult]


class ProposalRunnerError(ProposalStepError):
    """The audited proposal-execution contract was violated."""


@dataclass(frozen=True, init=False)
class ProposalExecution:
    """Immutable proposal result and detached outer-evaluation vectors.

    The tensor payload is kept private and each public vector access returns a
    clone.  A caller may therefore reshape, serialize, or mutate its local
    tensor without changing the sealed execution record.
    """

    result: ProposalStepResult
    _proxy_gradient: Tensor = field(repr=False, compare=False)
    _normalized_direction: Tensor = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        result: ProposalStepResult,
        proxy_gradient: Tensor,
        normalized_direction: Tensor,
    ) -> None:
        if not isinstance(result, ProposalStepResult):
            raise TypeError("result must be ProposalStepResult")
        gradient = _seal_vector(proxy_gradient, field_name="proxy_gradient")
        direction = _seal_vector(
            normalized_direction, field_name="normalized_direction"
        )
        if gradient.shape != direction.shape:
            raise ProposalRunnerError(
                "proxy gradient and normalized direction shapes differ"
            )
        if gradient.numel() != result.scalar_parameter_count:
            raise ProposalRunnerError(
                "execution vector length differs from result topology"
            )
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "_proxy_gradient", gradient)
        object.__setattr__(self, "_normalized_direction", direction)

    @property
    def proxy_gradient(self) -> Tensor:
        """Return an independent ordered CPU-float64 proxy-gradient vector."""

        return self._proxy_gradient.clone()

    @property
    def normalized_direction(self) -> Tensor:
        """Return an independent ordered CPU-float64 clipped direction."""

        return self._normalized_direction.clone()


@dataclass(frozen=True)
class _SourceState:
    name: str
    parameter: nn.Parameter = field(repr=False, compare=False)
    value: Tensor = field(repr=False, compare=False)
    gradient: Tensor | None = field(repr=False, compare=False)


def _seal_vector(value: Tensor, *, field_name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.layout != torch.strided
        or value.ndim != 1
        or not value.is_contiguous()
    ):
        raise ProposalRunnerError(
            f"{field_name} must be a contiguous one-dimensional CPU float64 tensor"
        )
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise ProposalRunnerError(f"{field_name} contains NaN or Inf")
    return value.detach().clone().contiguous()


def _materialize_named_parameters(
    named_parameters: NamedParameters,
) -> tuple[tuple[str, nn.Parameter], ...]:
    if isinstance(named_parameters, Mapping):
        try:
            materialized = tuple(
                (name, named_parameters[name]) for name in sorted(named_parameters)
            )
        except Exception as exc:
            raise ProposalRunnerError(
                "named-parameter mapping could not be ordered"
            ) from exc
    else:
        if isinstance(named_parameters, (str, bytes, bytearray)):
            raise ProposalRunnerError(
                "named_parameters must contain (name, nn.Parameter) pairs"
            )
        try:
            materialized = tuple(named_parameters)
        except TypeError as exc:
            raise ProposalRunnerError(
                "named_parameters must be a mapping or iterable of pairs"
            ) from exc

    if not materialized:
        raise ProposalRunnerError("named_parameters must not be empty")

    names: list[str] = []
    parameter_ids: list[int] = []
    for index, item in enumerate(materialized):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ProposalRunnerError(
                f"named_parameters[{index}] must be a (name, parameter) pair"
            )
        name, parameter = item
        if not isinstance(name, str) or not name or "\0" in name:
            raise ProposalRunnerError(
                f"named_parameters[{index}] has an invalid name"
            )
        if not isinstance(parameter, nn.Parameter):
            raise ProposalRunnerError(
                f"named_parameters[{index}] must contain nn.Parameter"
            )
        if (
            parameter.layout != torch.strided
            or parameter.is_sparse
            or not torch.is_floating_point(parameter)
            or parameter.is_complex()
            or parameter.numel() <= 0
        ):
            raise ProposalRunnerError(
                f"parameter {name!r} must be a non-empty real strided tensor"
            )
        if not parameter.is_leaf:
            raise ProposalRunnerError(f"parameter {name!r} must be a leaf tensor")
        if not parameter.requires_grad:
            raise ProposalRunnerError(
                f"parameter {name!r} must have requires_grad=True"
            )
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise ProposalRunnerError(f"parameter {name!r} contains NaN or Inf")
        names.append(name)
        parameter_ids.append(id(parameter))

    if len(set(names)) != len(names):
        raise ProposalRunnerError("named parameter names must be unique")
    if len(set(parameter_ids)) != len(parameter_ids):
        raise ProposalRunnerError("named parameter objects must be unique")
    return materialized


def _snapshot_source(
    materialized: tuple[tuple[str, nn.Parameter], ...],
) -> tuple[_SourceState, ...]:
    states: list[_SourceState] = []
    for name, parameter in materialized:
        gradient = parameter.grad
        if gradient is not None:
            if (
                gradient.layout != torch.strided
                or gradient.is_sparse
                or not torch.is_floating_point(gradient)
                or gradient.is_complex()
                or gradient.shape != parameter.shape
                or gradient.dtype != parameter.dtype
                or gradient.device != parameter.device
                or not bool(torch.isfinite(gradient.detach()).all().item())
            ):
                raise ProposalRunnerError(
                    f"pre-existing gradient for {name!r} is not a finite dense "
                    "parameter-shaped tensor"
                )
            gradient = gradient.detach().clone()
        states.append(
            _SourceState(
                name=name,
                parameter=parameter,
                value=parameter.detach().clone(),
                gradient=gradient,
            )
        )
    return tuple(states)


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


def _gradient_is_exact(state: _SourceState) -> bool:
    current = state.parameter.grad
    if state.gradient is None:
        return current is None
    return current is not None and _tensor_bit_exact(current, state.gradient)


def _restore_source(states: tuple[_SourceState, ...]) -> None:
    try:
        with torch.no_grad():
            for state in states:
                state.parameter.copy_(state.value)
                if state.gradient is None:
                    state.parameter.grad = None
                elif (
                    state.parameter.grad is None
                    or state.parameter.grad.shape != state.gradient.shape
                    or state.parameter.grad.dtype != state.gradient.dtype
                    or state.parameter.grad.device != state.gradient.device
                    or state.parameter.grad.layout != state.gradient.layout
                ):
                    state.parameter.grad = state.gradient.detach().clone()
                else:
                    state.parameter.grad.copy_(state.gradient)
    except Exception as exc:
        raise ProposalRunnerError(
            "failed to restore wrapper Source snapshot"
        ) from exc
    if any(
        not _tensor_bit_exact(state.parameter, state.value)
        or not _gradient_is_exact(state)
        for state in states
    ):
        raise ProposalRunnerError("wrapper Source restoration is not bit-exact")


def _captured_gradient_hook(
    *,
    name: str,
    parameter: nn.Parameter,
    captured: dict[str, Tensor],
    capture_counts: dict[str, int],
    on_capture_complete: Callable[[], None] | None = None,
) -> Callable[[Tensor], Tensor]:
    def capture(gradient: Tensor) -> Tensor:
        capture_counts[name] += 1
        if capture_counts[name] != 1:
            raise ProposalRunnerError(
                f"proxy gradient was produced more than once for {name!r}"
            )
        if not isinstance(gradient, Tensor):
            raise ProposalRunnerError(
                f"proxy gradient for {name!r} is not a tensor"
            )
        if (
            gradient.layout != torch.strided
            or gradient.is_sparse
            or not torch.is_floating_point(gradient)
            or gradient.is_complex()
            or gradient.shape != parameter.shape
            or gradient.dtype != parameter.dtype
            or gradient.device != parameter.device
        ):
            raise ProposalRunnerError(
                f"proxy gradient topology differs for {name!r}"
            )
        captured[name] = gradient.detach().clone().contiguous()
        if on_capture_complete is not None and all(
            count == 1 for count in capture_counts.values()
        ):
            # All requested leaf gradients are now materialized, so the known
            # CUDA backward has crossed its final parameter boundary.  Restore
            # before returning control to proposal construction.
            on_capture_complete()
        # Observation only: never replace, scale, or detach the live gradient.
        return gradient

    return capture


def _has_cuda_parameters(
    materialized: tuple[tuple[str, nn.Parameter], ...],
) -> bool:
    return any(parameter.device.type == "cuda" for _, parameter in materialized)


class _CudaBackwardDeterminismWindow:
    """Bound the known nondeterministic CUDA allowance to gradient building.

    ``proposal_step.propose_and_backtrack`` deliberately exposes a monolithic
    public call.  Its first loss-closure invocation builds the Source graph and
    is followed by its only ``autograd.grad``.  A transparent autograd root
    opens this controller when that backward actually begins.  Later closure
    wrappers restore it before evaluating an Armijo candidate or its safety
    checks.  No model forward executes under the temporary override.
    """

    def __init__(
        self,
        materialized: tuple[tuple[str, nn.Parameter], ...],
        *,
        allowed: bool,
    ) -> None:
        if type(allowed) is not bool:
            raise ProposalRunnerError(
                "allow_cuda_nondeterministic_backward must be bool"
            )
        self._enabled = allowed and _has_cuda_parameters(materialized)
        self._enabled_before = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        self._warn_only_before = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        self._active = False

    def open(self) -> None:
        if self._active:
            raise ProposalRunnerError(
                "CUDA backward determinism window is already active"
            )
        if not self._enabled:
            return
        torch.use_deterministic_algorithms(False)
        self._active = True

    def restore(self) -> None:
        if not self._active:
            return
        try:
            torch.use_deterministic_algorithms(
                self._enabled_before, warn_only=self._warn_only_before
            )
        finally:
            self._active = False
        if (
            bool(torch.are_deterministic_algorithms_enabled())
            != self._enabled_before
            or bool(torch.is_deterministic_algorithms_warn_only_enabled())
            != self._warn_only_before
        ):
            raise ProposalRunnerError(
                "CUDA backward determinism policy was not restored"
            )


class _OpenCudaBackwardWindow(torch.autograd.Function):
    """Identity whose backward opens the tightly scoped CUDA allowance."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        loss: Tensor,
        controller: _CudaBackwardDeterminismWindow,
    ) -> Tensor:
        # ``controller`` is intentionally non-tensor state carried only to the
        # root backward node.  Returning a view preserves scalar dtype/device
        # and connectivity without executing any model operation.
        setattr(ctx, "controller", controller)
        return loss.view_as(loss)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: object, gradient: Tensor
    ) -> tuple[Tensor, None]:
        controller = getattr(ctx, "controller")
        if not isinstance(controller, _CudaBackwardDeterminismWindow):
            raise ProposalRunnerError(
                "CUDA backward determinism controller topology is invalid"
            )
        controller.open()
        return gradient, None


def _open_cuda_window_on_backward(
    loss: object, controller: _CudaBackwardDeterminismWindow
) -> object:
    """Attach the policy switch without weakening Source-forward execution."""

    if not isinstance(loss, Tensor) or not loss.requires_grad:
        # Let the bound low-level validator produce its canonical type/graph
        # error.  In particular, a non-finite loss is rejected without ever
        # opening the allowance because no backward is attempted.
        return loss
    return _OpenCudaBackwardWindow.apply(loss, controller)


def _global_norm(values: Iterable[Tensor]) -> float:
    component_norms = tuple(
        float(
            torch.linalg.vector_norm(
                value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            ).item()
        )
        for value in values
    )
    return math.hypot(*component_norms)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def _build_execution(
    *,
    result: ProposalStepResult,
    materialized: tuple[tuple[str, nn.Parameter], ...],
    captured: dict[str, Tensor],
    capture_counts: dict[str, int],
    epsilon: float,
) -> tuple[ProposalExecution, tuple[Tensor, ...]]:
    names = tuple(name for name, _ in materialized)
    scalar_count = sum(parameter.numel() for _, parameter in materialized)
    if (
        result.parameter_names != names
        or result.scalar_parameter_count != scalar_count
    ):
        raise ProposalRunnerError(
            "proposal result topology differs from requested named parameters"
        )
    if set(captured) != set(names) or any(
        capture_counts[name] != 1 for name in names
    ):
        missing = tuple(name for name in names if capture_counts[name] == 0)
        repeated = tuple(name for name in names if capture_counts[name] > 1)
        raise ProposalRunnerError(
            "proxy-gradient capture topology differs: "
            f"missing={missing}, repeated={repeated}"
        )

    native_gradients = tuple(captured[name] for name in names)
    if any(
        gradient.shape != parameter.shape
        or gradient.dtype != parameter.dtype
        or gradient.device != parameter.device
        or not bool(torch.isfinite(gradient).all().item())
        for gradient, (_, parameter) in zip(
            native_gradients, materialized, strict=True
        )
    ):
        raise ProposalRunnerError(
            "captured proxy gradients are non-finite or violate topology"
        )

    gradient_norm = _global_norm(native_gradients)
    if not math.isfinite(gradient_norm):
        raise ProposalRunnerError("captured proxy-gradient norm is non-finite")
    if not _close(gradient_norm, result.gradient_norm):
        raise ProposalRunnerError(
            "captured proxy-gradient norm differs from ProposalStepResult"
        )
    if not _close(gradient_norm, result.raw_step_norm):
        raise ProposalRunnerError(
            "captured raw-step norm differs from ProposalStepResult"
        )

    scale = min(1.0, result.trust_radius / (gradient_norm + epsilon))
    native_directions = tuple(-gradient * scale for gradient in native_gradients)
    if any(
        not bool(torch.isfinite(direction).all().item())
        for direction in native_directions
    ):
        raise ProposalRunnerError("normalized direction contains NaN or Inf")
    direction_norm = _global_norm(native_directions)
    if not _close(direction_norm, result.normalized_step_norm):
        raise ProposalRunnerError(
            "normalized-direction norm differs from ProposalStepResult"
        )
    derivative = math.fsum(
        float(
            torch.sum(
                gradient.detach().to(device="cpu", dtype=torch.float64)
                * direction.detach().to(device="cpu", dtype=torch.float64)
            ).item()
        )
        for gradient, direction in zip(
            native_gradients, native_directions, strict=True
        )
    )
    if not _close(derivative, result.directional_derivative):
        raise ProposalRunnerError(
            "directional derivative differs from ProposalStepResult"
        )

    flat_gradient = torch.cat(
        tuple(
            gradient.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1)
            for gradient in native_gradients
        )
    ).contiguous()
    flat_direction = torch.cat(
        tuple(
            direction.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1)
            for direction in native_directions
        )
    ).contiguous()
    execution = ProposalExecution(
        result=result,
        proxy_gradient=flat_gradient,
        normalized_direction=flat_direction,
    )
    return execution, native_directions


def _assert_terminal_state(
    *,
    result: ProposalStepResult,
    states: tuple[_SourceState, ...],
    native_directions: tuple[Tensor, ...],
) -> None:
    if any(not _gradient_is_exact(state) for state in states):
        raise ProposalRunnerError("proposal execution polluted parameter .grad state")

    if not result.accepted:
        if any(
            not _tensor_bit_exact(state.parameter, state.value) for state in states
        ):
            raise ProposalRunnerError(
                "rejected proposal did not leave exact Source parameters"
            )
        return

    if result.backtracking_index is None or result.coefficient <= 0.0:
        raise ProposalRunnerError("accepted result has invalid backtracking metadata")
    for state, direction in zip(states, native_directions, strict=True):
        expected = torch.add(state.value, direction, alpha=result.coefficient)
        if not _tensor_bit_exact(state.parameter, expected):
            raise ProposalRunnerError(
                f"accepted endpoint differs from Source plus direction: {state.name}"
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
    allow_cuda_nondeterministic_backward: bool = False,
) -> ProposalExecution:
    """Execute one Source-anchored proposal and capture its exact gradient.

    This function has the same explicit optimization inputs as the low-level
    proposal routine.  Accepted endpoints remain applied.  Every rejected or
    exceptional execution is restored to the wrapper's exact Source snapshot;
    temporary gradient hooks are always removed.
    """

    if not callable(loss_closure):
        raise ProposalRunnerError("loss_closure must be callable")
    if not callable(safety_closure):
        raise ProposalRunnerError("safety_closure must be callable")

    materialized = _materialize_named_parameters(named_parameters)
    source_states = _snapshot_source(materialized)
    captured: dict[str, Tensor] = {}
    capture_counts = {name: 0 for name, _ in materialized}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    determinism_window = _CudaBackwardDeterminismWindow(
        materialized,
        allowed=allow_cuda_nondeterministic_backward,
    )
    source_loss_pending = True

    def controlled_loss_closure() -> object:
        nonlocal source_loss_pending
        if source_loss_pending:
            source_loss_pending = False
            source_loss = loss_closure()
            return _open_cuda_window_on_backward(
                source_loss, determinism_window
            )
        # This is an Armijo-candidate forward.  The normal successful path has
        # already restored in the last parameter hook; the explicit call is a
        # defensive boundary for incomplete/error topology paths.
        determinism_window.restore()
        return loss_closure()

    def controlled_safety_closure() -> SafetyClosureResult:
        # Safety may itself run a model forward, so retain the same hard policy
        # boundary even if a future low-level implementation changes ordering.
        determinism_window.restore()
        return safety_closure()

    try:
        for name, parameter in materialized:
            handles.append(
                parameter.register_hook(
                    _captured_gradient_hook(
                        name=name,
                        parameter=parameter,
                        captured=captured,
                        capture_counts=capture_counts,
                        on_capture_complete=determinism_window.restore,
                    )
                )
            )

        try:
            result = _propose_and_backtrack(
                named_parameters=materialized,
                loss_closure=controlled_loss_closure,
                safety_closure=controlled_safety_closure,
                relative_radius=relative_radius,
                absolute_radius=absolute_radius,
                coefficients=coefficients,
                armijo_c=armijo_c,
                epsilon=epsilon,
            )
        finally:
            # Covers autograd failure, disconnected/zero/non-finite gradients,
            # closure errors, and candidate paths that never reach safety.
            determinism_window.restore()
        # The low-level function has validated epsilon on every normal return.
        epsilon_value = float(epsilon)
        execution, native_directions = _build_execution(
            result=result,
            materialized=materialized,
            captured=captured,
            capture_counts=capture_counts,
            epsilon=epsilon_value,
        )
        _assert_terminal_state(
            result=result,
            states=source_states,
            native_directions=native_directions,
        )
        return execution
    except BaseException:
        _restore_source(source_states)
        raise
    finally:
        for handle in reversed(handles):
            handle.remove()


__all__ = [
    "ProposalExecution",
    "ProposalRunnerError",
    "propose_and_backtrack",
]
