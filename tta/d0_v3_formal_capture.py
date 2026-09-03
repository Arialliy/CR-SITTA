"""Lossless first-step tensors for formal train-Pilot D0 diagnostics.

This observer is deliberately separate from :mod:`tta.d0_v2_native_step`.
The latter is part of an already published engineering-smoke seal.  Formal
P3 needs the actual gradient and parameter endpoint tensors (not only their
hashes) so that a later, label-isolated outer evaluator can compute
per-module alignment and metric effects.

The intended nesting is::

    with NativeFirstStepObserver(optimizer, ...) as native:
        with FormalFirstStepCapture(optimizer, ...) as capture:
            runner.run_one_image(...)

``FormalFirstStepCapture`` therefore wraps the already installed native-step
observer, snapshots the live tensors immediately around that one real
``optimizer.step()``, and leaves both the optimizer implementation and all
gradients unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn


class D0V3FormalCaptureError(RuntimeError):
    """The formal first-step tensor capture contract was violated."""


NamedParameters = Sequence[tuple[str, nn.Parameter]]


def _materialize_named_parameters(
    value: NamedParameters,
) -> tuple[tuple[str, nn.Parameter], ...]:
    if isinstance(value, (str, bytes)):
        raise D0V3FormalCaptureError("named_parameters must be a sequence")
    result = tuple(value)
    if not result:
        raise D0V3FormalCaptureError("named_parameters cannot be empty")
    names: list[str] = []
    objects: list[nn.Parameter] = []
    for index, item in enumerate(result):
        if not isinstance(item, tuple) or len(item) != 2:
            raise D0V3FormalCaptureError(
                f"named_parameters[{index}] must be a (name, Parameter) tuple"
            )
        name, parameter = item
        if not isinstance(name, str) or not name:
            raise D0V3FormalCaptureError(
                f"named_parameters[{index}] has an invalid name"
            )
        if not isinstance(parameter, nn.Parameter):
            raise D0V3FormalCaptureError(
                f"named_parameters[{index}] is not an nn.Parameter"
            )
        names.append(name)
        objects.append(parameter)
    if len(set(names)) != len(names):
        raise D0V3FormalCaptureError("parameter names must be unique")
    if len({id(value) for value in objects}) != len(objects):
        raise D0V3FormalCaptureError("parameter objects must be unique")
    return result


def _clone_cpu(value: Tensor, *, label: str) -> Tensor:
    if value.layout != torch.strided or not torch.is_floating_point(value):
        raise D0V3FormalCaptureError(
            f"{label} must be a strided floating-point tensor"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise D0V3FormalCaptureError(f"{label} contains NaN/Inf")
    return value.detach().cpu().contiguous().clone()


def _storage_identity(value: Tensor) -> tuple[str, int, int, int, str]:
    return (
        str(value.device),
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
        int(value.numel()),
        str(value.dtype),
    )


@dataclass(frozen=True)
class FormalFirstStepTensors:
    """Immutable CPU snapshots plus live-storage ownership evidence."""

    parameter_names: tuple[str, ...]
    parameters_before: tuple[tuple[str, Tensor], ...]
    gradients: tuple[tuple[str, Tensor], ...]
    parameters_after: tuple[tuple[str, Tensor], ...]
    adaptation_step: tuple[tuple[str, Tensor], ...]
    parameter_live_storage: tuple[tuple[str, tuple[str, int, int, int, str]], ...]
    gradient_live_storage: tuple[tuple[str, tuple[str, int, int, int, str]], ...]
    optimizer_state_live_storage: tuple[
        tuple[str, str, tuple[str, int, int, int, str]], ...
    ]
    step_norm_l2: float

    def before_dict(self) -> dict[str, Tensor]:
        return dict(self.parameters_before)

    def gradient_dict(self) -> dict[str, Tensor]:
        return dict(self.gradients)

    def after_dict(self) -> dict[str, Tensor]:
        return dict(self.parameters_after)

    def step_dict(self) -> dict[str, Tensor]:
        return dict(self.adaptation_step)


class FormalFirstStepCapture:
    """One-shot wrapper capturing tensors around an actual optimizer step."""

    _MARKER = "_cr_sitta_d0_v3_formal_first_step_capture"

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        named_parameters: NamedParameters,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise D0V3FormalCaptureError("optimizer must be torch.optim.Optimizer")
        self._optimizer = optimizer
        self._named_parameters = _materialize_named_parameters(named_parameters)
        self._entered = False
        self._closed = False
        self._step_count = 0
        self._original_bound_step: Callable[..., Any] | None = None
        self._had_instance_step = False
        self._original_instance_step: Any = None
        self._tensors: FormalFirstStepTensors | None = None

    @property
    def tensors(self) -> FormalFirstStepTensors:
        if self._tensors is None:
            raise D0V3FormalCaptureError(
                "formal first-step tensor capture is not complete"
            )
        return self._tensors

    def __enter__(self) -> "FormalFirstStepCapture":
        if self._entered or self._closed:
            raise D0V3FormalCaptureError("formal capture cannot be reused")
        if hasattr(self._optimizer, self._MARKER):
            raise D0V3FormalCaptureError("optimizer is already formally captured")
        self._entered = True
        self._had_instance_step = "step" in self._optimizer.__dict__
        self._original_instance_step = self._optimizer.__dict__.get("step")
        self._original_bound_step = self._optimizer.step
        setattr(self._optimizer, self._MARKER, self)
        self._optimizer.step = self._captured_step  # type: ignore[method-assign]
        return self

    def _captured_step(self, *args: Any, **kwargs: Any) -> Any:
        if not self._entered or self._closed:
            raise D0V3FormalCaptureError("formal capture is not active")
        if self._step_count != 0:
            raise D0V3FormalCaptureError(
                "formal candidate must call optimizer.step exactly once"
            )
        if args or kwargs:
            raise D0V3FormalCaptureError(
                "formal first step forbids closures and step arguments"
            )
        if self._optimizer.state:
            raise D0V3FormalCaptureError(
                "formal candidate optimizer state must be empty before step"
            )
        self._step_count = 1

        before: list[tuple[str, Tensor]] = []
        gradients: list[tuple[str, Tensor]] = []
        parameter_storage: list[
            tuple[str, tuple[str, int, int, int, str]]
        ] = []
        gradient_storage: list[
            tuple[str, tuple[str, int, int, int, str]]
        ] = []
        for name, parameter in self._named_parameters:
            gradient = parameter.grad
            if gradient is None:
                raise D0V3FormalCaptureError(
                    f"formal candidate gradient is missing: {name}"
                )
            before.append((name, _clone_cpu(parameter, label=f"before.{name}")))
            gradients.append((name, _clone_cpu(gradient, label=f"gradient.{name}")))
            parameter_storage.append((name, _storage_identity(parameter)))
            gradient_storage.append((name, _storage_identity(gradient)))

        if self._original_bound_step is None:
            raise D0V3FormalCaptureError("wrapped optimizer step is unavailable")
        result = self._original_bound_step()

        after: list[tuple[str, Tensor]] = []
        steps: list[tuple[str, Tensor]] = []
        squared_norms: list[float] = []
        for (name, parameter), (before_name, before_tensor) in zip(
            self._named_parameters, before, strict=True
        ):
            if name != before_name:
                raise D0V3FormalCaptureError("parameter order changed during step")
            after_tensor = _clone_cpu(parameter, label=f"after.{name}")
            step_tensor = after_tensor - before_tensor
            if not bool(torch.isfinite(step_tensor).all().item()):
                raise D0V3FormalCaptureError(
                    f"formal adaptation step contains NaN/Inf: {name}"
                )
            after.append((name, after_tensor))
            steps.append((name, step_tensor))
            squared_norms.append(
                float(
                    torch.sum(
                        step_tensor.to(dtype=torch.float64).square()
                    ).item()
                )
            )

        state_storage: list[
            tuple[str, str, tuple[str, int, int, int, str]]
        ] = []
        parameter_to_name = {
            id(parameter): name for name, parameter in self._named_parameters
        }
        if set(map(id, self._optimizer.state)) != set(parameter_to_name):
            raise D0V3FormalCaptureError(
                "optimizer state parameter ownership differs after step"
            )
        for parameter, state in self._optimizer.state.items():
            name = parameter_to_name[id(parameter)]
            if not isinstance(state, dict) or not state:
                raise D0V3FormalCaptureError(
                    f"optimizer state is empty or invalid after step: {name}"
                )
            for state_name, state_value in sorted(state.items()):
                if not isinstance(state_name, str) or not isinstance(
                    state_value, Tensor
                ):
                    raise D0V3FormalCaptureError(
                        f"optimizer state tensor schema differs: {name}"
                    )
                if not bool(torch.isfinite(state_value).all().item()):
                    raise D0V3FormalCaptureError(
                        f"optimizer state contains NaN/Inf: {name}.{state_name}"
                    )
                state_storage.append(
                    (name, state_name, _storage_identity(state_value))
                )

        step_norm = math.sqrt(math.fsum(squared_norms))
        if not math.isfinite(step_norm) or step_norm < 0.0:
            raise D0V3FormalCaptureError("formal step norm is invalid")
        self._tensors = FormalFirstStepTensors(
            parameter_names=tuple(name for name, _ in self._named_parameters),
            parameters_before=tuple(before),
            gradients=tuple(gradients),
            parameters_after=tuple(after),
            adaptation_step=tuple(steps),
            parameter_live_storage=tuple(parameter_storage),
            gradient_live_storage=tuple(gradient_storage),
            optimizer_state_live_storage=tuple(state_storage),
            step_norm_l2=step_norm,
        )
        return result

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if not self._entered or self._closed:
            raise D0V3FormalCaptureError("formal capture exit state is invalid")
        try:
            if self._had_instance_step:
                self._optimizer.step = self._original_instance_step
            else:
                self._optimizer.__dict__.pop("step", None)
            if getattr(self._optimizer, self._MARKER, None) is self:
                delattr(self._optimizer, self._MARKER)
        finally:
            self._closed = True
        if exc_type is None:
            if self._step_count != 1 or self._tensors is None:
                raise D0V3FormalCaptureError(
                    "formal capture ended without exactly one completed step"
                )
        return False


__all__ = [
    "D0V3FormalCaptureError",
    "FormalFirstStepCapture",
    "FormalFirstStepTensors",
]
