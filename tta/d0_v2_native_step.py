"""Same-device, native-dtype first-step hard gate for D0-v2.

The sealed D0-v1 runner must not be imported to obtain its private optimizer
checks.  This module independently observes the *actual* ``optimizer.step``
call used by one D0-v2 candidate.  Immediately inside that call it:

* binds the runtime to torch 2.1.2 and the exact frozen Adam/SGD defaults;
* requires one ordered parameter group and an empty optimizer state;
* exposes the live ``parameter.grad`` objects to an optional ownership-ledger
  callback (never clones in place of those objects);
* builds the public same-device/native-dtype PyTorch first-step reference;
* executes the original optimizer step exactly once; and
* verifies every parameter endpoint and optimizer-state tensor bit-exactly.

The returned observation is engineering evidence only.  It does not perform
candidate selection and it does not authorize any later scientific stage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor, nn

from analysis.analyze_tent_optimizer_geometry import (
    OptimizerFirstStepSpec,
    OptimizerGeometryError,
    PYTORCH_REFERENCE_VERSION as GEOMETRY_PYTORCH_REFERENCE_VERSION,
    pytorch_first_step_reference,
)


SCHEMA_VERSION = 2
PROTOCOL_ID = "cr-sitta-d0-v2-same-device-native-first-step"
REFERENCE_NAME = (
    "pytorch_2_1_2_single_tensor_same_device_native_dtype_bit_exact_after_d0_v2"
)
PYTORCH_REFERENCE_VERSION = "2.1.2"
NAMED_TENSOR_HASH_CONTRACT = "cr-sitta-d0-v2-named-tensor-bundle-sha256-v1"
OPTIMIZER_STATE_HASH_CONTRACT = (
    "cr-sitta-d0-v2-ordered-optimizer-state-bundle-sha256-v1"
)

NamedParameters: TypeAlias = Sequence[tuple[str, nn.Parameter]]
LiveNamedGradients: TypeAlias = tuple[tuple[str, Tensor], ...]
GradientOwnershipCallback: TypeAlias = Callable[[LiveNamedGradients], None]


class D0V2NativeStepError(RuntimeError):
    """The actual first optimizer step violates the frozen D0-v2 contract."""


def _finite_positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D0V2NativeStepError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise D0V2NativeStepError(f"{field} must be finite and positive")
    return result


def frozen_first_step_spec(
    name: Literal["Adam", "SGD"], learning_rate: float
) -> OptimizerFirstStepSpec:
    """Build the exact frozen Binary-TENT empty-state first-step spec."""

    learning_rate = _finite_positive_float(
        learning_rate, field="learning_rate"
    )
    common: dict[str, Any] = {
        "name": name,
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "maximize": False,
        "initial_optimizer_state_empty": True,
    }
    if name == "Adam":
        mapping = {
            **common,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "decoupled_weight_decay": False,
        }
    elif name == "SGD":
        mapping = {
            **common,
            "momentum": 0.9,
            "dampening": 0.0,
            "nesterov": True,
        }
    else:
        raise D0V2NativeStepError("name must be exactly 'Adam' or 'SGD'")
    try:
        return OptimizerFirstStepSpec.from_mapping(mapping)
    except OptimizerGeometryError as exc:  # Defensive public-boundary wrapping.
        raise D0V2NativeStepError("frozen optimizer spec is invalid") from exc


def _same_exact(observed: Any, expected: Any) -> bool:
    """Compare optimizer fields without bool/int or list/tuple coercion."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(observed) == len(expected) and all(
            _same_exact(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return bool(observed == expected)


def _expected_runtime_fields(spec: OptimizerFirstStepSpec) -> dict[str, Any]:
    if not isinstance(spec, OptimizerFirstStepSpec):
        raise D0V2NativeStepError(
            "expected_spec must be OptimizerFirstStepSpec"
        )
    if not spec.initial_optimizer_state_empty:
        raise D0V2NativeStepError("expected optimizer state must be empty")
    if not spec.is_current_binary_tent_configuration:
        raise D0V2NativeStepError(
            "expected_spec differs from the frozen Binary-TENT configuration"
        )
    if spec.name == "Adam":
        return {
            "lr": spec.learning_rate,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        }
    return {
        "lr": spec.learning_rate,
        "momentum": 0.9,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "nesterov": True,
        "maximize": False,
        "foreach": False,
        "differentiable": False,
    }


def _materialize_named_parameters(
    values: NamedParameters,
) -> tuple[tuple[str, nn.Parameter], ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise D0V2NativeStepError(
            "named_parameters must be an ordered sequence of pairs"
        )
    materialized = tuple(values)
    if not materialized:
        raise D0V2NativeStepError("named_parameters must not be empty")
    names: list[str] = []
    parameter_ids: list[int] = []
    devices: set[torch.device] = set()
    for index, item in enumerate(materialized):
        if not isinstance(item, tuple) or len(item) != 2:
            raise D0V2NativeStepError(
                f"named_parameters[{index}] must be a (name, parameter) pair"
            )
        name, parameter = item
        if not isinstance(name, str) or not name or "\0" in name:
            raise D0V2NativeStepError(
                f"named_parameters[{index}] has an invalid name"
            )
        if not isinstance(parameter, nn.Parameter):
            raise D0V2NativeStepError(
                f"named_parameters[{index}] must contain nn.Parameter"
            )
        if (
            parameter.layout != torch.strided
            or parameter.is_sparse
            or not torch.is_floating_point(parameter)
            or parameter.is_complex()
            or parameter.numel() <= 0
        ):
            raise D0V2NativeStepError(
                f"parameter {name!r} must be a non-empty real strided tensor"
            )
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise D0V2NativeStepError(f"parameter {name!r} is non-finite")
        names.append(name)
        parameter_ids.append(id(parameter))
        devices.add(parameter.device)
    if len(set(names)) != len(names):
        raise D0V2NativeStepError("parameter names must be unique")
    if len(set(parameter_ids)) != len(parameter_ids):
        raise D0V2NativeStepError("parameter objects must be unique")
    if len(devices) != 1:
        raise D0V2NativeStepError(
            "all observed parameters must reside on one device"
        )
    return materialized


def _tensor_payload(value: Tensor, *, field: str) -> tuple[bytes, bytes, bytes]:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.is_sparse
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise D0V2NativeStepError(f"{field} must be a real strided tensor")
    tensor = value.detach()
    if not bool(torch.isfinite(tensor).all().item()):
        raise D0V2NativeStepError(f"{field} contains NaN or Inf")
    cpu = tensor.cpu().contiguous()
    dtype = str(cpu.dtype).encode("ascii")
    shape = ",".join(str(item) for item in cpu.shape).encode("ascii")
    payload = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return dtype, shape, payload


def _named_tensor_bundle_sha256(
    values: Sequence[tuple[str, Tensor]], *, contract: str
) -> str:
    materialized = tuple(values)
    if not materialized:
        raise D0V2NativeStepError("named tensor hash bundle must not be empty")
    names = tuple(name for name, _ in materialized)
    if (
        len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise D0V2NativeStepError("named tensor hash topology is invalid")
    digest = hashlib.sha256()
    digest.update(contract.encode("ascii") + b"\0")
    for name, value in materialized:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        for component in _tensor_payload(value, field=f"hash_bundle.{name}"):
            digest.update(len(component).to_bytes(8, "big"))
            digest.update(component)
    return digest.hexdigest()


def _optimizer_state_items(
    names: tuple[str, ...],
    parameters: tuple[nn.Parameter, ...],
    state: Mapping[Any, Any],
) -> tuple[tuple[str, Tensor], ...]:
    if len(state) != len(parameters):
        raise D0V2NativeStepError(
            "optimizer state must contain exactly one entry per parameter"
        )
    flattened: list[tuple[str, Tensor]] = []
    for name, parameter in zip(names, parameters, strict=True):
        entry = state.get(parameter)
        if not isinstance(entry, Mapping) or not entry:
            raise D0V2NativeStepError(
                f"optimizer state is missing for parameter {name!r}"
            )
        for field in sorted(entry):
            if not isinstance(field, str) or not field or "\0" in field:
                raise D0V2NativeStepError(
                    f"optimizer state field is invalid for {name!r}"
                )
            value = entry[field]
            if not isinstance(value, Tensor):
                raise D0V2NativeStepError(
                    f"optimizer state {name}.{field} must be a tensor"
                )
            flattened.append((f"{name}\0{field}", value))
    return tuple(flattened)


def _verify_runtime_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    spec: OptimizerFirstStepSpec,
    parameters: tuple[nn.Parameter, ...],
) -> dict[str, Any]:
    observed_version = str(torch.__version__).split("+", 1)[0]
    if GEOMETRY_PYTORCH_REFERENCE_VERSION != PYTORCH_REFERENCE_VERSION:
        raise D0V2NativeStepError(
            "public geometry reference version differs from D0-v2"
        )
    if observed_version != PYTORCH_REFERENCE_VERSION:
        raise D0V2NativeStepError(
            "D0-v2 native reference requires torch=="
            f"{PYTORCH_REFERENCE_VERSION}, observed {torch.__version__}"
        )
    default_scalar = torch.empty(())
    if default_scalar.dtype != torch.float32 or default_scalar.device.type != "cpu":
        raise D0V2NativeStepError(
            "D0-v2 native reference requires CPU float32 default tensors"
        )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise D0V2NativeStepError("optimizer must be torch.optim.Optimizer")
    if not all(parameter.requires_grad for parameter in parameters):
        raise D0V2NativeStepError(
            "all observed parameters must be trainable at the actual step"
        )
    if optimizer.state:
        raise D0V2NativeStepError("optimizer state must be empty before first step")
    if len(optimizer.param_groups) != 1:
        raise D0V2NativeStepError("optimizer must contain exactly one param group")
    expected_class = torch.optim.Adam if spec.name == "Adam" else torch.optim.SGD
    if type(optimizer) is not expected_class:
        raise D0V2NativeStepError(
            f"optimizer class must be exactly torch.optim.{spec.name}"
        )
    expected = _expected_runtime_fields(spec)
    group = optimizer.param_groups[0]
    if set(optimizer.defaults) != set(expected):
        raise D0V2NativeStepError(
            "optimizer.defaults fields differ from frozen contract"
        )
    if set(group) != {"params", *expected}:
        raise D0V2NativeStepError(
            "optimizer param_group fields differ from frozen contract"
        )
    runtime_parameters = group.get("params")
    if not isinstance(runtime_parameters, list) or tuple(
        id(value) for value in runtime_parameters
    ) != tuple(id(value) for value in parameters):
        raise D0V2NativeStepError(
            "optimizer param_group parameter identity/order differs"
        )
    for source_name, source in (
        ("defaults", optimizer.defaults),
        ("param_group", group),
    ):
        mismatches = {
            key: {"expected": expected_value, "observed": source.get(key)}
            for key, expected_value in expected.items()
            if not _same_exact(source.get(key), expected_value)
        }
        if mismatches:
            raise D0V2NativeStepError(
                f"optimizer {source_name} differs from frozen contract: {mismatches}"
            )
    return {
        "optimizer_class": type(optimizer).__name__,
        "optimizer_config_from_actual_param_group": spec.to_dict(),
        "implementation_flags_from_actual_param_group": {
            "foreach": group["foreach"],
            "fused": group.get("fused"),
            "capturable": group.get("capturable"),
            "differentiable": group["differentiable"],
        },
        "defaults_from_actual_optimizer": {
            key: list(optimizer.defaults[key])
            if isinstance(optimizer.defaults[key], tuple)
            else optimizer.defaults[key]
            for key in sorted(expected)
        },
    }


def _live_gradients(
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
) -> LiveNamedGradients:
    result: list[tuple[str, Tensor]] = []
    gradient_ids: list[int] = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if not isinstance(gradient, Tensor):
            raise D0V2NativeStepError(
                f"parameter {name!r} must have one real gradient tensor"
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
            raise D0V2NativeStepError(
                f"gradient {name!r} must match parameter shape/dtype/device"
            )
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise D0V2NativeStepError(f"gradient {name!r} contains NaN or Inf")
        result.append((name, gradient))
        gradient_ids.append(id(gradient))
    if len(set(gradient_ids)) != len(gradient_ids):
        raise D0V2NativeStepError(
            "gradient tensor objects must be unique within one candidate"
        )
    return tuple(result)


def _clone_named(
    values: Sequence[tuple[str, Tensor]],
) -> tuple[tuple[str, Tensor], ...]:
    return tuple(
        (
            name,
            value.detach().clone(memory_format=torch.preserve_format),
        )
        for name, value in values
    )


def _assert_parameters_unchanged(
    *,
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
    snapshot: tuple[tuple[str, Tensor], ...],
    context: str,
) -> None:
    if tuple(name for name, _ in named_parameters) != tuple(
        name for name, _ in snapshot
    ):
        raise D0V2NativeStepError("parameter topology changed internally")
    for (name, parameter), (_, expected) in zip(
        named_parameters, snapshot, strict=True
    ):
        if not torch.equal(parameter.detach(), expected):
            raise D0V2NativeStepError(f"{context} mutated parameter: {name}")


def _assert_live_gradients_unchanged(
    *,
    named_parameters: tuple[tuple[str, nn.Parameter], ...],
    live: LiveNamedGradients,
    snapshot: tuple[tuple[str, Tensor], ...],
    context: str,
) -> None:
    if tuple(name for name, _ in live) != tuple(name for name, _ in snapshot):
        raise D0V2NativeStepError("gradient topology changed internally")
    for (name, parameter), (_, original), (_, expected) in zip(
        named_parameters, live, snapshot, strict=True
    ):
        if parameter.grad is not original:
            raise D0V2NativeStepError(
                f"{context} replaced the live gradient object: {name}"
            )
        if not torch.equal(original.detach(), expected):
            raise D0V2NativeStepError(
                f"{context} mutated the live gradient value: {name}"
            )


def _assert_state_exact(
    *,
    optimizer: torch.optim.Optimizer,
    spec: OptimizerFirstStepSpec,
    names: tuple[str, ...],
    parameters: tuple[nn.Parameter, ...],
    reference_state: Mapping[str, Mapping[str, Tensor]],
) -> tuple[tuple[tuple[str, Tensor], ...], int]:
    if tuple(reference_state) != names or len(optimizer.state) != len(parameters):
        raise D0V2NativeStepError(
            "actual/reference optimizer state parameter topology differs"
        )
    exact_count = 0
    for name, parameter in zip(names, parameters, strict=True):
        actual = optimizer.state.get(parameter)
        expected = reference_state[name]
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise D0V2NativeStepError(
                f"optimizer state fields differ for parameter {name!r}"
            )
        required = (
            ({"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
             if spec.amsgrad else {"step", "exp_avg", "exp_avg_sq"})
            if spec.name == "Adam"
            else {"momentum_buffer"}
        )
        if set(actual) != required:
            raise D0V2NativeStepError(
                f"optimizer state fields are not frozen for parameter {name!r}"
            )
        for field in sorted(required):
            observed = actual[field]
            wanted = expected[field]
            if not isinstance(observed, Tensor) or not isinstance(wanted, Tensor):
                raise D0V2NativeStepError(
                    f"optimizer state {name}.{field} must be a tensor"
                )
            if (
                observed.shape != wanted.shape
                or observed.dtype != wanted.dtype
                or observed.device != wanted.device
                or not bool(torch.isfinite(observed.detach()).all().item())
                or not torch.equal(observed.detach(), wanted)
            ):
                raise D0V2NativeStepError(
                    f"optimizer state is not bit-exact: {name}.{field}"
                )
            exact_count += 1
        if spec.name == "Adam":
            step = actual["step"]
            if (
                step.shape != torch.Size([])
                or step.dtype != torch.float32
                or step.device.type != "cpu"
                or float(step.item()) != 1.0
            ):
                raise D0V2NativeStepError(
                    f"Adam step counter is not frozen at one: {name}"
                )
    return _optimizer_state_items(names, parameters, optimizer.state), exact_count


@dataclass(frozen=True)
class NativeFirstStepObservation:
    """Immutable evidence returned after one verified actual optimizer step."""

    optimizer_name: Literal["Adam", "SGD"]
    learning_rate: float
    parameter_names: tuple[str, ...]
    device: str
    runtime_optimizer: Mapping[str, Any]
    parameter_before_bundle_sha256: str
    gradient_bundle_sha256: str
    reference_parameter_after_bundle_sha256: str
    actual_parameter_after_bundle_sha256: str
    parameter_delta_bundle_sha256: str
    reference_optimizer_state_bundle_sha256: str
    actual_optimizer_state_bundle_sha256: str
    parameter_tensor_count: int
    gradient_tensor_count: int
    scalar_parameter_count: int
    changed_parameter_tensor_count: int
    native_reference_parameter_tensor_count: int
    native_reference_optimizer_state_tensor_count: int
    optimizer_state_parameter_count: int
    optimizer_state_tensor_count: int
    bit_exact_parameter_tensor_count: int
    bit_exact_optimizer_state_tensor_count: int
    step_norm_l2: float
    gradient_callback_invoked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "reference_name": REFERENCE_NAME,
            "pytorch_version_required": PYTORCH_REFERENCE_VERSION,
            "pytorch_version_observed": str(torch.__version__).split("+", 1)[0],
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "parameter_names": list(self.parameter_names),
            "device": self.device,
            "runtime_optimizer": dict(self.runtime_optimizer),
            "hash_contracts": {
                "named_tensors": NAMED_TENSOR_HASH_CONTRACT,
                "optimizer_state": OPTIMIZER_STATE_HASH_CONTRACT,
            },
            "hashes": {
                "parameter_before_bundle_sha256": (
                    self.parameter_before_bundle_sha256
                ),
                "gradient_bundle_sha256": self.gradient_bundle_sha256,
                "reference_parameter_after_bundle_sha256": (
                    self.reference_parameter_after_bundle_sha256
                ),
                "actual_parameter_after_bundle_sha256": (
                    self.actual_parameter_after_bundle_sha256
                ),
                "parameter_delta_bundle_sha256": (
                    self.parameter_delta_bundle_sha256
                ),
                "reference_optimizer_state_bundle_sha256": (
                    self.reference_optimizer_state_bundle_sha256
                ),
                "actual_optimizer_state_bundle_sha256": (
                    self.actual_optimizer_state_bundle_sha256
                ),
            },
            "counts": {
                "optimizer_step_call_count": 1,
                "parameter_tensor_count": self.parameter_tensor_count,
                "gradient_tensor_count": self.gradient_tensor_count,
                "scalar_parameter_count": self.scalar_parameter_count,
                "changed_parameter_tensor_count": (
                    self.changed_parameter_tensor_count
                ),
                "native_reference_parameter_tensor_count": (
                    self.native_reference_parameter_tensor_count
                ),
                "native_reference_optimizer_state_tensor_count": (
                    self.native_reference_optimizer_state_tensor_count
                ),
                "optimizer_state_parameter_count": (
                    self.optimizer_state_parameter_count
                ),
                "optimizer_state_tensor_count": self.optimizer_state_tensor_count,
                "bit_exact_parameter_tensor_count": (
                    self.bit_exact_parameter_tensor_count
                ),
                "bit_exact_optimizer_state_tensor_count": (
                    self.bit_exact_optimizer_state_tensor_count
                ),
            },
            "step_norm_l2": self.step_norm_l2,
            "gates": {
                "single_param_group_verified": True,
                "ordered_parameter_identity_verified": True,
                "initial_optimizer_state_empty": True,
                "live_gradient_objects_captured_inside_actual_step": True,
                "gradient_callback_invoked": self.gradient_callback_invoked,
                "live_gradient_identity_preserved": True,
                "live_gradient_values_preserved": True,
                "original_optimizer_step_called_once": True,
                "all_parameter_endpoints_bit_exact": True,
                "all_optimizer_state_tensors_bit_exact": True,
                "finite": True,
            },
            "authorization": {
                "engineering_observation_only": True,
                "scientific_gate_status": "unresolved",
                "scientific_selection_performed": False,
                "stage2_authorized": False,
            },
        }


class NativeFirstStepObserver:
    """One-shot context manager wrapping an actual ``optimizer.step`` call."""

    _MARKER = "_cr_sitta_d0_v2_native_first_step_observer"

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        named_parameters: NamedParameters,
        expected_spec: OptimizerFirstStepSpec,
        gradient_callback: GradientOwnershipCallback | None = None,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise D0V2NativeStepError("optimizer must be torch.optim.Optimizer")
        if gradient_callback is not None and not callable(gradient_callback):
            raise D0V2NativeStepError("gradient_callback must be callable or None")
        self._optimizer = optimizer
        self._named_parameters = _materialize_named_parameters(named_parameters)
        self._parameters = tuple(value for _, value in self._named_parameters)
        self._names = tuple(name for name, _ in self._named_parameters)
        self._expected_spec = expected_spec
        _expected_runtime_fields(expected_spec)
        self._gradient_callback = gradient_callback
        self._entered = False
        self._closed = False
        self._step_started = False
        self._observation: NativeFirstStepObservation | None = None
        self._original_bound_step: Callable[..., Any] | None = None
        self._had_instance_step = False
        self._original_instance_step: Any = None

    @property
    def observation(self) -> NativeFirstStepObservation:
        if self._observation is None:
            raise D0V2NativeStepError(
                "native first-step observation is not complete"
            )
        return self._observation

    def __enter__(self) -> "NativeFirstStepObserver":
        if self._entered or self._closed:
            raise D0V2NativeStepError("native first-step observer cannot be reused")
        if hasattr(self._optimizer, self._MARKER):
            raise D0V2NativeStepError("optimizer is already observed")
        self._entered = True
        self._had_instance_step = "step" in self._optimizer.__dict__
        self._original_instance_step = self._optimizer.__dict__.get("step")
        self._original_bound_step = self._optimizer.step
        setattr(self._optimizer, self._MARKER, self)
        self._optimizer.step = self._observed_step  # type: ignore[method-assign]
        return self

    def _observed_step(self, *args: Any, **kwargs: Any) -> Any:
        if not self._entered or self._closed:
            raise D0V2NativeStepError("observer is not active")
        if self._step_started:
            raise D0V2NativeStepError(
                "actual optimizer.step may be called exactly once"
            )
        self._step_started = True
        if args or kwargs:
            raise D0V2NativeStepError(
                "D0-v2 first step forbids optimizer closure/step arguments"
            )

        runtime = _verify_runtime_optimizer(
            self._optimizer,
            spec=self._expected_spec,
            parameters=self._parameters,
        )
        live_gradients = _live_gradients(self._named_parameters)
        parameters_before = _clone_named(
            tuple(
                (name, parameter.detach())
                for name, parameter in self._named_parameters
            )
        )
        gradient_snapshot = _clone_named(live_gradients)

        callback_invoked = self._gradient_callback is not None
        if self._gradient_callback is not None:
            callback_result = self._gradient_callback(live_gradients)
            if callback_result is not None:
                raise D0V2NativeStepError(
                    "gradient_callback must return None"
                )
        _assert_live_gradients_unchanged(
            named_parameters=self._named_parameters,
            live=live_gradients,
            snapshot=gradient_snapshot,
            context="gradient_callback",
        )
        _assert_parameters_unchanged(
            named_parameters=self._named_parameters,
            snapshot=parameters_before,
            context="gradient_callback",
        )
        # The callback is an ownership-registration hook, not an opportunity
        # to alter optimizer configuration or materialize optimizer state.
        _verify_runtime_optimizer(
            self._optimizer,
            spec=self._expected_spec,
            parameters=self._parameters,
        )

        try:
            reference = pytorch_first_step_reference(
                parameters_before=parameters_before,
                gradients=gradient_snapshot,
                optimizer=self._expected_spec,
            )
        except OptimizerGeometryError as exc:
            raise D0V2NativeStepError(
                "same-device native first-step reference failed"
            ) from exc
        if tuple(reference.parameters_after) != self._names or tuple(
            reference.optimizer_state
        ) != self._names:
            raise D0V2NativeStepError("native reference topology differs")

        if self._original_bound_step is None:
            raise D0V2NativeStepError("original optimizer step is unavailable")
        result = self._original_bound_step()

        # A supported non-differentiable optimizer must retain the exact raw
        # grad objects and values.  This also detects an unsafe custom step.
        _assert_live_gradients_unchanged(
            named_parameters=self._named_parameters,
            live=live_gradients,
            snapshot=gradient_snapshot,
            context="optimizer.step",
        )
        _verify_runtime_optimizer_after_step_configuration(
            self._optimizer,
            spec=self._expected_spec,
            parameters=self._parameters,
        )

        actual_after = tuple(
            (name, parameter.detach()) for name, parameter in self._named_parameters
        )
        exact_parameter_count = 0
        for name, actual in actual_after:
            expected = reference.parameters_after[name]
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.device != expected.device
                or not bool(torch.isfinite(actual).all().item())
                or not torch.equal(actual, expected)
            ):
                raise D0V2NativeStepError(
                    f"parameter endpoint is not bit-exact: {name}"
                )
            exact_parameter_count += 1

        actual_state_items, exact_state_count = _assert_state_exact(
            optimizer=self._optimizer,
            spec=self._expected_spec,
            names=self._names,
            parameters=self._parameters,
            reference_state=reference.optimizer_state,
        )
        reference_state_items = tuple(
            (f"{name}\0{field}", reference.optimizer_state[name][field])
            for name in self._names
            for field in sorted(reference.optimizer_state[name])
        )
        deltas = tuple(
            (name, actual - before)
            for (name, actual), (_, before) in zip(
                actual_after, parameters_before, strict=True
            )
        )
        changed_count = sum(
            not torch.equal(actual, before)
            for (_, actual), (_, before) in zip(
                actual_after, parameters_before, strict=True
            )
        )
        squared_norm = sum(
            float(
                torch.sum(delta.detach().to(dtype=torch.float64).square())
                .cpu()
                .item()
            )
            for _, delta in deltas
        )
        step_norm_l2 = math.sqrt(squared_norm)
        if not math.isfinite(step_norm_l2):
            raise D0V2NativeStepError("parameter step norm is non-finite")

        before_hash = _named_tensor_bundle_sha256(
            parameters_before, contract=NAMED_TENSOR_HASH_CONTRACT
        )
        gradient_hash = _named_tensor_bundle_sha256(
            gradient_snapshot, contract=NAMED_TENSOR_HASH_CONTRACT
        )
        reference_after_items = tuple(reference.parameters_after.items())
        reference_after_hash = _named_tensor_bundle_sha256(
            reference_after_items, contract=NAMED_TENSOR_HASH_CONTRACT
        )
        actual_after_hash = _named_tensor_bundle_sha256(
            actual_after, contract=NAMED_TENSOR_HASH_CONTRACT
        )
        if reference_after_hash != actual_after_hash:
            raise D0V2NativeStepError(
                "actual/reference parameter endpoint bundle hash differs"
            )
        reference_state_hash = _named_tensor_bundle_sha256(
            reference_state_items, contract=OPTIMIZER_STATE_HASH_CONTRACT
        )
        actual_state_hash = _named_tensor_bundle_sha256(
            actual_state_items, contract=OPTIMIZER_STATE_HASH_CONTRACT
        )
        if reference_state_hash != actual_state_hash:
            raise D0V2NativeStepError(
                "actual/reference optimizer state bundle hash differs"
            )

        self._observation = NativeFirstStepObservation(
            optimizer_name=self._expected_spec.name,
            learning_rate=self._expected_spec.learning_rate,
            parameter_names=self._names,
            device=str(self._parameters[0].device),
            runtime_optimizer=runtime,
            parameter_before_bundle_sha256=before_hash,
            gradient_bundle_sha256=gradient_hash,
            reference_parameter_after_bundle_sha256=reference_after_hash,
            actual_parameter_after_bundle_sha256=actual_after_hash,
            parameter_delta_bundle_sha256=_named_tensor_bundle_sha256(
                deltas, contract=NAMED_TENSOR_HASH_CONTRACT
            ),
            reference_optimizer_state_bundle_sha256=reference_state_hash,
            actual_optimizer_state_bundle_sha256=actual_state_hash,
            parameter_tensor_count=len(self._parameters),
            gradient_tensor_count=len(live_gradients),
            scalar_parameter_count=sum(
                parameter.numel() for parameter in self._parameters
            ),
            changed_parameter_tensor_count=changed_count,
            native_reference_parameter_tensor_count=len(
                reference.parameters_after
            ),
            native_reference_optimizer_state_tensor_count=len(
                reference_state_items
            ),
            optimizer_state_parameter_count=len(self._optimizer.state),
            optimizer_state_tensor_count=len(actual_state_items),
            bit_exact_parameter_tensor_count=exact_parameter_count,
            bit_exact_optimizer_state_tensor_count=exact_state_count,
            step_norm_l2=step_norm_l2,
            gradient_callback_invoked=callback_invoked,
        )
        return result

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del traceback
        if self._entered and not self._closed:
            if self._had_instance_step:
                self._optimizer.__dict__["step"] = self._original_instance_step
            else:
                self._optimizer.__dict__.pop("step", None)
            if getattr(self._optimizer, self._MARKER, None) is self:
                delattr(self._optimizer, self._MARKER)
            self._closed = True
        if exc_type is None and self._observation is None:
            raise D0V2NativeStepError(
                "context ended without one verified optimizer.step"
            )
        return False


def _verify_runtime_optimizer_after_step_configuration(
    optimizer: torch.optim.Optimizer,
    *,
    spec: OptimizerFirstStepSpec,
    parameters: tuple[nn.Parameter, ...],
) -> None:
    """Re-check exact class/group/defaults after the original step.

    ``_verify_runtime_optimizer`` also requires empty state, so the post-step
    configuration check is kept separate and intentionally small.
    """

    expected_class = torch.optim.Adam if spec.name == "Adam" else torch.optim.SGD
    expected = _expected_runtime_fields(spec)
    if type(optimizer) is not expected_class or len(optimizer.param_groups) != 1:
        raise D0V2NativeStepError("optimizer topology changed during step")
    group = optimizer.param_groups[0]
    if set(optimizer.defaults) != set(expected) or set(group) != {
        "params",
        *expected,
    }:
        raise D0V2NativeStepError("optimizer fields changed during step")
    if tuple(id(value) for value in group["params"]) != tuple(
        id(value) for value in parameters
    ):
        raise D0V2NativeStepError("optimizer parameter order changed during step")
    for source_name, source in (
        ("defaults", optimizer.defaults),
        ("param_group", group),
    ):
        if any(
            not _same_exact(source.get(key), expected_value)
            for key, expected_value in expected.items()
        ):
            raise D0V2NativeStepError(
                f"optimizer {source_name} changed during step"
            )


def execute_observed_native_first_step(
    optimizer: torch.optim.Optimizer,
    *,
    named_parameters: NamedParameters,
    expected_spec: OptimizerFirstStepSpec,
    gradient_callback: GradientOwnershipCallback | None = None,
) -> NativeFirstStepObservation:
    """Execute and verify exactly one actual optimizer step."""

    observer = NativeFirstStepObserver(
        optimizer,
        named_parameters=named_parameters,
        expected_spec=expected_spec,
        gradient_callback=gradient_callback,
    )
    with observer:
        optimizer.step()
    return observer.observation


__all__ = [
    "D0V2NativeStepError",
    "GradientOwnershipCallback",
    "LiveNamedGradients",
    "NAMED_TENSOR_HASH_CONTRACT",
    "NativeFirstStepObservation",
    "NativeFirstStepObserver",
    "OPTIMIZER_STATE_HASH_CONTRACT",
    "PROTOCOL_ID",
    "PYTORCH_REFERENCE_VERSION",
    "REFERENCE_NAME",
    "SCHEMA_VERSION",
    "execute_observed_native_first_step",
    "frozen_first_step_spec",
]
