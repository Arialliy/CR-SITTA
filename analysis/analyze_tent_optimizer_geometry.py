#!/usr/bin/env python3
"""Audit first-step SGD/Adam geometry for source-train Binary TENT runs.

The implementation mirrors the explicit optimizer settings in
``tta.binary_tent.build_binary_tent_optimizer`` but does not hard-code them:
every hyperparameter that changes the first step is parsed into the receipt.
Only an empty-state first optimizer step is accepted by this analysis.  Two
references are deliberately kept separate:

* the verification reference replays PyTorch's single-tensor operation order
  in each parameter's native storage dtype and includes the final in-place
  parameter rounding; and
* a continuous float64 reference is retained only to explain optimizer
  geometry.  It is never used for the formula-verification gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.source_train_provenance import SourceTrainAnalysisProvenance
from tta.parameter_vector import (
    NamedTensorItems,
    ParameterVectorError,
    ParameterVectorLayout,
    vectorize_named_tensors,
)


ANALYSIS_SCHEMA_VERSION = 3
ANALYSIS_TYPE = "tent_optimizer_first_step_geometry"
PYTORCH_REFERENCE_VERSION = "2.1.2"
RUNTIME_HARD_GATE_REFERENCE = (
    "pytorch_2_1_2_single_tensor_same_device_native_dtype_bit_exact_after_v1"
)
CROSS_BACKEND_CPU_STORAGE_REPLAY_REFERENCE = (
    "pytorch_2_1_2_cpu_snapshot_native_dtype_cross_backend_storage_replay_diagnostic_v1"
)
CONTINUOUS_IDEAL_REFERENCE = "continuous_float64_empty_state_first_step_v1"
CURRENT_SGD = {
    "learning_rate": None,
    "momentum": 0.9,
    "dampening": 0.0,
    "weight_decay": 0.0,
    "nesterov": True,
    "maximize": False,
}
CURRENT_ADAM = {
    "learning_rate": None,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "maximize": False,
    "decoupled_weight_decay": False,
}


class OptimizerGeometryError(ValueError):
    """Optimizer evidence cannot represent the requested first-step audit."""


@dataclass(frozen=True)
class NativeFirstStepReference:
    """Device-preserving parameter endpoint and optimizer-state reference."""

    parameters_after: dict[str, Tensor]
    optimizer_state: dict[str, dict[str, Tensor]]


def _finite_float(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OptimizerGeometryError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        requirement = "finite and non-negative" if nonnegative else "finite"
        raise OptimizerGeometryError(f"{label} must be {requirement}")
    return result


def _exact_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise OptimizerGeometryError(f"{label} must be bool")
    return value


@dataclass(frozen=True)
class OptimizerFirstStepSpec:
    """All optimizer fields that influence a supported empty-state first step."""

    name: Literal["SGD", "Adam"]
    learning_rate: float
    weight_decay: float
    maximize: bool
    initial_optimizer_state_empty: bool
    momentum: float | None = None
    dampening: float | None = None
    nesterov: bool | None = None
    betas: tuple[float, float] | None = None
    eps: float | None = None
    amsgrad: bool | None = None
    decoupled_weight_decay: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OptimizerFirstStepSpec":
        if not isinstance(value, Mapping):
            raise OptimizerGeometryError("optimizer must be a mapping")
        name = value.get("name")
        if name not in ("SGD", "Adam"):
            raise OptimizerGeometryError("optimizer.name must be 'SGD' or 'Adam'")
        learning_rate = _finite_float(
            value.get("learning_rate"), "optimizer.learning_rate"
        )
        if learning_rate <= 0.0:
            raise OptimizerGeometryError("optimizer.learning_rate must be positive")
        weight_decay = _finite_float(
            value.get("weight_decay"),
            "optimizer.weight_decay",
            nonnegative=True,
        )
        maximize = _exact_bool(value.get("maximize"), "optimizer.maximize")
        empty = _exact_bool(
            value.get("initial_optimizer_state_empty"),
            "optimizer.initial_optimizer_state_empty",
        )
        if not empty:
            raise OptimizerGeometryError(
                "first-step geometry requires an explicitly empty optimizer state"
            )

        if name == "SGD":
            momentum = _finite_float(
                value.get("momentum"), "optimizer.momentum", nonnegative=True
            )
            dampening = _finite_float(
                value.get("dampening"), "optimizer.dampening", nonnegative=True
            )
            nesterov = _exact_bool(value.get("nesterov"), "optimizer.nesterov")
            if nesterov and (momentum <= 0.0 or dampening != 0.0):
                raise OptimizerGeometryError(
                    "Nesterov SGD requires positive momentum and zero dampening"
                )
            return cls(
                name="SGD",
                learning_rate=learning_rate,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=nesterov,
                maximize=maximize,
                initial_optimizer_state_empty=True,
            )

        betas_raw = value.get("betas")
        if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
            raise OptimizerGeometryError("optimizer.betas must contain two values")
        beta1 = _finite_float(betas_raw[0], "optimizer.betas[0]")
        beta2 = _finite_float(betas_raw[1], "optimizer.betas[1]")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise OptimizerGeometryError("optimizer betas must lie in [0, 1)")
        eps = _finite_float(value.get("eps"), "optimizer.eps")
        if eps <= 0.0:
            raise OptimizerGeometryError("optimizer.eps must be positive")
        amsgrad = _exact_bool(value.get("amsgrad"), "optimizer.amsgrad")
        decoupled = value.get("decoupled_weight_decay", False)
        decoupled = _exact_bool(
            decoupled, "optimizer.decoupled_weight_decay"
        )
        if decoupled:
            raise OptimizerGeometryError(
                "decoupled Adam weight decay is outside the current Binary TENT contract"
            )
        return cls(
            name="Adam",
            learning_rate=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            initial_optimizer_state_empty=True,
            decoupled_weight_decay=False,
        )

    @property
    def is_current_binary_tent_configuration(self) -> bool:
        if self.name == "SGD":
            return (
                self.momentum == CURRENT_SGD["momentum"]
                and self.dampening == CURRENT_SGD["dampening"]
                and self.weight_decay == CURRENT_SGD["weight_decay"]
                and self.nesterov is CURRENT_SGD["nesterov"]
                and self.maximize is CURRENT_SGD["maximize"]
            )
        return (
            self.betas == CURRENT_ADAM["betas"]
            and self.eps == CURRENT_ADAM["eps"]
            and self.weight_decay == CURRENT_ADAM["weight_decay"]
            and self.amsgrad is CURRENT_ADAM["amsgrad"]
            and self.maximize is CURRENT_ADAM["maximize"]
            and self.decoupled_weight_decay
            is CURRENT_ADAM["decoupled_weight_decay"]
        )

    def to_dict(self) -> dict[str, Any]:
        common = {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "maximize": self.maximize,
            "initial_optimizer_state_empty": True,
        }
        if self.name == "SGD":
            return {
                **common,
                "momentum": self.momentum,
                "dampening": self.dampening,
                "nesterov": self.nesterov,
            }
        return {
            **common,
            "betas": list(self.betas or ()),
            "eps": self.eps,
            "amsgrad": self.amsgrad,
            "decoupled_weight_decay": self.decoupled_weight_decay,
        }


def _active_gradient_mask(
    layout: ParameterVectorLayout,
    gradients: NamedTensorItems,
) -> Tensor:
    items = dict(gradients.items()) if isinstance(gradients, Mapping) else dict(gradients)
    # ``layout.flatten`` performs the authoritative topology validation.
    mask_parts: list[Tensor] = []
    for name, numel in zip(layout.names, layout.numels, strict=True):
        mask_parts.append(
            torch.full((numel,), items[name] is not None, dtype=torch.bool)
        )
    return torch.cat(mask_parts)


def _native_items(
    values: NamedTensorItems,
    *,
    label: str,
    allow_none: bool,
) -> tuple[tuple[str, Tensor | None], ...]:
    """Validate tensors without changing their device, dtype, shape, or order."""

    if isinstance(values, Mapping):
        materialized = tuple((name, values[name]) for name in sorted(values))
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        materialized = tuple(values)
    else:
        raise OptimizerGeometryError(f"{label} must be a mapping or sequence of pairs")
    if not materialized:
        raise OptimizerGeometryError(f"{label} must not be empty")
    names: list[str] = []
    validated: list[tuple[str, Tensor | None]] = []
    for index, item in enumerate(materialized):
        if not isinstance(item, tuple) or len(item) != 2:
            raise OptimizerGeometryError(
                f"{label}[{index}] must be a (name, tensor) pair"
            )
        name, value = item
        if not isinstance(name, str) or not name:
            raise OptimizerGeometryError(f"{label}[{index}] has an invalid name")
        names.append(name)
        if value is None:
            if not allow_none:
                raise OptimizerGeometryError(f"{label}.{name} must be a tensor")
            validated.append((name, None))
            continue
        if not isinstance(value, Tensor):
            raise OptimizerGeometryError(f"{label}.{name} must be a tensor")
        if value.layout != torch.strided or value.is_sparse:
            raise OptimizerGeometryError(f"{label}.{name} must use dense strided layout")
        if not torch.is_floating_point(value) or value.is_complex():
            raise OptimizerGeometryError(
                f"{label}.{name} must be a real floating-point tensor"
            )
        if value.numel() <= 0:
            raise OptimizerGeometryError(f"{label}.{name} must not be empty")
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise OptimizerGeometryError(f"{label}.{name} contains NaN or Inf")
        validated.append((name, value.detach()))
    if len(set(names)) != len(names):
        raise OptimizerGeometryError(f"{label} contains duplicate parameter names")
    return tuple(validated)


def _aligned_native_inputs(
    parameters_before: NamedTensorItems,
    gradients: NamedTensorItems,
    *,
    parameters_after: NamedTensorItems | None = None,
) -> tuple[
    tuple[str, ...],
    dict[str, Tensor],
    dict[str, Tensor | None],
    dict[str, Tensor] | None,
]:
    """Return exact native tensors aligned to the before-parameter topology."""

    before_items = _native_items(
        parameters_before, label="parameters_before", allow_none=False
    )
    gradient_items = _native_items(gradients, label="gradients", allow_none=True)
    before = {name: value for name, value in before_items}
    gradient = dict(gradient_items)
    names = tuple(before)
    if set(gradient) != set(names):
        raise OptimizerGeometryError("gradient name topology differs from parameters")
    after: dict[str, Tensor] | None = None
    if parameters_after is not None:
        after_items = _native_items(
            parameters_after, label="parameters_after", allow_none=False
        )
        after = {name: value for name, value in after_items}
        if set(after) != set(names):
            raise OptimizerGeometryError(
                "after-parameter name topology differs from parameters"
            )

    typed_before: dict[str, Tensor] = {}
    typed_after: dict[str, Tensor] | None = {} if after is not None else None
    for name in names:
        before_value = before[name]
        assert isinstance(before_value, Tensor)
        gradient_value = gradient[name]
        if gradient_value is not None and (
            gradient_value.shape != before_value.shape
            or gradient_value.dtype != before_value.dtype
            or gradient_value.device != before_value.device
        ):
            raise OptimizerGeometryError(
                f"gradients.{name} must match parameter shape, dtype, and device"
            )
        if after is not None:
            after_value = after[name]
            assert isinstance(after_value, Tensor)
            if (
                after_value.shape != before_value.shape
                or after_value.dtype != before_value.dtype
                or after_value.device != before_value.device
            ):
                raise OptimizerGeometryError(
                    f"parameters_after.{name} must match before shape, dtype, and device"
                )
            assert typed_after is not None
            typed_after[name] = after_value
        typed_before[name] = before_value
    return names, typed_before, gradient, typed_after


def pytorch_first_step_reference(
    *,
    parameters_before: NamedTensorItems,
    gradients: NamedTensorItems,
    optimizer: OptimizerFirstStepSpec,
) -> NativeFirstStepReference:
    """Replay PyTorch 2.1.2's empty-state single-tensor first step.

    The replay is device preserving.  Every intermediate moment/direction and
    the final parameter write use the parameter's native storage dtype.  This
    function is therefore suitable both for the runner's same-device bit-exact
    hard gate and for CPU-snapshot storage-visible diagnostics.
    """

    if not isinstance(optimizer, OptimizerFirstStepSpec):
        raise OptimizerGeometryError("optimizer must be OptimizerFirstStepSpec")
    names, before, gradient, _ = _aligned_native_inputs(
        parameters_before, gradients
    )
    result: dict[str, Tensor] = {}
    state_result: dict[str, dict[str, Tensor]] = {}
    with torch.no_grad():
        for name in names:
            parameter = before[name].clone(memory_format=torch.preserve_format)
            raw_gradient = gradient[name]
            if raw_gradient is None:
                result[name] = parameter
                state_result[name] = {}
                continue
            direction = -raw_gradient if optimizer.maximize else raw_gradient
            if optimizer.weight_decay != 0.0:
                direction = direction.add(
                    parameter, alpha=optimizer.weight_decay
                )

            if optimizer.name == "SGD":
                assert optimizer.momentum is not None
                assert optimizer.nesterov is not None
                if optimizer.momentum != 0.0:
                    # Empty-state PyTorch SGD initializes the buffer from the
                    # effective gradient; dampening applies only after step 1.
                    momentum_buffer = direction.clone().detach()
                    state_result[name] = {
                        "momentum_buffer": momentum_buffer.clone()
                    }
                    if optimizer.nesterov:
                        direction = direction.add(
                            momentum_buffer, alpha=optimizer.momentum
                        )
                    else:
                        direction = momentum_buffer
                else:
                    state_result[name] = {}
                parameter.add_(direction, alpha=-optimizer.learning_rate)
                result[name] = parameter
                continue

            assert optimizer.betas is not None
            assert optimizer.eps is not None
            beta1, beta2 = optimizer.betas
            exp_avg = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            exp_avg_sq = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            exp_avg.lerp_(direction, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                direction, direction.conj(), value=1.0 - beta2
            )
            parameter_state = {
                # PyTorch 2.1.2 creates this scalar on CPU for the frozen
                # capturable=False/fused=False path.  The runner independently
                # freezes the process default dtype to float32.
                "step": torch.tensor(1.0, dtype=torch.float32, device="cpu"),
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
            }
            bias_correction1 = 1.0 - beta1
            bias_correction2 = 1.0 - beta2
            step_size = optimizer.learning_rate / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)
            if optimizer.amsgrad:
                max_exp_avg_sq = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format
                )
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                parameter_state["max_exp_avg_sq"] = max_exp_avg_sq
                denominator = (
                    max_exp_avg_sq.sqrt() / bias_correction2_sqrt
                ).add_(optimizer.eps)
            else:
                denominator = (
                    exp_avg_sq.sqrt() / bias_correction2_sqrt
                ).add_(optimizer.eps)
            parameter.addcdiv_(exp_avg, denominator, value=-step_size)
            result[name] = parameter
            state_result[name] = parameter_state
    return NativeFirstStepReference(
        parameters_after=result,
        optimizer_state=state_result,
    )


def pytorch_first_step_reference_after(
    *,
    parameters_before: NamedTensorItems,
    gradients: NamedTensorItems,
    optimizer: OptimizerFirstStepSpec,
) -> dict[str, Tensor]:
    """Compatibility wrapper returning only the storage-visible endpoint."""

    return pytorch_first_step_reference(
        parameters_before=parameters_before,
        gradients=gradients,
        optimizer=optimizer,
    ).parameters_after


def expected_first_step(
    *,
    parameters_before: Tensor,
    gradients: Tensor,
    active_gradient_mask: Tensor,
    optimizer: OptimizerFirstStepSpec,
) -> tuple[Tensor, Tensor, str]:
    """Return the continuous ideal delta/gradient/formula in float64.

    This explanatory reference intentionally excludes native parameter-storage
    rounding.  Formula verification uses :func:`pytorch_first_step_reference_after`.
    """

    if not (
        parameters_before.ndim
        == gradients.ndim
        == active_gradient_mask.ndim
        == 1
    ) or not (
        parameters_before.numel()
        == gradients.numel()
        == active_gradient_mask.numel()
    ):
        raise OptimizerGeometryError("first-step vectors must be aligned 1-D tensors")
    signed_gradient = -gradients if optimizer.maximize else gradients
    effective = signed_gradient + optimizer.weight_decay * parameters_before
    effective = torch.where(active_gradient_mask, effective, torch.zeros_like(effective))

    if optimizer.name == "SGD":
        assert optimizer.momentum is not None
        assert optimizer.nesterov is not None
        if optimizer.momentum > 0.0:
            # PyTorch initializes the first momentum buffer to g_eff itself;
            # dampening is applied only to subsequent buffer updates.
            buffer = effective
            direction = (
                effective + optimizer.momentum * buffer
                if optimizer.nesterov
                else buffer
            )
        else:
            direction = effective
        delta = -optimizer.learning_rate * direction
        if (
            optimizer.is_current_binary_tent_configuration
            and optimizer.nesterov
            and optimizer.momentum == 0.9
        ):
            formula = (
                "delta_theta_expected = -learning_rate * (1 + 0.9) * gradient "
                "= -1.9 * learning_rate * gradient"
            )
        else:
            formula = (
                "empty-state PyTorch SGD: apply maximize/weight_decay, initialize "
                "momentum_buffer=g_eff, then apply optional Nesterov direction"
            )
        return delta, effective, formula

    assert optimizer.betas is not None
    assert optimizer.eps is not None
    beta1, beta2 = optimizer.betas
    first_moment = (1.0 - beta1) * effective
    second_moment = (1.0 - beta2) * effective.square()
    corrected_first = first_moment / (1.0 - beta1)
    corrected_second = second_moment / (1.0 - beta2)
    delta = -optimizer.learning_rate * corrected_first / (
        torch.sqrt(corrected_second) + optimizer.eps
    )
    formula = (
        "empty-state Adam: delta_theta_expected = -learning_rate * g_eff / "
        "(|g_eff| + eps); beta bias correction is explicit"
    )
    return delta, effective, formula


def _norm(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def _cosine(left: Tensor, right: Tensor) -> float | None:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        return None
    value = torch.dot(left, right) / (left_norm * right_norm)
    return float(torch.clamp(value, -1.0, 1.0).item())


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _geometry_metrics(
    actual: Tensor,
    storage_expected: Tensor,
    continuous_ideal_expected: Tensor,
    *,
    learning_rate: float,
    active: Tensor,
    small_gradient: Tensor,
    optimizer_name: str,
    near_sign_threshold: float,
) -> dict[str, Any]:
    actual_norm = _norm(actual)
    storage_expected_norm = _norm(storage_expected)
    storage_residual = actual - storage_expected
    ideal_expected_norm = _norm(continuous_ideal_expected)
    ideal_residual = actual - continuous_ideal_expected
    active_count = int(active.sum().item())
    small_count = int((small_gradient & active).sum().item())
    result: dict[str, Any] = {
        "scalar_count": actual.numel(),
        "active_gradient_scalar_count": active_count,
        "actual_step_norm": actual_norm,
        "cross_backend_cpu_storage_replay_expected_step_norm": storage_expected_norm,
        "cross_backend_cpu_storage_replay_cos_actual_expected": _cosine(
            actual, storage_expected
        ),
        "cross_backend_cpu_storage_replay_actual_to_expected_norm_ratio": _ratio(
            actual_norm, storage_expected_norm
        ),
        "cross_backend_cpu_storage_replay_residual_norm": _norm(storage_residual),
        "cross_backend_cpu_storage_replay_max_abs_residual": float(
            torch.max(torch.abs(storage_residual)).item()
        ),
        "continuous_ideal_expected_step_norm": ideal_expected_norm,
        "continuous_ideal_cos_actual_expected": _cosine(
            actual, continuous_ideal_expected
        ),
        "continuous_ideal_actual_to_expected_norm_ratio": _ratio(
            actual_norm, ideal_expected_norm
        ),
        "continuous_ideal_residual_norm": _norm(ideal_residual),
        "continuous_ideal_max_abs_residual": float(
            torch.max(torch.abs(ideal_residual)).item()
        ),
        "small_gradient_scalar_count": small_count,
        "small_gradient_fraction": (
            None if active_count == 0 else small_count / active_count
        ),
    }
    if optimizer_name == "Adam":
        near_sign = (torch.abs(actual) / learning_rate) > near_sign_threshold
        near_sign_count = int((near_sign & active).sum().item())
        result.update(
            {
                "near_sign_step_threshold": near_sign_threshold,
                "near_sign_step_scalar_count": near_sign_count,
                "near_sign_step_fraction": (
                    None if active_count == 0 else near_sign_count / active_count
                ),
            }
        )
    else:
        result.update(
            {
                "near_sign_step_threshold": None,
                "near_sign_step_scalar_count": None,
                "near_sign_step_fraction": None,
            }
        )
    return result


def analyze_optimizer_first_step(
    *,
    parameters_before: NamedTensorItems,
    gradients: NamedTensorItems,
    parameters_after: NamedTensorItems,
    optimizer_config: Mapping[str, Any],
    provenance: SourceTrainAnalysisProvenance,
    parameter_groups: Mapping[str, str] | None = None,
    small_gradient_threshold: float = 1e-8,
    near_sign_step_threshold: float = 0.9,
    verification_rtol: float = 1e-7,
    verification_atol: float = 1e-12,
) -> dict[str, Any]:
    """Diagnose an actual empty-state step against two non-gating references."""

    if not isinstance(provenance, SourceTrainAnalysisProvenance):
        raise OptimizerGeometryError("provenance must be SourceTrainAnalysisProvenance")
    if provenance.oracle_analysis or provenance.outer_evaluator_label_accesses != 0:
        raise OptimizerGeometryError("optimizer geometry must be label-free")
    spec = OptimizerFirstStepSpec.from_mapping(optimizer_config)
    small_threshold = _finite_float(
        small_gradient_threshold, "small_gradient_threshold", nonnegative=True
    )
    sign_threshold = _finite_float(
        near_sign_step_threshold,
        "near_sign_step_threshold",
        nonnegative=True,
    )
    if sign_threshold >= 1.0:
        raise OptimizerGeometryError("near_sign_step_threshold must be below one")
    rtol = _finite_float(verification_rtol, "verification_rtol", nonnegative=True)
    atol = _finite_float(verification_atol, "verification_atol", nonnegative=True)

    native_names, native_before, native_gradients, native_after = (
        _aligned_native_inputs(
            parameters_before,
            gradients,
            parameters_after=parameters_after,
        )
    )
    layout, before = vectorize_named_tensors(
        parameters_before, label="parameters_before"
    )
    if tuple(layout.names) != native_names:
        raise OptimizerGeometryError("native and analysis parameter order differs")
    gradient = layout.flatten(
        gradients, label="gradients", none_as_zero=True
    )
    after = layout.flatten(parameters_after, label="parameters_after")
    active = _active_gradient_mask(layout, gradients)
    actual = after - before
    continuous_ideal_expected, effective_gradient, formula = expected_first_step(
        parameters_before=before,
        gradients=gradient,
        active_gradient_mask=active,
        optimizer=spec,
    )
    storage_reference_after = pytorch_first_step_reference_after(
        parameters_before=native_before,
        gradients=native_gradients,
        optimizer=spec,
    )
    storage_after = layout.flatten(
        storage_reference_after, label="storage_reference_after"
    )
    storage_expected = storage_after - before
    assert native_after is not None
    small = torch.abs(gradient) < small_threshold
    global_metrics = _geometry_metrics(
        actual,
        storage_expected,
        continuous_ideal_expected,
        learning_rate=spec.learning_rate,
        active=active,
        small_gradient=small,
        optimizer_name=spec.name,
        near_sign_threshold=sign_threshold,
    )
    parameter_norm = _norm(before)
    global_metrics.update(
        {
            "gradient_norm": _norm(gradient),
            "effective_gradient_norm": _norm(effective_gradient),
            "parameter_norm_before": parameter_norm,
            "relative_actual_step_norm": _ratio(
                global_metrics["actual_step_norm"], parameter_norm
            ),
            "cpu_storage_replay_within_frozen_tolerance": bool(
                torch.allclose(
                    actual, storage_expected, rtol=rtol, atol=atol
                )
            ),
        }
    )

    groups: dict[str, Any] = {}
    assignment = layout.validated_group_assignment(parameter_groups)
    assignment_by_name = dict(assignment)
    for group, indices in layout.group_indices(assignment_by_name).items():
        group_metrics = _geometry_metrics(
            actual[indices],
            storage_expected[indices],
            continuous_ideal_expected[indices],
            learning_rate=spec.learning_rate,
            active=active[indices],
            small_gradient=small[indices],
            optimizer_name=spec.name,
            near_sign_threshold=sign_threshold,
        )
        group_metrics.update(
            {
                "parameter_tensor_count": sum(
                    assigned_group == group for _, assigned_group in assignment
                ),
                "gradient_norm": _norm(gradient[indices]),
                "effective_gradient_norm": _norm(effective_gradient[indices]),
                "parameter_norm_before": _norm(before[indices]),
                "relative_actual_step_norm": _ratio(
                    group_metrics["actual_step_norm"], _norm(before[indices])
                ),
                "cpu_storage_replay_within_frozen_tolerance": bool(
                    torch.allclose(
                        actual[indices],
                        storage_expected[indices],
                        rtol=rtol,
                        atol=atol,
                    )
                ),
            }
        )
        groups[group] = group_metrics

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "scope": provenance.to_dict(),
        "optimizer": spec.to_dict(),
        "gate_policy": {
            "optimizer_correctness_acceptance_gate": "runtime_same_device_hard_gate",
            "runtime_same_device_hard_gate_is_sole_acceptance_gate": True,
            "cpu_storage_replay_used_for_hard_gate": False,
            "continuous_ideal_used_for_hard_gate": False,
        },
        "references": {
            "cross_backend_cpu_storage_replay": {
                "name": CROSS_BACKEND_CPU_STORAGE_REPLAY_REFERENCE,
                "role": "cross_backend_diagnostic_only",
                "used_for_hard_gate": False,
                "pytorch_version": PYTORCH_REFERENCE_VERSION,
                "single_tensor_operation_order": True,
                "native_storage_dtype": True,
                "final_parameter_storage_rounding_included": True,
            },
            "continuous_ideal": {
                "name": CONTINUOUS_IDEAL_REFERENCE,
                "role": "scientific_geometry_explanation_only",
                "dtype": "torch.float64",
                "used_for_hard_gate": False,
            },
        },
        "first_step_formula": formula,
        "first_step_formula_role": "continuous_float64_explanation_only",
        "current_binary_tent_optimizer_configuration": (
            spec.is_current_binary_tent_configuration
        ),
        "empty_optimizer_state_verified": True,
        "parameter_layout": {
            "parameter_tensor_count": len(layout.names),
            "parameter_scalar_count": layout.total_numel,
            "parameter_names_sha256": layout.parameter_names_sha256,
            "topology_sha256": layout.topology_sha256,
        },
        "parameter_storage": {
            "storage_dtype_counts": dict(
                sorted(
                    {
                        dtype: sum(
                            str(native_before[name].dtype) == dtype
                            for name in native_names
                        )
                        for dtype in {
                            str(native_before[name].dtype) for name in native_names
                        }
                    }.items()
                )
            ),
        },
        "thresholds": {
            "small_gradient_absolute_threshold": small_threshold,
            "adam_near_sign_step_ratio_threshold": sign_threshold,
            "verification_rtol": rtol,
            "verification_atol": atol,
        },
        "global": global_metrics,
        "per_group": groups,
    }


def _tensor_mapping(value: Any, label: str) -> dict[str, Tensor | None]:
    if not isinstance(value, Mapping) or not value:
        raise OptimizerGeometryError(f"{label} must be a non-empty mapping")
    result: dict[str, Tensor | None] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not name:
            raise OptimizerGeometryError(f"{label} has invalid parameter name")
        if raw is None:
            result[name] = None
            continue
        try:
            result[name] = torch.tensor(raw, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise OptimizerGeometryError(f"{label}.{name} is not numeric") from exc
    return result


def analyze_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise OptimizerGeometryError("input root must be a JSON object")
    provenance = SourceTrainAnalysisProvenance.from_mapping(
        payload.get("provenance"), require_outer_oracle=False
    )
    groups = payload.get("parameter_groups")
    if groups is not None and not isinstance(groups, Mapping):
        raise OptimizerGeometryError("parameter_groups must be a mapping")
    return analyze_optimizer_first_step(
        parameters_before=_tensor_mapping(
            payload.get("parameters_before"), "parameters_before"
        ),
        gradients=_tensor_mapping(payload.get("gradients"), "gradients"),
        parameters_after=_tensor_mapping(
            payload.get("parameters_after"), "parameters_after"
        ),
        optimizer_config=payload.get("optimizer"),
        provenance=provenance,
        parameter_groups=groups,
        small_gradient_threshold=payload.get("small_gradient_threshold", 1e-8),
        near_sign_step_threshold=payload.get("near_sign_step_threshold", 0.9),
        verification_rtol=payload.get("verification_rtol", 1e-7),
        verification_atol=payload.get("verification_atol", 1e-12),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        result = analyze_payload(payload)
    except (OSError, json.JSONDecodeError, ValueError, ParameterVectorError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "ANALYSIS_TYPE",
    "CONTINUOUS_IDEAL_REFERENCE",
    "CROSS_BACKEND_CPU_STORAGE_REPLAY_REFERENCE",
    "NativeFirstStepReference",
    "OptimizerFirstStepSpec",
    "OptimizerGeometryError",
    "PYTORCH_REFERENCE_VERSION",
    "RUNTIME_HARD_GATE_REFERENCE",
    "analyze_optimizer_first_step",
    "analyze_payload",
    "expected_first_step",
    "main",
    "pytorch_first_step_reference",
    "pytorch_first_step_reference_after",
]
