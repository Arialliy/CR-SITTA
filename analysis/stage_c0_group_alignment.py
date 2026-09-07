"""Pure vector geometry for the train-only Stage-C0 outer evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor


class StageC0AlignmentError(ValueError):
    """A proxy/task/metric gradient violates the Stage-C0 vector contract."""


@dataclass(frozen=True, slots=True)
class GroupAlignmentAudit:
    proxy_gradient_norm: float
    task_gradient_norm: float
    both_gradients_nonzero: bool
    outer_task_gradient_cosine: float | None
    normalized_virtual_step_task_directional_derivative: float | None
    candidate_absolute_response_derivative: float | None
    candidate_local_contrast_derivative: float | None
    virtual_step_norm: float
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vector(value: Tensor, *, name: str, expected_numel: int | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point() or value.is_complex() or value.numel() == 0:
        raise StageC0AlignmentError(f"{name} must be a non-empty real vector")
    flattened = value.reshape(-1)
    if expected_numel is not None and flattened.numel() != expected_numel:
        raise StageC0AlignmentError(f"{name} scalar count differs")
    if not bool(torch.isfinite(flattened).all().item()):
        raise StageC0AlignmentError(f"{name} contains NaN or Inf")
    return flattened


def flatten_gradients(
    gradients: tuple[Tensor | None, ...] | list[Tensor | None],
    parameters: tuple[Tensor, ...] | list[Tensor],
) -> Tensor:
    """Flatten gradients in parameter order, representing unused entries by 0."""

    if len(gradients) != len(parameters) or not parameters:
        raise StageC0AlignmentError(
            "gradients and parameters must be equally sized non-empty sequences"
        )
    pieces: list[Tensor] = []
    reference_device = parameters[0].device
    reference_dtype = parameters[0].dtype
    for index, (gradient, parameter) in enumerate(
        zip(gradients, parameters, strict=True)
    ):
        if not isinstance(parameter, Tensor) or parameter.numel() == 0:
            raise StageC0AlignmentError(f"parameter {index} is invalid")
        if parameter.device != reference_device or parameter.dtype != reference_dtype:
            raise StageC0AlignmentError("parameter layout must share device/dtype")
        if gradient is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
            continue
        if (
            not isinstance(gradient, Tensor)
            or gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or gradient.dtype != parameter.dtype
        ):
            raise StageC0AlignmentError(f"gradient {index} layout differs")
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise StageC0AlignmentError(f"gradient {index} contains NaN or Inf")
        pieces.append(gradient.detach().reshape(-1))
    return torch.cat(pieces)


def _positive_finite(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise StageC0AlignmentError(f"{name} must be finite and positive")
    return result


def clipped_descent_direction(
    proxy_gradient: Tensor,
    *,
    radius: float,
    nonzero_epsilon: float = 1.0e-12,
) -> Tensor:
    """Return ``-g * min(1, radius / (||g|| + eps))`` exactly."""

    proxy = _vector(proxy_gradient, name="proxy_gradient")
    checked_radius = _positive_finite(radius, name="radius")
    checked_epsilon = _positive_finite(
        nonzero_epsilon, name="nonzero_epsilon"
    )
    norm = float(torch.linalg.vector_norm(proxy).item())
    if norm <= checked_epsilon:
        return torch.zeros_like(proxy)
    scale = min(1.0, checked_radius / (norm + checked_epsilon))
    result = -proxy * scale
    if not bool(torch.isfinite(result).all().item()):
        raise StageC0AlignmentError("clipped descent direction is non-finite")
    return result


def analyze_group_alignment(
    proxy_gradient: Tensor,
    task_gradient: Tensor,
    *,
    candidate_absolute_gradient: Tensor | None = None,
    candidate_contrast_gradient: Tensor | None = None,
    virtual_step_radius: float = 1.0,
    nonzero_epsilon: float = 1.0e-12,
) -> GroupAlignmentAudit:
    """Measure gradients along the frozen clipped proxy descent direction.

    The task gradient is supplied only by the separate train-target outer
    evaluator.  A negative task directional derivative is an improving first
    order direction; non-negative candidate derivatives mean the proxy step
    does not erode the corresponding label-free response.
    """

    proxy = _vector(proxy_gradient, name="proxy_gradient")
    task = _vector(
        task_gradient, name="task_gradient", expected_numel=proxy.numel()
    )
    checked_radius = _positive_finite(
        virtual_step_radius, name="virtual_step_radius"
    )
    checked_epsilon = _positive_finite(
        nonzero_epsilon, name="nonzero_epsilon"
    )
    proxy_norm = float(torch.linalg.vector_norm(proxy).item())
    task_norm = float(torch.linalg.vector_norm(task).item())
    proxy_nonzero = proxy_norm > checked_epsilon
    task_nonzero = task_norm > checked_epsilon
    both = proxy_nonzero and task_nonzero
    if proxy_nonzero:
        direction = clipped_descent_direction(
            proxy,
            radius=checked_radius,
            nonzero_epsilon=checked_epsilon,
        )
        step_norm = float(torch.linalg.vector_norm(direction).item())
    else:
        direction = torch.zeros_like(proxy)
        step_norm = 0.0

    def metric_derivative(value: Tensor | None, name: str) -> float | None:
        if value is None or not proxy_nonzero:
            return None
        metric = _vector(value, name=name, expected_numel=proxy.numel())
        return float(torch.dot(metric, direction).item())

    cosine = (
        float(torch.dot(proxy, task).item() / (proxy_norm * task_norm))
        if both
        else None
    )
    task_derivative = (
        float(torch.dot(task, direction).item()) if both else None
    )
    absolute_derivative = metric_derivative(
        candidate_absolute_gradient, "candidate_absolute_gradient"
    )
    contrast_derivative = metric_derivative(
        candidate_contrast_gradient, "candidate_contrast_gradient"
    )
    numeric = (
        proxy_norm,
        task_norm,
        step_norm,
        *(value for value in (
            cosine,
            task_derivative,
            absolute_derivative,
            contrast_derivative,
        ) if value is not None),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise StageC0AlignmentError("alignment audit produced NaN or Inf")
    return GroupAlignmentAudit(
        proxy_gradient_norm=proxy_norm,
        task_gradient_norm=task_norm,
        both_gradients_nonzero=both,
        outer_task_gradient_cosine=cosine,
        normalized_virtual_step_task_directional_derivative=task_derivative,
        candidate_absolute_response_derivative=absolute_derivative,
        candidate_local_contrast_derivative=contrast_derivative,
        virtual_step_norm=step_norm,
        finite=True,
    )


__all__ = [
    "GroupAlignmentAudit",
    "StageC0AlignmentError",
    "analyze_group_alignment",
    "clipped_descent_direction",
    "flatten_gradients",
]
